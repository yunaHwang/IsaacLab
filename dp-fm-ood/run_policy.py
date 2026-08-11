# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy in an Isaac Lab environment.

Dispatches to one of two policy backbones depending on --ood_detection_metric: a robomimic
diffusion policy (run_dp_policy, scored via diffdagger-style diffusion loss) or a fm flow
policy (run_fm_policy, scored via density non-conformity).

Args:
    task: Name of the environment.
    dp_checkpoint: Path to the robomimic diffusion policy checkpoint. Required when
        --ood_detection_metric=diffdaggerloss.
    fm_checkpoint: Path or hub id of a trained fm policy checkpoint. Required when
        --ood_detection_metric=density.
    ood_detection_metric: Which OOD signal to compute: 'diffdaggerloss' or 'density'.
    blending_mechanism: Which gloves action-blending mechanism to use once blending is
        wired up ('gloves_fpas', 'gloves_feeg', or 'gloves_ifae').
    blend: If set, enable action blending (not_blend=False); otherwise the policy's own
        actions are executed directly.
    num_samples: Number of (noise, timestep) draws to average the live diffusion loss over.
    loss_range: If provided, path to a reference loss-distribution CSV to score the live
        diffusion loss's percentile against.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os
import sys

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

sys.path.append(os.path.join(os.path.dirname(__file__), "ood-signal-baseline-papers", "reconstruction-loss"))
from get_diffloss_diffdagger import compute_diffusion_loss, load_loss_reference

sys.path.append(os.path.join(os.path.dirname(__file__), "ood-signal-baseline-papers", "density"))
from density_nonconformity_score_calc import compute_density_score, generate_action_chunk, DEFAULT_STATE_KEYS

from state_ood_signal_impl import diffdagger_loss_state_ood, fm_diffdagger_loss_state_ood, density_state_ood, cf_prediction_loss_state_ood, smoothness_loss_state_ood, perturb_loss_state_ood
from lerobot_policy_gloves.modeling_gloves import DiTPolicy

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")

parser.add_argument("--device", type=str, default="cpu")

parser.add_argument("--ood_detection_metric", type=str, choices=['diffdaggerloss', 'density']) # NOTE: add as I go
parser.add_argument("--blending_mechanism", type=str, choices=['gloves_fpas', 'gloves_feeg', 'gloves_ifae']) # NOTE: add as I go
# FOR now: have not blend as a default, meaning that without this flag, then it will default to not blendd
parser.add_argument("--blend", action = 'store_true')


parser.add_argument(
    "--num_samples", type=int, default=512,
    help="Number of (noise, timestep) draws to average the live diffusion loss over "
    "(diffdagger's Nb in noise_estimation_loss_nb_infer). The live action is fixed for "
    "the duration of a step, so averaging multiple draws reduces the loss's variance; "
    "the offline --loss_range sweep in get_diffloss_diffdagger.py doesn't need this since it scores "
    "many distinct dataset (obs, action) pairs instead of resampling the same one.",
)

# Types of models and related parameters grouped together
# DP
parser.add_argument("--dp_checkpoint", type=str, default=None,
    help="Path of a trained diffusion policy checkpoint (see "
    "ood-signal/reconstruction-loss/get_diffloss_diffdagger.py). Required when "
    "--ood_detection_metric=diffdaggerloss")
parser.add_argument(
    "--loss_state_range", type=str, default=None,
    help="Path to a reference loss-distribution CSV produced by get_diffloss_diffdagger.py "
    "(python get_diffloss_diffdagger.py --hdf5_dataset ... --save_file_name ...). Used to score "
    "how in-distribution a live rollout's diffusion loss is via its CDF percentile.",
)
parser.add_argument(
    "--loss_action_range", type=str, default=None,
    help="Path to a reference loss-distribution CSV produced by get_diffloss_diffdagger.py "
    "(python get_diffloss_diffdagger.py --hdf5_dataset ... --save_file_name ...). Used to score "
    "how in-distribution a live rollout's diffusion loss is via its CDF percentile.",
)

# FM
parser.add_argument(
    "--fm_checkpoint", type=str, default=None,
    help="Path or hub id of a trained fm policy checkpoint (see "
    "ood-signal/density/density_nonconformity_score_calc.py). Required when "
    "--ood_detection_metric=density; scores the live human action via fm' density "
    "non-conformity score instead of the diffdagger-style diffusion loss.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab_tasks.utils import parse_env_cfg

from collections import deque


def run_dp_policy(
    policy,
    env,
    success_term,
    horizon,
    device,
    teleop_interface=None,
    loss_state_cdf=None,
    loss_action_cdf=None,
    not_blend = True
):
    """Perform action blending by 1) grabbing the action from the policy, 2) registering user action,
    3) function call for some metric (loss), 4) save metric value if necessary.

    Args:
        policy: The robomimic diffusion policy to play.
        env: The environment to play in.
        success_term: The extracted success-termination term (env_cfg.terminations.success),
            called each step to check whether the rollout succeeded.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.
        teleop_interface: A device (e.g. Se3SpaceMouse) to read the live human action from.
            Required when not_blend=False.
        loss_state_cdf: Optional get_diffloss_diffdagger.LossCDF built from a reference loss-distribution
            CSV (see --loss_state_range), used to report the live loss's percentile against it.
        loss_action_cdf: Optional get_diffloss_diffdagger.LossCDF built from a reference loss-distribution
            CSV (see --loss_action_range), used to report the live loss's percentile against it.
        not_blend: If True, execute the policy's own actions directly. If False, blend with
            the live human action (see teleop_interface) instead - not wired up yet.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    if teleop_interface is not None:
        teleop_interface.reset()


    observation_horizon = 2

    obs_history = deque(maxlen=observation_horizon)


    traj = dict(policy_actions=[], blended_actions = [], obs=[], next_obs=[])

    # Prepare first observation
    obs = copy.deepcopy(obs_dict["policy"])

    obs = {
        k: obs[k]
        for k in ["eef_pos", "gripper_pos", "object", "eef_quat"]
    }

    for ob in obs:
        obs[ob] = torch.squeeze(obs[ob])


    # Initialize history with repeated first observation
    for _ in range(observation_horizon):
        obs_history.append(obs)

    
    for i in range(horizon):
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])
        obs = {
            k: obs[k]
            for k in ["eef_pos", "gripper_pos", "object", "eef_quat"]
        }

        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # Check if environment image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image


        # Add current observation to history
        obs_history.append(obs)

        # Convert observation history into diffusion-policy input
        obs_seq = {}

        for key in obs_history[0].keys():
            obs_seq[key] = torch.stack(
                [o[key] for o in obs_history],
                dim=0
            ).unsqueeze(0).to(device)

        # Debug once
        if i == 0:
            print("Observation shapes sent to policy:")
            for k, v in obs_seq.items():
                print(k, v.shape)

        traj["obs"].append(obs_seq)

        policy_actions = policy(obs_seq, batched_ob = True)

        
        # print("actions, ", actions)
        # print(len(actions))

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            policy_actions = (
                (policy_actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        policy_actions = torch.from_numpy(policy_actions).to(device=device).view(1, env.action_space.shape[1])

        ###############
        # State OOD: score the model's own action against its own state (as opposed to the
        # teleop user_action scored in the not_blend=False branch below)
        state_ood_score = diffdagger_loss_state_ood(policy, obs_seq, policy_actions, num_samples=args_cli.num_samples)
        msg = f"[state OOD] diffusion loss for model's own action: {state_ood_score.item():.4f}"
        if loss_state_cdf is not None:
            percentile = loss_state_cdf(state_ood_score.item())
            msg += f" ([STATE] percentile vs. reference loss range: {percentile:.3f})"
        print(msg)

        ###############
        # Apply actions
        if not_blend:
            obs_dict, _, terminated, truncated, _ = env.step(policy_actions)
            obs = obs_dict["policy"]

            # Record trajectory
            traj["policy_actions"].append(policy_actions.tolist())
            traj["next_obs"].append(obs)
        else:
            # reference loss-distribution calibration is a separate offline step, see
            # get_diffloss_diffdagger.py - this branch only scores the live human action against
            # the trained policy (and, if --loss_range was given, against that reference).
            if teleop_interface is None:
                raise RuntimeError(
                    "run_dp_policy(..., not_blend=False) requires a teleop_interface "
                    "(e.g. Se3SpaceMouse) to read the live human action from."
                )
            # [7]: [x, y, z, rx, ry, rz, gripper] delta-pose command, already on `device`
            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html
            user_action = teleop_interface.advance()

            loss = None
            if args_cli.ood_detection_metric == "diffdaggerloss":
                # diffdagger-style Nb-sample averaging: the action is fixed for this step,
                # so average multiple (noise, timestep) draws to reduce the loss's variance
                sample_losses = torch.stack(
                    [compute_diffusion_loss(policy, obs_seq, user_action) for _ in range(args_cli.num_samples)]
                )
                loss = sample_losses.mean(dim=0)
                msg = f"diffusion loss for current human action: {loss.item():.4f}"
                if loss_action_cdf is not None:
                    percentile = loss_action_cdf(loss.item())
                    msg += f" ([ACTION] percentile vs. reference loss range: {percentile:.3f})"
                print(msg)

            # TODO: define blended_actions from policy_actions + user_action + loss/gamma
            # (see compute_linear_gamma / compute_sigmoid_gamma in joystick_diffdagger.py
            # for the gamma-blend pattern to adapt here).
            raise NotImplementedError(
                "Action blending (not_blend=False) isn't wired up yet: user_action and its "
                "diffusion loss are now available above, but the blend rule that turns them "
                "into blended_actions still needs to be chosen."
            )

            obs_dict, _, terminated, truncated, _ = env.step(blended_actions)
            obs = obs_dict["policy"]
            
            # Record trajectory
            traj["blended_actions"].append(blended_actions.tolist())
            traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj

def run_fm_policy(fm_policy, obs_history, device, teleop_interface, not_blend = True):
    """Score the live human action against a fm flow policy's density non-conformity
    score s(x) = ||z_hat(x)||^2 (paper Eq. 6; see compute_density_score/compute_z_hat in
    density_nonconformity_score_calc.py), analogous to run_dp_policy's diffdaggerloss branch.

    Not implemented yet: unlike run_dp_policy, this function has no env/success_term/horizon
    parameters, so it has no rollout loop of its own to reset the env, step actions, check
    success, or build up obs_history/traj across steps - obs_history must be supplied by the
    caller for now (built the same way as run_dp_policy's own obs_history, from the same
    eef_pos/gripper_pos/object/eef_quat keys - see compute_density_score's docstring), and
    this only handles a single step rather than looping over `horizon` steps. Actually
    stepping the env with the generated/blended actions (in both branches below) still needs
    env/success_term/horizon wired in before it can run - see the NotImplementedError raise
    in the not_blend=False branch, and the TODO comment in the not_blend=True branch.

    Args:
        fm_policy: The trained fm DiTPolicy to score actions against (see
            --fm_checkpoint).
        obs_history: iterable of `n_obs_steps` obs dicts, in compute_density_score's expected
            format (same keys/shape as run_dp_policy's own obs_history deque). Required when
            not_blend=True (to generate fm_policy's own action chunk).
        device: The device to run the policy on.
        teleop_interface: A device (e.g. Se3SpaceMouse) to read the live human action from.
            Required when not_blend=False.
        not_blend: If True, generate and return the policy's own action chunk (env-stepping
            not wired up yet, see TODO below). If False, score the live human action's
            density.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    if not_blend:
        if obs_history is None:
            raise RuntimeError(
                "run_fm_policy(..., not_blend=True) requires obs_history (an n_obs_steps-long "
                "iterable of obs dicts, see this function's docstring) to generate the fm "
                "policy's own action chunk."
            )

        # Actually run F_theta forward (DiTFlowModel.conditional_sample's ODE solve) to get
        # the policy's own action chunk. Sampled once as the full (B, horizon, ac_dim) chunk
        # -- rather than calling generate_action_chunk twice, once truncated and once full,
        # which would draw two independent samples -- so the action actually meant for
        # execution and the action scored for state OOD below are the same sample.
        full_model_actions = generate_action_chunk(
            fm_policy.dit_flow, obs_history, device=device, full_chunk=True
        )
        start = fm_policy.config.n_obs_steps - 1
        end = start + fm_policy.config.n_action_steps
        model_actions = full_model_actions[:, start:end]
        print(f"fm policy generated action chunk of shape {tuple(model_actions.shape)}")

        # TODO: env.step through `model_actions`, mirroring run_dp_policy's not_blend=True
        # branch. Needs env/success_term/horizon wired into this function first (see
        # docstring).
        traj = dict(policy_actions=[model_actions.tolist()], blended_actions=[], obs=[], next_obs=[])

        ###############
        # State OOD: score the model's own action against its own state (as opposed to the
        # teleop user_action scored in the not_blend=False branch below). Scored against the
        # full (B, horizon, ac_dim) chunk -- compute_density_score's ac_chunk assert expects
        # that shape, and it's what the paper's z_hat is defined over -- not the
        # n_action_steps slice above.
        state_ood_score = density_state_ood(fm_policy.dit_flow, obs_history, full_model_actions)
        msg = f"[state OOD] fm density for model's own action: {state_ood_score.item():.4f}"
        print(msg)

        # fm analog of run_dp_policy's diffdagger-style noise-prediction loss (Nb-sample
        # averaged flow-matching MSE instead of DDPM noise-prediction MSE), scored against
        # the same full action chunk as the density score above.
        fm_loss = fm_diffdagger_loss_state_ood(
            fm_policy.dit_flow, obs_history, full_model_actions, num_samples=args_cli.num_samples
        )
        print(f"[state OOD] fm diffdagger-style loss for model's own action: {fm_loss.item():.4f}")

        return False, traj

    if teleop_interface is None:
        raise RuntimeError(
            "run_fm_policy(..., not_blend=False) requires a teleop_interface "
            "(e.g. Se3SpaceMouse) to read the live human action from."
        )
    # [7]: [x, y, z, rx, ry, rz, gripper] delta-pose command, already on `device`
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html
    user_action = teleop_interface.advance()

    score = compute_density_score(fm_policy.dit_flow, obs_history, user_action, device=device)
    print(f"[ACTION] density non-conformity score for current human action: {score.item():.4f}")

    # TODO: define blended_actions from fm_policy's own action + user_action + score/gamma
    # (see compute_linear_gamma / compute_sigmoid_gamma in joystick_diffdagger.py for the
    # gamma-blend pattern to adapt here), then env.step(blended_actions) and record traj -
    # needs env/success_term/horizon wired in first (see docstring above).
    raise NotImplementedError(
        "Action blending (not_blend=False) isn't wired up yet: user_action and its density "
        "score are now available above, but env-stepping/blending needs env/success_term/"
        "horizon plumbed into this function first."
    )
    

def main():
    """Run a trained policy - robomimic diffusion policy or fm flow policy, chosen via
    --ood_detection_metric - in an Isaac Lab environment."""

    import gymnasium as gym

    import isaaclab_mimic.envs

    print("Mimic envs:")
    for name in gym.registry.keys():
        if "Mimic" in name:
            print(name)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # optional reference loss distribution (see get_diffloss_diffdagger.py) to score live losses
    # against via CDF percentile
    loss_cdf = None
    if args_cli.loss_range is not None and args_cli.ood_detection_metric == "diffdaggerloss":
        loss_cdf = load_loss_reference(args_cli.loss_range)
        print(f"[INFO] Loaded reference loss CDF from {args_cli.loss_range} (range: {loss_cdf.min:.4f} - {loss_cdf.max:.4f})")

    # Wire up a input device
    teleop_interface = None
    teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(sim_device=device))
    print(teleop_interface)

    not_blend = True if not args_cli.blend else False


    # Check if necesasry arguments appear together (i.e., checkpoints with the required metrics)
    if args_cli.ood_detection_metric == "density":
        if args_cli.fm_checkpoint is None:
            raise ValueError("--fm_checkpoint is required when --ood_detection_metric=density")

        policy_backbone = "fm"

    elif args_cli.ood_detection_metric == "diffdaggerloss":
        if args_cli.dp_checkpoint is None:
            raise ValueError("--dp_checkpoint is required when --ood_detection_metric=diffdaggerloss")

        policy_backbone = "diffusion_policy"

    else:
        raise ValueError(
            "--ood_detection_metric must be 'density' or 'diffdaggerloss', "
            f"got {args_cli.ood_detection_metric!r}"
        )


    # Run policy on live actions input from Isaac Lab
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        if policy_backbone == "fm":
            policy = DiTPolicy.from_pretrained(args_cli.fm_checkpoint)
            policy.to(device)
            policy.eval()

            # Build an initial n_obs_steps-long obs_history from the first observation,
            # mirroring run_dp_policy's own history init (repeat the first obs to fill the
            # window). TODO: once run_fm_policy grows a real rollout loop (see its
            # docstring), obs_history should instead be threaded/updated step-to-step there.
            obs_dict, _ = env.reset()
            obs = copy.deepcopy(obs_dict["policy"])
            obs = {k: torch.squeeze(obs[k]) for k in DEFAULT_STATE_KEYS}
            obs_history = deque([obs] * policy.config.n_obs_steps, maxlen=policy.config.n_obs_steps)

            terminated, traj = run_fm_policy(policy, obs_history, device, teleop_interface, not_blend)

        elif policy_backbone == "diffusion_policy":

            policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.dp_checkpoint, device=device)

            terminated, traj = run_dp_policy(
                policy, env, success_term, args_cli.horizon, device,
                teleop_interface, loss_cdf, not_blend
            )

        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")
        #print("traj, ", traj)

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
