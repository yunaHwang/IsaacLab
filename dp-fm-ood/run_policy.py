# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy in an Isaac Lab environment.

Dispatches to one of two policy backbones depending on --ood_detection_metric: a robomimic
diffusion policy (run_dp_policy, scored via diffdagger-style diffusion loss) or a GLOVES flow
policy (run_gloves_fm_policy, scored via density non-conformity).

Args:
    task: Name of the environment.
    dp_checkpoint: Path to the robomimic diffusion policy checkpoint. Required when
        --ood_detection_metric=diffdaggerloss.
    fm_checkpoint: Path or hub id of a trained GLOVES policy checkpoint. Required when
        --ood_detection_metric=density.
    ood_detection_metric: Which OOD signal to compute: 'diffdaggerloss' or 'density'.
    blending_mechanism: Which GLOVES action-blending mechanism to use once blending is
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

sys.path.append(os.path.join(os.path.dirname(__file__), "ood-signal", "reconstruction-loss"))
from get_diffloss_diffdagger import compute_diffusion_loss, load_loss_reference

sys.path.append(os.path.join(os.path.dirname(__file__), "ood-signal", "density"))
from density_nonconformity_score_calc import compute_density_score
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
    "the offline --loss_range sweep in diffusion_loss.py doesn't need this since it scores "
    "many distinct dataset (obs, action) pairs instead of resampling the same one.",
)

# Types of models and related parameters grouped together
# DP
parser.add_argument("--dp_checkpoint", type=str, default=None,
    help="Path of a trained diffusion policy checkpoint (see "
    "ood-signal/reconstruction-loss/get_diffloss_diffdagger.py). Required when "
    "--ood_detection_metric=diffdaggerloss")
parser.add_argument(
    "--loss_range", type=str, default=None,
    help="Path to a reference loss-distribution CSV produced by diffusion_loss.py "
    "(python diffusion_loss.py --hdf5_dataset ... --save_file_name ...). Used to score "
    "how in-distribution a live rollout's diffusion loss is via its CDF percentile.",
)

# FM
parser.add_argument(
    "--fm_checkpoint", type=str, default=None,
    help="Path or hub id of a trained GLOVES policy checkpoint (see "
    "ood-signal/density/density_nonconformity_score_calc.py). Required when "
    "--ood_detection_metric=density; scores the live human action via GLOVES' density "
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
    loss_cdf=None,
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
        loss_cdf: Optional diffusion_loss.LossCDF built from a reference loss-distribution
            CSV (see --loss_range), used to report the live loss's percentile against it.
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

        # Apply actions
        if not_blend:
            obs_dict, _, terminated, truncated, _ = env.step(policy_actions)
            obs = obs_dict["policy"]

            # Record trajectory
            traj["policy_actions"].append(policy_actions.tolist())
            traj["next_obs"].append(obs)
        else:
            # reference loss-distribution calibration is a separate offline step, see
            # diffusion_loss.py - this branch only scores the live human action against
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
                msg = f"[OOD] diffusion loss for current human action: {loss.item():.4f}"
                if loss_cdf is not None:
                    percentile = loss_cdf(loss.item())
                    msg += f" (percentile vs. reference loss range: {percentile:.3f})"
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

def run_gloves_fm_policy(gloves_policy, device, teleop_interface, not_blend = True):
    """Score the live human action against a GLOVES flow policy's density non-conformity
    score, analogous to run_dp_policy's diffdaggerloss branch.

    Not implemented yet: unlike run_dp_policy, this function has no env/success_term/horizon
    parameters, so it has no rollout loop to reset the env, step actions, check success, or
    build up obs_history/traj from - that still needs to be added before the body below
    (which references obs_history and traj without defining either) can run.

    Args:
        gloves_policy: The trained GLOVES DiTPolicy to score actions against (see
            --fm_checkpoint).
        device: The device to run the policy on.
        teleop_interface: A device (e.g. Se3SpaceMouse) to read the live human action from.
        not_blend: If True, execute the policy's own actions directly. If False, score the
            live human action's density and blend with it instead - not wired up yet.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    raise NotImplementedError("This function is not implemented yet.")

    user_action = teleop_interface.advance()

    # TODO - need to define obs_history here
    score = compute_density_score(gloves_policy.dit_flow, obs_history, user_action)
    loss = score
    print(f"[OOD] density non-conformity score for current human action: {score.item():.4f}")

    return False, traj
    

def main():
    """Run a trained policy - robomimic diffusion policy or GLOVES flow policy, chosen via
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

    # optional reference loss distribution (see diffusion_loss.py) to score live losses
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

        policy_backbone = "gloves"

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

        if policy_backbone == "gloves":
            policy = DiTPolicy.from_pretrained(args_cli.fm_checkpoint)
            policy.to(device)
            policy.eval()

            terminated, traj = run_gloves_fm_policy(policy, device, teleop_interface, not_blend)

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
