# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained robomimic diffusion policy in an Isaac Lab
environment.

Args:
    task: Name of the environment.
    device: Torch device for the simulation.
    disable_fabric: If set, disable fabric and use USD I/O operations.
    dp_checkpoint: Path to the robomimic diffusion policy checkpoint.
    blend: If set, enable action blending (not_blend=False); otherwise the policy's own
        actions are executed directly.
    num_samples: Number of (noise, timestep) draws to average the live diffusion loss over.
    loss_state_range: If provided, path to a reference loss-distribution CSV to score the
        live per-step state-OOD diffusion loss's percentile against.
    loss_action_range: If provided, path to a reference loss-distribution CSV to score the
        live human-action diffusion loss's percentile against (not_blend=False only).
    horizon: The step horizon of each rollout.
    num_rollouts: Number of rollouts to run.
    seed: Random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

# NOTE. to be run in separate isaaclab + robomimic environment
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from ood_signal_baseline_papers.reconstruction_loss.get_loss_diffdagger import load_loss_reference

from ood_signal import dp_loss

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a robomimic diffusion policy for Isaac Lab environment.")

parser.add_argument("--task", type=str, required=True, help="Name of the environment.")
# parser.add_argument("--device", type=str, default="cpu")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)

# FOR now: have not blend as a default, meaning that without this flag, then it will default to not blend
parser.add_argument("--blend", action='store_true')

parser.add_argument(
    "--num_samples", type=int, default=512,
    help="Number of (noise, timestep) draws to average the live diffusion loss over "
    "(diffdagger's Nb in noise_estimation_loss_nb_infer). The live action is fixed for "
    "the duration of a step, so averaging multiple draws reduces the loss's variance"
)

parser.add_argument(
    "--dp_checkpoint", type=str, required=True,
    help="Path of a trained diffusion policy checkpoint"
)
parser.add_argument(
    "--loss_state_range", type=str, default=None,
    help="Path to a reference STATE loss-distribution CSV produced by get_loss_diffdagger.py "
)
parser.add_argument(
    "--loss_action_range", type=str, default=None,
    help="Path to a reference ACTION loss-distribution CSV produced by get_loss_diffdagger.py "
)

parser.add_argument("--horizon", type=int, default=500, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts to run.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None,
    help="Minimum value of the action space normalization factor (unnormalizes the "
    "policy's own actions before stepping the env if both norm_factor_min/max are set).",
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None,
    help="Maximum value of the action space normalization factor.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import copy
import random
import time

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
    not_blend=True
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
        loss_state_cdf: Optional get_loss_diffdagger.LossCDF built from a reference loss-distribution
            CSV (see --loss_state_range), used to report the live loss's percentile against it.
        loss_action_cdf: Optional get_loss_diffdagger.LossCDF built from a reference loss-distribution
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

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            policy_actions = (
                (policy_actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        policy_actions = torch.from_numpy(policy_actions).to(device=device).view(1, env.action_space.shape[1])

        ###############
        # State OOD: score the model's own action against its own state (as opposed to the
        # teleop user_action scored in the not_blend=False branch below)
        state_ood_score = dp_loss(policy, obs_seq, policy_actions, num_samples=args_cli.num_samples)
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
            # get_loss_diffdagger.py - this branch only scores the live human action against
            # the trained policy (and, if --loss_action_range was given, against that reference).
            if teleop_interface is None:
                raise RuntimeError(
                    "run_dp_policy(..., not_blend=False) requires a teleop_interface "
                    "(e.g. Se3SpaceMouse) to read the live human action from."
                )
            # [7]: [x, y, z, rx, ry, rz, gripper] delta-pose command, already on `device`
            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html
            user_action = teleop_interface.advance()

            # diffdagger-style Nb-sample averaging: the action is fixed for this step, so
            # average multiple (noise, timestep) draws to reduce the loss's variance
            loss = dp_loss(policy, obs_seq, user_action, num_samples=args_cli.num_samples)
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


def main():
    """Run a trained robomimic diffusion policy in an Isaac Lab environment."""

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

    # reference loss distributions (see get_loss_diffdagger.py) to score live
    # losses against via CDF percentile
    loss_state_cdf = None
    if args_cli.loss_state_range is not None:
        loss_state_cdf = load_loss_reference(args_cli.loss_state_range)
        print(f"[INFO] Loaded reference state-loss CDF from {args_cli.loss_state_range} (range: {loss_state_cdf.min:.4f} - {loss_state_cdf.max:.4f})")

    loss_action_cdf = None
    if args_cli.loss_action_range is not None:
        loss_action_cdf = load_loss_reference(args_cli.loss_action_range)
        print(f"[INFO] Loaded reference action-loss CDF from {args_cli.loss_action_range} (range: {loss_action_cdf.min:.4f} - {loss_action_cdf.max:.4f})")

    # Wire up an input device
    # teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(sim_device=device))
    # print(teleop_interface)

    # not_blend = True if not args_cli.blend else False

    # # Run policy on live actions input from Isaac Lab
    # results = []
    # for trial in range(args_cli.num_rollouts):
    #     print(f"[INFO] Starting trial {trial}")

    #     policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.dp_checkpoint, device=device)

    #     terminated, traj = run_dp_policy(
    #         policy, env, success_term, args_cli.horizon, device,
    #         teleop_interface=teleop_interface,
    #         loss_state_cdf=loss_state_cdf,
    #         loss_action_cdf=loss_action_cdf,
    #         not_blend=not_blend,
    #     )

    #     results.append(terminated)
    #     print(f"[INFO] Trial {trial}: {terminated}\n")
    #     #print("traj, ", traj)

    # print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    # print(f"Success rate: {results.count(True) / len(results)}")
    # print(f"Trial Results: {results}\n")

    # Keep the sim window open for ~30 seconds
    start_time = time.time()
    while simulation_app.is_running() and (time.time() - start_time) < 30.0:
        print("[INFO]: Running...")
        simulation_app.update()

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
