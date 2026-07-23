# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--ood_detection_metric", type=str, choices=['diffdaggerloss', 'density'])
parser.add_argument("--dataset", type=str, default=None) # use when calculating loss range for rollouts and not user actions
parser.add_argument("--save_to_file", action="store_true")
parser.add_argument("--save_file_name", type=str, default=None)

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
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

from isaaclab_tasks.utils import parse_env_cfg

from collections import deque


def run_policy(policy, env, success_term, horizon, device, is_diffusion_policy = False, not_blend = True):
    from robomimic.algo.diffusion_policy import compute_diffusion_loss #TODO - this name might have to change based on the actual name once i have robomimic as my submodule

    import h5py

    import pandas as pd

    """Perform action blending by 1) grabbing the action from the policy, 2) registering user action, 
    3) function call for some metric (loss), 4) save metric value if necessary.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()


    # if is_diffusion_policy -> obs has to be dimension 2
    if is_diffusion_policy:
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

    if is_diffusion_policy:
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


        # if is_diffusion_policy -> obs has to be dimension 2
        if is_diffusion_policy:
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

        else:
            traj["obs"].append(obs)

            # Compute actions
            policy_actions = policy(obs)
        
        
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
            # TODO: register user_action
            if args_cli.metric == "diffdaggerloss":
                if args_cli.metric == "dataset":
                    # use existing dataset (ID/OOD's obs and actions)
                    all_demo_losses = dict()
                    
                    with h5py.File(args_cli.dataset, "r") as f:
                        data = f["data"]
                        for demo in data.keys():
                            actions = demo["actions"]
                            obs = demo["obs"]
                            states = demo["states"]
            
                            loss = compute_diffusion_loss(obs, states)
                            all_demo_losses[demo] = loss 
            
                    if args_cli.save_to_file:
                        all_demo_losses_df = pd.DataFrame(all_demo_losses)
                        all_demo_losses_df.to_csv(args_cli.save_file_name + ".csv", index = False)
                else:
                    # use user input from an input device
                    loss = compute_diffusion_loss(obs, user_actions)

            # TODO: define blended actions here

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
    """Run a trained policy from robomimic with Isaac Lab environment."""

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

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)

        #NOTE - yuna added
        is_diffusion_policy = True
        not_blend = True

        terminated, traj = run_policy(policy, env, success_term, args_cli.horizon, device, is_diffusion_policy, not_blend)
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
