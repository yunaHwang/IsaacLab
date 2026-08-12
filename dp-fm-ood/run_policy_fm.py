# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained fm (flow-matching) policy in an Isaac Lab
environment. 

Args:
    task: Name of the environment.
    device: Torch device for the simulation.
    disable_fabric: If set, disable fabric and use USD I/O operations.
    fm_checkpoint: Path or hub id of a trained fm policy checkpoint.
    fm_backbone: Which fm policy implementation to run: 'gloves' or 'multitask_dit'.
    blending_mechanism: Which gloves action-blending mechanism to use once blending is
        wired up ('gloves_fpas', 'gloves_feeg', or 'gloves_ifae').
    blend: If set, enable action blending (not_blend=False); otherwise the policy's own
        actions are executed directly.
    num_samples: Number of (noise, t) draws to average the live fm diffdagger-style loss over.
    horizon: The step horizon of each rollout.
    num_rollouts: Number of rollouts to run.
    seed: Random seed.
"""

# TODO - wire in conformal prediction and smoothness code in

import argparse

from ood_signal_baseline_papers.density.get_nonconformity_gloves import generate_action_chunk, DEFAULT_STATE_KEYS

from ood_signal import *

# NOTE. to be run in separate isaaclab + lerobot environment
from lerobot_policy_gloves.modeling_gloves import DiTPolicy
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an fm (flow-matching) policy for Isaac Lab environment.")

parser.add_argument("--task", type=str, required=True, help="Name of the environment.")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)

parser.add_argument(
    "--fm_backbone", type=str, choices=["gloves", "multitask_dit"], default="multitask_dit",
    help="Which fm policy implementation to run"
)
parser.add_argument(
    "--blending_mechanism", type=str, choices=['gloves_fpas', 'gloves_feeg', 'gloves_ifae'],
    default=None,
    help="Which gloves action-blending mechanism to use once blending is wired up.",
)  # NOTE: add as I go
# FOR now: have not blend as a default, meaning that without this flag, then it will default to not blend
parser.add_argument("--blend", action='store_true')

parser.add_argument(
    "--num_samples", type=int, default=512,
    help="Number of (noise, t) draws to average the live fm diffdagger-style loss over "
    "(the fm analog of diffdagger's Nb in noise_estimation_loss_nb_infer). The live action is "
    "fixed for the duration of a step, so averaging multiple draws reduces the loss's variance.",
)

parser.add_argument(
    "--fm_checkpoint", type=str, required=True,
    help="Path or hub id of a trained fm policy checkpoint"
)

parser.add_argument("--horizon", type=int, default=500, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts to run.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab_tasks.utils import parse_env_cfg

from collections import deque


def run_gloves_policy(fm_policy, obs_history, device, teleop_interface, not_blend=True):
    """
    Perform action blending by 1) grabbing the action from the policy, 2) registering user action,
    3) function call for some metric (loss), 4) save metric value if necessary.

    Score the live human action against a GLOVES flow policy's density non-conformity
    score s(x) = ||z_hat(x)||^2 (paper Eq. 6; see compute_density_score/compute_z_hat in
    get_nonconformity_gloves.py).

    Args:
        fm_policy: The trained GLOVES DiTPolicy to score actions against (see
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
                "run_gloves_policy(..., not_blend=True) requires obs_history to "
                "generate the fm policy's own action chunk."
            )

        
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
        # State OOD: score the model's own action against its own state using density loss. 
        state_ood_score = gloves_density(fm_policy.dit_flow, obs_history, full_model_actions)
        msg = f"[state OOD] fm density for model's own action: {state_ood_score.item():.4f}"
        print(msg)

        # fm diffdagger-style noise-prediction loss 
        fm_loss = gloves_loss(
            fm_policy.dit_flow, obs_history, full_model_actions, num_samples=args_cli.num_samples
        )
        print(f"[state OOD] gloves loss for model's own action: {fm_loss.item():.4f}")

        return False, traj

    if teleop_interface is None:
        raise RuntimeError(
            "run_gloves_policy(..., not_blend=False) requires a teleop_interface "
            "(e.g. Se3SpaceMouse) to read the live human action from."
        )
    # [7]: [x, y, z, rx, ry, rz, gripper] delta-pose command, already on `device`
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html
    user_action = teleop_interface.advance()

    score = gloves_density(fm_policy.dit_flow, obs_history, user_action)
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


def run_multitask_ditpolicy(
    policy,
    obs_history,
    device,
    teleop_interface=None,
    not_blend=True,
):
    """
    Run the trained LeRobot MultiTaskDiTPolicy.

    Checkpoint training schema:
        observation.state              [9]
        observation.images.table_cam   [3, 200, 200]
        observation.images.wrist_cam   [3, 200, 200]

    n_obs_steps = 2
    n_action_steps = 24
    action_dim = 7
    """

    policy.eval()

    obs_history = list(obs_history)

    # ---------------------------------------------------------
    # Validate history
    # ---------------------------------------------------------

    if len(obs_history) != policy.config.n_obs_steps:
        raise ValueError(
            f"Expected {policy.config.n_obs_steps} observations, "
            f"got {len(obs_history)}."
        )

    required_keys = [
        "observation.state",
        "observation.images.table_cam",
        "observation.images.wrist_cam",
    ]

    for i, obs in enumerate(obs_history):
        for key in required_keys:
            if key not in obs:
                raise KeyError(
                    f"Observation {i} is missing {key!r}"
                )

    # ---------------------------------------------------------
    # Stack temporal observations
    # ---------------------------------------------------------

    # [2, 9] -> [1, 2, 9]
    state = torch.stack(
        [
            obs["observation.state"]
            for obs in obs_history
        ],
        dim=0,
    ).unsqueeze(0)

    # [2, 3, H, W] -> [1, 2, 3, H, W]
    table_cam = torch.stack(
        [
            obs["observation.images.table_cam"]
            for obs in obs_history
        ],
        dim=0,
    ).unsqueeze(0)

    wrist_cam = torch.stack(
        [
            obs["observation.images.wrist_cam"]
            for obs in obs_history
        ],
        dim=0,
    ).unsqueeze(0)

    # ---------------------------------------------------------
    # Construct LeRobot batch
    # ---------------------------------------------------------

    batch = {
        "observation.state": state.to(device),
        "observation.images.table_cam": table_cam.to(device),
        "observation.images.wrist_cam": wrist_cam.to(device),
    }

    # ---------------------------------------------------------
    # Native LeRobot preprocessing
    # ---------------------------------------------------------

    batch = policy._prepare_batch(batch)

    # ---------------------------------------------------------
    # Flow-matching inference
    # ---------------------------------------------------------

    with torch.no_grad():
        policy_actions = policy._generate_actions(batch)

    print(
        "[MultiTaskDiT] generated action chunk:",
        tuple(policy_actions.shape),
    )

    # Expected:
    #
    # [1, 24, 7]

    # ---------------------------------------------------------
    # LOSSES
    # ---------------------------------------------------------
    ###############
    # State OOD: score the model's own action against its own state (as opposed to the
    # teleop user_action scored in the not_blend=False branch below), mirroring
    # run_gloves_policy's state-OOD call.
    state_ood_loss = multitask_dit_loss(
        policy, obs_history, policy_actions, num_samples=args_cli.num_samples
    )
    print(f"[state OOD] MultiTaskDiT loss for model's own action: {state_ood_loss.item():.4f}")

    # ---------------------------------------------------------
    # Human action
    # ---------------------------------------------------------

    user_action = None

    if not not_blend:

        if teleop_interface is None:
            raise RuntimeError(
                "not_blend=False requires teleop_interface."
            )

        user_action = teleop_interface.advance()

        action_ood_loss = multitask_dit_loss(
            policy, obs_history, user_action, num_samples=args_cli.num_samples
        )
        print(f"[ACTION] MultiTaskDiT loss for current human action: {action_ood_loss.item():.4f}")


    # ---------------------------------------------------------
    # DENSITIES
    # ---------------------------------------------------------
    state_ood_density = multitask_dit_density(policy, obs_history, policy_actions)
    print(f"[state OOD] MultiTaskDiT density for model's own action: {state_ood_density.item():.4f}")

    return policy_actions, user_action


def make_lerobot_obs(obs_dict):
    """
    Convert the current Isaac Lab observation into the exact
    observation format used by the LeRobot MultiTaskDiT dataset.

    Expected Isaac Lab fields:
        obs_dict["joint_pos"]   -> [9]
        obs_dict["table_cam"]   -> [3, H, W]
        obs_dict["wrist_cam"]   -> [3, H, W]
    """

    obs = obs_dict["policy"] if "policy" in obs_dict else obs_dict

    state = obs["joint_pos"]

    table_cam = obs["table_cam"]
    wrist_cam = obs["wrist_cam"]

    # Remove batch dimension if Isaac Lab gives [1, ...]
    if state.ndim > 1 and state.shape[0] == 1:
        state = state.squeeze(0)

    if table_cam.ndim == 4 and table_cam.shape[0] == 1:
        table_cam = table_cam.squeeze(0)

    if wrist_cam.ndim == 4 and wrist_cam.shape[0] == 1:
        wrist_cam = wrist_cam.squeeze(0)

    return {
        "observation.state": state,
        "observation.images.table_cam": table_cam,
        "observation.images.wrist_cam": wrist_cam,
    }


def main():
    """Run a trained fm policy - GLOVES DiTPolicy or LeRobot MultiTaskDiTPolicy, chosen via
    --fm_backbone - in an Isaac Lab environment."""

    import isaaclab_mimic.envs

    print("Mimic envs:")
    for name in gym.registry.keys():
        if "Mimic" in name:
            print(name)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode
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
    device = torch.device(args_cli.device)

    # Wire up an input device
    teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(sim_device=device))
    print(teleop_interface)

    not_blend = True if not args_cli.blend else False

    # Run policy on live actions input from Isaac Lab
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        if args_cli.fm_backbone == "gloves":
            policy = DiTPolicy.from_pretrained(args_cli.fm_checkpoint)
            policy.to(device)
            policy.eval()

            # Build an initial n_obs_steps-long obs_history from the first observation,
            # mirroring run_dp_policy's own history init (repeat the first obs to fill the
            # window). TODO: once run_gloves_policy grows a real rollout loop (see its
            # docstring), obs_history should instead be threaded/updated step-to-step there.
            obs_dict, _ = env.reset()
            obs = copy.deepcopy(obs_dict["policy"])
            obs = {k: torch.squeeze(obs[k]) for k in DEFAULT_STATE_KEYS}
            obs_history = deque([obs] * policy.config.n_obs_steps, maxlen=policy.config.n_obs_steps)

            terminated, traj = run_gloves_policy(policy, obs_history, device, teleop_interface, not_blend)

        elif args_cli.fm_backbone == "multitask_dit":
            policy = MultiTaskDiTPolicy.from_pretrained(args_cli.fm_checkpoint)
            policy.to(device)
            policy.eval()

            obs_dict, _ = env.reset()
            obs = make_lerobot_obs(obs_dict)
            obs_history = deque([obs] * policy.config.n_obs_steps, maxlen=policy.config.n_obs_steps)

            policy_actions, user_action = run_multitask_ditpolicy(
                policy=policy,
                obs_history=obs_history,
                device=device,
                teleop_interface=teleop_interface,
                not_blend=not_blend,
            )
            # TODO: env.step through policy_actions/blend with user_action, mirroring
            # run_gloves_policy's not_blend=True branch - not wired up yet.
            terminated = False
            traj = dict(policy_actions=[policy_actions.tolist()], blended_actions=[], obs=[], next_obs=[])

        else:
            raise ValueError(f"Unknown --fm_backbone {args_cli.fm_backbone!r}")

        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
