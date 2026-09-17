# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded HDF5 dataset in Isaac Lab (so you can SEE the robot move, same as
scripts/tools/replay_demos.py) while also scoring each recorded (state, action) pair against
a trained MultiTaskDiTPolicy via multitask_dit_server.py - the "action OOD" case from
ood_signal.py's module docstring (is this externally-supplied, real expert action consistent
with what the model learned), computed live as you watch the replay rather than offline like
eval_training_calibration.py.

This is NOT a variant of eval_training_calibration.py - that script deliberately has no Isaac
Sim dependency, replaying the already-converted LeRobot dataset directly. This script replays
the RAW HDF5 (pre-LeRobot-conversion) through the actual Isaac Lab env, exactly like
replay_demos.py, so you get the visual sim alongside the same reconstruction-loss/density
numbers - useful for eyeballing which specific moments in a demo (a fumbled grasp, a stall) the
metrics flag as high-loss/high-density.

Two-process split, same reason as run_policy_fm.py + multitask_dit_server.py: Isaac Lab
(this script, Python 3.11) can't import lerobot's MultiTaskDiTPolicy (Python 3.12+) in the same
process. Terminal 2 must already be running multitask_dit_server.py with your checkpoint loaded
before you start this script.

    Terminal 1 (LeRobot, Python 3.12):
        conda activate lerobot_0.6.1_multitask_dit
        cd dp-fm-ood
        python multitask_dit_server.py --checkpoint /path/to/pretrained_model --port 5555

    Terminal 2 (Isaac Lab, Python 3.11):
        conda activate leisaac  (or yuna_env - whichever run_policy_fm.py uses)
        cd dp-fm-ood
        python replay_dataset_with_scoring.py \
            --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
            --dataset_file ../datasets/visuomotor-based/0824-gen-ID-small-30.hdf5 \
            --task_instruction "stack cubes" \
            --mdit_server_port 5555

Only --num_envs 1 is supported (the server's obs_history/policy.reset() are single-episode
state, same limitation run_policy_fm.py already has - see multitask_dit_server.py's docstring).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay an Isaac Lab HDF5 dataset while live-scoring each recorded action "
    "against a trained MultiTaskDiTPolicy via multitask_dit_server.py."
)
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_success_rate",
    action="store_true",
    default=False,
    help="Validate the replay success rate using the task environment termination criteria",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

parser.add_argument(
    "--mdit_server_host", type=str, default="127.0.0.1",
    help="Host multitask_dit_server.py is listening on.",
)
parser.add_argument(
    "--mdit_server_port", type=int, default=5555,
    help="Port multitask_dit_server.py is listening on.",
)
parser.add_argument(
    "--mdit_server_authkey", type=str, default="mdit-ipc",
    help="Shared secret for the multiprocessing.connection handshake - must match "
    "multitask_dit_server.py's --authkey.",
)
parser.add_argument(
    "--task_instruction", type=str, default="stack cubes",
    help="Language task label the MultiTaskDiT checkpoint was trained on (see the training "
    "dataset's meta/tasks.parquet) - required for the server's preprocessor to tokenize "
    "task-conditioning.",
)
parser.add_argument(
    "--state_mode", type=str, choices=["auto", "joint_pos", "eef_pose"], default="auto",
    help="Which observation.state layout to build: 'joint_pos' (9 dims, pre-0908 checkpoints) "
    "or 'eef_pose' (8 dims, 0908-onward EE-pose checkpoints). 'auto' (default) takes it from "
    "the dim the server reports for whichever checkpoint it loaded. See lerobot_obs.py.",
)
parser.add_argument("--output_csv", type=str, default=None, help="If set, write one row per (episode, step) with action_loss/action_density to this CSV path.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import csv
import os
from multiprocessing.connection import Client

import gymnasium as gym
import torch

from lerobot_obs import make_lerobot_obs, resolve_state_mode

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


# make_lerobot_obs lives in lerobot_obs.py, shared with run_policy_fm.py and
# replay_training_rollouts.py - it used to be copied into all three (importing run_policy_fm.py
# re-runs its module-level AppLauncher, so the copy was deliberate), which meant the
# joint_pos -> eef_pose state-layout fix had to land three separate times.


def main():
    """Replay episodes loaded from a file, live-scoring each recorded action against the
    checkpoint multitask_dit_server.py has loaded."""
    global is_paused

    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = 1  # single-episode server state (obs_history/policy.reset()) - see module docstring

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    success_term = None
    if args_cli.validate_success_rate:
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print(
                "No success termination term was found in the environment."
                " Will not be able to mark recorded demos as successful."
            )

    # Dictionary mode so make_lerobot_obs can pull "joint_pos"/"table_cam"/"wrist_cam" by key,
    # same as run_policy_fm.py.
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = {}
    env_cfg.terminations = {}

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # One persistent connection to multitask_dit_server.py - each new episode below sends its
    # own {"cmd": "reset"} to clear the server-side obs_history, same pattern as
    # run_policy_fm.py's run_multitask_ditpolicy.
    conn = Client(
        (args_cli.mdit_server_host, args_cli.mdit_server_port),
        authkey=args_cli.mdit_server_authkey.encode(),
    )

    obs_dict, _ = env.reset()
    teleop_interface.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    recorded_episode_count = 0

    current_episode_indices = [None] * num_envs
    failed_demo_ids = []
    csv_rows = []

    # Provisional until the first {"cmd": "reset"} comes back with the server's authoritative
    # observation.state dim (see the reset handler in the loop below); resolved here too so
    # the name is always bound, and so an explicit --state_mode is honoured either way.
    state_mode, _why = resolve_state_mode(requested=args_cli.state_mode)

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            episode_ended = [False] * num_envs
            episode_step = [0] * num_envs
            while has_next_action:
                actions = idle_action
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        if (
                            (success_term is not None)
                            and (current_episode_indices[env_id]) is not None
                            and (not episode_ended[env_id])
                        ):
                            if bool(success_term.func(env, **success_term.params)[env_id]):
                                recorded_episode_count += 1
                                plural_trailing_s = "s" if recorded_episode_count > 1 else ""
                                print(
                                    f"Successfully replayed {recorded_episode_count} episode{plural_trailing_s} out"
                                    f" of {replayed_episode_count} demos."
                                )
                            else:
                                if current_episode_indices[env_id] is not None:
                                    failed_episode_name = episode_names[current_episode_indices[env_id]]
                                    if failed_episode_name not in failed_demo_ids:
                                        failed_demo_ids.append(failed_episode_name)
                            episode_ended[env_id] = True

                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                episode_ended[env_id] = False
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            replayed_episode_count += 1
                            current_episode_indices[env_id] = next_episode_index
                            episode_step[env_id] = 0
                            print(
                                f"{replayed_episode_count:4}: Loading {episode_names[next_episode_index]} episode"
                                f" to env_{env_id}"
                            )
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)

                            # New episode -> clear the server's obs_history/policy.reset(),
                            # same as run_policy_fm.py does per trial.
                            conn.send({"cmd": "reset"})
                            reset_response = conn.recv()
                            if not reset_response.get("ok", False):
                                raise RuntimeError(f"multitask_dit_server reset failed: {reset_response.get('error')}")
                            # Layout of observation.state comes from the SERVER's loaded
                            # checkpoint (9 = joint_pos, 8 = the 0908-onward EE pose); this
                            # script has no checkpoint path of its own. See lerobot_obs.py.
                            state_mode, why = resolve_state_mode(
                                requested=args_cli.state_mode,
                                state_dim=reset_response.get("state_dim"),
                            )
                            print(f"[INFO] observation.state layout: {state_mode} ({why})")

                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue

                # Score BEFORE stepping - obs_dict here is the state the recorded action was
                # actually taken from (matches eval_training_calibration.py's teacher-forced
                # convention: real state -> real action, not the state the action leads to).
                for env_id in range(num_envs):
                    if not has_next_action and episode_ended[env_id]:
                        continue
                    raw_obs = make_lerobot_obs(obs_dict, args_cli.task_instruction, state_mode)
                    conn.send({"cmd": "score_action", "obs": raw_obs, "action": actions[env_id].tolist()})
                    score_response = conn.recv()
                    if not score_response.get("ok", False):
                        print(f"[replay scoring] server error: {score_response.get('error')}")
                        continue

                    loss = score_response.get("action_loss")
                    density = score_response.get("action_density")
                    print(
                        f"[replay scoring] episode={current_episode_indices[env_id]} "
                        f"step={episode_step[env_id]:4} action_loss={loss} action_density={density}"
                    )
                    if args_cli.output_csv:
                        csv_rows.append({
                            "episode_index": current_episode_indices[env_id],
                            "step": episode_step[env_id],
                            "action_loss": loss,
                            "action_density": density,
                        })
                    episode_step[env_id] += 1

                obs_dict, _, _, _, _ = env.step(actions)
            break

    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")

    if success_term is not None:
        print(f"Successfully replayed: {recorded_episode_count}/{replayed_episode_count}")
        if failed_demo_ids:
            print(f"\nFailed demo IDs ({len(failed_demo_ids)} total):")
            print(f"  {sorted(failed_demo_ids, key=lambda name: int(name.split('_')[-1]))}")

    if args_cli.output_csv and csv_rows:
        fieldnames = sorted({key for row in csv_rows for key in row})
        with open(args_cli.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote {len(csv_rows)} scored steps to {args_cli.output_csv}")

    conn.send({"cmd": "close"})
    conn.recv()
    conn.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
