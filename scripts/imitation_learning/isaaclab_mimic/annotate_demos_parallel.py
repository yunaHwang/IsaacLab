# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-annotate mimic demos across multiple parallel envs (no manual/keyboard mode).

This is a parallel-env variant of annotate_demos.py --auto: instead of replaying one
episode at a time through a single env, it round-robins all episodes across --num_envs
simultaneously-stepped envs, the same dispatch pattern replay_demos.py uses.

Per-episode output:
  - CLI prints one line per episode: demo name, success, and one True/False per subtask
    signal (True if that signal was ever raised during the episode).
  - The full per-timestep boolean arrays for every subtask signal are written to a
    companion .txt file next to --output_file (not printed to the CLI).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Auto-annotate mimic demos across parallel envs.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument("--num_envs", type=int, default=25, help="Number of parallel envs to annotate with.")
parser.add_argument(
    "--annotate_subtask_start_signals",
    action="store_true",
    default=False,
    help="Also annotate start points of subtasks (in addition to termination signals).",
)
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class PreStepDatagenInfoRecorder(RecorderTerm):
    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)
        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskStartsObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskStartsObservationsRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsObservationsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def main():
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    txt_path = os.path.join(output_dir, f"{output_file_name}_subtask_signals.txt")

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)
    env_cfg.env_name = env_name

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")
    env_cfg.terminations = None

    env_cfg.recorders = MimicRecorderManagerCfg()
    if not args_cli.annotate_subtask_start_signals:
        env_cfg.recorders.record_pre_step_subtask_start_signals = None
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
        raise NotImplementedError(
            "The environment does not implement get_subtask_term_signals required for auto annotation."
        )

    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    env.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    episode_indices_to_replay = list(range(episode_count))

    env_episode_data_map: list[EpisodeData | None] = [None] * num_envs
    current_episode_indices: list[int | None] = [None] * num_envs

    exported_count = 0
    processed_count = 0
    successful_count = 0
    failed_demo_names: list[str] = []

    with open(txt_path, "w") as txt_file:
        while True:
            actions = idle_action.clone()
            has_next_action = False
            for env_id in range(num_envs):
                env_next_action = None
                if env_episode_data_map[env_id] is not None:
                    env_next_action = env_episode_data_map[env_id].get_next_action()

                if env_next_action is None:
                    # the episode running on this env_id (if any) just finished -- evaluate it
                    if current_episode_indices[env_id] is not None:
                        episode_name = episode_names[current_episode_indices[env_id]]
                        success = bool(success_term.func(env, **success_term.params)[env_id])

                        annotated_episode = env.recorder_manager.get_episode(env_id)
                        subtask_dict = (
                            annotated_episode.data.get("obs", {}).get("datagen_info", {}).get("subtask_term_signals", {})
                        )
                        subtask_summary = {}
                        all_subtasks_ok = True
                        for signal_name, signal_flags in subtask_dict.items():
                            flags_t = torch.as_tensor(signal_flags)
                            any_true = bool(torch.any(flags_t))
                            subtask_summary[signal_name] = any_true
                            all_subtasks_ok = all_subtasks_ok and any_true
                            txt_file.write(f"{episode_name} | {signal_name}: {flags_t.tolist()}\n")
                        txt_file.flush()

                        processed_count += 1
                        subtask_str = "  ".join(f"{k}={v}" for k, v in subtask_summary.items())
                        ok = success and all_subtasks_ok
                        print(
                            f"[{processed_count:4}/{episode_count}] {episode_name:<12} success={success!s:<6} "
                            f"{subtask_str}  -> {'EXPORTED' if ok else 'SKIPPED'}"
                        )

                        if ok:
                            env.recorder_manager.set_success_to_episodes(
                                [env_id], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                            )
                            demo_num = episode_name.split("_")[-1]
                            demo_id = int(demo_num) if demo_num.isdigit() else None
                            env.recorder_manager.export_episodes(
                                [env_id], demo_ids=[demo_id] if demo_id is not None else None
                            )
                            exported_count += 1
                            successful_count += 1
                        else:
                            failed_demo_names.append(episode_name)
                        current_episode_indices[env_id] = None

                    # assign the next unprocessed episode to this env_id
                    next_episode_index = episode_indices_to_replay.pop(0) if episode_indices_to_replay else None
                    if next_episode_index is not None:
                        current_episode_indices[env_id] = next_episode_index
                        episode_data = dataset_file_handler.load_episode(episode_names[next_episode_index], env.device)
                        env_episode_data_map[env_id] = episode_data
                        env.recorder_manager.reset(env_ids=[env_id])
                        initial_state = episode_data.get_initial_state()
                        env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                        env_next_action = env_episode_data_map[env_id].get_next_action()
                        has_next_action = True
                    else:
                        env_episode_data_map[env_id] = None
                else:
                    has_next_action = True

                if env_next_action is not None:
                    actions[env_id] = env_next_action

            if not has_next_action:
                break
            env.step(actions)

    print(f"\nProcessed {processed_count} episodes.")
    print(f"Exported (success + all subtasks detected): {exported_count}")
    print(f"Successful task completions (success term only): {successful_count}")
    if failed_demo_names:
        print(f"\nNot exported ({len(failed_demo_names)} total): {failed_demo_names}")
    print(f"\nFull per-timestep subtask signal arrays written to: {txt_path}")
    print(f"Annotated dataset written to: {args_cli.output_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
