# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministically annotate mimic demos WITHOUT re-simulating.

Unlike annotate_demos.py / annotate_demos_parallel.py, this script never calls
env.step().  Instead it forces the sim to each recorded per-timestep state via
scene.reset_to(state[t]) and then evaluates the datagen_info + subtask term
signals + success term on that exact recorded state.

Because there is no PhysX integration, the trajectory cannot drift: the
annotation reflects exactly the states that generate_dataset.py recorded (and
proved successful).  num_envs therefore has no effect on the result; it only
parallelizes throughput.

Output (per demo that passes the export filter):
  - full verbatim copy of the source demo group (actions, states, all obs
    INCLUDING the original table_cam / wrist_cam frames)
  - obs/datagen_info/{object_pose,eef_pose,target_eef_pose,subtask_term_signals}
  - demo attr success

A companion _subtask_signals.txt gets the full per-timestep boolean arrays.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministically annotate mimic demos via state forcing (no env.step).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--input_file", type=str, default="./datasets/dataset.hdf5", help="Dataset to annotate.")
parser.add_argument("--output_file", type=str, default="./datasets/dataset_annotated.hdf5", help="Annotated output.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (throughput only; result is identical).")
parser.add_argument(
    "--annotate_subtask_start_signals", action="store_true", default=False, help="Also annotate subtask start signals."
)
parser.add_argument(
    "--success_mode",
    type=str,
    default="final",
    choices=["final", "ever"],
    help="'final': success term must hold on the last recorded frame (matches annotate_demos*). "
    "'ever': success term held on any frame (matches generate_dataset.py).",
)
parser.add_argument(
    "--export_all", action="store_true", default=False, help="Export every demo (still annotated), not just passing ones."
)
parser.add_argument(
    "--export_selection",
    type=str,
    default="pass",
    choices=["pass", "final_fail"],
    help="'pass': export demos that pass (success AND all subtask signals). "
    "'final_fail': export ONLY demos that completed all subtasks but FAILED the final-frame "
    "success term (for visually inspecting whether the final-frame failure matters).",
)
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Enable Pinocchio.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.datasets import HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _to_np(x):
    return x.detach().to("cpu").numpy()


def main():
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    output_dir = os.path.dirname(args_cli.output_file)
    output_stem = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    txt_path = os.path.join(output_dir, f"{output_stem}_subtask_signals.txt")

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

    # we do not use the recorder manager (no env.step); disable it entirely
    env_cfg.recorders = {}

    # Drop the RGB cameras entirely so the run needs no rendering / --enable_cameras:
    #  - remove the CameraCfg sensors from the scene (creating them needs the RTX pipeline)
    #  - remove the matching image obs terms so observation_manager.compute() won't look for them
    # The original rendered frames are copied verbatim from the source file instead.
    for cam_name in ("table_cam", "wrist_cam", "table_high_cam", "front_cam"):
        if hasattr(env_cfg.scene, cam_name):
            setattr(env_cfg.scene, cam_name, None)
        if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, cam_name):
            setattr(env_cfg.observations.policy, cam_name, None)
    if hasattr(env_cfg, "image_obs_list"):
        env_cfg.image_obs_list = []

    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")
    if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
        raise NotImplementedError("The environment does not implement get_subtask_term_signals.")

    env.reset()
    device = env.device
    dt = env.physics_dt

    fin = h5py.File(args_cli.input_file, "r")
    fout = h5py.File(args_cli.output_file, "w")
    for ak, av in fin.attrs.items():
        fout.attrs[ak] = av
    dgrp_out = fout.create_group("data")
    for ak, av in fin["data"].attrs.items():
        dgrp_out.attrs[ak] = av

    processed = exported = succ_count = 0
    failed_names = []

    with open(txt_path, "w") as txt_file:
        # round-robin episodes across envs purely for throughput
        pending = list(range(episode_count))
        while pending:
            batch = [pending.pop(0) for _ in range(min(num_envs, len(pending)))]
            batch_eps = {env_id: dataset_file_handler.load_episode(episode_names[idx], device) for env_id, idx in enumerate(batch)}

            # per (env_id) accumulators
            lengths = {env_id: len(ep._data["actions"]) for env_id, ep in batch_eps.items()}
            max_len = max(lengths.values())
            sig_acc = {env_id: {} for env_id in batch_eps}
            start_acc = {env_id: {} for env_id in batch_eps}
            succ_acc = {env_id: [] for env_id in batch_eps}
            dg_obj = {env_id: {} for env_id in batch_eps}
            dg_eef = {env_id: {} for env_id in batch_eps}
            dg_tgt = {env_id: {} for env_id in batch_eps}

            for t in range(max_len):
                active = [env_id for env_id in batch_eps if t < lengths[env_id]]
                # force each active env to its recorded state[t]
                for env_id in active:
                    state_t = batch_eps[env_id].get_state(t)
                    env.scene.reset_to(state_t, torch.tensor([env_id], device=device), is_relative=True)
                env.sim.forward()
                env.scene.update(dt)
                env.obs_buf = env.observation_manager.compute(update_history=False)

                # datagen info (per active env)
                object_poses = env.get_object_poses()  # dict name -> (num_envs, 4, 4)
                eef_names = list(env.cfg.subtask_configs.keys())
                eef_poses = {name: env.get_robot_eef_pose(eef_name=name) for name in eef_names}
                actions_now = torch.zeros_like(env.action_manager.action)
                for env_id in active:
                    actions_now[env_id] = batch_eps[env_id].get_action(t)
                target_poses = env.action_to_target_eef_pose(actions_now)

                term = env.get_subtask_term_signals()  # dict name -> (num_envs,)
                start = env.get_subtask_start_signals() if args_cli.annotate_subtask_start_signals else {}
                succ_now = success_term.func(env, **success_term.params)

                for env_id in active:
                    for k, v in term.items():
                        sig_acc[env_id].setdefault(k, []).append(bool(v[env_id]))
                    for k, v in start.items():
                        start_acc[env_id].setdefault(k, []).append(bool(v[env_id]))
                    succ_acc[env_id].append(bool(succ_now[env_id]))
                    for k, v in object_poses.items():
                        dg_obj[env_id].setdefault(k, []).append(_to_np(v[env_id]))
                    for k, v in eef_poses.items():
                        dg_eef[env_id].setdefault(k, []).append(_to_np(v[env_id]))
                    for k, v in target_poses.items():
                        dg_tgt[env_id].setdefault(k, []).append(_to_np(v[env_id]))

            # evaluate + write each episode in the batch
            for env_id, idx in enumerate(batch):
                name = episode_names[idx]
                flags = succ_acc[env_id]
                final_success = flags[-1]
                ever_success = any(flags)
                success = ever_success if args_cli.success_mode == "ever" else final_success

                summary = {}
                all_ok = True
                for k, arr in sig_acc[env_id].items():
                    any_true = any(arr)
                    summary[k] = any_true
                    all_ok = all_ok and any_true
                    txt_file.write(f"{name} | {k}: {[int(x) for x in arr]}\n")
                txt_file.flush()

                processed += 1
                if ever_success:
                    succ_count += 1

                if args_cli.export_selection == "final_fail":
                    # demos that did every subtask but are NOT a success state on the last frame
                    do_export = all_ok and (not final_success)
                else:
                    do_export = success and all_ok
                do_export = do_export or args_cli.export_all

                sub_str = "  ".join(f"{k}={v}" for k, v in summary.items())
                print(
                    f"[{processed:4}/{episode_count}] {name:<12} success={success!s:<6} "
                    f"(final={final_success} ever={ever_success})  {sub_str}  "
                    f"-> {'EXPORTED' if do_export else 'SKIPPED'}"
                )

                if not do_export:
                    failed_names.append(name)
                    continue

                # verbatim copy of the source demo group, then attach datagen_info
                fin.copy(f"data/{name}", dgrp_out, name)
                gout = dgrp_out[name]
                gout.attrs["success"] = bool(success)
                gout.attrs["final_frame_success"] = bool(final_success)
                gout.attrs["ever_success"] = bool(ever_success)
                gout.attrs["all_subtasks_detected"] = bool(all_ok)
                if "num_samples" not in gout.attrs:
                    gout.attrs["num_samples"] = lengths[env_id]

                dg = gout["obs"].require_group("datagen_info")
                for group, payload in (
                    ("object_pose", dg_obj[env_id]),
                    ("eef_pose", dg_eef[env_id]),
                    ("target_eef_pose", dg_tgt[env_id]),
                ):
                    sub = dg.require_group(group)
                    for k, lst in payload.items():
                        sub.create_dataset(k, data=np.stack(lst).astype(np.float32))
                st = dg.require_group("subtask_term_signals")
                for k, arr in sig_acc[env_id].items():
                    st.create_dataset(k, data=np.array(arr, dtype=bool))
                if args_cli.annotate_subtask_start_signals:
                    ss = dg.require_group("subtask_start_signals")
                    for k, arr in start_acc[env_id].items():
                        ss.create_dataset(k, data=np.array(arr, dtype=bool))
                exported += 1

    dgrp_out.attrs["total"] = int(sum(dgrp_out[n].attrs["num_samples"] for n in dgrp_out))
    fout.close()
    fin.close()

    print(f"\nProcessed {processed} episodes.")
    print(f"Exported: {exported}")
    print(f"Ever-successful (success term true on some frame): {succ_count}")
    if failed_names:
        print(f"\nNot exported ({len(failed_names)}): {failed_names}")
    print(f"\nPer-timestep subtask signal arrays: {txt_path}")
    print(f"Annotated dataset: {args_cli.output_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
