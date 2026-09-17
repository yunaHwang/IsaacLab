# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll the POLICY out in Isaac Sim from a TRAINING episode's initial state, so you can watch
whether the model reproduces a demo it was fitted on.

This is the visual counterpart to replay_check.py / generalization_check.py. Those score the
model offline against recorded actions; this one puts the model in the loop in the actual sim,
starting from a recorded demo's exact initial cube/robot configuration, and reports whether it
still gets there.

HOW THIS DIFFERS FROM THE TWO EXISTING REPLAY SCRIPTS
  - scripts/tools/replay_demos.py and dp-fm-ood/replay_dataset_with_scoring.py both step the env
    with the RECORDED actions and (in the latter) score them. The robot always completes the
    task, because the demo's own actions are driving it.
  - This script steps the env with the MODEL'S actions. Nothing guarantees the robot completes
    anything - which is the entire point.

  --actions recorded reproduces the old behaviour and exists as a CONTROL: it proves the
  initial-state restore, env config and success term are sound before you read anything into a
  policy failure. Run it once per dataset. If the recorded actions do not succeed from
  reset_to(), the policy numbers below are meaningless and the bug is in the harness.

WHAT CANNOT BE REPRODUCED (read before interpreting a failure)
  env.reset_to() calls _reset_idx(), which re-runs the reset-mode event terms BEFORE overwriting
  the physics state. For this task those terms are randomize_light (1500-10000 intensity, 11 HDR
  skies), randomize_table_visual_material (12 textures) and randomize_robot_arm_visual_texture
  (13 textures) - see stack_ik_rel_visuomotor_env_cfg.py's EventCfg. Appearance is NOT part of
  the recorded state, so it is drawn fresh and will NOT match the appearance the demo was
  recorded under.

  So the default run gives you: GEOMETRY from the demo, APPEARANCE freshly randomized. That is
  a deliberate and useful split -
      succeeds here          -> the model handles novel appearance; the geometry is what it
                                memorized, and the closed-loop failure is elsewhere.
      fails here             -> appearance is the axis it cannot generalize over, since the
                                cube layout is one it was trained on.
  --disable_visual_dr strips those three terms so appearance is the fixed default instead of a
  fresh draw. It still is not the demo's appearance; it just holds appearance CONSTANT across
  episodes so runs are comparable to each other. Use it to separate "fails on every appearance"
  from "fails on unlucky draws".

  --episode_seed makes the appearance draw reproducible run-to-run (same seed -> same textures),
  which is what you want when comparing two checkpoints on the same episode.

TWO-PROCESS SPLIT, same as run_policy_fm.py + multitask_dit_server.py - Isaac Lab (this script,
Python 3.11) cannot import lerobot's MultiTaskDiTPolicy (3.12+) in the same process.

    Terminal 1 (LeRobot, Python 3.12):
        conda activate lerobot_0.6.1_multitask_dit
        cd dp-fm-ood
        python multitask_dit_server.py --checkpoint <ckpt>/pretrained_model --port 5555

    Terminal 2 (Isaac Lab, Python 3.11) - watch it, so no --headless:
        conda activate leisaac
        cd dp-fm-ood
        python replay_training_rollouts.py \
            --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
            --dataset_file ../datasets/visuomotor-based/0901_repro_gen_100_topped_up.hdf5 \
            --select_episodes 0 1 2 3 4 --enable_cameras

    --enable_cameras is REQUIRED even with a GUI: the policy conditions on table_cam/wrist_cam,
    and without it Isaac never instantiates the camera sensors, so the obs dict has no images.

Only one env is supported (the server's obs_history/policy.reset() are single-episode state -
same limitation run_policy_fm.py has).
"""

import argparse
import csv
import os
from multiprocessing.connection import Client

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Roll a trained MultiTaskDiTPolicy out in Isaac Lab from training episodes' "
    "initial states, and report whether it reproduces them."
)
parser.add_argument("--task", type=str, required=True, help="Name of the environment.")
parser.add_argument(
    "--dataset_file", type=str, required=True,
    help="RAW HDF5 the LeRobot training set was converted from (e.g. "
    "../datasets/visuomotor-based/0901_repro_gen_100_topped_up.hdf5). Needed for the recorded "
    "initial_state, which the LeRobot parquet does not carry.",
)
parser.add_argument(
    "--select_episodes", type=int, nargs="+", default=[],
    help="Episode indices to roll out. Empty = all of them (slow; start with a handful).",
)
parser.add_argument(
    "--actions", type=str, choices=["policy", "recorded"], default="policy",
    help="'policy' steps the env with the model's actions (the real test). 'recorded' steps it "
    "with the demo's own actions - the control run that validates the harness.",
)
parser.add_argument(
    "--disable_visual_dr", action="store_true",
    help="Strip the randomize_light / randomize_table_visual_material / "
    "randomize_robot_arm_visual_texture reset events, holding appearance constant across "
    "episodes. See the module docstring on what this does and does not isolate.",
)
parser.add_argument(
    "--episode_seed", type=int, default=None,
    help="Seed passed to reset_to() per episode, making the appearance draw reproducible "
    "run-to-run. Use when comparing two checkpoints on the same episodes.",
)
parser.add_argument(
    "--horizon_slack", type=float, default=1.5,
    help="Step budget per episode as a multiple of the demo's own length. The policy needs "
    "longer than the expert, so 1.0 would score timing rather than success.",
)
parser.add_argument(
    "--settle_steps", type=int, default=30,
    help="Extra steps to hold the demo's LAST action after its recorded actions run out, in "
    "--actions recorded mode. Required, not cosmetic: mdp.object_stacked ANDs the stacked test "
    "with a gripper-open test, and the Franka's binary gripper takes several sim steps to "
    "travel 0.0 -> 0.04. At the final recorded step the fingers are typically still mid-open "
    "and the cube still settling, so success reads False on a demo that in fact succeeded. "
    "scripts/tools/replay_demos.py avoids this by stepping idle_action after the episode ends "
    "and only then checking. Holding the last action (rather than idle/zero) keeps the "
    "gripper-open command asserted while things settle.",
)
parser.add_argument(
    "--state_mode", type=str, choices=["auto", "joint_pos", "eef_pose"], default="auto",
    help="Which observation.state layout to build: 'joint_pos' (9 dims, pre-0908 checkpoints) "
    "or 'eef_pose' (8 dims, 0908-onward EE-pose checkpoints). 'auto' (default) takes it from "
    "the dim the server reports for whichever checkpoint it loaded. See lerobot_obs.py.",
)
parser.add_argument("--out_csv", type=str, default="./outputs/training_rollouts.csv")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--task_instruction", type=str,
    default="grab red block and stack on top of blue block, then grab green block and stack on top of red block",
)
parser.add_argument("--mdit_server_host", type=str, default="127.0.0.1")
parser.add_argument("--mdit_server_port", type=int, default=5555)
parser.add_argument("--mdit_server_authkey", type=str, default="mdit-ipc")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.datasets import HDF5DatasetFileHandler

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# make_lerobot_obs lives in lerobot_obs.py, shared with run_policy_fm.py and
# replay_dataset_with_scoring.py - it used to be copied into all three (importing
# run_policy_fm.py re-runs its module-level AppLauncher, so the copy was deliberate), which
# meant the joint_pos -> eef_pose state-layout fix had to land three separate times.
from lerobot_obs import make_lerobot_obs, resolve_state_mode

DR_EVENT_TERMS = (
    "randomize_light",
    "randomize_table_visual_material",
    "randomize_robot_arm_visual_texture",
)


def rollout_episode(conn, env, success_term, episode_data, device, horizon, use_policy):
    """One episode: restore its initial state, then drive the env to `horizon` steps.

    Returns (succeeded, n_steps, mean_action_err, max_action_err). The action errors compare the
    model's action against the demo's recorded action at the same step index, and are None in
    --actions recorded mode. They are only meaningful early in the episode: once the policy
    diverges, step t of the rollout and step t of the demo are no longer the same situation, so
    read the mean as "did it start off in the right direction", not as a trajectory score.
    """
    initial_state = episode_data.get_initial_state()
    env.reset_to(
        initial_state,
        torch.tensor([0], device=env.device),
        seed=args_cli.episode_seed,
        is_relative=True,
    )

    conn.send({"cmd": "reset"})
    reset_response = conn.recv()
    if not reset_response.get("ok", False):
        raise RuntimeError(f"multitask_dit_server reset failed: {reset_response.get('error')}")

    # Which observation.state layout to build comes from the SERVER's loaded checkpoint (9 =
    # joint_pos, 8 = the 0908-onward EE pose), since this script has no --fm_checkpoint of its
    # own - the server owns the path. --state_mode overrides. See lerobot_obs.py.
    state_mode, why = resolve_state_mode(
        requested=args_cli.state_mode, state_dim=reset_response.get("state_dim")
    )
    if use_policy:
        print(f"[INFO] observation.state layout: {state_mode} ({why})")

    obs_dict = env.obs_buf
    errs = []

    # In recorded mode the run is the demo's own length plus a settle tail; in policy mode the
    # policy never "ends", so it just runs to the horizon.
    demo_len = episode_data.data["actions"].shape[0]
    budget = horizon if use_policy else demo_len + args_cli.settle_steps
    last_recorded = None

    for t in range(budget):
        recorded = episode_data.get_action(t)

        if use_policy:
            obs = make_lerobot_obs(obs_dict, args_cli.task_instruction, state_mode)
            conn.send({"cmd": "step", "obs": obs})
            response = conn.recv()
            if not response.get("ok", False):
                raise RuntimeError(f"multitask_dit_server error: {response.get('error')}")
            actions = torch.tensor(response["action"], device=device)
            if recorded is not None:
                errs.append(torch.linalg.norm(actions.reshape(-1) - recorded.reshape(-1)).item())
        elif recorded is not None:
            last_recorded = recorded.reshape(1, -1).to(device)
            actions = last_recorded
        else:
            # Settle tail. The arm action is a RELATIVE pose delta (IK-Rel), so repeating the
            # demo's last action would re-apply that delta every step and walk the arm right
            # through the stack it just built - demo_0's final action alone is dy=-0.14,
            # drz=-0.20, which over 30 steps is a large sweep. Zero the pose delta to hold
            # position, and carry only the gripper command forward so the fingers stay open
            # for object_stacked's gripper-open test while the cube settles.
            actions = torch.zeros_like(last_recorded)
            actions[0, 6] = last_recorded[0, 6]

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, t + 1, errs
        if terminated or truncated:
            return False, t + 1, errs

    return False, budget, errs


def main():
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")

    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.dataset_file)
    episode_names = list(handler.get_episode_names())
    # get_episode_names() yields HDF5 group order ("demo_10" before "demo_2"), so sort numerically
    # to make --select_episodes N mean demo_N.
    episode_names.sort(key=lambda s: int(s.split("_")[-1]))
    episode_count = len(episode_names)
    print(f"[INFO] {episode_count} demos in {args_cli.dataset_file}")

    indices = args_cli.select_episodes or list(range(episode_count))
    indices = [i for i in indices if i < episode_count]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    if args_cli.disable_visual_dr:
        for term in DR_EVENT_TERMS:
            if hasattr(env_cfg.events, term):
                setattr(env_cfg.events, term, None)
                print(f"[INFO] disabled reset event: {term}")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = torch.device(args_cli.device)

    conn = Client(
        (args_cli.mdit_server_host, args_cli.mdit_server_port),
        authkey=args_cli.mdit_server_authkey.encode(),
    )

    use_policy = args_cli.actions == "policy"
    print(f"[INFO] stepping the env with {'the MODEL' if use_policy else 'the RECORDED'} actions")

    # Both scripts/tools/replay_demos.py and replay_dataset_with_scoring.py do a full env.reset()
    # before their first reset_to(). Going straight to reset_to() leaves the sim un-warmed for
    # the first episode, which is its own source of a spurious first-episode failure.
    env.reset()

    rows = []
    successes = 0
    with torch.inference_mode():
        for n, idx in enumerate(indices):
            if not simulation_app.is_running() or simulation_app.is_exiting():
                break
            episode_data = handler.load_episode(episode_names[idx], env.device)
            demo_len = episode_data.data["actions"].shape[0]
            horizon = int(demo_len * args_cli.horizon_slack)
            budget = horizon if use_policy else demo_len + args_cli.settle_steps

            ok, steps, errs = rollout_episode(
                conn, env, success_term, episode_data, device, horizon, use_policy
            )
            successes += int(ok)
            mean_err = sum(errs) / len(errs) if errs else None
            first10 = sum(errs[:10]) / len(errs[:10]) if errs else None

            rows.append({
                "episode": episode_names[idx],
                "demo_len": demo_len,
                "steps": steps,
                "success": ok,
                "action_err_mean": f"{mean_err:.4f}" if mean_err is not None else "",
                "action_err_first10": f"{first10:.4f}" if first10 is not None else "",
            })
            err_txt = f"  action err: first10={first10:.4f} mean={mean_err:.4f}" if errs else ""
            print(
                f"[{n + 1}/{len(indices)}] {episode_names[idx]}: "
                f"{'SUCCESS' if ok else 'fail   '}  {steps}/{budget} steps "
                f"(demo was {demo_len}){err_txt}"
            )

    conn.send({"cmd": "close"})
    conn.recv()
    conn.close()

    os.makedirs(os.path.dirname(args_cli.out_csv), exist_ok=True)
    with open(args_cli.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"\n{'=' * 60}")
    print(f"  {args_cli.actions} actions, visual DR {'OFF' if args_cli.disable_visual_dr else 'ON'}")
    print(f"  {successes}/{n} succeeded from TRAINING initial states"
          f"{f' = {successes / n:.2f}' if n else ''}")
    if use_policy and n:
        print("  (control: re-run with --actions recorded; that must be ~1.00 or the harness is broken)")
    print(f"  wrote {args_cli.out_csv}")
    print(f"{'=' * 60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
