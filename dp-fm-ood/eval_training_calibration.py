#!/usr/bin/env python3
"""Replay a LeRobot training dataset (no Isaac Sim, no live rollout) through a trained
MultiTaskDiTPolicy and score every frame's OWN recorded (state, action) pair with the same
reconstruction-loss / density metrics multitask_dit_server.py reports during a live rollout
(ood_signal.multitask_dit_loss / multitask_dit_density).

Why: run_policy_fm.py's state_ood_loss/state_ood_density numbers are only meaningful relative
to what "in-distribution" looks like. This script builds that baseline by scoring the actual
expert demonstrations the checkpoint was trained on - if those score high or noisy, the live
metrics aren't trustworthy as an OOD signal; if they score low and tight, that's a real
calibration set (see ood_signal.cf_prediction_score_ood, which expects exactly this kind of
array of in-distribution scores).

Runs entirely in the LeRobot conda env (Python 3.12+) - scoring recorded data needs no
Isaac Lab/Isaac Sim import, unlike multitask_dit_server.py which only avoids Isaac Lab because
run_policy_fm.py is the one that needs the live env.

Observation windowing mirrors multitask_dit_server.py's step handler exactly: one raw
single-timestep observation per frame -> preprocessor(obs) -> append to a maxlen=n_obs_steps
deque (obs_history) - the same, already-proven code path the server uses, not a hand-rolled
alternative. The ground-truth action CHUNK (not a tiled single action - see
configuration_multi_task_dit.py's action_delta_indices) comes from a second LeRobotDataset
view constructed with the policy's own action_delta_indices, so it is the exact [horizon,
action_dim] window training itself used at that frame, boundary padding included.

Usage:
    conda activate lerobot_0.6.1_multitask_dit
    cd dp-fm-ood
    python eval_training_calibration.py \\
        --checkpoint ../outputs/0824_more_data_30/multitask_dit_training/checkpoints/last/pretrained_model \\
        --repo_id local/ID-visuomotor-based \\
        --dataset_root ../lerobot_dataset_0824_more_data_30/ID-visuomotor-based \\
        --output_csv scores_30.csv

Cost: multitask_dit_loss forwards the full CLIP-vision + CLIP-text + DiT stack --num_samples
times per scored frame (default 8, vs. the server's 32 - this sweeps thousands of frames, not
one live step). For a quick smoke test before committing to a full sweep, add
`--stride 10 --episodes 0,1`.
"""

import argparse
import csv
import statistics
from collections import deque

import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ood_signal import multitask_dit_density, multitask_dit_loss


def build_raw_obs(item):
    return {
        "observation.state": item["observation.state"],
        "observation.images.table_cam": item["observation.images.table_cam"],
        "observation.images.wrist_cam": item["observation.images.wrist_cam"],
        "task": item["task"],
    }


def summarize(name, values):
    if not values:
        print(f"  {name}: no successful scores")
        return
    print(
        f"  {name}: n={len(values)} mean={statistics.mean(values):.4f} "
        f"std={statistics.pstdev(values):.4f} min={min(values):.4f} max={max(values):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path or hub id of a trained MultiTaskDiTPolicy checkpoint.")
    parser.add_argument("--repo_id", type=str, default="local/ID-visuomotor-based")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of the LeRobot dataset to replay, e.g. ../lerobot_dataset_0824_more_data_30/ID-visuomotor-based")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num_samples", type=int, default=8,
        help="Nb draws averaged into each reconstruction-loss score (ood_signal.multitask_dit_loss). "
        "Lower than multitask_dit_server.py's default of 32 since this sweeps thousands of frames.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Score every Nth frame within each episode.")
    parser.add_argument("--episodes", type=str, default=None, help="Comma-separated episode indices to score (default: all episodes).")
    parser.add_argument(
        "--include_model_action", action="store_true",
        help="Also generate the policy's OWN action chunk at each scored frame (select_action's "
        "full num_integration_steps Euler generation, no queue reuse like a live rollout gets) and "
        "score its loss/density too - a self-consistency check, separate from 'is the recorded "
        "expert action in-distribution'. Expensive; forces --stride 1 so select_action's internal "
        "queue stays continuous across frames like a real rollout.",
    )
    parser.add_argument("--output_csv", type=str, default="training_calibration_scores.csv")
    args = parser.parse_args()

    if args.include_model_action and args.stride != 1:
        raise ValueError("--include_model_action requires --stride 1 (select_action's internal queue assumes consecutive frames).")

    device = torch.device(args.device)
    policy = MultiTaskDiTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    last_chunk = {"value": None}
    if args.include_model_action:
        # Same capture trick as multitask_dit_server.py: grabs the full generated chunk
        # select_action() already produces internally, instead of a second, redundant
        # generate_action_chunk() call.
        _orig_conditional_sample = policy.objective.conditional_sample

        def _conditional_sample_and_capture(*a, **kw):
            result = _orig_conditional_sample(*a, **kw)
            last_chunk["value"] = result.detach()
            return result

        policy.objective.conditional_sample = _conditional_sample_and_capture

    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    # Plain, un-windowed dataset: one raw single-timestep observation per item, the same shape
    # a live env.step() obs has. The n_obs_steps window is built by hand below (obs_history),
    # not requested from LeRobotDataset directly, to stay on the exact code path
    # multitask_dit_server.py already uses for scoring.
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)

    # Ground-truth action CHUNK only - the real [horizon, action_dim] window training itself
    # used at each frame (policy.config.action_delta_indices), not a tiled single action.
    action_delta_timestamps = {"action": [i / dataset.meta.fps for i in policy.config.action_delta_indices]}
    action_dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=action_delta_timestamps)

    wanted_episodes = {int(x) for x in args.episodes.split(",")} if args.episodes else None

    rows = []
    obs_history = deque(maxlen=policy.config.n_obs_steps)
    current_episode = None

    n = len(dataset)
    indices = range(0, n, args.stride)
    for idx in tqdm(indices, desc="Scoring frames"):
        item = dataset[idx]
        ep_idx = item["episode_index"].item()
        frame_idx = item["frame_index"].item()

        if wanted_episodes is not None and ep_idx not in wanted_episodes:
            continue

        if ep_idx != current_episode:
            policy.reset()
            obs_history.clear()
            current_episode = ep_idx

        raw_obs = build_raw_obs(item)
        row = {"episode_index": ep_idx, "frame_index": frame_idx, "task": raw_obs["task"]}

        with torch.no_grad():
            batch = preprocessor(raw_obs)

            norm_obs_step = {
                key: batch[key].squeeze(0)
                for key in (
                    "observation.state",
                    "observation.images.table_cam",
                    "observation.images.wrist_cam",
                )
            }
            norm_obs_step[OBS_LANGUAGE_TOKENS] = batch[OBS_LANGUAGE_TOKENS]
            norm_obs_step[OBS_LANGUAGE_ATTENTION_MASK] = batch[OBS_LANGUAGE_ATTENTION_MASK]
            obs_history.append(norm_obs_step)
            while len(obs_history) < policy.config.n_obs_steps:
                obs_history.append(norm_obs_step)

            gt_chunk = action_dataset[idx]["action"].unsqueeze(0)  # [1, horizon, action_dim]

            try:
                gt_loss = multitask_dit_loss(policy, obs_history, gt_chunk, num_samples=args.num_samples)
                row["gt_action_loss"] = gt_loss.item()
            except Exception as e:
                row["gt_action_loss_error"] = str(e)

            try:
                gt_density, z_hat = multitask_dit_density(
                    policy, obs_history, gt_chunk, tile_single_action=False, return_z_hat=True
                )
                row["gt_action_density"] = gt_density.item()
                z_flat = z_hat.flatten(start_dim=1)
                row["gt_zhat_mean"] = z_flat.mean().item()
                row["gt_zhat_std"] = z_flat.std().item()
            except Exception as e:
                row["gt_action_density_error"] = str(e)

            if args.include_model_action:
                try:
                    raw_action = policy.select_action(batch)
                    model_chunk = last_chunk["value"]
                    row["model_action_loss"] = multitask_dit_loss(
                        policy, obs_history, raw_action, num_samples=args.num_samples
                    ).item()
                    row["model_action_density"] = multitask_dit_density(
                        policy, obs_history, model_chunk, tile_single_action=False
                    ).item()
                except Exception as e:
                    row["model_action_error"] = str(e)

        rows.append(row)
        torch.cuda.empty_cache()

    fieldnames = sorted({key for row in rows for key in row})
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} scored frames to {args.output_csv}")

    print("\nSummary (in-distribution calibration scores - the training data scored against its own policy):")
    summarize("gt_action_loss", [r["gt_action_loss"] for r in rows if "gt_action_loss" in r])
    summarize("gt_action_density", [r["gt_action_density"] for r in rows if "gt_action_density" in r])
    if args.include_model_action:
        summarize("model_action_loss", [r["model_action_loss"] for r in rows if "model_action_loss" in r])
        summarize("model_action_density", [r["model_action_density"] for r in rows if "model_action_density" in r])

    n_errors = sum(1 for r in rows if "gt_action_loss_error" in r or "gt_action_density_error" in r)
    if n_errors:
        print(f"\n{n_errors} frame(s) had scoring errors - see the CSV's *_error columns.")


if __name__ == "__main__":
    main()
