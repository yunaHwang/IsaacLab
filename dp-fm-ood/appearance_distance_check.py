#!/usr/bin/env python
"""Is the held-out failure an APPEARANCE-generalization failure?

THE QUESTION THIS ANSWERS
    Every run so far fits training data (arm R^2 0.966) and scores ~0 on held-out episodes, at
    every checkpoint from 45k to 180k steps. The 0904 investigation concluded generalization was
    never acquired, and named a suspect it never tested: the datagen/eval env randomizes dome
    light (1500-10000, 11 HDR skies), table texture (12) and arm texture (13) on EVERY reset --
    1716 appearance combinations against a few hundred demos.

    If that suspect is right, held-out R^2 should depend on how far a held-out episode's
    APPEARANCE is from the nearest training episode. This measures exactly that, offline, on
    checkpoints that already exist. No training run required.

HOW TO READ THE RESULT
    R^2 high for appearance-NEAR episodes, ~0 for FAR ones
        -> confirmed appearance-generalization failure. The fix is data coverage (more demos
           spanning the randomization) or less randomization. Dropout / weight decay / freezing
           / more steps CANNOT fix this, which retires the whole family of sweeps run so far.

    R^2 flat ~0 regardless of distance
        -> appearance is NOT the driver. Something more basic is wrong (action representation,
           or the task is not observable from these two views). That is a more important finding
           and redirects the work entirely.

    R^2 uniformly high
        -> the split or the scoring is wrong, not the model. Check that the checkpoint was
           actually trained without these episodes.

COST / CONTENTION WARNING
    Phase A (descriptors + distances) is CPU-only, a few seconds, and safe to run any time.
    Phase B (per-episode R^2) runs the policy: flow matching integrates 100 Euler steps per
    sample, so it is not free. It shares the GPU with any training in flight. Start with
    --phase a, and run Phase B with a modest --per-episode.

USAGE
    conda activate lerobot_0.6.1_multitask_dit
    # cheap, no GPU, answers "are held-out episodes even appearance-far?"
    python dp-fm-ood/appearance_distance_check.py --phase a --eval_split 0.1

    # full test on an existing checkpoint
    python dp-fm-ood/appearance_distance_check.py \
        outputs/run_0902_norm/n100_k6_s180k_lr3e4_crop_aug_sepenc/checkpoints/180000/pretrained_model \
        --train-n 100 --eval-n 300 --per-episode 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generalization_check import (  # noqa: E402
    ARM,
    BASE,
    REPO_ID,
    load,
    split_episodes,
    tail_split,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


# ---------------------------------------------------------------------------------------
# Phase A -- appearance descriptors
# ---------------------------------------------------------------------------------------
def episode_descriptors(n: str, episodes: list[int], frames_per_ep: int, seed: int = 0):
    """Per-episode appearance descriptor: per-channel mean and std of sampled frames.

    Deliberately cheap and interpretable rather than a learned embedding. Dome-light intensity,
    HDR sky and table/arm texture all move channel means and spreads, which is precisely the
    randomization under suspicion. A CLIP embedding would be more sensitive to semantics but
    needs the GPU and is harder to reason about when the answer is ambiguous.

    Returns (episode_ids, descriptor matrix [n_eps, n_cams*6]).
    """
    root = str(BASE / f"lerobot_dataset_0901_topped_up_{n}" / "ID-visuomotor-based")
    ds = LeRobotDataset(repo_id=REPO_ID, root=root, episodes=episodes)
    cams = ds.meta.camera_keys
    rng = np.random.default_rng(seed)

    # Group frame indices by episode using the item's own episode_index -- robust to whatever
    # order `episodes=` produces internally.
    by_ep: dict[int, list[int]] = {}
    probe_n = ds.num_frames
    stride = max(1, probe_n // (len(episodes) * frames_per_ep * 3))
    for i in range(0, probe_n, stride):
        ep = int(ds[i]["episode_index"])
        by_ep.setdefault(ep, []).append(i)

    ep_ids, rows = [], []
    for ep in sorted(by_ep):
        pool = by_ep[ep]
        pick = rng.choice(len(pool), min(frames_per_ep, len(pool)), replace=False)
        feats = []
        for cam in cams:
            vals = []
            for j in pick:
                img = ds[pool[int(j)]][cam].float()
                if img.dtype == torch.uint8 or img.max() > 1.5:
                    img = img / 255.0
                if img.ndim == 4:      # (T,C,H,W) -> last frame
                    img = img[-1]
                vals.append(
                    torch.cat([img.mean(dim=(-2, -1)), img.std(dim=(-2, -1))]).numpy()
                )
            feats.append(np.stack(vals).mean(axis=0))
        ep_ids.append(ep)
        rows.append(np.concatenate(feats))
    return np.array(ep_ids), np.stack(rows)


def trajectory_descriptors(n: str, episodes: list[int], n_knots: int = 8):
    """Per-episode descriptor from the JOINT TRAJECTORY, not the pixels.

    WHY THIS IS THE BETTER DESCRIPTOR HERE
        The appearance descriptor showed held-out episodes are NOT visual outliers, yet arm R^2
        is ~0 on them. What actually determines the correct trajectory in a stacking task is the
        INITIAL CUBE CONFIGURATION -- and that is not recorded in this dataset (observation.state
        is only the 9 Franka joints). The joint trajectory is, however, a direct consequence of
        where the cubes are: the arm goes where it must reach. So sampling the joint state at
        fixed fractions of the episode gives a proxy for task configuration.

    READ THE RESULT CAREFULLY -- there is a circularity to respect
        The model predicts actions, and this measures distance in something close to action
        space, so a correlation is not surprising on its own. The INFORMATIVE outcome is the
        flat one:
          R^2 decays with trajectory distance -> the model only reproduces trajectories close to
              ones it memorized. Consistent with retrieval, and it says more demos covering the
              configuration space should help.
          R^2 ~0 even for trajectory-NEAR episodes -> it cannot reproduce a familiar trajectory
              on a new episode at all. That is a deeper failure than coverage and more data will
              not fix it.

    Reads columns straight off hf_dataset, so NO video decoding -- far cheaper than the
    appearance descriptor.
    """
    root = str(BASE / f"lerobot_dataset_0901_topped_up_{n}" / "ID-visuomotor-based")
    ds = LeRobotDataset(repo_id=REPO_ID, root=root, episodes=episodes)
    hf = ds.hf_dataset.with_format(None)
    ep_col = np.asarray(hf["episode_index"])
    st_col = np.asarray(hf["observation.state"], dtype=np.float32)

    ep_ids, rows = [], []
    for ep in sorted(set(ep_col.tolist())):
        idx = np.flatnonzero(ep_col == ep)
        if idx.size == 0:
            continue
        knots = np.linspace(0, idx.size - 1, n_knots).round().astype(int)
        rows.append(st_col[idx[knots]].reshape(-1))
        ep_ids.append(int(ep))
    return np.array(ep_ids), np.stack(rows)


def nearest_distances(train_D: np.ndarray, eval_D: np.ndarray, exclude_self: bool = False):
    """Standardize on the TRAINING distribution, then nearest-neighbour distance per eval ep.

    `exclude_self` is required when eval_D IS train_D: otherwise every episode matches itself at
    distance 0 and the reference scale is silently all-zeros, which makes any held-out distance
    look enormous by comparison.
    """
    mu, sd = train_D.mean(axis=0), train_D.std(axis=0) + 1e-8
    a, b = (train_D - mu) / sd, (eval_D - mu) / sd
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=-1)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


# ---------------------------------------------------------------------------------------
# Phase B -- per-episode R^2
# ---------------------------------------------------------------------------------------
def per_episode_r2(ckpt, n, episodes, per_episode, device, seed=0, batch=8):
    """Arm R^2 for each episode separately. Batched -- the one-at-a-time loop in
    generalization_check.py is fine for 64 samples but not for 30 episodes x N."""
    cfg, ds, policy, pre = load(ckpt, n, device, episodes)
    rng = np.random.default_rng(seed)

    by_ep: dict[int, list[int]] = {}
    for i in range(ds.num_frames):
        by_ep.setdefault(int(ds.hf_dataset[i]["episode_index"]), []).append(i)

    out = {}
    for ep in sorted(by_ep):
        pool = by_ep[ep]
        pick = rng.choice(len(pool), min(per_episode, len(pool)), replace=False)
        P, G = [], []
        for s in range(0, len(pick), batch):
            items = [ds[pool[int(j)]] for j in pick[s : s + batch]]
            bt = {}
            for k in items[0]:
                v0 = items[0][k]
                if isinstance(v0, torch.Tensor):
                    bt[k] = torch.stack([it[k] for it in items]).to(device)
                elif isinstance(v0, str):
                    bt[k] = [it[k] for it in items]
            for ck in ds.meta.camera_keys:
                if ck in bt and bt[ck].dtype == torch.uint8:
                    bt[ck] = bt[ck].float() / 255.0
            pb = pre(dict(bt))
            obs = {k: v for k, v in pb.items() if k != "action"}
            with torch.no_grad():
                if hasattr(policy, "_generate_actions"):
                    chunk = policy._generate_actions(policy._prepare_batch(obs))
                else:
                    policy.reset()
                    chunk = policy.predict_action_chunk(obs)
            m = min(chunk.shape[1], pb["action"].shape[1])
            P.append(chunk[:, :m].float().cpu().numpy())
            G.append(pb["action"][:, :m].float().cpu().numpy())
        P, G = np.concatenate(P), np.concatenate(G)
        num = ((P[..., ARM] - G[..., ARM]) ** 2).mean()
        den = ((G[..., ARM] - G[..., ARM].mean(axis=(0, 1), keepdims=True)) ** 2).mean()
        out[ep] = float(1 - num / den)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default=None)
    p.add_argument("--train-n", default="100", choices=["50", "100"])
    p.add_argument("--eval-n", default="300", choices=["300"])
    p.add_argument("--eval_split", type=float, default=None,
                   help="use lerobot's tail split (for checkpoints trained WITH --dataset.eval_split)")
    p.add_argument("--phase", default="both", choices=["a", "b", "both"])
    p.add_argument("--descriptor", default="appearance", choices=["appearance", "trajectory"],
                   help="appearance = per-channel image mean/std (tests domain randomization); "
                        "trajectory = joint state at fixed episode fractions (proxy for the "
                        "initial cube configuration, which is NOT recorded in this dataset)")
    p.add_argument("--frames-per-ep", type=int, default=8)
    p.add_argument("--per-episode", type=int, default=12)
    p.add_argument("--max-episodes", type=int, default=24,
                   help="cap held-out episodes scored in phase b. Flow matching runs 100 Euler "
                        "steps per sample, so the hash split's 200 held-out episodes is hours. "
                        "Episodes are picked to SPAN the distance range, not at random, so the "
                        "near/far contrast survives the subsample.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.eval_split is not None:
        seen, unseen = tail_split(args.eval_n, args.eval_split)
        how = f"lerobot tail split, eval_split={args.eval_split}"
    else:
        seen, unseen = split_episodes(args.train_n, args.eval_n)
        how = f"hash match against N={args.train_n}"
    print(f"split: {how}  ->  {len(seen)} train, {len(unseen)} held-out")

    print(f"\nPhase A -- {args.descriptor} descriptors (CPU)")
    if args.descriptor == "trajectory":
        _, train_D = trajectory_descriptors(args.eval_n, seen)
        ev_ids, eval_D = trajectory_descriptors(args.eval_n, unseen)
    else:
        _, train_D = episode_descriptors(args.eval_n, seen, args.frames_per_ep)
        ev_ids, eval_D = episode_descriptors(args.eval_n, unseen, args.frames_per_ep)
    dist = nearest_distances(train_D, eval_D)
    # How far apart are TRAINING episodes from each other? Without this the held-out distances
    # have no scale: "far" only means something relative to the spread the model already saw.
    ref = nearest_distances(train_D, train_D, exclude_self=True)
    print(f"  held-out nearest-train distance : min {dist.min():.2f}  med {np.median(dist):.2f}  max {dist.max():.2f}")
    print(f"  train-to-train  (same metric)   : min {ref.min():.2f}  med {np.median(ref):.2f}  max {ref.max():.2f}")
    if np.median(dist) <= np.median(ref) * 1.2:
        print(f"  -> held-out episodes are NOT {args.descriptor}-far from training. If R^2 is")
        print(f"     still ~0, {args.descriptor} novelty is probably NOT the driver.")
    else:
        print(f"  -> held-out episodes sit outside the training {args.descriptor} spread; the")
        print("     hypothesis is live. Run phase b to see whether R^2 tracks this distance.")

    if args.phase == "a":
        return 0
    if args.checkpoint is None:
        print("\nphase b needs a checkpoint path"); return 1

    # Subsample to span the distance range rather than uniformly at random: the whole point is
    # the near-vs-far contrast, so keep both tails and thin the middle.
    score_eps = list(unseen)
    if len(score_eps) > args.max_episodes:
        order = np.argsort(dist)
        keep = np.linspace(0, len(order) - 1, args.max_episodes).round().astype(int)
        score_eps = [int(ev_ids[order[k]]) for k in keep]
        print(f"\n  scoring {len(score_eps)} of {len(unseen)} held-out episodes, "
              f"spanning distance {dist[order[keep[0]]]:.2f} to {dist[order[keep[-1]]]:.2f}")

    print(f"\nPhase B -- per-episode arm R^2 ({args.per_episode} samples/episode, GPU)")
    r2 = per_episode_r2(args.checkpoint, args.eval_n, score_eps, args.per_episode, args.device)

    common = [e for e in ev_ids if e in r2]
    d = np.array([dist[list(ev_ids).index(e)] for e in common])
    r = np.array([r2[e] for e in common])
    print(f"\n  {'episode':>8} {args.descriptor + ' dist':>16} {'arm R^2':>9}")
    for e, dd, rr in sorted(zip(common, d, r), key=lambda t: t[1]):
        print(f"  {e:>8} {dd:>16.2f} {rr:>9.3f}")

    order = np.argsort(d)
    half = len(order) // 2
    near, far = r[order[:half]], r[order[half:]]
    corr = float(np.corrcoef(d, r)[0, 1]) if len(d) > 2 else float("nan")
    print(f"\n  nearest half : arm R^2 {near.mean():+.3f}")
    print(f"  farthest half: arm R^2 {far.mean():+.3f}")
    print(f"  corr(distance, R^2) = {corr:+.3f}")
    print("\n  VERDICT")
    if near.mean() - far.mean() > 0.15 and corr < -0.3:
        print(f"   {args.descriptor.upper()}-DRIVEN. R^2 decays with {args.descriptor} distance.")
        print("   Points at data COVERAGE -- more demos spanning this axis should help, and")
        print("   regularization/steps will not.")
        print("   CHECK FOR LEVERAGE: a single far outlier can manufacture this correlation.")
        print("   Look at the per-episode table above before believing it.")
    elif abs(near.mean()) < 0.15 and abs(far.mean()) < 0.15:
        print(f"   NOT {args.descriptor}-driven. R^2 is ~0 even for the {args.descriptor}-NEAREST")
        print("   held-out episodes, so the model fails on episodes that are ordinary by this")
        print("   measure. Not a coverage problem along this axis; more data will not fix it.")
    else:
        print("   Mixed / inconclusive. Raise --per-episode and re-run before concluding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
