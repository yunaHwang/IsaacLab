#!/usr/bin/env python3
"""Score a trained checkpoint against the TRAINING data it was fitted on. No Isaac, no rollout.

If a policy cannot reproduce actions it was trained on, nothing downstream can work, so this is
the cheapest health check available -- about a minute per checkpoint versus a full sim rollout.

It reports R^2 per action dimension. The aggregate number is misleading on this task: the
gripper dim carries ~5x the normalized variance of all five arm dims combined (a consequence of
MIN_MAX normalization on a binary channel), so a policy that only learns open/close still scores
a healthy-looking total. Watch the ARM row.

Normalization note: the pre/post processors are loaded from the CHECKPOINT (`pretrained_path`),
so each model is scored with exactly the constants it trained with. In particular lerobot
substitutes ImageNet mean/std for camera stats when `use_imagenet_stats=True` (the default,
datasets/factory.py), which is what the checkpoints carry -- the dataset's own meta/stats.json
image values are never used.

Usage:
    conda activate lerobot_0.6.1_multitask_dit
    python replay_check.py <checkpoint_dir> --dataset 100 [--label "pre-fix"] [--n 64]

<checkpoint_dir> is a .../checkpoints/<step>/pretrained_model directory.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import lerobot.policies.multi_task_dit.configuration_multi_task_dit  # noqa: F401  (registers the subclass)
import lerobot.policies.diffusion.configuration_diffusion  # noqa: F401  (ditto, for --policy.type diffusion)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from patch_tokens import assert_fully_loaded, maybe_install_patch_tokens  # noqa: E402


BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
REPO_ID = "ID/visuomotor-based"
# Action layout is 6-DOF end-effector delta + gripper (the task is IK-Rel, i.e. relative
# inverse kinematics), so the arm is dims 0-5 and the gripper is dim 6.
#
# This was slice(0, 5) until 2026-09-08, which silently EXCLUDED dim 5 -- the third rotation
# component -- from every "arm R^2" ever reported, while also not counting it as gripper. It was
# the best-scoring arm dim at the time it was found (held-out per-dim at 090000:
# [-0.04, -0.067, 0.242, -0.173, 0.078, 0.353, 0.906], where 0.353 is dim 5). The old value looks
# like a leftover from a 5-joint-arm assumption (SO-101 style), which does not apply to a 7-DOF
# Franka commanded in EE space. Numbers from before this change are not comparable.
ARM = slice(0, 6)


def load(ckpt: str, n: str, device: str):
    root = str(BASE / f"lerobot_dataset_0901_topped_up_{n}" / "ID-visuomotor-based")
    # MUST precede make_policy(): it patches the encoder CLASS. Without it a PATCH_TOKENS
    # checkpoint loads onto the CLS encoder and silently drops patch_proj (see patch_tokens.py).
    note = maybe_install_patch_tokens(ckpt)
    if note:
        print(f"    [patch] {note}")
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device

    probe = LeRobotDataset(repo_id=REPO_ID, root=root)
    fps = probe.fps
    obs_d = [i / fps for i in range(1 - cfg.n_obs_steps, 1)]
    dt = {"observation.state": obs_d, "action": [i / fps for i in range(cfg.horizon)]}
    for ck in probe.meta.camera_keys:
        dt[ck] = obs_d
    del probe

    ds = LeRobotDataset(repo_id=REPO_ID, root=root, delta_timestamps=dt)
    policy = make_policy(cfg, ds_meta=ds.meta).eval().to(device)
    assert_fully_loaded(policy, ckpt)
    pre, _ = make_pre_post_processors(cfg, pretrained_path=ckpt, dataset_stats=ds.meta.stats)
    return cfg, ds, policy, pre


def score(ckpt: str, n: str, n_samples: int, device: str, seed: int = 0) -> dict:
    cfg, ds, policy, pre = load(ckpt, n, device)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(ds.num_frames, n_samples, replace=False)

    P, G = [], []
    for i in idxs:
        item = ds[int(i)]
        batch = {}
        for k, v in item.items():
            batch[k] = v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else (
                [v] if isinstance(v, str) else v)
        for ck in ds.meta.camera_keys:
            if ck in batch and batch[ck].dtype == torch.uint8:
                batch[ck] = batch[ck].float() / 255.0
        pb = pre(dict(batch))
        obs_only = {k: v for k, v in pb.items() if k != "action"}
        with torch.no_grad():
            if hasattr(policy, "_generate_actions"):
                # MultiTaskDiT: predict_action_chunk unconditionally stacks from its internal
                # queues, which are empty here, so go one level down to _generate_actions.
                chunk = policy._generate_actions(policy._prepare_batch(obs_only))
            else:
                # DiffusionPolicy: predict_action_chunk has an offline branch that takes a
                # dataloader batch directly when the queues are empty.
                policy.reset()
                chunk = policy.predict_action_chunk(obs_only)
        m = min(chunk.shape[1], pb["action"].shape[1])
        P.append(chunk[0, :m].float().cpu().numpy())
        G.append(pb["action"][0, :m].float().cpu().numpy())

    P, G = np.stack(P), np.stack(G)

    def r2(p, g):
        return 1 - ((p - g) ** 2).mean() / ((g - g.mean(axis=(0, 1), keepdims=True)) ** 2).mean()

    per_dim = [float(r2(P[..., d:d + 1], G[..., d:d + 1])) for d in range(P.shape[-1])]
    return {
        "all": float(r2(P, G)),
        "arm": float(r2(P[..., ARM], G[..., ARM])),
        "gripper": per_dim[-1],
        "per_dim": per_dim,
        "mse": float(((P - G) ** 2).mean()),
        "n": P.shape[0],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--dataset", default="100", choices=["50", "100", "300"])
    p.add_argument("--label", default=None)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    r = score(args.checkpoint, args.dataset, args.n, args.device)
    label = args.label or Path(args.checkpoint).parents[2].name
    print(f"\n=== {label} ===")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  dataset    : N={args.dataset}   samples={r['n']}")
    print(f"  R2 all dims: {r['all']:.3f}")
    print(f"  R2 ARM 0-4 : {r['arm']:.3f}      <-- the number that matters")
    print(f"  R2 gripper : {r['gripper']:.3f}")
    print(f"  MSE        : {r['mse']:.5f}")
    print(f"  per-dim R2 : {[round(x, 3) for x in r['per_dim']]}")


if __name__ == "__main__":
    main()
