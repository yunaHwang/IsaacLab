#!/usr/bin/env python
"""Is the model RETRIEVING a memorized trajectory, predicting the MEAN, or neither?

THE QUESTION
    Established so far on the s180k checkpoint:
      - training arm R^2 0.968, held-out ~0
      - it genuinely uses vision (zeroing images drops seen arm R^2 0.971 -> -0.505)
      - held-out episodes are NOT novel in appearance (1.68 vs 1.73 train-to-train)
      - held-out episodes are NOT novel in trajectory (6.75 vs 6.45), corr(dist, R^2) = -0.005

    So it fails on episodes that are ordinary by every measure available. That leaves two
    concrete stories, and they imply completely different fixes:

      RETRIEVAL   it recognises which training scene this resembles and replays THAT episode's
                  trajectory. Then its output should match the nearest TRAINING episode's
                  actions better than the true ones.
      MEAN        it has collapsed to the average trajectory, ignoring the scene. Then its
                  output should match the phase-conditioned MEAN of training actions. (The 0904
                  notes already suspected this -- "~mean-action predictors on any scene they did
                  not train on" -- but it was never tested against a retrieval alternative.)
      NEITHER     it matches none of them, which points at the ACTION REPRESENTATION. These are
                  IK-Rel deltas: only meaningful relative to a current state the model may be
                  misjudging, in which case per-step deltas would look like noise.

THE CONTROL THAT MAKES THIS INTERPRETABLE
    If the nearest training episode's trajectory is already very close to the held-out one, then
    "matches retrieved" and "matches truth" are the same statement and the test says nothing.
    So MSE(truth_nearest_train, truth_heldout) is reported alongside, and the verdict refuses to
    conclude when that control is too small.

Everything is computed in the checkpoint's NORMALIZED action space (MIN_MAX -> [-1,1] using the
checkpoint's own min/max), which is the space the model actually predicts in, and the same space
generalization_check.py scores in.

USAGE
    conda activate lerobot_0.6.1_multitask_dit
    python dp-fm-ood/retrieval_check.py \
        outputs/run_0902_norm/n100_k6_s180k_lr3e4_crop_aug_sepenc/checkpoints/180000/pretrained_model \
        --train-n 100 --eval-n 300 --episodes 12 --per-episode 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from appearance_distance_check import nearest_distances, trajectory_descriptors  # noqa: E402
import generalization_check as _gc  # noqa: E402
from generalization_check import ARM, BASE, REPO_ID, load, split_episodes, tail_split  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def load_actions(n: str):
    """Raw action array + episode index, straight from the parquet -- no video decode.

    Uses generalization_check's DATASET_TEMPLATE (settable via --dataset-template) rather than
    the old hardcoded joint-state path: an 8-dim EEF checkpoint has to be scored against the
    0908 EEF export it trained on, and the 9-dim joint dataset would mis-shape the input.
    """
    root = str(BASE / _gc.DATASET_TEMPLATE.format(n=n) / "ID-visuomotor-based")
    ds = LeRobotDataset(repo_id=REPO_ID, root=root)
    hf = ds.hf_dataset.with_format(None)
    return (
        np.asarray(hf["action"], dtype=np.float32),
        np.asarray(hf["episode_index"]),
    )


def minmax_normalizer(ckpt: str):
    """The checkpoint's own ACTION MIN_MAX stats -> the [-1,1] mapping the policy was trained in."""
    f = Path(ckpt) / "policy_preprocessor_step_4_normalizer_processor.safetensors"
    d = load_file(str(f))
    lo = d["action.min"].float().numpy()
    hi = d["action.max"].float().numpy()
    rng = np.where((hi - lo) == 0, 1.0, hi - lo)
    return lambda a: 2.0 * (a - lo) / rng - 1.0


def chunk_at(actions_n, start, end, pos, horizon):
    """Action chunk of `horizon` steps from `pos`, clamped inside [start,end) by repeating the
    last real action -- the same padding idea lerobot uses at episode boundaries."""
    idx = np.minimum(np.arange(pos, pos + horizon), end - 1)
    return actions_n[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--train-n", default="100", choices=["50", "100"])
    p.add_argument("--eval-n", default="300", choices=["300"])
    p.add_argument("--eval_split", type=float, default=None)
    p.add_argument("--episodes", type=int, default=12, help="held-out episodes to test")
    p.add_argument("--per-episode", type=int, default=8)
    p.add_argument("--phase-bins", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset-template", default=_gc.DATASET_TEMPLATE,
                   help="dataset directory template with a {n} placeholder. Use "
                        "'lerobot_dataset_0908_eef_pos_{n}' for checkpoints trained on the "
                        "8-dim EEF-state re-export. Mirrors generalization_check.py's flag.")
    a = p.parse_args()

    # load(), tail_split() and load_actions() all read this module-level global at call time,
    # so rebinding it here is what actually redirects every dataset read.
    _gc.DATASET_TEMPLATE = a.dataset_template

    if a.eval_split is not None:
        seen, unseen = tail_split(a.eval_n, a.eval_split)
    else:
        seen, unseen = split_episodes(a.train_n, a.eval_n)
    print(f"split -> {len(seen)} train, {len(unseen)} held-out")

    actions, ep_idx = load_actions(a.eval_n)
    norm = minmax_normalizer(a.checkpoint)
    A = norm(actions)
    bounds = {int(e): (int(np.flatnonzero(ep_idx == e)[0]), int(np.flatnonzero(ep_idx == e)[-1]) + 1)
              for e in np.unique(ep_idx)}

    # nearest training episode for each held-out episode, by trajectory descriptor
    tr_ids, tr_D = trajectory_descriptors(a.eval_n, seen)
    ev_ids, ev_D = trajectory_descriptors(a.eval_n, unseen)
    mu, sd = tr_D.mean(0), tr_D.std(0) + 1e-8
    d = np.linalg.norm(((ev_D - mu) / sd)[:, None] - ((tr_D - mu) / sd)[None], axis=-1)
    nearest = {int(ev_ids[i]): int(tr_ids[int(d[i].argmin())]) for i in range(len(ev_ids))}
    dist = d.min(axis=1)

    cfg, ds, policy, pre = load(a.checkpoint, a.eval_n, a.device, unseen)
    H = cfg.horizon

    # phase-conditioned MEAN training chunk: the "collapsed to the average" hypothesis
    mean_bins = np.zeros((a.phase_bins, H, A.shape[1]), dtype=np.float64)
    cnt = np.zeros(a.phase_bins)
    for e in seen:
        s, t = bounds[int(e)]
        for b in range(a.phase_bins):
            pos = s + int(round((b + 0.5) / a.phase_bins * (t - s - 1)))
            mean_bins[b] += chunk_at(A, s, t, pos, H)
            cnt[b] += 1
    mean_bins /= np.maximum(cnt, 1)[:, None, None]

    pick_eps = [int(ev_ids[i]) for i in np.argsort(dist)[
        np.linspace(0, len(ev_ids) - 1, min(a.episodes, len(ev_ids))).round().astype(int)]]

    # Map (episode -> local frame offset -> index into the held-out-only dataset) ONCE.
    # Doing this inside the sampling loop re-materialises the whole column per frame.
    ds_ep = np.asarray(ds.hf_dataset.with_format(None)["episode_index"])
    ds_pos = {int(e): np.flatnonzero(ds_ep == e) for e in np.unique(ds_ep)}

    rng = np.random.default_rng(0)
    rows = []
    for e in pick_eps:
        s, t = bounds[e]
        ts, tt = bounds[nearest[e]]
        positions = s + rng.choice(max(t - s - 1, 1), min(a.per_episode, max(t - s - 1, 1)),
                                   replace=False)
        P, T_self, T_retr, T_mean = [], [], [], []
        for pos in positions:
            phase = (pos - s) / max(t - s - 1, 1)
            # locate this global frame inside the held-out-only dataset the policy reads from
            gi = int(ds_pos[e][pos - s])
            item = ds[gi]
            bt = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor):
                    bt[k] = v.unsqueeze(0).to(a.device)
                elif isinstance(v, str):
                    bt[k] = [v]
            for ck in ds.meta.camera_keys:
                if ck in bt and bt[ck].dtype == torch.uint8:
                    bt[ck] = bt[ck].float() / 255.0
            pb = pre(dict(bt))
            obs = {k: v for k, v in pb.items() if k != "action"}
            with torch.no_grad():
                if hasattr(policy, "_generate_actions"):
                    ch = policy._generate_actions(policy._prepare_batch(obs))
                else:
                    policy.reset()
                    ch = policy.predict_action_chunk(obs)
            P.append(ch[0, :H].float().cpu().numpy())
            T_self.append(chunk_at(A, s, t, pos, H))
            T_retr.append(chunk_at(A, ts, tt, ts + int(round(phase * (tt - ts - 1))), H))
            T_mean.append(mean_bins[min(int(phase * a.phase_bins), a.phase_bins - 1)])

        P, T_self = np.stack(P), np.stack(T_self)
        T_retr, T_mean = np.stack(T_retr), np.stack(T_mean)
        # The policy emits n_action_steps (24) steps, not horizon (32). Trim every target to the
        # prediction length -- same treatment generalization_check.py applies.
        m = min(P.shape[1], T_self.shape[1], T_retr.shape[1], T_mean.shape[1])
        P, T_self, T_retr, T_mean = P[:, :m], T_self[:, :m], T_retr[:, :m], T_mean[:, :m]
        var = T_self[..., ARM].var()

        def nmse(x, y):
            return float(((x[..., ARM] - y[..., ARM]) ** 2).mean() / var)

        rows.append((e, nearest[e], dist[list(ev_ids).index(e)],
                     nmse(P, T_self), nmse(P, T_retr), nmse(P, T_mean), nmse(T_retr, T_self)))

    print(f"\n  normalized MSE on arm dims (lower = closer). horizon={H}")
    print(f"  {'ep':>5} {'near':>5} {'dist':>6} | {'vs TRUTH':>9} {'vs RETRIEVED':>13} "
          f"{'vs MEAN':>8} | {'ctrl retr-vs-truth':>19}")
    for r in rows:
        print(f"  {r[0]:>5} {r[1]:>5} {r[2]:>6.2f} | {r[3]:>9.3f} {r[4]:>13.3f} "
              f"{r[5]:>8.3f} | {r[6]:>19.3f}")

    m = np.array([[r[3], r[4], r[5], r[6]] for r in rows]).mean(axis=0)
    print(f"\n  MEAN  vs truth {m[0]:.3f} | vs retrieved {m[1]:.3f} | vs mean-traj {m[2]:.3f}"
          f" | control(retrieved vs truth) {m[3]:.3f}")

    print("\n  VERDICT")
    if m[3] < 0.35:
        print("   INCONCLUSIVE: the nearest training episode is already very close to the held-out")
        print("   one (control < 0.35), so 'matches retrieved' and 'matches truth' are not")
        print("   distinguishable. Re-run with a stricter nearest-neighbour or more episodes.")
    elif m[2] < m[0] * 0.8 and m[2] < m[1] * 0.8:
        print("   MEAN-COLLAPSE. The output is closest to the phase-conditioned AVERAGE training")
        print("   trajectory -- it is ignoring the scene and replaying the average motion.")
        print("   Confirms the 0904 suspicion. The scene->action mapping was never learned;")
        print("   more data or regularization will not change that on its own.")
    elif m[1] < m[0] * 0.8:
        print("   RETRIEVAL. The output matches a MEMORIZED OTHER EPISODE better than the truth.")
        print("   The model is doing scene recognition -> trajectory lookup. More distinct demos")
        print("   should help, since coverage is what retrieval depends on.")
    else:
        print("   NEITHER. The output matches no training trajectory, average or specific. Look")
        print("   at the ACTION REPRESENTATION: these are IK-Rel deltas, meaningful only relative")
        print("   to a state the model may be misjudging. Consider absolute/EE-space actions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
