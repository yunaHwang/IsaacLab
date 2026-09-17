#!/usr/bin/env python3
"""Dense per-anchor, per-subtask-phase error analysis for a MultiTaskDiT checkpoint.

WHAT THIS IS, VS generalization_check.py
    generalization_check.py samples a handful of frames per episode, runs the policy ONCE on
    each, and reports split-level R^2/MSE. It is deliberately cheap and it is what produced
    every number in eefpos_vs_bf16.txt / r2_progress_0910_alleps.txt. This script does not
    touch it -- those numbers stay reproducible.

    This one is the expensive diagnostic:
      * ANCHORS ON A STRIDE, not random frames. Anchor t covers actions [t, t+23]; stride 4
        means t = 0, 4, 8, ... Consecutive chunks overlap in 20 of 24 steps (83%), so 64
        anchors per episode is far fewer than 64 independent measurements -- read the density
        as coverage of the trajectory, not as sample size.
      * MULTIPLE STOCHASTIC DRAWS per anchor. Flow matching integrates from x = torch.randn()
        (modeling_multi_task_dit.py), so the policy is stochastic: the same observation gives a
        different action chunk every call. `--draws 5` runs it 5 times against the SAME ground
        truth, which separates the policy's own sampling variance from its bias.
      * SUBTASK PHASE LABELS, per episode, exact. Not a global average fraction.

THE HIERARCHY IT PRODUCES
    draw          one (P, G) pair -> mse per dim (7) and total (1)
    anchor        `draws` draws -> mean and std over draws, per dim and total
    phase         all anchors whose START frame is in that subtask, within an episode
    split         all episodes -> per-phase, per-dim and total

SUBTASK PHASES -- WHERE THEY COME FROM
    The hdf5 records obs/datagen_info/subtask_term_signals/{grasp_1, stack_1, grasp_2} as a
    per-frame boolean. The first True is the frame that subtask completes. Four phases:
        P0  start   -> grasp_1    (approach + grasp red)     mean 21% of the episode
        P1  grasp_1 -> stack_1    (lift + stack red on blue) mean 23%
        P2  stack_1 -> grasp_2    (approach + grasp green)   mean 22%
        P3  grasp_2 -> end        (stack green on red)       mean 34%
    Boundaries are stable across the 300 episodes (std 0.011-0.024 as a fraction) but they are
    read PER EPISODE here, not from that average.

TWO INDEX OFFSETS, BOTH VERIFIED -- getting either wrong mislabels every phase
    1. EPISODE ORDER. The converter iterated get_episode_names() in raw HDF5 group order, which
       is LEXICOGRAPHIC: demo_0, demo_1, demo_10, demo_100, ... So LeRobot episode_index i maps
       to the hdf5 key at lexicographic position i -- NOT to demo_i. (demo_2 is LeRobot episode
       202, not 2.)
    2. FRAME OFFSET. 0908_eef_pos_isaac2lerobot.py:217 is `for frame_index in range(5, ...)` --
       it skips the first 5 frames of every episode. So hdf5_frame = lerobot_frame + 5.
    Both are asserted at load time (hdf5_len - 5 == lerobot_len, per episode). A future export
    with a different offset fails loudly instead of silently shifting the labels by ~2% of an
    episode, which is the same size as the boundary spread and would smear rather than break.

CHUNKS CAN CROSS A BOUNDARY
    A 24-step chunk is ~8.6% of a ~280-frame episode, and there are 3 internal boundaries, so
    ~27% of anchors produce a chunk spanning two subtasks. Each anchor is labelled by the phase
    of its START frame, and `crosses_boundary_frac` records how much of its chunk falls outside that
    phase. Filter on it to compare clean within-phase chunks against boundary-crossing ones
    rather than having that choice baked in.

Usage:
    python dp-fm-ood/phase_error_analysis.py <ckpt> --stride 4 --draws 5 --out-dir DIR
    python dp-fm-ood/phase_error_analysis.py <ckpt> --stride 4 --draws 5 --smoke-episodes 1  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generalization_check as gc  # noqa: E402

CHUNK = 24            # n_action_steps -- what _generate_actions returns
FRAME_OFFSET = 5      # converter skips hdf5 frames [0, 5)
SUBTASKS = ["grasp_1", "stack_1", "grasp_2"]
PHASE_NAMES = ["P0 approach+grasp red", "P1 stack red on blue",
               "P2 approach+grasp green", "P3 stack green on red"]
DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
ARM = slice(0, 6)


def load_phase_boundaries(hdf5_path: Path, lerobot_lengths: list[int]):
    """-> (boundaries, names). boundaries[i] = [b0, b1, b2] in LEROBOT frame coords.

    See the module docstring on the two offsets. Both are asserted here.
    """
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f["data"].keys())          # lexicographic == LeRobot episode order
        if len(keys) != len(lerobot_lengths):
            raise SystemExit(f"episode count mismatch: hdf5 {len(keys)} vs "
                             f"lerobot {len(lerobot_lengths)}")
        bounds = []
        for i, k in enumerate(keys):
            g = f["data"][k]
            T = int(g.attrs["num_samples"])
            if T - FRAME_OFFSET != lerobot_lengths[i]:
                raise SystemExit(
                    f"frame-offset assertion failed on {k} (lerobot episode {i}): "
                    f"hdf5 {T} - {FRAME_OFFSET} != lerobot {lerobot_lengths[i]}. "
                    f"The converter's skip may have changed; re-check "
                    f"0908_eef_pos_isaac2lerobot.py's frame loop before trusting phases.")
            st = g["obs"]["datagen_info"]["subtask_term_signals"]
            b = []
            for name in SUBTASKS:
                idx = np.flatnonzero(np.asarray(st[name]))
                # first True = the frame that subtask completes, shifted into LeRobot coords
                b.append(int(idx[0]) - FRAME_OFFSET if len(idx) else -1)
            bounds.append(b)
    return np.array(bounds), keys


def phase_timeline(b, L: int, width: int = 58):
    """ASCII bar per phase, showing where each subtask sits along one episode's frames."""
    cuts = [0] + [int(c) for c in b if c >= 0] + [int(L)]
    out = []
    for i in range(len(cuts) - 1):
        s0, e0 = cuts[i], cuts[i + 1]
        a = int(round(s0 / L * width))
        z = max(a + 1, int(round(e0 / L * width)))
        bar = " " * a + "|" + "-" * max(0, z - a - 2) + "|" + " " * max(0, width - z)
        out.append((i, s0, e0, bar[:width]))
    return out


def phase_of(frame: int, b) -> int:
    """Which of the 4 phases a frame falls in, given that episode's 3 boundaries."""
    for i, cut in enumerate(b):
        if cut >= 0 and frame < cut:
            return i
    return len(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--stride", type=int, default=4,
                   help="anchor every Nth frame. Chunks overlap in (24-N)/24 of their steps.")
    p.add_argument("--draw-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                   help="One torch seed per stochastic draw, re-applied at EVERY anchor. Flow "
                        "matching integrates from x = torch.randn() (modeling_multi_task_dit), "
                        "so the policy is stochastic; fixing the seed per draw index means "
                        "draw k uses the same noise everywhere, which makes anchors and "
                        "checkpoints directly comparable and the whole run reproducible. "
                        "Number of draws = number of seeds.")
    p.add_argument("--eval-n", default="300")
    p.add_argument("--eval_split", type=float, default=0.2)
    p.add_argument("--split", default="both", choices=["both", "seen", "held-out"])
    p.add_argument("--dataset-template", default="lerobot_dataset_0908_eef_pos_{n}")
    p.add_argument("--source-hdf5", type=Path,
                   default=gc.BASE / "datasets/visuomotor-based/0901_repro_gen_300_topped_up.hdf5")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-episodes", type=int, default=None,
                   help="SMOKE TEST ONLY: process just the first N episodes of each "
                        "split. Unrelated to --stride, which controls how many "
                        "anchors are taken WITHIN an episode. Omit for the real run "
                        "(all 240 seen / 60 held-out).")
    p.add_argument("--txt-episodes", type=int, default=2,
                   help="how many episodes to write per-anchor detail for in the .txt. The full "
                        "per-anchor arrays always go to the .npz; at stride 4 a complete text "
                        "dump would be ~300k lines per split.")
    a = p.parse_args()
    a.draws = len(a.draw_seeds)

    gc.DATASET_TEMPLATE = a.dataset_template
    a.out_dir.mkdir(parents=True, exist_ok=True)

    seen, unseen = gc.tail_split(a.eval_n, a.eval_split)
    if a.smoke_episodes:
        seen, unseen = seen[:a.smoke_episodes], unseen[:a.smoke_episodes]

    # Episode lengths straight from the LeRobot metadata, for the offset assertion.
    info = json.load(open(gc.BASE / gc.DATASET_TEMPLATE.format(n=a.eval_n)
                          / "ID-visuomotor-based" / "meta" / "info.json"))
    n_eps_total = info["total_episodes"]

    wanted = {"both": ("seen", "held-out"), "seen": ("seen",),
              "held-out": ("held-out",)}[a.split]
    splits = {"seen": seen, "held-out": unseen}

    for label in wanted:
        eps = splits[label]
        t0 = time.time()
        cfg, ds, policy, pre = gc.load(a.checkpoint, a.eval_n, a.device, eps)

        lengths = [int(x) for x in ds.meta.episodes["length"]]
        all_lengths = lengths if len(lengths) == n_eps_total else None
        if all_lengths is None:
            # `ds` holds only this split's episodes; the assertion needs all of them.
            full = gc.LeRobotDataset(repo_id=gc.REPO_ID,
                                     root=str(gc.BASE / gc.DATASET_TEMPLATE.format(n=a.eval_n)
                                              / "ID-visuomotor-based"))
            all_lengths = [int(x) for x in full.meta.episodes["length"]]
            del full
        bounds, hdf5_keys = load_phase_boundaries(a.source_hdf5, all_lengths)

        ep_col = np.asarray(ds.hf_dataset.with_format(None)["episode_index"])
        fr_col = np.asarray(ds.hf_dataset.with_format(None)["frame_index"])

        rec = []          # per-anchor records
        P_all, G_all = [], []
        for e in np.unique(ep_col):
            sel = np.flatnonzero(ep_col == e)
            order = sel[np.argsort(fr_col[sel])]
            L = len(order)
            b = bounds[int(e)]
            for t in range(0, L - CHUNK + 1, a.stride):
                i = int(order[t])
                item = ds[i]
                batch = {}
                for k, v in item.items():
                    batch[k] = (v.unsqueeze(0).to(a.device) if isinstance(v, torch.Tensor)
                                else ([v] if isinstance(v, str) else v))
                for ck in ds.meta.camera_keys:
                    if ck in batch and batch[ck].dtype == torch.uint8:
                        batch[ck] = batch[ck].float() / 255.0
                pb = pre(dict(batch))
                obs_only = {k: v for k, v in pb.items() if k != "action"}
                G = pb["action"][0, :CHUNK].float().cpu().numpy()
                with torch.no_grad():
                    # Encode the observation ONCE (identical across draws), then run all
                    # `draws` integrations as a SINGLE batch.
                    #
                    # Why this is exact, not an approximation: conditional_sample() would do
                    #     x = torch.randn((1, horizon, action_dim)); _euler_integrate(...)
                    # per call. Seeding before each torch.randn and stacking the results gives
                    # byte-identical noise to the sequential version, and _euler_integrate
                    # takes x_init explicitly, so batching only changes how many rows the DiT
                    # sees per step -- not the arithmetic on any one row. Verified equal to the
                    # sequential implementation (max|diff| = 0).
                    #
                    # Why it is much faster: the 100 Euler steps are sequential and cannot be
                    # parallelised in time, but at batch 1 a 6-layer/512-wide DiT leaves the
                    # GPU almost idle. Batching the draws fills it, turning 5 x 100 sequential
                    # model calls into 100.
                    pbatch = policy._prepare_batch(obs_only)
                    cond = policy.observation_encoder.encode(pbatch)
                    obj = policy.objective
                    dtype = next(policy.noise_predictor.parameters()).dtype
                    xs = []
                    for sd in a.draw_seeds:
                        torch.manual_seed(sd)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(sd)
                        xs.append(torch.randn((1, obj.horizon, obj.action_dim),
                                              dtype=dtype, device=a.device))
                    x0 = torch.cat(xs, dim=0)                       # (draws, horizon, act_dim)
                    cond_b = cond.expand(a.draws, *cond.shape[1:])
                    tg = torch.linspace(0, 1, obj.config.num_integration_steps + 1,
                                        device=a.device)
                    full = obj._euler_integrate(policy.noise_predictor, x0, tg, cond_b)
                    start = policy.config.n_obs_steps - 1
                    end = start + policy.config.n_action_steps
                    P = full[:, start:end][:, :CHUNK].float().cpu().numpy()
                per_draw_dim = ((P - G[None]) ** 2).mean(axis=1)   # (draws, 7)
                ph = phase_of(t, b)
                # how much of this chunk leaves the phase its start frame is in
                crosses_boundary_frac = float(np.mean([phase_of(t + j, b) != ph for j in range(CHUNK)]))
                rec.append(dict(episode=int(e), frame=t, phase=ph, crosses_boundary_frac=crosses_boundary_frac,
                                mse_dim_mean=per_draw_dim.mean(axis=0),
                                mse_dim_std=per_draw_dim.std(axis=0),
                                mse_mean=float(per_draw_dim.mean()),
                                mse_std=float(per_draw_dim.mean(axis=1).std())))
                P_all.append(P.astype(np.float32))
                G_all.append(G.astype(np.float32))

        lens_by_ep = {int(e): int((ep_col == e).sum()) for e in np.unique(ep_col)}
        bounds_by_ep = {int(e): bounds[int(e)] for e in np.unique(ep_col)}
        write_outputs(a, label, rec, P_all, G_all, hdf5_keys, time.time() - t0,
                      lens_by_ep, bounds_by_ep)
        del policy, ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_outputs(a, label, rec, P_all, G_all, hdf5_keys, elapsed,
                  lens_by_ep, bounds_by_ep):
    stem = a.out_dir / label.replace("-", "_")
    ep = np.array([r["episode"] for r in rec])
    fr = np.array([r["frame"] for r in rec])
    ph = np.array([r["phase"] for r in rec])
    cb = np.array([r["crosses_boundary_frac"] for r in rec])
    dm = np.stack([r["mse_dim_mean"] for r in rec])     # (anchors, 7)
    dsd = np.stack([r["mse_dim_std"] for r in rec])
    tot = np.array([r["mse_mean"] for r in rec])
    tsd = np.array([r["mse_std"] for r in rec])

    np.savez_compressed(f"{stem}.npz", draw_seeds=np.array(a.draw_seeds),
                        P=np.stack(P_all), G=np.stack(G_all),
                        episode=ep, frame=fr, phase=ph, crosses_boundary_frac=cb,
                        mse_dim_mean=dm, mse_dim_std=dsd, mse_mean=tot, mse_std=tsd)

    with open(f"{stem}.txt", "w") as f:
        w = f.write
        w(f"# checkpoint : {a.checkpoint}\n# split      : {label}\n")
        w(f"# anchors    : {len(rec)}   stride {a.stride}   draws/anchor {a.draws}   "
          f"chunk {CHUNK} steps\n")
        w(f"# draw seeds : {a.draw_seeds}  (re-applied at every anchor)\n")
        w(f"# forward passes: {len(rec) * a.draws}   elapsed {elapsed/60:.1f} min\n")
        w(f"# P = PREDICTED chunks ({a.draws}, {CHUNK}, 7) per anchor; G = GROUND TRUTH "
          f"({CHUNK}, 7). Normalized [-1,1].\n")
        w(f"# action dims: {', '.join(DIMS)}  (6-DoF relative EEF pose delta + binary gripper)\n")
        w(f"# phases (per-episode subtask boundaries, hdf5 frame - {FRAME_OFFSET}):\n")
        for i, n in enumerate(PHASE_NAMES):
            w(f"#   {i} = {n}\n")
        w(f"# crosses_boundary_frac = fraction of the 24-step chunk falling outside its start phase\n#\n")

        w("=== PHASE LAYOUT (where each subtask sits, mean over this split) ===\n")
        meanL = float(np.mean([lens_by_ep[int(e)] for e in np.unique(ep)]))
        meanb = np.mean([bounds_by_ep[int(e)] for e in np.unique(ep)], axis=0)
        w(f"  episode length (mean): {meanL:.0f} frames\n")
        w(f"  {'':<26}{'frames':>14}  0{' ' * 52}end\n")
        for i, s0, e0, bar in phase_timeline(meanb, meanL):
            w(f"  {PHASE_NAMES[i]:<26}{f'{s0}-{e0}':>14}  {bar}  "
              f"{(e0 - s0) / meanL * 100:.0f}%\n")

        w("\n=== PER-PHASE ERROR (every anchor is assigned to the phase of its start frame) ===\n")
        w(f"{'phase':<26}{'anchors':>8}{'mse':>10}{'sd/draw':>9}   "
          + "".join(f"{d:>9}" for d in DIMS) + "\n")
        for i, n in enumerate(PHASE_NAMES):
            m = ph == i
            if not m.any():
                continue
            w(f"{n:<26}{m.sum():>8}{tot[m].mean():>10.5f}{tsd[m].mean():>9.5f}   "
              + "".join(f"{v:>9.5f}" for v in dm[m].mean(axis=0)) + "\n")
        w(f"{'ALL':<26}{len(rec):>8}{tot.mean():>10.5f}{tsd.mean():>9.5f}   "
          + "".join(f"{v:>9.5f}" for v in dm.mean(axis=0)) + "\n")
        w("\n(crosses_boundary_frac -- how much of each 24-step chunk extends past its own\n"
          " phase -- is kept per anchor in the .npz for anyone who wants to filter on it,\n"
          " but it is not broken out here: a chunk belongs to the phase it starts in.)\n")

        w(f"\n=== PER-ANCHOR DETAIL (first {a.txt_episodes} episodes; all of them in "
          f"{Path(stem).name}.npz) ===\n")
        for e in np.unique(ep)[:a.txt_episodes]:
            m = np.flatnonzero(ep == e)
            w(f"\n--- episode {int(e)} (hdf5 {hdf5_keys[int(e)]})   {len(m)} anchors\n")
            for j in m:
                w(f"  t={fr[j]:>4} phase {ph[j]}  "
                  f"mse {tot[j]:.5f} (sd over {a.draws} draws {tsd[j]:.5f})\n")
                w("      per-dim mse : " + " ".join(f"{v:.5f}" for v in dm[j]) + "\n")
                w("      per-dim sd  : " + " ".join(f"{v:.5f}" for v in dsd[j]) + "\n")
            for i in range(4):
                mm = m[ph[m] == i]
                if len(mm):
                    w(f"  -- phase {i} mean over {len(mm)} anchors: {tot[mm].mean():.5f}\n")
            w(f"  -- episode mean over {len(m)} anchors: {tot[m].mean():.5f}\n")
        w(f"\nSPLIT MEAN over {len(np.unique(ep))} episodes / {len(rec)} anchors: "
          f"{tot.mean():.6f}\n")
    print(f"  [{label}] {len(rec)} anchors x {a.draws} draws = {len(rec)*a.draws} passes "
          f"in {elapsed/60:.1f} min -> {stem}.txt / .npz")


if __name__ == "__main__":
    main()
