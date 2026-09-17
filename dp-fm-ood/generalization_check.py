#!/usr/bin/env python3
"""Score a checkpoint on episodes it was NEVER trained on. No Isaac, no rollout.

replay_check.py samples the whole training dataset, and every train config here has
`eval_split: 0.0`, so its R^2 is a TRAINING R^2 -- it proves the model can fit, and says
nothing about whether it generalizes. A policy that memorizes 27929 frames scores ~0.99
there and still fails every rollout.

The N=300 set turns out to contain all 100 of the N=100 set's episodes (reordered) plus 200
new ones from the same generator, so those 200 are a free held-out split for any checkpoint
trained on N=100. This script scores the same checkpoint twice -- on the seen episodes and on
the unseen ones -- and reports the gap.

Read the gap, not either number alone:
  - both high                -> the model generalizes; the failure is closed-loop
                               (compounding error / harness), not the fit.
  - seen high, unseen low    -> memorization. More steps makes it worse, not better.
  - both low                 -> underfitting (this is what the pre-180k runs showed).

Usage:
    conda activate lerobot_0.6.1_multitask_dit
    python generalization_check.py <checkpoint_dir> [--n 64]
"""

import argparse
import sys
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

import lerobot.policies.multi_task_dit.configuration_multi_task_dit  # noqa: F401
import lerobot.policies.diffusion.configuration_diffusion  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tokens import assert_fully_loaded, maybe_install_patch_tokens  # noqa: E402

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
REPO_ID = "ID/visuomotor-based"

# Dataset directory template, {n} = the --eval-n size. Overridable with --dataset-template so a
# checkpoint trained on a RE-EXPORTED dataset can be scored against that dataset. The 0908 EEF
# export has an 8-dim observation.state; loading it against the default 9-dim joint dataset
# would mis-shape the policy input.
DATASET_TEMPLATE = "lerobot_dataset_0901_topped_up_{n}"
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


def split_episodes(train_n: str = "100", eval_n: str = "300"):
    """Episode indices of `eval_n` that do / do not appear in `train_n`.

    Matched on a hash of the episode's action sequence rather than on the index, because the
    two sets were generated independently and the shared episodes are in a different order.
    """
    import glob

    import pandas as pd

    def hashes(n):
        f = sorted(glob.glob(str(BASE / DATASET_TEMPLATE.format(n=n)
                                 / "ID-visuomotor-based/data/**/*.parquet"), recursive=True))
        df = pd.concat([pd.read_parquet(p) for p in f])
        return {int(e): hashlib.md5(np.stack(g["action"].to_numpy()).tobytes()).hexdigest()
                for e, g in df.groupby("episode_index")}

    train, ev = hashes(train_n), hashes(eval_n)
    seen_h = set(train.values())
    unseen = sorted(e for e, h in ev.items() if h not in seen_h)
    seen = sorted(e for e, h in ev.items() if h in seen_h)
    return seen, unseen


def tail_split(eval_n: str, eval_split: float):
    """Reproduce lerobot's own train/eval episode split, for checkpoints trained WITH
    --dataset.eval_split.

    make_train_eval_datasets (datasets/factory.py) holds out the LAST
    ceil(n_episodes * eval_split) episodes per task. This dataset has a single task
    (task_index is 0 everywhere), so that reduces to the tail of the episode list.

    Needed because the hash-matching split above assumes the checkpoint was trained on the
    N=100 set. A model trained on N=300 has seen every episode in that set, so hash-matching
    would report training episodes as "held out" and silently produce a meaningless number.
    """
    import json

    info = json.load(open(BASE / DATASET_TEMPLATE.format(n=eval_n)
                          / "ID-visuomotor-based/meta/info.json"))
    n_eps = info["total_episodes"]
    n_eval = math.ceil(n_eps * eval_split)
    seen = list(range(n_eps - n_eval))
    unseen = list(range(n_eps - n_eval, n_eps))
    return seen, unseen


def load(ckpt: str, n: str, device: str, episodes=None):
    root = str(BASE / DATASET_TEMPLATE.format(n=n) / "ID-visuomotor-based")
    # MUST happen before make_policy(): it monkeypatches the encoder CLASS, so the policy has
    # to be constructed afterwards or the extra tensors are dropped again.
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

    ds = LeRobotDataset(repo_id=REPO_ID, root=root, delta_timestamps=dt, episodes=episodes)
    policy = make_policy(cfg, ds_meta=ds.meta).eval().to(device)
    assert_fully_loaded(policy, ckpt)
    # Normalization constants come from the CHECKPOINT, so the held-out split is scored with
    # exactly the constants the model trained with -- not stats refit on the eval episodes.
    pre, _ = make_pre_post_processors(cfg, pretrained_path=ckpt, dataset_stats=ds.meta.stats)
    return cfg, ds, policy, pre


def dump_samples(P, G, EP, FR, out_stem: Path, label: str, ckpt: str, per_episode):
    """Write every sampled (P, G) pair out for inspection.

    TWO FILES, because they answer different questions:
      <stem>.txt   readable. One block per EPISODE, one row per sample inside it, showing the
                   first action vector of each chunk (P[t=0] vs G[t=0]), that sample's MSE, and
                   the per-dim MSE over the whole chunk. Ends with the episode mean, then the
                   split mean. Enough to eyeball WHY a sample is bad without loading anything.
      <stem>.npz   the FULL arrays -- P and G at (n_samples, chunk_steps, 7) plus the episode
                   and frame index of each sample. The txt only shows t=0 of a 32-step chunk;
                   this has all of it, for anything the eye cannot do.

    P and G are in NORMALIZED [-1, 1] action space (the preprocessor's MIN_MAX is applied and
    the unnormalizing postprocessor deliberately is not), which is the space the R^2 and MSE
    are computed in -- so numbers here match the reported ones exactly.
    """
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    per_sample = ((P - G) ** 2).mean(axis=(1, 2))
    per_sample_dim = ((P - G) ** 2).mean(axis=1)          # (n_samples, 7)

    np.savez_compressed(f"{out_stem}.npz", P=P, G=G, episode=EP, frame=FR,
                        mse_per_sample=per_sample, mse_per_sample_dim=per_sample_dim)

    dims = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    uniq = np.unique(EP)
    with open(f"{out_stem}.txt", "w") as f:
        f.write(f"# checkpoint : {ckpt}\n")
        f.write(f"# split      : {label}\n")
        f.write(f"# episodes   : {len(uniq)}   frames sampled per episode: {per_episode}   "
                f"total samples: {len(per_sample)}\n")
        f.write(f"# P = PREDICTED action chunk, G = GROUND-TRUTH action chunk, "
                f"both {P.shape[1]}x{P.shape[2]} in normalized [-1,1]\n")
        f.write(f"# action dims: {', '.join(dims)}  "
                f"(6-DoF relative EEF pose delta + binary gripper)\n")
        f.write(f"# sample mse = mean of (P-G)^2 over that sample's {P.shape[1]} chunk steps "
                f"x {P.shape[2]} dims\n")
        f.write(f"# full {P.shape[1]}-step chunks are in {out_stem.name}.npz\n#\n")
        for e in uniq:
            sel = np.flatnonzero(EP == e)
            f.write(f"=== episode {int(e)}   ({len(sel)} samples)\n")
            for j in sel:
                f.write(f"  frame {int(FR[j]):>4}   mse {per_sample[j]:.6f}\n")
                f.write("    P[t=0] : " + " ".join(f"{v:+.4f}" for v in P[j, 0]) + "\n")
                f.write("    G[t=0] : " + " ".join(f"{v:+.4f}" for v in G[j, 0]) + "\n")
                f.write("    mse/dim: " + " ".join(f"{v:.5f}" for v in per_sample_dim[j]) + "\n")
            f.write(f"  -- episode mean mse over {len(sel)} samples: "
                    f"{per_sample[sel].mean():.6f}\n")
        f.write(f"\nSPLIT MEAN over {len(uniq)} episodes / {len(per_sample)} samples: "
                f"{per_sample.mean():.6f}\n")
        f.write("(this equals the reported MSE for this split)\n")


def score(ckpt: str, n: str, episodes, n_samples: int, device: str, seed: int = 0,
          ablate_images: bool = False, ablate_state: bool = False,
          per_episode: int | None = None, dump_stem: Path | None = None,
          dump_label: str = "") -> dict:
    cfg, ds, policy, pre = load(ckpt, n, device, episodes)
    # Seed TORCH, not just numpy. Flow matching integrates from a random initial sample
    # (modeling_multi_task_dit.py:766, `x = torch.randn(...)`), so the policy's prediction for a
    # given frame is STOCHASTIC. Without this, re-scoring the same checkpoint on the same frames
    # returns a different R^2 every time -- measured 0.028 vs 0.042 arm held-out on
    # refpatch_0908 @090000, a 0.014 swing from sampler noise alone, which is the same size as
    # the effects being compared. Seeding makes a score reproducible and arms comparable.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    if per_episode:
        # STRATIFIED: exactly `per_episode` frames from EVERY episode in the split, so all of
        # them are represented. The default flat draw below samples n_samples frames uniformly
        # from the whole split pool, which leaves some episodes untouched purely by chance --
        # 256 frames over 240 episodes lands in ~162 of them (coupon collector), which is why
        # the report says "(162 eps)" and not "(240 eps)". That is a property of the sampler,
        # not a filter on the data. Stratifying removes it, at `per_episode * n_episodes`
        # forward passes instead of n_samples.
        ep_col = np.asarray(ds.hf_dataset.with_format(None)["episode_index"])
        picked = []
        for e in np.unique(ep_col):
            pool = np.flatnonzero(ep_col == e)
            picked.append(rng.choice(pool, min(per_episode, len(pool)), replace=False))
        idxs = np.concatenate(picked)
    else:
        idxs = rng.choice(ds.num_frames, min(n_samples, ds.num_frames), replace=False)

    # Episode lengths, for turning frame_index into a normalized phase in [0, 1]. Column access
    # on the HF Dataset, not row iteration -- 300 rows either way, but this is O(1) calls.
    _eps = ds.meta.episodes
    ep_len = dict(zip(_eps["episode_index"], _eps["length"]))

    P, G, EP, PH, FR = [], [], [], [], []
    for i in idxs:
        item = ds[int(i)]
        _e = int(item["episode_index"])
        _f = int(item["frame_index"])
        EP.append(_e)
        FR.append(_f)
        PH.append(_f / max(ep_len.get(_e, 1) - 1, 1))
        batch = {}
        for k, v in item.items():
            batch[k] = v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else (
                [v] if isinstance(v, str) else v)
        for ck in ds.meta.camera_keys:
            if ck in batch and batch[ck].dtype == torch.uint8:
                batch[ck] = batch[ck].float() / 255.0
        pb = pre(dict(batch))
        if ablate_images:
            # Zero AFTER normalization. Images use MEAN_STD, so 0 is exactly the mean image -
            # a valid, information-free input rather than an out-of-range black frame that
            # would itself be OOD and confound the result.
            for ck in ds.meta.camera_keys:
                if ck in pb:
                    pb[ck] = torch.zeros_like(pb[ck])
        if ablate_state:
            # Mirror of ablate_images, for observation.state. NOTE the normalization differs:
            # STATE is MIN_MAX (config normalization_mapping), so 0 in normalized space is the
            # MIDPOINT of the observed joint range, not the mean pose. It is in-range and
            # information-free, which is what this test needs, but it is not the exact analogue
            # of the mean image used above.
            #
            # WHY: on held-out episodes the model emits the phase-conditioned MEAN trajectory
            # (retrieval_check.py, 2026-09-07). These are mimic demos, so the 9 joint values
            # encode episode PHASE almost perfectly -- the model may be taking a
            # state -> phase -> mean-action shortcut and never learning pixels -> cube pose.
            # If held-out predictions barely move when state is removed, that shortcut is NOT
            # what is carrying them; if they collapse, it is.
            if "observation.state" in pb:
                pb["observation.state"] = torch.zeros_like(pb["observation.state"])
        obs_only = {k: v for k, v in pb.items() if k != "action"}
        with torch.no_grad():
            if hasattr(policy, "_generate_actions"):
                chunk = policy._generate_actions(policy._prepare_batch(obs_only))
            else:
                policy.reset()
                chunk = policy.predict_action_chunk(obs_only)
        m = min(chunk.shape[1], pb["action"].shape[1])
        P.append(chunk[0, :m].float().cpu().numpy())
        G.append(pb["action"][0, :m].float().cpu().numpy())

    P, G = np.stack(P), np.stack(G)
    EP, PH, FR = np.array(EP), np.array(PH), np.array(FR)
    if dump_stem is not None:
        dump_samples(P, G, EP, FR, dump_stem, dump_label, ckpt, per_episode)

    def r2(p, g):
        """Standard R^2: baseline is the per-dimension GLOBAL mean action over this split."""
        return 1 - ((p - g) ** 2).mean() / ((g - g.mean(axis=(0, 1), keepdims=True)) ** 2).mean()

    def phase_baseline(g, ph, n_bins: int = 20):
        """Predicted action chunk of a PHASE-CONDITIONED mean predictor, leave-one-out.

        WHY THIS IS THE BASELINE THAT MATTERS
            The 0907 diagnosis is MEAN-COLLAPSE: on unseen scenes the policy emits the
            phase-conditioned average training trajectory (retrieval_check.py -- predictions sit
            ~2x closer to the mean trajectory than to the truth). Standard R^2 scores against a
            single global mean, which that collapsed predictor BEATS, because episode phase is
            real signal. So a positive standard R^2 does not rule out mean-collapse.

            Scoring against this baseline does. R^2_phase ~ 0 means "no better than knowing how
            far through the episode we are" -- i.e. the policy contributes nothing beyond phase,
            which is exactly mean-collapse. R^2_phase > 0 means it is using something else, and
            the only other thing available is the scene.

        LEAVE-ONE-OUT because the bin means are computed from the same ground truth we score
        against. Including sample i in its own bin mean would let the baseline partly memorize
        the target, weakening it and flattering the model. LOO removes that.
        """
        bins = np.clip((ph * n_bins).astype(int), 0, n_bins - 1)
        out = np.empty_like(g)
        gmean = g.mean(axis=0, keepdims=True)
        for b in range(n_bins):
            m = bins == b
            k = int(m.sum())
            if k >= 2:
                out[m] = (g[m].sum(axis=0, keepdims=True) - g[m]) / (k - 1)
            elif k == 1:
                out[m] = gmean            # singleton bin: no LOO possible, fall back to global
        return out

    def r2_vs_phase(p, g, base):
        """1 - SS(model)/SS(phase-conditioned mean). Beats-the-phase-predictor score."""
        denom = ((base - g) ** 2).mean()
        return 1 - ((p - g) ** 2).mean() / denom if denom > 0 else float("nan")

    # (n_samples,) -- collapse each frame's [chunk_steps, 7] error block to one number.
    per_sample = ((P - G) ** 2).mean(axis=(1, 2))

    BASE = phase_baseline(G, PH)

    def boot(fn, n_boot: int = 1000, seed_: int = 0):
        """Percentile CI, resampled BY EPISODE.

        Frames inside one episode are heavily correlated -- consecutive states, one scene, one
        cube layout -- so a frame-level bootstrap treats ~n_samples independent draws where there
        are really only as many as there are episodes, and reports an interval several times too
        narrow. Resampling whole episodes respects that clustering.
        """
        uniq = np.unique(EP)
        if len(uniq) < 2:
            return (float("nan"), float("nan"))
        rb = np.random.default_rng(seed_)
        idx_by_ep = {e: np.flatnonzero(EP == e) for e in uniq}
        vals = []
        for _ in range(n_boot):
            pick = rb.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([idx_by_ep[e] for e in pick])
            v = fn(sel)
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    arm_lo, arm_hi = boot(lambda s: r2(P[s][..., ARM], G[s][..., ARM]))
    ph_lo, ph_hi = boot(lambda s: r2_vs_phase(P[s][..., ARM], G[s][..., ARM], BASE[s][..., ARM]))

    return {
        "all": float(r2(P, G)),
        "arm": float(r2(P[..., ARM], G[..., ARM])),
        "gripper": float(r2(P[..., -1:], G[..., -1:])),
        "per_dim": [float(r2(P[..., d:d + 1], G[..., d:d + 1])) for d in range(P.shape[-1])],
        "mse": float(((P - G) ** 2).mean()),
        # Per-SAMPLE MSE: one value per sampled frame (mean over that frame's chunk steps and
        # all 7 dims), so the spread ACROSS frames is visible instead of only the grand mean.
        # This is a distribution over the data, NOT a confidence interval on the mean -- the
        # bootstrap CIs above are the latter, and the two answer different questions.
        # Percentiles rather than min/max as the headline: per-frame MSE is strongly
        # right-skewed (most frames easy, a few near a grasp transition very hard), so min/max
        # tracks a single outlier. Both are recorded; the plot picks.
        "mse_per_sample": {
            "mean": float(per_sample.mean()),
            "p10": float(np.percentile(per_sample, 10)),
            "p25": float(np.percentile(per_sample, 25)),
            "p50": float(np.percentile(per_sample, 50)),
            "p75": float(np.percentile(per_sample, 75)),
            "p90": float(np.percentile(per_sample, 90)),
            "min": float(per_sample.min()),
            "max": float(per_sample.max()),
        },
        "n": P.shape[0],
        "n_episodes": int(len(np.unique(EP))),
        "arm_ci": (arm_lo, arm_hi),
        # How much of the standard R^2 was available from phase alone.
        "phase_arm": float(r2_vs_phase(P[..., ARM], G[..., ARM], BASE[..., ARM])),
        "phase_arm_ci": (ph_lo, ph_hi),
        "phase_baseline_arm": float(r2(BASE[..., ARM], G[..., ARM])),
    }


def main() -> None:
    global DATASET_TEMPLATE
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--train-n", default="100", choices=["50", "100"])
    p.add_argument("--eval-n", default="300", choices=["300"])
    p.add_argument("--eval_split", type=float, default=None,
                   help="Use ONLY for a checkpoint trained on --eval-n WITH "
                        "--dataset.eval_split=<this same value>. Reproduces lerobot's own split "
                        "(the last ceil(n*split) episodes are held out) instead of hash-matching "
                        "against --train-n. Required when training on N=300: that set contains "
                        "every N=100 episode, so hash-matching would score training episodes as "
                        "'held out' and return a meaningless number.")
    p.add_argument("--dump-dir", type=Path, default=None,
                   help="Write every sampled (P, G) pair here: <dir>/<split>.txt readable, "
                        "<dir>/<split>.npz with the full (n_samples, chunk, 7) arrays. "
                        "See dump_samples().")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Use only the first N episodes of EACH split. For smoke tests -- "
                        "N=1 with --per-episode 2 is 2 forward passes per split, not 1500.")
    p.add_argument("--per-episode", type=int, default=None,
                   help="Stratified sampling: take exactly this many frames from EVERY episode "
                        "in the split, instead of --n frames drawn uniformly from the whole "
                        "pool. Guarantees every episode is represented (the '(162 eps)' in the "
                        "default report is a coupon-collector artifact of uniform sampling, not "
                        "a filter). Costs per_episode * n_episodes forward passes: 20 on this "
                        "dataset is 4800 seen + 1200 held-out vs 256 + 256.")
    p.add_argument("--n", type=int, default=64,
                   help="Number of eval FRAMES sampled per split (Monte-Carlo sample count for "
                        "the R^2 estimate). NOT a batch size -- frames are scored one at a time. "
                        "The held-out pool is ~8.4k frames, so 64 samples <1%% of it.")
    p.add_argument("--seed", type=int, default=0,
                   help="Frame-sampling seed. Fixed across arms on purpose: base and patch are "
                        "scored on the SAME frames, so the arm-vs-arm DIFFERENCE is paired and "
                        "far tighter than either absolute number. Vary it to gauge noise.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset-template", default=DATASET_TEMPLATE,
                   help="dataset directory template with a {n} placeholder; default "
                        "'lerobot_dataset_0901_topped_up_{n}'. Use "
                        "'lerobot_dataset_0908_eef_pos_{n}' to score checkpoints trained on the "
                        "8-dim EEF-state re-export.")
    p.add_argument("--split", default="both", choices=["both", "seen", "held-out"],
                   help="Score only one split. 'held-out' halves the runtime when sweeping "
                        "checkpoints for the step at which memorization sets in.")
    p.add_argument("--ablate_state", action="store_true",
                   help="Also score with observation.state replaced by the midpoint of "
                        "its range, and report the drop. Tests whether the model is "
                        "riding a state->phase shortcut instead of reading the scene.")
    p.add_argument("--ablate_images", action="store_true",
                   help="Also score each split with both cameras replaced by the mean image, and "
                        "report the drop. This asks whether the model USES vision. Read the two "
                        "drops together:\n"
                        "  seen drops a lot + held-out drops ~0  -> the vision encoder memorized "
                        "training frames and contributes nothing on new scenes. Freeze it "
                        "(vision_encoder_lr_multiplier=0.0).\n"
                        "  seen drops ~0                         -> the model never used vision at "
                        "all; it is riding state/phase, and the encoder is not the lever.\n"
                        "  held-out drops a lot                  -> vision does transfer, and the "
                        "problem is elsewhere.")
    args = p.parse_args()
    DATASET_TEMPLATE = args.dataset_template

    if args.eval_split is not None:
        seen, unseen = tail_split(args.eval_n, args.eval_split)
        how = f"lerobot tail split, eval_split={args.eval_split}"
    else:
        seen, unseen = split_episodes(args.train_n, args.eval_n)
        how = "matched by action hash"
    print(f"\n=== {Path(args.checkpoint).parents[2].name} ===")
    print(f"  checkpoint : {args.checkpoint}")
    if args.max_episodes:
        seen, unseen = seen[:args.max_episodes], unseen[:args.max_episodes]
        print(f"  NOTE       : --max-episodes {args.max_episodes} -- smoke test, not a real score")

    print(f"  split      : {len(seen)} seen / {len(unseen)} held-out episodes "
          f"(N={args.eval_n} set, {how})")
    print(f"  sampling   : " + (f"stratified, {args.per_episode} frames from EVERY episode"
                                if args.per_episode else
                                f"{args.n} frames drawn uniformly from the whole split pool"))

    wanted = {"both": ("seen", "held-out"), "seen": ("seen",), "held-out": ("held-out",)}[args.split]
    rows = []
    for key, label, eps in (("seen", "SEEN (trained on)", seen),
                            ("held-out", "HELD-OUT (never seen)", unseen)):
        if key not in wanted:
            continue
        r = score(args.checkpoint, args.eval_n, eps, args.n, args.device, seed=args.seed,
                  per_episode=args.per_episode,
                  dump_stem=(args.dump_dir / key.replace("-", "_")) if args.dump_dir else None,
                  dump_label=label)
        rows.append((label, r))
        print(f"\n  {label}   samples={r['n']}")
        print(f"    R2 all     : {r['all']:.3f}")
        print(f"    R2 ARM 0-5 : {r['arm']:.3f}"
              + (f"   95% CI [{r['arm_ci'][0]:+.3f}, {r['arm_ci'][1]:+.3f}]"
                 f"  ({r.get('n_episodes','?')} eps)" if r.get("arm_ci") else ""))
        # The discriminating number: does the policy beat a phase-conditioned mean predictor?
        if r.get("phase_arm") is not None:
            print(f"    R2 ARM vs PHASE-MEAN : {r['phase_arm']:+.3f}"
                  + (f"   95% CI [{r['phase_arm_ci'][0]:+.3f}, {r['phase_arm_ci'][1]:+.3f}]"
                     if r.get("phase_arm_ci") else ""))
            print(f"      (the phase-mean predictor itself scores R2 {r['phase_baseline_arm']:+.3f}"
                  " against the global mean)")
            lo, hi = r.get("phase_arm_ci", (float('nan'), float('nan')))
            if hi == hi and hi <= 0:
                print("      -> CI entirely <= 0: NOT beating phase alone. Mean-collapse.")
            elif lo == lo and lo > 0:
                print("      -> CI entirely > 0: the policy uses more than phase.")
        print(f"    R2 gripper : {r['gripper']:.3f}")
        print(f"    MSE        : {r['mse']:.5f}")
        _ps = r["mse_per_sample"]
        print(f"    MSE per-sample : mean {_ps['mean']:.5f}  p10 {_ps['p10']:.5f}  "
              f"p25 {_ps['p25']:.5f}  p50 {_ps['p50']:.5f}  p75 {_ps['p75']:.5f}  "
              f"p90 {_ps['p90']:.5f}  min {_ps['min']:.5f}  max {_ps['max']:.5f}  "
              f"n {r['n']}")
        print(f"    per-dim R2 : {[round(x, 3) for x in r['per_dim']]}")
        if args.ablate_images:
            rb = score(args.checkpoint, args.eval_n, eps, args.n, args.device, seed=args.seed, ablate_images=True)
            print(f"    -- cameras replaced by the mean image --")
            print(f"    R2 ARM 0-4 : {rb['arm']:.3f}   (drop {r['arm'] - rb['arm']:+.3f})")
            print(f"    R2 gripper : {rb['gripper']:.3f}   (drop {r['gripper'] - rb['gripper']:+.3f})")
        if args.ablate_state:
            rs = score(args.checkpoint, args.eval_n, eps, args.n, args.device, seed=args.seed, ablate_state=True)
            print("    -- observation.state replaced by its range midpoint --")
            print(f"    R2 ARM 0-4 : {rs['arm']:.3f}   (drop {r['arm'] - rs['arm']:+.3f})")
            print(f"    R2 gripper : {rs['gripper']:.3f}   (drop {r['gripper'] - rs['gripper']:+.3f})")

    if len(rows) < 2:
        return
    gap = rows[0][1]["arm"] - rows[1][1]["arm"]
    print(f"\n  ARM generalization gap: {gap:+.3f}  "
          f"({rows[0][1]['arm']:.3f} seen -> {rows[1][1]['arm']:.3f} held-out)")
    if rows[1][1]["arm"] > 0.8:
        print("  -> generalizes. The failure is closed-loop, not the fit.")
    elif gap > 0.3:
        print("  -> MEMORIZATION. Training longer makes this worse, not better.")
    else:
        print("  -> underfitting on both splits.")


if __name__ == "__main__":
    main()
