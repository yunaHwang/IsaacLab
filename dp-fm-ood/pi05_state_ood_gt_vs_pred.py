"""Pi0.5 state-OOD reconstruction loss (ood_signal.py metric 1, Nb draws) on training episodes:

    gt group        3 stack_2_cubes episodes, every frame: loss of (obs, RECORDED next-16 action chunk)
    predicted group 2 OTHER stack episodes,  every frame: loss of (obs, MODEL'S OWN sampled chunk)

Output (in --out_dir):
    per_episode/ep<N>_<group>.csv   frame, loss               (written as each episode finishes)
    state_ood_gt_vs_pred.csv        frame, gt_min, gt_max, gt_n, gt_ep*, pred_min, pred_max, pred_n, pred_ep*
    state_ood_gt_vs_pred.txt        "frame 0 -> gt range: x - y (n=3), predicted range: x - y (n=2)"
Range = min-max across the group's episodes at that frame index (episodes differ in length, so
late frames can have fewer episodes; n says how many).

Loss = Pi0.5's training objective, the same quantity multitask_dit_loss() computes for
MultiTaskDiT: x_t = t*noise + (1-t)*a, target u = noise - a, MSE(v_theta(x_t, t | obs), u) over the
6 real action dims, averaged over Nb fresh (noise, t ~ Beta(1.5,1)*0.999+0.001) draws. Actions are
quantile-normalized exactly like the checkpoint's preprocessor. Frames within 16 of the episode end
repeat the last recorded action to fill the gt chunk (LeRobot's end-of-episode padding).

Run (lerobot 0.6.1 env, needs the dataset files the script downloads on demand):
    conda activate lerobot_0.6.1_multitask_dit
    python pi05_state_ood_gt_vs_pred.py --out_dir outputs/pi05_state_ood
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_server import load_policy, resolve_checkpoint, to_policy_obs
from lerobot.policies import make_pre_post_processors

REPO = "https://huggingface.co/datasets/Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi/resolve/main"
STACK_EPISODES = range(600, 900)
CHUNK = 16

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/tmp/claude-1000/-home-wisc-rt2-trimanual/56b8fc73-fa60-43fd-96f0-a456704b5bd3/scratchpad/scrape",
                    help="Where dataset parquet/meta/video files live (downloaded here if missing).")
parser.add_argument("--out_dir", default="outputs/pi05_state_ood")
parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent / "checkpoints" / "pi05_isaaclab_so101_multitask_1ep"))
parser.add_argument("--num_samples", type=int, default=32, help="Nb (noise, t) draws per frame, as in ood_signal.py.")
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gt_episodes", type=int, nargs="*", default=None)
parser.add_argument("--pred_episodes", type=int, nargs="*", default=None)
args = parser.parse_args()

D, OUT = Path(args.data_dir), Path(args.out_dir)
(OUT / "per_episode").mkdir(parents=True, exist_ok=True)
D.mkdir(parents=True, exist_ok=True)


def fetch(rel, local):
    if not local.exists():
        subprocess.run(["curl", "-sfL", "-o", str(local), f"{REPO}/{rel}"], check=True)
    return local


rng = np.random.default_rng(args.seed)
picked = [int(x) for x in rng.choice(list(STACK_EPISODES), 5, replace=False)]
gt_eps = args.gt_episodes or picked[:3]
pred_eps = args.pred_episodes or picked[3:]
assert not set(gt_eps) & set(pred_eps), "gt and predicted episodes must differ"
print(f"gt episodes (recorded actions): {gt_eps}\npredicted episodes (model actions): {pred_eps}", flush=True)

eps = pd.read_parquet(fetch("meta/episodes/chunk-000/file-000.parquet", D / "episodes.parquet")).set_index("episode_index")
meta = {}
for line in open(fetch("meta/scrape_episode_metadata.jsonl", D / "scrape_episode_metadata.jsonl")):
    m = json.loads(line)
    meta[m["dataset_episode_index"]] = m
(OUT / "episodes.json").write_text(json.dumps({
    "gt": gt_eps, "pred": pred_eps, "num_samples": args.num_samples,
    "instructions": {"gt": {str(e): meta[e]["instruction"] for e in gt_eps},
                     "pred": {str(e): meta[e]["instruction"] for e in pred_eps}},
}, indent=2))
data_cache = {}


def episode_arrays(ep):
    fi = int(eps.loc[ep, "data/file_index"])
    if fi not in data_cache:
        path = fetch(f"data/chunk-000/file-{fi:03d}.parquet", D / f"data{fi}.parquet")
        data_cache[fi] = pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state", "action"])
    e = data_cache[fi]
    e = e[e.episode_index == ep].sort_values("frame_index")
    return np.stack(e["observation.state"]).astype(np.float32), np.stack(e["action"]).astype(np.float32)


def episode_frames(ep, cam, n):
    fi = int(eps.loc[ep, f"videos/observation.images.{cam}/file_index"])
    t0 = float(eps.loc[ep, f"videos/observation.images.{cam}/from_timestamp"])
    path = fetch(f"videos/observation.images.{cam}/chunk-000/file-{fi:03d}.mp4", D / f"{cam}_file{fi:03d}.mp4")
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t0:.4f}", "-i", str(path), "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 480, 640, 3)


ck = resolve_checkpoint(args.checkpoint)
policy = load_policy(ck, torch.device("cuda"))
pre, post = make_pre_post_processors(policy.config, pretrained_path=ck)
st = {k: v.cuda().float() for k, v in pre.steps[2]._tensor_stats["action"].items()}
MAX_DIM, NB = policy.config.max_action_dim, args.num_samples
g = torch.Generator(device="cuda").manual_seed(args.seed)


@torch.no_grad()
def reconstruction_loss(batch, chunk_norm):
    """ood_signal.py metric 1 for Pi0.5: mean flow-matching MSE over NB (noise, t) draws."""
    bl = {k: (v.expand(NB, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
    images, img_masks = policy._preprocess_images(bl)
    acts = torch.nn.functional.pad(chunk_norm, (0, MAX_DIM - chunk_norm.shape[-1]))[None].expand(NB, -1, -1)
    noise = torch.randn(NB, CHUNK, MAX_DIM, device="cuda", generator=g)
    t = torch.distributions.Beta(1.5, 1.0).sample((NB,)).cuda() * 0.999 + 0.001
    losses = policy.model.forward(images, img_masks, bl["observation.language.tokens"],
                                  bl["observation.language.attention_mask"], acts, noise, t)
    return losses[:, :, :6].float().mean().item()


def run_episode(ep, group):
    out_csv = OUT / "per_episode" / f"ep{ep}_{group}.csv"
    if out_csv.exists():
        print(f"[{group} ep{ep}] already done, reusing {out_csv}", flush=True)
        return pd.read_csv(out_csv)
    S, A = episode_arrays(ep)
    tops, wrists = episode_frames(ep, "top", len(S)), episode_frames(ep, "left_wrist", len(S))
    n = min(len(S), len(tops), len(wrists))
    task = meta[ep]["instruction"]
    print(f"[{group} ep{ep}] {task!r}, {n} frames", flush=True)
    rows, t0 = [], time.time()
    for idx in range(0, n, args.stride):
        batch = pre(to_policy_obs({"observation.state": S[idx], "observation.images.top": tops[idx],
                                   "observation.images.left_wrist": wrists[idx], "task": task}))
        with torch.no_grad():
            if group == "gt":
                chunk = A[idx:idx + CHUNK]
                if len(chunk) < CHUNK:
                    chunk = np.concatenate([chunk, np.repeat(A[-1:], CHUNK - len(chunk), 0)])
                chunk_norm = 2 * (torch.as_tensor(chunk, device="cuda") - st["q01"]) / (st["q99"] - st["q01"] + 1e-8) - 1
            else:
                torch.manual_seed(args.seed * 1_000_000 + ep * 10_000 + idx)
                chunk_norm = policy.predict_action_chunk(batch)[0, :, :6].float()
        rows.append({"frame": idx, "loss": reconstruction_loss(batch, chunk_norm)})
        if idx % 100 == 0:
            print(f"[{group} ep{ep}] frame {idx}/{n} loss={rows[-1]['loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[{group} ep{ep}] done in {time.time() - t0:.0f}s -> {out_csv}", flush=True)
    return df


results = {"gt": {ep: run_episode(ep, "gt") for ep in gt_eps}, "pred": {ep: run_episode(ep, "pred") for ep in pred_eps}}

table = None
for group, per_ep in results.items():
    for ep, df in per_ep.items():
        col = df.set_index("frame")["loss"].rename(f"{group}_ep{ep}")
        table = col.to_frame() if table is None else table.join(col, how="outer")
table = table.sort_index()
for group in ("gt", "pred"):
    cols = [c for c in table.columns if c.startswith(f"{group}_ep")]
    table[f"{group}_min"], table[f"{group}_max"], table[f"{group}_n"] = table[cols].min(1), table[cols].max(1), table[cols].notna().sum(1)
order = ["gt_min", "gt_max", "gt_n"] + [c for c in table.columns if c.startswith("gt_ep")] + \
        ["pred_min", "pred_max", "pred_n"] + [c for c in table.columns if c.startswith("pred_ep")]
table = table[order]
table.index.name = "frame"
table.to_csv(OUT / "state_ood_gt_vs_pred.csv", float_format="%.5f")


def rng_str(lo, hi, k):
    return "n/a" if k == 0 else f"{lo:.4f} - {hi:.4f} (n={k})"


with open(OUT / "state_ood_gt_vs_pred.txt", "w") as f:
    f.write(f"Pi0.5 state-OOD reconstruction loss, Nb={NB}. gt = recorded demo chunks, episodes {gt_eps}; "
            f"predicted = model's own sampled chunks, episodes {pred_eps}\n")
    for frame, r in table.iterrows():
        f.write(f"frame {frame} -> gt range: {rng_str(r.gt_min, r.gt_max, r.gt_n)}, "
                f"predicted range: {rng_str(r.pred_min, r.pred_max, r.pred_n)}\n")

gt_all = table[[c for c in table.columns if c.startswith("gt_ep")]].values.ravel()
pr_all = table[[c for c in table.columns if c.startswith("pred_ep")]].values.ravel()
gt_all, pr_all = gt_all[~np.isnan(gt_all)], pr_all[~np.isnan(pr_all)]
print("\n================ SUMMARY ================")
print(f"gt   (recorded, eps {gt_eps}): mean {gt_all.mean():.4f}  median {np.median(gt_all):.4f}  p5-p95 {np.percentile(gt_all, 5):.4f} - {np.percentile(gt_all, 95):.4f}")
print(f"pred (model,    eps {pred_eps}): mean {pr_all.mean():.4f}  median {np.median(pr_all):.4f}  p5-p95 {np.percentile(pr_all, 5):.4f} - {np.percentile(pr_all, 95):.4f}")
both = table.dropna(subset=["gt_min", "pred_min"])
print(f"frames where the gt and predicted ranges overlap: {100 * ((both.gt_min <= both.pred_max) & (both.pred_min <= both.gt_max)).mean():.1f}% of {len(both)}")
print(f"wrote {OUT / 'state_ood_gt_vs_pred.csv'} and {OUT / 'state_ood_gt_vs_pred.txt'}")
