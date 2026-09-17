"""Pi0.5 state-OOD reconstruction loss on the SAME training episode, recorded vs predicted actions.

For every frame of each episode (same observation for all four scores):
    gt_loss        loss of (obs, RECORDED next-16 action chunk)
    pred_loss_1..3 loss of (obs, MODEL'S OWN sampled chunk), 3 independent samples
Each loss is the flow-matching reconstruction loss (ood_signal.py metric 1) averaged over Nb
(noise, t) draws; gt and the 3 predictions of a frame share the same Nb draws, so the columns
differ only by the action being scored.

Output (--out_dir):
    ep<N>.csv        frame, gt_loss, pred_loss_1, pred_loss_2, pred_loss_3   (written per episode)
    all_episodes.csv same columns plus episode
    episodes.json    episodes, instructions, Nb

    conda activate lerobot_0.6.1_multitask_dit
    python pi05_state_ood_same_episode.py --episodes 789 752 680 --num_samples 3
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pi05_server import load_policy, resolve_checkpoint, to_policy_obs
from lerobot.policies import make_pre_post_processors

REPO = "https://huggingface.co/datasets/Cache-SCA/Isaaclab-so101_11task_baseCaP_3300epi/resolve/main"
CHUNK = 16

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, nargs="+", default=[789, 752, 680])
parser.add_argument("--num_samples", type=int, default=3, help="Nb (noise, t) draws per loss.")
parser.add_argument("--num_predictions", type=int, default=3, help="Model action chunks sampled per frame.")
parser.add_argument("--episodes_dir", default=str(HERE / "data" / "scrape_so101_stack2cubes"),
                    help="Output of extract_scrape_stack2cubes_episodes.py (episodes.json/.npz).")
parser.add_argument("--video_dir", default=str(HERE / "data" / "scrape_so101_stack2cubes" / "raw"),
                    help="Where episode videos are cached (downloaded if missing).")
parser.add_argument("--out_dir", default=str(HERE / "outputs" / "pi05_state_ood_same_episode"))
parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent / "checkpoints" / "pi05_isaaclab_so101_multitask_1ep"))
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

EPD, VID, OUT = Path(args.episodes_dir), Path(args.video_dir), Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
VID.mkdir(parents=True, exist_ok=True)
info = json.loads((EPD / "episodes.json").read_text())
arrays = np.load(EPD / "episodes.npz")
eps_meta = pd.read_parquet(EPD / "raw" / "meta__episodes__chunk-000__file-000.parquet").set_index("episode_index")
(OUT / "episodes.json").write_text(json.dumps({
    "episodes": args.episodes, "num_samples": args.num_samples, "num_predictions": args.num_predictions,
    "instructions": {str(e): info[str(e)]["instruction"] for e in args.episodes}}, indent=2))


def episode_frames(ep, cam, n):
    fi = int(eps_meta.loc[ep, f"videos/observation.images.{cam}/file_index"])
    t0 = float(eps_meta.loc[ep, f"videos/observation.images.{cam}/from_timestamp"])
    path = VID / f"videos__observation.images.{cam}__chunk-000__file-{fi:03d}.mp4"
    if not path.exists():
        subprocess.run(["curl", "-sfL", "-o", str(path), f"{REPO}/videos/observation.images.{cam}/chunk-000/file-{fi:03d}.mp4"], check=True)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t0:.4f}", "-i", str(path), "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 480, 640, 3)


ck = resolve_checkpoint(args.checkpoint)
policy = load_policy(ck, torch.device("cuda"))
pre, _ = make_pre_post_processors(policy.config, pretrained_path=ck)
st = {k: v.cuda().float() for k, v in pre.steps[2]._tensor_stats["action"].items()}
MAX_DIM, NB, NP = policy.config.max_action_dim, args.num_samples, args.num_predictions
g = torch.Generator(device="cuda").manual_seed(args.seed)


@torch.no_grad()
def losses_on_shared_draws(batch, chunks_norm):
    """chunks_norm [C, 16, 6] -> [C] reconstruction losses, every chunk on the same NB (noise, t) draws."""
    C = chunks_norm.shape[0]
    bl = {k: (v.expand(C * NB, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
    images, img_masks = policy._preprocess_images(bl)
    acts = torch.nn.functional.pad(chunks_norm, (0, MAX_DIM - 6)).repeat_interleave(NB, 0)
    noise = torch.randn(NB, CHUNK, MAX_DIM, device="cuda", generator=g).repeat(C, 1, 1)
    t = (torch.distributions.Beta(1.5, 1.0).sample((NB,)).cuda() * 0.999 + 0.001).repeat(C)
    l = policy.model.forward(images, img_masks, bl["observation.language.tokens"],
                             bl["observation.language.attention_mask"], acts, noise, t)
    return l[:, :, :6].float().mean(dim=(1, 2)).view(C, NB).mean(1).cpu().numpy()


all_rows = []
for ep in args.episodes:
    out_csv = OUT / f"ep{ep}.csv"
    S, A = arrays[f"state_{ep}"], arrays[f"action_{ep}"]
    tops, wrists = episode_frames(ep, "top", len(S)), episode_frames(ep, "left_wrist", len(S))
    n = min(len(S), len(tops), len(wrists))
    task = info[str(ep)]["instruction"]
    print(f"[ep{ep}] {task!r}, {n} frames", flush=True)
    rows, t0 = [], time.time()
    csv_f = open(out_csv, "w", buffering=1)  # rows appear as they are computed
    csv_f.write("frame,gt_loss," + ",".join(f"pred_loss_{j + 1}" for j in range(NP)) + "\n")
    for idx in range(n):
        batch = pre(to_policy_obs({"observation.state": S[idx], "observation.images.top": tops[idx],
                                   "observation.images.left_wrist": wrists[idx], "task": task}))
        with torch.no_grad():
            gt = A[idx:idx + CHUNK]
            if len(gt) < CHUNK:  # end of episode: repeat last recorded action
                gt = np.concatenate([gt, np.repeat(A[-1:], CHUNK - len(gt), 0)])
            gt_norm = 2 * (torch.as_tensor(gt, device="cuda") - st["q01"]) / (st["q99"] - st["q01"] + 1e-8) - 1
            bp = {k: (v.expand(NP, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.manual_seed(args.seed * 1_000_000 + ep * 10_000 + idx)
            pred_norm = policy.predict_action_chunk(bp)[:, :, :6].float()          # [NP, 16, 6]
            losses = losses_on_shared_draws(batch, torch.cat([gt_norm[None], pred_norm], 0))
        row = {"frame": idx, "gt_loss": float(losses[0])}
        row.update({f"pred_loss_{j + 1}": float(losses[j + 1]) for j in range(NP)})
        rows.append(row)
        csv_f.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
        if idx % 100 == 0:
            print(f"[ep{ep}] frame {idx}/{n} gt={losses[0]:.4f} pred={np.round(losses[1:], 4)} ({time.time() - t0:.0f}s)", flush=True)
    csv_f.close()
    df = pd.DataFrame(rows)
    all_rows.append(df.assign(episode=ep))
    print(f"[ep{ep}] done in {time.time() - t0:.0f}s -> {out_csv}", flush=True)

alldf = pd.concat(all_rows)[["episode", "frame", "gt_loss"] + [f"pred_loss_{j + 1}" for j in range(NP)]]
alldf.to_csv(OUT / "all_episodes.csv", index=False, float_format="%.5f")
P = alldf[[f"pred_loss_{j + 1}" for j in range(NP)]].values
print("\n================ SUMMARY ================")
print(f"episodes {args.episodes}, Nb={NB}, {NP} predictions per frame, {len(alldf)} frames")
print(f"gt loss   mean {alldf.gt_loss.mean():.4f}  median {alldf.gt_loss.median():.4f}")
print(f"pred loss mean {P.mean():.4f}  median {np.median(P):.4f}")
print(f"frames where gt loss is inside [min, max] of the {NP} predicted losses: "
      f"{100 * ((alldf.gt_loss >= P.min(1)) & (alldf.gt_loss <= P.max(1))).mean():.1f}%")
print(f"wrote {OUT / 'all_episodes.csv'}")
