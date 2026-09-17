"""Does Pi0.5's self-scored state OOD react to OOD observations?

Take clean training frames and deliberately corrupt the observation. For each condition the model
samples its own chunks from the corrupted observation and scores them (reconstruction loss, shared
Nb draws). If pred_loss / sample spread rise vs "clean", self-scored state OOD carries signal for
this checkpoint; if they stay at the clean level, it is blind.

Conditions: clean, black_images, noise_images, other_episode_images, swapped_cameras,
            state_shift_+40, other_episode_state, wrong_prompt

    conda activate lerobot_0.6.1_multitask_dit
    python pi05_state_ood_perturbation_test.py --episode 789 --other_episode 752 --frames 100 300 500 700 900
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pi05_server import load_policy, resolve_checkpoint, to_policy_obs
from lerobot.policies import make_pre_post_processors

CHUNK = 16
parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=789)
parser.add_argument("--other_episode", type=int, default=752)
parser.add_argument("--frames", type=int, nargs="+", default=[100, 300, 500, 700, 900])
parser.add_argument("--num_samples", type=int, default=8, help="Nb (noise, t) draws per loss.")
parser.add_argument("--num_predictions", type=int, default=3)
parser.add_argument("--checkpoint", default=str(HERE / "checkpoints" / "pi05_isaaclab_so101_multitask_1ep"))
parser.add_argument("--out_dir", default=str(HERE / "outputs" / "pi05_state_ood_perturbation"))
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

EPD = HERE / "data" / "scrape_so101_stack2cubes"
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
info = json.loads((EPD / "episodes.json").read_text())
arr = np.load(EPD / "episodes.npz")
meta = pd.read_parquet(EPD / "raw" / "meta__episodes__chunk-000__file-000.parquet").set_index("episode_index")


def image(ep, cam, frame):
    fi = int(meta.loc[ep, f"videos/observation.images.{cam}/file_index"])
    t = float(meta.loc[ep, f"videos/observation.images.{cam}/from_timestamp"]) + frame / 30.0
    path = EPD / "raw" / f"videos__observation.images.{cam}__chunk-000__file-{fi:03d}.mp4"
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(480, 640, 3).copy()


ck = resolve_checkpoint(args.checkpoint)
policy = load_policy(ck, torch.device("cuda"))
pre, post = make_pre_post_processors(policy.config, pretrained_path=ck)
st = {k: v.cuda().float() for k, v in pre.steps[2]._tensor_stats["action"].items()}
MAX_DIM, NB, NP = policy.config.max_action_dim, args.num_samples, args.num_predictions
g = torch.Generator(device="cuda").manual_seed(args.seed)
rng = np.random.default_rng(args.seed)


@torch.no_grad()
def losses_on_shared_draws(batch, chunks_norm):
    C = chunks_norm.shape[0]
    bl = {k: (v.expand(C * NB, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
    images, img_masks = policy._preprocess_images(bl)
    acts = torch.nn.functional.pad(chunks_norm, (0, MAX_DIM - 6)).repeat_interleave(NB, 0)
    noise = torch.randn(NB, CHUNK, MAX_DIM, device="cuda", generator=g).repeat(C, 1, 1)
    t = (torch.distributions.Beta(1.5, 1.0).sample((NB,)).cuda() * 0.999 + 0.001).repeat(C)
    l = policy.model.forward(images, img_masks, bl["observation.language.tokens"],
                             bl["observation.language.attention_mask"], acts, noise, t)
    return l[:, :, :6].float().mean(dim=(1, 2)).view(C, NB).mean(1).cpu().numpy()


S, A = arr[f"state_{args.episode}"], arr[f"action_{args.episode}"]
S_other = arr[f"state_{args.other_episode}"]
task = info[str(args.episode)]["instruction"]
rows = []
for frame in args.frames:
    top, wrist = image(args.episode, "top", frame), image(args.episode, "left_wrist", frame)
    otop, owrist = image(args.other_episode, "top", frame), image(args.other_episode, "left_wrist", frame)
    state = S[frame]
    shifted = state.copy()
    shifted[:5] = np.clip(shifted[:5] + 40.0, -100, 100)
    conditions = {
        "clean": (state, top, wrist, task),
        "black_images": (state, np.zeros_like(top), np.zeros_like(wrist), task),
        "noise_images": (state, rng.integers(0, 256, top.shape, dtype=np.uint8), rng.integers(0, 256, wrist.shape, dtype=np.uint8), task),
        "other_episode_images": (state, otop, owrist, task),
        "swapped_cameras": (state, wrist, top, task),
        "state_shift_+40": (shifted, top, wrist, task),
        "other_episode_state": (S_other[min(frame, len(S_other) - 1)], top, wrist, task),
        "wrong_prompt": (state, top, wrist, "Press the red button."),
    }
    gt = torch.as_tensor(A[frame:frame + CHUNK], device="cuda")
    gt_norm = 2 * (gt - st["q01"]) / (st["q99"] - st["q01"] + 1e-8) - 1
    for name, (s, im_top, im_wrist, prompt) in conditions.items():
        batch = pre(to_policy_obs({"observation.state": s, "observation.images.top": im_top,
                                   "observation.images.left_wrist": im_wrist, "task": prompt}))
        with torch.no_grad():
            bp = {k: (v.expand(NP, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
            torch.manual_seed(args.seed * 1_000_000 + frame)  # same sampling noise across conditions
            pred_norm = policy.predict_action_chunk(bp)[:, :, :6].float()
            pred = post(pred_norm).float().cuda()
            losses = losses_on_shared_draws(batch, torch.cat([gt_norm[None], pred_norm], 0))
        first = pred[:, 0].mean(0).cpu().numpy()
        row = {"frame": frame, "condition": name, "gt_loss": losses[0],
               **{f"pred_loss_{j + 1}": losses[j + 1] for j in range(NP)},
               "pred_loss_mean": float(np.mean(losses[1:])),
               "sample_spread": float(pred.std(0).mean()),
               "pred_vs_gt_err": float((pred - gt[None]).abs().mean()),
               **{f"pred_a0_{n}": float(v) for n, v in zip(["pan", "lift", "elbow", "wflex", "wroll", "grip"], first)}}
        rows.append(row)
        print(f"frame {frame:4d} {name:22s} pred_loss={np.round(losses[1:], 4)} spread={row['sample_spread']:.2f} "
              f"gt_loss={losses[0]:.3f} first_action={np.round(first, 1)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / f"ep{args.episode}_perturbations.csv", index=False, float_format="%.5f")
summary = df.groupby("condition", sort=False)[["pred_loss_mean", "sample_spread", "gt_loss", "pred_vs_gt_err",
                                               "pred_a0_lift", "pred_a0_elbow", "pred_a0_grip"]].mean()
clean = summary.loc["clean", "pred_loss_mean"]
summary["pred_loss_vs_clean"] = summary["pred_loss_mean"] / clean
print(f"\n================ SUMMARY (episode {args.episode}, frames {args.frames}, Nb={NB}) ================")
print(summary.round(4).to_string())
summary.to_csv(OUT / f"ep{args.episode}_perturbations_summary.csv", float_format="%.5f")
print(f"\nwrote {OUT / f'ep{args.episode}_perturbations.csv'} and _summary.csv")
