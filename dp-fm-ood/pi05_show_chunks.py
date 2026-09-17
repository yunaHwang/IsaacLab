"""Print one frame's recorded action chunk next to 3 Pi0.5-sampled chunks, plus the state-OOD
reconstruction loss of each - the sanity check that the "predicted" column really scores the
model's own actions.

    conda activate lerobot_0.6.1_multitask_dit
    python pi05_show_chunks.py --episode 789 [--frame 412] [--num_samples 3]
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
JOINTS = ["pan", "lift", "elbow", "w_flex", "w_roll", "grip"]
parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=789)
parser.add_argument("--frame", type=int, default=None, help="Default: random frame.")
parser.add_argument("--num_samples", type=int, default=3)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out", default=str(HERE / "outputs" / "pi05_show_chunks"))
args = parser.parse_args()

EPD = HERE / "data" / "scrape_so101_stack2cubes"
info = json.loads((EPD / "episodes.json").read_text())[str(args.episode)]
arr = np.load(EPD / "episodes.npz")
S, A = arr[f"state_{args.episode}"], arr[f"action_{args.episode}"]
rng = np.random.default_rng(args.seed)
frame = args.frame if args.frame is not None else int(rng.integers(0, len(S) - CHUNK))
meta = pd.read_parquet(EPD / "raw" / "meta__episodes__chunk-000__file-000.parquet").set_index("episode_index")


def image(cam):
    fi = int(meta.loc[args.episode, f"videos/observation.images.{cam}/file_index"])
    t = float(meta.loc[args.episode, f"videos/observation.images.{cam}/from_timestamp"]) + frame / 30.0
    path = EPD / "raw" / f"videos__observation.images.{cam}__chunk-000__file-{fi:03d}.mp4"
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(480, 640, 3).copy()


ck = resolve_checkpoint("Cache-SCA/Pi0.5-IsaacLab-Multi-Task-1epochs")
policy = load_policy(ck, torch.device("cuda"))
pre, post = make_pre_post_processors(policy.config, pretrained_path=ck)
st = {k: v.cuda().float() for k, v in pre.steps[2]._tensor_stats["action"].items()}
top, wrist = image("top"), image("left_wrist")
batch = pre(to_policy_obs({"observation.state": S[frame], "observation.images.top": top,
                           "observation.images.left_wrist": wrist, "task": info["instruction"]}))

with torch.no_grad():
    gt = torch.as_tensor(A[frame:frame + CHUNK], device="cuda")
    gt_norm = 2 * (gt - st["q01"]) / (st["q99"] - st["q01"] + 1e-8) - 1
    # normalization round trip through the checkpoint's own unnormalizer: must give back gt
    roundtrip_err = (post(gt_norm[None]).float().cuda()[0] - gt).abs().max().item()

    bp = {k: (v.expand(3, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(frame)
    pred_norm = policy.predict_action_chunk(bp)[:, :, :6].float()
    pred = post(pred_norm).float().cuda()

    NB, MAX = args.num_samples, policy.config.max_action_dim
    chunks = torch.cat([gt_norm[None], pred_norm], 0)
    C = chunks.shape[0]
    bl = {k: (v.expand(C * NB, *v.shape[1:]) if torch.is_tensor(v) else v) for k, v in batch.items()}
    images, masks = policy._preprocess_images(bl)
    noise = torch.randn(NB, CHUNK, MAX, device="cuda").repeat(C, 1, 1)
    t = (torch.distributions.Beta(1.5, 1.0).sample((NB,)).cuda() * 0.999 + 0.001).repeat(C)
    l = policy.model.forward(images, masks, bl["observation.language.tokens"], bl["observation.language.attention_mask"],
                             torch.nn.functional.pad(chunks, (0, MAX - 6)).repeat_interleave(NB, 0), noise, t)
    losses = l[:, :, :6].float().mean(dim=(1, 2)).view(C, NB).mean(1).cpu().numpy()

lines = [f"episode {args.episode} frame {frame}: {info['instruction']!r}",
         f"current state (motor units): {np.round(S[frame], 1).tolist()}",
         f"normalization round-trip max error on gt chunk: {roundtrip_err:.2e} (should be ~0)",
         f"losses (Nb={NB}, shared draws): gt={losses[0]:.4f}  pred1={losses[1]:.4f}  pred2={losses[2]:.4f}  pred3={losses[3]:.4f}",
         f"mean |pred - gt| per sample (motor units): {[round(float(x), 2) for x in (pred - gt[None]).abs().mean(dim=(1, 2)).cpu()]}",
         ""]
header = "step | " + " | ".join(f"{name:>28s}" for name in ["GT (recorded)", "pred 1", "pred 2", "pred 3"])
lines += ["actions in LeRobot motor units, joints = " + " ".join(JOINTS), header, "-" * len(header)]
fmt = lambda v: " ".join(f"{x:6.1f}" for x in v)
for k in range(CHUNK):
    lines.append(f"{k:4d} | {fmt(gt[k].cpu().numpy()):>28s} | " + " | ".join(f"{fmt(pred[j, k].cpu().numpy()):>28s}" for j in range(3)))
text = "\n".join(lines)
print(text)
Path(args.out).mkdir(parents=True, exist_ok=True)
(Path(args.out) / f"ep{args.episode}_frame{frame}.txt").write_text(text + "\n")
from PIL import Image
Image.fromarray(np.concatenate([top, wrist], 1)).save(Path(args.out) / f"ep{args.episode}_frame{frame}.png")
print(f"\nsaved {Path(args.out) / f'ep{args.episode}_frame{frame}.txt'} (+ .png of the two camera images)")
