#!/usr/bin/env python
"""Exact runtime trace of how the (48,4,4) patch feature map reaches the DiT.

Prints each real variable from modeling_multi_task_dit.py, in execution order, with its line
number and actual shape. Run twice to compare:

    python dp-fm-ood/trace_patch_flow.py            # STOCK: (768,1,1)
    PATCH=1 python dp-fm-ood/trace_patch_flow.py    # PATCHED: (48,4,4)

CPU-only, random weights, no checkpoint. Safe during training.
"""
import os, sys
from pathlib import Path
import torch

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
sys.path.insert(0, str(BASE / "lerobot/src/lerobot/scripts"))
PATCHED = bool(os.environ.get("PATCH"))
if PATCHED:
    import lerobot_train_frozen as LTF
    # Deliberately hardcoded: this trace forces the patch encoder onto a CLS checkpoint to
    # compare code paths, so it must NOT auto-detect from the checkpoint (that would no-op).
    LTF._install_patch_tokens(4, 48)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.multi_task_dit import modeling_multi_task_dit as M

CKPT = str(BASE / "outputs/run_0902_norm/n100_k6_s180k_lr3e4_crop_aug_sepenc/checkpoints/180000/pretrained_model")
ROOT = str(BASE / "lerobot_dataset_0901_topped_up_300/ID-visuomotor-based")

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path, cfg.device = None, "cpu"
cfg.use_separate_rgb_encoder_per_camera = False   # the branch the running arms take
cfg.num_layers, cfg.hidden_dim = 4, 384

probe = LeRobotDataset(repo_id="ID/visuomotor-based", root=ROOT, episodes=[0])
obs_d = [i / probe.fps for i in range(1 - cfg.n_obs_steps, 1)]
dt = {"observation.state": obs_d, "action": [i / probe.fps for i in range(cfg.horizon)]}
for ck in probe.meta.camera_keys:
    dt[ck] = obs_d
ds = LeRobotDataset(repo_id="ID/visuomotor-based", root=ROOT, delta_timestamps=dt, episodes=[0])
policy = make_policy(cfg, ds_meta=ds.meta).eval()
# The preprocessor is what TOKENISES the task string. Without it the language
# branch contributes 0 and conditioning_vec is 3090 instead of 3858, which the DiT
# rejects: 'mat1 (1x3346) and mat2 (4114x2304)'.
pre, _post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
enc, dit = policy.observation_encoder, policy.noise_predictor

L = []
def say(line, var, val, note=""):
    shp = tuple(val.shape) if torch.is_tensor(val) else val
    L.append(f"  L{line:<4} {var:<26} {str(shp):<22} {note}")

# --- hook the vision encoder (the ONLY thing that differs) ---
ve = enc.vision_encoder
_vf = ve.forward
def vf(x):
    out = _vf(x)
    say(216 if not PATCHED else 238, "images_flat (input)", x, "(b*s*n) images")
    say(219 if not PATCHED else 248, "vision_encoder(...) OUT", out,
        "<<< THE DIFFERENCE" )
    return out
ve.forward = vf

_enc = enc.encode
def enc_fn(batch):
    say(345, "batch[OBS_IMAGES]", batch[M.OBS_IMAGES], "b,s,n,c,h,w")
    out = _enc(batch)
    say(383, "encode() -> conditioning_vec", out, "flatten(start_dim=1)")
    return out
enc.encode = enc_fn

_dit = dit.forward
def dit_fn(x, timestep, conditioning_vec):
    say(0, "x (noisy action chunk)", x, "b,horizon,action_dim")
    say(0, "conditioning_vec", conditioning_vec, "= encode() output")
    out = _dit(x, timestep, conditioning_vec)
    return out
dit.forward = _dit  # keep real one; we hook the block instead

blk = dit.transformer_blocks[0]
_bf = blk.forward
def bf(x, features):
    say(597, "features (= cond_features)", features, "timestep_emb + conditioning_vec")
    mod = blk.adaLN_modulation(features)
    say(534, "adaLN_modulation(features)", mod, "Linear(num_features, 6*hidden)")
    say(535, "  .chunk(6) -> each", mod.chunk(6, dim=1)[0], "shift/scale/gate x2")
    say(541, "x (hidden_seq) in block", x, "b,horizon,hidden_dim")
    return _bf(x, features)
blk.forward = bf

item = ds[0]
batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else ([v] if isinstance(v, str) else v))
         for k, v in item.items()}
for ck in ds.meta.camera_keys:
    if ck in batch and batch[ck].dtype == torch.uint8:
        batch[ck] = batch[ck].float() / 255.0

batch = pre(dict(batch))
print(f"\n=========== {'PATCHED (48,4,4)' if PATCHED else 'STOCK (768,1,1)'} ===========")
with torch.no_grad():
    policy.forward(batch)
print("\n".join(L))
lin = blk.adaLN_modulation[1]
print(f"\n  adaLN Linear: in={lin.in_features}  out={lin.out_features}"
      f"  params={sum(p.numel() for p in lin.parameters())/1e6:.2f}M per block x {cfg.num_layers} blocks")
