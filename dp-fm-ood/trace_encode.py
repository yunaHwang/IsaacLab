#!/usr/bin/env python
"""Trace the tensors inside ObservationEncoder.encode(), with and without PATCH_TOKENS.

Answers "what actually flows into the DiT, and where does the CLS bottleneck bite?" by wrapping
the real encode() (modeling_multi_task_dit.py:337) and reporting every piece it concatenates.

Run it twice to compare:
    python dp-fm-ood/trace_encode.py            # stock: CLS token
    PATCH=1 python dp-fm-ood/trace_encode.py    # patched: 4x4 grid

CPU-only, random-init weights (shapes do not depend on weights), no checkpoint needed -- safe to
run while training is in flight.
"""
import os, sys
from pathlib import Path
import torch

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
sys.path.insert(0, str(BASE / "lerobot/src/lerobot/scripts"))

if os.environ.get("PATCH"):
    import lerobot_train_frozen as LTF
    LTF._install_patch_tokens(int(os.environ.get("PATCH_GRID", "4")),
                              int(os.environ.get("PATCH_DIM", "48")))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy
from lerobot.policies.multi_task_dit import modeling_multi_task_dit as M

CKPT = str(BASE / "outputs/run_0902_norm/n100_k6_s180k_lr3e4_crop_aug_sepenc/checkpoints/180000/pretrained_model")
ROOT = str(BASE / "lerobot_dataset_0901_topped_up_300/ID-visuomotor-based")

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.pretrained_path = None          # random init: we only care about SHAPES
cfg.device = "cpu"
# Match the arms actually running (run_reg_frozen_0907.sh), not the s180k checkpoint.
cfg.use_separate_rgb_encoder_per_camera = False
cfg.num_layers, cfg.hidden_dim = 4, 384

probe = LeRobotDataset(repo_id="ID/visuomotor-based", root=ROOT, episodes=[0])
fps = probe.fps
obs_d = [i / fps for i in range(1 - cfg.n_obs_steps, 1)]
dt = {"observation.state": obs_d, "action": [i / fps for i in range(cfg.horizon)]}
for ck in probe.meta.camera_keys:
    dt[ck] = obs_d
ds = LeRobotDataset(repo_id="ID/visuomotor-based", root=ROOT, delta_timestamps=dt, episodes=[0])
policy = make_policy(cfg, ds_meta=ds.meta).eval()

enc = policy.observation_encoder
_orig = enc.encode

def traced(batch):
    print(f"\n  {'INPUT to encode()':<34}")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"    {k:<40} {tuple(v.shape)}")
    ve = enc.vision_encoder or (enc.vision_encoders[0] if enc.vision_encoders else None)
    if ve is not None:
        print(f"\n    vision_encoder.get_output_shape() = {ve.get_output_shape()}")
    out = _orig(batch)
    print(f"\n  {'conditioning_dim (declared)':<40} {enc.conditioning_dim}")
    print(f"  {'encode() OUTPUT (what the DiT sees)':<40} {tuple(out.shape)}")
    return out

enc.encode = traced

item = ds[0]
batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else ([v] if isinstance(v, str) else v))
         for k, v in item.items()}
for ck in ds.meta.camera_keys:
    if ck in batch and batch[ck].dtype == torch.uint8:
        batch[ck] = batch[ck].float() / 255.0
batch[M.OBS_IMAGES] = torch.stack([batch[k] for k in cfg.image_features], dim=-4)

mode = "PATCH_TOKENS (4x4 grid)" if os.environ.get("PATCH") else "STOCK (CLS token only)"
print(f"\n================ {mode} ================")
with torch.no_grad():
    enc.encode(batch)

dit = policy.noise_predictor
adaln = sum(p.numel() for n, p in dit.named_parameters() if "adaLN" in n)
print(f"  {'DiT cond_dim (timestep+conditioning)':<40} {dit.cond_dim}")
print(f"  {'adaLN params across all blocks':<40} {adaln/1e6:.1f}M")
print(f"  {'total policy params':<40} {sum(p.numel() for p in policy.parameters())/1e6:.1f}M")
