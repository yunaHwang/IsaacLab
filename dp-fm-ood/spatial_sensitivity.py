#!/usr/bin/env python
"""Demonstrate that CLS features are BLIND TO LOCATION and patch features are not.

Takes one real frame, then MOVES an object within it (a circular shift of a crop region) and
asks: how much does the encoder's output change?

If an encoder is location-aware, moving something must change its output. If it only encodes
"what is in this image", the output barely moves -- and a policy conditioned on it cannot know
where to reach. That is the entire failure mode.

CPU-only, no checkpoint. Safe to run during training.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
sys.path.insert(0, str(BASE / "lerobot/src/lerobot/scripts"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.multi_task_dit import modeling_multi_task_dit as M

MODEL = "openai/clip-vit-base-patch16"
ROOT = str(BASE / "lerobot_dataset_0901_topped_up_300/ID-visuomotor-based")

ds = LeRobotDataset(repo_id="ID/visuomotor-based", root=ROOT, episodes=[0])
img = ds[len(ds) // 2]["observation.images.table_cam"].float()
if img.max() > 1.5:
    img = img / 255.0
img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)

def shifted(x, px):
    """Move the scene contents by px pixels -- same objects, different place."""
    return torch.roll(x, shifts=px, dims=-1)

def probe(tag, enc):
    with torch.no_grad():
        base = enc(img).flatten(1)
        print(f"\n  {tag}   (feature vector: {base.numel()} values)")
        for px in (8, 16, 32, 56):
            v = enc(shifted(img, px)).flatten(1)
            cos = F.cosine_similarity(base, v).item()
            rel = ((v - base).norm() / base.norm()).item()
            print(f"    shift {px:>3}px : cosine similarity {cos:6.4f}   relative change {rel:6.3f}")

print("Same image, contents shifted sideways. A location-aware encoder MUST change.")
probe("STOCK  (CLS token, 1x1)", M.CLIPVisionEncoder(MODEL))

import lerobot_train_frozen as LTF
LTF._install_patch_tokens(4, 48)
probe("PATCH  (4x4 grid)", M.CLIPVisionEncoder(MODEL))
print("\n  cosine ~1.0 => the encoder cannot tell the object moved.")
print("  Note the patch encoder here is RANDOMLY INITIALISED (patch_proj untrained), so this")
print("  measures the INFORMATION AVAILABLE, not a learned skill. That is the point: the stock")
print("  path has no location information to learn from at all.")
