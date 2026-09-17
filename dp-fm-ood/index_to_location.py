#!/usr/bin/env python
"""Show that a flattened index corresponds to a FIXED image location (and that CLS has none).

Puts a bright blob in one image region, encodes, flattens, and asks which vector indices moved.
For (48,4,4) the flatten layout is index = c*16 + row*4 + col, so cell (r,c) owns the fixed
index set {k*16 + r*4 + c}. If the blob's location determines WHICH indices change, the DiT's
per-index weights can specialise to locations. CPU-only, random patch_proj.
"""
import sys
from pathlib import Path
import torch

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
sys.path.insert(0, str(BASE / "lerobot/src/lerobot/scripts"))
from lerobot.policies.multi_task_dit import modeling_multi_task_dit as M

MODEL = "openai/clip-vit-base-patch16"
def img_with_blob(r, c):
    """224px image, bright square centred in cell (r,c) of a 4x4 grid."""
    x = torch.full((1, 3, 224, 224), 0.45)
    s = 224 // 4
    x[:, :, r*s:(r+1)*s, c*s:(c+1)*s] = 0.95
    return x

def which_cells_moved(enc, tag, spatial):
    base = enc(torch.full((1, 3, 224, 224), 0.45)).flatten(1)[0]
    print(f"\n  {tag}")
    for (r, c) in [(0, 0), (0, 3), (3, 0), (3, 3)]:
        v = enc(img_with_blob(r, c)).flatten(1)[0]
        d = (v - base).abs()
        top = torch.topk(d, 60).indices            # the 60 most-changed indices
        if spatial:
            cells = [(int(i) % 16 // 4, int(i) % 16 % 4) for i in top]
            from collections import Counter
            win, n = Counter(cells).most_common(1)[0]
            print(f"    blob in cell {(r,c)} -> most-changed indices map to cell {win}"
                  f"  ({n}/60 of them)   {'MATCH' if win == (r, c) else 'mismatch'}")
        else:
            print(f"    blob in cell {(r,c)} -> indices carry no cell mapping at all "
                  f"(mean |delta| {d.mean():.4f})")

print("A blob is placed in one 4x4 cell. Which flattened indices react?")
which_cells_moved(M.CLIPVisionEncoder(MODEL), "STOCK (768,1,1) - one global vector", spatial=False)
import lerobot_train_frozen as LTF
LTF._install_patch_tokens(4, 48)
which_cells_moved(M.CLIPVisionEncoder(MODEL), "PATCHED (48,4,4) - index = c*16 + row*4 + col", spatial=True)
