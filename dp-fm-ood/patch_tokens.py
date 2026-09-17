#!/usr/bin/env python3
"""Reinstall the PATCH_TOKENS vision encoder before a checkpoint is loaded.

WHY THIS EXISTS
---------------
`_install_patch_tokens` (lerobot/src/lerobot/scripts/lerobot_train_frozen.py) monkeypatches
`CLIPVisionEncoder.__init__/forward/get_output_shape` at TRAINING time. Any script that builds
a policy through the stock `make_policy` / `from_pretrained` gets the CLS-only encoder instead,
and lerobot loads with `strict=False` (policies/pretrained.py), so the four extra tensors

    observation_encoder.vision_encoder.patch_proj.{weight,bias}
    observation_encoder.vision_encoder.patch_cls_proj.{weight,bias}

are SILENTLY DROPPED. The model then runs the CLS path it was never trained for and produces
garbage -- with no error and no shape mismatch.

There is no shape mismatch BY CONSTRUCTION: PATCH_DIM * PATCH_GRID^2 == 48 * 4 * 4 == 768 ==
CLIP's embed_dim was chosen so `conditioning_dim` stays 3858 and base-vs-patch is a controlled
comparison. Identical conditioning width means every DiT tensor matches, so the load is clean.

This bug cost ~36h of the 0907 base-vs-patch comparison: the patch arm scored -0.451/-0.438 arm
R^2 when it actually scores +0.073/-0.186 (better than base on both splits).

USAGE -- call BEFORE make_policy()/from_pretrained(), since it patches the CLASS:

    from patch_tokens import maybe_install_patch_tokens
    maybe_install_patch_tokens(ckpt_dir)      # no-op for CLS checkpoints
    policy = make_policy(cfg, ds_meta=ds.meta)
"""

from __future__ import annotations

import json
import logging
import struct
import sys
from pathlib import Path

LEROBOT_SCRIPTS = Path(__file__).resolve().parents[1] / "lerobot/src/lerobot/scripts"

_INSTALLED: tuple[int, int] | None = None


def patch_geometry(ckpt: str | Path) -> tuple[int, int] | None:
    """(grid, dim) if `ckpt` is a PATCH_TOKENS checkpoint, else None.

    Read straight out of the safetensors header, so this costs one small read and never
    builds a model. BOTH values are recovered from the weights because the config does NOT
    record PATCH_TOKENS -- base and patch config.json are byte-identical:

        dim  = <any>.patch_proj.weight.shape[0]           # out-channels of the 1x1 conv
        grid = sqrt(embed_dim / dim)                      # c*h*w is pinned to embed_dim
    """
    ckpt = Path(ckpt)
    f = ckpt / "model.safetensors"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))

    # Match by SUFFIX, not by an exact key. With use_separate_rgb_encoder_per_camera=true the
    # module is `observation_encoder.vision_encoders.0` / `.1` (ModuleList), not the singular
    # `observation_encoder.vision_encoder` -- an exact-key test silently misses every patch+
    # sepenc checkpoint and reintroduces the strict=False drop this module exists to prevent.
    keys = sorted(k for k in hdr if k.endswith(".patch_proj.weight"))
    if not keys:
        return None
    shapes = {tuple(hdr[k]["shape"]) for k in keys}
    if len(shapes) != 1:
        raise RuntimeError(f"per-camera patch_proj shapes disagree: "
                           + ", ".join(f"{k}->{hdr[k]['shape']}" for k in keys))

    dim, embed_dim = hdr[keys[0]]["shape"][:2]
    grid = int(round((embed_dim / dim) ** 0.5))
    if dim * grid * grid != embed_dim:
        raise RuntimeError(f"cannot infer PATCH_GRID from {key} shape {hdr[key]['shape']}")
    return grid, dim


def maybe_install_patch_tokens(ckpt: str | Path, quiet: bool = False) -> str | None:
    """Install the patch-token encoder if `ckpt` needs it. Returns a note, or None for a no-op.

    Idempotent. `_install_patch_tokens` captures the CURRENT `__init__` as `_orig_init`, so
    calling it twice nests the patch -- and callers legitimately load more than once (e.g.
    generalization_check.score() loads once per split).
    """
    global _INSTALLED

    geom = patch_geometry(ckpt)
    if geom is None:
        return None
    grid, dim = geom

    if _INSTALLED == geom:
        return None
    if _INSTALLED is not None:
        raise RuntimeError(
            f"patch geometry changed in-process: {_INSTALLED} -> {geom}. The encoder class is "
            "patched globally and cannot be un-patched; load one geometry per process."
        )

    sys.path.insert(0, str(LEROBOT_SCRIPTS))
    import lerobot_train_frozen as LTF

    LTF._install_patch_tokens(grid, dim)
    _INSTALLED = geom
    note = f"PATCH_TOKENS encoder reinstalled: {dim}-d x {grid}x{grid} grid per camera"
    if not quiet:
        logging.info("[patch] %s", note)
    return note


def assert_fully_loaded(policy, ckpt: str | Path) -> None:
    """Fail loudly if any checkpoint tensor did not land on the policy.

    The backstop for the whole class of bug: strict=False means a silent partial load is the
    default everywhere in lerobot. Cheap -- compares key sets, not values.
    """
    f = Path(ckpt) / "model.safetensors"
    if not f.exists():
        return
    with open(f, "rb") as fh:
        hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    ck = {k for k in hdr if k != "__metadata__"}
    have = set(policy.state_dict().keys())
    missing = sorted(ck - have)
    if missing:
        raise RuntimeError(
            f"{len(missing)} checkpoint tensor(s) were NOT loaded into the policy "
            f"(strict=False hid this): {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
