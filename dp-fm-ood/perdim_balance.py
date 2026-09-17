#!/usr/bin/env python3
"""Which checkpoint is BALANCED across action dims, not just highest on average?

THE QUESTION
    "R2 ARM 0-5" is a single pooled number, so a checkpoint can post a good arm R^2 while one
    or two dims sit at or below zero and the rest carry it. This ranks checkpoints by how
    EVENLY the per-dim R^2 is spread instead -- "all reasonable all around".

WHY ARM DIMS ONLY (0-5)
    Dim 6 is the gripper, and it scores ~0.87-0.98 held-out from episode phase structure alone
    with no spatial understanding (see generalization_check.py / the mean-collapse note). Left
    in, it dominates every spread statistic: max-min would just be measuring "gripper minus
    worst arm dim" at every checkpoint, which is the same story each time and tells you nothing
    about balance among the dims that actually encode where the cubes are. It is reported
    separately instead.

    Dims 0-5 are the IK-Rel end-effector deltas: 0-2 translation (dx, dy, dz), 3-5 rotation.

THE THREE STATISTICS, AND WHICH ONE ANSWERS THE QUESTION
    min     the WORST dim. This is the maximin reading of "all reasonable" -- a checkpoint is
            only as good as its weakest axis, so ranking by this finds the one with no
            catastrophic dim. This is the primary sort.
    spread  max - min. Low spread = dims agree with each other. On its own it is a TRAP:
            uniformly terrible scores the best possible spread of 0. Only meaningful read
            together with min/mean.
    std     the same idea, less sensitive to a single outlier dim.

Usage:
    python dp-fm-ood/perdim_balance.py                       # default alleps log, held-out
    python dp-fm-ood/perdim_balance.py --split seen
    python dp-fm-ood/perdim_balance.py --log <other txt>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
DEFAULT_LOG = BASE / ("outputs/refpatch_0908/n300_unfrz_sepenc_L6h512_ev20_patch_eefpos"
                      "/r2_progress_0910_alleps.txt")

RE_BLOCK = re.compile(r"^@@ (\S+)\s+(\d+)")
WANT_TAG = "EEFPOS"
RE_PERDIM = re.compile(r"per-dim R2\s*:\s*\[([^\]]*)\]")
RE_ARM = re.compile(r"R2 ARM 0-5\s*:\s*(-?[\d.]+)")

ARM = slice(0, 6)
DIM_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]


def parse(path: Path):
    """-> {step: {"seen": [7 floats], "held": [...], "seen_arm": f, "held_arm": f}}"""
    rows: dict[int, dict] = {}
    step = split = None
    for line in path.read_text(errors="replace").splitlines():
        m = RE_BLOCK.match(line.strip())
        if m:
            if m.group(1) != WANT_TAG:
                step = split = None
                continue
            step = int(m.group(2))
            rows.setdefault(step, {})
            split = None
            continue
        if step is None:
            continue
        if "SEEN (trained on)" in line:
            split = "seen"
            continue
        if "HELD-OUT (never seen)" in line:
            split = "held"
            continue
        m = RE_ARM.search(line)
        if m and split and f"{split}_arm" not in rows[step]:
            rows[step][f"{split}_arm"] = float(m.group(1))
            continue
        m = RE_PERDIM.search(line)
        if m and split and split not in rows[step]:
            rows[step][split] = [float(x) for x in m.group(1).split(",")]
    return {k: v for k, v in sorted(rows.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--split", choices=["held", "seen"], default="held")
    a = p.parse_args()

    rows = parse(a.log)
    rows = {k: v for k, v in rows.items() if a.split in v}
    if not rows:
        raise SystemExit(f"no per-dim rows for split {a.split!r} in {a.log}")

    print(f"{a.log.name}  --  split: {a.split}\n")
    hdr = f"{'step':>7} " + " ".join(f"{n:>7}" for n in DIM_NAMES)
    hdr += f" |{'armR2':>7}{'min':>7}{'max':>7}{'spread':>8}{'std':>7}"
    print(hdr)
    print("-" * len(hdr))

    stats = []
    for step, v in rows.items():
        d = np.array(v[a.split])
        arm = d[ARM]
        s = dict(step=step, arm_r2=v.get(f"{a.split}_arm", float("nan")),
                 mn=arm.min(), mx=arm.max(), spread=arm.max() - arm.min(), sd=arm.std(),
                 dims=d)
        stats.append(s)
        print(f"{step:>7} " + " ".join(f"{x:>7.3f}" for x in d)
              + f" |{s['arm_r2']:>7.3f}{s['mn']:>7.3f}{s['mx']:>7.3f}"
                f"{s['spread']:>8.3f}{s['sd']:>7.3f}")

    print("\nRanked by WORST arm dim (maximin -- 'no dim left behind'):")
    for s in sorted(stats, key=lambda s: -s["mn"])[:5]:
        worst = DIM_NAMES[int(np.argmin(s["dims"][ARM]))]
        print(f"  {s['step']:>7}   min {s['mn']:+.3f} ({worst})   spread {s['spread']:.3f}   "
              f"arm R2 {s['arm_r2']:.3f}")

    print("\nRanked by SMALLEST spread across arm dims (most uniform):")
    for s in sorted(stats, key=lambda s: s["spread"])[:5]:
        print(f"  {s['step']:>7}   spread {s['spread']:.3f}   min {s['mn']:+.3f}   "
              f"arm R2 {s['arm_r2']:.3f}")
    print("  NOTE: uniformly bad scores a perfect spread of 0 -- read this column with min.")

    # How often is each dim the weakest? A dim that is always last is a structural problem,
    # not checkpoint-to-checkpoint noise.
    print("\nHow often each arm dim is the WORST of the six:")
    worst_counts = np.zeros(6, dtype=int)
    for s in stats:
        worst_counts[int(np.argmin(s["dims"][ARM]))] += 1
    for i, n in enumerate(worst_counts):
        if n:
            print(f"  {DIM_NAMES[i]:>4}: {n:>2}/{len(stats)} checkpoints")

    print("\nPer-dim mean across all checkpoints (is a dim systematically weak?):")
    allo = np.stack([s["dims"] for s in stats])
    for i, n in enumerate(DIM_NAMES):
        print(f"  {n:>4}: {allo[:, i].mean():+.3f}   (min {allo[:, i].min():+.3f}, "
              f"max {allo[:, i].max():+.3f})")


if __name__ == "__main__":
    main()
