#!/usr/bin/env python3
"""Compare a rollout's raw observation.state against the training range it was normalized with.

Usage:
    STATE_LOG=/tmp/rollout1.csv python run_policy_fm.py ...      # produce the log
    python check_state_log.py /tmp/rollout1.csv                  # then read it
    python check_state_log.py /tmp/rollout1.csv --n 300          # against the N=300 stats

The policy normalizes observation.state with MIN_MAX against the training set's per-dim min/max
(`meta/stats.json`). A rollout value outside that range normalizes outside [-1, 1] and is an
out-of-distribution input the model never saw. Dims with a very narrow training range are the
ones to watch: joint_7 spans ~0.024 and joint_8 ~0.026 in the N=50 set, so MIN_MAX stretches
them ~80x and a tiny offset becomes a full-scale swing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")

parser = argparse.ArgumentParser()
parser.add_argument("log", help="CSV written by run_policy_fm.py with STATE_LOG set")
parser.add_argument("--n", default="50", choices=["50", "100", "300"],
                    help="which training set's stats to compare against (default: 50)")
args = parser.parse_args()

stats_path = BASE / f"lerobot_dataset_0901_topped_up_{args.n}/ID-visuomotor-based/meta/stats.json"
stats = json.load(open(stats_path))["observation.state"]
lo, hi = np.array(stats["min"]), np.array(stats["max"])

df = pd.read_csv(args.log)
print(f"{len(df)} steps from {args.log}   vs N={args.n} training stats\n")
print(f"{'dim':>8} {'rollout min':>12} {'rollout max':>12} {'train min':>11} {'train max':>11} "
      f"{'train range':>12} {'% OOR':>7} {'amplif.':>8}")

flagged = []
for i, col in enumerate(df.columns):
    v = df[col].to_numpy()
    span = hi[i] - lo[i]
    oor = float(((v < lo[i]) | (v > hi[i])).mean() * 100)
    # MIN_MAX maps a training span to width 2, so this is the factor applied to raw noise.
    amp = 2.0 / span if span > 0 else float("inf")
    mark = "  <-- " if oor > 0.5 else ""
    print(f"{col:>8} {v.min():>12.5f} {v.max():>12.5f} {lo[i]:>11.5f} {hi[i]:>11.5f} "
          f"{span:>12.5f} {oor:>6.1f}% {amp:>8.1f}x{mark}")
    if oor > 0.5:
        flagged.append((col, oor, amp))

print()
if flagged:
    print("OUT OF DISTRIBUTION:")
    for col, oor, amp in flagged:
        print(f"  {col}: {oor:.1f}% of steps outside the training range, "
              f"amplified {amp:.0f}x by MIN_MAX")
    print("\nThese dims are fed to the policy as values outside [-1, 1], which it never saw in\n"
          "training. Narrow-range dims are the usual offenders.")
else:
    print("All dims stayed within their training range -- state normalization is not the problem.")
