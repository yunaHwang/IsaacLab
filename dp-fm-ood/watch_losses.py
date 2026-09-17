#!/usr/bin/env python3
"""Track train vs eval loss per eval interval, for a live or finished lerobot run.

WHY THIS EXISTS
    The eval/train gap is the only cheap signal that says whether a run has started
    overfitting, and lerobot prints the two numbers in DIFFERENT log lines with different step
    formats -- eval as an exact `step 12000: eval_loss=...`, train as a rounded
    `step:12K ... epch:5.91 loss:...` emitted every log_freq steps. Eyeballing them together is
    error prone: grepping `step:12K` returns every line from 12000-12999, so the naive match
    pairs an eval at 12000 with a train loss from ~800 steps later.

    This pairs them on EPOCH, which lerobot computes itself and prints at full precision, so the
    two columns describe the same point in training.

READ IT WITH THE CAVEAT
    Eval loss is flow-matching noise-prediction MSE over all 7 action dims. The gripper dim
    alone reaches held-out R^2 ~0.81-0.85 from phase structure, with no spatial understanding at
    all, so a large share of this number is predictable regardless. A mean-collapsed policy
    scores nearly the SAME on both splits -- so a small gap is NOT evidence of generalization.
    Only held-out arm R^2 (generalization_check.py) discriminates that. What this view is good
    for is the opposite: a RISING eval loss is real evidence of overfitting.

Usage:
    python dp-fm-ood/watch_losses.py                       # every run under outputs/refpatch_0908
    python dp-fm-ood/watch_losses.py path/to/run.log       # one specific log
    python dp-fm-ood/watch_losses.py --watch 120           # refresh every 120 s
    python dp-fm-ood/watch_losses.py --out FILE            # also write the table to FILE
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
DEFAULT_GLOB = str(BASE / "outputs/refpatch_0908/*.log")

# `INFO ... ot_train.py:720 step 12000: eval_loss=0.1289`
RE_EVAL = re.compile(r"step (\d+): eval_loss=([0-9.]+)")
# `INFO ... ot_train.py:689 step:12K smpl:... ep:... epch:5.91 loss:0.120 grdn:0.317 ...`
RE_TRAIN = re.compile(r"epch:([0-9.]+)\s+loss:([0-9.]+)(?:\s+grdn:([0-9.]+))?")
RE_CFG = {
    "batch_size": re.compile(r"'batch_size':\s*(\d+)"),
    "eval_split": re.compile(r"'eval_split':\s*([0-9.]+)"),
    "steps": re.compile(r"'steps':\s*(\d+)"),
    "save_freq": re.compile(r"'save_freq':\s*(\d+)"),
    "root": re.compile(r"'root':\s*'([^']+)'"),
}


def parse(log: Path):
    text = log.read_text(errors="replace")
    cfg = {}
    for k, rx in RE_CFG.items():
        m = rx.search(text)
        if m:
            cfg[k] = m.group(1)

    evals = [(int(s), float(v)) for s, v in RE_EVAL.findall(text)]
    # Train lines carry a precise epoch; keep them ordered so we can pair by nearest epoch.
    trains = [(float(e), float(l), float(g) if g else None)
              for e, l, g in RE_TRAIN.findall(text)]
    trains.sort(key=lambda t: t[0])
    return cfg, evals, trains


def dataset_frames(cfg):
    """Training-set frame count, from the dataset's own metadata and the run's eval_split."""
    root = cfg.get("root")
    if not root:
        return None, None, None
    info = Path(root if Path(root).is_absolute() else BASE / root) / "meta/info.json"
    if not info.exists():
        return None, None, None
    d = json.loads(info.read_text())
    tot_e, tot_f = d["total_episodes"], d["total_frames"]
    split = float(cfg.get("eval_split", 0) or 0)
    # Mirrors make_train_eval_datasets: the LAST ceil(n*split) episodes are held out.
    import math
    held = math.ceil(tot_e * split)
    train_e = tot_e - held
    return train_e, tot_f * train_e / tot_e, held


def nearest_train(trains, epoch):
    """Train loss at the epoch closest to `epoch` -- the pairing this whole script exists for."""
    if not trains:
        return None, None
    best = min(trains, key=lambda t: abs(t[0] - epoch))
    return best[1], best[2]


def render(log: Path) -> str:
    cfg, evals, trains = parse(log)
    out = [f"### {log.name}"]
    if not evals:
        note = "no eval_loss lines yet"
        if cfg.get("eval_split", "0") in ("0", "0.0"):
            note = "eval_split=0 -- this run has no eval split"
        out.append(f"    ({note}; {len(trains)} train points)")
        return "\n".join(out) + "\n"

    train_e, train_f, held_e = dataset_frames(cfg)
    bs = int(cfg.get("batch_size", 0) or 0)
    per_interval = None
    if train_f and bs and len(evals) > 1:
        per_interval = (evals[1][0] - evals[0][0]) * bs / train_f

    out.append(f"{'step':<8}{'epoch':<8}{'eval_loss':<11}{'train_loss':<12}{'gap':<9}dEpoch")
    prev_ep = None
    best = min(evals, key=lambda e: e[1])
    for step, ev in evals:
        # Prefer lerobot's own epoch for this step, derived from the training-set size.
        ep = step * bs / train_f if (train_f and bs) else None
        tr, _ = nearest_train(trains, ep) if ep is not None else (None, None)
        gap = f"{ev - tr:+.4f}" if tr is not None else "-"
        d = f"(+{ep - prev_ep:.2f})" if (ep is not None and prev_ep is not None) else ""
        mark = "  <- min eval" if (step, ev) == best else ""
        out.append(f"{step:<8}{ep if ep is None else f'{ep:<8.2f}'}"
                   f"{ev:<11.4f}{'-' if tr is None else f'{tr:<12.3f}'}{gap:<9}{d}{mark}")
        prev_ep = ep

    last_step, last_ev = evals[-1]
    out.append("")
    out.append(f"  min eval {best[1]:.4f} at step {best[0]}"
               + (f" (epoch {best[0]*bs/train_f:.1f})" if train_f and bs else "")
               + f";  latest {last_ev:.4f} at step {last_step}"
               + f"  [{last_ev - best[1]:+.4f} from min]")
    if last_ev > best[1] and last_step > best[0]:
        out.append("  -> eval loss is ABOVE its minimum: overfitting has begun.")

    if train_f:
        out.append("")
        out.append("  dataset arithmetic")
        out.append(f"    train {train_e} eps / ~{train_f:.0f} frames"
                   f"   (eval_split {cfg.get('eval_split','?')} -> {held_e} eps held out)")
        if per_interval and len(evals) > 1:
            iv = evals[1][0] - evals[0][0]
            out.append(f"    {iv} steps x batch {bs} = {iv*bs} samples"
                       f"  ->  {per_interval:.2f} epochs per eval interval")
    if cfg.get("save_freq"):
        out.append(f"    save_freq {cfg['save_freq']} of {cfg.get('steps','?')} steps"
                   f"  ->  {int(cfg.get('steps',0))//int(cfg['save_freq'])} checkpoints"
                   " (only these can be scored for R^2)")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", help="log files (default: outputs/refpatch_0908/*.log)")
    p.add_argument("--watch", type=int, metavar="SEC", help="refresh every SEC seconds")
    p.add_argument("--out", help="also write the rendered table here")
    a = p.parse_args()

    while True:
        paths = [Path(x) for x in (a.logs or sorted(glob.glob(DEFAULT_GLOB)))]
        paths = [q for q in paths if q.exists()]
        # Keep only actual training logs. The run directory also holds the scorer's stderr and
        # the watcher's own tee, which have no loss lines and would render as empty sections.
        paths = [q for q in paths if RE_TRAIN.search(q.read_text(errors="replace")[:400000])
                 or RE_EVAL.search(q.read_text(errors="replace")[:400000])]
        body = (f"loss monitor  --  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + "=" * 72 + "\n")
        body += ("\n".join(render(q) for q in paths) if paths
                 else "  no logs found\n")
        print("\033[2J\033[H" + body if a.watch else body, flush=True)
        if a.out:
            Path(a.out).write_text(body)
        if not a.watch:
            return
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
