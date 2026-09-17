#!/usr/bin/env python3
"""Plot seen vs held-out arm R^2 across checkpoints, from r2_progress_0909.txt.

WHY PARSE THE LOG RATHER THAN HARDCODE THE NUMBERS
    The .txt is the canonical accumulating record the 0909 runner appends to, so this stays
    correct as more checkpoints land -- rerun it and the figure updates. The numbers are NOT
    baked in. (r2_history_0909.csv holds the same rows; --csv reads that instead.)

WHAT IT PLOTS
    "R2 ARM 0-5" -- arm dims 0-5 of the 7-dim IK-Rel action -- for both splits, with the 95%
    CI as a shaded band. Deliberately NOT "R2 all": that number is inflated by the gripper
    dim, which reaches ~0.87 held-out from phase structure alone with no spatial understanding
    (see generalization_check.py). Arm R^2 is the number that discriminates.

    The step-180000 anchor comes from the header's "where this run left off" line. It is the
    pre-resume value and carries no CI, so it is drawn as a hollow marker: the drop from it to
    195000 is the cosine warm restart (lr jumped ~0 -> ~2.4e-4 when --steps changed), not a
    training result.

ONE AXIS ON PURPOSE
    Both series are R^2 on the same scale, so they share one y-axis. The generalization gap is
    the vertical distance between them; --gap adds it as a second stacked panel sharing x
    (small multiples), never as a second y-scale on the same axes.

Usage:
    python dp-fm-ood/plot_r2_progress.py
    python dp-fm-ood/plot_r2_progress.py --gap                # add the gap panel
    python dp-fm-ood/plot_r2_progress.py --dark               # dark-surface variant
    python dp-fm-ood/plot_r2_progress.py --log <path> -o out.png
    python dp-fm-ood/plot_r2_progress.py --csv <r2_history.csv>
"""

from __future__ import annotations

import argparse
import csv as _csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path("/home/wisc-rt2-trimanual/isaac-sim/IsaacLab")
# Both files use the same "@@ EEFPOS <step>" block format, so they merge cleanly:
#   eefpos_vs_bf16.txt   the original run, 45000 -> 180000
#   r2_progress_0909.txt the 0909 continuation, 195000 -> onwards
# eefpos_vs_bf16.txt ALSO contains "@@ BF16" blocks for the joint-state run; those are a
# different model and are skipped by tag (see parse_log) rather than silently folded in.
_RUN = BASE / "outputs/refpatch_0908"
DEFAULT_LOGS = [
    _RUN / "eefpos_vs_bf16.txt",
    _RUN / "n300_unfrz_sepenc_L6h512_ev20_patch_eefpos/r2_progress_0909.txt",
]

# Categorical slots 1 and 2 of the validated default palette, in fixed order -- never cycled,
# never reassigned by rank. Dark steps are the same two hues re-stepped for the dark surface.
# The 7 action dims, in order. Arm 0-5 are the relative EEF pose delta; 6 is the binary gripper.
DIMS_NOTE = ("arm action: dx, dy, dz, drx, dry, drz, grip"
             "   (6-DoF relative EEF pose delta + binary gripper)")

THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#8a8880",
                  grid="#e4e3df", seen="#2a78d6", held="#eb6834"),
    "dark":  dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#8a8880",
                  grid="#333331", seen="#3987e5", held="#d95926"),
}

# Tagged so a "@@ BF16 ..." block cannot leak its numbers into the preceding EEFPOS step.
RE_BLOCK = re.compile(r"^@@ (\S+)\s+(\d+)")
WANT_TAG = "EEFPOS"
RE_SEEN = re.compile(r"SEEN \(trained on\)")
RE_HELD = re.compile(r"HELD-OUT \(never seen\)")
# "R2 ARM 0-5 : 0.652   95% CI [+0.612, +0.688]  (162 eps)"
RE_ARM = re.compile(r"R2 ARM 0-5\s*:\s*(-?[\d.]+)"
                    r"(?:\s+95% CI \[\s*([+-]?[\d.]+),\s*([+-]?[\d.]+)\s*\])?")
# "      MSE        : 0.03315"  -- flow-matching MSE over all 7 action dims, no CI reported.
RE_MSE = re.compile(r"^\s*MSE\s*:\s*([\d.eE+-]+)\s*$")
# "MSE per-sample : mean 0.03876  p10 0.00984 ... max 0.16616  n 60"
# The spread of per-frame MSE across the sampled frames. This is a DISTRIBUTION OVER THE DATA,
# not a confidence interval on the mean -- wider than a CI and answering a different question
# ("how variable are individual frames?" vs "how well pinned is the average?").
RE_MSE_PS = re.compile(r"MSE per-sample\s*:\s*(.+)$")
# "# 0909 EEFPOS CONTINUATION -- resumed from step 180000 at ..." -- marks the warm restart.
RE_RESUME = re.compile(r"resumed from step\s+(\d+)")


def parse_logs(paths):
    """-> (rows, resume_step). rows = [{step, seen, seen_lo, seen_hi, held, held_lo, held_hi}].

    Merges any number of log files keyed on step. The eval's stdout interleaves large CLIP
    load reports between the SEEN and HELD-OUT blocks, so this tracks which split it is inside
    rather than relying on line offsets, and takes the FIRST "R2 ARM 0-5" after each marker.
    """
    rows, resume = {}, None
    for path in paths:
        if not Path(path).exists():
            print(f"  (skipping missing {path})")
            continue
        r, rs = _parse_one(Path(path), rows)
        resume = resume or rs
        print(f"  {Path(path).name}: {r} EEFPOS checkpoints")
    complete = [v for v in rows.values() if "seen" in v and "held" in v]
    return sorted(complete, key=lambda v: v["step"]), resume


def _parse_one(path: Path, rows: dict):
    found = 0
    step = split = None
    resume = None
    for line in path.read_text(errors="replace").splitlines():
        m = RE_RESUME.search(line)
        if m:
            resume = int(m.group(1))
            continue
        m = RE_BLOCK.match(line.strip())
        if m:
            # A block of a different run (e.g. "@@ BF16") clears `step`, so its SEEN/HELD-OUT
            # numbers are dropped instead of attaching to the last EEFPOS step.
            if m.group(1) != WANT_TAG:
                step = split = None
                continue
            step = int(m.group(2))
            if step not in rows:
                found += 1
            rows.setdefault(step, {"step": step})
            split = None
            continue
        if step is None:
            continue
        if RE_SEEN.search(line):
            split = "seen"
            continue
        if RE_HELD.search(line):
            split = "held"
            continue
        m = RE_ARM.search(line)
        if m and split and split not in rows[step]:
            rows[step][split] = float(m.group(1))
            if m.group(2):
                rows[step][f"{split}_lo"] = float(m.group(2))
                rows[step][f"{split}_hi"] = float(m.group(3))
            continue
        m = RE_MSE.match(line)
        if m and split and f"{split}_mse" not in rows[step]:
            rows[step][f"{split}_mse"] = float(m.group(1))
            continue
        m = RE_MSE_PS.search(line)
        if m and split and f"{split}_mse_p10" not in rows[step]:
            for k, v in re.findall(r"(mean|p10|p25|p50|p75|p90|min|max)\s+([\d.eE+-]+)",
                                   m.group(1)):
                rows[step][f"{split}_mse_{k}"] = float(v)
    return found, resume


def parse_csv(path: Path):
    rows = []
    with open(path) as f:
        for r in _csv.DictReader(f):
            try:
                rows.append({
                    "step": int(r["step"]),
                    "seen": float(r["seen_r2"]), "seen_lo": float(r["seen_ci_lo"]),
                    "seen_hi": float(r["seen_ci_hi"]),
                    "held": float(r["heldout_r2"]), "held_lo": float(r["heldout_ci_lo"]),
                    "held_hi": float(r["heldout_ci_hi"]),
                })
            except (ValueError, KeyError):
                continue
    return sorted(rows, key=lambda r: r["step"]), None


# Each metric becomes its OWN figure -- never two y-scales on one axes. R^2 is
# higher-is-better and bounded [0,1]; MSE is lower-is-better and unbounded, so putting them
# together would need a second scale, which is the one thing a chart must not do.
METRICS = {
    "r2": dict(
        suffix="seen", ci=True, ylim=(0, 1), fmt="{:.3f}", best=max,
        ylabel="Arm R$^2$ (action dims 0\u20135)",
        title="EEF-pose run: seen R$^2$ keeps climbing, held-out never leaves ~0.2",
        note="higher the better",
    ),
    "mse": dict(
        suffix="mse", ci=True, ylim=(0, None), fmt="{:.4f}", best=min,
        ylabel="MSE (all 7 action dims)",
        title="EEF-pose run: seen MSE keeps falling, held-out is flat",
        note="lower the better",
    ),
}


def series(rows, key, metric="r2", band="p10-p90"):
    """(xs, ys, lo, hi) for one split.

    R^2: ys is the point estimate, lo/hi the bootstrap 95% CI (uncertainty on the MEAN).
    MSE: ys is the mean over sampled frames, lo/hi the requested spread of the PER-FRAME
         values (variation in the data itself). Falls back to a flat band when the log has no
         per-sample line -- i.e. was written before that was recorded.
    """
    col = key if metric == "r2" else f"{key}_mse"
    pts = [r for r in rows if col in r]
    xs = [r["step"] for r in pts]
    ys = [r[col] for r in pts]
    if metric == "r2":
        lo = [r.get(f"{key}_lo", r[col]) for r in pts]
        hi = [r.get(f"{key}_hi", r[col]) for r in pts]
    else:
        a, b = {"p10-p90": ("p10", "p90"), "p25-p75": ("p25", "p75"),
                "minmax": ("min", "max")}[band]
        lo = [r.get(f"{key}_mse_{a}", r[col]) for r in pts]
        hi = [r.get(f"{key}_mse_{b}", r[col]) for r in pts]
    return xs, ys, lo, hi


def draw(rows, resume, metric, c, out, a):
    """One figure for one metric. Called once per metric so each gets its own file -- the two
    are on different scales and different polarities, so they must not share an axes."""
    M = METRICS[metric]
    col = (lambda k: k) if metric == "r2" else (lambda k: f"{k}_mse")

    if a.gap:
        fig, (ax, axg) = plt.subplots(
            2, 1, figsize=(9.5, 7.0), sharex=True, layout="constrained",
            gridspec_kw=dict(height_ratios=[3, 1]))
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.4), layout="constrained")
        axg = None
    fig.patch.set_facecolor(c["surface"])

    for axis in filter(None, (ax, axg)):
        axis.set_facecolor(c["surface"])
        # Recessive grid, behind the data.
        axis.grid(True, color=c["grid"], linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(c["grid"])
        axis.tick_params(colors=c["secondary"], labelsize=9, length=0)

    # Colour follows the ENTITY, not the metric: seen is always slot 1, held-out always slot 2,
    # in both figures, so the two can be read side by side.
    for key, color, label in (("seen", c["seen"], "Train set (N=240)"),
                              ("held", c["held"], "Test set (N=60)")):
        xs, ys, lo, hi = series(rows, key, metric, a.mse_band)
        if not xs:
            continue
        if M["ci"]:
            ax.fill_between(xs, lo, hi, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o",
                markersize=9 if metric == "mse" else 7,
                markeredgecolor=c["surface"], markeredgewidth=2, label=label, zorder=3)
        # Direct label at the line end; text stays in ink, the line terminating at it carries
        # identity (<=4 series, so both are labeled).
        ax.annotate(f"{label.split(' (')[0]}  {M['fmt'].format(ys[-1])}",
                    xy=(xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9.5, color=c["primary"], zorder=4)

    # The run was stopped and resumed here, and --steps changed, so the cosine LR was rebuilt
    # and jumped ~0 -> ~2.4e-4. The jump straight after is that warm restart, not a training
    # result -- mark the boundary so it is not read as the model getting worse.
    if resume is not None:
        before = [r for r in rows if r["step"] <= resume and col("seen") in r]
        after = [r for r in rows if r["step"] > resume and col("seen") in r]
        if before and after:
            for key, color in (("seen", c["seen"]), ("held", c["held"])):
                seg_x = [before[-1]["step"], after[0]["step"]]
                seg_y = [before[-1][col(key)], after[0][col(key)]]
                ax.plot(seg_x, seg_y, color=c["surface"], linewidth=3.5, zorder=3)
                ax.plot(seg_x, seg_y, color=color, linewidth=2, linestyle=(0, (3, 2)), zorder=3)
        ax.axvline(resume, color=c["muted"], linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.annotate("stopped & resumed\n(LR warm restart)", xy=(resume, 0.99),
                    xycoords=("data", "axes fraction"), xytext=(-6, -4),
                    textcoords="offset points", va="top", ha="right",
                    fontsize=8.5, color=c["muted"], zorder=4)

    # Best held-out for this metric -- max for R^2, min for MSE.
    pts = [r for r in rows if col("held") in r]
    if pts:
        best = M["best"](pts, key=lambda r: r[col("held")])
        ax.annotate(f"best held-out {M['fmt'].format(best[col('held')])}",
                    xy=(best["step"], best[col("held")]), xytext=(0, -22),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=c["secondary"],
                    arrowprops=dict(arrowstyle="-", color=c["muted"], linewidth=1), zorder=4)

    ax.set_ylabel(M["ylabel"], color=c["secondary"], fontsize=10)
    ax.set_ylim(*M["ylim"])
    ax.set_title(M["title"], color=c["primary"], fontsize=13, loc="left", pad=44)
    sub = f"MultiTaskDiT, Train-test split=8:2 (train=240, test=60) \u00b7 {M['note']}"
    if M["ci"]:
        sub += " \u00b7 95% CI (sampling flow matching policy 5 times per datapoint)"
    ax.annotate(sub, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 21),
                textcoords="offset points", fontsize=9, color=c["secondary"], va="bottom")
    # Second line: what the 7 action dims ARE. The policy emits a 6-DoF RELATIVE end-effector
    # pose delta plus a binary gripper (DifferentialInverseKinematicsActionCfg,
    # use_relative_mode=True) -- not an absolute pose, and never joint angles.
    ax.annotate(DIMS_NOTE, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", fontsize=8.5, color=c["muted"], va="bottom")
    leg = ax.legend(loc="center left", frameon=False, fontsize=9.5,
                    bbox_to_anchor=(0.03, 0.45), handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(c["primary"])

    if axg is not None:
        pts = [r for r in rows if col("seen") in r and col("held") in r]
        xs = [r["step"] for r in pts]
        gaps = [abs(r[col("seen")] - r[col("held")]) for r in pts]
        axg.plot(xs, gaps, color=c["muted"], linewidth=2, marker="o", markersize=6,
                 markeredgecolor=c["surface"], markeredgewidth=2, zorder=3)
        axg.set_ylabel("gap", color=c["secondary"], fontsize=10)
        axg.annotate(M["fmt"].format(gaps[-1]), xy=(xs[-1], gaps[-1]), xytext=(8, 0),
                     textcoords="offset points", va="center", fontsize=9.5, color=c["primary"])
        axg.set_ylim(0, max(gaps) * 1.25)

    bottom = axg if axg is not None else ax
    bottom.set_xlabel("training step", color=c["secondary"], fontsize=10)
    bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

    # Headroom on the right for the direct labels.
    left = rows[0]["step"]
    span = rows[-1]["step"] - left
    ax.set_xlim(left - span * 0.04, rows[-1]["step"] + span * 0.20)

    fig.savefig(out, dpi=a.dpi, facecolor=c["surface"], bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, nargs="+", default=DEFAULT_LOGS,
                   help="one or more @@ EEFPOS logs; merged on step")
    p.add_argument("--csv", type=Path, default=None, help="read r2_history CSV instead of the txt")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--gap", action="store_true", help="add a second panel with the seen-heldout gap")
    p.add_argument("--mse-band", choices=["p10-p90", "p25-p75", "minmax"], default="p10-p90",
                   help="Which spread of the PER-FRAME MSE to shade. Default p10-p90: the "
                        "distribution is strongly right-skewed (a few frames near grasp "
                        "transitions are far harder than the rest), so minmax tracks a single "
                        "outlier rather than the bulk.")
    p.add_argument("--resume-step", type=int, default=None,
                   help="Mark a stop/resume boundary at this step. Auto-detected from a log "
                        "carrying 'resumed from step N' (the 0909 continuation header); a "
                        "re-scoring file such as r2_progress_0910_alleps.txt has no such line, "
                        "so pass it explicitly to keep the warm-restart discontinuity labelled.")
    p.add_argument("--metric", choices=["r2", "mse", "both"], default="both",
                   help="which figure(s) to write. Each metric gets its OWN png -- "
                        "R^2 and MSE have different scales and opposite polarity, so "
                        "they are never combined onto one axes.")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    a = p.parse_args()

    if a.csv:
        rows, resume = parse_csv(a.csv)
        src = a.csv
    else:
        print("parsing:")
        rows, resume = parse_logs(a.log)
        src = a.log[-1]
    if not rows:
        raise SystemExit(f"no complete EEFPOS checkpoint blocks parsed from {a.csv or a.log}")
    print(f"{len(rows)} checkpoints total: {rows[0]['step']} -> {rows[-1]['step']}"
          + (f", resumed at {resume}" if resume else ""))

    if a.resume_step is not None:
        resume = a.resume_step
    c = THEME["dark" if a.dark else "light"]
    wanted = list(METRICS) if a.metric == "both" else [a.metric]
    stem = Path(a.out).with_suffix("") if a.out else (
        Path(src).parent / ("r2_progress_0910" + ("_dark" if a.dark else "")))
    for metric in wanted:
        if not any(("seen" if metric == "r2" else "seen_mse") in r for r in rows):
            print(f"  (no {metric} values in the log -- skipping that figure)")
            continue
        out = Path(f"{stem}_{metric}.png")
        draw(rows, resume, metric, c, out, a)
        print(f"wrote {out}")

    # Table view -- identity is never color-alone, and the numbers stay readable as text.
    has_mse = any("seen_mse" in r for r in rows)
    hdr = f"\n{'step':>8}  {'seen R2':>8}  {'held R2':>8}  {'gap':>6}"
    if has_mse:
        hdr += f"  {'seen MSE':>9}  {'held MSE':>9}"
    print(hdr)
    best_r2 = max((r for r in rows if "held" in r), key=lambda r: r["held"], default=None)
    for r in rows:
        line = (f"{r['step']:>8}  {r.get('seen', float('nan')):>8.3f}  "
                f"{r.get('held', float('nan')):>8.3f}  "
                f"{r.get('seen', 0) - r.get('held', 0):>6.3f}")
        if has_mse:
            line += f"  {r.get('seen_mse', float('nan')):>9.5f}  {r.get('held_mse', float('nan')):>9.5f}"
        if best_r2 is not None and r is best_r2:
            line += "   <- best held-out R2"
        if resume is not None and r["step"] == resume:
            line += "   (resumed after this)"
        print(line)


if __name__ == "__main__":
    main()
