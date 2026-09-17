"""state_ood_loss with one subplot per seed, on shared axes so the seeds compare directly.

    --kind density (default)  x = loss, y = density: density-normalized histogram + kernel density estimate
    --kind step               x = step, y = loss: a line through each step's value

Each panel's subtitle carries that seed's n, median, std, p90 and p99.

    python plot_state_ood_loss_per_seed.py
    python plot_state_ood_loss_per_seed.py --kind step
    python plot_state_ood_loss_per_seed.py --csv outputs/state_id_loss_0915.csv --seeds 10 17 18 19 30
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
FILL = "#b7d3f6"        # blue 150 - histogram
LINE = "#1c5cab"        # blue 550 - KDE / step line
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3dd"


def style(ax, grid_axis):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(HERE / "outputs" / "state_id_loss_0915.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 17, 18, 19, 30])
    parser.add_argument("--kind", choices=["density", "step"], default="density")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--out", default=None, help="Output image (default: next to the CSV).")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["seed", "trial", "step", "state_ood_loss"]).dropna(subset=["state_ood_loss"])
    df = df[df.seed.isin(args.seeds)]
    missing = sorted(set(args.seeds) - set(df.seed))
    if missing:
        raise SystemExit(f"seeds {missing} have no state_ood_loss rows in {args.csv}")

    loss_max = float(df.state_ood_loss.max())
    bins = np.linspace(0.0, loss_max, args.bins + 1)
    x = np.linspace(0.0, loss_max, 400)
    step_max = int(df.step.max())

    ncols = 3 if len(args.seeds) > 4 else len(args.seeds)
    nrows = math.ceil(len(args.seeds) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    axes = axes.ravel()

    print(f"{'seed':>4s} {'n':>5s} {'min':>7s} {'max':>7s} {'median':>7s} {'std':>7s} {'p75':>7s} {'p90':>7s} {'p95':>7s} {'p99':>7s}")
    for ax, seed in zip(axes, args.seeds):
        d = df[df.seed == seed].sort_values("step")
        v = d.state_ood_loss.to_numpy()
        p = {q: np.percentile(v, q) for q in (75, 90, 95, 99)}
        print(f"{seed:4d} {len(v):5d} {v.min():7.3f} {v.max():7.3f} {np.median(v):7.3f} {v.std(ddof=1):7.3f} "
              f"{p[75]:7.3f} {p[90]:7.3f} {p[95]:7.3f} {p[99]:7.3f}")

        if args.kind == "density":
            ax.hist(v, bins=bins, density=True, color=FILL, edgecolor=SURFACE, linewidth=1.2, zorder=2)
            if len(v) > 1:
                ax.plot(x, gaussian_kde(v)(x), color=LINE, linewidth=2, zorder=3)
            style(ax, "y")
        else:
            ax.plot(d.step, v, color=LINE, linewidth=1.5, zorder=2)
            style(ax, "y")

        ax.set_title(f"seed {seed}", loc="left", fontsize=10, color=TEXT_PRIMARY, pad=16)
        ax.text(0.0, 1.02, f"n {len(v)}   median {np.median(v):.3f}   std {v.std(ddof=1):.3f}   "
                f"p90 {p[90]:.3f}   p99 {p[99]:.3f}", transform=ax.transAxes, fontsize=7.5, color=TEXT_SECONDARY,
                va="bottom")

    for ax in axes[len(args.seeds):]:
        ax.set_visible(False)
    xlabel, ylabel = ("state_ood_loss", "density") if args.kind == "density" else ("step", "state_ood_loss")
    for i, ax in enumerate(axes[:len(args.seeds)]):
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
        # bottom-most visible panel in each column gets the x label and tick labels
        if i + ncols >= len(args.seeds):
            ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
            ax.xaxis.set_tick_params(labelbottom=True)
    if args.kind == "density":
        axes[0].set_xlim(0, loss_max)
    else:
        axes[0].set_xlim(0, step_max + 2)
        axes[0].set_ylim(0, loss_max * 1.05)

    what = "density (bars: histogram, line: kernel density estimate)" if args.kind == "density" else "per step"
    fig.suptitle(f"state_ood_loss {what}, one panel per seed  ({Path(args.csv).name})", x=0.01, ha="left",
                 fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    suffix = "_per_seed_density.png" if args.kind == "density" else "_per_seed_step.png"
    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + suffix)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
