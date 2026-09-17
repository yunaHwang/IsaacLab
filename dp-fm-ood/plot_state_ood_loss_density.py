"""Density of state_ood_loss pooled over several seeds and every step, in one panel: x = loss, y = density.

Reads the CSV that multitask_dit_server.py --ood_csv writes (see ood_csv_logger.py) - a policy CSV or a
SpaceMouse one. Step order and seed are ignored. Bars are a density-normalized histogram (area = 1); the
line is a kernel density estimate of the same values (turn it off with --no_kde). The subtitle summarizes
all plotted values.

    python plot_state_ood_loss_density.py
    python plot_state_ood_loss_density.py --csv outputs/state_id_loss_0915.csv --seeds 10 17 18 19 30 \
        --out outputs/state_ood_loss_density.png
    python plot_state_ood_loss_density.py --csv outputs/spacemouse_state_ood_loss.csv --seeds all
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
FILL = "#b7d3f6"        # blue 150 - histogram
LINE = "#1c5cab"        # blue 550 - KDE
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3dd"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(HERE / "outputs" / "state_id_loss_0915.csv"))
    parser.add_argument("--seeds", nargs="+", default=["10", "17", "18", "19", "30"],
                        help="Seeds to pool, or 'all' for every seed in the CSV.")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--no_kde", action="store_true", help="Histogram only, no KDE line.")
    parser.add_argument("--out", default=None, help="Output image (default: next to the CSV).")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["seed", "trial", "step", "state_ood_loss"]).dropna(subset=["state_ood_loss"])
    if args.seeds != ["all"]:
        seeds = [int(s) for s in args.seeds]
        missing = sorted(set(seeds) - set(df.seed))
        if missing:
            raise SystemExit(f"seeds {missing} have no state_ood_loss rows in {args.csv}")
        df = df[df.seed.isin(seeds)]
    seeds = sorted(df.seed.unique().tolist())
    losses = df.state_ood_loss.to_numpy()

    summary = [("min", losses.min()), ("max", losses.max()), ("median", np.median(losses)),
               ("std", losses.std(ddof=1))]
    summary += [(f"p{q}", np.percentile(losses, q)) for q in (75, 90, 95, 99)]
    print("  ".join(f"{name}={value:.4f}" for name, value in summary))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    bins = np.linspace(0.0, float(losses.max()), args.bins + 1)
    ax.hist(losses, bins=bins, density=True, color=FILL, edgecolor=SURFACE, linewidth=1.5, zorder=2,
            label="histogram (density)")
    if not args.no_kde and len(losses) > 1:
        x = np.linspace(0.0, float(losses.max()), 500)
        ax.plot(x, gaussian_kde(losses)(x), color=LINE, linewidth=2, zorder=3, label="kernel density estimate")

    ax.set_xlim(0, bins[-1])
    ax.set_xlabel("state_ood_loss", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("density", fontsize=10, color=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    seed_label = ", ".join(str(s) for s in seeds)
    fig.suptitle(f"state_ood_loss density, seeds {seed_label} pooled over all steps  (n = {len(losses)})",
                 x=0.01, y=0.99, ha="left", fontsize=11, color=TEXT_PRIMARY)
    ax.set_title("   ".join(f"{name} {value:.3f}" for name, value in summary), loc="left", fontsize=9,
                 color=TEXT_SECONDARY, pad=8)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_density.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"seeds {seeds}: n={len(losses)}; wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
