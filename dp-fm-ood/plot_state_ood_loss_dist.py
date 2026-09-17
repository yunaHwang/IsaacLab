"""Distribution of state_ood_loss pooled over several seeds and every step, in one panel.

Reads the CSV that multitask_dit_server.py --ood_csv writes (see ood_csv_logger.py). Step order and
seed are ignored - every row of the selected seeds is one sample. The y-axis is the loss value, the
x-axis is how many steps fall in each loss bin (a sideways histogram).

    python plot_state_ood_loss_dist.py
    python plot_state_ood_loss_dist.py --csv outputs/state_id_loss_0915.csv --seeds 10 17 18 19 30 \
        --out outputs/state_ood_loss_dist.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
FILL = "#2a78d6"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3dd"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(HERE / "outputs" / "state_id_loss_0915.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 17, 18, 19, 30])
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--out", default=None, help="Output image (default: next to the CSV).")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["seed", "trial", "step", "state_ood_loss"])
    df = df[df.seed.isin(args.seeds)].dropna(subset=["state_ood_loss"])
    missing = sorted(set(args.seeds) - set(df.seed))
    if missing:
        raise SystemExit(f"seeds {missing} have no state_ood_loss rows in {args.csv}")
    losses = df.state_ood_loss.to_numpy()

    mean, median = float(losses.mean()), float(np.median(losses))
    p90 = float(np.quantile(losses, 0.9))
    print(f"seeds {args.seeds}: n={len(losses)} mean={mean:.4f} median={median:.4f} std={losses.std(ddof=1):.4f} "
          f"p90={p90:.4f} min={losses.min():.4f} max={losses.max():.4f}")

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    bins = np.linspace(0.0, float(losses.max()), args.bins + 1)
    ax.hist(losses, bins=bins, orientation="horizontal", color=FILL, edgecolor=SURFACE, linewidth=1.5, zorder=2)

    for value, label, style in ((mean, "mean", "-"), (median, "median", (0, (4, 3))), (p90, "90th pct", (0, (1, 2)))):
        ax.axhline(value, color=TEXT_PRIMARY, linewidth=1, linestyle=style, zorder=3)
        ax.text(1.0, value, f" {label} {value:.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9, color=TEXT_SECONDARY)

    ax.set_ylabel("state_ood_loss", fontsize=10, color=TEXT_SECONDARY)
    ax.set_xlabel("steps (count)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylim(0, bins[-1])
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    seeds = ", ".join(str(s) for s in args.seeds)
    ax.set_title(f"state_ood_loss, seeds {seeds} pooled over all steps  (n = {len(losses)})",
                 loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_loss_dist.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
