"""state_ood_loss over rollout steps for several seeds, in one panel: every (seed, step) value as a dot,
the mean across seeds per step as a line, and the min-max range across seeds per step shaded. The
subtitle summarizes all plotted values pooled (min, max, median, std, 75/90/95/99th percentiles).

Reads the CSV that multitask_dit_server.py --ood_csv writes (see ood_csv_logger.py). Seeds are not
colored individually. Rollouts end at different steps, so later steps average over fewer seeds - a
dotted guide marks where each seed's rollout ends, and the band is only drawn where >= 2 seeds remain.

    python plot_state_ood_loss_per_step.py
    python plot_state_ood_loss_per_step.py --csv outputs/state_id_loss_0915.csv --seeds 10 17 18 19 30 \
        --out outputs/state_ood_loss_per_step.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
LINE = "#1c5cab"        # blue 550 - mean line
DOTS = "#2a78d6"        # blue 450 - individual values
BAND = "#cde2fb"        # blue 100 - min-max range
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3dd"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(HERE / "outputs" / "state_id_loss_0915.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 17, 18, 19, 30])
    parser.add_argument("--out", default=None, help="Output image (default: next to the CSV).")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["seed", "trial", "step", "state_ood_loss"])
    df = df[df.seed.isin(args.seeds)].dropna(subset=["state_ood_loss"])
    missing = sorted(set(args.seeds) - set(df.seed))
    if missing:
        raise SystemExit(f"seeds {missing} have no state_ood_loss rows in {args.csv}")

    per_step = df.groupby("step").state_ood_loss.agg(["mean", "min", "max", "count"]).reset_index()
    ends = df.groupby("seed").step.max().sort_values()
    banded = per_step[per_step["count"] >= 2]

    # Summary over every plotted value (all seeds, all steps pooled).
    losses = df.state_ood_loss.to_numpy()
    summary = [("min", losses.min()), ("max", losses.max()), ("median", np.median(losses)),
               ("std", losses.std(ddof=1))]
    summary += [(f"p{q}", np.percentile(losses, q)) for q in (75, 90, 95, 99)]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    ax.fill_between(banded.step, banded["min"], banded["max"], color=BAND, linewidth=0, zorder=1,
                    label="min-max across seeds")
    ax.scatter(df.step, df.state_ood_loss, s=9, color=DOTS, alpha=0.45, linewidths=0, zorder=2,
               label="each seed's value")
    ax.plot(per_step.step, per_step["mean"], color=LINE, linewidth=2, zorder=3, label="mean across seeds")

    ymax = float(df.state_ood_loss.max()) * 1.08
    for seed, last in ends.items():
        ax.axvline(last, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (1, 3)), zorder=0)
        ax.text(last, ymax, f"seed {seed} ends ", rotation=90, ha="right", va="top", fontsize=8,
                color=TEXT_SECONDARY)

    ax.set_xlim(0, int(per_step.step.max()) + 2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("step", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("state_ood_loss", fontsize=10, color=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=3, frameon=False, fontsize=9,
                       markerscale=2)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    seeds = ", ".join(str(s) for s in args.seeds)
    fig.suptitle(f"state_ood_loss per step, seeds {seeds}  (n = {len(df)} steps)", x=0.01, y=0.995, ha="left",
                 fontsize=11, color=TEXT_PRIMARY)
    ax.set_title("all values:   " + "   ".join(f"{name} {value:.3f}" for name, value in summary), loc="left",
                 fontsize=9, color=TEXT_SECONDARY, pad=10)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_per_step.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("  ".join(f"{name}={value:.4f}" for name, value in summary))
    print(f"seeds {args.seeds}: {len(df)} rows; rollout lengths "
          f"{ {int(s): int(e) + 1 for s, e in ends.items()} }; wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
