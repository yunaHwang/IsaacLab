"""Per-frame distribution plot of pi05_state_ood_perturbation_test.py output.

One panel per corrupted condition. x = frame, y = predicted state-OOD loss (model's own chunk).
In every panel, clean (gray) and the condition (color) are each drawn as the 3 sample points per
frame, their mean line, and a mean ± 1 std band - so each frame reads as two small distributions
side by side.

    python plot_pi05_perturbation_bands.py [--csv outputs/pi05_state_ood_perturbation/ep789_perturbations.csv] [--log]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=str(HERE / "outputs" / "pi05_state_ood_perturbation" / "ep789_perturbations.csv"))
parser.add_argument("--log", action="store_true")
parser.add_argument("--overlay", action="store_true", help="All conditions on one axes instead of one panel each.")
args = parser.parse_args()

df = pd.read_csv(args.csv)
pred_cols = [c for c in df.columns if c.startswith("pred_loss_") and c[-1].isdigit()]
conditions = [c for c in dict.fromkeys(df["condition"]) if c != "clean"]
frames = sorted(df["frame"].unique())

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
CLEAN, TEXT, MUTED, GRID = "#8a8a85", "#1f1f1e", "#6b6b66", "#e6e5df"


def stats(cond):
    sub = df[df.condition == cond].set_index("frame").loc[frames, pred_cols].values  # [frames, samples]
    return sub, sub.mean(1), sub.std(1)


clean_vals, clean_mean, clean_std = stats("clean")
ymax = df[pred_cols].values.max() * 1.08

if args.overlay:
    # One axes: clean (gray, emphasized) + all corrupted conditions, each as mean line, mean ± 1 std
    # band and its 3 sample points; conditions are dodged along x so their points don't pile up.
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=130)
    x = np.array(frames, dtype=float)
    series = [("clean", CLEAN)] + list(zip(conditions, SERIES))
    dodge = np.linspace(-40, 40, len(series))
    for (cond, color), dx in zip(series, dodge):
        vals, mean, std = stats(cond)
        is_clean = cond == "clean"
        ax.fill_between(x + dx, np.maximum(mean - std, 1e-4), mean + std, color=color,
                        alpha=0.22 if is_clean else 0.10, lw=0, zorder=1 if is_clean else 0)
        ax.plot(x + dx, mean, color=color, lw=3 if is_clean else 1.8, zorder=5 if is_clean else 3,
                label=f"{cond.replace('_', ' ')}" + ("" if is_clean else f"  ({mean.mean() / clean_mean.mean():.2f}× clean)"))
        for j in range(vals.shape[1]):
            ax.scatter(x + dx, vals[:, j], s=30 if is_clean else 20, color=color, edgecolor="white",
                       linewidth=0.9, zorder=6 if is_clean else 4)
    ax.set_xticks(frames)
    ax.set_xlabel("frame", color=TEXT)
    ax.set_ylabel("predicted state-OOD loss (model's own chunk)", color=TEXT)
    if args.log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, ymax)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="condition (mean ± 1 std, 3 samples/frame)",
              title_fontsize=9.5)
    ax.set_title("Pi0.5 predicted state-OOD loss per frame: clean vs 7 corrupted observations", color=TEXT, loc="left", fontsize=11.5)
    fig.tight_layout()
    out = Path(args.csv).with_name(Path(args.csv).stem + ("_overlay_log.png" if args.log else "_overlay.png"))
    fig.savefig(out)
    print(f"saved {out}")
    raise SystemExit

ncols = 4
nrows = int(np.ceil(len(conditions) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), dpi=130, sharex=True, sharey=True)
axes = axes.ravel()
x = np.array(frames, dtype=float)
jitter = np.array([-12, 0, 12])[: len(pred_cols)]

for ax, cond, color in zip(axes, conditions, SERIES):
    vals, mean, std = stats(cond)
    for v, m, s, c, dx, label in ((clean_vals, clean_mean, clean_std, CLEAN, -18, "clean"),
                                  (vals, mean, std, color, 18, cond.replace("_", " "))):
        ax.fill_between(x, np.maximum(m - s, 1e-4), m + s, color=c, alpha=0.18, lw=0)
        ax.plot(x, m, color=c, lw=2, label=f"{label} (mean ± 1 std)")
        for j in range(v.shape[1]):
            ax.scatter(x + dx + jitter[j] * 0.5, v[:, j], s=26, color=c, edgecolor="white", linewidth=1, zorder=3)
    ratio = mean.mean() / clean_mean.mean()
    above = int((vals > clean_vals.max(1, keepdims=True)).sum())
    ax.set_title(f"{cond.replace('_', ' ')}\n{ratio:.2f}× clean mean · {above}/{vals.size} > same-frame clean max",
                 fontsize=9.5, color=TEXT, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, labelbottom=True)
    ax.set_xticks(frames)

for ax in axes[len(conditions):]:
    ax.axis("off")
if args.log:
    axes[0].set_yscale("log")
else:
    axes[0].set_ylim(0, ymax)
for ax in axes[::ncols]:
    ax.set_ylabel("predicted state-OOD loss", color=TEXT, fontsize=9.5)
for ax in axes[:len(conditions)]:
    ax.set_xlabel("frame", color=TEXT, fontsize=9.5)
fig.suptitle(f"Pi0.5 predicted state-OOD loss per frame: clean (gray) vs each corrupted observation — "
             f"{len(pred_cols)} model samples per frame", fontsize=11.5, color=TEXT, x=0.01, ha="left")
fig.tight_layout()
out = Path(args.csv).with_name(Path(args.csv).stem + ("_bands_log.png" if args.log else "_bands.png"))
fig.savefig(out)
print(f"saved {out}")
