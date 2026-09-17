"""Strip plot of pi05_state_ood_perturbation_test.py output: the 15 clean pred_loss values
(5 frames x 3 model samples) as a gray reference band, against the 15 values of each corrupted
condition.

    python plot_pi05_perturbation.py [--csv outputs/pi05_state_ood_perturbation/ep789_perturbations.csv]
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
parser.add_argument("--log", action="store_true", help="Log-scale y axis.")
args = parser.parse_args()

df = pd.read_csv(args.csv)
pred_cols = [c for c in df.columns if c.startswith("pred_loss_") and c[-1].isdigit()]
long = df.melt(id_vars=["frame", "condition"], value_vars=pred_cols, var_name="sample", value_name="pred_loss")
conditions = list(dict.fromkeys(df["condition"]))            # file order, clean first
frames = sorted(df["frame"].unique())

# Categorical slots in fixed order (dataviz reference palette); clean is the gray reference.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
colors = {"clean": "#8a8a85"}
colors.update({c: SERIES[i] for i, c in enumerate(x for x in conditions if x != "clean")})
TEXT, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e6e5df"

clean = long.loc[long.condition == "clean", "pred_loss"]
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=130)
fig.patch.set_facecolor("white")
ax.axhspan(clean.min(), clean.max(), color="#8a8a85", alpha=0.14, lw=0, zorder=0)
ax.axhline(clean.median(), color="#8a8a85", lw=1.5, ls="--", zorder=1)
ax.text(len(conditions) - 0.35, clean.max(), f"clean max\n{clean.max():.3f}", ha="left", va="center", fontsize=9, color=MUTED)
ax.text(len(conditions) - 0.35, clean.min(), f"clean min\n{clean.min():.3f}", ha="left", va="center", fontsize=9, color=MUTED)
ax.text(len(conditions) - 0.35, clean.median(), f"clean median\n{clean.median():.3f}", ha="left", va="center", fontsize=9, color=MUTED)

offsets = np.linspace(-0.3, 0.3, len(frames))                 # frame position inside each column
rng = np.random.default_rng(0)
for i, cond in enumerate(conditions):
    sub = long[long.condition == cond]
    for k, fr in enumerate(frames):
        y = sub.loc[sub.frame == fr, "pred_loss"].values
        x = i + offsets[k] + rng.uniform(-0.025, 0.025, len(y))
        ax.scatter(x, y, s=46, color=colors[cond], edgecolor="white", linewidth=1.2, zorder=3)
    med = sub["pred_loss"].median()
    ax.plot([i - 0.38, i + 0.38], [med, med], color=TEXT, lw=2, zorder=4, solid_capstyle="round")
    above = (sub["pred_loss"] > clean.max()).sum()
    ax.text(i, 1.0, f"{above}/{len(sub)} above\nclean max", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color=MUTED)

ax.set_xticks(range(len(conditions)))
ax.set_xticklabels([c.replace("_", " ").replace(" ", "\n", 1) for c in conditions], fontsize=9.5, color=TEXT)
ax.set_xlim(-0.6, len(conditions) - 0.4)
ax.set_clip_on(False)
if args.log:
    ax.set_yscale("log")
ax.set_ylabel("predicted state-OOD loss (model's own chunk)", color=TEXT)
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
nb = "Nb=8" if "perturbation" in args.csv else ""
ax.set_title(f"Predicted state-OOD loss: clean vs corrupted observations (15 points each = {len(frames)} frames × 3 samples; "
             f"frames {frames[0]}–{frames[-1]} left→right; black bar = median)",
             fontsize=10.5, color=TEXT, pad=34, loc="left")
fig.tight_layout(rect=(0, 0, 0.93, 1))
out = Path(args.csv).with_name(Path(args.csv).stem + ("_strip_log.png" if args.log else "_strip.png"))
fig.savefig(out)
print(f"saved {out}")
