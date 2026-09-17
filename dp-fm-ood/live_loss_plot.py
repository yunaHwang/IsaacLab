"""Live matplotlib window for SpaceMouse rollouts: the policy's recorded state_ood_loss for the same seed as a
fixed reference line, with the SpaceMouse loss drawn on top as it arrives - a line for past steps and a dot
for the current one.

Runs as its OWN process, launched by run_policy_fm.py (LiveLossPlot below): Isaac Sim owns the main thread
of that process, and multiprocessing would re-run its module-level AppLauncher in the child. The client
writes one line per event to this script's stdin:
    csv <path>                the server's --ood_csv: plots are saved next to it, named after it
    reset <seed> <trial>      a new rollout starts (clears the SpaceMouse trace)
    <step> <loss>             the loss the server returned for this step's SpaceMouse command
    interrupted               the run was stopped with Ctrl+C
When stdin closes (run finished, interrupted, or the client process died) the window stays open until you
close it. The viewer runs in its own session, so a Ctrl+C in the Isaac Sim terminal does not kill it.

Every rollout that got at least one step is saved as a PNG, next to the CSV when the server reported one
(otherwise in --save_dir, default dp-fm-ood/outputs/):
    <csv stem>_seed<seed>_trial<trial>[_interrupted]_<YYYYmmdd-HHMMSS>.png           loss per step
    <csv stem>_seed<seed>_trial<trial>[_interrupted]_<YYYYmmdd-HHMMSS>_density.png   ID vs this run
written when the next rollout resets or when the run ends, and covering the same steps as the CSV. The density
figure is drawn off-screen (never shown during the run): the ID distribution is the policy's state_ood_loss
pooled over --id_seeds of --reference_csv, overlaid with this rollout's SpaceMouse losses.

Standalone (e.g. to look at a reference without a run):
    python live_loss_plot.py --reference_csv outputs/state_id_loss_0915.csv --reference_seed 10 < /dev/null
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
SPACEMOUSE = "#2a78d6"   # blue 450
REFERENCE = "#8a8983"    # neutral gray - context, not the live series
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3dd"


class LiveLossPlot:
    """Client side, used from run_policy_fm.py. Never imports matplotlib; if the viewer cannot start or is
    closed mid-run, plotting silently turns off and the rollout carries on."""

    def __init__(self, reference_csv, reference_seed):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--reference_csv", str(reference_csv),
               "--reference_seed", str(reference_seed)]
        try:
            # start_new_session: Ctrl+C in the Isaac terminal must not reach the viewer, or it would die
            # before saving the plot.
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
        except OSError as e:
            print(f"[live_plot] could not start viewer: {e}")
            self._proc = None

    def _send(self, line):
        if self._proc is None:
            return
        if self._proc.poll() is not None:  # window closed or viewer crashed
            print("[live_plot] viewer exited - live plotting off for the rest of the run")
            self._proc = None
            return
        try:
            self._proc.stdin.write(line + "\n")
        except (BrokenPipeError, OSError):
            self._proc = None

    def set_csv(self, path):
        if path:
            self._send(f"csv {path}")

    def interrupted(self):
        self._send("interrupted")

    def reset(self, seed, trial):
        self._send(f"reset {seed} {trial}")

    def push(self, step, loss):
        self._send(f"{step} {loss:.8g}")

    def close(self):
        """Signal end of run; the window stays open for inspection."""
        if self._proc is not None and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass


def save_density_comparison(out, id_losses, id_label, run_losses, run_label, subtitle_run):
    """ID vs run loss distributions, saved off-screen (Figure + Agg canvas, no pyplot window). Bars are
    density-normalized histograms on shared bins, lines are kernel density estimates."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    id_losses, run_losses = np.asarray(id_losses, float), np.asarray(run_losses, float)
    fig = Figure(figsize=(9, 5.2), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    hi = max([v.max() for v in (id_losses, run_losses) if len(v)] + [0.5])
    bins = np.linspace(0.0, hi, 50)
    x = np.linspace(0.0, hi, 600)

    for values, color, label, alpha in ((id_losses, REFERENCE, id_label, 0.35), (run_losses, SPACEMOUSE, run_label, 0.3)):
        if not len(values):
            continue
        ax.hist(values, bins=bins, density=True, color=color, alpha=alpha, linewidth=0, zorder=2)
        if len(values) > 1 and values.std() > 0:
            from scipy.stats import gaussian_kde
            # Silverman's robust bandwidth (min of std and IQR/1.34): heavy-tailed loss runs otherwise get a
            # std-driven bandwidth that smears the main peak. gaussian_kde takes it as a factor of the std.
            iqr = np.subtract(*np.percentile(values, [75, 25]))
            spread = min(values.std(ddof=1), iqr / 1.34) if iqr > 0 else values.std(ddof=1)
            factor = 0.9 * spread * len(values) ** -0.2 / values.std(ddof=1)
            ax.plot(x, gaussian_kde(values, bw_method=factor)(x), color=color, linewidth=2, zorder=3, label=label)
        else:
            ax.plot([], [], color=color, linewidth=2, label=label)
        ax.axvline(np.median(values), color=color, linewidth=1, linestyle=(0, (4, 3)), zorder=3)

    def stats(v):
        if not len(v):
            return "n 0"
        return (f"n {len(v)}   median {np.median(v):.3f}   p90 {np.percentile(v, 90):.3f}   "
                f"p99 {np.percentile(v, 99):.3f}   max {v.max():.3f}")

    lines = [f"ID: {stats(id_losses)}", f"run: {stats(run_losses)}"]
    if len(id_losses) and len(run_losses):
        p95, p99 = np.percentile(id_losses, 95), np.percentile(id_losses, 99)
        lines.append(f"run steps above ID p95 ({p95:.3f}): {np.mean(run_losses > p95) * 100:.1f}%   "
                     f"above ID p99 ({p99:.3f}): {np.mean(run_losses > p99) * 100:.1f}%")
    ax.text(0.0, 1.02, "\n".join(lines), transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=TEXT_SECONDARY, linespacing=1.5)
    fig.suptitle(f"state_ood_loss density: ID vs {subtitle_run}   (dashed: medians)", x=0.01, ha="left",
                 fontsize=11, color=TEXT_PRIMARY)

    ax.set_xlim(0, hi)
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
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def place_window(fig, corner, margin):
    """Move the figure window to a screen corner. The window manager would otherwise put it top-left,
    on top of the Isaac Sim viewport. Best effort: TkAgg (yuna_env's backend) and Qt are handled."""
    window = getattr(fig.canvas.manager, "window", None)
    if window is None:
        return
    try:
        if hasattr(window, "winfo_screenwidth"):  # TkAgg
            def apply():
                window.update_idletasks()
                # reqwidth/height include the toolbar; before the window is mapped winfo_width is 1
                w = max(window.winfo_width(), window.winfo_reqwidth())
                h = max(window.winfo_height(), window.winfo_reqheight())
                sw, sh = window.winfo_screenwidth(), window.winfo_screenheight()
                x = sw - w - margin if "right" in corner else margin
                y = sh - h - margin if "bottom" in corner else margin
                window.geometry(f"+{max(x, 0)}+{max(y, 0)}")

            apply()
            window.after(300, apply)  # again once mapped, with the real window size
        elif hasattr(window, "move"):  # Qt
            screen = window.screen().availableGeometry()
            w, h = window.width(), window.height()
            x = screen.right() - w - margin if "right" in corner else screen.left() + margin
            y = screen.bottom() - h - margin if "bottom" in corner else screen.top() + margin
            window.move(x, y)
    except Exception as e:  # never let placement break the plot
        print(f"[live_plot] could not position window: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference_csv", default=str(HERE / "outputs" / "state_id_loss_0915.csv"))
    parser.add_argument("--reference_seed", type=int, default=10)
    parser.add_argument("--interval_ms", type=int, default=100)
    parser.add_argument("--id_seeds", type=int, nargs="+", default=[10, 17, 18, 19, 30],
                        help="Seeds of --reference_csv pooled into the ID distribution of the saved density plot.")
    parser.add_argument("--corner", choices=["bottom-right", "bottom-left", "top-right", "top-left"],
                        default="bottom-right", help="Screen corner to open the window in (away from Isaac Sim's).")
    parser.add_argument("--margin", type=int, default=40, help="Pixels between the window and the screen edges.")
    parser.add_argument("--save_dir", default=str(HERE / "outputs"),
                        help="Where each finished rollout's plot is saved as a PNG.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.animation import FuncAnimation

    ref_steps, ref_loss, ref_note = [], [], ""
    try:
        ref = pd.read_csv(args.reference_csv, usecols=["seed", "trial", "step", "state_ood_loss"])
        ref = ref[ref.seed == args.reference_seed].dropna(subset=["state_ood_loss"])
        if len(ref):
            ref = ref[ref.trial == ref.trial.min()].sort_values("step")
            ref_steps, ref_loss = ref.step.tolist(), ref.state_ood_loss.tolist()
        else:
            ref_note = f"  (no seed {args.reference_seed} rows in {Path(args.reference_csv).name})"
    except (OSError, ValueError) as e:
        ref_note = f"  (reference unavailable: {e})"

    id_losses, id_label = [], "ID (reference unavailable)"
    try:
        idf = pd.read_csv(args.reference_csv, usecols=["seed", "state_ood_loss"]).dropna(subset=["state_ood_loss"])
        idf = idf[idf.seed.isin(args.id_seeds)]
        id_losses = idf.state_ood_loss.tolist()
        present = sorted(idf.seed.unique().tolist())
        id_label = f"ID: policy, seeds {', '.join(map(str, present))}" if present else "ID (no rows for --id_seeds)"
    except (OSError, ValueError):
        pass

    state = {"steps": [], "loss": [], "seed": None, "trial": None, "done": False, "to_save": [],
             "csv": None, "interrupted": False}
    lock = threading.Lock()

    def finish_current():
        # Queue the rollout on screen for saving. matplotlib is not thread-safe, so the GUI thread
        # (update() below) does the actual savefig.
        if state["steps"]:
            state["to_save"].append((list(state["steps"]), list(state["loss"]), state["seed"], state["trial"],
                                     state["csv"], state["interrupted"]))

    def read_stdin():
        for raw in sys.stdin:
            parts = raw.split()
            with lock:
                if raw.startswith("csv "):
                    state["csv"] = raw[4:].strip()
                elif parts == ["interrupted"]:
                    state["interrupted"] = True
                elif len(parts) == 3 and parts[0] == "reset":
                    finish_current()
                    state.update(steps=[], loss=[], seed=parts[1], trial=parts[2])
                elif len(parts) == 2:
                    try:
                        state["steps"].append(int(parts[0]))
                        state["loss"].append(float(parts[1]))
                    except ValueError:
                        pass
        with lock:
            finish_current()
            state["done"] = True

    threading.Thread(target=read_stdin, daemon=True).start()

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    fig.canvas.manager.set_window_title("state_ood_loss - SpaceMouse vs policy")
    ax.plot(ref_steps, ref_loss, color=REFERENCE, linewidth=1.5, zorder=2,
            label=f"policy, seed {args.reference_seed} (recorded)")
    (trace,) = ax.plot([], [], color=SPACEMOUSE, linewidth=2, zorder=3, label="SpaceMouse (live)")
    (dot,) = ax.plot([], [], "o", color=SPACEMOUSE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
                     zorder=4)
    readout = ax.text(0.99, 0.97, "", transform=ax.transAxes, ha="right", va="top", fontsize=10,
                      color=TEXT_PRIMARY)

    ax.set_xlabel("step", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("state_ood_loss", fontsize=10, color=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    title = ax.set_title("waiting for rollout...", loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.2)
    place_window(fig, args.corner, args.margin)

    save_dir = Path(args.save_dir)

    def draw(steps, loss, seed, trial, done, interrupted=False):
        trace.set_data(steps, loss)
        dot.set_data(steps[-1:], loss[-1:])
        x_hi = max([ref_steps[-1] if ref_steps else 0, steps[-1] if steps else 0]) + 5
        y_hi = max(ref_loss + loss + [0.5]) * 1.08
        ax.set_xlim(0, x_hi)
        ax.set_ylim(0, y_hi)
        run = f"SpaceMouse seed {seed} trial {trial}" if seed is not None else "waiting for rollout..."
        status = "   [interrupted]" if interrupted else ("   [run finished]" if done else "")
        title.set_text(f"{run} vs policy reference{ref_note}{status}")
        readout.set_text(f"step {steps[-1]}   loss {loss[-1]:.3f}" if steps else "")

    def update(_):
        with lock:
            pending, state["to_save"] = state["to_save"], []
            steps, loss = list(state["steps"]), list(state["loss"])
            seed, trial, done, interrupted = state["seed"], state["trial"], state["done"], state["interrupted"]
        for snap_steps, snap_loss, snap_seed, snap_trial, snap_csv, snap_interrupted in pending:
            draw(snap_steps, snap_loss, snap_seed, snap_trial, True, snap_interrupted)
            folder, stem = (Path(snap_csv).parent, Path(snap_csv).stem) if snap_csv else (save_dir, "spacemouse_live")
            folder.mkdir(parents=True, exist_ok=True)
            tag = "_interrupted" if snap_interrupted else ""
            out = folder / f"{stem}_seed{snap_seed}_trial{snap_trial}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.png"
            try:
                fig.savefig(out, dpi=150, facecolor=SURFACE)
                print(f"[live_plot] saved {out}", flush=True)
            except OSError as e:
                print(f"[live_plot] could not save {out}: {e}", flush=True)
            density_out = out.with_name(out.stem + "_density.png")
            try:
                run_name = f"SpaceMouse seed {snap_seed} trial {snap_trial}{' (interrupted)' if snap_interrupted else ''}"
                save_density_comparison(density_out, id_losses, id_label, snap_loss, run_name, run_name)
                print(f"[live_plot] saved {density_out}", flush=True)
            except Exception as e:  # never let the extra figure break the live window
                print(f"[live_plot] could not save {density_out}: {e}", flush=True)
        draw(steps, loss, seed, trial, done, interrupted)
        return trace, dot, readout, title

    _anim = FuncAnimation(fig, update, interval=args.interval_ms, cache_frame_data=False)  # noqa: F841 keep ref
    plt.show()


if __name__ == "__main__":
    main()
