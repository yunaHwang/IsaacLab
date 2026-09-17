"""CSV logger for multitask_dit_server.py's state-OOD scoring: one row per "step" request, holding
the loss's OUTPUT (state_ood_loss) next to its INPUTS (obs_history, raw_action) and the rollout's
seed.

WHERE THIS RUNS
    Server-side (multitask_dit_server.py), because that is the only process where obs_history and
    raw_action exist - the client (run_policy_fm.py) only ever sees the postprocessed physical
    action and the loss scalar. The seed is the opposite: only the client has it (--seed), so the
    client sends it (plus the trial index) in its {"cmd": "reset"} request and the server hands it
    to begin_episode() here. No lerobot import, so it is importable from either conda env.

WHAT IS IN A ROW
    seed, trial, step                         - identify the row. One run_policy_fm.py invocation
                                                seeds the env ONCE and then runs num_rollouts
                                                trials, so seed alone is not unique; trial is.
    state_ood_loss, state_ood_loss_error      - the output (error is set instead when it raised).
    raw_action_{d}                            - select_action's NORMALIZED action, [action_dim] -
                                                the exact tensor multitask_dit_loss scored.
    obs_hist{k}_state_{d}                     - obs_history[k]["observation.state"], normalized,
                                                k=0 oldest .. n_obs_steps-1 newest. At the start
                                                of an episode these repeat (the server pads the
                                                window with the first obs).
    obs_hist{k}_{cam}_{mean,std,min,max}      - per-camera image summary (images are 3xHxW each,
                                                too big for a CSV cell).
    tokens, attention_mask                    - the language conditioning, JSON lists; constant across the window, so one copy.

    All rows from every run, seed and trial accumulate in ONE csv (append mode); filter by
    seed/trial when reading it back.

    Everything is the normalized/preprocessed representation, i.e. exactly what the loss saw -
    un-normalized state is already available separately via STATE_LOG (lerobot_obs.py).
"""

import csv
import json
import os

import numpy as np
import torch

STATE_KEY = "observation.state"
IMAGE_KEYS = ("observation.images.table_cam", "observation.images.wrist_cam")


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    return np.asarray(x)


def _cam_name(key):
    return key.rsplit(".", 1)[-1]  # "observation.images.table_cam" -> "table_cam"


class StateOODCsvLogger:
    """Append-only CSV writer. The header is fixed by the first row written (dims depend on the
    checkpoint), so one CSV file should hold one checkpoint's runs - a later row with different
    dims raises rather than silently misaligning columns."""

    def __init__(self, path):
        self.path = path
        self.seed = None
        self.trial = None
        self.step = 0
        self._header = None
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, newline="") as f:
                self._header = next(csv.reader(f))

    def begin_episode(self, seed=None, trial=None):
        """Call on every {"cmd": "reset"}. Resets the per-episode step counter."""
        self.seed = seed
        self.trial = trial
        self.step = 0

    def log_step(self, obs_history, raw_action, state_ood_loss=None, state_ood_loss_error=None, extra=None):
        """Write one row. obs_history/raw_action are the same objects passed to
        multitask_dit_loss; state_ood_loss is the float response["state_ood_loss"]. extra: optional
        {column: value} placed right after the raw_action columns (the server's score_action uses it
        for the physical scored action)."""
        row = {
            "seed": "" if self.seed is None else self.seed,
            "trial": "" if self.trial is None else self.trial,
            "step": self.step,
            "state_ood_loss": "" if state_ood_loss is None else f"{float(state_ood_loss):.8g}",
            "state_ood_loss_error": state_ood_loss_error or "",
        }

        for d, v in enumerate(_np(raw_action).reshape(-1)):
            row[f"raw_action_{d}"] = f"{v:.8g}"
        if extra:
            row.update(extra)

        history = list(obs_history)
        for k, o in enumerate(history):
            for d, v in enumerate(_np(o[STATE_KEY]).reshape(-1)):
                row[f"obs_hist{k}_state_{d}"] = f"{v:.8g}"
            for key in IMAGE_KEYS:
                img = _np(o[key])
                cam = _cam_name(key)
                row[f"obs_hist{k}_{cam}_mean"] = f"{img.mean():.6g}"
                row[f"obs_hist{k}_{cam}_std"] = f"{img.std():.6g}"
                row[f"obs_hist{k}_{cam}_min"] = f"{img.min():.6g}"
                row[f"obs_hist{k}_{cam}_max"] = f"{img.max():.6g}"

        # Everything else in obs_history is the task/language conditioning (OBS_LANGUAGE_TOKENS /
        # OBS_LANGUAGE_ATTENTION_MASK). Constant across the window, so one copy from the newest
        # step - same as multitask_dit_loss.
        if history:
            for key, value in history[-1].items():
                if key != STATE_KEY and key not in IMAGE_KEYS:
                    row[key.rsplit(".", 1)[-1] if key.startswith("observation.language.") else key] = (
                        json.dumps(_np(value).reshape(-1).tolist())
                    )

        self._write(row)
        self.step += 1

    def _write(self, row):
        # A file deleted or emptied while the server keeps running would otherwise get rows with no
        # header (the header is cached from the first write) - start a fresh header in that case.
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            self._header = None
        write_header = self._header is None
        if write_header:
            self._header = list(row.keys())
        elif set(row) - set(self._header):
            raise ValueError(
                f"{self.path}: row has columns not in the existing header "
                f"({sorted(set(row) - set(self._header))[:5]}...) - different checkpoint dims? "
                "Point --ood_csv at a separate file for this checkpoint."
            )
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._header, restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

