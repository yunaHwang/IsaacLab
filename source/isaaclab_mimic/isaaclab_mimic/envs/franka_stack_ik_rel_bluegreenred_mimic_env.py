# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic env for the OOD blue -> green -> red stacking order.

WHY THIS EXISTS
---------------
``FrankaCubeStackIKRelMimicEnv.get_subtask_term_signals`` hardcodes the ID signal names::

    signals["grasp_1"] = subtask_terms["grasp_1"][env_ids]
    signals["grasp_2"] = subtask_terms["grasp_2"][env_ids]
    signals["stack_1"] = subtask_terms["stack_1"][env_ids]

The OOD env emits ``grasp_3`` / ``stack_3`` / ``grasp_2`` instead, so annotating with the OOD task
raised ``KeyError: 'grasp_1'``. That method is the fourth place the ID/OOD ordering was encoded by
commenting blocks in and out (the others being ``stack_env_cfg.py``,
``stack_ik_rel_visuomotor_env_cfg.py`` and ``mdp/terminations.py``).

Rather than add a fifth hardcoded list, this override derives the names from
``cfg.subtask_configs`` - the same ``subtask_term_signal`` strings the datagen pool looks up. The
env can then never disagree with its own cfg, so a new ordering needs no change here at all.
"""

from collections.abc import Sequence

import torch

from .franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class FrankaCubeStackBlueGreenRedIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    """Franka cube stack mimic env whose subtask signals follow cfg.subtask_configs."""

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Termination signal per subtask, named exactly as ``cfg.subtask_configs`` declares.

        The final subtask carries ``subtask_term_signal=None`` (its end needs no detection) and is
        skipped, matching the base class, which likewise emits no signal for the last subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        signals = dict()
        for eef_subtask_configs in self.cfg.subtask_configs.values():
            for subtask_config in eef_subtask_configs:
                name = subtask_config.subtask_term_signal
                if name is None:
                    continue
                if name not in subtask_terms:
                    raise KeyError(
                        f"subtask_term_signal '{name}' is declared in cfg.subtask_configs but the environment's"
                        f" observation group 'subtask_terms' only provides {sorted(subtask_terms.keys())}."
                        " The mimic cfg and the task's SubtaskCfg disagree."
                    )
                signals[name] = subtask_terms[name][env_ids]
        return signals
