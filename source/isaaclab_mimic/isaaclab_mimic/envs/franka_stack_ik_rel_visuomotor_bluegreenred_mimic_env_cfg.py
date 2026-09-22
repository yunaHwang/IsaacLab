# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic env cfg for the OOD blue -> green -> red stacking order, with cameras.

Before this file, ``Isaac-Stack-Cube-BlueGreenRed-Franka-IK-Rel-Visuomotor-Mimic-v0`` was registered
against ``FrankaCubeStackIKRelVisuomotorMimicEnvCfg`` - the SAME cfg as the in-distribution
``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0``. It was a no-op alias: generating with it
produced in-distribution data under an OOD name. This cfg makes that env id mean what it says.

It pairs:
  * scene/observations/terminations from ``FrankaCubeStackBlueGreenRedVisuomotorEnvCfg``
    (cameras + grasp_3/stack_3/grasp_2 + OOD success criterion), and
  * the four ``SubTaskConfig`` entries below, whose ``subtask_term_signal`` strings are resolved by
    name against that observation group.

Datagen settings are inherited unchanged from the ID visuomotor mimic cfg, so any OOD-vs-ID
comparison differs only in the goal ordering, not in generation hyperparameters.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_bluegreenred_env_cfg import (
    FrankaCubeStackBlueGreenRedVisuomotorEnvCfg,
)

from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg


@configclass
class FrankaCubeStackBlueGreenRedIKRelVisuomotorMimicEnvCfg(
    FrankaCubeStackBlueGreenRedVisuomotorEnvCfg, MimicEnvCfg
):
    """Isaac Lab Mimic env cfg: Franka cube stack, cameras, OOD blue -> green -> red."""

    def __post_init__(self):
        # post init of parents: cameras + OOD observations/terminations, then MimicEnvCfg defaults
        super().__post_init__()

        # Reuse every datagen setting from the ID visuomotor mimic cfg so OOD vs ID differs only in
        # the goal ordering. Copied field-by-field rather than by inheritance because this class
        # already inherits the OOD env cfg, and inheriting the ID mimic cfg as well would re-apply
        # the ID observations/terminations through its __post_init__ and silently undo the override.
        id_defaults = FrankaCubeStackIKRelVisuomotorMimicEnvCfg()
        id_defaults.__post_init__()
        self.datagen_config = id_defaults.datagen_config.copy()
        self.datagen_config.name = "isaac_lab_franka_stack_ik_rel_visuomotor_bluegreenred_D0"

        # OOD ordering: cube_1 = blue, cube_2 = red, cube_3 = green.
        # grasp green -> stack green on blue -> grasp red -> stack red on green (final, implicit).
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="grasp_3",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp green cube",
                next_subtask_description="Stack green cube on top of blue cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="stack_3",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack green cube on top of blue cube",
                next_subtask_description="Grasp red cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp red cube",
                next_subtask_description="Stack red cube on top of green cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                # End of the final subtask does not need to be detected.
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack red cube on top of green cube",
            )
        )
        self.subtask_configs["franka"] = subtask_configs
