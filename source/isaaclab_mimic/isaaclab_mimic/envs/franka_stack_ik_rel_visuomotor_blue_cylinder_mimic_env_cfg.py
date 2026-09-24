# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic env cfg for the blue-cylinder task: the stack is built on the cylinder, not the blue cube.

added! Yuna. Datagen settings (seed, noise, selection strategy, interpolation) are inherited
unchanged from the ID visuomotor mimic cfg, so an ID-vs-cylinder comparison differs only in which
object the stack sits on - not in generation hyperparameters.

Three things are overridden, and all three are needed:
  * ``observations`` - the cylinder-based subtask group, so ``stack_1`` fires on "red on cylinder".
  * ``terminations``  - success (and a dropping guard) rebased on the cylinder.
  * ``subtask_configs`` - the mimic segmentation, whose second subtask must be relative to the
    cylinder's frame. Inheriting it would leave ``object_ref="cube_1"``: datagen would then warp the
    red cube's placement to the blue CUBE's pose while ``stack_1`` waited for it to land on the
    CYLINDER, so every trial would run to the end and be scored a failure.

``object_ref="blue_cylinder"`` needs no new env class: ``get_object_poses`` iterates over every rigid
object in the scene, so the cylinder's pose is already exposed to data generation.
"""

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_blue_cylinder_env_cfg import (
    BlueCylinderObservationsCfg,
    add_blue_cylinder,
    use_blue_cylinder_as_base,
)

from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg


@configclass
class FrankaCubeStackBlueCylinderIKRelVisuomotorMimicEnvCfg(FrankaCubeStackIKRelVisuomotorMimicEnvCfg):
    """Isaac Lab Mimic env cfg: Franka stack with cameras, built on the blue cylinder."""

    observations: BlueCylinderObservationsCfg = BlueCylinderObservationsCfg()

    def __post_init__(self):
        # post init of parents: cameras, ID datagen settings, ID subtask_configs
        super().__post_init__()

        add_blue_cylinder(self)
        use_blue_cylinder_as_base(self)
        self.datagen_config.name = "isaac_lab_franka_stack_ik_rel_visuomotor_blue_cylinder_D0"

        # Replaces the ID subtask_configs set by the parent's __post_init__. Identical to them in
        # every generation parameter; the only change is the second subtask's object_ref, which moves
        # from cube_1 (blue cube) to blue_cylinder. Signal NAMES are unchanged - grasp_1 / stack_1 /
        # grasp_2 are the three keys FrankaCubeStackIKRelMimicEnv.get_subtask_term_signals reads, and
        # the cylinder observations emit exactly those.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp red cube",
                next_subtask_description="Stack red cube on top of blue cylinder",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # THE CHANGE: was cube_1. This subtask's motion is generated relative to the frame of
                # the object being stacked ONTO, which is now the cylinder.
                object_ref="blue_cylinder",
                subtask_term_signal="stack_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack red cube on top of blue cylinder",
                next_subtask_description="Grasp green cube",
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
                description="Grasp green cube",
                next_subtask_description="Stack green cube on top of red cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                # End of the final subtask does not need to be detected.
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack green cube on top of red cube",
            )
        )
        self.subtask_configs["franka"] = subtask_configs
