# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic env cfg for the yellow task: blue -> yellow -> green, not the ID blue -> red -> green.

added! Yuna. Datagen settings (seed, noise, selection strategy, interpolation) are inherited
unchanged from the ID visuomotor mimic cfg, so an ID-vs-yellow comparison differs only in which
block goes in the middle - not in generation hyperparameters.

Three things are overridden, and all three are needed:
  * ``observations`` - the yellow subtask group, so grasp_1/stack_1 track cube_4 rather than cube_2.
  * ``terminations``  - success (and a dropping guard) rebased on cube_4.
  * ``subtask_configs`` - the mimic segmentation. Inheriting it would leave the first and last
    subtasks pointing at ``cube_2``: datagen would warp the motion to the RED block's pose while
    grasp_1/stack_1 waited on the YELLOW one, so every trial would run long and score a failure.

``object_ref="cube_4"`` needs no new env class: ``get_object_poses`` iterates over every rigid object
in the scene, so cube_4's pose is already exposed to data generation.
"""

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_yellow_env_cfg import (
    YellowObservationsCfg,
    add_yellow_block,
    use_yellow_as_middle,
)

from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg


@configclass
class FrankaCubeStackYellowIKRelVisuomotorMimicEnvCfg(FrankaCubeStackIKRelVisuomotorMimicEnvCfg):
    """Isaac Lab Mimic env cfg: Franka stack with cameras, blue -> yellow -> green."""

    observations: YellowObservationsCfg = YellowObservationsCfg()

    def __post_init__(self):
        # post init of parents: cameras, ID datagen settings, ID subtask_configs
        super().__post_init__()

        add_yellow_block(self)
        use_yellow_as_middle(self)
        self.datagen_config.name = "isaac_lab_franka_stack_ik_rel_visuomotor_yellow_D0"

        # Replaces the ID subtask_configs set by the parent's __post_init__. Identical to them in
        # every generation parameter; the only change is cube_2 -> cube_4 in the first and last
        # subtasks. Signal NAMES are unchanged - grasp_1 / stack_1 / grasp_2 are the three keys
        # FrankaCubeStackIKRelMimicEnv.get_subtask_term_signals reads, and the yellow observations
        # emit exactly those.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_4",  # THE CHANGE: was cube_2 (red)
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp yellow cube",
                next_subtask_description="Stack yellow cube on top of blue cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Unchanged: this subtask is generated relative to the object being stacked ONTO,
                # which is still the blue cube.
                object_ref="cube_1",
                subtask_term_signal="stack_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack yellow cube on top of blue cube",
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
                next_subtask_description="Stack green cube on top of yellow cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_4",  # THE CHANGE: was cube_2 - green now lands on the yellow block
                # End of the final subtask does not need to be detected.
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack green cube on top of yellow cube",
            )
        )
        self.subtask_configs["franka"] = subtask_configs
