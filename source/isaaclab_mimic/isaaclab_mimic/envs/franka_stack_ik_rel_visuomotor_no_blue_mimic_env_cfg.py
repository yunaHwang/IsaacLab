# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# added! Yuna - same as FrankaCubeStackIKRelVisuomotorMimicEnvCfg (subtasks, seed, datagen settings)
# but the blue block (cube_1) is an invisible, non-colliding phantom. For policy evaluation only:
# Mimic generation can't succeed here (stack_1 / success need a real blue block).

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_no_blue_env_cfg import (
    hide_blue_block,
)

from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg


@configclass
class FrankaCubeStackNoBlueIKRelVisuomotorMimicEnvCfg(FrankaCubeStackIKRelVisuomotorMimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Cube Stack IK Rel Visuomotor env without the blue block.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        hide_blue_block(self)
        self.datagen_config.name = "isaac_lab_franka_stack_ik_rel_visuomotor_no_blue_D0"
