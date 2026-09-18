# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# added! Yuna - same as FrankaCubeStackIKRelVisuomotorMimicEnvCfg (subtasks, seed, datagen settings)
# plus the yellow distractor block (cube_4).

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_yellow_env_cfg import (
    add_yellow_block,
)

from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg


@configclass
class FrankaCubeStackYellowIKRelVisuomotorMimicEnvCfg(FrankaCubeStackIKRelVisuomotorMimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Cube Stack IK Rel Visuomotor env with a yellow block.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        add_yellow_block(self)
        self.datagen_config.name = "isaac_lab_franka_stack_ik_rel_visuomotor_yellow_D0"
