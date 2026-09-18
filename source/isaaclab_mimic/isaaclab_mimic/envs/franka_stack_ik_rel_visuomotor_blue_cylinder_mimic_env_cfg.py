# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# added! Yuna - same as FrankaCubeStackIKRelVisuomotorMimicEnvCfg (subtasks, seed, datagen settings)
# plus the blue cylinder distractor (blue_cylinder).

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_blue_cylinder_env_cfg import (
    add_blue_cylinder,
)

from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg


@configclass
class FrankaCubeStackBlueCylinderIKRelVisuomotorMimicEnvCfg(FrankaCubeStackIKRelVisuomotorMimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Cube Stack IK Rel Visuomotor env with a blue cylinder.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        add_blue_cylinder(self)
        self.datagen_config.name = "isaac_lab_franka_stack_ik_rel_visuomotor_blue_cylinder_D0"
