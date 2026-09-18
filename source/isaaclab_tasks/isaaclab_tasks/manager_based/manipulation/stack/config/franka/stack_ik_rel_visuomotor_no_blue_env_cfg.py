# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# added! Yuna - Rel-Visuomotor stack env where the blue block (cube_1) is "gone" from the scene (eval only).
# cube_1 is NOT deleted (self.scene.cube_1 = None would KeyError in every term that names it: randomization,
# cube_1_dropping, stack_1, success, cube obs). Instead it stays as an invisible, non-colliding, kinematic
# phantom: the cameras don't see it, nothing touches it, and gravity doesn't move it. Subtasks, success,
# observations and terminations are unchanged, so every inference/replay script runs as is.
# Consequences: success (red on blue) can never fire; red/green keep the training layout distribution
# because the phantom is still part of randomize_cube_positions (min_separation included).

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .stack_ik_rel_visuomotor_env_cfg import FrankaCubeStackVisuomotorEnvCfg


def hide_blue_block(cfg):
    """Turn cube_1 into an invisible, non-colliding, kinematic phantom. Call after super().__post_init__()."""
    spawn = cfg.scene.cube_1.spawn
    spawn.visible = False
    spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    # new cfg object (not in-place) so the shared cube rigid_props of cube_2/cube_3 stay dynamic
    spawn.rigid_props = spawn.rigid_props.replace(kinematic_enabled=True)


@configclass
class FrankaCubeStackVisuomotorNoBlueEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        hide_blue_block(self)
