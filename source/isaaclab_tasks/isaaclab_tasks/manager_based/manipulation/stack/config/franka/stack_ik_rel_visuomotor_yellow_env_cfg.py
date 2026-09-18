# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# added! Yuna - Rel-Visuomotor stack env with an extra yellow block (cube_4) as a distractor.
# cube_4 is not referenced by any subtask, success, termination or state observation term;
# it only shows up in the camera images and in the recorded states / datagen_info.

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .stack_ik_rel_visuomotor_env_cfg import FrankaCubeStackVisuomotorEnvCfg


def add_yellow_block(cfg):
    """Add yellow cube_4 as a distractor. Call after super().__post_init__()."""
    cfg.scene.cube_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_4",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0.10, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/yellow_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cfg.scene.cube_1.spawn.rigid_props,  # same physics as cube_1..3
            semantic_tags=[("class", "cube_4")],
        ),
    )
    # new list (not .append) so the shared default EventCfg is never mutated
    cfg.events.randomize_cube_positions.params["asset_cfgs"] = [
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
        SceneEntityCfg("cube_4"),
    ]


@configclass
class FrankaCubeStackVisuomotorYellowEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        add_yellow_block(self)
