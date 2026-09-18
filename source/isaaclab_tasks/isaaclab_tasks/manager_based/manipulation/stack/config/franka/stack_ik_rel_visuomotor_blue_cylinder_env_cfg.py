# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# added! Yuna - Rel-Visuomotor stack env with an extra blue cylinder (blue_cylinder) as a distractor.
# Same color/texture as the blue cube (cube_1), different shape. Like the yellow block, it is not
# referenced by any subtask, success, termination or state observation term; it only shows up in the
# camera images and in the recorded states / datagen_info.

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Gf, Sdf, Usd, UsdShade

from .stack_ik_rel_visuomotor_env_cfg import FrankaCubeStackVisuomotorEnvCfg

# OmniPBR materials used by the Nucleus blocks (texture * tint): /Looks/{Red,Green,Blue,Yellow}
BLOCK_MATERIALS_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/Materials/Materials.usd"


def spawn_block_material(prim_path: str, cfg: "BlockMaterialCfg") -> Usd.Prim:
    """Reference one of the Nucleus block materials, so the shape gets the exact cube texture and tint.

    Isaac Lab's MdlFileCfg can't set texture inputs (string values are rejected), so the material is
    referenced from the blocks' own Materials.usd instead. Relative texture paths resolve against that file.
    """
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    prim = stage.DefinePrim(prim_path)
    prim.GetReferences().AddReference(BLOCK_MATERIALS_USD, f"/Looks/{cfg.color}")
    # optional: project the texture in object space (for shapes without UVs)
    if cfg.project_uvw is not None:
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"{prim_path}/Shader"))
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(cfg.project_uvw)
        shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
        if cfg.texture_scale is not None:
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*cfg.texture_scale))
    return prim


@configclass
class BlockMaterialCfg(sim_utils.VisualMaterialCfg):
    """Visual material taken from the Nucleus block materials."""

    func: Callable = spawn_block_material
    color: str = "Blue"
    """One of "Red", "Green", "Blue", "Yellow"."""
    project_uvw: bool | None = None
    texture_scale: tuple[float, float] | None = None


def add_blue_cylinder(cfg):
    """Add a blue cylinder as a distractor. Call after super().__post_init__()."""
    cfg.scene.blue_cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Blue_Cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0.10, 0.0203], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CylinderCfg(
            # same footprint/height as the blocks: basic_block.usd mesh is +-0.94 * 0.025 = 4.7 cm
            radius=0.0235,
            height=0.047,
            axis="Z",
            rigid_props=cfg.scene.cube_1.spawn.rigid_props,  # same physics as cube_1..3
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),  # block mass (basic_block.usd)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # block physics material (basic_block.usd /Root/PhysicsMaterial)
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=1.0, restitution=0.0, friction_combine_mode="max"
            ),
            # the implicit Cylinder has no UVs -> project the texture in object space, ~1 tile per 4.7 cm
            visual_material=BlockMaterialCfg(color="Blue", project_uvw=True, texture_scale=(21.3, 21.3)),
            semantic_tags=[("class", "blue_cylinder")],
        ),
    )
    # new list (not .append) so the shared default EventCfg is never mutated
    cfg.events.randomize_cube_positions.params["asset_cfgs"] = [
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
        SceneEntityCfg("blue_cylinder"),
    ]


@configclass
class FrankaCubeStackVisuomotorBlueCylinderEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        add_blue_cylinder(self)
