# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rel-Visuomotor stack env whose stack BASE is a blue cylinder instead of the blue cube.

added! Yuna. The scene is the ID scene plus ``blue_cylinder``: same colour and texture as the blue
cube (cube_1), different shape. It began as a pure visual distractor, referenced by no subtask,
success or termination term. It is now the OBJECT THE STACK IS BUILT ON, which is what makes this
task distinct from the ID one:

    ID   : grasp red (cube_2) -> stack on blue CUBE (cube_1)     -> grasp green (cube_3) -> stack on red
    here : grasp red (cube_2) -> stack on blue CYLINDER          -> grasp green (cube_3) -> stack on red

The blue cube (cube_1) stays in the scene and becomes the distractor - the mirror of the old setup.
A policy trained on ID has to tell the two blue objects apart by shape, which is the point.

WHY THIS FILE DEFINES ITS OWN SUBTASK + SUCCESS TERMS
-----------------------------------------------------
``stack_ik_rel_visuomotor_env_cfg.ObservationsCfg.SubtaskCfg`` and ``mdp.cubes_stacked`` are shared
with the ID task and are toggled by hand between stacking orders (both files carry commented-out
alternatives). Inheriting them would mean this env's goal silently follows whatever the ID files were
last toggled to - and they cannot express "stack on the cylinder" at all, since they name cube_1.
So the two terms that define the goal are written out explicitly below, the same way
``stack_ik_rel_visuomotor_bluegreenred_env_cfg.py`` does it for the OOD ordering."""

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Gf, Sdf, Usd, UsdShade

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from .stack_ik_rel_visuomotor_env_cfg import FrankaCubeStackVisuomotorEnvCfg
from .stack_ik_rel_visuomotor_env_cfg import ObservationsCfg as IDObservationsCfg

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


def use_blue_cylinder_as_base(cfg):
    """Point the success + dropping terminations at the cylinder. Call after ``add_blue_cylinder``.

    Kept as a function so the plain env cfg and the mimic env cfg apply exactly the same terms;
    duplicating them is how the two drifted apart on the BlueGreenRed task.
    """
    # Same stacking maths as the ID task, with the cylinder substituted for the bottom object.
    # cubes_stacked() only reads root_pos_w of whatever entities it is handed, so a cylinder works
    # as cube_1_cfg; its height (0.047 m) equals the block height the default height_diff assumes.
    cfg.terminations.success = DoneTerm(
        func=mdp.cubes_stacked,
        params={
            "cube_1_cfg": SceneEntityCfg("blue_cylinder"),  # bottom: the CYLINDER, not cube_1
            "cube_2_cfg": SceneEntityCfg("cube_2"),  # middle: red
            "cube_3_cfg": SceneEntityCfg("cube_3"),  # top: green
        },
    )
    # The base of the stack falling off the table is unrecoverable, so end the episode - the same
    # guard cube_1 gets in StackEnvCfg.TerminationsCfg, which no longer covers the base object here.
    cfg.terminations.blue_cylinder_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("blue_cylinder")},
    )


@configclass
class BlueCylinderObservationsCfg(IDObservationsCfg):
    """ID visuomotor observations with the subtask group rebased on the cylinder.

    Subclassing keeps PolicyCfg - both cameras, their resolution, every state term - byte-for-byte
    identical to the training distribution. Only ``subtask_terms`` changes, so an ID-vs-this
    comparison differs in the goal object alone.
    """

    @configclass
    class SubtaskCfg(ObsGroup):
        """Signals for: grasp red -> stack red on the blue CYLINDER -> grasp green (-> stack on red).

        The names stay grasp_1 / stack_1 / grasp_2 because they are looked up by string in three
        places that must agree: the mimic cfg's ``subtask_term_signal`` values, and
        ``FrankaCubeStackIKRelMimicEnv.get_subtask_term_signals`` which reads exactly these three
        keys. Only the objects they refer to differ from ID.
        """

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),  # red - same as ID
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),  # red
                "lower_object_cfg": SceneEntityCfg("blue_cylinder"),  # THE CHANGE: was cube_1
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),  # green - same as ID
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaCubeStackVisuomotorBlueCylinderEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """Franka cube stack, cameras on, stacked on the blue cylinder instead of the blue cube."""

    observations: BlueCylinderObservationsCfg = BlueCylinderObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        add_blue_cylinder(self)
        # terminations come from StackEnvCfg and are untouched by the visuomotor cfg, so without
        # this the env would render a cylinder-based scene while still scoring the ID cube stack.
        use_blue_cylinder_as_base(self)
