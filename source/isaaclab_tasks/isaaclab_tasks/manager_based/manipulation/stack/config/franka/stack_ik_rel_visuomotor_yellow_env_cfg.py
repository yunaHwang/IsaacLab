# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rel-Visuomotor stack env whose MIDDLE block is yellow (cube_4) instead of red (cube_2).

added! Yuna. The scene is the ID scene plus ``cube_4``, a yellow block. It began as a pure visual
distractor, referenced by no subtask, success or termination term. It is now part of the goal, which
is what makes this task distinct from the ID one:

    ID   : blue -> RED   -> green   i.e. grasp red (cube_2),    stack on blue, grasp green, stack on red
    here : blue -> YELLOW -> green  i.e. grasp yellow (cube_4), stack on blue, grasp green, stack on yellow

The red block (cube_2) stays in the scene and becomes the distractor - the mirror of the old setup.
A policy trained on ID has to notice the middle block changed colour, which is the point.

WHY THIS FILE DEFINES ITS OWN SUBTASK + SUCCESS TERMS
-----------------------------------------------------
``stack_ik_rel_visuomotor_env_cfg.ObservationsCfg.SubtaskCfg`` and ``mdp.cubes_stacked`` are shared
with the ID task and are toggled by hand between stacking orders (both files carry commented-out
alternatives). Inheriting them would mean this env's goal silently follows whatever the ID files were
last toggled to - and they cannot express "stack the yellow block" at all, since they name cube_2.
So the two terms that define the goal are written out explicitly below, the same way
``stack_ik_rel_visuomotor_blue_cylinder_env_cfg.py`` and the BlueGreenRed cfg do."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from .stack_ik_rel_visuomotor_env_cfg import FrankaCubeStackVisuomotorEnvCfg
from .stack_ik_rel_visuomotor_env_cfg import ObservationsCfg as IDObservationsCfg


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


def use_yellow_as_middle(cfg):
    """Point the success + dropping terminations at cube_4. Call after ``add_yellow_block``.

    Kept as a function so the plain env cfg and the mimic env cfg apply exactly the same terms;
    duplicating them is how the two drifted apart on the BlueGreenRed task.
    """
    # Same stacking maths as ID, with cube_4 substituted for the middle object. cubes_stacked()
    # checks cube_1 below cube_2_cfg below cube_3_cfg, reading only root_pos_w, so any block works
    # in any slot; cube_4 is a standard block, so the default height_diff still applies.
    cfg.terminations.success = DoneTerm(
        func=mdp.cubes_stacked,
        params={
            "cube_1_cfg": SceneEntityCfg("cube_1"),  # bottom: blue
            "cube_2_cfg": SceneEntityCfg("cube_4"),  # middle: YELLOW, not red
            "cube_3_cfg": SceneEntityCfg("cube_3"),  # top: green
        },
    )
    # cube_1..3 already get this guard in StackEnvCfg.TerminationsCfg; cube_4 is now load-bearing
    # too, and a dropped middle block makes the episode unrecoverable.
    cfg.terminations.cube_4_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_4")},
    )


@configclass
class YellowObservationsCfg(IDObservationsCfg):
    """ID visuomotor observations with the subtask group rebuilt around the yellow block.

    Subclassing keeps PolicyCfg - both cameras, their resolution, every state term - byte-for-byte
    identical to the training distribution. Only ``subtask_terms`` changes, so an ID-vs-this
    comparison differs in the middle block alone.
    """

    @configclass
    class SubtaskCfg(ObsGroup):
        """Signals for: grasp yellow -> stack yellow on blue -> grasp green (-> stack on yellow).

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
                "object_cfg": SceneEntityCfg("cube_4"),  # THE CHANGE: yellow, was cube_2 (red)
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_4"),  # THE CHANGE: yellow, was cube_2
                "lower_object_cfg": SceneEntityCfg("cube_1"),  # blue - same as ID
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
class FrankaCubeStackVisuomotorYellowEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """Franka cube stack, cameras on, blue -> yellow -> green goal ordering."""

    observations: YellowObservationsCfg = YellowObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        add_yellow_block(self)
        # terminations come from StackEnvCfg and are untouched by the visuomotor cfg, so without
        # this the env would render a yellow-goal scene while still scoring the ID red stack.
        use_yellow_as_middle(self)
