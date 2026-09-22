# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Visuomotor (camera) env cfg for the OOD blue -> green -> red stacking order.

WHY THIS FILE EXISTS
--------------------
The OOD ordering already existed in two places, but neither was usable with cameras:

  * ``stack_ik_rel_env_cfg.FrankaCubeStackBlueGreenRedEnvCfg`` has the right terminations and
    ``subtask_configs``, but inherits the STATE-only observation group - no table_cam/wrist_cam.
  * ``stack_env_cfg.ObservationsCfg.SubtaskCfg`` has ``grasp_3``/``stack_3`` live, but that class is
    DEAD CODE on the visuomotor path: ``stack_ik_rel_visuomotor_env_cfg`` declares its own
    ``ObservationsCfg`` (a fresh class, not a subclass) and reassigns ``observations`` on
    ``FrankaCubeStackVisuomotorEnvCfg``, so the base group is shadowed and never instantiated.

So a camera env asking for ``subtask_term_signal="grasp_3"`` would find no such signal: the
visuomotor observation group only emits ``grasp_1``/``stack_1``/``grasp_2``. This file supplies a
visuomotor-side ``SubtaskCfg`` that emits the OOD signals, as real terms rather than the
commented-out block the two files have been toggling by hand.

WHAT IS OVERRIDDEN, AND WHY BOTH ARE NEEDED
-------------------------------------------
  * ``observations`` - new SubtaskCfg (grasp_3 / stack_3 / grasp_2). PolicyCfg is reused verbatim
    from the ID visuomotor cfg, so the cameras, their resolution and the state terms are byte-for-byte
    the same as the training distribution. Only the SUBTASK GOAL differs, which is the point of the
    OOD comparison.
  * ``terminations.success`` - ``cubes_stacked`` remapped to cube_1 <- cube_3 <- cube_2. This is NOT
    inherited from ``FrankaCubeStackBlueGreenRedEnvCfg`` (we descend from the visuomotor cfg
    instead), and it is NOT covered by the observations override: terminations come from
    ``StackEnvCfg`` and the visuomotor cfg never touches them. Miss this and the env would render
    OOD scenes while still scoring success by the ID stacking order.

CUBE COLOURS (from stack_joint_pos_env_cfg): cube_1 = blue, cube_2 = red, cube_3 = green.
  ID  (blue -> red -> green): grasp red,   stack on blue, grasp green, stack on red.
  OOD (blue -> green -> red): grasp green, stack on blue, grasp red,   stack on green.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from .stack_ik_rel_visuomotor_env_cfg import FrankaCubeStackVisuomotorEnvCfg
from .stack_ik_rel_visuomotor_env_cfg import ObservationsCfg as IDObservationsCfg


@configclass
class BlueGreenRedObservationsCfg(IDObservationsCfg):
    """ID visuomotor observations with the subtask group swapped for the OOD ordering.

    Subclassing IDObservationsCfg keeps PolicyCfg (both cameras + all state terms) identical to the
    training distribution; only ``subtask_terms`` is replaced below.
    """

    @configclass
    class SubtaskCfg(ObsGroup):
        """OOD variant (blue -> green -> red).

        grasp_3: grasp green (cube_3); stack_3: green on blue (cube_3 on cube_1);
        grasp_2: grasp red (cube_2); the final stack (red on green) is implicit, as in the ID cfg.

        Names are kept as grasp_3/stack_3/grasp_2 to match
        ``stack_ik_rel_env_cfg.FrankaCubeStackBlueGreenRedEnvCfg``'s ``subtask_term_signal`` values -
        the mimic subtask configs look these up by string, so they must agree exactly.
        """

        grasp_3 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )
        stack_3 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_3"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaCubeStackBlueGreenRedVisuomotorEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """Franka cube stack, cameras on, OOD blue -> green -> red goal ordering."""

    observations: BlueGreenRedObservationsCfg = BlueGreenRedObservationsCfg()

    def __post_init__(self):
        # post init of parent: robot, IK action, cameras, events
        super().__post_init__()

        # Success must follow the OOD order too. terminations is inherited from StackEnvCfg and is
        # NOT touched by the visuomotor cfg, so without this the env would score the ID order.
        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": SceneEntityCfg("cube_2"),
            },
        )
