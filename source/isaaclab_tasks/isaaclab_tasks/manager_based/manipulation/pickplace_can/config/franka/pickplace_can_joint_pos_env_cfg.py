# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda pick-and-place-can, joint-position control.

robosuite's ``PickPlaceCan`` is itself a Panda task by default, so the robot choice matches the
source environment exactly. Only the controller differs: robosuite ships OSC_POSE, whereas this
is the joint-position variant. See the ``IK-Rel`` config for the closer analogue.
"""

import omni.log

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pickplace_can import mdp
from isaaclab_tasks.manager_based.manipulation.pickplace_can.can_asset import (
    CONVERSION_HINT,
    ROBOSUITE_CAN_USD,
    can_usd_available,
)
from isaaclab_tasks.manager_based.manipulation.pickplace_can.mdp import events as can_events
from isaaclab_tasks.manager_based.manipulation.pickplace_can.pickplace_can_env_cfg import PickPlaceCanEnvCfg
from isaaclab_tasks.manager_based.manipulation.pickplace_can.robosuite_layout import (
    BIN_SURFACE_Z,
    CAN_HEIGHT,
    CAN_MASS,
    CAN_RADIUS,
    CAN_REST_Z,
    CAN_SPAWN_X_RANGE,
    CAN_SPAWN_Y_RANGE,
    CAN_SPAWN_YAW_RANGE,
    target_bin_bounds,
)

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


# Solved by damped-least-squares IK against the Panda's modified-DH model: puts the TCP at
# (0.600, -0.150, 0.108), i.e. 0.20 m directly above the centre of the source bin, gripper
# pointing straight down, with the fingers open at 0.04.
FRANKA_READY_POSE = [-0.1156, 0.5173, -0.1333, -1.9566, 0.1054, 2.4676, 0.6952, 0.0400, 0.0400]


@configclass
class EventCfg:
    """Reset events, mirroring robosuite's ``_reset_internal`` for ``PickPlaceCan``."""

    init_franka_arm_pose = EventTerm(
        func=can_events.set_default_joint_pose,
        mode="reset",
        params={"default_pose": FRANKA_READY_POSE},
    )

    randomize_franka_joint_state = EventTerm(
        func=can_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Uniform over the source bin with a full random yaw -- robosuite's UniformRandomSampler.
    randomize_can_pose = EventTerm(
        func=can_events.randomize_can_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("can"),
            "pose_range": {
                "x": CAN_SPAWN_X_RANGE,
                "y": CAN_SPAWN_Y_RANGE,
                "z": (CAN_REST_Z, CAN_REST_Z),
                "yaw": CAN_SPAWN_YAW_RANGE,
            },
        },
    )


@configclass
class FrankaPickPlaceCanEnvCfg(PickPlaceCanEnvCfg):
    """Configuration for the Franka pick-and-place-can environment."""

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.plane.semantic_tags = [("class", "ground")]

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Used by the gripper-state checks in mdp/observations.py and mdp/terminations.py.
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # robosuite gives the can geom solimp="0.998 0.998 0.001" solref="0.001 1", i.e. a very
        # stiff contact. The PhysX analogue is a high solver iteration count.
        can_rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        # robosuite: density=100 on a 1.4646e-4 m^3 mesh -> 14.65 g.
        can_mass_props = MassPropertiesCfg(mass=CAN_MASS)
        # robosuite: friction="0.95 0.3 0.1" (sliding, torsional, rolling). PhysX only models
        # the sliding term.
        can_physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.95, dynamic_friction=0.95, restitution=0.0
        )

        if can_usd_available():
            can_spawn = UsdFileCfg(
                usd_path=ROBOSUITE_CAN_USD,
                scale=(1.0, 1.0, 1.0),
                rigid_props=can_rigid_props,
                mass_props=can_mass_props,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                semantic_tags=[("class", "can")],
            )
        else:
            omni.log.warn(CONVERSION_HINT)
            can_spawn = sim_utils.CylinderCfg(
                radius=CAN_RADIUS,
                height=CAN_HEIGHT,
                rigid_props=can_rigid_props,
                mass_props=can_mass_props,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=can_physics_material,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.72, 0.11, 0.11), roughness=0.35, metallic=0.5
                ),
                semantic_tags=[("class", "can")],
            )

        self.scene.can = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Can",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[CAN_SPAWN_X_RANGE[0], CAN_SPAWN_Y_RANGE[0], CAN_REST_Z], rot=[1, 0, 0, 0]
            ),
            spawn=can_spawn,
        )

        # End-effector frames, identical to the stack task's.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )


@configclass
class FrankaPickPlaceCanNearBinEnvCfg(FrankaPickPlaceCanEnvCfg):
    """Reachability-friendly variant: target the *near* quadrant of bin2.

    This is a deliberate deviation from robosuite, not a port of it.

    In robosuite the can is object id 3, which ``not_in_bin`` maps to the +x/+y quadrant of
    bin2 -- the corner furthest from the robot. With the Panda at robosuite's own base offset
    that quadrant is only reachable near its inner corner: IK against the Panda's DH model
    solves at (0.600, 0.380) and (0.660, 0.430), but fails at the quadrant centre (0.860 m
    radial) and its far corner (1.011 m). The task is still solvable as shipped -- a
    demonstrator just has to release the can over the inner corner -- but the usable drop zone
    is a narrow wedge, which makes teleoperation and mimic data generation fiddly.

    This variant reassigns the can to quadrant 0 (the one robosuite gives the milk carton),
    spanning x in (0.405, 0.600) and y in (0.135, 0.380). Every corner of it solves, the
    furthest being 0.710 m radial. Nothing else changes: same arena, same can, same dynamics.
    """

    def __post_init__(self):
        super().__post_init__()

        x_range, y_range = target_bin_bounds(bin_id=0)
        target_center = ((x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2, BIN_SURFACE_Z)

        self.terminations.success = DoneTerm(
            func=mdp.can_in_target_bin,
            params={"x_range": x_range, "y_range": y_range},
        )
        # Keep the goal the policy observes consistent with the goal it is scored against.
        self.observations.policy.target_position.params = {"target_center": target_center}
        self.observations.policy.object.params = {"target_center": target_center}
