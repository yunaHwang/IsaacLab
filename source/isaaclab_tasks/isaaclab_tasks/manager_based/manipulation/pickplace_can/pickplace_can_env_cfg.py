# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for an Isaac Lab port of robosuite's ``PickPlaceCan``.

This is the robot-agnostic half: the two-bin arena, the observation/termination structure, and
the simulation settings. The robot, its actions and the can asset are filled in by the configs
under ``config/<robot>/``.

See :mod:`.robosuite_layout` for where every geometric constant comes from.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp
from .robosuite_layout import (
    BIN1_POS,
    BIN1_WALLS,
    BIN2_POS,
    BIN2_WALLS,
    BIN_FLOOR_HALF,
    BIN_LEG_HALF_HEIGHT,
    BIN_LEG_OFFSETS,
    BIN_LEG_RADIUS,
    GROUND_Z,
)

##
# robosuite's bins are textured wood (light-wood for the source bin, dark-wood for the target).
# Textures are not reproduced here -- flat colours in the same family keep the two bins visually
# distinguishable without dragging robosuite's PNGs into the USD stage.
##
LIGHT_WOOD_COLOR = (0.78, 0.63, 0.44)
DARK_WOOD_COLOR = (0.35, 0.22, 0.13)
METAL_COLOR = (0.55, 0.56, 0.58)


def _add_bin(
    scene: InteractiveSceneCfg,
    name: str,
    position: tuple[float, float, float],
    walls: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
    color: tuple[float, float, float],
) -> None:
    """Attach one robosuite bin to ``scene`` as a set of static collider prims.

    robosuite's bins are a handful of MuJoCo box geoms, so they are rebuilt here out of
    ``CuboidCfg`` primitives rather than shipped as a USD asset -- that keeps the collision
    geometry exactly as the demonstrations saw it and needs no asset conversion.

    Each geom is spawned with ``collision_props`` but no ``rigid_props``, which makes it static
    world geometry. MuJoCo ``size`` is a half-extent while ``CuboidCfg.size`` is a full extent,
    hence the factors of two.

    Prim paths are flat (``Bin1_Wall_0``) rather than nested under a ``Bin1`` Xform, because
    ``InteractiveScene`` spawns each entity independently and cannot create an intermediate
    parent prim for them.
    """
    floor_cfg = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}_Floor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=tuple(2 * h for h in BIN_FLOOR_HALF),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.9),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # robosuite: friction="1 0.005 0.0001" on the bin geoms.
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            semantic_tags=[("class", name.lower())],
        ),
    )
    setattr(scene, f"{name.lower()}_floor", floor_cfg)

    for i, (offset, half) in enumerate(walls):
        wall_cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}_Wall_{i}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(p + o for p, o in zip(position, offset))),
            spawn=sim_utils.CuboidCfg(
                size=tuple(2 * h for h in half),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.9),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
                semantic_tags=[("class", name.lower())],
            ),
        )
        setattr(scene, f"{name.lower()}_wall_{i}", wall_cfg)

    # Visual-only pedestal legs, matching robosuite's contype=0/conaffinity=0 cylinders. They run
    # from the bin underside down to the floor plane.
    for i, (dx, dy) in enumerate(BIN_LEG_OFFSETS):
        leg_cfg = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}_Leg_{i}",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(position[0] + dx, position[1] + dy, position[2] - BIN_LEG_HALF_HEIGHT)
            ),
            spawn=sim_utils.CylinderCfg(
                radius=BIN_LEG_RADIUS,
                height=2 * BIN_LEG_HALF_HEIGHT,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=METAL_COLOR, metallic=0.8, roughness=0.3
                ),
            ),
        )
        setattr(scene, f"{name.lower()}_leg_{i}", leg_cfg)


##
# Scene definition
##
@configclass
class CanBinsSceneCfg(InteractiveSceneCfg):
    """The robosuite ``BinsArena``: a source bin, a four-quadrant target bin, and a robot.

    The robot, the ``ee_frame`` sensor and the ``can`` rigid object are supplied by the
    robot-specific configs.
    """

    # Populated by the agent env cfg.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # robosuite's floor is at world z=0, which is 0.912 m below the Panda base.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, GROUND_Z]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        # Bins are added procedurally because each one is a dozen separate collider prims.
        _add_bin(self, "Bin1", BIN1_POS, BIN1_WALLS, LIGHT_WOOD_COLOR)
        _add_bin(self, "Bin2", BIN2_POS, BIN2_WALLS, DARK_WOOD_COLOR)


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Low-dimensional state observations.

        The term names match the ``stack`` task's so the same robomimic configs
        (``eef_pos``/``eef_quat``/``gripper_pos``/``object``) and the same
        ``isaaclab_mimic`` plumbing apply unchanged.
        """

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        can_position = ObsTerm(func=mdp.can_position_in_world_frame)
        can_orientation = ObsTerm(func=mdp.can_orientation_in_world_frame)
        target_position = ObsTerm(func=mdp.target_bin_position)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Camera observations. Populated by the visuomotor variant."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Binary subtask signals consumed by ``isaaclab_mimic`` data generation.

        ``PickPlaceCan`` is a two-stage task: grasp the can, then place it in the target bin.
        Only the first needs a termination signal -- mimic infers the end of the final subtask
        from the end of the episode.
        """

        grasp_can = ObsTerm(
            func=mdp.can_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "can_cfg": SceneEntityCfg("can"),
            },
        )
        lift_can = ObsTerm(
            func=mdp.can_lifted,
            params={"can_cfg": SceneEntityCfg("can")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # The can fell off the bins onto the floor. The floor is 0.912 m below the base; anything
    # below the bin underside is unrecoverable.
    can_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": BIN1_POS[2] - 0.05, "asset_cfg": SceneEntityCfg("can")},
    )

    success = DoneTerm(func=mdp.can_in_target_bin)


@configclass
class PickPlaceCanEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the robosuite-style pick-and-place-can environment."""

    scene: CanBinsSceneCfg = CanBinsSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers -- this is a demonstration/imitation task, not an RL one. robosuite does
    # define a shaped reach/grasp/lift/hover reward for PickPlace; it is deliberately not ported
    # because the robomimic `can` datasets are behaviour cloning and never use it.
    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 5
        # robosuite's PickPlace uses horizon=500 at control_freq=20 Hz, i.e. 25 s.
        self.episode_length_s = 25.0

        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
