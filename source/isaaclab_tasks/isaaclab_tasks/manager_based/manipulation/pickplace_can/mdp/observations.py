# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the robosuite-style pick-and-place-can task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from ..robosuite_layout import CAN_TARGET_CENTER

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def can_position_in_world_frame(
    env: ManagerBasedRLEnv, can_cfg: SceneEntityCfg = SceneEntityCfg("can")
) -> torch.Tensor:
    """The position of the can in the world frame."""
    can: RigidObject = env.scene[can_cfg.name]
    return can.data.root_pos_w


def can_orientation_in_world_frame(
    env: ManagerBasedRLEnv, can_cfg: SceneEntityCfg = SceneEntityCfg("can")
) -> torch.Tensor:
    """The orientation of the can in the world frame."""
    can: RigidObject = env.scene[can_cfg.name]
    return can.data.root_quat_w


def target_bin_position(
    env: ManagerBasedRLEnv,
    target_center: tuple[float, float, float] = CAN_TARGET_CENTER,
) -> torch.Tensor:
    """Centre of the can's target quadrant of bin2, in the env-local frame.

    Constant per environment, but exposed as an observation so a policy trained here transfers
    to variants that move the goal (see the ``NearBin`` config, which retargets the quadrant).
    """
    return torch.tensor(target_center, device=env.device).repeat(env.num_envs, 1)


def object_obs(
    env: ManagerBasedRLEnv,
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_center: tuple[float, float, float] = CAN_TARGET_CENTER,
):
    """Object observations, mirroring robosuite's ``object-state`` for ``PickPlaceCan``.

    Layout (23 dims)::

        can pos (env-local)        3
        can quat                   4
        gripper-to-can             3
        can-to-target              3
        target pos (env-local)     3
        gripper pos (env-local)    3
        gripper quat               4

    robosuite exposes ``Can_pos``, ``Can_quat``, ``Can_to_robot0_eef_pos`` and
    ``Can_to_robot0_eef_quat``. The can-to-target and target terms are added because, unlike
    robosuite, nothing else in the observation tells the policy where the goal bin is.
    """
    can: RigidObject = env.scene[can_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    can_pos_w = can.data.root_pos_w
    can_quat_w = can.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    target_pos = target_bin_position(env, target_center=target_center)
    target_pos_w = target_pos + env.scene.env_origins

    gripper_to_can = can_pos_w - ee_pos_w
    can_to_target = target_pos_w - can_pos_w

    return torch.cat(
        (
            can_pos_w - env.scene.env_origins,
            can_quat_w,
            gripper_to_can,
            can_to_target,
            target_pos,
            ee_pos_w - env.scene.env_origins,
            ee_quat_w,
        ),
        dim=1,
    )


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Both finger joint positions, sign-flipped on the second finger as in the ``stack`` task."""
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env.cfg, "gripper_joint_names"):
        raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Observation gripper_pos only supports a parallel gripper"
    finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def can_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Subtask signal: the can is between the fingers and the gripper is closed on it."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    can: RigidObject = env.scene[can_cfg.name]

    pose_diff = torch.linalg.vector_norm(can.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=1)

    if not hasattr(env.cfg, "gripper_joint_names"):
        raise ValueError("No gripper_joint_names found in environment config")

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Observations only support a parallel gripper"

    open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
    grasped = pose_diff < diff_threshold
    for joint_id in gripper_joint_ids:
        grasped = torch.logical_and(
            grasped,
            torch.abs(robot.data.joint_pos[:, joint_id] - open_val) > env.cfg.gripper_threshold,
        )
    return grasped


def can_lifted(
    env: ManagerBasedRLEnv,
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    min_height: float = 0.02,
) -> torch.Tensor:
    """Subtask signal: the can has cleared the rim of the source bin.

    The bin walls stand 0.08 m above the bin floor, so ``min_height`` is measured relative to
    the resting height and defaults to a conservative 0.02 m above the rim.
    """
    from ..robosuite_layout import BIN_SURFACE_Z, BIN_WALL_HEIGHT_HALF, CAN_REST_Z

    can: RigidObject = env.scene[can_cfg.name]
    can_z = can.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rim_z = BIN_SURFACE_Z + 2 * BIN_WALL_HEIGHT_HALF - (BIN_SURFACE_Z - CAN_REST_Z)
    return can_z > (rim_z + min_height)


def can_placed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
) -> torch.Tensor:
    """Subtask signal: the can is in its target quadrant and the gripper has let go."""
    from .terminations import can_in_target_bin

    return can_in_target_bin(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, can_cfg=can_cfg)


__all__ = [
    "can_grasped",
    "can_lifted",
    "can_orientation_in_world_frame",
    "can_placed",
    "can_position_in_world_frame",
    "ee_frame_pos",
    "ee_frame_quat",
    "gripper_pos",
    "object_obs",
    "target_bin_position",
]
