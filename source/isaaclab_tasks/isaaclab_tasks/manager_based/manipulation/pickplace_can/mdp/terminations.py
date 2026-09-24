# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the robosuite-style pick-and-place-can task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from ..robosuite_layout import (
    CAN_TARGET_X_RANGE,
    CAN_TARGET_Y_RANGE,
    CAN_TARGET_Z_RANGE,
    RELEASE_DISTANCE,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def can_in_target_bin(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    can_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    x_range: tuple[float, float] = CAN_TARGET_X_RANGE,
    y_range: tuple[float, float] = CAN_TARGET_Y_RANGE,
    z_range: tuple[float, float] = CAN_TARGET_Z_RANGE,
    release_distance: float = RELEASE_DISTANCE,
) -> torch.Tensor:
    """Success condition, transcribed from ``PickPlace._check_success`` for the can.

    robosuite requires two things:

    1. ``not not_in_bin(can_pos, 3)`` -- the can's origin lies inside the axis-aligned box of its
       assigned quadrant of bin2.
    2. ``r_reach < 0.6`` where ``r_reach = 1 - tanh(10 * dist)`` -- the gripper is at least
       ``atanh(0.4)/10 ~= 0.0424`` m away from the can, i.e. it has actually let go rather than
       still dangling it over the bin.

    Both are reproduced here. Note ``single_object_mode=2`` (which ``PickPlaceCan`` uses) makes
    success a single-object check, so there is no all-objects variant to mirror.
    """
    can: RigidObject = env.scene[can_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # robosuite's bin bounds are in its world frame; ours are relative to the env origin.
    can_pos = can.data.root_pos_w - env.scene.env_origins
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    in_bin = (can_pos[:, 0] > x_range[0]) & (can_pos[:, 0] < x_range[1])
    in_bin &= (can_pos[:, 1] > y_range[0]) & (can_pos[:, 1] < y_range[1])
    in_bin &= (can_pos[:, 2] > z_range[0]) & (can_pos[:, 2] < z_range[1])

    released = torch.linalg.vector_norm(can.data.root_pos_w - ee_pos_w, dim=1) > release_distance

    return torch.logical_and(in_bin, released)


__all__ = ["can_in_target_bin"]
