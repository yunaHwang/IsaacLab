# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the robosuite-style pick-and-place-can task.

These mirror ``manipulation/stack/mdp/franka_stack_events.py``. They are duplicated rather than
imported so this task does not break if the ``stack`` task's events are edited.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Overwrite the articulation's default joint pose for every environment."""
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Perturb the arm's reset pose with Gaussian noise, leaving the fingers untouched."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper joints.
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_can_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("can"),
    pose_range: dict[str, tuple[float, float]] | None = None,
):
    """Drop the can at a uniformly random pose inside the source bin.

    Equivalent to robosuite's ``UniformRandomSampler`` over bin1 with ``rotation_axis="z"`` and
    ``rotation=None``. robosuite rejects samples whose horizontal radius would cross a bin wall;
    here the sampling box is pre-shrunk by that radius in :mod:`..robosuite_layout`, so every
    sample is valid and no rejection loop is needed.
    """
    if env_ids is None:
        return
    if pose_range is None:
        pose_range = {}

    asset: RigidObject = env.scene[asset_cfg.name]
    ranges = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]

    for cur_env in env_ids.tolist():
        sample = [random.uniform(lo, hi) for lo, hi in ranges]
        pose_tensor = torch.tensor([sample], device=env.device)
        positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
        orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
        env_index = torch.tensor([cur_env], device=env.device)
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_index)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_index)


__all__ = ["randomize_can_pose", "randomize_joint_by_gaussian_offset", "set_default_joint_pose"]
