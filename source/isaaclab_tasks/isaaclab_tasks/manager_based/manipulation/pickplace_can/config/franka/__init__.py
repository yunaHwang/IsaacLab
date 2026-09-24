# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for the Franka pick-and-place-can environments."""

import gymnasium as gym

from . import agents

##
# Joint Position Control
##

gym.register(
    id="Isaac-PickPlace-Can-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_can_joint_pos_env_cfg:FrankaPickPlaceCanEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Can-NearBin-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_can_joint_pos_env_cfg:FrankaPickPlaceCanNearBinEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
#
# This is the controller that matches how the robomimic `can` demos were collected
# (robosuite OSC_POSE delta actions).
##

gym.register(
    id="Isaac-PickPlace-Can-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_can_ik_rel_env_cfg:FrankaPickPlaceCanEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Can-NearBin-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_can_ik_rel_env_cfg:FrankaPickPlaceCanNearBinEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-PickPlace-Can-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_can_ik_abs_env_cfg:FrankaPickPlaceCanEnvCfg",
    },
    disable_env_checker=True,
)
