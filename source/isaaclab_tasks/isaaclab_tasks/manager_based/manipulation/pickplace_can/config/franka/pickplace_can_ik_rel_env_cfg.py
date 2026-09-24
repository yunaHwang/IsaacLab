# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda pick-and-place-can under relative end-effector (IK) control.

This is the closest analogue to how the robomimic ``can`` demonstrations were actually
collected: robosuite drives ``PickPlaceCan`` with an OSC_POSE controller taking delta-pose
actions, and the Isaac Lab equivalent is a differential-IK action with ``use_relative_mode``.
The action space is therefore 6 delta-pose terms plus 1 binary gripper, matching robosuite's
7-dim ``OSC_POSE`` action.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import pickplace_can_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaPickPlaceCanEnvCfg(pickplace_can_joint_pos_env_cfg.FrankaPickPlaceCanEnvCfg):
    """Faithful robosuite layout, relative IK control."""

    def __post_init__(self):
        super().__post_init__()

        # Stiffer PD gains track IK targets much more closely.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class FrankaPickPlaceCanNearBinEnvCfg(
    pickplace_can_joint_pos_env_cfg.FrankaPickPlaceCanNearBinEnvCfg, FrankaPickPlaceCanEnvCfg
):
    """Relative IK control with the reachability-friendly target quadrant.

    The MRO puts the ``NearBin`` ``__post_init__`` first, and it calls ``super()``, so the IK
    action setup still runs before the success/observation retargeting is applied.
    """

    pass
