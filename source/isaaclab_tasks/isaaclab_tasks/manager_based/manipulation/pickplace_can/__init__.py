# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab port of robosuite's ``PickPlaceCan`` -- the task behind robomimic's ``can`` datasets.

NOTE on the package name: it is ``pickplace_can``, not ``pick_place_can``, on purpose.
``isaaclab_tasks/__init__.py`` blacklists the substring ``"pick_place"`` from auto-registration,
and the blacklist is a substring match, so a package named ``pick_place_can`` would silently
never register its gym environments.
"""
