# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry of the robosuite ``PickPlaceCan`` task, transcribed for Isaac Lab.

This module is the single source of truth for the scene layout. Every number here was read
directly out of the robosuite 1.5.2 sources so the Isaac Lab scene is metrically identical to
the one the robomimic ``can`` demonstrations were collected in:

* ``robosuite/models/assets/arenas/bins_arena.xml``  -- bin geoms
* ``robosuite/models/arenas/bins_arena.py``          -- ``table_full_size``
* ``robosuite/environments/manipulation/pick_place.py`` -- bin poses, sampling, ``not_in_bin``
* ``robosuite/models/robots/manipulators/panda_robot.py`` -- ``base_xpos_offset["bins"]``
* ``robosuite/models/assets/bases/rethink_mount.xml``  -- pedestal height

Frame convention
----------------
robosuite expresses everything in a world frame whose origin is on the floor. Isaac Lab
manager-based manipulation tasks instead put the robot base at the environment origin (see the
``stack`` task), and the mimic/IK machinery assumes that. So every robosuite world coordinate
below is re-expressed **relative to the Franka base link**:

    env_local = robosuite_world - ROBOSUITE_PANDA_BASE_W

with ``ROBOSUITE_PANDA_BASE_W = (-0.5, -0.1, 0.912)``:

* ``(-0.5, -0.1)`` is ``PandaRobot.base_xpos_offset["bins"]``.
* ``0.912`` is the height the Panda root body ends up at once the Rethink pedestal is attached:
  ``set_base_xpos`` shifts the root by ``-bottom_offset``, and for the Rethink mount
  ``bottom_offset - top_offset = -0.922 - (-0.01) = -0.912``.

Because the transform is a pure translation, all *relative* geometry (robot-to-bin, bin-to-bin,
object-to-target) is preserved exactly. Note that the bin surface sits 0.092 m **below** the
robot base plane here, unlike the Isaac Lab ``stack`` task where the table top is level with the
base -- that offset is real and is what robosuite has.
"""

from __future__ import annotations

##
# Frame transform from the robosuite world frame to the Franka-base-relative frame.
##

ROBOSUITE_PANDA_BASE_W = (-0.5, -0.1, 0.912)
"""World pose of the Panda root body in robosuite's ``PickPlace`` scene."""


def from_robosuite_world(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Re-express a robosuite world position in the Franka-base-relative (env-local) frame."""
    return tuple(p - o for p, o in zip(pos, ROBOSUITE_PANDA_BASE_W))  # type: ignore[return-value]


##
# Bins. MuJoCo ``size`` is a half-extent, Isaac Lab ``CuboidCfg.size`` is a full extent.
##

BIN1_POS = from_robosuite_world((0.1, -0.25, 0.80))
"""Centre of the source bin (light wood) that the can starts in. -> (0.6, -0.15, -0.112)"""

BIN2_POS = from_robosuite_world((0.1, 0.28, 0.80))
"""Centre of the target bin (dark wood), divided into four quadrants. -> (0.6, 0.38, -0.112)"""

GROUND_Z = -ROBOSUITE_PANDA_BASE_W[2]
"""Height of the floor plane. robosuite's floor is at world z=0, i.e. -0.912 here."""

BIN_FLOOR_HALF = (0.20, 0.25, 0.02)
"""Half-extents of a bin's base plate."""

BIN_SURFACE_Z = BIN1_POS[2] + BIN_FLOOR_HALF[2]
"""Top surface of both bins, i.e. the height an object rests on. -> -0.092"""

BIN_WALL_HEIGHT_HALF = 0.05
BIN_WALL_THICKNESS_HALF = 0.01

# (pos, half_extent) for each collision geom of bin1, relative to the bin centre.
BIN1_WALLS = (
    ((0.0, 0.25, 0.05), (0.21, 0.01, 0.05)),
    ((0.0, -0.25, 0.05), (0.21, 0.01, 0.05)),
    ((0.20, 0.0, 0.05), (0.01, 0.25, 0.05)),
    ((-0.20, 0.0, 0.05), (0.01, 0.25, 0.05)),
)

# bin2 additionally carries the two dividers at x=0 and y=0 that split it into four quadrants.
BIN2_WALLS = (
    ((0.0, 0.25, 0.05), (0.21, 0.01, 0.05)),
    ((0.0, -0.25, 0.05), (0.21, 0.01, 0.05)),
    ((0.20, 0.0, 0.05), (0.01, 0.25, 0.05)),
    ((-0.20, 0.0, 0.05), (0.01, 0.25, 0.05)),
    ((0.0, 0.0, 0.05), (0.20, 0.01, 0.05)),  # divider splitting y
    ((0.0, 0.0, 0.05), (0.01, 0.25, 0.05)),  # divider splitting x
)

BIN_LEG_OFFSETS = ((0.15, 0.20), (-0.15, 0.20), (-0.15, -0.20), (0.15, -0.20))
BIN_LEG_RADIUS = 0.01
BIN_LEG_HALF_HEIGHT = 0.40
"""Visual-only pedestal legs; they run from the bin underside down to the floor."""


##
# The can. Measured from robosuite/models/assets/objects/meshes/can.obj.
##

CAN_BBOX_MIN = (-0.024984, -0.025091, -0.040297)
CAN_BBOX_MAX = (0.025016, 0.024909, 0.039703)
CAN_HEIGHT = CAN_BBOX_MAX[2] - CAN_BBOX_MIN[2]  # 0.0806
CAN_RADIUS = 0.025
CAN_VOLUME = 1.464592e-04
"""Closed-mesh volume in m^3, from the divergence theorem over the .obj triangles."""

CAN_MASS = CAN_VOLUME * 100.0
"""14.65 g -- robosuite gives the can geom ``density="100"`` and no explicit mass."""

CAN_REST_Z = BIN_SURFACE_Z - CAN_BBOX_MIN[2]
"""Height of the can *origin* when it rests upright on a bin floor. -> -0.0517

The mesh origin is not at the geometric centre: its base is 0.0403 m below the origin.
"""

# robosuite's site-defined horizontal radius: ``horizontal_radius_site`` sits at (0.025, 0.025, 0).
CAN_HORIZONTAL_RADIUS = (CAN_RADIUS**2 + CAN_RADIUS**2) ** 0.5  # 0.035355


##
# Initial-state sampling, matching ``PickPlace._get_placement_initializer``.
##

# ``bin_size`` in pick_place.py is BinsArena.table_full_size, NOT the geom extents. This is a
# quirk of robosuite (0.39/0.49 vs the 0.40/0.50 the geoms actually span) but it is what the
# demonstrations were generated against, so it is reproduced verbatim.
TABLE_FULL_SIZE = (0.39, 0.49, 0.82)

_SAMPLE_X_HALF = TABLE_FULL_SIZE[0] / 2 - 0.05  # 0.145
_SAMPLE_Y_HALF = TABLE_FULL_SIZE[1] / 2 - 0.05  # 0.195

# ``ensure_object_boundary_in_range=True`` shrinks the sampling box by the object's horizontal
# radius so the can never spawns clipping a bin wall.
CAN_SPAWN_X_RANGE = (
    BIN1_POS[0] - (_SAMPLE_X_HALF - CAN_HORIZONTAL_RADIUS),
    BIN1_POS[0] + (_SAMPLE_X_HALF - CAN_HORIZONTAL_RADIUS),
)
CAN_SPAWN_Y_RANGE = (
    BIN1_POS[1] - (_SAMPLE_Y_HALF - CAN_HORIZONTAL_RADIUS),
    BIN1_POS[1] + (_SAMPLE_Y_HALF - CAN_HORIZONTAL_RADIUS),
)
CAN_SPAWN_YAW_RANGE = (-3.141592653589793, 3.141592653589793)
"""robosuite samples ``z_rotation=None`` -> uniform over a full turn about z."""


##
# Success region, matching ``PickPlace.not_in_bin`` for the can.
##

CAN_BIN_ID = 3
"""``object_to_id`` in pick_place.py is {milk: 0, bread: 1, cereal: 2, can: 3}."""


def target_bin_bounds(bin_id: int = CAN_BIN_ID) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the ``(x_lo, x_hi), (y_lo, y_hi)`` quadrant of bin2 assigned to ``bin_id``.

    This is a direct transcription of ``PickPlace.not_in_bin``.
    """
    x_lo, y_lo = BIN2_POS[0], BIN2_POS[1]
    if bin_id == 0 or bin_id == 2:
        x_lo -= TABLE_FULL_SIZE[0] / 2
    if bin_id < 2:
        y_lo -= TABLE_FULL_SIZE[1] / 2
    return (x_lo, x_lo + TABLE_FULL_SIZE[0] / 2), (y_lo, y_lo + TABLE_FULL_SIZE[1] / 2)


CAN_TARGET_X_RANGE, CAN_TARGET_Y_RANGE = target_bin_bounds(CAN_BIN_ID)
"""-> x in (0.600, 0.795), y in (0.380, 0.625)"""

CAN_TARGET_Z_RANGE = (BIN2_POS[2], BIN2_POS[2] + 0.1)
"""-> (-0.112, -0.012). robosuite checks ``bin2_pos[2] < z < bin2_pos[2] + 0.1``."""

CAN_TARGET_CENTER = (
    (CAN_TARGET_X_RANGE[0] + CAN_TARGET_X_RANGE[1]) / 2,
    (CAN_TARGET_Y_RANGE[0] + CAN_TARGET_Y_RANGE[1]) / 2,
    BIN_SURFACE_Z,
)
"""Centre of the can's target quadrant, used for the relative observation term."""

RELEASE_DISTANCE = 0.042365
"""Minimum gripper-to-can distance required for success.

robosuite's ``_check_success`` additionally demands ``r_reach < 0.6`` where
``r_reach = 1 - tanh(10 * dist)``. Solving gives ``dist > atanh(0.4)/10``. Without this the task
would count as solved while the gripper is still holding the can over the bin.
"""
