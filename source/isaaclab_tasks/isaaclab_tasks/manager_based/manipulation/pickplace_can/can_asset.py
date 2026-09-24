# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The can asset, ported from robosuite's MuJoCo mesh.

How robosuite does it
---------------------
robosuite renders the can with MuJoCo, straight out of MJCF. ``models/assets/objects/can.xml``
declares a single mesh geom::

    <mesh file="meshes/can.msh" name="can_mesh"/>
    <texture file="../textures/soda.png" name="tex-can" type="2d"/>
    <material name="coke" reflectance="0.7" texrepeat="5 5" texture="tex-can" texuniform="true"/>
    <geom mesh="can_mesh" type="mesh" density="100" friction="0.95 0.3 0.1" material="coke" condim="4" .../>

So the can is a 956-triangle mesh with a soda-can texture, *not* a primitive. MuJoCo uses the
same mesh for collision (it convex-hulls it internally) and for rendering.

How this port does it
---------------------
Isaac Lab consumes USD, so the same mesh has to be converted once, offline::

    ./isaaclab.sh -p scripts/tools/convert_robosuite_can.py

which wraps ``scripts/tools/convert_mesh.py`` and writes ``assets/robosuite/can.usd``. The OBJ
carries a ``.mtl`` pointing at ``soda.png``, and ``omni.kit.asset_converter`` picks that up, so
the USD keeps robosuite's texture.

If the USD has not been generated yet the task falls back to a cylinder primitive of identical
bounding dimensions (r=0.025, h=0.0806) so the environment is still runnable. The fallback is
geometrically very close -- the mesh is a slightly tapered can -- but it is not textured, so
anything visuomotor should use the converted mesh.
"""

from __future__ import annotations

import os

from isaaclab_tasks import ISAACLAB_TASKS_EXT_DIR

ISAACLAB_ROOT = os.path.abspath(os.path.join(ISAACLAB_TASKS_EXT_DIR, "..", ".."))

ROBOSUITE_CAN_USD = os.environ.get(
    "ROBOSUITE_CAN_USD", os.path.join(ISAACLAB_ROOT, "assets", "robosuite", "can.usd")
)
"""Path to the converted can. Override with the ``ROBOSUITE_CAN_USD`` environment variable."""


def can_usd_available() -> bool:
    """Whether the converted can mesh exists on disk."""
    return os.path.isfile(ROBOSUITE_CAN_USD)


CONVERSION_HINT = (
    f"[pickplace_can] Converted can mesh not found at '{ROBOSUITE_CAN_USD}'.\n"
    "               Falling back to an untextured cylinder primitive of the same size.\n"
    "               To use robosuite's actual mesh, run:\n"
    "                 ./isaaclab.sh -p scripts/tools/convert_robosuite_can.py\n"
    "               or point ROBOSUITE_CAN_USD at an already-converted can.usd."
)
