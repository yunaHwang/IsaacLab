# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert robosuite's can mesh into a USD asset for the Isaac Lab pick-and-place-can task.

robosuite renders the can in MuJoCo straight from MJCF: a 956-triangle mesh
(``objects/meshes/can.msh`` for physics, ``can.obj``/``can.stl`` for geometry) wearing a
``soda.png`` texture through the ``coke`` material. Isaac Lab needs USD, so the mesh has to be
converted once, offline. This wraps :mod:`scripts.tools.convert_mesh` with the right defaults.

The OBJ references ``can.mtl``, which in turn references ``../../textures/soda.png``, and
``omni.kit.asset_converter`` resolves both, so the resulting USD keeps robosuite's texture.

Usage::

    ./isaaclab.sh -p scripts/tools/convert_robosuite_can.py
    ./isaaclab.sh -p scripts/tools/convert_robosuite_can.py --robosuite-root /path/to/robosuite

The output lands at ``<IsaacLab>/assets/robosuite/can.usd``, which is where
``pickplace_can/can_asset.py`` looks for it by default.
"""

import argparse
import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Where robosuite tends to live on this machine; --robosuite-root overrides.
_DEFAULT_ROBOSUITE_ROOTS = [
    os.path.expanduser("~/yuna/robosuite_source/robosuite"),
    os.path.expanduser("~/robosuite/robosuite"),
]


def find_robosuite_root(explicit: str | None) -> str:
    """Locate the ``robosuite`` package directory containing ``models/assets``."""
    candidates = [explicit] if explicit else []
    # Prefer an installed robosuite if there is one.
    try:
        import robosuite  # noqa: PLC0415

        candidates.append(os.path.dirname(robosuite.__file__))
    except ImportError:
        pass
    candidates.extend(_DEFAULT_ROBOSUITE_ROOTS)

    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "models", "assets", "objects", "can.xml")):
            return candidate
    raise FileNotFoundError(
        "Could not locate a robosuite source tree containing models/assets/objects/can.xml.\n"
        f"Tried: {[c for c in candidates if c]}\n"
        "Pass --robosuite-root explicitly."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robosuite-root", type=str, default=None, help="Path to the robosuite package directory.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_ISAACLAB_ROOT, "assets", "robosuite", "can.usd"),
        help="Destination USD path.",
    )
    parser.add_argument(
        "--collision-approximation",
        type=str,
        default="convexHull",
        help=(
            "Collision approximation. Defaults to convexHull because MuJoCo also convex-hulls"
            " mesh geoms for contact, so this matches robosuite's physics most closely. Use"
            " convexDecomposition if you need the concave rim modelled."
        ),
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=0.01464592,
        help="Mass in kg. Default is robosuite's density=100 over the 1.4646e-4 m^3 mesh.",
    )
    args, extra = parser.parse_known_args()

    robosuite_root = find_robosuite_root(args.robosuite_root)
    mesh_path = os.path.join(robosuite_root, "models", "assets", "objects", "meshes", "can.obj")
    texture_path = os.path.join(robosuite_root, "models", "assets", "textures", "soda.png")

    print(f"[INFO] robosuite root : {robosuite_root}")
    print(f"[INFO] input mesh     : {mesh_path}")
    print(f"[INFO] texture        : {texture_path} ({'found' if os.path.isfile(texture_path) else 'MISSING'})")
    print(f"[INFO] output USD     : {args.output}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Delegate to Isaac Lab's own converter so we inherit its AppLauncher handling.
    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "convert_mesh.py"),
        mesh_path,
        args.output,
        "--collision-approximation",
        args.collision_approximation,
        "--mass",
        str(args.mass),
        "--headless",
        *extra,
    ]
    print(f"[INFO] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"\n[INFO] Done. The task will pick this up automatically from {args.output}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
