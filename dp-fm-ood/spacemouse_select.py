# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pick a specific SpaceMouse: a named local HID device, or one over spacemouse_bridge.py.

WHY THIS EXISTS
---------------
``isaaclab.devices.Se3SpaceMouse`` finds its device by vendor/product id. Several SpaceMouse
Compacts are plugged into this workstation; they share both ids and report no serial, so plain
``Se3SpaceMouse(cfg)`` opens whichever one hidapi happens to list first. That is often not the one
being held, and it silently reads all zeros -- the robot just never moves, with no error.

The classes below were written for dp-fm-ood/run_policy_fm.py and are lifted here unchanged so
scripts/tools/record_demos.py can share them instead of growing a second copy. run_policy_fm.py
still carries its own definitions; it launches Isaac Sim at import time (it calls parse_args() and
AppLauncher at module level), so it cannot be imported by another tool, and this module is the
seam. If run_policy_fm.py is ever refactored, it should import from here.

IMPORT ORDER: importing this module pulls in isaaclab.devices, so it must be imported AFTER
AppLauncher has started the simulator -- import it inside a function, not at module top level.

USAGE
    from spacemouse_select import make_spacemouse
    teleop = make_spacemouse(Se3SpaceMouseCfg(...), choice="auto")
"""

from __future__ import annotations

from multiprocessing.connection import Client

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices import Se3SpaceMouse


def list_local_spacemice() -> list[bytes]:
    """Local SpaceMouse HID paths (one per physical device), e.g. [b'/dev/hidraw6', b'/dev/hidraw0']."""
    import hid

    names = ("SpaceMouse Compact", "SpaceMouse Wireless", "3Dconnexion Universal Receiver")
    paths = []
    for d in hid.enumerate():
        if d["product_string"] in names and d["path"] not in paths:
            paths.append(d["path"])
    return paths


class PathSe3SpaceMouse(Se3SpaceMouse):
    """Se3SpaceMouse bound to ONE HID path. Isaac Lab's Se3SpaceMouse._find_device opens by vendor/product
    id, and several SpaceMouse Compacts share both (and report no serial), so with more than one plugged
    in it opens whichever hidapi lists first - often not the one being moved, which reads all zeros."""

    def __init__(self, cfg, path):
        self._hid_path = path.encode() if isinstance(path, str) else path
        super().__init__(cfg)  # calls the _find_device below

    def _find_device(self):
        import hid

        match = [d for d in hid.enumerate() if d["path"] == self._hid_path]
        if not match:
            raise OSError(
                f"No SpaceMouse at {self._hid_path.decode()}; connected: "
                f"{[p.decode() for p in list_local_spacemice()]}"
            )
        self._device.open_path(self._hid_path)
        self._device_name = match[0]["product_string"]

    def __str__(self) -> str:
        return super().__str__().replace("\n", f" [{self._hid_path.decode()}]\n", 1)


class NetworkSe3SpaceMouse:
    """Reads a SpaceMouse over spacemouse_bridge.py instead of local HID hardware - for
    when the physical device is on your client machine but this script runs on a remote SSH
    server that has no direct access to it. See spacemouse_bridge.py's module docstring for
    setup (run it on your client machine, then `ssh -R <port>:localhost:<port>` when
    connecting to this server).

    Drop-in replacement for isaaclab.devices.spacemouse.Se3SpaceMouse's public interface
    (advance/reset/__str__). pos_sensitivity/rot_sensitivity are applied here (server-side,
    from Se3SpaceMouseCfg) rather than by the bridge, so retuning sensitivity doesn't require
    restarting the bridge - see spacemouse_bridge.py's docstring.
    """

    def __init__(self, cfg, host, port, authkey):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        self._conn = Client((host, port), authkey=authkey.encode())

    def __str__(self) -> str:
        msg = f"Spacemouse Controller for SE(3): {self.__class__.__name__} (via network bridge)\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tLeft button: toggle gripper command (open/close)\n"
        msg += "\tMove mouse laterally: move arm horizontally in x-y plane\n"
        msg += "\tMove mouse vertically: move arm vertically\n"
        msg += "\tTwist mouse about an axis: rotate arm about a corresponding axis"
        return msg

    def reset(self):
        self._conn.send({"cmd": "reset"})
        self._conn.recv()

    def advance(self) -> torch.Tensor:
        self._conn.send({"cmd": "advance"})
        response = self._conn.recv()
        if not response.get("ok", False):
            raise RuntimeError(f"spacemouse_bridge error: {response.get('error')}")

        delta_pos = np.asarray(response["delta_pos"]) * self.pos_sensitivity
        delta_rot = np.asarray(response["delta_rot"]) * self.rot_sensitivity
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        command = np.concatenate([delta_pos, rot_vec])
        if self.gripper_term:
            gripper_value = -1.0 if response["close_gripper"] else 1.0
            command = np.append(command, gripper_value)

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    # record_demos.py registers reset/callback keys on the teleop interface; the bridge has no
    # key events of its own, so accept and ignore them rather than crashing the caller.
    def add_callback(self, key, func):
        pass


def make_spacemouse(
    cfg,
    choice: str = "auto",
    bridge_host: str = "127.0.0.1",
    bridge_port: int = 6060,
    bridge_authkey: str = "spacemouse-ipc",
):
    """Build the SpaceMouse interface named by ``choice``.

    Args:
        cfg: an ``Se3SpaceMouseCfg`` (pos/rot sensitivity, gripper_term, sim_device).
        choice: ``"auto"`` (the single local device, or the bridge if none is connected),
            a HID path like ``"/dev/hidraw5"``, or ``"bridge"``.
        bridge_host/bridge_port/bridge_authkey: where spacemouse_bridge.py is listening.

    Raises:
        RuntimeError: if ``choice`` is "auto" and several local SpaceMice are connected - which one
            you are holding cannot be guessed, and picking wrong fails silently (all-zero reads).
    """
    local = [p.decode() for p in list_local_spacemice()]

    if choice == "auto":
        if len(local) > 1:
            raise RuntimeError(
                f"{len(local)} local SpaceMice connected ({', '.join(local)}) - pick the one you are "
                "holding with --spacemouse /dev/hidrawN (run `python dp-fm-ood/spacemouse_identify.py` "
                "and move it to see which), or --spacemouse bridge to use spacemouse_bridge.py."
            )
        choice = local[0] if local else "bridge"

    if choice == "bridge":
        print(
            f"[INFO] Using the SpaceMouse network bridge at {bridge_host}:{bridge_port}. Run "
            "spacemouse_bridge.py on your client machine and forward the port with "
            "`ssh -R <port>:localhost:<port>` if you haven't already."
        )
        return NetworkSe3SpaceMouse(cfg, host=bridge_host, port=bridge_port, authkey=bridge_authkey)

    print(f"[INFO] Using local SpaceMouse {choice} (connected: {', '.join(local) or 'none'})")
    return PathSe3SpaceMouse(cfg, choice)
