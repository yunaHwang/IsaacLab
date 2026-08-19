#!/usr/bin/env python3
"""Standalone bridge that reads a physically-connected 3Dconnexion SpaceMouse via HID and
serves its readings over multiprocessing.connection, so a SpaceMouse plugged into your LOCAL
client machine can be read by run_policy_fm.py running on a REMOTE workstation over SSH
(where the device itself isn't attached) - mirrors multitask_dit_server.py's client/server
split, just with the roles of "who owns the hardware" reversed.

Only depends on the `hid` package (pip install hidapi) plus the stdlib - no IsaacLab/Isaac
Sim install is needed on the machine the SpaceMouse is actually plugged into. The HID
parsing below is a standalone copy of isaaclab.devices.spacemouse's
Se3SpaceMouse._find_device/_run_device + spacemouse/utils.convert_buffer (verified against
that source), so behavior matches the local-device path exactly.

Setup:
    1. On your LOCAL machine (SpaceMouse physically connected here):
        pip install hidapi
        python spacemouse_bridge.py --port 6060

    2. When SSH'ing into the remote workstation, add a *remote* port forward so the remote
       side's localhost:6060 tunnels back to this script running on your machine:
        ssh -R 6060:localhost:6060 user@remote-workstation

    3. On the remote workstation, run_policy_fm.py --blend automatically falls back to this
       bridge (via NetworkSe3SpaceMouse) whenever no SpaceMouse is found on local HID.

Protocol (mirrors multitask_dit_server.py's style):
    Requests:  {"cmd": "advance"} | {"cmd": "reset"} | {"cmd": "close"}
    Responses: {"ok": True, "delta_pos": [x,y,z], "delta_rot": [rx,ry,rz], "close_gripper": bool}
               | {"ok": False, "error": "..."}

Reports RAW normalized [-1, 1] readings (via convert_buffer, unscaled) - NOT
pos_sensitivity/rot_sensitivity-scaled. That scaling is applied server-side by
NetworkSe3SpaceMouse (run_policy_fm.py) from its Se3SpaceMouseCfg, so retuning sensitivity
doesn't require restarting this bridge.
"""

import argparse
import threading
import time
from multiprocessing.connection import Listener

import hid

DEFAULT_AUTHKEY = "spacemouse-ipc"

KNOWN_PRODUCT_STRINGS = ("SpaceMouse Compact", "SpaceMouse Wireless", "3Dconnexion Universal Receiver")


def _to_int16(y1, y2):
    """Convert two 8-bit bytes to a signed 16-bit integer (verbatim from isaaclab's
    spacemouse/utils.py)."""
    x = y1 | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def _scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """Normalize a raw HID reading to [-1, 1] (verbatim from isaaclab's spacemouse/utils.py)."""
    x = x / axis_scale
    return min(max(x, min_v), max_v)


def convert_buffer(b1, b2):
    return _scale_to_control(_to_int16(b1, b2))


class SpaceMouseReader:
    """Background HID reader - standalone port of isaaclab's Se3SpaceMouse._find_device/
    _run_device, but unscaled (raw convert_buffer output, not multiplied by
    pos_sensitivity/rot_sensitivity - see this file's module docstring for why)."""

    def __init__(self):
        self._device = hid.device()
        self.device_name = None
        self._find_device()

        self.delta_pos = [0.0, 0.0, 0.0]
        self.delta_rot = [0.0, 0.0, 0.0]
        self.close_gripper = False
        self._read_rotation = False
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _find_device(self):
        found = False
        for _ in range(5):
            for device in hid.enumerate():
                if device["product_string"] in KNOWN_PRODUCT_STRINGS:
                    found = True
                    self._device.close()
                    self._device.open(device["vendor_id"], device["product_id"])
                    self.device_name = device["product_string"]
            if not found:
                time.sleep(1.0)
            else:
                break
        if not found:
            raise OSError("No SpaceMouse found on this (client) machine. Is it connected?")

    def _run(self):
        while True:
            if self.device_name == "3Dconnexion Universal Receiver":
                data = self._device.read(7 + 6)
            else:
                data = self._device.read(7)
            if data is None:
                continue

            with self._lock:
                if self.device_name == "3Dconnexion Universal Receiver":
                    if data[0] == 1:
                        self.delta_pos[1] = convert_buffer(data[1], data[2])
                        self.delta_pos[0] = convert_buffer(data[3], data[4])
                        self.delta_pos[2] = convert_buffer(data[5], data[6]) * -1.0
                        self.delta_rot[1] = convert_buffer(data[1 + 6], data[2 + 6])
                        self.delta_rot[0] = convert_buffer(data[3 + 6], data[4 + 6])
                        self.delta_rot[2] = convert_buffer(data[5 + 6], data[6 + 6]) * -1.0
                else:
                    if data[0] == 1:
                        self.delta_pos[1] = convert_buffer(data[1], data[2])
                        self.delta_pos[0] = convert_buffer(data[3], data[4])
                        self.delta_pos[2] = convert_buffer(data[5], data[6]) * -1.0
                    elif data[0] == 2 and not self._read_rotation:
                        self.delta_rot[1] = convert_buffer(data[1], data[2])
                        self.delta_rot[0] = convert_buffer(data[3], data[4])
                        self.delta_rot[2] = convert_buffer(data[5], data[6]) * -1.0

                if data[0] == 3:
                    if data[1] == 1:
                        self.close_gripper = not self.close_gripper
                    if data[1] == 2:
                        self._reset_locked()
                    if data[1] == 3:
                        self._read_rotation = not self._read_rotation

    def _reset_locked(self):
        self.close_gripper = False
        self.delta_pos = [0.0, 0.0, 0.0]
        self.delta_rot = [0.0, 0.0, 0.0]

    def reset(self):
        with self._lock:
            self._reset_locked()

    def snapshot(self):
        with self._lock:
            return {
                "ok": True,
                "delta_pos": list(self.delta_pos),
                "delta_rot": list(self.delta_rot),
                "close_gripper": self.close_gripper,
            }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6060)
    parser.add_argument(
        "--authkey", type=str, default=DEFAULT_AUTHKEY,
        help="Shared secret for the multiprocessing.connection handshake - must match "
        "run_policy_fm.py's --spacemouse_bridge_authkey.",
    )
    args = parser.parse_args()

    reader = SpaceMouseReader()
    print(f"[spacemouse_bridge] found device: {reader.device_name}")

    listener = Listener((args.host, args.port), authkey=args.authkey.encode())
    print(
        f"[spacemouse_bridge] listening on {args.host}:{args.port} - forward this port to "
        f"your remote workstation with: ssh -R {args.port}:localhost:{args.port} user@remote-workstation"
    )

    try:
        while True:
            conn = listener.accept()
            print(f"[spacemouse_bridge] client connected from {listener.last_accepted}")
            try:
                while True:
                    try:
                        request = conn.recv()
                    except EOFError:
                        break

                    cmd = request.get("cmd")
                    if cmd == "advance":
                        conn.send(reader.snapshot())
                    elif cmd == "reset":
                        reader.reset()
                        conn.send({"ok": True})
                    elif cmd == "close":
                        conn.send({"ok": True})
                        break
                    else:
                        conn.send({"ok": False, "error": f"unknown cmd {cmd!r}"})
            except Exception as e:
                print(f"[spacemouse_bridge] error: {e}")
                try:
                    conn.send({"ok": False, "error": str(e)})
                except OSError:
                    pass
            finally:
                conn.close()
                print("[spacemouse_bridge] client disconnected")
    except KeyboardInterrupt:
        print("\n[spacemouse_bridge] shutting down")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
