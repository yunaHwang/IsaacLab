"""Which /dev/hidrawN is the SpaceMouse in your hand? Opens every local SpaceMouse and prints the ones
that report motion or button presses. Move the one you want to use, then pass its path to
run_policy_fm.py --spacemouse /dev/hidrawN.

    python spacemouse_identify.py            # Ctrl+C to stop
    python spacemouse_identify.py --seconds 15

Needed because several identical SpaceMouse Compacts (same vendor/product id, no serial) can be plugged
in, and Isaac Lab's Se3SpaceMouse opens an arbitrary one of them. Only needs the `hid` package.
"""
import argparse
import time

import hid

NAMES = ("SpaceMouse Compact", "SpaceMouse Wireless", "3Dconnexion Universal Receiver")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=None, help="Stop after this long (default: until Ctrl+C).")
    args = parser.parse_args()

    devices = {}
    for d in hid.enumerate():
        if d["product_string"] in NAMES and d["path"] not in devices:
            dev = hid.device()
            try:
                dev.open_path(d["path"])
            except OSError as e:
                print(f"{d['path'].decode()}: cannot open ({e}) - check permissions on the hidraw node")
                continue
            dev.set_nonblocking(True)
            devices[d["path"]] = (dev, d["product_string"])
    if not devices:
        raise SystemExit("no local SpaceMouse found")
    print("connected:", ", ".join(f"{p.decode()} ({name})" for p, (_, name) in devices.items()))
    print("move or press a button on the SpaceMouse you want to use...")

    last = {}
    start = time.time()
    try:
        while args.seconds is None or time.time() - start < args.seconds:
            for path, (dev, name) in devices.items():
                data = dev.read(13)
                if data and (data[0] == 3 or any(data[1:])):
                    kind = {1: "translation", 2: "rotation", 3: "button"}.get(data[0], f"report {data[0]}")
                    if time.time() - last.get(path, 0) > 0.3:  # rate-limit per device
                        print(f"  {path.decode()}  {kind}")
                        last[path] = time.time()
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        for dev, _ in devices.values():
            dev.close()


if __name__ == "__main__":
    main()
