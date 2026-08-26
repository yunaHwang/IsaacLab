# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Copy the first N demos (by literal name, demo_0 .. demo_{N-1}) of an Isaac Lab hdf5
dataset into a new file, leaving the source file untouched.

Demo group names in the hdf5 are not guaranteed to iterate in numeric order (h5py returns
group keys in lexicographic order, so e.g. "demo_10" sorts before "demo_2") - this script
selects demos by their literal name instead of positional order, so "first 20 demos" always
means demo_0 through demo_19 specifically.
"""

import argparse
import os

import h5py

parser = argparse.ArgumentParser(description="Copy the first N demos of an hdf5 dataset into a new file.")
parser.add_argument("input_file", type=str, help="Path to the source hdf5 dataset file.")
parser.add_argument("output_file", type=str, help="Path to write the truncated hdf5 dataset file.")
parser.add_argument("--num_demos", type=int, default=20, help="Number of demos to keep (demo_0 .. demo_{N-1}).")
args_cli = parser.parse_args()


def main():
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")

    demo_names = [f"demo_{i}" for i in range(args_cli.num_demos)]

    with h5py.File(args_cli.input_file, "r") as src:
        src_data = src["data"]

        missing = [name for name in demo_names if name not in src_data]
        if missing:
            raise ValueError(
                f"{args_cli.input_file} is missing {len(missing)} of the requested demos: {missing}"
            )

        out_dir = os.path.dirname(args_cli.output_file)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        with h5py.File(args_cli.output_file, "w") as dst:
            dst_data = dst.create_group("data")
            for key, value in src_data.attrs.items():
                dst_data.attrs[key] = value

            total_samples = 0
            for name in demo_names:
                src_data.copy(name, dst_data)
                total_samples += int(dst_data[name].attrs.get("num_samples", 0))

            dst_data.attrs["total"] = total_samples

    print(f"Wrote {len(demo_names)} demos ({demo_names[0]}..{demo_names[-1]}) to {args_cli.output_file}")


if __name__ == "__main__":
    main()
