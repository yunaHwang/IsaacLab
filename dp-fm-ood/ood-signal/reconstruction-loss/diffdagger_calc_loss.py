# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

"""
call compute_diffusion_loss() inside diffusion_policy.py
and use that output to define k.

input to compute_diffusion_loss(): user actions
"""
import argparse
import h5py
import pandas as pd

from isaaclab.app import AppLauncher

from robomimic.algo.diffusion_policy import compute_diffusion_loss

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dataset", type=str, default=None) # use when calculating loss range for rollouts and not user actions
parser.add_argument("--save_to_file", action="store_true")
parser.add_argument("--save_file_name", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

def main():
    if args_cli.dataset:

        all_demo_losses = dict()

        with h5py.File(args_cli.dataset, "r") as f:
            data = f["data"]
            for demo in data.keys():
                actions = demo["actions"]
                obs = demo["obs"]
                states = demo["states"]

                loss = compute_diffusion_loss(obs, states)
                all_demo_losses[demo] = loss 

        if args_cli.save_to_file:
            all_demo_losses_df = pd.DataFrame(all_demo_losses)
            all_demo_losses_df.to_csv(args_cli.save_file_name + ".csv", index = False)
            

    else:
        try:
            all_losses = []

            while True:
                loss = compute_diffusion_loss(isaac_obs, isaac_action)
                all_losses.append(loss)

        except KeyboardInterrupt:
            print("Stopping streaming...")

            if args_cli.save_to_file:
                df = pd(all_losses, column="loss")
                df.to_csv(args_cli.save_file_name + ".csv", index = False)

# TODO - git clone robomimic because i am changing original codebase (make sure it's 0.5.0)
if __name__ == "__main__":
    main()

