"""Sanity check for so101_stack2cubes_env_cfg.py: build the env (no gym.register), pose the scene
like dataset episode 600 (gray on pink), and save top/left_wrist renders next to the dataset frames.

    conda activate yuna_env
    python check_so101_stack2cubes_env.py --headless --enable_cameras \
        --reference_top top_0.png --reference_wrist wrist_0.png --out_dir outputs/so101_env_check

Prints the sim EE (URDF gripper_frame_link) in SCRAPE's robot frame (= env origin) at the rest pose - the
dataset's observation.ee_pos.robot_xyzrpy at frame 0 is (0.169, 0.005, 0.026), which is how the robot
base pose / joint convention is verified.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference_top", type=str, default=None, help="Dataset top frame (png) to place next to the render.")
parser.add_argument("--reference_wrist", type=str, default=None, help="Dataset wrist frame (png) to place next to the render.")
parser.add_argument("--out_dir", type=str, default="outputs/so101_env_check")
parser.add_argument("--settle_steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from so101_stack2cubes_env_cfg import CUBE_SIZE, GRIPPER_FRAME_OFFSET, REST_JOINT_POS, SO101Stack2CubesEnvCfg, randomize_appearance, widen_joint_limits_to_dataset


def main():
    cfg = SO101Stack2CubesEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.terminations.success = None
    env = ManagerBasedRLEnv(cfg=cfg)
    widen_joint_limits_to_dataset(env)
    env.reset()

    # Episode 600 layout: gray (top) grasped at (0.345, 0.085), released above pink at (0.237, -0.023).
    appearance = randomize_appearance(env, top_color="gray", base_color="pink", light_scale=1.122)
    origin = env.scene.env_origins[0]
    for name, xy in (("cube_top", (0.345, 0.085)), ("cube_base", (0.237, -0.023))):
        pose = torch.tensor([[xy[0], xy[1], CUBE_SIZE / 2, 1.0, 0.0, 0.0, 0.0]], device=env.device)
        pose[:, :3] += origin
        env.scene[name].write_root_pose_to_sim(pose)

    action = torch.tensor([REST_JOINT_POS], device=env.device)
    for _ in range(args_cli.settle_steps):
        obs, *_ = env.step(action)

    robot = env.scene["robot"]
    names = robot.data.body_names
    print(f"[check] bodies: {names}")
    i = names.index("gripper")
    ee_w = robot.data.body_pos_w[:, i] + quat_apply(
        robot.data.body_quat_w[:, i], torch.tensor([GRIPPER_FRAME_OFFSET], device=env.device))
    p = (ee_w - env.scene.env_origins)[0].cpu().numpy()
    print(f"[check] EE (gripper_frame_link) in SCRAPE robot frame: {p.round(3)}  (dataset EE at rest: [0.169 0.005 0.026])")
    print(f"[check] joint_pos (rad): {obs['policy']['joint_pos'][0].cpu().numpy().round(3)}  target {np.round(REST_JOINT_POS, 3)}")
    print(f"[check] appearance: {appearance}")

    out = Path(args_cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, ref in (("top", args_cli.reference_top), ("left_wrist", args_cli.reference_wrist)):
        img = obs["policy"][key][0].cpu().numpy().astype(np.uint8)
        panels = [img]
        if ref:
            panels.insert(0, np.asarray(Image.open(ref).convert("RGB").resize((img.shape[1], img.shape[0]))))
        Image.fromarray(np.concatenate(panels, axis=1)).save(out / f"{key}.png")
        print(f"[check] saved {out / f'{key}.png'} ({'dataset | sim' if ref else 'sim'})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
