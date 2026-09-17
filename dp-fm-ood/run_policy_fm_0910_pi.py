# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Pi0.5 SO-101 checkpoint (Cache-SCA/Pi0.5-IsaacLab-Multi-Task-1epochs) in the rebuilt
stack_2_cubes scene, optionally recreating a specific dataset episode.

No gym.register(): the env is built straight from its config class,
    env = ManagerBasedRLEnv(cfg=SO101Stack2CubesEnvCfg())
see so101_stack2cubes_env_cfg.py. There is no --task.

Two terminals, same pattern as run_policy_fm.py + multitask_dit_server.py:

    Terminal 1 - Pi0.5 server (owns the checkpoint):
        conda activate lerobot_0.6.1_multitask_dit
        cd ~/isaac-sim/IsaacLab/dp-fm-ood
        python pi05_server.py --port 5556

    Terminal 2 - Isaac Lab client:
        ssh -X -R 6060:localhost:6060 wisc-rt2-trimanual@128.105.102.149
        conda activate yuna_env
        cd ~/isaac-sim/IsaacLab/dp-fm-ood
        export DISPLAY=:1
        python run_policy_fm_0910_pi.py --episode 674 --enable_cameras [--blend]

--episode N (600-899) recreates dataset episode N: cube colors, light scale, cube positions
(SCRAPE's perception labels) and the matching instruction, from
data/scrape_so101_stack2cubes/episodes.json (see extract_scrape_stack2cubes_episodes.py).
Omit it for a random layout and random colors.

--mode replay executes the episode's RECORDED actions open-loop instead of the policy (no server
needed). It is the check for the scene/unit conversion itself: if the replay does not stack, the
scene or calibration is off, independent of the checkpoint.

--blend reads and prints the SpaceMouse every step (local HID, falling back to
spacemouse_bridge.py over the network) - logging only, no blending, same as run_policy_fm.py.

Units: Pi0.5 consumes/produces LeRobot normalized motor units; the sim uses joint radians. The
exact per-joint linear map (normalized_to_radians / radians_to_normalized) lives in
so101_stack2cubes_env_cfg.py.
"""

import argparse
import json
from multiprocessing.connection import Client
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Run Pi0.5 in the SO-101 stack_2_cubes scene.")
parser.add_argument("--episode", type=int, default=None,
                    help="Dataset episode (600-899) whose scene to recreate. Omit for a random layout.")
parser.add_argument("--episodes_dir", type=str, default=str(HERE / "data" / "scrape_so101_stack2cubes"),
                    help="Output of extract_scrape_stack2cubes_episodes.py (episodes.json / episodes.npz).")
parser.add_argument("--mode", type=str, choices=["policy", "replay"], default="policy",
                    help="'policy' = Pi0.5 via pi05_server.py; 'replay' = the episode's recorded actions (needs --episode).")
parser.add_argument("--task_instruction", type=str, default=None,
                    help="Override the instruction sent to the policy (default: the episode's own, or the sampled colors').")
parser.add_argument("--pi_server_host", type=str, default="127.0.0.1")
parser.add_argument("--pi_server_port", type=int, default=5556)
parser.add_argument("--pi_server_authkey", type=str, default="pi05-ipc", help="Must match pi05_server.py's --authkey.")
parser.add_argument("--blend", action="store_true", help="Read and log the SpaceMouse every step (no blending yet).")
parser.add_argument("--spacemouse_bridge_host", type=str, default="127.0.0.1")
parser.add_argument("--spacemouse_bridge_port", type=int, default=6060)
parser.add_argument("--spacemouse_bridge_authkey", type=str, default="spacemouse-ipc")
parser.add_argument("--horizon", type=int, default=None,
                    help="Max steps per rollout (30 Hz). Default: episode length + 150 with --episode, else 1200.")
parser.add_argument("--settle_steps", type=int, default=10, help="Steps holding the rest pose after placing the scene.")
parser.add_argument("--success_hold_steps", type=int, default=30,
                    help="Cubes must stay stacked this many consecutive steps (1 s) to count as success.")
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_every", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # the policy needs both camera images

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnv

from so101_stack2cubes_env_cfg import (
    CUBE_SIZE,
    REST_JOINT_POS,
    SO101Stack2CubesEnvCfg,
    cubes_stacked,
    normalized_to_radians,
    radians_to_normalized,
    randomize_appearance,
    widen_joint_limits_to_dataset,
)


class NetworkSe3SpaceMouse:
    """SpaceMouse over spacemouse_bridge.py - same class as in run_policy_fm.py (copied, since
    importing run_policy_fm.py would re-run its AppLauncher)."""

    def __init__(self, cfg, host, port, authkey):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        self._conn = Client((host, port), authkey=authkey.encode())

    def __str__(self) -> str:
        return "Spacemouse Controller for SE(3): NetworkSe3SpaceMouse (via network bridge)"

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
        command = np.concatenate([delta_pos, Rotation.from_euler("XYZ", delta_rot).as_rotvec()])
        if self.gripper_term:
            command = np.append(command, -1.0 if response["close_gripper"] else 1.0)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)


def make_teleop(device):
    cfg = Se3SpaceMouseCfg(sim_device=device)
    try:
        return Se3SpaceMouse(cfg)
    except OSError:
        print(f"[INFO] No local SpaceMouse - using spacemouse_bridge.py at "
              f"{args_cli.spacemouse_bridge_host}:{args_cli.spacemouse_bridge_port}")
        return NetworkSe3SpaceMouse(cfg, args_cli.spacemouse_bridge_host, args_cli.spacemouse_bridge_port,
                                    args_cli.spacemouse_bridge_authkey)


def load_episode(episode):
    if episode is None:
        return None, None
    d = Path(args_cli.episodes_dir)
    info = json.loads((d / "episodes.json").read_text())
    if str(episode) not in info:
        raise ValueError(f"episode {episode} not in {d / 'episodes.json'} (stack_2_cubes episodes are 600-899)")
    arrays = np.load(d / "episodes.npz")
    return info[str(episode)], {"state": arrays[f"state_{episode}"], "action": arrays[f"action_{episode}"]}


def setup_scene(env, ep_info, generator):
    """Place cubes / colors / light for this rollout, let physics settle, return (obs, instruction)."""
    if ep_info is not None:
        origin = env.scene.env_origins[0]
        for name, xy in (("cube_top", ep_info["top_cube_xy"]), ("cube_base", ep_info["base_cube_xy"])):
            # Cube yaw is not in the dataset; axis-aligned.
            pose = torch.tensor([[xy[0], xy[1], CUBE_SIZE / 2, 1.0, 0.0, 0.0, 0.0]], device=env.device)
            pose[:, :3] += origin
            env.scene[name].write_root_pose_to_sim(pose)
            env.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        appearance = randomize_appearance(env, top_color=ep_info["top_color"], base_color=ep_info["base_color"],
                                          light_scale=ep_info["light_scale"])
        instruction = ep_info["instruction"]
    else:
        appearance = randomize_appearance(env, generator=generator)  # positions already randomized by the reset event
        instruction = appearance["instruction"]
    print(f"[scene] {appearance}")

    hold = torch.tensor([REST_JOINT_POS], device=env.device)
    obs = None
    for _ in range(args_cli.settle_steps):
        obs, *_ = env.step(hold)
    return obs, args_cli.task_instruction or instruction


def rollout(env, conn, ep_info, ep_arrays, teleop, generator, horizon):
    obs, _ = env.reset()
    if teleop is not None:
        teleop.reset()
    obs, instruction = setup_scene(env, ep_info, generator)
    print(f"[rollout] mode={args_cli.mode} instruction={instruction!r} horizon={horizon}")

    if conn is not None:
        conn.send({"cmd": "reset"})
        reply = conn.recv()
        if not reply.get("ok", False):
            raise RuntimeError(f"pi05_server reset failed: {reply.get('error')}")

    stacked_for = 0
    for step in range(horizon):
        pol = obs["policy"]
        joint_rad = pol["joint_pos"][0]

        if args_cli.mode == "policy":
            request_obs = {
                "observation.state": radians_to_normalized(joint_rad).cpu().numpy(),
                "observation.images.top": pol["top"][0].to(torch.uint8).cpu().numpy(),
                "observation.images.left_wrist": pol["left_wrist"][0].to(torch.uint8).cpu().numpy(),
                "task": instruction,
            }
            conn.send({"cmd": "step", "obs": request_obs})
            response = conn.recv()
            if not response.get("ok", False):
                raise RuntimeError(f"pi05_server error: {response.get('error')}")
            action_norm = torch.tensor(response["action"], dtype=torch.float32, device=env.device)
        else:
            recorded = ep_arrays["action"]
            action_norm = torch.as_tensor(recorded[min(step, len(recorded) - 1)], device=env.device)

        action_rad = normalized_to_radians(action_norm).unsqueeze(0)

        # if teleop is not None:
        #     # print(f"[SpaceMouse] raw 7-DoF action: {teleop.advance().tolist()}")

        if step % args_cli.log_every == 0:
            print(f"[step {step}] state(norm)={radians_to_normalized(joint_rad).cpu().numpy().round(1)} "
                  f"action(norm)={action_norm.cpu().numpy().round(1)}")

        obs, _, terminated, truncated, _ = env.step(action_rad)

        stacked_for = stacked_for + 1 if bool(cubes_stacked(env)[0]) else 0
        if stacked_for >= args_cli.success_hold_steps:
            print(f"[rollout] SUCCESS at step {step}")
            return True
        if bool(terminated[0]) or bool(truncated[0]):
            return False
    return False


def main():
    if args_cli.mode == "replay" and args_cli.episode is None:
        raise ValueError("--mode replay needs --episode")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    generator = torch.Generator().manual_seed(args_cli.seed)

    ep_info, ep_arrays = load_episode(args_cli.episode)
    horizon = args_cli.horizon or (ep_info["length"] + 150 if ep_info else 1200)

    cfg = SO101Stack2CubesEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    # Manual step loop like run_policy_fm.py: no time-out, success checked here (with a hold).
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    env = ManagerBasedRLEnv(cfg=cfg)
    widen_joint_limits_to_dataset(env)  # dataset poses exceed the USD limits - see so101_stack2cubes_env_cfg.py

    teleop = make_teleop(env.device) if args_cli.blend else None
    if teleop is not None:
        print(teleop)

    conn = None
    if args_cli.mode == "policy":
        conn = Client((args_cli.pi_server_host, args_cli.pi_server_port), authkey=args_cli.pi_server_authkey.encode())

    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        results.append(rollout(env, conn, ep_info, ep_arrays, teleop, generator, horizon))
        print(f"[INFO] Trial {trial}: {results[-1]}\n")

    if conn is not None:
        conn.send({"cmd": "close"})
        conn.recv()
        conn.close()

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
