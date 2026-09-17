"""Why does --mode replay not grasp? Replays a dataset episode's recorded actions and compares the sim
with what SCRAPE recorded, step by step.

    conda activate yuna_env
    python diagnose_so101_replay.py --episode 674 --headless --enable_cameras

Writes outputs/so101_replay_diag/ep<N>/:
    trace.csv       per step: commanded vs actual joints (rad), recorded joint state (rad),
                    sim gripper/jaw position in the robot base frame vs dataset EE position, cube positions
    frame<k>.png    [dataset top | sim top] over [dataset wrist | sim wrist] at key frames
and prints: joint tracking error (tests weak actuators), EE error (tests frame/zero offsets), cube lift.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=674)
parser.add_argument("--snap_frames", type=int, nargs="*", default=[0, 100, 190, 300, 344, 400, 550, 700, 900])
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--top_cube_yaw_deg", type=float, default=0.0,
                    help="Top cube yaw about z (deg). The dataset does not record cube yaw; default axis-aligned. "
                         "Non-zero writes to outputs/so101_replay_diag/ep<N>_yaw<deg>/ instead.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import json
import subprocess

import numpy as np
import pandas as pd
import torch
from PIL import Image

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from so101_stack2cubes_env_cfg import (
    CUBE_SIZE, GRIPPER_FRAME_OFFSET, JOINT_NAMES, REST_JOINT_POS, SO101Stack2CubesEnvCfg, normalized_to_radians,
    randomize_appearance, widen_joint_limits_to_dataset,
)

EPD = HERE / "data" / "scrape_so101_stack2cubes"
OUT = HERE / "outputs" / "so101_replay_diag" / (
    f"ep{args_cli.episode}" + (f"_yaw{args_cli.top_cube_yaw_deg:g}" if args_cli.top_cube_yaw_deg else ""))
OUT.mkdir(parents=True, exist_ok=True)
ep = args_cli.episode
info = json.loads((EPD / "episodes.json").read_text())[str(ep)]
arr = np.load(EPD / "episodes.npz")
rec_state, rec_action = arr[f"state_{ep}"], arr[f"action_{ep}"]
meta = pd.read_parquet(EPD / "raw" / "meta__episodes__chunk-000__file-000.parquet").set_index("episode_index")
fi = int(meta.loc[ep, "data/file_index"])
raw = pd.read_parquet(EPD / "raw" / f"data__chunk-000__file-{fi:03d}.parquet",
                      columns=["episode_index", "frame_index", "observation.ee_pos.robot_xyzrpy", "observation.state.radian_urdf0"])
raw = raw[raw.episode_index == ep].sort_values("frame_index")
rec_ee = np.stack(raw["observation.ee_pos.robot_xyzrpy"])[:, :3]
rec_rad = np.stack(raw["observation.state.radian_urdf0"])


def dataset_frame(cam, k):
    vfi = int(meta.loc[ep, f"videos/observation.images.{cam}/file_index"])
    t = float(meta.loc[ep, f"videos/observation.images.{cam}/from_timestamp"]) + k / 30.0
    path = EPD / "raw" / f"videos__observation.images.{cam}__chunk-000__file-{vfi:03d}.mp4"
    if not path.exists():
        return np.zeros((480, 640, 3), np.uint8)
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(480, 640, 3) if len(out) == 480 * 640 * 3 else np.zeros((480, 640, 3), np.uint8)


cfg = SO101Stack2CubesEnvCfg()
cfg.sim.device = args_cli.device
cfg.terminations.time_out = None
cfg.terminations.success = None
env = ManagerBasedRLEnv(cfg=cfg)
widen_joint_limits_to_dataset(env)
obs, _ = env.reset()

robot = env.scene["robot"]
print(f"[diag] robot bodies: {robot.body_names}")
print(f"[diag] robot joints: {robot.joint_names}")
print(f"[diag] joint limits (rad): {dict(zip(robot.joint_names, robot.data.joint_pos_limits[0].cpu().numpy().round(3).tolist()))}")
print(f"[diag] actuator stiffness/damping: {getattr(robot.data, 'joint_stiffness', None)} / {getattr(robot.data, 'joint_damping', None)}")
print(f"[diag] base pose world: pos={robot.data.root_pos_w[0].cpu().numpy().round(3)} quat(wxyz)={robot.data.root_quat_w[0].cpu().numpy().round(3)}")

origin = env.scene.env_origins[0]
for name, xy in (("cube_top", info["top_cube_xy"]), ("cube_base", info["base_cube_xy"])):
    half = np.radians(args_cli.top_cube_yaw_deg if name == "cube_top" else 0.0) / 2
    pose = torch.tensor([[xy[0], xy[1], CUBE_SIZE / 2, np.cos(half), 0.0, 0.0, np.sin(half)]], device=env.device)
    pose[:, :3] += origin
    env.scene[name].write_root_pose_to_sim(pose)
    env.scene[name].write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
randomize_appearance(env, top_color=info["top_color"], base_color=info["base_color"], light_scale=info["light_scale"])
hold = torch.tensor([REST_JOINT_POS], device=env.device)
for _ in range(10):
    obs, *_ = env.step(hold)

names = robot.body_names
bi = names.index("base")
body_ids = {b: names.index(b) for b in ("gripper", "jaw") if b in names}
# SCRAPE's EE point (URDF gripper_frame_link) in the gripper body frame - see so101_stack2cubes_env_cfg.py.
EE_OFFSET = torch.tensor([GRIPPER_FRAME_OFFSET], device=env.device)


def in_base(pos_w):
    """Position in SCRAPE's robot frame = env world frame relative to the env origin. The robot root is
    spawned yawed and offset (ROBOT_ROOT_POS) so that this IS the URDF base_link frame - not the root."""
    return (pos_w[0] - origin).cpu().numpy()


rows = []
steps = min(len(rec_action), args_cli.max_steps or len(rec_action))
for k in range(steps):
    pol = obs["policy"]
    q = pol["joint_pos"][0].cpu().numpy()
    row = {"frame": k}
    for j, n in enumerate(JOINT_NAMES):
        row[f"q_sim_{n}"] = q[j]
        row[f"q_rec_{n}"] = rec_rad[k, j]
        row[f"q_cmd_{n}"] = float(normalized_to_radians(torch.as_tensor(rec_action[max(k - 1, 0)]))[j])
    for b, i in body_ids.items():
        p = in_base(robot.data.body_pos_w[:, i])
        qw = robot.data.body_quat_w[0, i].cpu().numpy()
        row.update({f"{b}_x": p[0], f"{b}_y": p[1], f"{b}_z": p[2], f"{b}_qw": qw[0], f"{b}_qx": qw[1], f"{b}_qy": qw[2], f"{b}_qz": qw[3]})
    if "gripper" in body_ids:
        i = body_ids["gripper"]
        p = in_base(robot.data.body_pos_w[:, i] + quat_apply(robot.data.body_quat_w[:, i], EE_OFFSET))
        row.update({"ee_sim_x": p[0], "ee_sim_y": p[1], "ee_sim_z": p[2]})
    row.update({"ee_rec_x": rec_ee[k, 0], "ee_rec_y": rec_ee[k, 1], "ee_rec_z": rec_ee[k, 2]})
    for c in ("cube_top", "cube_base"):
        p = in_base(env.scene[c].data.root_pos_w)
        row.update({f"{c}_x": p[0], f"{c}_y": p[1], f"{c}_z": p[2]})
    rows.append(row)

    if k in args_cli.snap_frames:
        top = np.concatenate([dataset_frame("top", k), pol["top"][0].to(torch.uint8).cpu().numpy()], 1)
        wrist = np.concatenate([dataset_frame("left_wrist", k), pol["left_wrist"][0].to(torch.uint8).cpu().numpy()], 1)
        Image.fromarray(np.concatenate([top, wrist], 0)).resize((960, 720)).save(OUT / f"frame{k:04d}.png")

    action = normalized_to_radians(torch.as_tensor(rec_action[k], device=env.device)).unsqueeze(0)
    obs, *_ = env.step(action)

df = pd.DataFrame(rows)
df.to_csv(OUT / "trace.csv", index=False, float_format="%.5f")

print("\n================ DIAG SUMMARY ================")
print(f"episode {ep} {info['instruction']!r}, {len(df)} steps; top cube at {info['top_cube_xy']}, base cube at {info['base_cube_xy']}")
print("joint tracking, sim vs recorded state (deg): mean abs err | max abs err")
for n in JOINT_NAMES:
    e = np.degrees(np.abs(df[f"q_sim_{n}"] - df[f"q_rec_{n}"]))
    print(f"   {n:14s} {e.mean():7.2f} | {e.max():7.2f}")
print(f"sim joints at frame 0 (rad): {df.loc[0, [f'q_sim_{n}' for n in JOINT_NAMES]].values.round(3)}  recorded: {rec_rad[0].round(3)}")
for b in body_ids:
    d = np.linalg.norm(df[[f"{b}_x", f"{b}_y", f"{b}_z"]].values - rec_ee, axis=1)
    print(f"{b} body (base frame) vs dataset EE: mean {d.mean() * 100:.1f} cm, frame0 sim={df.loc[0, [f'{b}_x', f'{b}_y', f'{b}_z']].values.round(3)} rec={rec_ee[0].round(3)}")
if "ee_sim_x" in df:
    d = rec_ee[:len(df)] - df[["ee_sim_x", "ee_sim_y", "ee_sim_z"]].values
    print(f"sim gripper_frame_link vs dataset EE: mean {np.linalg.norm(d, axis=1).mean() * 100:.1f} cm, "
          f"rec - sim per axis mean {(d.mean(0) * 100).round(2)} cm / std {(d.std(0) * 100).round(2)} cm")
for k in (190, 344, 400):
    if k < len(df):
        print(f"frame {k}: gripper={df.loc[k, ['gripper_x', 'gripper_y', 'gripper_z']].values.round(3)} rec_ee={rec_ee[k].round(3)} "
              f"cube_top={df.loc[k, ['cube_top_x', 'cube_top_y', 'cube_top_z']].values.round(3)}")
print(f"cube_top max height: {df.cube_top_z.max():.3f} m (start {df.cube_top_z.iloc[0]:.3f}); "
      f"final top-base offset: {(df[['cube_top_x', 'cube_top_y', 'cube_top_z']].values[-1] - df[['cube_base_x', 'cube_base_y', 'cube_base_z']].values[-1]).round(3)}")
from scipy.spatial.transform import Rotation as _R
for b in body_ids:
    P = df[[f"{b}_x", f"{b}_y", f"{b}_z"]].values
    Rm = _R.from_quat(df[[f"{b}_qx", f"{b}_qy", f"{b}_qz", f"{b}_qw"]].values).as_matrix()   # scipy wants xyzw
    A = np.concatenate([Rm, np.repeat(np.eye(3)[None], len(P), 0)], axis=2).reshape(-1, 6)    # rec - p = R d + t
    sol, *_ = np.linalg.lstsq(A, (rec_ee[:len(P)] - P).reshape(-1), rcond=None)
    fit = P + np.einsum("nij,j->ni", Rm, sol[:3]) + sol[3:]
    err = np.linalg.norm(fit - rec_ee[:len(P)], axis=1)
    print(f"{b}: best EE point offset in {b} frame d={sol[:3].round(3)} m, residual base offset t={sol[3:].round(3)} m, "
          f"fit error mean {err.mean() * 100:.1f} cm / max {err.max() * 100:.1f} cm")
print(f"wrote {OUT}")
env.close()
simulation_app.close()
