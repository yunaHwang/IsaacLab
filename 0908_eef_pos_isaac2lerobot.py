# """This script converts IsaacLab HDF5 datasets into LeRobot Dataset v2 format.

# Since LeRobot is evolving rapidly, compatibility with the latest LeRobot versions is not guaranteed.
# Please install the following specific versions of the dependencies:

# pip install lerobot==0.3.3
# pip install numpy==1.26.0

# """

import argparse
import os

from isaaclab.app import AppLauncher
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot Dataset v2.")
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--task_type",
    type=str,
    default="keyboard",
    help=(
        "Specify task type. If your dataset is recorded with keyboard/gamepad, you should set it to"
        " 'keyboard'/'gamepad', otherwise not to set it and keep default value None."
    ),
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="ID/visuomotor-based", #NOTE - change here
    help="Repository ID",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames per second",
)
parser.add_argument(
    "--dataset_suffix",
    type=str,
    required=True,
    help=(
        "Suffix for the output directory: the dataset is written to"
        " ./lerobot_dataset_<dataset_suffix>/<repo_id with '/' replaced by '-'>."
        " e.g. --dataset_suffix 0901_topped_up_300 -> ./lerobot_dataset_0901_topped_up_300/ID-visuomotor-based"
    ),
)
parser.add_argument(
    "--dataset_root",
    type=str,
    default=".",
    help="Directory the lerobot_dataset_<suffix> folder is created in.",
)
parser.add_argument(
    "--hdf5_root",
    type=str,
    default="./datasets",
    help="HDF5 root directory",
)
parser.add_argument(
    "--hdf5_files",
    type=str,
    default=None,
    help="HDF5 files (comma-separated). If not provided, uses dataset.hdf5 in hdf5_root",
)
parser.add_argument(
    "--task_description",
    type=str,
    default=None,
    help="Task description. If not provided, will use the description defined in the task.",
)
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push to hub",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# default arguments
default_args = {
    "headless": True,
    "enable_cameras": True,
}
app_launcher_args = vars(args_cli)
app_launcher_args.update(default_args)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_mimic.envs
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.env_utils import get_task_type
from leisaac.utils.robot_utils import build_feature_from_env


def eef_state(eef_pos, eef_quat, gripper_pos):
    """End-effector state as [pos(3), axis-angle(3), gripper(2)] = 8 dims -- LIBERO's layout.

    WHY THIS REPLACED joint_pos
        The task is `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`: `IK-Rel` means the
        ACTIONS are relative end-effector deltas (3 translation + 3 rotation + 1 gripper). But
        leisaac's converter always wrote `joint_pos` as observation.state -- it is built for
        SO101Leader teleoperation, where the action IS a joint target so joint state and action
        share a frame. Its own comment admits the assumption. On an IK-Rel Franka that breaks:
        the policy reads joint space and writes EE space, so it has to learn forward kinematics
        just to connect its input to its output. The reference model
        (marvin-oh/multitask_dit_libero_plus) and LIBERO both use EE state with EE actions.

    NO FORWARD KINEMATICS NEEDED. IsaacLab already logs obs/eef_pos and obs/eef_quat every
    frame. LeRobot's ForwardKinematicsJointsToEE exists for REAL robots that only report joint
    encoders and must recompute the pose from a URDF; recomputing a measured quantity here would
    only add a URDF dependency and risk a tool-frame mismatch with IsaacLab's EE definition.

    QUATERNION ORDER IS A SILENT TRAP. IsaacLab is scalar-FIRST (w, x, y, z)
    (isaaclab/utils/math.py); scipy's Rotation.from_quat wants scalar-LAST (x, y, z, w). Passing
    eef_quat straight through produces wrong rotations with no error.

    ON AXIS-ANGLE. It is degenerate at 180 deg, where +pi*n and -pi*n are the same rotation, so a
    tiny wobble flips the encoding across its whole range. This gripper points down (~180 deg from
    the base frame) for 96.5% of frames, and its yaw also wraps: measured over all 85,411 frames,
    raw axis-angle spans 2*pi per dim, and even measured relative to "straight down" it reaches
    3.139 rad against pi=3.142. So the wrapping here is REAL.

    We use axis-angle anyway, on purpose: LIBERO has the same pathology (its state dim 4 spans
    7.202 > 2*pi, dim 3 has mean 2.972 against pi=3.142) and Pi0.5 still scores 97.5% on it. A
    6D rotation (first two columns of the rotation matrix, 9-dim state) removes the discontinuity
    entirely and is the cheap one-variable ablation if this is ever suspected of costing accuracy
    -- but matching the reference comes first.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    q_wxyz = np.asarray(eef_quat, dtype=np.float64).reshape(4)
    q_xyzw = q_wxyz[[1, 2, 3, 0]]                      # IsaacLab -> scipy
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    axis_angle = R.from_quat(q_xyzw).as_rotvec()       # (3,)

    return np.concatenate([
        np.asarray(eef_pos, dtype=np.float32).reshape(3),
        axis_angle.astype(np.float32),
        np.asarray(gripper_pos, dtype=np.float32).reshape(2),
    ]).astype(np.float32)


def patch_state_feature(features, state_dim=8):
    """Override leisaac's joint-space state feature with the 8-dim EE state.

    No object/cube columns are written: LIBERO's observation.state is purely proprioceptive and
    the dataset carries no object pose, so this keeps the schema identical to the reference.
    Cube pose remains available in the source hdf5 (obs/cube_positions, obs/cube_orientations,
    obs/datagen_info/object_pose/cube_*) if a probe or auxiliary loss ever needs it -- join on
    episode and frame index rather than re-exporting.

    build_feature_from_env hardcodes observation.state to len(default_feature_joint_names) (9 for
    this Franka). Writing an 8-dim state against a 9-dim declaration makes LeRobotDataset reject
    the frame, so the declaration has to be corrected too.
    """
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_1", "gripper_2"],
    }
    return features




def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
    def slice_at_index(data, idx: int):
        """Take the idx-th frame from the nested data structure."""
        if isinstance(data, dict):
            return {k: slice_at_index(v, idx) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            safe_idx = idx if idx < data.shape[0] else 0
            return [data[safe_idx]]
        return data

    full_data = episode.data
    sub_episodes: list[EpisodeData] = []
    for idx in range(num_frames):
        sub_episode = EpisodeData()
        sub_episode.data = slice_at_index(full_data, idx)
        sub_episodes.append(sub_episode)

    return sub_episodes


def add_episode(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    env: ManagerBasedRLEnv | DirectRLEnv,
    dataset_cfg: LeRobotDatasetCfg,
    task: str,
):
    all_data = episode.data
    num_frames = all_data["actions"].shape[0]
    if num_frames < 10:
        print(f"Episode {episode.env_id} has less than 10 frames, skip it")
        return False

    episode_list = split_episode(episode, num_frames)
    # skip the first 5 frames
    for frame_index in tqdm(range(5, num_frames), desc="Processing each frame"):
        frame = env.cfg.build_lerobot_frame(episode_list[frame_index], dataset_cfg)
        # print("frame keys", frame.keys())

        predefined_task = frame["task"]   # don't pop!

        # do this for 0.3.3
        # predefined_task = frame.pop("task")
        
        dataset.add_frame(frame)

        # do this for 0.3.3
        #dataset.add_frame(frame=frame, task=predefined_task if task is None else task)
    return True


def convert_isaaclab_to_lerobot():
    import isaaclab_mimic.envs

    print("Mimic envs:")
    for name in gym.registry.keys():
        if "Mimic" in name:
            print(name)
    
    """automatically build features and dataset"""
    env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=1)


    # print("[OBS] env_cfg.observations ", env_cfg.observations)
    # print("policy too, ", env_cfg.observations.policy)

    task_type = get_task_type(args_cli.task_name, args_cli.task_type)
    # env_cfg.use_teleop_device(task_type)
    env_cfg.teleop_devices = task_type

    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task_name, cfg=env_cfg).unwrapped
    # import pdb
    # import traceback

    # try:
    #     env = gym.make(args_cli.task_name, cfg=env_cfg, render_mode="rgb_array").unwrapped
    # except Exception as e:
    #     print("Exception type:", type(e))
    #     print("Exception:", e)
    #     print("Cause:", repr(e.__cause__))
    #     print("Context:", repr(e.__context__))

    #     if e.__cause__:
    #         traceback.print_exception(type(e.__cause__), e.__cause__, e.__cause__.__traceback__)
    #     elif e.__context__:
    #         traceback.print_exception(type(e.__context__), e.__context__, e.__context__.__traceback__)

    #     raise

    from types import MethodType

    # def build_lerobot_frame(self, frame_data, dataset_cfg):
    #     # convert IsaacLab Mimic frame -> LeRobot frame
    #     print(type(frame_data))
    #     print(dir(frame_data))

    #     print(type(frame_data.data))
    #     print(frame_data.data.keys())

    #     obs = frame_data["obs"]

    #     return {
    #         "observation.state": obs["joint_pos"].astype("float32"),
    #         "action": frame_data["actions"].astype("float32"),
    #     }

    # def build_lerobot_frame(self, frame_data, dataset_cfg):

    #     import numpy as np
    #     import torch

    #     data = frame_data.data

    #     obs = data["obs"]
    #     actions = data["actions"]

    #     frame = {}

    #     print(type(actions))
    #     print(type(actions[0]))
    #     print(actions[0].device if torch.is_tensor(actions[0]) else None)

    #     # # # LeRobot action
    #     # # if isinstance(actions, torch.Tensor):
    #     # #     actions = actions.cpu().numpy()

    #     # # print(type(actions), getattr(actions, "device", None))
    #     # # frame["action"] = np.asarray(actions, dtype=np.float32)

    #     # # # Robot proprioception
    #     # joint_pos = obs["joint_pos"]
    #     # # if isinstance(joint_pos, torch.Tensor):
    #     # #     joint_pos = joint_pos.cpu().numpy()

    #     # # print(type(joint_pos), getattr(joint_pos, "device", None))
    #     # frame["observation.state"] = np.asarray(joint_pos.cpu(), dtype=np.float32)

    #     # frame["task"] = "stack cubes"

    #     return frame

    # def to_numpy(x):

    #     import numpy as np
    #     import torch
    #     """
    #     Convert IsaacLab data (CUDA tensors, CPU tensors, lists of tensors, lists)
    #     into CPU numpy arrays suitable for LeRobot.
    #     """
    #     if torch.is_tensor(x):
    #         return x.detach().cpu().numpy().astype(np.float32)

    #     if isinstance(x, list):
    #         if len(x) > 0 and torch.is_tensor(x[0]):
    #             return torch.stack(x).detach().cpu().numpy().astype(np.float32)
    #         else:
    #             return np.asarray(x, dtype=np.float32)

    #     return np.asarray(x, dtype=np.float32)

    def to_numpy(x):

        import numpy as np
        import torch
        
        """
        Convert IsaacLab data (CUDA tensors, CPU tensors, lists of tensors, lists)
        into CPU numpy arrays suitable for LeRobot.
        """
        if isinstance(x, list):
            if len(x) > 0 and torch.is_tensor(x[0]):
                x = torch.stack(x)
            else:
                x = np.asarray(x, dtype=np.float32)

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        x = np.asarray(x, dtype=np.float32)

        # Remove single-frame/batch dimension
        if x.ndim > 1 and x.shape[0] == 1:
            x = x.squeeze(0)

        return x


    def build_lerobot_frame(self, frame_data, dataset_cfg):

        import numpy as np

        data = frame_data.data

        obs = data["obs"]
        # print("obs keys, ", obs.keys())
        actions = data["actions"]

        frame = {}

        # LeRobot action
        frame["action"] = to_numpy(actions)

        # Robot proprioception -- END-EFFECTOR pose, matching the LIBERO 8-dim layout
        # (eef position, axis-angle orientation, gripper qpos). See eef_state() below for why
        # this replaced joint_pos.
        frame["observation.state"] = eef_state(
            to_numpy(obs["eef_pos"]), to_numpy(obs["eef_quat"]), to_numpy(obs["gripper_pos"])
        )


        # NOTE: yuna add - image
        frame["observation.images.table_cam"] = (
            obs["table_cam"][0]
            .detach()
            .cpu()
            .numpy()
        )

        frame["observation.images.wrist_cam"] = (
            obs["wrist_cam"][0]
            .detach()
            .cpu()
            .numpy()
        )

        # frame["observation.images.table_cam"] = to_numpy(obs["table_cam"])
        # frame["observation.images.wrist_cam"] = to_numpy(obs["wrist_cam"])

        # table_img = obs["table_cam"]

        # table_img = obs["table_cam"]

        # print(type(table_img))
        # print(len(table_img))
        # print(type(table_img[0]))
        # print(getattr(table_img[0], "shape", None))
        # print(getattr(table_img[0], "dtype", None))


        # if torch.is_tensor(table_img):
        #     table_img = table_img.detach().cpu().numpy()

        # frame["observation.images.table_cam"] = table_img.astype(np.uint8)

        # wrist_img = obs["wrist_cam"]
        # if torch.is_tensor(wrist_img):
        #     wrist_img = wrist_img.detach().cpu().numpy()

        # frame["observation.images.wrist_cam"] = wrist_img.astype(np.uint8)

        # LeRobot task description
        frame["task"] = args_cli.task_description if args_cli.task_description else "stack cubes"

        return frame

    env_cfg.build_lerobot_frame = MethodType(
        build_lerobot_frame,
        env_cfg,
    )

    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        #robot_type=env_cfg.robot_name,
        robot_type="franka"
    )
    env_cfg.default_feature_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    dataset_cfg.features = build_feature_from_env(env, dataset_cfg)
    # Replace the hardcoded joint-space state with the EE state, and add the diagnostic columns.
    dataset_cfg.features = patch_state_feature(dataset_cfg.features)

    #print("keys", dataset_cfg.features.keys())

    # ./lerobot_dataset_<suffix>/<repo_id>, with repo_id's '/' flattened to '-' so it is one dir level
    dataset_root = os.path.join(
        args_cli.dataset_root,
        f"lerobot_dataset_{args_cli.dataset_suffix}",
        dataset_cfg.repo_id.replace("/", "-"),
    )
    print(f"[isaaclab2lerobot] writing LeRobot dataset to: {dataset_root}")

    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
        root=dataset_root,
    )

    if args_cli.hdf5_files is None:
        hdf5_files_list = [os.path.join(args_cli.hdf5_root, "dataset.hdf5")]
    else:
        hdf5_files_list = [
            os.path.join(args_cli.hdf5_root, f.strip()) if not os.path.isabs(f.strip()) else f.strip()
            for f in args_cli.hdf5_files.split(",")
        ]

    now_episode_index = 0

    for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
        print(
            f"[{hdf5_id+1}/{len(hdf5_files_list)}] "
            f"Processing hdf5 file: {hdf5_file}"
        )

        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(hdf5_file)

        episode_names = dataset_file_handler.get_episode_names()
        print(f"Found {len(episode_names)} episodes: {episode_names}")

        for episode_name in tqdm(
            episode_names,
            desc="Processing each episode"
        ):
            episode = dataset_file_handler.load_episode(
                episode_name,
                device=args_cli.device
            )

            if not episode.success:
                print(
                    f"Episode {episode_name} "
                    f"is not successful, skip it"
                )
                continue

            valid = add_episode(
                dataset,
                episode,
                env,
                dataset_cfg,
                args_cli.task_description
            )

            if valid:
                now_episode_index += 1
                dataset.save_episode()
                print(
                    f"Saving episode "
                    f"{now_episode_index} successfully"
                )
            else:
                dataset.clear_episode_buffer()

        dataset_file_handler.close()

    # IMPORTANT: finalize ONCE, after all episodes
    print("Finalizing LeRobot dataset...")
    dataset.finalize()
    print("Dataset finalized.")

    from pathlib import Path

    root = Path(dataset.root)

    for p in root.rglob("*.parquet"):
        print(p)

    # now_episode_index = 0
    # for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
    #     print(f"[{hdf5_id+1}/{len(hdf5_files_list)}] Processing hdf5 file: {hdf5_file}")

    #     dataset_file_handler = HDF5DatasetFileHandler()
    #     dataset_file_handler.open(hdf5_file)

    #     episode_names = dataset_file_handler.get_episode_names()
    #     print(f"Found {len(episode_names)} episodes: {episode_names}")
    #     for episode_name in tqdm(episode_names, desc="Processing each episode"):
    #         episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)
    #         if not episode.success:
    #             print(f"Episode {episode_name} is not successful, skip it")
    #             continue
    #         valid = add_episode(dataset, episode, env, dataset_cfg, args_cli.task_description)
    #         if valid:
    #             now_episode_index += 1
    #             dataset.save_episode()                

    #             print(f"Saving episode {now_episode_index} successfully")
    #         else:
    #             dataset.clear_episode_buffer()

    #     dataset_file_handler.close()


    #     # adding for 0.4.4
    #     print("Closing parquet writer...")
    #     # dataset._close_writer()
    #     # dataset.finalize()

    #     # dataset_file_handler.close()

    #     from pathlib import Path

    #     print("Finished processing episodes")

    #     from pathlib import Path

    #     root = Path(dataset.root)

    #     for p in root.rglob("*.parquet"):
    #         print(p)

    
    # print("Finalizing LeRobot dataset...")
    # dataset.finalize()
    # print("Dataset finalized.")



    if args_cli.push_to_hub:
        dataset.push_to_hub()

    print("Finished converting IsaacLab dataset to LeRobot dataset")
    env.close()


if __name__ == "__main__":
    convert_isaaclab_to_lerobot()
