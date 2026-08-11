# """This script converts IsaacLab HDF5 datasets into LeRobot Dataset v3 format.

# Since LeRobot is evolving rapidly, compatibility with the latest LeRobot versions is not guaranteed.
# Please install the following specific versions of the dependencies:

# pip install lerobot==0.4.2
# pip install numpy==1.26.0

# """

# import argparse
# import os

# from isaaclab.app import AppLauncher
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from tqdm import tqdm

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot Dataset v3.")
# parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
# parser.add_argument(
#     "--task_type",
#     type=str,
#     default=None,
#     help=(
#         "Specify task type. If your dataset is recorded with keyboard/gamepad, you should set it to"
#         " 'keyboard'/'gamepad', otherwise not to set it and keep default value None."
#     ),
# )
# parser.add_argument(
#     "--repo_id",
#     type=str,
#     default="EverNorif/so101_test_orange_pick",
#     help="Repository ID",
# )
# parser.add_argument(
#     "--fps",
#     type=int,
#     default=30,
#     help="Frames per second",
# )
# parser.add_argument(
#     "--hdf5_root",
#     type=str,
#     default="./datasets",
#     help="HDF5 root directory",
# )
# parser.add_argument(
#     "--hdf5_files",
#     type=str,
#     default=None,
#     help="HDF5 files (comma-separated). If not provided, uses dataset.hdf5 in hdf5_root",
# )
# parser.add_argument(
#     "--task_description",
#     type=str,
#     default=None,
#     help="Task description. If not provided, will use the description defined in the task.",
# )
# parser.add_argument(
#     "--push_to_hub",
#     action="store_true",
#     help="Push to hub",
# )

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()
# # default arguments
# default_args = {
#     "headless": True,
#     "enable_cameras": True,
# }
# app_launcher_args = vars(args_cli)
# app_launcher_args.update(default_args)

# # launch omniverse app
# app_launcher = AppLauncher(app_launcher_args)
# simulation_app = app_launcher.app


# import gymnasium as gym
# import torch
# from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
# from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
# from isaaclab_tasks.utils import parse_env_cfg
# from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
# from leisaac.utils.env_utils import get_task_type
# from leisaac.utils.robot_utils import build_feature_from_env


# def split_episode(episode: EpisodeData, num_frames: int) -> list[EpisodeData]:
#     def slice_at_index(data, idx: int):
#         """Take the idx-th frame from the nested data structure."""
#         if isinstance(data, dict):
#             return {k: slice_at_index(v, idx) for k, v in data.items()}
#         if isinstance(data, torch.Tensor):
#             safe_idx = idx if idx < data.shape[0] else 0
#             return [data[safe_idx]]
#         return data

#     full_data = episode.data
#     sub_episodes: list[EpisodeData] = []
#     for idx in range(num_frames):
#         sub_episode = EpisodeData()
#         sub_episode.data = slice_at_index(full_data, idx)
#         sub_episodes.append(sub_episode)

#     return sub_episodes


# def add_episode(
#     dataset: LeRobotDataset,
#     episode: EpisodeData,
#     env: ManagerBasedRLEnv | DirectRLEnv,
#     dataset_cfg: LeRobotDatasetCfg,
#     task: str,
# ):
#     all_data = episode.data
#     num_frames = all_data["actions"].shape[0]
#     if num_frames < 10:
#         print(f"Episode {episode.env_id} has less than 10 frames, skip it")
#         return False

#     episode_list = split_episode(episode, num_frames)
#     # skip the first 5 frames
#     for frame_index in tqdm(range(5, num_frames), desc="Processing each frame"):
#         frame = env.cfg.build_lerobot_frame(episode_list[frame_index], dataset_cfg)
#         if task is not None:
#             frame["task"] = task
#         dataset.add_frame(frame=frame)
#     return True


# def convert_isaaclab_to_lerobot():
#     """automatically build features and dataset"""
#     env_cfg = parse_env_cfg(args_cli.task_name, device=args_cli.device, num_envs=1)
#     task_type = get_task_type(args_cli.task_name, args_cli.task_type)
#     env_cfg.use_teleop_device(task_type)

#     env: ManagerBasedRLEnv | DirectRLEnv = gym.make(args_cli.task_name, cfg=env_cfg).unwrapped

#     dataset_cfg = LeRobotDatasetCfg(
#         repo_id=args_cli.repo_id,
#         fps=args_cli.fps,
#         robot_type=env_cfg.robot_name,
#     )
#     dataset_cfg.features = build_feature_from_env(env, dataset_cfg)

#     dataset = LeRobotDataset.create(
#         repo_id=dataset_cfg.repo_id,
#         fps=dataset_cfg.fps,
#         robot_type=dataset_cfg.robot_type,
#         features=dataset_cfg.features,
#     )

#     if args_cli.hdf5_files is None:
#         hdf5_files_list = [os.path.join(args_cli.hdf5_root, "dataset.hdf5")]
#     else:
#         hdf5_files_list = [
#             os.path.join(args_cli.hdf5_root, f.strip()) if not os.path.isabs(f.strip()) else f.strip()
#             for f in args_cli.hdf5_files.split(",")
#         ]

#     now_episode_index = 0
#     for hdf5_id, hdf5_file in enumerate(hdf5_files_list):
#         print(f"[{hdf5_id+1}/{len(hdf5_files_list)}] Processing hdf5 file: {hdf5_file}")

#         dataset_file_handler = HDF5DatasetFileHandler()
#         dataset_file_handler.open(hdf5_file)

#         episode_names = dataset_file_handler.get_episode_names()
#         print(f"Found {len(episode_names)} episodes: {episode_names}")
#         for episode_name in tqdm(episode_names, desc="Processing each episode"):
#             episode = dataset_file_handler.load_episode(episode_name, device=args_cli.device)
#             if not episode.success:
#                 print(f"Episode {episode_name} is not successful, skip it")
#                 continue
#             valid = add_episode(dataset, episode, env, dataset_cfg, args_cli.task_description)
#             if valid:
#                 now_episode_index += 1
#                 dataset.save_episode()
#                 print(f"Saving episode {now_episode_index} successfully")
#             else:
#                 dataset.clear_episode_buffer()

#         dataset_file_handler.close()

#     dataset.finalize()

#     if args_cli.push_to_hub:
#         dataset.push_to_hub()

#     print("Finished converting IsaacLab dataset to LeRobot dataset")
#     env.close()


# if __name__ == "__main__":
#     convert_isaaclab_to_lerobot()

#!/usr/bin/env python3

"""
Convert an Isaac Lab HDF5 dataset to LeRobot Dataset v3.

Target versions:
    lerobot==0.4.2
    numpy==1.26.0

Panda-specific mapping:

    Isaac Lab                         LeRobot
    --------------------------------------------------------
    obs["joint_pos"]             ->    observation.state
    data["actions"]              ->    action
    obs["table_cam"]             ->    observation.images.table_cam
    obs["wrist_cam"]             ->    observation.images.wrist_cam

The resulting dataset is written locally.

IMPORTANT:
    This uses LeRobot Dataset v3.
    dataset.finalize() MUST be called after all episodes.
"""

import argparse
import os
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Convert IsaacLab HDF5 dataset to LeRobot Dataset v3."
)

parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    help="Isaac Lab task name.",
)

parser.add_argument(
    "--task_type",
    type=str,
    default=None,
    help=(
        "Task type if using keyboard/gamepad teleoperation. "
        "Otherwise leave unset."
    ),
)

parser.add_argument(
    "--repo_id",
    type=str,
    default="local/panda_visuomotor",
    help=(
        "LeRobot dataset identifier. "
        "This does not need to be a Hugging Face repository "
        "when root is provided."
    ),
)

parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Dataset FPS.",
)

parser.add_argument(
    "--hdf5_root",
    type=str,
    default="./datasets",
    help="HDF5 root directory.",
)

parser.add_argument(
    "--hdf5_files",
    type=str,
    default=None,
    help=(
        "HDF5 files, comma-separated. "
        "If omitted, uses dataset.hdf5."
    ),
)

parser.add_argument(
    "--task_description",
    type=str,
    default="stack cubes",
    help="Task description.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./lerobot_dataset_0810/ID-visuomotor-based",
    help="Local output directory.",
)

parser.add_argument(
    "--push_to_hub",
    action="store_true",
    help="Push dataset to Hugging Face Hub.",
)


# ============================================================
# Isaac Lab AppLauncher
# ============================================================

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

default_args = {
    "headless": True,
    "enable_cameras": True,
}

app_launcher_args = vars(args_cli)
app_launcher_args.update(default_args)

app_launcher = AppLauncher(
    app_launcher_args
)

simulation_app = app_launcher.app


# ============================================================
# Imports that require Isaac Lab / simulation
# ============================================================

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from isaaclab.envs import (
    DirectRLEnv,
    ManagerBasedRLEnv,
)

from isaaclab.utils.datasets import (
    EpisodeData,
    HDF5DatasetFileHandler,
)

from isaaclab_tasks.utils import (
    parse_env_cfg,
)

from leisaac.enhance.datasets.lerobot_dataset_handler import (
    LeRobotDatasetCfg,
)

from leisaac.utils.env_utils import (
    get_task_type,
)

from leisaac.utils.robot_utils import (
    build_feature_from_env,
)

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)


# ============================================================
# Panda configuration
# ============================================================

PANDA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


# ============================================================
# Utilities
# ============================================================

def to_numpy(x):
    """
    Convert Isaac Lab CUDA/CPU tensors or lists of tensors
    into CPU numpy arrays suitable for LeRobot.
    """

    if isinstance(x, list):

        if len(x) > 0 and torch.is_tensor(x[0]):

            x = torch.stack(x)

        else:

            x = np.asarray(
                x,
                dtype=np.float32,
            )

    if torch.is_tensor(x):

        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    # Remove the single environment/frame dimension.
    if x.ndim > 1 and x.shape[0] == 1:
        x = x.squeeze(0)

    return x


def split_episode(
    episode: EpisodeData,
    num_frames: int,
):
    """
    Split one Isaac Lab EpisodeData into individual
    frame-sized EpisodeData objects.
    """

    def slice_at_index(
        data,
        idx,
    ):

        if isinstance(data, dict):

            return {
                key: slice_at_index(
                    value,
                    idx,
                )
                for key, value in data.items()
            }

        if torch.is_tensor(data):

            safe_idx = (
                idx
                if idx < data.shape[0]
                else 0
            )

            return [
                data[safe_idx]
            ]

        return data

    full_data = episode.data

    sub_episodes = []

    for idx in range(num_frames):

        sub_episode = EpisodeData()

        sub_episode.data = (
            slice_at_index(
                full_data,
                idx,
            )
        )

        sub_episodes.append(
            sub_episode
        )

    return sub_episodes


# ============================================================
# Build one LeRobot frame
# ============================================================

def build_lerobot_frame(
    self,
    frame_data,
    dataset_cfg,
):
    """
    Convert one Panda Isaac Lab frame into
    LeRobot v3 representation.
    """

    data = frame_data.data

    obs = data["obs"]
    actions = data["actions"]

    frame = {}

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    frame["action"] = to_numpy(
        actions
    )

    # --------------------------------------------------------
    # Panda joint positions
    # --------------------------------------------------------

    frame["observation.state"] = (
        to_numpy(
            obs["joint_pos"]
        )
    )

    # --------------------------------------------------------
    # Table camera
    # --------------------------------------------------------

    table_cam = obs["table_cam"]

    if torch.is_tensor(table_cam):

        table_cam = (
            table_cam.detach()
            .cpu()
            .numpy()
        )

    else:

        table_cam = np.asarray(
            table_cam
        )

    # Original Panda code used obs["table_cam"][0].
    #
    # For one environment this removes the env dimension.
    if (
        table_cam.ndim == 4
        and table_cam.shape[0] == 1
    ):
        table_cam = table_cam[0]

    frame[
        "observation.images.table_cam"
    ] = table_cam

    # --------------------------------------------------------
    # Wrist camera
    # --------------------------------------------------------

    wrist_cam = obs["wrist_cam"]

    if torch.is_tensor(wrist_cam):

        wrist_cam = (
            wrist_cam.detach()
            .cpu()
            .numpy()
        )

    else:

        wrist_cam = np.asarray(
            wrist_cam
        )

    if (
        wrist_cam.ndim == 4
        and wrist_cam.shape[0] == 1
    ):
        wrist_cam = wrist_cam[0]

    frame[
        "observation.images.wrist_cam"
    ] = wrist_cam

    # --------------------------------------------------------
    # Task
    # --------------------------------------------------------

    frame["task"] = (
        args_cli.task_description
    )

    return frame


# ============================================================
# Add one episode
# ============================================================

def add_episode(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    env,
    dataset_cfg: LeRobotDatasetCfg,
    task: str,
):
    """
    Add one Isaac Lab episode to the LeRobot dataset.
    """

    all_data = episode.data

    num_frames = (
        all_data["actions"].shape[0]
    )

    if num_frames < 10:

        print(
            f"Episode {episode.env_id} "
            f"has less than 10 frames; skipping."
        )

        return False

    # --------------------------------------------------------
    # Split episode into frames
    # --------------------------------------------------------

    episode_list = split_episode(
        episode,
        num_frames,
    )

    # --------------------------------------------------------
    # Preserve your original behavior:
    # skip first 5 frames
    # --------------------------------------------------------

    for frame_index in tqdm(
        range(5, num_frames),
        desc=f"Frames in {episode.env_id}",
        leave=True,
    ):

        frame = (
            env.cfg.build_lerobot_frame(
                episode_list[frame_index],
                dataset_cfg,
            )
        )

        # The frame builder already provides task.
        # Override it only if the CLI task was supplied.
        if task is not None:
            frame["task"] = task

        dataset.add_frame(
            frame=frame
        )

    return True


# ============================================================
# Main conversion
# ============================================================

def convert_isaaclab_to_lerobot():

    print()
    print("=" * 70)
    print("Isaac Lab -> LeRobot Dataset v3")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Isaac Lab environment configuration
    # --------------------------------------------------------

    print("Loading Isaac Lab task configuration...")

    env_cfg = parse_env_cfg(
        args_cli.task_name,
        device=args_cli.device,
        num_envs=1,
    )

    # --------------------------------------------------------
    # Task type
    # --------------------------------------------------------

    task_type = get_task_type(
        args_cli.task_name,
        args_cli.task_type,
    )

    # Your previous script used:
    #
    # env_cfg.teleop_devices = task_type
    #
    # rather than use_teleop_device().
    #
    env_cfg.teleop_devices = task_type

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    print("Creating Isaac Lab environment...")

    env = gym.make(
        args_cli.task_name,
        cfg=env_cfg,
    ).unwrapped

    print("Isaac Lab environment created.")
    print()

    # --------------------------------------------------------
    # Install Panda frame builder
    # --------------------------------------------------------

    env_cfg.build_lerobot_frame = MethodType(
        build_lerobot_frame,
        env_cfg,
    )

    # --------------------------------------------------------
    # Panda feature configuration
    # --------------------------------------------------------

    dataset_cfg = LeRobotDatasetCfg(
        repo_id=args_cli.repo_id,
        fps=args_cli.fps,
        robot_type="franka",
    )

    env_cfg.default_feature_joint_names = (
        PANDA_JOINT_NAMES
    )

    print(
        "Building LeRobot feature definitions..."
    )

    dataset_cfg.features = (
        build_feature_from_env(
            env,
            dataset_cfg,
        )
    )

    print(
        "Features:"
    )

    for key, feature in (
        dataset_cfg.features.items()
    ):

        print(
            f"  {key}: {feature}"
        )

    print()

    # --------------------------------------------------------
    # HDF5 files
    # --------------------------------------------------------

    if args_cli.hdf5_files is None:

        hdf5_files_list = [
            os.path.join(
                args_cli.hdf5_root,
                "dataset.hdf5",
            )
        ]

    else:

        hdf5_files_list = [

            (
                os.path.join(
                    args_cli.hdf5_root,
                    f.strip(),
                )
                if not os.path.isabs(
                    f.strip()
                )
                else f.strip()
            )

            for f in (
                args_cli.hdf5_files.split(",")
            )
        ]

    print(
        "HDF5 files:"
    )

    for hdf5_file in hdf5_files_list:

        print(
            f"  {hdf5_file}"
        )

        if not os.path.isfile(
            hdf5_file
        ):

            raise FileNotFoundError(
                f"HDF5 file does not exist:\n"
                f"{hdf5_file}"
            )

    print()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        args_cli.output_dir
    )

    if output_dir.exists():

        raise FileExistsError(
            "\nOutput directory already exists:\n"
            f"  {output_dir}\n\n"
            "Delete it before running again:\n"
            f"  rm -rf {output_dir}\n"
        )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create LeRobot v3 dataset
    # --------------------------------------------------------

    print(
        "Creating LeRobot v3 dataset..."
    )

    print(
        f"  repo_id: {dataset_cfg.repo_id}"
    )

    print(
        f"  root:    {output_dir}"
    )

    print(
        f"  fps:     {dataset_cfg.fps}"
    )

    print()

    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=dataset_cfg.robot_type,
        features=dataset_cfg.features,
        root=output_dir,
        use_videos=True,
    )

    print(
        "Dataset created."
    )

    print()

    # --------------------------------------------------------
    # Convert episodes
    # --------------------------------------------------------

    now_episode_index = 0

    for hdf5_id, hdf5_file in enumerate(
        hdf5_files_list
    ):

        print()
        print("=" * 70)

        print(
            f"Processing HDF5 "
            f"{hdf5_id + 1}/"
            f"{len(hdf5_files_list)}"
        )

        print(
            f"  {hdf5_file}"
        )

        print("=" * 70)
        print()

        dataset_file_handler = (
            HDF5DatasetFileHandler()
        )

        dataset_file_handler.open(
            hdf5_file
        )

        episode_names = (
            dataset_file_handler
            .get_episode_names()
        )

        print(
            f"Found {len(episode_names)} episodes:"
        )

        print(
            episode_names
        )

        print()

        # ----------------------------------------------------
        # Process each episode
        # ----------------------------------------------------

        for episode_name in tqdm(
            episode_names,
            desc="Processing episodes",
        ):

            print()
            print(
                f"Episode: {episode_name}"
            )

            episode = (
                dataset_file_handler
                .load_episode(
                    episode_name,
                    device=args_cli.device,
                )
            )

            if not episode.success:

                print(
                    f"Episode {episode_name} "
                    f"was unsuccessful; skipping."
                )

                continue

            num_frames = (
                episode.data[
                    "actions"
                ].shape[0]
            )

            print(
                f"  Frames: {num_frames}"
            )

            valid = add_episode(
                dataset,
                episode,
                env,
                dataset_cfg,
                args_cli.task_description,
            )

            if valid:

                now_episode_index += 1

                print(
                    f"Saving episode "
                    f"{now_episode_index}..."
                )

                dataset.save_episode()

                print(
                    f"Episode "
                    f"{now_episode_index} saved."
                )

            else:

                dataset.clear_episode_buffer()

        dataset_file_handler.close()

        print()
        print(
            f"Finished HDF5 file: "
            f"{hdf5_file}"
        )

    # ========================================================
    # IMPORTANT: FINALIZE
    # ========================================================

    print()
    print("=" * 70)
    print("FINALIZING DATASET")
    print("=" * 70)
    print()

    dataset.finalize()

    print(
        "dataset.finalize() completed."
    )

    # ========================================================
    # Verify Parquet files
    # ========================================================

    print()
    print("=" * 70)
    print("VERIFYING PARQUET FILES")
    print("=" * 70)
    print()

    import pyarrow.parquet as pq

    parquet_files = sorted(
        output_dir.rglob(
            "*.parquet"
        )
    )

    if not parquet_files:

        raise RuntimeError(
            "No Parquet files were generated."
        )

    for parquet_file in parquet_files:

        print(
            f"Checking: {parquet_file}"
        )

        try:

            pq.read_schema(
                parquet_file
            )

        except Exception as exc:

            raise RuntimeError(
                "\nINVALID PARQUET FILE:\n"
                f"{parquet_file}\n\n"
                f"{exc}"
            ) from exc

    print()
    print(
        "ALL PARQUET FILES ARE VALID."
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Output:\n"
        f"  {output_dir}"
    )

    print(
        f"\nEpisodes:\n"
        f"  {now_episode_index}"
    )

    print(
        f"\nParquet files:\n"
        f"  {len(parquet_files)}"
    )

    print()

    # --------------------------------------------------------
    # Optional Hub upload
    # --------------------------------------------------------

    if args_cli.push_to_hub:

        print(
            "Pushing dataset to Hugging Face..."
        )

        dataset.push_to_hub()

        print(
            "Push complete."
        )

    env.close()

    print(
        "Done."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:

        convert_isaaclab_to_lerobot()

    finally:

        simulation_app.close()