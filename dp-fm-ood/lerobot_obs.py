# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab observation -> LeRobot MultiTaskDiT observation, shared by every client script.

WHY THIS FILE EXISTS
    run_policy_fm.py, replay_training_rollouts.py and replay_dataset_with_scoring.py each had
    their own copy of make_lerobot_obs(), because importing run_policy_fm.py re-runs its
    module-level AppLauncher. Three copies meant a state-layout fix had to land in three
    places, and the copies had already drifted (only one logged raw state). This module has NO
    argparse and NO AppLauncher, so all three can import it safely - but import it AFTER the
    AppLauncher block, alongside the other post-launch imports, since it pulls in torch.

STATE LAYOUT - THE PART THAT ACTUALLY MATTERS
    observation.state is normalized with MIN_MAX against the training dataset's
    meta/stats.json, so the layout the client builds must be the layout that wrote the
    dataset. Two exist in this project:

      joint_pos (pre-0908 checkpoints, lerobot_dataset_0810/):
          9 dims - obs["joint_pos"], 7 arm joints + 2 fingers.
      eef_pose (0908 onward, lerobot_dataset_0908_eef_pos_300/, written by
      0908_eef_pos_isaac2lerobot.py):
          8 dims - [x, y, z, rx, ry, rz, gripper_1, gripper_2], see eef_state() below.

    Sending the wrong one is not a silent degradation, it is the error
        "The size of tensor a (9) must match the size of tensor b (8) at non-singleton
         dimension 1"
    raised server-side when the normalizer broadcasts state against mean/std of the other
    dim. (a = what the client sent, b = what the checkpoint expects - the numbers swap
    depending on which way round the mismatch is.)

    Which layout to build is NOT guessed from the checkpoint folder name: multitask_dit_server
    reports the checkpoint's own config.json `input_features["observation.state"].shape[0]` in
    its reset reply, and resolve_state_mode() maps that dim to a layout. A name heuristic and
    an explicit override both exist as fallbacks for an old server that doesn't send the dim.
"""

import csv
import json
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Dim -> layout. This is the whole mapping; it works because the two layouts happen to differ
# in width. If a third layout ever lands with a colliding dim (e.g. a 9-dim 6D-rotation EE
# state, which is the documented next ablation in 0908_eef_pos_isaac2lerobot.py's eef_state),
# this must stop being inferred from the dim alone - pass --state_mode explicitly then.
STATE_MODE_BY_DIM = {9: "joint_pos", 8: "eef_pose"}
STATE_MODES = ("auto", "joint_pos", "eef_pose")

STATE_COLUMN_NAMES = {
    "joint_pos": [f"joint_{i}" for i in range(9)],
    "eef_pose": ["x", "y", "z", "rx", "ry", "rz", "gripper_1", "gripper_2"],
}


def resolve_state_mode(requested="auto", state_dim=None, checkpoint=None):
    """Decide which observation.state layout to build.

    Resolution order, most authoritative first:
      1. `requested` if it is not "auto" - an explicit --state_mode always wins.
      2. `state_dim`, the dim the SERVER reported for the loaded checkpoint (reset reply).
         This is the checkpoint's own config.json, so it cannot disagree with the model.
      3. `checkpoint`'s config.json read directly off disk, when the caller has a checkpoint
         path (run_policy_fm.py does; the replay scripts don't - the server owns the path).
      4. "eef" appearing in the checkpoint path - a last-resort name heuristic, since the
         0908-onward runs are named ..._eefpos. Only reached when 2 and 3 both failed.
      5. "joint_pos", the historical default.

    Returns:
        (mode, why) - `why` is a short human-readable reason, printed by the callers so a
        silently-wrong layout is at least a visible line in the log.
    """
    if requested and requested != "auto":
        if requested not in STATE_COLUMN_NAMES:
            raise ValueError(f"unknown state_mode {requested!r}, expected one of {STATE_MODES}")
        return requested, f"--state_mode {requested} (explicit)"

    if state_dim is not None:
        mode = STATE_MODE_BY_DIM.get(int(state_dim))
        if mode is None:
            raise ValueError(
                f"server reported observation.state dim {state_dim}, which matches neither the "
                f"9-dim joint_pos nor the 8-dim eef_pose layout. Pass --state_mode explicitly "
                f"and add the layout to lerobot_obs.py."
            )
        return mode, f"server reported observation.state dim {state_dim}"

    dim, config_path = _state_dim_from_checkpoint(checkpoint)
    if dim is not None:
        mode = STATE_MODE_BY_DIM.get(dim)
        if mode is not None:
            return mode, f"{config_path} declares observation.state dim {dim}"

    if checkpoint and "eef" in str(checkpoint).lower():
        return "eef_pose", f"'eef' in checkpoint path {checkpoint!r} (name heuristic - unverified)"

    return "joint_pos", "default (no server dim, no readable config.json, no 'eef' in the path)"


def _state_dim_from_checkpoint(checkpoint):
    """(dim, config_path) from a checkpoint's config.json, or (None, None) if unreadable.

    Accepts either a `.../pretrained_model` dir or its `.../checkpoints/<step>` parent. Never
    raises: a hub id, a missing file or an unexpected schema all just fall through to None so
    resolve_state_mode() can try the next source.
    """
    if not checkpoint:
        return None, None
    for candidate in (
        os.path.join(str(checkpoint), "config.json"),
        os.path.join(str(checkpoint), "pretrained_model", "config.json"),
    ):
        try:
            with open(candidate) as f:
                cfg = json.load(f)
            return int(cfg["input_features"]["observation.state"]["shape"][0]), candidate
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None, None


def eef_state(eef_pos, eef_quat, gripper_pos):
    """End-effector state as [pos(3), axis-angle(3), gripper(2)] = 8 dims -- LIBERO's layout.

    This is the INFERENCE-side twin of 0908_eef_pos_isaac2lerobot.py's `eef_state()`, and the
    two must stay equivalent: whatever layout/convention wrote observation.state into the
    training dataset is the only thing the checkpoint's MIN_MAX stats are valid for. See that
    converter's docstring for WHY EE state replaced joint_pos (actions are relative EE deltas,
    so joint-space state made the policy learn state->action across an unmodelled IK gap) and
    for the axis-angle discontinuity caveat that LIBERO shares.

    Convention trap, same as in the converter: IsaacLab's eef_quat is scalar-FIRST (w, x, y,
    z) (isaaclab/utils/math.py); scipy's Rotation.from_quat wants scalar-LAST (x, y, z, w).
    Passing eef_quat straight through produces wrong rotations with no error.

    Args:
        eef_pos: obs["eef_pos"] -- [3] or [1, 3], end-effector position in world frame.
        eef_quat: obs["eef_quat"] -- [4] or [1, 4], (w, x, y, z).
        gripper_pos: obs["gripper_pos"] -- [2] or [1, 2], the two finger joint positions.

    Returns:
        A [8] float32 tensor on the same device as `eef_pos` (the images stay on the sim
        device too, and the server's preprocessor moves the whole batch itself).
    """
    device = eef_pos.device

    q_wxyz = eef_quat.detach().to(torch.float64).cpu().numpy().reshape(4)
    q_xyzw = q_wxyz[[1, 2, 3, 0]]                        # IsaacLab -> scipy
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    axis_angle = Rotation.from_quat(q_xyzw).as_rotvec()  # (3,)

    return torch.cat([
        eef_pos.detach().reshape(3).to(torch.float32),
        torch.as_tensor(axis_angle, dtype=torch.float32, device=device),
        gripper_pos.detach().reshape(2).to(torch.float32),
    ]).to(device)


def log_raw_state(state, mode="joint_pos"):
    """Append one raw (un-normalized) observation.state row to $STATE_LOG, if that is set.

    Worth logging because the checkpoint normalizes state with MIN_MAX against
    `meta/stats.json`, which rescales each dim by its TRAINING extremes. Under the joint_pos
    layout two dims are nearly constant in the training set -- joint_7 spans [-0.0243, 0.0]
    and joint_8 spans [-0.0262, 0.0] -- so a value a hair outside that range maps outside
    [-1, 1] and is fed to the policy as an out-of-distribution input. A ~0.025-wide range also
    means MIN_MAX amplifies whatever noise those dims carry by ~80x. The eef_pose layout has
    the same pathology in its two gripper dims. Neither is visible after normalization, hence
    logging the raw values here.

    No-op unless STATE_LOG is set, so the normal eval path is unaffected.
    """
    path = os.environ.get("STATE_LOG")
    if not path:
        return
    values = state.detach().cpu().numpy().ravel()
    header = STATE_COLUMN_NAMES.get(mode) or []
    if len(header) != len(values):
        header = [f"dim_{i}" for i in range(len(values))]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([f"{v:.6f}" for v in values])


def make_lerobot_obs(obs_dict, task_instruction, state_mode="joint_pos"):
    """Convert the current Isaac Lab observation into the LeRobot MultiTaskDiT format.

    Expected Isaac Lab fields (all present in ObservationsCfg.PolicyCfg, see
    stack_ik_rel_visuomotor_env_cfg.py - which is also the source the dataset converter read,
    so no forward kinematics is needed on either side):
        obs_dict["joint_pos"]           -> [9]           (state_mode="joint_pos")
        obs_dict["eef_pos"/"eef_quat"/"gripper_pos"]     (state_mode="eef_pose")
        obs_dict["table_cam"]           -> [H, W, 3] uint8
        obs_dict["wrist_cam"]           -> [H, W, 3] uint8

    Args:
        obs_dict: Isaac Lab's observation dict for this step.
        task_instruction: Language task label the checkpoint was trained on (e.g. "stack
            cubes" - see the training dataset's meta/tasks.parquet). MultiTaskDiT's LeRobot
            preprocessor requires this "task" key to tokenize task-conditioning.
        state_mode: "joint_pos" or "eef_pose" - see this module's docstring, and get it from
            resolve_state_mode() rather than hardcoding it.
    """
    obs = obs_dict["policy"] if "policy" in obs_dict else obs_dict

    if state_mode == "eef_pose":
        # 8-dim EE pose: the 0908-onward checkpoints.
        state = eef_state(obs["eef_pos"], obs["eef_quat"], obs["gripper_pos"])
    elif state_mode == "joint_pos":
        # 9-dim joint space: the pre-0908 checkpoints. Kept working, not deleted - the older
        # runs are still the comparison baseline.
        state = obs["joint_pos"]
    else:
        raise ValueError(f"unknown state_mode {state_mode!r}, expected 'joint_pos' or 'eef_pose'")

    table_cam = obs["table_cam"]
    wrist_cam = obs["wrist_cam"]

    # Remove batch dimension if Isaac Lab gives [1, ...]. eef_state already returns [8].
    if state.ndim > 1 and state.shape[0] == 1:
        state = state.squeeze(0)

    # Raw, pre-normalization -- this is the only place the un-rescaled values exist.
    log_raw_state(state, state_mode)

    if table_cam.ndim == 4 and table_cam.shape[0] == 1:
        table_cam = table_cam.squeeze(0)

    if wrist_cam.ndim == 4 and wrist_cam.shape[0] == 1:
        wrist_cam = wrist_cam.squeeze(0)

    # Isaac Lab's Camera sensor returns "rgb" as (H, W, 3) uint8 (see
    # isaaclab.sensors.camera.Camera._process_annotator_output), but the LeRobot dataset these
    # checkpoints were trained on stores images as (3, H, W) float32 in [0, 1] - sending the
    # raw uint8 tensor through as-is both scores the model on the wrong pixel format and (via
    # NormalizerProcessorStep's dtype-matching in normalize_processor.py) corrupts the float
    # normalization stats by re-casting them to uint8, which is what raised "value cannot be
    # converted to type uint8 without overflow" server-side.
    def _to_chw_float(img):
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if img.shape[-1] == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        return img.contiguous()

    return {
        "observation.state": state,
        "observation.images.table_cam": _to_chw_float(table_cam),
        "observation.images.wrist_cam": _to_chw_float(wrist_cam),
        "task": task_instruction,
    }
