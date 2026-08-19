# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained fm (flow-matching) policy in an Isaac Lab
environment. 

Args:
    task: Name of the environment.
    device: Torch device for the simulation.
    disable_fabric: If set, disable fabric and use USD I/O operations.
    fm_checkpoint: Path or hub id of a trained fm policy checkpoint.
    fm_backbone: Which fm policy implementation to run: 'gloves' or 'multitask_dit'.
    blending_mechanism: Which gloves action-blending mechanism to use once blending is
        wired up ('gloves_fpas', 'gloves_feeg', or 'gloves_ifae').
    blend: If set, enable action blending (not_blend=False); otherwise the policy's own
        actions are executed directly.
    num_samples: Number of (noise, t) draws to average the live fm diffdagger-style loss over.
    horizon: The step horizon of each rollout.
    num_rollouts: Number of rollouts to run.
    seed: Random seed.
"""

# TODO - wire in conformal prediction and smoothness code in

import argparse
from multiprocessing.connection import Client

from ood_signal import *

# NOTE: this script runs in Isaac Lab's env (yuna_env, Python 3.11). Neither the GLOVES
# DiTPolicy nor LeRobot's MultiTaskDiTPolicy are importable here - they live in separate
# conda envs and are reached over multiprocessing.connection instead. See
# multitask_dit_server.py (and, once written, gloves_server.py) for the model-owning side.

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an fm (flow-matching) policy for Isaac Lab environment.")

parser.add_argument("--task", type=str, required=True, help="Name of the environment.")
# parser.add_argument("--device", type=str, default="cpu")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)

parser.add_argument(
    "--fm_backbone", type=str, choices=["gloves", "multitask_dit"], default="multitask_dit",
    help="Which fm policy implementation to run"
)
parser.add_argument(
    "--blending_mechanism", type=str, choices=['gloves_fpas', 'gloves_feeg', 'gloves_ifae'],
    default=None,
    help="Which gloves action-blending mechanism to use once blending is wired up.",
)  # NOTE: add as I go
# FOR now: have not blend as a default, meaning that without this flag, then it will default to not blend
parser.add_argument("--blend", action='store_true')

parser.add_argument(
    "--num_samples", type=int, default=32, # was 512 and ran out of memory
    help="Number of (noise, t) draws to average the live fm diffdagger-style loss over "
    "(the fm analog of diffdagger's Nb in noise_estimation_loss_nb_infer). The live action is "
    "fixed for the duration of a step, so averaging multiple draws reduces the loss's variance.",
)

parser.add_argument(
    "--fm_checkpoint", type=str, required=True,
    help="Path or hub id of a trained fm policy checkpoint"
)

parser.add_argument(
    "--mdit_server_host", type=str, default="127.0.0.1",
    help="Host multitask_dit_server.py is listening on.",
)
parser.add_argument(
    "--mdit_server_port", type=int, default=5555,
    help="Port multitask_dit_server.py is listening on.",
)
parser.add_argument(
    "--mdit_server_authkey", type=str, default="mdit-ipc",
    help="Shared secret for the multiprocessing.connection handshake - must match "
    "multitask_dit_server.py's --authkey.",
)
parser.add_argument(
    "--task_instruction", type=str, default="stack cubes",
    help="Language task label the MultiTaskDiT checkpoint was trained on (see the training "
    "dataset's meta/tasks.parquet) - required for the server's preprocessor to tokenize "
    "task-conditioning.",
)

parser.add_argument(
    "--spacemouse_bridge_host", type=str, default="127.0.0.1",
    help="Host spacemouse_bridge.py is listening on (after SSH -R port-forwarding). Only "
    "used as a fallback when no SpaceMouse is found on local HID - see spacemouse_bridge.py.",
)
parser.add_argument(
    "--spacemouse_bridge_port", type=int, default=6060,
    help="Port spacemouse_bridge.py is listening on.",
)
parser.add_argument(
    "--spacemouse_bridge_authkey", type=str, default="spacemouse-ipc",
    help="Shared secret for the multiprocessing.connection handshake - must match "
    "spacemouse_bridge.py's --authkey.",
)

parser.add_argument("--horizon", type=int, default=500, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts to run.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import random

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab_tasks.utils import parse_env_cfg


class NetworkSe3SpaceMouse:
    """Reads a SpaceMouse over spacemouse_bridge.py instead of local HID hardware - for
    when the physical device is on your client machine but this script runs on a remote SSH
    server that has no direct access to it. See spacemouse_bridge.py's module docstring for
    setup (run it on your client machine, then `ssh -R <port>:localhost:<port>` when
    connecting to this server).

    Drop-in replacement for isaaclab.devices.spacemouse.Se3SpaceMouse's public interface
    (advance/reset/__str__) as used by this script. pos_sensitivity/rot_sensitivity are
    applied here (server-side, from Se3SpaceMouseCfg) rather than by the bridge, so retuning
    sensitivity doesn't require restarting the bridge - see spacemouse_bridge.py's docstring.
    """

    def __init__(self, cfg, host, port, authkey):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        self._conn = Client((host, port), authkey=authkey.encode())

    def __str__(self) -> str:
        msg = f"Spacemouse Controller for SE(3): {self.__class__.__name__} (via network bridge)\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tLeft button: toggle gripper command (open/close)\n"
        msg += "\tMove mouse laterally: move arm horizontally in x-y plane\n"
        msg += "\tMove mouse vertically: move arm vertically\n"
        msg += "\tTwist mouse about an axis: rotate arm about a corresponding axis"
        return msg

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
        rot_vec = Rotation.from_euler("XYZ", delta_rot).as_rotvec()
        command = np.concatenate([delta_pos, rot_vec])
        if self.gripper_term:
            gripper_value = -1.0 if response["close_gripper"] else 1.0
            command = np.append(command, gripper_value)

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)


def run_gloves_policy(fm_policy, obs_history, device, teleop_interface, not_blend=True):
    """
    Perform action blending by 1) grabbing the action from the policy, 2) registering user action,
    3) function call for some metric (loss), 4) save metric value if necessary.

    Score the live human action against a GLOVES flow policy's density non-conformity
    score s(x) = ||z_hat(x)||^2 (paper Eq. 6; see compute_density_score/compute_z_hat in
    get_nonconformity_gloves.py).

    Args:
        fm_policy: The trained GLOVES DiTPolicy to score actions against (see
            --fm_checkpoint).
        obs_history: iterable of `n_obs_steps` obs dicts, in compute_density_score's expected
            format (same keys/shape as run_dp_policy's own obs_history deque). Required when
            not_blend=True (to generate fm_policy's own action chunk).
        device: The device to run the policy on.
        teleop_interface: A device (e.g. Se3SpaceMouse) to read the live human action from.
            Required when not_blend=False.
        not_blend: If True, generate and return the policy's own action chunk (env-stepping
            not wired up yet, see TODO below). If False, score the live human action's
            density.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    from ood_signal_baseline_papers.density.get_nonconformity_gloves import generate_action_chunk

    if not_blend:
        if obs_history is None:
            raise RuntimeError(
                "run_gloves_policy(..., not_blend=True) requires obs_history to "
                "generate the fm policy's own action chunk."
            )


        full_model_actions = generate_action_chunk(
            fm_policy.dit_flow, obs_history, device=device, full_chunk=True
        )
        start = fm_policy.config.n_obs_steps - 1
        end = start + fm_policy.config.n_action_steps
        model_actions = full_model_actions[:, start:end]
        print(f"fm policy generated action chunk of shape {tuple(model_actions.shape)}")

        # TODO: env.step through `model_actions`, mirroring run_dp_policy's not_blend=True
        # branch. Needs env/success_term/horizon wired into this function first (see
        # docstring).
        traj = dict(policy_actions=[model_actions.tolist()], blended_actions=[], obs=[], next_obs=[])

        ###############
        # State OOD: score the model's own action against its own state using density loss. 
        state_ood_score = gloves_density(fm_policy.dit_flow, obs_history, full_model_actions)
        msg = f"[state OOD] fm density for model's own action: {state_ood_score.item():.4f}"
        print(msg)

        # fm diffdagger-style noise-prediction loss 
        fm_loss = gloves_loss(
            fm_policy.dit_flow, obs_history, full_model_actions, num_samples=args_cli.num_samples
        )
        print(f"[state OOD] gloves loss for model's own action: {fm_loss.item():.4f}")

        return False, traj

    if teleop_interface is None:
        raise RuntimeError(
            "run_gloves_policy(..., not_blend=False) requires a teleop_interface "
            "(e.g. Se3SpaceMouse) to read the live human action from."
        )
    # [7]: [x, y, z, rx, ry, rz, gripper] delta-pose command, already on `device`
    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.devices.html
    user_action = teleop_interface.advance()
    print(f"[SpaceMouse] raw 7-DoF action: {user_action.tolist()}")

    score = gloves_density(fm_policy.dit_flow, obs_history, user_action)
    print(f"[ACTION] density non-conformity score for current human action: {score.item():.4f}")

    # TODO: define blended_actions from fm_policy's own action + user_action + score/gamma
    # (see compute_linear_gamma / compute_sigmoid_gamma in joystick_diffdagger.py for the
    # gamma-blend pattern to adapt here), then env.step(blended_actions) and record traj -
    # needs env/success_term/horizon wired in first (see docstring above).
    raise NotImplementedError(
        "Action blending (not_blend=False) isn't wired up yet: user_action and its density "
        "score are now available above, but env-stepping/blending needs env/success_term/"
        "horizon plumbed into this function first."
    )


def run_multitask_ditpolicy(
    conn,
    obs,
    teleop_interface=None,
    not_blend=True,
):
    """
    Run the trained LeRobot MultiTaskDiTPolicy via multitask_dit_server.py.

    The server owns the model, the LeRobot preprocessor/postprocessor, and the
    n_obs_steps window (see multitask_dit_server.py's module docstring -
    `select_action` takes one raw timestep per call and pads/queues internally). This
    function is therefore just an IPC round-trip for a single timestep's observation,
    not a batch of history - `conn` must already have sent {"cmd": "reset"} once at the
    start of the episode (see main()).

    Args:
        conn: An open multiprocessing.connection.Client connection to
            multitask_dit_server.py.
        obs: A single timestep's observation dict, in make_lerobot_obs's format
            (raw, un-batched - the server's preprocessor adds the batch dim).
        teleop_interface: A device (e.g. Se3SpaceMouse) to read the live human action
            from. Required when not_blend=False.
        not_blend: If True, only the policy's own action chunk is returned/scored. If
            False, the live human action is also read (its OOD scoring against the
            server-owned model isn't wired up yet - see TODO below).

    Returns:
        policy_actions: The action chunk returned by the server (already
            unnormalized/postprocessed).
        user_action: The live teleop action, or None if not_blend=True.
    """

    conn.send({"cmd": "step", "obs": obs})
    response = conn.recv()

    if not response.get("ok", False):
        raise RuntimeError(f"multitask_dit_server error: {response.get('error')}")

    # Server sends "action" as a plain nested list, not a torch.Tensor - see
    # multitask_dit_server.py's comment on why (avoids the cross-process
    # multiprocessing shared-memory reducer/authkey mismatch).
    policy_actions = torch.tensor(response["action"])

    print(
        "[FM client] generated action chunk:",
        tuple(policy_actions.shape),
    )
    print(policy_actions)

    # if "state_ood_loss" in response:
    #     print(f"[state OOD] MultiTaskDiT loss for model's own action: {response['state_ood_loss']:.4f}")
    # elif "state_ood_loss_error" in response:
    #     print(f"[state OOD] MultiTaskDiT loss errored server-side: {response['state_ood_loss_error']}")

    # if "state_ood_density" in response:
    #     print(f"[state OOD] MultiTaskDiT density for model's own action: {response['state_ood_density']:.4f}")
    # elif "state_ood_density_error" in response:
    #     print(f"[state OOD] MultiTaskDiT density errored server-side: {response['state_ood_density_error']}")

    # ---------------------------------------------------------
    # Human action
    # ---------------------------------------------------------

    user_action = None

    if not not_blend:

        if teleop_interface is None:
            raise RuntimeError(
                "not_blend=False requires teleop_interface."
            )

        user_action = teleop_interface.advance()
        print(f"[SpaceMouse] raw 7-DoF action: {user_action.tolist()}")

        # TODO: scoring the live human action's OOD loss/density against the
        # server-owned model isn't wired up yet - would need a dedicated server "cmd"
        # (e.g. "score_action") since the model/preprocessor only live in that process.

    return policy_actions, user_action


def make_lerobot_obs(obs_dict, task_instruction):
    """
    Convert the current Isaac Lab observation into the exact
    observation format used by the LeRobot MultiTaskDiT dataset.

    Expected Isaac Lab fields:
        obs_dict["joint_pos"]   -> [9]
        obs_dict["table_cam"]   -> [3, H, W]
        obs_dict["wrist_cam"]   -> [3, H, W]

    Args:
        obs_dict: Isaac Lab's observation dict for this step.
        task_instruction: Language task label the checkpoint was trained on (e.g. "stack
            cubes" - see the training dataset's meta/tasks.parquet). MultiTaskDiT's LeRobot
            preprocessor requires this "task" key to tokenize task-conditioning.
    """

    obs = obs_dict["policy"] if "policy" in obs_dict else obs_dict

    state = obs["joint_pos"]

    table_cam = obs["table_cam"]
    wrist_cam = obs["wrist_cam"]

    # Remove batch dimension if Isaac Lab gives [1, ...]
    if state.ndim > 1 and state.shape[0] == 1:
        state = state.squeeze(0)

    if table_cam.ndim == 4 and table_cam.shape[0] == 1:
        table_cam = table_cam.squeeze(0)

    if wrist_cam.ndim == 4 and wrist_cam.shape[0] == 1:
        wrist_cam = wrist_cam.squeeze(0)

    # Isaac Lab's Camera sensor returns "rgb" as (H, W, 3) uint8 (see
    # isaaclab.sensors.camera.Camera._process_annotator_output), but the LeRobot dataset
    # this checkpoint was trained on stores images as (3, H, W) float32 in [0, 1] - sending
    # the raw uint8 tensor through as-is both scores the model on the wrong pixel format and
    # (via NormalizerProcessorStep's dtype-matching in normalize_processor.py) corrupts the
    # float normalization stats by re-casting them to uint8, which is what raised "value
    # cannot be converted to type uint8 without overflow" server-side.
    def _to_chw_float(img):
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if img.shape[-1] == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)
        return img.contiguous()

    table_cam = _to_chw_float(table_cam)
    wrist_cam = _to_chw_float(wrist_cam)

    return {
        "observation.state": state,
        "observation.images.table_cam": table_cam,
        "observation.images.wrist_cam": wrist_cam,
        "task": task_instruction,
    }


def main():
    """Run a trained fm policy - GLOVES DiTPolicy or LeRobot MultiTaskDiTPolicy, chosen via
    --fm_backbone - in an Isaac Lab environment."""

    import isaaclab_mimic.envs

    print("Mimic envs:")
    for name in gym.registry.keys():
        if "Mimic" in name:
            print(name)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = torch.device(args_cli.device)

    not_blend = True if not args_cli.blend else False

    # Wire up an input device - only needed for action blending (--blend), which reads the
    # live human action via teleop_interface.advance(). Skipped entirely when not_blend=True
    # (the default). When blending IS requested, try the local HID SpaceMouse first (the
    # normal case when this script runs on the same machine the device is plugged into); if
    # none is found (e.g. this script is running on a remote SSH workstation and the
    # SpaceMouse is on your local client machine instead), fall back to spacemouse_bridge.py
    # over the network - see that file's module docstring for the one-time SSH -R setup.
    teleop_interface = None
    if not not_blend:
        cfg = Se3SpaceMouseCfg(sim_device=device)
        try:
            teleop_interface = Se3SpaceMouse(cfg)
        except OSError:
            print(
                "[INFO] No local SpaceMouse found - falling back to the network bridge at "
                f"{args_cli.spacemouse_bridge_host}:{args_cli.spacemouse_bridge_port}. Run "
                "spacemouse_bridge.py on your client machine and forward the port with "
                "`ssh -R <port>:localhost:<port>` if you haven't already - see "
                "spacemouse_bridge.py's module docstring."
            )
            teleop_interface = NetworkSe3SpaceMouse(
                cfg,
                host=args_cli.spacemouse_bridge_host,
                port=args_cli.spacemouse_bridge_port,
                authkey=args_cli.spacemouse_bridge_authkey,
            )
        print(teleop_interface)

    # Run policy on live actions input from Isaac Lab
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")

        if args_cli.fm_backbone == "gloves":
            # gloves_server.py (mirroring multitask_dit_server.py's client/server split)
            # doesn't exist yet - DiTPolicy can't be imported here (yuna_env has no
            # lerobot_policy_gloves installed), so this backbone isn't runnable yet.
            raise NotImplementedError(
                "gloves backbone requires gloves_server.py (not yet written) - see "
                "multitask_dit's client/server split (multitask_dit_server.py + "
                "run_multitask_ditpolicy) for the pattern to follow."
            )

        elif args_cli.fm_backbone == "multitask_dit":
            conn = Client(
                (args_cli.mdit_server_host, args_cli.mdit_server_port),
                authkey=args_cli.mdit_server_authkey.encode(),
            )
            conn.send({"cmd": "reset"})
            reset_response = conn.recv()
            if not reset_response.get("ok", False):
                raise RuntimeError(f"multitask_dit_server reset failed: {reset_response.get('error')}")

            obs_dict, _ = env.reset()
            obs = make_lerobot_obs(obs_dict, args_cli.task_instruction)

            policy_actions, user_action = run_multitask_ditpolicy(
                conn=conn,
                obs=obs,
                teleop_interface=teleop_interface,
                not_blend=not_blend,
            )

            conn.send({"cmd": "close"})
            conn.recv()
            conn.close()
            # TODO: env.step through policy_actions/blend with user_action, mirroring
            # run_gloves_policy's not_blend=True branch - not wired up yet.
            terminated = False
            traj = dict(policy_actions=[policy_actions.tolist()], blended_actions=[], obs=[], next_obs=[])

        else:
            raise ValueError(f"Unknown --fm_backbone {args_cli.fm_backbone!r}")

        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
