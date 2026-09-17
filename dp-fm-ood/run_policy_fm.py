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

CURRENT MODE: the executed action is the SpaceMouse, not the policy. The run_multitask_ditpolicy() call
is commented out in main(); run_spacemouse_teleop() steps the env with the SpaceMouse command and sends
each (obs, command) to multitask_dit_server.py's "score_action", which scores it with the same
multitask_dit_loss as state_ood_loss and, when started with --ood_csv, writes the row (loss in the
state_ood_loss column, normalized command in raw_action_*, physical command in physical_action_*).
"""

# TODO - wire in conformal prediction and smoothness code in

import argparse
import csv
import os
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
    "--task_instruction", type=str, default="grab red block and stack on top of blue block, then grab green block and stack on top of red block",
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
parser.add_argument(
    "--spacemouse", type=str, default="auto",
    help="Which SpaceMouse drives the robot: a local HID path like /dev/hidraw5 (find yours with "
    "spacemouse_identify.py), 'bridge' for spacemouse_bridge.py over the network, or 'auto' (the one "
    "local SpaceMouse if exactly one is connected, the bridge if none, and an error listing them if "
    "several are - Isaac Lab's own lookup silently opens an arbitrary one of identical devices).",
)

parser.add_argument(
    "--state_mode", type=str, choices=["auto", "joint_pos", "eef_pose"], default="auto",
    help="Which observation.state layout to build: 'joint_pos' (9 dims, pre-0908 "
    "checkpoints) or 'eef_pose' (8 dims, 0908-onward EE-pose checkpoints). 'auto' (default) "
    "takes it from the dim the server reports for its loaded checkpoint, falling back to "
    "--fm_checkpoint's own config.json. See lerobot_obs.py.",
)

parser.add_argument("--horizon", type=int, default=500, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts to run.")  # default=10
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--no_live_plot", action="store_true",
    help="SpaceMouse mode: don't open the live loss window (live_loss_plot.py).",
)
parser.add_argument(
    "--live_plot_reference_csv", type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "state_id_loss_0915.csv"),
    help="Policy CSV whose per-step state_ood_loss is drawn as the reference line in the live window.",
)
parser.add_argument(
    "--live_plot_reference_seed", type=int, default=None,
    help="Seed of the reference line (default: --seed, i.e. the same initial scene).",
)
parser.add_argument(
    "--score_density", action="store_true",
    help="SpaceMouse mode: also have the server compute the density score s(x) for each command "
    "(slower per step; the loss is always computed).",
)

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

# Imported after the AppLauncher block, with the other post-launch imports, since it pulls in
# torch. Shared with replay_training_rollouts.py / replay_dataset_with_scoring.py so the
# observation.state layout is defined in exactly one place - see lerobot_obs.py.
from lerobot_obs import make_lerobot_obs, resolve_state_mode
from live_loss_plot import LiveLossPlot


def list_local_spacemice():
    """Local SpaceMouse HID paths (one per physical device), e.g. [b'/dev/hidraw6', b'/dev/hidraw0']."""
    import hid

    names = ("SpaceMouse Compact", "SpaceMouse Wireless", "3Dconnexion Universal Receiver")
    paths = []
    for d in hid.enumerate():
        if d["product_string"] in names and d["path"] not in paths:
            paths.append(d["path"])
    return paths


class PathSe3SpaceMouse(Se3SpaceMouse):
    """Se3SpaceMouse bound to ONE HID path. Isaac Lab's Se3SpaceMouse._find_device opens by vendor/product
    id, and several SpaceMouse Compacts share both (and report no serial), so with more than one plugged
    in it opens whichever hidapi lists first - often not the one being moved, which reads all zeros."""

    def __init__(self, cfg, path):
        self._hid_path = path.encode() if isinstance(path, str) else path
        super().__init__(cfg)  # calls the _find_device below

    def _find_device(self):
        import hid

        match = [d for d in hid.enumerate() if d["path"] == self._hid_path]
        if not match:
            raise OSError(f"No SpaceMouse at {self._hid_path.decode()}; connected: "
                          f"{[p.decode() for p in list_local_spacemice()]}")
        self._device.open_path(self._hid_path)
        self._device_name = match[0]["product_string"]

    def __str__(self) -> str:
        return super().__str__().replace("\n", f" [{self._hid_path.decode()}]\n", 1)


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
    env,
    success_term,
    horizon,
    device,
    task_instruction,
    teleop_interface=None,
    state_mode="auto",
    checkpoint=None,
    seed=None,
    trial=None,
):
    """
    Run a full rollout (up to `horizon` steps) of the trained LeRobot MultiTaskDiTPolicy via
    multitask_dit_server.py, mirroring run_dp_policy's per-step loop/success/termination
    structure (run_policy_dp.py) so both backbones share the same rollout semantics.

    The server owns the model, the LeRobot preprocessor/postprocessor, and the
    n_obs_steps window (see multitask_dit_server.py's module docstring -
    `select_action` takes one raw timestep per call and pads/queues internally, and
    returns ONE action per call, not a pre-chunked horizon). So each loop iteration here
    is exactly one obs -> server round trip -> one env.step() - no chunk-splitting needed.

    Every step always executes the policy's own action directly (no blending math is
    implemented yet - see run_dp_policy/run_gloves_policy's identical gap). When
    `teleop_interface` is given (--blend), the live SpaceMouse action is also read and
    printed every step alongside it, purely for visibility/logging - unrestricted, not
    capped to one read like the earlier single-step scaffold, and not blocking the rollout.

    Args:
        conn: An open multiprocessing.connection.Client connection to
            multitask_dit_server.py.
        env: The environment to roll out in.
        success_term: The extracted success-termination term (env_cfg.terminations.success),
            called each step to check whether the rollout succeeded.
        horizon: The step horizon of this rollout (env_cfg.terminations.time_out is
            disabled in main(), so this manual Python loop is the only real step cap).
        device: The device to run the policy/step actions on.
        task_instruction: Language task label passed to make_lerobot_obs (see
            --task_instruction).
        teleop_interface: A device (e.g. Se3SpaceMouse) to read the live human action
            from, each step, purely for logging. Optional.
        state_mode: Requested observation.state layout ("auto"/"joint_pos"/"eef_pose", see
            --state_mode). Resolved against the dim the server reports below.
        checkpoint: --fm_checkpoint, used only as the fallback source for the state layout
            when an older server doesn't report its dim.
        seed: --seed, forwarded to the server on reset so its --ood_csv rows record it.
        trial: Trial index within this run, forwarded alongside seed.

    Returns:
        terminated: True if success_term fired (success), False otherwise (env
            termination/truncation, or ran out of horizon).
        traj: The trajectory of the rollout.
    """

    conn.send({"cmd": "reset", "seed": seed, "trial": trial})
    reset_response = conn.recv()
    if not reset_response.get("ok", False):
        raise RuntimeError(f"multitask_dit_server reset failed: {reset_response.get('error')}")

    # Resolve the state layout from the SERVER's loaded checkpoint, not from this script's
    # idea of it - the server is the only side that has actually read the model. A pre-0908
    # (9-dim joint_pos) and a 0908-onward (8-dim eef_pose) checkpoint are both valid targets
    # and are switched between purely by which one the server was started with.
    resolved_state_mode, why = resolve_state_mode(
        requested=state_mode,
        state_dim=reset_response.get("state_dim"),
        checkpoint=checkpoint,
    )
    print(f"[FM client] observation.state layout: {resolved_state_mode} ({why})")

    obs_dict, _ = env.reset()
    if teleop_interface is not None:
        teleop_interface.reset()

    traj = dict(policy_actions=[], blended_actions=[], obs=[], next_obs=[])

    for i in range(horizon):
        obs = make_lerobot_obs(obs_dict, task_instruction, resolved_state_mode)
        print("obs, ", obs)
        traj["obs"].append(obs)

        conn.send({"cmd": "step", "obs": obs})
        response = conn.recv()
        if not response.get("ok", False):
            raise RuntimeError(f"multitask_dit_server error: {response.get('error')}")

        # Plain nested list, not a torch.Tensor - see multitask_dit_server.py's comment on
        # why (avoids the cross-process multiprocessing shared-memory reducer/authkey
        # mismatch).
        policy_actions = torch.tensor(response["action"], device=device)

        if i == 0:
            print("[FM client] generated action chunk:", tuple(policy_actions.shape))

        if "state_ood_loss" in response:
            print(f"[state OOD] MultiTaskDiT loss for model's own action: {response['state_ood_loss']:.4f}")
        elif "state_ood_loss_error" in response:
            print(f"[state OOD] MultiTaskDiT loss errored server-side: {response['state_ood_loss_error']}")

        if "state_ood_density" in response:
            print(f"[state OOD] MultiTaskDiT non-conformity score s(x)=||z_hat||^2 for model's own action: {response['state_ood_density']:.4f}")
        elif "state_ood_density_error" in response:
            print(f"[state OOD] MultiTaskDiT non-conformity score errored server-side: {response['state_ood_density_error']}")

        # Purely for visibility/logging - not blocking, not capped, and not blended into
        # the executed action (no blend rule is implemented yet - see TODO in
        # run_dp_policy/run_gloves_policy's identical not_blend=False gap).
        if teleop_interface is not None:
            user_action = teleop_interface.advance()
            print(f"[SpaceMouse] raw 7-DoF action: {user_action.tolist()}")

        obs_dict, _, terminated, truncated, _ = env.step(policy_actions)

        traj["policy_actions"].append(policy_actions.tolist())
        traj["next_obs"].append(obs_dict["policy"])

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def run_spacemouse_teleop(
    conn,
    env,
    success_term,
    horizon,
    task_instruction,
    teleop_interface,
    state_mode="auto",
    checkpoint=None,
    seed=None,
    trial=None,
    score_density=False,
    live_plot=None,
):
    """
    Same rollout loop as run_multitask_ditpolicy, but the executed action is the live SpaceMouse
    command. The policy never drives: each step sends (obs, command) to the server's "score_action",
    which returns (and, with --ood_csv, logs) the loss of that command under the model.

    Args:
        conn: open multiprocessing.connection.Client to multitask_dit_server.py.
        env / success_term / horizon / task_instruction: as in run_multitask_ditpolicy.
        teleop_interface: Se3SpaceMouse or NetworkSe3SpaceMouse - its 7-D [dx, dy, dz, rx, ry, rz,
            gripper] command goes to env.step directly (the IK-Rel action space).
        state_mode / checkpoint: observation.state layout, resolved against the server's dim.
        seed / trial: sent on reset so the server's CSV rows record them.
        score_density: also compute the density score each step (slower).
        live_plot: optional LiveLossPlot; receives each step's loss.

    Returns:
        terminated: True if success_term fired, False otherwise.
        traj: The trajectory of the rollout.
    """
    conn.send({"cmd": "reset", "seed": seed, "trial": trial})
    reset_response = conn.recv()
    if not reset_response.get("ok", False):
        raise RuntimeError(f"multitask_dit_server reset failed: {reset_response.get('error')}")

    resolved_state_mode, why = resolve_state_mode(
        requested=state_mode, state_dim=reset_response.get("state_dim"), checkpoint=checkpoint,
    )
    print(f"[teleop] observation.state layout: {resolved_state_mode} ({why})")

    obs_dict, _ = env.reset()
    teleop_interface.reset()
    if live_plot is not None:
        live_plot.set_csv(reset_response.get("ood_csv"))
        live_plot.reset(seed, trial)

    # The only randomness in this task's reset is the cube poses (python `random`) and a small
    # gaussian offset on the Franka joints (torch) - print both so runs with the same seed can be
    # compared by eye.
    policy_obs = obs_dict["policy"]
    print(f"[teleop] seed={seed} trial={trial} cube_positions={policy_obs['cube_positions'][0].cpu().numpy().round(4).tolist()}")
    print(f"[teleop] seed={seed} trial={trial} joint_pos_rel={policy_obs['joint_pos'][0].cpu().numpy().round(4).tolist()}")

    traj = dict(policy_actions=[], blended_actions=[], obs=[], next_obs=[])

    for i in range(horizon):
        obs = make_lerobot_obs(obs_dict, task_instruction, resolved_state_mode)
        traj["obs"].append(obs)

        # [7] delta-pose + gripper command, already on the sim device
        action = teleop_interface.advance().reshape(1, -1)
        print(f"[SpaceMouse] raw 7-DoF action: {action.reshape(-1).tolist()}")

        conn.send({"cmd": "score_action", "obs": obs, "action": action.reshape(-1).tolist(), "density": score_density})
        try:
            response = conn.recv()
        except KeyboardInterrupt:
            # Ctrl+C while waiting on the server: it writes this step's CSV row before replying, so wait
            # briefly for the reply and plot it too - the saved plot then ends on the same step as the CSV.
            try:
                if conn.poll(5):
                    late = conn.recv()
                    if live_plot is not None and "action_loss" in late:
                        live_plot.push(i, late["action_loss"])
            except (EOFError, OSError, KeyboardInterrupt):
                pass
            raise
        if not response.get("ok", False):
            raise RuntimeError(f"multitask_dit_server error: {response.get('error')}")
        if "action_loss" in response:
            print(f"[teleop step {i}] SpaceMouse action loss: {response['action_loss']:.4f}")
            if live_plot is not None:
                live_plot.push(i, response["action_loss"])
        elif "action_loss_error" in response:
            print(f"[teleop step {i}] SpaceMouse action loss errored server-side: {response['action_loss_error']}")

        obs_dict, _, terminated, truncated, _ = env.step(action)

        traj["blended_actions"].append(action.tolist())
        traj["next_obs"].append(obs_dict["policy"])

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


# make_lerobot_obs / eef_state / log_raw_state now live in lerobot_obs.py, imported above.
# They were moved out because replay_training_rollouts.py and replay_dataset_with_scoring.py
# each carried their own copy (they cannot import this module - the AppLauncher at its top
# level would re-run), so the joint_pos -> eef_pose state-layout fix had to land three times.


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
    # SpaceMouse-driven rollouts need the device regardless of --blend (was: only created when
    # blending was requested).
    teleop_interface = None
    # if not not_blend:
    if True:
        cfg = Se3SpaceMouseCfg(sim_device=device)
        # --spacemouse picks the device explicitly. Several identical SpaceMouse Compacts are plugged
        # into this workstation, and Isaac Lab's Se3SpaceMouse(cfg) opens an arbitrary one of them
        # (and never falls back to the bridge while any local one exists) - see PathSe3SpaceMouse.
        local = [p.decode() for p in list_local_spacemice()]
        choice = args_cli.spacemouse
        if choice == "auto":
            if len(local) > 1:
                raise RuntimeError(
                    f"{len(local)} local SpaceMice connected ({', '.join(local)}) - pick the one you are "
                    "holding with --spacemouse /dev/hidrawN (run `python spacemouse_identify.py` and move "
                    "it to see which), or --spacemouse bridge to use spacemouse_bridge.py."
                )
            choice = local[0] if local else "bridge"
        if choice == "bridge":
            print(
                "[INFO] Using the SpaceMouse network bridge at "
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
        else:
            print(f"[INFO] Using local SpaceMouse {choice} (connected: {', '.join(local) or 'none'})")
            teleop_interface = PathSe3SpaceMouse(cfg, choice)
        print(teleop_interface)

    if args_cli.fm_backbone == "gloves":
        # gloves_server.py (mirroring multitask_dit_server.py's client/server split)
        # doesn't exist yet - DiTPolicy can't be imported here (yuna_env has no
        # lerobot_policy_gloves installed), so this backbone isn't runnable yet.
        raise NotImplementedError(
            "gloves backbone requires gloves_server.py (not yet written) - see "
            "multitask_dit's client/server split (multitask_dit_server.py + "
            "run_multitask_ditpolicy) for the pattern to follow."
        )
    elif args_cli.fm_backbone != "multitask_dit":
        raise ValueError(f"Unknown --fm_backbone {args_cli.fm_backbone!r}")

    # One persistent connection, reused for every trial (not reopened per trial) - each
    # trial's own {"cmd": "reset"} resets the server-side policy/obs_history for that episode.
    # Still needed in SpaceMouse mode: the server scores every command.
    conn = Client(
        (args_cli.mdit_server_host, args_cli.mdit_server_port),
        authkey=args_cli.mdit_server_authkey.encode(),
    )

    live_plot = None
    if not args_cli.no_live_plot:
        ref_seed = args_cli.seed if args_cli.live_plot_reference_seed is None else args_cli.live_plot_reference_seed
        live_plot = LiveLossPlot(args_cli.live_plot_reference_csv, ref_seed)

    results = []
    interrupted = False
    try:
        for trial in range(args_cli.num_rollouts):
            print(f"[INFO] Starting trial {trial}")

            # POLICY-DRIVEN ROLLOUT (commented out - the SpaceMouse drives below, the server only scores).
            # terminated, traj = run_multitask_ditpolicy(
            #     conn=conn,
            #     env=env,
            #     success_term=success_term,
            #     horizon=args_cli.horizon,
            #     device=device,
            #     task_instruction=args_cli.task_instruction,
            #     teleop_interface=teleop_interface,
            #     state_mode=args_cli.state_mode,
            #     checkpoint=args_cli.fm_checkpoint,
            #     seed=args_cli.seed,
            #     trial=trial,
            # )
            terminated, traj = run_spacemouse_teleop(
                conn=conn,
                env=env,
                success_term=success_term,
                horizon=args_cli.horizon,
                task_instruction=args_cli.task_instruction,
                teleop_interface=teleop_interface,
                state_mode=args_cli.state_mode,
                checkpoint=args_cli.fm_checkpoint,
                seed=args_cli.seed,
                trial=trial,
                score_density=args_cli.score_density,
                live_plot=live_plot,
            )

            results.append(terminated)
            print(f"[INFO] Trial {trial}: {terminated}\n")
    except KeyboardInterrupt:
        # Ctrl+C: every step scored so far is already in the server's --ood_csv; the live plot is saved
        # up to the same step when the viewer sees this (see live_loss_plot.py).
        interrupted = True
        print("\n[INFO] Interrupted - CSV rows so far are kept by the server; saving the live plot.")
        if live_plot is not None:
            live_plot.interrupted()
    finally:
        if live_plot is not None:
            live_plot.close()

    try:
        conn.send({"cmd": "close"})
        if conn.poll(5):
            conn.recv()
        conn.close()
    except (OSError, EOFError):
        pass

    if results:
        print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
        print(f"Success rate: {results.count(True) / len(results)}")
        print(f"Trial Results: {results}\n")
    if interrupted:
        print("[INFO] Run was interrupted before all trials finished.")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
