#!/usr/bin/env python3
"""Standalone server that owns a LeRobot MultiTaskDiTPolicy and serves it over
multiprocessing.connection, so run_policy_fm.py (Isaac Lab, Python 3.11) never has to
import lerobot's MultiTaskDiT code - which needs Python 3.12+ and therefore can't live in
the same process as Isaac Lab/Isaac Sim.

Architecture: two separate OS processes, each with its own conda env, talking over
127.0.0.1 - Isaac Lab owns the simulation/env.step(), this process owns the policy/GPU
inference. Neither process imports the other's heavy dependencies.

    Terminal 1 (Isaac Lab, Python 3.11):
        conda activate lerobot_isaaclab
        python run_policy_fm.py --fm_backbone multitask_dit --fm_checkpoint ... \
            --mdit_server_host 127.0.0.1 --mdit_server_port 5555

    Terminal 2 (LeRobot, Python 3.12):
        conda activate lerobot
        python multitask_dit_server.py --checkpoint /path/to/pretrained_model --port 5555

IPC: multiprocessing.connection.Listener/Client (stdlib) - pickle-serialized objects over a
plain socket. Same transport/serialization a raw socket + pickle would use, just without
hand-writing the length-prefix framing. --authkey must match run_policy_fm.py's
--mdit_server_authkey (multiprocessing.connection.Listener performs an HMAC handshake using
this shared key - see docs.python.org/3/library/multiprocessing.html#multiprocessing-listeners-and-clients).
Requests: {"cmd": "reset"} | {"cmd": "step", "obs": {...}} | {"cmd": "close"}.
Responses: {"ok": True, ...} | {"ok": False, "error": "..."}.

Inference pattern verified against huggingface/lerobot's actual source (not guessed):
  - MultiTaskDiTPolicy.select_action(batch) (modeling_multi_task_dit.py) expects ONE
    timestep's worth of observation per call - NOT a pre-stacked n_obs_steps window. It
    internally calls populate_queues(self._queues, batch) (policies/utils.py), which
    appends batch[key] to a per-key deque (maxlen=n_obs_steps for observation keys,
    maxlen=n_action_steps for "action") that reset() already pre-creates; on the first call
    after reset() the deque is empty, so populate_queues' while-loop repeats that single
    first observation until the deque is full - an implicit "pad the start of the episode",
    the same pattern this codebase's own run_gloves_policy/run_dp_policy build by hand.
    So: call policy.reset() once per episode, then policy.select_action(batch) once per
    step with just that step's observation - the n_obs_steps window is handled internally.
  - The proper way to build `batch` for select_action is NOT to hand-construct normalized
    tensors ourselves - it's LeRobot's own processor pipeline (verbatim pattern from
    https://huggingface.co/docs/lerobot/en/introduction_processors):
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=checkpoint, dataset_stats=...
        )
        batch = preprocessor(raw_obs)          # rename -> add batch dim -> tokenize task
                                                # -> device -> normalize
        action = policy.select_action(batch)
        action = postprocessor(action)          # unnormalize -> cpu

KNOWN GAP (flagged, not fixed here): ood_signal.py's multitask_dit_loss/multitask_dit_density
(called below, per-step) build their own batch by hand rather than going through
`preprocessor` above, so unlike the action itself, those two numbers are NOT normalized the
way the model was trained on - and for the image-conditioned case specifically, that's
likely not just miscalibrated but meaningless (the vision encoder never saw un-normalized
pixels during training). Treat state_ood_loss/state_ood_density as a rough/relative signal
until that's addressed - see this file's TODO markers on the two try/except blocks below.
"""

import argparse
from collections import deque
from multiprocessing.connection import Listener

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy

from ood_signal import multitask_dit_density, multitask_dit_loss

DEFAULT_AUTHKEY = "mdit-ipc"


def main():
    parser = argparse.ArgumentParser(
        description="Serve a trained MultiTaskDiTPolicy over multiprocessing.connection, so "
        "the Isaac Lab process (run_policy_fm.py, Python 3.11) never needs to import "
        "lerobot's MultiTaskDiT code (Python 3.12+)."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path or hub id of a trained MultiTaskDiTPolicy checkpoint.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--authkey", type=str, default=DEFAULT_AUTHKEY,
        help="Shared secret for multiprocessing.connection's authentication handshake - "
        "must match run_policy_fm.py's --mdit_server_authkey.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=512,
        help="Nb for the diffdagger-style loss computed alongside each action (see "
        "ood_signal.multitask_dit_loss).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = MultiTaskDiTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    # Official LeRobot inference pattern - see this file's module docstring. dataset_stats
    # omitted: relies on normalization stats bundled with the checkpoint at pretrained_path
    # (the standard from_pretrained layout). If your checkpoint doesn't bundle stats, pass a
    # real dataset's `.meta.stats` here instead.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint
    )

    # A second, separate raw-observation window - NOT policy._queues (which select_action
    # owns internally, and which only ever holds already-preprocessed tensors) - kept
    # purely so multitask_dit_loss/multitask_dit_density (ood_signal.py) have something to
    # score against, mirroring the same n_obs_steps history select_action is conditioning
    # on. See this file's module docstring for the normalization caveat on those two.
    obs_history = deque(maxlen=policy.config.n_obs_steps)

    listener = Listener((args.host, args.port), authkey=args.authkey.encode())
    print(f"[multitask_dit_server] listening on {args.host}:{args.port}, checkpoint={args.checkpoint}, device={device}")

    try:
        while True:
            conn = listener.accept()
            print(f"[multitask_dit_server] client connected from {listener.last_accepted}")
            try:
                while True:
                    try:
                        request = conn.recv()
                    except EOFError:
                        break

                    cmd = request.get("cmd")
                    if cmd == "reset":
                        policy.reset()
                        obs_history.clear()
                        conn.send({"ok": True})

                    elif cmd == "step":
                        obs = request["obs"]
                        obs_history.append(obs)
                        while len(obs_history) < policy.config.n_obs_steps:
                            obs_history.append(obs)

                        with torch.no_grad():
                            batch = preprocessor(obs)
                            action = policy.select_action(batch)
                            action = postprocessor(action)
                        action = action.cpu()

                        # Sent as a plain list, not a torch.Tensor: pickling a raw tensor
                        # through this Connection routes through torch's multiprocessing
                        # shared-memory reducer (resource_sharer), which authenticates with
                        # this process's own default multiprocessing authkey - not the
                        # --authkey this Listener uses. Since the client is a separate
                        # interpreter with a different default authkey, that side-channel
                        # handshake fails with AuthenticationError even though this response
                        # itself sends fine. Plain data avoids the reducer entirely.
                        response = {"ok": True, "action": action.tolist()}

                        # TODO (see module docstring KNOWN GAP): bypasses `preprocessor`'s
                        # normalization - relative/trend signal only until fixed.
                        try:
                            state_ood_loss = multitask_dit_loss(
                                policy, obs_history, action, num_samples=args.num_samples
                            )
                            response["state_ood_loss"] = state_ood_loss.item()
                        except Exception as e:
                            response["state_ood_loss_error"] = str(e)

                        # TODO (see module docstring KNOWN GAP): same normalization caveat.
                        try:
                            state_ood_density = multitask_dit_density(
                                policy, obs_history, action, tile_single_action=True
                            )
                            response["state_ood_density"] = state_ood_density.item()
                        except Exception as e:
                            response["state_ood_density_error"] = str(e)

                        print(
                            f"[multitask_dit_server] action={action.tolist()} "
                            f"loss={response.get('state_ood_loss')} "
                            f"density={response.get('state_ood_density')}"
                        )
                        conn.send(response)

                    elif cmd == "close":
                        conn.send({"ok": True})
                        break

                    else:
                        conn.send({"ok": False, "error": f"unknown cmd {cmd!r}"})
            except Exception as e:
                # A bad request/inference error shouldn't kill the whole server - report it
                # to this client and keep serving.
                print(f"[multitask_dit_server] error: {e}")
                try:
                    conn.send({"ok": False, "error": str(e)})
                except OSError:
                    pass
            finally:
                conn.close()
                print("[multitask_dit_server] client disconnected")
    except KeyboardInterrupt:
        print("\n[multitask_dit_server] shutting down")
    finally:
        listener.close()


if __name__ == "__main__":
    main()
