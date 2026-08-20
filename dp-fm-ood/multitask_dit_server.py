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

NORMALIZATION (fixed): ood_signal.py's multitask_dit_loss/multitask_dit_density are scored
against obs_history/raw_action below, which are the SAME normalized, preprocessed
representation `batch = preprocessor(obs)` already produces for select_action - not raw
`obs` and not the postprocessor's unnormalized action. See the step handler below for how
obs_history/raw_action are built (reusing `batch`, no second preprocessor(obs) call).

TASK/LANGUAGE CONDITIONING (fixed): obs_history also carries OBS_LANGUAGE_TOKENS/
OBS_LANGUAGE_ATTENTION_MASK, pulled from that same `batch` (this checkpoint's
ObservationEncoder always expects them - text_encoder_name is set, so conditioning_dim
includes text_dim unconditionally, see modeling_multi_task_dit.py - without them the
conditioning vector comes out text_dim short and size-mismatches downstream). This
checkpoint was trained on a single constant task string ("stack cubes" - see
lerobot_dataset_0810/ID-visuomotor-based/meta/tasks.parquet), matching run_policy_fm.py's
--task_instruction default, so whatever obs["task"] the client sends is already the right
string to tokenize.
"""

import argparse
from collections import deque
from multiprocessing.connection import Listener

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

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
        "--num_samples", type=int, default=32,
        help="Nb for the diffdagger-style loss computed alongside each action (see "
        "ood_signal.multitask_dit_loss).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = MultiTaskDiTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    # select_action() only regenerates a fresh chunk once every n_action_steps calls
    # (config.json: horizon=32, n_action_steps=24, num_integration_steps=100 Euler steps -
    # a full generation is expensive), reusing the queued remainder otherwise. Calling
    # generate_action_chunk() separately every step for OOD scoring would redo that same
    # 100-step Euler integration on every single step regardless of whether select_action
    # needed one - ~24x the intended generation cost, which is enough to OOM the GPU over
    # a long rollout. Monkey-patching conditional_sample instead captures the exact full
    # [1, horizon, action_dim] chunk select_action() already produces internally, for free -
    # same cadence, zero extra generation calls, and it's still the real chunk (never
    # tiled/fabricated) the currently-queued/executing action actually came from.
    last_chunk = {"value": None}
    _orig_conditional_sample = policy.objective.conditional_sample

    def _conditional_sample_and_capture(*args, **kwargs):
        result = _orig_conditional_sample(*args, **kwargs)
        last_chunk["value"] = result.detach()
        return result

    policy.objective.conditional_sample = _conditional_sample_and_capture

    # Official LeRobot inference pattern - see this file's module docstring. dataset_stats
    # omitted: relies on normalization stats bundled with the checkpoint at pretrained_path
    # (the standard from_pretrained layout). If your checkpoint doesn't bundle stats, pass a
    # real dataset's `.meta.stats` here instead.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint
    )

    # A second, separate observation window - NOT policy._queues (which select_action owns
    # internally) - kept purely so multitask_dit_loss/multitask_dit_density (ood_signal.py)
    # have something to score against, mirroring the same n_obs_steps history select_action
    # is conditioning on. Holds the same normalized, preprocessed per-step tensors `batch`
    # already has (batch dim stripped) - built once per step below, no second
    # preprocessor(obs) call. See this file's module docstring for the still-open
    # task/language conditioning gap on those two.
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
                        print(f"[multitask_dit_server] obs: state={obs['observation.state'].tolist()} task={obs['task']!r}")

                        with torch.no_grad():
                            batch = preprocessor(obs)

                            # obs_history stores the SAME normalized, per-step tensors
                            # `batch` holds (batch dim stripped) - reusing `batch` rather
                            # than re-deriving anything, so this is not a second
                            # preprocessor(obs) call.
                            norm_obs_step = {
                                key: batch[key].squeeze(0)
                                for key in (
                                    "observation.state",
                                    "observation.images.table_cam",
                                    "observation.images.wrist_cam",
                                )
                            }
                            # Task/language conditioning: one tokenized task string per
                            # step call, constant across the whole n_obs_steps window (not
                            # per-timestep like state/images) - kept with its batch dim
                            # intact since ObservationEncoder.encode() expects [B, seq_len].
                            norm_obs_step[OBS_LANGUAGE_TOKENS] = batch[OBS_LANGUAGE_TOKENS]
                            norm_obs_step[OBS_LANGUAGE_ATTENTION_MASK] = batch[OBS_LANGUAGE_ATTENTION_MASK]
                            obs_history.append(norm_obs_step)
                            while len(obs_history) < policy.config.n_obs_steps:
                                obs_history.append(norm_obs_step)

                            # raw_action is select_action's output BEFORE postprocessor
                            # unnormalizes it - the same normalized space
                            # multitask_dit_loss/multitask_dit_density need (and the space
                            # the model's forward/noise_predictor were trained on). `action`
                            # is the physical-unit version sent back to the client below.
                            raw_action = policy.select_action(batch)
                            action = postprocessor(raw_action)
                        action = action.cpu()
                        print(
                            f"[multitask_dit_server] action: normalized={raw_action.tolist()} "
                            f"physical={action.tolist()}"
                        )

                        # Sent as a plain list, not a torch.Tensor: pickling a raw tensor
                        # through this Connection routes through torch's multiprocessing
                        # shared-memory reducer (resource_sharer), which authenticates with
                        # this process's own default multiprocessing authkey - not the
                        # --authkey this Listener uses. Since the client is a separate
                        # interpreter with a different default authkey, that side-channel
                        # handshake fails with AuthenticationError even though this response
                        # itself sends fine. Plain data avoids the reducer entirely.
                        response = {"ok": True, "action": action.tolist()}

                        try:
                            state_ood_loss = multitask_dit_loss(
                                policy, obs_history, raw_action, num_samples=args.num_samples
                            )
                            response["state_ood_loss"] = state_ood_loss.item()
                            print(f"[multitask_dit_server] state_ood_loss={response['state_ood_loss']:.6f}")
                        except Exception as e:
                            response["state_ood_loss_error"] = str(e)
                            print(f"[multitask_dit_server] state_ood_loss FAILED: {e}")

                        try:
                            # Real full flow-generated chunk [1, horizon, action_dim] -
                            # captured for free via the conditional_sample monkey-patch
                            # above (no tiling/fabrication, no redundant generation call).
                            # select_action() always populates this at least once before
                            # this point is reached (first call after reset(), when its
                            # action queue is empty), so this is never None here.
                            real_chunk = last_chunk["value"]

                            state_ood_score, z_hat = multitask_dit_density(
                                policy, obs_history, real_chunk, tile_single_action=False, return_z_hat=True
                            )
                            # response key name is unchanged (state_ood_density, matching the
                            # paper's own "density"-inspired naming/existing IPC contract) -
                            # only the printed label below is fixed: what this actually is is
                            # the non-conformity score s(x) = ||z_hat||^2 (paper Eq. 6), not a
                            # probability density itself.
                            response["state_ood_density"] = state_ood_score.item()
                            z_flat = z_hat.flatten(start_dim=1)
                            d = z_flat.shape[1]  # horizon * action_dim
                            print(
                                f"[multitask_dit_server] nonconformity_score s(x)="
                                f"{response['state_ood_density']:.6f} "
                                # f"(in-distribution reference: chi-sq({d}) mean={d}, std={(2 * d) ** 0.5:.2f})"
                            )
                            print(
                                f"[multitask_dit_server] z_hat: mean={z_flat.mean().item():.4f} std={z_flat.std().item():.4f} "
                                f"min={z_flat.min().item():.4f} max={z_flat.max().item():.4f} "
                            #     f"(in-distribution reference: mean~0, std~1 per element)"
                            # )
                        except Exception as e:
                            response["state_ood_density_error"] = str(e)
                            print(f"[multitask_dit_server] nonconformity_score FAILED: {e}")

                        conn.send(response)

                        # multitask_dit_loss batches num_samples copies of both camera
                        # images through the model in one forward pass - PyTorch's caching
                        # allocator keeps that peak-sized block reserved for reuse rather
                        # than returning it to the OS, so without this the process's VRAM
                        # footprint ratchets up to (and stays at) that peak instead of
                        # settling back down between steps, starving other GPU users (e.g.
                        # the Isaac Lab viewport) even at idle.
                        torch.cuda.empty_cache()

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
