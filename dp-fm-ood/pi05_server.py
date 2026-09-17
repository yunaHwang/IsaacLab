#!/usr/bin/env python3
"""Standalone server that owns a LeRobot PI05Policy (Pi0.5) and serves it over
multiprocessing.connection, so run_policy_fm_0910_pi.py (Isaac Lab, Python 3.11) never has to
import lerobot 0.6.1 / transformers 5.x - same two-process split as multitask_dit_server.py.

    Terminal 1 (LeRobot, Python 3.12):
        conda activate lerobot_0.6.1_multitask_dit
        python pi05_server.py --port 5556   # default --checkpoint: ./checkpoints/pi05_isaaclab_so101_multitask_1ep

    Terminal 2 (Isaac Lab, Python 3.11):
        conda activate yuna_env
        python run_policy_fm_0910_pi.py --pi_server_port 5556 --enable_cameras

Checkpoint: Cache-SCA/Pi0.5-IsaacLab-Multi-Task-1epochs (mirror of CoRL2026-CSI/...), a full
fine-tune of lerobot/pi05_base on CoRL2026-CSI/Isaaclab-so101_11task_baseCaP_3300epi (SO-101,
11 SCRAPE-IsaacLab tasks). Expected raw observation (what the dataset stored):
    observation.state            [6]          LeRobot normalized motor units (-100..100,
                                               gripper 0..~30), NOT radians
    observation.images.top       [3, 480, 640] float in [0, 1]
    observation.images.left_wrist[3, 480, 640] float in [0, 1]
    task                         str          e.g. "Stack the gray block on the pink block."
The returned action is in the same normalized motor units as observation.state; the client
converts it to sim joint radians (see so101_stack2cubes_env_cfg.py).

The client sends images as HWC uint8 numpy arrays and state as a numpy array (plain pickled
numpy, not torch tensors - see multitask_dit_server.py's note on torch's shared-memory reducer
and the authkey mismatch); the conversion to CHW float tensors happens here.

Requests: {"cmd": "reset"} | {"cmd": "step", "obs": {...}} | {"cmd": "close"}.
Responses: {"ok": True, ...} | {"ok": False, "error": "..."}.

Chunking: PI05Policy.select_action() predicts a chunk_size=16 chunk and pops one action per
call until n_action_steps are used, so each "step" request is exactly one env.step() on the
client. --n_action_steps overrides how much of each chunk is executed before replanning.
OOD scoring (ood_signal.py) is MultiTaskDiT-specific and is not wired in for Pi0.5 yet.
"""

import argparse
from pathlib import Path
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

DEFAULT_AUTHKEY = "pi05-ipc"
IMAGE_KEYS = ("observation.images.top", "observation.images.left_wrist")


def resolve_checkpoint(checkpoint):
    """Return a local checkpoint dir whose config.json only has fields this lerobot's PI05Config
    knows. The Cache-SCA checkpoint was saved by a newer lerobot (e.g. `profile_forward_timing`),
    and draccus rejects unknown fields outright. The filtered copy lives next to the HF snapshot,
    with every other file symlinked, so the 9 GB of weights are not duplicated."""
    import dataclasses
    import json
    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    src = Path(checkpoint) if Path(checkpoint).is_dir() else Path(snapshot_download(checkpoint))
    cfg = json.loads((src / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
    dropped = sorted(k for k in cfg if k not in known)
    if not dropped:
        return str(src)

    dst = src.parent / f"{src.name}_lerobot_compat"
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.name != "config.json" and not (dst / f.name).exists():
            os.symlink(f.resolve(), dst / f.name)
    (dst / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k in known}, indent=4))
    print(f"[pi05_server] dropped config fields unknown to this lerobot: {dropped} -> {dst}")
    return str(dst)


def load_weights_remapped(policy, checkpoint, weights_dir=None):
    """Stream model.safetensors into `policy` tensor by tensor, with the SigLIP vision-tower keys
    remapped, and fail loudly on anything missing.

    Why not PI05Policy.from_pretrained(): the checkpoint stores the vision tower as
    `...vision_tower.encoder/embeddings/post_layernorm` (older transformers layout) while
    transformers 5.x nests it under `...vision_tower.vision_model.`. from_pretrained() loads
    strict=True, which raises on the mismatch AFTER copying every key that does match - and then
    swallows the exception with a one-line "Warning: Could not load state dict". The result is a
    policy whose language model and action expert are fine-tuned but whose vision encoder is
    randomly initialized. It also materializes the whole 9 GB file plus a CPU copy of the model,
    which the kernel OOM-killed on this 30 GB machine.

    Streaming with safe_open(device=<param device>) keeps host RAM near zero: each tensor goes
    straight from the file to its parameter's device and dtype.
    """
    from pathlib import Path

    from safetensors import safe_open

    params = policy.state_dict()  # tensors share storage with the live parameters
    loaded = set()
    device = next(policy.parameters()).device
    with safe_open(str(Path(weights_dir or checkpoint) / "model.safetensors"), framework="pt", device=str(device)) as f:
        for file_key in f.keys():
            key = file_key if file_key.startswith("model.") else f"model.{file_key}"
            if key not in params and ".vision_tower." in key:
                key = key.replace(".vision_tower.", ".vision_tower.vision_model.", 1)
            if key not in params:
                raise RuntimeError(f"pi05 checkpoint tensor {file_key!r} has no matching parameter")
            with torch.no_grad():
                params[key].copy_(f.get_tensor(file_key))
            loaded.add(key)

    # lm_head is tied to embed_tokens in PaliGemma; a missing lm_head alone is harmless.
    missing = [k for k in params if k not in loaded and not k.endswith("lm_head.weight")]
    if missing:
        raise RuntimeError(f"pi05 weights did not fully load: missing={missing[:10]} (total {len(missing)})")
    print(f"[pi05_server] all {len(loaded)} checkpoint tensors loaded (vision tower keys remapped)")


def load_policy(checkpoint, device, weights_dir=None):
    """Build PI05Policy directly on `device` from the (compat) config, then stream the weights in."""
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = str(device)
    with torch.device(device):
        policy = PI05Policy(config)
    load_weights_remapped(policy, checkpoint, weights_dir)
    return policy.eval()


def to_policy_obs(obs):
    """Client numpy obs -> the raw (pre-normalization) tensor dict the preprocessor expects."""
    out = {"observation.state": torch.as_tensor(np.asarray(obs["observation.state"]), dtype=torch.float32)}
    for key in IMAGE_KEYS:
        img = torch.as_tensor(np.asarray(obs[key]))
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        out[key] = img.contiguous()
    out["task"] = obs["task"]
    return out


def main():
    parser = argparse.ArgumentParser(description="Serve a Pi0.5 (PI05Policy) checkpoint over multiprocessing.connection.")
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parent / "checkpoints" / "pi05_isaaclab_so101_multitask_1ep"),
                        help="Path or hub id of the PI05Policy checkpoint.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Override how many actions of each 16-step chunk are executed before "
                        "replanning (inference-only). Omit to use the checkpoint's value (16).")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Override the number of flow-matching integration steps (checkpoint: 10).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--authkey", type=str, default=DEFAULT_AUTHKEY,
                        help="Must match run_policy_fm_0910_pi.py's --pi_server_authkey.")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    policy = load_policy(args.checkpoint, device)
    if args.n_action_steps is not None:
        print(f"[pi05_server] overriding n_action_steps {policy.config.n_action_steps} -> {args.n_action_steps}")
        policy.config.n_action_steps = args.n_action_steps
    if args.num_inference_steps is not None:
        print(f"[pi05_server] overriding num_inference_steps {policy.config.num_inference_steps} -> {args.num_inference_steps}")
        policy.config.num_inference_steps = args.num_inference_steps
    policy.to(device)
    policy.eval()

    # Same device_processor override as multitask_dit_server.py: the saved processor config
    # records device="cuda" and make_pre_post_processors() rebuilds the step from it.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    state_dim = int(policy.config.output_features["action"].shape[0])
    listener = Listener((args.host, args.port), authkey=args.authkey.encode())
    print(f"[pi05_server] listening on {args.host}:{args.port}, checkpoint={args.checkpoint}, device={device}, "
          f"chunk_size={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")

    try:
        while True:
            conn = listener.accept()
            print(f"[pi05_server] client connected from {listener.last_accepted}")
            try:
                while True:
                    try:
                        request = conn.recv()
                    except EOFError:
                        break

                    cmd = request.get("cmd")
                    if cmd == "reset":
                        policy.reset()
                        conn.send({"ok": True, "state_dim": state_dim,
                                   "n_action_steps": policy.config.n_action_steps})

                    elif cmd == "step":
                        obs = to_policy_obs(request["obs"])
                        replanned = len(policy._action_queue) == 0
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            batch = preprocessor(obs)
                            raw_action = policy.select_action(batch)
                            action = postprocessor(raw_action)
                        action = action.float().cpu().squeeze(0)
                        if replanned:
                            print(f"[pi05_server] new chunk in {time.perf_counter() - t0:.3f}s task={obs['task']!r} "
                                  f"state={obs['observation.state'].tolist()}")
                        conn.send({"ok": True, "action": action.tolist(), "replanned": replanned})

                    elif cmd == "close":
                        conn.send({"ok": True})
                        break

                    else:
                        conn.send({"ok": False, "error": f"unknown cmd {cmd!r}"})
            except Exception as e:
                # A bad request/inference error shouldn't kill the whole server.
                print(f"[pi05_server] error: {e!r}")
                try:
                    conn.send({"ok": False, "error": repr(e)})
                except Exception:
                    pass
            finally:
                conn.close()
                torch.cuda.empty_cache()
    finally:
        listener.close()


if __name__ == "__main__":
    main()
