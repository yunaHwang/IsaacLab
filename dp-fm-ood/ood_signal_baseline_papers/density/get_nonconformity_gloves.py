#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GLOVES density-based non-conformity score (paper "Flow-based Policy Adaptation without
Policy Updates", Eq. 6): s(x) = ||z_hat(x)||_2^2, where z_hat(x) is a one-step
backward-Euler approximation of F_theta^-1(x):

    z_hat(x) ~= x - v_theta(x, t=1, context)

F_theta is the push-forward map induced by integrating the trained GLOVES velocity field
v_theta (DiTFlowModel.velocity_net) from a Gaussian prior to an action chunk; it is not
solved for exactly here, matching the paper. The paper's full density formula (Eq. 5b) has
an additional log|det J_{F_theta}(z)| correction term, but the paper itself drops it for
this score -- computing/estimating that Jacobian is expensive in high dimensions, so its
effect is instead treated as a frozen, model-dependent scaling constant and absorbed into
conformal calibration/thresholding downstream of this script, rather than computed here.

Intended use: run over a calibration set of expert action chunks to build up scores for
conformal-prediction thresholding, or over agent-proposed action chunks at inference time
to flag OOD actions (large s(x) -> maps to a low-likelihood region of the Gaussian prior).

Standalone usage: run this file directly to sweep a LeRobotDataset and save the resulting
non-conformity scores to a CSV (a calibration-set score distribution for
cf_prediction_score_state_ood in ood_signal.py):

    python get_nonconformity_gloves.py --checkpoint gloves_model.pth \
        --dataset my_dataset_repo_id --save-to-file --save-file-name nonconformity_scores.csv

Optional flags: --dataset-root (local root, if --dataset isn't a hub repo id), --device
(default: cpu), --batch-size (default: 32). See main() below for the full list.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_gloves.modeling_gloves import DiTFlowModel, DiTPolicy

DEFAULT_STATE_KEYS = ["eef_pos", "gripper_pos", "object", "eef_quat"]


@torch.no_grad()
def compute_z_hat(
    velocity_net: torch.nn.Module, actions: torch.Tensor, context: torch.Tensor
) -> torch.Tensor:
    """One-step backward-Euler approximation of F_theta^-1(actions) (paper Eq. 6).

    Args:
        velocity_net: the trained, frozen v_theta (DiTFlowModel.velocity_net).
        actions: (B, T, ac_dim) action chunk to score.
        context: (seq_len, B, context_dim) conditioning tokens for these actions.
    Returns:
        (B, T, ac_dim) estimate of the latent z each action chunk maps back to.
    """
    t_one = torch.ones(actions.shape[0], device=actions.device, dtype=actions.dtype)
    return actions - velocity_net(actions, t_one, context)


@torch.no_grad()
def nonconformity_score(
    velocity_net: torch.nn.Module, actions: torch.Tensor, context: torch.Tensor
) -> torch.Tensor:
    """s(x) = ||z_hat(x)||_2^2 (paper Eq. 6), summed over the whole action chunk."""
    z_hat = compute_z_hat(velocity_net, actions, context)
    return z_hat.flatten(start_dim=1).pow(2).sum(dim=1)


@torch.no_grad()
def nonconformity_score_from_batch(
    dit_flow: DiTFlowModel, batch: dict[str, torch.Tensor], action_key: str = ACTION
) -> torch.Tensor:
    """Same as `nonconformity_score`, but builds `context` from a raw observation batch via
    the policy's own encoders, and reads the action chunk to score from `batch[action_key]`.
    """
    context = dit_flow._prepare_context_tokens(batch)
    return nonconformity_score(dit_flow.velocity_net, batch[action_key], context)


def _stack_obs_history_state(
    obs_history, state_keys: list[str], device
) -> torch.Tensor:
    """Concatenate `state_keys` from each step of a live `obs_history` (e.g. run_policy.py's
    `obs_history` deque) into a [1, To, state_dim] proprioceptive state batch, matching the
    `observation.state` shape DiTFlowModel's encoders expect."""
    return torch.stack(
        [torch.cat([torch.as_tensor(o[k]).flatten() for k in state_keys]) for o in obs_history],
        dim=0,
    ).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, To, state_dim]


@torch.no_grad()
def generate_action_chunk(
    dit_flow: DiTFlowModel,
    obs_history,
    state_keys: list[str] = DEFAULT_STATE_KEYS,
    device=None,
    full_chunk: bool = False,
) -> torch.Tensor:
    """Generate the trained GLOVES flow model's own action chunk from a live `obs_history`
    (e.g. run_policy.py's `obs_history` deque), by actually running F_theta forward
    (conditional_sample's ODE solve) instead of scoring an existing action against it (see
    compute_density_score below for that).

    Args:
        dit_flow: the trained, frozen GLOVES flow model (DiTPolicy.dit_flow).
        obs_history: iterable of `n_obs_steps` obs dicts, each mapping every key in
            `state_keys` to a tensor for that step (same shape/keys as run_policy.py's
            per-step `obs`).
        state_keys: obs dict keys to concatenate into the proprioceptive state vector fed
            to DiTFlowModel's state encoder -- must match the keys the checkpoint was
            trained on.
        device: device to run on. Defaults to dit_flow's own device.
        full_chunk: if True, return F_theta's full (B, horizon, ac_dim) sample -- the shape
            compute_density_score's ac_chunk assert expects, and what the paper's z_hat is
            defined over -- via DiTFlowModel.conditional_sample directly, skipping
            DiTFlowModel.generate_actions' truncation to n_action_steps. If False (default),
            return the (B, n_action_steps, ac_dim) slice meant for env execution, matching
            DiTFlowModel.generate_actions' own return value.

    Returns:
        tensor [1, horizon, ac_dim] if full_chunk else [1, n_action_steps, ac_dim].
    """
    device = device or next(dit_flow.parameters()).device
    state = _stack_obs_history_state(obs_history, state_keys, device)
    if not full_chunk:
        return dit_flow.generate_actions({OBS_STATE: state})

    context = dit_flow._prepare_context_tokens({OBS_STATE: state})
    return dit_flow.conditional_sample(context=context)


@torch.no_grad()
def compute_density_score(
    dit_flow: DiTFlowModel,
    obs_history,
    action: torch.Tensor,
    state_keys: list[str] = DEFAULT_STATE_KEYS,
    device=None,
) -> torch.Tensor:
    """Score a live action against GLOVES' density non-conformity score (Eq. 6), built from a
    live `obs_history` (e.g. run_policy.py's `obs_history` deque) instead of a LeRobotDataset
    batch. Mirrors get_loss_diffdagger.compute_diffusion_loss's role for the density
    metric: run_policy.py calls this once per step to score the live human action.

    Args:
        dit_flow: the trained, frozen GLOVES flow model (DiTPolicy.dit_flow).
        obs_history: iterable of `n_obs_steps` obs dicts, each mapping every key in
            `state_keys` to a tensor for that step (same shape/keys as run_policy.py's
            per-step `obs`).
        action: [Da], [B, Da], or [B, T, Da] action to score (e.g. a spacemouse
            user_action). A single-step action is tiled across dit_flow's action chunk
            length, same tiling convention as compute_diffusion_loss.
        state_keys: obs dict keys to concatenate into the proprioceptive state vector fed
            to DiTFlowModel's state encoder -- must match the keys the checkpoint was
            trained on.

    Returns:
        tensor [B]: s(x) = ||z_hat(x)||^2 for this action.
    """
    device = device or next(dit_flow.parameters()).device

    state = _stack_obs_history_state(obs_history, state_keys, device)
    context = dit_flow._prepare_context_tokens({OBS_STATE: state})

    ac_chunk = dit_flow.velocity_net.ac_chunk
    ac_dim = dit_flow.velocity_net.ac_dim

    action = action.to(device=device, dtype=torch.float32)
    if action.ndim == 1:
        action = action.unsqueeze(0)  # [Da] -> [1, Da]
    if action.ndim == 2:
        # single-step action -> tile across the action chunk, matching
        # compute_diffusion_loss's tiling convention
        action = action.unsqueeze(1).expand(-1, ac_chunk, -1).contiguous()
    assert action.shape[-2:] == (ac_chunk, ac_dim), (
        f"expected action shape (B,{ac_chunk},{ac_dim}), got {tuple(action.shape)}"
    )

    return nonconformity_score(dit_flow.velocity_net, action, context)


def _stack_image_features(policy: DiTPolicy, batch: dict[str, torch.Tensor]) -> dict:
    if not policy.config.image_features:
        return batch
    batch = dict(batch)
    batch[OBS_IMAGES] = torch.stack(
        [batch[key] for key in policy.config.image_features], dim=-4
    )
    return batch


def main():
    parser = argparse.ArgumentParser(
        description="Compute the GLOVES density non-conformity score s(x) = ||z_hat(x)||^2 "
        "(paper Eq. 6) for every action chunk in a LeRobot dataset, using a trained GLOVES "
        "checkpoint's frozen velocity field as v_theta."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path or hub id of a trained GLOVES policy checkpoint.")

    # these two ones are for reading straight from a hdf5 file, where in that case can be used as a standalone
    # but you can also use this as a python file with necessary functions in it, at the same time when you are running a policy (`run_policy.py`)
    parser.add_argument("--dataset", type=str, required=True, help="LeRobotDataset repo id or local root.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Local root, if --dataset is not a hub repo id.")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-to-file", action="store_true")
    parser.add_argument("--save-file-name", type=str, default="nonconformity_scores.csv")
    args = parser.parse_args()

    if args.dataset is not None or args.dataset_root is not None:
        device = torch.device(args.device)
        policy = DiTPolicy.from_pretrained(args.checkpoint)
        policy.to(device)
        policy.eval()

        ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
        delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
        dataset = LeRobotDataset(
            args.dataset, root=args.dataset_root, delta_timestamps=delta_timestamps
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        scores = []
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = _stack_image_features(policy, batch)
            batch_scores = nonconformity_score_from_batch(policy.dit_flow, batch)
            scores.extend(batch_scores.cpu().tolist())

        if args.save_to_file:
            out_path = Path(args.save_file_name)
            with out_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["score"])
                writer.writerows([[s] for s in scores])
            print(f"Saved {len(scores)} scores to {out_path}")
        else:
            print(scores)


if __name__ == "__main__":
    main()
