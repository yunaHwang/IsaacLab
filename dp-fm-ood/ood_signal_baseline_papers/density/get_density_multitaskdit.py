#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Density-based non-conformity score for a native LeRobot MultiTaskDiTPolicy.

This is the native-MultiTaskDiT equivalent of the previous GLOVES density
implementation. It intentionally has NO imports from:

    - lerobot.datasets
    - lerobot.utils.constants
    - lerobot_policy_gloves
    - robomimic
    - diffusers

The only model object required is an already-loaded
MultiTaskDiTPolicy.

For the flow-matching model, the score follows the same one-step
backward-Euler approximation used by the previous implementation:

    z_hat(x) = x - v_theta(x, t=1, context)

    s(x) = ||z_hat(x)||_2^2

where:

    v_theta = policy.noise_predictor
    context = policy.observation_encoder.encode(observation_batch)

The score is computed in the action space presented to the policy's
flow-matching model. Therefore, `action` must be in the SAME normalized
action space used by the trained model. If your live action is still in
physical/robot units, normalize it before calling compute_density_score.

The observation batch must also already be in the representation expected
by the policy (including any state/image normalization handled by your
LeRobot preprocessing pipeline).

Standalone usage: NONE - unlike get_loss_diffdagger.py/get_nonconformity_gloves.py,
this file has no main()/argparse/`if __name__ == "__main__":` entry point. It's library code
only, imported by ood_signal.py's multitask_dit_density. Add a dataset-sweep CLI here
(mirroring get_nonconformity_gloves.py's main()) if you need an offline calibration
score distribution for MultiTaskDiTPolicy.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def build_multitask_dit_context(
    policy,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Build the conditioning vector using the native MultiTaskDiTPolicy.

    Args:
        policy:
            Loaded MultiTaskDiTPolicy.

        batch:
            Observation batch in native LeRobot feature-key format.
            For your checkpoint this should contain at least:
                "observation.state"
                "observation.images.wrist_cam"
                "observation.images.table_cam"

            Expected state shape:
                [B, n_obs_steps, 9]

            Expected image shapes before _prepare_batch:
                [B, n_obs_steps, 3, H, W]

    Returns:
        Conditioning tensor produced by the policy's own
        ObservationEncoder.
    """
    batch = policy._prepare_batch(batch)
    return policy.observation_encoder.encode(batch)


@torch.no_grad()
def compute_z_hat(
    policy,
    action_chunk: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """
    One-step backward-Euler approximation of the inverse flow.

        z_hat(x) ~= x - v_theta(x, t=1, context)

    Args:
        policy:
            Loaded MultiTaskDiTPolicy.

        action_chunk:
            Tensor [B, T, action_dim].

        context:
            Tensor returned by build_multitask_dit_context().

    Returns:
        Tensor [B, T, action_dim].
    """
    if not getattr(policy.config, "is_flow_matching", False):
        raise ValueError(
            "compute_z_hat() requires a flow-matching MultiTaskDiTPolicy."
        )

    if action_chunk.ndim != 3:
        raise ValueError(
            f"Expected action_chunk [B, T, action_dim], "
            f"got {tuple(action_chunk.shape)}"
        )

    # The native MultiTaskDiT flow-matching model uses continuous
    # timesteps in [0, 1]. The trained velocity field is therefore
    # evaluated at t=1 exactly as in the original density score.
    t_one = torch.ones(
        action_chunk.shape[0],
        device=action_chunk.device,
        dtype=action_chunk.dtype,
    )

    velocity = policy.noise_predictor(
        action_chunk,
        t_one,
        conditioning_vec=context,
    )

    return action_chunk - velocity


@torch.no_grad()
def nonconformity_score(
    policy,
    action_chunk: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the density-based non-conformity score:

        s(x) = ||z_hat(x)||_2^2

    summed over the complete action chunk.

    Returns:
        Tensor [B], one score per action chunk.
    """
    z_hat = compute_z_hat(
        policy=policy,
        action_chunk=action_chunk,
        context=context,
    )

    return z_hat.flatten(start_dim=1).pow(2).sum(dim=1)


def _get_action_shape(policy) -> tuple[int, int]:
    """
    Return (horizon, action_dim) from the loaded policy.
    """
    horizon = int(policy.config.horizon)
    action_dim = int(policy.config.action_feature.shape[0])
    return horizon, action_dim


@torch.no_grad()
def compute_density_score(
    policy,
    batch: dict[str, torch.Tensor],
    action: torch.Tensor,
    *,
    tile_single_action: bool = True,
) -> torch.Tensor:
    """
    Compute the density non-conformity score for an action under the
    conditioning observation batch.

    This is the main function intended for run_policy.py.

    Args:
        policy:
            Already-loaded native LeRobot MultiTaskDiTPolicy.

        batch:
            Observation batch in the same format used for policy inference.

            For your checkpoint:
                observation.state              [B, 2, 9]
                observation.images.wrist_cam   [B, 2, 3, H, W]
                observation.images.table_cam   [B, 2, 3, H, W]

        action:
            Action to score. Accepted shapes:

                [action_dim]
                    One action for one environment.

                [B, action_dim]
                    One action per batch element. If tile_single_action=True,
                    it is repeated over the policy horizon.

                [B, T, action_dim]
                    Full action chunk. T must equal policy.config.horizon.

        tile_single_action:
            If True, a single-step action is repeated across the full
            policy horizon. This matches the convention used by the old
            GLOVES density implementation.

    Returns:
        Tensor [B].
    """
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    # Make sure observations are on the policy device/dtype where
    # appropriate. Image tensors and state tensors should remain floating
    # point; integer image inputs are intentionally not cast here because
    # the caller may be using the policy's expected image representation.
    prepared_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared_batch[key] = value.to(device=device)
        else:
            prepared_batch[key] = value

    context = build_multitask_dit_context(
        policy=policy,
        batch=prepared_batch,
    )

    horizon, action_dim = _get_action_shape(policy)

    action = action.to(device=device)

    if not torch.is_floating_point(action):
        action = action.float()

    # Convert action to the policy/model dtype.
    action = action.to(dtype=dtype)

    if action.ndim == 1:
        # [Da] -> [1, Da]
        action = action.unsqueeze(0)

    if action.ndim == 2:
        # [B, Da] -> [B, T, Da]
        if action.shape[-1] != action_dim:
            raise ValueError(
                f"Expected action_dim={action_dim}, "
                f"got {action.shape[-1]}."
            )

        if not tile_single_action:
            raise ValueError(
                "Received [B, action_dim], but tile_single_action=False. "
                "Provide a full [B, T, action_dim] action chunk instead."
            )

        action = action.unsqueeze(1).expand(
            -1, horizon, -1
        ).contiguous()

    elif action.ndim == 3:
        if action.shape[-1] != action_dim:
            raise ValueError(
                f"Expected action_dim={action_dim}, "
                f"got {action.shape[-1]}."
            )

        if action.shape[-2] != horizon:
            raise ValueError(
                f"Expected full action horizon={horizon}, "
                f"got T={action.shape[-2]}."
            )

    else:
        raise ValueError(
            "Action must have shape [Da], [B, Da], or [B, T, Da]. "
            f"Got {tuple(action.shape)}."
        )

    # The context batch size and action batch size must agree.
    if context.shape[0] != action.shape[0]:
        if context.shape[0] == 1 and action.shape[0] > 1:
            context = context.expand(action.shape[0], -1)
        else:
            raise ValueError(
                f"Context batch size ({context.shape[0]}) does not match "
                f"action batch size ({action.shape[0]})."
            )

    return nonconformity_score(
        policy=policy,
        action_chunk=action,
        context=context,
    )


@torch.no_grad()
def generate_action_chunk(
    policy,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Generate the native MultiTaskDiT policy's full flow-matching action
    trajectory.

    IMPORTANT:
        This returns the complete [B, horizon, action_dim] flow output,
        not the policy's n_action_steps execution slice.

    This is useful if you want to compare the generated action chunk
    against a density score defined over the full flow horizon.

    Returns:
        Tensor [B, horizon, action_dim].
    """
    device = next(policy.parameters()).device

    prepared_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared_batch[key] = value.to(device=device)
        else:
            prepared_batch[key] = value

    prepared_batch = policy._prepare_batch(prepared_batch)

    batch_size = prepared_batch["observation.state"].shape[0]

    conditioning_vec = policy.observation_encoder.encode(
        prepared_batch
    )

    # For flow matching, conditional_sample performs the same Euler/RK4
    # integration configured in policy.config.
    return policy.objective.conditional_sample(
        policy.noise_predictor,
        batch_size,
        conditioning_vec,
    )


@torch.no_grad()
def generate_executable_action_chunk(
    policy,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Generate the action chunk normally returned by native
    MultiTaskDiTPolicy._generate_actions().

    Returns:
        Tensor [B, n_action_steps, action_dim].
    """
    device = next(policy.parameters()).device

    prepared_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared_batch[key] = value.to(device=device)
        else:
            prepared_batch[key] = value

    prepared_batch = policy._prepare_batch(prepared_batch)

    return policy._generate_actions(prepared_batch)

