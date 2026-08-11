import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "ood-signal-baseline-papers", "reconstruction-loss"))
from get_diffloss_diffdagger import compute_diffusion_loss, diffusion_action_shape

sys.path.append(os.path.join(os.path.dirname(__file__), "ood-signal-baseline-papers", "density"))
from density_nonconformity_score_calc import (
    compute_density_score,
    DEFAULT_STATE_KEYS,
    _stack_obs_history_state,
)
from lerobot.utils.constants import OBS_STATE


def _expand_to_batch(tensor, n):
    """Expand a single-sample tensor to batch size n (broadcast view, no data copy - then
    made contiguous since some ops, e.g. conv-based obs encoders, expect real strides)."""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.expand(n, *tensor.shape[1:]).contiguous()


def diffdagger_loss_state_ood(policy, state, model_action, num_samples=512):
    """Diffdagger-style noise-prediction loss for a (state, action) pair (see
    get_diffloss_diffdagger.compute_diffusion_loss), averaged over `num_samples`
    independent (noise, diffusion-timestep) draws (diffdagger's Nb, Eq. 2:
    L(o,a,pi) = E_{eps~N(0,I), t~mu(1,T)} L_pi(o,a,eps,t)).

    Rather than calling compute_diffusion_loss in a Python loop num_samples times - which
    would silently redraw its own (noise, timesteps) deep inside that function on every
    call - this draws the whole batch of num_samples (noise, timestep) pairs explicitly
    here, and scores all of them in a single vectorized compute_diffusion_loss call: the
    same (state, action) pair is repeated num_samples times along the batch dim, so one
    forward pass computes the whole Nb-sample expectation at once.

    Args:
        policy: robomimic diffusion policy (as returned by FileUtils.policy_from_checkpoint).
        state: windowed obs_seq, {key: tensor [1, To, ...]} - same format the diffusion
            policy itself consumes (a single (unbatched) observation).
        model_action: [Da] or [1, Da] - the single action to score.
        num_samples: Nb, how many independent (noise, timestep) draws to average the loss
            over - a single draw is noisy since (noise, timestep) are random.

    Returns:
        scalar tensor: mean noise-prediction MSE loss over num_samples draws.
    """
    Nb = num_samples
    Tp, action_dim, num_train_timesteps, device = diffusion_action_shape(policy)

    # repeat the single (state, action) pair Nb times along the batch dim
    batched_state = {k: _expand_to_batch(v, Nb) for k, v in state.items()}
    batched_action = _expand_to_batch(model_action, Nb)

    # explicitly draw the whole batch of (noise, timestep) pairs here, visible at this
    # level, instead of it happening invisibly inside compute_diffusion_loss
    noise = torch.randn(Nb, Tp, action_dim, device=device)
    timesteps = torch.randint(0, num_train_timesteps, (Nb,), device=device).long()

    sample_losses = compute_diffusion_loss(
        policy, batched_state, batched_action, noise=noise, timesteps=timesteps
    )
    return sample_losses.mean(dim=0)


def fm_diffdagger_loss_state_ood(
    dit_flow, state, model_action, num_samples=512, state_keys=DEFAULT_STATE_KEYS
):
    """Diffdagger-style flow-matching loss for a (state, action) pair - the fm analog of
    diffdagger_loss_state_ood above. Reuses DiTFlowModel.compute_loss's training objective
    (continuous-time flow matching: x_t = (1-t)*noise + t*action, predict velocity
    v_theta(x_t, t, context), MSE against the target velocity action-noise) instead of dp's
    discrete-timestep DDPM noise-prediction loss - fm's continuous t in [0,1] standing in for
    dp's discrete diffusion timestep. Averaged over `num_samples` independent (noise, t)
    draws for the same fixed (state, action) pair, the same Nb-sample-averaging idea as
    diffdagger_loss_state_ood (Eq. 2).

    Args:
        dit_flow: trained, frozen GLOVES flow model (DiTPolicy.dit_flow).
        state: obs_history - iterable of `n_obs_steps` obs dicts, matching
            compute_density_score's expected format.
        model_action: [Da], [B, Da], or [B, T, Da] action to score - a single-step action is
            tiled across dit_flow's action chunk length, same convention as
            compute_density_score.
        num_samples: Nb, how many independent (noise, t) draws to average the loss over - a
            single draw is noisy since (noise, t) are random.
        state_keys: obs dict keys to concatenate into the proprioceptive state vector fed to
            DiTFlowModel's state encoder - must match the keys the checkpoint was trained on.

    Returns:
        scalar tensor: mean flow-matching MSE loss over num_samples draws.
    """
    Nb = num_samples
    device = next(dit_flow.parameters()).device

    # build context once from the fixed state, then repeat it across the Nb batch (the state
    # doesn't change across draws, only (noise, t) do)
    state_batch = _stack_obs_history_state(state, state_keys, device)  # [1, To, state_dim]
    context = dit_flow._prepare_context_tokens({OBS_STATE: state_batch})  # [seq_len, 1, dc]
    context = context.expand(-1, Nb, -1).contiguous()  # [seq_len, Nb, dc]

    ac_chunk = dit_flow.velocity_net.ac_chunk
    ac_dim = dit_flow.velocity_net.ac_dim

    action = model_action.to(device=device, dtype=torch.float32)
    if action.ndim == 1:
        action = action.unsqueeze(0)  # [Da] -> [1, Da]
    if action.ndim == 2:
        # single-step action -> tile across the action chunk, matching
        # compute_density_score's tiling convention
        action = action.unsqueeze(1).expand(-1, ac_chunk, -1).contiguous()
    assert action.shape[-2:] == (ac_chunk, ac_dim), (
        f"expected action shape (B,{ac_chunk},{ac_dim}), got {tuple(action.shape)}"
    )
    action = _expand_to_batch(action, Nb)

    # explicitly draw the whole batch of (noise, t) pairs here, visible at this level,
    # mirroring diffdagger_loss_state_ood's explicit draw rather than letting compute_loss's
    # formula silently draw its own single (noise, t) pair per call
    noise = torch.randn(Nb, ac_chunk, ac_dim, device=device)
    timesteps = dit_flow.noise_distribution.sample((Nb,)).to(device=device)

    noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * action
    pred = dit_flow.velocity_net(noisy_actions=noisy_trajectory, time=timesteps, context=context)
    target = action - noise
    sample_losses = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))  # [Nb]
    return sample_losses.mean(dim=0)


def density_state_ood(dit_flow, state, model_action):
    """GLOVES density non-conformity score s(x) = ||z_hat(x)||^2 for a (state, action) pair
    (paper Eq. 6; see density_nonconformity_score_calc.compute_density_score).

    Args:
        dit_flow: trained, frozen GLOVES flow model (DiTPolicy.dit_flow).
        state: obs_history - iterable of per-step obs dicts, matching
            compute_density_score's expected format.
        model_action: [Da], [B, Da], or [B, T, Da] action to score.

    Returns:
        tensor [B]: s(x) for this action.
    """
    return compute_density_score(dit_flow, state, model_action)


def cf_prediction_loss_state_ood(score, calibration_scores):
    """Conformal-prediction p-value for a non-conformity score, calibrated against a
    held-out calibration set of the same score computed over in-distribution/expert
    (state, action) pairs (see the paper's "we apply conformal prediction with a calibration
    set of small amount expert data on score s(x) to make p-value OOD prediction").

    This does not compute a score itself - `score` is expected to already come from one of
    diffdagger_loss_state_ood / density_state_ood / etc., and `calibration_scores` from the
    same score function run over the calibration set. A small p-value means `score` is
    unusually large relative to the calibration set, i.e. likely OOD.

    p(score) = (1 + #{calibration_scores >= score}) / (n_calib + 1)

    the standard split/inductive-conformal-prediction p-value formula (the "+1"s are what
    make it finite-sample valid, unlike a plain empirical-CDF percentile).

    Args:
        score: scalar non-conformity score (float, or a tensor with a single element).
        calibration_scores: 1D array-like of non-conformity scores from the calibration set.

    Returns:
        float p-value in (0, 1].
    """
    if torch.is_tensor(score):
        score = score.item()
    calibration_scores = np.asarray(calibration_scores)
    n_calib = calibration_scores.shape[0]
    num_at_least_as_extreme = np.sum(calibration_scores >= score)
    return (1 + num_at_least_as_extreme) / (n_calib + 1)


def smoothness_loss_state_ood(score_fn, state, model_action, prev_state, prev_model_action):
    """Temporal smoothness of a non-conformity score across consecutive timesteps:
    |score_fn(state, action) - score_fn(prev_state, prev_action)|. A large jump flags an
    abrupt change in how OOD the trajectory looks step-to-step, independent of the absolute
    score level.

    Args:
        score_fn: any (state, model_action) -> scalar score function from this file, with
            its model handle already bound, e.g.
            functools.partial(diffdagger_loss_state_ood, policy) or
            functools.partial(density_state_ood, dit_flow).
        state, model_action: this timestep's (state, action) pair.
        prev_state, prev_model_action: the previous timestep's (state, action) pair.

    Returns:
        scalar (same type score_fn returns): absolute score difference between timesteps.
    """
    score = score_fn(state, model_action)
    prev_score = score_fn(prev_state, prev_model_action)
    return abs(score - prev_score)


def _perturb_state(state, sigma):
    """Add isotropic Gaussian noise (std=sigma) to every tensor value in `state`, whether
    `state` is a single {key: tensor} obs dict/obs_seq (diffdagger's format) or an iterable
    of per-step obs dicts (density's obs_history format)."""

    def _perturb_dict(d):
        return {
            k: v + sigma * torch.randn_like(v) if torch.is_tensor(v) else v
            for k, v in d.items()
        }

    if isinstance(state, dict):
        return _perturb_dict(state)
    return [_perturb_dict(step_obs) for step_obs in state]


def perturb_loss_state_ood(score_fn, state, model_action, sigma=0.05, n_perturbations=8):
    """Sensitivity of a non-conformity score to small random perturbations of `state`: how
    much does score_fn's output change if the state is nudged by isotropic Gaussian noise?
    A score that swings wildly under tiny state perturbations suggests the model's OOD
    judgment near this state is unstable, independent of the raw score value itself.

    Args:
        score_fn: see smoothness_loss_state_ood's docstring.
        state: obs dict/obs_seq or obs_history (whatever score_fn expects).
        model_action: action to score (held fixed - only `state` is perturbed).
        sigma: std of the isotropic Gaussian noise added to each tensor in `state`.
        n_perturbations: number of perturbed copies of `state` to average over.

    Returns:
        mean absolute score deviation, averaged over `n_perturbations` perturbed states.
    """
    base_score = score_fn(state, model_action)

    deviations = []
    for _ in range(n_perturbations):
        perturbed_state = _perturb_state(state, sigma)
        perturbed_score = score_fn(perturbed_state, model_action)
        deviations.append(abs(perturbed_score - base_score))

    return sum(deviations) / len(deviations)
