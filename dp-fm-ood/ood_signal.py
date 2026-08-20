"""OOD-scoring functions shared by run_policy_dp.py and run_policy_fm.py.

Every function here is called with EITHER the model's own action (a "state OOD" check - is
the model's own action consistent with its own state?) OR a live human/teleop action (an
"action OOD" check - is this externally-supplied action consistent with what the model has
learned?), depending solely on what the caller passes as the action argument. None of these
functions know or care which case they're being used for - that distinction lives entirely
at the call site (see run_policy_dp.py/run_policy_fm.py's "[STATE]"/"[ACTION]" log labels).

Each backbone (dp, GLOVES, MultiTaskDiT) gets its own function per metric, since their
underlying models have genuinely different internals (see each function's docstring). For
the density metric, the actual math lives in ood_signal_baseline_papers/density/*.py
(one file per backbone) and the functions here are thin wrappers: reshape a live
obs_history/obs_seq into the batch format the primitive expects, call it, return the result.
For reconstruction loss, no such per-backbone primitive file exists, so dp_loss/gloves_loss/
multitask_dit_loss do real work of their own - not just reshaping data, but also diffdagger's
Nb-sample averaging (Eq. 2), and for gloves/multitask_dit, the training-objective math itself.
"""

import numpy as np
import torch
import torch.nn.functional as F


# -- Utility functions --
def _expand_to_batch(tensor, n):
    """Expand a single-sample tensor to batch size n (broadcast view, no data copy - then
    made contiguous since some ops, e.g. conv-based obs encoders, expect real strides)."""
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.expand(n, *tensor.shape[1:]).contiguous()

#####################################
# -- Metric 1. Reconstruction loss --
#####################################

## -- DP --
def dp_loss(policy, state, action, num_samples=512):
    """Diffdagger-style noise-prediction loss for a (state, action) pair. Calling
    get_loss_diffdagger.compute_diffusion_loss) multiple times.

    Args:
        policy: robomimic diffusion policy.
        state: windowed obs_seq, {key: tensor [1, To, ...]}.
        action: [Da] or [1, Da] - the single action to score.
        num_samples: Nb, how many independent (noise, timestep) draws to average the loss
            over - a single draw is noisy since (noise, timestep) are random.

    Returns:
        scalar tensor: mean noise-prediction MSE loss over num_samples draws.
    """
    from ood_signal_baseline_papers.reconstruction_loss.get_loss_diffdagger import (
        compute_diffusion_loss,
        diffusion_action_shape,
    )

    Nb = num_samples
    Tp, action_dim, num_train_timesteps, device = diffusion_action_shape(policy)

    # repeat the single (state, action) pair Nb times along the batch dim
    batched_state = {k: _expand_to_batch(v, Nb) for k, v in state.items()}
    batched_action = _expand_to_batch(action, Nb)

    noise = torch.randn(Nb, Tp, action_dim, device=device)
    timesteps = torch.randint(0, num_train_timesteps, (Nb,), device=device).long()

    sample_losses = compute_diffusion_loss(
        policy, batched_state, batched_action, noise=noise, timesteps=timesteps
    )
    return sample_losses.mean(dim=0)

## -- GLOVES --
def gloves_loss(
    dit_flow, state, action, num_samples=512, state_keys=None
):
    """Diffdagger-style flow-matching loss for a (state, action) pair. 
    Reuses DiTFlowModel.compute_loss's training objective
    (continuous-time flow matching: x_t = (1-t)*noise + t*action, predict velocity
    v_theta(x_t, t, context), MSE against the target velocity action-noise).
    Averaged over `num_samples` independent (noise, t) draws for the same fixed (state, action) pair.

    Args:
        dit_flow: trained, frozen GLOVES flow model (DiTPolicy.dit_flow).
        state: obs_history - iterable of `n_obs_steps` obs dicts, matching
            compute_density_score's expected format.
        action: [Da], [B, Da], or [B, T, Da] action to score.
        num_samples: Nb, how many independent (noise, t) draws to average the loss over - a
            single draw is noisy since (noise, t) are random.
        state_keys: obs dict keys to concatenate into the proprioceptive state vector fed to
            DiTFlowModel's state encoder.

    Returns:
        scalar tensor: mean flow-matching MSE loss over num_samples draws.
    """

    # call necessary utility functions from script, despite not using the key functions regarding density calculation
    from ood_signal_baseline_papers.density.get_nonconformity_gloves import (
        DEFAULT_STATE_KEYS,
        _stack_obs_history_state,
    )

    from lerobot.utils.constants import OBS_STATE

    if state_keys is None:
        state_keys = DEFAULT_STATE_KEYS

    Nb = num_samples
    device = next(dit_flow.parameters()).device

    state_batch = _stack_obs_history_state(state, state_keys, device)  # [1, To, state_dim]
    context = dit_flow._prepare_context_tokens({OBS_STATE: state_batch})  # [seq_len, 1, dc]
    context = context.expand(-1, Nb, -1).contiguous()  # [seq_len, Nb, dc]

    ac_chunk = dit_flow.velocity_net.ac_chunk
    ac_dim = dit_flow.velocity_net.ac_dim

    action = action.to(device=device, dtype=torch.float32)
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

    noise = torch.randn(Nb, ac_chunk, ac_dim, device=device)
    timesteps = dit_flow.noise_distribution.sample((Nb,)).to(device=device)

    noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * action
    pred = dit_flow.velocity_net(noisy_actions=noisy_trajectory, time=timesteps, context=context)
    target = action - noise
    sample_losses = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))  # [Nb]
    return sample_losses.mean(dim=0)

## -- MultiTaskDiT --
def multitask_dit_loss(policy, obs_history, action, num_samples=512):
    """Loss for a (state, action) pair, for LeRobot's MultiTaskDiTPolicy.

    Args:
        policy: trained, frozen MultiTaskDiTPolicy.
        obs_history: iterable of `n_obs_steps` obs dicts, already normalized/preprocessed
            the way multitask_dit_server.py's step handler builds them - observation.state,
            observation.images.table_cam, observation.images.wrist_cam (per-timestep,
            batch dim stripped) plus observation.language.tokens/attention_mask (one
            tokenized task string per call, constant across the window, batch dim intact).
        action: [Da], [T, Da], or [B, T, Da] action to score.
        num_samples: Nb, how many independent (noise, timestep) draws to average the loss
            over.

    Returns:
        scalar tensor: mean loss over num_samples draws (whichever objective the policy was
        configured with).
    """
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    Nb = num_samples
    device = next(policy.parameters()).device
    obs_history = list(obs_history)

    state = torch.stack(
        [o["observation.state"] for o in obs_history], dim=0
    ).unsqueeze(0)  # [1, To, state_dim]
    table_cam = torch.stack(
        [o["observation.images.table_cam"] for o in obs_history], dim=0
    ).unsqueeze(0)  # [1, To, 3, H, W]
    wrist_cam = torch.stack(
        [o["observation.images.wrist_cam"] for o in obs_history], dim=0
    ).unsqueeze(0)  # [1, To, 3, H, W]
    # Task/language conditioning is constant across the window, so one copy from the
    # latest step covers the whole obs_history - not stacked per-timestep like state/images.
    language_tokens = obs_history[-1][OBS_LANGUAGE_TOKENS]  # [1, seq_len]
    language_attention_mask = obs_history[-1][OBS_LANGUAGE_ATTENTION_MASK]  # [1, seq_len]

    action = action.to(device=device, dtype=torch.float32)
    if action.ndim == 1:
        action = action.unsqueeze(0)  # [Da] -> [1, Da]
    if action.ndim == 2:
        # single-step action -> tile across the training horizon, same convention as
        # dp_loss/gloves_loss
        action = action.unsqueeze(1).expand(-1, policy.config.horizon, -1).contiguous()

    batch = {
        "observation.state": _expand_to_batch(state, Nb).to(device),
        "observation.images.table_cam": _expand_to_batch(table_cam, Nb).to(device),
        "observation.images.wrist_cam": _expand_to_batch(wrist_cam, Nb).to(device),
        OBS_LANGUAGE_TOKENS: _expand_to_batch(language_tokens, Nb).to(device),
        OBS_LANGUAGE_ATTENTION_MASK: _expand_to_batch(language_attention_mask, Nb).to(device),
        ACTION: _expand_to_batch(action, Nb).to(device),
    }
    # This is a scoring-only forward pass (.item()'d by the caller, never backprop'd) - without
    # no_grad, autograd retains the full activation graph through the DiT + both CLIP vision
    # encoders + CLIP text encoder for all Nb samples, which is enough to OOM a 32GB GPU within
    # a couple of steps (confirmed via smoke test).
    with torch.no_grad():
        loss, _ = policy(batch) # https://github.com/huggingface/lerobot/blob/9e14584904d14eb79dc960b27bb40220f85bb993/src/lerobot/policies/multi_task_dit/modeling_multi_task_dit.py
    return loss

#####################################
# -- Metric 2. Density --
#####################################
# Currently FM only

## -- GLOVES --
def gloves_density(dit_flow, state, action):
    """GLOVES density non-conformity score s(x) = ||z_hat(x)||^2 for a (state, action) pair
    (paper Eq. 6; see get_nonconformity_gloves.compute_density_score).

    Args:
        dit_flow: trained, frozen GLOVES flow model (DiTPolicy.dit_flow).
        state: obs_history - iterable of per-step obs dicts, matching
            compute_density_score's expected format.
        action: [Da], [B, Da], or [B, T, Da] action to score.

    Returns:
        tensor [B]: s(x) for this action.
    """
    from ood_signal_baseline_papers.density.get_nonconformity_gloves import (
        compute_density_score,
    )

    return compute_density_score(dit_flow, state, action)


## -- MultiTaskDiT --
def multitask_dit_density(policy, obs_history, action, tile_single_action=False, return_z_hat=False):
    """MultiTaskDiTPolicy analog of gloves_density - a thin wrapper around
    get_density_multitaskdit.compute_density_score (ood_signal_baseline_papers/
    density/get_density_multitaskdit.py), which already implements the same
    one-step backward-Euler z_hat(x) = x - v_theta(x, t=1, context) score (paper Eq. 6) for
    MultiTaskDiTPolicy's own FlowMatchingObjective/noise_predictor - mirrors how
    gloves_density wraps get_nonconformity_gloves.compute_density_score for GLOVES.

    Args:
        policy: trained, frozen MultiTaskDiTPolicy, configured with a FlowMatchingObjective
            (compute_density_score raises otherwise, via policy.config.is_flow_matching).
        obs_history: iterable of `n_obs_steps` obs dicts, already normalized/preprocessed
            the way multitask_dit_server.py's step handler builds them - observation.state,
            observation.images.table_cam, observation.images.wrist_cam (per-timestep,
            batch dim stripped) plus observation.language.tokens/attention_mask (one
            tokenized task string per call, constant across the window, batch dim intact).
        action: [Da], [B, Da], or [B, T, Da] action to score.
        tile_single_action: if True, a [Da]/[B, Da] action is tiled across the full
            policy.config.horizon (see compute_density_score's docstring). If False
            (default), action must already be a full [B, T, Da] chunk (T ==
            policy.config.horizon) - matches gloves_density's convention of scoring the
            model's own already-generated chunk without implicit tiling.
        return_z_hat: if True, also return ẑ(x) itself (see compute_density_score's
            docstring for why ẑ is not itself a density value).

    Returns:
        tensor [B]: s(x) for this action, or (tensor [B], tensor [B, horizon, action_dim])
        if return_z_hat=True.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    from ood_signal_baseline_papers.density.get_density_multitaskdit import (
        compute_density_score,
    )

    obs_history = list(obs_history)
    batch = {
        "observation.state": torch.stack(
            [o["observation.state"] for o in obs_history], dim=0
        ).unsqueeze(0),
        "observation.images.table_cam": torch.stack(
            [o["observation.images.table_cam"] for o in obs_history], dim=0
        ).unsqueeze(0),
        "observation.images.wrist_cam": torch.stack(
            [o["observation.images.wrist_cam"] for o in obs_history], dim=0
        ).unsqueeze(0),
        # Constant across the window - one copy from the latest step, not stacked
        # per-timestep like state/images (see obs_history's docstring above).
        OBS_LANGUAGE_TOKENS: obs_history[-1][OBS_LANGUAGE_TOKENS],
        OBS_LANGUAGE_ATTENTION_MASK: obs_history[-1][OBS_LANGUAGE_ATTENTION_MASK],
    }
    return compute_density_score(
        policy, batch, action, tile_single_action=tile_single_action, return_z_hat=return_z_hat
    )

#####################################
# -- Metric 3. Interpretation of losses and/or densities --
#####################################

## -- 1. Conformal Prediction --
def cf_prediction_score_ood(score, calibration_scores):
    """Conformal-prediction p-value for a non-conformity score, calibrated against a
    held-out calibration set of the same score computed over in-distribution/expert
    (state, action) pairs (see the paper's "we apply conformal prediction with a calibration
    set of small amount expert data on score s(x) to make p-value OOD prediction").

    This does not compute a score itself - `score` is expected to already come from one of
    dp_loss / gloves_density / etc., and `calibration_scores` from the
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

## -- 2. Score Smoothness --
def smoothness_score_ood(score_fn, state, action, prev_state, prev_action):
    """Temporal smoothness of a non-conformity score across consecutive timesteps:
    |score_fn(state, action) - score_fn(prev_state, prev_action)|. A large jump flags an
    abrupt change in how OOD the trajectory looks step-to-step, independent of the absolute
    score level.

    Args:
        score_fn: any (state, action) -> scalar score function from this file, with
            its model handle already bound, e.g.
            functools.partial(dp_loss, policy) or
            functools.partial(gloves_density, dit_flow).
        state, action: this timestep's (state, action) pair.
        prev_state, prev_action: the previous timestep's (state, action) pair.

    Returns:
        scalar (same type score_fn returns): absolute score difference between timesteps.
    """
    score = score_fn(state, action)
    prev_score = score_fn(prev_state, prev_action)
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


def perturb_score_ood(score_fn, state, action, sigma=0.05, n_perturbations=8):
    """Sensitivity of a non-conformity score to small random perturbations of `state`: how
    much does score_fn's output change if the state is nudged by isotropic Gaussian noise?
    A score that swings wildly under tiny state perturbations suggests the model's OOD
    judgment near this state is unstable, independent of the raw score value itself.

    Args:
        score_fn: see smoothness_loss_state_ood's docstring.
        state: obs dict/obs_seq or obs_history (whatever score_fn expects).
        action: action to score (held fixed - only `state` is perturbed).
        sigma: std of the isotropic Gaussian noise added to each tensor in `state`.
        n_perturbations: number of perturbed copies of `state` to average over.

    Returns:
        mean absolute score deviation, averaged over `n_perturbations` perturbed states.
    """
    base_score = score_fn(state, action)

    deviations = []
    for _ in range(n_perturbations):
        perturbed_state = _perturb_state(state, sigma)
        perturbed_score = score_fn(perturbed_state, action)
        deviations.append(abs(perturbed_score - base_score))

    return sum(deviations) / len(deviations)
