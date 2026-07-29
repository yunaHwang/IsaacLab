"""Diffusion-policy noise-prediction loss utilities.

Deliberately has no isaaclab imports, so it runs standalone off an hdf5 dataset without
booting Isaac Sim. Run this file directly to compute an in-distribution loss range from a
dataset, for later use as run_policy.py's --loss_range:

    python diffusion_loss.py --checkpoint model.pth --hdf5_dataset data.hdf5 \
        --save_file_name losses
"""

import numpy as np
import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils

DEFAULT_OBS_KEYS = ["eef_pos", "gripper_pos", "object", "eef_quat"]


# NOTE. should implement only pure state, too.
def compute_diffusion_loss(policy, obs_seq, action, device=None):
    """Score a candidate action under a robomimic DiffusionPolicyUNet's noise-prediction
    objective, without training or modifying robomimic's installed source.

    Reproduces the inference-relevant half of DiffusionPolicyUNet.train_on_batch (obs
    encoding -> add noise to the action -> predict the noise -> MSE), skipping the
    optimizer/EMA update, so it can be called as a pure "how OOD is this state-conditioned action" score.

    This is a single (noise, timestep) draw, no averaging. For a fixed dataset of
    (obs, action) pairs (see compute_dataset_loss_distribution below) one draw per pair is
    enough. For scoring the same live action repeatedly (diffdagger's Nb-sample averaging,
    e.g. Nb=512), call this in a loop and average externally - see run_policy.py.

    Note: the diffusion policy scores an action *chunk* of length Tp conditioned on To
    past observations, not a single timestep. A single-step action gets tiled across
    the prediction horizon; pass a real [B, Tp, Da] action history for a more faithful
    score if you have one buffered (e.g. the last Tp blended/executed actions).

    Args:
        policy: robomimic RolloutPolicy (or bare DiffusionPolicyUNet algo) as returned
            by FileUtils.policy_from_checkpoint.
        obs_seq (dict): {key: tensor [B, To, ...]} - windowed observation history in the
            same format the policy itself consumes.
        action (tensor): [Da], [B, Da], or [B, Tp, Da].

    Returns:
        tensor [B]: noise-prediction MSE loss per batch element, for this one draw.
    """
    algo = policy.policy if hasattr(policy, "policy") else policy
    nets = algo.ema.averaged_model if algo.ema is not None else algo.nets
    device = device or algo.device

    Tp = algo.algo_config.horizon.prediction_horizon
    action_dim = algo.ac_dim

    was_training = nets.training
    nets.eval()

    with torch.no_grad():
        for k in algo.obs_shapes:
            assert obs_seq[k].ndim - 2 == len(algo.obs_shapes[k]), (
                f"obs_seq['{k}'] must be [B, To, ...], got shape {tuple(obs_seq[k].shape)}"
            )

        # obs encoding + conditioning, mirrors DiffusionPolicyUNet.train_on_batch
        inputs = {"obs": obs_seq, "goal": None}
        obs_features = TensorUtils.time_distributed(
            inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
        )  # [B, To, D]
        obs_cond = obs_features.flatten(start_dim=1)  # [B, To*D]
        B = obs_cond.shape[0]

        action = action.to(device=device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)  # [Da] -> [1, Da]
        if action.ndim == 2:
            # single-step action -> tile across the prediction horizon
            action = action.unsqueeze(1).expand(-1, Tp, -1).contiguous()
        assert action.shape == (B, Tp, action_dim), (
            f"expected action shape ({B},{Tp},{action_dim}), got {tuple(action.shape)}"
        )

        noise = torch.randn(action.shape, device=device)
        timesteps = torch.randint(
            0, algo.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        noisy_action = algo.noise_scheduler.add_noise(action, noise, timesteps)
        noise_pred = nets["policy"]["noise_pred_net"](
            noisy_action, timesteps, global_cond=obs_cond
        )
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2))  # [B]

    return loss


def compute_dataset_loss_distribution(
    policy,
    dataset_path,
    obs_keys,
    observation_horizon,
    save_to_file=False,
    save_file_name=None,
):
    """Compute the diffusion loss for every (obs, action) step of every demo in an hdf5
    dataset, to establish an in-distribution loss range to compare live rollout losses
    against (mirrors diffdagger's build_cdfs()/reference-CSV approach).

    Each (obs, action) pair here is fixed (it's straight from the dataset), so this uses
    a single (noise, timestep) draw per step rather than diffdagger's Nb-sample averaging
    - that averaging matters when repeatedly re-scoring the same live action, which is
    run_policy.py's job, not this offline sweep's.

    Assumes the dataset was written with the same obs keys as the live env observation
    (demo["obs"][key] of shape [T, ...]) and demo["actions"] of shape [T, Da]. Adjust the
    windowing below if your hdf5 layout differs.
    """
    import h5py
    import pandas as pd

    To = observation_horizon
    all_demo_losses = {}

    with h5py.File(dataset_path, "r") as f:
        data = f["data"]
        for demo_name in data.keys():
            demo = data[demo_name]
            actions = np.array(demo["actions"])  # [T, Da]
            obs_group = demo["obs"]

            T = actions.shape[0]
            demo_losses = []
            for t in range(T):
                lo = max(0, t - To + 1)
                obs_window = {}
                for k in obs_keys:
                    vals = np.array(obs_group[k][lo : t + 1])  # [<=To, ...]
                    if vals.shape[0] < To:
                        # pad the start of the episode by repeating the first frame
                        pad = np.repeat(vals[:1], To - vals.shape[0], axis=0)
                        vals = np.concatenate([pad, vals], axis=0)
                    obs_window[k] = torch.from_numpy(vals).unsqueeze(0).float()  # [1, To, ...]

                action_t = torch.from_numpy(actions[t]).float()
                loss = compute_diffusion_loss(policy, obs_window, action_t)
                demo_losses.append(loss.item())

            all_demo_losses[demo_name] = demo_losses

    if save_to_file:
        assert save_file_name, "save_file_name must be set when save_to_file=True"
        # demos may have different lengths, so build columns from a dict of ragged lists
        all_demo_losses_df = pd.DataFrame.from_dict(all_demo_losses, orient="index").T
        all_demo_losses_df.to_csv(save_file_name + ".csv", index=False)

    return all_demo_losses


class LossCDF:
    """Empirical CDF over a reference loss distribution.

    Callable percentile lookup: loss_cdf(value) returns the fraction of reference losses
    <= value, i.e. how "in-distribution" a live loss is relative to the reference sweep.
    Self-contained numpy implementation of the same role as diffdagger's util.cdf.CDF.
    """

    def __init__(self, reference_losses):
        self._sorted = np.sort(np.asarray(reference_losses, dtype=np.float64))

    def __call__(self, value):
        # NOTE: calling LossCDF instance with some loss values, hit __Call__ and then returns percentile values directly
        # NOTE (super-unrelated to this, but related eureka with model(input) == forward): https://medium.com/@ashishpandey2062/inside-pytorchs-nn-module-forward-pass-call-and-composition-482d9772c10c

        return float(np.searchsorted(self._sorted, value, side="right") / self._sorted.size) # hoho numpy.searchsorted() is so fun!!

    @property
    def min(self):
        return float(self._sorted[0])

    @property
    def max(self):
        return float(self._sorted[-1])

def load_loss_reference(csv_path):
    """Load a loss-distribution CSV produced by compute_dataset_loss_distribution (one
    column per demo, one row per timestep, ragged demos padded with NaN) and return it as
    a LossCDF for percentile lookups against live losses.
    """
    import pandas as pd

    values = pd.read_csv(csv_path).to_numpy().flatten()
    values = values[~np.isnan(values)]
    return LossCDF(values)


def main():
    import argparse

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    parser = argparse.ArgumentParser(
        description=(
            "Compute a robomimic diffusion policy's noise-prediction loss over an hdf5 "
            "dataset (no Isaac Sim required) and save it as a reference loss distribution."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Robomimic diffusion policy checkpoint.")
    parser.add_argument("--hdf5_dataset", type=str, required=True, help="hdf5 dataset to grab obs/actions from.") # TODO: add options, mkdir if needed, one ID and the other OOD
    parser.add_argument(
        "--obs_keys", type=str, nargs="+", default=DEFAULT_OBS_KEYS,
        help="Observation keys to read from the dataset - must match the live env obs.",
    )
    parser.add_argument(
        "--observation_horizon", type=int, default=2,
        help="Number of past observation frames (To) the policy conditions on.",
    )
    parser.add_argument(
        "--save_file_name", type=str, required=True,
        help="Output CSV path (without .csv extension). This is what run_policy.py's --loss_range reads back in.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: auto-detect CUDA).")
    args = parser.parse_args()

    device = args.device or TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=device)

    all_demo_losses = compute_dataset_loss_distribution(
        policy,
        args.hdf5_dataset,
        args.obs_keys,
        args.observation_horizon,
        save_to_file=True,
        save_file_name=args.save_file_name,
    )
    print(f"[INFO] Computed diffusion loss distribution over {len(all_demo_losses)} demos from {args.hdf5_dataset}")
    print(f"[INFO] Saved to {args.save_file_name}.csv")


if __name__ == "__main__":
    main()
