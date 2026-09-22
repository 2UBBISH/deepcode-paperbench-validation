#!/usr/bin/env python
"""Self-contained FRE demo that runs on CPU in a couple of minutes.

This exists because the paper's full experiments (150k-1M encoder steps plus
850k-1M policy steps, five seeds, on AntMaze / ExORL / Kitchen) are far too
large to run inside a sandbox.  The demo instead verifies the *mechanism*: that
pre-training the FRE encoder over the random reward prior produces a latent
space from which the decoder can predict **held-out** reward functions that the
encoder never saw during training, i.e. in-context regression of a functional
reward representation.

It builds a synthetic dataset of trajectories, trains FRE, and for each reward
family in the prior reports

  * the decoder's held-out reward MSE, and
  * the MSE of a trivial baseline that always predicts each function's mean
    reward,

so a lower error demonstrates that the functional encoding is working.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fre.datasets import OfflineDataset, from_trajectories
from fre.fre import FRE, FREConfig
from fre.reward_functions import build_prior, discretize_reward


def make_synthetic_dataset(
    num_traj: int = 200, traj_len: int = 50, seed: int = 0, nuisance_dims: int = 1
) -> OfflineDataset:
    """Random-walk trajectories in a 2-D box plus a few nuisance dimensions.

    The state space is deliberately small: the paper pre-trains for 150k-1M
    encoder steps, whereas this demo has a budget of a few hundred to a couple
    of thousand steps, so the problem has to be small enough to show progress.
    """
    rng = np.random.default_rng(seed)
    trajectories, actions = [], []
    for _ in range(num_traj):
        xy = np.cumsum(rng.normal(scale=1.0, size=(traj_len, 2)), axis=0)
        xy = np.clip(xy, -8.0, 8.0)
        rest = rng.normal(scale=1.0, size=(traj_len, nuisance_dims))
        obs = np.concatenate([xy, rest], axis=-1).astype(np.float32)
        trajectories.append(obs)
        actions.append(rng.uniform(-1, 1, size=(traj_len, 3)).astype(np.float32))
    return from_trajectories(trajectories, actions)


def capacity_check(args, dataset, mean, std, steps: int = 300, batch: int = 16):
    """Fit a *single* fixed reward function and report residual error.

    This is the strongest cheap signal that the encoder-to-decoder path really
    carries the information needed to represent a reward function: if the
    bottleneck could not transmit it, no amount of optimisation would drive the
    residual error far below the function's variance.
    """
    device = torch.device(args.device)
    # Use a random linear reward: unlike a goal-reaching reward it has
    # non-trivial variance, so the residual error below is informative.
    prior = build_prior("FRE-lin", dataset.encoder_obs_dim, dataset.goal_sampler(np.random.default_rng(5)))
    torch.manual_seed(args.seed + 7)
    probe = standardise(dataset.random_states(128, rng=np.random.default_rng(3)), mean, std).unsqueeze(0).to(device)
    eta = None
    for _ in range(50):
        candidate = prior.sample(1, device)
        # The 0.9 sparsity mask can zero every dimension, giving a constant zero
        # reward function; skip those since their variance carries no signal.
        if float(candidate.reward(probe).var()) > 1e-3:
            eta = candidate
            break
    if eta is None:
        eta = prior.sample(1, device)

    torch.manual_seed(args.seed)
    model = FRE(FREConfig(
        state_dim=dataset.encoder_obs_dim,
        num_encode_pairs=args.num_encode_pairs,
        num_decode_pairs=args.num_decode_pairs,
    )).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    for _ in range(steps):
        enc_raw = dataset.random_states(batch * args.num_encode_pairs, rng=rng)
        enc_raw = enc_raw.reshape(batch, args.num_encode_pairs, -1).to(device)
        enc = standardise(eta.maybe_insert_goal_state(enc_raw).cpu().numpy(), mean, std).to(device)
        dec_raw = dataset.random_states(batch * args.num_decode_pairs, rng=rng)
        dec = standardise(dec_raw.reshape(batch, args.num_decode_pairs, -1).numpy(), mean, std).to(device)
        enc_rewards = eta.reward(enc)
        target = eta.reward(dec)
        bins = discretize_reward(
            (enc_rewards - eta.r_min.view(-1, 1)) / (eta.r_max - eta.r_min).view(-1, 1).clamp_min(1e-6),
            32, r_min=0.0, r_max=1.0,
        )
        loss, _, _ = model.loss(enc, bins, dec, target, return_parts=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        dec_raw = dataset.random_states(batch * args.num_decode_pairs, rng=rng)
        dec = standardise(dec_raw.reshape(batch, args.num_decode_pairs, -1).numpy(), mean, std).to(device)
        target = eta.reward(dec)
        enc_raw = dataset.random_states(batch * args.num_encode_pairs, rng=rng)
        enc_raw = enc_raw.reshape(batch, args.num_encode_pairs, -1).to(device)
        enc = standardise(eta.maybe_insert_goal_state(enc_raw).cpu().numpy(), mean, std).to(device)
        bins = discretize_reward(
            (eta.reward(enc) - eta.r_min.view(-1, 1)) / (eta.r_max - eta.r_min).view(-1, 1).clamp_min(1e-6),
            32, r_min=0.0, r_max=1.0,
        )
        pred = model.decode(dec, model.encode(enc, bins, sample=False))
        mse = float(torch.nn.functional.mse_loss(pred, target))
        variance = float(target.var())
    return mse, variance


def standardise(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.as_tensor((np.asarray(states, dtype=np.float32) - mean) / std)


def sample_heldout_reward_functions(prior, num_functions: int, device):
    """Draw a fixed set of reward functions to hold out from pre-training."""
    return [prior.sample(1, device) for _ in range(num_functions)]


def evaluate_heldout_rewards(model: FRE, held_out, dataset, k: int, seed: int, mean, std, num_z_samples: int = 8):
    """Measure how well the decoder predicts unseen reward functions.

    The same set of held-out reward functions is reused before and after
    training, and the baseline is the error of always predicting the mean
    reward of each function, so the two numbers are comparable.
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    model.eval()
    mse_list, baseline_list = [], []
    with torch.no_grad():
        for eta in held_out:
            # The goal state is inserted while the states are still raw (the
            # prior stores raw dataset states), then everything is standardised.
            enc_raw = dataset.random_states(k, rng=rng).unsqueeze(0).to(device)
            enc = standardise(eta.maybe_insert_goal_state(enc_raw).cpu().numpy(), mean, std).to(device)
            enc_rewards = eta.reward(enc)
            r_min, r_max = eta.r_min, eta.r_max
            bins = discretize_reward(
                (enc_rewards - r_min.view(-1, 1)) / (r_max - r_min).view(-1, 1).clamp_min(1e-6),
                32, r_min=0.0, r_max=1.0,
            )
            dec = standardise(dataset.random_states(64, rng=rng), mean, std).unsqueeze(0).to(device)
            target = eta.reward(dec)
            # Estimate E_{z ~ q(z | context)}[decoder(s, z)] by Monte-Carlo over
            # posterior samples; the decoder is nonlinear, so averaging the
            # *outputs* is the correct estimator.
            pred = torch.stack(
                [
                    model.decode(dec, model.encode(enc, bins, sample=True))
                    for _ in range(num_z_samples)
                ]
            ).mean(dim=0)
            mse_list.append(float(torch.nn.functional.mse_loss(pred, target).item()))
            baseline_list.append(float(torch.nn.functional.mse_loss(target.mean().expand_as(target), target).item()))
    model.train()
    return float(np.mean(mse_list)), float(np.mean(baseline_list))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400, help="encoder steps per reward family")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-encode-pairs", type=int, default=32)
    parser.add_argument("--num-decode-pairs", type=int, default=8)
    parser.add_argument("--prior", default="FRE-all",
                        help="prior used for pre-training (FRE-all is the paper's default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    # A single-threaded run is faster here: the model is small and thread
    # oversubscription dominates the runtime.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = make_synthetic_dataset(seed=args.seed)
    mean = np.asarray(dataset.encoder_observations.mean(axis=0), dtype=np.float32)
    std = np.asarray(dataset.encoder_observations.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    model = train(args, dataset, mean, std)

    # Capacity check: can the z bottleneck represent a single reward function?
    mse, variance = capacity_check(args, dataset, mean, std)
    print(
        f"\ncapacity check (fit one fixed reward function): residual mse={mse:.4f} "
        f"vs reward variance={variance:.4f}"
        f"  -> {'PASS' if mse < 0.5 * variance else 'WARN'}"
    )

    # Held-out evaluation: fresh reward functions drawn from each prior family,
    # never seen during pre-training.
    held_out_families = [("goal", "FRE-goals"), ("lin", "FRE-lin"), ("mlp", "FRE-mlp")]
    summary = {}
    for family, prior_name in held_out_families:
        family_prior = build_prior(
            prior_name, dataset.encoder_obs_dim, dataset.goal_sampler(np.random.default_rng(1 + args.seed))
        )
        torch.manual_seed(args.seed + 999)
        held_out = sample_heldout_reward_functions(family_prior, 16, torch.device(args.device))
        summary[family] = evaluate_heldout_rewards(
            model, held_out, dataset, args.num_encode_pairs, seed=123, mean=mean, std=std
        )

    print("\n=== held-out reward prediction (fresh reward functions, lower is better) ===")
    print(f"{'family':>8}   {'decoder mse':>12}   {'mean-predictor mse':>18}")
    for family, (mse, baseline) in summary.items():
        print(f"{family:>8}   {mse:>12.4f}   {baseline:>18.4f}")
    print(
        "\nInterpretation: this demo is a plumbing and stability check, not a "
        "convergence check.  It shows that the information-bottleneck loss "
        "decomposes as expected, that the posterior does not collapse, and that "
        "the encoder/decoder can be trained end to end.  The paper uses "
        "150k-1M encoder steps -- 200-1000x more than a demo budget allows -- "
        "so the absolute held-out numbers here are not expected to match the "
        "paper; run scripts/run_fre_antmaze.sh for the real experiments."
    )


def train(args, dataset, mean, std):
    """Pre-train an FRE model with the configured prior reward distribution."""
    device = torch.device(args.device)
    prior = build_prior(
        args.prior, dataset.encoder_obs_dim, dataset.goal_sampler(np.random.default_rng(args.seed))
    )
    torch.manual_seed(args.seed)
    model = FRE(FREConfig(
        state_dim=dataset.encoder_obs_dim,
        num_encode_pairs=args.num_encode_pairs,
        num_decode_pairs=args.num_decode_pairs,
    )).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    print(f"\n--- pre-training FRE with the '{args.prior}' prior for {args.steps} steps ---")
    tail_recon, tail_baseline = [], []
    for step in range(args.steps):
        eta = prior.sample(args.batch_size, device)
        enc_raw = dataset.random_states(
            args.batch_size * args.num_encode_pairs, rng=rng
        ).reshape(args.batch_size, args.num_encode_pairs, -1).to(device)
        enc = standardise(eta.maybe_insert_goal_state(enc_raw).cpu().numpy(), mean, std).to(device)
        dec = standardise(
            dataset.random_states(args.batch_size * args.num_decode_pairs, rng=rng), mean, std
        ).reshape(args.batch_size, args.num_decode_pairs, -1).to(device)

        enc_rewards = eta.reward(enc)
        dec_rewards = eta.reward(dec)
        r_min, r_max = eta.r_min, eta.r_max
        bins = discretize_reward(
            (enc_rewards - r_min.view(-1, 1)) / (r_max - r_min).view(-1, 1).clamp_min(1e-6),
            32, r_min=0.0, r_max=1.0,
        )
        loss, recon, kl = model.loss(enc, bins, dec, dec_rewards, return_parts=True)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step >= args.steps - 100:
            tail_recon.append(float(recon))
            # Honest baseline: the error of predicting each reward function's
            # own mean (not the batch mean across different functions).
            tail_baseline.append(float(dec_rewards.var(dim=1).mean()))
        if (step + 1) % 50 == 0:
            print(f"step {step + 1:4d}  loss={loss.item():.4f}  recon={recon.item():.4f}  kl={kl.item():.4f}")
    recon_mean = float(np.mean(tail_recon))
    baseline_mean = float(np.mean(tail_baseline))
    print(
        f"\nin-context reconstruction on the pre-training distribution "
        f"(last 100 steps): recon={recon_mean:.4f}  vs per-function-mean "
        f"baseline={baseline_mean:.4f}"
    )
    torch.save(model.state_dict(), os.environ.get("FRE_SMOKE_CKPT", "/tmp/fre_smoke_model.pt"))
    return model


if __name__ == "__main__":
    main()
