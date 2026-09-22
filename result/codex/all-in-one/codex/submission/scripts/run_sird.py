#!/usr/bin/env python
"""Sec. 4.3 / Fig. 6: inference in an infinite dimensional parameter space (SIRD).

Inference is performed for the global parameters (recovery / death rate) and the
*time dependent* contact rate ``beta(t)``; the Simformer is queried at an
arbitrary grid of time points and the expected coverage is computed for the
posterior of randomly selected time points (as in Appendix Fig. A13).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from common import RESULTS, ensure_dir, generate_simulations, load_checkpoint, \
    save_checkpoint
from simformer import Simformer, SimformerConfig, get_task
from simformer.metrics import expected_coverage
from simformer.plotting import plot_function_posterior, plot_posterior_marginals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-simulations", type=int, default=100000)
    parser.add_argument("--n-beta", type=int, default=10)
    parser.add_argument("--n-obs", type=int, default=6)
    parser.add_argument("--inference-n-beta", type=int, default=21)
    parser.add_argument("--inference-n-obs", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mask-mode", default="directed",
                        choices=["dense", "undirected", "directed"])
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-coverage", type=int, default=20)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=str(RESULTS / "sird"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(Path(args.out))
    task = get_task("sird", n_beta=args.n_beta, n_obs=args.n_obs)
    rng = np.random.default_rng(args.seed)

    if args.checkpoint:
        model = load_checkpoint(task, Path(args.checkpoint))
    else:
        model = Simformer(task.problem(), SimformerConfig(
            mask_mode=args.mask_mode, n_layers=args.n_layers,
            num_steps=args.num_steps, seed=args.seed))
        theta, x, index, metadata = generate_simulations(
            task, args.n_simulations, seed=args.seed)
        model.fit(theta, x, index, metadata, batch_size=1000, lr=1e-4,
                  max_epochs=args.max_epochs, patience=20,
                  max_steps=args.max_steps, verbose=True)
        save_checkpoint(model, out / "model.pt")

    # The trained transformer is a set function over tokens: the same network is
    # queried on a finer grid of contact rate time points (Sec. 4.3).
    inference_problem = task.inference_problem(args.inference_n_beta,
                                               args.inference_n_obs)
    model.set_problem(inference_problem)

    # ------------------------------------------------------------- ground truth
    gamma_true, mu_true = 0.2, 0.05
    beta_times = task.default_beta_times(args.inference_n_beta)
    beta_hat_true = 1.5 * np.sin(2 * np.pi * beta_times / 60.0) + 0.5
    beta_true = 1.0 / (1.0 + np.exp(-beta_hat_true))
    obs_times = np.linspace(5.0, 55.0, args.inference_n_obs)
    observations = task.synthetic_observation(
        np.array([gamma_true, mu_true]), beta_true, beta_times, obs_times,
        rng=rng)

    n_global = task.n_global_params
    n_params = inference_problem.n_params
    n_variables = inference_problem.n_variables
    value = np.zeros(n_variables)
    state = np.zeros(n_variables)
    # condition on the observations of I, R and D
    for k in range(3):
        start = n_params + k * args.inference_n_obs
        value[start:start + args.inference_n_obs] = observations[:, k]
        state[start:start + args.inference_n_obs] = 1.0
    index = np.zeros(n_variables)
    index[n_global:n_params] = beta_times
    for k in range(3):
        start = n_params + k * args.inference_n_obs
        index[start:start + args.inference_n_obs] = obs_times

    samples = model.sample([(value, state)], num_samples=args.n_samples,
                           num_steps=args.num_steps, index=index[None])[0]
    global_samples = samples[:, :n_global]
    beta_samples = samples[:, n_global:n_params]

    plot_posterior_marginals(global_samples, ["gamma", "mu"],
                             out / "global_posterior.png",
                             true_values=np.array([gamma_true, mu_true]),
                             title="global parameters (SIRD)")
    plot_function_posterior(beta_samples, beta_times,
                            out / "contact_rate_posterior.png",
                            true_values=beta_true,
                            title="time dependent contact rate")
    np.savez_compressed(out / "posterior_samples.npz", samples=samples,
                        beta_times=beta_times, obs_times=obs_times,
                        observations=observations, beta_true=beta_true)

    # ------------------------------------------------- coverage (Sec. 4.3/A13)
    if args.n_coverage > 0:
        coverages = coverage_analysis(model, task, args)
        np.savez_compressed(out / "coverage.npz", **coverages)
        print("[sird] expected coverage of the global parameters: "
              f"{np.round(coverages['global'], 3).tolist()}")
        print("[sird] expected coverage of the contact rate: "
              f"{np.round(coverages['beta'], 3).tolist()}")
    print(f"[sird] results written to {out}")


def coverage_analysis(model, task, args, n_samples: Optional[int] = None):
    """Expected coverage of the posterior (Appendix Fig. A13).

    For ``n_coverage`` synthetic data sets we sample the posterior with the
    Simformer and compute the rank of the ground truth parameter / contact rate /
    observation within the posterior samples.  A well calibrated model has
    ``coverage(alpha) = alpha``.
    """
    from simformer.metrics import expected_coverage

    rng = np.random.default_rng(args.seed + 1)
    problem = model.problem
    n_global = task.n_global_params
    n_params = problem.n_params
    n_obs = args.inference_n_obs
    n_beta = args.inference_n_beta
    beta_times = task.default_beta_times(n_beta)
    n_repeats = args.n_coverage
    n_samples = n_samples or max(200, args.n_samples // 10)
    global_samples = np.empty((n_repeats, n_samples, n_global))
    global_true = np.empty((n_repeats, n_global))
    beta_samples = np.empty((n_repeats, n_samples))
    beta_true_values = np.empty(n_repeats)
    for rep in range(n_repeats):
        global_true[rep] = task.prior_sample(1, rng)[0]
        beta_true = task.sample_beta(beta_times, 1, rng)[0]
        obs_times = np.sort(rng.uniform(0.0, task.t_end, size=n_obs))
        observations = task.synthetic_observation(
            global_true[rep], beta_true, beta_times, obs_times, rng=rng)
        value = np.zeros(problem.n_variables)
        state = np.zeros(problem.n_variables)
        for k in range(3):
            start = n_params + k * n_obs
            value[start:start + n_obs] = observations[:, k]
            state[start:start + n_obs] = 1.0
        index = np.zeros(problem.n_variables)
        index[n_global:n_params] = beta_times
        for k in range(3):
            start = n_params + k * n_obs
            index[start:start + n_obs] = obs_times
        samples = model.sample([(value, state)], num_samples=n_samples,
                               num_steps=model.config.num_steps,
                               index=index[None])[0]
        global_samples[rep] = samples[:, :n_global]
        beta_samples[rep] = samples[:, n_global:n_params].mean(axis=-1)
        beta_true_values[rep] = beta_true.mean()
    return {
        "global": expected_coverage(global_samples, global_true),
        "beta": expected_coverage(beta_samples[..., None],
                                  beta_true_values[:, None]),
        "alphas": np.linspace(0.05, 0.95, 10),
    }


if __name__ == "__main__":
    main()
