#!/usr/bin/env python
"""Sec. 4.2 / Fig. 5: inference with unstructured observations (Lotka-Volterra).

The script

1. trains a Simformer on ``1e5`` simulations of the Lotka-Volterra model
   (if no checkpoint is given),
2. creates two synthetic observation scenarios: (a) four prey measurements
   placed irregularly in time and (b) the same with nine additional predator
   measurements,
3. infers the posterior / posterior predictive for both scenarios with the
   *same* trained network (no additional simulator runs), and
4. evaluates the posterior with C2ST against MCMC reference samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import RESULTS, ensure_dir, generate_simulations, load_checkpoint, \
    save_checkpoint
from simformer import Simformer, SimformerConfig, get_task
from simformer.plotting import plot_posterior_predictive, plot_posterior_marginals
from simformer.reference import sample_conditional
from simformer.metrics import c2st_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-simulations", type=int, default=100000)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mask-mode", default="directed",
                        choices=["dense", "undirected", "directed"])
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-reference", type=int, default=500,
                        help="MCMC chains for the ground truth posterior")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=str(RESULTS / "lotka_volterra"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(Path(args.out))
    task = get_task("lotka_volterra")
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

    # ---------------------------------------------------------------- scenario
    theta_true = task.prior_sample(1, rng)[0]
    scenarios = {"prey_only": dict(n_prey=4, n_predator=0),
                 "prey_and_predator": dict(n_prey=4, n_predator=9)}
    trajectories = {}
    for name, kwargs in scenarios.items():
        x_obs, value, state = task.unstructured_observation(
            theta_true, rng=rng, **kwargs)
        samples = model.sample([(value, state)], num_samples=args.n_samples,
                               num_steps=args.num_steps)[0]
        theta_samples = samples[:, :task.n_params]
        posterior_predictive = samples[:, task.n_params:]

        observed_prey = state[task.n_params:task.n_params + task.n_grid] > 0.5
        prey_indices = np.flatnonzero(observed_prey)
        plot_posterior_predictive(
            posterior_predictive[:, :task.n_grid],
            value[task.n_params:task.n_params + task.n_grid][observed_prey],
            prey_indices, out / f"posterior_predictive_prey_{name}.png",
            title=f"prey (Simformer, {name.replace('_', ' ')})")
        plot_posterior_marginals(
            theta_samples, ["alpha", "beta", "delta", "gamma"],
            out / f"posterior_{name}.png", true_values=theta_true,
            title=f"posterior ({name.replace('_', ' ')})")
        trajectories[name] = dict(samples=samples, theta_samples=theta_samples,
                                  value=value, state=state, x_obs=x_obs)

        # ------- MCMC ground truth posterior for the C2ST evaluation (Fig. 5c)
        if args.n_reference > 0:
            theta_ref, _ = sample_conditional(
                task, state, value, n_samples=args.n_reference, rng=rng,
                n_chains=args.n_reference)
            c2st = c2st_accuracy(theta_samples, theta_ref)
            print(f"[lotka_volterra] {name}: C2ST of the posterior = {c2st:.3f}")
            np.savez_compressed(out / f"reference_{name}.npz",
                                theta_ref=theta_ref, theta_samples=theta_samples,
                                value=value, state=state, c2st=c2st)
    print(f"[lotka_volterra] results written to {out}")


if __name__ == "__main__":
    main()
