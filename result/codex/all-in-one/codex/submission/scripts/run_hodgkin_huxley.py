#!/usr/bin/env python
"""Sec. 4.4 / Fig. 7: inference with observation intervals (Hodgkin-Huxley).

1. Train a Simformer on the Hodgkin-Huxley task (7 parameters, 7 voltage summary
   statistics and the energy consumption as an additional statistic).
2. Infer the posterior given the voltage summary statistics only (Fig. 7b) and
   generate posterior predictive samples of the energy consumption with the same
   network -- without running the simulator (Fig. 7c).
3. Define an interval constraint on the energy consumption (lowest 10% quantile
   of the posterior predictive) and use *guided diffusion* to infer the
   energy constrained posterior (Fig. 7e/f).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import RESULTS, ensure_dir, generate_simulations, load_checkpoint, \
    save_checkpoint
from simformer import (IntervalUpperBound, Simformer, SimformerConfig, get_task)
from simformer.plotting import plot_energy_posterior_predictive, \
    plot_posterior_marginals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-simulations", type=int, default=100000)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mask-mode", default="directed",
                        choices=["dense", "undirected", "directed"])
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--self-recurrence", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--energy-quantile", type=float, default=0.1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=str(RESULTS / "hodgkin_huxley"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(Path(args.out))
    task = get_task("hodgkin_huxley")
    rng = np.random.default_rng(args.seed)
    n_params = task.n_params
    energy_index = n_params + task.n_voltage_stats      # the 8th statistic

    # ------------------------------------------------------------------ train
    if args.checkpoint:
        model = load_checkpoint(task, Path(args.checkpoint))
    else:
        model = Simformer(task.problem(), SimformerConfig(
            mask_mode=args.mask_mode, n_layers=args.n_layers,
            num_steps=args.num_steps, self_recurrence=args.self_recurrence,
            seed=args.seed))
        theta, x, index, metadata = generate_simulations(
            task, args.n_simulations, seed=args.seed)
        model.fit(theta, x, index, metadata, batch_size=1000, lr=1e-4,
                  max_epochs=args.max_epochs, patience=20,
                  max_steps=args.max_steps, verbose=True)
        save_checkpoint(model, out / "model.pt")

    # ------------------------------------------------------- synthetic experim.
    theta_true = task.prior_sample(1, rng)[0]
    x_true = task.simulate(theta_true[None], rng)[0]
    value = np.zeros(task.n_variables)
    state = np.zeros(task.n_variables)
    value[n_params:n_params + task.n_voltage_stats] = \
        x_true[:task.n_voltage_stats]
    state[n_params:n_params + task.n_voltage_stats] = 1.0

    samples = model.sample([(value, state)], num_samples=args.n_samples,
                           num_steps=args.num_steps)[0]
    theta_samples = samples[:, :n_params]
    energy_predictive = samples[:, energy_index]
    plot_posterior_marginals(theta_samples, task.param_names,
                             out / "posterior_marginals.png",
                             true_values=theta_true,
                             title="posterior given voltage statistics")

    # energy from *simulator* outputs of the posterior samples (Fig. 7c/d)
    simulator_energy = task.simulate(theta_samples[:200], rng)[:,
                                                               task.n_voltage_stats]
    plot_energy_posterior_predictive(energy_predictive, simulator_energy,
                                     out / "energy_predictive.png",
                                     title="posterior predictive energy")
    np.savez_compressed(out / "posterior_predictive.npz",
                        samples=samples, theta_true=theta_true, x_true=x_true)

    # -------------------------------------------------- guided diffusion (7e/f)
    threshold = float(np.quantile(energy_predictive, args.energy_quantile))
    constraint = IntervalUpperBound([energy_index], threshold)
    guided = model.sample([(value, state)], num_samples=args.n_samples,
                          num_steps=args.num_steps,
                          guidance=constraint,
                          self_recurrence=args.self_recurrence)[0]
    guided_theta = guided[:, :n_params]
    guided_energy = guided[:, energy_index]
    plot_posterior_marginals(guided_theta, task.param_names,
                             out / "posterior_marginals_guided.png",
                             true_values=theta_true,
                             title="posterior with energy constraint")
    guided_simulator_energy = task.simulate(guided_theta[:200], rng)[:,
                                                                    task.n_voltage_stats]
    plot_energy_posterior_predictive(guided_energy, guided_simulator_energy,
                                     out / "energy_predictive_guided.png",
                                     threshold=threshold,
                                     title="energy constrained posterior")
    fraction_below = float(np.mean(guided_energy <= threshold))
    print(f"[hodgkin_huxley] energy constraint = {threshold:.3f} uJ/s; "
          f"{100 * fraction_below:.1f}% of the guided samples satisfy it.")
    np.savez_compressed(out / "guided.npz", samples=guided,
                        threshold=threshold, fraction_below=fraction_below)
    print(f"[hodgkin_huxley] results written to {out}")


if __name__ == "__main__":
    main()
