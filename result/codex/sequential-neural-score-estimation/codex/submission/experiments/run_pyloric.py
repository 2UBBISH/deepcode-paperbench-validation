#!/usr/bin/env python3
"""Real-world neuroscience experiment: pyloric network (Section 5.3, Figure 4).

Applies TSNPSE to the 31-parameter pyloric network model of the stomatogastric
ganglion of *Cancer borealis* (Prinz et al., 2003, 2004), inferring the
posterior given experimental observations (Haddad & Marder, 2021).

Following Section 5.3 and Appendix E.2:

* the same score network architecture is used as in the benchmark experiments;
* inference runs over 9 rounds, with 30000 initial simulations and 20000 added
  simulations per round;
* the VP SDE is used as the forward noising process;
* invalid summary statistics are replaced by a value two standard deviations
  below the prior predictive of the corresponding statistic;
* we report the percentage of valid summary statistics per round (Figure 4c)
  and draw a posterior predictive sample (Figure 4a).

Requires the ``pyloric`` package (see ``scripts/setup_third_party.sh``) and a
working NEURON installation.

Example
-------
::

    python experiments/run_pyloric.py --output-dir results/pyloric --rounds 9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.pyloric_adapter import PyloricProblem  # noqa: E402
from snpse import TSNPSE, TSNPSEConfig, TrainingConfig  # noqa: E402


class ValidityTrackingSimulator:
    """Wraps the pyloric simulator and records the fraction of valid summary
    statistics (Figure 4c)."""

    def __init__(self, problem: PyloricProblem) -> None:
        self.problem = problem
        self.valid = np.ones(problem.dim_summary_statistics, dtype=bool)
        self.history = []

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        stats, valid = self.problem.simulate_batch(theta)
        if self.problem._replacement_values is None:
            self.problem.fit_prior_predictive()
        stats = torch.as_tensor(
            np.where(valid, stats.numpy(), self.problem._replacement_values), dtype=torch.float32
        )
        self.history.append(
            {
                "num_simulations": int(theta.shape[0]),
                "fraction_valid": float(valid.mean()),
            }
        )
        return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--num-initial", type=int, default=30000)
    p.add_argument("--sims-per-round", type=int, default=20000)
    p.add_argument("--num-posterior-samples", type=int, default=10000)
    p.add_argument("--num-predictives", type=int, default=100)
    p.add_argument("--sde", type=str, default="vp", choices=["vp", "ve"])
    p.add_argument("--num-prior-predictive", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="results/pyloric")
    p.add_argument("--smoke", action="store_true", help="tiny settings for a quick test")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    problem = PyloricProblem(dim_summary_statistics=18)
    if args.smoke:
        args.rounds = 2
        args.num_initial = 20
        args.sims_per_round = 20
        args.num_posterior_samples = 50
        args.num_predictives = 5
        args.num_prior_predictive = 20

    print("Estimating the prior predictive of the summary statistics ...")
    problem.fit_prior_predictive(args.num_prior_predictive, seed=args.seed)

    tracking_simulator = ValidityTrackingSimulator(problem)

    config = TSNPSEConfig(
        num_rounds=args.rounds,
        num_simulations=args.sims_per_round,
        num_simulations_initial=args.num_initial,
        hpr_num_samples=2000 if args.smoke else 20000,
    )
    training = TrainingConfig(seed=args.seed, batch_size=500)
    if args.smoke:
        training.max_iters = 60
        training.patience = 20
    estimator = TSNPSE(
        dim_parameters=problem.dim_parameters,
        dim_data=problem.dim_summary_statistics,
        prior=problem.numerical_prior,
        simulator=tracking_simulator,
        sde=args.sde,
        sigma_min=0.05,
        config=config,
        training_config=training,
        seed=args.seed,
    )

    t0 = time.time()
    estimator.run(problem.observation)
    elapsed = time.time() - t0

    # ---- posterior predictive sample (Figure 4a) ---------------------------
    samples = estimator.npse.sample(problem.observation, args.num_posterior_samples)
    with torch.no_grad():
        predictive, valid = problem.simulate_batch(samples[: args.num_predictives])

    results = {
        "rounds": args.rounds,
        "num_initial": args.num_initial,
        "sims_per_round": args.sims_per_round,
        "sde": args.sde,
        "seconds": elapsed,
        "diagnostics": estimator.diagnostics,
        "simulator_validity": tracking_simulator.history,
        "fraction_valid_by_round": _valid_fraction_by_round(tracking_simulator.history, args),
        "posterior_predictive_valid_fraction": float(valid.mean()),
        "posterior_mean": samples.mean(0).tolist(),
        "observation": problem.observation.reshape(-1).tolist(),
        "posterior_predictive": predictive.numpy().tolist(),
    }
    out = os.path.join(args.output_dir, "pyloric_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    np.save(os.path.join(args.output_dir, "posterior_samples.npy"), samples.numpy())
    print(f"Saved results to {out}")
    print(f"Final fraction of valid summary statistics: {results['fraction_valid_by_round'][-1]:.3f}")
    return 0


def _valid_fraction_by_round(history, args):
    """Aggregate the per-batch validity fractions into per-round values."""
    out = []
    idx = 0
    sizes = [args.num_initial] + [args.sims_per_round] * (args.rounds - 1)
    for size in sizes:
        acc, n = 0.0, 0
        while n < size and idx < len(history):
            acc += history[idx]["fraction_valid"] * history[idx]["num_simulations"]
            n += history[idx]["num_simulations"]
            idx += 1
        out.append(acc / max(1, n))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
