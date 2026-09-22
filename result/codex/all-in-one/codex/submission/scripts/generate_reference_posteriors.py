#!/usr/bin/env python
"""Generate ground truth samples with MCMC (Appendix A2.2).

Two kinds of reference samples are generated:

* ``--mode posterior``: for 10 synthetic observations per task, samples of the
  ground truth posterior ``p(theta | x = x_obs)`` (used for Fig. 4a),
* ``--mode conditional``: for 100 randomly selected conditionals of the joint
  distribution (a random subset of the variables is conditioned on values drawn
  from the joint), samples of the corresponding conditional (used for Fig. 4b).

The MCMC settings follow the paper: random direction slice sampling + MH for the
benchmark tasks and HMC for the tree / HMM tasks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import RESULTS, ensure_dir
from simformer import get_task
from simformer.reference import sample_conditional


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all")
    parser.add_argument("--mode", default="posterior",
                        choices=["posterior", "conditional", "both"])
    parser.add_argument("--n-observations", type=int, default=10)
    parser.add_argument("--n-conditionals", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def reference_posteriors(task, n_observations: int, n_samples: int,
                         seed: int = 0, n_chains=None, verbose=True):
    """Reference posterior samples for ``n_observations`` synthetic data sets."""
    rng = np.random.default_rng(seed)
    theta_true = task.prior_sample(n_observations, rng)
    x_obs = task.simulate(theta_true, rng)
    samples = np.empty((n_observations, n_samples, task.n_variables))
    for i in range(n_observations):
        value, state = task.posterior_condition(theta_true[i], x_obs[i])
        theta_samples, x_samples = sample_conditional(
            task, state, value, n_samples=n_samples, rng=rng,
            n_chains=n_chains or n_samples, verbose=False)
        samples[i] = np.concatenate([theta_samples, x_samples], axis=-1)
        if verbose:
            print(f"  posterior {i + 1}/{n_observations}")
    return theta_true, x_obs, samples


def reference_conditionals(task, n_conditionals: int, n_samples: int,
                           seed: int = 0, min_conditioned: int = 1,
                           verbose=True):
    """Reference samples of ``n_conditionals`` randomly selected conditionals."""
    rng = np.random.default_rng(seed)
    joint_theta, joint_x, _, _ = task.joint_sample(n_conditionals, rng)
    joint = np.concatenate([joint_theta, joint_x], axis=-1)
    values, states, samples = [], [], []
    for i in range(n_conditionals):
        state = (rng.random(task.n_variables) < 0.5).astype(float)
        if state.sum() < min_conditioned:
            idx = rng.choice(task.n_variables, size=min_conditioned,
                             replace=False)
            state[idx] = 1.0
        if state.sum() == task.n_variables:
            state[rng.integers(task.n_variables)] = 0.0
        value = np.where(state > 0.5, joint[i], 0.0)
        theta_samples, x_samples = sample_conditional(
            task, state, value, n_samples=n_samples, rng=rng,
            n_chains=n_samples, verbose=False)
        values.append(value)
        states.append(state)
        samples.append(np.concatenate([theta_samples, x_samples], axis=-1))
        if verbose:
            print(f"  conditional {i + 1}/{n_conditionals} "
                  f"({int(state.sum())} conditioned variables)")
    return {"values": np.asarray(values), "states": np.asarray(states),
            "samples": np.asarray(samples), "true_joint": joint}


def main():
    args = parse_args()
    out_dir = Path(args.out) if args.out else RESULTS / "reference"
    ensure_dir(out_dir)
    names = (["gaussian_linear", "gaussian_mixture", "two_moons", "slcp",
              "tree", "hmm"] if args.task == "all" else [args.task])
    for name in names:
        task = get_task(name)
        print(f"[reference] {name}")
        if args.mode in ("posterior", "both"):
            theta_true, x_obs, samples = reference_posteriors(
                task, args.n_observations, args.n_samples, args.seed,
                args.n_chains)
            np.savez_compressed(out_dir / f"{name}_posterior.npz",
                                theta_true=theta_true, x_obs=x_obs,
                                samples=samples)
        if args.mode in ("conditional", "both"):
            payload = reference_conditionals(
                task, args.n_conditionals, args.n_samples, args.seed)
            np.savez_compressed(out_dir / f"{name}_conditional.npz", **payload)
        print(f"[reference] {name} written to {out_dir}")


if __name__ == "__main__":
    main()
