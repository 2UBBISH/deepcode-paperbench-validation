#!/usr/bin/env python
"""Short end-to-end sanity check of the reproduction (fast, CPU friendly).

For the Gaussian linear task the ground truth posterior is available in closed
form (``theta | x ~ N(x / 2, 0.05 I)`` because both the prior and the likelihood
are Gaussian with the same variance).  This script trains a *small* Simformer for
a few epochs and reports the C2ST accuracy between the Simformer posterior and
the analytic posterior -- a value that should decrease towards ``0.5`` as the
model learns (the full paper configuration requires ``1e5`` simulations and
training to convergence; see the README).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import RESULTS, ensure_dir, generate_simulations
from simformer import Simformer, SimformerConfig, get_task
from simformer.metrics import c2st_accuracy


def analytic_posterior_samples(x_obs: np.ndarray, n: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Exact posterior of the Gaussian linear task: N(x / 2, 0.05 I)."""
    mean = 0.5 * np.asarray(x_obs)
    std = np.sqrt(0.05)
    return mean[None] + std * rng.normal(size=(n, mean.size))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-simulations", type=int, default=20000)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--mask-mode", default="directed")
    parser.add_argument("--out", default=str(RESULTS / "reproduction_check"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(Path(args.out))
    task = get_task("gaussian_linear")
    rng = np.random.default_rng(args.seed)
    theta, x, index, metadata = generate_simulations(
        task, args.n_simulations, seed=args.seed)
    model = Simformer(task.problem(), SimformerConfig(
        mask_mode=args.mask_mode, n_layers=args.n_layers,
        num_steps=args.num_steps, seed=args.seed))

    history, c2st_history = [], []
    checkpoint_epochs = [1, max(2, args.max_epochs // 4),
                         max(3, args.max_epochs // 2), args.max_epochs]
    epoch_offset = 0
    for target_epoch in checkpoint_epochs:
        model.fit(theta, x, index, metadata, batch_size=args.batch_size,
                  lr=args.lr, max_epochs=target_epoch - epoch_offset,
                  val_fraction=0.1, patience=target_epoch, verbose=False)
        epoch_offset = target_epoch
        theta_test = task.prior_sample(10, rng)
        x_obs = task.simulate(theta_test, rng)
        scores = []
        for i in range(10):
            value = np.concatenate([np.zeros(task.n_params), x_obs[i]])
            state = np.concatenate([np.zeros(task.n_params),
                                    np.ones(task.n_data)])
            samples = model.sample([(value, state)], num_samples=args.n_samples,
                                   num_steps=args.num_steps)[0][:, :task.n_params]
            reference = analytic_posterior_samples(x_obs[i], args.n_samples, rng)
            scores.append(c2st_accuracy(samples, reference, n_trees=100))
        c2st_history.append({"epochs": target_epoch,
                             "c2st": float(np.mean(scores))})
        history.extend(model.history)
        print(f"[check] after {target_epoch} epochs: C2ST = "
              f"{np.mean(scores):.3f} (0.5 is perfect)")

    (out / "reproduction_check.json").write_text(json.dumps({
        "task": "gaussian_linear",
        "n_simulations": args.n_simulations,
        "mask_mode": args.mask_mode,
        "config": {k: str(v) for k, v in vars(model.config).items()},
        "c2st_history": c2st_history,
    }, indent=2))
    print(f"[check] written to {out / 'reproduction_check.json'}")


if __name__ == "__main__":
    main()
