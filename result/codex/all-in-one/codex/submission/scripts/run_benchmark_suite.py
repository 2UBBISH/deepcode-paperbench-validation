#!/usr/bin/env python
"""Sec. 4.1 / Fig. 4a: benchmark suite (Simformer vs NPE).

For every benchmark task (Gaussian Linear, Gaussian Mixture, Two Moons, SLCP) and
every training set size (1k / 10k / 100k simulations) the script trains

* the Simformer (dense, undirected and directed attention mask) and
* NPE (sbi, neural spline flow, defaults),

and evaluates the posterior with the C2ST metric against MCMC reference samples.
The trained models and the results are stored under ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import RESULTS, ensure_dir, generate_simulations, save_checkpoint
from simformer import BENCHMARK_TASKS, Simformer, SimformerConfig, get_task
from simformer.baselines import SBIBaseline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=BENCHMARK_TASKS)
    parser.add_argument("--n-simulations", nargs="+", type=int,
                        default=[1000, 10000, 100000])
    parser.add_argument("--mask-modes", nargs="+",
                        default=["dense", "undirected", "directed"])
    parser.add_argument("--sde", default="vesde", choices=["vesde", "vpsde"])
    parser.add_argument("--baselines", nargs="+", default=["npe"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=str(RESULTS / "benchmark_suite"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate against MCMC reference samples")
    return parser.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(Path(args.out))
    results = {}
    for task_name in args.tasks:
        task = get_task(task_name)
        results[task_name] = {"simformer": {}, "npe": {}}
        for n_sims in args.n_simulations:
            theta, x, index, metadata = generate_simulations(
                task, n_sims, seed=args.seed)
            for mask_mode in args.mask_modes:
                tag = f"{mask_mode}_{n_sims}"
                dir_out = ensure_dir(out / task_name / tag)
                model = Simformer(task.problem(), SimformerConfig(
                    mask_mode=mask_mode, sde=args.sde, n_layers=6,
                    seed=args.seed))
                model.fit(theta, x, index, metadata, batch_size=1000, lr=1e-4,
                          max_epochs=args.max_epochs, patience=20,
                          max_steps=args.max_steps, verbose=False)
                save_checkpoint(model, dir_out / "model.pt",
                                {"n_simulations": n_sims})
                if args.evaluate:
                    from evaluate_c2st import evaluate_posterior
                    reference = (RESULTS / "reference" /
                                 f"{task_name}_posterior.npz")
                    if reference.exists():
                        metrics = evaluate_posterior(
                            model, task, reference, args.n_samples,
                            model.config.num_steps)
                        results[task_name]["simformer"][tag] = metrics
                        print(f"[suite] {task_name} {tag} C2ST="
                              f"{metrics['c2st_simformer']:.3f}")
            for baseline_name in args.baselines:
                tag = f"{baseline_name}_{n_sims}"
                baseline = SBIBaseline(task, method=baseline_name)
                baseline.train(theta, x, training_batch_size=1000,
                               show_progress_bar=False)
                if args.evaluate:
                    reference = (RESULTS / "reference" /
                                 f"{task_name}_posterior.npz")
                    import torch
                    data = np.load(reference)
                    x_obs, ref = data["x_obs"], data["samples"]
                    from simformer.metrics import c2st_accuracy
                    scores = []
                    for i in range(len(x_obs)):
                        samples = baseline.sample_posterior(x_obs[i], args.n_samples)
                        scores.append(c2st_accuracy(
                            samples[0], ref[i, :, :task.n_params]))
                    results[task_name][baseline_name][tag] = float(np.mean(scores))
                    print(f"[suite] {task_name} {tag} C2ST="
                          f"{np.mean(scores):.3f}")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[suite] results written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
