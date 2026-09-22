#!/usr/bin/env python3
"""Benchmark (TS)NPSE on the eight sbibm tasks (Section 5.2, Figures 2 and 3).

For every (task, budget, method) combination this script

1. runs the inference algorithm with the specified simulation budget,
2. draws ``num_posterior_samples`` posterior samples for the observation,
3. evaluates the classification-based two-sample test (C2ST) score against the
   reference posterior samples provided by ``sbibm``, and
4. appends a JSON record to the results file.

Example
-------
::

    python experiments/run_benchmark.py \
        --tasks two_moons slcp --budgets 1000 --methods npse_ve tsnpse_ve \
        --output results/benchmark.jsonl

The full grid reported in the paper is
``--tasks all --budgets 1000 10000 100000 --methods npse_ve npse_vp tsnpse_ve tsnpse_vp``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (  # noqa: E402
    BUDGETS,
    TASK_SPECS,
    append_result,
    build_snpse_method,
    draw_posterior_samples,
    get_observation,
    get_reference_posterior_samples,
    get_task,
    run_npse,
    run_sequential,
)
from experiments.configs import get_config  # noqa: E402
from snpse.metrics import c2st  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=["two_moons"], help="`all` or sbibm task names")
    p.add_argument(
        "--budgets",
        nargs="+",
        default=["1000"],
        help="simulation budgets, or `all` for 1000 10000 100000",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["npse_ve", "npse_vp", "tsnpse_ve", "tsnpse_vp"],
        help="npse_ve npse_vp tsnpse_ve tsnpse_vp snpse_a snpse_b snpse_c",
    )
    p.add_argument("--observation", type=int, default=1)
    p.add_argument("--num-posterior-samples", type=int, default=None,
                   help="defaults to task.num_posterior_samples (10000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="results/benchmark.jsonl")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smoke", action="store_true", help="tiny settings for a quick test")
    p.add_argument("--save-samples", action="store_true", help="store posterior samples in the record")
    p.add_argument(
        "--config",
        type=str,
        default="paper",
        choices=["paper", "extended", "paper_literal_embedding", "smoke"],
        help="named hyperparameter configuration (see experiments/configs.py)",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="override TrainingConfig.max_iters (paper: 3000 gradient steps)",
    )
    p.add_argument("--lr", type=float, default=None, help="override TrainingConfig.lr (paper: 1e-4)")
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="number of rounds for sequential methods (paper: 10)",
    )
    p.add_argument(
        "--t-scale",
        type=float,
        default=None,
        help="sinusoidal time-embedding scale (default 1000; paper's literal formula is 1)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_config = get_config(args.config)
    if run_config.smoke:
        args.smoke = True
    max_iters = args.max_iters if args.max_iters is not None else run_config.max_iters
    lr = args.lr if args.lr is not None else run_config.lr
    t_scale = args.t_scale if args.t_scale is not None else run_config.t_scale
    tasks = list(TASK_SPECS) if args.tasks == ["all"] else args.tasks
    budgets = BUDGETS if "all" in args.budgets else [int(b) for b in args.budgets]

    for task_name in tasks:
        spec = TASK_SPECS[task_name]
        task = get_task(task_name)
        x_obs = get_observation(task, args.observation)
        c2st_seed = 0

        for budget in budgets:
            if args.smoke and budget > 2000:
                continue
            for method in args.methods:
                t0 = time.time()
                estimator, kind = build_snpse_method(
                    method,
                    task,
                    spec.dim_parameters,
                    spec.dim_data,
                    budget,
                    spec.sigma_min,
                    seed=args.seed,
                    device=args.device,
                    smoke=args.smoke,
                    max_iters=max_iters,
                    lr=lr,
                    t_scale=t_scale,
                    rounds=args.rounds,
                )
                print(f"[{task_name}] budget={budget} method={method}")
                if kind == "npse":
                    run_npse(estimator, task, budget, verbose=not args.smoke)
                    diagnostics = [estimator.train_info]
                else:
                    estimator, diagnostics = run_sequential(
                        estimator, x_obs, verbose=not args.smoke, smoke=args.smoke
                    )

                n_samples = args.num_posterior_samples or task.num_posterior_samples
                samples = draw_posterior_samples(estimator, method, x_obs, n_samples, seed=args.seed)
                reference = get_reference_posterior_samples(
                    task, args.observation, num_samples=n_samples
                )
                score = c2st(samples, reference, seed=c2st_seed)

                record = {
                    "task": task_name,
                    "method": method,
                    "budget": budget,
                    "observation": args.observation,
                    "seed": args.seed,
                    "config": args.config,
                    "num_posterior_samples": int(n_samples),
                    "c2st": score,
                    "seconds": time.time() - t0,
                    "diagnostics": _clean_diagnostics(diagnostics),
                }
                if args.save_samples:
                    record["posterior_samples"] = samples.detach().cpu()
                append_result(args.output, record)
                print(f"  -> C2ST = {score:.4f}  ({record['seconds']:.1f}s)")

    return 0


def _clean_diagnostics(diagnostics):
    """Drop the (large) per-iteration training history."""
    cleaned = []
    for d in diagnostics:
        if isinstance(d, dict):
            cleaned.append({k: v for k, v in d.items() if k != "history" and k != "train_idx" and k != "val_idx"})
    return cleaned


if __name__ == "__main__":
    raise SystemExit(main())
