#!/usr/bin/env python3
"""Baselines for the benchmark experiments (Figures 2 and 3).

* ``npe``     -- non-sequential NPE (Papamakarios & Murray, 2016), via sbibm
* ``snpe_c``  -- SNPE-C (Greenberg et al., 2019), via sbibm
* ``tsnpe``   -- TSNPE (Deistler et al., 2022a), via the
  ``mackelab/tsnpe_neurips`` repository (see the addendum)

All baselines are evaluated with the same C2ST metric as (TS)NPSE.

Example
-------
::

    python experiments/run_baselines.py --tasks two_moons slcp \
        --budgets 1000 --methods npe snpe_c tsnpe --output results/baselines.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.sbibm_baselines import run_baseline  # noqa: E402
from baselines.tsnpe_adapter import run_tsnpe  # noqa: E402
from experiments.common import (  # noqa: E402
    BUDGETS,
    TASK_SPECS,
    append_result,
    get_reference_posterior_samples,
    get_task,
)
from snpse.metrics import c2st  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=["two_moons"])
    p.add_argument("--budgets", nargs="+", default=["1000"])
    p.add_argument("--methods", nargs="+", default=["npe", "snpe_c", "tsnpe"])
    p.add_argument("--observation", type=int, default=1)
    p.add_argument("--num-posterior-samples", type=int, default=10000)
    p.add_argument("--num-rounds", type=int, default=None,
                   help="defaults to 1 for npe and 10 for the sequential baselines")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="results/baselines.jsonl")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = list(TASK_SPECS) if args.tasks == ["all"] else args.tasks
    budgets = BUDGETS if "all" in args.budgets else [int(b) for b in args.budgets]

    for task_name in tasks:
        task = get_task(task_name)
        for budget in budgets:
            for method in args.methods:
                if args.smoke:
                    budget = min(budget, 1000)
                t0 = time.time()
                print(f"[{task_name}] budget={budget} baseline={method}")
                if method == "tsnpe":
                    samples, num_calls = run_tsnpe(
                        task,
                        num_samples=args.num_posterior_samples,
                        num_simulations=budget,
                        num_observation=args.observation,
                        num_rounds=args.num_rounds or 10,
                        seed=args.seed,
                    )
                else:
                    samples, num_calls = run_baseline(
                        method,
                        task,
                        num_samples=args.num_posterior_samples,
                        num_simulations=budget,
                        num_observation=args.observation,
                        num_rounds=args.num_rounds,
                        seed=args.seed,
                    )
                reference = get_reference_posterior_samples(
                    task, args.observation, num_samples=args.num_posterior_samples
                )
                score = c2st(samples, reference, seed=0)
                record = {
                    "task": task_name,
                    "method": method,
                    "budget": budget,
                    "observation": args.observation,
                    "seed": args.seed,
                    "num_posterior_samples": args.num_posterior_samples,
                    "num_simulations": num_calls,
                    "c2st": score,
                    "seconds": time.time() - t0,
                    "source": "sbibm" if method != "tsnpe" else "mackelab/tsnpe_neurips",
                }
                append_result(args.output, record)
                print(f"  -> C2ST = {score:.4f}  ({record['seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
