#!/usr/bin/env python
"""Train an sbi baseline (NPE / NLE / NRE) on one of the benchmark tasks.

Appendix A2.1: default sbi parameters, a more expressive neural spline flow for
NPE and NLE, batch size 1000, Adam, early stopping on the validation loss.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from common import RESULTS, ensure_dir, generate_simulations
from simformer import get_task
from simformer.baselines import SBIBaseline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", default="npe", choices=["npe", "nle", "nre"])
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--training-batch-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--max-num-epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out) if args.out else (
        RESULTS / f"baseline_{args.method}_{args.task}_{args.n_simulations}")
    ensure_dir(out)
    task = get_task(args.task)
    theta, x, _, _ = generate_simulations(task, args.n_simulations,
                                          seed=args.seed)
    baseline = SBIBaseline(task, method=args.method)
    baseline.train(theta, x, training_batch_size=args.training_batch_size,
                   learning_rate=args.learning_rate,
                   stop_after_epochs=args.stop_after_epochs,
                   max_num_epochs=args.max_num_epochs,
                   show_progress_bar=False)
    with open(out / "baseline.pkl", "wb") as f:
        pickle.dump({"method": args.method, "task": args.task,
                     "posterior": baseline.posterior,
                     "likelihood": baseline.likelihood}, f)
    print(f"[baseline] saved {args.method} for {args.task} to {out}")


if __name__ == "__main__":
    main()
