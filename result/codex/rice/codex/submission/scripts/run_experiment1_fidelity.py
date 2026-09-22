#!/usr/bin/env python3
"""Experiment I: fidelity scores (Figure 5) and mask training cost (Table 4)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.experiments.common import ckpt_path  # noqa: E402
from rice.experiments.explanation import run_experiment_i  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--mask-ours", default=None)
    parser.add_argument("--mask-statemask", default=None)
    parser.add_argument("--trajectories", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    agent = args.agent or ckpt_path("{}_seed{}.pt".format(args.env, args.seed), kind="agents")
    paths = {}
    if args.mask_ours:
        paths["ours"] = args.mask_ours
    if args.mask_statemask:
        paths["statemask"] = args.mask_statemask
    result = run_experiment_i(
        args.env,
        agent,
        explanation_paths=paths,
        n_trajectories=args.trajectories,
        n_seeds=args.seeds,
        seed=args.seed,
        device=args.device,
        train_if_missing=not args.no_train,
    )
    print("\n=== fidelity scores ({}) ===".format(result.env))
    for method, values in result.results.items():
        print(
            "{:>10}: {}".format(
                method,
                "  ".join(
                    "K={:.0%}: {:.3f}".format(k, m)
                    for k, m in zip(values["k"], values["mean"])
                ),
            )
        )
    print("\n=== mask training cost ===")
    for method, values in result.efficiency.items():
        print("{:>10}: {}".format(method, values))


if __name__ == "__main__":
    main()
