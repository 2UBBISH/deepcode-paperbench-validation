#!/usr/bin/env python3
"""Sec. 6.4 / Figures 7 and 8: state-diversity of SAPG vs PPO vs random policy.

State batches are written by the trainers (``algo.record_states_interval``) into
``<run_dir>/states/``.  This script also records a randomly initialised policy
so that the "randomly initialized policy" curve of the paper can be reproduced.

Example::

    python scripts/run_diversity_analysis.py \
        --run sapg=runs/sapg_reorientation_seed0 \
        --run ppo=runs/ppo_reorientation_seed0 \
        --random-config sapg/configs/toy_sapg.yaml \
        --out figures/fig7_reorientation.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.analysis.mlp_diversity import analyse_mlp_diversity, plot_mlp_diversity  # noqa: E402
from sapg.analysis.pca_diversity import analyse_pca_diversity, plot_pca_diversity  # noqa: E402
from sapg.envs import make_env  # noqa: E402
from sapg.utils.config import load_config  # noqa: E402
from sapg.utils.state_dataset import collect_random_policy_states, load_state_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="State-diversity analysis")
    parser.add_argument("--run", action="append", default=[], help="label=run_dir")
    parser.add_argument("--random-config", default=None, help="config used to roll out a random policy")
    parser.add_argument("--num-random-steps", type=int, default=2000)
    parser.add_argument("--components", type=int, nargs="*", default=list(range(1, 33)))
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--transitions", type=int, default=400_000)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--out", default="figures/fig7_diversity.png")
    parser.add_argument("--out-mlp", default="figures/fig8_diversity.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: Dict[str, np.ndarray] = {}
    for item in args.run:
        label, _, run_dir = item.partition("=")
        datasets[label] = load_state_dataset(run_dir)
    if args.random_config:
        cfg = load_config(args.random_config)
        env = make_env(cfg, device="cpu")
        batches = collect_random_policy_states(env, num_steps=args.num_random_steps, seed=0)
        datasets["random"] = np.concatenate(batches, axis=0)
    if not datasets:
        print("[diversity] nothing to analyse: pass --run label=dir and/or --random-config", file=sys.stderr)
        return

    pca_results = analyse_pca_diversity(
        datasets, components=args.components, max_samples=args.max_samples
    )
    plot_pca_diversity(pca_results, args.out)
    print(f"[diversity] wrote {args.out}")

    mlp_results = analyse_mlp_diversity(
        {k: v[: args.max_samples] for k, v in datasets.items()},
        hidden_sizes=args.hidden_sizes,
        num_transitions=args.transitions,
    )
    plot_mlp_diversity(mlp_results, args.out_mlp)
    print(f"[diversity] wrote {args.out_mlp}")


if __name__ == "__main__":
    main()
