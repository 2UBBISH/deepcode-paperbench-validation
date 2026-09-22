#!/usr/bin/env python3
"""Experiment V: sensitivity of p / lambda (Figures 6-8) and alpha (Figure 9)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.experiments.common import ckpt_path  # noqa: E402
from rice.experiments.hyperparams import (  # noqa: E402
    ALPHA_VALUES,
    run_alpha_sweep,
    run_p_lambda_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--mask-ours", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--skip-alpha", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    agent = args.agent or ckpt_path("{}_seed0.pt".format(args.env), kind="agents")
    mask = args.mask_ours or ckpt_path(
        "{}_ours_seed0.pt".format(args.env), kind="masks"
    )
    print("=== p / lambda sweep ===")
    run_p_lambda_grid(
        args.env,
        agent,
        mask,
        total_steps=args.steps,
        seeds=args.seeds,
        device=args.device,
    )
    if not args.skip_alpha:
        print("=== alpha sweep (fidelity) ===")
        run_alpha_sweep(args.env, agent, ALPHA_VALUES, device=args.device)


if __name__ == "__main__":
    main()
