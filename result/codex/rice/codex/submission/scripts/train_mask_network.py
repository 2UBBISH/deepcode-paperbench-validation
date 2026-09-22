#!/usr/bin/env python3
"""Train a mask network (Algorithm 1) for one application.

Examples
--------
    # our explanation method
    python scripts/train_mask_network.py --env Hopper --method ours \
        --agent checkpoints/agents/Hopper_seed0.pt

    # StateMask baseline (same sample budget, timing is reported -> Table 4)
    python scripts/train_mask_network.py --env Hopper --method statemask \
        --agent checkpoints/agents/Hopper_seed0.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.experiments.common import ckpt_path, save_json  # noqa: E402
from rice.experiments.explanation import train_explanation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", required=True, help="pre-trained target agent")
    parser.add_argument("--method", default="ours", choices=["ours", "statemask"])
    parser.add_argument("--samples", type=int, default=None, help="sample budget")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = train_explanation(
        args.env,
        args.agent,
        method=args.method,
        seed=args.seed,
        device=args.device,
        alpha=args.alpha,
        max_samples=args.samples,
        iterations=args.iterations,
    )
    payload = {
        "env": args.env,
        "method": args.method,
        "samples": result.samples,
        "wall_time": result.wall_time,
        "checkpoint": getattr(result, "checkpoint", None),
        "history": result.history,
    }
    out_path = args.out or ckpt_path(
        "{}_{}_seed{}.json".format(args.env, args.method, args.seed), kind="results"
    )
    save_json(payload, out_path)
    print(
        "{} mask network trained on {} samples in {:.1f}s -> {}".format(
            args.method, result.samples, result.wall_time, out_path
        )
    )


if __name__ == "__main__":
    main()
