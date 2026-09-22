#!/usr/bin/env python3
"""Figure 2: PPO performance as a function of the batch size (+ SAPG reference).

Example::

    python scripts/run_batch_size_study.py --env regrasping \
        --env-counts 128 512 2048 8192 24576 --seeds 0 1 2 \
        --ppo-config sapg/configs/ppo_allegrokuka.yaml \
        --sapg-config sapg/configs/sapg_allegrokuka_regrasping.yaml \
        --out figures/fig2_regrasping.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.analysis.batch_size_study import plot_batch_size_study, run_batch_size_study  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-size study (Figure 2)")
    parser.add_argument("--env", default="regrasping")
    parser.add_argument("--env-counts", type=int, nargs="*", default=[128, 512, 2048, 8192, 24576])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--ppo-config", default="sapg/configs/ppo_allegrokuka.yaml")
    parser.add_argument("--sapg-config", default="sapg/configs/sapg_allegrokuka_regrasping.yaml")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument(
        "--total-env-steps",
        type=float,
        default=None,
        help="hold the number of samples collected constant across batch sizes",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logroot", default="runs/batch_size_study")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_batch_size_study(
        env_name=args.env,
        env_counts=args.env_counts,
        seeds=args.seeds,
        config_paths={"ppo": args.ppo_config, "sapg": args.sapg_config},
        num_iterations=args.num_iterations,
        total_env_steps=args.total_env_steps,
        device=args.device,
        logroot=args.logroot,
    )
    out = args.out or os.path.join("figures", f"fig2_{args.env}.png")
    plot_batch_size_study(results, out, title=f"{args.env}: PPO vs batch size")
    with open(out.replace(".png", ".json"), "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"[batch_size_study] wrote {out}")


if __name__ == "__main__":
    main()
