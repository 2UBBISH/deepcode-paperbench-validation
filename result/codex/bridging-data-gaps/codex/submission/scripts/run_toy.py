#!/usr/bin/env python3
"""Section 5.1 / Figure 2: the 2-D toy study.

Example
-------
python scripts/run_toy.py --output-dir outputs/toy --source-iterations 10000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpms_ant.toy_experiment import ToyConfig, run_toy_experiment
from dpms_ant.utils import Logger, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/toy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dim", type=int, default=2, help="the paper uses 2")
    parser.add_argument("--source-iterations", type=int, default=10000)
    parser.add_argument("--transfer-iterations", type=int, default=300)
    parser.add_argument("--num-target-samples", type=int, default=10)
    parser.add_argument("--guidance-rms", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.2)
    parser.add_argument("--num-adversarial-steps", type=int, default=10)
    parser.add_argument(
        "--gradient-timestep-fraction",
        type=float,
        default=0.2,
        help="timestep (as a fraction of T) at which Figure 2(a) is computed",
    )
    parser.add_argument("--heatmap-samples", type=int, default=20000)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = str(get_device(args.device))
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, "log.txt"))

    config = ToyConfig(
        data_dim=args.data_dim,
        num_target_samples=args.num_target_samples,
        source_iterations=args.source_iterations,
        transfer_iterations=args.transfer_iterations,
        guidance_rms=args.guidance_rms,
        gradient_timestep_fraction=args.gradient_timestep_fraction,
        num_samples_for_heatmap=args.heatmap_samples,
        seed=args.seed,
    )
    config.adversary.step_size = args.omega
    config.adversary.num_steps = args.num_adversarial_steps

    summary = run_toy_experiment(
        config, device=device, output_dir=args.output_dir, logger=logger, quick=args.quick
    )
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.log("[toy] angle errors (deg): " + json.dumps(summary["angle_error_deg"]))
    logger.log(f"[toy] figures written to {args.output_dir}")


if __name__ == "__main__":
    main()

