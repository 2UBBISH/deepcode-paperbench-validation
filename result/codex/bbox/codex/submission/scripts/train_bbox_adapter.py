#!/usr/bin/env python
"""Train BBOX-ADAPTER with the online adaptation framework (Algorithm 1).

Example
-------
python scripts/train_bbox_adapter.py --config configs/strategyqa.yaml \
    --output-dir runs/strategyqa/ground_truth_0.1B
"""

from __future__ import annotations

import os
import sys

from _common import base_parser, build_config, write_json

from bbox_adapter.pipeline import run_online_adaptation


def main() -> None:
    parser = base_parser("Train BBOX-ADAPTER (online adaptation, Algorithm 1).")
    parser.add_argument("--iterations", type=int, default=None, help="Number of online iterations T.")
    parser.add_argument("--positive-source", type=str, default=None,
                        choices=["ground_truth", "ai_feedback", "combined"])
    parser.add_argument("--train-steps", type=int, default=None,
                        help="Adapter updates per iteration (default 6000).")
    parser.add_argument("--mode", type=str, default=None, choices=["full", "single"],
                        help="Adapted inference variant used while sampling candidates "
                             "(Table 4 compares 'full-step' with 'single-step').")
    args = parser.parse_args()

    config = build_config(args)
    if args.iterations is not None:
        config.online.num_iterations = args.iterations
    if args.positive_source is not None:
        config.online.positive_source = args.positive_source
    if args.train_steps is not None:
        config.adapter.num_train_steps = args.train_steps
    if args.mode is not None:
        config.inference.mode = args.mode

    result = run_online_adaptation(
        config,
        dataset=args.dataset,
        adapter_size=args.adapter_size,
        data_dir=args.data_dir,
        max_train=args.limit_train,
        max_test=args.limit_test,
        device=args.device,
    )
    write_json(os.path.join(config.output_dir, "results.json"), result)
    print(f"[done] results written to {config.output_dir}/results.json")


if __name__ == "__main__":
    main()
