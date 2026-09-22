"""Shared CLI plumbing for the experiment drivers."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wwmf.config import ExperimentConfig  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model", default="flan_t5_large",
        choices=["bart0_large", "flan_t5_large", "flan_t5_3b", "flan_t5_small"],
    )
    parser.add_argument("--tuning-mode", default=None, choices=["head", "lora", "full_ft"])
    parser.add_argument("--refinement-data", default=None, choices=["mmlu", "p3_test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-upstream", type=int, default=None)
    parser.add_argument("--max-online", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None,
                        help="override the number of forecasting training steps (debug/smoke runs)")
    parser.add_argument("--examples-per-upstream-task", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model = args.model
    cfg.seed = args.seed
    cfg.tuning_mode = args.tuning_mode or ("head" if args.model.startswith("bart0") else "lora")
    cfg.refinement_data = args.refinement_data or ("p3_test" if args.model.startswith("bart0") else "mmlu")
    if args.data_root:
        cfg.data_root = args.data_root
    if args.cache_root:
        cfg.cache_root = args.cache_root
    if args.output_root:
        cfg.output_root = args.output_root
    if args.examples_per_upstream_task:
        cfg.examples_per_upstream_task = args.examples_per_upstream_task
    return cfg


def print_result(obj) -> None:
    print(json.dumps(obj, indent=2))
