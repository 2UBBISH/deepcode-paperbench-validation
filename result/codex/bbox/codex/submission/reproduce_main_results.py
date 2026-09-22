#!/usr/bin/env python
"""Reproduce the main results of Table 2 end to end.

Runs BBOX-ADAPTER with the three sources of positive samples (ground truth, AI
feedback, combined) for every dataset and adapter size, and writes a summary
CSV/JSON that mirrors the layout of Table 2.

This is the long-running entry point: 4 datasets x 3 settings x 2 adapter sizes
of online adaptation.  Use ``--limit-train`` / ``--datasets`` for partial runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bbox_adapter.config import load_run_config  # noqa: E402
from bbox_adapter.pipeline import run_online_adaptation  # noqa: E402


DATASETS = ["strategyqa", "gsm8k", "truthfulqa", "scienceqa"]
SETTINGS = ["ground_truth", "ai_feedback", "combined"]
SIZES = ["0.1B", "0.3B"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--output-root", type=str, default="runs/table2")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    parser.add_argument("--settings", type=str, default=",".join(SETTINGS))
    parser.add_argument("--sizes", type=str, default=",".join(SIZES))
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    rows = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for setting in [s.strip() for s in args.settings.split(",") if s.strip()]:
            for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
                config = load_run_config(os.path.join(args.config_dir, f"{dataset}.yaml"))
                config.online.positive_source = setting
                config.adapter.adapter_size = size
                config.output_dir = os.path.join(args.output_root, dataset, f"{setting}_{size}")
                if args.iterations is not None:
                    config.online.num_iterations = args.iterations
                if args.train_steps is not None:
                    config.adapter.num_train_steps = args.train_steps

                result = run_online_adaptation(
                    config,
                    dataset=dataset,
                    adapter_size=size,
                    data_dir=args.data_dir,
                    max_train=args.limit_train,
                    max_test=args.limit_test,
                    device=args.device,
                )
                rows.append({
                    "dataset": dataset,
                    "setting": setting,
                    "adapter_size": size,
                    "accuracy": result["test_accuracy"],
                    "training_cost_usd": result.get("training_cost_usd"),
                    "output_dir": config.output_dir,
                })
                print(f"[table2] {dataset} {setting} {size}: {rows[-1]['accuracy']:.2f}%")

    os.makedirs(args.output_root, exist_ok=True)
    with open(os.path.join(args.output_root, "table2_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with open(os.path.join(args.output_root, "table2_summary.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table2] wrote {args.output_root}/table2_summary.csv")


if __name__ == "__main__":
    main()
