#!/usr/bin/env python
"""Cache the four datasets as JSONL so that runs can be fully offline.

The paper uses (Appendix F.1):

* GSM8K       -- 7473 train / 1319 test
* StrategyQA  -- 2059 train / 229 test
* TruthfulQA  -- 717 train / 100 test (random split of the 817 validation items)
* ScienceQA   -- 2000 train / 500 test, image questions removed

The resulting files (``<dataset>_train.jsonl`` / ``<dataset>_test.jsonl``) are
exactly what ``--data-dir`` expects, including the deterministic sampling used
by the paper.
"""

from __future__ import annotations

import argparse
import json
import os

from _common import base_parser

from bbox_adapter.data.loaders import DATASET_SPECS, load_dataset_examples


def main() -> None:
    parser = base_parser("Download/cache the four datasets as JSONL.")
    parser.add_argument("--datasets", type=str, default=",".join(DATASET_SPECS))
    args = parser.parse_args()

    output_dir = args.data_dir or args.output_dir or "data"
    os.makedirs(output_dir, exist_ok=True)
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for split, limit in (("train", args.limit_train), ("test", args.limit_test)):
            examples = load_dataset_examples(
                dataset, split, seed=args.seed or 0, cache_dir=output_dir, max_samples=limit
            )
            path = os.path.join(output_dir, f"{dataset}_{split}.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for example in examples:
                    handle.write(
                        json.dumps(
                            {
                                "question": example.question,
                                "answer": example.answer,
                                "solution": example.solution,
                                "choices": example.choices,
                                "split": split,
                            }
                        )
                        + "\n"
                    )
            print(f"[data] {path}: {len(examples)} examples")


if __name__ == "__main__":
    main()
