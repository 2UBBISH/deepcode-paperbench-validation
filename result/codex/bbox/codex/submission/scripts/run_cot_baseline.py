#!/usr/bin/env python
"""CoT baseline: the un-adapted black-box LLM with the prompts of Appendix J."""

from __future__ import annotations

import os
import sys

from _common import base_parser, build_config, write_json

from bbox_adapter.baselines.cot import CoTBaseline
from bbox_adapter.data.loaders import load_train_test
from bbox_adapter.eval.evaluator import evaluate_texts
from bbox_adapter.llm import build_llm
from bbox_adapter.utils.seed import set_seed


def main() -> None:
    parser = base_parser("Evaluate the un-adapted black-box LLM (CoT).")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    dataset = (args.dataset or config.dataset).lower()
    _, test = load_train_test(dataset, data_dir=args.data_dir, seed=config.seed,
                              cache_dir=os.path.join(config.output_dir, "hf_cache"),
                              max_test=args.limit_test)
    llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))
    baseline = CoTBaseline(llm, dataset, max_new_tokens=config.llm.max_new_tokens,
                           temperature=args.temperature, num_samples=args.num_samples)
    generations = baseline.generate(test)
    report = evaluate_texts(
        dataset,
        [texts[0] for texts in generations],
        [example.answer for example in test],
        [example.num_choices for example in test],
    )
    print(f"[cot] {dataset}: accuracy {report.accuracy:.2f}% over {report.num_examples} examples")
    write_json(
        os.path.join(config.output_dir, "cot_results.json"),
        {
            "dataset": dataset,
            "accuracy": report.accuracy,
            "num_examples": report.num_examples,
            "usage": llm.usage.as_dict(),
            "cost_usd": llm.usage.cost(config.llm.name),
        },
    )


if __name__ == "__main__":
    main()
