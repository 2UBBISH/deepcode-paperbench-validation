#!/usr/bin/env python
"""Evaluate a trained BBOX-ADAPTER (also used for plug-and-play, Table 3).

The same adapter can be attached to any other black-box LLM by changing
``--llm-model``: it only steers the generation through its energies, so it is
independent of the LLM's internal parameters.
"""

from __future__ import annotations

import os
import sys

from _common import base_parser, build_config, write_json

from bbox_adapter.adapter.energy import EnergyAdapter
from bbox_adapter.adapter.mlm_adapter import MLMAdapter
from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.eval.evaluator import Evaluator
from bbox_adapter.inference.adaptive_inference import AdaptedInference
from bbox_adapter.llm import build_llm
from bbox_adapter.utils.seed import set_seed


def load_adapter(path: str, device=None):
    with open(os.path.join(path, "adapter_config.json"), "r", encoding="utf-8") as handle:
        import json

        config = json.load(handle)
    if config.get("loss_type") == "mlm":
        return MLMAdapter.load(path, device=device)
    return EnergyAdapter.load(path, device=device)


def main() -> None:
    parser = base_parser("Evaluate a trained adapter (plug-and-play for other LLMs).")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default=None, choices=["full", "single"])
    parser.add_argument("--beam-size", type=int, default=None)
    args = parser.parse_args()

    config = build_config(args)
    if args.beam_size:
        config.inference.beam_size = args.beam_size
    set_seed(config.seed)
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)

    test = load_dataset_examples(
        dataset, "test", data_dir=args.data_dir, seed=config.seed,
        cache_dir=os.path.join(config.output_dir, "hf_cache"), max_samples=args.limit_test,
    )
    llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))
    adapter = load_adapter(args.adapter_path, device=args.device)
    inference = AdaptedInference(
        llm=llm,
        adapter=adapter,
        dataset=dataset,
        beam_size=config.inference.beam_size,
        num_samples_per_beam=config.inference.num_samples_per_beam,
        max_sentence_steps=config.inference.max_sentence_steps,
        max_new_tokens=config.inference.max_new_tokens,
        max_solution_tokens=config.inference.max_solution_tokens,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        mode=args.mode or config.inference.mode,
        num_single_step_candidates=config.inference.num_single_step_candidates,
    )
    evaluator = Evaluator(dataset, inference, output_dir=config.output_dir)
    report = evaluator.evaluate(test)
    print(f"[eval] {dataset} with {config.llm.name}: {report.accuracy:.2f}%")
    write_json(
        os.path.join(config.output_dir, "eval_results.json"),
        {
            "dataset": dataset,
            "llm": config.llm.name,
            "accuracy": report.accuracy,
            "extra": report.extra,
            "usage": llm.usage.as_dict(),
            "cost_usd": llm.usage.cost_or_none(config.llm.name),
        },
    )


if __name__ == "__main__":
    main()
