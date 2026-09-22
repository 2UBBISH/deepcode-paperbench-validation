#!/usr/bin/env python
"""Plug-and-play adaptation (Section 4.3, Table 3).

A BBOX-ADAPTER tuned while adapting gpt-3.5-turbo is attached, without any
retraining, to another black-box LLM (davinci-002 or Mixtral-8x7B).  We report
the un-adapted accuracy of the pluggee and the accuracy obtained with the
plugged-in adapter.
"""

from __future__ import annotations

import os

from _common import base_parser, build_config, write_json

from bbox_adapter.baselines.cot import CoTBaseline
from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.eval.evaluator import Evaluator, evaluate_texts
from bbox_adapter.inference.adaptive_inference import AdaptedInference
from bbox_adapter.llm import build_llm
from bbox_adapter.utils.seed import set_seed

from evaluate_adapter import load_adapter


def main() -> None:
    parser = base_parser("Plug-and-play evaluation (Table 3).")
    parser.add_argument("--plugger-adapter", type=str, required=True,
                        help="Directory of the adapter tuned on gpt-3.5-turbo.")
    parser.add_argument("--pluggees", type=str,
                        default="davinci-002,mistralai/Mixtral-8x7B-v0.1",
                        help="Comma separated list of black-box LLMs to plug into.")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)
    test = load_dataset_examples(dataset, "test", data_dir=args.data_dir, seed=config.seed,
                                 cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                 max_samples=args.limit_test)
    adapter = load_adapter(args.plugger_adapter, device=args.device)

    rows = []
    for pluggee in [name.strip() for name in args.pluggees.split(",") if name.strip()]:
        config.llm.name = pluggee
        config.llm.provider = "openai" if "davinci" in pluggee else "huggingface"
        llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))

        baseline = CoTBaseline(llm, dataset, max_new_tokens=config.llm.max_new_tokens,
                               temperature=0.0)
        generations = baseline.generate(test)
        base_report = evaluate_texts(
            dataset, [g[0] for g in generations],
            [example.answer for example in test],
            [example.num_choices for example in test],
        )
        inference = AdaptedInference(
            llm=llm, adapter=adapter, dataset=dataset,
            beam_size=config.inference.beam_size,
            num_samples_per_beam=config.inference.num_samples_per_beam,
            max_sentence_steps=config.inference.max_sentence_steps,
            max_new_tokens=config.inference.max_new_tokens,
            max_solution_tokens=config.inference.max_solution_tokens,
            temperature=config.inference.temperature,
            mode=config.inference.mode,
        )
        plugged_report = Evaluator(dataset, inference).evaluate(test)
        rows.append({
            "pluggee": pluggee,
            "base_accuracy": base_report.accuracy,
            "plugged_accuracy": plugged_report.accuracy,
            "delta": plugged_report.accuracy - base_report.accuracy,
        })
        print(f"[plug-and-play] {pluggee}: {base_report.accuracy:.2f} -> "
              f"{plugged_report.accuracy:.2f} ({rows[-1]['delta']:+.2f})")

    write_json(os.path.join(config.output_dir, "plug_and_play.json"), rows)


if __name__ == "__main__":
    main()
