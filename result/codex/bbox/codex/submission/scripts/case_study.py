#!/usr/bin/env python
"""Case study output (Section 4.8 / Figure 4).

Generates the CoT answer of the un-adapted black-box LLM and the adapted answer
produced by BBOX-ADAPTER for a few questions, together with the top-ranked beam
hypotheses and the adapter scores.  This is the textual equivalent of Figure 4:
for the reported GSM8K example the CoT answer is wrong while the adapted
inference (sentence-level beam search steered by ``g_theta``) finds the correct
solution.
"""

from __future__ import annotations

import json
import os

from _common import base_parser, build_config, write_json

from bbox_adapter.baselines.cot import CoTBaseline
from bbox_adapter.data.answer_extraction import extract_answer
from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.inference.adaptive_inference import AdaptedInference
from bbox_adapter.llm import build_llm
from bbox_adapter.utils.seed import set_seed

from evaluate_adapter import load_adapter


def main() -> None:
    parser = base_parser("Case study: CoT vs. adapted inference.")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)
    examples = load_dataset_examples(
        dataset, "test", data_dir=args.data_dir, seed=config.seed,
        cache_dir=os.path.join(config.output_dir, "hf_cache"),
    )[args.start_index:args.start_index + args.num_examples]
    llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))
    adapter = load_adapter(args.adapter_path, device=args.device)

    cot = CoTBaseline(llm, dataset, max_new_tokens=config.llm.max_new_tokens, temperature=0.0)
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

    records = []
    for example in examples:
        cot_text = cot.answer(example)
        adapted_text, ranked = inference.answer_with_details(example.question, example.choices)
        record = {
            "question": example.question,
            "gold_answer": example.answer,
            "gold_solution": example.solution,
            "cot_text": cot_text,
            "cot_answer": extract_answer(dataset, cot_text, example.num_choices),
            "adapted_text": adapted_text,
            "adapted_answer": extract_answer(dataset, adapted_text, example.num_choices),
            "ranked_candidates": [
                {"text": text, "energy": score} for text, score in ranked[:3]
            ],
        }
        records.append(record)
        print("=" * 80)
        print("Q:", example.question)
        print("gold:", example.answer)
        print("CoT:", record["cot_answer"], "->", (cot_text or "")[:200])
        print("Adapted:", record["adapted_answer"], "->", (adapted_text or "")[:200])

    write_json(os.path.join(config.output_dir, "case_study.json"), records)


if __name__ == "__main__":
    main()
