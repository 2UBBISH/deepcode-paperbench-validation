#!/usr/bin/env python
"""Cost analysis: accuracy and $/1k questions (Section 4.4, Table 4).

Training cost of BBOX-ADAPTER is the API cost of the black-box LLM sampling
during the online adaptation (initialisation + one sampling round per iteration)
plus the cost of the gpt-4 rater when AI feedback is used; the adapter itself is
trained locally.  Inference cost is the API cost per 1,000 questions of the
chosen inference variant (single step / full step).
"""

from __future__ import annotations

import json
import os

from _common import base_parser, build_config, write_json

from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.eval.evaluator import Evaluator
from bbox_adapter.inference.adaptive_inference import AdaptedInference
from bbox_adapter.llm import build_llm
from bbox_adapter.baselines.cot import CoTBaseline
from bbox_adapter.utils.cost import UsageTracker

from evaluate_adapter import load_adapter


def evaluate_mode(llm, adapter, config, dataset, test, mode):
    llm.reset_usage()
    inference = AdaptedInference(
        llm=llm, adapter=adapter, dataset=dataset,
        beam_size=config.inference.beam_size,
        num_samples_per_beam=config.inference.num_samples_per_beam,
        max_sentence_steps=config.inference.max_sentence_steps,
        max_new_tokens=config.inference.max_new_tokens,
        max_solution_tokens=config.inference.max_solution_tokens,
        temperature=config.inference.temperature,
        mode=mode,
        num_single_step_candidates=config.inference.num_single_step_candidates,
    )
    report = Evaluator(dataset, inference).evaluate(test)
    return report, llm.usage


def main() -> None:
    parser = base_parser("Cost/accuracy table (Table 4).")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--training-results", type=str, default=None,
                        help="results.json of train_bbox_adapter.py (training cost).")
    parser.add_argument("--pricing-model", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--sft-training-cost", type=float, default=None,
                        help="Fine-tuning cost of the Azure-SFT baseline (USD), e.g. 153.00.")
    parser.add_argument("--sft-inference-cost-per-1k", type=float, default=None,
                        help="Inference cost of the Azure-SFT baseline (USD/1k questions).")
    args = parser.parse_args()

    config = build_config(args)
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)
    test = load_dataset_examples(dataset, "test", data_dir=args.data_dir, seed=config.seed,
                                 cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                 max_samples=args.limit_test)
    llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))
    adapter = load_adapter(args.adapter_path, device=args.device) if args.adapter_path else None

    rows = []

    # ---- base model ------------------------------------------------------
    llm.reset_usage()
    baseline = CoTBaseline(llm, dataset, max_new_tokens=config.llm.max_new_tokens,
                           temperature=0.0)
    generations = baseline.generate(test)
    from bbox_adapter.eval.evaluator import evaluate_texts

    base_report = evaluate_texts(
        dataset, [g[0] for g in generations],
        [example.answer for example in test],
        [example.num_choices for example in test],
    )
    rows.append({
        "method": "gpt-3.5-turbo",
        "accuracy": base_report.accuracy,
        "training_cost_usd": 0.0,
        "inference_cost_per_1k": llm.usage.cost_per_1k_questions(
            args.pricing_model, len(test)),
    })

    if args.sft_training_cost is not None:
        rows.append({
            "method": "Azure-SFT",
            "accuracy": None,
            "training_cost_usd": args.sft_training_cost,
            "inference_cost_per_1k": None,
        })

    # ---- BBOX-ADAPTER single step / full step ---------------------------
    if adapter is not None:
        for mode in ("single", "full"):
            report, usage = evaluate_mode(llm, adapter, config, dataset, test, mode)
            rows.append({
                "method": f"BBox-Adapter ({'Single-step' if mode == 'single' else 'Full-step'})",
                "accuracy": report.accuracy,
                "training_cost_usd": None,
                "inference_cost_per_1k": usage.cost_per_1k_questions(
                    args.pricing_model, len(test)),
            })

    # ---- training cost from the adaptation run --------------------------
    if args.training_results and os.path.exists(args.training_results):
        with open(args.training_results, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        training_cost = payload.get("training_cost_usd")
        for row in rows:
            if row["method"].startswith("BBox-Adapter"):
                row["training_cost_usd"] = training_cost

    # Cost-reduction ratios quoted in the abstract and Section 4.4
    # ("31.30x training and 1.84x inference" for the full-step variant).
    if args.sft_training_cost is not None:
        for row in rows:
            if row["method"].startswith("BBox-Adapter") and row["training_cost_usd"]:
                row["training_cost_reduction_x"] = (
                    args.sft_training_cost / row["training_cost_usd"]
                )
            if (args.sft_inference_cost_per_1k and row["inference_cost_per_1k"]
                    and row["method"].startswith("BBox-Adapter")):
                row["inference_cost_reduction_x"] = (
                    args.sft_inference_cost_per_1k / row["inference_cost_per_1k"]
                )
    write_json(os.path.join(config.output_dir, "cost_results.json"), rows)
    header = f"{'method':<32}{'acc(%)':>10}{'train($)':>12}{'infer($/1k)':>14}"
    print(header)
    for row in rows:
        train_cost = row["training_cost_usd"]
        print(f"{row['method']:<32}{row['accuracy']:>10.2f}"
              f"{(train_cost if train_cost is not None else float('nan')):>12.2f}"
              f"{row['inference_cost_per_1k']:>14.2f}")


if __name__ == "__main__":
    main()
