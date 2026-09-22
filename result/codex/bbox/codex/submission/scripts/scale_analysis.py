#!/usr/bin/env python
"""Scale analysis: number of beams and number of online iterations (Figure 3)."""

from __future__ import annotations

import csv
import json
import os

from _common import base_parser, build_config, write_json

from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.eval.evaluator import Evaluator
from bbox_adapter.inference.adaptive_inference import AdaptedInference
from bbox_adapter.llm import build_llm
from bbox_adapter.pipeline import build_adapter
from bbox_adapter.utils.seed import set_seed

from evaluate_adapter import load_adapter


def evaluate_with(llm, adapter, config, dataset, test, beam_size, num_samples_per_beam=None):
    inference = AdaptedInference(
        llm=llm,
        adapter=adapter,
        dataset=dataset,
        beam_size=beam_size,
        num_samples_per_beam=num_samples_per_beam or max(beam_size, config.inference.num_samples_per_beam),
        max_sentence_steps=config.inference.max_sentence_steps,
        max_new_tokens=config.inference.max_new_tokens,
        max_solution_tokens=config.inference.max_solution_tokens,
        temperature=config.inference.temperature,
        mode="full",
    )
    return Evaluator(dataset, inference).evaluate(test).accuracy


def main() -> None:
    parser = base_parser("Scale analysis on StrategyQA (Figure 3).")
    parser.add_argument("--sizes", type=str, default="0.1B,0.3B")
    parser.add_argument("--beam-sizes", type=str, default="1,3,5")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Directory with adapter_iter{0..T-1} checkpoints.")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)
    test = load_dataset_examples(dataset, "test", data_dir=args.data_dir, seed=config.seed,
                                 cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                 max_samples=args.limit_test)
    llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))

    rows = []
    for size in args.sizes.split(","):
        size = size.strip()
        run_dir = args.run_dir or os.path.join(config.output_dir, f"adapters_{size}")

        # ---- number of online iterations (T = 0 .. max-iterations) --------
        for iteration in range(args.max_iterations + 1):
            checkpoint = os.path.join(run_dir, f"adapter_iter{iteration - 1}")
            if iteration == 0:
                # T = 0 is the randomly initialised (un-finetuned) adapter.
                adapter = build_adapter(config, dataset, size, device=args.device)
            elif os.path.isdir(checkpoint):
                adapter = load_adapter(checkpoint, device=args.device)
            else:
                print(f"[scale] missing {checkpoint}; skipping T={iteration}")
                continue
            accuracy = evaluate_with(llm, adapter, config, dataset, test,
                                     config.inference.beam_size)
            rows.append({"adapter_size": size, "axis": "iterations",
                         "value": iteration, "accuracy": accuracy})
            print(f"[scale] size={size} T={iteration}: {accuracy:.2f}%")

        # ---- number of beams ---------------------------------------------
        trained_dir = os.path.join(run_dir, f"adapter_iter{config.online.num_iterations - 1}")
        if os.path.isdir(trained_dir):
            adapter = load_adapter(trained_dir, device=args.device)
            for beam_size in [int(b) for b in args.beam_sizes.split(",")]:
                accuracy = evaluate_with(llm, adapter, config, dataset, test, beam_size,
                                         num_samples_per_beam=beam_size)
                rows.append({"adapter_size": size, "axis": "beams",
                             "value": beam_size, "accuracy": accuracy})
                print(f"[scale] size={size} beam={beam_size}: {accuracy:.2f}%")

    write_json(os.path.join(config.output_dir, "scale_results.json"), rows)
    csv_path = os.path.join(config.output_dir, "scale_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["adapter_size", "axis", "value", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[scale] wrote {csv_path}")

    # Figure 3 of the paper: (a) beam size, (b) number of iterations.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for axis, title in (("beams", "number of beams"), ("iterations", "online iteration T")):
            subset = [row for row in rows if row["axis"] == axis]
            if not subset:
                continue
            figure, ax = plt.subplots(figsize=(4.5, 3.2))
            for size in sorted({row["adapter_size"] for row in subset}):
                points = sorted(
                    (row["value"], row["accuracy"]) for row in subset if row["adapter_size"] == size
                )
                ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                        label=f"adapter {size}")
            ax.set_xlabel(title)
            ax.set_ylabel("accuracy (%)")
            ax.set_title(f"StrategyQA: {title}")
            ax.legend()
            figure.tight_layout()
            figure.savefig(os.path.join(config.output_dir, f"figure3_{axis}.png"), dpi=200)
            print(f"[scale] wrote figure3_{axis}.png")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[scale] matplotlib unavailable, skipping plots ({exc})")


if __name__ == "__main__":
    main()
