#!/usr/bin/env python
"""Azure OpenAI fine-tuning baseline (Appendix F.2, Table 9 grid)."""

from __future__ import annotations

import os

from _common import base_parser, build_config, write_json

from bbox_adapter.baselines.cot import CoTBaseline
from bbox_adapter.baselines.sft_azure import AzureSFTConfig, run_azure_sft
from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.eval.evaluator import evaluate_texts
from bbox_adapter.llm import build_llm
from bbox_adapter.utils.seed import set_seed


def main() -> None:
    parser = base_parser("Azure-SFT baseline (fine-tune gpt-3.5-turbo through the API).")
    parser.add_argument("--epochs", type=int, default=3,
                        help="3 by default; 5 for the best setting of Table 9.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr-multiplier", type=float, default=None)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)
    train = load_dataset_examples(dataset, "train", data_dir=args.data_dir, seed=config.seed,
                                  cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                  max_samples=args.limit_train)
    sft_config = AzureSFTConfig(
        output_dir=config.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate_multiplier=args.lr_multiplier,
    )
    model_name = run_azure_sft(dataset, train, sft_config)
    print(f"[azure-sft] fine-tuned model: {model_name}")

    if args.evaluate:
        test = load_dataset_examples(dataset, "test", data_dir=args.data_dir, seed=config.seed,
                                     cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                     max_samples=args.limit_test)
        config.llm.name = model_name
        config.llm.deployment = model_name
        llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))
        baseline = CoTBaseline(llm, dataset, max_new_tokens=config.llm.max_new_tokens,
                               temperature=0.0)
        generations = baseline.generate(test)
        report = evaluate_texts(
            dataset, [g[0] for g in generations],
            [example.answer for example in test],
            [example.num_choices for example in test],
        )
        print(f"[azure-sft] {dataset}: {report.accuracy:.2f}%")
        write_json(os.path.join(config.output_dir, "results.json"),
                   {"accuracy": report.accuracy, "model": model_name})


if __name__ == "__main__":
    main()
