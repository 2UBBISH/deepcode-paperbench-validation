#!/usr/bin/env python
"""SFT-LoRA baseline (Table 2/6, Appendix F.2)."""

from __future__ import annotations

import os

from _common import base_parser, build_config, write_json

from bbox_adapter.baselines.cot import CoTBaseline
from bbox_adapter.baselines.sft_lora import LoRASFTConfig, run_lora_sft
from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.eval.evaluator import evaluate_texts
from bbox_adapter.utils.seed import set_seed


def main() -> None:
    parser = base_parser("LoRA supervised fine-tuning baseline.")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--lora-r", type=int, default=128,
                        help="128 matches the 0.1B adapter, 384 the 0.3B adapter.")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)
    dataset = (args.dataset or config.dataset).lower()
    output_dir = os.path.join(config.output_dir, f"lora_r{args.lora_r}")
    os.makedirs(output_dir, exist_ok=True)

    train = load_dataset_examples(dataset, "train", data_dir=args.data_dir, seed=config.seed,
                                  cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                  max_samples=args.limit_train)
    test = load_dataset_examples(dataset, "test", data_dir=args.data_dir, seed=config.seed,
                                 cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                 max_samples=args.limit_test)
    sft_config = LoRASFTConfig(
        model_name=args.model,
        output_dir=output_dir,
        lora_r=args.lora_r,
        num_epochs=args.epochs,
        per_device_batch_size=8,
    )
    model = run_lora_sft(dataset, train, sft_config)

    from transformers import AutoTokenizer  # type: ignore

    # CoTBaseline expects a ``generate``-like client; reuse the HuggingFace
    # client with the fine-tuned model/tokenizer attached.
    from bbox_adapter.llm.hf_client import HuggingFaceClient

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    client = HuggingFaceClient(name=args.model)
    client._model = model
    client._tokenizer = tokenizer
    baseline = CoTBaseline(client, dataset, max_new_tokens=config.llm.max_new_tokens,
                           temperature=0.0)
    generations = baseline.generate(test)
    report = evaluate_texts(
        dataset, [g[0] for g in generations],
        [example.answer for example in test],
        [example.num_choices for example in test],
    )
    print(f"[sft-lora r={args.lora_r}] {dataset}: {report.accuracy:.2f}%")
    write_json(os.path.join(output_dir, "results.json"),
               {"accuracy": report.accuracy, "lora_r": args.lora_r})


if __name__ == "__main__":
    main()
