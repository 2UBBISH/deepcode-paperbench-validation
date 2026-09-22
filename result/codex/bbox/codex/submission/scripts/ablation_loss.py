#!/usr/bin/env python
"""Ablation: ranking based NCE loss versus MLM loss (Section 4.5, Table 5).

Both adapters are trained on exactly the same positive/negative supervision
(the sample bank produced by the online initialisation) and evaluated with the
same adapted inference, for the 0.1B and the 0.3B backbone.
"""

from __future__ import annotations

import os

from _common import base_parser, build_config, write_json

from bbox_adapter.adapter.trainer import AdapterTrainer
from bbox_adapter.data.loaders import load_dataset_examples
from bbox_adapter.eval.evaluator import Evaluator
from bbox_adapter.inference.adaptive_inference import AdaptedInference
from bbox_adapter.llm import build_llm
from bbox_adapter.online.bank import SampleBank, SampleEntry
from bbox_adapter.online.feedback import GroundTruthFeedback
from bbox_adapter.pipeline import build_adapter
from bbox_adapter.data.prompts import build_generator_prompt
from bbox_adapter.utils.seed import set_seed


def build_supervision(config, dataset, train, llm):
    bank = SampleBank(dataset, positive_source="ground_truth")
    for example in train:
        bank.add(SampleEntry(key=example.key, question=example.question,
                             gold_answer=example.answer, gold_solution=example.solution,
                             choices=example.choices))
    prompts = [build_generator_prompt(dataset, ex.question, ex.choices) for ex in train]
    results = llm.generate(prompts, n=config.online.init_candidates_per_question,
                           temperature=1.0, max_new_tokens=config.inference.max_solution_tokens)
    candidates = {ex.key: result.texts for ex, result in zip(train, results)}
    bank.initialize(candidates, selector=GroundTruthFeedback(dataset))
    return bank


def evaluate(adapter, config, dataset, test, llm, tag):
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
    evaluator = Evaluator(dataset, inference)
    report = evaluator.evaluate(test)
    print(f"[ablation] {tag}: {report.accuracy:.2f}%")
    return report.accuracy


def main() -> None:
    parser = base_parser("MLM vs ranking-based NCE loss (Table 5).")
    parser.add_argument("--sizes", type=str, default="0.1B,0.3B")
    parser.add_argument("--train-steps", type=int, default=None)
    args = parser.parse_args()

    config = build_config(args)
    set_seed(config.seed)
    dataset = (args.dataset or config.dataset).lower()
    os.makedirs(config.output_dir, exist_ok=True)
    if args.train_steps:
        config.adapter.num_train_steps = args.train_steps

    train = load_dataset_examples(dataset, "train", data_dir=args.data_dir, seed=config.seed,
                                  cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                  max_samples=args.limit_train)
    test = load_dataset_examples(dataset, "test", data_dir=args.data_dir, seed=config.seed,
                                 cache_dir=os.path.join(config.output_dir, "hf_cache"),
                                 max_samples=args.limit_test)
    llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))
    bank = build_supervision(config, dataset, train, llm)
    examples = bank.to_training_examples()

    results = {}
    for size in args.sizes.split(","):
        size = size.strip()
        for loss_type in ("nce", "mlm"):
            config.adapter.loss_type = loss_type
            adapter = build_adapter(config, dataset, size, device=args.device)
            if loss_type == "mlm":
                # The MLM adapter trains with a masked-language-modelling head
                # (Section 4.5) instead of the ranking-based NCE loss.
                adapter.fit(examples, num_steps=config.adapter.num_train_steps,
                            config=config.adapter)
            else:
                trainer = AdapterTrainer(adapter, config=config.adapter, device=args.device)
                trainer.fit(examples, num_steps=config.adapter.num_train_steps,
                            output_dir=os.path.join(config.output_dir, loss_type))
            adapter.save(os.path.join(config.output_dir, f"{loss_type}_{size}"))
            accuracy = evaluate(adapter, config, dataset, test, llm, f"{loss_type}-{size}")
            results[f"{loss_type}_{size}"] = accuracy
    write_json(os.path.join(config.output_dir, "ablation_results.json"), results)


if __name__ == "__main__":
    main()
