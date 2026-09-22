"""End-to-end smoke test on a toy RoBERTa.

Runs the complete APT recipe (prune+distill stage, recovery stage, physical
pruning, efficiency measurement) plus one baseline, on a small local model and a
synthetic SST-2-like dataset.  It takes a few seconds on CPU and requires the
``roberta-base`` *tokenizer* only (no model download).

    python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import torch
from datasets import Dataset, DatasetDict

warnings.filterwarnings("ignore")

from apt.trainer import APTConfig, APTTrainer  # noqa: E402
from data.tasks import build_task  # noqa: E402


def build_dataset():
    texts = ["a great movie", "terrible film", "i loved it", "boring and dull"] * 16
    labels = [1, 0, 1, 0] * 16
    ds = Dataset.from_dict({"sentence": texts, "label": labels})
    split = ds.train_test_split(test_size=0.25, seed=0)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def build_toy_model():
    from transformers import RobertaConfig, RobertaForSequenceClassification

    conf = RobertaConfig(
        vocab_size=50265,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
        num_labels=2,
    )
    return RobertaForSequenceClassification(conf)


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = build_dataset()
    task = build_task("sst2", dataset, tokenizer)
    train_features, eval_features = task.tokenize("train"), task.tokenize("validation")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = APTConfig(
            model_name="toy-roberta",
            task="sst2",
            output_dir=tmp,
            device="cpu",
            target_sparsity=0.6,
            epochs=2,
            distill_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            adjust_interval=3,
            eval_interval=3,
            eval_batch_size=8,
            target_rank=16,
            seed=0,
        )
        t0 = time.time()
        trainer = APTTrainer(build_toy_model(), tokenizer, task, cfg, train_features, eval_features)
        result = trainer.train()
        elapsed = time.time() - t0

    print(f"achieved sparsity : {result['achieved_sparsity']:.3f} (target {cfg.target_sparsity})")
    print(f"final metrics     : {result['final_metrics']}")
    print(f"parameters        : {result['n_parameters_dense']} -> {result['n_parameters_pruned']}")
    print(f"inference         : {result['inference'].get('relative')}")
    print(f"wall time         : {elapsed:.1f}s")

    assert result["achieved_sparsity"] >= cfg.target_sparsity - 1e-6
    assert result["n_parameters_pruned"] < result["n_parameters_dense"]
    assert result["final_metrics"]["accuracy"] >= 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
