#!/usr/bin/env python3
"""End-to-end smoke test of the reproduction pipeline without downloads.

Builds a *randomly initialised* tiny GPT-2, fabricates a small synthetic
multiple-choice benchmark in the shape of the P3/harness documents, runs the
full zero-shot sweep (several guidance strengths), then runs the FLOP/ANCOVA
analysis on the produced records.  It exercises exactly the code paths used
by the real experiments and finishes in a few seconds on CPU.

    python experiments/smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfglm import harness, tasks as task_module  # noqa: E402
from cfglm.flops import flops_per_token  # noqa: E402


class WhitespaceTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def _enc(self, text):
        return [((abs(hash(w)) % (self.vocab_size - 2)) + 2) for w in text.split()]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self._enc(text)
        if return_tensors == "pt":
            return type("E", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})
        return type("E", (), {"input_ids": ids})

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(i) for i in row.tolist()) for row in ids]


def synthetic_docs(n: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        correct = rng.integers(0, 3)
        docs.append(
            {
                "question": f"question number {i} about topic {rng.integers(0, 10)}",
                "choices": [f" answer option {j}" for j in range(3)],
                "gold": int(correct),
            }
        )
    return docs


def main() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=256, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    tokenizer = WhitespaceTokenizer(config.vocab_size)

    task = task_module.Task(
        name="synthetic_mc",
        dataset_path="synthetic",
        doc_to_text=lambda d: d["question"],
        doc_to_choices=lambda d: d["choices"],
        doc_to_gold=lambda d: d["gold"],
    )
    task_module.TASKS[task.name] = task

    docs = synthetic_docs()
    harness.load_examples = lambda task, limit=None, split=None: docs  # type: ignore

    records = []
    for gamma in (1.0, 1.1, 1.25, 1.5, 1.75):
        result = harness.evaluate_task(model, tokenizer, "synthetic_mc", gamma, batch_size=4)
        records.append(
            {
                "task": "synthetic_mc",
                "model": "tiny-random-gpt2",
                "gamma": gamma,
                "cfg": gamma != 1.0,
                "acc": result.metrics["acc"],
                "acc_norm": result.metrics["acc_norm"],
                "flops_per_token": flops_per_token(config, 64, n_passes=2 if gamma != 1.0 else 1),
            }
        )
        print(f"gamma={gamma}: acc={result.metrics['acc']:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "zeroshot.json")
        with open(path, "w") as fh:
            json.dump(records, fh)
        from run_flops_ancova import analyse_task, load_points

        by_task = load_points(path)
        analysis = analyse_task("synthetic_mc", by_task["synthetic_mc"])
        print("ANCOVA:", {k: round(v, 4) if isinstance(v, float) else v for k, v in analysis.items()})

    print("\nSmoke test finished: the full sweep + ANCOVA pipeline runs end to end.")


if __name__ == "__main__":
    main()
