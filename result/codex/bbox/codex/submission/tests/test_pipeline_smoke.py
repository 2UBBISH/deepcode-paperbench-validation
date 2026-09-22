"""End-to-end smoke test: Algorithm 1 with a mock LLM and a tiny adapter."""

from __future__ import annotations

import json
import os

import pytest

from bbox_adapter.config import RunConfig
from bbox_adapter.pipeline import run_online_adaptation

from ._tiny import save_tiny_encoder


QUESTIONS = [
    ("What is 2 + 2?", "4"),
    ("What is 3 + 5?", "8"),
    ("What is 10 - 4?", "6"),
    ("What is 6 * 2?", "12"),
]


def write_local_dataset(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    for split, items in (("train", QUESTIONS), ("test", QUESTIONS[:2])):
        with open(os.path.join(directory, f"gsm8k_{split}.jsonl"), "w", encoding="utf-8") as handle:
            for question, answer in items:
                handle.write(
                    json.dumps(
                        {
                            "question": question,
                            "answer": f"Adding the numbers gives {answer}.\n#### {answer}",
                        }
                    )
                    + "\n"
                )


@pytest.mark.slow
def test_online_adaptation_smoke(tmp_path):
    data_dir = tmp_path / "data"
    write_local_dataset(str(data_dir))
    adapter_dir = save_tiny_encoder(str(tmp_path / "tiny_encoder"))

    config = RunConfig(dataset="gsm8k")
    config.adapter.model_name = adapter_dir
    config.adapter.max_length = 64
    config.adapter.num_train_steps = 4
    config.adapter.batch_size = 4
    config.inference.beam_size = 2
    config.inference.num_samples_per_beam = 2
    config.inference.max_sentence_steps = 2
    config.inference.max_solution_tokens = 16
    config.online.num_iterations = 2
    config.online.init_candidates_per_question = 3
    config.online.num_candidates_per_question = 2
    config.online.positive_source = "ground_truth"
    config.llm.provider = "mock"
    config.llm.name = "mock-llm"
    config.output_dir = str(tmp_path / "run")

    result = run_online_adaptation(
        config,
        dataset="gsm8k",
        data_dir=str(data_dir),
        max_train=4,
        max_test=2,
        logger=lambda message: None,
    )

    assert len(result["history"]) == 2
    assert os.path.exists(os.path.join(config.output_dir, "adapter_final", "adapter_config.json"))
    assert os.path.exists(os.path.join(config.output_dir, "online_history.json"))
    assert os.path.exists(os.path.join(config.output_dir, "bank_iter0.jsonl"))
    assert result["usage"]["prompt_tokens"] > 0

    with open(os.path.join(config.output_dir, "online_history.json"), "r", encoding="utf-8") as handle:
        history = json.load(handle)
    assert history[0]["iteration"] == 0
    assert history[0]["bank_positives_per_question"] >= 1.0
