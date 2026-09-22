"""The energy adapter scores pairs and its training raises the gap."""

from __future__ import annotations

import torch

from bbox_adapter.adapter.base import TrainingExample
from bbox_adapter.adapter.trainer import AdapterTrainer
from bbox_adapter.config import AdapterConfig

from ._tiny import build_tiny_adapter


def test_score_batch_shapes():
    adapter = build_tiny_adapter()
    scores = adapter.score_batch(
        ["Is the sky blue?", "Is the sky blue?"],
        ["Yes, the sky is blue.", "No, the sky is green."],
    )
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_energy_head_starts_near_zero():
    adapter = build_tiny_adapter()
    scores = adapter.score_lists("Q", ["A", "B longer answer", "C"])
    assert max(abs(value) for value in scores) < 1.0


def test_training_increases_positive_energy_margin():
    torch.manual_seed(0)
    adapter = build_tiny_adapter(
        ["four", "five", "answer", "the", "is", "4", "5", "1", "2", "3", "6", "7"]
    )
    examples = [
        TrainingExample(
            key=f"k{i}",
            question="What is 2 + 2?",
            positive=f"The answer is four (four) {i}.",
            negatives=[f"The answer is five (five) {i}."],
        )
        for i in range(8)
    ]
    before = adapter.score_batch(
        [ex.question for ex in examples], [ex.positive for ex in examples]
    ) - adapter.score_batch(
        [ex.question for ex in examples], [ex.negatives[0] for ex in examples]
    )
    config = AdapterConfig(num_train_steps=60, batch_size=4, learning_rate=5e-4,
                           weight_decay=0.0, alpha=0.0, seed=0, max_length=128)
    trainer = AdapterTrainer(adapter, config=config)
    trainer.fit(examples, num_steps=60, log_every=20)
    after = adapter.score_batch(
        [ex.question for ex in examples], [ex.positive for ex in examples]
    ) - adapter.score_batch(
        [ex.question for ex in examples], [ex.negatives[0] for ex in examples]
    )
    assert after.mean().item() > before.mean().item()
    assert trainer.log.loss[-1] < trainer.log.loss[0]


def test_save_and_load_roundtrip(tmp_path):
    adapter = build_tiny_adapter()
    directory = str(tmp_path / "adapter")
    adapter.save(directory)
    from bbox_adapter.adapter.energy import EnergyAdapter

    restored = EnergyAdapter.load(directory)
    original = adapter.score_lists("Q", ["A", "B"])
    reloaded = restored.score_lists("Q", ["A", "B"])
    assert all(abs(a - b) < 1e-6 for a, b in zip(original, reloaded))


def test_listwise_softmax_training_runs_and_separates():
    torch.manual_seed(0)
    adapter = build_tiny_adapter(["four", "five", "six", "7", "8", "9"])
    examples = [
        TrainingExample(
            key=f"k{i}",
            question="How many?",
            positive=f"The answer is four (four) {i}.",
            negatives=[
                f"The answer is five (five) {i}.",
                f"The answer is six (six) {i}.",
                f"The answer is 8 (eight) {i}.",
            ],
        )
        for i in range(6)
    ]
    config = AdapterConfig(num_train_steps=50, batch_size=4, learning_rate=5e-4,
                           weight_decay=0.0, alpha=0.0, seed=0, max_length=128,
                           listwise_softmax=True, num_softmax_negatives=4)
    trainer = AdapterTrainer(adapter, config=config)
    log = trainer.fit(examples, num_steps=50, log_every=25)
    assert log.loss[-1] < log.loss[0]
    positive = adapter.score_batch(
        [ex.question for ex in examples], [ex.positive for ex in examples]
    )
    negative = adapter.score_batch(
        [ex.question for ex in examples], [ex.negatives[0] for ex in examples]
    )
    assert positive.mean().item() > negative.mean().item()
