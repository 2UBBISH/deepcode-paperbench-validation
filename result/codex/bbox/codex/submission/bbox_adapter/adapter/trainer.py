"""Adapter update (Eq. 7) with AdamW, lr 5e-6 / bs 64 / 6000 steps."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from .base import TrainLog, TrainingExample
from .losses import nce_accuracy, ranking_nce_loss, ranking_nce_softmax_loss


class AdapterTrainer:
    """Runs the inner optimisation of the online adaptation loop."""

    def __init__(self, adapter, config=None, device: Optional[str] = None) -> None:
        from ..config import AdapterConfig

        self.adapter = adapter
        self.config = config or AdapterConfig(
            model_name=getattr(adapter, "name", "adapter"),
            max_length=getattr(adapter, "max_length", 512),
        )
        self.device = torch.device(device) if device else adapter.device
        self.log = TrainLog()

    # ------------------------------------------------------------------ data
    @staticmethod
    def _expand_examples(examples: Sequence[TrainingExample]):
        """Yield (question, positive, negatives, extra_positives) records."""

        for example in examples:
            yield example

    @staticmethod
    def _sample_batch(examples: Sequence[TrainingExample], batch_size: int,
                      rng: random.Random) -> List[TrainingExample]:
        if len(examples) >= batch_size:
            return rng.sample(list(examples), batch_size)
        return [rng.choice(list(examples)) for _ in range(batch_size)]

    # ------------------------------------------------------------------- fit
    def fit(self, examples: Sequence[TrainingExample], num_steps: Optional[int] = None,
            log_every: int = 100, save_curves: bool = True,
            output_dir: Optional[str] = None) -> TrainLog:
        cfg = self.config
        num_steps = int(num_steps or cfg.num_train_steps)
        examples = list(examples)
        if not examples:
            raise ValueError("Cannot fit the adapter without training examples")

        rng = random.Random(cfg.seed)
        adapter = self.adapter
        adapter.train()
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = None
        if cfg.warmup_ratio > 0:
            warmup_steps = max(1, int(cfg.warmup_ratio * num_steps))

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        log = TrainLog()
        for step in range(1, num_steps + 1):
            batch = self._sample_batch(examples, cfg.batch_size, rng)
            if cfg.listwise_softmax:
                # Eq. (2): one positive and K-1 negatives form the candidate set
                # whose posterior is normalised with a softmax.
                k = max(2, cfg.num_softmax_negatives)
                questions, positive_answers, groups = [], [], []
                for example in batch:
                    candidates = example.negatives or [""]
                    positives = example.all_positives() or [example.positive]
                    group = []
                    for _ in range(k):
                        group.append(rng.choice(candidates))
                    questions.append(example.question)
                    positive_answers.append(rng.choice(positives))
                    groups.append([rng.choice(positives)] + group)
                flat_questions = [q for q, group in zip(questions, groups) for _ in group]
                flat_answers = [answer for group in groups for answer in group]
                batch_encoding = adapter.tokenize(
                    adapter_pair_texts(flat_questions, flat_answers)
                )
            else:
                # Eq. (3): pairwise ranking with one positive and one negative.
                questions, positive_answers, negative_answers = [], [], []
                for example in batch:
                    candidates = example.negatives or [""]
                    positives = example.all_positives() or [example.positive]
                    for _ in range(max(1, cfg.negatives_per_example)):
                        questions.append(example.question)
                        # Outcome supervision and the "combined" setting
                        # contribute extra positives; draw uniformly from them.
                        positive_answers.append(rng.choice(positives))
                        negative_answers.append(rng.choice(candidates))
                batch_encoding = adapter.tokenize(
                    adapter_pair_texts(questions, positive_answers)
                    + adapter_pair_texts(questions, negative_answers)
                )
            batch_encoding = {k: v.to(self.device) for k, v in batch_encoding.items()}
            forward = getattr(adapter, "_forward_batch", None)
            if forward is not None:
                energies = forward(batch_encoding)
            else:
                energies = adapter(**batch_encoding)

            if cfg.listwise_softmax:
                grouped = energies.view(len(questions), len(groups[0]))
                positive_energies = grouped[:, 0]
                # For the learning curves we report the mean negative energy of
                # every candidate set; the loss itself uses the full (B, K)
                # tensor (Eq. 2).
                negative_energies = grouped[:, 1:].mean(dim=1)
                loss = ranking_nce_softmax_loss(grouped, alpha=cfg.alpha)
            else:
                half = len(questions)
                positive_energies = energies[:half]
                negative_energies = energies[half:]
                loss = ranking_nce_loss(
                    positive_energies, negative_energies, alpha=cfg.alpha
                )

            normalized = loss / max(1, cfg.grad_accumulation_steps)
            normalized.backward()
            if step % max(1, cfg.grad_accumulation_steps) == 0:
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            if step % log_every == 0 or step == 1 or step == num_steps:
                log.append(
                    step=step,
                    loss=float(loss.detach().cpu()),
                    positive_energy=float(positive_energies.mean().detach().cpu()),
                    negative_energy=float(negative_energies.mean().detach().cpu()),
                    pairwise_accuracy=nce_accuracy(
                        positive_energies.detach(), negative_energies.detach()
                    ),
                )

        adapter.eval()
        self.log = log
        if save_curves and output_dir:
            self.save_curves(os.path.join(output_dir, "train_curves.csv"))
        return log

    # ------------------------------------------------------------------ misc
    def save_curves(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "loss", "positive_energy", "negative_energy", "pairwise_accuracy"])
            for row in zip(
                self.log.steps,
                self.log.loss,
                self.log.positive_energy,
                self.log.negative_energy,
                self.log.pairwise_accuracy,
            ):
                writer.writerow(row)


def adapter_pair_texts(questions: Sequence[str], answers: Sequence[str]) -> List[str]:
    """Deferred import of :func:`bbox_adapter.adapter.energy.format_pairs`."""

    from .energy import format_pairs

    return format_pairs(questions, answers)
