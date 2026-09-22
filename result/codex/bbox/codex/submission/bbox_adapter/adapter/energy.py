"""The energy function ``g_theta(x, y)`` (Section 3.1).

The adapter is a small encoder-only language model (DeBERTa-v3-base/large for
StrategyQA, GSM8K and ScienceQA and BERT-base-cased for TruthfulQA, see
Appendix H.2) whose pooled representation of the (question, answer) pair is
projected onto a scalar energy:

    g_theta(x, y) = w^T * pool(Encoder(x, y)) + b

The energy parameterises the adapted distribution
``p_theta(y|x) = p_LLM(y|x) exp(g_theta(x,y)) / Z`` (Eq. 1); training pushes the
energy of target-domain (positive) generations up and the energy of the
LLM's own generations (negative samples) down.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .base import BaseAdapter, TrainLog, TrainingExample

PAIR_TEMPLATE = "Question: {question}\nAnswer: {answer}"


def load_tokenizer(model_name: str):
    """Load a tokenizer, tolerating environments without the fast backend.

    ``microsoft/deberta-v3-base`` ships a sentencepiece vocabulary; the fast
    tokenizer needs ``sentencepiece``/``protobuf`` to be converted, so we fall
    back to the slow tokenizer when the fast one is unavailable.
    """

    from transformers import AutoTokenizer  # type: ignore

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:  # pragma: no cover - depends on the local installation
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def format_pairs(questions: Sequence[str], answers: Sequence[str]) -> List[str]:
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")
    return [
        PAIR_TEMPLATE.format(question=(question or "").strip(), answer=(answer or "").strip())
        for question, answer in zip(questions, answers)
    ]


class EnergyAdapter(BaseAdapter, nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        name: str = "energy-adapter",
        max_length: int = 512,
        pooling: str = "cls",
        dropout: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        nn.Module.__init__(self)
        BaseAdapter.__init__(self, name=name, max_length=max_length, device=device)
        self.encoder = encoder
        self.tokenizer = tokenizer
        hidden_size = getattr(encoder.config, "hidden_size", 768)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.energy_head = nn.Linear(hidden_size, 1)
        # Small initialisation keeps the initial energies close to zero, which
        # is the identity adapter p_theta = p_LLM at t = 0.
        nn.init.normal_(self.energy_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.energy_head.bias)
        if self._device is not None:
            self.to(self._device)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_length: int = 512,
        pooling: str = "cls",
        dropout: float = 0.0,
        device: Optional[str] = None,
        freeze_encoder_layers: int = 0,
    ) -> "EnergyAdapter":
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        tokenizer = load_tokenizer(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder_layers > 0:
            for index, layer in enumerate(encoder.encoder.layer):  # type: ignore[attr-defined]
                if index < freeze_encoder_layers:
                    for parameter in layer.parameters():
                        parameter.requires_grad = False
        adapter = cls(
            encoder=encoder,
            tokenizer=tokenizer,
            name=model_name,
            max_length=max_length,
            pooling=pooling,
            dropout=dropout,
            device=device,
        )
        return adapter

    # ------------------------------------------------------------------ core
    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:  # some checkpoints return a tuple
            hidden = outputs[0]
        pooled = self.pool(hidden, attention_mask)
        energy = self.energy_head(self.dropout(pooled)).squeeze(-1)
        return energy

    # --------------------------------------------------------------- scoring
    def tokenize(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            # DeBERTa-v3 is trained without token type embeddings
            # (``type_vocab_size == 0``) and raises when they are passed.
            return_token_type_ids=False,
        )

    def _forward_batch(self, batch):
        # Defensive: some tokenizers always return token_type_ids.
        if getattr(self.encoder.config, "type_vocab_size", 0) in (0, None):
            batch.pop("token_type_ids", None)
        return self.forward(**batch)

    @torch.no_grad()
    def score_batch(self, questions: Sequence[str], answers: Sequence[str]) -> torch.Tensor:
        self.eval()
        texts = format_pairs(questions, answers)
        batch = self.tokenize(texts)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        energies = self._forward_batch(batch)
        return energies.detach()

    def score_lists(self, question: str, answers: Sequence[str]) -> List[float]:
        if not answers:
            return []
        questions = [question] * len(answers)
        return self.score_batch(questions, answers).tolist()

    # ------------------------------------------------------------------- fit
    def fit(self, examples: Sequence[TrainingExample], num_steps: Optional[int] = None) -> TrainLog:
        from .trainer import AdapterTrainer  # local import avoids a cycle

        trainer = AdapterTrainer(self)
        return trainer.fit(examples, num_steps=num_steps)

    # ---------------------------------------------------------------- storage
    def save(self, directory: str) -> None:
        directory = self._ensure_dir(directory)
        self.encoder.save_pretrained(os.path.join(directory, "encoder"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(directory, "encoder"))
        torch.save(self.energy_head.state_dict(), os.path.join(directory, "energy_head.pt"))
        with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "name": self.name,
                    "max_length": self.max_length,
                    "pooling": self.pooling,
                    "loss_type": self.loss_type,
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "EnergyAdapter":
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        encoder_dir = os.path.join(directory, "encoder")
        with open(os.path.join(directory, "adapter_config.json"), "r", encoding="utf-8") as handle:
            config = json.load(handle)
        encoder = AutoModel.from_pretrained(encoder_dir)
        tokenizer = load_tokenizer(encoder_dir)
        adapter = cls(
            encoder=encoder,
            tokenizer=tokenizer,
            name=config.get("name", encoder_dir),
            max_length=config.get("max_length", 512),
            pooling=config.get("pooling", "cls"),
            device=device,
        )
        state = torch.load(
            os.path.join(directory, "energy_head.pt"), map_location="cpu"
        )
        adapter.energy_head.load_state_dict(state)
        return adapter
