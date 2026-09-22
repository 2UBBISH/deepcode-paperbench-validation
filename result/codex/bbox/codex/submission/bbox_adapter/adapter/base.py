"""Interfaces shared by the NCE adapter and the MLM ablation adapter."""

from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class TrainingExample:
    """One question together with its positive and negative samples.

    ``positive`` plays the role of ``y_+ ~ p_data`` and ``negatives`` the role of
    ``y_- ~ p_theta`` in Eq. (3).
    """

    key: str
    question: str
    positive: str
    negatives: List[str] = field(default_factory=list)
    extra_positives: List[str] = field(default_factory=list)

    def all_positives(self) -> List[str]:
        return [self.positive] + list(self.extra_positives)


@dataclass
class TrainLog:
    """Learning curves (Appendix K)."""

    steps: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    positive_energy: List[float] = field(default_factory=list)
    negative_energy: List[float] = field(default_factory=list)
    pairwise_accuracy: List[float] = field(default_factory=list)

    def append(self, step: int, loss: float, positive_energy: float,
               negative_energy: float, pairwise_accuracy: float) -> None:
        self.steps.append(step)
        self.loss.append(loss)
        self.positive_energy.append(positive_energy)
        self.negative_energy.append(negative_energy)
        self.pairwise_accuracy.append(pairwise_accuracy)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "steps": self.steps,
            "loss": self.loss,
            "positive_energy": self.positive_energy,
            "negative_energy": self.negative_energy,
            "pairwise_accuracy": self.pairwise_accuracy,
        }


class BaseAdapter(abc.ABC):
    """Common API of every adapter used as an evaluator during inference."""

    #: Name of the adapter family, e.g. ``"nce"`` or ``"mlm"``.
    loss_type: str = "nce"

    def __init__(self, name: str, max_length: int = 512, device: Optional[str] = None) -> None:
        self.name = name
        self.max_length = max_length
        self._device = device

    # ------------------------------------------------------------------ device
    @property
    def device(self):
        import torch

        if self._device is not None:
            return torch.device(self._device)
        try:
            return next(self.parameters()).device  # type: ignore[attr-defined]
        except (AttributeError, StopIteration):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------- api
    @abc.abstractmethod
    def score_batch(self, questions: Sequence[str], answers: Sequence[str]):
        """Return the adapter score of every (question, answer) pair."""

    @abc.abstractmethod
    def fit(self, examples: Sequence[TrainingExample], num_steps: Optional[int] = None) -> TrainLog:
        """Update the adapter parameters on positives/negatives (Eq. 3/7)."""

    @abc.abstractmethod
    def save(self, directory: str) -> None:
        ...

    # ----------------------------------------------------------------- helper
    @staticmethod
    def _ensure_dir(directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        return directory
