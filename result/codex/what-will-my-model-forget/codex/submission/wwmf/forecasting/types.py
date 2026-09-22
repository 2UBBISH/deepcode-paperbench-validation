"""Data structures shared by the three forecasting methods (Sec. 3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data.types import Dataset, Example


@dataclass
class CandidateVocab:
    """A pruned output vocabulary.

    Sec. 3.2: "In practice, we only cache top k = 100 largest logits for each
    token in y_j."  We collect the top-k ids of every upstream example (plus the
    gold token ids) into one candidate set and store all logits restricted to it.
    """

    vocab_ids: np.ndarray            # [C] original vocabulary ids
    topk: int = 100
    max_size: int = 512

    def __len__(self) -> int:
        return int(self.vocab_ids.shape[0])

    def reduce(self, logits: np.ndarray) -> np.ndarray:
        return logits[:, self.vocab_ids]

    def to_dict(self) -> Dict:
        return {"vocab_ids": self.vocab_ids.tolist(), "topk": self.topk, "max_size": self.max_size}

    @classmethod
    def from_dict(cls, obj: Dict) -> "CandidateVocab":
        return cls(np.asarray(obj["vocab_ids"], dtype=np.int64), int(obj["topk"]), int(obj["max_size"]))


@dataclass
class UpstreamExampleCache:
    """Everything that can be pre-computed for one upstream example x_j."""

    example: Example
    reduced_logits: np.ndarray        # [T_j, C] pre-softmax logits of the gold tokens of f_0
    gold_ids: np.ndarray              # [T_j] gold token ids
    mask: np.ndarray                  # [T_j] bool, True for non-padding positions
    decoder_reps: np.ndarray          # [T_j, H] frozen decoder representations (fixed logit kernel)
    pooled_rep: np.ndarray            # [H] mean-pooled representation of <x_j, y_j>
    correct: bool                     # f_0(x_j) == y_j  (i.e. x_j is in D_hat_PT)


@dataclass
class UpstreamCache:
    """Cached logits / representations of ``D_PT`` (all methods reuse these)."""

    examples: Dataset
    vocab: CandidateVocab
    items: List[UpstreamExampleCache] = field(default_factory=list)
    base_em: float = float("nan")

    def __len__(self) -> int:
        return len(self.items)

    @property
    def correct_mask(self) -> np.ndarray:
        return np.asarray([item.correct for item in self.items], dtype=bool)

    def subset(self, indices) -> "UpstreamCache":
        return UpstreamCache(
            examples=[self.examples[i] for i in indices],
            vocab=self.vocab,
            items=[self.items[i] for i in indices],
            base_em=self.base_em,
        )

    def pooled_matrix(self) -> np.ndarray:
        return np.stack([item.pooled_rep for item in self.items], axis=0)


@dataclass
class OnlineArtifact:
    """What one online learning example contributes to forecasting."""

    example: Example
    delta_logits: np.ndarray          # [T_i, C] f_i(x_i) - f_0(x_i), reduced to the candidate vocab
    gold_ids: np.ndarray              # [T_i] gold token ids of y_i
    mask: np.ndarray                  # [T_i] bool
    labels: Optional[np.ndarray] = None      # [N_PT] ground truth z_ij (None for test-time input)
    edit_success: bool = False               # f_i(x_i) == y_i after the update
    n_steps: int = 0
    #: frozen representation h(x_i, y_i) of the *base* PTLM, needed by the
    #: fixed logit-based forecaster of Sec. 4.2 (None unless requested).
    frozen_token_reps: Optional[np.ndarray] = None

    @property
    def key(self) -> str:
        return self.example.key
