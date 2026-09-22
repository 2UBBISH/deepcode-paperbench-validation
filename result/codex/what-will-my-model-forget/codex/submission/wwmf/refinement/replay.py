"""Example selection for replay-based model refinement (Sec. 4.2 / Sec. 5.2).

Every variant replays the *same number* of upstream examples at the *same*
intervals; they only differ in how the examples are selected:

* RandomReplay   -- uniform sample from D_PT.
* ScoreReplay    -- the examples forecasted to be forgotten (threshold-, logit-
  or representation-based prediction).
* GTForgetReplay -- the ground-truth forgotten examples (upper bound; needs
  inference with the updated LM and is therefore infeasible in practice).

MIR (Aljundi et al., 2019a) and OCS (Yoon et al., 2022) are declared out of
scope by the addendum and are therefore not part of the default runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..data.types import Example
from ..forecasting.types import OnlineArtifact, UpstreamCache
from ..models.tuning import ReplayBatch


@dataclass
class ReplayPool:
    """D_PT examples plus the base PTLM's cached logits, ready for replay."""

    cache: UpstreamCache
    max_target_len: int

    @property
    def examples(self) -> List[Example]:
        return list(self.cache.examples)

    def __len__(self) -> int:
        return len(self.cache.examples)

    def batch(self, indices: Sequence[int]) -> ReplayBatch:
        """Distillation targets of the selected replay examples.

        Logits are taken from the candidate vocabulary cached with D_PT (the same
        top-k reduced vocabulary that the forecasting models use), which keeps the
        memory of the replay buffer tractable.
        """
        items = [self.cache.items[i] for i in indices]
        logits = np.stack(
            [
                np.pad(
                    item.reduced_logits.astype(np.float32),
                    ((0, max(0, self.max_target_len - item.reduced_logits.shape[0])), (0, 0)),
                )[: self.max_target_len]
                for item in items
            ],
            axis=0,
        )
        return ReplayBatch(
            inputs=[item.example.input for item in items],
            targets=[item.example.target for item in items],
            logits=logits,
            vocab_ids=self.cache.vocab.vocab_ids,
        )

    def candidate_vocab(self) -> np.ndarray:
        return self.cache.vocab.vocab_ids


class RandomReplay:
    """Baseline: replay a uniformly random mini-batch of D_PT."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, k: int, pool: ReplayPool, **kwargs) -> List[int]:
        n = len(pool)
        k = min(k, n)
        return sorted(self.rng.choice(n, size=k, replace=False).tolist())


class ScoreReplay:
    """Replay the examples predicted to be forgotten by a forecasting model."""

    name = "score"

    def __init__(
        self,
        forecaster,
        artifact_lookup: Callable[[Example], Optional[OnlineArtifact]],
        seed: int = 0,
        fill_random: bool = True,
    ) -> None:
        self.forecaster = forecaster
        self.artifact_lookup = artifact_lookup
        self.rng = np.random.default_rng(seed)
        self.fill_random = fill_random

    def select(self, k: int, pool: ReplayPool, online_example: Optional[Example] = None, **kwargs) -> List[int]:
        artifact = self.artifact_lookup(online_example) if online_example is not None else None
        if artifact is None:
            return RandomReplay(seed=int(self.rng.integers(1 << 30))).select(k, pool)
        pred = np.asarray(self.forecaster.predict(artifact)).reshape(-1)
        positive = np.flatnonzero(pred == 1).tolist()
        if len(positive) >= k:
            chosen = self.rng.choice(positive, size=k, replace=False).tolist()
        elif self.fill_random:
            chosen = positive + list(self.rng.choice(len(pool), size=k - len(positive), replace=False).tolist())
        else:
            chosen = positive
        return sorted(dict.fromkeys(chosen))


class GTForgetReplay:
    """Upper bound: replay the ground-truth forgotten upstream examples."""

    name = "gt_forget"

    def __init__(self, label_lookup: Callable[[Example], Optional[np.ndarray]], seed: int = 0) -> None:
        self.label_lookup = label_lookup
        self.rng = np.random.default_rng(seed)

    def select(self, k: int, pool: ReplayPool, online_example: Optional[Example] = None, **kwargs) -> List[int]:
        labels = self.label_lookup(online_example) if online_example is not None else None
        if labels is None:
            return RandomReplay(seed=int(self.rng.integers(1 << 30))).select(k, pool)
        positive = np.flatnonzero(np.asarray(labels).reshape(-1) == 1).tolist()
        if len(positive) >= k:
            chosen = self.rng.choice(positive, size=k, replace=False).tolist()
        else:
            chosen = positive + list(self.rng.choice(len(pool), size=k - len(positive), replace=False).tolist())
        return sorted(dict.fromkeys(chosen))


def make_replay_callback(strategy, pool: ReplayPool, online_example: Example, batch_size: int):
    """Return the ``replay_pool(step) -> ReplayBatch`` callable expected by
    :func:`wwmf.models.tuning.fix_single_error`."""

    def _callback(step: int) -> ReplayBatch:
        indices = strategy.select(batch_size, pool, online_example=online_example)
        return pool.batch(indices)

    return _callback
