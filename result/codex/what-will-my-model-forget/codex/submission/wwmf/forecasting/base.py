"""Common interface and helpers of the forecasting models (Sec. 3)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import FORECAST_TRAIN, TOPK_CACHED_LOGITS, ExperimentConfig
from ..evaluation.metrics import binary_f1_scores
from ..utils import ensure_dir
from .types import OnlineArtifact, UpstreamCache


@dataclass
class ForecastContext:
    """Everything a forecasting model is allowed to use."""

    cfg: ExperimentConfig
    upstream_cache: UpstreamCache
    train_artifacts: List[OnlineArtifact] = field(default_factory=list)
    device: str = "cpu"
    hidden_dim: int = FORECAST_TRAIN["hidden_dim"]
    topk: int = TOPK_CACHED_LOGITS
    seed: int = 0
    verbose: bool = True
    output_dir: Optional[str] = None

    @property
    def train_labels(self) -> np.ndarray:
        return np.stack([a.labels for a in self.train_artifacts], axis=0).astype(np.int8)

    @property
    def eval_mask(self) -> np.ndarray:
        """D_hat_PT: upstream examples that f_0 answers correctly (addendum)."""
        return self.upstream_cache.correct_mask


class BaseForecaster:
    """``g: (<x_i, y_i>, <x_j, y_j>) -> z_ij in {0, 1}`` (Sec. 2)."""

    name = "base"
    #: True when the method has parameters to learn (used by the tables).
    trainable = False

    def fit(self, ctx: ForecastContext) -> "BaseForecaster":  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, artifact: OnlineArtifact) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def scores(self, artifact: OnlineArtifact) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    # ----------------------------------------------------------------------------------
    def evaluate(
        self,
        artifacts: Sequence[OnlineArtifact],
        ctx: ForecastContext,
    ) -> Dict[str, float]:
        """F1 / precision / recall over all (online, upstream) pairs of the split."""
        mask = ctx.eval_mask
        y_true: List[int] = []
        y_pred: List[int] = []
        for art in artifacts:
            if art.labels is None:
                raise ValueError("ground-truth forgetting labels are required for evaluation")
            pred = self.predict(art)[mask]
            truth = np.asarray(art.labels)[mask]
            y_true.extend(truth.tolist())
            y_pred.extend(pred.tolist())
        metrics = binary_f1_scores(y_true, y_pred)
        metrics["n_pairs"] = float(len(y_true))
        metrics["positive_rate"] = float(np.mean(y_true)) if y_true else 0.0
        return metrics


# --------------------------------------------------------------------------------------
# pair sampling (Appendix B: 8 positive and 8 negative pairs per mini-batch)
# --------------------------------------------------------------------------------------
def sample_pairs(
    labels: np.ndarray,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
    allowed_upstream: Optional[np.ndarray] = None,
    online_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample positive and negative ``(i, j)`` pairs.

    ``i`` indexes the online learned examples, ``j`` the upstream examples.
    """
    labels = np.asarray(labels)
    n_online, n_upstream = labels.shape
    if allowed_upstream is None:
        allowed_upstream = np.ones(n_upstream, dtype=bool)
    online_indices = np.arange(n_online) if online_indices is None else np.asarray(online_indices)
    sub = labels[np.ix_(online_indices, np.arange(n_upstream)[allowed_upstream])]
    pos_local = np.argwhere(sub == 1)
    neg_local = np.argwhere(sub == 0)
    upstream_ids = np.arange(n_upstream)[allowed_upstream]

    def _draw(pool: np.ndarray, k: int) -> np.ndarray:
        if len(pool) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        take = rng.choice(len(pool), size=min(k, len(pool)), replace=len(pool) < k)
        rows = pool[take]
        return np.stack([online_indices[rows[:, 0]], upstream_ids[rows[:, 1]]], axis=1)

    return _draw(pos_local, n_pos), _draw(neg_local, n_neg)


def frequency_prior(labels: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """``b_j = log p(z_j = 1) - log p(z_j = 0)`` over ``D_R^Train`` (Sec. 3.3)."""
    labels = np.asarray(labels, dtype=np.float64)
    p1 = labels.mean(axis=0)
    p1 = np.clip(p1, epsilon, 1 - epsilon)
    return np.log(p1) - np.log(1 - p1)


def save_metrics(path: str, metrics: Dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf8") as fh:
        json.dump(metrics, fh, indent=2)
