"""Sec. 3.1 -- Frequency-threshold based forecasting.

    g(<x_i, y_i>, <x_j, y_j>) = 1[ |{i in D_R^Train : z_ij = 1}| >= gamma ]

``gamma`` is tuned to maximise the F1 on ``D_R^Train``.  The method ignores the
interaction between the two examples and only uses how often the upstream
example was forgotten.  Note the paper's ``J = |D_PT|`` is a typo: the count runs
over the online learned examples of ``D_R^Train`` (see addendum for Sec. 3.1).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..evaluation.metrics import precision_recall_f1
from .base import BaseForecaster, ForecastContext


class ThresholdForecaster(BaseForecaster):
    name = "threshold"
    trainable = False

    def __init__(self, gamma: Optional[int] = None) -> None:
        self.gamma = gamma
        self.counts: Optional[np.ndarray] = None

    def fit(self, ctx: ForecastContext) -> "ThresholdForecaster":
        labels = ctx.train_labels                       # [n_online, n_upstream]
        self.counts = labels.sum(axis=0).astype(np.int64)
        if self.gamma is None:
            self.gamma = self._tune_gamma(labels, ctx)
        return self

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _tune_gamma(labels: np.ndarray, ctx: ForecastContext) -> int:
        """Maximise the F1 on ``D_R^Train`` over integer thresholds.

        Implemented in O(n_upstream log n_upstream) per candidate using the fact
        that the prediction only depends on the forget count of x_j.
        """
        n_online, n_upstream = labels.shape
        counts = labels.sum(axis=0).astype(np.int64)
        total_pos = int(counts.sum())
        best_gamma, best_f1 = 1, -1.0
        order = np.argsort(-counts)
        sorted_counts = counts[order]
        # running sums as gamma decreases
        tp = 0
        predicted_pos = 0
        idx = 0
        for gamma in range(n_online + 1, 0, -1):
            while idx < n_upstream and sorted_counts[idx] >= gamma:
                tp += int(sorted_counts[idx])
                predicted_pos += n_online
                idx += 1
            fn = total_pos - tp
            fp = predicted_pos - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if idx > 0 and f1 >= best_f1:
                best_gamma, best_f1 = gamma, f1
        return best_gamma

    def scores(self, artifact) -> np.ndarray:
        if self.counts is None:
            raise RuntimeError("call fit() first")
        return self.counts.astype(np.float64)

    def predict(self, artifact) -> np.ndarray:
        return (self.scores(artifact) >= float(self.gamma or 1)).astype(int)

    def to_dict(self) -> Dict:
        return {"name": self.name, "gamma": float(self.gamma or 1), "n_upstream": int(len(self.counts or []))}
