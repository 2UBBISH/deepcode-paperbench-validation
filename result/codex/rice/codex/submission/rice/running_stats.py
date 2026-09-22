"""Running statistics (used by RND normalisation and observation normalisation)."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Incremental mean/variance estimate (Welford / parallel algorithm)."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None, ...]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count: float) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-8))

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = (x - self.mean) / self.std
        if clip is not None:
            out = np.clip(out, -clip, clip)
        return out.astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
