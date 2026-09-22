"""Running mean/std, used for observation normalisation."""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """Welford / parallel-variance running statistics."""

    def __init__(self, shape=(), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == self.mean.ndim:
            x = x[None]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

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
        return np.sqrt(self.var)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class Normalizer:
    """Normalise a tensor with (frozen) running statistics."""

    def __init__(self, shape, device: str = "cpu", clip: float = 5.0) -> None:
        self.stats = RunningMeanStd(shape)
        self.device = device
        self.clip = clip

    def update(self, x) -> None:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.stats.update(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.stats.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.stats.std, dtype=x.dtype, device=x.device)
        out = (x - mean) / (std + 1e-8)
        if self.clip is not None:
            out = torch.clamp(out, -self.clip, self.clip)
        return out
