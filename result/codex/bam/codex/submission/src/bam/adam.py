"""A tiny dependency-free Adam optimizer (used by the gradient-based baselines).

The paper's baselines use Adam (Appendix E.1); ``optax`` is not required.
"""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params: np.ndarray, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr = float(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0

    def update(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        mhat = self.m / (1.0 - self.beta1**self.t)
        vhat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * mhat / (np.sqrt(vhat) + self.eps)
