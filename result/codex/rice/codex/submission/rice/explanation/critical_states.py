"""Turning mask probabilities into critical states / critical windows."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def importance_scores(mask_net, states: np.ndarray) -> np.ndarray:
    """Importance of every step: ``P(a^m = 0 | s_t)`` (higher = more critical)."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    return np.asarray(mask_net.importance(states), dtype=np.float64).reshape(-1)


def most_critical_state(scores: Sequence[float]) -> int:
    """Index of the single most critical step (Algorithm 2, roll-in step)."""
    scores = np.asarray(scores, dtype=np.float64)
    return int(np.argmax(scores))


def topk_critical_states(scores: Sequence[float], k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    k = int(max(1, min(k, len(scores))))
    return np.argsort(-scores)[:k]


def most_critical_window(
    scores: Sequence[float], window: int, stride: int = 1
) -> Tuple[int, int]:
    """Sliding window of width ``window`` with the highest average importance.

    This is the selection rule of the fidelity score (Section 4.1 / StateMask).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    window = int(max(1, min(window, n)))
    if window >= n:
        return 0, n
    csum = np.concatenate([[0.0], np.cumsum(scores)])
    starts = np.arange(0, n - window + 1, stride)
    averages = (csum[starts + window] - csum[starts]) / window
    best = int(starts[int(np.argmax(averages))])
    return best, best + window


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
