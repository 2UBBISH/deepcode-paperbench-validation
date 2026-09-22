"""Metrics used across the RICE evaluation protocols.

Two groups of metrics live here:

* generic RL statistics (``discounted_return``, ``episode_returns``,
  ``RunningMeanStd``, ``normalize``) reused by the refinement loop and the
  RND intrinsic-reward normalisation;
* the **fidelity score** of Experiment I (paper Sec. 4.1 "Evaluation
  Metrics")::

      fidelity = log(d / d_max) - log(l / L)

  where ``d = |R' - R|`` is the absolute difference between the return of
  the explanation-driven rollout ``R'`` and the reference return ``R``,
  ``d_max`` is the environment's maximum achievable single-episode reward
  and ``l = L x K`` is the length of the randomised window (``K`` in
  ``{10%, 20%, 30%, 40%}``).  Higher is better.

The helper :func:`sliding_window_windows` enumerates the candidate windows
of width ``l`` over a trajectory of ``L`` steps so that the caller can
select the one with the highest *average* step-level importance.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "discounted_return",
    "compute_returns",
    "episode_returns",
    "mean_std",
    "print_mean_std",
    "normalize",
    "RunningMeanStd",
    "fidelity_score",
    "sliding_window_windows",
    "best_window",
    "window_report",
]

# ---------------------------------------------------------------------------
# Generic RL statistics
# ---------------------------------------------------------------------------


def discounted_return(rewards: Sequence[float], gamma: float = 0.99,
                      normalize_by_discount: bool = False) -> float:
    """Discounted return ``sum_t gamma^t r_t``."""
    out = 0.0
    for t, r in enumerate(rewards):
        out += (gamma ** t) * float(r)
    if normalize_by_discount and len(rewards) > 0:
        denom = sum(gamma ** t for t in range(len(rewards)))
        if denom > 0:
            out /= denom
    return float(out)


def compute_returns(rewards: Sequence[float], gamma: float = 0.99,
                    normalize_by_discount: bool = False) -> np.ndarray:
    """Discounted return-to-go for every timestep (used by the PPO refiner)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    out = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    if normalize_by_discount and len(rewards) > 0:
        discounts = gamma ** np.arange(len(rewards))
        out = out / discounts
    return out


def episode_returns(episode_rewards: Iterable[Sequence[float]]) -> List[float]:
    """Undiscounted episode returns (the quantity reported in Table 1)."""
    return [float(np.sum(np.asarray(r, dtype=np.float64))) for r in episode_rewards]


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    """``(mean, std)`` over ``values`` (empty -> ``(nan, nan)``)."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def print_mean_std(values: Iterable[float], name: str = "metric", decimals: int = 2) -> str:
    """Return ``"name: mean +- std"`` (paper tables report mean +- std)."""
    mean, std = mean_std(values)
    text = f"{name}: {mean:.{decimals}f} +- {std:.{decimals}f}"
    print(text)
    return text


def normalize(array, mean, std, eps: float = 1e-8):
    """``(array - mean) / (std + eps)`` (observation normalisation, Walker2d/HalfCheetah)."""
    return (np.asarray(array, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / (
        np.asarray(std, dtype=np.float64) + eps
    )


class RunningMeanStd:
    """Streaming mean/variance (Welford / parallel-variance accumulation).

    Used for two purposes in RICE:

    * observation normalisation for Walker2d and HalfCheetah
      (``rice/envs/normalizer.py``);
    * normalising the RND intrinsic reward, whose exact form is not given in
      the paper, so the plan's default (divide by the running std of the
      prediction error) is implemented here.
    """

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == self.mean.ndim and x.shape != self.mean.shape and x.size == self.mean.size:
            x = x.reshape(self.mean.shape)
        if x.ndim == self.mean.ndim + 1:  # a batch of independent samples
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = x.shape[0]
        else:
            batch_mean = x
            batch_var = np.zeros_like(self.mean)
            batch_count = 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def __call__(self, x, clip: float = 10.0):
        return np.clip((np.asarray(x, dtype=np.float64) - self.mean) /
                       (self.std + 1e-8), -clip, clip)


# ---------------------------------------------------------------------------
# Fidelity (Experiment I, paper Sec. 4.1 "Evaluation Metrics")
# ---------------------------------------------------------------------------


def fidelity_score(d: float, d_max: float, l: int, L: int, eps: float = 1e-12) -> float:
    """Paper's fidelity score ``log(d / d_max) - log(l / L)``.

    ``d = |R' - R|`` (return gap w.r.t. the reference trajectory),
    ``d_max`` the environment's maximum single-episode reward,
    ``l`` the randomised-window length and ``L`` the full trajectory length.
    Higher is better: a smaller return gap and a shorter explained window.
    """
    d = float(max(abs(d), eps))
    d_max = float(max(abs(d_max), eps))
    ratio_d = max(d / d_max, eps)
    ratio_l = max(float(l) / float(max(L, 1)), eps)
    return float(math.log(ratio_d) - math.log(ratio_l))


def window_length(L: int, K) -> int:
    """Window width ``l = L x K``; ``K`` may be a fraction or a percentage."""
    k = float(K)
    if k > 1.0:
        k = k / 100.0
    return max(1, int(round(float(L) * k)))


def sliding_window_windows(L: int, K) -> List[Tuple[int, int]]:
    """All ``(start, end)`` half-open windows of width ``l = L x K`` over ``[0, L)``."""
    l = window_length(L, K)
    if l >= L:
        return [(0, L)]
    return [(start, start + l) for start in range(0, L - l + 1)]


def best_window(importance: Sequence[float], L: Optional[int] = None, K=0.1):
    """Select the window with the highest **average** step-level importance.

    Implements the paper's rule: "sweep a sliding window of width ``l = L x
    K`` and pick the window with highest average importance".  Returns
    ``(start, end, avg_importance)``; ties are broken by the *earlier*
    window so that results are deterministic across seeds.
    """
    importance = np.asarray(importance, dtype=np.float64)
    L = int(L or importance.shape[0])
    l = window_length(L, K)
    if importance.size == 0:
        return 0, min(l, L), 0.0

    best = None
    for start, end in sliding_window_windows(L, K):
        seg = importance[start:min(end, importance.size)]
        if seg.size == 0:
            continue
        avg = float(seg.mean())
        if best is None or avg > best[2] + 1e-12:
            best = (start, min(end, importance.size), avg)
    if best is None:
        return 0, min(l, L), float(importance.mean())
    return best


def window_report(importance: Sequence[float], L: Optional[int] = None, K=0.1) -> dict:
    """Convenience wrapper returning a JSON-serialisable window report."""
    start, end, avg = best_window(importance, L=L, K=K)
    importance = np.asarray(importance, dtype=np.float64) if len(importance) else np.zeros(0)
    return {
        "K": K if isinstance(K, float) else float(K) / 100.0,
        "l": int(end - start),
        "start": int(start),
        "end": int(end),
        "avg_importance": float(avg),
        "max_importance": float(importance.max()) if importance.size else 0.0,
        "mean_importance": float(importance.mean()) if importance.size else 0.0,
    }
