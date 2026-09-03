"""Metric helpers for FRE evaluation and benchmarking.

This module contains small, dependency-light utilities used to turn raw
episode returns, success indicators, and per-seed task scores into the
normalized 0-100 metrics reported in the FRE paper (Table 1, Figures 5/6).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Discounting
# ---------------------------------------------------------------------------
def discounted_cumsum(
    values: Union[Sequence[float], np.ndarray, torch.Tensor],
    discount: float = 1.0,
) -> np.ndarray:
    """Compute discounted cumulative sums, from last timestep backwards.

    Args:
        values: Scalar rewards for a single trajectory.
        discount: Discount factor (commonly 1.0 for undiscounted sums or
            ``gamma`` for RL-style discounted returns).

    Returns:
        Numpy array with the same length as ``values`` where element ``t`` is
        ``sum_{k>=t} discount**(k-t) * values[k]``.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(arr)
    running = 0.0
    for t in range(len(arr) - 1, -1, -1):
        running = arr[t] + discount * running
        out[t] = running
    return out


def compute_discounted_return(
    rewards: Union[Sequence[float], np.ndarray, torch.Tensor],
    discount: float = 1.0,
) -> float:
    """Return the discounted return of a reward sequence."""
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().cpu().numpy()
    return float(discounted_cumsum(rewards, discount=discount)[0]) if len(rewards) else 0.0


# ---------------------------------------------------------------------------
# Basic statistical aggregation
# ---------------------------------------------------------------------------
def mean_std(
    values: Union[Sequence[float], np.ndarray, torch.Tensor],
    axis: Optional[int] = None,
) -> Tuple[float, float]:
    """Return ``(mean, standard deviation)`` of a collection of scalar scores.

    Population standard deviation (``ddof=0``) is used to match common
    benchmark reporting.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr, axis=axis)), float(np.std(arr, axis=axis))


def aggregate_seed_metrics(
    scores_by_seed: Union[Sequence[float], np.ndarray],
) -> Dict[str, float]:
    """Aggregate per-seed scores into mean/std/min/max metrics."""
    mean, std = mean_std(scores_by_seed)
    arr = np.asarray(scores_by_seed, dtype=np.float64)
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "n": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Normalized scores
# ---------------------------------------------------------------------------
def normalize_score_0_100(
    score: float,
    min_score: float = 0.0,
    max_score: Optional[float] = None,
) -> float:
    """Map a raw score to the paper's ``[0, 100]`` reporting scale.

    Args:
        score: Raw score (e.g., mean episode return or success fraction).
        min_score: Value corresponding to 0 on the normalized scale.
        max_score: Value corresponding to 100 on the normalized scale. If
            ``None``, ``score`` is assumed to already be in ``[0, 1]`` and is
            multiplied by 100.

    Returns:
        Normalized score clipped to ``[0, 100]`` unless ``clip`` is disabled.
    """
    if max_score is None:
        normalized = score * 100.0
    else:
        denom = max_score - min_score
        normalized = (score - min_score) / denom * 100.0 if denom != 0 else 0.0
    return float(np.clip(normalized, 0.0, 100.0))


def normalize_episode_returns(
    episode_returns: Union[Sequence[float], np.ndarray],
    max_return: Optional[float] = None,
    min_return: float = 0.0,
) -> np.ndarray:
    """Normalize a collection of episode returns into the 0-100 scale."""
    arr = np.asarray(episode_returns, dtype=np.float64)
    if max_return is None:
        max_return = float(np.max(arr)) if arr.size else 1.0
    return np.array(
        [normalize_score_0_100(float(v), min_score=min_return, max_score=max_return) for v in arr],
        dtype=np.float64,
    )


def success_rate(episode_successes: Union[Sequence[bool], Sequence[float], np.ndarray]) -> float:
    """Return the fraction of successful episodes in ``[0, 1]``."""
    arr = np.asarray(episode_successes, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr > 0.5))


# ---------------------------------------------------------------------------
# Task-level aggregation
# ---------------------------------------------------------------------------
def format_mean_std(mean: float, std: float, precision: int = 1) -> str:
    """Format a mean/std pair as ``"mean ± std"`` with fixed precision."""
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def summarize_task_scores(
    scores: Union[Sequence[float], np.ndarray],
    precision: int = 1,
) -> Dict[str, Union[float, str]]:
    """Summarize repeated scores (e.g., across seeds) for a single task."""
    stats = aggregate_seed_metrics(scores)
    stats["mean_std"] = format_mean_std(stats["mean"], stats["std"], precision=precision)
    return stats


def summarize_evaluation(
    results: Dict[str, Sequence[float]],
    precision: int = 1,
) -> Dict[str, Dict[str, Union[float, str]]]:
    """Summarize per-task score collections.

    Args:
        results: Mapping from task name to a sequence of per-seed scores.

    Returns:
        Mapping from task name to aggregation metrics (mean, std, min, max,
        and a formatted ``mean_std`` string).
    """
    summary: Dict[str, Dict[str, Union[float, str]]] = {}
    for task, scores in results.items():
        summary[task] = summarize_task_scores(scores, precision=precision)
    return summary


def aggregate_all_tasks(
    results: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    """Aggregate per-task mean scores into a single overall mean/std.

    This treats each task's mean as one observation, matching the paper's
    domain-wide averages (``antmaze-all``, ``exorl-all``, ``all``).
    """
    task_means = [float(np.mean(scores)) for scores in results.values() if len(scores) > 0]
    if not task_means:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return aggregate_seed_metrics(task_means)


# ---------------------------------------------------------------------------
# Miscellaneous trajectory helpers
# ---------------------------------------------------------------------------
def compute_episode_length(
    terminals: Union[Sequence[bool], np.ndarray],
    max_steps: Optional[int] = None,
) -> int:
    """Infer episode length from terminal indicators."""
    arr = np.asarray(terminals, dtype=bool)
    if arr.size == 0:
        return 0
    if np.any(arr):
        return int(np.argmax(arr) + 1)
    return int(arr.size) if max_steps is None else int(min(arr.size, max_steps))


def aggregate_rollout_metrics(
    episode_returns: Sequence[float],
    episode_successes: Optional[Sequence[bool]] = None,
    episode_lengths: Optional[Sequence[int]] = None,
    max_return: Optional[float] = None,
) -> Dict[str, float]:
    """Aggregate raw rollout data into a consistent metrics dictionary."""
    returns = np.asarray(episode_returns, dtype=np.float64)
    metrics: Dict[str, float] = {
        "mean_return": float(np.mean(returns)) if returns.size else 0.0,
        "std_return": float(np.std(returns)) if returns.size else 0.0,
        "normalized_score": float(normalize_score_0_100(float(np.mean(returns)) if returns.size else 0.0,
                                                         max_score=max_return)),
    }
    if episode_successes is not None:
        metrics["success_rate"] = success_rate(episode_successes)
    if episode_lengths is not None:
        lengths = np.asarray(episode_lengths, dtype=np.float64)
        metrics["mean_episode_length"] = float(np.mean(lengths)) if lengths.size else 0.0
    return metrics


__all__ = [
    "discounted_cumsum",
    "compute_discounted_return",
    "mean_std",
    "aggregate_seed_metrics",
    "normalize_score_0_100",
    "normalize_episode_returns",
    "success_rate",
    "format_mean_std",
    "summarize_task_scores",
    "summarize_evaluation",
    "aggregate_all_tasks",
    "compute_episode_length",
    "aggregate_rollout_metrics",
]
