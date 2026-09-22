"""Evaluation metrics for FRE zero-shot offline RL.

This module implements the return-processing pipeline described in
Section 5.2 / Appendix C of the paper:

* each agent is evaluated with a mean over **20 evaluation episodes**;
* each agent is trained with **5 random seeds** and the standard deviation
  across seeds is reported;
* returns are normalized to the ``[0, 100]`` scale used in Table 1;
* for the prior-subsets scaling study (Table 4 / Figure 5) returns are
  additionally normalized so that the best agent on each task set scores ``1.0``.

The helpers here are deliberately dependency-light (numpy only) so that they can
be imported by the training/eval orchestration code without pulling in torch,
gym or dm_control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # episode-level aggregation
    "EpisodeResult",
    "aggregate_episodes",
    "compute_episode_metrics",
    # return normalization
    "normalize_return",
    "normalize_returns",
    "denormalize_returns",
    "NORMALIZED_MIN",
    "NORMALIZED_MAX",
    # seed-level aggregation
    "aggregate_seeds",
    "SeedResult",
    "TaskResult",
    # suite-level aggregation
    "aggregate_tasks",
    "relative_normalize",
    "aggregate_task_sets",
    "compute_scores",
    "build_score_table",
    "format_metric",
    "format_table",
    # defaults
    "NUM_EVAL_EPISODES",
    "NUM_SEEDS",
    "SUCCESS_THRESHOLD",
]

# --------------------------------------------------------------------------------------
# Paper constants
# --------------------------------------------------------------------------------------

#: Number of evaluation episodes (Section 5.2: "a mean over twenty evaluation episodes").
NUM_EVAL_EPISODES: int = 20

#: Number of training seeds (Section 5.2: "each agent is trained using five random seeds").
NUM_SEEDS: int = 5

#: Range of normalized returns used in Table 1.
NORMALIZED_MIN: float = 0.0
NORMALIZED_MAX: float = 100.0

#: Default success threshold used when deciding whether a goal was reached
#: (Appendix C: Euclidean distance threshold 0.1 for ExORL goal tasks).
SUCCESS_THRESHOLD: float = 0.1


# --------------------------------------------------------------------------------------
# Episode-level metrics
# --------------------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode.

    Parameters
    ----------
    return_
        Total (undiscounted) reward accumulated during the episode.
    length
        Number of environment steps taken.
    success
        Optional boolean flag indicating task success (e.g. goal reached).
    success_steps
        Optional index of the first step at which the task was solved.
    info
        Optional free-form dictionary with extra information.
    """

    return_: float
    length: int = 0
    success: bool = False
    success_steps: Optional[int] = None
    info: Dict[str, Any] = field(default_factory=dict)

    # Convenience aliases -------------------------------------------------------------
    @property
    def reward(self) -> float:
        """Alias for :attr:`return_`."""
        return self.return_

    @property
    def score(self) -> float:
        """Alias for :attr:`return_`."""
        return self.return_

    def to_dict(self) -> Dict[str, Any]:
        return {
            "return": float(self.return_),
            "length": int(self.length),
            "success": bool(self.success),
            "success_steps": self.success_steps,
        }


def _as_episode_result(episode: Any) -> EpisodeResult:
    """Coerce a flexible episode representation into :class:`EpisodeResult`."""
    if isinstance(episode, EpisodeResult):
        return episode
    if isinstance(episode, Mapping):
        ret = episode.get("return", episode.get("return_", episode.get("reward", episode.get("score", 0.0))))
        return EpisodeResult(
            return_=float(ret),
            length=int(episode.get("length", episode.get("steps", 0)) or 0),
            success=bool(episode.get("success", episode.get("done", False))),
            success_steps=episode.get("success_steps", episode.get("steps_to_goal")),
            info=episode.get("info", {}) or {},
        )
    if isinstance(episode, (tuple, list)) and len(episode) >= 1:
        length = int(episode[1]) if len(episode) > 1 else 0
        success = bool(episode[2]) if len(episode) > 2 else False
        return EpisodeResult(return_=float(episode[0]), length=length, success=success)
    # Assume a numeric return.
    return EpisodeResult(return_=float(episode))


def aggregate_episodes(
    episodes: Sequence[Any],
    success_threshold: float = SUCCESS_THRESHOLD,
) -> Dict[str, float]:
    """Aggregate a sequence of evaluation episodes.

    Returns a dictionary with ``mean_return``, ``std_return``, ``max_return``,
    ``min_return``, ``mean_length``, ``success_rate`` and ``median_return``.
    """
    results = [_as_episode_result(e) for e in episodes]
    if not results:
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "median_return": 0.0,
            "max_return": 0.0,
            "min_return": 0.0,
            "mean_length": 0.0,
            "success_rate": 0.0,
            "num_episodes": 0,
            "num_success": 0,
        }

    returns = np.asarray([r.return_ for r in results], dtype=np.float64)
    lengths = np.asarray([r.length for r in results], dtype=np.float64)
    successes = np.asarray([1.0 if r.success else 0.0 for r in results], dtype=np.float64)

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=0)),
        "median_return": float(np.median(returns)),
        "max_return": float(returns.max()),
        "min_return": float(returns.min()),
        "mean_length": float(lengths.mean()),
        "success_rate": float(successes.mean()),
        "num_episodes": int(len(results)),
        "num_success": int(successes.sum()),
    }


def compute_episode_metrics(
    episodes: Sequence[Any],
    success_threshold: float = SUCCESS_THRESHOLD,
) -> Dict[str, float]:
    """Alias of :func:`aggregate_episodes` (kept for API symmetry)."""
    return aggregate_episodes(episodes, success_threshold=success_threshold)


# --------------------------------------------------------------------------------------
# Return normalization
# --------------------------------------------------------------------------------------


def normalize_return(
    returns: Any,
    ref_min: float = 0.0,
    ref_max: float = 1.0,
    out_min: float = NORMALIZED_MIN,
    out_max: float = NORMALIZED_MAX,
    clip: bool = False,
    eps: float = 1e-8,
) -> Any:
    """Linearly map raw returns onto the normalized ``[out_min, out_max]`` scale.

    This is the analogue of D4RL's ``normalize_score`` but is applied to the
    *task reward* of the FRE prior/evaluation tasks (whose natural range is
    typically ``[-1, 0]`` for goal-reaching or ``[0, 1]`` for velocity tasks).
    """
    arr = np.asarray(returns, dtype=np.float64)
    scale = (out_max - out_min) / max(float(ref_max) - float(ref_min), eps)
    out = (arr - float(ref_min)) * scale + float(out_min)
    if clip:
        out = np.clip(out, out_min, out_max)
    if np.isscalar(returns) or (isinstance(returns, np.ndarray) and returns.ndim == 0):
        return float(out)
    return out


def normalize_returns(returns: Any, **kwargs: Any) -> Any:
    """Alias of :func:`normalize_return` (plural form for readability)."""
    return normalize_return(returns, **kwargs)


def denormalize_returns(normalized: Any, ref_min: float = 0.0, ref_max: float = 1.0,
                       out_min: float = NORMALIZED_MIN, out_max: float = NORMALIZED_MAX,
                       eps: float = 1e-8) -> Any:
    """Inverse of :func:`normalize_return`."""
    arr = np.asarray(normalized, dtype=np.float64)
    scale = max(float(ref_max) - float(ref_min), eps) / max(float(out_max) - float(out_min), eps)
    out = (arr - float(out_min)) * scale + float(ref_min)
    if np.isscalar(normalized):
        return float(out)
    return out


def d4rl_normalize(score: Any, min_score: float, max_score: float, eps: float = 1e-8) -> Any:
    """D4RL-style normalization to ``[0, 100]``."""
    return normalize_return(score, min_score, max_score, eps=eps)


# --------------------------------------------------------------------------------------
# Seed-level aggregation
# --------------------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Aggregated result of one seed (already averaged over ``NUM_EVAL_EPISODES``)."""

    seed: int
    mean_return: float
    std_return: float = 0.0
    normalized: Optional[float] = None
    success_rate: float = 0.0
    num_episodes: int = NUM_EVAL_EPISODES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "mean_return": float(self.mean_return),
            "std_return": float(self.std_return),
            "normalized": None if self.normalized is None else float(self.normalized),
            "success_rate": float(self.success_rate),
            "num_episodes": int(self.num_episodes),
        }


def _as_seed_result(seed: Any, value: Any) -> SeedResult:
    """Coerce ``(seed, value)`` pairs into :class:`SeedResult`."""
    if isinstance(value, SeedResult):
        return value
    if isinstance(value, Mapping):
        return SeedResult(
            seed=int(seed),
            mean_return=float(value.get("mean_return", value.get("mean", 0.0))),
            std_return=float(value.get("std_return", value.get("std", 0.0))),
            normalized=value.get("normalized"),
            success_rate=float(value.get("success_rate", 0.0)),
            num_episodes=int(value.get("num_episodes", NUM_EVAL_EPISODES)),
        )
    if isinstance(value, (tuple, list)) and len(value) >= 1:
        std = float(value[1]) if len(value) > 1 else 0.0
        return SeedResult(seed=int(seed), mean_return=float(value[0]), std_return=std)
    return SeedResult(seed=int(seed), mean_return=float(value))


def aggregate_seeds(
    per_seed: Any,
    ref_min: Optional[float] = None,
    ref_max: Optional[float] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Aggregate per-seed episode-averaged returns.

    ``per_seed`` can be:

    * a sequence of :class:`SeedResult` / mappings / floats,
    * a mapping ``{seed: value}``,
    * a sequence of sequences of episodes (one list per seed, in which case each
      seed's episodes are first averaged over the 20 evaluation episodes).

    Returns a dictionary with ``mean``, ``std`` (across seeds), ``normalized_mean``,
    ``normalized_std``, ``per_seed`` and ``num_seeds``.  ``std`` is the standard
    deviation across seeds, exactly as reported in Table 1.
    """
    results: List[SeedResult] = []

    if isinstance(per_seed, Mapping):
        for seed, value in per_seed.items():
            results.append(_as_seed_result(seed, value))
    else:
        seq = list(per_seed)
        for idx, value in enumerate(seq):
            if isinstance(value, Mapping):
                seed = value.get("seed", idx)
                results.append(_as_seed_result(seed, value))
            elif isinstance(value, SeedResult):
                results.append(value)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple, Mapping, EpisodeResult)):
                # Sequence of episodes for this seed.
                agg = aggregate_episodes(value)
                results.append(
                    SeedResult(
                        seed=idx,
                        mean_return=agg["mean_return"],
                        std_return=agg["std_return"],
                        success_rate=agg["success_rate"],
                    )
                )
            else:
                results.append(_as_seed_result(idx, value))

    if not results:
        return {
            "mean": 0.0,
            "std": 0.0,
            "normalized_mean": 0.0,
            "normalized_std": 0.0,
            "success_rate": 0.0,
            "per_seed": [],
            "num_seeds": 0,
        }

    means = np.asarray([r.mean_return for r in results], dtype=np.float64)
    success = np.asarray([r.success_rate for r in results], dtype=np.float64)

    mean = float(means.mean())
    std = float(means.std(ddof=0))

    norm_mean = math.nan
    norm_std = math.nan
    if normalize:
        if ref_min is None:
            ref_min = min(0.0, float(means.min()))
        if ref_max is None:
            ref_max = float(means.max())
        if abs(float(ref_max) - float(ref_min)) < 1e-8:
            ref_max = float(ref_min) + 1.0
        normed = normalize_return(means, ref_min, ref_max)
        normed = np.atleast_1d(normed)
        norm_mean = float(normed.mean())
        norm_std = float(normed.std(ddof=0))

    return {
        "mean": mean,
        "std": std,
        "normalized_mean": norm_mean,
        "normalized_std": norm_std,
        "success_rate": float(success.mean()),
        "per_seed": [r.to_dict() for r in results],
        "num_seeds": len(results),
    }


# --------------------------------------------------------------------------------------
# Task / suite level aggregation
# --------------------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Zero-shot result for a single evaluation task."""

    name: str
    mean: float
    std: float = 0.0
    normalized: Optional[float] = None
    task_set: str = "all"
    success_rate: float = 0.0
    num_seeds: int = NUM_SEEDS
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_set": self.task_set,
            "mean": float(self.mean),
            "std": float(self.std),
            "normalized": None if self.normalized is None else float(self.normalized),
            "success_rate": float(self.success_rate),
            "num_seeds": int(self.num_seeds),
        }


def _as_task_result(task_set: str, name: str, value: Any) -> TaskResult:
    if isinstance(value, TaskResult):
        return value
    if isinstance(value, Mapping):
        return TaskResult(
            name=name,
            task_set=str(value.get("task_set", task_set)),
            mean=float(value.get("mean", value.get("mean_return", 0.0))),
            std=float(value.get("std", value.get("std_return", 0.0))),
            normalized=value.get("normalized"),
            success_rate=float(value.get("success_rate", 0.0)),
            num_seeds=int(value.get("num_seeds", NUM_SEEDS)),
        )
    if isinstance(value, (tuple, list)) and len(value) >= 1:
        std = float(value[1]) if len(value) > 1 else 0.0
        return TaskResult(name=name, task_set=task_set, mean=float(value[0]), std=std)
    return TaskResult(name=name, task_set=task_set, mean=float(value))


def aggregate_tasks(
    tasks: Any,
    normalize: bool = True,
    ref_min: Optional[float] = None,
    ref_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate a collection of per-task results into a task-set summary.

    ``tasks`` may be a mapping ``{"task-name": result}`` or an iterable of
    :class:`TaskResult`.  Returns ``mean``/``std`` over tasks plus a ``tasks``
    list with the individual results.
    """
    results: List[TaskResult] = []
    if isinstance(tasks, Mapping):
        for name, value in tasks.items():
            results.append(_as_task_result("all", name, value))
    else:
        for idx, value in enumerate(tasks):
            if isinstance(value, TaskResult):
                results.append(value)
            elif isinstance(value, Mapping):
                name = str(value.get("name", idx))
                results.append(_as_task_result(str(value.get("task_set", "all")), name, value))
            elif isinstance(value, (tuple, list)) and len(value) >= 2 and isinstance(value[0], str):
                results.append(_as_task_result("all", str(value[0]), value[1:]))
            else:
                results.append(_as_task_result("all", str(idx), value))

    if not results:
        return {"mean": 0.0, "std": 0.0, "normalized_mean": 0.0, "tasks": [], "num_tasks": 0}

    means = np.asarray([r.mean for r in results], dtype=np.float64)
    mean = float(means.mean())
    std = float(means.std(ddof=0))

    normed = None
    if normalize:
        lo = float(means.min()) if ref_min is None else float(ref_min)
        hi = float(means.max()) if ref_max is None else float(ref_max)
        if abs(hi - lo) < 1e-8:
            hi = lo + 1.0
        normed = np.atleast_1d(normalize_return(means, lo, hi))

    for i, r in enumerate(results):
        if normed is not None and r.normalized is None:
            r.normalized = float(normed[i])

    return {
        "mean": mean,
        "std": std,
        "normalized_mean": float(np.mean(normed)) if normed is not None else math.nan,
        "tasks": [r.to_dict() for r in results],
        "num_tasks": len(results),
    }


def relative_normalize(
    scores: Mapping[str, Any],
    higher_is_better: bool = True,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Normalize a mapping of methods -> score so the best method scores ``1.0``.

    This reproduces Figure 5 / Table 4 of the paper, where "returns are
    normalized so the best agent on each task set scores 1.0".
    """
    if not scores:
        return {}
    values = {k: float(v) for k, v in scores.items()}
    best = max(values.values()) if higher_is_better else min(values.values())
    if abs(best) < eps:
        # Degenerate (all-zero or near-zero) row: return zeros.
        return {k: 0.0 for k in values}
    if higher_is_better:
        return {k: v / best for k, v in values.items()}
    return {k: best / v if abs(v) > eps else 0.0 for k, v in values.items()}


def aggregate_task_sets(
    task_results: Mapping[str, Any],
    task_to_set: Optional[Mapping[str, str]] = None,
    normalize: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Group per-task results into task sets and average within each set.

    ``task_results`` maps task name -> per-task result (or mean).  ``task_to_set``
    maps task name -> task-set name; when omitted, the ``task_set`` field of the
    result is used, falling back to ``"all"``.
    """
    grouped: Dict[str, List[float]] = {}
    for name, value in task_results.items():
        result = _as_task_result("all", name, value)
        set_name = str(task_to_set.get(name, result.task_set)) if task_to_set else result.task_set
        # Domain-level mapping (e.g. ant-goal-reaching) when available in metadata.
        grouped.setdefault(set_name or "all", []).append(result.mean)

    out: Dict[str, Dict[str, float]] = {}
    for set_name, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        entry = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "num_tasks": len(values)}
        if normalize and len(values) > 1:
            entry["normalized_mean"] = float(np.mean(normalize_return(arr, arr.min(), arr.max())))
        out[set_name] = entry
    return out


def compute_scores(
    per_seed: Any,
    ref_min: Optional[float] = None,
    ref_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: aggregate per-seed returns and report ``mean +/- std``.

    Mirrors the Table 1 reporting format (normalized return 0-100, mean over 20
    episodes, std across 5 seeds).
    """
    agg = aggregate_seeds(per_seed, ref_min=ref_min, ref_max=ref_max, normalize=True)
    return {
        "mean": agg["mean"],
        "std": agg["std"],
        "normalized_mean": agg["normalized_mean"],
        "normalized_std": agg["normalized_std"],
        "num_seeds": agg["num_seeds"],
        "per_seed": agg["per_seed"],
        "display": format_metric(agg["normalized_mean"], agg["normalized_std"]),
    }


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------


def build_score_table(
    results: Mapping[str, Mapping[str, Any]],
    metric: str = "normalized_mean",
    digits: int = 1,
    relative: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Build a printable score table.

    Parameters
    ----------
    results
        Mapping ``{method: {task_set: score-or-dict}}``.  Values may either be
        plain numbers or dictionaries containing ``metric`` / ``"std"``.
    metric
        Which key to read from dictionary values (default ``"normalized_mean"``).
    relative
        When ``True``, normalize each task-set column so the best method scores
        ``1.0`` (Figure 5 / Table 4 convention).
    """
    # Collect raw values per task set.
    columns: Dict[str, Dict[str, float]] = {}
    stds: Dict[str, Dict[str, float]] = {}
    for method, row in results.items():
        for task_set, value in row.items():
            if isinstance(value, Mapping):
                val = float(value.get(metric, value.get("mean", 0.0)))
                std = float(value.get("std", value.get("normalized_std", 0.0)))
            else:
                val = float(value)
                std = 0.0
            columns.setdefault(task_set, {})[method] = val
            stds.setdefault(task_set, {})[method] = std

    if relative:
        for task_set, values in columns.items():
            for method, val in relative_normalize(values).items():
                columns[task_set][method] = val

    # Column means (the "all" aggregate column).
    table: Dict[str, Dict[str, str]] = {}
    for method in results:
        row: Dict[str, str] = {}
        row_values: List[float] = []
        for task_set, values in columns.items():
            if method in values:
                val = values[method]
                row[task_set] = format_metric(val, stds[task_set].get(method, 0.0), digits=digits)
                row_values.append(val)
        row["average"] = format_metric(float(np.mean(row_values)) if row_values else 0.0, 0.0, digits=digits)
        table[method] = row
    return table


def format_metric(mean: float, std: Optional[float] = None, digits: int = 1) -> str:
    """Format ``mean +/- std`` the way Table 1 reports it."""
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def format_table(
    table: Mapping[str, Mapping[str, str]],
    title: Optional[str] = None,
    float_fmt: str = "{:>18}",
) -> str:
    """Pretty-print a score table produced by :func:`build_score_table`."""
    if not table:
        return title + "\n(empty)\n" if title else "(empty)\n"

    task_sets: List[str] = []
    for row in table.values():
        for key in row:
            if key not in task_sets:
                task_sets.append(key)

    header = "{:<24}".format("method") + "".join("{:>20}".format(c) for c in task_sets)
    lines = []
    if title:
        lines.append(title)
    lines.append(header)
    lines.append("-" * len(header))
    for method, row in table.items():
        line = "{:<24}".format(method) + "".join("{:>20}".format(row.get(c, "")) for c in task_sets)
        lines.append(line)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Rollout conversion helpers (used by fre/evaluation/evaluate.py)
# --------------------------------------------------------------------------------------


def rewards_to_returns(
    rewards: Sequence[float],
    discount: float = 1.0,
    normalize: bool = False,
    ref_min: float = 0.0,
    ref_max: float = 1.0,
) -> float:
    """Compute the (optionally discounted) return of a reward sequence."""
    total = 0.0
    gamma = 1.0
    for r in rewards:
        total += gamma * float(r)
        gamma *= discount
    if normalize:
        return normalize_return(total, ref_min, ref_max)
    return total


def first_success_step(
    successes: Iterable[bool],
    offset: int = 0,
) -> Optional[int]:
    """Return the first index at which ``success`` is ``True`` (plus ``offset``)."""
    for i, ok in enumerate(successes):
        if ok:
            return i + offset
    return None


def episode_success(successes: Sequence[bool]) -> bool:
    """Whether the task was solved at any point during the episode."""
    return bool(any(bool(s) for s in successes))
