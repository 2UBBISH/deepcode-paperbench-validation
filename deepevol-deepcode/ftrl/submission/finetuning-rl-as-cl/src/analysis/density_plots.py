"""NetHack level-visitation density analysis (Figure 5).

This module reproduces the qualitative and quantitative claims attached to
Figure 5 of Wolczyk et al. (2024), *Fine-tuning Reinforcement Learning Models is
Secretly a Forgetting Mitigation Problem*:

    "the agent fine-tuned without any retention mechanism spends most of its
    time on the shallow dungeon levels it already knew from pre-training,
    while methods that retain the pre-trained capabilities keep exploring and
    reach deeper levels".

In other words, the *state coverage* of the fine-tuned policy shrinks towards
the pre-training distribution, and this collapse of the visitation density is
the visible symptom of Forgetting of Pre-trained Capabilities (FPC).

What is implemented here
------------------------
``LevelVisitTracker``
    Online accumulator placed inside the NetHack fine-tuning loop.  At every
    logged evaluation it receives the ``dlvl`` (dungeon level) attained by each
    episode and maintains

      * ``counts[level]``           -- how many episodes finished/were observed at
                                       that level (a discrete visitation density),
      * ``steps[level]``            -- how many environment steps were spent on
                                       that level (the continuous variant used by
                                       Figure 5's density plots),
      * per-checkpoint snapshots    -- so that the *evolution* of the density over
                                       training can be plotted as a heatmap
                                       (level vs. environment steps, as in the
                                       paper's per-method panels).

``level_density`` / ``density_matrix`` / ``density_from_records``
    Pure-numeric helpers turning visit bookkeeping into normalised densities and
    into a ``(method, level)`` matrix ready for plotting/JSON export.

``LevelDensityAnalyzer``
    Offline counterpart: loads ``density.json`` / ``summary.json`` files written
    by the NetHack trainer (or by this module's ``save``), aggregates them across
    seeds with 90% confidence intervals and produces the Figures 5 panels.

Plotting
--------
``plot_level_density``            -- per-method bar/step density over dungeon levels.
``plot_density_heatmap``          -- level vs. training-step heatmap (single run).
``plot_density_grid``             -- the Figure 5 grid: one heatmap per method,
                                     sharing the same level axis and colour scale.
``plot_mean_depth``               -- mean dungeon depth over training steps with
                                     90% CI, one curve per retention method.
``plot_density_comparison``       -- side-by-side density curves per method.

Everything heavy (NumPy, matplotlib) is imported lazily/optionally so that the
bookkeeping and JSON logic can be exercised in minimal environments (mirroring
``src/analysis/plotting.py`` and ``src/analysis/cka.py``).

CLI
---
``python -m src.analysis.density_plots --results-dir <dir> --output-dir <dir>``
rebuilds Figure 5 from a directory of per-method/per-seed JSON results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - exercised implicitly by the environment
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # pragma: no cover
    from src.analysis.plotting import (  # type: ignore
        DEFAULT_CONFIDENCE,
        figure_dpi,
        method_style,
        set_plot_style,
        z_for as _plotting_z_for,
    )
except Exception:  # pragma: no cover
    DEFAULT_CONFIDENCE = 0.90
    method_style = None  # type: ignore
    set_plot_style = None  # type: ignore
    _plotting_z_for = None  # type: ignore
    figure_dpi = None  # type: ignore


__all__ = [
    "LevelVisitTracker",
    "LevelDensityAnalyzer",
    "LevelDensity",
    "DensityRecord",
    "level_density",
    "normalize_density",
    "density_matrix",
    "density_from_records",
    "mean_depth_from_counts",
    "expected_level",
    "active_levels",
    "aggregate_density",
    "aggregate_mean_depth",
    "summarize",
    "z_for",
    "plot_level_density",
    "plot_density_heatmap",
    "plot_density_grid",
    "plot_mean_depth",
    "plot_density_comparison",
    "load_density_records",
    "collect_densities",
    "main",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_CONFIDENCE = DEFAULT_CONFIDENCE if isinstance(DEFAULT_CONFIDENCE, float) else 0.90
DEFAULT_MAX_LEVEL = 50          # NetHack dungeon goes deeper, but densities are ~0 past 30
DEFAULT_LEVELS = tuple(range(1, DEFAULT_MAX_LEVEL + 1))
DEPTH_EVERY = 25_000_000        # paper logs per-level metrics every 25M env steps
DENSITY_KEYS = ("counts", "steps", "episodes", "levels")
METHOD_LABELS = {
    "scratch": "from scratch",
    "none": "vanilla fine-tuning",
    "vanilla": "vanilla fine-tuning",
    "ewc": "+ EWC",
    "bc": "+ BC",
    "ks": "+ KS",
    "kickstarting": "+ KS",
    "em": "+ EM",
}
METHOD_ORDER = ("scratch", "none", "ewc", "bc", "ks", "em")


# --------------------------------------------------------------------------- #
# Small numeric helpers (work with or without NumPy)
# --------------------------------------------------------------------------- #


def _is_numpy(value: Any) -> bool:
    return _np is not None and isinstance(value, _np.ndarray)


def _to_list(values: Any) -> List[float]:
    """Best-effort conversion of a sequence/array to a flat list of floats."""
    if values is None:
        return []
    if _is_numpy(values):
        return [float(v) for v in values.reshape(-1)]
    if isinstance(values, Mapping):
        # dense mapping {level: value} -> dense list (sorted by level)
        return [float(values[k]) for k in sorted(values, key=_as_level_key)]
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _as_level_key(key: Any) -> float:
    try:
        return float(key)
    except (TypeError, ValueError):
        return float("inf")


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    values = list(values)
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return float(math.sqrt(max(var, 0.0)))


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile (0.90 -> 1.6449) with a table + approximation."""
    if _plotting_z_for is not None:
        try:
            return float(_plotting_z_for(confidence))
        except Exception:
            pass
    table = {0.50: 0.6745, 0.68: 0.9945, 0.80: 1.2816, 0.90: 1.6449,
             0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758, 0.999: 3.2905}
    for key in sorted(table):
        if abs(confidence - key) < 1e-9:
            return table[key]
    p = (1.0 + float(confidence)) / 2.0
    # Acklam's inverse normal CDF approximation.
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def summarize(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean and normal-approximation confidence interval over seeds."""
    values = [float(v) for v in values if v is not None and not _is_nan(v)]
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "half_width": float("nan"), "n": 0, "lo": float("nan"), "hi": float("nan")}
    m = _mean(values)
    if n < 2:
        return {"mean": m, "std": 0.0, "half_width": 0.0, "n": 1, "lo": m, "hi": m}
    sd = _std(values)
    hw = z_for(confidence) * sd / math.sqrt(n)
    return {"mean": m, "std": sd, "half_width": hw, "n": n, "lo": m - hw, "hi": m + hw}


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------- #
# Density primitives
# --------------------------------------------------------------------------- #


def normalize_density(values: Sequence[float], total: Optional[float] = None) -> List[float]:
    """Normalise a non-negative vector into a probability density.

    ``total`` may be supplied explicitly (e.g. when the caller holds the number
    of episodes the counts came from).  When every entry is zero the result is a
    vector of zeros rather than ``nan``.
    """
    vals = [max(float(v), 0.0) for v in _to_list(values)]
    if not vals:
        return []
    denom = float(total) if total is not None else float(sum(vals))
    if denom <= 0.0:
        return [0.0 for _ in vals]
    return [v / denom for v in vals]


def level_density(
    counts: Union[Mapping[int, float], Sequence[float]],
    levels: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Convert level visit counts into a dense, optionally normalised density.

    Returns a dict with ``levels``, ``counts``, ``density``, ``total`` and
    ``mean_depth`` (expected dungeon level under the density).  Levels missing
    from ``counts`` are filled with zero, matching Figure 5's shared axis.
    """
    if isinstance(counts, Mapping):
        levels_list = [int(lv) for lv in (levels if levels is not None else sorted(counts, key=_as_level_key))]
        dense = [float(counts.get(lv, counts.get(str(lv), 0.0))) for lv in levels_list]
    else:
        dense = _to_list(counts)
        if levels is not None:
            levels_list = [int(lv) for lv in levels]
        else:
            levels_list = list(range(1, len(dense) + 1))
    total = float(sum(dense))
    density = normalize_density(dense, total) if normalize else list(dense)
    mean_depth = mean_depth_from_counts(dense, levels_list)
    return {
        "levels": levels_list,
        "counts": dense,
        "density": density,
        "total": total,
        "mean_depth": mean_depth,
    }


def mean_depth_from_counts(
    counts: Sequence[float], levels: Optional[Sequence[int]] = None
) -> float:
    """Expected dungeon level ``E[dlvl]`` under the (unnormalised) density."""
    vals = _to_list(counts)
    if not vals:
        return float("nan")
    lv = [int(l) for l in levels] if levels is not None else list(range(1, len(vals) + 1))
    if len(lv) != len(vals):
        lv = list(range(1, len(vals) + 1))
    total = float(sum(vals))
    if total <= 0.0:
        return float("nan")
    return float(sum(v * l for v, l in zip(vals, lv)) / total)


def expected_level(counts: Sequence[float], levels: Optional[Sequence[int]] = None) -> float:
    """Alias of :func:`mean_depth_from_counts` (readability helper)."""
    return mean_depth_from_counts(counts, levels)


def active_levels(density: Sequence[float], levels: Optional[Sequence[int]] = None,
                  threshold: float = 0.01) -> List[int]:
    """Levels carrying at least ``threshold`` of the total visitation mass."""
    vals = _to_list(density)
    lv = [int(l) for l in levels] if levels is not None else list(range(1, len(vals) + 1))
    return [l for l, v in zip(lv, vals) if v >= threshold]


def density_matrix(
    densities: Mapping[str, Union[Mapping[int, float], Sequence[float]]],
    levels: Optional[Sequence[int]] = None,
    normalize: bool = True,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a ``(method, level)`` matrix from per-method level counts."""
    keys = list(order) if order is not None else list(densities.keys())
    keys = [k for k in keys if k in densities]
    for k in densities:
        if k not in keys:
            keys.append(k)
    all_levels: List[int] = []
    if levels is not None:
        all_levels = [int(l) for l in levels]
    else:
        for k in keys:
            cnt = densities[k]
            if isinstance(cnt, Mapping):
                all_levels.extend(int(l) for l in cnt.keys())
            else:
                all_levels.extend(range(1, len(_to_list(cnt)) + 1))
        all_levels = sorted(set(all_levels))
    matrix: List[List[float]] = []
    rows: Dict[str, Dict[str, Any]] = OrderedDict()
    for k in keys:
        info = level_density(densities[k], levels=all_levels, normalize=normalize)
        matrix.append(info["density"] if normalize else info["counts"])
        rows[k] = info
    return {"methods": keys, "levels": all_levels, "matrix": matrix, "rows": rows}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class DensityRecord:
    """One (method, seed, step) level-visitation observation."""

    method: str = "none"
    seed: int = 0
    step: int = 0
    levels: List[int] = field(default_factory=list)
    density: List[float] = field(default_factory=list)
    counts: List[float] = field(default_factory=list)
    mean_depth: float = float("nan")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "step": int(self.step),
            "levels": list(self.levels),
            "density": [float(v) for v in self.density],
            "counts": [float(v) for v in self.counts],
            "mean_depth": float(self.mean_depth),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DensityRecord":
        return cls(
            method=str(payload.get("method", "none")),
            seed=int(payload.get("seed", 0)),
            step=int(payload.get("step", 0)),
            levels=[int(l) for l in payload.get("levels", [])],
            density=_to_list(payload.get("density", [])),
            counts=_to_list(payload.get("counts", [])),
            mean_depth=float(payload.get("mean_depth", float("nan"))),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class LevelDensity:
    """Aggregated density of one method (optionally across seeds)."""

    method: str = "none"
    levels: List[int] = field(default_factory=list)
    density: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    half_width: List[float] = field(default_factory=list)
    mean_depth: float = float("nan")
    mean_depth_half_width: float = float("nan")
    seeds: List[int] = field(default_factory=list)
    n: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "levels": list(self.levels),
            "density": [float(v) for v in self.density],
            "std": [float(v) for v in self.std],
            "half_width": [float(v) for v in self.half_width],
            "mean_depth": float(self.mean_depth),
            "mean_depth_half_width": float(self.mean_depth_half_width),
            "seeds": list(self.seeds),
            "n": int(self.n),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LevelDensity":
        return cls(
            method=str(payload.get("method", "none")),
            levels=[int(l) for l in payload.get("levels", [])],
            density=_to_list(payload.get("density", [])),
            std=_to_list(payload.get("std", [])),
            half_width=_to_list(payload.get("half_width", [])),
            mean_depth=float(payload.get("mean_depth", float("nan"))),
            mean_depth_half_width=float(payload.get("mean_depth_half_width", float("nan"))),
            seeds=[int(s) for s in payload.get("seeds", [])],
            n=int(payload.get("n", 0)),
        )


# --------------------------------------------------------------------------- #
# Online tracking
# --------------------------------------------------------------------------- #


class LevelVisitTracker:
    """Online accumulator of dungeon-level visitation for one training run.

    Typical use inside the NetHack fine-tuning loop::

        tracker = LevelVisitTracker(method="bc", seed=0, every=DEPTH_EVERY)
        ...
        for episode in eval_episodes:
            tracker.observe_level(episode["dlvl"])
        if step % DEPTH_EVERY == 0:
            tracker.record(step)

    ``counts`` counts episodes per level, ``steps`` accumulates environment
    steps per level (used when the trajectory's per-level step counts are known,
    which the evaluation harness reports as ``steps``/``time``).
    """

    def __init__(
        self,
        method: str = "none",
        seed: int = 0,
        levels: Optional[Sequence[int]] = None,
        max_level: int = DEFAULT_MAX_LEVEL,
        every: Optional[int] = DEPTH_EVERY,
        name: str = "density",
    ) -> None:
        self.method = str(method)
        self.seed = int(seed)
        self.max_level = int(max_level)
        self.levels = [int(l) for l in levels] if levels is not None else list(range(1, max_level + 1))
        self.every = int(every) if every else None
        self.name = name

        self.counts: Dict[int, float] = {lv: 0.0 for lv in self.levels}
        self.steps: Dict[int, float] = {lv: 0.0 for lv in self.levels}
        self.episodes: int = 0
        self.total_steps: int = 0
        self.records: List[DensityRecord] = []
        self._episode_levels: List[int] = []

    # -- ingestion -------------------------------------------------------- #

    def observe_level(self, level: Any, steps: float = 1.0, weight: float = 1.0) -> None:
        """Register that an episode visited (or finished at) dungeon ``level``."""
        lv = int(level) if level is not None else 1
        lv = max(1, min(lv, self.max_level))
        self.counts[lv] = self.counts.get(lv, 0.0) + float(weight)
        self.steps[lv] = self.steps.get(lv, 0.0) + float(steps) * float(weight)
        self.episodes += 1
        self.total_steps += int(max(steps, 0.0))
        self._episode_levels.append(lv)

    # Alias used by generic trainers/loggers.
    def add(self, level: Any, steps: float = 1.0, weight: float = 1.0) -> None:
        self.observe_level(level, steps=steps, weight=weight)

    def observe_episode(
        self,
        dlvl: Any = None,
        steps: float = 0.0,
        levels: Optional[Sequence[int]] = None,
        level_steps: Optional[Sequence[float]] = None,
    ) -> None:
        """Register a whole episode.

        ``levels`` (if given) is the sequence of levels traversed by the episode
        and ``level_steps`` the number of steps spent on each of them; the final
        level is taken as the episode's achievement.
        """
        if levels:
            seq = [int(l) for l in levels]
            st = [float(s) for s in level_steps] if level_steps is not None else [1.0] * len(seq)
            if len(st) != len(seq):
                st = [float(steps) / max(len(seq), 1)] * len(seq)
            for lv, s in zip(seq, st):
                self.observe_level(lv, steps=s)
            return
        self.observe_level(dlvl if dlvl is not None else 1, steps=steps)

    def observe_batch(
        self,
        dlvls: Sequence[Any],
        steps: Optional[Sequence[float]] = None,
        steps_per_level: Optional[Mapping[int, float]] = None,
    ) -> None:
        """Register a batch of episode outcomes returned by one evaluation."""
        if steps_per_level:
            for lv, s in steps_per_level.items():
                self.steps[int(lv)] = self.steps.get(int(lv), 0.0) + float(s)
        steps = list(steps) if steps is not None else [1.0] * len(dlvls)
        for lv, s in zip(dlvls, steps):
            self.observe_level(lv, steps=s)

    # -- snapshots -------------------------------------------------------- #

    def should_record(self, step: int) -> bool:
        if self.every is None or self.every <= 0:
            return False
        return int(step) % int(self.every) == 0

    def snapshot(self, step: int = 0, use_steps: bool = False) -> DensityRecord:
        """Density at the current point in training (without mutating counters)."""
        source = self.steps if use_steps else self.counts
        dense = [float(source.get(lv, 0.0)) for lv in self.levels]
        info = level_density(dense, levels=self.levels, normalize=True)
        return DensityRecord(
            method=self.method,
            seed=self.seed,
            step=int(step),
            levels=list(info["levels"]),
            density=list(info["density"]),
            counts=list(info["counts"]),
            mean_depth=float(info["mean_depth"]),
            metadata={"episodes": self.episodes, "use_steps": bool(use_steps)},
        )

    def record(self, step: int = 0, use_steps: bool = False, reset: bool = True) -> DensityRecord:
        """Append a snapshot to the history (and optionally reset the counters)."""
        rec = self.snapshot(step=step, use_steps=use_steps)
        self.records.append(rec)
        return rec

    def reset_counters(self) -> None:
        self.counts = {lv: 0.0 for lv in self.levels}
        self.steps = {lv: 0.0 for lv in self.levels}
        self.episodes = 0
        self.total_steps = 0
        self._episode_levels = []

    # -- derived quantities ----------------------------------------------- #

    def density(self, use_steps: bool = False) -> List[float]:
        source = self.steps if use_steps else self.counts
        return normalize_density([source.get(lv, 0.0) for lv in self.levels])

    def mean_depth(self, use_steps: bool = False) -> float:
        source = self.steps if use_steps else self.counts
        return mean_depth_from_counts([source.get(lv, 0.0) for lv in self.levels], self.levels)

    def max_level(self) -> int:
        """Deepest level with non-zero visitation."""
        visited = [lv for lv in self.levels if self.counts.get(lv, 0.0) > 0]
        return int(max(visited)) if visited else 0

    def active_levels(self, threshold: float = 0.01) -> List[int]:
        return active_levels(self.density(), self.levels, threshold=threshold)

    def coverage(self, total_levels: Optional[int] = None) -> float:
        """Number of visited levels divided by the maximum reachable depth."""
        denom = float(total_levels if total_levels is not None else self.max_level)
        if denom <= 0:
            return 0.0
        return float(len(self.active_levels(threshold=0.0))) / denom

    # -- persistence ------------------------------------------------------ #

    def to_dict(self, include_records: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "method": self.method,
            "seed": self.seed,
            "levels": list(self.levels),
            "counts": [float(self.counts.get(lv, 0.0)) for lv in self.levels],
            "steps": [float(self.steps.get(lv, 0.0)) for lv in self.levels],
            "episodes": int(self.episodes),
            "total_steps": int(self.total_steps),
            "mean_depth": self.mean_depth(),
            "mean_depth_steps": self.mean_depth(use_steps=True),
            "density": self.density(),
            "records": [r.as_dict() for r in self.records] if include_records else [],
        }
        return payload

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "LevelVisitTracker":
        with open(path) as handle:
            payload = json.load(handle)
        levels = [int(l) for l in payload.get("levels", [])] or None
        tracker = cls(
            method=payload.get("method", "none"),
            seed=int(payload.get("seed", 0)),
            levels=levels,
            name=payload.get("name", "density"),
        )
        for lv, c, s in zip(tracker.levels,
                            _to_list(payload.get("counts", [])),
                            _to_list(payload.get("steps", []))):
            tracker.counts[lv] = float(c)
            tracker.steps[lv] = float(s)
        tracker.episodes = int(payload.get("episodes", 0))
        tracker.total_steps = int(payload.get("total_steps", 0))
        tracker.records = [DensityRecord.from_dict(r) for r in payload.get("records", [])]
        return tracker

    @classmethod
    def from_records(cls, records: Sequence[Any], method: str = "none",
                     seed: int = 0, levels: Optional[Sequence[int]] = None) -> "LevelVisitTracker":
        tracker = cls(method=method, seed=seed, levels=levels)
        for rec in records:
            if isinstance(rec, DensityRecord):
                tracker.records.append(rec)
            else:
                tracker.records.append(DensityRecord.from_dict(rec))
        for rec in tracker.records:
            if rec.levels and not levels:
                tracker.levels = list(rec.levels)
                tracker.counts = {lv: 0.0 for lv in tracker.levels}
                tracker.steps = {lv: 0.0 for lv in tracker.levels}
        return tracker


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #


def aggregate_density(
    records: Sequence[Any],
    levels: Optional[Sequence[int]] = None,
    confidence: float = DEFAULT_CONFIDENCE,
    method: Optional[str] = None,
    step: Optional[Union[int, str]] = None,
    use_steps: bool = False,
) -> LevelDensity:
    """Average densities across seeds with 90% CIs.

    ``records`` may contain :class:`DensityRecord`, :class:`LevelVisitTracker`
    instances, or raw mappings.  ``step`` selects a specific checkpoint
    (``"final"``/``None`` means the deepest available step per seed); when the
    records carry no step information all of them are averaged.
    """
    parsed = _coerce_records(records, use_steps=use_steps)
    if method is not None:
        parsed = [r for r in parsed if r.method == method]
    if not parsed:
        return LevelDensity(method=method or "none")

    if step is None or step == "final":
        # one density per seed: the last recorded step (deepest training point)
        by_seed: Dict[int, DensityRecord] = {}
        for rec in parsed:
            prev = by_seed.get(rec.seed)
            if prev is None or rec.step >= prev.step:
                by_seed[rec.seed] = rec
        selected = [by_seed[s] for s in sorted(by_seed)]
    else:
        step = int(step)
        selected = [r for r in parsed if int(r.step) == step]
        if not selected:
            # ``records`` may be simple end-of-training density entries
            selected = parsed if all(r.step == parsed[0].step for r in parsed) else []

    if levels is None:
        levels = list(selected[0].levels) if selected and selected[0].levels else list(DEFAULT_LEVELS)
    levels = [int(l) for l in levels]

    per_seed: List[List[float]] = []
    depths: List[float] = []
    seeds: List[int] = []
    for rec in selected:
        if rec.density and len(rec.density) == len(rec.levels):
            lookup = {int(l): float(d) for l, d in zip(rec.levels, rec.density)}
        else:
            lookup = {int(l): float(c) for l, c in zip(rec.levels, rec.counts)}
        vec = [lookup.get(lv, 0.0) for lv in levels]
        vec = normalize_density(vec)
        per_seed.append(vec)
        depths.append(mean_depth_from_counts(vec, levels))
        seeds.append(rec.seed)

    if not per_seed:
        return LevelDensity(method=method or "none", levels=levels)

    mean_vec, hw_vec, std_vec = _vector_stats(per_seed, confidence)
    depth_stats = summarize(depths, confidence)
    return LevelDensity(
        method=parsed[0].method if method is None else method,
        levels=levels,
        density=mean_vec,
        std=std_vec,
        half_width=hw_vec,
        mean_depth=float(depth_stats["mean"]),
        mean_depth_half_width=float(depth_stats["half_width"]),
        seeds=seeds,
        n=len(per_seed),
    )


def _vector_stats(vectors: Sequence[Sequence[float]], confidence: float
                  ) -> Tuple[List[float], List[float], List[float]]:
    if not vectors:
        return [], [], []
    width = min(len(v) for v in vectors)
    vectors = [list(v)[:width] for v in vectors]
    n = len(vectors)
    mean = [sum(v[i] for v in vectors) / n for i in range(width)]
    if n < 2:
        return mean, [0.0] * width, [0.0] * width
    std = []
    for i in range(width):
        var = sum((v[i] - mean[i]) ** 2 for v in vectors) / (n - 1)
        std.append(math.sqrt(max(var, 0.0)))
    z = z_for(confidence)
    half = [z * s / math.sqrt(n) for s in std]
    return mean, half, std


def _coerce_records(records: Sequence[Any], use_steps: bool = False) -> List[DensityRecord]:
    out: List[DensityRecord] = []
    for rec in records:
        if isinstance(rec, DensityRecord):
            out.append(rec)
        elif isinstance(rec, LevelVisitTracker):
            snap = rec.snapshot(step=0, use_steps=use_steps)
            snap.method = rec.method
            snap.seed = rec.seed
            out.append(snap)
            out.extend(rec.records)
        elif isinstance(rec, Mapping):
            if "levels" in rec or "density" in rec or "counts" in rec:
                out.append(DensityRecord.from_dict(rec))
            elif "records" in rec:
                out.extend(DensityRecord.from_dict(r) for r in rec["records"])
            else:
                # dense mapping {level: count}
                info = level_density(rec, normalize=True)
                out.append(DensityRecord(
                    method=str(rec.get("method", "none")),
                    levels=list(info["levels"]),
                    density=list(info["density"]),
                    counts=list(info["counts"]),
                    mean_depth=float(info["mean_depth"]),
                ))
        else:
            dens = _to_list(getattr(rec, "density", []))
            if dens:
                lv = _to_list(getattr(rec, "levels", [])) or list(range(1, len(dens) + 1))
                out.append(DensityRecord(
                    method=str(getattr(rec, "method", "none")),
                    seed=int(getattr(rec, "seed", 0)),
                    step=int(getattr(rec, "step", 0)),
                    levels=[int(l) for l in lv],
                    density=dens,
                    counts=_to_list(getattr(rec, "counts", dens)),
                    mean_depth=float(getattr(rec, "mean_depth", float("nan"))),
                ))
    return out


def aggregate_mean_depth(
    records: Sequence[Any],
    confidence: float = DEFAULT_CONFIDENCE,
    use_steps: bool = False,
) -> Dict[str, Any]:
    """Mean dungeon depth (with CI) per step and per method -> Figure 5 trend."""
    parsed = _coerce_records(records, use_steps=use_steps)
    per_method: Dict[str, Dict[int, List[float]]] = OrderedDict()
    for rec in parsed:
        depth = rec.mean_depth
        if _is_nan(depth):
            depth = mean_depth_from_counts(rec.counts or rec.density, rec.levels or None)
        per_method.setdefault(rec.method, OrderedDict()).setdefault(int(rec.step), []).append(float(depth))
    result: Dict[str, Any] = OrderedDict()
    for method, by_step in per_method.items():
        steps = sorted(by_step)
        stats = [summarize(by_step[s], confidence) for s in steps]
        result[method] = {
            "steps": steps,
            "mean": [s["mean"] for s in stats],
            "half_width": [s["half_width"] for s in stats],
            "std": [s["std"] for s in stats],
            "n": [s["n"] for s in stats],
            "final": stats[-1] if stats else {},
        }
    return result


# --------------------------------------------------------------------------- #
# Offline analyzer
# --------------------------------------------------------------------------- #


class LevelDensityAnalyzer:
    """Aggregate and plot level-visitation densities across methods and seeds."""

    def __init__(
        self,
        results_dir: Optional[str] = None,
        methods: Sequence[str] = METHOD_ORDER,
        confidence: float = DEFAULT_CONFIDENCE,
        max_level: int = DEFAULT_MAX_LEVEL,
        every: Optional[int] = DEPTH_EVERY,
        name: str = "density",
    ) -> None:
        self.results_dir = results_dir
        self.methods = list(methods)
        self.confidence = float(confidence)
        self.max_level = int(max_level)
        self.every = every
        self.name = name
        self.records: List[DensityRecord] = []
        self.aggregates: Dict[str, LevelDensity] = OrderedDict()
        self.depth_curves: Dict[str, Any] = OrderedDict()

    # -- loading ---------------------------------------------------------- #

    def add_records(self, records: Sequence[Any]) -> None:
        self.records.extend(_coerce_records(records))

    def add_tracker(self, tracker: LevelVisitTracker) -> None:
        self.records.extend(_coerce_records([tracker]))

    def load(self, results_dir: Optional[str] = None,
             methods: Optional[Sequence[str]] = None) -> int:
        """Discover ``density.json`` (or density blocks in ``summary.json``)."""
        results_dir = results_dir or self.results_dir
        if not results_dir:
            raise ValueError("LevelDensityAnalyzer.load requires a results_dir")
        self.results_dir = results_dir
        methods = list(methods) if methods is not None else self.methods
        loaded = 0
        for payload in load_density_records(results_dir, methods=methods):
            self.records.extend(_coerce_records([payload]))
            loaded += 1
        return loaded

    # -- aggregation ------------------------------------------------------ #

    def aggregate(self, step: Optional[Union[int, str]] = None,
                  use_steps: bool = False,
                  methods: Optional[Sequence[str]] = None) -> Dict[str, LevelDensity]:
        methods = list(methods) if methods is not None else self.methods
        present = {r.method for r in self.records}
        self.aggregates = OrderedDict()
        for method in [m for m in methods if m in present] + \
                      sorted(m for m in present if m not in methods):
            agg = aggregate_density(self.records, levels=list(range(1, self.max_level + 1)),
                                   confidence=self.confidence, method=method,
                                   step=step, use_steps=use_steps)
            self.aggregates[method] = agg
        self.depth_curves = aggregate_mean_depth(self.records, self.confidence, use_steps=use_steps)
        return self.aggregates

    def table(self, methods: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Numeric Figure 5 summary: mean depth, coverage, density per level."""
        methods = list(methods) if methods is not None else list(self.aggregates)
        out: Dict[str, Any] = OrderedDict()
        for method in methods:
            agg = self.aggregates.get(method)
            if agg is None:
                continue
            out[method] = {
                "mean_depth": agg.mean_depth,
                "mean_depth_half_width": agg.mean_depth_half_width,
                "n": agg.n,
                "levels": list(agg.levels),
                "density": list(agg.density),
                "half_width": list(agg.half_width),
                "coverage": len(active_levels(agg.density, agg.levels, 0.01)),
                "max_level": max(active_levels(agg.density, agg.levels, 0.0), default=0),
            }
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "methods": list(self.aggregates),
            "table": self.table(),
            "mean_depth": {m: agg.mean_depth for m, agg in self.aggregates.items()},
            "mean_depth_half_width": {m: agg.mean_depth_half_width for m, agg in self.aggregates.items()},
            "n_seeds": {m: agg.n for m, agg in self.aggregates.items()},
            "confidence": self.confidence,
        }

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "densities": {m: agg.as_dict() for m, agg in self.aggregates.items()},
            "depth_curves": self.depth_curves,
            "records": [r.as_dict() for r in self.records],
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        return path


# --------------------------------------------------------------------------- #
# Loading from result directories
# --------------------------------------------------------------------------- #


def _read_json(path: str) -> Any:
    with open(path) as handle:
        return json.load(handle)


def load_density_records(results_dir: str, methods: Sequence[str] = METHOD_ORDER) -> List[Dict[str, Any]]:
    """Collect level-density payloads produced by the NetHack trainer.

    Recognised layouts (all optional)::

        <results_dir>/<method>/seed_<s>/density.json     (LevelVisitTracker dump)
        <results_dir>/<method>/seed_<s>/summary.json     (key "level_density"/"levels")
        <results_dir>/<method>/density.json
        <results_dir>/density_<method>_seed<s>.json

    Each returned payload is a mapping with (at least) ``method``, ``seed``,
    ``levels`` and either ``density`` or ``counts`` (plus optional ``records``
    for the training-time evolution).
    """
    payloads: List[Dict[str, Any]] = []
    if not results_dir or not os.path.isdir(results_dir):
        return payloads

    def _absorb(payload: Any, method: str, seed: int) -> None:
        if not isinstance(payload, Mapping):
            return
        base = dict(payload)
        base.setdefault("method", method)
        base.setdefault("seed", seed)
        if "level_density" in base and "density" not in base:
            base["density"] = base["level_density"]
        if "levels" not in base and "level_counts" in base:
            base["counts"] = base["level_counts"]
            base["levels"] = [int(k) for k in sorted(base["level_counts"], key=_as_level_key)]
        if "density" in base or "counts" in base or "records" in base:
            if "records" in base and base["records"]:
                for rec in base["records"]:
                    rec.setdefault("method", base.get("method", method))
                    rec.setdefault("seed", int(base.get("seed", seed)))
                    payloads.append(rec)
            else:
                if "levels" not in base and "density" in base:
                    base["levels"] = list(range(1, len(_to_list(base["density"])) + 1))
                payloads.append(base)

    for method in methods:
        method_dir = os.path.join(results_dir, method)
        if os.path.isdir(method_dir):
            top = os.path.join(method_dir, "density.json")
            if os.path.exists(top):
                _absorb(_read_json(top), method, 0)
            for entry in sorted(os.listdir(method_dir)):
                seed_dir = os.path.join(method_dir, entry)
                if not os.path.isdir(seed_dir):
                    continue
                seed = int(entry.split("_")[-1]) if entry.startswith("seed") else 0
                for fname in ("density.json", "level_density.json"):
                    fpath = os.path.join(seed_dir, fname)
                    if os.path.exists(fpath):
                        _absorb(_read_json(fpath), method, seed)
                summary = os.path.join(seed_dir, "summary.json")
                if os.path.exists(summary):
                    _absorb(_read_json(summary), method, seed)
        for entry in sorted(os.listdir(results_dir)):
            if entry.startswith(f"density_{method}"):
                _absorb(_read_json(os.path.join(results_dir, entry)), method, 0)
    payloads.sort(key=lambda p: (str(p.get("method")), int(p.get("seed", 0)), int(p.get("step", 0))))
    return payloads


def collect_densities(results_dir: str, methods: Sequence[str] = METHOD_ORDER,
                      confidence: float = DEFAULT_CONFIDENCE,
                      max_level: int = DEFAULT_MAX_LEVEL,
                      step: Optional[Union[int, str]] = None,
                      use_steps: bool = False) -> Dict[str, Any]:
    """One-shot helper: load a results directory and return the Figure 5 numbers."""
    analyzer = LevelDensityAnalyzer(results_dir=results_dir, methods=methods,
                                    confidence=confidence, max_level=max_level)
    analyzer.load()
    analyzer.aggregate(step=step, use_steps=use_steps)
    return {
        "summary": analyzer.summary(),
        "densities": {m: agg.as_dict() for m, agg in analyzer.aggregates.items()},
        "depth_curves": analyzer.depth_curves,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _import_pyplot():
    try:  # pragma: no cover
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:  # pragma: no cover
        return None


def _save_figure(fig, path: Optional[str]) -> None:
    if path and fig is not None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        dpi = figure_dpi() if callable(figure_dpi) else 150
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _style_for(method: str, index: int = 0) -> Dict[str, Any]:
    if method_style is not None:
        try:
            return dict(method_style(method, index))
        except Exception:
            pass
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    return {"color": palette[index % len(palette)], "linestyle": "-", "marker": "o"}


def _label_for(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def density_dict(densities: Any) -> Dict[str, Any]:
    """Normalise many accepted density containers into ``{method: LevelDensity}``."""
    out: Dict[str, Any] = OrderedDict()
    if densities is None:
        return out
    if isinstance(densities, LevelDensityAnalyzer):
        return OrderedDict(densities.aggregates)
    if isinstance(densities, (list, tuple)):
        for rec in _coerce_records(densities):
            out.setdefault(rec.method, rec)
        return out
    if isinstance(densities, Mapping):
        for key, value in densities.items():
            if isinstance(value, LevelDensity):
                out[key] = value
            elif isinstance(value, Mapping):
                out[key] = LevelDensity.from_dict(dict(value, method=value.get("method", key)))
            elif isinstance(value, LevelVisitTracker):
                snap = value.snapshot()
                out[key] = LevelDensity(method=key, levels=snap.levels, density=snap.density,
                                        counts=None if False else snap.counts,
                                        mean_depth=snap.mean_depth, n=1)
    return out


def plot_level_density(
    densities: Any,
    path: Optional[str] = None,
    levels: Optional[Sequence[int]] = None,
    kind: str = "bar",
    confidence_band: bool = True,
    title: str = "NetHack level visitation density",
    xlabel: str = "dungeon level (dlvl)",
    ylabel: str = "visitation density",
    log_y: bool = False,
    methods: Optional[Sequence[str]] = None,
    show: bool = False,
    ax: Any = None,
    **kwargs: Any,
):
    """Per-method visitation density over dungeon levels (Figure 5 bottom-left)."""
    plt = _import_pyplot()
    data = density_dict(densities)
    if methods is not None:
        data = OrderedDict((m, data[m]) for m in methods if m in data)
    if plt is None:  # pragma: no cover
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7.0, 4.0)))
    for index, (method, agg) in enumerate(data.items()):
        lv = [int(l) for l in (levels if levels is not None else agg.levels)]
        lookup = {int(l): float(d) for l, d in zip(agg.levels, agg.density)}
        hw_lookup = {int(l): float(d) for l, d in zip(agg.levels, agg.half_width)}
        vals = [lookup.get(l, 0.0) for l in lv]
        style = _style_for(method, index)
        if kind == "bar":
            width = 0.8 / max(len(data), 1)
            offset = (index - (len(data) - 1) / 2.0) * width
            ax.bar([l + offset for l in lv], vals, width=width, alpha=0.75,
                   label=_label_for(method), color=style.get("color"), **kwargs)
        else:
            ax.plot(lv, vals, label=_label_for(method), color=style.get("color"),
                    marker=style.get("marker"), markersize=3, **kwargs)
            if confidence_band and any(agg.half_width):
                lo = [max(v - hw_lookup.get(l, 0.0), 0.0) for l, v in zip(lv, vals)]
                hi = [v + hw_lookup.get(l, 0.0) for l, v in zip(lv, vals)]
                ax.fill_between(lv, lo, hi, color=style.get("color"), alpha=0.20)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save_figure(ax.figure, path)
    if show:  # pragma: no cover
        plt.show()
    return ax.figure


def plot_density_heatmap(
    records: Any,
    path: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "environment steps",
    ylabel: str = "dungeon level (dlvl)",
    cmap: str = "viridis",
    max_level: int = DEFAULT_MAX_LEVEL,
    every: Optional[int] = DEPTH_EVERY,
    show: bool = False,
    ax: Any = None,
    annotate: bool = False,
    **kwargs: Any,
):
    """Level vs. training-step heatmap of the visitation density of one run."""
    plt = _import_pyplot()
    recs = _coerce_records(records if isinstance(records, (list, tuple)) else [records])
    recs = [r for r in recs if r.step is not None]
    if not recs:
        return None
    if plt is None:  # pragma: no cover
        return None
    steps = sorted({int(r.step) for r in recs})
    levels = list(range(1, max_level + 1))
    if _np is not None:
        grid = _np.zeros((len(levels), len(steps)), dtype=float)
    else:  # pragma: no cover
        grid = [[0.0] * len(steps) for _ in range(len(levels))]
    for rec in recs:
        j = steps.index(int(rec.step))
        vec = [0.0] * len(levels)
        if rec.density and len(rec.density) == len(rec.levels):
            for l, d in zip(rec.levels, rec.density):
                if 1 <= int(l) <= len(levels):
                    vec[int(l) - 1] = float(d)
        else:
            for l, c in zip(rec.levels, rec.counts):
                if 1 <= int(l) <= len(levels):
                    vec[int(l) - 1] = float(c)
        vec = normalize_density(vec)
        for i in range(len(levels)):
            if _np is not None:
                grid[i, j] = vec[i]
            else:  # pragma: no cover
                grid[i][j] = vec[i]
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6.0, 4.0)))
    extent = None
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, extent=extent)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{s/1e6:.0f}M" for s in steps], fontsize=7)
    yticks = list(range(0, len(levels), max(len(levels) // 10, 1)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(levels[i]) for i in yticks], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "level visitation density")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("density", fontsize=8)
    if annotate:
        for i in range(len(levels)):
            for j in range(len(steps)):
                value = grid[i, j] if _np is not None else grid[i][j]
                if value > 0.05:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5, color="w")
    _save_figure(ax.figure, path)
    if show:  # pragma: no cover
        plt.show()
    return ax.figure


def plot_density_grid(
    histories: Mapping[str, Any],
    path: Optional[str] = None,
    methods: Optional[Sequence[str]] = None,
    ncols: int = 2,
    title: str = "Level visitation density during fine-tuning",
    max_level: int = DEFAULT_MAX_LEVEL,
    cmap: str = "viridis",
    show: bool = False,
    **kwargs: Any,
):
    """Figure 5 grid: one level/step heatmap per method on a shared colour scale."""
    plt = _import_pyplot()
    if plt is None:  # pragma: no cover
        return None
    keys = list(methods) if methods is not None else list(histories.keys())
    keys = [k for k in keys if k in histories]
    if not keys:
        return None
    ncols = max(1, min(int(ncols), len(keys)))
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=kwargs.pop("figsize", (4.0 * ncols, 3.2 * nrows)),
                             squeeze=False)
    for index, method in enumerate(keys):
        ax = axes[index // ncols][index % ncols]
        plot_density_heatmap(histories[method], ax=ax, max_level=max_level, cmap=cmap,
                             title=_label_for(method), show=False)
    for index in range(len(keys), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_mean_depth(
    curves: Any,
    path: Optional[str] = None,
    confidence_band: bool = True,
    title: str = "Mean dungeon depth reached during fine-tuning",
    xlabel: str = "environment steps",
    ylabel: str = "mean dungeon level (dlvl)",
    log_x: bool = True,
    show: bool = False,
    ax: Any = None,
    **kwargs: Any,
):
    """Mean depth over training steps, one curve per method (Figure 5 top row)."""
    plt = _import_pyplot()
    data: Dict[str, Any] = OrderedDict()
    if isinstance(curves, LevelDensityAnalyzer):
        data = OrderedDict(curves.depth_curves)
    elif isinstance(curves, Mapping):
        data = OrderedDict(curves)
    else:
        data = OrderedDict(aggregate_mean_depth(curves))
    if plt is None:  # pragma: no cover
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7.0, 4.0)))
    for index, (method, curve) in enumerate(data.items()):
        if isinstance(curve, Mapping) and "steps" in curve:
            steps = list(curve["steps"])
            mean = list(curve["mean"])
            hw = list(curve.get("half_width", [0.0] * len(mean)))
        else:
            steps = list(range(len(curve)))
            mean = _to_list(curve)
            hw = [0.0] * len(mean)
        style = _style_for(method, index)
        ax.plot(steps, mean, label=_label_for(method), color=style.get("color"),
                marker=style.get("marker"), markersize=3, **kwargs)
        if confidence_band and any(hw):
            ax.fill_between(steps, [m - h for m, h in zip(mean, hw)],
                            [m + h for m, h in zip(mean, hw)],
                            color=style.get("color"), alpha=0.20)
    if log_x:
        ax.set_xscale("symlog", linthresh=1e5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save_figure(ax.figure, path)
    if show:  # pragma: no cover
        plt.show()
    return ax.figure


def plot_density_comparison(
    densities: Any,
    path: Optional[str] = None,
    levels: Optional[Sequence[int]] = None,
    confidence_band: bool = True,
    title: str = "Visitation density: retention methods vs. vanilla fine-tuning",
    show: bool = False,
    **kwargs: Any,
):
    """Side-by-side density curves for all methods on a single axis."""
    return plot_level_density(densities, path=path, levels=levels, kind="line",
                              confidence_band=confidence_band, title=title,
                              show=show, **kwargs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetHack level-visitation density analysis (paper Figure 5).")
    parser.add_argument("--results-dir", default=None,
                        help="directory with <method>/seed_<s>/{density,summary}.json")
    parser.add_argument("--methods", default=",".join(METHOD_ORDER),
                        help="comma separated retention methods")
    parser.add_argument("--output-dir", default=None, help="where to write figures/JSON")
    parser.add_argument("--step", default="final",
                        help="checkpoint step to aggregate ('final' or an integer)")
    parser.add_argument("--max-level", type=int, default=DEFAULT_MAX_LEVEL)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--use-steps", action="store_true",
                        help="weight the density by environment steps instead of episodes")
    parser.add_argument("--plot", action="store_true", help="render Figure 5 panels")
    parser.add_argument("--show", action="store_true", help="display figures interactively")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    try:
        step: Optional[Union[int, str]] = None if args.step in ("None", "none", "") else (
            "final" if isinstance(args.step, str) and args.step == "final" else int(args.step))
    except (TypeError, ValueError):
        step = "final"

    analyzer = LevelDensityAnalyzer(results_dir=args.results_dir, methods=methods,
                                    confidence=args.confidence, max_level=args.max_level)
    if args.results_dir:
        loaded = analyzer.load()
        print(f"loaded {loaded} density payload(s) from {args.results_dir}")
    analyzer.aggregate(step=step, use_steps=bool(args.use_steps))
    summary = analyzer.summary()
    print(json.dumps(summary, indent=2))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        analyzer.save(os.path.join(args.output_dir, "density_summary.json"))
        if args.plot:
            plot_level_density(analyzer.aggregates, path=os.path.join(args.output_dir, "figure5_density.png"),
                               show=args.show)
            plot_density_grid(
                {m: analyzer.records if False else _records_for_method(analyzer.records, m)
                 for m in analyzer.aggregates},
                path=os.path.join(args.output_dir, "figure5_heatmap.png"), show=args.show)
            plot_mean_depth(analyzer, path=os.path.join(args.output_dir, "figure5_mean_depth.png"),
                            show=args.show)
    return 0


def _records_for_method(records: Sequence[DensityRecord], method: str) -> List[DensityRecord]:
    return [r for r in records if r.method == method]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
