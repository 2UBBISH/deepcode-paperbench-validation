"""Metric logging, history tracking and seed aggregation utilities for FRE.

This module is infrastructure (``kind: glue`` in the reproduction plan): it carries no
paper obligation of its own, but it must satisfy the interfaces required by the rest of
the code base:

* ``fre.utils.__init__`` re-exports :class:`MetricLogger` and :func:`aggregate_seeds`.
* ``fre.training.strided`` passes a user supplied ``logger`` object and dispatches to the
  first available method among ``log``, ``log_stats``, ``log_metrics`` and ``record``
  (see ``StridedTrainer._log_stats``), so all four must exist on our logger.
* ``fre.eval.zero_shot_eval`` produces per-seed ``EvalReport``/``SuiteResult`` objects
  (plain dicts via ``to_dict()``); the helpers here aggregate those into the paper's
  "mean +- std over five random seeds" rows (Section 5.2, Tables 1/4) and render them in
  the same ``48.8 +- 6`` style used by the paper.

Everything is dependency free (stdlib + numpy only) so that the logging layer can be
imported without pulling in ``torch``/``D4RL``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_SMOOTHING",
    "DEFAULT_DIGITS",
    "MetricLogger",
    "RunningAverage",
    "EpisodeStats",
    "SeedStatistics",
    "aggregate_seeds",
    "aggregate_seed_values",
    "seed_statistics",
    "format_mean_std",
    "format_summary",
    "get_logger",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
#: Default interval (in training steps) between console log lines. Mirrors the "log every
#: 1k steps" behaviour used by the training entry points (Table 3 budgets are in the
#: 150k/850k (AntMaze) and 1M/1M (ExORL/Kitchen) range).
DEFAULT_LOG_INTERVAL = 1000
#: Default exponential-moving-average smoothing for console output.
DEFAULT_SMOOTHING = 0.9
#: Default number of decimal digits for paper style ``mean +- std`` rendering.
DEFAULT_DIGITS = 1
#: Non-paper default: how many (step, value) pairs to keep per key when tracking history.
DEFAULT_MAX_HISTORY = 100_000


# --------------------------------------------------------------------------------------
# Running statistics
# --------------------------------------------------------------------------------------
@dataclass
class RunningAverage:
    """Exponential moving average with an optional windowed mean.

    Used for smooth console output of noisy training losses; the un-smoothed values are
    still available through :class:`MetricLogger` histories.
    """

    smoothing: float = DEFAULT_SMOOTHING
    value: Optional[float] = None
    count: int = 0

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None or not math.isfinite(self.value):
            self.value = x
        else:
            self.value = self.smoothing * self.value + (1.0 - self.smoothing) * x
        self.count += 1
        return self.value

    def reset(self) -> None:
        self.value = None
        self.count = 0

    def get(self) -> float:
        return float("nan") if self.value is None else float(self.value)


@dataclass
class EpisodeStats:
    """Aggregate statistics of a batch of evaluation episodes."""

    returns: List[float] = field(default_factory=list)
    lengths: List[float] = field(default_factory=list)
    successes: List[float] = field(default_factory=list)

    def add(self, episode_return: float, length: Optional[float] = None, success: Optional[float] = None) -> None:
        self.returns.append(float(episode_return))
        if length is not None:
            self.lengths.append(float(length))
        if success is not None:
            self.successes.append(float(success))

    @property
    def num_episodes(self) -> int:
        return len(self.returns)

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else float("nan")

    @property
    def std_return(self) -> float:
        return float(np.std(self.returns)) if self.returns else float("nan")

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else float("nan")

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else float("nan")

    def to_dict(self) -> Dict[str, float]:
        return {
            "return": self.mean_return,
            "return_std": self.std_return,
            "length": self.mean_length,
            "success_rate": self.success_rate,
            "num_episodes": float(self.num_episodes),
        }


@dataclass
class SeedStatistics:
    """Mean/std/min/max of one metric across random seeds (paper: 5 seeds)."""

    mean: float = float("nan")
    std: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    num_seeds: int = 0
    values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "num_seeds": self.num_seeds,
        }

    def format(self, digits: int = DEFAULT_DIGITS) -> str:
        return format_mean_std(self.mean, self.std, digits=digits)


def seed_statistics(values: Sequence[float], ddof: int = 0) -> Tuple[float, float]:
    """Return ``(mean, std)`` of per-seed numbers, ignoring non-finite entries.

    ``ddof=0`` (population std) is the non-paper default chosen to match the plain
    ``np.std`` convention used by most benchmark tables; ``ddof=1`` gives the unbiased
    sample standard deviation.
    """

    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    arr = np.asarray(finite, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=ddof))


# --------------------------------------------------------------------------------------
# Metric logger
# --------------------------------------------------------------------------------------
class MetricLogger:
    """Lightweight step-indexed metric tracker with smoothing and JSON persistence.

    Typical use (matches ``fre.training.strided.StridedTrainer``)::

        logger = MetricLogger(log_interval=1000)
        logger.log(step, loss=loss, kl=kl)          # positional or keyword step
        logger.log_stats({"step": step, "loss": loss})   # dict form
        logger.log_metrics(step=step, loss=loss)
        logger.record(step=step, loss=loss)

    All four entry points accept the same shapes (a mapping and/or keyword arguments,
    with an optional ``step``), so any of them can be used interchangeably.
    """

    def __init__(
        self,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        smoothing: float = DEFAULT_SMOOTHING,
        name: str = "fre",
        verbose: bool = True,
        print_fn: Optional[Callable[[str], None]] = None,
        track_history: bool = True,
        max_history: int = DEFAULT_MAX_HISTORY,
        log_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.log_interval = int(log_interval)
        self.smoothing = float(smoothing)
        self.name = name
        self.verbose = bool(verbose)
        self.print_fn = print_fn if print_fn is not None else print
        self.track_history = bool(track_history)
        self.max_history = int(max_history)
        self.log_dir = log_dir
        self.seed = seed

        self._averages: Dict[str, RunningAverage] = {}
        self._history: Dict[str, List[Tuple[int, float]]] = {}
        self._latest: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._step: int = 0
        self._last_logged_step: int = -1

    # ------------------------------------------------------------------ logging entry
    def log(
        self,
        step_or_metrics: Optional[Union[int, Mapping[str, Any]]] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        step: Optional[int] = None,
        prefix: str = "",
        force: bool = False,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Record one step of metrics and print periodically.

        ``step`` may be passed positionally (first argument), directly as ``step=``, or as
        a ``"step"`` key inside a metrics mapping. Values that are not finite scalars are
        skipped (so tensor scalars, ``nan``/``inf`` and nested structures are harmless).
        """

        merged: Dict[str, Any] = {}
        if isinstance(step_or_metrics, Mapping):
            merged.update(step_or_metrics)
        elif isinstance(step_or_metrics, (int, np.integer)):
            step = int(step_or_metrics)
        if metrics:
            merged.update(metrics)
        merged.update(kwargs)

        # A "step" key inside the mapping is consumed as the step index.
        if step is None and "step" in merged:
            raw_step = merged.pop("step")
            try:
                step = int(raw_step)
            except (TypeError, ValueError):
                merged["step"] = raw_step
        else:
            merged.pop("step", None)

        if step is not None:
            self._step = int(step)

        recorded: Dict[str, float] = {}
        for key, value in merged.items():
            scalar = _as_scalar(value)
            if scalar is None:
                continue
            full_key = f"{prefix}{key}" if prefix else str(key)
            self._record_scalar(full_key, scalar)
            recorded[full_key] = scalar

        if step is None:
            self._step += 1

        if self._should_log(force=force):
            self.flush()

        return recorded

    # Aliases -----------------------------------------------------------------------
    def log_stats(self, step_or_metrics=None, metrics=None, step=None, prefix="", force=False, **kwargs):
        """Alias of :meth:`log` (used by ``StridedTrainer`` dispatch)."""
        return self.log(step_or_metrics, metrics=metrics, step=step, prefix=prefix, force=force, **kwargs)

    def log_metrics(self, step_or_metrics=None, metrics=None, step=None, prefix="", force=False, **kwargs):
        """Alias of :meth:`log` (used by ``StridedTrainer`` dispatch)."""
        return self.log(step_or_metrics, metrics=metrics, step=step, prefix=prefix, force=force, **kwargs)

    def record(self, step_or_metrics=None, metrics=None, step=None, prefix="", force=False, **kwargs):
        """Alias of :meth:`log` (used by ``StridedTrainer`` dispatch)."""
        return self.log(step_or_metrics, metrics=metrics, step=step, prefix=prefix, force=force, **kwargs)

    def update(self, **metrics: Any) -> Dict[str, float]:
        """Record keyword metrics at the current (incremented) step without printing."""
        return self.log(step=None, force=False, **metrics)

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return self.log(*args, **kwargs)

    # ------------------------------------------------------------------ internals
    def _record_scalar(self, key: str, value: float) -> None:
        self._latest[key] = value
        self._counts[key] = self._counts.get(key, 0) + 1
        avg = self._averages.get(key)
        if avg is None:
            avg = RunningAverage(smoothing=self.smoothing)
            self._averages[key] = avg
        avg.update(value)
        if self.track_history:
            hist = self._history.setdefault(key, [])
            hist.append((self._step, value))
            if len(hist) > self.max_history:
                del hist[: len(hist) - self.max_history]

    def _should_log(self, force: bool = False) -> bool:
        if force:
            return True
        if not self.verbose:
            return False
        if self.log_interval <= 0:
            return False
        if self._step == self._last_logged_step:
            return False
        return self._step % self.log_interval == 0

    # ------------------------------------------------------------------ accessors
    def latest(self, key: str, default: float = float("nan")) -> float:
        return float(self._latest.get(key, default))

    def mean(self, key: str, default: float = float("nan")) -> float:
        """Mean of all recorded values for ``key`` (un-smoothed)."""
        hist = self._history.get(key)
        if not hist:
            return float(self._latest.get(key, default))
        return float(np.mean([v for _, v in hist]))

    def smoothed(self, key: str, default: float = float("nan")) -> float:
        avg = self._averages.get(key)
        if avg is None:
            return default
        value = avg.get()
        return default if not math.isfinite(value) else value

    #: Alias kept for symmetry with :meth:`smoothed`.
    ema = smoothed

    def history(self, key: str) -> List[Tuple[int, float]]:
        return list(self._history.get(key, []))

    def values(self, key: str) -> List[float]:
        return [v for _, v in self._history.get(key, [])]

    def steps(self, key: str) -> List[int]:
        return [s for s, _ in self._history.get(key, [])]

    @property
    def keys(self) -> List[str]:
        return sorted(self._latest.keys())

    @property
    def step(self) -> int:
        return self._step

    @property
    def num_records(self) -> Dict[str, int]:
        return dict(self._counts)

    # ------------------------------------------------------------------ summaries
    def summary(self, smoothed: bool = False, prefix: str = "") -> Dict[str, float]:
        """Return ``{key: value}`` for every tracked metric.

        With ``smoothed=True`` the exponential moving averages are returned (nice for
        console tables); otherwise the mean over the whole history is used.
        """
        out: Dict[str, float] = {}
        for key in self.keys:
            value = self.smoothed(key) if smoothed else self.mean(key)
            out[f"{prefix}{key}" if prefix else key] = value
        return out

    def to_dict(self, smoothed: bool = False, prefix: str = "") -> Dict[str, float]:
        return self.summary(smoothed=smoothed, prefix=prefix)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "step": self._step,
            "log_interval": self.log_interval,
            "smoothing": self.smoothing,
            "latest": dict(self._latest),
            "counts": dict(self._counts),
            "history": {k: list(v) for k, v in self._history.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._step = int(state.get("step", 0))
        self._latest = {k: float(v) for k, v in dict(state.get("latest", {})).items()}
        self._counts = {k: int(v) for k, v in dict(state.get("counts", {})).items()}
        self._history = {k: [(int(s), float(v)) for s, v in v_list] for k, v_list in dict(state.get("history", {})).items()}
        self._averages = {}
        for key, value in self._latest.items():
            avg = RunningAverage(smoothing=self.smoothing)
            avg.value = float(value)
            avg.count = self._counts.get(key, 1)
            self._averages[key] = avg

    # ------------------------------------------------------------------ output
    def format_line(self, smoothed: bool = True, digits: int = 4, extra: Optional[str] = None) -> str:
        parts = [f"step {self._step}"]
        metrics = self.summary(smoothed=smoothed)
        for key in sorted(metrics):
            parts.append(f"{key}={_format_number(metrics[key], digits)}")
        if extra:
            parts.append(str(extra))
        header = f"[{self.name}]" if self.name else ""
        return " ".join(p for p in [header] + parts if p)

    def flush(self, smoothed: bool = True, digits: int = 4) -> str:
        """Print (and return) one console line for the current step."""
        line = self.format_line(smoothed=smoothed, digits=digits)
        self._last_logged_step = self._step
        if self.verbose and self.print_fn is not None:
            self.print_fn(line)
        return line

    #: Print an unconditional line (used at the end of a stage).
    def print(self, message: str) -> None:
        if self.verbose and self.print_fn is not None:
            self.print_fn(message)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Optional[str] = None, smoothed: bool = False) -> str:
        """Persist the full logger state (histories included) as JSON."""
        target = path
        if target is None:
            if self.log_dir is None:
                raise ValueError("MetricLogger.save requires a path (or log_dir at construction)")
            os.makedirs(self.log_dir, exist_ok=True)
            target = os.path.join(self.log_dir, f"metrics_{self.name}.json")
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = self.state_dict()
        payload["summary"] = self.summary(smoothed=smoothed)
        with open(target, "w") as handle:
            json.dump(payload, handle, indent=2)
        return target

    def load(self, path: str) -> "MetricLogger":
        with open(path, "r") as handle:
            payload = json.load(handle)
        self.load_state_dict(payload)
        return self

    # ------------------------------------------------------------------ lifecycle
    def reset(self, keep_history: bool = False) -> None:
        self._averages = {}
        self._latest = {}
        self._counts = {}
        if not keep_history:
            self._history = {}

    def close(self) -> None:
        if self.verbose:
            self.flush()


def get_logger(
    name: str = "fre",
    verbose: bool = True,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    log_dir: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> MetricLogger:
    """Factory mirroring the ``make_*`` helpers used elsewhere in the code base."""
    return MetricLogger(
        name=name,
        verbose=verbose,
        log_interval=log_interval,
        log_dir=log_dir,
        seed=seed,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# Seed aggregation (paper protocol: mean +- std over 5 random seeds)
# --------------------------------------------------------------------------------------
def aggregate_seed_values(values: Sequence[float], ddof: int = 0, digits: Optional[int] = None) -> Dict[str, Any]:
    """Aggregate a list of per-seed values into a mean/std/min/max summary dict."""
    stats = SeedStatistics(
        mean=float("nan"),
        std=float("nan"),
        minimum=float("nan"),
        maximum=float("nan"),
        num_seeds=0,
        values=[float(v) for v in values],
    )
    mean, std = seed_statistics(values, ddof=ddof)
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    stats.mean = mean
    stats.std = std
    stats.minimum = float(min(finite)) if finite else float("nan")
    stats.maximum = float(max(finite)) if finite else float("nan")
    stats.num_seeds = len(finite)
    out = stats.to_dict()
    if digits is not None:
        out["formatted"] = stats.format(digits=digits)
    return out


def _collect_seed_rows(results: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, List[Any]]:
    """Normalize per-seed results into ``{key: [value_per_seed]}``.

    Accepts either a list of per-seed metric dicts (e.g. one ``EvalReport.to_dict()`` per
    seed) or an already transposed mapping ``{key: [values...]}``. Nested mappings are
    flattened with ``"_"`` joins, so ``{"task_a": {"score": 48.8}}`` becomes
    ``{"task_a_score": 48.8}``.
    """

    collected: Dict[str, List[Any]] = {}

    def _absorb(mapping: Mapping[str, Any], prefix: str) -> None:
        for key, value in mapping.items():
            full_key = f"{prefix}{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                _absorb(value, f"{full_key}_")
            else:
                collected.setdefault(full_key, []).append(value)

    if isinstance(results, Mapping):
        for key, value in results.items():
            if isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, (str, bytes)):
                collected.setdefault(str(key), []).extend(list(value))
            else:
                collected.setdefault(str(key), []).append(value)
        return collected

    for row in results:
        if isinstance(row, Mapping):
            _absorb(row, "")
        elif hasattr(row, "to_dict"):
            _absorb(row.to_dict(), "")
        else:
            raise TypeError(f"aggregate_seeds expects mappings or objects with to_dict(), got {type(row)!r}")
    return collected


def aggregate_seeds(
    results: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]],
    ddof: int = 0,
    flat: bool = False,
    digits: Optional[int] = None,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Aggregate per-seed results into ``mean +- std`` summaries.

    Parameters
    ----------
    results:
        Either a list of per-seed metric dicts (for example ``SuiteResult.to_dict()`` for
        seeds ``0..4``) or a transposed mapping ``{key: [values across seeds]}``.
    ddof:
        Delta degrees of freedom for the standard deviation (0 = population std, the
        default; 1 = unbiased sample std).
    flat:
        When ``True`` a flat mapping is returned with ``key`` -> mean and
        ``key_std`` -> std (convenient for JSON tables). When ``False`` (default) the
        result maps ``key`` -> ``{"mean", "std", "min", "max", "num_seeds"}``.
    digits:
        If given, each entry additionally carries a paper-style ``"formatted"`` string
        of the form ``"48.8 +- 6.0"``.
    keys:
        Optional subset of keys to aggregate.

    Returns
    -------
    dict
        The aggregated summary described above. Non-numeric entries (strings, nested
        structures that could not be reduced) are skipped.
    """

    collected = _collect_seed_rows(results)
    out: Dict[str, Any] = {}
    for key, values in collected.items():
        if keys is not None and key not in set(keys):
            continue
        numeric: List[float] = []
        ok = True
        for value in values:
            scalar = _as_scalar(value)
            if scalar is None:
                ok = False
                break
            numeric.append(scalar)
        if not ok or not numeric:
            continue
        summary = aggregate_seed_values(numeric, ddof=ddof, digits=digits)
        if flat:
            out[key] = summary["mean"]
            out[f"{key}_std"] = summary["std"]
        else:
            out[key] = summary
    return out


def format_mean_std(mean: float, std: float, digits: int = DEFAULT_DIGITS) -> str:
    """Render ``mean +- std`` in the paper's style (e.g. ``48.8 +- 6.0``)."""
    if mean is None or not math.isfinite(float(mean)):
        return "n/a"
    if std is None or not math.isfinite(float(std)):
        return _format_number(float(mean), digits)
    return f"{_format_number(float(mean), digits)} +- {_format_number(float(std), digits)}"


def format_summary(summary: Mapping[str, Any], digits: int = DEFAULT_DIGITS) -> str:
    """Pretty print an :func:`aggregate_seeds` result as an aligned text table."""
    rows: List[Tuple[str, str]] = []
    for key in sorted(summary):
        entry = summary[key]
        if isinstance(entry, Mapping):
            rows.append((str(key), format_mean_std(entry.get("mean", float("nan")), entry.get("std", float("nan")), digits)))
        else:
            rows.append((str(key), _format_number(entry, digits)))
    if not rows:
        return "(no metrics)"
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"{name.ljust(width)}  {value}" for name, value in rows)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _as_scalar(value: Any) -> Optional[float]:
    """Coerce a metric value into a finite python float, or ``None`` if not possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    # Torch tensors / numpy arrays without importing torch eagerly.
    if hasattr(value, "detach") and hasattr(value, "numel"):
        try:
            tensor = value.detach()
            if tensor.numel() != 1:
                return None
            return _as_scalar(tensor.item())
        except Exception:  # pragma: no cover - defensive, tensor quirks only
            return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return _as_scalar(value.reshape(-1)[0])
    if hasattr(value, "mean") and callable(getattr(value, "mean")):
        try:
            return _as_scalar(value.mean())
        except Exception:  # pragma: no cover
            return None
    try:
        return _as_scalar(float(value))
    except (TypeError, ValueError):
        return None


def _format_number(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    value = float(value)
    if value != 0 and (abs(value) >= 1e4 or abs(value) < 1e-3):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"
