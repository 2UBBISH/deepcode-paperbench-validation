"""Logging utilities for SAPG experiments.

The paper (Sec. 5.2 / Sec. 6) reports metrics as *mean and standard error* over
5 seeds where the shaded band is::

    band(t) = (2 / sqrt(n)) * sum_i ( y(t) - y_i(t) ) ** 2

with ``n`` the number of seeds.  This module implements:

* :class:`MetricLogger` -- an aggregation-friendly logger that stores scalar
  series against the number of environment transitions collected (the x-axis of
  every figure in the paper), and can emit the paper's shaded band.
* :class:`AverageMeter` / :class:`RunningMeanStd` -- small statistics helpers
  used by the rollout collector and the analysis scripts.
* :class:`ExperimentLogger` -- convenience wrapper that optionally tees results
  to TensorBoard (``torch.utils.tensorboard``) or a JSONL file, without making
  either dependency mandatory.
* :func:`make_logger` / :func:`get_logger` -- factories used by the training
  scripts.

All heavy dependencies (torch, tensorboard, numpy) are imported lazily so this
module stays importable in stripped environments (e.g. unit tests on CPU).

Paper references
----------------
* Sec. 5.2 -- "we report the mean and standard error across 5 seeds" with the
  band formula above.
* Sec. 6.1 -- performance vs. number of samples collected (2e10 transitions).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "PAPER_NUM_SEEDS",
    "PAPER_BAND_SCALE",
    "AverageMeter",
    "RunningMeanStd",
    "MetricLogger",
    "Logger",
    "ExperimentLogger",
    "make_logger",
    "get_logger",
    "paper_standard_error",
    "paper_band",
    "aggregate_curves",
    "format_metrics",
    "DEFAULT_LOG_DIR",
]

#: Number of seeds used for every reported result (Sec. 5.2).
PAPER_NUM_SEEDS: int = 5
#: Constant factor in the paper's shaded band formula ``2 / sqrt(n)``.
PAPER_BAND_SCALE: float = 2.0
DEFAULT_LOG_DIR: str = "runs"

_SEPARATOR = "-" * 72


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
class AverageMeter:
    """Tracks the running mean/standard-deviation of a scalar stream.

    Used for rollout diagnostics (episode returns, episode lengths, success
    counts) where values become available asynchronously at episode ends.
    """

    __slots__ = ("name", "count", "sum", "_sumsq", "last", "window", "_values")

    def __init__(self, name: str = "meter", window: Optional[int] = None) -> None:
        self.name = name
        self.window = window
        self.reset()

    def reset(self) -> None:
        """Forget all accumulated statistics."""
        self.count = 0
        self.sum = 0.0
        self._sumsq = 0.0
        self.last = 0.0
        self._values: List[float] = []

    def update(self, value: Union[float, Sequence[float], Any], n: int = 1) -> float:
        """Add ``value`` (scalar or sequence) with weight ``n`` per element."""
        vals = _flatten_scalars(value)
        if not vals:
            vals = [float("nan")]
        for v in vals:
            self.count += 1
            self.sum += v
            self._sumsq += v * v
            self.last = v
            if self.window is not None:
                self._values.append(v)
                if len(self._values) > self.window:
                    del self._values[0]
        return self.mean

    @property
    def mean(self) -> float:
        if self.window is not None and self._values:
            return float(sum(self._values) / len(self._values))
        if self.count == 0:
            return 0.0
        return float(self.sum / self.count)

    @property
    def var(self) -> float:
        if self.count == 0:
            return 0.0
        mu = self.sum / self.count
        return max(0.0, float(self._sumsq / self.count - mu * mu))

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the meter."""
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum,
            "sumsq": self._sumsq,
            "last": self.last,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`."""
        self.name = state.get("name", self.name)
        self.count = int(state.get("count", 0))
        self.sum = float(state.get("sum", 0.0))
        self._sumsq = float(state.get("sumsq", 0.0))
        self.last = float(state.get("last", 0.0))

    def __float__(self) -> float:
        return self.mean

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"AverageMeter({self.name}, mean={self.mean:.4g}, count={self.count})"


class RunningMeanStd:
    """Welford-style running mean/variance estimator for observations.

    Kept dependency free (no numpy/bottleneck) so observation normalization can
    run in the environment wrapper on any host.
    """

    __slots__ = ("mean", "var", "count", "epsilon", "_m2")

    def __init__(self, epsilon: float = 1e-4, shape: Any = ()) -> None:
        self.epsilon = float(epsilon)
        shape = tuple(shape) if not isinstance(shape, int) else (shape,)
        self.mean = _zeros(shape)
        self._m2 = _zeros(shape)
        self.var = _ones(shape)
        self.count = float(epsilon)

    def update(self, x: Union[float, Sequence[float], Any], n: int = 1) -> None:
        """Merge a batch of observations ``x`` into the running statistics."""
        flat = _flatten_scalars(x)
        if not flat:
            return
        batch_mean = sum(flat) / len(flat)
        batch_var = 0.0
        if len(flat) > 1:
            batch_var = sum((v - batch_mean) ** 2 for v in flat) / len(flat)
        batch_count = float(len(flat) * max(1, n))
        self._merge(batch_mean, batch_var, batch_count)

    def _merge(self, batch_mean: float, batch_var: float, batch_count: float) -> None:
        delta = batch_mean - (self.mean if not isinstance(self.mean, list) else 0.0)
        total = self.count + batch_count
        if total <= 0:
            return
        if isinstance(self.mean, list):
            # shape-aware update
            for i in range(len(self.mean)):
                m = self.mean[i]
                v = self._m2[i]
                d = delta if not isinstance(delta, list) else delta[i]
                new_m = m + d * batch_count / total
                new_v = v + d * d * self.count * batch_count / total
                self.mean[i] = new_m
                self._m2[i] = new_v
                self.var[i] = new_v / total
        else:
            new_m = self.mean + delta * batch_count / total
            new_v = self._m2 + delta * delta * self.count * batch_count / total
            self.mean = new_m
            self._m2 = new_v
            self.var = new_v / total
        self.count = total

    @property
    def std(self) -> Any:
        return _sqrt_list(self.var)

    def normalize(self, x: Any, clip: float = 10.0) -> Any:
        """Normalize ``x`` by the running statistics and clip to ``clip``."""
        std = self.std
        if isinstance(x, list):
            out = [max(-clip, min(clip, (v - self.mean[i]) / (std[i] + 1e-8))) for i, v in enumerate(x)]
            return out
        return max(-clip, min(clip, (x - self.mean) / (std + 1e-8)))

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable snapshot."""
        return {"mean": self.mean, "var": self.var, "count": self.count, "m2": self._m2}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`."""
        self.mean = state.get("mean", self.mean)
        self.var = state.get("var", self.var)
        self.count = float(state.get("count", self.count))
        self._m2 = state.get("m2", self._m2)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"RunningMeanStd(count={self.count:.0f})"


# ---------------------------------------------------------------------------
# paper aggregation helpers
# ---------------------------------------------------------------------------
def paper_standard_error(curves: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Compute the paper's mean curve and shaded-band half-width.

    Implements the formula from Sec. 5.2 verbatim::

        mean(t) = (1/n) * sum_i y_i(t)
        band(t) = (2 / sqrt(n)) * sum_i ( y(t) - y_i(t) ) ** 2

    Parameters
    ----------
    curves:
        Sequence of ``n`` equally long per-seed curves.

    Returns
    -------
    ``(mean, band)`` lists of length ``T``.
    """
    curves = [list(map(float, c)) for c in curves if c is not None]
    if not curves:
        return [], []
    n = len(curves)
    length = min(len(c) for c in curves)
    mean: List[float] = []
    band: List[float] = []
    for t in range(length):
        vals = [c[t] for c in curves]
        mu = sum(vals) / n
        sq = sum((mu - v) ** 2 for v in vals)
        mean.append(mu)
        band.append(PAPER_BAND_SCALE / math.sqrt(n) * sq)
    return mean, band


#: Alias, matching the paper's terminology ("shaded region").
paper_band = paper_standard_error


def aggregate_curves(
    seed_curves: Sequence[Sequence[float]],
    samples: Optional[Sequence[float]] = None,
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate per-seed curves into ``{samples, mean, band, seed_curves}``.

    If ``samples`` (x-axis: transitions collected) is given together with
    ``num_bins``, every seed curve is interpolated onto a common grid so that
    seeds that ran a different number of iterations can still be averaged.
    """
    curves = [list(map(float, c)) for c in seed_curves if c is not None]
    out: Dict[str, Any] = {"seed_curves": curves, "num_seeds": len(curves)}
    if not curves:
        out.update({"samples": [], "mean": [], "band": []})
        return out

    if samples is not None and num_bins is not None and len(samples) > 1:
        grid = _linspace(float(samples[0]), float(samples[-1]), int(num_bins))
        resampled = [_interp(samples, c, grid) for c in curves]
        mean, band = paper_standard_error(resampled)
        out.update({"samples": grid, "mean": mean, "band": band, "seed_curves": resampled})
        return out

    mean, band = paper_standard_error(curves)
    out.update(
        {
            "samples": list(samples) if samples is not None else list(range(len(mean))),
            "mean": mean,
            "band": band,
        }
    )
    return out


# ---------------------------------------------------------------------------
# metric logger
# ---------------------------------------------------------------------------
class MetricLogger:
    """Records scalar series keyed by (name, step) and provides aggregations.

    Two step counters are tracked: an *iteration* counter (outer training
    iterations) and a *samples* counter (environment transitions collected --
    the x-axis used throughout the paper).  ``log`` writes both.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        verbose: bool = False,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        print_every: int = 1,
        seed: Optional[int] = None,
        run_name: Optional[str] = None,
        writer: Any = None,
    ) -> None:
        self.log_dir = log_dir
        self.verbose = bool(verbose)
        self.print_every = max(1, int(print_every))
        self.seed = seed
        self.run_name = run_name or self._default_run_name(seed)
        self.iteration = 0
        self.samples = 0
        self.frames = 0
        self.episodes = 0
        self.start_time = time.time()
        self.history: Dict[str, List[float]] = {}
        self.sample_history: Dict[str, List[float]] = {}
        self.iteration_history: Dict[str, List[int]] = {}
        self._latest: Dict[str, float] = {}
        self._writer = writer
        self._tb = None
        self._wandb = None
        self._jsonl_path: Optional[str] = None
        self._closed = False

        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self._jsonl_path = os.path.join(self.log_dir, "metrics.jsonl")
            except OSError:  # pragma: no cover - read-only filesystem
                self._jsonl_path = None

        if use_tensorboard and self._writer is None and self.log_dir:
            self._tb = self._make_tensorboard_writer(self.log_dir)

        if use_wandb:
            self._wandb = self._init_wandb()

    # -- naming / writers --------------------------------------------------
    def _default_run_name(self, seed: Optional[int]) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return f"sapg-{stamp}" + (f"-seed{seed}" if seed is not None else "")

    @staticmethod
    def _make_tensorboard_writer(log_dir: str) -> Any:
        try:  # pragma: no cover - optional dependency
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir)
        except Exception:  # pragma: no cover - tensorboard unavailable
            return None

    def _init_wandb(self) -> Any:  # pragma: no cover - optional dependency
        try:
            import wandb

            return wandb
        except Exception:
            return None

    # -- logging -----------------------------------------------------------
    def log(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
        samples: Optional[int] = None,
        step: Optional[int] = None,
        prefix: str = "",
        to_file: bool = True,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Record a dictionary of scalars.

        Parameters
        ----------
        metrics:
            Mapping name -> value.  None values are skipped.
        iteration / samples:
            Explicit counters; if omitted the internal counters are used.
        step:
            Alias for ``samples`` (or ``iteration`` if samples are unknown).
        prefix:
            String prepended to every metric name (e.g. ``"train/"``).
        """
        payload: Dict[str, Any] = {}
        if metrics:
            payload.update(metrics)
        payload.update(kwargs)

        if iteration is not None:
            self.iteration = int(iteration)
        if samples is not None:
            self.samples = int(samples)
        elif step is not None:
            self.samples = int(step) if self.samples else int(step)

        flat = _flatten_dict(payload, prefix=prefix)
        clean: Dict[str, float] = {}
        for key, value in flat.items():
            num = _to_float(value)
            if num is None or not math.isfinite(num):
                continue
            clean[key] = num
            self.history.setdefault(key, []).append(num)
            self.sample_history.setdefault(key, []).append(float(self.samples))
            self.iteration_history.setdefault(key, []).append(int(self.iteration))
            self._latest[key] = num

        if self._tb is not None:
            for key, value in clean.items():
                try:
                    self._tb.add_scalar(key, value, self.samples if self.samples else self.iteration)
                except Exception:  # pragma: no cover
                    pass
        if self._wandb is not None:  # pragma: no cover - optional dependency
            try:
                self._wandb.log(dict(clean), step=self.samples)
            except Exception:
                pass
        if to_file and self._jsonl_path:
            self._append_jsonl(clean)

        if self.verbose and self.iteration % self.print_every == 0:
            self.print(clean)
        return clean

    def add_scalar(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Single-scalar convenience wrapper around :meth:`log`."""
        self.log({name: value}, step=step)

    def log_episode_stats(
        self,
        episode_return: Any = None,
        episode_length: Any = None,
        successes: Any = None,
        num_episodes: Optional[int] = None,
        prefix: str = "train/",
        **extra: Any,
    ) -> Dict[str, float]:
        """Log the end-of-iteration episode statistics used by the paper.

        ``successes`` is the AllegroKuka ``successes/episode`` metric that drives
        the 7.5cm -> 1cm tolerance curriculum (Appendix A).
        """
        stats: Dict[str, Any] = {}
        if episode_return is not None:
            stats["episode_return"] = episode_return
        if episode_length is not None:
            stats["episode_length"] = episode_length
        if successes is not None:
            stats["successes"] = successes
        if num_episodes is not None:
            self.episodes += int(num_episodes)
            stats["num_episodes"] = int(num_episodes)
        stats.update(extra)
        return self.log(stats, prefix=prefix)

    def update_counters(self, samples: int = 0, episodes: int = 0, increment_iteration: bool = True) -> None:
        """Advance the sample/episode counters (and the iteration counter)."""
        self.samples += int(samples)
        self.episodes += int(episodes)
        if increment_iteration:
            self.iteration += 1

    # -- queries -----------------------------------------------------------
    def latest(self, key: Optional[str] = None, default: Any = None) -> Any:
        """Most recently recorded value (or the full dict when ``key`` is None)."""
        if key is None:
            return dict(self._latest)
        return self._latest.get(key, default)

    def get(self, key: str, default: Any = None) -> Any:
        """Alias of :meth:`latest` for a single key."""
        return self._latest.get(key, default)

    def curve(self, key: str) -> List[float]:
        """Recorded values of ``key`` in insertion order."""
        return list(self.history.get(key, []))

    def sample_curve(self, key: str) -> List[float]:
        """x-axis sample counts for ``key``."""
        return list(self.sample_history.get(key, []))

    def keys(self) -> List[str]:
        """All metric names recorded so far."""
        return sorted(self.history.keys())

    def mean(self, key: str) -> float:
        """Mean of all recorded values of ``key``."""
        vals = self.history.get(key, [])
        return float(sum(vals) / len(vals)) if vals else 0.0

    def summary(self, keys: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Mean of the last 10% of every (or the given) metric.

        This mirrors the "asymptotic performance" numbers reported in Table 1.
        """
        keys = list(keys) if keys is not None else self.keys()
        out: Dict[str, float] = {}
        for key in keys:
            vals = self.history.get(key)
            if not vals:
                continue
            n = max(1, len(vals) // 10)
            tail = vals[-n:]
            out[key] = float(sum(tail) / len(tail))
        return out

    def aggregate(
        self,
        key: str,
        num_bins: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Paper-style aggregation is per-seed; single-run returns raw curve."""
        return aggregate_curves([self.curve(key)], self.sample_curve(key), num_bins)

    # -- output ------------------------------------------------------------
    def print(self, metrics: Optional[Dict[str, float]] = None, header: bool = False) -> None:
        """Print a human readable line of metrics."""
        metrics = metrics if metrics is not None else dict(self._latest)
        if header:
            print(_SEPARATOR)
        elapsed = time.time() - self.start_time
        parts = " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items()) if _to_float(v) is not None)
        print(
            f"[iter {self.iteration:>5d} | samples {self.samples:>12d} | {elapsed:7.1f}s] {parts}",
            flush=True,
        )

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Store (and forward to tensorboard) the run hyperparameters."""
        self.hparams = dict(params)
        if self._tb is not None:
            try:  # pragma: no cover - optional dependency
                self._tb.add_text("hparams", json.dumps(_jsonable(params), sort_keys=True))
            except Exception:
                pass

    def save(self, path: Optional[str] = None) -> Optional[str]:
        """Persist the recorded history as JSON; returns the written path."""
        if path is None:
            if not self.log_dir:
                return None
            path = os.path.join(self.log_dir, "history.json")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "history": self.history,
                        "sample_history": self.sample_history,
                        "iteration_history": self.iteration_history,
                        "samples": self.samples,
                        "iteration": self.iteration,
                        "episodes": self.episodes,
                        "run_name": self.run_name,
                        "seed": self.seed,
                    },
                    fh,
                )
            return path
        except OSError:  # pragma: no cover - read-only filesystem
            return None

    def load(self, path: str) -> "MetricLogger":
        """Restore a history previously written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        self.history = {k: list(v) for k, v in blob.get("history", {}).items()}
        self.sample_history = {k: list(v) for k, v in blob.get("sample_history", {}).items()}
        self.iteration_history = {k: list(v) for k, v in blob.get("iteration_history", {}).items()}
        self.samples = int(blob.get("samples", 0))
        self.iteration = int(blob.get("iteration", 0))
        self.episodes = int(blob.get("episodes", 0))
        for key, vals in self.history.items():
            if vals:
                self._latest[key] = vals[-1]
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the logger state."""
        return {
            "iteration": self.iteration,
            "samples": self.samples,
            "episodes": self.episodes,
            "history": self.history,
            "sample_history": self.sample_history,
            "iteration_history": self.iteration_history,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`."""
        self.iteration = int(state.get("iteration", 0))
        self.samples = int(state.get("samples", 0))
        self.episodes = int(state.get("episodes", 0))
        self.history = {k: list(v) for k, v in state.get("history", {}).items()}
        self.sample_history = {k: list(v) for k, v in state.get("sample_history", {}).items()}
        self.iteration_history = {k: list(v) for k, v in state.get("iteration_history", {}).items()}

    def close(self) -> None:
        """Flush and release any external writers."""
        if self._closed:
            return
        self._closed = True
        if self._tb is not None:
            try:  # pragma: no cover - optional dependency
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
            self._tb = None

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- internals ---------------------------------------------------------
    def _append_jsonl(self, metrics: Dict[str, float]) -> None:
        if not self._jsonl_path:  # pragma: no cover - defensive
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "iteration": self.iteration,
                            "samples": self.samples,
                            "metrics": metrics,
                        }
                    )
                    + "\n"
                )
        except OSError:  # pragma: no cover - read-only filesystem
            self._jsonl_path = None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"MetricLogger(run={self.run_name}, iter={self.iteration}, samples={self.samples})"


#: ``Logger`` is the historical name used by the trainers.
Logger = MetricLogger


# ---------------------------------------------------------------------------
# experiment logger
# ---------------------------------------------------------------------------
class ExperimentLogger:
    """Facade combining a :class:`MetricLogger` with console/checkpoint helpers.

    The SAPG trainers (`SAPGTrainer`, `PPOTrainer`, the baselines) accept a
    duck-typed logger and only require ``log(...)`` / ``print(...)``, so this
    class can be used interchangeably with :class:`MetricLogger`.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        name: str = "sapg",
        config: Any = None,
        verbose: bool = True,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        seed: Optional[int] = None,
        print_every: int = 1,
    ) -> None:
        self.name = name
        self.config = config
        self.seed = seed
        self.log_dir = _resolve_log_dir(log_dir, name, seed)
        self.verbose = bool(verbose)
        self.metrics = MetricLogger(
            log_dir=self.log_dir,
            verbose=verbose,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb,
            seed=seed,
            run_name=name,
            print_every=print_every,
        )
        if config is not None:
            self.log_hyperparameters(config)

    # -- delegation --------------------------------------------------------
    def log(self, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, float]:
        """Delegate to the underlying :class:`MetricLogger`."""
        return self.metrics.log(metrics, **kwargs)

    def print(self, metrics: Optional[Dict[str, float]] = None, header: bool = False) -> None:
        """Delegate to the underlying :class:`MetricLogger`."""
        self.metrics.print(metrics, header=header)

    def update_counters(self, **kwargs: Any) -> None:
        """Delegate counter bookkeeping."""
        self.metrics.update_counters(**kwargs)

    def log_hyperparameters(self, params: Any) -> None:
        """Record the run configuration on both the console and the writer."""
        payload = _jsonable(params)
        self.metrics.log_hyperparameters(payload)
        if self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                with open(os.path.join(self.log_dir, "config.json"), "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, sort_keys=True)
            except (OSError, TypeError):  # pragma: no cover - defensive
                pass

    def log_metrics(self, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Alias kept for scripts that call ``logger.log_metrics``."""
        self.metrics.log(metrics, **kwargs)

    def info(self, message: str) -> None:
        """Print an informational line when verbose."""
        if self.verbose:
            print(f"[{self.name}] {message}", flush=True)

    def warning(self, message: str) -> None:
        """Print a warning to stderr."""
        print(f"[{self.name}] WARNING: {message}", file=sys.stderr, flush=True)

    @property
    def iteration(self) -> int:
        return self.metrics.iteration

    @property
    def samples(self) -> int:
        return self.metrics.samples

    @property
    def history(self) -> Dict[str, List[float]]:
        return self.metrics.history

    def summary(self, keys: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Mean of the tail of every metric (asymptotic performance, Table 1)."""
        return self.metrics.summary(keys)

    def save(self, path: Optional[str] = None) -> Optional[str]:
        """Persist the metric history."""
        return self.metrics.save(path)

    def state_dict(self) -> Dict[str, Any]:
        """Serialisable logger state (includes the config)."""
        state = self.metrics.state_dict()
        state["config"] = _jsonable(self.config)
        state["name"] = self.name
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore logger state (config is kept as-is)."""
        self.metrics.load_state_dict(state)

    def close(self) -> None:
        """Flush writers and save the history."""
        self.metrics.save()
        self.metrics.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        # Only called when normal attribute lookup fails: forward unknown
        # attributes (e.g. `curve`, `add_scalar`, `keys`) to the metric logger.
        metrics = self.__dict__.get("metrics")
        if metrics is not None and hasattr(metrics, item):
            return getattr(metrics, item)
        raise AttributeError(item)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ExperimentLogger(name={self.name}, log_dir={self.log_dir})"


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------
def make_logger(
    config: Any = None,
    log_dir: Optional[str] = None,
    name: str = "sapg",
    verbose: bool = True,
    use_tensorboard: bool = False,
    use_wandb: bool = False,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> ExperimentLogger:
    """Build an :class:`ExperimentLogger` from a config and/or explicit args.

    ``config`` may be a :class:`sapg.utils.config.SAPGConfig`, a plain dict or
    ``None``; its ``log_dir``/``seed``/``task``/``method`` fields are used as
    defaults when the corresponding argument is not supplied.
    """
    cfg = _as_mapping(config)
    if log_dir is None:
        log_dir = cfg.get("log_dir")
    if seed is None:
        seed = cfg.get("seed")
    method = str(cfg.get("method", "sapg") or "sapg")
    task = cfg.get("task")
    if name == "sapg":
        name = method if not task else f"{method}_{task}"
    return ExperimentLogger(
        log_dir=log_dir,
        name=name,
        config=config,
        verbose=verbose,
        use_tensorboard=use_tensorboard or bool(cfg.get("use_tensorboard", False)),
        use_wandb=use_wandb or bool(cfg.get("use_wandb", False)),
        seed=seed,
        print_every=int(cfg.get("print_every", kwargs.get("print_every", 1)) or 1),
    )


#: Process-wide default logger used by :func:`get_logger`.
_LOGGERS: Dict[str, ExperimentLogger] = {}


def get_logger(
    name: str = "sapg",
    log_dir: Optional[str] = None,
    config: Any = None,
    **kwargs: Any,
) -> ExperimentLogger:
    """Return a cached logger; creates it on first request for ``name``."""
    if name not in _LOGGERS:
        _LOGGERS[name] = make_logger(config=config, log_dir=log_dir, name=name, **kwargs)
    return _LOGGERS[name]


def reset_loggers() -> None:
    """Close and forget every cached logger (used between experiments)."""
    for logger in _LOGGERS.values():
        try:
            logger.close()
        except Exception:  # pragma: no cover - defensive
            pass
    _LOGGERS.clear()


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------
def format_metrics(metrics: Dict[str, Any], precision: int = 4, prefix: str = "") -> str:
    """Format a metric dict as ``key=value`` pairs for console output."""
    parts = []
    for key, value in sorted(metrics.items()):
        num = _to_float(value)
        if num is None:
            continue
        parts.append(f"{prefix}{key}={num:.{precision}g}")
    return " ".join(parts)


def print_header(title: str, width: int = 72) -> None:
    """Print a centered section header surrounded by separators."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n", flush=True)


def format_table(rows: Sequence[Sequence[Any]], headers: Sequence[str] = ()) -> str:
    """Render a simple fixed-width text table (used for Table-1 style output)."""
    rows = [list(map(_fmt_cell, r)) for r in rows]
    cols = max([len(headers)] + [len(r) for r in rows]) if (rows or headers) else 0
    if cols == 0:
        return ""
    widths = [len(str(headers[i])) if i < len(headers) else 0 for i in range(cols)]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]) if i < len(r) else 0)
    lines: List[str] = []
    if headers:
        lines.append("  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
        lines.append("  ".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        lines.append("  ".join((r[i] if i < len(r) else "").ljust(widths[i]) for i in range(cols)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------
def _fmt_cell(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return str(value)
    if abs(num) >= 1000:
        return f"{num:.4g}"
    return f"{num:.3f}"


def _resolve_log_dir(log_dir: Optional[str], name: str, seed: Optional[int]) -> Optional[str]:
    if log_dir is None:
        return None
    path = os.path.join(log_dir, name)
    if seed is not None:
        path = os.path.join(path, f"seed_{seed}")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:  # pragma: no cover - read-only filesystem
        remove = getattr(os.environ, "SAPG_ALLOW_MISSING_LOG_DIR", None)
        if not remove:
            return path
    return path


def _as_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            out = to_dict()
            if isinstance(out, dict):
                return out
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        import dataclasses

        if dataclasses.is_dataclass(config):
            return {k: v for k, v in dataclasses.asdict(config).items()}
    except Exception:  # pragma: no cover - defensive
        pass
    return {k: v for k, v in vars(config).items() if not k.startswith("_")} if hasattr(config, "__dict__") else {}


def _flatten_dict(data: Dict[str, Any], prefix: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten nested dicts into ``a/b/c`` keys.

    Values that look like reducible statistics (``mean``/``std``/``min``/``max``)
    are expanded to ``key/mean`` etc. only when the mapping holds no other
    scalar leaves, mirroring common RL logging conventions.
    """
    out: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        name = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, name + sep, sep))
        else:
            out[name] = value
    return out


def _flatten_scalars(value: Any) -> List[float]:
    """Extract finite floats from scalars / tensors / sequences."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        f = _to_float(value)
        return [f] if f is not None else []
    if hasattr(value, "detach"):  # torch tensors
        try:
            value = value.detach()
            if hasattr(value, "numel") and value.numel() > 1:
                value = value.reshape(-1)
            return [float(v) for v in value.tolist()] if hasattr(value, "tolist") else [float(value)]
        except Exception:  # pragma: no cover - defensive
            return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for v in value:
            out.extend(_flatten_scalars(v))
        return out
    f = _to_float(value)
    return [f] if f is not None else []


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of configs/tensors to JSON-serialisable data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        import dataclasses

        if dataclasses.is_dataclass(value):
            return _jsonable(dataclasses.asdict(value))
    except Exception:  # pragma: no cover - defensive
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def _interp(xs: Sequence[float], ys: Sequence[float], grid: Sequence[float]) -> List[float]:
    """Monotone-x linear interpolation of ``ys(xs)`` onto ``grid``."""
    if not xs or not ys:
        return [0.0 for _ in grid]
    out: List[float] = []
    n = min(len(xs), len(ys))
    xs = list(map(float, xs[:n]))
    ys = list(map(float, ys[:n]))
    j = 0
    for g in grid:
        while j < n - 2 and xs[j + 1] < g:
            j += 1
        x0, x1 = xs[j], xs[min(j + 1, n - 1)]
        y0, y1 = ys[j], ys[min(j + 1, n - 1)]
        if x1 == x0:
            out.append(y1)
        else:
            t = (g - x0) / (x1 - x0)
            t = max(0.0, min(1.0, t))
            out.append(y0 + t * (y1 - y0))
    return out


def _zeros(shape: Tuple[int, ...]) -> Any:
    if not shape:
        return 0.0
    return [0.0 for _ in range(shape[0])]


def _ones(shape: Tuple[int, ...]) -> Any:
    if not shape:
        return 1.0
    return [1.0 for _ in range(shape[0])]


def _sqrt_list(value: Any) -> Any:
    if isinstance(value, list):
        return [math.sqrt(max(0.0, v)) for v in value]
    return math.sqrt(max(0.0, float(value)))
