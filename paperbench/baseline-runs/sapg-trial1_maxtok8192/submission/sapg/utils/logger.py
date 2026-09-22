"""Metric logging utilities for SAPG.

Provides a lightweight, dependency-optional :class:`MetricLogger` that tracks
running statistics (mean / min / max / count) for scalar metrics and can
optionally forward them to TensorBoard.

The paper reports the following metrics:

* ``successes`` / ``episode_successes`` -- number of successes per episode
  (AllegroKuka hard tasks).
* ``episode_reward`` -- net episode reward (Shadow Hand / Allegro Hand easy
  tasks).
* ``delta`` -- current curriculum tolerance.
* PPO diagnostics (policy loss, value loss, entropy, approx KL, ...).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["MetricLogger", "RunningMeanStd"]


class RunningMeanStd:
    """Welford-style running mean / variance tracker for scalars."""

    def __init__(self) -> None:
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float, n: float = 1.0) -> None:
        value = float(value)
        self.count += n
        delta = value - self.mean
        self.mean += (n / self.count) * delta
        self.m2 += n * delta * (value - self.mean)
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    @property
    def var(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.var ** 0.5

    def reset(self) -> None:
        self.__init__()

    def state_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "min": self.min,
            "max": self.max,
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.count = state.get("count", 0.0)
        self.mean = state.get("mean", 0.0)
        self.m2 = state.get("m2", 0.0)
        self.min = state.get("min", float("inf"))
        self.max = state.get("max", float("-inf"))


class MetricLogger:
    """Accumulates scalar metrics and periodically flushes them.

    Parameters
    ----------
    log_dir:
        Directory for the optional JSONL log file and TensorBoard events.
    use_tensorboard:
        If ``True`` and ``torch.utils.tensorboard`` is importable, metrics are
        also written to TensorBoard.
    print_every:
        Print aggregated metrics every ``print_every`` flushes (0 disables).
    prefix:
        Optional prefix prepended to every metric name.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_tensorboard: bool = False,
        print_every: int = 1,
        prefix: str = "",
        verbose: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.prefix = prefix
        self.print_every = print_every
        self.verbose = verbose

        self._stats: Dict[str, RunningMeanStd] = defaultdict(RunningMeanStd)
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._flush_count = 0
        self._start_time = time.time()

        self._writer = None
        if use_tensorboard and log_dir is not None:
            try:  # pragma: no cover - optional dependency
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=log_dir)
            except Exception:  # pragma: no cover
                self._writer = None

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def log(self, key: str, value: Any, n: float = 1.0) -> None:
        """Record a single scalar value under ``key``."""
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if value != value:  # NaN guard
            return
        self._stats[key].update(value, n=n)

    def log_dict(self, metrics: Dict[str, Any], n: float = 1.0) -> None:
        """Record every scalar entry of ``metrics``."""
        for key, value in metrics.items():
            self.log(key, value, n=n)

    def log_scalars(self, metrics: Dict[str, Any], step: int) -> None:
        """Directly write scalars to TensorBoard at ``step`` (no aggregation)."""
        if self._writer is None:
            return
        for key, value in metrics.items():
            try:
                self._writer.add_scalar(self._prefixed(key), float(value), step)
            except (TypeError, ValueError):
                continue

    # ------------------------------------------------------------------
    # Aggregation / flushing
    # ------------------------------------------------------------------
    def _prefixed(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    def mean(self, key: str) -> float:
        return self._stats[key].mean if key in self._stats else 0.0

    def get(self, key: str) -> Optional[float]:
        if key not in self._stats:
            return None
        return self._stats[key].mean

    def aggregate(self) -> Dict[str, float]:
        """Return the current mean of every tracked metric."""
        return {k: v.mean for k, v in self._stats.items()}

    def flush(self, step: Optional[int] = None, reset: bool = True) -> Dict[str, float]:
        """Emit aggregated metrics, optionally resetting accumulators."""
        aggregated = self.aggregate()
        self._flush_count += 1

        if self._writer is not None and step is not None:
            for key, value in aggregated.items():
                self._writer.add_scalar(self._prefixed(key), value, step)
            self._writer.flush()

        if self.log_dir is not None:
            record = {"step": step, "time": time.time() - self._start_time}
            record.update({self._prefixed(k): v for k, v in aggregated.items()})
            path = os.path.join(self.log_dir, "metrics.jsonl")
            try:
                with open(path, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
            except OSError:  # pragma: no cover
                pass

        for key, value in aggregated.items():
            self._history[key].append(value)

        if (
            self.verbose
            and self.print_every > 0
            and self._flush_count % self.print_every == 0
        ):
            self._print(aggregated, step)

        if reset:
            self.reset()
        return aggregated

    def _print(self, aggregated: Dict[str, float], step: Optional[int]) -> None:
        step_str = f"step={step}" if step is not None else f"flush={self._flush_count}"
        parts = [f"{k}={v:.4g}" for k, v in sorted(aggregated.items())]
        print(f"[MetricLogger] {step_str} | " + " | ".join(parts), flush=True)

    def reset(self) -> None:
        self._stats = defaultdict(RunningMeanStd)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def history(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._history.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "stats": {k: v.state_dict() for k, v in self._stats.items()},
            "history": {k: list(v) for k, v in self._history.items()},
            "flush_count": self._flush_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._stats = defaultdict(RunningMeanStd)
        for key, stat in state.get("stats", {}).items():
            self._stats[key].load_state_dict(stat)
        self._history = defaultdict(list)
        for key, values in state.get("history", {}).items():
            self._history[key] = list(values)
        self._flush_count = state.get("flush_count", 0)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
