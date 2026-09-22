"""Logging utilities for SAPG.

Provides:
  * :class:`Logger` -- a lightweight metric logger that tracks scalar series
    (mean / standard error), writes to TensorBoard when available, dumps JSON
    summaries, and plots learning curves with shaded standard-error bands.
  * :func:`plot_curves` -- standalone curve plotting helper used to reproduce
    Figures 2, 5 and 6 of the SAPG paper.
  * :func:`standard_error` -- helper implementing the shaded-width convention
    described in the paper: ``2 / sqrt(n) * sum_t (y(t) - y_i(t))^2`` style
    aggregation is exposed through :func:`aggregate_runs`.

The logger is intentionally dependency-light: TensorBoard and matplotlib are
optional imports so the training loop can run headless without them.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - tensorboard optional
    SummaryWriter = None  # type: ignore

try:  # optional
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None  # type: ignore


__all__ = [
    "Logger",
    "standard_error",
    "aggregate_runs",
    "plot_curves",
    "MovingAverage",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def standard_error(values: Sequence[float]) -> float:
    """Standard error of the mean: ``std / sqrt(n)`` (0 for n <= 1)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def aggregate_runs(
    runs: Sequence[Sequence[float]],
    num_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate several runs (e.g. 5 seeds) into mean and standard error.

    Args:
        runs: list of per-seed curves. Curves may have different lengths; they
            are linearly interpolated onto a common grid when ``num_points`` is
            given, otherwise truncated to the shortest run.
        num_points: number of points on the common x-grid.

    Returns:
        ``(x, mean, stderr)`` arrays. ``stderr`` is the standard error of the
        mean across seeds, matching the shaded band convention of the paper.
    """
    runs = [np.asarray(r, dtype=np.float64) for r in runs if len(r) > 0]
    if not runs:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    if num_points is None:
        length = min(len(r) for r in runs)
        stacked = np.stack([r[:length] for r in runs], axis=0)
        x = np.arange(length, dtype=np.float64)
    else:
        x = np.linspace(0.0, 1.0, num_points)
        interp = []
        for r in runs:
            src_x = np.linspace(0.0, 1.0, len(r))
            interp.append(np.interp(x, src_x, r))
        stacked = np.stack(interp, axis=0)

    mean = stacked.mean(axis=0)
    if stacked.shape[0] > 1:
        stderr = stacked.std(axis=0, ddof=1) / math.sqrt(stacked.shape[0])
    else:
        stderr = np.zeros_like(mean)
    return x, mean, stderr


class MovingAverage:
    """Exponential moving average tracker for scalar metrics."""

    def __init__(self, alpha: float = 0.99) -> None:
        self.alpha = float(alpha)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * x
        return self.value

    def reset(self) -> None:
        self.value = None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Logger:
    """Metric logger with TensorBoard, JSON and matplotlib backends.

    Args:
        log_dir: directory where logs/plots are written.
        use_tensorboard: enable TensorBoard ``SummaryWriter`` if installed.
        verbose: print metrics to stdout.
        plot_interval: how often (in ``log()`` calls) to refresh the curve
            plot. ``0`` disables plotting.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        use_tensorboard: bool = True,
        verbose: bool = True,
        plot_interval: int = 0,
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.verbose = verbose
        self.plot_interval = int(plot_interval)

        self._writer = None
        if use_tensorboard and SummaryWriter is not None:
            try:
                self._writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self._writer = None

        # history[tag] = list of (step, value)
        self.history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self._n_logs = 0
        self._start_time = time.time()
        self._last_step = 0

    # -- core API ----------------------------------------------------------
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Record a dict of scalar metrics at ``step``."""
        step = int(step)
        self._last_step = step
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(scalar):
                continue
            self.history[key].append((step, scalar))
            if self._writer is not None:
                self._writer.add_scalar(key, scalar, step)

        self._n_logs += 1
        if self.verbose:
            elapsed = time.time() - self._start_time
            pretty = "  ".join(
                f"{k}={float(v):.4g}"
                for k, v in metrics.items()
                if _is_number(v)
            )
            print(f"[step {step:>10d}] ({elapsed:7.1f}s) {pretty}", flush=True)

        if self.plot_interval and self._n_logs % self.plot_interval == 0:
            self.plot()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.log({tag: value}, step)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """Alias for :meth:`log_scalar` (TensorBoard-style naming)."""
        self.log_scalar(tag, value, step)

    # -- accessors ---------------------------------------------------------
    def get(self, tag: str) -> List[Tuple[int, float]]:
        return self.history.get(tag, [])

    def last(self, tag: str, default: float = 0.0) -> float:
        series = self.history.get(tag)
        if not series:
            return default
        return series[-1][1]

    def mean(self, tag: str, last_n: Optional[int] = None) -> float:
        series = self.history.get(tag)
        if not series:
            return 0.0
        values = [v for _, v in series]
        if last_n is not None:
            values = values[-last_n:]
        return float(np.mean(values))

    def stderr(self, tag: str, last_n: Optional[int] = None) -> float:
        series = self.history.get(tag)
        if not series:
            return 0.0
        values = [v for _, v in series]
        if last_n is not None:
            values = values[-last_n:]
        return standard_error(values)

    # -- persistence -------------------------------------------------------
    def save_json(self, filename: str = "metrics.json") -> str:
        path = os.path.join(self.log_dir, filename)
        payload = {
            tag: {"steps": [s for s, _ in series], "values": [v for _, v in series]}
            for tag, series in self.history.items()
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def load_json(self, filename: str = "metrics.json") -> None:
        path = os.path.join(self.log_dir, filename)
        if not os.path.exists(path):
            return
        with open(path, "r") as fh:
            payload = json.load(fh)
        self.history = defaultdict(list)
        for tag, series in payload.items():
            self.history[tag] = list(zip(series["steps"], series["values"]))

    # -- plotting ----------------------------------------------------------
    def plot(
        self,
        tags: Optional[Iterable[str]] = None,
        filename: str = "curves.png",
        smooth: int = 1,
    ) -> Optional[str]:
        """Plot all (or selected) tracked metrics into a single figure."""
        if plt is None:
            return None
        tags = list(tags) if tags is not None else list(self.history.keys())
        tags = [t for t in tags if self.history.get(t)]
        if not tags:
            return None

        n = len(tags)
        ncols = min(3, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        for idx, tag in enumerate(tags):
            ax = axes[idx // ncols][idx % ncols]
            steps = [s for s, _ in self.history[tag]]
            values = [v for _, v in self.history[tag]]
            if smooth > 1 and len(values) >= smooth:
                values = _smooth(values, smooth)
            ax.plot(steps, values, linewidth=1.5)
            ax.set_title(tag, fontsize=9)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.tight_layout()
        path = os.path.join(self.log_dir, filename)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
            self._writer = None

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Standalone plotting (Figures 2 / 5 / 6)
# ---------------------------------------------------------------------------
def plot_curves(
    curves: Dict[str, Sequence[Sequence[float]]],
    xlabel: str = "transitions",
    ylabel: str = "successes",
    title: str = "",
    filename: Optional[str] = None,
    x_values: Optional[Sequence[float]] = None,
    num_points: int = 200,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> Optional[str]:
    """Plot multiple methods with shaded standard-error bands.

    Args:
        curves: mapping ``method_name -> list of per-seed curves``.
        xlabel / ylabel / title: axis labels.
        filename: if given, save the figure to this path.
        x_values: optional common x-axis values (defaults to normalized grid).
        num_points: number of points on the interpolation grid.

    Returns:
        The saved filename (or ``None`` if matplotlib is unavailable).
    """
    if plt is None:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    for name, runs in curves.items():
        x, mean, stderr = aggregate_runs(runs, num_points=num_points)
        if x_values is not None:
            x = np.asarray(x_values, dtype=np.float64)
            if len(x) != len(mean):
                x = np.linspace(float(x[0]), float(x[-1]), len(mean))
        ax.plot(x, mean, label=name, linewidth=1.8)
        ax.fill_between(x, mean - stderr, mean + stderr, alpha=0.25)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        fig.savefig(filename, dpi=140)
        plt.close(fig)
        return filename
    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _smooth(values: Sequence[float], window: int) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or arr.size < window:
        return list(arr)
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, arr[0]), arr])
    return list(np.convolve(padded, kernel, mode="valid"))
