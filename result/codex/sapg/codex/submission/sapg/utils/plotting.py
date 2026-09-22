"""Plotting / aggregation helpers for the paper's figures and tables.

The paper reports "the mean and standard error in the plots.  In each plot, the
solid line is y(t) = 1/n sum_i y_i(t) while the width of the shaded region is
determined by standard error" (Sec. 5.2), so every helper here returns the mean
and the standard error over seeds; curves are plotted with a shaded region of
+/- 2 standard errors.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_progress(csv_path: str) -> Dict[str, np.ndarray]:
    """Load a ``progress.csv`` keeping every column row-aligned.

    Evaluation rows only fill the ``eval*`` columns, so missing entries become
    NaN (they are dropped when a curve is built).
    """
    with open(csv_path, "r") as handle:
        rows = list(csv.DictReader(handle))
    columns: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key is None:
                continue
            try:
                columns.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                columns.setdefault(key, []).append(float("nan"))
    for key, values in columns.items():  # pad columns introduced late
        if len(values) < len(rows):
            values.extend([float("nan")] * (len(rows) - len(values)))
    return {key: np.asarray(values) for key, values in columns.items()}


def has_metric(data: Dict[str, np.ndarray], metric: str) -> bool:
    """True if ``metric`` (or its per-policy ``eval*`` variants) is present."""
    if metric in data:
        return True
    suffix = metric.split("/")[-1]
    return any(key.startswith("eval") and key.endswith("/" + suffix) for key in data)


def standard_error(values: np.ndarray, axis: int = 0) -> np.ndarray:
    n = values.shape[axis]
    if n <= 1:
        return np.zeros_like(values.mean(axis=axis))
    return values.std(axis=axis, ddof=1) / np.sqrt(n)


def _resample(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate (and forward-fill) a seed's curve onto a common x grid."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size == 1:
        return np.full_like(grid, y[0], dtype=np.float64)
    out = np.interp(grid, x, y, left=y[0], right=y[-1])
    return out


def curve_stats(
    run_dirs: Sequence[str],
    metric: str,
    x_metric: str = "env_steps",
    num_points: int = 200,
    reduce: str = "mean",
    csv_name: str = "progress.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error of ``metric`` across seeds."""
    curves = []
    grid: Optional[np.ndarray] = None
    for run_dir in run_dirs:
        path = os.path.join(run_dir, csv_name)
        if not os.path.exists(path):
            continue
        data = load_progress(path)
        if not has_metric(data, metric) or x_metric not in data:
            continue
        y = _reduce_policies(data, metric, reduce)
        x = data[x_metric]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            continue
        if grid is None:
            grid = np.linspace(x.min(), x.max(), num_points)
        curves.append(_resample(x, y, grid))
    if not curves or grid is None:
        return np.asarray([]), np.asarray([]), np.asarray([])
    stacked = np.vstack(curves)
    return grid, stacked.mean(axis=0), 2.0 * standard_error(stacked, axis=0)


def _reduce_policies(data: Dict[str, np.ndarray], metric: str, reduce: str) -> np.ndarray:
    """Handle multi-policy metrics such as ``eval3/successes``.

    DexPBT evaluates its whole population; the reported number for a
    population-based method is the performance of the *best* member (that is
    what population-based training selects), while SAPG reports its leader.
    """
    if metric in data:
        return data[metric]
    suffix = metric.split("/")[-1]
    candidates = sorted(key for key in data if key.startswith("eval") and key.endswith("/" + suffix))
    if not candidates:
        raise KeyError(f"Metric '{metric}' not found in progress.csv")
    stacked = np.vstack([data[key] for key in candidates])
    if reduce == "max":
        return np.nanmax(stacked, axis=0)
    if reduce == "min":
        return np.nanmin(stacked, axis=0)
    return np.nanmean(stacked, axis=0)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    kernel = np.ones(window) / float(window)
    return np.convolve(values, kernel, mode="same")


def plot_curves(
    series: Dict[str, Sequence[str]],
    out_path: str,
    metric: str = "eval0/successes",
    x_metric: str = "env_steps",
    xlabel: str = "environment steps",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    smooth_window: int = 1,
    logx: bool = False,
    reduce: str = "mean",
    reduce_map: Optional[Dict[str, str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> str:
    """Plot one curve per method (mean over seeds with a +/-2 SE band)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, run_dirs in series.items():
        label_reduce = (reduce_map or {}).get(label, reduce)
        x, mean, band = curve_stats(run_dirs, metric, x_metric=x_metric, reduce=label_reduce)
        if x.size == 0:
            continue
        ax.plot(x, smooth(mean, smooth_window), label=label)
        ax.fill_between(x, smooth(mean - band, smooth_window), smooth(mean + band, smooth_window), alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def summarise_final(
    series: Dict[str, Sequence[str]],
    metric: str,
    x_metric: str = "env_steps",
    last_fraction: float = 0.1,
    reduce: str = "mean",
    reduce_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Mean +/- standard error of the final (asymptotic) performance."""
    summary: Dict[str, Dict[str, float]] = {}
    for label, run_dirs in series.items():
        finals: List[float] = []
        label_reduce = (reduce_map or {}).get(label, reduce)
        for run_dir in run_dirs:
            path = os.path.join(run_dir, "progress.csv")
            if not os.path.exists(path):
                continue
            data = load_progress(path)
            if not has_metric(data, metric):
                continue
            values = _reduce_policies(data, metric, label_reduce)
            steps = data.get(x_metric, np.arange(values.size))
            finite = np.isfinite(values) & np.isfinite(steps)
            values, steps = values[finite], steps[finite]
            if values.size == 0:
                continue
            keep = steps >= steps.max() * (1.0 - last_fraction)
            values = values[keep]
            if values.size:
                finals.append(float(np.mean(values)))
        if finals:
            summary[label] = {
                "mean": float(np.mean(finals)),
                "stderr": float(np.std(finals, ddof=1) / np.sqrt(len(finals))) if len(finals) > 1 else 0.0,
                "n_seeds": float(len(finals)),
            }
    return summary


def summary_to_markdown(summary: Dict[str, Dict[str, float]], float_format: str = "{:.3g}") -> str:
    lines = ["| method | mean | std. error | seeds |", "| --- | --- | --- | --- |"]
    for label, stats in summary.items():
        lines.append(
            f"| {label} | {float_format.format(stats['mean'])} | "
            f"{float_format.format(stats['stderr'])} | {int(stats['n_seeds'])} |"
        )
    return "\n".join(lines)


def summary_to_latex(summary: Dict[str, Dict[str, float]], float_format: str = "{:.3g}") -> str:
    lines = [r"\begin{tabular}{|c|c|c|}", r"\hline method & mean & std. error \\", r"\hline"]
    for label, stats in summary.items():
        lines.append(
            f"{label} & {float_format.format(stats['mean'])} & {float_format.format(stats['stderr'])} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)
