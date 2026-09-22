"""Plotting utilities for SAPG experiment results.

This module reproduces the paper's figures from the JSON result files written by
``experiments/train_sapg.py`` and ``experiments/train_baselines.py``:

* **Figure 5 / Table 1** -- learning curves and final performance for SAPG vs.
  baselines (PPO, PBT, PQL) across the five tasks.
* **Figure 6** -- ablation curves (entropy coefficient sweep, symmetric update,
  w/o off-policy, high off-policy ratio).
* **Figure 2** -- PPO saturation with batch size (optional helper).

The paper reports ``mean +/- standard error`` where the shaded band is computed
as ``2 / sqrt(n) * sum((y - y_i)^2)`` over seeds (i.e. 2 standard errors of the
mean).  All curves are smoothed with an exponential moving average for
readability, matching common RL plotting conventions.

Public interface
----------------
* :class:`PlotConfig` -- dataclass of plotting options.
* :func:`load_results` -- load JSON result files from a directory.
* :func:`smooth` -- exponential moving average smoothing.
* :func:`aggregate_curves` -- mean/std-error aggregation across seeds.
* :func:`plot_learning_curves` -- Figure 5 style comparison plot.
* :func:`plot_ablations` -- Figure 6 style ablation plot.
* :func:`plot_final_performance` -- Table 1 style bar chart.
* :func:`plot_all` -- convenience entry point used by ``main.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # matplotlib is optional at import time (headless CI)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - matplotlib missing
    plt = None  # type: ignore
    _HAS_MPL = False


__all__ = [
    "PlotConfig",
    "load_results",
    "smooth",
    "aggregate_curves",
    "plot_learning_curves",
    "plot_ablations",
    "plot_final_performance",
    "plot_all",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PlotConfig:
    """Configuration for result plotting.

    Attributes:
        results_dir: Directory containing JSON result files.
        figures_dir: Directory where figures are written.
        smoothing: EMA smoothing factor in ``(0, 1]`` (1.0 = no smoothing).
        tasks: Tasks to plot (defaults to the paper's five tasks).
        algos: Algorithms to include in the comparison plot.
        dpi: Output figure resolution.
        formats: File formats to write (e.g. ``["png", "pdf"]``).
        use_standard_error: If True, shade ``2 * SEM``; else shade ``std``.
        max_points: Optional down-sampling of the x-axis for readability.
        title: Optional figure title prefix.
    """

    results_dir: str = "results"
    figures_dir: str = "figures"
    smoothing: float = 0.9
    tasks: Sequence[str] = field(
        default_factory=lambda: [
            "allegrohand",
            "shadowhand",
            "regrasping",
            "throw",
            "reorientation",
        ]
    )
    algos: Sequence[str] = field(
        default_factory=lambda: ["sapg", "ppo", "pbt", "pql"]
    )
    dpi: int = 150
    formats: Sequence[str] = field(default_factory=lambda: ["png"])
    use_standard_error: bool = True
    max_points: Optional[int] = None
    title: str = ""

    # Pretty labels / colors used across figures.
    LABELS: Dict[str, str] = field(
        default_factory=lambda: {
            "sapg": "SAPG",
            "sapg_entropy": "SAPG (entropy coef)",
            "sapg_symmetric": "SAPG (symmetric)",
            "sapg_no_off_policy": "SAPG (w/o off-policy)",
            "sapg_high_off_policy_ratio": "SAPG (high off-policy ratio)",
            "ppo": "PPO",
            "pbt": "PBT",
            "pql": "PQL",
        },
        repr=False,
    )
    COLORS: Dict[str, str] = field(
        default_factory=lambda: {
            "sapg": "#1f77b4",
            "sapg_entropy": "#17becf",
            "sapg_symmetric": "#d62728",
            "sapg_no_off_policy": "#ff7f0e",
            "sapg_high_off_policy_ratio": "#9467bd",
            "ppo": "#2ca02c",
            "pbt": "#8c564b",
            "pql": "#e377c2",
        },
        repr=False,
    )

    def label(self, algo: str) -> str:
        return self.LABELS.get(algo, algo.upper())

    def color(self, algo: str) -> Optional[str]:
        return self.COLORS.get(algo)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files from ``results_dir`` (recursively).

    Returns a list of result dictionaries.  Files that fail to parse are
    skipped with a warning printed to stderr.
    """
    results: List[Dict[str, Any]] = []
    if not os.path.isdir(results_dir):
        return results
    for root, _dirs, files in os.walk(results_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r") as fh:
                    payload = json.load(fh)
            except Exception as exc:  # pragma: no cover - corrupt file
                print(f"[plot_results] skipping {path}: {exc}")
                continue
            if isinstance(payload, dict):
                payload.setdefault("_path", path)
                results.append(payload)
    return results


def _infer_algo(result: Dict[str, Any]) -> str:
    """Best-effort inference of the algorithm name from a result payload."""
    algo = result.get("algo")
    if algo:
        return str(algo).lower()
    path = str(result.get("_path", "")).lower()
    for key in ("sapg", "ppo", "pbt", "pql"):
        if key in path:
            return key
    return "unknown"


def _infer_task(result: Dict[str, Any]) -> str:
    task = result.get("task")
    if task:
        return str(task).lower()
    path = str(result.get("_path", "")).lower()
    for key in (
        "allegrohand",
        "shadowhand",
        "regrasping",
        "throw",
        "reorientation",
    ):
        if key in path:
            return key
    return "unknown"


def _extract_curve(
    result: Dict[str, Any], metric: str = "return"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``(x, y)`` arrays for a metric from a result payload.

    Supports two history layouts:

    * ``history`` is a list of per-iteration dicts containing ``total_samples``
      and the metric (e.g. ``mean_return`` / ``return`` / ``eval_return``).
    * ``history`` is a dict of metric -> list of values, with an optional
      ``samples`` / ``total_samples`` key for the x-axis.
    """
    history = result.get("history")
    if history is None:
        return np.zeros(0), np.zeros(0)

    metric_keys = [
        metric,
        f"eval_{metric}",
        f"mean_{metric}",
        "eval_return",
        "mean_return",
        "return",
        "episode_return",
        "reward",
    ]

    if isinstance(history, dict):
        xs = None
        for xkey in ("total_samples", "samples", "steps", "iteration"):
            if xkey in history:
                xs = np.asarray(history[xkey], dtype=np.float64)
                break
        for mkey in metric_keys:
            if mkey in history:
                ys = np.asarray(history[mkey], dtype=np.float64)
                if xs is None or len(xs) != len(ys):
                    xs = np.arange(len(ys), dtype=np.float64)
                return xs, ys
        return np.zeros(0), np.zeros(0)

    if isinstance(history, list) and history:
        xs: List[float] = []
        ys: List[float] = []
        for i, entry in enumerate(history):
            if not isinstance(entry, dict):
                continue
            x = None
            for xkey in ("total_samples", "samples", "steps", "iteration"):
                if xkey in entry and entry[xkey] is not None:
                    x = float(entry[xkey])
                    break
            if x is None:
                x = float(i)
            y = None
            for mkey in metric_keys:
                if mkey in entry and entry[mkey] is not None:
                    try:
                        y = float(entry[mkey])
                    except (TypeError, ValueError):
                        y = None
                    if y is not None:
                        break
            if y is None:
                continue
            xs.append(x)
            ys.append(y)
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    return np.zeros(0), np.zeros(0)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def smooth(values: np.ndarray, factor: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing.

    ``factor`` is the weight of the *previous* smoothed value (0 = no
    smoothing, close to 1 = heavy smoothing).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or factor <= 0.0:
        return values
    factor = float(min(max(factor, 0.0), 0.999))
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = factor * acc + (1.0 - factor) * v
        out[i] = acc
    return out


def _interp_to_grid(
    xs: np.ndarray, ys: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a curve onto a common x-grid (extrapolate flat)."""
    if xs.size == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    if xs.size == 1:
        return np.full_like(grid, ys[0], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # np.interp clamps outside the range to the boundary values.
    return np.interp(grid, xs, ys)


def aggregate_curves(
    curves: Sequence[Tuple[np.ndarray, np.ndarray]],
    smoothing: float = 0.9,
    use_standard_error: bool = True,
    num_points: int = 200,
) -> Dict[str, np.ndarray]:
    """Aggregate multiple seed curves into mean and shaded band.

    Args:
        curves: Sequence of ``(x, y)`` arrays (one per seed).
        smoothing: EMA factor applied to each seed curve before aggregation.
        use_standard_error: If True the band is ``2 * SEM`` (paper convention);
            otherwise it is the standard deviation across seeds.
        num_points: Number of points on the common x-grid.

    Returns:
        Dict with keys ``x``, ``mean``, ``band``, ``n``.
    """
    curves = [(np.asarray(x, float), np.asarray(y, float)) for x, y in curves]
    curves = [(x, y) for x, y in curves if x.size > 0 and y.size > 0]
    if not curves:
        return {
            "x": np.zeros(0),
            "mean": np.zeros(0),
            "band": np.zeros(0),
            "n": 0,
        }

    x_min = min(float(x.min()) for x, _ in curves)
    x_max = max(float(x.max()) for x, _ in curves)
    if x_max <= x_min:
        x_max = x_min + 1.0
    grid = np.linspace(x_min, x_max, num_points)

    stacked = np.stack(
        [_interp_to_grid(x, smooth(y, smoothing), grid) for x, y in curves],
        axis=0,
    )
    mean = np.nanmean(stacked, axis=0)
    n = stacked.shape[0]
    if n > 1:
        if use_standard_error:
            # Paper: shaded = 2 / sqrt(n) * sum((y - y_i)^2)  -> 2 * SEM
            var = np.nanmean((stacked - mean[None, :]) ** 2, axis=0)
            band = 2.0 * np.sqrt(var / n)
        else:
            band = np.nanstd(stacked, axis=0)
    else:
        band = np.zeros_like(mean)
    return {"x": grid, "mean": mean, "band": band, "n": n}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _save_figure(fig, figures_dir: str, name: str, cfg: PlotConfig) -> List[str]:
    _ensure_dir(figures_dir)
    paths: List[str] = []
    for fmt in cfg.formats:
        out = os.path.join(figures_dir, f"{name}.{fmt}")
        fig.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
        paths.append(out)
    if _HAS_MPL:
        plt.close(fig)
    return paths


def _group_results(
    results: Iterable[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Group results by ``(task, algo)``."""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for res in results:
        key = (_infer_task(res), _infer_algo(res))
        grouped.setdefault(key, []).append(res)
    return grouped


def plot_learning_curves(
    results: Sequence[Dict[str, Any]],
    cfg: Optional[PlotConfig] = None,
    metric: str = "return",
    name: str = "figure5_learning_curves",
) -> List[str]:
    """Reproduce Figure 5: per-task learning curves for all algorithms."""
    cfg = cfg or PlotConfig()
    if not _HAS_MPL:
        print("[plot_results] matplotlib unavailable; skipping learning curves")
        return []

    grouped = _group_results(results)
    tasks = [t for t in cfg.tasks if any(k[0] == t for k in grouped)]
    if not tasks:
        tasks = sorted({k[0] for k in grouped})
    if not tasks:
        print("[plot_results] no results found for learning curves")
        return []

    ncols = min(3, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False
    )

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        for algo in cfg.algos:
            seeds = grouped.get((task, algo), [])
            if not seeds:
                continue
            curves = [_extract_curve(r, metric) for r in seeds]
            agg = aggregate_curves(
                curves,
                smoothing=cfg.smoothing,
                use_standard_error=cfg.use_standard_error,
            )
            if agg["n"] == 0:
                continue
            color = cfg.color(algo)
            ax.plot(
                agg["x"],
                agg["mean"],
                label=f"{cfg.label(algo)} (n={agg['n']})",
                color=color,
                linewidth=1.8,
            )
            ax.fill_between(
                agg["x"],
                agg["mean"] - agg["band"],
                agg["mean"] + agg["band"],
                color=color,
                alpha=0.25,
                linewidth=0,
            )
        ax.set_title(task)
        ax.set_xlabel("samples")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Hide unused axes.
    for j in range(len(tasks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if cfg.title:
        fig.suptitle(cfg.title, fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, cfg.figures_dir, name, cfg)


def plot_ablations(
    results: Sequence[Dict[str, Any]],
    cfg: Optional[PlotConfig] = None,
    metric: str = "return",
    name: str = "figure6_ablations",
) -> List[str]:
    """Reproduce Figure 6: ablation curves (entropy, symmetric, off-policy)."""
    cfg = cfg or PlotConfig()
    if not _HAS_MPL:
        print("[plot_results] matplotlib unavailable; skipping ablations")
        return []

    ablation_algos = [
        "sapg",
        "sapg_entropy",
        "sapg_symmetric",
        "sapg_no_off_policy",
        "sapg_high_off_policy_ratio",
    ]
    grouped = _group_results(results)
    tasks = [t for t in cfg.tasks if any(k[0] == t for k in grouped)]
    if not tasks:
        tasks = sorted({k[0] for k in grouped})
    if not tasks:
        print("[plot_results] no results found for ablations")
        return []

    ncols = min(3, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False
    )

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        plotted = False
        for algo in ablation_algos:
            seeds = grouped.get((task, algo), [])
            if not seeds:
                continue
            curves = [_extract_curve(r, metric) for r in seeds]
            agg = aggregate_curves(
                curves,
                smoothing=cfg.smoothing,
                use_standard_error=cfg.use_standard_error,
            )
            if agg["n"] == 0:
                continue
            color = cfg.color(algo)
            ax.plot(
                agg["x"],
                agg["mean"],
                label=f"{cfg.label(algo)} (n={agg['n']})",
                color=color,
                linewidth=1.8,
            )
            ax.fill_between(
                agg["x"],
                agg["mean"] - agg["band"],
                agg["mean"] + agg["band"],
                color=color,
                alpha=0.25,
                linewidth=0,
            )
            plotted = True
        ax.set_title(task)
        ax.set_xlabel("samples")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, loc="best")

    for j in range(len(tasks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if cfg.title:
        fig.suptitle(cfg.title, fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, cfg.figures_dir, name, cfg)


def _final_performance(result: Dict[str, Any], metric: str = "return") -> float:
    """Extract the final performance value from a result payload."""
    final_eval = result.get("final_eval")
    if isinstance(final_eval, dict):
        for key in (
            f"mean_{metric}",
            "mean_return",
            "return",
            "eval_return",
            "mean_reward",
        ):
            if key in final_eval and final_eval[key] is not None:
                try:
                    return float(final_eval[key])
                except (TypeError, ValueError):
                    pass
    xs, ys = _extract_curve(result, metric)
    if ys.size:
        return float(ys[-1])
    return float("nan")


def plot_final_performance(
    results: Sequence[Dict[str, Any]],
    cfg: Optional[PlotConfig] = None,
    metric: str = "return",
    name: str = "table1_final_performance",
) -> List[str]:
    """Reproduce Table 1 as a grouped bar chart of final performance."""
    cfg = cfg or PlotConfig()
    if not _HAS_MPL:
        print("[plot_results] matplotlib unavailable; skipping final performance")
        return []

    grouped = _group_results(results)
    tasks = [t for t in cfg.tasks if any(k[0] == t for k in grouped)]
    if not tasks:
        tasks = sorted({k[0] for k in grouped})
    if not tasks:
        print("[plot_results] no results found for final performance")
        return []

    algos = [a for a in cfg.algos if any((t, a) in grouped for t in tasks)]
    if not algos:
        algos = sorted({k[1] for k in grouped})

    n_tasks = len(tasks)
    n_algos = len(algos)
    width = 0.8 / max(n_algos, 1)
    x = np.arange(n_tasks, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * n_tasks), 4.2))
    for ai, algo in enumerate(algos):
        means, errs = [], []
        for task in tasks:
            seeds = grouped.get((task, algo), [])
            vals = np.asarray(
                [_final_performance(r, metric) for r in seeds], dtype=np.float64
            )
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                means.append(0.0)
                errs.append(0.0)
                continue
            means.append(float(vals.mean()))
            if vals.size > 1:
                sem = float(vals.std(ddof=1) / np.sqrt(vals.size))
                errs.append(2.0 * sem if cfg.use_standard_error else float(vals.std()))
            else:
                errs.append(0.0)
        ax.bar(
            x + ai * width,
            means,
            width,
            yerr=errs,
            capsize=3,
            label=cfg.label(algo),
            color=cfg.color(algo),
            alpha=0.85,
        )

    ax.set_xticks(x + width * (n_algos - 1) / 2.0)
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel(f"final {metric}")
    ax.set_title(cfg.title or "Final performance (Table 1)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_figure(fig, cfg.figures_dir, name, cfg)


def plot_ppo_saturation(
    results: Sequence[Dict[str, Any]],
    cfg: Optional[PlotConfig] = None,
    metric: str = "return",
    name: str = "figure2_ppo_saturation",
) -> List[str]:
    """Reproduce Figure 2: PPO performance vs. batch size (num_envs)."""
    cfg = cfg or PlotConfig()
    if not _HAS_MPL:
        return []

    points: Dict[str, Dict[int, List[float]]] = {}
    for res in results:
        if _infer_algo(res) != "ppo":
            continue
        task = _infer_task(res)
        num_envs = res.get("num_envs")
        if num_envs is None:
            continue
        points.setdefault(task, {}).setdefault(int(num_envs), []).append(
            _final_performance(res, metric)
        )
    if not points:
        print("[plot_results] no PPO saturation data found")
        return []

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for task, by_env in sorted(points.items()):
        xs = sorted(by_env.keys())
        ys = [float(np.nanmean(by_env[n])) for n in xs]
        errs = [
            2.0 * float(np.nanstd(by_env[n]) / max(np.sqrt(len(by_env[n])), 1.0))
            for n in xs
        ]
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=task)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("num_envs (batch size)")
    ax.set_ylabel(f"final {metric}")
    ax.set_title(cfg.title or "PPO saturation with batch size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_figure(fig, cfg.figures_dir, name, cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def plot_all(cfg: Optional[PlotConfig] = None) -> Dict[str, List[str]]:
    """Load results and generate every figure.

    Returns a mapping ``figure_name -> list of written file paths``.
    """
    cfg = cfg or PlotConfig()
    results = load_results(cfg.results_dir)
    print(f"[plot_results] loaded {len(results)} result file(s) from {cfg.results_dir}")
    outputs: Dict[str, List[str]] = {}
    outputs["learning_curves"] = plot_learning_curves(results, cfg)
    outputs["ablations"] = plot_ablations(results, cfg)
    outputs["final_performance"] = plot_final_performance(results, cfg)
    outputs["ppo_saturation"] = plot_ppo_saturation(results, cfg)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """CLI entry point for standalone plotting."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot SAPG experiment results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--smoothing", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--std", action="store_true", help="shade std instead of 2*SEM")
    args = parser.parse_args(argv)

    cfg = PlotConfig(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        smoothing=args.smoothing,
        dpi=args.dpi,
        formats=args.formats,
        use_standard_error=not args.std,
    )
    return plot_all(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
