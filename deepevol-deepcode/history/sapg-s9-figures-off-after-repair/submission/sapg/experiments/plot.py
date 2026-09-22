"""Plotting / table-generation utilities for the SAPG paper reproduction.

This module consumes the artifacts produced by ``experiments/train.py``
(``summary.json`` and ``curves.npz``) and reproduces:

  * Fig. 2  -- PPO batch-size saturation (asymptotic performance vs. #envs).
  * Fig. 5  -- Learning curves for SAPG vs. baselines on the 5 tasks.
  * Fig. 6  -- Ablation curves (symmetric / high-off-policy / no-off-policy /
               entropy coefficient sweep).
  * Table 1 -- Final performance table (mean +/- std error) after ~2e10
               samples, compared against the paper's reported numbers.

The module is intentionally dependency-light: ``numpy`` and ``matplotlib``
are required, everything else is stdlib.  All functions degrade gracefully
when the expected input files are missing so that the plotting pipeline can
be smoke-tested without a full training run.

Usage
-----
    python -m experiments.plot --runs runs --output figures
    python -m experiments.plot --table-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root injection so the module can be executed as a script.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:  # pragma: no cover - matplotlib is optional at import time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS: Tuple[str, ...] = (
    "regrasping",
    "throw",
    "reorientation",
    "shadowhand",
    "allegrohand",
)

TASK_TITLES: Dict[str, str] = {
    "regrasping": "Regrasping",
    "throw": "Throw",
    "reorientation": "Reorientation",
    "shadowhand": "ShadowHand",
    "allegrohand": "AllegroHand",
}

#: Metric reported per task (Table 1).
TASK_METRIC: Dict[str, str] = {
    "regrasping": "successes",
    "throw": "successes",
    "reorientation": "successes",
    "shadowhand": "episode_reward",
    "allegrohand": "episode_reward",
}

ALGORITHMS: Tuple[str, ...] = ("sapg", "ppo", "pql", "dexpbt")

ALGO_LABELS: Dict[str, str] = {
    "sapg": "SAPG",
    "ppo": "PPO",
    "pql": "PQL",
    "dexpbt": "DexPBT",
}

ALGO_COLORS: Dict[str, str] = {
    "sapg": "#1f77b4",
    "ppo": "#d62728",
    "pql": "#2ca02c",
    "dexpbt": "#ff7f0e",
}

#: Ablation variants (Fig. 6).
ABLATION_VARIANTS: Tuple[str, ...] = (
    "leader_follower",
    "symmetric",
    "high_off_policy_ratio",
    "no_off_policy",
)

ABLATION_LABELS: Dict[str, str] = {
    "leader_follower": "SAPG (leader-follower)",
    "symmetric": "Symmetric aggregation",
    "high_off_policy_ratio": "High off-policy ratio",
    "no_off_policy": "No off-policy (indep. PPO)",
}

ABLATION_COLORS: Dict[str, str] = {
    "leader_follower": "#1f77b4",
    "symmetric": "#9467bd",
    "high_off_policy_ratio": "#8c564b",
    "no_off_policy": "#7f7f7f",
}

#: Entropy coefficient sweep values (Sec 4.5).
ENTROPY_COEFS: Tuple[float, ...] = (0.0, 0.003, 0.005)

#: PPO batch sizes used for the saturation study (Fig. 2).
PPO_BATCH_SIZES: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@dataclass
class Curve:
    """A single aggregated learning curve (mean +/- stderr over seeds)."""

    x: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray
    n: int = 1

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.stderr = np.asarray(self.stderr, dtype=np.float64)

    def __len__(self) -> int:
        return int(self.x.shape[0])


@dataclass
class RunRecord:
    """Metadata for one (task, algorithm, seed) run."""

    task: str
    algorithm: str
    seed: int
    run_dir: str
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def metric(self) -> str:
        return TASK_METRIC.get(self.task, "episode_reward")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def load_summary(runs_dir: str) -> Optional[Dict[str, Any]]:
    """Load ``summary.json`` produced by ``experiments/train.py``."""
    return _load_json(os.path.join(runs_dir, "summary.json"))


def load_curves(runs_dir: str) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Load ``curves.npz`` produced by ``experiments/train.py``.

    Returns a nested mapping ``task -> algorithm -> {x, mean, stderr, n}``.
    """
    path = os.path.join(runs_dir, "curves.npz")
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key in data.files:
        # Keys are stored as "<task>__<algorithm>__<field>".
        parts = key.split("__")
        if len(parts) != 3:
            continue
        task, algo, fld = parts
        out.setdefault(task, {}).setdefault(algo, {})[fld] = data[key]
    return out


def load_run_histories(runs_dir: str) -> List[RunRecord]:
    """Walk ``runs_dir`` and load every ``*_history.json`` file.

    The expected layout is ``<runs_dir>/<task>/<algorithm>/seed_<s>/...`` but
    the loader is tolerant of flatter layouts: it infers task/algorithm/seed
    from the directory names when possible.
    """
    records: List[RunRecord] = []
    if not os.path.isdir(runs_dir):
        return records

    for root, _dirs, files in os.walk(runs_dir):
        for fname in files:
            if not fname.endswith("_history.json"):
                continue
            path = os.path.join(root, fname)
            payload = _load_json(path)
            if payload is None:
                continue
            history = payload.get("history", payload) if isinstance(payload, dict) else payload
            if not isinstance(history, list):
                continue

            rel = os.path.relpath(root, runs_dir)
            parts = [p for p in rel.split(os.sep) if p not in (".", "")]
            task = parts[0] if len(parts) >= 1 else "unknown"
            algorithm = parts[1] if len(parts) >= 2 else "sapg"
            seed = 0
            for part in parts[2:]:
                if part.startswith("seed"):
                    try:
                        seed = int(part.replace("seed_", "").replace("seed", ""))
                    except ValueError:
                        seed = 0
            records.append(
                RunRecord(
                    task=task,
                    algorithm=algorithm,
                    seed=seed,
                    run_dir=root,
                    history=history,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _extract_series(
    history: Sequence[Dict[str, Any]],
    metric: str,
    x_key: str = "total_transitions",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) arrays from a run history."""
    xs: List[float] = []
    ys: List[float] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        x = entry.get(x_key)
        y = entry.get(metric)
        if x is None or y is None:
            continue
        try:
            xs.append(float(x))
            ys.append(float(y))
        except (TypeError, ValueError):
            continue
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def interpolate_curve(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate ``y(x)`` onto ``grid`` (flat extrapolation)."""
    if x.size == 0 or y.size == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    if x.size == 1:
        return np.full_like(grid, float(y[0]), dtype=np.float64)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Deduplicate x to keep np.interp well-defined.
    uniq_x, uniq_idx = np.unique(x_sorted, return_index=True)
    uniq_y = y_sorted[uniq_idx]
    if uniq_x.size == 1:
        return np.full_like(grid, float(uniq_y[0]), dtype=np.float64)
    return np.interp(grid, uniq_x, uniq_y)


def aggregate_records(
    records: Sequence[RunRecord],
    num_points: int = 100,
    x_key: str = "total_transitions",
) -> Dict[str, Dict[str, Curve]]:
    """Aggregate per-seed histories into mean/stderr curves.

    Returns ``task -> algorithm -> Curve``.
    """
    grouped: Dict[str, Dict[str, List[RunRecord]]] = {}
    for rec in records:
        grouped.setdefault(rec.task, {}).setdefault(rec.algorithm, []).append(rec)

    out: Dict[str, Dict[str, Curve]] = {}
    for task, algos in grouped.items():
        metric = TASK_METRIC.get(task, "episode_reward")
        out[task] = {}
        for algo, recs in algos.items():
            series: List[Tuple[np.ndarray, np.ndarray]] = []
            max_x = 0.0
            for rec in recs:
                x, y = _extract_series(rec.history, metric, x_key=x_key)
                if x.size == 0:
                    continue
                series.append((x, y))
                max_x = max(max_x, float(x.max()))
            if not series:
                continue
            grid = np.linspace(0.0, max_x if max_x > 0 else 1.0, num_points)
            stacked = np.stack([interpolate_curve(x, y, grid) for x, y in series], axis=0)
            mean = np.nanmean(stacked, axis=0)
            n = stacked.shape[0]
            if n > 1:
                stderr = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(n)
            else:
                stderr = np.zeros_like(mean)
            out[task][algo] = Curve(x=grid, mean=mean, stderr=stderr, n=n)
    return out


def curves_from_npz(
    curves: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, Curve]]:
    """Convert the ``curves.npz`` mapping into ``Curve`` objects."""
    out: Dict[str, Dict[str, Curve]] = {}
    for task, algos in curves.items():
        out[task] = {}
        for algo, fields in algos.items():
            if "x" not in fields or "mean" not in fields:
                continue
            stderr = fields.get("stderr")
            if stderr is None:
                stderr = np.zeros_like(fields["mean"])
            n = int(np.asarray(fields.get("n", 1)).reshape(-1)[0]) if "n" in fields else 1
            out[task][algo] = Curve(
                x=fields["x"], mean=fields["mean"], stderr=stderr, n=n
            )
    return out


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------


def build_table(
    curves: Dict[str, Dict[str, Curve]],
    expected: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Build the final-performance table (mean, stderr) per task/algorithm."""
    table: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for task in TASKS:
        table[task] = {}
        task_curves = curves.get(task, {})
        for algo in ALGORITHMS:
            curve = task_curves.get(algo)
            if curve is None or len(curve) == 0:
                continue
            table[task][algo] = (float(curve.mean[-1]), float(curve.stderr[-1]))
    return table


def format_table(
    table: Dict[str, Dict[str, Tuple[float, float]]],
    expected: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the final-performance table as a plain-text string."""
    lines: List[str] = []
    header = f"{'Task':<14}" + "".join(f"{ALGO_LABELS.get(a, a):>18}" for a in ALGORITHMS)
    lines.append(header)
    lines.append("-" * len(header))

    for task in TASKS:
        row = f"{TASK_TITLES.get(task, task):<14}"
        for algo in ALGORITHMS:
            entry = table.get(task, {}).get(algo)
            if entry is None:
                row += f"{'--':>18}"
            else:
                mean, stderr = entry
                row += f"{_fmt_value(mean, stderr):>18}"
        lines.append(row)

    if expected:
        lines.append("")
        lines.append("Paper-reported values (Table 1):")
        for key, values in expected.items():
            lines.append(f"  {key}:")
            for task in TASKS:
                if task in values:
                    mean, std = values[task]
                    lines.append(f"    {TASK_TITLES.get(task, task):<14} {mean:>12.4g} +/- {std:<10.4g}")
    return "\n".join(lines)


def _fmt_value(mean: float, stderr: float) -> str:
    if abs(mean) >= 1e4:
        return f"{mean:.3e}+/-{stderr:.2e}"
    return f"{mean:.2f}+/-{stderr:.2f}"


def write_table(
    table: Dict[str, Dict[str, Tuple[float, float]]],
    output_path: str,
    expected: Optional[Dict[str, Any]] = None,
) -> str:
    """Write the formatted table to ``output_path`` (creating dirs)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    text = format_table(table, expected=expected)
    with open(output_path, "w") as fh:
        fh.write(text + "\n")
    return text


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _require_matplotlib() -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        )


def _plot_curve(
    ax: Any,
    curve: Curve,
    label: str,
    color: Optional[str] = None,
    linestyle: str = "-",
    alpha: float = 0.25,
) -> None:
    """Plot a mean curve with a shaded +/- stderr band."""
    ax.plot(curve.x, curve.mean, label=label, color=color, linestyle=linestyle, linewidth=1.8)
    if curve.stderr is not None and np.any(curve.stderr > 0):
        ax.fill_between(
            curve.x,
            curve.mean - curve.stderr,
            curve.mean + curve.stderr,
            color=color,
            alpha=alpha,
            linewidth=0.0,
        )


def plot_learning_curves(
    curves: Dict[str, Dict[str, Curve]],
    output_path: str,
    tasks: Sequence[str] = TASKS,
    algorithms: Sequence[str] = ALGORITHMS,
    title: str = "SAPG vs. baselines",
    xlabel: str = "Environment transitions",
) -> Optional[str]:
    """Reproduce Fig. 5: learning curves for SAPG vs. baselines."""
    _require_matplotlib()
    tasks = [t for t in tasks if t in curves]
    if not tasks:
        return None

    ncols = min(3, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        for algo in algorithms:
            curve = curves.get(task, {}).get(algo)
            if curve is None:
                continue
            _plot_curve(ax, curve, ALGO_LABELS.get(algo, algo), ALGO_COLORS.get(algo))
        ax.set_title(TASK_TITLES.get(task, task))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(TASK_METRIC.get(task, "episode_reward"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_ppo_saturation(
    saturation: Dict[str, Dict[str, float]],
    output_path: str,
    tasks: Sequence[str] = TASKS,
    title: str = "PPO batch-size saturation (Fig. 2)",
) -> Optional[str]:
    """Reproduce Fig. 2: asymptotic PPO performance vs. number of envs.

    ``saturation`` maps ``task -> {str(batch_size): final_metric}``.
    """
    _require_matplotlib()
    tasks = [t for t in tasks if t in saturation]
    if not tasks:
        return None

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), squeeze=False)
    for idx, task in enumerate(tasks):
        ax = axes[0][idx]
        points = saturation[task]
        xs: List[int] = []
        ys: List[float] = []
        for key, value in points.items():
            try:
                xs.append(int(float(key)))
                ys.append(float(value))
            except (TypeError, ValueError):
                continue
        order = np.argsort(xs)
        xs_arr = np.asarray(xs, dtype=np.float64)[order]
        ys_arr = np.asarray(ys, dtype=np.float64)[order]
        ax.plot(xs_arr, ys_arr, marker="o", color=ALGO_COLORS["ppo"], linewidth=1.8)
        ax.set_xscale("log", base=2)
        ax.set_title(TASK_TITLES.get(task, task))
        ax.set_xlabel("Number of parallel envs")
        ax.set_ylabel(TASK_METRIC.get(task, "episode_reward"))
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_ablations(
    ablation_curves: Dict[str, Dict[str, Curve]],
    output_path: str,
    tasks: Sequence[str] = TASKS,
    variants: Sequence[str] = ABLATION_VARIANTS,
    title: str = "SAPG ablations (Fig. 6)",
) -> Optional[str]:
    """Reproduce Fig. 6: aggregation-variant ablation curves."""
    _require_matplotlib()
    tasks = [t for t in tasks if t in ablation_curves]
    if not tasks:
        return None

    ncols = min(3, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        for variant in variants:
            curve = ablation_curves.get(task, {}).get(variant)
            if curve is None:
                continue
            _plot_curve(
                ax,
                curve,
                ABLATION_LABELS.get(variant, variant),
                ABLATION_COLORS.get(variant),
            )
        ax.set_title(TASK_TITLES.get(task, task))
        ax.set_xlabel("Environment transitions")
        ax.set_ylabel(TASK_METRIC.get(task, "episode_reward"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_entropy_sweep(
    entropy_curves: Dict[str, Dict[str, Curve]],
    output_path: str,
    tasks: Sequence[str] = TASKS,
    coefs: Sequence[float] = ENTROPY_COEFS,
    title: str = "Entropy coefficient sweep (Sec 4.5)",
) -> Optional[str]:
    """Plot the entropy-coefficient sweep (sigma in {0, 0.003, 0.005})."""
    _require_matplotlib()
    tasks = [t for t in tasks if t in entropy_curves]
    if not tasks:
        return None

    cmap = plt.get_cmap("viridis")
    ncols = min(3, len(tasks))
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        for k, coef in enumerate(coefs):
            key = f"sigma_{coef}"
            curve = entropy_curves.get(task, {}).get(key)
            if curve is None:
                continue
            color = cmap(k / max(1, len(coefs) - 1))
            _plot_curve(ax, curve, f"sigma={coef}", color)
        ax.set_title(TASK_TITLES.get(task, task))
        ax.set_xlabel("Environment transitions")
        ax.set_ylabel(TASK_METRIC.get(task, "episode_reward"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(tasks), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_all_figures(
    runs_dir: str,
    output_dir: str,
    expected: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Generate every figure/table from the artifacts in ``runs_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    produced: Dict[str, Optional[str]] = {}

    curves = load_curves(runs_dir)
    if curves is not None:
        curve_objs = curves_from_npz(curves)
    else:
        records = load_run_histories(runs_dir)
        curve_objs = aggregate_records(records) if records else {}

    # Table 1
    table = build_table(curve_objs, expected=expected)
    table_path = os.path.join(output_dir, "table1.txt")
    write_table(table, table_path, expected=expected)
    produced["table1"] = table_path

    # Fig. 5
    produced["fig5"] = plot_learning_curves(
        curve_objs, os.path.join(output_dir, "fig5_learning_curves.png")
    )

    # Fig. 6 (ablation variants live under the same task keys)
    ablation_curves = {
        task: {k: v for k, v in algos.items() if k in ABLATION_VARIANTS}
        for task, algos in curve_objs.items()
    }
    produced["fig6"] = plot_ablations(
        ablation_curves, os.path.join(output_dir, "fig6_ablations.png")
    )

    # Entropy sweep
    entropy_curves = {
        task: {k: v for k, v in algos.items() if k.startswith("sigma_")}
        for task, algos in curve_objs.items()
    }
    produced["entropy"] = plot_entropy_sweep(
        entropy_curves, os.path.join(output_dir, "entropy_sweep.png")
    )

    # Fig. 2 (saturation) -- read from summary.json if present.
    summary = load_summary(runs_dir)
    saturation = None
    if summary is not None:
        saturation = summary.get("ppo_saturation")
    if saturation:
        produced["fig2"] = plot_ppo_saturation(
            saturation, os.path.join(output_dir, "fig2_ppo_saturation.png")
        )
    else:
        produced["fig2"] = None

    return produced


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot SAPG results (Figs. 2/5/6, Table 1).")
    parser.add_argument("--runs", type=str, default="runs", help="Directory with run artifacts.")
    parser.add_argument("--output", type=str, default="figures", help="Output directory.")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only produce Table 1 (skip figures).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    expected = None
    try:
        from sapg.config import EXPECTED_RESULTS  # type: ignore

        expected = EXPECTED_RESULTS
    except Exception:
        expected = None

    if args.table_only:
        curves = load_curves(args.runs)
        if curves is not None:
            curve_objs = curves_from_npz(curves)
        else:
            records = load_run_histories(args.runs)
            curve_objs = aggregate_records(records) if records else {}
        table = build_table(curve_objs, expected=expected)
        text = write_table(table, os.path.join(args.output, "table1.txt"), expected=expected)
        print(text)
        return 0

    produced = make_all_figures(args.runs, args.output, expected=expected)
    for name, path in produced.items():
        print(f"{name}: {path if path else '(skipped - no data)'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
