"""Figure-generation script for the NPSE reproduction project.

This script loads benchmark result JSON files produced by the training scripts
(``train_npse_benchmarks.py``, ``train_tsnpse_benchmarks.py``,
``run_snpse_variants.py``, ``run_nlse_comparison.py``, and ``run_pyloric.py``)
and produces publication-style summary figures:

* ``figure2_npse_benchmarks``   -- non-sequential NPSE C2ST/MMD across budgets
* ``figure3_tsnpse_benchmarks`` -- truncated sequential NPSE across budgets
* ``figure5_npse_vs_nlse``      -- NPSE vs NLSE comparison
* ``figure6_snpse_variants``    -- TSNPSE vs SNPSE-A/B/C ablation
* ``figure4_pyloric``           -- pyloric-network application marginals and
                                   posterior predictive summaries

All plotting is intentionally defensive: missing JSON files or missing fields
are skipped with a warning rather than aborting, so a partially completed
experiment can still produce whatever figures are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # noqa: F401  (non-interactive backend)
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - environment-dependent
    plt = None  # type: ignore
    _MPL_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures"

ALL_BENCHMARKS = [
    "gaussian_linear",
    "gaussian_mixture",
    "two_moons",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "slcp",
    "sir",
    "lotka_volterra",
]

BENCHMARK_LABELS = {
    "gaussian_linear": "Gaussian Linear",
    "gaussian_mixture": "Gaussian Mixture",
    "two_moons": "Two Moons",
    "gaussian_linear_uniform": "Gaussian Linear Uniform",
    "bernoulli_glm": "Bernoulli GLM",
    "slcp": "SLCP",
    "sir": "SIR",
    "lotka_volterra": "Lotka-Volterra",
}


# ---------------------------------------------------------------------------
# Result loading and normalization
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Load a JSON file, returning ``None`` if it is missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def discover_json_files(results_dir: Path) -> List[Path]:
    """Return all JSON files under ``results_dir``, sorted by path."""
    if not results_dir.exists():
        return []
    return sorted([p for p in results_dir.rglob("*.json") if p.is_file()])


def normalize_to_list(data: Any) -> List[Dict[str, Any]]:
    """Convert a variety of JSON payload layouts to a flat list of dicts.

    Accepted layouts:

    * a list of result dicts,
    * a dict with a ``"results"`` list,
    * a dict keyed by benchmark name, each value a dict keyed by budget,
    * a dict with a single result dict (wrapped),
    * a dict already containing a single result record.

    Returns an empty list for unrecognized/missing data.
    """
    if data is None:
        return []

    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]

    if not isinstance(data, dict):
        return []

    # Common wrapper: {"results": [...]}
    if "results" in data and isinstance(data["results"], list):
        return normalize_to_list(data["results"])

    # A single result record.
    if any(key in data for key in ("benchmark", "method", "c2st", "mmd", "budget")):
        return [data]

    # Nested benchmark -> budget/result mapping.
    flattened: List[Dict[str, Any]] = []
    for benchmark, benchmark_value in data.items():
        if not isinstance(benchmark_value, dict):
            continue
        if any(key in benchmark_value for key in ("c2st", "mmd", "budget", "status")):
            record = dict(benchmark_value)
            if "benchmark" not in record:
                record["benchmark"] = benchmark
            flattened.append(record)
            continue
        for budget, result in benchmark_value.items():
            if isinstance(result, dict):
                record = dict(result)
                if "benchmark" not in record:
                    record["benchmark"] = benchmark
                if "budget" not in record:
                    record["budget"] = budget
                flattened.append(record)
    return flattened


def _first_value(record: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record:
            value = record[key]
            if value is not None:
                return value
    return default


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def extract_rows(
    data: Any,
    method_key: str = "method",
    benchmark_key: str = "benchmark",
    budget_key: str = "budget",
    c2st_keys: Sequence[str] = ("c2st", "c2st_score", "c2st_accuracy"),
    mmd_keys: Sequence[str] = ("mmd", "mmd_score", "mmd2"),
) -> List[Dict[str, Any]]:
    """Normalize loaded JSON data into canonical result rows."""
    rows: List[Dict[str, Any]] = []
    for record in normalize_to_list(data):
        method = _first_value(record, (method_key,))
        benchmark = _first_value(record, (benchmark_key,))
        budget = _first_value(record, (budget_key,))
        c2st = _to_float(_first_value(record, c2st_keys))
        mmd = _to_float(_first_value(record, mmd_keys))
        status = _first_value(record, ("status",), default="ok")
        if benchmark is None and method is None and c2st is None and mmd is None:
            continue
        rows.append(
            {
                "method": str(method) if method is not None else "unknown",
                "benchmark": str(benchmark) if benchmark is not None else "unknown",
                "budget": budget,
                "c2st": c2st,
                "mmd": mmd,
                "status": str(status),
            }
        )
    return rows


def load_rows_from_file(path: Path, **kwargs: Any) -> List[Dict[str, Any]]:
    """Load and normalize a single result JSON file."""
    return extract_rows(load_json(path), **kwargs)


def aggregate_by(
    rows: Iterable[Dict[str, Any]], keys: Sequence[str]
) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def mean_std(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None, None
    vals_arr = np.asarray(vals, dtype=float)
    if len(vals_arr) == 1:
        return float(vals_arr[0]), 0.0
    return float(np.mean(vals_arr)), float(np.std(vals_arr))


def summary_by_benchmark(
    rows: List[Dict[str, Any]],
    metric: str = "c2st",
    budget_key: str = "budget",
) -> Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """Return ``{benchmark: {budget_label: (mean, std)}}`` for one metric."""
    out: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    for benchmark, group in aggregate_by(rows, ("benchmark",)).items():
        by_budget: Dict[str, List[float]] = {}
        for row in group:
            if row.get("status") == "error":
                continue
            value = row.get(metric)
            if value is None:
                continue
            budget = row.get(budget_key)
            label = str(budget)
            by_budget.setdefault(label, []).append(float(value))
        out[str(benchmark)] = {
            label: mean_std(values) for label, values in by_budget.items()
        }
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _require_mpl() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to generate figures. "
            f"Import failed with: {_MPL_IMPORT_ERROR}"
        )


def _save(fig: Any, figure_dir: Path, name: str, formats: Sequence[str] = ("pdf", "png")) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(figure_dir / f"{name}.{fmt}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_budget_lines(
    summary: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]],
    title: str,
    ylabel: str,
    figure_dir: Path,
    filename: str,
    benchmark_order: Sequence[str] = ALL_BENCHMARKS,
    baseline: float = 0.5,
) -> None:
    """Plot per-benchmark metric-vs-budget lines with error bars.

    ``summary`` has the layout produced by :func:`summary_by_benchmark`.
    """
    _require_mpl()
    benchmarks = [b for b in benchmark_order if b in summary and summary[b]]
    if not benchmarks:
        print(f"[make_figures] No data for {filename}; skipping.", file=sys.stderr)
        return

    n_cols = min(4, len(benchmarks))
    n_rows = int(np.ceil(len(benchmarks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # Build a global budget ordering for consistent x-axis.
    all_budget_labels: List[str] = []
    for bench_data in summary.values():
        for label in bench_data:
            if label not in all_budget_labels:
                all_budget_labels.append(label)
    all_budget_labels.sort(key=_budget_sort_key)

    for ax, benchmark in zip(axes, benchmarks):
        bench_data = summary[benchmark]
        xs: List[float] = []
        means: List[Optional[float]] = []
        stds: List[Optional[float]] = []
        for label in all_budget_labels:
            if label not in bench_data:
                continue
            mean, std = bench_data[label]
            if mean is None:
                continue
            xs.append(_budget_sort_key(label))
            means.append(mean)
            stds.append(std if std is not None else 0.0)

        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            linewidth=1.6,
            markersize=4.5,
        )
        if baseline is not None:
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(BENCHMARK_LABELS.get(benchmark, benchmark), fontsize=10)
        ax.set_xscale("log" if len(xs) > 2 and all(x > 0 for x in xs) else "linear")
        ax.set_xlabel("Simulation budget")
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=8)

    for ax in axes[len(benchmarks):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, figure_dir, filename)


def _budget_sort_key(label: Any) -> float:
    try:
        return float(label)
    except (TypeError, ValueError):
        try:
            return float(str(label).replace("_", ""))
        except (TypeError, ValueError):
            return float("inf")


# ---------------------------------------------------------------------------
# Figure 2: non-sequential NPSE benchmark sweep
# ---------------------------------------------------------------------------
def make_figure2(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Generate C2ST and MMD budget-sweep figures for non-sequential NPSE."""
    json_files = [p for p in discover_json_files(results_dir) if "npse" in p.name]
    rows: List[Dict[str, Any]] = []
    for path in json_files:
        rows.extend(load_rows_from_file(path))
    if not rows:
        print("[make_figures] No NPSE result files found.", file=sys.stderr)
        return

    c2st_summary = summary_by_benchmark(rows, metric="c2st")
    mmd_summary = summary_by_benchmark(rows, metric="mmd")

    _plot_budget_lines(
        c2st_summary,
        "NPSE (non-sequential): C2ST vs simulation budget",
        "C2ST accuracy",
        figure_dir,
        "figure2_npse_c2st",
    )
    _plot_budget_lines(
        mmd_summary,
        "NPSE (non-sequential): MMD vs simulation budget",
        "MMD$^2$",
        figure_dir,
        "figure2_npse_mmd",
        baseline=None,
    )


# ---------------------------------------------------------------------------
# Figure 3: TSNPSE benchmark sweep
# ---------------------------------------------------------------------------
def make_figure3(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Generate C2ST and MMD budget-sweep figures for TSNPSE."""
    json_files = [p for p in discover_json_files(results_dir) if "tsnpse" in p.name]
    rows: List[Dict[str, Any]] = []
    for path in json_files:
        rows.extend(load_rows_from_file(path))
    if not rows:
        print("[make_figures] No TSNPSE result files found.", file=sys.stderr)
        return

    _plot_budget_lines(
        summary_by_benchmark(rows, metric="c2st"),
        "TSNPSE: C2ST vs simulation budget",
        "C2ST accuracy",
        figure_dir,
        "figure3_tsnpse_c2st",
    )
    _plot_budget_lines(
        summary_by_benchmark(rows, metric="mmd"),
        "TSNPSE: MMD vs simulation budget",
        "MMD$^2$",
        figure_dir,
        "figure3_tsnpse_mmd",
        baseline=None,
    )


# ---------------------------------------------------------------------------
# Figure 5: NPSE vs NLSE comparison
# ---------------------------------------------------------------------------
def make_figure5(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Generate a grouped bar chart comparing NPSE and NLSE C2ST."""
    all_rows: List[Dict[str, Any]] = []
    for path in discover_json_files(results_dir):
        name = path.name.lower()
        if "nlse" in name or "npse" in name:
            all_rows.extend(load_rows_from_file(path))

    if not all_rows:
        print("[make_figures] No NPSE/NLSE comparison data found.", file=sys.stderr)
        return

    methods = sorted({str(r.get("method")) for r in all_rows})
    benchmarks = sorted({str(r.get("benchmark")) for r in all_rows})
    if not methods or not benchmarks:
        return

    _require_mpl()
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in all_rows:
        if row.get("status") == "error":
            continue
        if row.get("c2st") is None:
            continue
        grouped.setdefault((str(row["method"]), str(row["benchmark"])), []).append(
            float(row["c2st"])
        )

    methods = [m for m in methods if any((m, b) in grouped for b in benchmarks)]
    benchmarks = [b for b in benchmarks if any((m, b) in grouped for m in methods)]
    if not methods or not benchmarks:
        return

    x = np.arange(len(benchmarks))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(benchmarks)), 4.5))

    for i, method in enumerate(methods):
        means = []
        stds = []
        for benchmark in benchmarks:
            values = grouped.get((method, benchmark), [])
            mean, std = mean_std(values)
            means.append(mean if mean is not None else np.nan)
            stds.append(std if std is not None else 0.0)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=method)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABELS.get(b, b) for b in benchmarks], rotation=30, ha="right")
    ax.set_ylabel("C2ST accuracy")
    ax.set_title("NPSE vs NLSE")
    ax.legend()
    fig.tight_layout()
    _save(fig, figure_dir, "figure5_npse_vs_nlse")


# ---------------------------------------------------------------------------
# Figure 6: SNPSE variants ablation
# ---------------------------------------------------------------------------
def make_figure6(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Generate grouped bars comparing TSNPSE and SNPSE-A/B/C."""
    all_rows: List[Dict[str, Any]] = []
    for path in discover_json_files(results_dir):
        name = path.name.lower()
        if "snpse" in name or "tsnpse" in name:
            all_rows.extend(load_rows_from_file(path))

    if not all_rows:
        print("[make_figures] No SNPSE/TSNPSE variant data found.", file=sys.stderr)
        return

    methods = sorted({str(r.get("method")) for r in all_rows})
    benchmarks = sorted({str(r.get("benchmark")) for r in all_rows})
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in all_rows:
        if row.get("status") == "error":
            continue
        if row.get("c2st") is None:
            continue
        grouped.setdefault((str(row["method"]), str(row["benchmark"])), []).append(
            float(row["c2st"])
        )

    methods = [m for m in methods if any((m, b) in grouped for b in benchmarks)]
    benchmarks = [b for b in benchmarks if any((m, b) in grouped for m in methods)]
    if not methods or not benchmarks:
        return

    _require_mpl()
    x = np.arange(len(benchmarks))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(benchmarks)), 4.5))

    for i, method in enumerate(methods):
        means = []
        stds = []
        for benchmark in benchmarks:
            values = grouped.get((method, benchmark), [])
            mean, std = mean_std(values)
            means.append(mean if mean is not None else np.nan)
            stds.append(std if std is not None else 0.0)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=method)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABELS.get(b, b) for b in benchmarks], rotation=30, ha="right")
    ax.set_ylabel("C2ST accuracy")
    ax.set_title("TSNPSE vs SNPSE variants")
    ax.legend()
    fig.tight_layout()
    _save(fig, figure_dir, "figure6_snpse_variants")


# ---------------------------------------------------------------------------
# Figure 4: pyloric-network application
# ---------------------------------------------------------------------------
def _plot_marginals_from_arrays(
    posterior: np.ndarray,
    names: Optional[Sequence[str]] = None,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Plot per-dimension marginal histograms of posterior samples."""
    _require_mpl()
    posterior = np.asarray(posterior)
    dim = posterior.shape[1]
    n_cols = min(6, dim)
    n_rows = int(np.ceil(dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.0 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(dim):
        ax = axes[i]
        ax.hist(posterior[:, i], bins=40, density=True, alpha=0.8, color="tab:blue")
        if names is not None and i < len(names):
            ax.set_title(str(names[i]), fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes[dim:]:
        ax.axis("off")

    fig.suptitle("Pyloric-network approximate posterior marginals", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, figure_dir, "figure4_pyloric_marginals")


def make_figure4(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> None:
    """Generate pyloric-experiment figures from saved posterior/predictive arrays."""
    pyloric_json: Optional[Path] = None
    for path in discover_json_files(results_dir):
        if "pyloric" in path.name.lower():
            pyloric_json = path
            break

    if pyloric_json is None:
        print("[make_figures] No pyloric result file found.", file=sys.stderr)
        return

    data = load_json(pyloric_json)
    if not isinstance(data, dict):
        return

    # Posterior marginals. The run_pyloric script can save either an array or a
    # path to an .npy file under "posterior_samples".
    posterior = None
    raw_posterior = data.get("posterior_samples") or data.get("posterior")
    if isinstance(raw_posterior, list):
        try:
            posterior = np.asarray(raw_posterior, dtype=float)
        except (TypeError, ValueError):
            posterior = None
    elif isinstance(raw_posterior, str):
        candidate = Path(raw_posterior)
        if not candidate.is_absolute():
            candidate = pyloric_json.parent / candidate
        if candidate.exists():
            try:
                posterior = np.load(candidate)
            except (OSError, ValueError):
                posterior = None

    if posterior is not None and posterior.ndim == 2 and posterior.shape[1] > 0:
        names = data.get("theta_names") or data.get("parameter_names")
        _plot_marginals_from_arrays(posterior, names=names, figure_dir=figure_dir)

    # Posterior predictive summary plot.
    _require_mpl()
    predictive_mean = data.get("predictive_mean")
    predictive_std = data.get("predictive_std")
    observed = data.get("observed_summary") or data.get("x_obs")
    if predictive_mean is not None or observed is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        index = np.arange(max(len(predictive_mean or []), len(observed or [])))

        if predictive_mean is not None:
            predictive_mean = np.asarray(predictive_mean, dtype=float)
            predictive_std = np.asarray(predictive_std, dtype=float) if predictive_std is not None else np.zeros_like(predictive_mean)
            ax.errorbar(index[: len(predictive_mean)], predictive_mean, yerr=2 * predictive_std,
                        fmt="o", color="tab:blue", capsize=3, label="Posterior predictive")
        if observed is not None:
            observed = np.asarray(observed, dtype=float)
            ax.plot(index[: len(observed)], observed, "x", color="tab:red",
                    markersize=8, markeredgewidth=2, label="Observed data")

        ax.set_xlabel("Summary statistic")
        ax.set_ylabel("Value")
        ax.set_title("Pyloric-network posterior predictive check")
        ax.legend()
        fig.tight_layout()
        _save(fig, figure_dir, "figure4_pyloric_predictive")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NPSE reproduction figures from result JSON files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["2", "3", "5", "6", "4"],
        help="Figures to generate (e.g. 2 3 5 6 4).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    figure_map = {
        "2": make_figure2,
        "3": make_figure3,
        "5": make_figure5,
        "6": make_figure6,
        "4": make_figure4,
    }
    for figure in args.figures:
        func = figure_map.get(figure)
        if func is None:
            print(f"[make_figures] Unknown figure key: {figure!r}", file=sys.stderr)
            continue
        print(f"[make_figures] Generating figure {figure} ...")
        try:
            func(results_dir=args.results_dir, figure_dir=args.figure_dir)
        except Exception as exc:  # pragma: no cover - defensive CLI
            print(f"[make_figures] Failed to generate figure {figure}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
