"""Evaluate trained NPSE/TSNPSE/SNPSE benchmark results.

This script loads JSON result files produced by the training/experiment drivers
(``train_npse_benchmarks.py``, ``train_tsnpse_benchmarks.py``,
``run_snpse_variants.py``, and ``run_nlse_comparison.py``) and aggregates the
main evaluation metrics -- C2ST accuracy and MMD -- into per-benchmark,
per-budget summary tables.  The aggregated results are written to CSV and JSON
files so that ``make_figures.py`` can generate the paper's figures.

Typical usage::

    python -m npse.experiments.evaluate_benchmarks
    python -m npse.experiments.evaluate_benchmarks --results-dir results
    python -m npse.experiments.evaluate_benchmarks --output results/summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # pandas is optional but convenient for CSV output
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PANDAS = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Load a JSON file, returning ``None`` if the file is missing/invalid."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] could not load {path}: {exc}", file=sys.stderr)
        return None


def discover_result_files(results_dir: Path) -> List[Path]:
    """Return all JSON files found under ``results_dir``, sorted by path."""
    if not results_dir.exists():
        return []
    return sorted(p for p in results_dir.rglob("*.json") if p.is_file())


def normalize_to_list(data: Any) -> List[Dict[str, Any]]:
    """Convert a JSON result payload to a flat list of result dictionaries.

    Training scripts typically write a JSON list.  Some scripts may write a
    dict keyed by benchmark/budget; this function handles both cases, as well
    as the ``{"results": [...]}`` wrapper format.
    """
    if data is None:
        return []

    if isinstance(data, dict):
        # Common wrapper formats.
        for key in ("results", "items", "entries", "runs"):
            if key in data and isinstance(data[key], list):
                return [r for r in data[key] if isinstance(r, dict)]
        # A dict of {benchmark: {budget: {...}}} or {key: {...}}.
        flattened: List[Dict[str, Any]] = []
        for key, value in data.items():
            if isinstance(value, dict):
                if any(isinstance(v, dict) for v in value.values()):
                    for sub_value in value.values():
                        if isinstance(sub_value, dict):
                            row = dict(sub_value)
                            if "benchmark" not in row:
                                row["benchmark"] = key
                            flattened.append(row)
                else:
                    row = dict(value)
                    if "benchmark" not in row:
                        row["benchmark"] = key
                    flattened.append(row)
            elif isinstance(value, (int, float, str, bool, type(None))):
                flattened.append({"benchmark": key, "value": value})
        return flattened

    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]

    return []


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _first_present(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among ``keys``."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def extract_metric_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize result dicts into a canonical row format.

    The canonical row contains::

        method, benchmark, budget, c2st, mmd, status, wall_time
    """
    rows: List[Dict[str, Any]] = []
    for raw in results:
        method = str(_first_present(raw, "method", "algorithm", "variant", default="npse"))
        benchmark = str(
            _first_present(
                raw,
                "benchmark",
                "benchmark_name",
                "task",
                "task_name",
                default="unknown",
            )
        )
        budget = _first_present(raw, "budget", "total_budget", "simulation_budget", default=None)
        c2st = _first_present(raw, "c2st", "c2st_score", "c2st_accuracy", default=None)
        mmd = _first_present(raw, "mmd", "mmd_score", "mmd2", default=None)
        status = str(_first_present(raw, "status", "state", default="ok"))
        wall_time = _first_present(raw, "wall_time", "elapsed", "time", default=None)

        if c2st is None and mmd is None and status == "ok":
            # Nothing quantitative to aggregate; still record metadata.
            pass

        rows.append(
            {
                "method": method,
                "benchmark": benchmark,
                "budget": budget,
                "c2st": c2st,
                "mmd": mmd,
                "status": status,
                "wall_time": wall_time,
            }
        )
    return rows


def aggregate_metric(rows: Iterable[Dict[str, Any]], key: str) -> Tuple[int, float, float]:
    """Return ``(count, mean, std)`` for a numeric field across rows."""
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0, float("nan"), float("nan")
    values_arr = np.asarray(values, dtype=float)
    return int(values_arr.size), float(np.mean(values_arr)), float(np.std(values_arr))


def group_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, Any], List[Dict[str, Any]]]:
    """Group canonical rows by ``(benchmark, method, budget)``."""
    groups: Dict[Tuple[str, str, Any], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["benchmark"], row["method"], row["budget"])
        groups.setdefault(key, []).append(row)
    return groups


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate canonical rows into one summary row per group."""
    summary_rows: List[Dict[str, Any]] = []
    for (benchmark, method, budget), group in sorted(group_rows(rows).items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), _budget_sort_key(kv[0][2]))):
        c2st_n, c2st_mean, c2st_std = aggregate_metric(group, "c2st")
        mmd_n, mmd_mean, mmd_std = aggregate_metric(group, "mmd")
        statuses = sorted({row["status"] for row in group})
        wall_times = [row["wall_time"] for row in group if row["wall_time"] is not None]
        summary_rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "budget": budget,
                "c2st": c2st_mean if c2st_n else None,
                "c2st_std": c2st_std if c2st_n else None,
                "c2st_runs": c2st_n,
                "mmd": mmd_mean if mmd_n else None,
                "mmd_std": mmd_std if mmd_n else None,
                "mmd_runs": mmd_n,
                "status": ",".join(statuses) if statuses else "ok",
                "mean_wall_time": float(np.mean(wall_times)) if wall_times else None,
            }
        )
    return summary_rows


def _budget_sort_key(budget: Any) -> Tuple[int, Any]:
    """Sort key that places unknown budgets last while ordering integers naturally."""
    if isinstance(budget, (int, float)) or (isinstance(budget, str) and budget.isdigit()):
        return (0, float(budget))
    return (1, str(budget))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def format_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.{precision}f}"
    return str(value)


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    """Print a human-readable summary table grouped by benchmark."""
    if not rows:
        print("No aggregated result rows to display.")
        return

    benchmarks = sorted({row["benchmark"] for row in rows})
    for benchmark in benchmarks:
        bench_rows = [row for row in rows if row["benchmark"] == benchmark]
        print(f"\n=== {benchmark} ===")
        header = f"{'method':<12} {'budget':>9} {'C2ST':>10} {'C2ST std':>9} {'MMD':>10} {'MMD std':>9} {'runs':>5} {'status':>10}"
        print(header)
        print("-" * len(header))
        for row in bench_rows:
            print(
                f"{row['method']:<12} {format_value(row['budget']):>9} "
                f"{format_value(row['c2st']):>10} {format_value(row['c2st_std']):>9} "
                f"{format_value(row['mmd']):>10} {format_value(row['mmd_std']):>9} "
                f"{row['c2st_runs'] + row['mmd_runs']:>5} {row['status']:>10}"
            )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write summary rows to CSV, using pandas when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_PANDAS:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:  # pragma: no cover - minimal fallback
        import csv

        if not rows:
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        "benchmark",
                        "method",
                        "budget",
                        "c2st",
                        "c2st_std",
                        "c2st_runs",
                        "mmd",
                        "mmd_std",
                        "mmd_runs",
                        "status",
                        "mean_wall_time",
                    ]
                )
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate NPSE/TSNPSE/SNPSE benchmark evaluation metrics.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing result JSON files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit list of JSON files to load. If omitted, all JSON files under --results-dir are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to write aggregated CSV/JSON outputs.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Explicit CSV output path (overrides --output-dir for CSV).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Explicit JSON output path (overrides --output-dir for JSON).",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress the human-readable summary table.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.files:
        result_files = [p for p in args.files]
    else:
        result_files = discover_result_files(args.results_dir)

    if not result_files:
        print(f"[warn] no result JSON files found under {args.results_dir}", file=sys.stderr)
        return 0

    all_raw: List[Dict[str, Any]] = []
    for path in result_files:
        payload = load_json(path)
        raw_rows = normalize_to_list(payload)
        for row in raw_rows:
            # Preserve source file for traceability.
            if "source" not in row:
                row = dict(row)
                row["source"] = str(path)
            all_raw.append(row)

    if not all_raw:
        print("[warn] no parseable result rows found.", file=sys.stderr)
        return 0

    canonical = extract_metric_rows(all_raw)
    summary = summarize(canonical)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.csv if args.csv is not None else output_dir / "summary.csv"
    json_path = args.json if args.json is not None else output_dir / "summary.json"

    write_csv(summary, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    if not args.no_print:
        print_summary_table(summary)
        print(f"\nWrote aggregated CSV -> {csv_path}")
        print(f"Wrote aggregated JSON -> {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
