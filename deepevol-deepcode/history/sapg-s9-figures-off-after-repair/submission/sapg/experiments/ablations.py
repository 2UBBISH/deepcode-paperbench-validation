"""Ablation experiments for SAPG (Sec 4.2-4.3, 4.5; Fig. 6).

This module drives the ablation suite described in the SAPG paper:

* **Aggregation variants** (Sec 4.2-4.3):
    - ``leader_follower``      -- primary SAPG scheme (leader aggregates follower data)
    - ``symmetric``            -- every policy aggregates off-policy data from all others
    - ``high_off_policy_ratio``-- leader uses ALL off-policy data (no subsampling)
    - ``no_off_policy``        -- independent PPO per block (no aggregation at all)

* **Entropy coefficient sweep** (Sec 4.5):
    - ``sigma in {0.0, 0.003, 0.005}`` for follower policies, with the leader
      always receiving no entropy term.

The module reuses the training machinery from :mod:`experiments.train` so that
ablation runs are directly comparable to the main results (same seeds, same
sample budget, same logging format).  Results are written as per-run
``*_history.json`` files plus an aggregated ``ablation_summary.json`` and
``ablation_curves.npz`` that :mod:`experiments.plot` can consume.

Usage
-----
::

    python -m experiments.ablations --tasks regrasping throw reorientation \\
        --seeds 0 1 2 3 4 --output-dir runs_ablations

    # Only the aggregation variants
    python -m experiments.ablations --suite aggregation

    # Only the entropy sweep
    python -m experiments.ablations --suite entropy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-root path injection so the module works both as a package and a script.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sapg.config import (  # noqa: E402
    EXPECTED_RESULTS,
    SAPGConfig,
    get_config,
)
from experiments.train import (  # noqa: E402
    ALGORITHMS,
    TASKS,
    TASK_METADATA,
    RunSpec,
    aggregate_runs,
    build_env,
    build_trainer,
    final_metrics,
    run_single,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Aggregation variants ablated in Fig. 6 (left/middle panels).
AGGREGATION_VARIANTS: Tuple[str, ...] = (
    "leader_follower",
    "symmetric",
    "high_off_policy_ratio",
    "no_off_policy",
)

#: Human-readable labels for the aggregation variants.
AGGREGATION_LABELS: Dict[str, str] = {
    "leader_follower": "SAPG (leader-follower)",
    "symmetric": "Symmetric aggregation",
    "high_off_policy_ratio": "High off-policy ratio",
    "no_off_policy": "No off-policy (indep. PPO)",
}

#: Entropy coefficients swept in Sec 4.5 / Fig. 6 (right panel).
ENTROPY_COEFS: Tuple[float, ...] = (0.0, 0.003, 0.005)

#: Tasks used for the ablation suite by default (the three hard tasks).
DEFAULT_ABLATION_TASKS: Tuple[str, ...] = ("regrasping", "throw", "reorientation")

#: Default number of seeds for ablations (paper uses 5).
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

#: Default sample budget per run (paper: ~2e10 transitions).
DEFAULT_TOTAL_TRANSITIONS: float = 2e10


# ---------------------------------------------------------------------------
# Run specification for a single ablation run
# ---------------------------------------------------------------------------


@dataclass
class AblationSpec:
    """Describes a single ablation run.

    Attributes
    ----------
    task:
        Task name (``regrasping``/``throw``/``reorientation``/``shadowhand``/
        ``allegrohand``).
    variant:
        Aggregation variant name (one of :data:`AGGREGATION_VARIANTS`) or
        ``"entropy"`` for entropy-sweep runs.
    seed:
        Random seed.
    entropy_coef:
        Follower entropy coefficient ``sigma`` (only meaningful for the
        entropy sweep; ignored for aggregation variants which use the task
        default).
    num_iterations:
        Number of outer training iterations.
    total_transitions:
        Sample budget (used to derive ``num_iterations`` when not given).
    dry_run:
        Use the CPU ``DummyVectorEnv`` fallback.
    output_dir:
        Root directory for run artifacts.
    device:
        Torch device string.
    overrides:
        Extra config overrides applied on top of the task defaults.
    """

    task: str
    variant: str
    seed: int
    entropy_coef: Optional[float] = None
    num_iterations: Optional[int] = None
    total_transitions: Optional[float] = None
    dry_run: bool = False
    output_dir: str = "runs_ablations"
    device: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        """Unique, filesystem-safe name for this run."""
        if self.variant == "entropy":
            coef = 0.0 if self.entropy_coef is None else float(self.entropy_coef)
            tag = f"sigma_{coef:g}".replace(".", "p")
        else:
            tag = self.variant
        return f"{self.task}__{tag}__seed{self.seed}"

    @property
    def run_dir(self) -> str:
        """Directory where this run's artifacts are written."""
        return os.path.join(self.output_dir, self.run_name)

    def to_run_spec(self) -> RunSpec:
        """Convert to a :class:`experiments.train.RunSpec` for execution."""
        overrides = dict(self.overrides)
        if self.variant == "entropy":
            overrides["entropy_coef"] = (
                0.0 if self.entropy_coef is None else float(self.entropy_coef)
            )
            # Entropy sweep always uses the leader-follower aggregation.
            overrides.setdefault("aggregation", "leader_follower")
        else:
            overrides["aggregation"] = self.variant
            if self.variant == "high_off_policy_ratio":
                overrides["subsample_off_policy"] = False
            elif self.variant == "no_off_policy":
                overrides["subsample_off_policy"] = False
                overrides["off_policy_coef"] = 0.0

        return RunSpec(
            task=self.task,
            algorithm="sapg",
            seed=self.seed,
            num_iterations=self.num_iterations,
            total_transitions=self.total_transitions,
            dry_run=self.dry_run,
            output_dir=self.output_dir,
            device=self.device,
            overrides=overrides,
        )


# ---------------------------------------------------------------------------
# Suite construction
# ---------------------------------------------------------------------------


def build_aggregation_specs(
    tasks: Sequence[str] = DEFAULT_ABLATION_TASKS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    variants: Sequence[str] = AGGREGATION_VARIANTS,
    output_dir: str = "runs_ablations",
    num_iterations: Optional[int] = None,
    total_transitions: Optional[float] = None,
    dry_run: bool = False,
    device: Optional[str] = None,
) -> List[AblationSpec]:
    """Build the list of aggregation-variant ablation runs."""
    specs: List[AblationSpec] = []
    for task in tasks:
        for variant in variants:
            for seed in seeds:
                specs.append(
                    AblationSpec(
                        task=task,
                        variant=variant,
                        seed=seed,
                        num_iterations=num_iterations,
                        total_transitions=total_transitions,
                        dry_run=dry_run,
                        output_dir=output_dir,
                        device=device,
                    )
                )
    return specs


def build_entropy_specs(
    tasks: Sequence[str] = DEFAULT_ABLATION_TASKS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    coefs: Sequence[float] = ENTROPY_COEFS,
    output_dir: str = "runs_ablations",
    num_iterations: Optional[int] = None,
    total_transitions: Optional[float] = None,
    dry_run: bool = False,
    device: Optional[str] = None,
) -> List[AblationSpec]:
    """Build the list of entropy-coefficient sweep runs."""
    specs: List[AblationSpec] = []
    for task in tasks:
        for coef in coefs:
            for seed in seeds:
                specs.append(
                    AblationSpec(
                        task=task,
                        variant="entropy",
                        seed=seed,
                        entropy_coef=float(coef),
                        num_iterations=num_iterations,
                        total_transitions=total_transitions,
                        dry_run=dry_run,
                        output_dir=output_dir,
                        device=device,
                    )
                )
    return specs


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_ablation_suite(
    specs: Sequence[AblationSpec],
    output_dir: str = "runs_ablations",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute a list of ablation runs and aggregate their results.

    Each run is executed via :func:`experiments.train.run_single`, which writes
    ``history.json`` and ``final.pt`` into the run directory.  Failures are
    caught per-run so a single crash does not abort the whole suite.

    Returns
    -------
    dict
        ``{"runs": [...], "aggregation": {...}, "entropy": {...}}`` where the
        aggregation/entropy entries map ``task -> variant -> {x, mean, stderr,
        n}``.
    """
    os.makedirs(output_dir, exist_ok=True)

    records: List[Dict[str, Any]] = []
    t_start = time.time()
    for idx, spec in enumerate(specs, start=1):
        if verbose:
            print(
                f"[ablation {idx}/{len(specs)}] {spec.run_name} "
                f"(task={spec.task}, variant={spec.variant}, seed={spec.seed})",
                flush=True,
            )
        try:
            record = run_single(spec.to_run_spec())
            record["variant"] = spec.variant
            record["entropy_coef"] = spec.entropy_coef
            records.append(record)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  !! run failed: {exc}", file=sys.stderr, flush=True)
            records.append(
                {
                    "task": spec.task,
                    "algorithm": "sapg",
                    "seed": spec.seed,
                    "variant": spec.variant,
                    "entropy_coef": spec.entropy_coef,
                    "run_name": spec.run_name,
                    "run_dir": spec.run_dir,
                    "error": repr(exc),
                }
            )

    aggregation = aggregate_ablation_records(records, kind="aggregation")
    entropy = aggregate_ablation_records(records, kind="entropy")

    summary = {
        "runs": records,
        "aggregation": aggregation,
        "entropy": entropy,
        "wall_time": time.time() - t_start,
    }

    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2)

    _write_curves_npz(
        os.path.join(output_dir, "ablation_curves.npz"),
        aggregation=aggregation,
        entropy=entropy,
    )

    if verbose:
        print(f"\nWrote {summary_path}", flush=True)
        print(
            f"Wrote {os.path.join(output_dir, 'ablation_curves.npz')}",
            flush=True,
        )

    return summary


def aggregate_ablation_records(
    records: Sequence[Dict[str, Any]],
    kind: str = "aggregation",
    num_points: int = 100,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Aggregate per-seed ablation histories into mean/stderr curves.

    Parameters
    ----------
    records:
        Run records produced by :func:`run_ablation_suite`.
    kind:
        ``"aggregation"`` to group by aggregation variant, ``"entropy"`` to
        group by entropy coefficient (keys become ``sigma_<coef>``).
    num_points:
        Number of points on the common interpolation grid.

    Returns
    -------
    dict
        ``task -> variant_key -> {"x", "mean", "stderr", "n"}``.
    """
    # Group records by (task, variant_key).
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for rec in records:
        if "error" in rec:
            continue
        task = rec.get("task")
        if task is None:
            continue
        if kind == "entropy":
            if rec.get("variant") != "entropy":
                continue
            coef = rec.get("entropy_coef")
            coef = 0.0 if coef is None else float(coef)
            key = f"sigma_{coef:g}".replace(".", "p")
        else:
            variant = rec.get("variant")
            if variant in (None, "entropy"):
                continue
            key = variant
        grouped.setdefault(task, {}).setdefault(key, []).append(rec)

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for task, variants in grouped.items():
        out[task] = {}
        for key, recs in variants.items():
            curves = _aggregate_records(recs, num_points=num_points)
            if curves is not None:
                out[task][key] = curves
    return out


def _aggregate_records(
    records: Sequence[Dict[str, Any]],
    num_points: int = 100,
    x_key: str = "total_transitions",
) -> Optional[Dict[str, np.ndarray]]:
    """Aggregate a list of run records into a mean/stderr curve.

    Reuses :func:`experiments.train.aggregate_runs` when possible; falls back
    to a local implementation that reads ``history`` entries directly from the
    records (which is what :func:`run_single` returns).
    """
    # Preferred path: records carry a "history" list of per-iteration dicts.
    histories = [r.get("history") for r in records if r.get("history")]
    if histories:
        return _aggregate_histories(histories, num_points=num_points, x_key=x_key)

    # Fallback: delegate to train.aggregate_runs which reads from disk.
    try:
        agg = aggregate_runs(list(records), num_points=num_points)
    except Exception:
        return None

    # aggregate_runs returns task -> algo -> {x, mean, stderr, n}; unwrap.
    for task, algos in agg.items():
        for _algo, curve in algos.items():
            return curve
    return None


def _aggregate_histories(
    histories: Sequence[Sequence[Dict[str, Any]]],
    num_points: int = 100,
    x_key: str = "total_transitions",
) -> Dict[str, np.ndarray]:
    """Interpolate per-seed histories onto a common grid and compute stats."""
    # Determine the metric key from the first history entry.
    metric_key = _infer_metric_key(histories[0])

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for hist in histories:
        if not hist:
            continue
        x = np.asarray([h.get(x_key, i) for i, h in enumerate(hist)], dtype=np.float64)
        y = np.asarray([_extract_metric(h, metric_key) for h in hist], dtype=np.float64)
        if x.size < 2:
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        return {
            "x": np.zeros(0),
            "mean": np.zeros(0),
            "stderr": np.zeros(0),
            "n": np.asarray(0),
        }

    x_min = max(float(x[0]) for x in xs)
    x_max = min(float(x[-1]) for x in xs)
    if x_max <= x_min:
        x_max = x_min + 1.0
    grid = np.linspace(x_min, x_max, num_points)

    interp = np.stack(
        [_interp_flat(x, y, grid) for x, y in zip(xs, ys)], axis=0
    )  # [n_seeds, num_points]
    mean = interp.mean(axis=0)
    n = interp.shape[0]
    if n > 1:
        stderr = interp.std(axis=0, ddof=1) / np.sqrt(n)
    else:
        stderr = np.zeros_like(mean)

    return {
        "x": grid,
        "mean": mean,
        "stderr": stderr,
        "n": np.asarray(n),
    }


def _infer_metric_key(history: Sequence[Dict[str, Any]]) -> str:
    """Pick the metric key to plot from a history entry."""
    if not history:
        return "mean_episode_reward"
    sample = history[0]
    for key in ("mean_episode_reward", "successes", "mean_reward", "episode_reward"):
        if key in sample:
            return key
    return "mean_episode_reward"


def _extract_metric(entry: Dict[str, Any], key: str) -> float:
    """Extract a scalar metric from a history entry, tolerating nesting."""
    value = entry.get(key)
    if value is None:
        # Try common nested containers.
        for container in ("metrics", "stats", "eval"):
            sub = entry.get(container)
            if isinstance(sub, dict) and key in sub:
                value = sub[key]
                break
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _interp_flat(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation with flat extrapolation."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # np.interp already does flat extrapolation at both ends.
    return np.interp(grid, x_sorted, y_sorted)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def build_ablation_table(
    summary: Dict[str, Any],
    kind: str = "aggregation",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Extract final (mean, stderr) per task/variant from an ablation summary."""
    curves = summary.get(kind, {}) if isinstance(summary, dict) else {}
    table: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for task, variants in curves.items():
        table[task] = {}
        for key, curve in variants.items():
            mean = np.asarray(curve.get("mean", []), dtype=np.float64)
            stderr = np.asarray(curve.get("stderr", []), dtype=np.float64)
            if mean.size == 0:
                continue
            table[task][key] = (float(mean[-1]), float(stderr[-1]))
    return table


def format_ablation_table(
    table: Dict[str, Dict[str, Tuple[float, float]]],
    kind: str = "aggregation",
) -> str:
    """Render an ablation table as text."""
    lines: List[str] = []
    title = "Aggregation ablation" if kind == "aggregation" else "Entropy sweep"
    lines.append(f"=== {title} (final performance, mean +/- stderr) ===")
    for task in sorted(table.keys()):
        lines.append(f"\n[{task}]")
        for key in sorted(table[task].keys()):
            mean, stderr = table[task][key]
            lines.append(f"  {key:<28} {mean:>12.4g} +/- {stderr:<10.3g}")
    return "\n".join(lines)


def compare_to_expected(
    table: Dict[str, Dict[str, Tuple[float, float]]],
    kind: str = "aggregation",
) -> Dict[str, Dict[str, float]]:
    """Compare ablation results against the paper's reported values.

    For the entropy sweep, compares ``sigma_0`` and ``sigma_0p005`` against the
    ``sapg_coef0`` / ``sapg_coef0005`` entries of
    :data:`sapg.config.EXPECTED_RESULTS`.  For the aggregation ablation, compares
    ``leader_follower`` against the same SAPG entries and reports the relative
    drop of the other variants.
    """
    out: Dict[str, Dict[str, float]] = {}
    for task, variants in table.items():
        out[task] = {}
        if kind == "entropy":
            ref0 = _expected_value(task, "sapg_coef0")
            ref5 = _expected_value(task, "sapg_coef0005")
            if ref0 is not None and "sigma_0" in variants:
                out[task]["sigma_0_rel_err"] = _rel_err(variants["sigma_0"][0], ref0)
            if ref5 is not None and "sigma_0p005" in variants:
                out[task]["sigma_0p005_rel_err"] = _rel_err(
                    variants["sigma_0p005"][0], ref5
                )
        else:
            ref = _expected_value(task, "sapg_coef0")
            base = variants.get("leader_follower")
            if ref is not None and base is not None:
                out[task]["leader_follower_rel_err"] = _rel_err(base[0], ref)
            if base is not None:
                for key, (mean, _se) in variants.items():
                    if key == "leader_follower":
                        continue
                    denom = abs(base[0]) if abs(base[0]) > 1e-12 else 1.0
                    out[task][f"{key}_rel_drop"] = (base[0] - mean) / denom
    return out


def _expected_value(task: str, key: str) -> Optional[float]:
    """Look up a paper-reported expected value for a task."""
    entry = EXPECTED_RESULTS.get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get(task)
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return float(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rel_err(value: float, reference: float) -> float:
    """Relative error of ``value`` w.r.t. ``reference``."""
    denom = abs(reference) if abs(reference) > 1e-12 else 1.0
    return (value - reference) / denom


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _write_curves_npz(
    path: str,
    aggregation: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    entropy: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> None:
    """Write aggregated ablation curves to an ``.npz`` archive.

    Keys follow the ``<task>__<variant>__<field>`` convention used by
    :mod:`experiments.plot`.
    """
    arrays: Dict[str, np.ndarray] = {}
    for task, variants in aggregation.items():
        for variant, curve in variants.items():
            for field_name, arr in curve.items():
                arrays[f"{task}__{variant}__{field_name}"] = np.asarray(arr)
    for task, variants in entropy.items():
        for variant, curve in variants.items():
            for field_name, arr in curve.items():
                arrays[f"{task}__{variant}__{field_name}"] = np.asarray(arr)
    np.savez(path, **arrays)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the ablation suite."""
    parser = argparse.ArgumentParser(
        description="Run SAPG ablation experiments (aggregation variants + entropy sweep).",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "aggregation", "entropy"),
        default="all",
        help="Which ablation suite to run (default: all).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_ABLATION_TASKS),
        choices=list(TASKS),
        help="Tasks to ablate (default: the three hard AllegroKuka tasks).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(AGGREGATION_VARIANTS),
        choices=list(AGGREGATION_VARIANTS),
        help="Aggregation variants to ablate.",
    )
    parser.add_argument(
        "--entropy-coefs",
        nargs="+",
        type=float,
        default=list(ENTROPY_COEFS),
        help="Entropy coefficients to sweep (default: 0.0 0.003 0.005).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs_ablations",
        help="Directory for ablation run artifacts.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Override the number of training iterations per run.",
    )
    parser.add_argument(
        "--total-transitions",
        type=float,
        default=None,
        help="Sample budget per run (default: 2e10, as in the paper).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the CPU DummyVectorEnv fallback for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (e.g. 'cuda:0').",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only render tables from an existing ablation_summary.json.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the ablation suite."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.table_only:
        summary_path = os.path.join(args.output_dir, "ablation_summary.json")
        if not os.path.exists(summary_path):
            print(f"No summary found at {summary_path}", file=sys.stderr)
            return 1
        with open(summary_path) as fh:
            summary = json.load(fh)
        for kind in ("aggregation", "entropy"):
            table = build_ablation_table(summary, kind=kind)
            if table:
                print(format_ablation_table(table, kind=kind))
                print()
        return 0

    total_transitions = args.total_transitions
    if total_transitions is None and args.num_iterations is None:
        total_transitions = DEFAULT_TOTAL_TRANSITIONS

    specs: List[AblationSpec] = []
    if args.suite in ("all", "aggregation"):
        specs.extend(
            build_aggregation_specs(
                tasks=args.tasks,
                seeds=args.seeds,
                variants=args.variants,
                output_dir=args.output_dir,
                num_iterations=args.num_iterations,
                total_transitions=total_transitions,
                dry_run=args.dry_run,
                device=args.device,
            )
        )
    if args.suite in ("all", "entropy"):
        specs.extend(
            build_entropy_specs(
                tasks=args.tasks,
                seeds=args.seeds,
                coefs=args.entropy_coefs,
                output_dir=args.output_dir,
                num_iterations=args.num_iterations,
                total_transitions=total_transitions,
                dry_run=args.dry_run,
                device=args.device,
            )
        )

    if not specs:
        print("No ablation runs to execute.", file=sys.stderr)
        return 1

    print(
        f"Running {len(specs)} ablation runs "
        f"(suite={args.suite}, tasks={args.tasks}, seeds={args.seeds})",
        flush=True,
    )
    summary = run_ablation_suite(specs, output_dir=args.output_dir)

    for kind in ("aggregation", "entropy"):
        table = build_ablation_table(summary, kind=kind)
        if table:
            print()
            print(format_ablation_table(table, kind=kind))
            comparison = compare_to_expected(table, kind=kind)
            if comparison:
                print("\nComparison to paper-reported values:")
                for task, metrics in comparison.items():
                    for name, value in metrics.items():
                        print(f"  [{task}] {name}: {value:+.3f}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
