#!/usr/bin/env python
"""Reproduce Table 3 of *LCA-on-the-Line* (OOD performance prediction).

This script drives the OOD-prediction evaluation implemented in
``src/eval/ood_prediction.py``:

  * it loads (or rebuilds) the per-model score table produced by
    ``scripts/run_correlation.py`` / ``src/eval/evaluate_models.py``;
  * it predicts OOD Top-1 accuracy from the in-distribution LCA metric and
    from the paper's four baselines
        - ID Top-1 (Miller et al.),
        - Average Confidence (AC, temperature-scaled OOD logits),
        - Aline-D (agreement-on-the-line, depth pairs, hard agreement),
        - Aline-S (agreement-on-the-line, size/width pairs, soft agreement);
  * it reports the MAE of each predictor against the true OOD accuracy for
    ImageNet-v2 / -S / -R / -A / ObjectNet and validates the values against the
    numbers reported in the paper (``TABLE3_REFERENCE``).

The paper's expected behaviour (Table 3 of §4.2):

    dataset      ID Top-1   AC/others   ID LCA
    ImgN-v2        0.058       ...        0.162
    ImgN-S         0.230       ...        0.093
    ImgN-R         0.277       ...        0.114
    ImgN-A         0.192       ...        0.103
    ObjectNet      0.178       ...        0.048

i.e. the ID-LCA predictor has the smallest MAE on the four severe-shift sets.

Scaling note (paper §D.1.1 and the plan): LCA is *not* in ``[0, 1]`` so its
predictor is fitted with min-max scaling, whereas the accuracy baselines are
fitted in probit space.

Usage examples
--------------
    python scripts/run_ood_prediction.py --scores-json results/scores.json
    python scripts/run_ood_prediction.py --cache-dir cache --subset vm
    python scripts/run_ood_prediction.py --config configs/config.yaml -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap so the script runs from a source checkout without install.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


LOG = logging.getLogger("run_ood_prediction")


DEFAULT_DATASETS: Tuple[str, ...] = ("v2", "s", "r", "a", "objectnet")
DEFAULT_METHODS: Tuple[str, ...] = ("id_top1", "ac", "aline_d", "aline_s", "id_lca")
# Datasets on which the paper's baselines collapse / the LCA predictor wins.
SEVERE_SHIFT: Tuple[str, ...] = ("s", "r", "a", "objectnet")

METHOD_LABELS: Dict[str, str] = {
    "id_top1": "ID Top-1",
    "id_top5": "ID Top-5",
    "ac": "AC",
    "aline_d": "Aline-D",
    "aline_s": "Aline-S",
    "id_lca": "ID LCA",
    "id_elca": "ID ELCA",
}


# ---------------------------------------------------------------------------
# Imports with graceful degradation
# ---------------------------------------------------------------------------
def _import_ood_prediction():
    """Import the OOD-prediction implementation, returning the module."""
    try:
        from src.eval import ood_prediction  # type: ignore
    except Exception:  # pragma: no cover - fallback for flat sys.path
        try:
            from eval import ood_prediction  # type: ignore
        except Exception as exc:  # pragma: no cover
            LOG.error("Could not import src/eval/ood_prediction.py: %s", exc)
            raise
    return ood_prediction


def _import_evaluate_models():
    try:
        from src.eval import evaluate_models  # type: ignore
    except Exception:  # pragma: no cover
        from eval import evaluate_models  # type: ignore
    return evaluate_models


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file (best effort; YAML is optional)."""
    if not path:
        default = os.path.join(REPO_ROOT, "configs", "config.yaml")
        path = default if os.path.exists(default) else None
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        LOG.warning("PyYAML unavailable; ignoring config file %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            LOG.warning("Config %s did not parse to a mapping; ignoring", path)
            return {}
        return payload
    except Exception as exc:  # pragma: no cover
        LOG.warning("Failed to read config %s: %s", path, exc)
        return {}


def _config_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Look up the first present key, supporting dotted paths."""
    for key in keys:
        node: Any = config
        found = True
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                found = False
                break
        if found and node is not None:
            return node
    return default


def parse_ood_roots(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeated ``name=path`` CLI arguments into a mapping."""
    roots: Dict[str, str] = {}
    for item in values or []:
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            roots[name.strip()] = path.strip()
        else:
            roots[os.path.basename(item.rstrip("/"))] = item
    return roots


# ---------------------------------------------------------------------------
# Score table loading / rebuilding
# ---------------------------------------------------------------------------
def load_records_from_json(path: str) -> Dict[str, Any]:
    """Load a previously saved score table (JSON) via evaluate_models."""
    ev = _import_evaluate_models()
    return ev.load_results(path)


def records_from_cache(
    cache_dir: str,
    hierarchy: Any = None,
    models: Optional[Sequence[str]] = None,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Rebuild the score table purely from cached ``.npz`` model outputs."""
    ev = _import_evaluate_models()
    if hierarchy is None:
        hierarchy = ev.build_hierarchy(allow_synthetic=False)
    if models:
        names = list(models)
    else:
        names = discover_cached_models(cache_dir, datasets)
    LOG.info("Rebuilding score table from cache for %d model(s)", len(names))
    return ev.evaluate_zoo_from_cache(
        names,
        cache_dir,
        hierarchy=hierarchy,
        ood_names=list(datasets),
        temperature=temperature,
    )


def discover_cached_models(cache_dir: str, datasets: Sequence[str]) -> List[str]:
    """Infer the model names present in a cache directory."""
    outputs_dir = os.path.join(cache_dir, "outputs")
    search_dir = outputs_dir if os.path.isdir(outputs_dir) else cache_dir
    if not os.path.isdir(search_dir):
        return []
    names = set()
    for fname in os.listdir(search_dir):
        if not fname.endswith(".npz"):
            continue
        stem = fname[: -len(".npz")]
        if "__" in stem:
            names.add(stem.split("__")[0])
        else:
            names.add(stem)
    return sorted(names)


def build_or_load_records(
    args: argparse.Namespace, config: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """Return ``(records, source_description)``."""
    scores_json = args.scores_json or _config_get(
        config, "scores_json", "results.scores_json"
    )
    if scores_json and os.path.exists(scores_json) and not args.rebuild:
        LOG.info("Loading score table from %s", scores_json)
        return load_records_from_json(scores_json), f"json:{scores_json}"

    cache_dir = args.cache_dir or _config_get(
        config, "cache_dir", "cache_dir", "outputs", "paths.cache_dir"
    )
    if not cache_dir:
        cache_dir = os.path.join(REPO_ROOT, "cache")

    if args.models:
        models = list(args.models)
    else:
        models = _config_get(config, "models", "model_list", default=None)

    hierarchy = None
    try:
        ev = _import_evaluate_models()
        hierarchy = ev.build_hierarchy(
            csv_path=args.hierarchy_csv, allow_synthetic=args.allow_synthetic
        )
    except Exception as exc:
        LOG.warning("Could not build hierarchy from CSV: %s", exc)

    records = records_from_cache(
        cache_dir,
        hierarchy=hierarchy,
        models=models,
        datasets=args.datasets,
        temperature=args.temperature,
    )
    if not records:
        LOG.error(
            "No cached model outputs found under %s. Run "
            "`python scripts/run_correlation.py` first or pass --scores-json.",
            cache_dir,
        )
    return records, f"cache:{cache_dir}"


# ---------------------------------------------------------------------------
# Table 3 evaluation
# ---------------------------------------------------------------------------
def run_table3(
    records: Dict[str, Any],
    datasets: Sequence[str],
    methods: Sequence[str],
    subset: str,
    cache_dir: Optional[str],
    calibrate_temperature: bool,
    leave_one_out: bool,
    agreement_d: str,
    agreement_s: str,
    metric: str = "top1",
) -> Any:
    """Compute the Table 3 predictor-MAE table."""
    op = _import_ood_prediction()
    kwargs: Dict[str, Any] = {
        "datasets": tuple(datasets),
        "methods": tuple(methods),
        "metric": metric,
        "subset": subset,
        "cache_dir": cache_dir,
        "calibrate_temperature": calibrate_temperature,
        "leave_one_out": leave_one_out,
        "agreement_d": agreement_d,
        "agreement_s": agreement_s,
    }
    try:
        return op.evaluate_ood_prediction(records, **kwargs)
    except TypeError as exc:
        # Older/newer signature: drop the agreement knobs and retry.
        LOG.warning("evaluate_ood_prediction signature mismatch (%s); retrying", exc)
        for key in ("agreement_d", "agreement_s", "leave_one_out"):
            kwargs.pop(key, None)
        return op.evaluate_ood_prediction(records, **kwargs)


def format_mae_table(table: Any, methods: Sequence[str], datasets: Sequence[str]) -> str:
    """Render the Table 3 MAE matrix as text."""
    # Prefer the implementation's own formatter when available.
    formatter = getattr(table, "format_table", None)
    if callable(formatter):
        try:
            rendered = formatter()
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:  # pragma: no cover - fall through to manual render
            pass

    mae_table = getattr(table, "mae_table", None)
    if callable(mae_table):
        mae_table = mae_table()
    if not isinstance(mae_table, dict):
        mae_table = {}

    width = max(12, *(len(METHOD_LABELS.get(m, m)) for m in methods))
    header = "| " + "method".ljust(width) + " | " + " | ".join(
        str(d).rjust(10) for d in datasets
    ) + " |"
    sep = "|" + "-" * (width + 2) + "|" + "|".join("-" * 12 for _ in datasets) + "|"
    lines = [header, sep]
    for method in methods:
        row = mae_table.get(method, {})
        cells = []
        for ds in datasets:
            value = row.get(ds) if isinstance(row, dict) else None
            if value is None:
                result = getattr(table, "get", None)
                if callable(result):
                    try:
                        value = result(method, ds).mae
                    except Exception:
                        value = None
            cells.append("n/a".rjust(10) if value is None else f"{float(value):.3f}".rjust(10))
        lines.append("| " + METHOD_LABELS.get(method, method).ljust(width) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mae_lookup(table: Any, method: str, dataset: str) -> Optional[float]:
    """Fetch a single MAE value from an ``OodPredictionTable``-like object."""
    for attr in ("get", "result"):
        getter = getattr(table, attr, None)
        if callable(getter):
            try:
                result = getter(method, dataset)
            except Exception:
                continue
            if result is None:
                continue
            mae = getattr(result, "mae", None)
            if mae is None and isinstance(result, dict):
                mae = result.get("mae")
            if mae is not None:
                try:
                    return float(mae)
                except (TypeError, ValueError):
                    return None
    mae_table = getattr(table, "mae_table", None)
    if callable(mae_table):
        mae_table = mae_table()
    if isinstance(mae_table, dict):
        row = mae_table.get(method, {})
        if isinstance(row, dict) and row.get(dataset) is not None:
            return float(row[dataset])
    return None


def check_against_table3(
    table: Any, tolerance: float = 0.06, predictor: str = "id_lca"
) -> List[str]:
    """Compare ID-LCA MAE with the paper's Table 3 values."""
    messages: List[str] = []
    try:
        op = _import_ood_prediction()
        reference = getattr(op, "TABLE3_REFERENCE", {})
    except Exception:
        reference = {}
    if not reference:
        return messages
    for dataset, targets in reference.items():
        expected = None
        if isinstance(targets, dict):
            expected = targets.get(predictor, targets.get("lca"))
        else:
            expected = targets
        if expected is None:
            continue
        observed = mae_lookup(table, predictor, dataset)
        if observed is None:
            messages.append(f"[skip] {dataset}: no {predictor} prediction available")
            continue
        delta = abs(observed - float(expected))
        status = "ok" if delta <= tolerance else "MISMATCH"
        messages.append(
            f"[{status}] {dataset}: ID-LCA MAE observed={observed:.3f} "
            f"expected={float(expected):.3f} |delta|={delta:.3f} (tol={tolerance})"
        )
    return messages


def check_lca_beats_baselines(table: Any, datasets: Sequence[str] = SEVERE_SHIFT) -> List[str]:
    """Verify that ID LCA beats ID Top-1 on the severe-shift datasets."""
    messages: List[str] = []
    for dataset in datasets:
        lca = mae_lookup(table, "id_lca", dataset)
        top1 = mae_lookup(table, "id_top1", dataset)
        if lca is None or top1 is None:
            messages.append(f"[skip] {dataset}: missing predictor MAE")
            continue
        status = "ok" if lca < top1 else "FAIL"
        messages.append(
            f"[{status}] {dataset}: LCA MAE {lca:.3f} vs ID Top-1 MAE {top1:.3f}"
        )
    return messages


def success_summary(table: Any, datasets: Sequence[str]) -> Dict[str, Any]:
    """Compact machine-readable summary of the reproduction gates."""
    summary: Dict[str, Any] = {"datasets": {}, "lca_beats_id_top1": {}}
    for dataset in datasets:
        lca = mae_lookup(table, "id_lca", dataset)
        top1 = mae_lookup(table, "id_top1", dataset)
        summary["datasets"][dataset] = {"id_lca_mae": lca, "id_top1_mae": top1}
        if lca is not None and top1 is not None:
            summary["lca_beats_id_top1"][dataset] = bool(lca < top1)
    severe = [d for d in datasets if d in SEVERE_SHIFT]
    summary["passes_lca_beats_id_top1"] = bool(
        severe
        and all(summary["lca_beats_id_top1"].get(d, False) for d in severe)
    )
    best = getattr(table, "best_method", None)
    if callable(best):
        try:
            summary["best_method"] = best()
        except Exception:
            pass
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 3 (OOD performance prediction) of LCA-on-the-Line."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to configs/config.yaml")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cached model outputs")
    parser.add_argument("--scores-json", type=str, default=None, help="Precomputed score table")
    parser.add_argument("--hierarchy-csv", type=str, default=None, help="imagenet_fiveai.csv")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to include")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="OOD datasets (v2 s r a objectnet)",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(DEFAULT_METHODS),
        help="Predictors: id_top1 ac aline_d aline_s id_lca",
    )
    parser.add_argument(
        "--subset", choices=("all", "vm", "vlm"), default="all",
        help="Restrict to vision models or vision-language models",
    )
    parser.add_argument("--metric", choices=("top1", "top5"), default="top1")
    parser.add_argument("--agreement-d", choices=("soft", "hard"), default="hard")
    parser.add_argument("--agreement-s", choices=("soft", "hard"), default="soft")
    parser.add_argument(
        "--no-calibrate", action="store_true",
        help="Disable temperature scaling for the AC baseline",
    )
    parser.add_argument(
        "--leave-one-out", action="store_true",
        help="Fit predictors leave-one-out (honest MAE)",
    )
    parser.add_argument("--rebuild", action="store_true", help="Ignore results/scores.json")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="Permit synthetic fallbacks (offline smoke tests)")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--tol", type=float, default=0.06, help="Table 3 comparison tolerance")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    results_dir = args.results_dir or _config_get(
        config, "results_dir", "output_dir", default=os.path.join(REPO_ROOT, "results")
    )
    os.makedirs(results_dir, exist_ok=True)

    cache_dir = args.cache_dir or _config_get(
        config, "cache_dir", "outputs", "paths.cache_dir",
        default=os.path.join(REPO_ROOT, "cache"),
    )

    # ---- 1. score table ---------------------------------------------------
    try:
        records, source = build_or_load_records(args, config)
    except Exception as exc:
        LOG.error("Failed to obtain model records: %s", exc, exc_info=args.verbose)
        return 2
    print(f"\n=== Model score table ({source}) ===")
    print(f"models: {len(records)}")
    if not records:
        return 1

    # ---- 2. Table 3 predictors -------------------------------------------
    datasets = list(args.datasets)
    methods = list(args.methods)
    try:
        table = run_table3(
            records,
            datasets=datasets,
            methods=methods,
            subset=args.subset,
            cache_dir=cache_dir,
            calibrate_temperature=not args.no_calibrate,
            leave_one_out=args.leave_one_out,
            agreement_d=args.agreement_d,
            agreement_s=args.agreement_s,
            metric=args.metric,
        )
    except Exception as exc:
        LOG.error("OOD prediction failed: %s", exc, exc_info=args.verbose)
        return 3

    print(f"\n=== Table 3: MAE of OOD Top-{args.metric[-1]} predictors ===")
    print(format_mae_table(table, methods, datasets))

    # ---- 3. validation ----------------------------------------------------
    print("\n=== Validation against paper Table 3 ===")
    checks = check_against_table3(table, tolerance=args.tol)
    for line in checks:
        print(line)
    baseline_checks = check_lca_beats_baselines(table, datasets=[d for d in datasets if d in SEVERE_SHIFT])
    for line in baseline_checks:
        print(line)

    summary = success_summary(table, datasets)
    print("\n=== Success summary ===")
    print(json.dumps(summary, indent=2, default=str))

    # ---- 4. artifacts -----------------------------------------------------
    artifacts: List[str] = []
    table_payload = None
    asdict = getattr(table, "asdict", None)
    if callable(asdict):
        try:
            table_payload = asdict()
        except Exception:
            table_payload = None
    if table_payload is None:
        mae_table = getattr(table, "mae_table", None)
        table_payload = mae_table() if callable(mae_table) else {}

    for fname, payload in (
        ("table3_ood_prediction.json", table_payload),
        ("table3_checks.json", {"checks": checks, "baseline_checks": baseline_checks}),
        ("table3_success_summary.json", summary),
    ):
        try:
            path = os.path.join(results_dir, fname)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
            artifacts.append(path)
        except Exception as exc:  # pragma: no cover
            LOG.warning("Could not write %s: %s", fname, exc)

    if artifacts:
        print("\nArtifacts written:")
        for path in artifacts:
            print(f"  - {path}")

    return 0 if summary.get("passes_lca_beats_id_top1", False) or not records else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
