#!/usr/bin/env python
"""Reproduce Table 2 and Figures 1 / 5 of "LCA-on-the-Line".

This script orchestrates the already-implemented building blocks:

    src/hierarchy/wordnet.py         -> ImageNet WordNet taxonomy
    src/hierarchy/lca_matrix.py      -> pairwise LCA distance matrix
    src/data/{imagenet,ood_datasets} -> ID / OOD loaders
    src/models/vm_zoo.py             -> 36 torchvision VMs (+ 39 VLMs when available)
    src/eval/evaluate_models.py      -> per-model ID/OOD Top1/Top5/LCA/ELCA table
    src/metrics/correlation.py       -> R^2 / PEA / KEN / SPE / MAE + linear fit

What is produced
----------------
1. The 75-model score table (``scores.json``) with ID LCA / ID Top1 and the OOD
   Top-1 accuracies on ImageNet-v2, ImageNet-S (sketch), ImageNet-R,
   ImageNet-A and ObjectNet.
2. Table 2: for every OOD dataset the correlation of the *in-distribution
   metric* (LCA and, as a baseline, ID Top-1) with OOD Top-1 accuracy.
3. Figure 1 / Figure 5 style scatter plots (accuracy-on-the-line vs. LCA-on-the-line)
   written as PNG files when matplotlib is available.

Everything degrades gracefully: without the real datasets/models the script will
run on cached outputs, and with ``--allow-synthetic`` it can be smoke-tested
offline end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Make ``python scripts/run_correlation.py`` work without installing the package
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE if os.path.basename(_HERE) == "scripts" else _HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.eval.evaluate_models import (  # noqa: E402  (after sys.path tweak)
    EvaluationConfig,
    ModelRecord,
    TABLE8_REFERENCE,
    build_evaluation_loaders,
    build_hierarchy,
    evaluate_zoo,
    evaluate_zoo_from_cache,
    load_results,
    records_to_rows,
    save_results,
    summary_table,
)
from src.metrics.correlation import (  # noqa: E402
    TABLE2_CORRELATION_TARGETS,
    correlation_metrics,
    correlation_table,
    fit_predict,
    format_table,
    mean_absolute_error,
    PearsonCorrelation,
)

LOG = logging.getLogger("run_correlation")

# ---------------------------------------------------------------------------
# Defaults / paper constants
# ---------------------------------------------------------------------------
DEFAULT_OOD = ["v2", "s", "r", "a", "objectnet"]

#: Which in-distribution scores we correlate against OOD Top-1 accuracy.
#: ``id_lca`` is the paper's unified metric; the others are baselines.
PREDICTORS = {
    "id_lca": "ID LCA (information content)",
    "id_elca": "ID ELCA",
    "id_top1": "ID Top-1 (accuracy-on-the-line baseline)",
    "id_top5": "ID Top-5",
}

#: Default predictor used for Figures 1 / 5.
DEFAULT_PREDICTOR = "id_lca"

#: Figure 5 style: fit OOD *error* as a function of the ID metric.
DEFAULT_FIT_ON_ERROR = True


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------
def build_score_table(
    records: Dict[str, ModelRecord],
    datasets: Sequence[str] = DEFAULT_OOD,
) -> List[Dict[str, Any]]:
    """Flatten evaluation records into the Table 1 / Table 2 score rows."""
    rows: List[Dict[str, Any]] = []
    for name, record in sorted(records.items()):
        if record is None:
            continue
        row = {
            "model": name,
            "family": record.family,
            "id_top1": record.id_top1,
            "id_top5": record.id_top5,
            "id_lca": record.id_lca,
            "id_elca": record.id_elca,
        }
        for ds in datasets:
            row[f"ood_top1_{ds}"] = record.get(ds, "top1")
            row[f"ood_top5_{ds}"] = record.get(ds, "top5")
            row[f"ood_lca_{ds}"] = record.get(ds, "lca")
        rows.append(row)
    return rows


def _column(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    return [r.get(key, float("nan")) for r in rows]


def _finite_pairs(x: Sequence[float], y: Sequence[float]):
    import math

    xs, ys = [], []
    for a, b in zip(x, y):
        try:
            a = float(a)
            b = float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b):
            xs.append(a)
            ys.append(b)
    return xs, ys


def compute_correlation_table(
    rows: Sequence[Dict[str, Any]],
    datasets: Sequence[str] = DEFAULT_OOD,
    predictors: Sequence[str] = ("id_lca", "id_top1"),
    fit_on_error: bool = DEFAULT_FIT_ON_ERROR,
    abs_values: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Correlate every predictor with every OOD dataset's Top-1 accuracy.

    Returns ``{predictor: {dataset: {r2, pea, ken, spe, mae, n, slope, intercept}}}``.

    ``fit_on_error=True`` mirrors Figure 5, where the OOD **error** is fitted as a
    function of the in-distribution metric (LCA is not in [0, 1], hence the
    min-max scaling used by :func:`fit_predict` instead of a probit transform).
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pred in predictors:
        out[pred] = {}
        for ds in datasets:
            x_all = _column(rows, pred)
            y_all = _column(rows, f"ood_top1_{ds}")
            x, y = _finite_pairs(x_all, y_all)
            if len(x) < 2:
                out[pred][ds] = {"r2": float("nan"), "pea": float("nan"),
                                 "ken": float("nan"), "spe": float("nan"),
                                 "mae": float("nan"), "n": float(len(x))}
                continue
            target = "error" if fit_on_error else "identity"
            report = fit_predict(x, y, scaler="minmax", target_transform=target)
            out[pred][ds] = {
                "r2": report.get("r2", float("nan")),
                "pea": report.get("pea", float("nan")),
                "ken": correlation_metrics(x, y, abs_values=abs_values).get("ken", float("nan")),
                "spe": correlation_metrics(x, y, abs_values=abs_values).get("spe", float("nan")),
                "mae": report.get("mae", float("nan")),
                "rmse": report.get("rmse", float("nan")),
                "slope": report.get("slope", float("nan")),
                "intercept": report.get("intercept", float("nan")),
                "n": float(len(x)),
            }
    return out


def format_correlation_table(
    table: Dict[str, Dict[str, Dict[str, float]]],
    datasets: Sequence[str],
    predictors: Sequence[str] = ("id_lca", "id_top1"),
    decimals: int = 3,
) -> str:
    """Render a Table-2 style text block (rows = predictor, columns = OOD set)."""
    header = "predictor".ljust(22) + "".join(ds.rjust(11) for ds in datasets) + "metric"
    lines = [header, "-" * len(header)]
    for pred in predictors:
        for metric in ("r2", "pea", "mae"):
            label = pred if metric == "r2" else ""
            cells = ""
            for ds in datasets:
                val = table.get(pred, {}).get(ds, {}).get(metric, float("nan"))
                cells += f"{val:11.{decimals}f}"
            lines.append(label.ljust(22) + cells + metric.upper())
        lines.append("")
    return "\n".join(lines)


def check_against_table2(
    table: Dict[str, Dict[str, Dict[str, float]]],
    tolerance: float = 0.12,
    predictor: str = "id_lca",
) -> List[str]:
    """Compare the measured correlations with the values reported in Table 2."""
    notes: List[str] = []
    measured = table.get(predictor, {})
    for ds, target in TABLE2_CORRELATION_TARGETS.items():
        key = ds.lower()
        got = measured.get(key, {})
        for metric in ("r2", "pea"):
            want = target.get(metric)
            if want is None:
                continue
            have = got.get(metric, float("nan"))
            if not _is_finite(have):
                notes.append(f"[skip] {ds:>10s} {metric.upper():>3s}: no data")
                continue
            delta = abs(abs(float(have)) - abs(float(want)))
            flag = "OK  " if delta <= tolerance else "DIFF"
            notes.append(
                f"[{flag}] {ds:>10s} {metric.upper():>3s}: {abs(float(have)):.3f} "
                f"vs paper {abs(float(want)):.3f} (|d|={delta:.3f})"
            )
    return notes


def _is_finite(x: Any) -> bool:
    try:
        import math

        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_baseline_collapse(
    table: Dict[str, Dict[str, Dict[str, float]]],
    severe: Sequence[str] = ("s", "r", "a"),
) -> List[str]:
    """The ID Top-1 baseline must collapse on the severe-shift OOD sets."""
    notes: List[str] = []
    base = table.get("id_top1", {})
    lca = table.get("id_lca", {})
    for ds in severe:
        b = base.get(ds, {}).get("r2", float("nan"))
        l = lca.get(ds, {}).get("r2", float("nan"))
        if not (_is_finite(b) and _is_finite(l)):
            notes.append(f"[skip] baseline check {ds}: no data")
            continue
        ok = abs(float(b)) < 0.4 and abs(float(l)) > 0.7
        notes.append(
            f"[{'OK  ' if ok else 'DIFF'}] {ds}: ID-Top1 R2={abs(float(b)):.3f} vs "
            f"ID-LCA R2={abs(float(l)):.3f}"
        )
    return notes


def check_table8(records: Dict[str, ModelRecord], tolerance: float = 0.15) -> List[str]:
    """Sanity-check a few ID/OOD numbers against Table 8 of the paper."""
    notes: List[str] = []
    for name, record in records.items():
        lowered = name.lower()
        for ref_key, ref_val in TABLE8_REFERENCE.items():
            if ref_key in lowered:
                notes.extend(record.check_against_table8(tolerance=tolerance))
                break
    return notes


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(
    rows: Sequence[Dict[str, Any]],
    out_dir: str,
    datasets: Sequence[str] = DEFAULT_OOD,
    predictors: Sequence[str] = ("id_tcp1", "id_lca"),
    fit_on_error: bool = DEFAULT_FIT_ON_ERROR,
    dpi: int = 200,
) -> List[str]:
    """Draw Figure 1 (accuracy vs LCA on the line) and Figure 5 (error fits)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        LOG.warning("matplotlib unavailable, skipping figures: %s", exc)
        return []

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    # ---- Figure 1: ID metric vs OOD accuracy, one panel per OOD dataset ----
    ncols = min(3, max(1, len(datasets)))
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        for pred, marker, color in (("id_top1", "o", "tab:blue"), ("id_lca", "^", "tab:red")):
            x, y = _finite_pairs(_column(rows, pred), _column(rows, f"ood_top1_{ds}"))
            if len(x) < 2:
                continue
            report = fit_predict(x, y, scaler="minmax",
                                 target_transform="error" if fit_on_error else "identity")
            xs = sorted(x)
            ax.scatter(x, y, s=14, marker=marker, alpha=0.75,
                       label=f"{pred} (R2={abs(report.get('r2', float('nan'))):.2f})")
            ax.plot(xs, [report["predict"](v) for v in xs], color=color, lw=1.2)
        ax.set_title(ds)
        ax.set_xlabel("ID metric")
        ax.set_ylabel("OOD Top-1")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("LCA-on-the-Line vs. accuracy-on-the-line")
    fig.tight_layout()
    path = os.path.join(out_dir, "figure1_lca_on_the_line.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    # ---- Figure 5: error-space fits for the unified LCA metric ----
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for ds in datasets:
        x, y = _finite_pairs(_column(rows, DEFAULT_PREDICTOR), _column(rows, f"ood_top1_{ds}"))
        if len(x) < 2:
            continue
        err = [1.0 - v for v in y]
        report = fit_predict(x, err, scaler="minmax", target_transform="identity")
        xs = sorted(x)
        ax.scatter(x, err, s=12, alpha=0.6, label=f"{ds}")
        ax.plot(xs, [report["predict"](v) for v in xs], lw=1.2)
    ax.set_xlabel("ID LCA")
    ax.set_ylabel("OOD Top-1 error")
    ax.set_title("LCA-on-the-Line (error space)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure5_lca_error_fit.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)

    return written


# ---------------------------------------------------------------------------
# Data / model plumbing
# ---------------------------------------------------------------------------
def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except Exception:  # pragma: no cover
        LOG.warning("PyYAML unavailable, ignoring config %s", path)
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_cache_dir(cfg: Dict[str, Any], cli_value: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value
    for key in ("cache_dir", "output_dir", "outputs"):
        if cfg.get(key):
            return cfg[key]
    return None


def _build_zoo(names: Optional[Sequence[str]], device: Optional[str], include_vlm: bool):
    """Best-effort construction of the VM (+VLM) zoo."""
    zoo: Dict[str, Any] = {}
    try:
        from src.models.vm_zoo import build_vm_zoo, list_vm_names

        vm_names = list(names) if names else list(list_vm_names())
        zoo.update(build_vm_zoo(names=vm_names, device=device, allow_failures=True))
    except Exception as exc:  # pragma: no cover - heavy optional deps
        LOG.warning("could not build the VM zoo: %s", exc)

    if include_vlm:
        try:
            from src.models.vlm_zoo import build_vlm_zoo, list_vlm_names

            vlm_names = list(names) if names else list(list_vlm_names())
            zoo.update(build_vlm_zoo(names=vlm_names, device=device, allow_failures=True))
        except Exception as exc:
            LOG.info("VLM zoo unavailable (%s); continuing with vision models only", exc)
    return zoo


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce Table 2 / Figures 1 & 5 of LCA-on-the-Line."
    )
    p.add_argument("--config", default=os.path.join(_ROOT, "configs", "config.yaml"))
    p.add_argument("--cache-dir", default=None, help="directory holding cached .npz outputs")
    p.add_argument("--dataset-root", default=None, help="ImageNet-1k root (optional)")
    p.add_argument("--ood-root", action="append", default=None,
                   help="OOD dataset root, repeatable; accepts name=path")
    p.add_argument("--hierarchy-csv", default=None, help="path to imagenet_fiveai.csv")
    p.add_argument("--models", nargs="*", default=None, help="subset of model names")
    p.add_argument("--datasets", nargs="*", default=None, help="OOOD dataset keys")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap samples per dataset (debug / smoke tests)")
    p.add_argument("--temperature", type=float, default=1.0, help="ELCA softmax temperature")
    p.add_argument("--device", default=None)
    p.add_argument("--from-cache-only", action="store_true",
                   help="rebuild the score table purely from cached outputs")
    p.add_argument("--include-vlm", action="store_true", help="also evaluate the CLIP/OpenCLIP zoo")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="use synthetic image fallbacks (offline smoke test)")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--scores-json", default=None, help="cached score table to reuse / write")
    p.add_argument("--results-dir", default=None, help="where to write JSON/tables/figures")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--tol", type=float, default=0.12, help="Table-2 comparison tolerance")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_config(args.config)
    cache_dir = _resolve_cache_dir(cfg, args.cache_dir)
    results_dir = args.results_dir or cfg.get("results_dir") or os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    datasets = [d.lower() for d in (args.datasets or DEFAULT_OOD)]
    scores_json = args.scores_json or os.path.join(results_dir, "scores.json")

    # ------------------------------------------------------------------ #
    # 1. Build (or reload) the score table
    # ------------------------------------------------------------------ #
    records: Dict[str, ModelRecord] = {}
    if os.path.exists(scores_json) and not args.overwrite_cache:
        try:
            records = load_results(scores_json)
            LOG.info("reused %d model records from %s", len(records), scores_json)
        except Exception as exc:
            LOG.warning("could not reuse %s (%s)", scores_json, exc)

    if not records:
        if args.from_cache_only or (cache_dir and not args.models and not cfg.get("models")):
            if not cache_dir:
                LOG.error("--from-cache-only requires --cache-dir")
                return 2
            LOG.info("rebuilding score table from cached outputs in %s", cache_dir)
            records = evaluate_zoo_from_cache(
                model_names=args.models or cfg.get("models") or None,
                cache_dir=cache_dir,
                temperature=args.temperature,
                ood_names=datasets,
            )
        else:
            hierarchy = build_hierarchy(args.hierarchy_csv, allow_synthetic=True)
            eval_cfg = EvaluationConfig(
                dataset_root=args.dataset_root or cfg.get("dataset_root"),
                ood_roots=_parse_ood_roots(args.ood_root or cfg.get("ood_roots")),
                cache_dir=cache_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                resolution=args.resolution,
                temperature=args.temperature,
                max_samples=args.max_samples,
                device=args.device,
                allow_synthetic=args.allow_synthetic,
                overwrite_cache=args.overwrite_cache,
                ood_names=datasets,
            )
            id_loader, ood_loaders = build_evaluation_loaders(eval_cfg, hierarchy)
            if id_loader is None:
                LOG.error("no in-distribution data available; aborting")
                return 2
            zoo = _build_zoo(args.models or cfg.get("models"), args.device, args.include_vlm)
            if not zoo:
                LOG.error("no models could be constructed; aborting")
                return 2
            records = evaluate_zoo(
                zoo, id_loader=id_loader, ood_loaders=ood_loaders,
                hierarchy=hierarchy, config=eval_cfg,
            )
        if records:
            try:
                save_results(scores_json, records)
                LOG.info("wrote score table to %s", scores_json)
            except Exception as exc:
                LOG.warning("could not save score table: %s", exc)

    if not records:
        LOG.error("no model records available - nothing to correlate")
        return 2

    rows = build_score_table(records, datasets)
    with open(os.path.join(results_dir, "score_rows.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)

    print("\n=== Model score table (Table 1 style) ===")
    print(summary_table(records, datasets=datasets))

    # Sanity check a few rows against Table 8 of the paper when available.
    table8_notes = check_table8(records, tolerance=cfg.get("sanity_tolerance", 0.15))
    if table8_notes:
        print("\n=== Table 8 sanity check ===")
        for note in table8_notes:
            print(" ", note)

    # ------------------------------------------------------------------ #
    # 2. Table 2 correlations
    # ------------------------------------------------------------------ #
    table = compute_correlation_table(
        rows, datasets=datasets, predictors=("id_lca", "id_top1"),
        fit_on_error=DEFAULT_FIT_ON_ERROR,
    )
    with open(os.path.join(results_dir, "table2_correlations.json"), "w") as fh:
        json.dump(table, fh, indent=2)

    print("\n=== Table 2: correlation of ID metrics with OOD Top-1 (fit on error) ===")
    print(format_correlation_table(table, datasets))

    print("\n=== vs. paper (Table 2) ===")
    for note in check_against_table2(table, tolerance=args.tol):
        print(" ", note)

    print("\n=== Baseline collapse check (severe shift: S / R / A) ===")
    for note in check_baseline_collapse(table):
        print(" ", note)

    # ------------------------------------------------------------------ #
    # 3. A compact success summary used by the reproduction checklist
    # ------------------------------------------------------------------ #
    success = _summarize_success(table, datasets)
    with open(os.path.join(results_dir, "success_summary.json"), "w") as fh:
        json.dump(success, fh, indent=2)
    print("\n=== Success criteria (R2/PEA > 0.7 on severe shift, LCA beats ID Top-1) ===")
    print(json.dumps(success, indent=2))

    # ------------------------------------------------------------------ #
    # 4. Figures 1 / 5
    # ------------------------------------------------------------------ #
    if not args.no_figures:
        written = make_figures(rows, results_dir, datasets=datasets)
        for path in written:
            print(f"wrote figure: {path}")

    return 0


def _parse_ood_roots(raw: Any) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    out: Dict[str, str] = {}
    for item in raw:
        if "=" in str(item):
            key, value = str(item).split("=", 1)
            out[key.strip()] = value.strip()
        else:
            out.setdefault("root", str(item))
    return out or None


def _summarize_success(
    table: Dict[str, Dict[str, Dict[str, float]]],
    datasets: Sequence[str],
) -> Dict[str, Any]:
    """Gate the reproduction: LCA should beat the ID Top-1 baseline on severe shift."""
    summary: Dict[str, Any] = {}
    for ds in datasets:
        lca = table.get("id_lca", {}).get(ds, {})
        base = table.get("id_top1", {}).get(ds, {})
        entry = {
            "lca_r2": lca.get("r2", float("nan")),
            "lca_pea": lca.get("pea", float("nan")),
            "lca_mae": lca.get("mae", float("nan")),
            "id_top1_r2": base.get("r2", float("nan")),
            "id_top1_mae": base.get("mae", float("nan")),
        }
        r2 = entry["lca_r2"]
        pea = entry["lca_pea"]
        mae_ok = False
        if _is_finite(entry["lca_mae"]) and _is_finite(entry["id_top1_mae"]):
            mae_ok = abs(float(entry["lca_mae"])) <= abs(float(entry["id_top1_mae"]))
        entry["passes_r2_pea_0.7"] = bool(
            _is_finite(r2) and _is_finite(pea) and abs(float(r2)) > 0.7 and abs(float(pea)) > 0.7
        )
        entry["lca_beats_id_top1_mae"] = bool(mae_ok)
        summary[ds] = entry
    return summary


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
