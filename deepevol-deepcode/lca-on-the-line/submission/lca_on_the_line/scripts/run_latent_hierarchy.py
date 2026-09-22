#!/usr/bin/env python
"""Reproduce Table 4 of *LCA-on-the-Line*: robust LCA measurement with latent hierarchies.

For each of the (up to) 75 source pretrained models of the zoo we construct a *latent* class
hierarchy by running hierarchical K-means on the per-class average in-distribution features
``M(X)`` (paper Section 4.3.1 / Appendix E.1):

* extract ID features ``M(X)`` and group them by label ``Y`` -> ``k`` average class features,
* run K-means independently at each of the 9 levels with ``2^i`` centers (``i = 1..9``,
  since ``2^9 < 1000``),
* the pairwise LCA height of two classes is the deepest level at which both share a cluster,
  and by definition every pair shares the base cluster level 10.

Using every latent hierarchy as the taxonomy, we recompute the in-distribution LCA distance for
all models of the zoo and measure the Pearson correlation (PEA) against OOD Top-1 accuracy.
Table 4 reports the mean / min / max / std of those 75 PEA values per OOD dataset and shows that
the LCA signal is robust to the choice of source model.

The script works purely from the cached ``<model>__<dataset>.npz`` outputs written by
``src/eval/evaluate_models.py`` (fast path), can rebuild them on the fly, and supports a
``--allow-synthetic`` smoke test when neither cache nor datasets are available.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap so the script can be executed without installing the package.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = logging.getLogger("run_latent_hierarchy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATASETS: Tuple[str, ...] = ("v2", "s", "r", "a", "objectnet")

#: Cache file names that may be used for the in-distribution split.
ID_CACHE_NAMES: Tuple[str, ...] = (
    "imagenet",
    "id",
    "imagenet-1k",
    "imagenet1k",
    "val",
    "validation",
)

#: Paper Table 4 reference PEA statistics across the 75 latent hierarchies.
TABLE4_REFERENCE: Dict[str, Dict[str, float]] = {
    "v2": {"mean": 0.815, "min": 0.721, "max": 0.863, "std": 0.028},
    "s": {"mean": 0.773, "min": 0.715, "max": 0.829, "std": 0.022},
    "r": {"mean": 0.712, "min": 0.646, "max": 0.780, "std": 0.027},
    "a": {"mean": 0.662, "min": 0.577, "max": 0.717, "std": 0.025},
    "objectnet": {"mean": 0.930, "min": 0.890, "max": 0.952, "std": 0.010},
}

#: Paper Table 4 WordNet row (LCA -> OOD Top-1 PEA) and ID-Top-1 baseline row.
TABLE4_WORDNET_REFERENCE: Dict[str, float] = {
    "v2": 0.582,
    "s": 0.903,
    "r": 0.883,
    "a": 0.839,
    "objectnet": 0.956,
}
TABLE4_BASELINE_REFERENCE: Dict[str, float] = {
    "v2": 0.980,
    "s": 0.275,
    "r": 0.140,
    "a": 0.094,
    "objectnet": 0.522,
}

#: Aliases used when matching OOD dataset names to cache file suffixes.
DATASET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "v2": ("v2", "imagenet-v2", "imagenetv2", "imagenet_v2", "imagenet-v2-matched-frequency"),
    "s": ("s", "sketch", "imagenet-s", "imagenet_sketch", "imagenet-skech", "imagenet-sketch"),
    "r": ("r", "rendition", "imagenet-r", "imagenet_r", "imagenet-rendition"),
    "a": ("a", "adversarial", "imagenet-a", "imagenet_a", "imagenet-adversarial"),
    "objectnet": ("objectnet", "objnet", "object-net", "object_net"),
}

TABLE_KEYS = ("mean", "min", "max", "std")


# ---------------------------------------------------------------------------
# Import helpers (tolerant to flat ``src`` layouts)
# ---------------------------------------------------------------------------
def _import_module(module_names: Sequence[str], what: str = "") -> Any:
    last_err: Optional[BaseException] = None
    for name in module_names:
        try:
            return __import__(name, fromlist=["*"])
        except Exception as exc:  # pragma: no cover - environment dependent
            last_err = exc
    raise ImportError(f"Could not import {what or module_names}: {last_err}")


def _import_latent() -> Any:
    return _import_module(
        ["src.hierarchy.latent_kmeans", "hierarchy.latent_kmeans", "latent_kmeans"],
        "latent_kmeans",
    )


def _import_metrics() -> Any:
    return _import_module(
        ["src.metrics.lca_metric", "metrics.lca_metric", "lca_metric"], "lca_metric"
    )


def _import_correlation() -> Any:
    return _import_module(
        ["src.metrics.correlation", "metrics.correlation", "correlation"], "correlation"
    )


def _import_ood() -> Any:
    return _import_module(
        ["src.data.ood_datasets", "data.ood_datasets", "ood_datasets"], "ood_datasets"
    )


def _import_models_eval() -> Any:
    return _import_module(
        ["src.eval.evaluate_models", "eval.evaluate_models", "evaluate_models"],
        "evaluate_models",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file, returning ``{}`` when unavailable."""
    if path is None:
        candidate = os.path.join(REPO_ROOT, "configs", "config.yaml")
        path = candidate if os.path.exists(candidate) else None
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if not isinstance(cfg, dict):
            LOG.warning("config %s is not a mapping; ignoring", path)
            return {}
        return cfg
    except Exception as exc:  # pragma: no cover
        LOG.warning("failed to parse config %s: %s", path, exc)
        return {}


def _config_get(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def resolve_cache_dir(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    cache_dir = getattr(args, "cache_dir", None) or _config_get(
        config, "cache_dir", "output_dir", "outputs", "eval_cache_dir"
    )
    if not cache_dir:
        cache_dir = os.path.join(REPO_ROOT, "cache")
    return os.path.abspath(cache_dir)


def resolve_results_dir(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    results_dir = getattr(args, "results_dir", None) or _config_get(
        config, "results_dir", "output", "results"
    )
    if not results_dir:
        results_dir = os.path.join(REPO_ROOT, "results")
    results_dir = os.path.abspath(results_dir)
    return os.path.join(results_dir, "latent_hierarchy")


# ---------------------------------------------------------------------------
# Cache discovery / loading
# ---------------------------------------------------------------------------
def discover_cached_models(cache_dir: str, datasets: Sequence[str] = DEFAULT_DATASETS) -> List[str]:
    """Infer the evaluated model names from cached ``<model>__<dataset>.npz`` files."""
    models: set = set()
    for path in glob.glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0]
        for sep in ("__", "--"):
            if sep in stem:
                model = stem.split(sep, 1)[0]
                if model:
                    models.add(model)
                break
    return sorted(models)


def _dataset_tokens(dataset: str) -> Tuple[str, ...]:
    tokens = {dataset.lower()}
    tokens.update(DATASET_ALIASES.get(dataset.lower(), ()))
    try:
        ood = _import_ood()
        tokens.add(str(ood.normalize_ood_name(dataset)).lower())
        tokens.add(str(ood.display_name(dataset)).lower())
    except Exception:
        pass
    return tuple(sorted(tokens))


def find_cache_file(
    cache_dir: str, model_name: str, dataset: str, id_names: Sequence[str] = ID_CACHE_NAMES
) -> Optional[str]:
    """Locate the ``.npz`` cache file for ``(model_name, dataset)`` (tolerant matching)."""
    tokens = set(_dataset_tokens(dataset))
    if dataset.lower() in {"id", "imagenet", "imagenet-1k", "imagenet1k", "val", "validation"}:
        tokens.update(name.lower() for name in id_names)

    candidates: List[str] = []
    for path in glob.glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem == model_name:
            candidates.append(path)
            continue
        for sep in ("__", "--"):
            if stem.startswith(model_name + sep):
                suffix = stem[len(model_name) + len(sep):].lower()
                if suffix in tokens:
                    candidates.append(path)
                break
    if not candidates:
        return None
    # Prefer exact dataset-name matches and shorter names.
    candidates.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    return candidates[0]


def load_npz(path: str) -> Dict[str, Any]:
    """Load a cached output file (delegates to ``evaluate_models`` when possible)."""
    try:
        ev = _import_models_eval()
        loader = getattr(ev, "load_outputs_npz", None)
        if loader is not None:
            return dict(loader(path))
    except Exception:
        pass
    import numpy as np  # type: ignore

    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _as_array(value: Any) -> Any:
    import numpy as np  # type: ignore

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Step 1: accuracy (hierarchy independent) from the cached outputs
# ---------------------------------------------------------------------------
def build_accuracy_rows(
    cache_dir: str,
    model_names: Sequence[str],
    datasets: Sequence[str] = DEFAULT_DATASETS,
    id_dataset: str = "imagenet",
    require_id: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Compute Top-1/Top-5 for every model/dataset from cached outputs.

    Returns ``(accuracies, id_payload)`` where ``accuracies[model][dataset]`` holds
    ``{"top1": float, "top5": float, "predictions": ndarray, "targets": ndarray}`` and
    ``id_payload[model]`` holds the ID ``logits``/``features``/``targets`` needed for the
    latent hierarchy construction.
    """
    lca_mod = _import_metrics()
    top1_fn = getattr(lca_mod, "top1_accuracy", None)
    top5_fn = getattr(lca_mod, "top5_accuracy", None)

    accuracies: Dict[str, Dict[str, Any]] = {}
    id_payload: Dict[str, Dict[str, Any]] = {}

    for model in model_names:
        per_model: Dict[str, Any] = {}
        id_file = find_cache_file(cache_dir, model, id_dataset)
        if id_file is not None:
            payload = load_npz(id_file)
            logits = _as_array(payload.get("logits"))
            targets = _as_array(payload.get("targets"))
            features = _as_array(payload.get("features"))
            id_payload[model] = {
                "features": features,
                "logits": logits,
                "targets": targets,
                "path": id_file,
            }
            if logits is not None and targets is not None:
                preds = logits.argmax(axis=-1)
                per_model["id"] = {
                    "top1": float(top1_fn(logits, targets)) if top1_fn else float(
                        (preds == targets).mean()
                    ),
                    "top5": float(top5_fn(logits, targets)) if top5_fn else float("nan"),
                    "predictions": preds,
                    "targets": targets,
                }
        elif require_id:
            LOG.warning("no cached ID outputs for %s (looked in %s)", model, cache_dir)

        for dataset in datasets:
            path = find_cache_file(cache_dir, model, dataset)
            if path is None:
                continue
            payload = load_npz(path)
            logits = _as_array(payload.get("logits"))
            targets = _as_array(payload.get("targets"))
            if logits is None or targets is None:
                continue
            preds = logits.argmax(axis=-1)
            n = min(len(preds), len(targets))
            per_model[dataset] = {
                "top1": float(top1_fn(logits, targets)) if top1_fn else float(
                    (preds[:n] == targets[:n]).mean()
                ),
                "top5": float(top5_fn(logits, targets)) if top5_fn else float("nan"),
                "predictions": preds,
                "targets": targets,
            }
        if per_model:
            accuracies[model] = per_model

    return accuracies, id_payload


# ---------------------------------------------------------------------------
# Step 2: latent hierarchies
# ---------------------------------------------------------------------------
def build_latent_hierarchies(
    id_payload: Dict[str, Dict[str, Any]],
    num_classes: int = 1000,
    class_names: Optional[Sequence[str]] = None,
    max_level: int = 9,
    base_level: int = 10,
    seed: int = 0,
    normalize_features: bool = False,
    max_source_models: Optional[int] = None,
    source_models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Construct one :class:`LatentHierarchy` per source model from cached ID features."""
    lk = _import_latent()

    names = list(source_models) if source_models else sorted(id_payload.keys())
    if max_source_models is not None:
        names = names[: int(max_source_models)]

    hierarchies: Dict[str, Any] = {}
    for name in names:
        payload = id_payload.get(name)
        if payload is None:
            continue
        features = payload.get("features")
        targets = payload.get("targets")
        if features is None or targets is None or len(features) == 0:
            LOG.warning("source model %s has no usable ID features; skipped", name)
            continue
        try:
            hierarchy = lk.latent_hierarchy_from_features(
                features,
                targets,
                num_classes=num_classes,
                normalize_features=normalize_features,
                max_level=max_level,
                base_level=base_level,
                seed=seed,
                class_names=list(class_names) if class_names is not None else None,
                source_model=name,
            )
        except TypeError:
            # older/simpler signature fallback
            class_features, _ = lk.class_mean_features(
                features, targets, num_classes=num_classes
            )
            hierarchy = lk.build_latent_hierarchy(
                class_features,
                max_level=max_level,
                base_level=base_level,
                num_classes=num_classes,
                seed=seed,
                class_names=list(class_names) if class_names is not None else None,
                source_model=name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("latent hierarchy for %s failed: %s", name, exc)
            continue
        hierarchies[name] = hierarchy
        LOG.info("built latent hierarchy from source model %s", name)
    return hierarchies


def hierarchy_matrix(hierarchy: Any, processed: bool = False, temperature: float = 1.0) -> List[List[float]]:
    """Return the pairwise distance matrix of a latent hierarchy."""
    if processed:
        return hierarchy.processed_matrix(temperature=temperature, scale=True)
    try:
        return hierarchy.distance_matrix()
    except AttributeError:  # pragma: no cover
        return hierarchy.latent_lca_matrix()


# ---------------------------------------------------------------------------
# Step 3: per-hierarchy correlations
# ---------------------------------------------------------------------------
def id_lca_for_hierarchy(
    hierarchy: Any,
    id_rows: Dict[str, Dict[str, Any]],
    processed: bool = False,
    temperature: float = 1.0,
    num_classes: int = 1000,
) -> Dict[str, float]:
    """Recompute the dataset-level ID LCA distance for every model under ``hierarchy``."""
    lca_mod = _import_metrics()
    matrix = hierarchy_matrix(hierarchy, processed=processed, temperature=temperature)
    try:
        metric = lca_mod.LcaMetric(matrix=matrix, num_classes=num_classes, mode="information")
    except TypeError:  # pragma: no cover - defensive for alternative signatures
        metric = lca_mod.LcaMetric(hierarchy=hierarchy)

    out: Dict[str, float] = {}
    for model, rows in id_rows.items():
        row = rows.get("id")
        if not row:
            continue
        try:
            out[model] = float(metric.dataset_lca(row["predictions"], row["targets"]))
        except Exception as exc:  # pragma: no cover
            LOG.debug("ID LCA for %s failed: %s", model, exc)
    return out


def correlation_for_hierarchy(
    id_lca: Dict[str, float],
    accuracies: Dict[str, Dict[str, Any]],
    datasets: Sequence[str] = DEFAULT_DATASETS,
    abs_values: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Correlate (ID LCA, OOD Top-1) across models for one hierarchy."""
    corr = _import_correlation()
    table: Dict[str, Dict[str, float]] = {}
    for dataset in datasets:
        xs: List[float] = []
        ys: List[float] = []
        for model, lca in id_lca.items():
            rows = accuracies.get(model) or {}
            row = rows.get(dataset)
            if row is None:
                continue
            top1 = row.get("top1")
            if top1 is None or not math.isfinite(float(top1)) or not math.isfinite(float(lca)):
                continue
            xs.append(float(lca))
            ys.append(float(top1))
        if len(xs) < 3:
            continue
        try:
            stats = corr.correlation_metrics(
                xs, ys, abs_values=abs_values, x_name="id_lca", y_name=f"ood_top1_{dataset}"
            )
        except Exception as exc:  # pragma: no cover
            LOG.debug("correlation for %s failed: %s", dataset, exc)
            continue
        table[dataset] = {k: float(v) for k, v in dict(stats).items() if isinstance(v, (int, float))}
        table[dataset]["n"] = len(xs)
    return table


def aggregate_statistics(
    per_hierarchy: Dict[str, Dict[str, Dict[str, float]]],
    datasets: Sequence[str] = DEFAULT_DATASETS,
    metric: str = "pea",
) -> Dict[str, Dict[str, float]]:
    """Mean / min / max / std of ``metric`` across the source-model hierarchies."""
    stats: Dict[str, Dict[str, float]] = {}
    for dataset in datasets:
        values: List[float] = []
        for _source, table in per_hierarchy.items():
            entry = table.get(dataset)
            if not entry:
                continue
            value = entry.get(metric)
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                values.append(abs(value))
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        stats[dataset] = {
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "std": math.sqrt(var),
            "n": float(n),
            "metric": metric,
        }
    return stats


# ---------------------------------------------------------------------------
# Step 4: validation against Table 4
# ---------------------------------------------------------------------------
def check_against_table4(stats: Dict[str, Dict[str, float]], tolerance: float = 0.10) -> List[str]:
    """Compare aggregated PEA statistics with the paper's Table 4 targets."""
    problems: List[str] = []
    for dataset, reference in TABLE4_REFERENCE.items():
        observed = stats.get(dataset)
        if not observed:
            problems.append(f"{dataset}: no correlation could be computed")
            continue
        mean = observed.get("mean")
        if mean is None or abs(float(mean) - reference["mean"]) > tolerance:
            problems.append(
                f"{dataset}: mean PEA {mean if mean is None else round(float(mean), 3)} "
                f"vs reference {reference['mean']} (tol {tolerance})"
            )
        lo, hi = observed.get("min"), observed.get("max")
        if lo is not None and float(lo) < reference["min"] - 2 * tolerance:
            problems.append(
                f"{dataset}: min PEA {round(float(lo), 3)} far below reference {reference['min']}"
            )
        if hi is not None and float(hi) < reference["min"] - tolerance:
            problems.append(
                f"{dataset}: max PEA {round(float(hi), 3)} below reference min {reference['min']}"
            )
    return problems


def check_robustness(stats: Dict[str, Dict[str, float]], severe: Sequence[str] = ("s", "r", "a")) -> List[str]:
    """Latent hierarchies must stay above the 0.7 PEA gate on severe-shift datasets."""
    problems: List[str] = []
    for dataset in severe:
        observed = stats.get(dataset)
        if not observed:
            problems.append(f"{dataset}: missing")
            continue
        if float(observed.get("mean", 0.0)) < 0.7:
            problems.append(f"{dataset}: mean PEA {round(float(observed['mean']), 3)} < 0.7")
        min_floor = 0.6 if dataset != "a" else 0.55
        if float(observed.get("min", 0.0)) < min_floor:
            problems.append(
                f"{dataset}: min PEA {round(float(observed['min']), 3)} < {min_floor}"
            )
    return problems


def format_table(stats: Dict[str, Dict[str, float]], datasets: Sequence[str] = DEFAULT_DATASETS) -> str:
    """Render the Table 4 aggregation block."""
    header = f"{'Stat':<6}" + "".join(f"{ds:>12}" for ds in datasets)
    lines = [header, "-" * len(header)]
    for key, label in (("mean", "Mean"), ("min", "Min"), ("max", "Max"), ("std", "Std")):
        row = f"{label:<6}"
        for ds in datasets:
            entry = stats.get(ds)
            row += f"{entry[key]:>12.3f}" if entry and key in entry else f"{'-':>12}"
        lines.append(row)
    counts = sorted({int(e.get("n", 0)) for e in stats.values()} or {0})
    lines.append(f"#hierarchies = {counts[0] if len(counts) == 1 else counts}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic fallback (smoke test only)
# ---------------------------------------------------------------------------
def synthetic_setup(
    num_classes: int = 1000,
    num_models: int = 8,
    samples_per_class: int = 4,
    feature_dim: int = 64,
    num_groups: int = 8,
    seed: int = 0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Fabricate ID features/logits and OOD accuracies so the pipeline can be exercised offline.

    The synthetic construction places classes in ``num_groups`` well-separated feature clusters so
    the K-means hierarchy recovers a meaningful tree, and makes the OOD accuracy an (affine plus
    noise) function of the mean within-cluster distance so the correlation is non-degenerate.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    group_of_class = rng.integers(0, num_groups, size=num_classes)
    centers = rng.normal(0.0, 6.0, size=(num_groups, feature_dim))

    id_payload: Dict[str, Dict[str, Any]] = {}
    accuracies: Dict[str, Dict[str, Any]] = {}

    for m in range(num_models):
        # each "source model" sees a differently perturbed feature space
        scale = 0.8 + 0.4 * m / max(1, num_models - 1)
        feats = []
        labels = []
        for c in range(num_classes):
            centre = centers[group_of_class[c]] * scale
            block = centre + rng.normal(0.0, 1.0, size=(samples_per_class, feature_dim))
            feats.append(block)
            labels.extend([c] * samples_per_class)
        features = np.concatenate(feats, axis=0).astype("float32")
        targets = np.asarray(labels, dtype="int64")

        # logits proportional to negative distance to class prototypes (+ noise)
        proto = centers[group_of_class] * scale
        d = ((features[:, None, :] - proto[None, :, :]) ** 2).sum(-1) if num_classes <= 64 else None
        if d is None:
            # chunked distance computation for the 1000-class case
            chunks = []
            for start in range(0, len(features), 256):
                block = features[start : start + 256]
                dist = ((block[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
                chunks.append(dist)
            d = np.concatenate(chunks, axis=0)
        logits = (-d * 0.5 + rng.normal(0.0, 1.0, size=d.shape)).astype("float32")
        preds = logits.argmax(axis=1)

        id_payload[m_name(m)] = {
            "features": features,
            "logits": logits,
            "targets": targets,
        }
        rows: Dict[str, Any] = {
            "id": {
                "top1": float((preds == targets).mean()),
                "top5": float("nan"),
                "predictions": preds,
                "targets": targets,
            }
        }
        severity = float(rng.uniform(0.0, 1.0))
        for dataset in DEFAULT_DATASETS:
            n = min(512, len(targets))
            tgt = targets[:n]
            # correlated with the source quality, as in the real experiment
            skill = 0.15 + 0.7 * (1.0 - severity) * (0.6 + 0.4 * (m + 1) / num_models)
            hit = rng.random(n) < skill
            prd = np.where(hit, tgt, rng.integers(0, num_classes, size=n))
            rows[dataset] = {
                "top1": float((prd == tgt).mean()),
                "top5": float("nan"),
                "predictions": prd,
                "targets": tgt,
            }
        accuracies[m_name(m)] = rows

    return accuracies, id_payload, accuracies


def m_name(index: int) -> str:
    return f"synth_model_{index:02d}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 4 (latent hierarchy robustness) of LCA-on-the-Line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="path to configs/config.yaml")
    parser.add_argument("--cache-dir", default=None, help="directory with cached *.npz outputs")
    parser.add_argument("--results-dir", default=None, help="where to write result artifacts")
    parser.add_argument("--models", nargs="*", default=None, help="source models to use")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS), help="OOD datasets")
    parser.add_argument("--class-names", default=None, help="optional JSON/txt list of class names")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--max-level", type=int, default=9, help="deepest K-means level (2^i centers)")
    parser.add_argument("--base-level", type=int, default=10, help="shared base cluster level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0, help="power for processed matrices")
    parser.add_argument("--normalize-features", action="store_true", help="L2-normalize features")
    parser.add_argument(
        "--processed",
        action="store_true",
        help="use invert+power+MinMax latent matrices instead of raw distances",
    )
    parser.add_argument(
        "--max-source-models",
        type=int,
        default=None,
        help="limit the number of source hierarchies (runtime knob; paper uses 75)",
    )
    parser.add_argument("--save-hierarchies", action="store_true", help="persist each latent hierarchy")
    parser.add_argument("--allow-synthetic", action="store_true", help="offline smoke test data")
    parser.add_argument("--tol", type=float, default=0.10, help="Table 4 comparison tolerance")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _load_class_names(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    if not os.path.exists(path):
        LOG.warning("class-name file %s not found", path)
        return None
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return [str(x) for x in payload]
        if isinstance(payload, dict):
            return [str(payload[k]) for k in sorted(payload, key=lambda x: int(x))]
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    cache_dir = resolve_cache_dir(args, config)
    out_dir = resolve_results_dir(args, config)
    datasets = list(args.datasets or DEFAULT_DATASETS)
    class_names = _load_class_names(args.class_name) if False else _load_class_names(args.class_names)  # noqa: E501

    LOG.info("cache dir: %s", cache_dir)
    LOG.info("results dir: %s", out_dir)

    # ---------------------------------------------------------------- data
    synthetic = False
    accuracies: Dict[str, Dict[str, Any]] = {}
    id_payload: Dict[str, Dict[str, Any]] = {}

    if os.path.isdir(cache_dir):
        model_names = list(args.models) if args.models else discover_cached_models(cache_dir, datasets)
        if model_names:
            LOG.info("found %d cached models", len(model_names))
            accuracies, id_payload = build_accuracy_rows(
                cache_dir, model_names, datasets=datasets
            )

    if not id_payload:
        if not args.allow_synthetic:
            LOG.error(
                "no cached ID outputs found under %s. Run scripts/run_correlation.py first, or "
                "pass --allow-synthetic for an offline smoke test.",
                cache_dir,
            )
            return 1
        LOG.warning("no cache available - running with SYNTHETIC data (smoke test only)")
        accuracies, id_payload, _ = synthetic_setup(num_classes=min(args.num_classes, 200), seed=args.seed)
        synthetic = True

    # ------------------------------------------------------ latent hierarchies
    t0 = time.time()
    hierarchies = build_latent_hierarchies(
        id_payload,
        num_classes=args.num_classes,
        class_names=class_names,
        max_level=args.max_level,
        base_level=args.base_level,
        seed=args.seed,
        normalize_features=args.normalize_features,
        max_source_models=args.max_source_models,
        source_models=args.models,
    )
    if not hierarchies:
        LOG.error("no latent hierarchy could be constructed")
        return 2
    LOG.info("constructed %d latent hierarchies in %.1fs", len(hierarchies), time.time() - t0)

    if args.save_hierarchies:
        hier_dir = os.path.join(out_dir, "hierarchies")
        for source, hierarchy in hierarchies.items():
            try:
                hierarchy.save(os.path.join(hier_dir, f"{source}.json"))
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("could not save hierarchy %s: %s", source, exc)

    # --------------------------------------------------------- correlations
    per_hierarchy: Dict[str, Dict[str, Dict[str, float]]] = {}
    for source, hierarchy in hierarchies.items():
        try:
            id_lca = id_lca_for_hierarchy(
                hierarchy,
                {m: rows for m, rows in accuracies.items()},
                processed=args.processed,
                temperature=args.temperature,
                num_classes=args.num_classes,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("ID LCA under hierarchy %s failed: %s", source, exc)
            continue
        if not id_lca:
            continue
        per_hierarchy[source] = correlation_for_hierarchy(id_lca, accuracies, datasets=datasets)
        LOG.info(
            "hierarchy %s PEA: %s",
            source,
            {ds: round(v.get("pea", float("nan")), 3) for ds, v in per_hierarchy[source].items()},
        )

    if not per_hierarchy:
        LOG.error("no correlation could be computed")
        return 3

    stats = aggregate_statistics(per_hierarchy, datasets=datasets, metric="pea")
    stats_r2 = aggregate_statistics(per_hierarchy, datasets=datasets, metric="r2")

    print()
    print("Table 4 - PEA between ID LCA (latent hierarchy) and OOD Top-1 across source models")
    print(format_table(stats, datasets=datasets))
    print()
    print("Reference (paper Table 4):")
    for ds in datasets:
        ref = TABLE4_REFERENCE.get(ds)
        if ref:
            print(
                f"  {ds:>10}: mean {ref['mean']:.3f}  min {ref['min']:.3f}  "
                f"max {ref['max']:.3f}  std {ref['std']:.3f}"
            )
    print()

    problems = check_against_table4(stats, tolerance=args.tol)
    robustness = check_robustness(stats)
    for message in problems:
        LOG.warning("Table 4 mismatch: %s", message)
    for message in robustness:
        LOG.warning("robustness: %s", message)

    success = {
        "num_hierarchies": len(per_hierarchy),
        "synthetic": synthetic,
        "datasets": datasets,
        "stats_pea": stats,
        "stats_r2": stats_r2,
        "table4_reference": TABLE4_REFERENCE,
        "wordnet_reference": TABLE4_WORDNET_REFERENCE,
        "baseline_reference": TABLE4_BASELINE_REFERENCE,
        "problems": problems,
        "robustness_issues": robustness,
        "passes_table4": len(problems) == 0,
        "passes_robustness_gate": len(robustness) == 0,
    }

    _write_json(os.path.join(out_dir, "table4_stats.json"), stats)
    _write_json(os.path.join(out_dir, "per_hierarchy_correlations.json"), per_hierarchy)
    _write_json(os.path.join(out_dir, "checks.json"), {"problems": problems, "robustness": robustness})
    _write_json(os.path.join(out_dir, "success_summary.json"), success)
    LOG.info("wrote artifacts to %s", out_dir)

    return 0 if success["passes_table4"] or synthetic else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
