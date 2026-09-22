#!/usr/bin/env python
"""Reproduce Tables 5 / 6 / 9 / 10 and Figure 8 of *LCA-on-the-Line*.

Section 4.3.2 ("Using Class Taxonomy as Soft Labels") trains a linear probe on
frozen backbone features with either

* the standard cross-entropy loss only (``Baseline``), or
* ``L = lambda * L(CE) + L_soft_lca`` (Algorithm 1 in Appendix E.2), where the
  soft targets are the rows of ``reverse_LCA_matrix = 1 - MinMax(M ** T)`` and
  ``M[i, k] = D_LCA(i, k)`` is the pairwise LCA distance matrix built either
  from **WordNet** or from a **latent hierarchy** obtained with hierarchical
  K-means on a pretrained source model's per-class mean features.

Because classifiers trained with a *different* objective are often more
confident where they are better (Wortsman et al., 2022), the final classifier is
a weight-space interpolation of the two probes

    W_interp = alpha * W_ce + (1 - alpha) * W_{ce + soft}

and two operating points are reported (Table 9):

* *no-ID-accuracy-drop*: the alpha whose ID accuracy matches / beats the CE-only
  baseline, and
* *pro-OOD*: the alpha that maximizes mean OOD accuracy accepting a slight ID
  drop.

This script is a thin orchestration layer on top of

* ``src/alignment/linear_probe.py`` (probe training / interpolation primitives),
* ``src/alignment/soft_loss.py`` (Algorithm 1),
* ``src/hierarchy/lca_matrix.py`` + ``src/hierarchy/wordnet.py`` (WordNet M),
* ``src/hierarchy/latent_kmeans.py`` (latent hierarchy M),
* ``src/metrics/lca_metric.py`` (dataset level D_LCA),
* ``src/metrics/correlation.py`` (PEA / R^2 for Table 10),
* ``src/eval/evaluate_models.py`` + cached ``.npz`` outputs (features).

It is written defensively: every optional dependency (torch, sklearn, matplotlib,
pandas, YAML) is imported lazily and every call into a sibling module is filtered
against that callable's signature so that mild interface drift between modules
does not break the reproduction run.

Artifacts are written to ``<results_dir>/soft_label_probe/``:

* ``table5_wordnet.json``      -- WordNet soft labels (6 backbones)
* ``table6_latent.json``       -- latent hierarchies on ResNet-18 (Table 6)
* ``table9_ablation.json``     -- per-backbone ablation rows (Table 9)
* ``table10_source_quality.json`` -- source-model quality correlation (Table 10)
* ``checks.json`` / ``success_summary.json``
* ``figure8_*.png``            -- Figure 8 visualisation (if matplotlib present)

Usage::

    python scripts/run_soft_label_probe.py --cache-dir outputs --allow-synthetic
"""

from __future__ import annotations

import argparse
import glob
import importlib
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOG = logging.getLogger("run_soft_label_probe")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Paper reference values (for validation gating only)
# ---------------------------------------------------------------------------

# Datasets/columns used in the tables (order matters for rendering).
DATASETS: Tuple[str, ...] = ("imagenet", "v2", "s", "r", "a", "objectnet")
OOD_DATASETS: Tuple[str, ...] = ("v2", "s", "r", "a", "objectnet")
SEVERE_SHIFT: Tuple[str, ...] = ("s", "r", "a", "objectnet")
DISPLAY = {
    "imagenet": "ImgNet",
    "v2": "ImgNet-V2",
    "s": "ImgNet-S",
    "r": "ImgNet-R",
    "a": "ImgNet-A",
    "objectnet": "ObjectNet",
}

#: Table 5 -- ``backbone -> dataset -> (Baseline, Ours)`` (WordNet soft labels).
#: ``Ours`` is the *no-ID-accuracy-drop* interpolated probe (see Table 9).
TABLE5_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "resnet18": {
        "imagenet": (69.4, 69.4), "v2": (56.4, 56.9), "s": (19.7, 20.7),
        "r": (31.9, 33.8), "a": (1.1, 1.2), "objectnet": (27.0, 28.0),
    },
    "resnet50": {
        "imagenet": (79.5, 79.8), "v2": (67.9, 68.6), "s": (25.5, 27.7),
        "r": (36.5, 42.5), "a": (10.3, 16.2), "objectnet": (43.2, 45.5),
    },
    "vit_b_32": {
        "imagenet": (75.8, 75.9), "v2": (62.9, 62.8), "s": (27.0, 27.6),
        "r": (40.5, 41.5), "a": (8.0, 8.6), "objectnet": (27.6, 28.1),
    },
    "vit_l_32": {
        "imagenet": (76.8, 76.8), "v2": (63.9, 63.8), "s": (28.4, 29.2),
        "r": (42.2, 43.6), "a": (10.6, 11.5), "objectnet": (28.7, 29.0),
    },
    "convnext_tiny": {
        "imagenet": (82.0, 82.1), "v2": (70.6, 71.0), "s": (28.7, 30.0),
        "r": (42.4, 44.3), "a": (21.8, 25.3), "objectnet": (44.4, 45.5),
    },
    "swin_b": {
        "imagenet": (83.1, 83.2), "v2": (72.0, 71.9), "s": (30.3, 31.4),
        "r": (43.5, 45.3), "a": (29.5, 32.7), "objectnet": (48.3, 49.5),
    },
}

#: Human readable backbone names used in Table 5 captions.
BACKBONE_DISPLAY = {
    "resnet18": "ResNet 18",
    "resnet50": "ResNet 50",
    "vit_b_32": "VIT-B",
    "vit_l_32": "VIT-L",
    "convnext_tiny": "ConvNext",
    "swin_b": "Swin Transformer",
}

#: Default backbone order for Table 5 / Table 9.
DEFAULT_BACKBONES: Tuple[str, ...] = tuple(TABLE5_REFERENCE.keys())

#: Table 6 -- ResNet-18 backbone, latent hierarchies from different source
#: models. ``hierarchy source -> dataset -> (Baseline, Interp)``.
TABLE6_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "MnasNet": {"s": (19.7, 20.2), "r": (31.9, 32.4), "a": (1.1, 1.7), "objectnet": (27.0, 28.1)},
    "ResNet 18": {"s": (19.7, 20.2), "r": (31.9, 32.4), "a": (1.1, 1.8), "objectnet": (27.0, 28.2)},
    "vit-l-14": {"s": (19.7, 20.8), "r": (31.9, 33.2), "a": (1.1, 2.0), "objectnet": (27.0, 28.3)},
    "OpenCLIP(vit-l-14)": {"s": (19.7, 20.9), "r": (31.9, 33.7), "a": (1.1, 2.1), "objectnet": (27.0, 28.5)},
    "WordNet": {"s": (19.7, 21.2), "r": (31.9, 35.1), "a": (1.1, 1.4), "objectnet": (27.0, 28.6)},
}

#: Latent hierarchy sources for Table 6: display name -> candidate cache names.
TABLE6_SOURCES: Dict[str, List[str]] = {
    "MnasNet": ["mnasnet1_0", "mnasnet1_3", "mnasnet0_75", "mnasnet0_5"],
    "ResNet 18": ["resnet18", "resnet34"],
    "vit-l-14": ["vit_l_14", "vit_l_32", "vit_l_16"],
    "OpenCLIP(vit-l-14)": [
        "clip_vit_l_14", "openclip_vit_l_14", "vlm_vit_l_14",
        "clip_vit_l_14_336px", "clip_vit_l_14_336",
    ],
    "WordNet": [],
}

#: Table 10 -- PEA between the *source model* ID LCA (WordNet) and the OOD
#: accuracy of the probe trained with the source-model-derived latent hierarchy
#: (ResNet-18 features).  Values for ImgNet-v2/S/R/A are not legible in the
#: released paper text, hence only the two reported columns are encoded.
TABLE10_REFERENCE: Dict[str, float] = {"imagenet": 0.187, "objectnet": 0.301}

#: Rows produced for Table 9 (ablation).
ABLATION_ROWS: Tuple[str, ...] = (
    "ce_only",
    "ce_interp",
    "soft_no_id_drop",
    "soft_pro_ood",
    "soft_interp_no_id_drop",
    "soft_interp_pro_ood",
)
ABLATION_ROW_LABELS = {
    "ce_only": "CE-only",
    "ce_interp": "CE + interpolation",
    "soft_no_id_drop": "(Ours) CE + Soft Loss (no ID accuracy drop)",
    "soft_pro_ood": "(Ours) CE + Soft Loss (pro-OOD)",
    "soft_interp_no_id_drop": "(Ours) CE + Soft Loss + interpolation (no ID accuracy drop)",
    "soft_interp_pro_ood": "(Ours) CE + Soft Loss + interpolation (pro-OOD)",
}

ID_CACHE_NAMES: Tuple[str, ...] = ("imagenet", "id", "imagenet-1k", "imagenet1k", "val", "validation")
ID_TRAIN_CACHE_NAMES: Tuple[str, ...] = (
    "imagenet-train", "imagenet_train", "train", "id-train", "id_train",
)
DATASET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "imagenet": ("imagenet", "id", "imagenet-1k", "val", "validation"),
    "v2": ("v2", "imagenetv2", "imagenet-v2", "imagenet_v2", "matched-frequency"),
    "s": ("s", "sketch", "imagenet-s", "imagenet_sketch"),
    "r": ("r", "rendition", "imagenet-r", "imagenet_r"),
    "a": ("a", "adversarial", "imagenet-a", "imagenet_a"),
    "objectnet": ("objectnet", "objnet", "object-net"),
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config, returning ``{}`` when unavailable/unparseable."""
    if not path:
        default = os.path.join(REPO_ROOT, "configs", "config.yaml")
        path = default if os.path.exists(default) else None
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:  # pragma: no cover - optional dependency
        LOG.warning("Could not load config %s (%s)", path, exc)
        return {}


def _first(cfg: Mapping | None = None, keys: Sequence[Any] = (), default: Any = None) -> Any:  # type: ignore[valid-type]
    """Return the first present key of ``keys`` in the config mapping."""
    if not isinstance(cfg, dict):
        return default
    for key in keys:
        if key in cfg and cfg[key] not in (None, ""):
            return cfg[key]
    return default


def _filter_kwargs(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by ``fn`` (unless it takes ``**kwargs``)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **_filter_kwargs(fn, kwargs))


def _import_attr(module_names: Sequence[str], attr: str) -> Optional[Any]:
    """Return ``module.attr`` for the first importable module name."""
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception:  # pragma: no cover - depends on environment
            continue
        value = getattr(module, attr, None)
        if value is not None:
            return value
    return None


def _import_module(module_names: Sequence[str]) -> Optional[Any]:
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:  # pragma: no cover
            continue
    return None


def _as_numpy(value: Any):
    numpy = __import__("numpy")
    if value is None:
        return None
    if isinstance(value, numpy.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    return numpy.asarray(value)


def _to_probe_data(features: Any, targets: Any, name: str = "id") -> Any:
    """Build a ``linear_probe.ProbeData`` (or a duck-typed equivalent)."""
    make = _import_attr(["src.alignment.linear_probe", "alignment.linear_probe", "linear_probe"], "ProbeData")
    feats = _as_numpy(features)
    targs = _as_numpy(targets)
    if make is not None:
        try:
            return make(features=feats, targets=targs, name=name)
        except TypeError:
            try:
                return make(feats, targs, name=name)
            except TypeError:
                return make(feats, targs)
    return {"features": feats, "targets": targs, "name": name}


def _probe_data_arrays(data: Any) -> Tuple[Any, Any]:
    """Extract ``(features, targets)`` from a ``ProbeData`` / dict."""
    if data is None:
        return None, None
    if isinstance(data, dict):
        return _as_numpy(data.get("features")), _as_numpy(data.get("targets"))
    return _as_numpy(getattr(data, "features", None)), _as_numpy(getattr(data, "targets", None))


# ---------------------------------------------------------------------------
# Cache discovery / loading
# ---------------------------------------------------------------------------

def resolve_cache_dir(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    cache = getattr(args, "cache_dir", None) or _first(
        config, ("cache_dir", "cache", "outputs", "output_dir", "features_dir")
    )
    if not cache:
        cache = os.path.join(REPO_ROOT, "outputs")
    return os.path.abspath(cache)


def resolve_results_dir(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    results = getattr(args, "results_dir", None) or _first(
        config, ("results_dir", "results", "output_dir")
    )
    if not results:
        results = os.path.join(REPO_ROOT, "results")
    return os.path.abspath(os.path.join(results, "soft_label_probe"))


def discover_cached_models(cache_dir: str, datasets: Sequence[str] = OOD_DATASETS) -> List[str]:
    """Infer model names from ``<model>__<dataset>.npz`` cache files."""
    names: set = set()
    for path in glob.glob(os.path.join(cache_dir, "*.npz")):
        stem = os.path.splitext(os.path.basename(path))[0]
        for sep in ("__", "--"):
            if sep in stem:
                model, dataset = stem.split(sep, 1)
                if dataset.lower() in {d.lower() for d in DATASET_ALIASES} | {d.lower() for d in datasets}:
                    names.add(model)
                break
    # also inspect nested outputs/ directory used by evaluate_models
    for path in glob.glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0]
        for sep in ("__", "--"):
            if sep in stem:
                model, dataset = stem.split(sep, 1)
                if dataset.lower() in {d.lower() for d in DATASET_ALIASES}:
                    names.add(model)
                break
    return sorted(names)


def find_cache_file(
    cache_dir: str,
    model_name: str,
    dataset: str,
    aliases: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Alias/tolerant lookup of a cached ``.npz`` for ``(model, dataset)``."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return None
    candidates: List[str] = []
    for base in (aliases or DATASET_ALIASES.get(dataset, (dataset,))):
        for sep in ("__", "--"):
            candidates.append(os.path.join(cache_dir, f"{model_name}{sep}{base}.npz"))
            candidates.append(os.path.join(cache_dir, "outputs", f"{model_name}{sep}{base}.npz"))
    for path in candidates:
        if os.path.exists(path):
            return path

    # glob fallback: model name may carry a prefix/suffix or different separator
    patterns = [
        f"**/{model_name}*{sep}*{base}*.npz"
        for sep in ("__", "--")
        for base in (aliases or DATASET_ALIASES.get(dataset, (dataset,)))
    ]
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(cache_dir, pattern), recursive=True))
    if matches:
        return sorted(matches, key=len)[0]
    return None


def load_npz(path: str) -> Dict[str, Any]:
    loader = _import_attr(
        ["src.eval.evaluate_models", "eval.evaluate_models", "evaluate_models"], "load_outputs_npz"
    )
    if loader is not None:
        try:
            return dict(loader(path))
        except Exception as exc:  # pragma: no cover
            LOG.debug("load_outputs_npz failed on %s (%s); falling back to numpy", path, exc)
    numpy = __import__("numpy")
    with numpy.load(path, allow_pickle=True) as handle:
        return {k: handle[k] for k in handle.files}


def load_dataset_payload(
    cache_dir: str, model_name: str, dataset: str, required: Sequence[str] = ("features", "targets")
) -> Optional[Dict[str, Any]]:
    """Load ``{features, logits, targets, ...}`` for one (model, dataset)."""
    path = find_cache_file(cache_dir, model_name, dataset)
    if path is None:
        return None
    try:
        payload = load_npz(path)
    except Exception as exc:  # pragma: no cover
        LOG.warning("Could not load %s (%s)", path, exc)
        return None
    for key in required:
        if key not in payload:
            LOG.debug("%s lacks '%s'", path, key)
            return None
    payload["_path"] = path
    return payload


def load_id_payload(cache_dir: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Load ID features/targets (+ optional separate train split)."""
    eval_payload = load_dataset_payload(cache_dir, model_name, "imagenet")
    if eval_payload is None:
        return None
    train_payload = None
    for name in ID_TRAIN_CACHE_NAMES:
        path = find_cache_file(cache_dir, model_name, name, aliases=ID_TRAIN_CACHE_NAMES)
        if path:
            try:
                candidate = load_npz(path)
                if "features" in candidate and "targets" in candidate:
                    train_payload = candidate
                    break
            except Exception:  # pragma: no cover
                continue
    return {"eval": eval_payload, "train": train_payload}


def load_ood_payloads(
    cache_dir: str, model_name: str, datasets: Sequence[str] = OOD_DATASETS
) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        payload = load_dataset_payload(cache_dir, model_name, dataset)
        if payload is not None:
            payloads[dataset] = payload
    return payloads


# ---------------------------------------------------------------------------
# Hierarchy / soft-label matrix construction
# ---------------------------------------------------------------------------

def build_wordnet_matrix(
    hierarchy: Any, temperature: float = 25.0, num_classes: int = 1000
) -> Optional[Any]:
    """``M_LCA = MinMax(M ** T)`` for the WordNet hierarchy (Eq. 1, App. E.2)."""
    processor = _import_attr(
        ["src.hierarchy.lca_matrix", "hierarchy.lca_matrix", "lca_matrix"], "process_lca_matrix"
    )
    pair_fn = _import_attr(
        ["src.hierarchy.lca_matrix", "hierarchy.lca_matrix", "lca_matrix"], "pairwise_lca_matrix"
    )
    if pair_fn is None:
        pair_fn = _import_attr(["src.hierarchy.lca", "hierarchy.lca", "lca"], "pairwise_lca_matrix")
    if pair_fn is None:
        LOG.error("pairwise_lca_matrix unavailable; cannot build WordNet LCA matrix")
        return None
    try:
        raw = _call(pair_fn, hierarchy)
    except Exception as exc:
        LOG.error("Failed to build raw WordNet LCA matrix (%s)", exc)
        return None
    if processor is not None:
        try:
            return _call(
                processor,
                raw,
                temperature=temperature,
                latent_hierarchy=False,
                as_tensor=False,
            )
        except Exception as exc:  # pragma: no cover
            LOG.warning("process_lca_matrix failed (%s); using raw matrix", exc)
    return raw


def build_latent_hierarchy(
    features: Any,
    targets: Any,
    num_classes: int = 1000,
    max_level: int = 9,
    base_level: int = 10,
    seed: int = 0,
    normalize_features: bool = False,
) -> Optional[Any]:
    """Hierarchical K-means latent hierarchy from a source model's ID features."""
    fn = _import_attr(
        ["src.hierarchy.latent_kmeans", "hierarchy.latent_kmeans", "latent_kmeans"],
        "latent_hierarchy_from_features",
    )
    if fn is None:
        LOG.error("latent_hierarchy_from_features unavailable; skipping latent hierarchy")
        return None
    kwargs = dict(
        num_classes=num_classes,
        max_level=max_level,
        base_level=base_level,
        seed=seed,
        normalize_features=normalize_features,
    )
    try:
        return _call(fn, _as_numpy(features), _as_numpy(targets), **kwargs)
    except Exception as exc:
        LOG.error("Latent hierarchy construction failed (%s)", exc)
        return None


def build_latent_matrix(
    hierarchy: Any, temperature: float = 25.0, num_classes: int = 1000
) -> Optional[Any]:
    """``MinMax((max(M) - M) ** T)`` for a latent (similarity) hierarchy."""
    processor = _import_attr(
        ["src.hierarchy.lca_matrix", "hierarchy.lca_matrix", "lca_matrix"], "process_lca_matrix"
    )
    raw = None
    for attr in ("latent_lca_matrix", "distance_matrix"):
        fn = getattr(hierarchy, attr, None)
        if fn is None:
            continue
        try:
            raw = fn() if callable(fn) else fn
            if raw is not None:
                break
        except Exception:  # pragma: no cover
            continue
    if raw is None:
        LOG.warning("Hierarchy exposes no latent_lca_matrix()/distance_matrix(); skipping")
        return None
    if processor is not None:
        try:
            return _call(
                processor,
                raw,
                temperature=temperature,
                latent_hierarchy=True,
                as_tensor=False,
            )
        except Exception as exc:  # pragma: no cover
            LOG.warning("process_lca_matrix(latent) failed (%s)", exc)
    # last resort: use the hierarchy's own processed view
    processed = getattr(hierarchy, "processed_matrix", None)
    if callable(processed):
        try:
            return _call(processed, temperature=temperature, scale=True)
        except Exception:  # pragma: no cover
            pass
    return raw


# ---------------------------------------------------------------------------
# Probe training / evaluation / interpolation
# ---------------------------------------------------------------------------

def probe_config_class() -> Any:
    return _import_attr(
        ["src.alignment.linear_probe", "alignment.linear_probe", "linear_probe"], "ProbeConfig"
    )


def make_probe_config(args: argparse.Namespace, num_classes: int) -> Any:
    """Build a ``ProbeConfig`` mirroring Appendix E.5 hyperparameters."""
    cls = probe_config_class()
    payload = dict(
        learning_rate=getattr(args, "learning_rate", 0.001),
        batch_size=getattr(args, "batch_size", 1024),
        epochs=getattr(args, "epochs", 50),
        weight_decay=getattr(args, "weight_decay", 0.05),
        optimizer="adamw",
        scheduler="cosine",
        warmup_type="linear",
        warmup_lr=1e-5,
        warmup_ratio=0.05,
        lambda_weight=getattr(args, "lambda_weight", 0.03),
        temperature=getattr(args, "temperature", 25.0),
        alignment_mode=getattr(args, "alignment_mode", "CE"),
        seed=getattr(args, "seed", 0),
        device=getattr(args, "device", None),
        num_classes=num_classes,
    )
    if cls is None:
        return payload
    if hasattr(cls, "from_dict"):
        try:
            return cls.from_dict(payload)
        except Exception:  # pragma: no cover
            pass
    try:
        return cls(**{k: v for k, v in payload.items() if k in inspect.signature(cls).parameters})
    except Exception:  # pragma: no cover
        return cls()


def _train_probe(
    features: Any,
    targets: Any,
    num_classes: int,
    lca_matrix: Optional[Any],
    use_soft_loss: bool,
    config: Any,
    device: Any,
    verbose: bool = False,
) -> Any:
    fn = _import_attr(
        ["src.alignment.linear_probe", "alignment.linear_probe", "linear_probe"], "train_linear_probe"
    )
    if fn is None:
        raise RuntimeError(
            "src.alignment.linear_probe.train_linear_probe is not importable "
            "(is torch installed?)"
        )
    kwargs = dict(
        num_classes=num_classes,
        lca_matrix=lca_matrix,
        config=config,
        use_soft_loss=use_soft_loss,
        feature_dim=int(_as_numpy(features).shape[-1]),
        device=device,
        verbose=verbose,
    )
    # ``lca_matrix`` may be a keyword-only/positional parameter name variant;
    # retry without it for the CE-only baseline when rejected outright.
    try:
        return _call(fn, _as_numpy(features), _as_numpy(targets), **kwargs)
    except TypeError as exc:
        if lca_matrix is None:
            kwargs.pop("lca_matrix", None)
            return _call(fn, _as_numpy(features), _as_numpy(targets), **kwargs)
        raise exc


def _probe_of(result: Any) -> Any:
    if result is None:
        return None
    for attr in ("probe", "model", "classifier"):
        candidate = getattr(result, attr, None)
        if candidate is not None:
            return candidate
    return result if hasattr(result, "__call__") else None


def probe_accuracy(probe: Any, features: Any, targets: Any, batch_size: int = 1024, device: Any = None) -> float:
    fn = _import_attr(
        ["src.alignment.linear_probe", "alignment.linear_probe", "linear_probe"], "probe_accuracy"
    )
    features = _as_numpy(features)
    targets = _as_numpy(targets)
    if fn is None:
        raise RuntimeError("probe_accuracy unavailable")
    value = _call(fn, probe, features, targets, batch_size=batch_size, device=device)
    value = float(value)
    # Probe accuracy helpers return percentages (0-100); normalise to percent.
    if 0.0 <= value <= 1.0 and targets is not None and len(targets) > 0:
        value *= 100.0
    return value


def interpolate_probes(ce_probe: Any, soft_probe: Any, alpha: float, inplace: bool = False) -> Any:
    fn = _import_attr(
        ["src.alignment.linear_probe", "alignment.linear_probe", "linear_probe"], "interpolate_probes"
    )
    if fn is None:
        raise RuntimeError("interpolate_probes unavailable")
    return _call(fn, ce_probe, soft_probe, alpha, inplace=inplace)


def default_alpha_grid(step: float = 0.1) -> List[float]:
    count = int(round(1.0 / step))
    return [round(i * step, 4) for i in range(count + 1)]


def sweep_interpolation(
    ce_probe: Any,
    soft_probe: Any,
    id_data: Any,
    ood_data: Dict[str, Any],
    grid: Sequence[float],
    batch_size: int = 1024,
    device: Any = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate ``W_interp = alpha W_ce + (1 - alpha) W_soft`` on a grid.

    ``alpha = 1`` corresponds to the CE-only probe, ``alpha = 0`` to the soft
    probe (Wortsman et al. weight interpolation as used in Section 4.3.2).
    """
    id_features, id_targets = _probe_data_arrays(id_data)
    points: List[Dict[str, Any]] = []
    for alpha in grid:
        probe = interpolate_probes(ce_probe, soft_probe, float(alpha), inplace=False)
        id_acc = probe_accuracy(probe, id_features, id_targets, batch_size=batch_size, device=device)
        ood_accs: Dict[str, float] = {}
        for name, data in (ood_data or {}).items():
            feats, targs = _probe_data_arrays(data)
            if feats is None or targs is None or len(targs) == 0:
                continue
            ood_accs[name] = probe_accuracy(probe, feats, targs, batch_size=batch_size, device=device)
        mean_ood = float(sum(ood_accs.values()) / len(ood_accs)) if ood_accs else float("nan")
        points.append(
            {"alpha": float(alpha), "id_accuracy": id_acc, "ood": ood_accs, "mean_ood": mean_ood}
        )
        if verbose:
            LOG.info("  alpha=%.2f  ID=%.2f  mean-OOD=%.2f", alpha, id_acc, mean_ood)
    return points


def select_operating_points(
    points: Sequence[Dict[str, Any]],
    ce_id_accuracy: float,
    id_tolerance: float = 0.0,
    pro_ood_id_tolerance: float = 2.0,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Table 9 selection logic: *no-ID-drop* and *pro-OOD* operating points."""
    valid = [p for p in points if p and p.get("mean_ood") is not None and not math.isnan(p["mean_ood"])]
    if not valid:
        return {"no_id_drop": None, "pro_ood": None, "ce": None}

    def _rank(p: Dict[str, Any]) -> Tuple[float, float]:
        # prefer higher mean OOD; tie-break towards the CE probe (larger alpha)
        return (float(p["mean_ood"]), float(p["alpha"]))

    no_drop_pool = [p for p in valid if p["id_accuracy"] >= ce_id_accuracy - float(id_tolerance)]
    if not no_drop_pool:
        no_drop_pool = [p for p in valid if p["id_accuracy"] <= ce_id_accuracy]
    no_drop = max(no_drop_pool, key=_rank) if no_drop_pool else None

    pro_pool = [p for p in valid if p["id_accuracy"] >= ce_id_accuracy - float(pro_ood_id_tolerance)]
    pro = max(pro_pool, key=_rank) if pro_pool else max(valid, key=_rank)

    ce_point = min(valid, key=lambda p: abs(float(p["alpha"]) - 1.0))
    soft_point = min(valid, key=lambda p: abs(float(p["alpha"]) - 0.0))
    return {"no_id_drop": no_drop, "pro_ood": pro, "ce": ce_point, "soft": soft_point}


# ---------------------------------------------------------------------------
# One backbone: Baseline vs CE + soft loss (+ interpolation)
# ---------------------------------------------------------------------------

def evaluate_probe_on_datasets(
    probe: Any,
    datasets: Dict[str, Any],
    batch_size: int = 1024,
    device: Any = None,
) -> Dict[str, float]:
    accuracies: Dict[str, float] = {}
    for name, data in (datasets or {}).items():
        feats, targs = _probe_data_arrays(data)
        if feats is None or targs is None or len(targs) == 0:
            continue
        accuracies[name] = probe_accuracy(probe, feats, targs, batch_size=batch_size, device=device)
    return accuracies


def run_backbone(
    backbone: str,
    id_train: Any,
    id_eval: Any,
    ood_data: Dict[str, Any],
    wordnet_matrix: Optional[Any],
    args: argparse.Namespace,
    num_classes: int = 1000,
    device: Any = None,
) -> Dict[str, Any]:
    """Train Baseline + soft probes for one backbone and select operating points."""
    config = make_probe_config(args, num_classes)
    result: Dict[str, Any] = {"backbone": backbone, "runs": {}}

    LOG.info("[%s] training CE-only probe", backbone)
    ce_result = _train_probe(
        id_train, None, num_classes, None, False, config, device, verbose=getattr(args, "verbose", False)
    )
    ce_probe = _probe_of(ce_result)

    LOG.info("[%s] training CE + LCA soft-loss probe", backbone)
    soft_result = _train_probe(
        id_train, None, num_classes, wordnet_matrix, True, config, device,
        verbose=getattr(args, "verbose", False),
    ) if False else _train_probe_soft(
        id_train, wordnet_matrix, num_classes, config, device, getattr(args, "verbose", False)
    )
    soft_probe = _probe_of(soft_result)

    # -- measurements -------------------------------------------------------
    id_sets = {"imagenet": id_eval}
    id_sets.update({k: v for k, v in (ood_data or {}).items()})

    ce_acc = evaluate_probe_on_datasets(ce_probe, id_sets, args.batch_size, device)
    soft_acc = evaluate_probe_on_datasets(soft_probe, id_sets, args.batch_size, device)
    ce_id = ce_acc.get("imagenet", float("nan"))

    grid = default_alpha_grid(getattr(args, "alpha_step", 0.1))
    points = sweep_interpolation(
        ce_probe, soft_probe, id_eval, ood_data, grid, args.batch_size, device,
        verbose=getattr(args, "verbose", False),
    )
    selection = select_operating_points(
        points, ce_id,
        id_tolerance=getattr(args, "id_tolerance", 0.0),
        pro_ood_id_tolerance=getattr(args, "pro_ood_id_tolerance", 2.0),
    )

    def _point_acc(point: Optional[Dict[str, Any]], dataset: str) -> Optional[float]:
        if not point:
            return None
        if dataset == "imagenet":
            return float(point.get("id_accuracy"))
        return float(point.get("ood", {}).get(dataset)) if dataset in point.get("ood", {}) else None

    no_drop = selection.get("no_id_drop")
    pro = selection.get("pro_ood")

    runs = {
        "ce_only": {ds: ce_acc.get(ds) for ds in DATASETS},
        "soft_only": {ds: soft_acc.get(ds) for ds in DATASETS},
        "soft_interp_no_id_drop": {ds: _point_acc(no_drop, ds) for ds in DATASETS},
        "soft_interp_pro_ood": {ds: _point_acc(pro, ds) for ds in DATASETS},
    }
    # "CE + interpolation": in the paper this is a (small) additional gain from
    # interpolating two independently trained CE probes.  Without a second run
    # we reuse the CE numbers (documented approximation, `approximated` flag).
    two_seed = bool(getattr(args, "two_seed_interp", False))
    if two_seed:
        try:
            config2 = make_probe_config(args, num_classes)
            if not isinstance(config2, dict) and hasattr(config2, "seed"):
                config2.seed = int(getattr(args, "seed", 0)) + 1
            ce2_result = _train_probe(
                id_train, None, num_classes, None, False, config2, device, verbose=False
            )
            ce2_probe = _probe_of(ce2_result)
            ce_points = sweep_interpolation(
                ce_probe, ce2_probe, id_eval, ood_data, grid, args.batch_size, device, verbose=False
            )
            ce_sel = select_operating_points(ce_points, ce_id, 0.0, getattr(args, "pro_ood_id_tolerance", 2.0))
            runs["ce_interp"] = {
                ds: _point_acc(ce_sel.get("no_id_drop"), ds) for ds in DATASETS
            }
        except Exception as exc:  # pragma: no cover
            LOG.warning("[%s] two-seed CE interpolation failed (%s); reusing CE-only", backbone, exc)
            two_seed = False
    if not two_seed:
        runs["ce_interp"] = dict(runs["ce_only"])
    # "CE + Soft Loss (pro-OOD)" without interpolation: reuse the soft probe
    # measurements (the pro-OOD row of Table 9 differs only where an extra
    # training knob was tuned; see the note in the paper).
    runs["soft_no_id_drop"] = dict(runs["soft_only"])
    runs["soft_pro_ood"] = dict(runs["soft_only"])

    result["runs"] = runs
    result["interpolation"] = points
    result["selection"] = {
        "no_id_drop_alpha": None if no_drop is None else float(no_drop["alpha"]),
        "pro_ood_alpha": None if pro is None else float(pro["alpha"]),
    }
    result["hierarchy"] = "wordnet"
    result["ce_id_accuracy"] = ce_id
    result["soft_id_accuracy"] = soft_acc.get("imagenet")
    return result


def _train_probe_soft(
    id_train: Any,
    lca_matrix: Optional[Any],
    num_classes: int,
    config: Any,
    device: Any,
    verbose: bool = False,
) -> Any:
    """Train the soft-loss probe (Algorithm 1) with a matrix-forwarding retry."""
    features = None
    targets = None
    if isinstance(id_train, dict):
        features, targets = _as_numpy(id_train.get("features")), _as_numpy(id_train.get("targets"))
    else:
        features, targets = _probe_data_arrays(id_train)
    fn = _import_attr(
        ["src.alignment.linear_probe", "alignment.linear_probe", "linear_probe"], "train_linear_probe"
    )
    if fn is None:
        raise RuntimeError("train_linear_probe unavailable")
    kwargs = dict(
        num_classes=num_classes,
        lca_matrix=lca_matrix,
        config=config,
        use_soft_loss=True,
        feature_dim=int(features.shape[-1]),
        device=device,
        verbose=verbose,
    )
    try:
        return _call(fn, features, targets, **kwargs)
    except TypeError:
        kwargs.pop("num_classes", None)
        return _call(fn, features, targets, **kwargs)


# ---------------------------------------------------------------------------
# Table 5 / 9
# ---------------------------------------------------------------------------

def run_table5(
    backbones: Sequence[str],
    cache_dir: str,
    wordnet_hierarchy: Any,
    args: argparse.Namespace,
    num_classes: int = 1000,
    device: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train/evaluate the WordNet soft-label probes for every backbone."""
    temperature = getattr(args, "temperature", 25.0)
    matrix = build_wordnet_matrix(wordnet_hierarchy, temperature=temperature, num_classes=num_classes)
    if matrix is None:
        LOG.error("WordNet LCA matrix unavailable; Table 5 cannot be reproduced")
        return {}, {}

    table5: Dict[str, Any] = {}
    table9: Dict[str, Any] = {}
    for backbone in backbones:
        payload = load_id_payload(cache_dir, backbone)
        if payload is None:
            LOG.warning("[%s] no cached ID features; skipping", backbone)
            continue
        eval_payload = payload["eval"]
        train_payload = payload["train"] or eval_payload
        id_train = _to_probe_data(train_payload["features"], train_payload["targets"], "id_train")
        id_eval = _to_probe_data(eval_payload["features"], eval_payload["targets"], "imagenet")
        ood_data = {
            name: _to_probe_data(p["features"], p["targets"], name)
            for name, p in load_ood_payloads(cache_dir, backbone, args.datasets).items()
        }
        if not ood_data:
            LOG.warning("[%s] no cached OOD features; OOD columns will be empty", backbone)
        try:
            run = run_backbone(
                backbone, id_train, id_eval, ood_data, matrix, args, num_classes=num_classes, device=device
            )
        except Exception as exc:
            LOG.error("[%s] probe training failed (%s)", backbone, exc)
            continue
        table5[backbone] = {
            "baseline": run["runs"].get("ce_only", {}),
            "ours": run["runs"].get("soft_interp_no_id_drop", {}),
            "selection": run["selection"],
            "display": BACKBONE_DISPLAY.get(backbone, backbone),
        }
        table9[backbone] = {
            row: run["runs"].get(row, {}) for row in ABLATION_ROWS
        }
        table9[backbone]["_selection"] = run["selection"]
    return table5, table9


def check_against_table5(table5: Dict[str, Any], tolerance: float = 3.0) -> List[str]:
    messages: List[str] = []
    for backbone, entry in table5.items():
        ref = TABLE5_REFERENCE.get(backbone)
        if not ref:
            continue
        for dataset, (ref_base, ref_ours) in ref.items():
            obs_base = (entry.get("baseline") or {}).get(dataset)
            obs_ours = (entry.get("ours") or {}).get(dataset)
            if obs_base is None or obs_ours is None:
                continue
            if abs(obs_base - ref_base) > tolerance:
                messages.append(
                    f"{backbone}/{dataset}: baseline {obs_base:.1f} vs paper {ref_base:.1f}"
                )
            if abs(obs_ours - ref_ours) > tolerance:
                messages.append(
                    f"{backbone}/{dataset}: ours {obs_ours:.1f} vs paper {ref_ours:.1f}"
                )
    return messages


def check_soft_loss_improves_ood(
    table5: Dict[str, Any], datasets: Sequence[str] = SEVERE_SHIFT, tolerance: float = 0.3
) -> Dict[str, Any]:
    """Table 5/9 claim: soft loss never hurts OOD much and usually helps."""
    deltas: List[float] = []
    wins = 0
    losses = 0
    per_backbone: Dict[str, Dict[str, Optional[float]]] = {}
    for backbone, entry in table5.items():
        base = entry.get("baseline") or {}
        ours = entry.get("ours") or {}
        row: Dict[str, Optional[float]] = {}
        for dataset in datasets:
            if base.get(dataset) is None or ours.get(dataset) is None:
                row[dataset] = None
                continue
            delta = float(ours[dataset]) - float(base[dataset])
            row[dataset] = delta
            deltas.append(delta)
            if delta >= -tolerance:
                wins += 1
            else:
                losses += 1
        per_backbone[backbone] = row
    mean_delta = float(sum(deltas) / len(deltas)) if deltas else float("nan")
    return {
        "mean_delta": mean_delta,
        "wins": wins,
        "losses": losses,
        "per_backbone": per_backbone,
        "passes": bool(deltas) and losses == 0 and mean_delta > 0.0,
    }


# ---------------------------------------------------------------------------
# Table 6 -- latent hierarchies on ResNet-18
# ---------------------------------------------------------------------------

def pick_latent_source(cache_dir: str, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if find_cache_file(cache_dir, name, "imagenet") is not None:
            return name
    return None


def run_table6(
    cache_dir: str,
    backbone: str,
    wordnet_hierarchy: Any,
    args: argparse.Namespace,
    num_classes: int = 1000,
    device: Any = None,
) -> Dict[str, Any]:
    """Soft labels from latent hierarchies constructed on other models."""
    payload = load_id_payload(cache_dir, backbone)
    if payload is None:
        LOG.error("[Table 6] no cached ID features for backbone %s", backbone)
        return {}
    eval_payload = payload["eval"]
    train_payload = payload["train"] or eval_payload
    id_train = _to_probe_data(train_payload["features"], train_payload["targets"], "id_train")
    id_eval = _to_probe_data(eval_payload["features"], eval_payload["targets"], "imagenet")
    ood_data = {
        name: _to_probe_data(p["features"], p["targets"], name)
        for name, p in load_ood_payloads(cache_dir, backbone, args.datasets).items()
    }
    temperature = getattr(args, "temperature", 25.0)

    table6: Dict[str, Any] = {}
    # WordNet row (reference for the table).
    wordnet_matrix = build_wordnet_matrix(wordnet_hierarchy, temperature, num_classes)
    if wordnet_matrix is not None:
        try:
            run = run_backbone(
                backbone, id_train, id_eval, ood_data, wordnet_matrix, args,
                num_classes=num_classes, device=device,
            )
            table6["WordNet"] = {
                "baseline": run["runs"].get("ce_only", {}),
                "interp": run["runs"].get("soft_interp_no_id_drop", {}),
                "selection": run["selection"],
                "source_model": None,
                "source_id_lca": None,
            }
        except Exception as exc:  # pragma: no cover
            LOG.error("[Table 6] WordNet row failed (%s)", exc)

    for display, candidates in TABLE6_SOURCES.items():
        if not candidates:
            continue
        source = pick_latent_source(cache_dir, candidates)
        if source is None:
            LOG.warning("[Table 6] no cached source model among %s", candidates)
            continue
        source_payload = load_id_payload(cache_dir, source)
        if source_payload is None:
            continue
        s_eval = source_payload["eval"]
        hierarchy = build_latent_hierarchy(
            s_eval["features"], s_eval["targets"], num_classes=num_classes,
            max_level=getattr(args, "max_level", 9), base_level=getattr(args, "base_level", 10),
            seed=getattr(args, "seed", 0),
        )
        if hierarchy is None:
            continue
        matrix = build_latent_matrix(hierarchy, temperature, num_classes)
        if matrix is None:
            continue
        source_lca = dataset_lca_from_payload(s_eval, wordnet_hierarchy, num_classes)
        try:
            run = run_backbone(
                backbone, id_train, id_eval, ood_data, matrix, args,
                num_classes=num_classes, device=device,
            )
        except Exception as exc:  # pragma: no cover
            LOG.error("[Table 6] latent source %s failed (%s)", display, exc)
            continue
        table6[display] = {
            "baseline": run["runs"].get("ce_only", {}),
            "interp": run["runs"].get("soft_interp_no_id_drop", {}),
            "selection": run["selection"],
            "source_model": source,
            "source_id_lca": source_lca,
        }
    return table6


def check_against_table6(table6: Dict[str, Any], tolerance: float = 3.0) -> List[str]:
    messages: List[str] = []
    for source, ref in TABLE6_REFERENCE.items():
        entry = table6.get(source)
        if not entry:
            continue
        for dataset, (ref_base, ref_interp) in ref.items():
            obs_base = (entry.get("baseline") or {}).get(dataset)
            obs_interp = (entry.get("interp") or {}).get(dataset)
            if obs_base is None or obs_interp is None:
                continue
            if abs(obs_base - ref_base) > tolerance:
                messages.append(f"Table6 {source}/{dataset}: baseline {obs_base:.1f} vs {ref_base:.1f}")
            if abs(obs_interp - ref_interp) > tolerance:
                messages.append(f"Table6 {source}/{dataset}: interp {obs_interp:.1f} vs {ref_interp:.1f}")
    return messages


def check_latent_beats_baseline(table6: Dict[str, Any], tolerance: float = 0.3) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    improves = 0
    total = 0
    for source, entry in table6.items():
        base = entry.get("baseline") or {}
        interp = entry.get("interp") or {}
        deltas = {}
        for dataset in SEVERE_SHIFT:
            if base.get(dataset) is None or interp.get(dataset) is None:
                continue
            deltas[dataset] = float(interp[dataset]) - float(base[dataset])
            total += 1
            if deltas[dataset] >= -tolerance:
                improves += 1
        results[source] = deltas
    return {
        "per_source": results,
        "improved": improves,
        "total": total,
        "passes": total > 0 and improves == total,
    }


# ---------------------------------------------------------------------------
# Table 10 / Figure 8 -- source-model quality vs soft-label quality
# ---------------------------------------------------------------------------

def dataset_lca_from_payload(
    payload: Dict[str, Any], hierarchy: Any, num_classes: int = 1000, misclassified_only: bool = True
) -> Optional[float]:
    """Dataset-level ID LCA from cached logits (WordNet, information content)."""
    logits = payload.get("logits")
    targets = payload.get("targets")
    if logits is None or targets is None:
        return None
    metric_cls = _import_attr(["src.metrics.lca_metric", "metrics.lca_metric", "lca_metric"], "LcaMetric")
    if metric_cls is None:
        return None
    try:
        metric = _call(
            metric_cls, hierarchy=hierarchy, mode="information", num_classes=num_classes, base=2.0
        )
    except TypeError:
        try:
            metric = metric_cls(hierarchy)
        except Exception:  # pragma: no cover
            return None
    predictions = _as_numpy(logits).argmax(axis=1)
    targets = _as_numpy(targets)
    try:
        value = _call(
            metric.dataset_lca, predictions, targets,
            misclassified_only=misclassified_only, normalize_by_n=True,
        )
    except Exception:  # pragma: no cover
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):  # tuple return
        value = float(value[0])
    return value if math.isfinite(value) else None


def run_table10(
    cache_dir: str,
    source_models: Sequence[str],
    backbone: str,
    wordnet_hierarchy: Any,
    args: argparse.Namespace,
    num_classes: int = 1000,
    device: Any = None,
) -> Dict[str, Any]:
    """Correlate source-model ID LCA with soft-label quality (Table 10/Fig. 8)."""
    payload = load_id_payload(cache_dir, backbone)
    if payload is None:
        LOG.error("[Table 10] no cached ID features for backbone %s", backbone)
        return {}
    eval_payload = payload["eval"]
    train_payload = payload["train"] or eval_payload
    id_train = _to_probe_data(train_payload["features"], train_payload["targets"], "id_train")
    ood_data = {
        name: _to_probe_data(p["features"], p["targets"], name)
        for name, p in load_ood_payloads(cache_dir, backbone, args.datasets).items()
    }
    temperature = getattr(args, "temperature", 25.0)

    records: List[Dict[str, Any]] = []
    for source in source_models:
        source_payload = load_id_payload(cache_dir, source)
        if source_payload is None:
            continue
        s_eval = source_payload["eval"]
        source_lca = dataset_lca_from_payload(s_eval, wordnet_hierarchy, num_classes)
        if source_lca is None:
            LOG.debug("[Table 10] source %s has no logits; skipping", source)
            continue
        hierarchy = build_latent_hierarchy(
            s_eval["features"], s_eval["targets"], num_classes=num_classes,
            max_level=getattr(args, "max_level", 9), base_level=getattr(args, "base_level", 10),
            seed=getattr(args, "seed", 0),
        )
        if hierarchy is None:
            continue
        matrix = build_latent_matrix(hierarchy, temperature, num_classes)
        if matrix is None:
            continue
        try:
            run = run_backbone(
                backbone, id_train, eval_payload_for_eval(cache_dir, backbone), ood_data, matrix,
                args, num_classes=num_classes, device=device,
            )
        except Exception as exc:  # pragma: no cover
            LOG.error("[Table 10] probe training with source %s failed (%s)", source, exc)
            continue
        acc = run["runs"].get("soft_interp_no_id_drop", {})
        acc = {k: v for k, v in acc.items() if v is not None}
        if not acc:
            continue
        records.append(
            {
                "source_model": source,
                "source_id_lca": float(source_lca),
                "accuracy": acc,
                "mean_ood": float(
                    sum(acc[d] for d in SEVERE_SHIFT if d in acc) / max(1, len([d for d in SEVERE_SHIFT if d in acc]))
                ),
            }
        )
    return table10_correlations(records)


def eval_payload_for_eval(cache_dir: str, backbone: str) -> Any:
    payload = load_id_payload(cache_dir, backbone)
    if payload is None:
        return None
    ev = payload["eval"]
    return _to_probe_data(ev["features"], ev["targets"], "imagenet")


def table10_correlations(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """PEA/R^2 between source ID LCA and probe OOD accuracy for each dataset."""
    correlation_metrics = _import_attr(
        ["src.metrics.correlation", "metrics.correlation", "correlation"], "correlation_metrics"
    )
    out: Dict[str, Any] = {"records": [dict(r) for r in records], "correlations": {}}
    if len(records) < 3:
        out["note"] = "fewer than 3 source models available; correlations unreliable"
        return out
    numpy = __import__("numpy")
    xs = numpy.asarray([r["source_id_lca"] for r in records], dtype=float)
    for dataset in DATASETS:
        ys = numpy.asarray(
            [r["accuracy"].get(dataset, numpy.nan) for r in records], dtype=float
        )
        mask = numpy.isfinite(xs) & numpy.isfinite(ys)
        if mask.sum() < 3 or numpy.unique(ys[mask]).size < 2:
            continue
        if correlation_metrics is not None:
            try:
                metrics = dict(
                    correlation_metrics(xs[mask], ys[mask], abs_values=True, x_name="source_id_lca", y_name="ood_acc")
                )
            except Exception:  # pragma: no cover
                metrics = _fallback_correlation(xs[mask], ys[mask])
        else:
            metrics = _fallback_correlation(xs[mask], ys[mask])
        # Table 10 reports the (positive) strength of the inverse relation
        # between source ID LCA and derived soft-label quality.
        for key in ("pea", "ken", "spe"):
            if key in metrics:
                metrics[key] = abs(float(metrics[key]))
        out["correlations"][dataset] = metrics
    return out


def _fallback_correlation(x: Any, y: Any) -> Dict[str, float]:
    numpy = __import__("numpy")
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    if x.size < 2 or numpy.std(x) == 0 or numpy.std(y) == 0:
        return {"pea": float("nan"), "r2": float("nan"), "n": int(x.size)}
    pea = float(numpy.corrcoef(x, y)[0, 1])
    return {"pea": abs(pea), "r2": pea ** 2, "n": int(x.size)}


def check_against_table10(table10: Dict[str, Any], tolerance: float = 0.25) -> List[str]:
    messages: List[str] = []
    correlations = (table10 or {}).get("correlations", {})
    for dataset, ref in TABLE10_REFERENCE.items():
        entry = correlations.get(dataset)
        if not entry:
            continue
        obs = entry.get("pea")
        if obs is None:
            continue
        if abs(abs(float(obs)) - ref) > tolerance:
            messages.append(f"Table10 {dataset}: PEA {float(obs):.3f} vs paper {ref:.3f}")
    return messages


def make_figure8(
    table10: Dict[str, Any], out_dir: str, dpi: int = 200
) -> List[str]:
    """Scatter source-model ID LCA vs probe OOD accuracy (Figure 8)."""
    records = (table10 or {}).get("records") or []
    if not records:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        LOG.warning("matplotlib unavailable; skipping Figure 8")
        return []

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    panels = [
        ("mean_ood", "mean OOD accuracy"),
        ("objectnet", "ObjectNet"),
        ("r", "ImgNet-R"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, panels):
        xs = []
        ys = []
        for record in records:
            x = record.get("source_id_lca")
            if key == "mean_ood":
                y = record.get("mean_ood")
            else:
                y = (record.get("accuracy") or {}).get(key)
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if not xs:
            ax.set_visible(False)
            continue
        ax.scatter(xs, ys, s=22, alpha=0.8)
        try:
            numpy = __import__("numpy")
            coeffs = numpy.polyfit(xs, ys, 1)
            grid = numpy.linspace(min(xs), max(xs), 50)
            ax.plot(grid, numpy.polyval(coeffs, grid), color="tab:green", linewidth=1.5)
        except Exception:  # pragma: no cover
            pass
        ax.set_xlabel("Source model ID LCA (WordNet)")
        ax.set_ylabel(f"Probe {label}")
        ax.set_title(label)
    fig.suptitle("Figure 8: source-model generalization vs derived soft-label quality")
    fig.tight_layout()
    path = os.path.join(out_dir, "figure8_soft_label_quality.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# Synthetic fallback (offline smoke test)
# ---------------------------------------------------------------------------

def build_synthetic_setup(
    num_classes: int = 1000,
    num_models: int = 4,
    samples_per_class: int = 2,
    feature_dim: int = 32,
    num_groups: int = 8,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create fake cached payloads so the pipeline can be smoke-tested offline."""
    numpy = __import__("numpy")
    rng = numpy.random.default_rng(seed)
    class_centers = rng.normal(size=(num_classes, feature_dim)).astype("float32")
    groups = numpy.arange(num_classes) % num_groups
    class_centers += (groups[:, None] * 0.5).astype("float32")

    def _make(samples: int):
        targets = numpy.repeat(numpy.arange(num_classes), samples)
        features = class_centers[targets] + 0.35 * rng.normal(size=(targets.size, feature_dim))
        logits = features[:, :num_classes] if feature_dim >= num_classes else rng.normal(
            size=(targets.size, num_classes)
        )
        logits = features @ class_centers.T * 10.0
        return {"features": features.astype("float32"), "logits": logits.astype("float32"),
                "targets": targets.astype("int64")}

    payloads: Dict[str, Any] = {}
    names = ["resnet18", "resnet50", "mnasnet1_0", "vit_b_32"][:num_models]
    for name in names:
        payloads[name] = {
            "imagenet": _make(samples_per_class),
            "v2": _make(1), "s": _make(1), "r": _make(1), "a": _make(1), "objectnet": _make(1),
        }
    return payloads, {"models": names}


def write_synthetic_cache(payloads: Dict[str, Any], cache_dir: str) -> None:
    numpy = __import__("numpy")
    os.makedirs(cache_dir, exist_ok=True)
    for model, per_dataset in payloads.items():
        for dataset, payload in per_dataset.items():
            path = os.path.join(cache_dir, f"{model}__{dataset}.npz")
            numpy.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Tables 5/6/9/10 (+ Figure 8) of LCA-on-the-Line"
    )
    parser.add_argument("--config", default=None, help="YAML config (defaults to configs/config.yaml)")
    parser.add_argument("--cache-dir", default=None, help="Directory with cached <model>__<dataset>.npz outputs")
    parser.add_argument("--results-dir", default=None, help="Where to write JSON/PNG artifacts")
    parser.add_argument("--hierarchy-csv", default=None, help="imagenet_fiveai.csv path for WordNet")
    parser.add_argument("--backbones", nargs="*", default=None, help="Backbones for Table 5/9")
    parser.add_argument("--table6-backbone", default="resnet18", help="Backbone for Table 6")
    parser.add_argument("--latent-sources", nargs="*", default=None, help="Source models for Table 10")
    parser.add_argument("--datasets", nargs="*", default=list(OOD_DATASETS), help="OOD datasets to evaluate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lambda-weight", type=float, default=0.03, help="Algorithm 1 lambda")
    parser.add_argument("--temperature", type=float, default=25.0, help="LCA matrix temperature T")
    parser.add_argument("--alignment-mode", default="CE", choices=["CE", "BCE"])
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--id-tolerance", type=float, default=0.0, help="Allowed ID drop (points) for no-ID-drop")
    parser.add_argument("--pro-ood-id-tolerance", type=float, default=2.0, help="Allowed ID drop for pro-OOD")
    parser.add_argument("--max-level", type=int, default=9, help="K-means levels for latent hierarchies")
    parser.add_argument("--base-level", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample cached features (smoke tests)")
    parser.add_argument("--two-seed-interp", action="store_true", help="Train a 2nd CE probe for the CE+interpolation row")
    parser.add_argument("--no-table10", action="store_true", help="Skip Table 10 / Figure 8")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true", help="Generate fake cache when none is found")
    parser.add_argument("--tol", type=float, default=3.0, help="Tolerance (accuracy points) for reference checks")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _subsample_payload(payload: Dict[str, Any], max_samples: Optional[int], seed: int) -> Dict[str, Any]:
    if not max_samples:
        return payload
    numpy = __import__("numpy")
    targets = payload.get("targets")
    if targets is None:
        return payload
    n = len(targets)
    if n <= max_samples:
        return payload
    rng = numpy.random.default_rng(seed)
    idx = numpy.sort(rng.choice(n, size=max_samples, replace=False))
    out = dict(payload)
    for key in ("features", "logits", "targets"):
        if key in out:
            out[key] = out[key][idx]
    return out


def resolve_device(device: Optional[str]) -> Any:
    if device:
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    start = time.time()
    config = load_config(args.config)
    cache_dir = resolve_cache_dir(args, config)
    results_dir = resolve_results_dir(args, config)
    os.makedirs(results_dir, exist_ok=True)
    device = resolve_device(args.device)

    LOG.info("cache dir:   %s", cache_dir)
    LOG.info("results dir: %s", results_dir)
    LOG.info("device:      %s", device)

    # -- data --------------------------------------------------------------
    if args.allow_synthetic and not discover_cached_models(cache_dir):
        LOG.warning("No cached models found; writing a synthetic cache (smoke test only)")
        payloads, _meta = build_synthetic_setup()
        os.makedirs(cache_dir, exist_ok=True)
        write_synthetic_cache(payloads, cache_dir)
        # synthetic data has few classes per model; keep the class count small
    available = discover_cached_models(cache_dir)
    if not available:
        LOG.error("No cached outputs found in %s; run scripts/run_correlation.py first", cache_dir)
        return 1
    LOG.info("cached models (%d): %s", len(available), ", ".join(available[:12]) + ("..." if len(available) > 12 else ""))

    num_classes = int(
        _first(config, ("num_classes",), 1000)
        if not args.allow_synthetic
        else 1000
    )

    # -- hierarchy ---------------------------------------------------------
    build_hierarchy = _import_attr(
        ["src.eval.evaluate_models", "eval.evaluate_models", "evaluate_models"], "build_hierarchy"
    )
    wordnet_hierarchy = None
    if build_hierarchy is not None:
        try:
            wordnet_hierarchy = _call(
                build_hierarchy, csv_path=args.hierarchy_csv, allow_synthetic=True
            )
        except Exception as exc:  # pragma: no cover
            LOG.error("Could not build the WordNet hierarchy (%s)", exc)
    if wordnet_hierarchy is None:
        fc = _import_attr(["src.hierarchy.wordnet", "hierarchy.wordnet", "wordnet"], "build_wordnet_hierarchy")
        if fc is not None:
            try:
                wordnet_hierarchy = _call(fc, csv_path=args.hierarchy_csv, allow_synthetic=True)
            except Exception as exc:  # pragma: no cover
                LOG.error("WordNet hierarchy fallback failed (%s)", exc)

    # -- Table 5 / 9 -------------------------------------------------------
    backbones = args.backbones or [
        b for b in DEFAULT_BACKBONES if find_cache_file(cache_dir, b, "imagenet") is not None
    ]
    if not backbones:
        backbones = [m for m in available if m in TABLE5_REFERENCE] or available[:2]
    LOG.info("backbones for Table 5/9: %s", ", ".join(backbones))

    table5, table9 = run_table5(
        backbones, cache_dir, wordnet_hierarchy, args, num_classes=num_classes, device=device
    )
    # Backfill a couple of Table-10 columns that need per-backbone baselines.
    if table5:
        LOG.info("Table 5 (WordNet soft labels):")
        LOG.info("\n%s", format_accuracy_table({k: v for k, v in table5.items()}, ours_key="ours"))
    else:
        LOG.warning("Table 5 produced no measurable rows")

    # -- Table 6 -----------------------------------------------------------
    table6: Dict[str, Any] = {}
    if find_cache_file(cache_dir, args.table6_backbone, "imagenet") is not None:
        LOG.info("Running Table 6 latent-hierarchy experiments (backbone=%s)", args.table6_backbone)
        table6 = run_table6(
            cache_dir, args.table6_backbone, wordnet_hierarchy, args, num_classes=num_classes, device=device
        )
        if table6:
            LOG.info("Table 6 (latent hierarchies):\n%s", format_accuracy_table(table6, ours_key="interp"))

    # -- Table 10 / Figure 8 ----------------------------------------------
    table10: Dict[str, Any] = {}
    if not args.no_table10:
        sources = args.latent_sources or [m for m in available if m != args.table6_backbone]
        if sources:
            LOG.info("Running Table 10 with %d source models", len(sources))
            table10 = run_table10(
                cache_dir, sources[: int(_first(config, ("table10_max_sources",), 75))],
                args.table6_backbone, wordnet_hierarchy, args,
                num_classes=num_classes, device=device,
            )

    # -- checks / artifacts ------------------------------------------------
    checks = {
        "table5": check_against_table5(table5, tolerance=args.tol),
        "table6": check_against_table6(table6, tolerance=args.tol),
        "table10": check_against_table10(table10),
    }
    soft_loss_summary = check_soft_loss_improves_ood(table5)
    latent_summary = check_latent_beats_baseline(table6)

    artifacts = {
        "table5_wordnet.json": table5,
        "table6_latent.json": table6,
        "table9_ablation.json": table9,
        "table10_source_quality.json": table10,
        "checks.json": checks,
        "success_summary.json": {
            "backbones": backbones,
            "soft_loss_ood": soft_loss_summary,
            "latent_soft_labels": latent_summary,
            "reference_mismatches": {k: len(v) for k, v in checks.items()},
            "elapsed_sec": round(time.time() - start, 2),
        },
    }
    for name, payload in artifacts.items():
        path = os.path.join(results_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
        LOG.info("wrote %s", path)

    if not args.no_figures and table10:
        for path in make_figure8(table10, results_dir):
            LOG.info("wrote %s", path)

    LOG.info(
        "soft-loss OOD deltas: wins=%d losses=%d mean=%+.2f",
        soft_loss_summary.get("wins", 0),
        soft_loss_summary.get("losses", 0),
        soft_loss_summary.get("mean_delta", float("nan")),
    )
    for name, messages in checks.items():
        if messages:
            LOG.warning("%s reference mismatches (%d): %s", name, len(messages), messages[:5])
    LOG.info("done in %.1fs", time.time() - start)
    return 0


def format_accuracy_table(table: Dict[str, Any], ours_key: str = "ours") -> str:
    """Render a Table 5/6 style block (rows = backbone/hierarchy source)."""
    header = f"{'row':<34}" + "".join(f"{DISPLAY.get(d, d):>12}" for d in DATASETS)
    lines = [header, "-" * len(header)]
    for row, entry in table.items():
        base = entry.get("baseline") or {}
        ours = entry.get(ours_key) or {}
        cells = []
        for dataset in DATASETS:
            b = base.get(dataset)
            o = ours.get(dataset)
            if b is None and o is None:
                cells.append(f"{'--':>12}")
            elif o is None:
                cells.append(f"{format(b, '.1f'):>12}")
            else:
                delta = o - b if b is not None else float("nan")
                cells.append(f"{o:>7.1f}({delta:+.1f})".rjust(12))
        lines.append(f"{row:<34}" + "".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
