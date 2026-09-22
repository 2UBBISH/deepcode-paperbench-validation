"""StanfordCars failure-case study (paper Appendix D.4, Table 12).

The SMM paper reports an *ineffective* case of input visual reprogramming:

    Table 12. An Ineffective Case of Input Reprogramming - StanfordCars
    (Mean % +- Std %)
    ------------------------------------------------------------------
    Method      PAD      NARROW    MEDIUM    FULL      OURS
    RESNET-18   4.5+-0.1 3.6+-0.1  3.6+-0.1  3.4+-0.1  2.9+-0.2
    RESNET-50   4.7+-0.2 4.7+-0.1  4.7+-0.2  4.6+-0.1  3.0+-0.6
    ViT-B32     4.7+-0.6 7.7+-0.2  8.3+-0.3  5.0+-0.0  4.8+-0.9
    ------------------------------------------------------------------

Quoting Appendix D.4 ("SMM on An Ineffective Case of Input Reprogramming"):

    "All input visual reprogramming methods seem ineffective on fine-grained
    recognition tasks where subtle appearance differences should be detected.
    As shown in Table 12, in the classification of StanfordCars, where 196
    types of cars are to be classified, the accuracy of all input VR methods
    is below 10 %, indicating the failure of VR methods in this fine-grained
    recognition tasks. Adding our SMM module will not improve performance
    when VR methods fail."

This module therefore (i) runs SMM and the four shared-mask baselines
(Pad / Narrow / Medium / Full) on StanfordCars for every requested backbone
over the paper's three seeds, (ii) reports mean +- std top-1 accuracy, and
(iii) verifies the paper's claim that *every* method stays below ~10 % and
that SMM does not rescue a failing fine-grained task.

Its only real difference from the main-table runner is the dataset (196-class
StanfordCars, the 12th entry of Table 6) and the failure-oriented comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional project imports (guarded so the module stays importable while the
# package is being built incrementally).
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import plumbing
    from ..data.datasets import (  # type: ignore
        DEFAULT_BATCH_SIZES,
        build_dataloaders,
        num_classes as _num_classes,
    )
    _DATASETS_AVAILABLE = True
except Exception:  # pragma: no cover
    DEFAULT_BATCH_SIZES = {"default": 256, "dtd": 64, "oxfordpets": 64}
    build_dataloaders = None  # type: ignore
    _num_classes = None  # type: ignore
    _DATASETS_AVAILABLE = False

try:  # pragma: no cover
    from ..engine.seeds import resolve_seeds as _resolve_seeds  # type: ignore
    from ..engine.seeds import set_seed as _set_seed  # type: ignore
    _SEEDS_AVAILABLE = True
except Exception:  # pragma: no cover

    def _resolve_seeds(seeds=None, n_seeds=None):  # type: ignore
        if seeds is None:
            return [0, 1, 2]
        return [int(s) for s in seeds]

    def _set_seed(seed, **_kwargs):  # type: ignore
        return int(seed)

    _SEEDS_AVAILABLE = False

try:  # pragma: no cover
    from ..engine.metrics import (  # type: ignore
        RunResult,
        aggregate_seeds,
        format_mean_std,
    )
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    RunResult = None  # type: ignore

    def aggregate_seeds(values, ddof: int = 1):  # type: ignore
        vals = [float(v) for v in values]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) == 1 or ddof == 0:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return mean, math.sqrt(max(var, 0.0))

    def format_mean_std(mean, std, decimals: int = 2):  # type: ignore
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

    _METRICS_AVAILABLE = False

try:  # pragma: no cover
    from ..models.pretrained import build_classifier, input_size_for  # type: ignore
    _PRETRAINED_AVAILABLE = True
except Exception:  # pragma: no cover
    build_classifier = None  # type: ignore
    input_size_for = None  # type: ignore
    _PRETRAINED_AVAILABLE = False

try:  # pragma: no cover
    from ..modules.reprogram import build_smm_reprogram  # type: ignore
    _REPROGRAM_AVAILABLE = True
except Exception:  # pragma: no cover
    build_smm_reprogram = None  # type: ignore
    _REPROGRAM_AVAILABLE = False

try:  # pragma: no cover
    from ..engine.train_smm import (  # type: ignore
        MASK_LAYERS_BY_BACKBONE as _ENGINE_MASK_LAYERS,
        SMMTrainConfig,
        train_smm,
    )
    from ..engine.train_smm import build_label_mapping as _engine_build_label_mapping
    _TRAIN_AVAILABLE = True
except Exception:  # pragma: no cover
    _ENGINE_MASK_LAYERS = {"resnet18": 5, "resnet50": 5, "vit_b32": 6}
    SMMTrainConfig = None  # type: ignore
    train_smm = None  # type: ignore
    _engine_build_label_mapping = None  # type: ignore
    _TRAIN_AVAILABLE = False

try:  # pragma: no cover
    from ..label_mapping import build_label_mapping as _lm_build_label_mapping  # type: ignore
    _LABEL_MAPPING_AVAILABLE = True
except Exception:  # pragma: no cover
    _lm_build_label_mapping = None  # type: ignore
    _LABEL_MAPPING_AVAILABLE = False

try:  # pragma: no cover
    from ..methods.baselines import (  # type: ignore
        BaselineTrainConfig,
        build_shared_mask_baseline,
        train_baseline,
    )
    _BASELINES_AVAILABLE = True
except Exception:  # pragma: no cover
    BaselineTrainConfig = None  # type: ignore
    build_shared_mask_baseline = None  # type: ignore
    train_baseline = None  # type: ignore
    _BASELINES_AVAILABLE = False

try:  # pragma: no cover
    from ..data.dataset_stats import TABLE12_STANFORDCARS as _TABLE12_STATS  # type: ignore
except Exception:  # pragma: no cover
    _TABLE12_STATS = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The failing fine-grained target task (196 car classes, Appendix D.4 / Table 12).
STANFORDCARS_DATASET: str = "stanfordcars"

#: Number of target classes of StanfordCars (Table 6 / Table 12 caption).
STANFORDCARS_NUM_CLASSES: int = 196

#: Backbones evaluated in Table 12.
BACKBONES: Tuple[str, ...] = ("resnet18", "resnet50", "vit_b32")

#: Shared-mask baselines plus the proposed SMM ("ours").
BASELINE_METHODS: Tuple[str, ...] = ("pad", "narrow", "medium", "full")
SMM_METHOD: str = "ours"
METHOD_ORDER: Tuple[str, ...] = ("pad", "narrow", "medium", "full", "ours")

#: Human readable display names (matching Table 12 headers).
METHOD_DISPLAY: Dict[str, str] = {
    "pad": "Pad",
    "narrow": "Narrow",
    "medium": "Medium",
    "full": "Full",
    "ours": "Ours",
}

BACKBONE_DISPLAY: Dict[str, str] = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit_b32": "ViT-B32",
}

#: Table 12 of the paper: StanfordCars accuracy (mean, std) in percent.
TABLE12_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "resnet18": {
        "pad": (4.5, 0.1),
        "narrow": (3.6, 0.1),
        "medium": (3.6, 0.1),
        "full": (3.4, 0.1),
        "ours": (2.9, 0.2),
    },
    "resnet50": {
        "pad": (4.7, 0.2),
        "narrow": (4.7, 0.1),
        "medium": (4.7, 0.2),
        "full": (4.6, 0.1),
        "ours": (3.0, 0.6),
    },
    "vit_b32": {
        "pad": (4.7, 0.6),
        "narrow": (7.7, 0.2),
        "medium": (8.3, 0.3),
        "full": (5.0, 0.0),
        "ours": (4.8, 0.9),
    },
}

#: Convenience views of the Table 12 reference numbers.
TABLE12_MEANS: Dict[str, Dict[str, float]] = {
    bb: {m: v[0] for m, v in per_method.items()}
    for bb, per_method in TABLE12_REFERENCE.items()
}
TABLE12_STDS: Dict[str, Dict[str, float]] = {
    bb: {m: v[1] for m, v in per_method.items()}
    for bb, per_method in TABLE12_REFERENCE.items()
}

#: Table 12 claim: every method falls below ~10 % on StanfordCars.
FAILURE_THRESHOLD: float = 10.0

#: Training protocol (Table 9 / Sec. 5) - identical to the main tables so the
#: failure case is measured under a fair comparison.
TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": 200,
    "milestones": (100, 145),
    "alpha_delta": 0.01,
    "gamma_delta": 0.1,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0,
    "batch_size": 256,
    "patch_size": 8,
    "label_mapping": "ilm",
}

#: Mask-generator depth per backbone (5 for ResNets, 6 for ViT-B/32, Sec. 3.2).
MASK_LAYERS_BY_BACKBONE: Dict[str, int] = dict(_ENGINE_MASK_LAYERS)

#: Module level accumulator so callers can introspect the last run.
STANFORDCARS_RESULTS: Dict[str, Any] = {"table12": {}}


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

_BACKBONE_ALIASES = {
    "resnet18": "resnet18",
    "resnet_18": "resnet18",
    "r18": "resnet18",
    "resnet-18": "resnet18",
    "resnet50": "resnet50",
    "resnet_50": "resnet50",
    "r50": "resnet50",
    "resnet-50": "resnet50",
    "vit_b32": "vit_b32",
    "vitb32": "vit_b32",
    "vit-b32": "vit_b32",
    "vit_b_32": "vit_b32",
    "vit": "vit_b32",
}

_METHOD_ALIASES = {
    "pad": "pad",
    "padding": "pad",
    "narrow": "narrow",
    "medium": "medium",
    "mid": "medium",
    "full": "full",
    "ours": "ours",
    "smm": "ours",
    "proposed": "ours",
    "smm_specific": "ours",
    "ours_smm": "ours",
}


def canonical_backbone(name: Optional[str]) -> str:
    """Normalise a backbone spelling to a Table-12 registry key."""
    if name is None:
        return "resnet18"
    key = str(name).strip().lower().replace(" ", "_")
    canonical = _BACKBONE_ALIASES.get(key)
    if canonical:
        return canonical
    squashed = key.replace("-", "").replace("_", "").replace(".", "")
    for alias, value in _BACKBONE_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == squashed:
            return value
    return "resnet18"


def canonical_method(name: Optional[str]) -> str:
    """Normalise a method spelling to ``pad|narrow|medium|full|ours``."""
    if name is None:
        return "ours"
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _METHOD_ALIASES.get(key)
    if canonical:
        return canonical
    squashed = key.replace("_", "")
    for alias, value in _METHOD_ALIASES.items():
        if alias.replace("_", "") == squashed:
            return value
    raise ValueError(
        f"Unknown StanfordCars method {name!r}; expected one of {tuple(METHOD_ORDER)}"
    )


def num_mask_layers_for(backbone: str) -> int:
    """Mask-generator depth: 5 layers for ResNets, 6 for ViT-B/32."""
    return int(MASK_LAYERS_BY_BACKBONE.get(canonical_backbone(backbone), 5))


def input_size_for_backbone(backbone: str, imgsize: Optional[int] = None) -> int:
    """Reprogrammed input resolution (224 ResNets / 384 ViT-B32)."""
    if imgsize is not None:
        return int(imgsize)
    if input_size_for is not None:
        try:
            return int(input_size_for(canonical_backbone(backbone)))
        except Exception:
            pass
    return 384 if canonical_backbone(backbone).startswith("vit") else 224


def batch_size_for(dataset: str = STANFORDCARS_DATASET, backbone: Optional[str] = None) -> int:
    """Table 9 batch size (256 everywhere except DTD / OxfordPets at 64)."""
    key = str(dataset).strip().lower()
    if isinstance(DEFAULT_BATCH_SIZES, dict) and key in DEFAULT_BATCH_SIZES:
        return int(DEFAULT_BATCH_SIZES[key])
    if key in ("dtd", "oxfordpets", "oxford_iiit_pet"):
        return 64
    return int(TRAINING_DEFAULTS["batch_size"])


def mask_lr_schedule(
    backbone: str,
    num_mask_layers: Optional[int] = None,
    alpha_mask: Optional[float] = None,
    gamma_mask: Optional[float] = None,
) -> Tuple[float, float]:
    """Table 9 mask LR/gamma: 0.01/0.1 (5 layers), 0.001/1.0 (6 layers)."""
    layers = int(num_mask_layers) if num_mask_layers is not None else num_mask_layers_for(backbone)
    default_alpha = 0.001 if layers >= 6 else 0.01
    default_gamma = 1.0 if layers >= 6 else 0.1
    return (
        float(alpha_mask) if alpha_mask is not None else default_alpha,
        float(gamma_mask) if gamma_mask is not None else default_gamma,
    )


def num_target_classes(dataset: str = STANFORDCARS_DATASET) -> int:
    """Number of target classes (196 for StanfordCars)."""
    if _num_classes is not None:
        try:
            value = _num_classes(dataset)
            if value:
                return int(value)
        except Exception:
            pass
    return STANFORDCARS_NUM_CLASSES


def resolve_device(device: Optional[Any] = None) -> Any:
    """Return a ``torch.device``, defaulting to CUDA when available."""
    if device is not None and not isinstance(device, str):
        return device
    try:
        import torch  # local import keeps the module import-light
    except Exception:  # pragma: no cover
        return device or "cpu"
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


# ---------------------------------------------------------------------------
# Config construction (kwarg tolerant so signature drift cannot break a run)
# ---------------------------------------------------------------------------

def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys a dataclass does not declare (tolerant wiring)."""
    if cls is None:
        return {}
    try:
        allowed = {f.name for f in fields(cls)}
    except Exception:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


def make_train_config(
    dataset: str = STANFORDCARS_DATASET,
    backbone: str = "resnet18",
    *,
    label_mapping: str = "ilm",
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    patch_size: int = 8,
    **overrides: Any,
) -> Any:
    """Build the Algorithm-1 config used for the SMM row of Table 12."""
    backbone = canonical_backbone(backbone)
    layers = num_mask_layers_for(backbone)
    alpha_mask, gamma_mask = mask_lr_schedule(backbone, layers)
    kwargs: Dict[str, Any] = dict(
        epochs=int(TRAINING_DEFAULTS["epochs"]),
        milestones=tuple(TRAINING_DEFAULTS["milestones"]),
        alpha_delta=float(TRAINING_DEFAULTS["alpha_delta"]),
        gamma_delta=float(TRAINING_DEFAULTS["gamma_delta"]),
        optimizer=str(TRAINING_DEFAULTS["optimizer"]),
        momentum=float(TRAINING_DEFAULTS["momentum"]),
        weight_decay=float(TRAINING_DEFAULTS["weight_decay"]),
        batch_size=batch_size_for(dataset, backbone),
        backbone=backbone,
        input_size=input_size_for_backbone(backbone, imgsize),
        patch_size=int(patch_size),
        num_mask_layers=layers,
        label_mapping=str(label_mapping).lower(),
        mapping_refresh_every=1 if str(label_mapping).lower() == "ilm" else 0,
        alpha_mask=alpha_mask,
        gamma_mask=gamma_mask,
        seed=int(seed),
        device=device,
    )
    kwargs.update(overrides)
    if SMMTrainConfig is None:
        return kwargs
    return SMMTrainConfig(**_filter_kwargs(SMMTrainConfig, kwargs))


def make_baseline_config(
    name: str,
    dataset: str = STANFORDCARS_DATASET,
    backbone: str = "resnet18",
    *,
    label_mapping: str = "ilm",
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    **overrides: Any,
) -> Any:
    """Build the fair-comparison config for Pad/Narrow/Medium/Full."""
    name = canonical_method(name)
    backbone = canonical_backbone(backbone)
    kwargs: Dict[str, Any] = dict(
        name=name,
        backbone=backbone,
        input_size=input_size_for_backbone(backbone, imgsize),
        epochs=int(TRAINING_DEFAULTS["epochs"]),
        milestones=tuple(TRAINING_DEFAULTS["milestones"]),
        lr=float(TRAINING_DEFAULTS["alpha_delta"]),
        gamma=float(TRAINING_DEFAULTS["gamma_delta"]),
        momentum=float(TRAINING_DEFAULTS["momentum"]),
        optimizer=str(TRAINING_DEFAULTS["optimizer"]),
        batch_size=batch_size_for(dataset, backbone),
        label_mapping=str(label_mapping).lower(),
        seed=int(seed),
    )
    kwargs.update(overrides)
    if BaselineTrainConfig is None:
        return kwargs
    return BaselineTrainConfig(**_filter_kwargs(BaselineTrainConfig, kwargs))


def _build_label_mapping(
    dataset: str,
    backbone: str,
    classifier: Any,
    data_loader: Any,
    device: Any,
    seed: int,
    label_mapping: str,
) -> Any:
    """Construct ``f_out`` (Ilm by default) through whichever API is present."""
    n_classes = num_target_classes(dataset)
    mapping_name = str(label_mapping).lower()

    if _engine_build_label_mapping is not None:
        try:
            return _engine_build_label_mapping(
                mapping_name,
                classifier=classifier,
                data_loader=data_loader,
                num_target_classes=n_classes,
                device=device,
                seed=int(seed),
            )
        except TypeError:
            try:
                return _engine_build_label_mapping(
                    name=mapping_name,
                    classifier=classifier,
                    data_loader=data_loader,
                    num_target_classes=n_classes,
                    device=device,
                    seed=int(seed),
                )
            except Exception:
                pass
        except Exception:
            pass

    if _lm_build_label_mapping is not None:
        try:
            return _lm_build_label_mapping(
                mapping_name,
                classifier=classifier,
                data_loader=data_loader,
                num_target_classes=n_classes,
                device=device,
                seed=int(seed),
            )
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

def _history_accuracy(history: Any) -> Optional[float]:
    """Extract the final (or best) test accuracy in percent from a history."""
    if history is None:
        return None
    for attr in ("final_test_accuracy", "best_test_accuracy"):
        value = getattr(history, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    accuracies = getattr(history, "test_accuracies", None)
    if accuracies:
        try:
            return float(accuracies[-1])
        except (TypeError, ValueError, IndexError):
            pass
    return None


def run_single_run(
    backbone: str = "resnet18",
    method: str = "ours",
    *,
    dataset: str = STANFORDCARS_DATASET,
    seed: int = 0,
    device: Optional[Any] = None,
    label_mapping: str = "ilm",
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    imgsize: Optional[int] = None,
    patch_size: int = 8,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    logger: Any = None,
    classifier: Any = None,
    train_loader: Any = None,
    test_loader: Any = None,
) -> Dict[str, Any]:
    """Train one (backbone, method, seed) combination on StanfordCars.

    Returns a dict with the final test accuracy (percent), the training history
    and assorted metadata.  Accuracy is ``None`` when the required project
    modules are unavailable; the caller then falls back to the paper's Table 12
    reference numbers rather than inventing results.
    """
    backbone = canonical_backbone(backbone)
    method = canonical_method(method)
    device = resolve_device(device)
    imgsize = input_size_for_backbone(backbone, imgsize)
    n_classes = num_target_classes(dataset)

    _set_seed(seed)
    start = time.time()

    # --- frozen pre-trained classifier ------------------------------------
    if classifier is None:
        if build_classifier is None:
            return {
                "backbone": backbone, "method": method, "dataset": dataset,
                "seed": int(seed), "accuracy": None, "history": None,
                "num_classes": n_classes, "input_size": imgsize,
                "error": "models.pretrained unavailable",
                "elapsed_seconds": time.time() - start,
            }
        classifier = build_classifier(backbone, device=device)

    # --- data -------------------------------------------------------------
    if train_loader is None or test_loader is None:
        if build_dataloaders is None:
            return {
                "backbone": backbone, "method": method, "dataset": dataset,
                "seed": int(seed), "accuracy": None, "history": None,
                "num_classes": n_classes, "input_size": imgsize,
                "error": "data.datasets unavailable",
                "elapsed_seconds": time.time() - start,
            }
        train_loader, test_loader, spec = build_dataloaders(
            dataset,
            backbone=backbone,
            root=root,
            data_root=data_root,
            imgsize=imgsize,
            batch_size=batch_size_for(dataset, backbone),
            test_batch_size=batch_size_for(dataset, backbone),
            num_workers=num_workers,
            download=download,
            split_seed=split_seed,
            train_fraction=train_fraction,
        )
        n_classes = int(getattr(spec, "num_classes", n_classes) or n_classes)

    # --- output label mapping (Ilm by default, per Appendix A.4) ----------
    f_out = _build_label_mapping(
        dataset, backbone, classifier, train_loader, device, seed, label_mapping
    )

    history = None
    if method == SMM_METHOD:
        if build_smm_reprogram is None or train_smm is None:
            return {
                "backbone": backbone, "method": method, "dataset": dataset,
                "seed": int(seed), "accuracy": None, "history": None,
                "num_classes": n_classes, "input_size": imgsize,
                "error": "SMM reprogram/trainer unavailable",
                "elapsed_seconds": time.time() - start,
            }
        model = build_smm_reprogram(
            backbone=backbone, input_size=imgsize, patch_size=int(patch_size)
        )
        cfg = make_train_config(
            dataset, backbone, label_mapping=label_mapping, seed=seed,
            device=device, imgsize=imgsize, patch_size=patch_size,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            save_dir=save_dir, verbose=verbose,
            **(config_overrides or {}),
        )
        history = train_smm(
            model, classifier, train_loader,
            test_loader=test_loader,
            f_out=f_out,
            config=cfg,
            dataset=dataset,
            device=device,
            logger=logger,
        )
    else:
        if build_shared_mask_baseline is None or train_baseline is None:
            return {
                "backbone": backbone, "method": method, "dataset": dataset,
                "seed": int(seed), "accuracy": None, "history": None,
                "num_classes": n_classes, "input_size": imgsize,
                "error": "methods.baselines unavailable",
                "elapsed_seconds": time.time() - start,
            }
        model = build_shared_mask_baseline(method, backbone=backbone, input_size=imgsize)
        cfg = make_baseline_config(
            method, dataset, backbone, label_mapping=label_mapping, seed=seed,
            device=device, imgsize=imgsize, **(config_overrides or {}),
        )
        history = train_baseline(
            model, classifier, train_loader,
            test_loader=test_loader,
            f_out=f_out,
            num_classes=n_classes,
            config=cfg,
            dataset=dataset,
            device=device,
            logger=logger,
        )

    return {
        "backbone": backbone,
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "accuracy": _history_accuracy(history),
        "history": history,
        "num_classes": n_classes,
        "input_size": imgsize,
        "num_mask_layers": num_mask_layers_for(backbone),
        "elapsed_seconds": time.time() - start,
    }


def run_method_seeds(
    backbone: str = "resnet18",
    method: str = "ours",
    *,
    seeds: Optional[Sequence[int]] = None,
    dataset: str = STANFORDCARS_DATASET,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run one method across seeds and aggregate mean +- std."""
    backbone = canonical_backbone(backbone)
    method = canonical_method(method)
    seed_list = _resolve_seeds(seeds)

    per_seed: List[float] = []
    histories: List[Any] = []
    errors: List[str] = []
    for seed in seed_list:
        run = run_single_run(backbone, method, dataset=dataset, seed=seed, **kwargs)
        if run.get("accuracy") is None:
            errors.append(str(run.get("error", "unknown")))
            continue
        per_seed.append(float(run["accuracy"]))
        if run.get("history") is not None:
            histories.append(run["history"])

    mean, std = aggregate_seeds(per_seed) if per_seed else (None, None)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "seeds": list(seed_list),
        "per_seed_accuracy": per_seed,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std) if mean is not None else None,
        "histories": histories,
        "errors": sorted(set(errors)),
    }


def run_backbone(
    backbone: str = "resnet18",
    *,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    dataset: str = STANFORDCARS_DATASET,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run every requested method for one backbone on StanfordCars."""
    backbone = canonical_backbone(backbone)
    method_list = [canonical_method(m) for m in (methods or METHOD_ORDER)]
    results: Dict[str, Dict[str, Any]] = {}
    for method in method_list:
        if verbose:
            print(
                f"[stanfordcars] {BACKBONE_DISPLAY.get(backbone, backbone)} / "
                f"{METHOD_DISPLAY.get(method, method)} ...",
                flush=True,
            )
        results[method] = run_method_seeds(
            backbone, method, seeds=seeds, dataset=dataset, **kwargs
        )
        if verbose:
            if results[method]["formatted"] is not None:
                print(f"    -> {results[method]['formatted']} %", flush=True)
            elif results[method]["errors"]:
                print(f"    -> unavailable ({results[method]['errors'][0]})", flush=True)
    return {
        "backbone": backbone,
        "dataset": dataset,
        "methods": method_list,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Reporting / comparison (the paper's claim: everything below 10 %)
# ---------------------------------------------------------------------------

TABLE12_ORDER: Tuple[str, ...] = ("pad", "narrow", "medium", "full", "ours")


def format_table(
    payload: Dict[str, Any],
    *,
    backbones: Optional[Sequence[str]] = None,
    method_order: Sequence[str] = TABLE12_ORDER,
    decimals: int = 1,
) -> str:
    """Render a Table 12 style mean +- std text table."""
    lines: List[str] = []
    header = f"{'Method':<12}" + "".join(
        f"{METHOD_DISPLAY.get(m, m.upper()):>16}" for m in method_order
    )
    lines.append(header)
    lines.append("-" * len(header))

    backbone_list = backbones or [b for b in BACKBONES if b in payload]
    for bb in backbone_list:
        bb = canonical_backbone(bb)
        entry = payload.get(bb) or {}
        method_results = entry.get("results", entry)
        label = BACKBONE_DISPLAY.get(bb, bb)
        row = f"{label:<12}"
        for method in method_order:
            res = method_results.get(method)
            if not res or res.get("mean") is None:
                row += f"{'n/a':>16}"
            else:
                row += f"{format_mean_std(res['mean'], res.get('std') or 0.0, decimals):>16}"
        lines.append(row)
    return "\n".join(lines)


def compare_table12(
    payload: Dict[str, Any],
    *,
    tolerance: float = 3.0,
    threshold: float = FAILURE_THRESHOLD,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Verify the two claims of Table 12 / Appendix D.4.

    1. ``all_below_threshold`` - every measured method stays under ~10 %.
    2. ``smm_not_better`` - SMM does not rescue the failing task (SMM mean is
       not more than ``tolerance`` points above the best baseline mean).

    Also reports, per (backbone, method), the deviation from the paper numbers.
    """
    per_cell: Dict[str, Dict[str, Any]] = {}
    all_below = True
    smm_not_better = True
    compared_any = False

    backbone_list = [b for b in BACKBONES if b in payload]
    for bb in backbone_list:
        entry = payload.get(bb) or {}
        method_results = entry.get("results", entry)
        per_cell[bb] = {}
        baseline_means: List[float] = []
        smm_mean: Optional[float] = None
        for method in TABLE12_ORDER:
            res = method_results.get(method)
            mean = None if not res else res.get("mean")
            std = None if not res else res.get("std")
            reference = TABLE12_REFERENCE.get(bb, {}).get(method)
            if mean is not None:
                compared_any = True
                if float(mean) >= threshold:
                    all_below = False
                if method == SMM_METHOD:
                    smm_mean = float(mean)
                else:
                    baseline_means.append(float(mean))
            per_cell[bb][method] = {
                "mean": None if mean is None else float(mean),
                "std": None if std is None else float(std),
                "reference_mean": None if reference is None else reference[0],
                "reference_std": None if reference is None else reference[1],
                "deviation": (
                    None
                    if (mean is None or reference is None)
                    else float(mean) - float(reference[0])
                ),
            }
        if smm_mean is not None and baseline_means:
            best_baseline = max(baseline_means)
            if smm_mean > best_baseline + tolerance:
                smm_not_better = False

    return {
        "per_cell": per_cell,
        "all_below_threshold": bool(all_below),
        "threshold": float(threshold),
        "smm_not_better": bool(smm_not_better),
        "tolerance": float(tolerance),
        "compared_any": bool(compared_any),
        "passed": bool(compared_any and all_below and smm_not_better),
    }


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """Human readable rendering of :func:`compare_table12`."""
    lines = ["StanfordCars (Table 12) failure-case verification:"]
    if not comparison.get("compared_any"):
        lines.append(
            "  no measured results available - cannot verify "
            "(run the experiment first)"
        )
        return "\n".join(lines)
    lines.append(
        f"  all methods below {comparison['threshold']:.0f} % : "
        f"{comparison['all_below_threshold']}"
    )
    lines.append(
        "  SMM does not rescue the failing task "
        f"(<= best baseline + {comparison['tolerance']:.1f}) : "
        f"{comparison['smm_not_better']}"
    )
    for bb, cells in comparison.get("per_cell", {}).items():
        for method, cell in cells.items():
            if cell["mean"] is None:
                continue
            reference = cell["reference_mean"]
            deviation = cell["deviation"]
            lines.append(
                f"    {BACKBONE_DISPLAY.get(bb, bb):<10} "
                f"{METHOD_DISPLAY.get(method, method):<7} "
                f"measured={cell['mean']:.1f} "
                f"reference={'n/a' if reference is None else f'{reference:.1f}'} "
                f"deviation={'n/a' if deviation is None else f'{deviation:+.1f}'}"
            )
    lines.append(f"  PASSED: {comparison['passed']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def _strip_histories(obj: Any) -> Any:
    """Recursively remove non JSON-serialisable training histories."""
    if isinstance(obj, dict):
        return {
            k: _strip_histories(v)
            for k, v in obj.items()
            if k not in ("history", "histories", "config", "classifier", "model")
        }
    if isinstance(obj, (list, tuple)):
        if obj and hasattr(obj[0], "losses"):  # list of history objects
            return f"<{len(obj)} histories>"
        return [_strip_histories(v) for v in obj]
    if hasattr(obj, "losses") and hasattr(obj, "test_accuracies"):
        return "<history>"
    return obj


def run_stanfordcars_experiment(
    backbones: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    *,
    output_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
    device: Optional[Any] = None,
    label_mapping: str = "ilm",
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    imgsize: Optional[int] = None,
    patch_size: int = 8,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    logger: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Table 12: run all input-VR methods on StanfordCars.

    Returns a payload with per-(backbone, method) mean +- std accuracies, the
    Table-12 style text table, the failure-case verification, and - when
    ``save`` is true - the JSON artifact path.
    """
    backbone_list = [canonical_backbone(b) for b in (backbones or BACKBONES)]
    method_list = [canonical_method(m) for m in (methods or METHOD_ORDER)]
    seed_list = _resolve_seeds(seeds)

    run_kwargs: Dict[str, Any] = dict(
        device=device,
        label_mapping=label_mapping,
        root=root,
        data_root=data_root,
        num_workers=num_workers,
        download=download,
        imgsize=imgsize,
        patch_size=patch_size,
        train_fraction=train_fraction,
        split_seed=split_seed,
        config_overrides=config_overrides,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        logger=logger,
    )

    results: Dict[str, Any] = {}
    for bb in backbone_list:
        results[bb] = run_backbone(
            bb, methods=method_list, seeds=seed_list, verbose=verbose, **run_kwargs
        )

    comparison = compare_table12(results, verbose=verbose)
    table = format_table(results)

    payload: Dict[str, Any] = {
        "experiment": "stanfordcars",
        "dataset": STANFORDCARS_DATASET,
        "num_classes": num_target_classes(STANFORDCARS_DATASET),
        "backbones": backbone_list,
        "methods": method_list,
        "seeds": list(seed_list),
        "label_mapping": str(label_mapping).lower(),
        "results": results,
        "reference": TABLE12_REFERENCE,
        "comparison": comparison,
        "table": table,
        "report": format_comparison_report(comparison),
    }

    STANFORDCARS_RESULTS["table12"] = payload

    if verbose:
        print()
        print(table)
        print()
        print(payload["report"])

    if save:
        try:
            payload["path"] = save_stanfordcars_results(payload, output_dir)
            if verbose:
                print(f"\n[stanfordcars] results written to {payload['path']}")
        except Exception as exc:  # pragma: no cover - IO is best effort
            payload["save_error"] = str(exc)

    return payload


#: Registry alias used by ``smm_vr.experiments``.
run_stanfordcars_failure_case = run_stanfordcars_experiment


def run_single_stanfordcars(
    backbone: str = "resnet18", method: str = "ours", **kwargs: Any
) -> Dict[str, Any]:
    """Convenience wrapper: one backbone/method across seeds."""
    return run_method_seeds(backbone, method, **kwargs)


def save_stanfordcars_results(
    payload: Dict[str, Any],
    output_dir: Optional[str] = None,
    *,
    filename: str = "stanfordcars_table12.json",
) -> str:
    """Persist the Table 12 payload (histories stripped) as JSON."""
    out_dir = output_dir or os.path.join("outputs", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_strip_histories(payload), handle, indent=2, sort_keys=True)
    return path


def describe_stanfordcars_study(
    backbones: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Metadata describing the failure-case study (for ``--describe``)."""
    return {
        "name": "stanfordcars",
        "description": (
            "Appendix D.4 / Table 12 - an ineffective case of input visual "
            "reprogramming: 196-class fine-grained StanfordCars, where all "
            "input VR methods (Pad/Narrow/Medium/Full and SMM) stay below "
            "~10 % accuracy and SMM does not improve performance."
        ),
        "dataset": STANFORDCARS_DATASET,
        "num_classes": STANFORDCARS_NUM_CLASSES,
        "backbones": [canonical_backbone(b) for b in (backbones or BACKBONES)],
        "methods": [canonical_method(m) for m in (methods or METHOD_ORDER)],
        "seeds": _resolve_seeds(seeds),
        "training": dict(TRAINING_DEFAULTS),
        "failure_threshold": FAILURE_THRESHOLD,
        "reference_table12": TABLE12_REFERENCE,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for the StanfordCars failure-case experiment."""
    parser = argparse.ArgumentParser(
        prog="python -m smm_vr.experiments.run_stanfordcars",
        description=(
            "Reproduce Table 12 (Appendix D.4): the ineffective StanfordCars "
            "case of input visual reprogramming."
        ),
    )
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES),
                        help="backbones to evaluate (default: resnet18 resnet50 vit_b32)")
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER),
                        help="methods to run (default: pad narrow medium full ours)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(_resolve_seeds(None)),
                        help="random seeds (default: 0 1 2)")
    parser.add_argument("--label-mapping", default="ilm",
                        help="output label mapping (default: ilm)")
    parser.add_argument("--patch-size", type=int,
                        default=int(TRAINING_DEFAULTS["patch_size"]),
                        help="SMM patch size (default: 8)")
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    parser.add_argument("--data-root", default=None, help="dataset root directory")
    parser.add_argument("--root", default=None, help="torchvision dataset root")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true",
                        help="do not download the dataset")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="debug: use only this fraction of the training split")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--describe", action="store_true",
                        help="print the study description and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)

    if args.describe:
        print(json.dumps(
            describe_stanfordcars_study(args.backbones, args.methods, args.seeds),
            indent=2, sort_keys=True,
        ))
        return 0

    payload = run_stanfordcars_experiment(
        backbones=args.backbones,
        methods=args.methods,
        seeds=args.seeds,
        output_dir=args.output_dir,
        save=not args.no_save,
        verbose=not args.quiet,
        device=args.device,
        label_mapping=args.label_mapping,
        root=args.root,
        data_root=args.data_root,
        num_workers=args.num_workers,
        download=not args.no_download,
        patch_size=args.patch_size,
        train_fraction=args.train_fraction,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    comparison = payload.get("comparison", {})
    return 0 if comparison.get("passed", False) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
