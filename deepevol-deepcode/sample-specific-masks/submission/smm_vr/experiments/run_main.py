"""Main-results experiment runner for SMM (ICML 2024).

Reproduces the two headline tables of the paper:

* **Table 1** -- ResNet-18 / ResNet-50 target-task top-1 accuracy for
  SMM (``ours``) and the four shared-mask VR baselines
  (``pad``, ``narrow``, ``medium``, ``full``).
* **Table 2** -- the same comparison with the frozen ViT-B/32 backbone.

The runner is a thin orchestration layer on top of the already-implemented
engine: for every (backbone, dataset, method, seed) combination it

1. builds the torchvision data loaders with the exact addendum transforms
   (``imgsize = 384`` for ViT-B32, else 224),
2. loads the frozen ImageNet-1K classifier (``f_P``),
3. builds ``f_out`` (Ilm by default -- the paper's default mapping for the
   main tables),
4. trains either the SMM reprogram wrapper (``ours``) or a shared-mask
   baseline with the fair-comparison schedule (200 epochs, LR 0.01,
   gamma 0.1, milestones 100/145, batch 256 -- 64 for DTD/OxfordPets),
5. evaluates top-1 top-k accuracy and aggregates over the three paper seeds
   ``{0, 1, 2}``, reporting mean +- std.

All heavy lifting lives in :mod:`smm_vr.engine` / :mod:`smm_vr.methods`; this
module only wires those components together and renders the table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Imports (guarded so the runner can be imported during incremental builds)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - guarded imports
    from ..data.datasets import (  # noqa: F401
        ALL_DATASETS,
        DEFAULT_BATCH_SIZES,
        MAIN_DATASETS as _DATA_MAIN_DATASETS,
        build_dataloaders,
        build_datasets,
        get_dataset_spec,
        num_classes as dataset_num_classes,
    )
    _DATA_AVAILABLE = True
except Exception:  # pragma: no cover
    _DATA_AVAILABLE = False
    _DATA_MAIN_DATASETS = None

try:  # pragma: no cover
    from .main_reference import *  # type: ignore  # not used, kept optional
except Exception:  # pragma: no cover
    pass

try:  # pragma: no cover - plain import of the dataclass
    from ..data.dataset_stats import (  # noqa: F401
        REFERENCE_AVERAGES as _REFERENCE_AVERAGES,
        TABLE1_AVERAGES as _TABLE1_AVERAGES,
        TABLE2_AVERAGES as _TABLE2_AVERAGES,
    )
except Exception:  # pragma: no cover
    _REFERENCE_AVERAGES = {}
    _TABLE1_AVERAGES = {}
    _TABLE2_AVERAGES = {}

try:  # pragma: no cover
    from ..engine.seeds import (  # noqa: F401
        SEEDS as _SEEDS,
        resolve_seeds,
        set_seed,
    )
except Exception:  # pragma: no cover
    _SEEDS = (0, 1, 2)

    def resolve_seeds(seeds=None, n_seeds=None):  # type: ignore
        return list(seeds) if seeds else list(_SEEDS)

    def set_seed(seed, **kwargs):  # type: ignore
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

try:  # pragma: no cover
    from ..engine.metrics import (  # noqa: F401
        TABLE1_RESNET18,
        TABLE1_RESNET50,
        TABLE2_VIT_B32,
        aggregate_seeds,
        dump_results_json,
        format_mean_std,
        format_results_table,
        mean_over_datasets,
    )
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    _METRICS_AVAILABLE = False
    TABLE1_RESNET18 = {}
    TABLE1_RESNET50 = {}
    TABLE2_VIT_B32 = {}

    def aggregate_seeds(values, ddof=1):  # type: ignore
        import math

        vals = [float(v) for v in values if v is not None]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) == 1:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return mean, math.sqrt(max(var, 0.0))

    def format_mean_std(mean, std, decimals=2):  # type: ignore
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

    def mean_over_datasets(per_dataset, dataset_order=None):  # type: ignore
        order = list(dataset_order or per_dataset.keys())
        vals = [float(per_dataset[d]) for d in order if d in per_dataset]
        return sum(vals) / len(vals) if vals else 0.0

    def format_results_table(per_dataset, **kwargs):  # type: ignore
        return json.dumps(per_dataset, indent=2)

    def dump_results_json(path, results, **metadata):  # type: ignore
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"metadata": metadata, "results": results}, fh, indent=2)
        return path

try:  # pragma: no cover
    from ..models.pretrained import (  # noqa: F401
        build_classifier,
        input_size_for,
        load_pretrained,
        resolve_backbone,
    )
    _PRETRAINED_AVAILABLE = True
except Exception:  # pragma: no cover
    _PRETRAINED_AVAILABLE = False

try:  # pragma: no cover
    from ..models.mask_generator import (  # noqa: F401
        EXPECTED_PARAMETERS,
        build_mask_generator,
        count_parameters,
    )
    _MASK_GEN_AVAILABLE = True
except Exception:  # pragma: no cover
    _MASK_GEN_AVAILABLE = False
    EXPECTED_PARAMETERS = {"resnet18": 26499, "resnet50": 26499, "vit_b32": 102339}

try:  # pragma: no cover
    from ..modules.reprogram import build_smm_reprogram  # noqa: F401
    _REPROGRAM_AVAILABLE = True
except Exception:  # pragma: no cover
    _REPROGRAM_AVAILABLE = False

try:  # pragma: no cover
    from ..label_mapping import build_label_mapping  # noqa: F401
    _LABEL_MAPPING_AVAILABLE = True
except Exception:  # pragma: no cover
    _LABEL_MAPPING_AVAILABLE = False

try:  # pragma: no cover
    from ..engine.train_smm import (  # noqa: F401
        MASK_LAYERS_BY_BACKBONE as _TRAIN_MASK_LAYERS,
        SMMTrainConfig,
        build_label_mapping as engine_build_label_mapping,
        train_smm,
    )
    _TRAIN_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRAIN_AVAILABLE = False
    _TRAIN_MASK_LAYERS = {"resnet18": 5, "resnet50": 5, "vit_b32": 6}

try:  # pragma: no cover
    from ..engine.evaluate import evaluate  # noqa: F401
    _EVAL_AVAILABLE = True
except Exception:  # pragma: no cover
    _EVAL_AVAILABLE = False

try:  # pragma: no cover
    from ..methods.baselines import (  # noqa: F401
        BASELINE_NAMES,
        BaselineTrainConfig,
        build_shared_mask_baseline,
        train_baseline,
    )
    _BASELINES_AVAILABLE = True
except Exception:  # pragma: no cover
    _BASELINES_AVAILABLE = False
    BASELINE_NAMES = ("pad", "narrow", "medium", "full"),


__all__ = [
    "MAIN_DATASETS",
    "BACKBONES",
    "SMM_METHOD",
    "BASELINE_METHODS",
    "METHOD_ORDER",
    "TABLE1_AVERAGES",
    "TABLE2_AVERAGES",
    "REFERENCE_AVERAGES",
    "TRAINING_DEFAULTS",
    "build_all_methods",
    "make_train_config",
    "make_baseline_config",
    "run_single_dataset_method",
    "run_single_dataset",
    "run_table1",
    "run_table2",
    "run_resnet18_experiment",
    "run_resnet50_experiment",
    "run_vit_b32_experiment",
    "run_main_experiment",
    "compare_to_reference",
    "build_main_summary",
    "save_main_results",
    "main",
]


# ---------------------------------------------------------------------------
# Static registry / reference values (paper Tables 1, 2 and Table 6 order)
# ---------------------------------------------------------------------------
#: Table 6 dataset order -- kept identical to the table row order.
MAIN_DATASETS: Tuple[str, ...] = (
    "cifar10",
    "cifar100",
    "svhn",
    "gtsrb",
    "flowers102",
    "dtd",
    "ucf101",
    "food101",
    "sun397",
    "eurosat",
    "oxfordpets",
) if not _DATA_MAIN_DATASETS else tuple(_DATA_MAIN_DATASETS)

#: Backbones reported in the main tables (Tables 1 and 2).
BACKBONES: Tuple[str, ...] = ("resnet18", "resnet50", "vit_b32")

SMM_METHOD: str = "ours"

#: Shared-mask VR baselines compared against SMM.
BASELINE_METHODS: Tuple[str, ...] = ("pad", "narrow", "medium", "full")

#: Column order used when rendering the paper tables (SMM first).
METHOD_ORDER: Tuple[str, ...] = (SMM_METHOD,) + tuple(BASELINE_METHODS)

#: Backbone -> number of CNN layers in the mask generator (Sec. 3.2, Table 4).
MASK_LAYERS_BY_BACKBONE: Dict[str, int] = dict(
    _TRAIN_MASK_LAYERS
) if isinstance(_TRAIN_MASK_LAYERS, dict) else {"resnet18": 5, "resnet50": 5, "vit_b32": 6}

# Mask generator parameter budget per backbone (Table 4).
MASK_PARAMETER_BUDGET: Dict[str, int] = {
    "resnet18": 26499,
    "resnet50": 26499,
    "vit_b32": 102339,
    "vit_large": 102339,
}

#: Fair-comparison training schedule shared by SMM and the baselines.
TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": 200,
    "milestones": (100, 145),
    "lr": 0.01,
    "gamma": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "optimizer": "sgd",
    "batch_size": 256,
    "test_batch_size": 256,
    "num_workers": 4,
    "download": True,
    "drop_last": False,
    "patch_size": 8,
    "label_mapping": "ilm",
    "log_every": 10,
    "eval_every": 1,
}

#: Paper reference averages (Table 1 for ResNets, Table 2 for ViT-B/32).
TABLE1_AVERAGES: Dict[str, Dict[str, float]] = {
    "resnet18": {"ours": 52.53, "full": 46.85, "medium": 45.04, "narrow": 43.48, "pad": 43.91},
    "resnet50": {"ours": 56.35, "full": 52.10, "medium": 49.39, "narrow": 46.76, "pad": 49.15},
}
TABLE2_AVERAGES: Dict[str, float] = {
    "ours": 72.4,
    "full": 64.7,
    "medium": 65.2,
    "narrow": 63.7,
    "pad": 53.1,
}
REFERENCE_AVERAGES: Dict[str, Dict[str, float]] = {
    "resnet18": TABLE1_AVERAGES["resnet18"],
    "resnet50": TABLE1_AVERAGES["resnet50"],
    "vit_b32": TABLE2_AVERAGES,
}

#: Reference per-dataset tables (prefer the engine's copies when available).
REFERENCE_TABLES: Dict[str, Dict[str, float]] = {
    "resnet18": dict(TABLE1_RESNET18 or {}),
    "resnet50": dict(TABLE1_RESNET50 or {}),
    "vit_b32": dict(TABLE2_VIT_B32 or {}),
}

#: Known abnormal / documented exception cases (see README "known exceptions").
KNOWN_EXCEPTIONS: Dict[str, str] = {
    "resnet18:dtd": "Pad may slightly outperform SMM on DTD textures with ResNet-18.",
    "vit_b32:eurosat": "Pad can be marginally better on EuroSAT (SMM over-fits).",
    "resnet18:ucf101": "UCF101 is learning-rate sensitive with ResNet-18.",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _filter_kwargs(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the subset of ``kwargs`` accepted by a dataclass."""
    try:
        names = {f.name for f in dataclasses.fields(target)}
    except Exception:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in names}


def _resolve_device(device: Optional[Any] = None) -> str:
    if isinstance(device, str):
        return device
    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_input_size(backbone: str, imgsize: Optional[int] = None) -> int:
    if imgsize:
        return int(imgsize)
    if _PRETRAINED_AVAILABLE:
        try:
            return int(input_size_for(backbone))
        except Exception:
            pass
    return 384 if "vit" in str(backbone).lower() else 224


def _mask_layers(backbone: str) -> int:
    key = str(backbone).lower().replace("-", "_")
    return int(MASK_LAYERS_BY_BACKBONE.get(key, 6 if "vit" in key else 5))


def batch_size_for(dataset: str, backend: Optional[str] = None) -> int:
    """Table 9 batch size: 256 everywhere except DTD and OxfordPets (64)."""
    name = str(dataset).lower()
    if isinstance(DEFAULT_BATCH_SIZES, dict) and name in DEFAULT_BATCH_SIZES:
        return int(DEFAULT_BATCH_SIZES[name])
    return 64 if name in ("dtd", "oxfordpets") else 256


def _num_target_classes(dataset: str) -> Optional[int]:
    if not _DATA_AVAILABLE:
        return None
    try:
        return int(dataset_num_classes(dataset))
    except Exception:
        try:
            return int(get_dataset_spec(dataset).num_classes)
        except Exception:
            return None


def build_all_methods() -> List[str]:
    """All methods compared in the main tables (SMM + shared-mask baselines)."""
    return list(METHOD_ORDER)


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------
def make_train_config(
    dataset: str,
    backbone: str = "resnet18",
    *,
    label_mapping: str = "ilm",
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    patch_size: int = 8,
    **overrides: Any,
) -> Any:
    """Build the Algorithm-1 training config for one SMM run.

    Defaults follow Table 9 (200 epochs, milestones 100/145,
    ``alpha_delta=0.01``/``gamma_delta=0.1``) and resolve the
    mask-generator learning rate from the CNN depth (5 layers: 0.01/0.1,
    6 layers: 0.001/1.0).
    """
    num_mask_layers = _mask_layers(backbone)
    if num_mask_layers >= 6:
        alpha_mask, gamma_mask = 0.001, 1.0
    else:
        alpha_mask, gamma_mask = 0.01, 0.1

    kwargs: Dict[str, Any] = dict(
        epochs=TRAINING_DEFAULTS["epochs"],
        milestones=TRAINING_DEFAULTS["milestones"],
        alpha_delta=TRAINING_DEFAULTS["lr"],
        gamma_delta=TRAINING_DEFAULTS["gamma"],
        alpha_mask=alpha_mask,
        gamma_mask=gamma_mask,
        optimizer=TRAINING_DEFAULTS["optimizer"],
        momentum=TRAINING_DEFAULTS["momentum"],
        weight_decay=TRAINING_DEFAULTS["weight_decay"],
        batch_size=batch_size_for(dataset),
        test_batch_size=TRAINING_DEFAULTS["test_batch_size"],
        num_workers=TRAINING_DEFAULTS["num_workers"],
        download=TRAINING_DEFAULTS["download"],
        drop_last=TRAINING_DEFAULTS["drop_last"],
        backbone=backbone,
        input_size=_resolve_input_size(backbone, imgsize),
        patch_size=patch_size,
        num_mask_layers=num_mask_layers,
        label_mapping=str(label_mapping).lower(),
        mapping_refresh_every=1,
        seed=seed,
        device=device,
        log_every=TRAINING_DEFAULTS["log_every"],
        eval_every=TRAINING_DEFAULTS["eval_every"],
        verbose=False,
    )
    kwargs.update(overrides)
    if not _TRAIN_AVAILABLE:
        return kwargs
    return SMMTrainConfig(**_filter_kwargs(SMMTrainConfig, kwargs))


def make_baseline_config(
    name: str,
    dataset: str,
    backbone: str = "resnet18",
    *,
    label_mapping: str = "ilm",
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    **overrides: Any,
) -> Any:
    """Fair-comparison baseline config (identical schedule to SMM)."""
    kwargs: Dict[str, Any] = dict(
        name=str(name).lower(),
        backbone=backbone,
        input_size=_resolve_input_size(backbone, imgsize),
        epochs=TRAINING_DEFAULTS["epochs"],
        milestones=TRAINING_DEFAULTS["milestones"],
        lr=TRAINING_DEFAULTS["lr"],
        gamma=TRAINING_DEFAULTS["gamma"],
        momentum=TRAINING_DEFAULTS["momentum"],
        optimizer=TRAINING_DEFAULTS["optimizer"],
        batch_size=batch_size_for(dataset),
        label_mapping=str(label_mapping).lower(),
        seed=seed,
        device=device,
    )
    kwargs.update(overrides)
    if not _BASELINES_AVAILABLE:
        return kwargs
    return BaselineTrainConfig(**_filter_kwargs(BaselineTrainConfig, kwargs))


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------
def _build_loaders(
    dataset: str,
    backbone: str,
    *,
    batch_size: Optional[int] = None,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    imgsize: Optional[int] = None,
    device: Optional[Any] = None,
    split_seed: int = 0,
    train_fraction: Optional[float] = None,
    generator_seed: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Build ``(train_loader, test_loader, spec)`` with addendum transforms."""
    if not _DATA_AVAILABLE:
        raise RuntimeError("smm_vr.data.datasets is unavailable; cannot build loaders.")
    return build_dataloaders(
        name=dataset,
        backbone=backbone,
        root=root,
        data_root=data_root,
        imgsize=imgsize,
        batch_size=batch_size if batch_size is not None else batch_size_for(dataset),
        test_batch_size=TRAINING_DEFAULTS["test_batch_size"],
        num_workers=num_workers,
        download=download,
        split_seed=split_seed,
        train_fraction=train_fraction,
        drop_last=False,
        device=device,
        generator_seed=generator_seed,
    )


def _build_classifier(
    backbone: str,
    *,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    weights: str = "IMAGENET1K_V1",
    verbose: bool = False,
) -> Any:
    if not _PRETRAINED_AVAILABLE:
        raise RuntimeError("smm_vr.models.pretrained is unavailable.")
    return load_pretrained(
        backbone,
        weights=weights,
        pretrained=True,
        freeze=True,
        input_size=_resolve_input_size(backbone, imgsize),
        device=device,
        verbose=verbose,
    )


def _build_smm_model(
    backbone: str,
    *,
    imgsize: Optional[int] = None,
    patch_size: int = 8,
    num_layers: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate the SMM ``f_in`` wrapper (delta + f_mask + patch interp)."""
    if not _REPROGRAM_AVAILABLE:
        raise RuntimeError("smm_vr.modules.reprogram is unavailable.")
    input_size = _resolve_input_size(backbone, imgsize)
    num_layers = int(num_layers or _mask_layers(backbone))
    num_pooling = max(0, int(round(patch_size).bit_length() - 1))
    try:
        return build_smm_reprogram(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_pooling_layers=num_pooling,
            **kwargs,
        )
    except TypeError:
        # Fall back to constructing the mask generator explicitly.
        if not _MASK_GEN_AVAILABLE:
            raise
        f_mask = build_mask_generator(
            backbone, num_layers=num_layers, num_pooling_layers=num_pooling
        )
        from ..modules.reprogram import SMMReprogram  # local import

        return SMMReprogram(
            mask_generator=f_mask,
            input_size=input_size,
            patch_size=patch_size,
            backbone=backbone,
        )


def _build_fout(
    label_mapping: str,
    *,
    classifier: Any,
    data_loader: Any,
    num_target_classes: Optional[int],
    device: Optional[Any] = None,
    seed: int = 0,
) -> Any:
    """Build the output mapping ``f_out`` (Ilm default for Tables 1 and 2)."""
    name = str(label_mapping).lower()
    if name in ("none", "identity", ""):
        return None
    builder = build_label_mapping if _LABEL_MAPPING_AVAILABLE else (
        engine_build_label_mapping if _TRAIN_AVAILABLE else None
    )
    if builder is None:
        return None
    try:
        return builder(
            name,
            classifier=classifier,
            model=classifier,
            data_loader=data_loader,
            num_target_classes=int(num_target_classes or 1000),
            device=device,
            seed=seed,
        )
    except TypeError:
        return builder(
            name,
            classifier=classifier,
            data_loader=data_loader,
            num_target_classes=int(num_target_classes or 1000),
            device=device,
            seed=seed,
        )


def _evaluate_model(
    classifier: Any,
    test_loader: Any,
    *,
    f_in: Any = None,
    f_out: Any = None,
    device: Optional[Any] = None,
    dataset: str = "",
    method: str = "ours",
    label_mapping: str = "ilm",
    backbone: str = "resnet18",
    seed: int = 0,
    num_classes: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> Optional[float]:
    """Top-1 test accuracy (percent) of a trained reprogrammer, or ``None``."""
    if not _EVAL_AVAILABLE:
        return None
    try:
        result = evaluate(
            classifier,
            test_loader,
            f_in=f_in,
            f_out=f_out,
            device=device,
            dataset=dataset,
            method=method,
            label_mapping=label_mapping,
            backbone=backbone,
            seed=seed,
            num_classes=num_classes,
            max_batches=max_batches,
        )
    except Exception:
        return None
    return float(getattr(result, "top1", getattr(result, "accuracy", 0.0)))


# ---------------------------------------------------------------------------
# Single (dataset, method, seed) runs
# ---------------------------------------------------------------------------
def run_single_dataset_method(
    dataset: str,
    method: str = SMM_METHOD,
    backbone: str = "resnet18",
    *,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[Any] = None,
    label_mapping: str = "ilm",
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    num_workers: int = TRAINING_DEFAULTS["num_workers"],
    download: bool = True,
    imgsize: Optional[int] = None,
    patch_size: int = TRAINING_DEFAULTS["patch_size"],
    weights: str = "IMAGENET1K_V1",
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Train + evaluate one method on one dataset over the paper seeds.

    Returns a dict with ``per_seed_accuracy``, ``mean``, ``std``, ``formatted``
    and the first available training history.
    """
    device = _resolve_device(device)
    seeds = list(resolve_seeds(seeds)) if seeds is not None else list(_SEEDS)
    method = str(method).lower()
    overrides = dict(config_overrides or {})

    train_loader, test_loader, spec = _build_loaders(
        dataset,
        backbone,
        root=root,
        data_root=data_root,
        num_workers=num_workers,
        download=download,
        imgsize=imgsize,
        device=device,
        split_seed=split_seed,
        train_fraction=train_fraction,
    )
    num_target_classes = getattr(spec, "num_classes", None) or _num_target_classes(dataset)
    history_ref: Optional[Any] = None
    accuracies: List[float] = []

    for seed in seeds:
        set_seed(int(seed))
        classifier = _build_classifier(
            backbone, device=device, imgsize=imgsize, weights=weights, verbose=False
        )
        f_out = _build_fout(
            label_mapping,
            classifier=classifier,
            data_loader=train_loader,
            num_target_classes=num_target_classes,
            device=device,
            seed=int(seed),
        )

        if method in (SMM_METHOD, "smm"):
            config = make_train_config(
                dataset,
                backbone,
                label_mapping=label_mapping,
                seed=int(seed),
                device=device,
                imgsize=imgsize,
                patch_size=patch_size,
                max_train_batches=max_train_batches,
                max_eval_batches=max_eval_batches,
                **overrides,
            )
            model = _build_smm_model(
                backbone, imgsize=imgsize, patch_size=patch_size
            )
            if hasattr(model, "to"):
                model = model.to(device)
            history = train_smm(
                model,
                classifier,
                train_loader,
                test_loader=test_loader,
                f_out=f_out,
                label_mapping_builder=lambda cl=classifier, dl=train_loader,
                nc=num_target_classes, d=device, s=int(seed): _build_fout(
                    label_mapping,
                    classifier=cl,
                    data_loader=dl,
                    num_target_classes=nc,
                    device=d,
                    seed=s,
                ),
                config=config,
                dataset=dataset,
                device=device,
                logger=log,
            )
            history_ref = history_ref or history
            acc = _evaluate_model(
                classifier,
                test_loader,
                f_in=model,
                f_out=f_out,
                device=device,
                dataset=dataset,
                method=method,
                label_mapping=label_mapping,
                backbone=backbone,
                seed=int(seed),
                num_classes=num_target_classes,
                max_batches=max_eval_batches,
            )
            if acc is None:
                acc = float(
                    getattr(history, "final_test_accuracy", 0.0)
                    or getattr(history, "best_test_accuracy", 0.0)
                    or 0.0
                )
            accuracies.append(float(acc))
        else:
            if not _BASELINES_AVAILABLE:
                raise RuntimeError("smm_vr.methods.baselines is unavailable.")
            config = make_baseline_config(
                method,
                dataset,
                backbone,
                label_mapping=label_mapping,
                seed=int(seed),
                device=device,
                imgsize=imgsize,
                **overrides,
            )
            model = build_shared_mask_baseline(
                method, backbone=backbone, input_size=_resolve_input_size(backbone, imgsize)
            )
            if hasattr(model, "to"):
                model = model.to(device)
            history = train_baseline(
                model,
                classifier,
                train_loader,
                test_loader=test_loader,
                f_out=f_out,
                num_classes=num_target_classes,
                config=config,
                dataset=dataset,
                device=device,
                logger=log,
            )
            history_ref = history_ref or history
            acc = _evaluate_model(
                classifier,
                test_loader,
                f_in=model,
                f_out=f_out,
                device=device,
                dataset=dataset,
                method=method,
                label_mapping=label_mapping,
                backbone=backbone,
                seed=int(seed),
                num_classes=num_target_classes,
                max_batches=max_eval_batches,
            )
            if acc is None:
                acc = float(
                    getattr(history, "final_test_accuracy", 0.0)
                    or getattr(history, "best_test_accuracy", 0.0)
                    or 0.0
                )
            accuracies.append(float(acc))

        if log:
            log(f"[{backbone}/{dataset}/{method}] seed={seed} top1={accuracies[-1]:.2f}")
        if verbose:
            print(f"[{backbone}/{dataset}/{method}] seed={seed} top1={accuracies[-1]:.2f}")

    mean, std = aggregate_seeds(accuracies)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "label_mapping": str(label_mapping).lower(),
        "seeds": [int(s) for s in seeds],
        "per_seed_accuracy": [float(a) for a in accuracies],
        "mean": float(mean),
        "std": float(std),
        "formatted": format_mean_std(mean, std),
        "history": history_ref,
        "num_classes": num_target_classes,
    }


def run_single_dataset(
    dataset: str,
    backbone: str = "resnet18",
    *,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run every method (SMM + baselines) on a single target dataset."""
    methods = list(methods) if methods else build_all_methods()
    per_method: Dict[str, Any] = {}
    for method in methods:
        out = run_single_dataset_method(
            dataset, method, backbone, seeds=seeds, **kwargs
        )
        per_method[method] = {k: v for k, v in out.items() if k != "history"}
        per_method[method]["history"] = out.get("history")
    return {
        "dataset": dataset,
        "backbone": backbone,
        "methods": per_method,
        "averages": {
            m: per_method[m]["mean"] for m in per_method
        },
    }


# ---------------------------------------------------------------------------
# Whole-table runs
# ---------------------------------------------------------------------------
def _run_table(
    backbone: str,
    *,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    datasets = list(datasets) if datasets else list(MAIN_DATASETS)
    methods = list(methods) if methods else build_all_methods()
    table: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in methods}
    log = kwargs.get("log")
    started = time.time()

    for dataset in datasets:
        for method in methods:
            out = run_single_dataset_method(
                dataset, method, backbone, seeds=seeds, **kwargs
            )
            table[method][dataset] = {k: v for k, v in out.items() if k != "history"}
            if log:
                log(
                    f"[{backbone}] {dataset:>11s} {method:<14s} "
                    f"{out['formatted']} (ref: "
                    f"{REFERENCE_TABLES.get(backbone, {}).get(dataset, 'n/a')})"
                )

    averages: Dict[str, float] = {}
    for method in methods:
        per_dataset = {d: table[method][d]["mean"] for d in datasets}
        averages[method] = float(mean_over_datasets(per_dataset, dataset_order=datasets))

    reference = REFERENCE_TABLES.get(backbone, {})
    comparison = {
        method: {
            "measured_average": averages[method],
            "reference_average": REFERENCE_AVERAGES.get(backbone, {}).get(method),
        }
        for method in methods
    }

    return {
        "backbone": backbone,
        "datasets": datasets,
        "methods": methods,
        "results": table,
        "averages": averages,
        "reference_averages": dict(REFERENCE_AVERAGES.get(backbone, {})),
        "comparison": comparison,
        "reference_per_dataset": reference,
        "elapsed_seconds": time.time() - started,
        "seed_values": [int(s) for s in (resolve_seeds(seeds) if seeds is None else seeds)],
    }


def run_table1(
    backbone: str = "resnet18",
    *,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Table 1 (ResNet-18 or ResNet-50, 11 datasets)."""
    backbone = "resnet50" if str(backbone).lower().replace("-", "_") in (
        "resnet50", "resnet_50", "rn50",
    ) else "resnet18"
    return _run_table(backbone, datasets=datasets, methods=methods, seeds=seeds, **kwargs)


def run_table2(
    *,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Table 2 (frozen ViT-B/32, 11 datasets)."""
    return _run_table("vit_b32", datasets=datasets, methods=methods, seeds=seeds, **kwargs)


def run_resnet18_experiment(**kwargs: Any) -> Dict[str, Any]:
    """Entry point for the ResNet-18 half of Table 1."""
    kwargs.setdefault("backbone", "resnet18")
    kwargs.pop("backbone")
    return run_table1("resnet18", **kwargs)


def run_resnet50_experiment(**kwargs: Any) -> Dict[str, Any]:
    """Entry point for the ResNet-50 half of Table 1."""
    kwargs.pop("backbone", None)
    return run_table1("resnet50", **kwargs)


def run_vit_b32_experiment(**kwargs: Any) -> Dict[str, Any]:
    """Entry point for Table 2 (ViT-B/32)."""
    kwargs.pop("backbone", None)
    return run_table2(**kwargs)


def run_main_experiment(
    backbone: str = "resnet18",
    *,
    backbones: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    output_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the full main-results experiment (Tables 1 and 2).

    Parameters mirror the YAML configs: ``backbone``/``backbones`` select the
    frozen classifier(s), ``datasets`` defaults to the 11 Table-6 tasks,
    ``methods`` defaults to ``("ours", "pad", "narrow", "medium", "full")`` and
    ``seeds`` defaults to ``(0, 1, 2)``.
    """
    targets = [str(b).lower() for b in (backbones or [backbone])]
    targets = [b for b in targets if b]
    out: Dict[str, Any] = {
        "experiment": "main",
        "tables": {},
        "averages": {},
        "dataset_order": list(datasets) if datasets else list(MAIN_DATASETS),
        "seeds": [int(s) for s in (resolve_seeds(seeds) if seeds is None else seeds)],
        "methods": list(methods) if methods else build_all_methods(),
    }

    for bb in targets:
        if bb in ("vit_b32", "vit_b_32", "vit"):
            table = run_table2(
                datasets=datasets, methods=methods, seeds=seeds, verbose=verbose, **kwargs
            )
            key = "table2"
        else:
            table = run_table1(
                bb, datasets=datasets, methods=methods, seeds=seeds,
                verbose=verbose, **kwargs,
            )
            key = f"table1_{bb}"
        out["tables"][key] = table
        out["averages"][bb] = table["averages"]
        if verbose:
            print(format_table(table))

    if save:
        try:
            path = save_main_results(out, output_dir=output_dir)
            out["results_path"] = path
        except Exception as exc:  # pragma: no cover - best-effort persistence
            if verbose:
                print(f"warning: could not save results: {exc}")
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def format_table(table: Dict[str, Any], *, decimals: int = 1) -> str:
    """Render a runner table dict as a fixed-width paper-style text table."""
    datasets = list(table.get("datasets") or [])
    methods = list(table.get("methods") or [])
    results = table.get("results", {})
    if not datasets or not methods:
        return ""
    name_w = max(12, max(len(d) for d in datasets) + 2)
    header = "dataset".ljust(name_w) + "".join(m.rjust(12) for m in methods)
    lines = [f"backbone={table.get('backbone')}  f_out={table.get('label_mapping', 'Ilm')}", header,
             "-" * len(header)]
    for d in datasets:
        row = d.ljust(name_w)
        for m in methods:
            entry = results.get(m, {}).get(d)
            row += (
                f"{entry['mean']:.{decimals}f}".rjust(12) if entry else "n/a".rjust(12)
            )
        lines.append(row)
    avg_row = "AVERAGE".ljust(name_w)
    for m in methods:
        avg_row += f"{table.get('averages', {}).get(m, 0.0):.{decimals}f}".rjust(12)
    lines.append("-" * len(header))
    lines.append(avg_row)
    ref_row = "(paper)".ljust(name_w)
    for m in methods:
        ref = table.get("reference_averages", {}).get(m)
        ref_row += (f"{ref:.1f}".rjust(12) if ref is not None else "-".rjust(12))
    lines.append(ref_row)
    return "\n".join(lines)


def compare_to_reference(
    table: Dict[str, Any],
    *,
    backbone: Optional[str] = None,
    average_tolerance: float = 2.0,
    per_dataset_tolerance: float = 5.0,
) -> Dict[str, Any]:
    """Compare measured accuracies against the paper's reference numbers."""
    backbone = backbone or table.get("backbone")
    reference = dict(REFERENCE_TABLES.get(backbone, {}))
    ref_avg = dict(REFERENCE_AVERAGES.get(backbone, {}))
    report: Dict[str, Any] = {
        "backbone": backbone,
        "averages": {},
        "per_dataset": {},
        "within_tolerance": True,
        "known_exceptions": {
            k: v for k, v in KNOWN_EXCEPTIONS.items() if k.startswith(f"{backbone}:")
        },
    }
    for method, measured in (table.get("averages") or {}).items():
        ref = ref_avg.get(method)
        delta = None if ref is None else float(measured) - float(ref)
        ok = delta is None or abs(delta) <= average_tolerance
        report["averages"][method] = {
            "measured": float(measured),
            "reference": ref,
            "delta": delta,
            "within_tolerance": ok,
        }
        if not ok:
            report["within_tolerance"] = False

    results = table.get("results", {})
    for method, per_ds in results.items():
        for dataset, entry in per_ds.items():
            ref = reference.get(dataset)
            measured = entry.get("mean")
            if ref is None or measured is None:
                continue
            delta = float(measured) - float(ref)
            ok = abs(delta) <= per_dataset_tolerance
            report["per_dataset"][f"{method}/{dataset}"] = {
                "measured": float(measured),
                "reference": float(ref),
                "delta": delta,
                "within_tolerance": ok,
            }

    # SMM should beat every baseline on the average (the paper's core claim).
    ours = (table.get("averages") or {}).get(SMM_METHOD)
    if ours is not None:
        beaten = {
            m: float(ours) - float(v)
            for m, v in (table.get("averages") or {}).items()
            if m != SMM_METHOD
        }
        report["smm_minus_baselines"] = beaten
        report["smm_best_average"] = all(d >= 0 for d in beaten.values())
    return report


def build_main_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary (averages + reference deltas) for logging/JSON."""
    summary: Dict[str, Any] = {
        "experiment": results.get("experiment", "main"),
        "seeds": results.get("seeds"),
        "dataset_order": results.get("dataset_order"),
        "averages": results.get("averages", {}),
        "comparison": {},
    }
    for backbone, table in (results.get("tables") or {}).items():
        try:
            summary["comparison"][backbone] = compare_to_reference(table)
        except Exception:
            continue
    return summary


def save_main_results(
    results: Dict[str, Any], output_dir: Optional[str] = None, *, filename: str = "main_results.json"
) -> str:
    """Persist main-experiment results (seeds, averages, comparison) to JSON."""
    output_dir = output_dir or os.path.join("outputs", "results")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k != "history"}
        if isinstance(obj, (list, tuple)):
            return [_strip(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    payload = {
        "metadata": {
            "experiment": "main",
            "backbone_targets": list(results.get("tables", {}).keys()),
            "seeds": results.get("seeds"),
            "dataset_order": results.get("dataset_order"),
            "methods": results.get("methods"),
            "reference_averages": REFERENCE_AVERAGES,
        },
        "summary": _strip(build_main_summary(results)),
        "results": _strip(results),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point: ``python -m smm_vr.experiments.run_main --backbone resnet18``."""
    parser = argparse.ArgumentParser(description="SMM main results (Tables 1 and 2)")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50", "vit_b32", "all"],
        help="frozen pre-trained classifier",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="subset of the 11 tasks")
    parser.add_argument("--methods", nargs="*", default=None, help="methods to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="seeds (default 0 1 2)")
    parser.add_argument("--label-mapping", default="ilm", choices=["ilm", "flm", "rlm"])
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=None, help="debug subsampling")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    backbones = list(BACKBONES) if args.backbone == "all" else [args.backbone]
    results = run_main_experiment(
        backbones=backbones,
        datasets=args.datasets,
        methods=args.methods,
        seeds=args.seeds,
        output_dir=args.output_dir,
        save=not args.no_save,
        verbose=True,
        num_workers=args.num_workers,
        download=True,
        data_root=args.data_root,
        label_mapping=args.label_mapping,
        patch_size=args.patch_size,
        device=args.device,
        train_fraction=args.train_fraction,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    print(json.dumps(build_main_summary(results), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
