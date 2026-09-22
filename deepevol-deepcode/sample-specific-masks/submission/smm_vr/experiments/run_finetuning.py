"""Finetuning-based comparison experiments for SMM (ICML 2024).

This runner reproduces the two finetuning comparisons of the SMM paper:

* **Table 13 / Appendix E.1** -- *Advantages of VR in dealing with distorted input
  images*: LoRA for ViT (rank 6, LR 0.01, 10 epochs, ~0.60 M extra parameters)
  versus SMM (~0.54 M extra parameters) on the four ``32 x 32`` target tasks
  (CIFAR10, CIFAR100, SVHN, GTSRB).  ViT-Large with a ``384 x 384`` input is the
  well-trained model; the paper reports averages of ``77.9`` (LoRA) versus
  ``90.0`` (SMM) on the four low-resolution tasks, and ``83.4`` versus ``83.5``
  on the seven ``128 x 128`` tasks.  SMM is expected to be clearly better on the
  low-resolution tasks because up-scaling a ``32 x 32`` image to ``384 x 384``
  distorts it, and VR (input-space reprogramming) compensates for that
  distortion while LoRA does not.

* **Table 14 / Appendix E.2** -- *Advantages of VR in being orthogonal to
  finetuning-based methods*: Finetuning the fully-connected layer
  ("Finetuning-FC") on ResNet-50 with and without the SMM input module.  SMM is
  orthogonal to finetuning (it modifies the input space, finetuning modifies the
  model), so attaching SMM on top of Finetuning-FC raises the average accuracy
  from ``75.3`` to ``79.2``.

The module is deliberately tolerant: every project import is guarded and, when a
component is unavailable, the corresponding run reports ``accuracy=None`` with a
reason instead of fabricating numbers.  Reference values from the paper are kept
in :data:`TABLE13_REFERENCE` / :data:`TABLE14_REFERENCE` so results can always be
compared against them.

Training defaults follow the paper: 200 epochs, milestones ``(100, 145)``,
LR ``0.01`` with gamma ``0.1`` (SGD momentum 0.9), batch size 256 (64 for DTD and
OxfordPets), patch size 8 and label mapping ``Ilm`` for the SMM module.  For the
viability of a fair LoRA-vs-SMM comparison (Appendix E.1) epochs are set to 10
with LR 0.01 for both methods, matching "all training settings are kept the
same".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Guarded project imports (the module must import cleanly in partial builds)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly
    from ..data.datasets import (  # type: ignore
        DEFAULT_BATCH_SIZES,
        MAIN_DATASETS,
        build_dataloaders,
        num_classes as _data_num_classes,
    )
except Exception:  # pragma: no cover
    DEFAULT_BATCH_SIZES = {"dtd": 64, "oxfordpets": 64}
    MAIN_DATASETS = (
        "cifar10", "cifar100", "svhn", "gtsrb", "flowers102", "dtd",
        "ucf101", "food101", "sun397", "eurosat", "oxfordpets",
    )
    build_dataloaders = None
    _data_num_classes = None

try:  # pragma: no cover
    from ..engine.metrics import (  # type: ignore
        RunResult,
        aggregate_seeds,
        format_mean_std,
    )
except Exception:  # pragma: no cover
    RunResult = None

    def aggregate_seeds(values: Sequence[float], ddof: int = 1) -> Tuple[float, float]:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return (float("nan"), float("nan"))
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return (mean, 0.0)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return (mean, math.sqrt(max(var, 0.0)))

    def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

try:  # pragma: no cover
    from ..engine.seeds import resolve_seeds, set_seed  # type: ignore
except Exception:  # pragma: no cover
    SEEDS_FALLBACK = (0, 1, 2)

    def resolve_seeds(seeds=None, n_seeds=None) -> List[int]:
        if seeds is None:
            seeds = SEEDS_FALLBACK
        seeds = list(seeds)
        if n_seeds is not None and n_seeds > 0:
            seeds = seeds[:n_seeds]
        return seeds

    def set_seed(seed: int, **kwargs) -> int:  # noqa: D401
        return int(seed)

try:  # pragma: no cover
    from ..models.pretrained import build_classifier, input_size_for  # type: ignore
except Exception:  # pragma: no cover
    build_classifier = None
    input_size_for = None

try:  # pragma: no cover
    from ..methods.finetuning import (  # type: ignore
        DEFAULT_FINETUNE_EPOCHS,
        DEFAULT_FINETUNE_LR,
        DEFAULT_GAMMA,
        DEFAULT_LORA_ALPHA,
        DEFAULT_LORA_EPOCHS,
        DEFAULT_LORA_LR,
        DEFAULT_LORA_RANK,
        DEFAULT_MILESTONES,
        FEATURE_DIMS,
        FINETUNE_METHODS,
        FinetuneConfig,
        FinetuneHistory,
        build_finetune_fc,
        build_finetune_fc_with_smm,
        build_lora_model,
        describe_finetuning,
        list_finetuning_methods,
        train_finetune_fc,
        train_finetune_fc_with_smm,
        train_finetuning_method,
        train_lora,
    )
except Exception:  # pragma: no cover
    DEFAULT_LORA_RANK = 6
    DEFAULT_LORA_ALPHA = 6.0
    DEFAULT_LORA_LR = 0.01
    DEFAULT_LORA_EPOCHS = 10
    DEFAULT_FINETUNE_LR = 0.01
    DEFAULT_FINETUNE_EPOCHS = 200
    DEFAULT_MILESTONES = (100, 145)
    DEFAULT_GAMMA = 0.1
    FEATURE_DIMS = {"resnet18": 512, "resnet50": 2048, "vit_b32": 768, "vit_large": 1024}
    FINETUNE_METHODS = ("lora", "finetune_fc", "finetune_fc_smm")
    FinetuneConfig = None
    FinetuneHistory = None
    build_finetune_fc = None
    build_finetune_fc_with_smm = None
    build_lora_model = None
    describe_finetuning = None
    list_finetuning_methods = None
    train_finetune_fc = None
    train_finetune_fc_with_smm = None
    train_finetuning_method = None
    train_lora = None

try:  # pragma: no cover
    from ..data.dataset_stats import (  # type: ignore
        TABLE13_LORA as _REF_TABLE13_LORA,
        TABLE14_FINETUNE_FC as _REF_TABLE14_FC,
    )
except Exception:  # pragma: no cover
    _REF_TABLE13_LORA = None
    _REF_TABLE14_FC = None

try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

__all__ = [
    # constants
    "FINETUNE_COMPARISON_METHODS",
    "LOW_RES_DATASETS",
    "HIGH_RES_DATASETS",
    "TABLE13_DATASETS",
    "TABLE14_DATASET_ORDER",
    "TABLE13_LORA",
    "TABLE13_SMM",
    "TABLE13_AVERAGES",
    "TABLE13_REFERENCE",
    "TABLE13_EXTRA_PARAMETERS_M",
    "TABLE14_FINETUNE_FC",
    "TABLE14_FINETUNE_FC_SMM",
    "TABLE14_AVERAGES",
    "TABLE14_REFERENCE",
    "TRAINING_DEFAULTS",
    "LORA_TRAINING_DEFAULTS",
    "LORA_BACKBONE",
    "FINETUNE_FC_BACKBONE",
    "FINETUNING_RESULTS",
    # helpers
    "canonical_backbone",
    "canonical_method",
    "canonical_dataset",
    "input_size_for_backbone",
    "num_mask_layers_for",
    "mask_lr_schedule",
    "batch_size_for",
    "resolve_device",
    "num_target_classes",
    "make_lora_config",
    "make_finetune_fc_config",
    # runs
    "run_single_lora",
    "run_single_finetune_fc",
    "run_lora_seeds",
    "run_finetune_fc_seeds",
    "run_table13",
    "run_table14",
    "run_finetuning_experiment",
    "run_finetuning",
    "run_single_finetuning",
    # reporting
    "format_table13",
    "format_table14",
    "compare_table13",
    "compare_table14",
    "format_comparison_report",
    "save_finetuning_results",
    "describe_finetuning_study",
    "build_arg_parser",
    "main",
]

# ---------------------------------------------------------------------------
# Reference numbers (paper Tables 13 and 14 / Appendix E.1-E.2)
# ---------------------------------------------------------------------------
LORA_BACKBONE = "vit_large"
FINETUNE_FC_BACKBONE = "resnet50"

# Table 13: the four 32x32 tasks and the seven 128x128 tasks.
LOW_RES_DATASETS: Tuple[str, ...] = ("cifar10", "cifar100", "svhn", "gtsrb")
HIGH_RES_DATASETS: Tuple[str, ...] = (
    "flowers102", "dtd", "ucf101", "food101", "sun397", "eurosat", "oxfordpets",
)
TABLE13_DATASETS: Tuple[str, ...] = LOW_RES_DATASETS
TABLE14_DATASET_ORDER: Tuple[str, ...] = LOW_RES_DATASETS + HIGH_RES_DATASETS

TABLE13_LORA: Dict[str, float] = {
    "cifar10": 95.9, "cifar100": 83.6, "svhn": 65.3, "gtsrb": 66.6,
}
TABLE13_SMM: Dict[str, float] = {
    "cifar10": 97.4, "cifar100": 87.3, "svhn": 91.0, "gtsrb": 84.2,
}
TABLE13_EXTRA_PARAMETERS_M: Dict[str, float] = {"lora": 0.60, "ours": 0.54}
TABLE13_AVERAGES: Dict[str, Dict[str, float]] = {
    "lora": {"32x32": 77.9, "128x128": 83.4, "params_m": 0.60},
    "ours": {"32x32": 90.0, "128x128": 83.5, "params_m": 0.54},
}
TABLE13_REFERENCE: Dict[str, Dict[str, float]] = {
    "lora": TABLE13_LORA,
    "ours": TABLE13_SMM,
}

TABLE14_FINETUNE_FC: Dict[str, float] = {
    "cifar10": 90.1, "cifar100": 70.7, "svhn": 63.5, "gtsrb": 77.8,
    "flowers102": 90.9, "dtd": 67.6, "ucf101": 70.8, "food101": 57.6,
    "sun397": 53.5, "eurosat": 95.7, "oxfordpets": 90.4, "average": 75.3,
}
TABLE14_FINETUNE_FC_SMM: Dict[str, float] = {
    "cifar10": 91.2, "cifar100": 72.4, "svhn": 86.9, "gtsrb": 85.2,
    "flowers102": 90.9, "dtd": 68.2, "ucf101": 72.0, "food101": 59.6,
    "sun397": 57.9, "eurosat": 95.8, "oxfordpets": 90.6, "average": 79.2,
}
TABLE14_AVERAGES: Dict[str, float] = {
    "finetune_fc": 75.3,
    "finetune_fc_smm": 79.2,
    "gain": 3.9,
}
TABLE14_REFERENCE: Dict[str, Dict[str, float]] = {
    "finetune_fc": TABLE14_FINETUNE_FC,
    "finetune_fc_smm": TABLE14_FINETUNE_FC_SMM,
}

# Training defaults (paper Sec. 5 / Table 9; optimizer unspecified -> SGD 0.9).
TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": DEFAULT_FINETUNE_EPOCHS,
    "milestones": tuple(DEFAULT_MILESTONES),
    "lr": DEFAULT_FINETUNE_LR,
    "gamma": DEFAULT_GAMMA,
    "momentum": 0.9,
    "optimizer": "sgd",
    "weight_decay": 0.0,
    "batch_size": 256,
    "patch_size": 8,
    "label_mapping": "ilm",
}
# Appendix E.1: "All training settings are kept the same ... learning rate is
# 0.01, running 10 epochs in total" for the LoRA/SMM comparison.
LORA_TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": DEFAULT_LORA_EPOCHS,
    "milestones": (5, 8),
    "lr": DEFAULT_LORA_LR,
    "gamma": DEFAULT_GAMMA,
    "momentum": 0.9,
    "optimizer": "sgd",
    "rank": DEFAULT_LORA_RANK,
    "alpha": DEFAULT_LORA_ALPHA,
}

FINETUNE_COMPARISON_METHODS: Tuple[str, ...] = ("lora", "finetune_fc", "finetune_fc_smm")
FINETUNING_RESULTS: Dict[str, Any] = {"table13": {}, "table14": {}}

_SMALL_BATCH_DATASETS = ("dtd", "oxfordpets")
_DATASET_CLASS_FALLBACK: Dict[str, int] = {
    "cifar10": 10, "cifar100": 100, "svhn": 10, "gtsrb": 43, "flowers102": 102,
    "dtd": 47, "ucf101": 101, "food101": 101, "sun397": 397, "eurosat": 10,
    "oxfordpets": 37, "stanfordcars": 196,
}

_METHOD_ALIASES = {
    "lora": "lora",
    "finetuning_lora": "lora",
    "finetuning-lora": "lora",
    "finetune_fc": "finetune_fc",
    "finetuning_fc": "finetune_fc",
    "finetuning-fc": "finetune_fc",
    "fc": "finetune_fc",
    "smm": "finetune_fc_smm",
    "finetune_fc_smm": "finetune_fc_smm",
    "finetuning_fc_smm": "finetune_fc_smm",
    "finetuning-fc+smm": "finetune_fc_smm",
    "ours": "finetune_fc_smm",
    "our_smm": "finetune_fc_smm",
}

_DISPLAY = {
    "lora": "Finetuning-LoRA",
    "finetune_fc": "Finetuning-FC",
    "finetune_fc_smm": "Finetuning-FC + Our SMM",
    "ours": "Our SMM",
}

_DATASET_DISPLAY = {
    "cifar10": "CIFAR10", "cifar100": "CIFAR100", "svhn": "SVHN",
    "gtsrb": "GTSRB", "flowers102": "Flowers102", "dtd": "DTD",
    "ucf101": "UCF101", "food101": "Food101", "sun397": "SUN397",
    "eurosat": "EuroSAT", "oxfordpets": "OxfordPets",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _canonical(name: Optional[str], table: Dict[str, str], fallback: str = "") -> str:
    if name is None:
        return fallback
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key in table:
        return table[key]
    squashed = key.replace("_", "")
    for k, v in table.items():
        if k.replace("_", "") == squashed:
            return v
    return key


def canonical_backbone(name: Optional[str]) -> str:
    """Normalise a backbone spelling to ``resnet18``/``resnet50``/``vit_large``/..."""
    key = _canonical(name, {}, fallback="resnet18")
    key = key.replace("-", "_")
    if key in ("vitl", "vit_l", "vitl16", "vit_l16", "vit_large", "vit_large_16",
               "vitlarge", "vit_large_patch16_384"):
        return "vit_large"
    if key in ("vitb32", "vit_b32", "vit_b_32", "vitb_32", "vit"):
        return "vit_b32"
    if key in ("r18", "resnet_18", "resnet18"):
        return "resnet18"
    if key in ("r50", "resnet_50", "resnet50"):
        return "resnet50"
    return key


def canonical_method(name: Optional[str]) -> str:
    """Normalise a finetuning method spelling."""
    key = str(name if name is not None else "lora").strip().lower().replace(" ", "_")
    key = key.replace("-", "_")
    if key in _METHOD_ALIASES:
        return _METHOD_ALIASES[key]
    squashed = key.replace("_", "")
    for k, v in _METHOD_ALIASES.items():
        if k.replace("_", "") == squashed:
            return v
    raise ValueError(
        f"Unknown finetuning method {name!r}; expected one of {sorted(set(_METHOD_ALIASES.values()))}"
    )


def canonical_dataset(name: Optional[str]) -> str:
    """Normalise a dataset spelling to the registry keys used here."""
    key = str(name if name is not None else "cifar10").strip().lower()
    key = key.replace("-", "").replace("_", "").replace(" ", "")
    alias = {
        "cifar10": "cifar10", "cifar100": "cifar100", "svhn": "svhn",
        "gtsrb": "gtsrb", "flowers102": "flowers102", "dtd": "dtd",
        "ucf101": "ucf101", "food101": "food101", "sun397": "sun397",
        "eurosat": "eurosat", "oxfordpets": "oxfordpets",
        "oxfordiiitpet": "oxfordpets", "oxfordiiitpets": "oxfordpets",
        "stanfordcars": "stanfordcars", "cars": "stanfordcars",
    }
    return alias.get(key, key)


def input_size_for_backbone(backbone: str, imgsize: Optional[int] = None) -> int:
    """224 for ResNets, 384 for ViT (paper addendum / Appendix E.1)."""
    if imgsize is not None:
        return int(imgsize)
    if input_size_for is not None:
        try:
            return int(input_size_for(backbone))
        except Exception:
            pass
    return 384 if "vit" in str(backbone).lower() else 224


def num_mask_layers_for(backbone: str) -> int:
    """5 CNN layers for ResNet backbones, 6 for ViT (paper Sec. 3.2, Table 4)."""
    raw = str(backbone).lower()
    if "resnet" in raw or raw.startswith("r"):
        return 5
    return 6


def mask_lr_schedule(
    backbone: str,
    num_mask_layers: Optional[int] = None,
    alpha_mask: Optional[float] = None,
    gamma_mask: Optional[float] = None,
) -> Tuple[float, float]:
    """Mask-generator LR/gamma: 0.01/0.1 (5 layers) or 0.001/1.0 (6 layers)."""
    layers = num_mask_layers if num_mask_layers is not None else num_mask_layers_for(backbone)
    default_alpha = 0.01 if layers <= 5 else 0.001
    default_gamma = 0.1 if layers <= 5 else 1.0
    return (
        float(alpha_mask) if alpha_mask is not None else default_alpha,
        float(gamma_mask) if gamma_mask is not None else default_gamma,
    )


def batch_size_for(dataset: str, backbone: Optional[str] = None) -> int:
    """Batch size per Table 9: 256, except DTD and OxfordPets (64)."""
    name = canonical_dataset(dataset)
    if name in _SMALL_BATCH_DATASETS:
        return 64
    return int(DEFAULT_BATCH_SIZES.get(name, 256)) if isinstance(DEFAULT_BATCH_SIZES, dict) else 256


def resolve_device(device: Optional[Any] = None) -> Any:
    if device is not None:
        return device
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
        except Exception:
            pass
        return torch.device("cpu")
    return "cpu"


def num_target_classes(dataset: str) -> Optional[int]:
    name = canonical_dataset(dataset)
    if _data_num_classes is not None:
        try:
            return int(_data_num_classes(name))
        except Exception:
            pass
    return _DATASET_CLASS_FALLBACK.get(name)


def _filter_kwargs(factory: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys a dataclass/constructor does not declare (tolerates drift)."""
    if factory is None:
        return dict(kwargs)
    try:
        if is_dataclass(factory):
            names = {f.name for f in fields(factory)}
        else:
            import inspect

            signature = inspect.signature(factory)
            if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
                return dict(kwargs)
            names = set(signature.parameters)
    except Exception:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in names}


def _config_dict(**kwargs: Any) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
def make_lora_config(
    dataset: str,
    backbone: str = LORA_BACKBONE,
    *,
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    rank: int = DEFAULT_LORA_RANK,
    alpha: Optional[float] = None,
    lr: float = DEFAULT_LORA_LR,
    epochs: int = DEFAULT_LORA_EPOCHS,
    milestones: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    train_head: bool = True,
    **overrides: Any,
) -> Any:
    """Build a :class:`FinetuneConfig` (or plain dict) for the LoRA run."""
    payload = dict(
        method="lora",
        backbone=canonical_backbone(backbone),
        input_size=input_size_for_backbone(backbone, imgsize),
        epochs=int(epochs),
        lr=float(lr),
        gamma=DEFAULT_GAMMA,
        milestones=tuple(milestones) if milestones else tuple(DEFAULT_MILESTONES),
        weight_decay=0.0,
        momentum=0.9,
        optimizer="sgd",
        batch_size=int(batch_size if batch_size is not None else batch_size_for(dataset)),
        rank=int(rank),
        alpha=float(alpha) if alpha is not None else float(rank),
        lora_target="qkv",
        train_head=bool(train_head),
        patch_size=8,
        label_mapping="ilm",
        num_workers=int(num_workers),
        seed=int(seed),
        device=device,
        verbose=False,
    )
    payload.update(overrides)
    if FinetuneConfig is not None:
        return FinetuneConfig(**_filter_kwargs(FinetuneConfig, payload))
    return _config_dict(**payload)


def make_finetune_fc_config(
    dataset: str,
    backbone: str = FINETUNE_FC_BACKBONE,
    *,
    use_smm: bool = False,
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    lr: float = DEFAULT_FINETUNE_LR,
    epochs: int = DEFAULT_FINETUNE_EPOCHS,
    milestones: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    patch_size: int = 8,
    **overrides: Any,
) -> Any:
    """Build a :class:`FinetuneConfig` (or plain dict) for the Finetuning-FC run."""
    payload = dict(
        method="finetune_fc_smm" if use_smm else "finetune_fc",
        backbone=canonical_backbone(backbone),
        input_size=input_size_for_backbone(backbone, imgsize),
        epochs=int(epochs),
        lr=float(lr),
        gamma=DEFAULT_GAMMA,
        milestones=tuple(milestones) if milestones else tuple(DEFAULT_MILESTONES),
        weight_decay=0.0,
        momentum=0.9,
        optimizer="sgd",
        batch_size=int(batch_size if batch_size is not None else batch_size_for(dataset)),
        rank=DEFAULT_LORA_RANK,
        alpha=None,
        train_head=True,
        patch_size=int(patch_size),
        label_mapping="ilm",
        num_workers=int(num_workers),
        seed=int(seed),
        device=device,
        verbose=False,
    )
    payload.update(overrides)
    if FinetuneConfig is not None:
        return FinetuneConfig(**_filter_kwargs(FinetuneConfig, payload))
    return _config_dict(**payload)


# ---------------------------------------------------------------------------
# Data / model building
# ---------------------------------------------------------------------------
def build_loaders(
    dataset: str,
    backbone: str,
    *,
    device: Optional[Any] = None,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    imgsize: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    download: bool = True,
    seed: int = 0,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    drop_last: bool = False,
) -> Tuple[Any, Any]:
    """Build ``(train_loader, test_loader)`` with the addendum transforms."""
    if build_dataloaders is None:
        raise RuntimeError("smm_vr.data.datasets.build_dataloaders is unavailable")
    size = batch_size if batch_size is not None else batch_size_for(dataset)
    train_loader, test_loader, _spec = build_dataloaders(
        canonical_dataset(dataset),
        backbone=backbone,
        root=root,
        data_root=data_root,
        imgsize=input_size_for_backbone(backbone, imgsize),
        batch_size=size,
        test_batch_size=size,
        num_workers=num_workers,
        download=download,
        train_fraction=train_fraction,
        split_seed=split_seed,
        drop_last=drop_last,
        device=device,
        generator_seed=seed,
    )
    return train_loader, test_loader


def _final_accuracy(history: Any) -> Optional[float]:
    """Best-effort extraction of a test accuracy (percent) from a history object."""
    if history is None:
        return None
    if isinstance(history, (int, float)):
        return float(history)
    for attr in ("best_test_accuracy", "final_test_accuracy"):
        value = getattr(history, attr, None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    if isinstance(history, dict):
        for key in ("best_test_accuracy", "final_test_accuracy", "accuracy", "test_accuracy"):
            if history.get(key) is not None:
                try:
                    return float(history[key])
                except Exception:
                    continue
        epochs = history.get("epochs") or history.get("history")
        if isinstance(epochs, list) and epochs:
            last = epochs[-1]
            if isinstance(last, dict):
                for key in ("test_accuracy", "accuracy", "test_acc"):
                    if last.get(key) is not None:
                        return float(last[key])
            else:
                value = getattr(last, "test_accuracy", None)
                if value is not None:
                    return float(value)
    return None


def _fallback_train(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    *,
    config: Any,
    device: Optional[Any] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Minimal local training loop used only if the finetuning module is absent."""
    if torch is None or nn is None:
        return {"accuracy": None, "error": "torch unavailable"}
    dev = resolve_device(device if device is not None else getattr(config, "device", None))
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {"accuracy": None, "error": "no trainable parameters"}
    lr = float(getattr(config, "lr", 0.01))
    momentum = float(getattr(config, "momentum", 0.9))
    epochs = int(getattr(config, "epochs", 100))
    milestones = list(getattr(config, "milestones", (100, 145)) or [])
    gamma = float(getattr(config, "gamma", 0.1))
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    log_every = int(getattr(config, "log_every", 10) or 10)
    epochs_stats: List[Dict[str, Any]] = []
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            images, targets = batch[0], batch[1]
            images = images.to(dev)
            targets = targets.to(dev)
            optimizer.zero_grad()
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        acc = _evaluate_model(model, test_loader, dev)
        best = max(best, acc)
        epochs_stats.append({"epoch": epoch, "loss": running / max(seen, 1), "test_accuracy": acc})
        if logger is not None and (epoch % log_every == 0 or epoch == epochs):
            logger(f"    epoch {epoch}/{epochs} loss={running / max(seen, 1):.4f} test={acc:.2f}")
    return {"accuracy": best, "epochs": epochs_stats}


def _evaluate_model(model: Any, loader: Any, device: Any = None, max_batches: Optional[int] = None) -> float:
    """Top-1 accuracy (percent) of a model over a loader."""
    if torch is None:
        return float("nan")
    dev = device if device is not None else next(model.parameters()).device
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= int(max_batches):
                break
            images, targets = batch[0], batch[1]
            logits = model(images.to(dev))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            prediction = logits.argmax(dim=1)
            correct += int((prediction == targets.to(dev)).sum().item())
            total += int(targets.numel())
    return 100.0 * correct / total if total else float("nan")


# ---------------------------------------------------------------------------
# Single runs
# ---------------------------------------------------------------------------
def run_single_lora(
    dataset: str,
    *,
    backbone: str = LORA_BACKBONE,
    seed: int = 0,
    device: Optional[Any] = None,
    rank: int = DEFAULT_LORA_RANK,
    lr: float = DEFAULT_LORA_LR,
    epochs: int = DEFAULT_LORA_EPOCHS,
    milestones: Optional[Sequence[int]] = None,
    num_workers: int = 4,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    imgsize: Optional[int] = None,
    download: bool = True,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    batch_size: Optional[int] = None,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One LoRA-for-ViT run (Appendix E.1 / Table 13)."""
    dataset = canonical_dataset(dataset)
    backbone = canonical_backbone(backbone)
    started = time.time()
    log = logger if logger is not None else (lambda *_: None)
    set_seed(seed)
    classes = num_target_classes(dataset)
    result: Dict[str, Any] = {
        "dataset": dataset, "backbone": backbone, "method": "lora", "seed": int(seed),
        "rank": int(rank), "epochs": int(epochs), "accuracy": None,
    }
    try:
        train_loader, test_loader = build_loaders(
            dataset, backbone, device=device, root=root, data_root=data_root,
            imgsize=imgsize, batch_size=batch_size, num_workers=num_workers,
            download=download, seed=seed, train_fraction=train_fraction,
            split_seed=split_seed,
        )
    except Exception as exc:  # pragma: no cover
        result["error"] = f"failed to build data: {exc}"
        result["elapsed_seconds"] = time.time() - started
        return result
    config = make_lora_config(
        dataset, backbone, seed=seed, device=device, imgsize=imgsize, rank=rank,
        alpha=float(rank), lr=lr, epochs=epochs, milestones=milestones,
        batch_size=batch_size, num_workers=num_workers, **kwargs,
    )
    try:
        if build_lora_model is None or train_lora is None:
            raise RuntimeError("methods.finetuning unavailable")
        model = build_lora_model(
            backbone, num_classes=classes, rank=rank, alpha=float(rank),
            target="qkv", train_head=True, device=resolve_device(device),
        )
        history = train_lora(
            model, train_loader, test_loader=test_loader, config=config,
            dataset=dataset, device=resolve_device(device), history=None, logger=log,
        )
        result["accuracy"] = _final_accuracy(history)
        result["history"] = history
    except Exception as exc:
        log(f"  [warn] LoRA path failed ({exc}); falling back to local training loop")
        try:
            if build_lora_model is not None:
                model = build_lora_model(
                    backbone, num_classes=classes, rank=rank, alpha=float(rank),
                    target="qkv", train_head=True, device=resolve_device(device),
                )
                payload = _fallback_train(
                    model, train_loader, test_loader, config=config,
                    device=device, logger=log,
                )
                result["accuracy"] = payload.get("accuracy")
            else:
                result["error"] = str(exc)
        except Exception as exc2:
            result["error"] = f"{exc} | fallback failed: {exc2}"
    result["elapsed_seconds"] = time.time() - started
    return result


def run_single_finetune_fc(
    dataset: str,
    *,
    backbone: str = FINETUNE_FC_BACKBONE,
    use_smm: bool = False,
    seed: int = 0,
    device: Optional[Any] = None,
    lr: float = DEFAULT_FINETUNE_LR,
    epochs: int = DEFAULT_FINETUNE_EPOCHS,
    milestones: Optional[Sequence[int]] = None,
    num_workers: int = 4,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    imgsize: Optional[int] = None,
    download: bool = True,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    batch_size: Optional[int] = None,
    patch_size: int = 8,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One ``Finetuning-FC`` run with or without the SMM input module (Table 14)."""
    dataset = canonical_dataset(dataset)
    backbone = canonical_backbone(backbone)
    method = "finetune_fc_smm" if use_smm else "finetune_fc"
    started = time.time()
    log = logger if logger is not None else (lambda *_: None)
    set_seed(seed)
    classes = num_target_classes(dataset)
    result: Dict[str, Any] = {
        "dataset": dataset, "backbone": backbone, "method": method,
        "seed": int(seed), "epochs": int(epochs), "accuracy": None,
    }
    try:
        train_loader, test_loader = build_loaders(
            dataset, backbone, device=device, root=root, data_root=data_root,
            imgsize=imgsize, batch_size=batch_size, num_workers=num_workers,
            download=download, seed=seed, train_fraction=train_fraction,
            split_seed=split_seed,
        )
    except Exception as exc:  # pragma: no cover
        result["error"] = f"failed to build data: {exc}"
        result["elapsed_seconds"] = time.time() - started
        return result
    config = make_finetune_fc_config(
        dataset, backbone, use_smm=use_smm, seed=seed, device=device, imgsize=imgsize,
        lr=lr, epochs=epochs, milestones=milestones, batch_size=batch_size,
        num_workers=num_workers, patch_size=patch_size, **kwargs,
    )
    dev = resolve_device(device)
    try:
        if build_classifier is None:
            raise RuntimeError("models.pretrained.build_classifier unavailable")
        classifier = build_classifier(backbone, device=dev)
    except Exception as exc:
        result["error"] = f"failed to build classifier: {exc}"
        result["elapsed_seconds"] = time.time() - started
        return result

    model: Any = None
    try:
        if use_smm:
            if build_finetune_fc_with_smm is None:
                raise RuntimeError("build_finetune_fc_with_smm unavailable")
            model = build_finetune_fc_with_smm(
                classifier, classes, backbone=backbone,
                input_size=input_size_for_backbone(backbone, imgsize),
                patch_size=patch_size,
                num_mask_layers=num_mask_layers_for(backbone),
                feature_dim=FEATURE_DIMS.get(backbone), device=dev,
            )
        else:
            if build_finetune_fc is None:
                raise RuntimeError("build_finetune_fc unavailable")
            model = build_finetune_fc(
                classifier, classes, backbone=backbone,
                input_size=input_size_for_backbone(backbone, imgsize),
                feature_dim=FEATURE_DIMS.get(backbone), device=dev,
            )
    except Exception as exc:
        result["error"] = f"failed to build finetuning model: {exc}"
        result["elapsed_seconds"] = time.time() - started
        return result

    history = None
    trainer = train_finetune_fc_with_smm if use_smm else train_finetune_fc
    try:
        if trainer is not None:
            history = trainer(
                model, train_loader, test_loader=test_loader, config=config,
                dataset=dataset, device=dev, f_out=None, label_mapping=None,
                history=None, logger=log,
            )
        elif train_finetuning_method is not None:
            history = train_finetuning_method(
                method, classifier=classifier, model=model, train_loader=train_loader,
                test_loader=test_loader, num_classes=classes, dataset=dataset,
                backbone=backbone, config=config, device=dev, f_out=None,
                label_mapping=None, logger=log,
            )
        else:
            raise RuntimeError("no finetuning trainer available")
        result["accuracy"] = _final_accuracy(history)
        result["history"] = history
    except Exception as exc:
        log(f"  [warn] finetuning path failed ({exc}); falling back to local training loop")
        payload = _fallback_train(model, train_loader, test_loader, config=config, device=dev, logger=log)
        result["accuracy"] = payload.get("accuracy")
        if result["accuracy"] is None:
            result["error"] = str(exc)
    result["elapsed_seconds"] = time.time() - started
    return result


def run_lora_seeds(
    dataset: str,
    *,
    backbone: str = LORA_BACKBONE,
    seeds: Optional[Sequence[int]] = None,
    verbose: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """LoRA run over the three seeds with mean ± std aggregation."""
    seed_list = resolve_seeds(seeds)
    per_seed: List[Optional[float]] = []
    runs: List[Dict[str, Any]] = []
    for seed in seed_list:
        run = run_single_lora(
            dataset, backbone=backbone, seed=seed, logger=logger, **kwargs,
        )
        per_seed.append(run.get("accuracy"))
        runs.append({k: v for k, v in run.items() if k != "history"})
        if verbose:
            print(f"    lora seed={seed} acc={run.get('accuracy')}")
    values = [v for v in per_seed if v is not None]
    mean, std = aggregate_seeds(values)
    return {
        "dataset": canonical_dataset(dataset),
        "method": "lora",
        "backbone": canonical_backbone(backbone),
        "per_seed_accuracy": per_seed,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std),
        "runs": runs,
        "reference": TABLE13_LORA.get(canonical_dataset(dataset)),
    }


def run_finetune_fc_seeds(
    dataset: str,
    *,
    backbone: str = FINETUNE_FC_BACKBONE,
    use_smm: bool = False,
    seeds: Optional[Sequence[int]] = None,
    verbose: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Finetuning-FC (± SMM) over the three seeds with mean ± std aggregation."""
    seed_list = resolve_seeds(seeds)
    method = "finetune_fc_smm" if use_smm else "finetune_fc"
    per_seed: List[Optional[float]] = []
    runs: List[Dict[str, Any]] = []
    for seed in seed_list:
        run = run_single_finetune_fc(
            dataset, backbone=backbone, use_smm=use_smm, seed=seed,
            logger=logger, **kwargs,
        )
        per_seed.append(run.get("accuracy"))
        runs.append({k: v for k, v in run.items() if k != "history"})
        if verbose:
            print(f"    {method} seed={seed} acc={run.get('accuracy')}")
    values = [v for v in per_seed if v is not None]
    mean, std = aggregate_seeds(values)
    reference = TABLE14_REFERENCE[method]
    return {
        "dataset": canonical_dataset(dataset),
        "method": method,
        "backbone": canonical_backbone(backbone),
        "per_seed_accuracy": per_seed,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std),
        "runs": runs,
        "reference": reference.get(canonical_dataset(dataset)),
    }


# ---------------------------------------------------------------------------
# Table runners
# ---------------------------------------------------------------------------
def run_table13(
    datasets: Optional[Sequence[str]] = None,
    *,
    seeds: Optional[Sequence[int]] = None,
    backbone: str = LORA_BACKBONE,
    verbose: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Table 13 (LoRA vs SMM on the four ``32 x 32`` tasks)."""
    names = [canonical_dataset(d) for d in (datasets or TABLE13_DATASETS)]
    results: Dict[str, Dict[str, Any]] = {"lora": {}, "ours": {}}
    for dataset in names:
        if verbose:
            print(f"[table13] {dataset}: LoRA")
        results["lora"][dataset] = run_lora_seeds(
            dataset, backbone=backbone, seeds=seeds, verbose=verbose, logger=logger, **kwargs,
        )
        if verbose:
            print(f"[table13] {dataset}: SMM (+ finetuned head)")
        # "Since LoRA for ViT already includes finetuning the fully connected
        # layers, we also incorporate it in SMM." -> SMM with a finetuned head.
        results["ours"][dataset] = run_finetune_fc_seeds(
            dataset, backbone=backbone, use_smm=True, seeds=seeds,
            verbose=verbose, logger=logger, **kwargs,
        )
    averages = {
        "lora": _mean_of(results["lora"], names),
        "ours": _mean_of(results["ours"], names),
    }
    comparison = compare_table13(results, verbose=verbose)
    payload = {
        "experiment": "table13",
        "datasets": names,
        "backbone": canonical_backbone(backbone),
        "seeds": resolve_seeds(seeds),
        "results": results,
        "averages": averages,
        "comparison": comparison,
        "table": format_table13(results, averages),
        "reference": {
            "lora": TABLE13_LORA,
            "ours": TABLE13_SMM,
            "averages": TABLE13_AVERAGES,
            "extra_parameters_m": TABLE13_EXTRA_PARAMETERS_M,
        },
    }
    FINETUNING_RESULTS["table13"] = payload
    return payload


def run_table14(
    datasets: Optional[Sequence[str]] = None,
    *,
    seeds: Optional[Sequence[int]] = None,
    backbone: str = FINETUNE_FC_BACKBONE,
    verbose: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Table 14 (Finetuning-FC without/with the SMM module)."""
    names = [canonical_dataset(d) for d in (datasets or TABLE14_DATASET_ORDER)]
    results: Dict[str, Dict[str, Any]] = {"finetune_fc": {}, "finetune_fc_smm": {}}
    for dataset in names:
        if verbose:
            print(f"[table14] {dataset}: Finetuning-FC")
        results["finetune_fc"][dataset] = run_finetune_fc_seeds(
            dataset, backbone=backbone, use_smm=False, seeds=seeds,
            verbose=verbose, logger=logger, **kwargs,
        )
        if verbose:
            print(f"[table14] {dataset}: Finetuning-FC + SMM")
        results["finetune_fc_smm"][dataset] = run_finetune_fc_seeds(
            dataset, backbone=backbone, use_smm=True, seeds=seeds,
            verbose=verbose, logger=logger, **kwargs,
        )
    averages = {
        "finetune_fc": _mean_of(results["finetune_fc"], names),
        "finetune_fc_smm": _mean_of(results["finetune_fc_smm"], names),
    }
    averages["gain"] = averages["finetune_fc_smm"] - averages["finetune_fc"]
    comparison = compare_table14(results, verbose=verbose)
    payload = {
        "experiment": "table14",
        "datasets": names,
        "backbone": canonical_backbone(backbone),
        "seeds": resolve_seeds(seeds),
        "results": results,
        "averages": averages,
        "comparison": comparison,
        "table": format_table14(results, averages),
        "reference": {
            "finetune_fc": TABLE14_FINETUNE_FC,
            "finetune_fc_smm": TABLE14_FINETUNE_FC_SMM,
            "averages": TABLE14_AVERAGES,
        },
    }
    FINETUNING_RESULTS["table14"] = payload
    return payload


def _mean_of(block: Dict[str, Dict[str, Any]], names: Sequence[str]) -> float:
    values = [
        block[name]["mean"] for name in names
        if name in block and block[name].get("mean") is not None
        and not math.isnan(float(block[name]["mean"]))
    ]
    return float(sum(values) / len(values)) if values else float("nan")


def run_finetuning_experiment(
    *,
    mode: str = "all",
    datasets: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    backbone: Optional[str] = None,
    output_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the finetuning comparisons (Table 13 and/or Table 14)."""
    log = logger if logger is not None else (print if verbose else (lambda *_: None))
    key = str(mode or "all").strip().lower().replace("-", "_")
    if key in ("table13", "lora", "e13", "e.1"):
        key = "table13"
    elif key in ("table14", "fc", "finetune_fc", "e14", "e.2"):
        key = "table14"
    elif key not in ("all", "tables13_14"):
        raise ValueError(f"Unknown finetuning mode {mode!r}; expected all|table13|table14")
    else:
        key = "all"

    payload: Dict[str, Any] = {
        "experiment": "finetuning",
        "mode": key,
        "seeds": resolve_seeds(seeds),
    }
    if key in ("all", "table13"):
        payload["table13"] = run_table13(
            datasets=datasets, seeds=seeds,
            backbone=canonical_backbone(backbone or LORA_BACKBONE),
            verbose=verbose, logger=log, **kwargs,
        )
    if key in ("all", "table14"):
        payload["table14"] = run_table14(
            datasets=datasets, seeds=seeds,
            backbone=canonical_backbone(backbone or FINETUNE_FC_BACKBONE),
            verbose=verbose, logger=log, **kwargs,
        )
    if save:
        try:
            payload["path"] = save_finetuning_results(payload, output_dir)
        except Exception as exc:  # pragma: no cover
            log(f"[warn] could not save finetuning results: {exc}")
    return payload


# Registry alias used by smm_vr.experiments dispatch.
run_finetuning = run_finetuning_experiment


def run_single_finetuning(
    dataset: str,
    method: str = "lora",
    *,
    seeds: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience entry: run one finetuning method on one dataset over seeds."""
    name = canonical_method(method)
    if name == "lora":
        return run_lora_seeds(dataset, seeds=seeds, **kwargs)
    return run_finetune_fc_seeds(dataset, use_smm=(name == "finetune_fc_smm"), seeds=seeds, **kwargs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_table13(
    results: Dict[str, Dict[str, Any]],
    averages: Optional[Dict[str, float]] = None,
    *,
    datasets: Optional[Sequence[str]] = None,
    decimals: int = 1,
) -> str:
    """Render a Table-13 style text table (LoRA vs SMM)."""
    names = [canonical_dataset(d) for d in (datasets or TABLE13_DATASETS)]
    header = (
        f"{'method':<24}{'params':>9}"
        + "".join(f"{_DATASET_DISPLAY.get(d, d):>10}" for d in names)
        + f"{'average':>10}"
    )
    lines = [header, "-" * len(header)]
    for key, display in (("lora", "Finetuning-LoRA"), ("ours", "Our SMM")):
        params = TABLE13_EXTRA_PARAMETERS_M.get(key)
        cells = []
        for dataset in names:
            entry = results.get(key, {}).get(dataset, {})
            mean = entry.get("mean")
            cells.append("       n/a" if mean is None or (isinstance(mean, float) and math.isnan(mean))
                         else f"{mean:>10.{decimals}f}")
        average = (averages or {}).get(key)
        avg_cell = "       n/a" if average is None or math.isnan(float(average)) else f"{average:>10.{decimals}f}"
        lines.append(
            f"{display:<24}{('%0.2f M' % params) if params is not None else '   n/a':>9}"
            + "".join(cells) + avg_cell
        )
    ref = (
        f"{'paper (LoRA)':<24}{'':>9}"
        + "".join(f"{TABLE13_LORA.get(d, float('nan')):>10.{decimals}f}" for d in names)
        + f"{TABLE13_AVERAGES['lora']['32x32']:>10.{decimals}f}"
    )
    ref_smm = (
        f"{'paper (Ours)':<24}{'':>9}"
        + "".join(f"{TABLE13_SMM.get(d, float('nan')):>10.{decimals}f}" for d in names)
        + f"{TABLE13_AVERAGES['ours']['32x32']:>10.{decimals}f}"
    )
    lines.extend(["-" * len(header), ref, ref_smm])
    return "\n".join(lines)


def format_table14(
    results: Dict[str, Dict[str, Any]],
    averages: Optional[Dict[str, float]] = None,
    *,
    datasets: Optional[Sequence[str]] = None,
    decimals: int = 1,
) -> str:
    """Render a Table-14 style text table (Finetuning-FC ± SMM)."""
    names = [canonical_dataset(d) for d in (datasets or TABLE14_DATASET_ORDER)]
    header = f"{'method':<28}" + "".join(f"{_DATASET_DISPLAY.get(d, d):>11}" for d in names) + f"{'average':>10}"
    lines = [header, "-" * len(header)]
    for key, display in (("finetune_fc", "Finetuning-FC"), ("finetune_fc_smm", "Finetuning-FC + Ours")):
        cells = []
        for dataset in names:
            entry = results.get(key, {}).get(dataset, {})
            mean = entry.get("mean")
            cells.append("        n/a" if mean is None or (isinstance(mean, float) and math.isnan(mean))
                         else f"{mean:>11.{decimals}f}")
        average = (averages or {}).get(key)
        avg_cell = "       n/a" if average is None or math.isnan(float(average)) else f"{average:>10.{decimals}f}"
        lines.append(f"{display:<28}" + "".join(cells) + avg_cell)
    ref = f"{'paper (FC)':<28}" + "".join(f"{TABLE14_FINETUNE_FC.get(d, float('nan')):>11.{decimals}f}" for d in names) + f"{TABLE14_FINETUNE_FC['average']:>10.{decimals}f}"
    ref_smm = f"{'paper (FC + Ours)':<28}" + "".join(f"{TABLE14_FINETUNE_FC_SMM.get(d, float('nan')):>11.{decimals}f}" for d in names) + f"{TABLE14_FINETUNE_FC_SMM['average']:>10.{decimals}f}"
    lines.extend(["-" * len(header), ref, ref_smm])
    return "\n".join(lines)


def compare_table13(
    results: Dict[str, Dict[str, Any]],
    *,
    tolerance: float = 3.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare measured LoRA/SMM numbers against Table 13 references.

    The paper's claim for Appendix E.1 is that SMM clearly outperforms LoRA on
    the four low-resolution tasks (average 90.0 vs 77.9).  We therefore also
    report whether the expected ordering holds.
    """
    per_dataset: Dict[str, Dict[str, Any]] = {}
    wins = 0
    counted = 0
    for dataset in TABLE13_DATASETS:
        lora = results.get("lora", {}).get(dataset, {}).get("mean")
        ours = results.get("ours", {}).get(dataset, {}).get("mean")
        row: Dict[str, Any] = {
            "lora": lora, "ours": ours,
            "reference_lora": TABLE13_LORA[dataset],
            "reference_ours": TABLE13_SMM[dataset],
        }
        if lora is not None and ours is not None and not (math.isnan(lora) or math.isnan(ours)):
            row["delta_vs_lora"] = ours - lora
            row["smm_better"] = ours > lora
            counted += 1
            wins += int(ours > lora)
        if lora is not None and not math.isnan(lora):
            row["lora_deviation"] = lora - TABLE13_LORA[dataset]
        if ours is not None and not math.isnan(ours):
            row["ours_deviation"] = ours - TABLE13_SMM[dataset]
        per_dataset[dataset] = row
    averages = {
        "lora": _mean_of(results.get("lora", {}), TABLE13_DATASETS),
        "ours": _mean_of(results.get("ours", {}), TABLE13_DATASETS),
    }
    reference_gap = TABLE13_AVERAGES["ours"]["32x32"] - TABLE13_AVERAGES["lora"]["32x32"]
    measured_gap = (
        averages["ours"] - averages["lora"]
        if not (math.isnan(averages["ours"]) or math.isnan(averages["lora"])) else float("nan")
    )
    report = {
        "per_dataset": per_dataset,
        "measured_averages": averages,
        "reference_averages": {k: v for k, v in TABLE13_AVERAGES.items()},
        "reference_gap": reference_gap,
        "measured_gap": measured_gap,
        "smm_wins": wins,
        "smm_comparisons": counted,
        "smm_wins_majority": (counted > 0 and wins >= (counted + 1) // 2),
        "smm_favoured_low_res": (not math.isnan(measured_gap)) and measured_gap > 0,
        "within_tolerance": all(
            abs(row.get("ours_deviation", 0.0)) <= tolerance
            for row in per_dataset.values()
            if row.get("ours_deviation") is not None
        ) if per_dataset else False,
        "tolerance": tolerance,
    }
    if verbose:
        print(format_comparison_report({"table13": report}))
    return report


def compare_table14(
    results: Dict[str, Dict[str, Any]],
    *,
    dataset_order: Optional[Sequence[str]] = None,
    tolerance: float = 3.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare measured Finetuning-FC (± SMM) numbers against Table 14."""
    names = [canonical_dataset(d) for d in (dataset_order or TABLE14_DATASET_ORDER)]
    per_dataset: Dict[str, Dict[str, Any]] = {}
    wins = 0
    counted = 0
    for dataset in names:
        fc = results.get("finetune_fc", {}).get(dataset, {}).get("mean")
        fc_smm = results.get("finetune_fc_smm", {}).get(dataset, {}).get("mean")
        row: Dict[str, Any] = {
            "finetune_fc": fc, "finetune_fc_smm": fc_smm,
            "reference_finetune_fc": TABLE14_FINETUNE_FC.get(dataset),
            "reference_finetune_fc_smm": TABLE14_FINETUNE_FC_SMM.get(dataset),
        }
        if fc is not None and fc_smm is not None and not (math.isnan(fc) or math.isnan(fc_smm)):
            row["gain"] = fc_smm - fc
            row["smm_improves"] = fc_smm >= fc - 1e-9
            counted += 1
            wins += int(fc_smm >= fc - 1e-9)
        if fc is not None and not math.isnan(fc) and TABLE14_FINETUNE_FC.get(dataset) is not None:
            row["finetune_fc_deviation"] = fc - TABLE14_FINETUNE_FC[dataset]
        if fc_smm is not None and not math.isnan(fc_smm) and TABLE14_FINETUNE_FC_SMM.get(dataset) is not None:
            row["finetune_fc_smm_deviation"] = fc_smm - TABLE14_FINETUNE_FC_SMM[dataset]
        per_dataset[dataset] = row
    averages = {
        "finetune_fc": _mean_of(results.get("finetune_fc", {}), names),
        "finetune_fc_smm": _mean_of(results.get("finetune_fc_smm", {}), names),
    }
    measured_gain = (
        averages["finetune_fc_smm"] - averages["finetune_fc"]
        if not (math.isnan(averages["finetune_fc_smm"]) or math.isnan(averages["finetune_fc"]))
        else float("nan")
    )
    report = {
        "per_dataset": per_dataset,
        "measured_averages": averages,
        "reference_averages": dict(TABLE14_AVERAGES),
        "measured_gain": measured_gain,
        "reference_gain": TABLE14_AVERAGES["gain"],
        "smm_wins": wins,
        "smm_comparisons": counted,
        "smm_improves_majority": (counted > 0 and wins >= (counted + 1) // 2),
        "orthogonal_gain_positive": (not math.isnan(measured_gain)) and measured_gain > 0,
        "within_tolerance": all(
            abs(row.get("finetune_fc_smm_deviation", 0.0)) <= tolerance
            for row in per_dataset.values()
            if row.get("finetune_fc_smm_deviation") is not None
        ) if per_dataset else False,
        "tolerance": tolerance,
    }
    if verbose:
        print(format_comparison_report({"table14": report}))
    return report


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """Human-readable rendering of :func:`compare_table13` / :func:`compare_table14`."""
    lines: List[str] = []
    if "table13" in comparison or "per_dataset" in comparison and "reference_gap" in comparison:
        report = comparison.get("table13", comparison)
        lines.append("Table 13 (LoRA vs SMM, ViT-Large 384x384):")
        for dataset, row in (report.get("per_dataset") or {}).items():
            lines.append(
                f"  {dataset:<11} LoRA={_fmt(row.get('lora'))} SMM={_fmt(row.get('ours'))} "
                f"(paper {row.get('reference_lora')} / {row.get('reference_ours')})"
            )
        lines.append(
            f"  averages: LoRA={_fmt(report.get('measured_averages', {}).get('lora'))} "
            f"SMM={_fmt(report.get('measured_averages', {}).get('ours'))} "
            f"(paper gap +{report.get('reference_gap')})"
        )
        lines.append(f"  SMM wins {report.get('smm_wins')}/{report.get('smm_comparisons')} low-res tasks")
    if "table14" in comparison or (isinstance(comparison.get("per_dataset"), dict) and "reference_gain" in comparison):
        report = comparison.get("table14", comparison)
        lines.append("Table 14 (Finetuning-FC without/with SMM, ResNet-50):")
        for dataset, row in (report.get("per_dataset") or {}).items():
            lines.append(
                f"  {dataset:<11} FC={_fmt(row.get('finetune_fc'))} FC+SMM={_fmt(row.get('finetune_fc_smm'))} "
                f"(paper {row.get('reference_finetune_fc')} / {row.get('reference_finetune_fc_smm')})"
            )
        lines.append(
            f"  averages: FC={_fmt(report.get('measured_averages', {}).get('finetune_fc'))} "
            f"FC+SMM={_fmt(report.get('measured_averages', {}).get('finetune_fc_smm'))} "
            f"(paper gain +{report.get('reference_gain')})"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if math.isnan(float(value)):
            return "n/a"
        return f"{float(value):.1f}"
    except Exception:
        return str(value)


def _strip_histories(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_histories(v)
            for k, v in obj.items()
            if k not in ("history", "histories", "config", "model", "runs")
        }
    if isinstance(obj, (list, tuple)):
        return [_strip_histories(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(type(obj).__name__)


def save_finetuning_results(
    payload: Dict[str, Any],
    output_dir: Optional[str] = None,
    *,
    filename: str = "finetuning_results.json",
) -> str:
    """Persist finetuning results (histories stripped) to JSON."""
    directory = output_dir or os.path.join("outputs", "results")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_strip_histories(payload), handle, indent=2, sort_keys=False)
    return path


def describe_finetuning_study(
    datasets: Optional[Sequence[str]] = None,
    *,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Study metadata (used by the ``--describe`` CLI flag)."""
    return {
        "experiment": "finetuning",
        "tables": ["table13 (LoRA vs SMM, ViT-Large 384x384)",
                   "table14 (Finetuning-FC without/with SMM, ResNet-50)"],
        "lora_backbone": LORA_BACKBONE,
        "finetune_fc_backbone": FINETUNE_FC_BACKBONE,
        "lora_settings": dict(LORA_TRAINING_DEFAULTS),
        "finetune_fc_settings": dict(TRAINING_DEFAULTS),
        "datasets": list(datasets or TABLE14_DATASET_ORDER),
        "seeds": resolve_seeds(seeds),
        "reference_table13": TABLE13_REFERENCE,
        "reference_table14": TABLE14_REFERENCE,
        "notes": [
            "Appendix E.1: LoRA rank 6 (~0.60 M extra params) vs SMM (~0.54 M), "
            "LR 0.01, 10 epochs, ViT-Large with 384x384 input.",
            "SMM is combined with a finetuned FC head in Table 13 because LoRA "
            "already finetunes the FC layers ('we also incorporate it in SMM').",
            "Appendix E.2: Finetuning-FC on ResNet-50 averages 75.3 and "
            "79.2 with the SMM input module attached (orthogonality).",
        ],
    }


# Backwards/registry friendly aliases.
describe_finetuning_experiment = describe_finetuning_study
run_finetuning_experiment_registry = run_finetuning_experiment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the SMM finetuning comparisons (Tables 13 and 14).",
    )
    parser.add_argument("--mode", default="all",
                        choices=["all", "table13", "table14", "lora", "finetune_fc"],
                        help="Which comparison to run (default: all).")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Datasets to evaluate (default: the paper's lists).")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Seeds (default: 0 1 2).")
    parser.add_argument("--backbone", default=None,
                        help="Backbone override (default: vit_large for Table 13, resnet50 for Table 14).")
    parser.add_argument("--data-root", default=None, help="Dataset root directory.")
    parser.add_argument("--output-dir", default=None, help="Where to write JSON results.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="Deterministic class-balanced subsampling for debugging.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON results.")
    parser.add_argument("--describe", action="store_true", help="Print the study description and exit.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.describe:
        print(json.dumps(describe_finetuning_study(args.datasets, seeds=args.seeds), indent=2))
        return 0
    payload = run_finetuning_experiment(
        mode="all" if args.mode in ("lora", "finetune_fc") else args.mode,
        datasets=args.datasets,
        seeds=args.seeds,
        backbone=args.backbone,
        output_dir=args.output_dir,
        save=not args.no_save,
        verbose=not args.quiet,
        data_root=args.data_root,
        num_workers=args.num_workers,
        device=args.device,
        train_fraction=args.train_fraction,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    if not args.quiet:
        for key in ("table13", "table14"):
            if key in payload:
                print(payload[key].get("table", ""))
                print()
        if payload.get("path"):
            print(f"results written to {payload['path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
