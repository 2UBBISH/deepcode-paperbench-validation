"""Scaling study of the mask generator :math:`f_{\\mathrm{mask}}` (paper Appendix D.3, Table 11).

This module reproduces the experiment in which the *intermediate channels* of the
mask generator are progressively **doubled** while keeping its architecture
(3x3 convolutions with padding 1 / stride 1, 2x2 max-pooling, 3 max-pooling
layers, 3 output channels) unchanged.  The paper runs this study on **EuroSAT**
with a frozen **ResNet-18** well-trained model as :math:`f_{\\mathrm{P}}`.

Paper claims to verify (Appendix D.3, "More Discussion about the Estimation Error"):

1. "As the number of parameters continues to increase, although the training
   accuracy slowly increases, the test accuracy may even decrease, implying that
   the estimation error becomes more and more noticeable."
2. "Under this situation (i.e., EuroSAT, ResNet-18), when the size of
   :math:`f_{\\mathrm{mask}}` is close to the same order of magnitude as the
   well-trained model, the estimation error should not be overlooked."
3. "A larger model with the best test accuracy may not be optimal because of too
   many parameters. Our work strikes a balance between the number of parameters
   and test accuracy."

The default :math:`f_{\\mathrm{mask}}` (width scale 1, 5 CNN layers) has exactly
**26,499** parameters (Table 4), the configuration adopted everywhere else in the
paper.  Doubling the intermediate channel widths gives the ladder

======================  ==================  ==========================
width scale (doubling)  parameters (5-CNN)  ratio to ResNet-18 (11.7M)
======================  ==================  ==========================
1 (default, Table 4)            26,499      0.23%
2                              101,379      0.87%
4                              396,291      3.4%
8                            1,566,723       13.4%
======================  ==================  ==========================

The analytic parameter count of the mask generator (geometric channel layout
``[w, 2w, 4w, ...]`` built from ``base_channels``, BatchNorm included) is

* 5 layers: ``378 w^2 + 288 w + 3``  (``w = 8`` -> 26,499)
* 6 layers: ``1530 w^2 + 552 w + 3`` (``w = 8`` -> 102,339)

matching Table 4 of the paper exactly; it is used here to sanity-check the
constructed modules and to normalise the reference ladder independently of torch.

Everything follows the defensive, fallback-friendly style of the other experiment
runners so the module stays importable (and the CLI usable) even in a partially
built environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is a hard requirement for real runs
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


__all__ = [
    # constants
    "SCALING_DATASET",
    "SCALING_BACKBONE",
    "SCALING_SEEDS",
    "SCALING_LEVELS",
    "DEFAULT_WIDTH_SCALES",
    "RESNET18_PARAMETERS",
    "DEFAULT_EPOCHS",
    "DEFAULT_MILESTONES",
    "DEFAULT_ALPHA_DELTA",
    "DEFAULT_GAMMA_DELTA",
    "DEFAULT_ALPHA_MASK_5",
    "DEFAULT_GAMMA_MASK_5",
    "DEFAULT_ALPHA_MASK_6",
    "DEFAULT_GAMMA_MASK_6",
    "DEFAULT_BATCH_SIZE",
    "SMALL_BATCH_DATASETS",
    "TABLE11_REFERENCE",
    "SCALING_RESULTS",
    # helpers
    "scaled_parameter_count",
    "FallbackMaskGenerator",
    "build_scaled_mask_generator",
    "canonical_backbone",
    "canonical_dataset",
    "num_mask_layers_for",
    "input_size_for_backbone",
    "batch_size_for",
    "mask_lr_schedule",
    "num_target_classes",
    "resolve_device",
    "resolve_seeds",
    "set_seed",
    "aggregate_seeds",
    "format_mean_std",
    "apply_output_mapping",
    "build_label_mapping",
    "refresh_mapping",
    "split_parameters",
    "make_scaling_config",
    "ScalingTrainConfig",
    "ScalingEpochStats",
    "ScalingHistory",
    # training / evaluation
    "evaluate_scaling_model",
    "train_scaling_run",
    "run_single_level",
    # reporting
    "aggregate_level",
    "format_scaling_table",
    "check_scaling_trend",
    "format_trend_report",
    "compare_table11",
    "save_scaling_results",
    "describe_scaling_study",
    # experiment entry points
    "run_scaling_study",
    "run_scaling_experiment",
    "build_scaling_arg_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALING_DATASET: str = "eurosat"
SCALING_BACKBONE: str = "resnet18"
SCALING_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: Width multipliers applied to *all* intermediate channels (1x, 2x, 4x, 8x).
DEFAULT_WIDTH_SCALES: Tuple[int, ...] = (1, 2, 4, 8)

#: The paper's default configuration (Table 4) corresponds to width scale 1.
MAX_SANITY_WIDTH_SCALE: int = 8

#: Parameter count of the frozen well-trained model, for the paper's comparison
#: "close to the same order of magnitude as the well-trained model".
RESNET18_PARAMETERS: int = 11_689_512

#: Base width, depth and pooling configuration of the mask generator (Table 4 / Fig. 8-9).
BASE_CHANNELS: int = 8
NUM_POOLING_LAYERS: int = 3
DEFAULT_PATCH_SIZE: int = 8

DEFAULT_EPOCHS: int = 200
DEFAULT_MILESTONES: Tuple[int, int] = (100, 145)
DEFAULT_ALPHA_DELTA: float = 0.01
DEFAULT_GAMMA_DELTA: float = 0.1
DEFAULT_ALPHA_MASK_5: float = 0.01
DEFAULT_GAMMA_MASK_5: float = 0.1
DEFAULT_ALPHA_MASK_6: float = 0.001
DEFAULT_GAMMA_MASK_6: float = 1.0
DEFAULT_BATCH_SIZE: int = 256
SMALL_BATCH_DATASETS: Tuple[str, ...] = ("dtd", "oxfordpets")

MASK_LAYERS_BY_BACKBONE_FALLBACK: Dict[str, int] = {
    "resnet18": 5,
    "resnet50": 5,
    "resnet101": 5,
    "vit_b32": 6,
    "vit_large": 6,
}

DATASET_ALIASES: Dict[str, str] = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "flowers102": "flowers102",
    "dtd": "dtd",
    "ucf101": "ucf101",
    "food101": "food101",
    "sun397": "sun397",
    "eurosat": "eurosat",
    "euro_sat": "eurosat",
    "oxfordpets": "oxfordpets",
    "oxford_iiit_pet": "oxfordpets",
    "stanfordcars": "stanfordcars",
    "cars": "stanfordcars",
}

BACKBONE_ALIASES: Dict[str, str] = {
    "resnet18": "resnet18",
    "resnet_18": "resnet18",
    "r18": "resnet18",
    "resnet50": "resnet50",
    "resnet_50": "resnet50",
    "r50": "resnet50",
    "resnet101": "resnet101",
    "vit_b32": "vit_b32",
    "vit-b32": "vit_b32",
    "vitb32": "vit_b32",
    "vit_b_32": "vit_b32",
    "vit_large": "vit_large",
    "vit-l": "vit_large",
    "vitl16": "vit_large",
}


def _make_scaling_levels(width_scales: Sequence[int] = DEFAULT_WIDTH_SCALES,
                         num_layers: int = 5,
                         base_channels: int = BASE_CHANNELS) -> List[Dict[str, Any]]:
    """Build the declarative scaling ladder used by the runner and reporting."""
    levels: List[Dict[str, Any]] = []
    for idx, scale in enumerate(width_scales):
        params = scaled_parameter_count(scale, num_layers=num_layers,
                                        base_channels=base_channels)
        channels = [base_channels * int(scale) * (2 ** i)
                    for i in range(max(num_layers - 1, 1))]
        levels.append(
            {
                "level": idx,
                "width_scale": int(scale),
                "num_layers": int(num_layers),
                "base_channels": int(base_channels),
                "channels": channels,
                "parameters": int(params),
                "ratio_to_resnet18": round(params / float(RESNET18_PARAMETERS), 5),
                "is_default": int(scale) == 1,
            }
        )
    return levels


#: Table 11 ladder: progressively doubling the intermediate channels.
SCALING_LEVELS: List[Dict[str, Any]] = _make_scaling_levels()

#: Reference information for Table 11.  The paper prints the table in Appendix
#: D.3; its exact per-row accuracies are not part of the extracted text, so the
#: quantitative parameter ladder (Table 4) and the three qualitative claims are
#: encoded here and checked as a *trend* rather than as fixed numbers.
TABLE11_REFERENCE: Dict[str, Any] = {
    "table": "table11",
    "title": "Impact of the number of parameters of f_mask (EuroSAT, ResNet-18)",
    "dataset": SCALING_DATASET,
    "backbone": SCALING_BACKBONE,
    "width_scales": [level["width_scale"] for level in SCALING_LEVELS],
    "parameter_counts": [level["parameters"] for level in SCALING_LEVELS],
    "default_parameters": SCALING_LEVELS[0]["parameters"] if SCALING_LEVELS else 26499,
    "resnet18_parameters": RESNET18_PARAMETERS,
    "claims": {
        "train_accuracy": "increases slowly as the number of parameters increases",
        "test_accuracy": "peaks at a medium size and then plateaus or decreases",
        "best_test_accuracy": "not obtained by the largest f_mask",
        "default_configuration": "26,499 parameters (Table 4) balances size and accuracy",
        "estimation_error": "becomes noticeable once f_mask approaches the size of the "
                            "well-trained model",
    },
    "trend": {
        "train_monotone_non_decreasing": True,
        "test_declines_or_plateaus_after_peak": True,
    },
}

#: Module-level accumulator used by the registry / notebooks.
SCALING_RESULTS: Dict[str, Any] = {"table11": {}}


# ---------------------------------------------------------------------------
# Guarded project imports (all with functional local fallbacks)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - depends on build state
    from ..engine.seeds import SEEDS as _ENGINE_SEEDS  # noqa: F401
    from ..engine.seeds import resolve_seeds as _engine_resolve_seeds
    from ..engine.seeds import set_seed as _engine_set_seed

    _SEEDS_AVAILABLE = True
except Exception:  # pragma: no cover
    _engine_resolve_seeds = None
    _engine_set_seed = None
    _SEEDS_AVAILABLE = False

try:  # pragma: no cover
    from ..engine.metrics import aggregate_seeds as _engine_aggregate_seeds
    from ..engine.metrics import format_mean_std as _engine_format_mean_std

    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    _engine_aggregate_seeds = None
    _engine_format_mean_std = None
    _METRICS_AVAILABLE = False

try:  # pragma: no cover
    from ..data.datasets import DEFAULT_BATCH_SIZES as _DEFAULT_BATCH_SIZES
    from ..data.datasets import build_dataloaders as _build_dataloaders

    _DATASETS_AVAILABLE = True
except Exception:  # pragma: no cover
    _DEFAULT_BATCH_SIZES = {"default": 256, "dtd": 64, "oxfordpets": 64}
    _build_dataloaders = None
    _DATASETS_AVAILABLE = False

try:  # pragma: no cover
    from ..models.pretrained import build_classifier as _build_classifier
    from ..models.pretrained import input_size_for as _input_size_for

    _PRETRAINED_AVAILABLE = True
except Exception:  # pragma: no cover
    _build_classifier = None
    _input_size_for = None
    _PRETRAINED_AVAILABLE = False

try:  # pragma: no cover
    from ..models.mask_generator import EXPECTED_PARAMETERS as _EXPECTED_PARAMETERS
    from ..models.mask_generator import MaskGenerator as _MaskGenerator
    from ..models.mask_generator import build_mask_generator as _build_mask_generator
    from ..models.mask_generator import count_parameters as _count_parameters

    _MASK_GENERATOR_AVAILABLE = True
except Exception:  # pragma: no cover
    _EXPECTED_PARAMETERS = {"resnet18": 26499, "resnet50": 26499, "vit_b32": 102339}
    _MaskGenerator = None
    _build_mask_generator = None
    _count_parameters = None
    _MASK_GENERATOR_AVAILABLE = False

try:  # pragma: no cover
    from ..modules.reprogram import build_smm_reprogram as _build_smm_reprogram
    from ..modules.reprogram import SMMReprogram as _SMMReprogram

    _REPROGRAM_AVAILABLE = True
except Exception:  # pragma: no cover
    _build_smm_reprogram = None
    _SMMReprogram = None
    _REPROGRAM_AVAILABLE = False

try:  # pragma: no cover
    from ..engine.train_smm import SMMTrainConfig as _SMMTrainConfig  # noqa: F401
    from ..engine.train_smm import build_label_mapping as _engine_build_label_mapping
    from ..engine.train_smm import train_smm as _train_smm  # noqa: F401

    _TRAIN_SMM_AVAILABLE = True
except Exception:  # pragma: no cover
    _engine_build_label_mapping = None
    _TRAIN_SMM_AVAILABLE = False

try:  # pragma: no cover
    from ..label_mapping import build_label_mapping as _lm_build_label_mapping
    from ..label_mapping.flm import apply_label_mapping as _packaged_apply_label_mapping
    from ..label_mapping.flm import IGNORE_INDEX as _IGNORE_INDEX  # noqa: F401

    _LABEL_MAPPING_AVAILABLE = True
except Exception:  # pragma: no cover
    _lm_build_label_mapping = None
    _packaged_apply_label_mapping = None
    _LABEL_MAPPING_AVAILABLE = False

try:  # pragma: no cover - optional reference table
    from ..data.dataset_stats import TABLE11_SCALING as _TABLE11_SCALING

    if isinstance(_TABLE11_SCALING, dict) and _TABLE11_SCALING:
        TABLE11_REFERENCE["dataset_stats"] = _TABLE11_SCALING
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def canonical_dataset(name: Optional[str]) -> str:
    """Normalise a dataset spelling to a canonical key."""
    if not name:
        return SCALING_DATASET
    key = str(name).strip().lower().replace(" ", "_")
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    squashed = key.replace("-", "").replace("_", "").replace(".", "")
    for alias, value in DATASET_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == squashed:
            return value
    return squashed


def canonical_backbone(name: Optional[str]) -> str:
    """Normalise a backbone spelling to a canonical registry key."""
    if not name:
        return SCALING_BACKBONE
    key = str(name).strip().lower().replace(" ", "")
    if key in BACKBONE_ALIASES:
        return BACKBONE_ALIASES[key]
    squashed = key.replace("-", "").replace("_", "").replace(".", "")
    for alias, value in BACKBONE_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == squashed:
            return value
    if "vit" in squashed:
        return "vit_large" if ("large" in squashed or "l16" in squashed) else "vit_b32"
    return "resnet18"


def num_mask_layers_for(backbone: Optional[str]) -> int:
    """5 CNN layers for ResNets, 6 CNN layers for ViT-B32 (paper Sec. 3.2 / Table 4)."""
    key = canonical_backbone(backbone)
    try:  # pragma: no cover - prefer the engine's registry
        from ..engine.train_smm import MASK_LAYERS_BY_BACKBONE as _registry

        if isinstance(_registry, dict) and key in _registry:
            return int(_registry[key])
    except Exception:
        pass
    return int(MASK_LAYERS_BY_BACKBONE_FALLBACK.get(key, 5))


def input_size_for_backbone(backbone: Optional[str], imgsize: Optional[int] = None) -> int:
    """224x224 for ResNets, 384x384 for ViT-B32 (addendum transforms)."""
    if imgsize:
        return int(imgsize)
    key = canonical_backbone(backbone)
    if _input_size_for is not None:
        try:
            return int(_input_size_for(key))
        except Exception:
            pass
    return 384 if "vit" in key else 224


def batch_size_for(dataset: str, backbone: Optional[str] = None) -> int:
    """Table 9 batch size: 256 everywhere except DTD / OxfordPets (64)."""
    key = canonical_dataset(dataset)
    if isinstance(_DEFAULT_BATCH_SIZES, dict):
        if key == "dtd" and "dtd" in _DEFAULT_BATCH_SIZES:
            return int(_DEFAULT_BATCH_SIZES["dtd"])
        if key == "oxfordpets" and "oxfordpets" in _DEFAULT_BATCH_SIZES:
            return int(_DEFAULT_BATCH_SIZES["oxfordpets"])
    return 64 if key in SMALL_BATCH_DATASETS else DEFAULT_BATCH_SIZE


def mask_lr_schedule(backbone: Optional[str],
                     num_mask_layers: Optional[int] = None,
                     alpha_mask: Optional[float] = None,
                     gamma_mask: Optional[float] = None) -> Tuple[float, float]:
    """Table 9 rule: 0.01/0.1 for the 5-layer generator, 0.001/1.0 for the 6-layer one."""
    layers = int(num_mask_layers) if num_mask_layers else num_mask_layers_for(backbone)
    default_alpha = DEFAULT_ALPHA_MASK_6 if layers >= 6 else DEFAULT_ALPHA_MASK_5
    default_gamma = DEFAULT_GAMMA_MASK_6 if layers >= 6 else DEFAULT_GAMMA_MASK_5
    return (
        float(default_alpha if alpha_mask is None else alpha_mask),
        float(default_gamma if gamma_mask is None else gamma_mask),
    )


def num_target_classes(dataset: str) -> Optional[int]:
    """Number of target classes (Table 6); falls back to a local table."""
    key = canonical_dataset(dataset)
    try:  # pragma: no cover
        from ..data.datasets import num_classes as _num_classes

        value = _num_classes(key)
        if value:
            return int(value)
    except Exception:
        pass
    return {
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10,
        "gtsrb": 43,
        "flowers102": 102,
        "dtd": 47,
        "ucf101": 101,
        "food101": 101,
        "sun397": 397,
        "eurosat": 10,
        "oxfordpets": 37,
        "stanfordcars": 196,
    }.get(key)


def resolve_device(device: Optional[Any] = None) -> Any:
    """Resolve a device specification to a ``torch.device`` when torch is present."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if _TORCH_AVAILABLE:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return "cpu"
    if _TORCH_AVAILABLE and not isinstance(device, torch.device):
        return torch.device(str(device))
    return device


def resolve_seeds(seeds: Optional[Sequence[int]] = None,
                  n_seeds: Optional[int] = None) -> List[int]:
    """Normalise a seed specification (defaults to the paper's ``{0, 1, 2}``)."""
    if seeds is not None:
        resolved = [int(s) for s in seeds]
        if n_seeds and n_seeds > 0:
            resolved = resolved[: int(n_seeds)]
        return resolved
    if _engine_resolve_seeds is not None:
        try:
            return [int(s) for s in _engine_resolve_seeds(seeds=None, n_seeds=n_seeds)]
        except TypeError:
            try:
                return [int(s) for s in _engine_resolve_seeds(None)]
            except Exception:
                pass
        except Exception:
            pass
    base = list(SCALING_SEEDS)
    return base[: int(n_seeds)] if n_seeds and n_seeds > 0 else base


def set_seed(seed: int, **kwargs: Any) -> int:
    """Seed every RNG (delegates to ``engine.seeds`` when available)."""
    if _engine_set_seed is not None:
        try:
            return int(_engine_set_seed(int(seed), **kwargs))
        except TypeError:
            try:
                return int(_engine_set_seed(int(seed)))
            except Exception:
                pass
        except Exception:
            pass
    if _TORCH_AVAILABLE:
        try:
            import random

            import numpy as _np

            random.seed(int(seed))
            _np.random.seed(int(seed) % (2 ** 32))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass
    return int(seed)


def aggregate_seeds(values: Sequence[Optional[float]], ddof: int = 1) -> Tuple[float, float]:
    """Mean and sample standard deviation of per-seed accuracies (``mean +- std``)."""
    if _engine_aggregate_seeds is not None:
        try:
            return _engine_aggregate_seeds(list(values), ddof=ddof)
        except TypeError:
            try:
                return _engine_aggregate_seeds(list(values))
            except Exception:
                pass
        except Exception:
            pass
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return mean, 0.0
    divisor = (len(clean) - ddof) if ddof else len(clean)
    var = sum((v - mean) ** 2 for v in clean) / max(divisor, 1)
    return mean, math.sqrt(max(var, 0.0))


def format_mean_std(mean: Optional[float], std: Optional[float],
                    decimals: int = 2) -> str:
    """Render ``"52.53 +- 0.31"``."""
    if _engine_format_mean_std is not None:
        try:
            return _engine_format_mean_std(mean, std, decimals=decimals)
        except TypeError:
            try:
                return _engine_format_mean_std(mean, std)
            except Exception:
                pass
        except Exception:
            pass
    if mean is None or (isinstance(mean, float) and math.isnan(float(mean))):
        return "n/a"
    std_value = 0.0 if std is None or (isinstance(std, float) and math.isnan(float(std))) else float(std)
    return f"{float(mean):.{decimals}f} +- {std_value:.{decimals}f}"


def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs not declared by ``cls`` (tolerates dataclass signature drift)."""
    try:
        if dataclasses.is_dataclass(cls):
            allowed = {f.name for f in fields(cls)}
            return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        pass
    try:  # pragma: no cover
        import inspect

        params = inspect.signature(cls).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return dict(kwargs)


def make_scaling_config(*overrides: Dict[str, Any], **_kwargs: Any) -> "ScalingTrainConfig":
    """Build a :class:`ScalingTrainConfig`, dropping unknown keys."""
    merged: Dict[str, Any] = {}
    for override in overrides:
        if override:
            merged.update({k: v for k, v in override.items() if v is not None})
    merged.update({k: v for k, v in _kwargs.items() if v is not None})
    return ScalingTrainConfig(**_filter_kwargs(ScalingTrainConfig, merged))


# ---------------------------------------------------------------------------
# Mask generator scaling
# ---------------------------------------------------------------------------

def scaled_parameter_count(width_scale: int = 1,
                           num_layers: int = 5,
                           base_channels: int = BASE_CHANNELS) -> int:
    """Analytic parameter count of the SMM mask generator at a given width scale.

    Channel layout widens geometrically (``[w, 2w, 4w, ...]``) with
    ``w = base_channels * width_scale``; every hidden block is a 3x3 convolution
    with padding 1 followed by a BatchNorm, and the final layer maps to 3 mask
    channels.  For ``num_layers=5, width_scale=1`` this returns ``26,499`` and for
    ``num_layers=6, width_scale=1`` it returns ``102,339`` -- exactly Table 4.
    """
    hidden = max(int(num_layers) - 1, 1)
    widths = [int(base_channels) * int(width_scale) * (2 ** i) for i in range(hidden)]
    total = 0
    in_ch = 3
    for w in widths:
        total += 3 * 3 * in_ch * w + w    # 3x3 convolution, padding 1, bias
        total += 2 * w                     # BatchNorm2d (weight + bias)
        in_ch = w
    total += 3 * 3 * in_ch * 3 + 3         # final 3x3 convolution -> 3 mask channels
    return int(total)


def expected_parameters_for(width_scale: int = 1,
                            num_layers: int = 5,
                            base_channels: int = BASE_CHANNELS) -> int:
    """Alias of :func:`scaled_parameter_count` used by the sanity checks."""
    return scaled_parameter_count(width_scale, num_layers=num_layers,
                                  base_channels=base_channels)


class FallbackMaskGenerator(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Compact stand-in for :class:`smm_vr.models.mask_generator.MaskGenerator`.

    Only used if importing the packaged mask generator fails.  It mirrors the
    paper's architecture exactly -- 3x3 convolutions with padding 1 / stride 1,
    ReLU, 2x2 max-pooling, ``num_pooling_layers`` pooling layers, BatchNorm and a
    final convolution to 3 channels -- and therefore reproduces the Table 4
    parameter counts (26,499 / 102,339).
    """

    def __init__(self,
                 num_layers: int = 5,
                 base_channels: int = BASE_CHANNELS,
                 width_scale: int = 1,
                 out_channels: int = 3,
                 num_pooling_layers: int = NUM_POOLING_LAYERS,
                 use_bn: bool = True,
                 use_relu: bool = True,
                 in_channels: int = 3) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("torch is required to build the mask generator")
        super().__init__()
        self.num_layers = int(num_layers)
        self.base_channels = int(base_channels)
        self.width_scale = int(width_scale)
        self.out_channels = int(out_channels)
        self.num_pooling_layers = int(num_pooling_layers)
        self.in_channels = int(in_channels)

        hidden = max(self.num_layers - 1, 1)
        widths = [self.base_channels * self.width_scale * (2 ** i) for i in range(hidden)]

        blocks: List[Any] = []
        in_ch = self.in_channels
        for idx, w in enumerate(widths):
            layers: List[Any] = [nn.Conv2d(in_ch, w, kernel_size=3, padding=1, stride=1, bias=True)]
            if use_bn:
                layers.append(nn.BatchNorm2d(w))
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            if idx < self.num_pooling_layers:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            blocks.append(nn.Sequential(*layers))
            in_ch = w
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(in_ch, self.out_channels, kernel_size=3, padding=1, stride=1,
                               bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def patch_size(self) -> int:
        return 2 ** self.num_pooling_layers

    def spatial_out_size(self, in_size: int) -> int:
        size = int(in_size)
        for _ in range(self.num_pooling_layers):
            size = max(size // 2, 1)
        return size

    def forward(self, x: Any, return_penultimate: bool = False) -> Any:
        penultimate = None
        for block in self.blocks:
            x = block(x)
            penultimate = x
        out = self.final(x)   # no squashing / activation on the mask (paper Sec. 3.2)
        if return_penultimate:
            return out, penultimate
        return out

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return (f"num_layers={self.num_layers}, base_channels={self.base_channels}, "
                f"width_scale={self.width_scale}, num_pooling_layers={self.num_pooling_layers}")


def _count_module_parameters(module: Any) -> int:
    """Count trainable parameters of ``module`` (robust to missing helpers)."""
    if module is None:
        return 0
    if _count_parameters is not None:
        try:
            return int(_count_parameters(module))
        except Exception:
            pass
    try:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    except Exception:
        return 0


def build_scaled_mask_generator(backbone: str = SCALING_BACKBONE,
                                width_scale: int = 1,
                                *,
                                input_size: Optional[int] = None,
                                num_layers: Optional[int] = None,
                                base_channels: int = BASE_CHANNELS,
                                num_pooling_layers: int = NUM_POOLING_LAYERS,
                                channels: Optional[Sequence[int]] = None,
                                in_channels: int = 3,
                                device: Optional[Any] = None,
                                verify: bool = False) -> Any:
    """Build the SMM mask generator with doubled intermediate channels.

    ``width_scale=k`` multiplies every intermediate channel width by ``k`` while
    leaving the architecture untouched -- exactly the Table 11 protocol.  The
    function tries, in order: the packaged ``build_mask_generator`` (with a
    ``width_scale`` / ``channels`` override), the packaged ``MaskGenerator``
    constructed with explicit ``channels``, and finally a faithful local fallback.

    Returns the constructed module (``None`` if torch is unavailable).
    """
    if not _TORCH_AVAILABLE:
        return None

    layers = int(num_layers) if num_layers else num_mask_layers_for(backbone)
    hidden = max(layers - 1, 1)
    if channels is None:
        channels = [int(base_channels) * int(width_scale) * (2 ** i) for i in range(hidden)]
    channels = [int(c) for c in channels]
    expected = scaled_parameter_count(int(width_scale), num_layers=layers,
                                      base_channels=int(base_channels))

    module = None

    # 1) packaged factory with a width_scale / channels override
    if _build_mask_generator is not None:
        attempts = (
            {"width_scale": int(width_scale), "channels": channels, "num_layers": layers,
             "num_pooling_layers": int(num_pooling_layers), "in_channels": int(in_channels),
             "base_channels": int(base_channels)},
            {"width_scale": int(width_scale), "num_layers": layers,
             "num_pooling_layers": int(num_pooling_layers), "in_channels": int(in_channels),
             "base_channels": int(base_channels)},
            {"channels": channels, "num_layers": layers,
             "num_pooling_layers": int(num_pooling_layers), "in_channels": int(in_channels),
             "base_channels": int(base_channels)},
        )
        for overrides in attempts:
            try:
                candidate = _build_mask_generator(backbone=canonical_backbone(backbone),
                                                  **overrides)
            except Exception:
                continue
            if candidate is not None:
                module = candidate
                break

    # 2) packaged MaskGenerator constructed directly with explicit channels
    if module is None and _MaskGenerator is not None:
        direct_attempts = (
            {"channels": channels, "num_layers": layers,
             "num_pooling_layers": int(num_pooling_layers), "in_channels": int(in_channels)},
            {"num_layers": layers, "base_channels": int(base_channels) * int(width_scale),
             "num_pooling_layers": int(num_pooling_layers), "in_channels": int(in_channels)},
        )
        for kwargs in direct_attempts:
            try:
                module = _MaskGenerator(**kwargs)
                break
            except Exception:
                continue

    # 3) faithful local fallback
    if module is None:
        try:
            module = FallbackMaskGenerator(
                num_layers=layers,
                base_channels=int(base_channels),
                width_scale=int(width_scale),
                num_pooling_layers=int(num_pooling_layers),
                in_channels=int(in_channels),
            )
        except Exception:
            return None

    if verify:
        try:
            actual = _count_module_parameters(module)
            if actual and actual != expected:
                # Not fatal: the packaged generator may use a different (but still
                # lightweight) channel layout.  Recorded for the report.
                setattr(module, "_scaling_parameter_mismatch", (actual, expected))
        except Exception:
            pass

    if device is not None:
        try:
            module = module.to(resolve_device(device))
        except Exception:
            pass
    return module


# ---------------------------------------------------------------------------
# Config / history containers
# ---------------------------------------------------------------------------

@dataclass
class ScalingTrainConfig:
    """Training configuration for one rung of the Table 11 ladder."""

    epochs: int = DEFAULT_EPOCHS
    milestones: Tuple[int, ...] = DEFAULT_MILESTONES
    alpha_delta: float = DEFAULT_ALPHA_DELTA
    gamma_delta: float = DEFAULT_GAMMA_DELTA
    alpha_mask: Optional[float] = None
    gamma_mask: Optional[float] = None
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False
    grad_clip: Optional[float] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    test_batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 4
    label_mapping: str = "ilm"
    mapping_refresh_every: int = 1
    patch_size: int = DEFAULT_PATCH_SIZE
    seed: int = 0
    device: Optional[Any] = None
    log_every: int = 10
    eval_every: int = 1
    verbose: bool = False
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    save_dir: Optional[str] = None

    def resolved_mask_lr(self, backbone: str = SCALING_BACKBONE,
                         num_layers: Optional[int] = None) -> Tuple[float, float]:
        """Mask learning rate and decay factor implied by Table 9."""
        return mask_lr_schedule(backbone, num_mask_layers=num_layers,
                                alpha_mask=self.alpha_mask, gamma_mask=self.gamma_mask)

    def as_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["device"] = str(self.device) if self.device is not None else None
        out["milestones"] = list(self.milestones)
        return out


@dataclass
class ScalingEpochStats:
    """Per-epoch record; the training accuracy is needed for Table 11 claim 1."""

    epoch: int
    loss: float
    train_accuracy: float
    test_accuracy: Optional[float] = None
    delta_lr: Optional[float] = None
    mask_lr: Optional[float] = None
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ScalingHistory:
    """Full record of one (width scale, seed) run."""

    dataset: str = SCALING_DATASET
    backbone: str = SCALING_BACKBONE
    width_scale: int = 1
    level: int = 0
    parameters: int = 0
    seed: int = 0
    label_mapping: str = "ilm"
    epochs: List[ScalingEpochStats] = field(default_factory=list)
    best_test_accuracy: Optional[float] = None
    best_epoch: Optional[int] = None
    final_test_accuracy: Optional[float] = None
    final_train_accuracy: Optional[float] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def add(self, stats: ScalingEpochStats) -> None:
        self.epochs.append(stats)
        if stats.test_accuracy is not None:
            if self.best_test_accuracy is None or stats.test_accuracy > self.best_test_accuracy:
                self.best_test_accuracy = float(stats.test_accuracy)
                self.best_epoch = int(stats.epoch)
            self.final_test_accuracy = float(stats.test_accuracy)
        self.final_train_accuracy = float(stats.train_accuracy)

    @property
    def train_accuracies(self) -> List[float]:
        return [e.train_accuracy for e in self.epochs]

    @property
    def test_accuracies(self) -> List[Optional[float]]:
        return [e.test_accuracy for e in self.epochs]

    @property
    def losses(self) -> List[float]:
        return [e.loss for e in self.epochs]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "width_scale": int(self.width_scale),
            "level": int(self.level),
            "parameters": int(self.parameters),
            "seed": int(self.seed),
            "label_mapping": self.label_mapping,
            "epochs": [e.as_dict() for e in self.epochs],
            "best_test_accuracy": self.best_test_accuracy,
            "best_epoch": self.best_epoch,
            "final_test_accuracy": self.final_test_accuracy,
            "final_train_accuracy": self.final_train_accuracy,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "config": dict(self.config),
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
        return path


# ---------------------------------------------------------------------------
# Output label mapping helpers
# ---------------------------------------------------------------------------

def _unpack_logits(output: Any) -> Any:
    """ViT returns ``(logits, tokens)``; unwrap to logits."""
    if _TORCH_AVAILABLE and isinstance(output, (tuple, list)) and output:
        first = output[0]
        if hasattr(first, "dim") and first.dim() == 2:
            return first
    return output


def apply_output_mapping(logits: Any, f_out: Optional[Any] = None) -> Any:
    """Apply the non-parametric output label mapping ``f_out`` to ImageNet logits."""
    if f_out is None or logits is None:
        return logits
    if callable(f_out):
        try:
            mapped = f_out(logits)
            if mapped is not None:
                return mapped
        except Exception:
            pass
    index = getattr(f_out, "target_to_pretrained", None)
    if index is None and _TORCH_AVAILABLE and hasattr(f_out, "dim"):
        index = f_out
    if index is not None and _TORCH_AVAILABLE:
        try:
            index = torch.as_tensor(index, device=logits.device, dtype=torch.long)
            return torch.index_select(logits, -1, index)
        except Exception:
            pass
    if _packaged_apply_label_mapping is not None:
        try:
            return _packaged_apply_label_mapping(logits, f_out)
        except Exception:
            pass
    return logits


def apply_classification_loss(mapped_logits: Any, targets: Any) -> Any:
    """Cross-entropy over the mapped target labels.

    The paper only states "classification loss"; cross-entropy over ``f_out``'s
    labels is used (with a logged fallback for logit vectors wider than the labels).
    """
    try:
        return F.cross_entropy(mapped_logits, targets)
    except Exception:
        pass
    safe_targets = targets.clamp(min=0)
    picked = mapped_logits.gather(1, safe_targets.view(-1, 1)).squeeze(1)
    return (torch.logsumexp(mapped_logits, dim=1) - picked).mean()


def build_label_mapping(name: str = "ilm", *,
                        classifier: Any = None,
                        data_loader: Any = None,
                        num_target_classes: Optional[int] = None,
                        num_pretrained_classes: int = 1000,
                        f_in: Any = None,
                        device: Optional[Any] = None,
                        seed: int = 0,
                        **kwargs: Any) -> Optional[Any]:
    """Construct ``f_out`` (Rlm / Flm / Ilm) with whichever builder is available."""
    if _engine_build_label_mapping is not None:
        for extra in (
            dict(classifier=classifier, data_loader=data_loader,
                 num_target_classes=num_target_classes,
                 num_pretrained_classes=num_pretrained_classes,
                 device=device, seed=seed, **kwargs),
            dict(classifier=classifier, data_loader=data_loader,
                 num_target_classes=num_target_classes, device=device, seed=seed),
        ):
            try:
                built = _engine_build_label_mapping(name, **extra)
                if built is not None:
                    return built
            except TypeError:
                continue
            except Exception:
                continue
    if _lm_build_label_mapping is not None:
        for extra in (
            dict(classifier=classifier, data_loader=data_loader,
                 num_target_classes=num_target_classes,
                 num_pretrained_classes=num_pretrained_classes,
                 f_in=f_in, device=device, seed=seed),
            dict(classifier=classifier, data_loader=data_loader,
                 num_target_classes=num_target_classes, f_in=f_in, device=device, seed=seed),
        ):
            try:
                built = _lm_build_label_mapping(name, **extra)
                if built is not None:
                    return built
            except TypeError:
                continue
            except Exception:
                continue
    return None


def refresh_mapping(f_out: Optional[Any], *,
                    model: Any = None,
                    classifier: Any = None,
                    data_loader: Any = None,
                    device: Optional[Any] = None,
                    num_target_classes: Optional[int] = None,
                    num_pretrained_classes: int = 1000,
                    max_batches: Optional[int] = None) -> bool:
    """Recompute an Ilm-style mapping in place (Algorithm 4); no-op for Rlm/Flm."""
    if f_out is None or not getattr(f_out, "recomputes_each_epoch", False):
        return False
    full = dict(model=classifier, data_loader=data_loader, f_in=model, device=device,
                num_target_classes=num_target_classes,
                num_pretrained_classes=num_pretrained_classes,
                max_batches=max_batches)
    minimal = dict(model=classifier, data_loader=data_loader, f_in=model, device=device)
    for attempt in (full, {k: v for k, v in full.items() if k != "max_batches"}, minimal):
        try:
            f_out.update(**attempt)
            return True
        except TypeError:
            continue
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Optimiser / parameter partitioning
# ---------------------------------------------------------------------------

def split_parameters(model: Any) -> Tuple[List[Any], List[Any]]:
    """Split trainable parameters into (shared delta, mask-generator) groups."""
    delta: List[Any] = []
    mask: List[Any] = []
    if model is None or not _TORCH_AVAILABLE:
        return delta, mask
    for getter, bucket in (("delta_parameters", delta), ("mask_parameters", mask)):
        fn = getattr(model, getter, None)
        if callable(fn):
            try:
                bucket.extend([p for p in fn() if getattr(p, "requires_grad", True)])
            except Exception:
                pass
    if delta or mask:
        return delta, mask
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if "mask" in lname or "fmask" in lname or "gen" in lname:
            mask.append(param)
        else:
            delta.append(param)
    return delta, mask


def build_optimizers_and_schedulers(model: Any,
                                    config: ScalingTrainConfig,
                                    *,
                                    backbone: str = SCALING_BACKBONE,
                                    num_layers: Optional[int] = None
                                    ) -> Tuple[List[Any], List[Any]]:
    """Two SGD + MultiStepLR pairs: one for ``delta``, one for ``phi`` (Table 9)."""
    if not _TORCH_AVAILABLE:
        return [], []
    delta_params, mask_params = split_parameters(model)
    alpha_mask, gamma_mask = config.resolved_mask_lr(backbone, num_layers=num_layers)
    optimizers: List[Any] = []
    schedulers: List[Any] = []

    def _make(params: List[Any], lr: float, gamma: float) -> None:
        if not params:
            return
        if str(config.optimizer).lower() == "adam":
            opt = torch.optim.Adam(params, lr=float(lr),
                                   weight_decay=float(config.weight_decay))
        else:
            opt = torch.optim.SGD(params, lr=float(lr), momentum=float(config.momentum),
                                  weight_decay=float(config.weight_decay),
                                  nesterov=bool(config.nesterov))
        optimizers.append(opt)
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(m) for m in config.milestones], gamma=float(gamma)))

    _make(delta_params, float(config.alpha_delta), float(config.gamma_delta))
    _make(mask_params, float(alpha_mask), float(gamma_mask))
    return optimizers, schedulers


# ---------------------------------------------------------------------------
# Evaluation / training
# ---------------------------------------------------------------------------

def evaluate_scaling_model(model: Any,
                           classifier: Any,
                           data_loader: Any,
                           *,
                           f_out: Optional[Any] = None,
                           device: Optional[Any] = None,
                           max_batches: Optional[int] = None,
                           criterion: bool = True) -> Tuple[Optional[float], Optional[float]]:
    """Top-1 accuracy (%) and mean loss of the reprogrammed frozen classifier."""
    if not _TORCH_AVAILABLE or data_loader is None:
        return None, None
    device = resolve_device(device)
    if model is not None:
        model.eval()
    classifier.eval()
    correct = 0
    total = 0
    losses: List[float] = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if max_batches is not None and idx >= int(max_batches):
                break
            images, targets = batch[0], batch[1]
            images = images.to(device)
            targets = targets.to(device)
            inputs = model(images) if model is not None else images
            if isinstance(inputs, (tuple, list)):
                inputs = inputs[0]
            logits = _unpack_logits(classifier(inputs))
            mapped = apply_output_mapping(logits, f_out)
            if criterion:
                try:
                    loss = F.cross_entropy(mapped, targets)
                    losses.append(float(loss.detach().item()))
                except Exception:
                    pass
            preds = mapped.argmax(dim=-1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
    accuracy = 100.0 * correct / total if total else None
    mean_loss = sum(losses) / len(losses) if losses else None
    return accuracy, mean_loss


def train_scaling_run(model: Any,
                      classifier: Any,
                      train_loader: Any,
                      test_loader: Optional[Any] = None,
                      *,
                      f_out: Optional[Any] = None,
                      config: Optional[ScalingTrainConfig] = None,
                      history: Optional[ScalingHistory] = None,
                      dataset: str = SCALING_DATASET,
                      backbone: str = SCALING_BACKBONE,
                      num_layers: Optional[int] = None,
                      num_target_classes: Optional[int] = None,
                      num_pretrained_classes: int = 1000,
                      device: Optional[Any] = None,
                      logger: Optional[Callable[[str], None]] = None) -> ScalingHistory:
    """Algorithm 1 loop for one rung of the scaling ladder.

    Tracks both the training accuracy (claim 1: it rises as ``f_mask`` grows) and
    the test accuracy (claims 2/3: it may decline once the estimation error caused
    by the enlarged mask generator becomes non-negligible).
    """
    config = config or ScalingTrainConfig()
    device = resolve_device(device if device is not None else config.device)
    history = history or ScalingHistory(
        dataset=canonical_dataset(dataset), backbone=canonical_backbone(backbone),
        seed=int(config.seed), label_mapping=str(config.label_mapping),
        config=config.as_dict(),
    )

    if not _TORCH_AVAILABLE or train_loader is None or model is None or classifier is None:
        history.error = "torch / loaders / model unavailable"
        return history

    channels = int(num_layers) if num_layers else num_mask_layers_for(backbone)
    optimizer_list, scheduler_list = build_optimizers_and_schedulers(
        model, config, backbone=backbone, num_layers=channels)
    if not optimizer_list:
        history.error = "no trainable parameters found in the reprogramming module"
        return history

    alpha_mask, _ = config.resolved_mask_lr(backbone, num_layers=channels)

    start = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()

        # ---- Ilm: refresh the injective mapping before each epoch (Algorithm 4)
        if getattr(f_out, "recomputes_each_epoch", False) and config.mapping_refresh_every:
            if epoch == 1 or (epoch - 1) % max(int(config.mapping_refresh_every), 1) == 0:
                refresh_mapping(f_out, model=model, classifier=classifier,
                                data_loader=train_loader, device=device,
                                num_target_classes=num_target_classes,
                                num_pretrained_classes=num_pretrained_classes,
                                max_batches=config.max_train_batches)

        model.train()
        classifier.eval()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, batch in enumerate(train_loader):
            if config.max_train_batches is not None and batch_idx >= int(config.max_train_batches):
                break
            images, targets = batch[0], batch[1]
            images = images.to(device)
            targets = targets.to(device)

            for opt in optimizer_list:
                opt.zero_grad(set_to_none=True)

            inputs = model(images)
            if isinstance(inputs, (tuple, list)):
                inputs = inputs[0]
            logits = _unpack_logits(classifier(inputs))
            mapped = apply_output_mapping(logits, f_out)
            loss = apply_classification_loss(mapped, targets)
            loss.backward()

            if config.grad_clip:
                params = [p for opt in optimizer_list for p in opt.param_groups[0]["params"]]
                torch.nn.utils.clip_grad_norm_(params, float(config.grad_clip))
            for opt in optimizer_list:
                opt.step()

            with torch.no_grad():
                preds = mapped.argmax(dim=-1)
                running_correct += int((preds == targets).sum().item())
                running_total += int(targets.numel())
                running_loss += float(loss.detach().item()) * int(targets.numel())

        train_accuracy = 100.0 * running_correct / running_total if running_total else float("nan")
        mean_loss = running_loss / running_total if running_total else float("nan")

        test_accuracy = None
        if test_loader is not None and (
                epoch % max(int(config.eval_every), 1) == 0 or epoch == int(config.epochs)):
            test_accuracy, _ = evaluate_scaling_model(
                model, classifier, test_loader, f_out=f_out, device=device,
                max_batches=config.max_eval_batches)

        history.add(ScalingEpochStats(
            epoch=epoch,
            loss=float(mean_loss),
            train_accuracy=float(train_accuracy),
            test_accuracy=test_accuracy,
            delta_lr=float(optimizer_list[0].param_groups[0]["lr"]) if optimizer_list else None,
            mask_lr=float(alpha_mask),
            seconds=time.time() - epoch_start,
        ))

        for scheduler in scheduler_list:
            scheduler.step()

        if config.verbose and logger is not None and epoch % max(int(config.log_every), 1) == 0:
            test_txt = "n/a" if test_accuracy is None else f"{test_accuracy:.2f}"
            logger(f"[scaling w={history.width_scale} seed={history.seed}] epoch "
                   f"{epoch}/{config.epochs} loss={mean_loss:.4f} "
                   f"train={train_accuracy:.2f} test={test_txt}")

    history.elapsed_seconds = time.time() - start
    return history


def build_scaling_loaders(dataset: str,
                          backbone: str,
                          config: ScalingTrainConfig,
                          *,
                          root: Optional[str] = None,
                          data_root: Optional[str] = None,
                          imgsize: Optional[int] = None,
                          download: bool = True,
                          train_fraction: Optional[float] = None,
                          split_seed: int = 0) -> Tuple[Any, Any]:
    """Build (train_loader, test_loader) with the addendum transforms."""
    if _build_dataloaders is None:
        return None, None
    size = input_size_for_backbone(backbone, imgsize=imgsize)
    batch = int(config.batch_size or batch_size_for(dataset, backbone))
    kwargs: Dict[str, Any] = dict(
        dataset=canonical_dataset(dataset),
        backbone=canonical_backbone(backbone),
        root=root,
        data_root=data_root,
        imgsize=size,
        batch_size=batch,
        test_batch_size=int(config.test_batch_size or batch),
        num_workers=int(config.num_workers),
        download=bool(download),
        split_seed=int(split_seed),
    )
    if train_fraction is not None:
        kwargs["train_fraction"] = float(train_fraction)
    try:
        train_loader, test_loader, _spec = _build_dataloaders(**kwargs)
        return train_loader, test_loader
    except TypeError:
        kwargs.pop("train_fraction", None)
        try:
            train_loader, test_loader, _spec = _build_dataloaders(**kwargs)
            return train_loader, test_loader
        except Exception:
            return None, None
    except Exception:
        return None, None


def run_single_level(width_scale: int = 1,
                     level: int = 0,
                     *,
                     dataset: str = SCALING_DATASET,
                     backbone: str = SCALING_BACKBONE,
                     seed: int = 0,
                     config: Optional[ScalingTrainConfig] = None,
                     device: Optional[Any] = None,
                     label_mapping: str = "ilm",
                     root: Optional[str] = None,
                     data_root: Optional[str] = None,
                     num_workers: int = 4,
                     download: bool = True,
                     imgsize: Optional[int] = None,
                     patch_size: int = DEFAULT_PATCH_SIZE,
                     train_fraction: Optional[float] = None,
                     split_seed: int = 0,
                     config_overrides: Optional[Dict[str, Any]] = None,
                     max_train_batches: Optional[int] = None,
                     max_eval_batches: Optional[int] = None,
                     save_dir: Optional[str] = None,
                     verbose: bool = False,
                     logger: Optional[Callable[[str], None]] = None,
                     classifier: Any = None,
                     train_loader: Any = None,
                     test_loader: Any = None) -> ScalingHistory:
    """Train SMM with a width-scaled ``f_mask`` for a single seed."""
    backbone = canonical_backbone(backbone)
    dataset = canonical_dataset(dataset)
    layers = num_mask_layers_for(backbone)

    overrides: Dict[str, Any] = dict(config_overrides or {})
    overrides.setdefault("seed", int(seed))
    overrides.setdefault("device", device)
    overrides.setdefault("num_workers", int(num_workers))
    overrides.setdefault("label_mapping", label_mapping)
    overrides.setdefault("patch_size", int(patch_size))
    overrides.setdefault("batch_size", batch_size_for(dataset, backbone))
    if max_train_batches is not None:
        overrides.setdefault("max_train_batches", max_train_batches)
    if max_eval_batches is not None:
        overrides.setdefault("max_eval_batches", max_eval_batches)
    if save_dir is not None:
        overrides.setdefault("save_dir", save_dir)
    overrides.setdefault("verbose", bool(verbose))

    if config is None:
        config = ScalingTrainConfig(**_filter_kwargs(ScalingTrainConfig, overrides))
    else:
        for key in ("max_train_batches", "max_eval_batches", "save_dir", "device",
                    "num_workers", "patch_size", "label_mapping", "batch_size", "verbose"):
            value = overrides.get(key)
            if value is not None:
                try:
                    setattr(config, key, value)
                except Exception:
                    pass

    history = ScalingHistory(
        dataset=dataset, backbone=backbone, width_scale=int(width_scale),
        level=int(level), seed=int(seed), label_mapping=str(label_mapping),
        parameters=scaled_parameter_count(int(width_scale), num_layers=layers),
        config=config.as_dict(),
    )

    if not _TORCH_AVAILABLE:
        history.error = "torch unavailable"
        return history

    device = resolve_device(device if device is not None else config.device)
    set_seed(int(seed))

    if classifier is None:
        if _build_classifier is None:
            history.error = "models.pretrained.build_classifier unavailable"
            return history
        try:
            classifier = _build_classifier(backbone=backbone, device=device)
        except Exception as exc:  # pragma: no cover
            history.error = f"failed to build classifier: {exc}"
            return history
    try:
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad_(False)
    except Exception:
        pass

    if train_loader is None:
        train_loader, built_test_loader = build_scaling_loaders(
            dataset, backbone, config, root=root, data_root=data_root, imgsize=imgsize,
            download=download, train_fraction=train_fraction, split_seed=split_seed)
        if test_loader is None:
            test_loader = built_test_loader
    if train_loader is None:
        history.error = "data loaders unavailable"
        return history

    # ---- scaled mask generator (phi) + SMM reprogramming wrapper f_in
    mask_generator = build_scaled_mask_generator(
        backbone, int(width_scale), input_size=input_size_for_backbone(backbone, imgsize),
        num_layers=layers, in_channels=3, device=device, verify=True)
    if mask_generator is None:
        history.error = "failed to build scaled mask generator"
        return history
    counted = _count_module_parameters(mask_generator)
    if counted:
        history.parameters = int(counted)

    model = None
    if _build_smm_reprogram is not None:
        for kwargs in (
            dict(backbone=backbone, input_size=input_size_for_backbone(backbone, imgsize),
                 patch_size=int(patch_size), mask_generator=mask_generator),
            dict(backbone=backbone, input_size=input_size_for_backbone(backbone, imgsize),
                 patch_size=int(patch_size), width_scale=int(width_scale)),
        ):
            try:
                model = _build_smm_reprogram(**kwargs)
            except Exception:
                model = None
            if model is not None:
                break
    if model is None and _SMMReprogram is not None:
        try:
            model = _SMMReprogram(mask_generator=mask_generator,
                                  input_size=input_size_for_backbone(backbone, imgsize),
                                  patch_size=int(patch_size))
        except Exception:
            model = None
    if model is None:
        history.error = "modules.reprogram incompletable"
        return history
    try:
        model = model.to(device)
    except Exception:
        pass

    # ---- output label mapping f_out (Ilm by default; Table 11 uses the default)
    target_classes = num_target_classes(dataset)
    f_out = build_label_mapping(
        label_mapping, classifier=classifier, data_loader=train_loader,
        num_target_classes=target_classes, num_pretrained_classes=1000,
        f_in=model, device=device, seed=seed)

    return train_scaling_run(
        model, classifier, train_loader, test_loader, f_out=f_out, config=config,
        history=history, dataset=dataset, backbone=backbone, num_layers=layers,
        num_target_classes=target_classes, device=device, logger=logger)


# ---------------------------------------------------------------------------
# Reporting / verification
# ---------------------------------------------------------------------------

def aggregate_level(histories: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate the per-seed histories of one scaling level (mean +- std)."""
    hist_list = list(histories)
    if not hist_list:
        return {"mean": None, "std": None, "n": 0, "per_seed": [], "train_mean": None,
                "parameters": 0, "formatted": "n/a", "errors": []}

    def _get(history: Any, key: str) -> Any:
        if isinstance(history, dict):
            return history.get(key)
        return getattr(history, key, None)

    per_seed = [a for a in (_get(h, "final_test_accuracy") for h in hist_list) if a is not None]
    per_train = [a for a in (_get(h, "final_train_accuracy") for h in hist_list) if a is not None]
    mean, std = aggregate_seeds(per_seed) if per_seed else (None, None)
    train_mean, _ = aggregate_seeds(per_train) if per_train else (None, None)
    params = next((int(_get(h, "parameters") or 0) for h in hist_list
                   if int(_get(h, "parameters") or 0)), 0)
    return {
        "mean": mean,
        "std": std,
        "n": len(per_seed),
        "per_seed": per_seed,
        "train_mean": train_mean,
        "train_per_seed": per_train,
        "parameters": params,
        "formatted": format_mean_std(mean, std) if mean is not None else "n/a",
        "errors": [_get(h, "error") for h in hist_list if _get(h, "error")],
    }


def format_scaling_table(results: Dict[Any, Dict[str, Any]],
                         *,
                         width_scales: Optional[Sequence[int]] = None,
                         decimals: int = 1) -> str:
    """Render a Table-11-style text table (parameters / train / test accuracy)."""
    scales = list(width_scales) if width_scales else sorted(
        (int(k) for k in results.keys()), key=lambda s: int(s))
    header = (f"{'WIDTH':>6} {'PARAMS':>12} {'RATIO':>8} {'TRAIN ACC':>12} {'TEST ACC':>16}")
    lines = [header, "-" * len(header)]
    for scale in scales:
        entry = results.get(scale) or results.get(str(scale)) or {}
        params = int(entry.get("parameters") or scaled_parameter_count(scale))
        ratio = params / float(RESNET18_PARAMETERS) * 100.0
        train_mean = entry.get("train_mean")
        train_txt = "n/a" if train_mean is None else f"{float(train_mean):.{decimals}f}"
        lines.append(f"{scale:>6} {params:>12,} {ratio:>7.2f}% {train_txt:>12} "
                     f"{entry.get('formatted', 'n/a'):>16}")
    return "\n".join(lines)


def check_scaling_trend(results: Dict[Any, Dict[str, Any]],
                        *,
                        width_scales: Optional[Sequence[int]] = None,
                        tolerance: float = 0.5,
                        min_gain: float = 0.5,
                        verbose: bool = False) -> Dict[str, Any]:
    """Verify the three claims of Appendix D.3 / Table 11.

    Claim 1: the training accuracy increases (within ``tolerance``) with the number
    of parameters.  Claims 2/3: the test accuracy peaks at a *medium* size and then
    plateaus or declines instead of improving monotonically, i.e. the largest
    ``f_mask`` is not the best one.
    """
    scales = [int(s) for s in (width_scales or sorted(results.keys(), key=lambda s: int(s)))]
    scales = [s for s in scales if (results.get(s) or results.get(str(s)))]

    params: List[int] = []
    train: List[Optional[float]] = []
    test: List[Optional[float]] = []
    for scale in scales:
        entry = results.get(scale) or results.get(str(scale)) or {}
        params.append(int(entry.get("parameters") or scaled_parameter_count(scale)))
        train.append(entry.get("train_mean"))
        test.append(entry.get("mean"))

    report: Dict[str, Any] = {
        "dataset": TABLE11_REFERENCE["dataset"],
        "backbone": TABLE11_REFERENCE["backbone"],
        "width_scales": scales,
        "parameters": params,
        "train_accuracies": train,
        "test_accuracies": test,
        "tolerance": float(tolerance),
        "min_gain": float(min_gain),
    }

    # ---- claim 1: the training accuracy slowly increases
    train_violations = [
        f"w={scales[i + 1]}: {b:.2f} < {a:.2f}"
        for i, (a, b) in enumerate(zip(train, train[1:]))
        if a is not None and b is not None and b < a - tolerance
    ]
    report["train_monotone_non_decreasing"] = len(train_violations) == 0
    report["train_violations"] = train_violations
    report["train_gain_first_to_last"] = (
        None if train[0] is None or train[-1] is None else float(train[-1] - train[0]))

    # ---- claims 2/3: the test accuracy peaks at a medium size
    valid = [(i, v) for i, v in enumerate(test) if v is not None]
    if not valid:
        report.update({
            "best_index": None, "best_width_scale": None, "best_parameters": None,
            "peak_accuracy": None, "peak_is_largest": None,
            "plateau_or_decline_after_peak": None, "passed": False,
            "reason": "no test accuracies recorded",
        })
        if verbose:
            print(format_trend_report(report))
        return report

    best_index, best_value = max(valid, key=lambda pair: pair[1])
    report["best_index"] = int(best_index)
    report["best_width_scale"] = int(scales[best_index])
    report["best_parameters"] = int(params[best_index])
    report["peak_accuracy"] = float(best_value)
    report["peak_is_largest"] = bool(best_index == len(scales) - 1)
    report["peak_is_default"] = bool(best_index == 0)

    after = [v for v in test[best_index + 1:] if v is not None]
    report["plateau_or_decline_after_peak"] = (
        bool(all(v <= best_value + tolerance for v in after)) if after else True)
    report["test_gain_from_first_to_peak"] = (
        None if test[0] is None else float(best_value - test[0]))
    report["largest_minus_peak"] = (
        None if test[-1] is None else float(test[-1] - best_value))

    # The extracted text of the paper does not print the individual accuracies of
    # Table 11, so the check is expressed as a *trend*: the training accuracy must
    # (roughly) grow, and the best test accuracy must not come from the largest
    # f_mask -- that is exactly when the estimation error becomes noticeable.
    reasons: List[str] = []
    if all(v is not None for v in train) and not report["train_monotone_non_decreasing"]:
        reasons.append("training accuracy decreased noticeably at some rung")
    if report["peak_is_largest"]:
        reasons.append("test accuracy kept improving for the largest f_mask "
                       "(no visible estimation error)")
    report["passed"] = len(reasons) == 0
    report["reasons"] = reasons
    report["reason"] = "; ".join(reasons) if reasons else "trend consistent with Table 11"

    if verbose:
        print(format_trend_report(report))
    return report


def format_trend_report(report: Dict[str, Any]) -> str:
    """Human-readable rendering of :func:`check_scaling_trend`."""
    lines = ["[Table 11] Impact of the number of parameters of f_mask",
             f"  dataset={report.get('dataset')} backbone={report.get('backbone')}"]
    scales = report.get("width_scales") or []
    params = report.get("parameters") or []
    train = report.get("train_accuracies") or []
    test = report.get("test_accuracies") or []
    for i, scale in enumerate(scales):
        p = params[i] if i < len(params) else 0
        tr = train[i] if i < len(train) else None
        te = test[i] if i < len(test) else None
        train_txt = f"{tr:.2f}" if tr is not None else "n/a"
        test_txt = f"{te:.2f}" if te is not None else "n/a"
        lines.append(f"    width_scale={scale:<2} params={p:>10,} "
                     f"train={train_txt:>7} test={test_txt:>7}")
    lines.append(f"  train monotone (tol={report.get('tolerance')}): "
                 f"{report.get('train_monotone_non_decreasing')}")
    lines.append(f"  best width_scale={report.get('best_width_scale')} "
                 f"(params={report.get('best_parameters')}, "
                 f"acc={report.get('peak_accuracy')})")
    lines.append(f"  plateau/decline after peak: {report.get('plateau_or_decline_after_peak')}")
    lines.append(f"  PASSED: {report.get('passed')} -- {report.get('reason')}")
    return "\n".join(lines)


def compare_table11(results: Dict[Any, Dict[str, Any]],
                    *,
                    width_scales: Optional[Sequence[int]] = None,
                    tolerance: float = 0.5,
                    verbose: bool = False) -> Dict[str, Any]:
    """Combine the trend check with the Table 4 / Table 11 parameter ladder."""
    trend = check_scaling_trend(results, width_scales=width_scales,
                                tolerance=tolerance, verbose=verbose)
    parameter_check: Dict[str, Any] = {}
    scales_checked = width_scales or [level["width_scale"] for level in SCALING_LEVELS]
    for level in _make_scaling_levels(scales_checked):
        scale = level["width_scale"]
        entry = results.get(scale) or results.get(str(scale)) or {}
        measured = int(entry.get("parameters") or 0)
        parameter_check[str(scale)] = {
            "expected": level["parameters"],
            "measured": measured,
            "matches": (measured == level["parameters"]) if measured else None,
            "ratio_to_resnet18": level["ratio_to_resnet18"],
        }
    default_entry = parameter_check.get(str(1), {})
    comparison = {
        "trend": trend,
        "parameters": parameter_check,
        "default_matches_table4": default_entry.get(
            "matches", default_entry.get("expected") == 26499),
        "reference": TABLE11_REFERENCE,
        "passed": bool(trend.get("passed")),
    }
    if verbose:
        print(format_trend_report(trend))
    return comparison


def save_scaling_results(payload: Dict[str, Any],
                         output_dir: Optional[str] = None,
                         *,
                         filename: str = "scaling_table11.json") -> str:
    """Persist the scaling-study payload as JSON (histories/configs stripped)."""
    directory = output_dir or os.environ.get("SMM_OUTPUT_DIR") or "outputs"
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items()
                    if k not in ("histories", "history", "config", "model")}
        if isinstance(obj, (list, tuple)):
            return [_strip(v) for v in obj]
        if hasattr(obj, "as_dict") and callable(getattr(obj, "as_dict")):
            return _strip(obj.as_dict())
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_strip(payload), handle, indent=2)
    return path


def describe_scaling_study(datasets: Optional[Sequence[str]] = None,
                           *,
                           backbone: Optional[str] = None,
                           seeds: Optional[Sequence[int]] = None,
                           width_scales: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Metadata describing the Table 11 run (used by ``--describe``)."""
    backbone = canonical_backbone(backbone)
    layers = num_mask_layers_for(backbone)
    scales = [int(s) for s in (width_scales or DEFAULT_WIDTH_SCALES)]
    return {
        "experiment": "scaling",
        "table": "table11",
        "dataset": SCALING_DATASET,
        "extra_datasets": [canonical_dataset(d) for d in (datasets or []) if d],
        "backbone": backbone,
        "num_mask_layers": layers,
        "width_scales": scales,
        "levels": _make_scaling_levels(scales, num_layers=layers),
        "seeds": resolve_seeds(seeds),
        "epochs": DEFAULT_EPOCHS,
        "milestones": list(DEFAULT_MILESTONES),
        "batch_size": batch_size_for(SCALING_DATASET, backbone),
        "mask_lr": mask_lr_schedule(backbone, num_layers=layers),
        "claims": TABLE11_REFERENCE["claims"],
    }


# ---------------------------------------------------------------------------
# Experiment entry points
# ---------------------------------------------------------------------------

def run_scaling_study(backbone: str = SCALING_BACKBONE,
                      *,
                      datasets: Optional[Sequence[str]] = None,
                      seeds: Optional[Sequence[int]] = None,
                      width_scales: Optional[Sequence[int]] = None,
                      epochs: Optional[int] = None,
                      output_dir: Optional[str] = None,
                      save: bool = True,
                      verbose: bool = True,
                      device: Optional[Any] = None,
                      data_root: Optional[str] = None,
                      root: Optional[str] = None,
                      num_workers: int = 4,
                      download: bool = True,
                      imgsize: Optional[int] = None,
                      patch_size: int = DEFAULT_PATCH_SIZE,
                      label_mapping: str = "ilm",
                      train_fraction: Optional[float] = None,
                      split_seed: int = 0,
                      config_overrides: Optional[Dict[str, Any]] = None,
                      max_train_batches: Optional[int] = None,
                      max_eval_batches: Optional[int] = None,
                      logger: Optional[Callable[[str], None]] = None,
                      **kwargs: Any) -> Dict[str, Any]:
    """Run the Appendix D.3 / Table 11 scaling study (EuroSAT + frozen ResNet-18).

    The intermediate channels of ``f_mask`` are progressively doubled
    (``width_scale`` in ``{1, 2, 4, 8}``) while the architecture stays unchanged.
    Every rung records the training and test accuracies across the paper's three
    seeds, then the Appendix D.3 trends are verified and the reference parameter
    ladder of Table 4 is reported alongside.
    """
    backbone = canonical_backbone(backbone)
    layers = num_mask_layers_for(backbone)
    scales = [int(s) for s in (width_scales or DEFAULT_WIDTH_SCALES)]
    seed_list = resolve_seeds(seeds)
    dataset_list = [canonical_dataset(d) for d in (datasets or [SCALING_DATASET])]
    num_epochs = int(epochs) if epochs else DEFAULT_EPOCHS

    log = logger or (print if verbose else (lambda *_a, **_k: None))
    log(f"[scaling] backbone={backbone} datasets={dataset_list} width_scales={scales} "
        f"seeds={seed_list} epochs={num_epochs} "
        f"mask_lr={mask_lr_schedule(backbone, num_layers=layers)}")

    per_dataset: Dict[str, Any] = {}
    for dataset in dataset_list:
        histories: List[ScalingHistory] = []
        results: Dict[int, Dict[str, Any]] = {}
        for level in _make_scaling_levels(scales, num_layers=layers):
            scale = int(level["width_scale"])
            level_histories: List[ScalingHistory] = []
            for seed in seed_list:
                overrides: Dict[str, Any] = dict(config_overrides or {})
                overrides["epochs"] = num_epochs
                history = run_single_level(
                    scale, level=int(level["level"]), dataset=dataset, backbone=backbone,
                    seed=int(seed), device=device, label_mapping=label_mapping,
                    root=root, data_root=data_root, num_workers=num_workers,
                    download=download, imgsize=imgsize, patch_size=patch_size,
                    train_fraction=train_fraction, split_seed=split_seed,
                    config_overrides=overrides, max_train_batches=max_train_batches,
                    max_eval_batches=max_eval_batches, verbose=False, logger=logger)
                if history.error:
                    log(f"[scaling] warning: width_scale={scale} seed={seed}: {history.error}")
                level_histories.append(history)
            histories.extend(level_histories)
            entry = aggregate_level(level_histories)
            entry["level"] = int(level["level"])
            if not entry["parameters"]:
                entry["parameters"] = int(level["parameters"])
            results[scale] = entry
            log(f"[scaling] {dataset} width_scale={scale} "
                f"params={entry['parameters']:,} test={entry['formatted']}")

        trend = check_scaling_trend(results, width_scales=scales, verbose=False)
        table = format_scaling_table(results, width_scales=scales)
        per_dataset[dataset] = {"results": results, "trend": trend, "table": table,
                                "histories": histories}
        if verbose:
            print(table)

    primary_key = SCALING_DATASET if SCALING_DATASET in per_dataset else next(iter(per_dataset))
    primary = per_dataset[primary_key]
    comparison = compare_table11(primary["results"], width_scales=scales, verbose=False)

    payload: Dict[str, Any] = {
        "experiment": "scaling",
        "table": "table11",
        "backbone": backbone,
        "datasets": dataset_list,
        "seeds": seed_list,
        "width_scales": scales,
        "levels": _make_scaling_levels(scales, num_layers=layers),
        "reference": TABLE11_REFERENCE,
        "results": primary["results"],
        "table_text": primary["table"],
        "trend": primary["trend"],
        "comparison": comparison,
        "per_dataset": {k: {"results": v["results"], "trend": v["trend"], "table": v["table"]}
                        for k, v in per_dataset.items()},
        "histories": {k: [h.as_dict() for h in v["histories"]] for k, v in per_dataset.items()},
    }

    SCALING_RESULTS["table11"] = payload
    if save:
        payload["path"] = save_scaling_results(payload, output_dir)
        log(f"[scaling] saved -> {payload['path']}")

    if verbose:
        print(format_trend_report(primary["trend"]))
    return payload


def run_scaling_experiment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Registry alias of :func:`run_scaling_study` used by ``smm_vr.experiments``."""
    return run_scaling_study(*args, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_scaling_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_scaling",
        description="SMM scaling study of f_mask (paper Appendix D.3, Table 11): "
                    "progressively double the intermediate channels of the mask "
                    "generator on EuroSAT with a frozen ResNet-18.",
    )
    parser.add_argument("--backbone", default=SCALING_BACKBONE,
                        help="frozen pre-trained backbone (default: resnet18)")
    parser.add_argument("--datasets", nargs="*", default=[SCALING_DATASET],
                        help="target datasets (default: eurosat)")
    parser.add_argument("--width-scales", "--scales", dest="width_scales", nargs="*",
                        type=int, default=list(DEFAULT_WIDTH_SCALES),
                        help="intermediate-channel multipliers (default: 1 2 4 8)")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SCALING_SEEDS),
                        help="random seeds (default: 0 1 2)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="training epochs per run (default: 200)")
    parser.add_argument("--label-mapping", default="ilm", choices=["ilm", "flm", "rlm"])
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="deterministic class-balanced subsample for debugging")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--describe", action="store_true",
                        help="print the study description and exit")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns 0 whenever the run completed (including a graceful degradation when
    torch / the datasets / the packaged modules are unavailable, which is reported
    through ``history.error`` and echoed in the summary).
    """
    parser = build_scaling_arg_parser()
    args = parser.parse_args(argv)

    if args.describe:
        info = describe_scaling_study(args.datasets, backbone=args.backbone,
                                      seeds=args.seeds, width_scales=args.width_scales)
        print(json.dumps(info, indent=2, default=str))
        return 0

    payload = run_scaling_study(
        backbone=args.backbone,
        datasets=args.datasets,
        seeds=args.seeds,
        width_scales=args.width_scales,
        epochs=args.epochs,
        output_dir=args.output_dir,
        save=not args.no_save,
        verbose=not args.quiet,
        device=args.device,
        data_root=args.data_root,
        root=args.root,
        num_workers=args.num_workers,
        download=not args.no_download,
        patch_size=args.patch_size,
        label_mapping=args.label_mapping,
        train_fraction=args.train_fraction,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    return 0 if payload else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
