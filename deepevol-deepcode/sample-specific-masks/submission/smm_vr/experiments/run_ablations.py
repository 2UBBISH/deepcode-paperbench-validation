"""Ablation studies for Sample-specific Multi-channel Masks (SMM, ICML 2024).

Implements the two ablation experiments reported in Section 5 of the paper:

* **"Impact of Masking"** (Table 3, ResNet-18).  Four masking strategies:

  .. code-block:: text

      (i)   only delta       f_in(x_i) = r(x_i) + delta                  (M all-one)
      (ii)  only f_mask      f_in(x_i) = r(x_i) + f_mask(r(x_i))
      (iii) single-channel   f_in(x_i) = r(x_i) + delta (*) f_mask^s(r(x_i))
      (iv)  ours (full SMM)  f_in(x_i) = r(x_i) + delta (*) f_mask(r(x_i))

  where ``f_mask^s`` averages the penultimate-layer output of the mask
  generator (paper: "Single-channel version of SMM ... averaging the
  penultimate-layer output of the mask generator").

* **"Impact of Patch Size"** (Figure 4).  Sweeps the number of Max-Pooling
  layers ``l in {0, 1, 2, 3, 4}``, i.e. patch sizes ``2**l in {1, 2, 4, 8, 16}``,
  with ResNet-18 as the pre-trained model.  The paper reports that accuracy
  "increases first, followed by a plateau or decline", and that patch size 8 is
  used across all datasets ("Since the 5-layer mask generator neural network has
  at most 4 Max-Pooling layers ...").

Training protocol (shared with the main tables, see Section 5 "Baselines" and
Appendix C Table 9): 200 epochs, milestones at epochs 100/145, initial learning
rate 0.01 with decay 0.1 for the shared pattern ``delta``;

* 5-layer mask generator (ResNets): ``alpha_mask = 0.01``, ``gamma_mask = 0.1``;
* 6-layer mask generator (ViT-B32): ``alpha_mask = 0.001``, ``gamma_mask = 1``.

Reference numbers of Table 3 are embedded so that measured results can be
compared against the paper without re-typing them.

Paper ambiguities resolved here (see README):

* the exact optimiser is unspecified -> SGD (momentum 0.9, no weight decay);
* the loss is only called "classification loss" -> cross-entropy over the
  mapped target labels;
* variant (ii) has no ``delta`` at all, so the raw (patch-interpolated) output of
  the mask generator is added to the resized image, exactly as written in the
  paper.  Its magnitude is therefore controlled only by the learning rate;
  ``output_scale`` is exposed for debugging but defaults to 1.0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Guarded internal imports (the module must stay importable during partial
# builds; every dependency falls back to a local implementation).
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - import guard
    from ..data.datasets import (
        DEFAULT_BATCH_SIZES,
        MAIN_DATASETS,
        build_dataloaders,
        build_datasets,
    )
    from ..data.datasets import num_classes as _dataset_num_classes

    _DATA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _DATA_AVAILABLE = False

    MAIN_DATASETS = (
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
    )
    DEFAULT_BATCH_SIZES = {name: 256 for name in MAIN_DATASETS}
    DEFAULT_BATCH_SIZES["dtd"] = 64
    DEFAULT_BATCH_SIZES["oxfordpets"] = 64

    build_dataloaders = None  # type: ignore[assignment]
    build_datasets = None  # type: ignore[assignment]

    def _dataset_num_classes(name: str) -> int:  # type: ignore[misc]
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
        }.get(str(name).lower(), 10)


try:  # pragma: no cover - import guard
    from ..engine.metrics import (
        aggregate_seeds,
        format_mean_std,
        mean_over_datasets,
    )

    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _METRICS_AVAILABLE = False

    def aggregate_seeds(values: Sequence[float], ddof: int = 1) -> Tuple[float, float]:  # type: ignore[misc]
        vals = [float(v) for v in values]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) <= ddof:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return mean, math.sqrt(var)

    def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:  # type: ignore[misc]
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

    def mean_over_datasets(  # type: ignore[misc]
        per_dataset: Dict[str, Any], dataset_order: Optional[Sequence[str]] = None
    ) -> float:
        order = list(dataset_order) if dataset_order else [k for k in per_dataset if k != "average"]
        vals: List[float] = []
        for key in order:
            entry = per_dataset.get(key)
            if entry is None:
                continue
            if isinstance(entry, dict):
                entry = entry.get("mean", entry.get("accuracy"))
            if entry is None:
                continue
            vals.append(float(entry))
        return sum(vals) / len(vals) if vals else 0.0


try:  # pragma: no cover - import guard
    from ..engine.seeds import SEEDS, resolve_seeds, set_seed

    _SEEDS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _SEEDS_AVAILABLE = False
    SEEDS = (0, 1, 2)

    def resolve_seeds(  # type: ignore[misc]
        seeds: Optional[Sequence[int]] = None, n_seeds: Optional[int] = None
    ) -> List[int]:
        resolved = [int(s) for s in seeds] if seeds is not None else [int(s) for s in SEEDS]
        if n_seeds is not None and n_seeds > 0:
            resolved = resolved[:n_seeds]
        return resolved

    def set_seed(seed: int, **kwargs: Any) -> int:  # type: ignore[misc]
        import random

        random.seed(seed)
        try:  # pragma: no cover - optional dependency
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - gpu only
            torch.cuda.manual_seed_all(seed)
        return seed


try:  # pragma: no cover - import guard
    from ..models.pretrained import build_classifier, input_size_for

    _PRETRAINED_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _PRETRAINED_AVAILABLE = False

    build_classifier = None  # type: ignore[assignment]

    def input_size_for(backbone: str = "resnet18", imgsize: Optional[int] = None) -> int:  # type: ignore[misc]
        if imgsize is not None:
            return int(imgsize)
        return 384 if "vit" in str(backbone).lower() else 224


try:  # pragma: no cover - import guard
    from ..models.mask_generator import (
        DEFAULT_PATCH_SIZE,
        EXPECTED_PARAMETERS,
        build_mask_generator,
        count_parameters,
    )

    _MASK_GENERATOR_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _MASK_GENERATOR_AVAILABLE = False
    DEFAULT_PATCH_SIZE = 8
    EXPECTED_PARAMETERS = {"resnet18": 26499, "resnet50": 26499, "vit_b32": 102339}
    build_mask_generator = None  # type: ignore[assignment]

    def count_parameters(module: nn.Module, only_trainable: bool = True) -> int:  # type: ignore[misc]
        return int(
            sum(p.numel() for p in module.parameters() if (p.requires_grad or not only_trainable))
        )


try:  # pragma: no cover - import guard
    from ..modules.patch_interp import patch_wise_interpolate

    _PATCH_INTERP_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _PATCH_INTERP_AVAILABLE = False

    def patch_wise_interpolate(  # type: ignore[misc]
        mask: torch.Tensor, patch_size: int = 8, out_size: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """Fallback block replication: each pixel -> patch_size x patch_size patch."""
        patch_size = int(patch_size)
        if patch_size <= 1:
            upsampled = mask
        else:
            upsampled = mask.repeat_interleave(patch_size, dim=-2).repeat_interleave(
                patch_size, dim=-1
            )
        if out_size is not None:
            height, width = int(out_size[0]), int(out_size[1])
            upsampled = upsampled[..., :height, :width]
            pad_h = max(0, height - upsampled.shape[-2])
            pad_w = max(0, width - upsampled.shape[-1])
            if pad_h or pad_w:
                upsampled = F.pad(upsampled, (0, pad_w, 0, pad_h), mode="replicate")
        return upsampled


try:  # pragma: no cover - import guard
    from ..modules.reprogram import SMMReprogram

    _REPROGRAM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _REPROGRAM_AVAILABLE = False
    SMMReprogram = None  # type: ignore[assignment]


try:  # pragma: no cover - import guard
    from ..engine.train_smm import (
        DEFAULT_ALPHA_DELTA,
        DEFAULT_ALPHA_MASK_5,
        DEFAULT_ALPHA_MASK_6,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_GAMMA_DELTA,
        DEFAULT_GAMMA_MASK_5,
        DEFAULT_GAMMA_MASK_6,
        DEFAULT_MILESTONES,
        MASK_LAYERS_BY_BACKBONE,
        SMALL_BATCH_DATASETS,
    )

    _TRAIN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _TRAIN_AVAILABLE = False
    DEFAULT_EPOCHS = 200
    DEFAULT_MILESTONES = (100, 145)
    DEFAULT_ALPHA_DELTA = 0.01
    DEFAULT_GAMMA_DELTA = 0.1
    DEFAULT_ALPHA_MASK_5 = 0.01
    DEFAULT_GAMMA_MASK_5 = 0.1
    DEFAULT_ALPHA_MASK_6 = 0.001
    DEFAULT_GAMMA_MASK_6 = 1.0
    DEFAULT_BATCH_SIZE = 256
    SMALL_BATCH_DATASETS = ("dtd", "oxfordpets")
    MASK_LAYERS_BY_BACKBONE = {
        "resnet18": 5,
        "resnet50": 5,
        "resnet101": 5,
        "vit_b32": 6,
        "vit_large": 6,
    }


try:  # pragma: no cover - import guard
    from ..label_mapping import build_label_mapping as _package_build_label_mapping

    _LABEL_MAPPING_PACKAGE_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _LABEL_MAPPING_PACKAGE_AVAILABLE = False
    _package_build_label_mapping = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from ..engine.train_smm import build_label_mapping as _engine_build_label_mapping

    _LABEL_MAPPING_ENGINE_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    _LABEL_MAPPING_ENGINE_AVAILABLE = False
    _engine_build_label_mapping = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The four masking strategies of Table 3 (order matches the paper's columns).
MASKING_VARIANTS: Tuple[str, ...] = (
    "only_delta",
    "only_fmask",
    "single_channel_fmask",
    "ours",
)

#: Human readable labels, in the paper's column order.
VARIANT_DISPLAY: Dict[str, str] = {
    "only_delta": "ONLY delta",
    "only_fmask": "ONLY f_mask",
    "single_channel_fmask": "SINGLE-CHANNEL f_mask^s",
    "ours": "OURS",
}

#: Alias table for CLI / programmatic use.
VARIANT_ALIASES: Dict[str, str] = {
    "only_delta": "only_delta",
    "onlydelta": "only_delta",
    "delta": "only_delta",
    "delta_only": "only_delta",
    "shared": "only_delta",
    "shared_pattern": "only_delta",
    "only_fmask": "only_fmask",
    "fmask": "only_fmask",
    "onlyfmask": "only_fmask",
    "mask_only": "only_fmask",
    "fmask_only": "only_fmask",
    "single_channel_fmask": "single_channel_fmask",
    "single_channel": "single_channel_fmask",
    "singlechannel": "single_channel_fmask",
    "fmask_s": "single_channel_fmask",
    "ours": "ours",
    "smm": "ours",
    "full_smm": "ours",
    "proposed": "ours",
}

#: Patch sizes examined in Figure 4 (``2**l`` for ``l in {0,1,2,3,4}``).
PATCH_STUDY_L: Tuple[int, ...] = (0, 1, 2, 3, 4)
PATCH_SIZES: Tuple[int, ...] = tuple(2 ** l for l in PATCH_STUDY_L)  # (1, 2, 4, 8, 16)
DEFAULT_PATCH_SIZE_INDEX: int = 3  # l = 3 -> patch size 8 (paper default)

#: Datasets used by default for the patch-size sweep (Figure 4 plots one curve
#: per dataset; the CLI's ``--patch-datasets`` covers all eleven tasks).
PATCH_STUDY_DATASETS: Tuple[str, ...] = ("cifar10", "svhn", "flowers102", "eurosat")

#: The 5-layer mask generator has at most 4 Max-Pooling layers (Section 5).
MAX_POOLING_LAYERS_5 = 4

#: Table 3 (ResNet-18) reference per-dataset means.
#: Column order: only delta | only f_mask | single-channel f_mask^s | ours.
TABLE3_MEANS: Dict[str, Dict[str, float]] = {
    "cifar10": {
        "only_delta": 68.9,
        "only_fmask": 59.0,
        "single_channel_fmask": 72.6,
        "ours": 72.8,
    },
    "cifar100": {
        "only_delta": 33.8,
        "only_fmask": 32.1,
        "single_channel_fmask": 38.0,
        "ours": 39.4,
    },
    "svhn": {
        "only_delta": 78.3,
        "only_fmask": 51.1,
        "single_channel_fmask": 78.4,
        "ours": 84.4,
    },
    "gtsrb": {
        "only_delta": 76.8,
        "only_fmask": 55.7,
        "single_channel_fmask": 70.7,
        "ours": 80.4,
    },
    "flowers102": {
        "only_delta": 23.2,
        "only_fmask": 32.2,
        "single_channel_fmask": 30.2,
        "ours": 38.7,
    },
    "dtd": {
        "only_delta": 29.0,
        "only_fmask": 27.2,
        "single_channel_fmask": 32.7,
        "ours": 33.6,
    },
    "ucf101": {
        "only_delta": 24.4,
        "only_fmask": 25.7,
        "single_channel_fmask": 28.0,
        "ours": 28.7,
    },
    "food101": {
        "only_delta": 13.2,
        "only_fmask": 13.3,
        "single_channel_fmask": 15.8,
        "ours": 17.5,
    },
    "sun397": {
        "only_delta": 13.4,
        "only_fmask": 10.5,
        "single_channel_fmask": 15.9,
        "ours": 16.0,
    },
    "eurosat": {
        "only_delta": 84.3,
        "only_fmask": 89.2,
        "single_channel_fmask": 90.6,
        "ours": 92.2,
    },
    "oxfordpets": {
        "only_delta": 70.0,
        "only_fmask": 72.5,
        "single_channel_fmask": 73.8,
        "ours": 74.1,
    },
}

TABLE3_STDS: Dict[str, Dict[str, float]] = {
    "cifar10": {
        "only_delta": 0.4,
        "only_fmask": 1.6,
        "single_channel_fmask": 2.6,
        "ours": 0.7,
    },
    "cifar100": {
        "only_delta": 0.2,
        "only_fmask": 0.3,
        "single_channel_fmask": 0.6,
        "ours": 0.6,
    },
    "svhn": {
        "only_delta": 0.3,
        "only_fmask": 3.1,
        "single_channel_fmask": 0.2,
        "ours": 2.0,
    },
    "gtsrb": {
        "only_delta": 0.9,
        "only_fmask": 1.2,
        "single_channel_fmask": 0.8,
        "ours": 1.2,
    },
    "flowers102": {
        "only_delta": 0.5,
        "only_fmask": 0.4,
        "single_channel_fmask": 0.4,
        "ours": 0.7,
    },
    "dtd": {
        "only_delta": 0.7,
        "only_fmask": 0.5,
        "single_channel_fmask": 0.5,
        "ours": 0.4,
    },
    "ucf101": {
        "only_delta": 0.9,
        "only_fmask": 0.3,
        "single_channel_fmask": 0.3,
        "ours": 0.8,
    },
    "food101": {
        "only_delta": 0.1,
        "only_fmask": 0.1,
        "single_channel_fmask": 0.1,
        "ours": 0.1,
    },
    "sun397": {
        "only_delta": 0.2,
        "only_fmask": 0.1,
        "single_channel_fmask": 0.1,
        "ours": 0.3,
    },
    "eurosat": {
        "only_delta": 0.5,
        "only_fmask": 0.9,
        "single_channel_fmask": 0.5,
        "ours": 0.2,
    },
    "oxfordpets": {
        "only_delta": 0.6,
        "only_fmask": 0.3,
        "single_channel_fmask": 0.6,
        "ours": 0.4,
    },
}

#: Averages highlighted in grey in Table 3.
TABLE3_AVERAGES: Dict[str, float] = {
    "only_delta": 46.85,
    "only_fmask": 42.59,
    "single_channel_fmask": 49.70,
    "ours": 52.53,
}

#: Ordering claim of Section 5 / Table 3.
TABLE3_ORDERING: Tuple[str, ...] = (
    "ours",
    "single_channel_fmask",
    "only_delta",
    "only_fmask",
)

#: Default training protocol (Section 5 "Baselines", Appendix C Table 9).
TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": DEFAULT_EPOCHS,
    "milestones": tuple(DEFAULT_MILESTONES),
    "alpha_delta": DEFAULT_ALPHA_DELTA,
    "gamma_delta": DEFAULT_GAMMA_DELTA,
    "alpha_mask": None,  # -> 0.01 (5 layers) / 0.001 (6 layers)
    "gamma_mask": None,  # -> 0.1 (5 layers) / 1.0 (6 layers)
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0,
    "batch_size": DEFAULT_BATCH_SIZE,
    "label_mapping": "ilm",
    "patch_size": DEFAULT_PATCH_SIZE,
}

#: Accumulator consumed by :mod:`smm_vr.experiments` and the analysis helpers.
ABLATION_RESULTS: Dict[str, Any] = {"masking": {}, "patch_size": {}}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def canonical_backbone(name: Optional[str]) -> str:
    """Normalise a backbone spelling to the project's canonical key."""
    key = str(name or "resnet18").strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("resnet18", "resnet_18", "rn18"):
        return "resnet18"
    if key in ("resnet50", "resnet_50", "rn50"):
        return "resnet50"
    if key in ("vit_b32", "vit_b_32", "vitb32", "vit"):
        return "vit_b32"
    if key.startswith("vit"):
        return "vit_b32"
    return key


def num_mask_layers_for(backbone: str) -> int:
    """5 CNN layers for ResNet-18/50, 6 for ViT-B32 (Section 3.2)."""
    key = canonical_backbone(backbone)
    try:
        return int(MASK_LAYERS_BY_BACKBONE[key])
    except Exception:
        return 6 if key.startswith("vit") else 5


def canonical_variant(name: str) -> str:
    """Normalise a masking-variant name (raises ``ValueError`` when unknown)."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in VARIANT_ALIASES:
        return VARIANT_ALIASES[key]
    squashed = key.replace("_", "")
    for alias, canonical in VARIANT_ALIASES.items():
        if alias.replace("_", "") == squashed:
            return canonical
    raise ValueError(f"unknown masking variant {name!r}; expected one of {MASKING_VARIANTS}")


def mask_lr_schedule(
    backbone: str,
    num_mask_layers: Optional[int] = None,
    alpha_mask: Optional[float] = None,
    gamma_mask: Optional[float] = None,
) -> Tuple[float, float]:
    """Return ``(alpha_mask, gamma_mask)`` for the given mask-generator depth.

    Table 9: ``alpha = 0.01``, ``gamma = 0.1`` for the 5-layer generator and
    ``alpha = 0.001``, ``gamma = 1`` for the 6-layer generator.
    """
    layers = int(num_mask_layers) if num_mask_layers is not None else num_mask_layers_for(backbone)
    default_alpha = DEFAULT_ALPHA_MASK_5 if layers <= 5 else DEFAULT_ALPHA_MASK_6
    default_gamma = DEFAULT_GAMMA_MASK_5 if layers <= 5 else DEFAULT_GAMMA_MASK_6
    alpha = float(default_alpha if alpha_mask is None else alpha_mask)
    gamma = float(default_gamma if gamma_mask is None else gamma_mask)
    return alpha, gamma


def resolve_device(device: Optional[Any] = None) -> torch.device:
    """Resolve a device specification to a :class:`torch.device`."""
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = str(device)
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def batch_size_for(dataset: str, backbone: Optional[str] = None) -> int:
    """Table 9 batch size: 256 everywhere, 64 for DTD and OxfordPets."""
    key = str(dataset).strip().lower()
    override = DEFAULT_BATCH_SIZES.get(key) if isinstance(DEFAULT_BATCH_SIZES, dict) else None
    if override:
        return int(override)
    return 64 if key in SMALL_BATCH_DATASETS else int(DEFAULT_BATCH_SIZE)


def _match_spatial(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Resize ``tensor`` to ``reference``'s spatial resolution when needed."""
    if tuple(tensor.shape[-2:]) == tuple(reference.shape[-2:]):
        return tensor
    return F.interpolate(
        tensor, size=tuple(reference.shape[-2:]), mode="bilinear", align_corners=False
    )


def _as_logits(output: Any) -> torch.Tensor:
    """Unwrap tuple outputs (e.g. ViT ``(logits, tokens)``)."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


# --------------------------------------------------------------------------- #
# Ablation reprogramming modules
# --------------------------------------------------------------------------- #


class OnlyDeltaReprogram(nn.Module):
    """Ablation (i): ``f_in(x_i) = r(x_i) + delta`` with an all-one mask ``M``.

    The paper: "Shared-pattern VR ``f_in(x_i) = r(x_i) + delta``, with ``M``
    being an all-one matrix equal to the image dimension for maximal
    flexibility in ``delta``.  It defaults to the 'full watermarks' baseline
    without using ``f_mask``."

    Args:
        input_size: Side length of the reprogrammed input (224 or 384).
        in_channels: Number of image channels (3).
        delta_init: ``"zero"`` (Algorithm 1) or ``"normal"``.
        delta_std: Standard deviation when ``delta_init == "normal"``.
    """

    variant = "only_delta"
    uses_mask_generator = False

    def __init__(
        self,
        input_size: int = 224,
        in_channels: int = 3,
        delta_init: str = "zero",
        delta_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        shape = (self.in_channels, self.input_size, self.input_size)
        if str(delta_init).lower() == "normal":
            init = torch.randn(shape) * float(delta_std)
        else:
            init = torch.zeros(shape)
        self.delta = nn.Parameter(init)
        self.mask_generator = None
        self.register_buffer(
            "mask",
            torch.ones((1, self.in_channels, self.input_size, self.input_size)),
            persistent=False,
        )

    def delta_parameters(self) -> List[nn.Parameter]:
        return [self.delta]

    def mask_parameters(self) -> List[nn.Parameter]:
        return []

    def forward(self, images: torch.Tensor, return_mask: bool = False):
        images = _match_spatial(images, self.delta)
        output = images + self.delta.unsqueeze(0)
        if return_mask:
            return output, self.mask.expand_as(output)
        return output

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"variant={self.variant}, input_size={self.input_size}"


class OnlyFmaskReprogram(nn.Module):
    """Ablation (ii): ``f_in(x_i) = r(x_i) + f_mask(r(x_i))``.

    "Sample-specific pattern without masking": there is no shared pattern
    ``delta``; the (patch-interpolated) mask-generator output is added to the
    resized image directly.
    """

    variant = "only_fmask"
    uses_mask_generator = True

    def __init__(
        self,
        backbone: str = "resnet18",
        input_size: int = 224,
        patch_size: int = DEFAULT_PATCH_SIZE,
        num_mask_layers: Optional[int] = None,
        num_pooling_layers: Optional[int] = None,
        mask_generator: Optional[nn.Module] = None,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = canonical_backbone(backbone)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.output_scale = float(output_scale)
        self.num_mask_layers = int(
            num_mask_layers if num_mask_layers is not None else num_mask_layers_for(self.backbone)
        )
        self.mask_generator = (
            mask_generator
            if mask_generator is not None
            else _build_mask_generator(
                self.backbone,
                num_layers=self.num_mask_layers,
                num_pooling_layers=num_pooling_layers,
            )
        )

    def delta_parameters(self) -> List[nn.Parameter]:
        return []

    def mask_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.mask_generator.parameters() if p.requires_grad]

    def generate_mask(self, images: torch.Tensor) -> torch.Tensor:
        low_res = self.mask_generator(images)
        if isinstance(low_res, tuple):
            low_res = low_res[0]
        mask = patch_wise_interpolate(
            low_res, patch_size=self.patch_size, out_size=tuple(images.shape[-2:])
        )
        return mask * self.output_scale

    def forward(self, images: torch.Tensor, return_mask: bool = False):
        mask = self.generate_mask(images)
        output = images + mask
        if return_mask:
            return output, mask
        return output

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"variant={self.variant}, backbone={self.backbone}, "
            f"patch_size={self.patch_size}, output_scale={self.output_scale}"
        )


class SingleChannelFmaskReprogram(nn.Module):
    """Ablation (iii): ``f_in(x_i) = r(x_i) + delta (*) f_mask^s(r(x_i))``.

    ``f_mask^s`` is the penultimate-layer output of the mask generator averaged
    over its channel dimension ("Single-channel version of SMM ... averaging the
    penultimate-layer output of the mask generator"); the resulting single
    channel is broadcast to the three image channels so that ``delta (*) mask``
    keeps the multi-channel pattern's per-channel flexibility.
    """

    variant = "single_channel_fmask"
    uses_mask_generator = True

    def __init__(
        self,
        backbone: str = "resnet18",
        input_size: int = 224,
        patch_size: int = DEFAULT_PATCH_SIZE,
        num_mask_layers: Optional[int] = None,
        num_pooling_layers: Optional[int] = None,
        mask_generator: Optional[nn.Module] = None,
        in_channels: int = 3,
        delta_init: str = "zero",
    ) -> None:
        super().__init__()
        self.backbone = canonical_backbone(backbone)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.num_mask_layers = int(
            num_mask_layers if num_mask_layers is not None else num_mask_layers_for(self.backbone)
        )
        self.mask_generator = (
            mask_generator
            if mask_generator is not None
            else _build_mask_generator(
                self.backbone,
                num_layers=self.num_mask_layers,
                num_pooling_layers=num_pooling_layers,
            )
        )
        shape = (self.in_channels, self.input_size, self.input_size)
        init = (
            torch.randn(shape) * 0.01
            if str(delta_init).lower() == "normal"
            else torch.zeros(shape)
        )
        self.delta = nn.Parameter(init)

    def delta_parameters(self) -> List[nn.Parameter]:
        return [self.delta]

    def mask_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.mask_generator.parameters() if p.requires_grad]

    def generate_mask(self, images: torch.Tensor) -> torch.Tensor:
        result = self.mask_generator(images, return_penultimate=True)
        if isinstance(result, tuple):
            penultimate = result[1]
        else:  # pragma: no cover - defensive
            penultimate = result
        single = penultimate.mean(dim=1, keepdim=True)  # (B, 1, h, w)
        single = patch_wise_interpolate(
            single, patch_size=self.patch_size, out_size=tuple(images.shape[-2:])
        )
        return single.expand(-1, self.in_channels, -1, -1).contiguous()

    def forward(self, images: torch.Tensor, return_mask: bool = False):
        images = _match_spatial(images, self.delta)
        mask = self.generate_mask(images)
        output = images + self.delta.unsqueeze(0) * mask
        if return_mask:
            return output, mask
        return output

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"variant={self.variant}, backbone={self.backbone}, patch_size={self.patch_size}"


def _build_mask_generator(
    backbone: str,
    *,
    num_layers: Optional[int] = None,
    num_pooling_layers: Optional[int] = None,
    **kwargs: Any,
) -> nn.Module:
    """Build ``f_mask``; prefers the project implementation, else a local CNN."""
    key = canonical_backbone(backbone)
    layers = int(num_layers if num_layers is not None else num_mask_layers_for(key))
    if build_mask_generator is not None:
        overrides: Dict[str, Any] = {"num_layers": layers}
        if num_pooling_layers is not None:
            overrides["num_pooling_layers"] = int(num_pooling_layers)
        overrides.update(kwargs)
        try:
            return build_mask_generator(key, **overrides)
        except TypeError:
            return build_mask_generator(key)
    return _FallbackMaskGenerator(
        num_layers=layers,
        num_pooling_layers=int(
            num_pooling_layers
            if num_pooling_layers is not None
            else max(0, int(DEFAULT_PATCH_SIZE).bit_length() - 1)
        ),
    )


class _FallbackMaskGenerator(nn.Module):
    """Minimal stand-in for ``models.mask_generator.MaskGenerator``."""

    def __init__(
        self,
        num_layers: int = 5,
        base_channels: int = 8,
        out_channels: int = 3,
        num_pooling_layers: int = 3,
        in_channels: int = 3,
        **_ignored: Any,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_pooling = int(num_pooling_layers)
        widths = [base_channels * (2 ** i) for i in range(max(1, self.num_layers - 1))]
        blocks: List[nn.Module] = []
        current = int(in_channels)
        for index, width in enumerate(widths):
            blocks.append(nn.Conv2d(current, width, 3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
            if index < self.num_pooling:
                blocks.append(nn.MaxPool2d(2, 2))
            current = width
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(current, int(out_channels), 3, padding=1)

    def forward(self, x: torch.Tensor, return_penultimate: bool = False):
        features = self.features(x)
        mask = self.classifier(features)
        if return_penultimate:
            return mask, features
        return mask

    def spatial_out_size(self, in_size: int) -> int:
        return int(in_size) // (2 ** self.num_pooling)

    @property
    def patch_size(self) -> int:
        return 2 ** self.num_pooling


def build_ablation_reprogram(
    variant: str,
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    num_mask_layers: Optional[int] = None,
    num_pooling_layers: Optional[int] = None,
    in_channels: int = 3,
    delta_init: str = "zero",
    device: Optional[Any] = None,
) -> nn.Module:
    """Build the reprogramming module of one masking variant.

    Args:
        variant: One of :data:`MASKING_VARIANTS` (aliases accepted).
        backbone: Pre-trained backbone whose input resolution ``f_in`` targets.
        input_size: Explicit input side; defaults to 224 (ResNets) / 384 (ViT).
        patch_size: Patch size ``2**l`` (default 8).
        num_mask_layers: Mask-generator depth (5 for ResNets, 6 for ViT-B32).
        num_pooling_layers: Number of 2x2 max-pooling layers (``l``).
        in_channels: Image channels (3).
        delta_init: ``"zero"`` per Algorithm 1.
        device: Optional device to place the module on.

    Returns:
        An ``nn.Module`` implementing ``f_in`` with ``delta_parameters()`` and
        ``mask_parameters()`` helpers.  For ``variant == "ours"`` the project's
        :class:`~smm_vr.modules.reprogram.SMMReprogram` is returned.
    """
    key = canonical_backbone(backbone)
    canonical = canonical_variant(variant)
    size = int(input_size if input_size is not None else input_size_for(key))
    layers = int(num_mask_layers if num_mask_layers is not None else num_mask_layers_for(key))
    pooling = (
        int(num_pooling_layers)
        if num_pooling_layers is not None
        else int(round(math.log2(max(1, int(patch_size)))))
    )

    if canonical == "ours":
        generator = _build_mask_generator(key, num_layers=layers, num_pooling_layers=pooling)
        if SMMReprogram is None:  # pragma: no cover - guarded builds
            raise ImportError("SMMReprogram is unavailable; cannot build the full SMM variant")
        try:
            model: nn.Module = SMMReprogram(
                mask_generator=generator,
                input_size=size,
                patch_size=int(patch_size),
                backbone=key,
                in_channels=in_channels,
                delta_init=delta_init,
            )
        except TypeError:  # pragma: no cover - signature drift
            model = SMMReprogram(input_size=size, patch_size=int(patch_size), backbone=key)
        model.variant = "ours"
        return model.to(resolve_device(device))

    if canonical == "only_delta":
        model = OnlyDeltaReprogram(input_size=size, in_channels=in_channels, delta_init=delta_init)
        return model.to(resolve_device(device))

    if canonical == "only_fmask":
        model = OnlyFmaskReprogram(
            backbone=key,
            input_size=size,
            patch_size=int(patch_size),
            num_mask_layers=layers,
            num_pooling_layers=pooling,
        )
        return model.to(resolve_device(device))

    model = SingleChannelFmaskReprogram(
        backbone=key,
        input_size=size,
        patch_size=int(patch_size),
        num_mask_layers=layers,
        num_pooling_layers=pooling,
        in_channels=in_channels,
        delta_init=delta_init,
    )
    return model.to(resolve_device(device))


def build_patch_size_reprogram(
    l: int,
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    num_mask_layers: Optional[int] = None,
    in_channels: int = 3,
    device: Optional[Any] = None,
) -> nn.Module:
    """Full-SMM ``f_in`` with ``l`` max-pooling layers (patch size ``2**l``).

    Section 5 "Impact of Patch Size": "Since the 5-layer mask generator neural
    network has at most 4 Max-Pooling layers, we examine the impact of patch
    sizes in {2^0, 2^1, 2^2, 2^3, 2^4}."
    """
    l = int(l)
    if not 0 <= l <= MAX_POOLING_LAYERS_5:
        raise ValueError(f"l must be in [0, {MAX_POOLING_LAYERS_5}], got {l}")
    return build_ablation_reprogram(
        "ours",
        backbone=backbone,
        input_size=input_size,
        patch_size=2 ** l,
        num_mask_layers=num_mask_layers,
        num_pooling_layers=l,
        in_channels=in_channels,
        device=device,
    )


def build_variant_models(
    backbone: str = "resnet18",
    variants: Optional[Sequence[str]] = None,
    input_size: Optional[int] = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    in_channels: int = 3,
    device: Optional[Any] = None,
) -> Dict[str, nn.Module]:
    """Build one fresh module per masking variant."""
    names = [canonical_variant(v) for v in (variants or MASKING_VARIANTS)]
    return {
        name: build_ablation_reprogram(
            name,
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            device=device,
        )
        for name in names
    }


# --------------------------------------------------------------------------- #
# Training / evaluation utilities
# --------------------------------------------------------------------------- #


@dataclass
class AblationTrainConfig:
    """Hyperparameters of the ablation training loop (Algorithm-1 protocol)."""

    epochs: int = DEFAULT_EPOCHS
    milestones: Tuple[int, ...] = tuple(DEFAULT_MILESTONES)
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
    label_mapping: str = "ilm"
    mapping_refresh_every: int = 1
    log_every: int = 10
    eval_every: int = 1
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    num_workers: int = 4
    download: bool = True
    seed: int = 0
    device: Optional[str] = None
    verbose: bool = True
    patch_size: int = DEFAULT_PATCH_SIZE
    num_mask_layers: Optional[int] = None
    save_dir: Optional[str] = None

    def resolved_mask_lr(self, backbone: str = "resnet18") -> Tuple[float, float]:
        return mask_lr_schedule(backbone, self.num_mask_layers, self.alpha_mask, self.gamma_mask)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["milestones"] = list(self.milestones)
        return data


@dataclass
class AblationEpochStats:
    """Per-epoch record (loss / accuracies / learning rates)."""

    epoch: int
    loss: float
    train_accuracy: float
    test_accuracy: Optional[float]
    delta_lr: float
    mask_lr: float
    mapping_updated: bool = False
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationHistory:
    """Full record of one run (one variant, one dataset, one seed)."""

    dataset: str = ""
    backbone: str = "resnet18"
    variant: str = "ours"
    seed: int = 0
    label_mapping: str = "ilm"
    epochs: List[AblationEpochStats] = field(default_factory=list)
    best_test_accuracy: Optional[float] = None
    best_epoch: Optional[int] = None
    final_test_accuracy: Optional[float] = None
    train_accuracy: Optional[float] = None
    mask_parameters: int = 0
    delta_parameters: int = 0
    elapsed_seconds: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def add(self, stats: AblationEpochStats) -> None:
        self.epochs.append(stats)
        if stats.test_accuracy is not None:
            self.final_test_accuracy = float(stats.test_accuracy)
            if self.best_test_accuracy is None or stats.test_accuracy > self.best_test_accuracy:
                self.best_test_accuracy = float(stats.test_accuracy)
                self.best_epoch = int(stats.epoch)

    @property
    def losses(self) -> List[float]:
        return [e.loss for e in self.epochs]

    @property
    def test_accuracies(self) -> List[Optional[float]]:
        return [e.test_accuracy for e in self.epochs]

    @property
    def accuracy(self) -> float:
        """Reported accuracy: final-epoch test accuracy (paper protocol)."""
        if self.final_test_accuracy is not None:
            return float(self.final_test_accuracy)
        return float(self.best_test_accuracy or 0.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "variant": self.variant,
            "seed": self.seed,
            "label_mapping": self.label_mapping,
            "epochs": [e.as_dict() for e in self.epochs],
            "best_test_accuracy": self.best_test_accuracy,
            "best_epoch": self.best_epoch,
            "final_test_accuracy": self.final_test_accuracy,
            "train_accuracy": self.train_accuracy,
            "mask_parameters": self.mask_parameters,
            "delta_parameters": self.delta_parameters,
            "elapsed_seconds": self.elapsed_seconds,
            "config": self.config,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2, default=str)
        return path


def apply_output_mapping(logits: torch.Tensor, f_out: Optional[Any]) -> torch.Tensor:
    """Apply ``f_out`` (index tensor, callable or mapping module) to logits.

    Idempotent for logits already living in the target label space.
    """
    if f_out is None:
        return logits
    if isinstance(f_out, torch.Tensor):
        index = f_out.to(logits.device).long().view(-1)
        return logits.index_select(-1, index)
    if isinstance(f_out, (list, tuple)):
        index = torch.as_tensor(list(f_out), device=logits.device).long().view(-1)
        return logits.index_select(-1, index)
    if isinstance(f_out, nn.Module) or callable(f_out):
        try:
            return f_out(logits)
        except TypeError:  # pragma: no cover - signature drift
            pass
    index = getattr(f_out, "target_to_pretrained", None)
    if index is not None:
        index = torch.as_tensor(index, device=logits.device).long().view(-1)
        return logits.index_select(-1, index)
    return logits


def advance_classifier(
    model: nn.Module, classifier: nn.Module, images: torch.Tensor
) -> torch.Tensor:
    """``f_P(f_in(x))``: reprogram the batch, then run the frozen classifier."""
    reprogrammed = model(images)
    if isinstance(reprogrammed, (tuple, list)):
        reprogrammed = reprogrammed[0]
    return classifier(reprogrammed)


@torch.no_grad()
def evaluate_variant(
    model: nn.Module,
    classifier: nn.Module,
    data_loader: Any,
    *,
    f_out: Optional[Any] = None,
    device: Optional[Any] = None,
    max_batches: Optional[int] = None,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, Optional[float]]:
    """Top-1 accuracy (%) and (optional) mean loss of a variant on a split."""
    device = resolve_device(device)
    was_training = model.training
    model.eval()
    classifier.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss_sum = 0.0
    loss_count = 0
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        images, targets = batch[0], batch[1]
        images = images.to(device)
        targets = targets.to(device).long().view(-1)
        logits = _as_logits(advance_classifier(model, classifier, images))
        mapped = apply_output_mapping(logits, f_out)
        predictions = mapped.argmax(dim=-1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
        if int(targets.max().item()) < int(mapped.shape[-1]):
            loss_sum += float(criterion(mapped.float(), targets).item()) * int(targets.numel())
            loss_count += int(targets.numel())
    if was_training:
        model.train()
    accuracy = 100.0 * correct / total if total else 0.0
    mean_loss = loss_sum / loss_count if loss_count else None
    return accuracy, mean_loss


def _parameter_group(model: nn.Module, group: str) -> List[nn.Parameter]:
    """Collect trainable parameters of the ``delta`` or ``mask`` group."""
    accessor_name = "delta_parameters" if group == "delta" else "mask_parameters"
    accessor = getattr(model, accessor_name, None)
    if callable(accessor):
        try:
            return [p for p in accessor() if p.requires_grad]
        except Exception:  # pragma: no cover - defensive
            pass
    selected: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (name.split(".")[0] == "delta") == (group == "delta"):
            selected.append(param)
    return selected


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: Any,
    *,
    backbone: str = "resnet18",
    group: str = "delta",
) -> Tuple[Optional[torch.optim.Optimizer], Optional[Any]]:
    """SGD/Adam optimiser + MultiStepLR for one parameter group (Algorithm 1)."""
    if group == "delta":
        params = _parameter_group(model, "delta")
        lr = float(getattr(config, "alpha_delta", DEFAULT_ALPHA_DELTA))
        gamma = float(getattr(config, "gamma_delta", DEFAULT_GAMMA_DELTA))
    else:
        params = _parameter_group(model, "mask")
        lr, gamma = mask_lr_schedule(
            backbone,
            getattr(config, "num_mask_layers", None),
            getattr(config, "alpha_mask", None),
            getattr(config, "gamma_mask", None),
        )
    if not params:
        return None, None

    name = str(getattr(config, "optimizer", "sgd")).lower()
    weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
    if name == "adam":
        optimizer: torch.optim.Optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(getattr(config, "momentum", 0.9) or 0.0),
            weight_decay=weight_decay,
            nesterov=bool(getattr(config, "nesterov", False)),
        )
    milestones = list(getattr(config, "milestones", DEFAULT_MILESTONES) or DEFAULT_MILESTONES)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return optimizer, scheduler


def _make_label_mapping(
    name: str,
    *,
    classifier: nn.Module,
    data_loader: Any,
    num_target_classes: int,
    f_in: Optional[nn.Module] = None,
    device: Optional[Any] = None,
    seed: int = 0,
) -> Optional[Any]:
    """Build ``f_out`` through whichever label-mapping factory is available."""
    num_pretrained = 1000
    out_features = getattr(classifier, "out_features", None) or getattr(
        classifier, "num_classes", None
    )
    if out_features:
        num_pretrained = int(out_features)

    if _engine_build_label_mapping is not None:
        try:
            return _engine_build_label_mapping(
                name,
                classifier=classifier,
                data_loader=data_loader,
                num_target_classes=num_target_classes,
                device=device,
                seed=seed,
            )
        except Exception:
            pass

    if _package_build_label_mapping is not None:
        candidates = (
            {
                "name": name,
                "classifier": classifier,
                "model": classifier,
                "data_loader": data_loader,
                "num_target_classes": num_target_classes,
                "num_pretrained_classes": num_pretrained,
                "f_in": f_in,
                "device": device,
                "seed": seed,
            },
            {"name": name, "num_target_classes": num_target_classes, "seed": seed},
        )
        for kwargs in candidates:
            try:
                return _package_build_label_mapping(**kwargs)
            except Exception:
                continue
    return None


def refresh_mapping(
    f_out: Optional[Any],
    *,
    model: Optional[nn.Module] = None,
    classifier: Optional[nn.Module] = None,
    data_loader: Any = None,
    device: Optional[Any] = None,
    max_batches: Optional[int] = None,
) -> bool:
    """Refresh an Ilm-style mapping (Algorithm 4).

    Returns ``True`` when the mapping was recomputed.  Mappings flagged with
    ``recomputes_each_epoch == False`` (Rlm/Flm) are left untouched.
    """
    if f_out is None or not getattr(f_out, "recomputes_each_epoch", False):
        return False
    update = getattr(f_out, "update", None)
    if not callable(update):
        return False
    attempts = (
        {
            "model": classifier,
            "data_loader": data_loader,
            "f_in": model,
            "device": device,
            "max_batches": max_batches,
        },
        {"model": classifier, "data_loader": data_loader, "f_in": model, "device": device},
        {"model": classifier, "data_loader": data_loader, "f_in": model},
        {"classifier": classifier, "data_loader": data_loader, "f_in": model},
    )
    for kwargs in attempts:
        try:
            update(**kwargs)
            return True
        except TypeError:
            continue
        except Exception:
            return False
    return False


def train_variant(
    model: nn.Module,
    classifier: nn.Module,
    train_loader: Any,
    *,
    test_loader: Any = None,
    f_out: Optional[Any] = None,
    label_mapping_builder: Optional[Callable[..., Any]] = None,
    config: Optional[Any] = None,
    dataset: str = "",
    backbone: str = "resnet18",
    variant: Optional[str] = None,
    num_target_classes: Optional[int] = None,
    device: Optional[Any] = None,
    history: Optional[AblationHistory] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> AblationHistory:
    """Train one masking variant with the Algorithm-1 protocol.

    For every epoch ``j = 1..E`` the output mapping is optionally recomputed
    (Ilm, Algorithm 4), sample-specific masks are produced for the batch, the
    reprogrammed batch is pushed through the frozen ``f_P``, and the
    classification loss w.r.t. the mapped target labels is minimised with
    separate learning rates for ``delta`` and ``phi``.  The schedule follows
    Section 5 ("200 epochs ... the 100th and the 145th epochs are the
    milestones") and Appendix C Table 9.
    """
    cfg = config if config is not None else AblationTrainConfig()
    device_obj = resolve_device(
        device if device is not None else (getattr(cfg, "device", None) or "auto")
    )
    model = model.to(device_obj)
    classifier = classifier.to(device_obj)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)

    variant_name = variant or getattr(model, "variant", "ours")
    mapping_name = str(getattr(cfg, "label_mapping", "ilm") or "ilm")
    classes = int(
        num_target_classes
        if num_target_classes is not None
        else (_dataset_num_classes(dataset) if dataset else 10)
    )

    if f_out is None:
        if label_mapping_builder is not None:
            try:
                f_out = label_mapping_builder(
                    mapping_name,
                    classifier=classifier,
                    data_loader=train_loader,
                    num_target_classes=classes,
                    f_in=model,
                    device=device_obj,
                    seed=int(getattr(cfg, "seed", 0)),
                )
            except Exception:
                f_out = None
        if f_out is None:
            f_out = _make_label_mapping(
                mapping_name,
                classifier=classifier,
                data_loader=train_loader,
                num_target_classes=classes,
                f_in=model,
                device=device_obj,
                seed=int(getattr(cfg, "seed", 0)),
            )

    opt_delta, sched_delta = build_optimizer_and_scheduler(
        model, cfg, backbone=backbone, group="delta"
    )
    opt_mask, sched_mask = build_optimizer_and_scheduler(
        model, cfg, backbone=backbone, group="mask"
    )
    criterion = nn.CrossEntropyLoss()

    record = history if history is not None else AblationHistory()
    record.dataset = dataset
    record.backbone = canonical_backbone(backbone)
    record.variant = variant_name
    record.seed = int(getattr(cfg, "seed", 0))
    record.label_mapping = mapping_name
    record.config = cfg.as_dict() if hasattr(cfg, "as_dict") else {}
    record.delta_parameters = sum(p.numel() for p in _parameter_group(model, "delta"))
    record.mask_parameters = sum(p.numel() for p in _parameter_group(model, "mask"))

    epochs = int(getattr(cfg, "epochs", DEFAULT_EPOCHS))
    log_every = int(getattr(cfg, "log_every", 10) or 10)
    eval_every = max(1, int(getattr(cfg, "eval_every", 1) or 1))
    refresh_every = max(1, int(getattr(cfg, "mapping_refresh_every", 1) or 1))
    grad_clip = getattr(cfg, "grad_clip", None)
    verbose = bool(getattr(cfg, "verbose", True))
    max_train_batches = getattr(cfg, "max_train_batches", None)
    max_eval_batches = getattr(cfg, "max_eval_batches", None)

    start = time.time()
    for epoch in range(1, epochs + 1):
        mapping_updated = False
        if getattr(f_out, "recomputes_each_epoch", False) and (epoch - 1) % refresh_every == 0:
            mapping_updated = refresh_mapping(
                f_out,
                model=model,
                classifier=classifier,
                data_loader=train_loader,
                device=device_obj,
                max_batches=max_eval_batches,
            )

        model.train()
        classifier.eval()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.time()

        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= int(max_train_batches):
                break
            images, targets = batch[0], batch[1]
            images = images.to(device_obj, non_blocking=True)
            targets = targets.to(device_obj).long().view(-1)

            if opt_delta is not None:
                opt_delta.zero_grad(set_to_none=True)
            if opt_mask is not None:
                opt_mask.zero_grad(set_to_none=True)

            logits = _as_logits(advance_classifier(model, classifier, images))
            mapped = apply_output_mapping(logits, f_out)
            loss = criterion(mapped.float(), targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], float(grad_clip)
                )
            if opt_delta is not None:
                opt_delta.step()
            if opt_mask is not None:
                opt_mask.step()

            batch_size = int(targets.numel())
            running_loss += float(loss.item()) * batch_size
            running_correct += int((mapped.argmax(dim=-1) == targets).sum().item())
            running_total += batch_size

            if verbose and log_every and batch_index % log_every == 0 and batch_index == 0:
                message = (
                    f"[{dataset}/{variant_name}] epoch {epoch}/{epochs} "
                    f"batch {batch_index} loss {float(loss.item()):.4f}"
                )
                if logger:
                    logger(message)
                elif epoch == 1:
                    print(message, flush=True)

        train_loss = running_loss / running_total if running_total else 0.0
        train_accuracy = 100.0 * running_correct / running_total if running_total else 0.0

        if opt_delta is not None and sched_delta is not None:
            sched_delta.step()
        if opt_mask is not None and sched_mask is not None:
            sched_mask.step()

        test_accuracy: Optional[float] = None
        if test_loader is not None and epoch % eval_every == 0:
            test_accuracy, _ = evaluate_variant(
                model,
                classifier,
                test_loader,
                f_out=f_out,
                device=device_obj,
                max_batches=max_eval_batches,
            )

        stats = AblationEpochStats(
            epoch=epoch,
            loss=train_loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            delta_lr=float(opt_delta.param_groups[0]["lr"]) if opt_delta else 0.0,
            mask_lr=float(opt_mask.param_groups[0]["lr"]) if opt_mask else 0.0,
            mapping_updated=mapping_updated,
            seconds=time.time() - epoch_start,
        )
        record.add(stats)

        if verbose and (epoch % max(1, log_every) == 0 or epoch == epochs):
            message = (
                f"[{dataset}/{variant_name}] epoch {epoch}/{epochs} loss {train_loss:.4f} "
                f"train {train_accuracy:.2f}%"
                + (f" test {test_accuracy:.2f}%" if test_accuracy is not None else "")
            )
            if logger:
                logger(message)
            else:
                print(message, flush=True)

    record.train_accuracy = record.epochs[-1].train_accuracy if record.epochs else None
    record.elapsed_seconds = time.time() - start
    if record.final_test_accuracy is None and test_loader is not None:
        accuracy, _ = evaluate_variant(
            model,
            classifier,
            test_loader,
            f_out=f_out,
            device=device_obj,
            max_batches=max_eval_batches,
        )
        record.final_test_accuracy = float(accuracy)
        record.best_test_accuracy = float(accuracy)
        record.best_epoch = epochs
    return record


def _filtered_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs that are not fields of ``cls`` (tolerates signature drift)."""
    try:
        valid = {f.name for f in fields(cls)}
    except Exception:  # pragma: no cover - defensive
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in valid}


def run_single_variant(
    dataset: str,
    variant: str = "ours",
    backbone: str = "resnet18",
    *,
    seed: int = 0,
    device: Optional[Any] = None,
    label_mapping: str = "ilm",
    patch_size: int = DEFAULT_PATCH_SIZE,
    num_pooling_layers: Optional[int] = None,
    data_root: Optional[str] = None,
    root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    classifier: Optional[nn.Module] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> AblationHistory:
    """Train and evaluate one masking variant on one dataset for one seed."""
    canonical = canonical_variant(variant)
    name = canonical_backbone(backbone)
    device_obj = resolve_device(device)
    set_seed(seed)

    if build_dataloaders is None:
        raise ImportError("smm_vr.data.datasets.build_dataloaders is unavailable")
    batch_size = batch_size_for(dataset)
    train_loader, test_loader, spec = build_dataloaders(
        dataset,
        backbone=name,
        root=root,
        data_root=data_root,
        batch_size=batch_size,
        test_batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        train_fraction=train_fraction,
        split_seed=split_seed,
        device=str(device_obj),
    )
    classes = int(getattr(spec, "num_classes", None) or _dataset_num_classes(dataset))

    if classifier is None:
        if build_classifier is None:
            raise ImportError("smm_vr.models.pretrained.build_classifier is unavailable")
        classifier = build_classifier(backbone=name, device=str(device_obj))
    classifier = classifier.to(device_obj)

    model = build_ablation_reprogram(
        canonical,
        backbone=name,
        patch_size=patch_size,
        num_pooling_layers=num_pooling_layers,
        device=device_obj,
    )

    overrides = dict(config_overrides or {})
    overrides.setdefault("label_mapping", label_mapping)
    overrides.setdefault("patch_size", patch_size)
    overrides.setdefault("batch_size", batch_size)
    overrides.setdefault("num_workers", num_workers)
    overrides.setdefault("seed", seed)
    overrides.setdefault("device", str(device_obj))
    overrides.setdefault("verbose", verbose)
    overrides.setdefault("max_train_batches", max_train_batches)
    overrides.setdefault("max_eval_batches", max_eval_batches)
    overrides.setdefault("save_dir", save_dir)
    config = AblationTrainConfig(**_filtered_kwargs(AblationTrainConfig, overrides))

    history = train_variant(
        model,
        classifier,
        train_loader,
        test_loader=test_loader,
        f_out=None,
        config=config,
        dataset=dataset,
        backbone=name,
        variant=canonical,
        num_target_classes=classes,
        device=device_obj,
        logger=logger,
    )
    history.config.setdefault("dataset", dataset)
    history.config.setdefault("classes", classes)
    return history


# --------------------------------------------------------------------------- #
# Experiment 1 - masking strategies (Table 3)
# --------------------------------------------------------------------------- #


def run_masking_ablation(
    datasets: Optional[Sequence[str]] = None,
    backbone: str = "resnet18",
    variants: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    *,
    output_dir: Optional[str] = None,
    save: bool = True,
    data_root: Optional[str] = None,
    root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    label_mapping: str = "ilm",
    patch_size: int = DEFAULT_PATCH_SIZE,
    device: Optional[Any] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    verbose: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Table 3: masking strategies with ResNet-18, three seeds.

    Expected averages: OURS 52.53, single-channel 49.70, only-delta 46.85,
    only-f_mask 42.59.
    """
    dataset_list = [str(d).strip().lower() for d in (datasets or MAIN_DATASETS)]
    variant_list = [canonical_variant(v) for v in (variants or MASKING_VARIANTS)]
    seed_list = resolve_seeds(seeds)

    results: Dict[str, Dict[str, Any]] = {dataset: {} for dataset in dataset_list}
    histories: Dict[str, Dict[str, Any]] = {dataset: {} for dataset in dataset_list}
    classifier = None
    if build_classifier is not None:
        try:
            classifier = build_classifier(backbone=canonical_backbone(backbone))
        except Exception:  # pragma: no cover - weight download issues
            classifier = None

    for dataset in dataset_list:
        for variant in variant_list:
            per_seed: List[float] = []
            run_histories: List[AblationHistory] = []
            for seed in seed_list:
                history = run_single_variant(
                    dataset,
                    variant,
                    backbone,
                    seed=seed,
                    device=device,
                    label_mapping=label_mapping,
                    patch_size=patch_size,
                    data_root=data_root,
                    root=root,
                    num_workers=num_workers,
                    download=download,
                    train_fraction=train_fraction,
                    split_seed=split_seed,
                    config_overrides=config_overrides,
                    max_train_batches=max_train_batches,
                    max_eval_batches=max_eval_batches,
                    verbose=verbose,
                    classifier=classifier,
                    logger=logger,
                )
                per_seed.append(float(history.accuracy))
                run_histories.append(history)
            mean, std = aggregate_seeds(per_seed)
            results[dataset][variant] = {
                "per_seed_accuracy": per_seed,
                "mean": mean,
                "std": std,
                "formatted": format_mean_std(mean, std),
            }
            histories[dataset][variant] = [h.as_dict() for h in run_histories]
            if verbose:
                print(f"[masking] {dataset}/{variant}: {format_mean_std(mean, std)}", flush=True)

    averages: Dict[str, float] = {}
    for variant in variant_list:
        per_dataset = {
            dataset: results[dataset][variant]["mean"]
            for dataset in dataset_list
            if variant in results[dataset]
        }
        averages[variant] = mean_over_datasets(per_dataset, dataset_order=dataset_list)

    comparison = compare_table3(
        results, averages=averages, dataset_order=dataset_list, verbose=verbose
    )
    payload: Dict[str, Any] = {
        "experiment": "masking_ablation",
        "reference_table": "table3",
        "backbone": canonical_backbone(backbone),
        "datasets": dataset_list,
        "variants": variant_list,
        "seeds": seed_list,
        "label_mapping": label_mapping,
        "patch_size": int(patch_size),
        "results": results,
        "averages": averages,
        "reference_means": {d: TABLE3_MEANS.get(d, {}) for d in dataset_list},
        "reference_stds": {d: TABLE3_STDS.get(d, {}) for d in dataset_list},
        "reference_averages": dict(TABLE3_AVERAGES),
        "comparison": comparison,
        "verification": verify_masking_ablation(comparison),
        "table": format_ablation_table(
            results, averages, dataset_order=dataset_list, variants=variant_list
        ),
        "history": histories,
    }
    ABLATION_RESULTS["masking"] = payload
    if save:
        payload["path"] = save_ablation_results(
            payload, output_dir, filename="ablations_masking.json"
        )
    return payload


def format_ablation_table(
    results: Dict[str, Dict[str, Any]],
    averages: Optional[Dict[str, float]] = None,
    *,
    dataset_order: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    decimals: int = 1,
) -> str:
    """Render a Table-3-style fixed-width text table."""
    order = list(dataset_order) if dataset_order else list(results.keys())
    columns = [canonical_variant(v) for v in (variants or MASKING_VARIANTS)]
    header = f"{'DATASET':<14}" + "".join(f"{VARIANT_DISPLAY[c]:>24}" for c in columns)
    lines = [header, "-" * len(header)]
    for dataset in order:
        cells = []
        for column in columns:
            entry = results.get(dataset, {}).get(column)
            if entry is None:
                cells.append(f"{'-':>24}")
            else:
                cells.append(f"{format_mean_std(entry['mean'], entry['std'], decimals):>24}")
        lines.append(f"{dataset.upper():<14}" + "".join(cells))
    if averages:
        lines.append("-" * len(header))
        cells = [f"{averages.get(column, float('nan')):>23.{decimals}f} " for column in columns]
        lines.append(f"{'AVERAGE':<14}" + "".join(cells))
    return "\n".join(lines)


def compare_table3(
    results: Dict[str, Dict[str, Any]],
    *,
    averages: Optional[Dict[str, float]] = None,
    dataset_order: Optional[Sequence[str]] = None,
    tolerance: float = 2.0,
    ordering_tolerance: float = 0.5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare measured masking ablations against Table 3 of the paper."""
    order = [d for d in (dataset_order or list(results.keys())) if d in TABLE3_MEANS]
    per_dataset: Dict[str, Dict[str, float]] = {}
    worst = 0.0
    for dataset in order:
        measured = results.get(dataset, {})
        diffs: Dict[str, float] = {}
        for variant, reference in TABLE3_MEANS[dataset].items():
            entry = measured.get(variant)
            if entry is None:
                continue
            diff = float(entry["mean"]) - float(reference)
            diffs[variant] = diff
            worst = max(worst, abs(diff))
        if diffs:
            per_dataset[dataset] = diffs

    average_diffs: Dict[str, float] = {}
    if averages:
        for variant, reference in TABLE3_AVERAGES.items():
            if variant in averages:
                average_diffs[variant] = float(averages[variant]) - float(reference)

    # Ordering claim: "SMM consistently stands out as the best performer on all
    # datasets" -> ours >= every other variant on every dataset (small slack).
    ordering_failures: List[str] = []
    for dataset in order:
        measured = results.get(dataset, {})
        if "ours" not in measured:
            continue
        ours = float(measured["ours"]["mean"])
        for variant in MASKING_VARIANTS:
            if variant == "ours" or variant not in measured:
                continue
            if ours + ordering_tolerance < float(measured[variant]["mean"]):
                ordering_failures.append(f"{dataset}:{variant}")

    comparison = {
        "per_dataset_diff": per_dataset,
        "average_diff": average_diffs,
        "max_abs_diff": worst,
        "within_tolerance": worst <= float(tolerance),
        "tolerance": float(tolerance),
        "ordering_failures": ordering_failures,
        "ordering_holds": not ordering_failures,
        "reference_averages": dict(TABLE3_AVERAGES),
    }
    if verbose:
        print(format_comparison_report(comparison))
    return comparison


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """Human-readable summary of :func:`compare_table3`."""
    lines = ["Table 3 comparison (measured - reference):"]
    for variant, diff in sorted(comparison.get("average_diff", {}).items()):
        lines.append(f"  average {variant:<22}: {diff:+.2f}")
    max_diff = comparison.get("max_abs_diff")
    if max_diff is not None:
        lines.append(f"  max |per-dataset diff|: {max_diff:.2f}")
    if comparison.get("ordering_failures"):
        lines.append("  ordering failures: " + ", ".join(comparison["ordering_failures"]))
    else:
        lines.append("  ordering: OURS >= every other variant on every dataset")
    return "\n".join(lines)


def verify_masking_ablation(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise whether the measured ablations reproduce Table 3."""
    averages_ok = True
    for variant, diff in comparison.get("average_diff", {}).items():
        limit = 2.0 if variant == "ours" else 3.0
        if abs(diff) > limit:
            averages_ok = False
    return {
        "averages_within_tolerance": averages_ok,
        "ordering_holds": bool(comparison.get("ordering_holds", False)),
        "max_abs_diff": comparison.get("max_abs_diff"),
        "reference_averages": dict(TABLE3_AVERAGES),
        "note": (
            "Expected Table 3 averages: OURS 52.53, single-channel 49.70, "
            "only-delta 46.85, only-f_mask 42.59 (ResNet-18, Ilm, 3 seeds)."
        ),
    }


# --------------------------------------------------------------------------- #
# Experiment 2 - patch-size sweep (Figure 4)
# --------------------------------------------------------------------------- #


def run_patch_size_study(
    datasets: Optional[Sequence[str]] = None,
    l_values: Optional[Sequence[int]] = None,
    backbone: str = "resnet18",
    seeds: Optional[Sequence[int]] = None,
    *,
    output_dir: Optional[str] = None,
    save: bool = True,
    data_root: Optional[str] = None,
    root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    label_mapping: str = "ilm",
    device: Optional[Any] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    verbose: bool = True,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Figure 4: sweep ``l in {0,1,2,3,4}`` (patch sizes 1..16).

    Uses the full SMM reprogramming function with ``l`` 2x2 max-pooling layers
    in the mask generator, i.e. patch size ``2**l``, and ResNet-18 as the
    pre-trained model.
    """
    dataset_list = [str(d).strip().lower() for d in (datasets or PATCH_STUDY_DATASETS)]
    l_list = [int(l) for l in (l_values if l_values is not None else PATCH_STUDY_L)]
    for l in l_list:
        if not 0 <= l <= MAX_POOLING_LAYERS_5:
            raise ValueError(f"l must be in [0, {MAX_POOLING_LAYERS_5}], got {l}")
    seed_list = resolve_seeds(seeds)

    results: Dict[str, Dict[str, Any]] = {dataset: {} for dataset in dataset_list}
    histories: Dict[str, Dict[str, Any]] = {dataset: {} for dataset in dataset_list}
    classifier = None
    if build_classifier is not None:
        try:
            classifier = build_classifier(backbone=canonical_backbone(backbone))
        except Exception:  # pragma: no cover - weight download issues
            classifier = None

    for dataset in dataset_list:
        for l in l_list:
            per_seed: List[float] = []
            run_histories: List[AblationHistory] = []
            for seed in seed_list:
                history = run_single_variant(
                    dataset,
                    "ours",
                    backbone,
                    seed=seed,
                    device=device,
                    label_mapping=label_mapping,
                    patch_size=2 ** l,
                    num_pooling_layers=l,
                    data_root=data_root,
                    root=root,
                    num_workers=num_workers,
                    download=download,
                    train_fraction=train_fraction,
                    split_seed=split_seed,
                    config_overrides=config_overrides,
                    max_train_batches=max_train_batches,
                    max_eval_batches=max_eval_batches,
                    verbose=verbose,
                    classifier=classifier,
                    logger=logger,
                )
                per_seed.append(float(history.accuracy))
                run_histories.append(history)
            mean, std = aggregate_seeds(per_seed)
            results[dataset][str(l)] = {
                "patch_size": 2 ** l,
                "l": l,
                "per_seed_accuracy": per_seed,
                "mean": mean,
                "std": std,
                "formatted": format_mean_std(mean, std),
            }
            histories[dataset][str(l)] = [h.as_dict() for h in run_histories]
            if verbose:
                print(
                    f"[patch-size] {dataset} l={l} (patch {2 ** l}): "
                    f"{format_mean_std(mean, std)}",
                    flush=True,
                )

    averages = {
        str(l): mean_over_datasets(
            {d: results[d][str(l)]["mean"] for d in dataset_list if str(l) in results[d]},
            dataset_order=dataset_list,
        )
        for l in l_list
    }
    best_l = {
        dataset: int(max(results[dataset], key=lambda k: results[dataset][k]["mean"]))
        for dataset in dataset_list
        if results[dataset]
    }

    payload: Dict[str, Any] = {
        "experiment": "patch_size_study",
        "reference_figure": "figure4",
        "backbone": canonical_backbone(backbone),
        "datasets": dataset_list,
        "l_values": l_list,
        "patch_sizes": [2 ** l for l in l_list],
        "default_patch_size": DEFAULT_PATCH_SIZE,
        "default_l": DEFAULT_PATCH_SIZE_INDEX,
        "seeds": seed_list,
        "label_mapping": label_mapping,
        "results": results,
        "averages": averages,
        "best_l": best_l,
        "check": check_patch_size_trend(averages, best_l, verbose=verbose),
        "table": format_patch_size_table(
            results, averages, dataset_order=dataset_list, l_values=l_list
        ),
        "history": histories,
    }
    ABLATION_RESULTS["patch_size"] = payload
    if save:
        payload["path"] = save_ablation_results(
            payload, output_dir, filename="ablations_patch_size.json"
        )
        curve_path = patch_size_curve_path(output_dir)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(curve_path)) or ".", exist_ok=True)
            with open(curve_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "backbone": payload["backbone"],
                        "datasets": dataset_list,
                        "l_values": l_list,
                        "curves": {
                            dataset: [results[dataset][str(l)]["mean"] for l in l_list]
                            for dataset in dataset_list
                        },
                        "averages": [averages[str(l)] for l in l_list],
                    },
                    handle,
                    indent=2,
                )
            payload["curve_path"] = curve_path
        except Exception:  # pragma: no cover - non-fatal
            pass
    return payload


def patch_size_curve_path(output_dir: Optional[str] = None) -> str:
    """Where the patch-size curves are cached for :mod:`analysis.plot_patch_size`."""
    root = output_dir or os.path.join("outputs", "figures")
    return os.path.join(root, "patch_size_curves.json")


def format_patch_size_table(
    results: Dict[str, Dict[str, Any]],
    averages: Optional[Dict[str, float]] = None,
    *,
    dataset_order: Optional[Sequence[str]] = None,
    l_values: Optional[Sequence[int]] = None,
    decimals: int = 1,
) -> str:
    """Render the patch-size sweep as a text table (rows: datasets, cols: 2^l)."""
    order = list(dataset_order) if dataset_order else list(results.keys())
    l_list = [int(l) for l in (l_values if l_values is not None else PATCH_STUDY_L)]
    header = f"{'DATASET':<14}" + "".join(f"{'2^' + str(l):>12}" for l in l_list)
    lines = [header, "-" * len(header)]
    for dataset in order:
        cells = []
        for l in l_list:
            entry = results.get(dataset, {}).get(str(l))
            cells.append(f"{entry['mean']:>11.{decimals}f} " if entry else f"{'-':>12}")
        lines.append(f"{dataset.upper():<14}" + "".join(cells))
    if averages:
        lines.append("-" * len(header))
        cells = [f"{averages.get(str(l), float('nan')):>11.{decimals}f} " for l in l_list]
        lines.append(f"{'AVERAGE':<14}" + "".join(cells))
    return "\n".join(lines)


def check_patch_size_trend(
    averages: Dict[str, float],
    best_l: Optional[Dict[str, int]] = None,
    *,
    tolerance: float = 0.5,
    min_gain: float = 0.5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Verify the Figure 4 claim that accuracy rises then plateaus or declines.

    Args:
        averages: ``{str(l): mean accuracy}`` averaged over datasets.
        best_l: Per-dataset argmax ``l`` (optional).
        tolerance: Plateau slack allowed between the peak and the later values.
        min_gain: Minimum gain required between ``l=0`` and the peak.

    Returns:
        Dict with the trend, the argmax ``l``, and boolean checks including
        ``near_optimal_at_patch_size_8``.
    """
    curve = {int(l): float(v) for l, v in averages.items()}
    if not curve:
        return {"trend": {}, "best_l": None, "rises_then_plateaus_or_declines": False}
    peak_l = max(curve, key=lambda key: curve[key])
    peak_value = curve[peak_l]
    ordered = sorted(curve)
    initial_gain = curve[peak_l] - curve[ordered[0]]
    rises = initial_gain >= float(min_gain)
    # Values after the peak must not exceed the peak by more than `tolerance`.
    declines_ok = all(curve[l] <= peak_value + float(tolerance) for l in ordered)
    near_optimal = abs(curve.get(DEFAULT_PATCH_SIZE_INDEX, float("-inf")) - peak_value) <= max(
        tolerance, 1.0
    )
    result = {
        "trend": {str(l): curve[l] for l in ordered},
        "best_l": peak_l,
        "best_patch_size": 2 ** peak_l,
        "best_value": peak_value,
        "initial_gain": initial_gain,
        "rises_first": rises,
        "plateau_or_decline_after_peak": declines_ok,
        "rises_then_plateaus_or_declines": bool(rises and declines_ok),
        "near_optimal_at_patch_size_8": bool(near_optimal),
        "best_l_per_dataset": dict(best_l or {}),
        "note": (
            "Figure 4: accuracy increases with patch size, then plateaus or declines; "
            "the paper fixes patch size 8 for all datasets."
        ),
    }
    if verbose:
        print(
            f"[patch-size] trend {result['trend']} -> best l={peak_l} (2^{peak_l}), "
            f"patch-8 near-optimal: {result['near_optimal_at_patch_size_8']}"
        )
    return result


# --------------------------------------------------------------------------- #
# Top-level runners / persistence
# --------------------------------------------------------------------------- #

_MODES_MASKING = ("all", "masking", "table3", "table_3", "mask")
_MODES_PATCH = ("all", "patch", "patch_size", "patch-size", "figure4", "figure_4")


def run_ablation_experiment(
    *,
    mode: str = "all",
    datasets: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    l_values: Optional[Sequence[int]] = None,
    patch_datasets: Optional[Sequence[str]] = None,
    backbone: str = "resnet18",
    seeds: Optional[Sequence[int]] = None,
    output_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the masking ablation (Table 3), the patch-size study (Figure 4), or both.

    Args:
        mode: ``"masking"``/``"table3"``, ``"patch"``/``"figure4"``, or ``"all"``.

    Returns:
        Dict with the keys ``"masking"`` and/or ``"patch_size"``.
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in _MODES_MASKING + _MODES_PATCH:
        raise ValueError(f"unknown ablation mode {mode!r}")
    payload: Dict[str, Any] = {
        "experiment": "ablations",
        "mode": mode_key,
        "backbone": canonical_backbone(backbone),
    }
    if mode_key in _MODES_MASKING:
        payload["masking"] = run_masking_ablation(
            datasets=datasets,
            backbone=backbone,
            variants=variants,
            seeds=seeds,
            output_dir=output_dir,
            save=save,
            verbose=verbose,
            **kwargs,
        )
    if mode_key in _MODES_PATCH:
        payload["patch_size"] = run_patch_size_study(
            datasets=patch_datasets if patch_datasets is not None else datasets,
            l_values=l_values,
            backbone=backbone,
            seeds=seeds,
            output_dir=output_dir,
            save=save,
            verbose=verbose,
            **kwargs,
        )
    if save:
        payload["path"] = save_ablation_results(
            payload, output_dir, filename=f"ablations_{mode_key}.json"
        )
    return payload


def run_ablations_experiment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Alias of :func:`run_ablation_experiment` (experiment-registry entry point)."""
    return run_ablation_experiment(*args, **kwargs)


def save_ablation_results(
    payload: Dict[str, Any],
    output_dir: Optional[str] = None,
    *,
    filename: str = "ablations.json",
) -> str:
    """Persist ablation results to JSON (training histories are compacted)."""
    root = output_dir or os.path.join("outputs", "results")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_strip_histories(payload), handle, indent=2, default=str)
    return path


def _strip_histories(payload: Any) -> Any:
    """Recursively replace ``history`` entries by compact per-seed summaries."""
    if isinstance(payload, dict):
        return {
            key: (_compact_histories(value) if key == "history" else _strip_histories(value))
            for key, value in payload.items()
        }
    if isinstance(payload, (list, tuple)):
        return [_strip_histories(item) for item in payload]
    return payload


def _compact_histories(value: Any) -> Any:
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, list) and item and isinstance(item[0], dict):
                compact[key] = [
                    {
                        "seed": entry.get("seed"),
                        "final_test_accuracy": entry.get("final_test_accuracy"),
                        "best_test_accuracy": entry.get("best_test_accuracy"),
                        "mask_parameters": entry.get("mask_parameters"),
                    }
                    for entry in item
                ]
            else:
                compact[key] = _compact_histories(item)
        return compact
    if isinstance(value, list):
        return [_compact_histories(item) for item in value]
    return value


def describe_ablation(
    datasets: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    backbone: str = "resnet18",
    l_values: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Metadata describing the ablation experiments (for logging / dry runs)."""
    dataset_list = [str(d).lower() for d in (datasets or MAIN_DATASETS)]
    variant_list = [canonical_variant(v) for v in (variants or MASKING_VARIANTS)]
    l_list = [int(l) for l in (l_values if l_values is not None else PATCH_STUDY_L)]
    key = canonical_backbone(backbone)
    mask_params = None
    try:
        mask_params = count_parameters(build_ablation_reprogram("ours", backbone=key))
    except Exception:  # pragma: no cover - guarded builds
        mask_params = None
    alpha_mask, gamma_mask = mask_lr_schedule(key)
    return {
        "experiment": "ablations",
        "backbone": key,
        "num_mask_layers": num_mask_layers_for(key),
        "mask_parameters": mask_params,
        "expected_mask_parameters": EXPECTED_PARAMETERS.get(key),
        "datasets": dataset_list,
        "variants": variant_list,
        "l_values": l_list,
        "patch_sizes": [2 ** l for l in l_list],
        "default_patch_size": DEFAULT_PATCH_SIZE,
        "training": {
            "epochs": DEFAULT_EPOCHS,
            "milestones": list(DEFAULT_MILESTONES),
            "alpha_delta": DEFAULT_ALPHA_DELTA,
            "gamma_delta": DEFAULT_GAMMA_DELTA,
            "alpha_mask": alpha_mask,
            "gamma_mask": gamma_mask,
            "batch_size": DEFAULT_BATCH_SIZE,
        },
        "reference_table3_averages": dict(TABLE3_AVERAGES),
        "reference_table3_means": TABLE3_MEANS,
        "reference_table3_stds": TABLE3_STDS,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(
        description="SMM ablations: masking strategies (Table 3) and patch size (Figure 4)."
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "masking", "patch", "patch_size"],
        help="which ablation to run (default: all)",
    )
    parser.add_argument("--backbone", default="resnet18", help="pre-trained backbone")
    parser.add_argument("--datasets", nargs="*", default=None, help="datasets for Table 3")
    parser.add_argument(
        "--patch-datasets", nargs="*", default=None, help="datasets for the patch-size sweep"
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="masking variants (default: the four columns of Table 3)",
    )
    parser.add_argument("--l-values", nargs="*", type=int, default=None, help="pooling layers l")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="random seeds")
    parser.add_argument("--label-mapping", default="ilm", choices=["ilm", "flm", "rlm"])
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--data-root", default=None, help="dataset root directory")
    parser.add_argument("--output-dir", default=None, help="where JSON results are written")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=None, help="debug subsampling")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--describe", action="store_true", help="print experiment metadata only")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entry
    args = build_arg_parser().parse_args(argv)
    if args.describe:
        print(
            json.dumps(
                describe_ablation(args.datasets, args.variants, args.backbone, args.l_values),
                indent=2,
                default=str,
            )
        )
        return 0
    run_ablation_experiment(
        mode=args.mode,
        datasets=args.datasets,
        variants=args.variants,
        l_values=args.l_values,
        patch_datasets=args.patch_datasets,
        backbone=args.backbone,
        seeds=args.seeds,
        output_dir=args.output_dir,
        save=not args.no_save,
        verbose=not args.quiet,
        data_root=args.data_root,
        num_workers=args.num_workers,
        download=not args.no_download,
        train_fraction=args.train_fraction,
        label_mapping=args.label_mapping,
        patch_size=args.patch_size,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    return 0


__all__ = [
    # constants
    "MASKING_VARIANTS",
    "VARIANT_DISPLAY",
    "VARIANT_ALIASES",
    "PATCH_STUDY_L",
    "PATCH_SIZES",
    "DEFAULT_PATCH_SIZE_INDEX",
    "PATCH_STUDY_DATASETS",
    "MAX_POOLING_LAYERS_5",
    "TABLE3_MEANS",
    "TABLE3_STDS",
    "TABLE3_AVERAGES",
    "TABLE3_ORDERING",
    "TRAINING_DEFAULTS",
    "ABLATION_RESULTS",
    # modules
    "OnlyDeltaReprogram",
    "OnlyFmaskReprogram",
    "SingleChannelFmaskReprogram",
    # builders
    "build_ablation_reprogram",
    "build_patch_size_reprogram",
    "build_variant_models",
    "canonical_variant",
    "canonical_backbone",
    "num_mask_layers_for",
    "mask_lr_schedule",
    "batch_size_for",
    "resolve_device",
    # training
    "AblationTrainConfig",
    "AblationEpochStats",
    "AblationHistory",
    "train_variant",
    "run_single_variant",
    "evaluate_variant",
    "apply_output_mapping",
    "advance_classifier",
    "build_optimizer_and_scheduler",
    "refresh_mapping",
    # experiments
    "run_masking_ablation",
    "run_patch_size_study",
    "run_ablation_experiment",
    "run_ablations_experiment",
    # reporting
    "format_ablation_table",
    "format_patch_size_table",
    "compare_table3",
    "check_patch_size_trend",
    "verify_masking_ablation",
    "describe_ablation",
    "save_ablation_results",
    "patch_size_curve_path",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
