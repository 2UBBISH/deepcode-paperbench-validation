"""Finetuning-based transfer baselines for the SMM paper.

This module implements the finetuning comparison methods used in the paper to show
that input visual reprogramming (SMM) is *orthogonal* to finetuning-based transfer:

* **Finetuning-LoRA** (Hu et al., 2021; Zhu et al., 2023) -- low-rank adaptation of a
  pre-trained ViT.  The paper (Appendix E.1) reports rank ``6``, learning rate
  ``0.01``, ``10`` epochs and ``0.60 M`` extra parameters (not counting the fully
  connected layers), seen against ``0.54 M`` for SMM + ViT-Large (Table 13).
  "Since LoRA for ViT already includes finetuning the fully connected layers, we also
  incorporate it in SMM. All training settings are kept the same."

* **Finetuning-FC** (Appendix E.2) -- finetuning only the final fully connected layer of
  a frozen ResNet-50, optionally combined with the SMM input module
  ("Finetuning-FC + Our SMM"), improving the 11-dataset average from ``75.3`` to
  ``79.2`` (Table 14).

The module also carries the reference numbers of Table 12 (StanfordCars, the
ineffective case of input VR) and Table 13/14 so that experiment runners can verify
their measured accuracies.

Paper ambiguity notes (documented defaults, since the paper leaves them open):

* LoRA scaling ``alpha / r`` is not given; we default ``alpha = r`` (i.e. scaling 1.0).
  The low-rank branch is zero-initialised (``B = 0``) in the standard LoRA way, so the
  model starts from the pre-trained function regardless of the scaling choice.
* The LoRA target modules are not named; ``0.60 M`` extra parameters for ViT-L/16 at
  rank 6 is reproduced by adapting the fused ``qkv`` projection of every transformer
  block (24 blocks x (6x1024 + 3072x6) = 589,824 parameters).
* Finetuning-FC keeps the SMM training protocol (200 epochs, milestones 100/145,
  LR 0.01, gamma 0.1) because the paper only states that the settings are "kept the
  same" for the LoRA/SMM comparison (10 epochs there).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Guarded internal imports (keeps this module importable during incremental builds)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - defensive
    from ..models.pretrained import (
        PretrainedClassifier,
        build_pretrained_model,
        input_size_for,
    )
except Exception:  # pragma: no cover
    PretrainedClassifier = None  # type: ignore
    build_pretrained_model = None  # type: ignore

    def input_size_for(name: str = "resnet18", imgsize: Optional[int] = None) -> int:  # type: ignore
        if imgsize is not None:
            return int(imgsize)
        return 384 if "vit" in str(name).lower() else 224


try:  # pragma: no cover - defensive
    from ..modules.reprogram import build_smm_reprogram
except Exception:  # pragma: no cover
    build_smm_reprogram = None  # type: ignore

try:  # pragma: no cover - defensive
    from ..engine.metrics import RunResult, aggregate_seeds, format_mean_std
except Exception:  # pragma: no cover
    RunResult = None  # type: ignore

    def aggregate_seeds(values, ddof: int = 1):  # type: ignore
        vals = [float(v) for v in values]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return mean, math.sqrt(max(var, 0.0))

    def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:  # type: ignore
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"


try:  # pragma: no cover - defensive
    from ..engine.seeds import resolve_seeds, set_seed
except Exception:  # pragma: no cover

    def set_seed(seed: int, **kwargs) -> int:  # type: ignore
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def resolve_seeds(seeds=None, n_seeds=None):  # type: ignore
        if seeds is None:
            return [0, 1, 2]
        return list(seeds)


__all__ = [
    # constants
    "FINETUNE_METHODS",
    "LOW_RES_DATASETS",
    "HIGH_RES_DATASETS",
    "DEFAULT_LORA_RANK",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_LR",
    "DEFAULT_LORA_EPOCHS",
    "DEFAULT_FINETUNE_LR",
    "DEFAULT_FINETUNE_EPOCHS",
    "DEFAULT_MILESTONES",
    "TABLE12_STANFORDCARS",
    "TABLE13_LORA",
    "TABLE13_SMM",
    "TABLE13_REFERENCE",
    "TABLE14_FINETUNE_FC",
    "TABLE14_FINETUNE_FC_SMM",
    "TABLE14_REFERENCE",
    # LoRA
    "LoRALinear",
    "LoRAMultiheadAttention",
    "apply_lora",
    "freeze_except_lora",
    "lora_parameters",
    "count_lora_parameters",
    "build_lora_model",
    # finetuning-FC / SMM combination
    "FinetuneFC",
    "FinetuneFCWithSMM",
    "build_finetune_fc",
    "build_finetune_fc_with_smm",
    "smm_trainable_parameters",
    # training
    "FinetuneConfig",
    "FinetuneEpochStats",
    "FinetuneHistory",
    "train_lora",
    "train_finetune_fc",
    "train_finetune_fc_with_smm",
    "train_finetuning_method",
    "train_finetuning_with_seeds",
    # reporting
    "compare_table13",
    "compare_table14",
    "describe_finetuning",
    "list_finetuning_methods",
]


# ======================================================================================
# Reference constants from the paper
# ======================================================================================

#: All finetuning-based comparison methods implemented here.
FINETUNE_METHODS: Tuple[str, ...] = ("lora", "finetune_fc", "finetune_fc_smm")

#: Target tasks whose inputs are 32x32 (the "distorted input" group of Table 13).
LOW_RES_DATASETS: Tuple[str, ...] = ("cifar10", "cifar100", "svhn", "gtsrb")

#: Target tasks whose inputs are 128x128.
HIGH_RES_DATASETS: Tuple[str, ...] = (
    "flowers102",
    "dtd",
    "ucf101",
    "food101",
    "sun397",
    "eurosat",
    "oxfordpets",
)

#: LoRA hyper-parameters reported in Appendix E.1 ("rank ... six", "learning rate 0.01,
#: running 10 epochs in total").
DEFAULT_LORA_RANK: int = 6
DEFAULT_LORA_ALPHA: float = 6.0  # scaling = alpha / rank = 1.0 (paper does not specify)
DEFAULT_LORA_LR: float = 0.01
DEFAULT_LORA_EPOCHS: int = 10

#: Finetuning-FC schedule (paper keeps the SMM settings; Table 9).
DEFAULT_FINETUNE_LR: float = 0.01
DEFAULT_FINETUNE_EPOCHS: int = 200
DEFAULT_MILESTONES: Tuple[int, int] = (100, 145)
DEFAULT_GAMMA: float = 0.1
DEFAULT_WEIGHT_DECAY: float = 0.0
DEFAULT_MOMENTUM: float = 0.9
DEFAULT_BATCH_SIZE: int = 256

#: Extra (trainable) parameter budgets reported in the paper.
LORA_EXTRA_PARAMETERS_M = 0.60  # Table 13, Finetuning-LoRA (without FC layers)
SMM_EXTRA_PARAMETERS_M = 0.54  # Table 13, Our SMM (delta + 6-layer mask generator)

# Table 12 -- StanfordCars, the ineffective case of input visual reprogramming
# (mean +- std, per backbone, per method).
TABLE12_STANFORDCARS: Dict[str, Dict[str, Tuple[float, float]]] = {
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

# Table 13 -- Finetuning (LoRA) vs SMM on distorded inputs (ViT-Large, 384x384 input).
TABLE13_LORA: Dict[str, float] = {
    "extra_parameters_m": LORA_EXTRA_PARAMETERS_M,
    "cifar10": 95.9,
    "cifar100": 83.6,
    "svhn": 65.3,
    "gtsrb": 66.6,
    "average_32": 77.9,
    "average_128": 83.4,
}

TABLE13_SMM: Dict[str, float] = {
    "extra_parameters_m": SMM_EXTRA_PARAMETERS_M,
    "cifar10": 97.4,
    "cifar100": 87.3,
    "svhn": 91.0,
    "gtsrb": 84.2,
    "average_32": 90.0,
    "average_128": 83.5,
}

TABLE13_REFERENCE: Dict[str, Dict[str, float]] = {
    "lora": TABLE13_LORA,
    "smm": TABLE13_SMM,
}

# Table 14 -- Finetuning-FC without / with the SMM input module (ResNet-50).
TABLE14_FINETUNE_FC: Dict[str, float] = {
    "cifar10": 90.1,
    "cifar100": 70.7,
    "svhn": 63.5,
    "gtsrb": 77.8,
    "flowers102": 90.9,
    "dtd": 67.6,
    "ucf101": 70.8,
    "food101": 57.6,
    "sun397": 53.5,
    "eurosat": 95.7,
    "oxfordpets": 90.4,
    "average": 75.3,
}

TABLE14_FINETUNE_FC_SMM: Dict[str, float] = {
    "cifar10": 91.2,
    "cifar100": 72.4,
    "svhn": 86.9,
    "gtsrb": 85.2,
    "flowers102": 90.9,
    "dtd": 68.2,
    "ucf101": 72.0,
    "food101": 59.6,
    "sun397": 57.9,
    "eurosat": 95.8,
    "oxfordpets": 90.6,
    "average": 79.2,
}

TABLE14_REFERENCE: Dict[str, Dict[str, float]] = {
    "finetune_fc": TABLE14_FINETUNE_FC,
    "finetune_fc_smm": TABLE14_FINETUNE_FC_SMM,
}

#: Feature dimensionality of the pre-trained backbones (used to size the new head).
FEATURE_DIMS: Dict[str, int] = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "vit_b32": 768,
    "vit_b_32": 768,
    "vit_large": 1024,
    "vit_l_16": 1024,
    "vit_l32": 1024,
}

#: Mask-generator depth used by SMM for each backbone family (Sec. 3.2 / Appendix A.2).
MASK_LAYERS_BY_BACKBONE: Dict[str, int] = {
    "resnet18": 5,
    "resnet50": 5,
    "resnet101": 5,
    "vit_b32": 6,
    "vit_large": 6,
    "vit_l32": 6,
}


def _canonical_backbone(name: str) -> str:
    """Normalise a backbone spelling to a registry key."""

    key = str(name or "resnet18").strip().lower().replace("-", "_").replace("/", "_")
    aliases = {
        "resnet_18": "resnet18",
        "resnet_50": "resnet50",
        "resnet_101": "resnet101",
        "vit_b_32": "vit_b32",
        "vitb32": "vit_b32",
        "vit_b": "vit_b32",
        "vit_l_16": "vit_large",
        "vitl16": "vit_large",
        "vit_l": "vit_large",
    }
    if key in aliases:
        return aliases[key]
    if key in MASK_LAYERS_BY_BACKBONE:
        return key
    if "vit" in key:
        return "vit_large" if ("l" in key.split("vit")[-1][:3]) else "vit_b32"
    return key


# ======================================================================================
# LoRA
# ======================================================================================


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with a trainable zero-initialised low-rank branch.

    Implements ``y = W x + b + (alpha / r) * B A x`` (Hu et al., 2021).  ``B`` is
    zero-initialised so the adapted layer is exactly the frozen layer at step 0.
    """

    def __init__(self, base: nn.Linear, rank: int = DEFAULT_LORA_RANK, alpha: Optional[float] = None):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / float(self.rank)

        for param in self.base.parameters():
            param.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """Low-rank weight update ``(alpha / r) * B A``."""

        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(x, self.delta_weight().to(dtype=x.dtype))
        return out + delta

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:g}"


class LoRAMultiheadAttention(nn.Module):
    """Wraps a frozen ``nn.MultiheadAttention`` with rank-``r`` ``qkv`` adapters.

    Torchvision's ViT uses a *fused* ``in_proj_weight`` of shape ``(3E, E)``, so LoRA on
    the query/key/value projections corresponds to adding a stacked low-rank update
    ``[B_q A_q; B_k A_k; B_v A_v]`` to that fused weight.  The forward pass therefore
    calls :func:`torch.nn.functional.multi_head_attention_forward` with
    ``in_proj_weight + delta``.  With ``E = 1024`` and 24 blocks at rank 6 this yields
    ``24 * 6 * (1024 + 3072) = 589,824`` parameters, matching the ``0.60 M`` reported in
    Table 13 (the fully connected layers are counted separately).
    """

    def __init__(self, base: nn.MultiheadAttention, rank: int = DEFAULT_LORA_RANK, alpha: Optional[float] = None):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / float(self.rank)

        for param in self.base.parameters():
            param.requires_grad_(False)

        embed_dim = int(base.embed_dim)
        out_dim = int(base.in_proj_weight.shape[0]) if base.in_proj_weight is not None else 3 * embed_dim
        self.lora_A = nn.Parameter(torch.zeros(self.rank, embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **kwargs: Any,
    ):
        base = self.base
        in_proj_weight = getattr(base, "in_proj_weight", None)
        if in_proj_weight is None or query.shape[-1] != int(base.embed_dim):
            return base(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                **kwargs,
            )

        weight = in_proj_weight + self.delta_weight().to(dtype=in_proj_weight.dtype)
        try:
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                int(base.embed_dim),
                int(base.num_heads),
                weight,
                getattr(base, "in_proj_bias", None),
                getattr(base, "bias_k", None),
                getattr(base, "bias_v", None),
                bool(getattr(base, "add_zero_attn", False)),
                float(getattr(base, "dropout", 0.0)),
                base.out_proj.weight,
                base.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        except TypeError:
            # Older/newer torch signatures: fall back to the un-adapted attention.
            return base(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                **kwargs,
            )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:g}"


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace the submodule of ``model`` addressed by dotted ``name``."""

    if not name:
        raise ValueError("cannot replace the root module")
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, new_module)


def apply_lora(
    model: nn.Module,
    rank: int = DEFAULT_LORA_RANK,
    alpha: Optional[float] = None,
    target: str = "qkv",
    **kwargs: Any,
) -> Tuple[nn.Module, List[nn.Module]]:
    """Inject LoRA adapters into ``model`` (in place).

    ``target`` selects the adapted projections:

    * ``"qkv"`` (default) -- adapt the fused query/key/value projection of every
      ``nn.MultiheadAttention``.  This is the setting that reproduces the ``0.60 M``
      extra-parameter budget of Table 13 for ViT-L/16 at rank 6.
    * ``"proj"`` / ``"out_proj"`` -- adapt the attention output projection.
    * ``"all"`` -- adapt every linear layer of the transformer blocks.

    Returns ``(model, lora_modules)``.
    """

    target = str(target or "qkv").lower()
    lora_modules: List[nn.Module] = []

    if target in ("qkv", "attn", "attention", "self_attention"):
        for name, module in list(model.named_modules()):
            if isinstance(module, LoRAMultiheadAttention):
                continue
            if isinstance(module, nn.MultiheadAttention):
                wrapper = LoRAMultiheadAttention(module, rank=rank, alpha=alpha)
                _set_submodule(model, name, wrapper)
                lora_modules.append(wrapper)
        return model, lora_modules

    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        short = name.rsplit(".", 1)[-1]
        if target == "all":
            ok = True
        elif target in ("proj", "out_proj", "o_proj"):
            ok = short in ("proj", "out_proj", "o_proj")
        elif target in ("qkv", "q", "v", "qkv_proj"):
            ok = short in ("qkv", "qkv_proj", "q_proj", "k_proj", "v_proj", "in_proj")
        else:
            ok = short == target
        if ok:
            wrapper = LoRALinear(module, rank=rank, alpha=alpha)
            _set_submodule(model, name, wrapper)
            lora_modules.append(wrapper)

    return model, lora_modules


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return the trainable LoRA parameters of ``model``."""

    params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            params.append(param)
    return params


def count_lora_parameters(model: nn.Module) -> int:
    """Count the number of trainable LoRA parameters."""

    return int(sum(p.numel() for p in lora_parameters(model)))


def freeze_except_lora(model: nn.Module, *, train_head: bool = True, head_names: Sequence[str] = ("heads", "head", "fc", "classifier")) -> nn.Module:
    """Freeze everything except LoRA adapters (and optionally the classifier head)."""

    for param in model.parameters():
        param.requires_grad_(False)
    for param in lora_parameters(model):
        param.requires_grad_(True)

    if train_head:
        for name, module in model.named_modules():
            if name.endswith(head_names) or name in head_names:
                for param in module.parameters():
                    param.requires_grad_(True)
    return model


def _replace_head(model: nn.Module, num_classes: int, feature_dim: Optional[int] = None) -> nn.Module:
    """Replace the classifier head of ``model`` with a fresh ``num_classes`` head."""

    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        in_features = int(model.heads.head.in_features)
        model.heads.head = nn.Linear(in_features, int(num_classes))
        return model
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
        in_features = int(model.head.in_features)
        model.head = nn.Linear(in_features, int(num_classes))
        return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, int(num_classes))
        return model
    if feature_dim is None:
        raise ValueError("cannot infer the classifier head of the model; pass feature_dim")
    head = nn.Linear(int(feature_dim), int(num_classes))
    model.add_module("fc", head)
    return model


def build_lora_model(
    backbone: str = "vit_large",
    num_classes: int = 10,
    *,
    rank: int = DEFAULT_LORA_RANK,
    alpha: Optional[float] = None,
    target: str = "qkv",
    weights: str = "IMAGENET1K_V1",
    pretrained: bool = True,
    train_head: bool = True,
    feature_dim: Optional[int] = None,
    device: Optional[Any] = None,
) -> nn.Module:
    """Build a pre-trained model with LoRA adapters and a fresh classifier head.

    Mirrors Appendix E.1: "Since LoRA for ViT already includes finetuning the fully
    connected layers, we also incorporate it in SMM" -- i.e. the head stays trainable.
    """

    backbone = _canonical_backbone(backbone)
    if build_pretrained_model is None:  # pragma: no cover - torchvision missing
        raise ImportError("torchvision is required to build the LoRA model")

    model = build_pretrained_model(backbone, weights=weights, pretrained=pretrained)
    model = _replace_head(model, num_classes, feature_dim=feature_dim or FEATURE_DIMS.get(backbone))
    model, _ = apply_lora(model, rank=rank, alpha=alpha, target=target)
    freeze_except_lora(model, train_head=train_head)
    if device is not None:
        model = model.to(device)
    return model


# ======================================================================================
# Finetuning the fully connected layer (with / without the SMM input module)
# ======================================================================================


def _infer_feature_dim(classifier: nn.Module, input_size: int, backbone: str, device: Any) -> int:
    """Infer the pooled feature dimensionality via a dummy forward pass."""

    known = FEATURE_DIMS.get(_canonical_backbone(backbone))
    if known:
        return int(known)
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size, device=device)
            feats = classifier.features(dummy) if hasattr(classifier, "features") else classifier(dummy)
        return int(feats.flatten(1).shape[1])
    except Exception:  # pragma: no cover - defensive
        return 2048


class FinetuneFC(nn.Module):
    """Frozen pre-trained backbone + a trainable fully connected layer.

    This is the "Finetuning-FC" baseline of Appendix E.2 (Table 14).
    """

    def __init__(
        self,
        classifier: nn.Module,
        num_classes: int,
        *,
        backbone: str = "resnet50",
        input_size: int = 224,
        feature_dim: Optional[int] = None,
        device: Optional[Any] = None,
        use_smm: bool = False,
        patch_size: int = 8,
        num_mask_layers: Optional[int] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.backbone = _canonical_backbone(backbone)
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.use_smm = bool(use_smm)

        for param in self.classifier.parameters():
            param.requires_grad_(False)
        self.classifier.eval()

        if feature_dim is None:
            feature_dim = _infer_feature_dim(self.classifier, self.input_size, self.backbone, device or "cpu")
        self.head = nn.Linear(int(feature_dim), int(num_classes))

        self.f_in: Optional[nn.Module] = None
        if self.use_smm:
            if build_smm_reprogram is None:  # pragma: no cover - defensive
                raise ImportError("smm_vr.modules.reprogram.build_smm_reprogram is required for SMM")
            self.f_in = build_smm_reprogram(
                backbone=self.backbone,
                input_size=self.input_size,
                patch_size=int(patch_size),
                num_layers=num_mask_layers,
            )

    # -- helpers ------------------------------------------------------------------
    def features(self, images: torch.Tensor) -> torch.Tensor:
        out = self.classifier.features(images) if hasattr(self.classifier, "features") else self.classifier(images)
        return out.flatten(1) if out.dim() > 2 else out

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = list(self.head.parameters())
        if self.f_in is not None:
            params += [p for p in self.f_in.parameters() if p.requires_grad]
        return params

    def num_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))

    def forward(self, images: torch.Tensor, return_features: bool = False):
        if self.f_in is not None:
            images = self.f_in(images)
        feats = self.features(images)
        logits = self.head(feats)
        if return_features:
            return logits, feats
        return logits

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"backbone={self.backbone}, num_classes={self.num_classes}, "
            f"use_smm={self.use_smm}"
        )


def build_finetune_fc(
    classifier: nn.Module,
    num_classes: int,
    *,
    backbone: str = "resnet50",
    input_size: Optional[int] = None,
    feature_dim: Optional[int] = None,
    device: Optional[Any] = None,
) -> FinetuneFC:
    """Build the plain Finetuning-FC baseline (Appendix E.2, Table 14)."""

    backbone = _canonical_backbone(backbone)
    return FinetuneFC(
        classifier,
        num_classes,
        backbone=backbone,
        input_size=int(input_size or input_size_for(backbone)),
        feature_dim=feature_dim,
        device=device,
        use_smm=False,
    )


def build_finetune_fc_with_smm(
    classifier: nn.Module,
    num_classes: int,
    *,
    backbone: str = "resnet50",
    input_size: Optional[int] = None,
    patch_size: int = 8,
    num_mask_layers: Optional[int] = None,
    feature_dim: Optional[int] = None,
    device: Optional[Any] = None,
) -> FinetuneFC:
    """Build "Finetuning-FC + Our SMM": the SMM input module on top of a frozen
    backbone whose fully connected layer is finetuned (Appendix E.2, Table 14)."""

    backbone = _canonical_backbone(backbone)
    return FinetuneFC(
        classifier,
        num_classes,
        backbone=backbone,
        input_size=int(input_size or input_size_for(backbone)),
        feature_dim=feature_dim,
        device=device,
        use_smm=True,
        patch_size=int(patch_size),
        num_mask_layers=num_mask_layers,
    )


def smm_trainable_parameters(
    backbone: str = "vit_large",
    input_size: Optional[int] = None,
    num_mask_layers: Optional[int] = None,
) -> Dict[str, int]:
    """Breakdown of SMM's trainable parameters (delta + mask generator).

    For ViT-Large at 384x384 this is ``3 * 384 * 384 = 442,368`` (delta) plus the
    ``102,339`` parameters of the 6-layer mask generator, i.e. ``544,707 ~= 0.54 M``,
    matching the budget reported in Table 13.
    """

    backbone = _canonical_backbone(backbone)
    size = int(input_size or input_size_for(backbone))
    layers = int(num_mask_layers or MASK_LAYERS_BY_BACKBONE.get(backbone, 5))

    delta_params = 3 * size * size
    try:
        from ..models.mask_generator import EXPECTED_PARAMETERS  # pylint: disable=import-outside-toplevel

        mask_params = int(EXPECTED_PARAMETERS.get("vit_b32" if layers == 6 else "resnet18", 0))
    except Exception:  # pragma: no cover - defensive
        mask_params = 102339 if layers == 6 else 26499

    return {
        "delta": delta_params,
        "mask_generator": mask_params,
        "total": delta_params + mask_params,
        "input_size": size,
        "num_mask_layers": layers,
    }


# ======================================================================================
# Training utilities
# ======================================================================================


@dataclass
class FinetuneConfig:
    """Configuration of a finetuning-based comparison run."""

    method: str = "lora"
    backbone: str = "vit_large"
    input_size: Optional[int] = None
    epochs: int = DEFAULT_LORA_EPOCHS
    lr: float = DEFAULT_LORA_LR
    gamma: float = DEFAULT_GAMMA
    milestones: Sequence[int] = DEFAULT_MILESTONES
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    momentum: float = DEFAULT_MOMENTUM
    optimizer: str = "sgd"
    batch_size: int = DEFAULT_BATCH_SIZE
    rank: int = DEFAULT_LORA_RANK
    alpha: Optional[float] = DEFAULT_LORA_ALPHA
    lora_target: str = "qkv"
    train_head: bool = True
    patch_size: int = 8
    label_mapping: Optional[str] = None
    num_workers: int = 4
    seed: int = 0
    device: Optional[Any] = None
    log_every: int = 1
    eval_every: int = 1
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    save_dir: Optional[str] = None
    verbose: bool = True

    def __post_init__(self) -> None:
        self.method = canonical_method_name(self.method)
        self.backbone = _canonical_backbone(self.backbone)
        self.input_size = int(self.input_size or input_size_for(self.backbone))
        if self.method == "lora" and self.epochs in (None, 0):
            self.epochs = DEFAULT_LORA_EPOCHS
            self.lr = DEFAULT_LORA_LR

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "backbone": self.backbone,
            "input_size": self.input_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "gamma": self.gamma,
            "milestones": list(self.milestones),
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "rank": self.rank,
            "alpha": self.alpha,
            "lora_target": self.lora_target,
            "patch_size": self.patch_size,
            "label_mapping": self.label_mapping,
            "seed": self.seed,
        }


@dataclass
class FinetuneEpochStats:
    """Per-epoch record of a finetuning run."""

    epoch: int
    loss: float
    train_accuracy: float = float("nan")
    test_accuracy: float = float("nan")
    lr: float = float("nan")
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "lr": self.lr,
            "seconds": self.seconds,
        }


@dataclass
class FinetuneHistory:
    """Full record of a single-seed finetuning run."""

    method: str = "lora"
    dataset: str = ""
    backbone: str = "vit_large"
    seed: int = 0
    epochs: List[FinetuneEpochStats] = field(default_factory=list)
    trainable_parameters: int = 0
    extra_parameters_millions: float = 0.0
    best_test_accuracy: float = float("nan")
    best_epoch: int = -1
    final_test_accuracy: float = float("nan")
    elapsed_seconds: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def add(self, stats: FinetuneEpochStats) -> None:
        self.epochs.append(stats)
        if not math.isnan(stats.test_accuracy):
            if math.isnan(self.best_test_accuracy) or stats.test_accuracy > self.best_test_accuracy:
                self.best_test_accuracy = float(stats.test_accuracy)
                self.best_epoch = int(stats.epoch)
        self.final_test_accuracy = stats.test_accuracy

    @property
    def losses(self) -> List[float]:
        return [e.loss for e in self.epochs]

    @property
    def test_accuracies(self) -> List[float]:
        return [e.test_accuracy for e in self.epochs]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "seed": self.seed,
            "best_test_accuracy": self.best_test_accuracy,
            "best_epoch": self.best_epoch,
            "final_test_accuracy": self.final_test_accuracy,
            "trainable_parameters": self.trainable_parameters,
            "extra_parameters_millions": self.extra_parameters_millions,
            "elapsed_seconds": self.elapsed_seconds,
            "config": self.config,
            "epochs": [e.as_dict() for e in self.epochs],
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2)
        return path


def canonical_method_name(name: str) -> str:
    """Normalise a finetuning method name/alias."""

    key = str(name or "lora").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lora": "lora",
        "finetuning_lora": "lora",
        "finetune_lora": "lora",
        "fc": "finetune_fc",
        "finetune_fc": "finetune_fc",
        "finetuning_fc": "finetune_fc",
        "finetune_fc_smm": "finetune_fc_smm",
        "fc_smm": "finetune_fc_smm",
        "finetuning_fc_smm": "finetune_fc_smm",
        "fc+ours": "finetune_fc_smm",
        "finetune_fc+ours": "finetune_fc_smm",
    }
    if key in aliases:
        return aliases[key]
    if "lora" in key:
        return "lora"
    if "smm" in key or "our" in key:
        return "finetune_fc_smm"
    if "fc" in key:
        return "finetune_fc"
    return key


def _maybe_update_mapping(mapping: Any, model: Any, loader: Any, device: Any) -> bool:
    """Refresh an epoch-wise output label mapping (Ilm) if one is provided."""

    if mapping is None or not getattr(mapping, "recomputes_each_epoch", False):
        return False
    updater = getattr(mapping, "update", None)
    if updater is None:
        return False
    try:
        return bool(updater(model=model, data_loader=loader, device=device))
    except TypeError:  # pragma: no cover - tolerant signature
        try:
            return bool(updater(model, loader))
        except Exception:
            return False
    except Exception:  # pragma: no cover - defensive
        return False


def _map_logits(logits: torch.Tensor, f_out: Any) -> torch.Tensor:
    """Apply an output label mapping to ImageNet logits (identity when ``f_out`` is None)."""

    if f_out is None:
        return logits
    if callable(f_out) and not isinstance(f_out, torch.Tensor):
        return f_out(logits)
    if isinstance(f_out, torch.Tensor):
        index = f_out.to(device=logits.device).long()
        return logits.index_select(-1, index)
    if hasattr(f_out, "target_to_pretrained"):
        index = torch.as_tensor(f_out.target_to_pretrained, device=logits.device).long()
        return logits.index_select(-1, index)
    return logits


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    raise TypeError(f"unexpected batch structure: {type(batch)!r}")


def build_optimizer_and_scheduler(model: nn.Module, config: FinetuneConfig):
    """SGD/Adam + MultiStepLR over the *trainable* parameters of ``model``."""

    params: Iterable[nn.Parameter]
    if hasattr(model, "trainable_parameters"):
        params = model.trainable_parameters()  # type: ignore[assignment]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters found for the finetuning run")

    name = str(config.optimizer or "sgd").lower()
    if name == "adam":
        optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config.milestones), gamma=config.gamma
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: Any,
    *,
    f_out: Any = None,
    device: Any = None,
    max_batches: Optional[int] = None,
    criterion: Optional[Callable[..., torch.Tensor]] = None,
) -> Tuple[float, Optional[float]]:
    """Top-1 accuracy (percent) and optional mean loss on a data loader."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    losses: List[float] = []
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images, targets = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        mapped = _map_logits(logits, f_out)
        preds = mapped.argmax(dim=-1)
        correct += int((preds == targets).sum().item())
        total += int(targets.numel())
        if criterion is not None and mapped.shape[-1] == int(targets.max().item()) + 1:
            losses.append(float(criterion(mapped, targets).item()))

    if was_training:
        model.train()
    accuracy = 100.0 * correct / total if total else 0.0
    mean_loss = sum(losses) / len(losses) if losses else None
    return accuracy, mean_loss


def train_epoch(
    model: nn.Module,
    data_loader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    f_out: Any = None,
    device: Any = None,
    criterion: Optional[Callable[..., torch.Tensor]] = None,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """One optimisation pass; returns ``(mean_loss, train_accuracy_percent)``."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion or nn.CrossEntropyLoss()
    model.train()

    running_loss = 0.0
    seen = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images, targets = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        mapped = _map_logits(logits, f_out)
        loss = criterion(mapped, targets)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        seen += int(images.size(0))
        correct += int((mapped.argmax(dim=-1) == targets).sum().item())
        total += int(targets.numel())

    mean_loss = running_loss / seen if seen else float("nan")
    accuracy = 100.0 * correct / total if total else 0.0
    return mean_loss, accuracy


def _run_training(
    model: nn.Module,
    train_loader: Any,
    *,
    test_loader: Any = None,
    f_out: Any = None,
    config: Optional[FinetuneConfig] = None,
    method: str = "lora",
    dataset: str = "",
    device: Any = None,
    history: Optional[FinetuneHistory] = None,
    logger: Optional[Callable[[str], None]] = None,
    label_mapping: Any = None,
) -> FinetuneHistory:
    """Shared training loop for every finetuning-based method."""

    config = config or FinetuneConfig(method=method)
    device = device or config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    criterion = nn.CrossEntropyLoss()

    trainable = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    history = history or FinetuneHistory(
        method=canonical_method_name(method),
        dataset=dataset,
        backbone=config.backbone,
        seed=int(config.seed),
        trainable_parameters=trainable,
        extra_parameters_millions=trainable / 1e6,
        config=config.as_dict(),
    )
    history.trainable_parameters = trainable
    history.extra_parameters_millions = trainable / 1e6

    started = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.time()

        mapping_updated = _maybe_update_mapping(label_mapping, model, train_loader, device)
        loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            f_out=f_out,
            device=device,
            criterion=criterion,
            max_batches=config.max_train_batches,
        )
        scheduler.step()

        test_acc = float("nan")
        if test_loader is not None and (epoch % max(int(config.eval_every), 1) == 0):
            test_acc, _ = evaluate_model(
                model,
                test_loader,
                f_out=f_out,
                device=device,
                max_batches=config.max_eval_batches,
                criterion=None,
            )

        stats = FinetuneEpochStats(
            epoch=epoch,
            loss=loss,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=time.time() - epoch_start,
        )
        history.add(stats)

        if config.verbose and (epoch % max(int(config.log_every), 1) == 0 or epoch == int(config.epochs)):
            message = (
                f"[{method}][{dataset}][seed {config.seed}] epoch {epoch}/{config.epochs} "
                f"loss {loss:.4f} train {train_acc:.2f} test {test_acc:.2f}"
                + (" (mapping updated)" if mapping_updated else "")
            )
            print(message, flush=True)
            if logger is not None:
                logger(message)

    history.elapsed_seconds = time.time() - started
    if config.save_dir:
        history.save(os.path.join(config.save_dir, f"{method}_{dataset}_seed{config.seed}.json"))
    return history


def train_lora(
    model: nn.Module,
    train_loader: Any,
    *,
    test_loader: Any = None,
    config: Optional[FinetuneConfig] = None,
    dataset: str = "",
    device: Any = None,
    history: Optional[FinetuneHistory] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> FinetuneHistory:
    """Train a LoRA-adapted model (Appendix E.1: rank 6, LR 0.01, 10 epochs)."""

    config = config or FinetuneConfig(method="lora")
    config.method = "lora"
    return _run_training(
        model,
        train_loader,
        test_loader=test_loader,
        f_out=None,
        config=config,
        method="lora",
        dataset=dataset,
        device=device,
        history=history,
        logger=logger,
    )


def train_finetune_fc(
    model: nn.Module,
    train_loader: Any,
    *,
    test_loader: Any = None,
    config: Optional[FinetuneConfig] = None,
    dataset: str = "",
    device: Any = None,
    f_out: Any = None,
    label_mapping: Any = None,
    history: Optional[FinetuneHistory] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> FinetuneHistory:
    """Train the fully connected layer on top of a frozen backbone (Table 14)."""

    config = config or FinetuneConfig(method="finetune_fc", epochs=DEFAULT_FINETUNE_EPOCHS)
    config.method = "finetune_fc"
    return _run_training(
        model,
        train_loader,
        test_loader=test_loader,
        f_out=f_out,
        config=config,
        method="finetune_fc",
        dataset=dataset,
        device=device,
        history=history,
        logger=logger,
        label_mapping=label_mapping,
    )


def train_finetune_fc_with_smm(
    model: nn.Module,
    train_loader: Any,
    *,
    test_loader: Any = None,
    config: Optional[FinetuneConfig] = None,
    dataset: str = "",
    device: Any = None,
    f_out: Any = None,
    label_mapping: Any = None,
    history: Optional[FinetuneHistory] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> FinetuneHistory:
    """Train "Finetuning-FC + Our SMM": the SMM input module plus the FC layer."""

    config = config or FinetuneConfig(method="finetune_fc_smm", epochs=DEFAULT_FINETUNE_EPOCHS)
    config.method = "finetune_fc_smm"
    return _run_training(
        model,
        train_loader,
        test_loader=test_loader,
        f_out=f_out,
        config=config,
        method="finetune_fc_smm",
        dataset=dataset,
        device=device,
        history=history,
        logger=logger,
        label_mapping=label_mapping,
    )


def train_finetuning_method(
    method: str,
    *,
    classifier: Optional[nn.Module] = None,
    model: Optional[nn.Module] = None,
    train_loader: Any = None,
    test_loader: Any = None,
    num_classes: Optional[int] = None,
    dataset: str = "",
    backbone: str = "resnet50",
    config: Optional[FinetuneConfig] = None,
    device: Optional[Any] = None,
    f_out: Any = None,
    label_mapping: Any = None,
    logger: Optional[Callable[[str], None]] = None,
) -> FinetuneHistory:
    """Unified entry point dispatching to the requested finetuning method.

    ``method`` is one of ``"lora"``, ``"finetune_fc"`` or ``"finetune_fc_smm"``.
    When ``model`` is ``None`` it is built from the frozen ``classifier``.
    """

    method = canonical_method_name(method)
    config = config or FinetuneConfig(method=method, backbone=backbone)
    config.method = method
    backbone = _canonical_backbone(config.backbone or backbone)

    if model is None:
        if method == "lora":
            model = build_lora_model(
                backbone,
                int(num_classes or 10),
                rank=config.rank,
                alpha=config.alpha,
                target=config.lora_target,
                train_head=config.train_head,
            )
        else:
            if classifier is None:
                raise ValueError("a frozen classifier is required for the Finetuning-FC methods")
            if method == "finetune_fc_smm":
                model = build_finetune_fc_with_smm(
                    classifier,
                    int(num_classes or 10),
                    backbone=backbone,
                    input_size=config.input_size,
                    patch_size=config.patch_size,
                )
            else:
                model = build_finetune_fc(
                    classifier,
                    int(num_classes or 10),
                    backbone=backbone,
                    input_size=config.input_size,
                )

    return _run_training(
        model,
        train_loader,
        test_loader=test_loader,
        f_out=f_out,
        config=config,
        method=method,
        dataset=dataset,
        device=device,
        logger=logger,
        label_mapping=label_mapping,
    )


def train_finetuning_with_seeds(
    method: str,
    *,
    build_fn: Callable[..., Any],
    datasets: Any = None,
    dataset: str = "",
    backbone: str = "resnet50",
    config: Optional[FinetuneConfig] = None,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[Any] = None,
    logger: Optional[Callable[[str], None]] = None,
    **build_kwargs: Any,
) -> Dict[str, Any]:
    """Train one finetuning method with the paper's seeds and aggregate mean +- std.

    ``build_fn(seed, **kwargs)`` must return a ``(model, train_loader, test_loader)``
    triple (or a dict with those keys).
    """

    config = config or FinetuneConfig(method=method, backbone=backbone)
    seeds = list(resolve_seeds(seeds))
    accuracies: List[float] = []
    histories: List[FinetuneHistory] = []

    for seed in seeds:
        set_seed(int(seed))
        built = build_fn(seed=int(seed), **build_kwargs)
        if isinstance(built, dict):
            model = built["model"]
            train_loader = built.get("train_loader") or built.get("train")
            test_loader = built.get("test_loader") or built.get("test")
        else:
            model, train_loader, test_loader = built

        run_config = copy.deepcopy(config)
        run_config.seed = int(seed)
        history = train_finetuning_method(
            method,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            dataset=dataset,
            backbone=backbone,
            config=run_config,
            device=device,
            logger=logger,
        )
        histories.append(history)
        accuracies.append(float(history.best_test_accuracy))

    mean, std = aggregate_seeds(accuracies)
    result = {
        "method": canonical_method_name(method),
        "dataset": dataset,
        "backbone": backbone,
        "seeds": seeds,
        "per_seed_accuracy": accuracies,
        "mean": mean,
        "std": std,
        "formatted": format_mean_std(mean, std),
        "histories": histories,
    }
    if RunResult is not None:
        result["result"] = RunResult(
            dataset=dataset,
            backbone=backbone,
            method=canonical_method_name(method),
            seed=int(seeds[0]) if seeds else 0,
            accuracy=mean,
            extra={"std": std, "per_seed_accuracy": accuracies},
        )
    return result


# ======================================================================================
# Reporting helpers
# ======================================================================================


def _summary_of(dataset_accuracies: Dict[str, float], keys: Sequence[str]) -> Optional[float]:
    values = [dataset_accuracies[key] for key in keys if key in dataset_accuracies]
    if not values:
        return None
    return sum(values) / len(values)


def compare_table13(
    results: Dict[str, Dict[str, float]],
    *,
    verbose: bool = False,
    tolerance: float = 2.0,
) -> Dict[str, Any]:
    """Compare measured LoRA/SMM accuracies against Table 13 (Appendix E.1).

    ``results`` maps a method name (``"lora"`` or ``"smm"``) to per-dataset accuracies;
    the ``32x32`` and ``128x128`` averages are computed from ``LOW_RES_DATASETS`` and
    ``HIGH_RES_DATASETS`` when not supplied explicitly.
    """

    report: Dict[str, Any] = {"table": "table13", "methods": {}}
    for method, measured in results.items():
        key = "smm" if "smm" in str(method).lower() else "lora"
        reference = TABLE13_REFERENCE[key]
        row: Dict[str, Any] = {"measured": {}, "reference": {}, "delta": {}}
        for dataset, ref_value in reference.items():
            if dataset == "extra_parameters_m":
                continue
            value = measured.get(dataset)
            if value is None and dataset == "average_32":
                value = _summary_of(measured, LOW_RES_DATASETS)
            if value is None and dataset == "average_128":
                value = _summary_of(measured, HIGH_RES_DATASETS)
            if value is None:
                continue
            row["measured"][dataset] = float(value)
            row["reference"][dataset] = float(ref_value)
            row["delta"][dataset] = float(value) - float(ref_value)
        if "average_32" in row["measured"] and abs(row["delta"].get("average_32", 0.0)) <= tolerance:
            row["average_32_ok"] = True
        report["methods"][key] = row
        if verbose:
            print(f"Table 13 -- {key}")
            for dataset, value in row["measured"].items():
                print(
                    f"  {dataset:12s} measured {value:6.2f}  reference "
                    f"{row['reference'][dataset]:6.2f}  delta {row['delta'][dataset]:+6.2f}"
                )
    return report


def compare_table14(
    results: Dict[str, Dict[str, float]],
    *,
    dataset_order: Optional[Sequence[str]] = None,
    verbose: bool = False,
    tolerance: float = 2.0,
) -> Dict[str, Any]:
    """Compare measured Finetuning-FC (+SMM) accuracies against Table 14 (Appendix E.2)."""

    order = list(dataset_order or (LOW_RES_DATASETS + HIGH_RES_DATASETS))
    report: Dict[str, Any] = {"table": "table14", "methods": {}}
    for method, measured in results.items():
        key = "finetune_fc_smm" if ("smm" in str(method).lower() or "our" in str(method).lower()) else "finetune_fc"
        reference = TABLE14_REFERENCE[key]
        row: Dict[str, Any] = {"measured": {}, "reference": {}, "delta": {}}
        for dataset in order + ["average"]:
            if dataset not in reference:
                continue
            value = measured.get(dataset)
            if value is None and dataset == "average":
                value = _summary_of(measured, order)
            if value is None:
                continue
            row["measured"][dataset] = float(value)
            row["reference"][dataset] = float(reference[dataset])
            row["delta"][dataset] = float(value) - float(reference[dataset])
        if "average" in row["measured"] and abs(row["delta"]["average"]) <= tolerance:
            row["average_ok"] = True
        report["methods"][key] = row
        if verbose:
            print(f"Table 14 -- {key}")
            for dataset, value in row["measured"].items():
                print(
                    f"  {dataset:12s} measured {value:6.2f}  reference "
                    f"{row['reference'][dataset]:6.2f}  delta {row['delta'][dataset]:+6.2f}"
                )
    return report


def describe_finetuning(method: str = "lora", backbone: str = "vit_large") -> Dict[str, Any]:
    """Static description of a finetuning method (for logging / config dumps)."""

    method = canonical_method_name(method)
    backbone = _canonical_backbone(backbone)
    info: Dict[str, Any] = {
        "method": method,
        "backbone": backbone,
        "input_size": input_size_for(backbone),
        "reference_extra_parameters_m": LORA_EXTRA_PARAMETERS_M if method == "lora" else SMM_EXTRA_PARAMETERS_M,
    }
    if method == "lora":
        info.update(
            {
                "rank": DEFAULT_LORA_RANK,
                "alpha": DEFAULT_LORA_ALPHA,
                "target": "qkv",
                "lr": DEFAULT_LORA_LR,
                "epochs": DEFAULT_LORA_EPOCHS,
                "note": "Table 13: LoRA includes finetuning the fully connected layers",
            }
        )
    else:
        info.update(
            {
                "lr": DEFAULT_FINETUNE_LR,
                "epochs": DEFAULT_FINETUNE_EPOCHS,
                "milestones": list(DEFAULT_MILESTONES),
                "note": "Table 14: finetune the FC layer, with/without the SMM input module",
            }
        )
        if method == "finetune_fc_smm":
            info["smm_parameter_breakdown"] = smm_trainable_parameters(backbone)
    return info


def list_finetuning_methods() -> List[str]:
    """Return the implemented finetuning-based comparison methods."""

    return list(FINETUNE_METHODS)
