"""Frozen ImageNet-1K pre-trained classifiers used as ``f_P`` in SMM.

Paper references
----------------
* §5 "Pre-trained Models and Target Tasks": *"Following Chen et al. (2023), we use
  ResNet-18, and ResNet-50 (He et al., 2016) as the pre-trained model. Performance on
  pre-trained ViT-B32 (Dosovitskiy et al., 2020) is also tested. All these models are
  pre-trained on ImageNet-1K ... "*
* §5: experiments run on a single A100 GPU with three seeds.
* Appendix E.1: *"ViT-Large with the input size being 384x384 is applied"* for the
  finetuning (LoRA) vs. SMM comparison (Table 13).

Design notes (paper ambiguities)
--------------------------------
The paper states the models are pre-trained on ImageNet-1K but does not name the exact
torchvision weight tag.  We default to ``IMAGENET1K_V1`` (the canonical ImageNet-1K
checkpoint) and make it configurable.

All parameters are frozen with ``requires_grad_(False)`` (``model.freeze`` in the
configs) because SMM only learns the input-space parameters ``delta`` (shared pattern)
and ``phi`` (mask generator); gradients still flow *through* the frozen network to those
input-space tensors, which requires the forward pass to stay differentiable.

Input resolutions follow the addendum transforms: ``imgsize = 384`` for ViT_B32 (and
ViT-Large with 384x384 input) else ``224`` for the ResNets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

try:  # torchvision >= 0.13 exposes weight enums; older versions expose plain functions
    from torchvision.models import (
        ResNet18_Weights,
        ResNet50_Weights,
        ViT_B_32_Weights,
        ViT_L_16_Weights,
    )

    _WEIGHT_ENUMS = {
        "resnet18": ResNet18_Weights,
        "resnet50": ResNet50_Weights,
        "vit_b_32": ViT_B_32_Weights,
        "vit_b32": ViT_B_32_Weights,
        "vit_l_16": ViT_L_16_Weights,
        "vit_large": ViT_L_16_Weights,
    }
except ImportError:  # pragma: no cover - very old torchvision
    _WEIGHT_ENUMS = {}

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "BACKBONE_SPECS",
    "BackboneSpec",
    "PretrainedClassifier",
    "SUPPORTED_BACKBONES",
    "build_classifier",
    "build_pretrained_model",
    "count_trainable_parameters",
    "feature_layers_of",
    "freeze_model",
    "get_backbone_spec",
    "input_size_for",
    "load_pretrained",
    "num_classes_of",
    "resolve_backbone",
]


# --------------------------------------------------------------------------------------
# Backbone registry
# --------------------------------------------------------------------------------------
class BackboneSpec:
    """Static description of a pre-trained backbone used as ``f_P``."""

    __slots__ = (
        "name",
        "torchvision_name",
        "input_size",
        "num_classes",
        "weights",
        "feature_layer",
        "display_name",
        "extra_parameters_millions",
    )

    def __init__(
        self,
        name: str,
        torchvision_name: str,
        input_size: int,
        num_classes: int = 1000,
        weights: str = "IMAGENET1K_V1",
        feature_layer: str = "avgpool",
        display_name: str = "",
        extra_parameters_millions: Optional[float] = None,
    ) -> None:
        self.name = name
        self.torchvision_name = torchvision_name
        # ImageNet-1K pre-training uses 224x224 for ResNets; ViT-B/32 is pre-trained at
        # 224 but the paper reprograms it at 384x384 (§5 / addendum transforms).
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = weights
        self.feature_layer = feature_layer
        self.display_name = display_name or name
        self.extra_parameters_millions = extra_parameters_millions

    # -- convenience ---------------------------------------------------------------
    @property
    def is_vit(self) -> bool:
        return "vit" in self.name.lower()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "torchvision_name": self.torchvision_name,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "weights": self.weights,
            "feature_layer": self.feature_layer,
            "display_name": self.display_name,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"BackboneSpec(name={self.name!r}, torchvision_name={self.torchvision_name!r}, "
            f"input_size={self.input_size}, num_classes={self.num_classes}, "
            f"weights={self.weights!r})"
        )


#: Registry of backbones referenced by the paper. ``input_size`` mirrors the addendum
#: transforms (384 for ViT, 224 for ResNets).
BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        "resnet18", "resnet18", input_size=224, feature_layer="avgpool",
        display_name="ResNet-18 (ImageNet-1K)",
    ),
    "resnet50": BackboneSpec(
        "resnet50", "resnet50", input_size=224, feature_layer="avgpool",
        display_name="ResNet-50 (ImageNet-1K)",
    ),
    "vit_b32": BackboneSpec(
        "vit_b32", "vit_b_32", input_size=384, feature_layer="heads.head",
        display_name="ViT-B/32 (ImageNet-1K)",
    ),
    # Appendix E.1: ViT-Large at 384x384 with LoRA (rank 6, 0.60M extra params).
    "vit_large": BackboneSpec(
        "vit_large", "vit_l_16", input_size=384, feature_layer="heads.head",
        display_name="ViT-L/16 (ImageNet-1K, 384x384)",
        extra_parameters_millions=0.60,
    ),
}

#: Aliases tolerated by :func:`resolve_backbone`.
BACKBONE_ALIASES: Dict[str, str] = {
    "resnet-18": "resnet18",
    "resnet_18": "resnet18",
    "r18": "resnet18",
    "resnet-50": "resnet50",
    "resnet_50": "resnet50",
    "r50": "resnet50",
    "vit": "vit_b32",
    "vit-b32": "vit_b32",
    "vit_b_32": "vit_b32",
    "vitb32": "vit_b32",
    "vit_b/32": "vit_b32",
    "vit-b/32": "vit_b32",
    "vit-large": "vit_large",
    "vit_l_16": "vit_large",
    "vitl": "vit_large",
    "vit_l32": "vit_large",
}

SUPPORTED_BACKBONES: Tuple[str, ...] = tuple(BACKBONE_SPECS.keys())


def resolve_backbone(name: Optional[str]) -> str:
    """Normalise a backbone spelling to a registry key (defaults to ``resnet18``)."""
    if name is None:
        return "resnet18"
    key = str(name).strip().lower()
    if key in BACKBONE_SPECS:
        return key
    if key in BACKBONE_ALIASES:
        return BACKBONE_ALIASES[key]
    compact = key.replace("-", "").replace("_", "").replace("/", "").replace(".", "")
    for canonical in BACKBONE_SPECS:
        if compact == canonical.replace("-", "").replace("_", ""):
            return canonical
        if compact in ("vitb32", "vitbase32") and canonical == "vit_b32":
            return canonical
        if compact in ("vitlarge16", "vitlarge") and canonical == "vit_large":
            return canonical
    raise ValueError(
        f"Unknown backbone {name!r}; supported: {list(SUPPORTED_BACKBONES)}"
    )


def get_backbone_spec(name: Optional[str] = "resnet18", input_size: Optional[int] = None) -> BackboneSpec:
    """Return the :class:`BackboneSpec` for ``name`` (``input_size`` may override)."""
    spec = BACKBONE_SPECS[resolve_backbone(name)]
    if input_size is not None and int(input_size) != spec.input_size:
        return BackboneSpec(
            spec.name,
            spec.torchvision_name,
            input_size=int(input_size),
            num_classes=spec.num_classes,
            weights=spec.weights,
            feature_layer=spec.feature_layer,
            display_name=spec.display_name,
            extra_parameters_millions=spec.extra_parameters_millions,
        )
    return spec


def input_size_for(name: Optional[str] = "resnet18", imgsize: Optional[int] = None) -> int:
    """Input resolution for ``name`` (224 ResNets / 384 ViT-B32 & ViT-Large)."""
    if imgsize is not None:
        return int(imgsize)
    return BACKBONE_SPECS[resolve_backbone(name)].input_size


def feature_layers_of(name: Optional[str] = "resnet18") -> str:
    """Name of the module whose output is used for t-SNE feature extraction."""
    return BACKBONE_SPECS[resolve_backbone(name)].feature_layer


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------
def _weight_enum(torchvision_name: str, weights: Union[str, None]):
    """Resolve a torchvision weight enum for ``torchvision_name``.

    ``weights=None`` means "randomly initialised" (used for tests / ablation sanity
    checks); ``weights="DEFAULT"`` uses the torchvision default (ImageNet-1K).
    """
    if weights is None:
        return None
    key = torchvision_name.lower().replace("-", "_")
    enum_cls = _WEIGHT_ENUMS.get(key)
    if enum_cls is None:
        return None
    tag = str(weights).upper()
    if tag in ("DEFAULT", "AUTO", "TRUE", "IMAGENET"):
        # torchvision default == best available ImageNet-1K checkpoint.
        return enum_cls.DEFAULT
    if hasattr(enum_cls, tag):
        return getattr(enum_cls, tag)
    # Fall back to the versioned ImageNet-1K weight if the tag is unknown.
    for candidate in ("IMAGENET1K_V1", "IMAGENET1K_V2", "DEFAULT"):
        if hasattr(enum_cls, candidate):
            return getattr(enum_cls, candidate)
    return enum_cls.DEFAULT


def build_pretrained_model(
    backbone: str = "resnet18",
    *,
    weights: Union[str, None] = "IMAGENET1K_V1",
    num_classes: Optional[int] = None,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate the torchvision model named by ``backbone``.

    Parameters
    ----------
    backbone:
        Registry name or alias (``resnet18``, ``resnet50``, ``vit_b32``, ``vit_large``).
    weights:
        torchvision weight tag (``IMAGENET1K_V1``); ``None`` disables pre-trained weights.
    num_classes:
        Optional replacement classifier width.  The paper always keeps the full
        ImageNet-1K label space (1000 outputs) because the label mappings select a
        subset of ImageNet labels; this argument exists only for debugging.
    pretrained:
        Convenience flag: when ``False`` the model is randomly initialised.
    """
    spec = BACKBONE_SPECS[resolve_backbone(backbone)]
    torchvision_name = spec.torchvision_name
    weight_enum = _weight_enum(torchvision_name, weights if pretrained else None)

    factory_name = {
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "vit_b_32": "vit_b_32",
        "vit_l_16": "vit_l_16",
    }.get(torchvision_name, torchvision_name)
    factory = getattr(torchvision, factory_name)

    kwargs: Dict[str, Any] = {}
    if num_classes is not None:
        kwargs["num_classes"] = int(num_classes)
    model = factory(weights=weight_enum, **kwargs) if weight_enum is not None else factory(**kwargs)
    return model


def freeze_model(model: nn.Module) -> nn.Module:
    """Set ``requires_grad=False`` on every parameter and switch to eval mode.

    The model stays differentiable (gradients propagate *through* it to the
    reprogramming tensors), but no parameter of ``f_P`` is ever updated.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of parameters with ``requires_grad=True`` (0 for a frozen ``f_P``)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class PretrainedClassifier(nn.Module):
    """Frozen ImageNet-1K classifier wrapper exposing logits and metadata.

    Parameters
    ----------
    backbone:
        Registry name/alias (``resnet18`` | ``resnet50`` | ``vit_b32`` | ``vit_large``).
    weights:
        torchvision weight tag, default ``IMAGENET1K_V1`` (ImageNet-1K).
    input_size:
        Overrides the default resolution (224 ResNets / 384 ViT-B32).
    freeze:
        When ``True`` (paper setting) all parameters get ``requires_grad_(False)``.
    pretrained:
        Whether to load ImageNet-1K weights.
    num_classes:
        Optional head width override (default: full ImageNet-1K label space, 1000).
    feature_layer:
        Module name used by :meth:`forward_features` (t-SNE analysis).

    Notes
    -----
    ``forward`` returns logits over the *ImageNet* label space; the target-task output
    mapping ``f_out`` (Rlm/Flm/Ilm) is applied on top of these logits.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        *,
        weights: Union[str, None] = "IMAGENET1K_V1",
        input_size: Optional[int] = None,
        freeze: bool = True,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        feature_layer: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.spec = get_backbone_spec(backbone, input_size=input_size)
        self.backbone = self.spec.name
        self.input_size = self.spec.input_size
        self.num_pretrained_classes = int(
            num_classes if num_classes is not None else self.spec.num_classes
        )
        self.weights = weights if pretrained else None
        self.freeze = bool(freeze)
        self.feature_layer = feature_layer or self.spec.feature_layer

        if model is None:
            model = build_pretrained_model(
                self.backbone,
                weights=weights,
                num_classes=num_classes,
                pretrained=pretrained,
            )
        self.model = model

        if freeze:
            freeze_model(self.model)
        if device is not None:
            self.to(device)

        if verbose:  # pragma: no cover - informational
            total = sum(p.numel() for p in self.model.parameters())
            print(
                f"[PretrainedClassifier] {self.spec.display_name}: "
                f"{total/1e6:.2f}M params, {count_trainable_parameters(self.model)} trainable, "
                f"input {self.input_size}x{self.input_size}, "
                f"{self.num_pretrained_classes} classes"
            )

    # -- forward ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ImageNet logits ``f_P(x)`` for a (reprogrammed) batch ``(B,3,H,W)``.

        The wrapper forces ``eval()`` behaviour (frozen ``f_P``); gradients are *not*
        disabled so they keep flowing to the input-space SMM parameters.
        """
        was_training = self.model.training
        if was_training:
            self.model.eval()
        logits = self.model(x)
        if was_training:
            self.model.train()
        return logits

    # -- analysis helpers ---------------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract output-layer features *before* the label mapping (t-SNE, §5)."""
        return self.forward_features(x)

    def _unwrap(self, x: torch.Tensor) -> torch.Tensor:
        """torchvision ViT returns ``(logits, tokens)``; keep only the logits."""
        if isinstance(x, (tuple, list)):
            return x[0]
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate/output-layer features (used by ``analysis/tsne_features.py``)."""
        model = self.model
        name = self.feature_layer

        if name and name != "logits":
            module = model
            try:
                for part in name.split("."):
                    module = getattr(module, part)
            except AttributeError:
                module = None
            if module is not None:
                captured: List[torch.Tensor] = []

                def _hook(_mod, _inp, out):  # pragma: no cover - hook
                    captured.append(self._unwrap(out))

                handle = module.register_forward_hook(_hook)
                try:
                    with torch.no_grad():
                        self.model(x)
                finally:
                    handle.remove()
                if captured:
                    return _flatten_features(captured[-1])

        # ResNet-style / fallback: pooled features then logits.
        if hasattr(model, "avgpool") and hasattr(model, "fc"):
            z = model.conv1(x)
            z = model.bn1(z)
            z = model.relu(z)
            z = model.maxpool(z)
            z = model.layer1(z)
            z = model.layer2(z)
            z = model.layer3(z)
            z = model.layer4(z)
            z = model.avgpool(z)
            return _flatten_features(z)

        if hasattr(model, "forward_features"):
            try:
                feats = model.forward_features(x)
            except TypeError:  # pragma: no cover - some ViT variants
                feats = model.forward_features(x, None)
            return _flatten_features(self._unwrap(feats))

        return _flatten_features(self._unwrap(self.forward(x)))

    # -- metadata -----------------------------------------------------------------
    @property
    def label_space_size(self) -> int:
        """Size of the pre-trained ImageNet label space ``|Y^P|``."""
        return self.num_pretrained_classes

    @property
    def num_classes(self) -> int:
        return self.num_pretrained_classes

    @property
    def out_features(self) -> int:
        """Alias used by ``label_mapping.frequency`` for inference."""
        return self.num_pretrained_classes

    def parameters_are_frozen(self) -> bool:
        return count_trainable_parameters(self.model) == 0

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return (
            f"backbone={self.backbone!r}, input_size={self.input_size}, "
            f"weights={self.weights!r}, freeze={self.freeze}, "
            f"num_pretrained_classes={self.num_pretrained_classes}"
        )


def _flatten_features(feats: torch.Tensor) -> torch.Tensor:
    """Flatten ``(B, ...)`` features to ``(B, D)`` for t-SNE / linear probes."""
    if feats.dim() == 1:
        return feats.unsqueeze(0)
    if feats.dim() > 2:
        return feats.flatten(1)
    return feats


def num_classes_of(model: Union[nn.Module, PretrainedClassifier]) -> int:
    """Best-effort ImageNet label-space size of a (possibly bare) classifier."""
    if isinstance(model, PretrainedClassifier):
        return model.num_pretrained_classes
    if hasattr(model, "out_features"):
        return int(model.out_features)  # type: ignore[arg-type]
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "out_features"):
        return int(head.out_features)
    heads = getattr(model, "heads", None)
    if heads is not None and hasattr(heads, "head") and hasattr(heads.head, "out_features"):
        return int(heads.head.out_features)
    fc = getattr(model, "fc", None)
    if fc is not None and hasattr(fc, "out_features"):
        return int(fc.out_features)
    raise ValueError(f"Cannot infer label-space size from {type(model)!r}")


def load_pretrained(
    backbone: str = "resnet18",
    *,
    weights: Union[str, None] = "IMAGENET1K_V1",
    pretrained: bool = True,
    freeze: bool = True,
    input_size: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> PretrainedClassifier:
    """Load a frozen ImageNet-1K classifier ready to be used as ``f_P``.

    Mirrors the config keys ``model.backbone`` / ``model.weights`` / ``model.freeze``
    / ``model.input_size``.
    """
    return PretrainedClassifier(
        backbone,
        weights=weights,
        pretrained=pretrained,
        freeze=freeze,
        input_size=input_size,
        device=device,
        verbose=verbose,
        **kwargs,
    )


def build_classifier(
    backbone: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> PretrainedClassifier:
    """Config-dict aware factory for ``f_P``.

    Accepts either a ``model`` block (``{"backbone": ..., "weights": ...}``) or a full
    merged config (``{"model": {...}}``), matching the layered YAML files.  Explicit
    keyword arguments win over config values so experiment runners can override.
    """
    cfg: Dict[str, Any] = {}
    if config:
        cfg = dict(config.get("model", config) or {})
    name = backbone or cfg.get("backbone") or cfg.get("name") or "resnet18"
    merged = {
        "weights": cfg.get("weights", "IMAGENET1K_V1"),
        "pretrained": cfg.get("pretrained", True),
        "freeze": cfg.get("freeze", True),
        "input_size": cfg.get("input_size"),
    }
    merged.update(kwargs)
    if merged.get("input_size") is None:
        merged.pop("input_size", None)
    return load_pretrained(name, device=device, **merged)
