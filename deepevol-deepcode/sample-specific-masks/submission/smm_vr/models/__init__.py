"""SMM model zoo: frozen pre-trained classifiers ``f_P`` and the mask generator.

This package aggregates the two model-side components of SMM:

* :mod:`smm_vr.models.pretrained` - the frozen ImageNet-1K pre-trained
  classifiers (ResNet-18, ResNet-50, ViT-B/32, optional ViT-L/16) used as
  ``f_P`` in the paper (Sec. 5 "Pre-trained Models and Target Tasks",
  Appendix E.1).  All parameters are frozen (``requires_grad=False``) while the
  forward pass remains differentiable w.r.t. the reprogrammed input, so
  gradients still reach the shared pattern ``delta`` and the mask generator
  ``phi``.
* :mod:`smm_vr.models.mask_generator` - the lightweight CNN ``f_mask`` that
  produces a sample-specific 3-channel low-resolution mask (Sec. 3.2,
  Appendix A.2), with the exact parameter budgets from Table 4
  (26,499 params for the 5-layer variant, 102,339 for the 6-layer variant).

The module uses guarded ``try/except ImportError`` re-exports so that
``import smm_vr.models`` never fails during incremental development.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

__all__: list = []


def _extend(names) -> None:
    """Append ``names`` to ``__all__`` without creating duplicates."""
    for name in names:
        if name not in __all__:
            __all__append(name)


def __all__append(name: str) -> None:  # pragma: no cover - tiny helper
    globals()["__all__"].append(name)


# ---------------------------------------------------------------------------
# Frozen pre-trained classifiers (models/pretrained.py)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - guarded re-export
    from .pretrained import (  # noqa: F401
        BACKBONE_ALIASES,
        BACKBONE_SPECS,
        SUPPORTED_BACKBONES,
        BackboneSpec,
        PretrainedClassifier,
        build_classifier,
        build_pretrained_model,
        count_trainable_parameters,
        feature_layers_of,
        freeze_model,
        get_backbone_spec,
        input_size_for,
        load_pretrained,
        num_classes_of,
        resolve_backbone,
    )

    _PRETRAINED_AVAILABLE = True
    _extend(
        [
            "BACKBONE_ALIASES",
            "BACKBONE_SPECS",
            "SUPPORTED_BACKBONES",
            "BackboneSpec",
            "PretrainedClassifier",
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
    )
except ImportError:  # pragma: no cover
    _PRETRAINED_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lightweight CNN mask generator f_mask (models/mask_generator.py)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - guarded re-export
    from .mask_generator import (  # noqa: F401
        DEFAULT_PATCH_SIZE,
        EXPECTED_PARAMETERS,
        MASK_GENERATOR_CONFIGS,
        MaskGenerator,
        build_mask_generator,
        count_parameters,
        verify_parameters,
    )

    _MASK_GENERATOR_AVAILABLE = True
    _extend(
        [
            "DEFAULT_PATCH_SIZE",
            "EXPECTED_PARAMETERS",
            "MASK_GENERATOR_CONFIGS",
            "MaskGenerator",
            "build_mask_generator",
            "count_parameters",
            "verify_parameters",
        ]
    )
except ImportError:  # pragma: no cover
    _MASK_GENERATOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Backbone -> mask-generator depth convention (Sec. 3.2 / Appendix A.2)
# ---------------------------------------------------------------------------
#: Number of CNN layers in ``f_mask`` per backbone: 5 for the ResNets, 6 for
#: ViT-B/32 (the paper's Table 4 budgets 26,499 vs 102,339 parameters).
MASK_LAYERS_BY_BACKBONE: Dict[str, int] = {
    "resnet18": 5,
    "resnet50": 5,
    "resnet101": 5,
    "vit_b32": 6,
    "vit_b_32": 6,
    "vit_large": 6,
    "vit_l_16": 6,
    "vit_l32": 6,
}
_extend(["MASK_LAYERS_BY_BACKBONE"])


def resolve_backbone_name(backbone: Optional[str]) -> str:
    """Normalise a backbone spelling to a canonical registry key.

    Prefers :func:`smm_vr.models.pretrained.resolve_backbone` when available and
    falls back to a small local alias table otherwise.  Returns ``"resnet18"``
    for an unknown / empty name.
    """
    if not backbone:
        return "resnet18"
    try:
        return resolve_backbone(backbone)  # type: ignore[name-defined]
    except Exception:  # pragma: no cover - defensive
        pass

    key = str(backbone).strip().lower().replace("-", "_").replace(".", "_")
    aliases = {
        "resnet_18": "resnet18",
        "resnet_50": "resnet50",
        "resnet_101": "resnet101",
        "vit_b_32": "vit_b32",
        "vitb32": "vit_b32",
        "vit_base_patch32": "vit_b32",
        "vit_l_16": "vit_large",
        "vit_large_384": "vit_large",
    }
    key = aliases.get(key, key)
    if key in MASK_LAYERS_BY_BACKBONE:
        return key
    if "vit" in key:
        return "vit_b32" if "b" in key or "base" in key else "vit_large"
    if "resnet" in key:
        return "resnet18"
    return "resnet18"


def num_mask_layers_for(backbone: Optional[str]) -> int:
    """Return the mask-generator depth for a backbone (5 ResNets, 6 ViT-B32)."""
    return MASK_LAYERS_BY_BACKBONE.get(resolve_backbone_name(backbone), 5)


def input_size_for_backbone(backbone: Optional[str] = None, imgsize: Optional[int] = None) -> int:
    """Input resolution: 224 for the ResNets, 384 for ViT-B/32 (addendum)."""
    if imgsize is not None:
        return int(imgsize)
    try:
        return int(input_size_for(resolve_backbone_name(backbone), imgsize))  # type: ignore[name-defined]
    except Exception:  # pragma: no cover - defensive
        key = resolve_backbone_name(backbone)
        return 384 if key.startswith("vit") else 224


def build_models(
    backbone: str = "resnet18",
    *,
    input_size: Optional[int] = None,
    patch_size: int = 8,
    device=None,
    weights: str = "IMAGENET1K_V1",
    freeze: bool = True,
    pretrained: bool = True,
    verbose: bool = False,
    **mask_kwargs,
) -> Tuple["PretrainedClassifier", "MaskGenerator"]:  # noqa: F821
    """Build the frozen classifier ``f_P`` together with the mask generator.

    Convenience glue used by the experiment runners: resolves the backbone's
    input resolution (224 for ResNets, 384 for ViT-B/32) and mask-generator
    depth (5 vs 6 layers), then instantiates both components consistently.

    Returns
    -------
    (classifier, mask_generator)
        ``classifier`` is a :class:`PretrainedClassifier` in eval mode with all
        parameters frozen; ``mask_generator`` is a :class:`MaskGenerator`
        producing a low-resolution 3-channel mask.
    """
    key = resolve_backbone_name(backbone)
    size = input_size_for_backbone(key, input_size)

    classifier = build_classifier(  # type: ignore[name-defined]
        backbone=key,
        device=device,
        pretrained=pretrained,
        weights=weights,
        freeze=freeze,
        input_size=size,
        verbose=verbose,
    )
    mask_generator = build_mask_generator(  # type: ignore[name-defined]
        backbone=key,
        **mask_kwargs,
    )
    return classifier, mask_generator


def describe_backbone(backbone: str = "resnet18") -> Dict[str, object]:
    """Human-readable description of a backbone for logging/config dumps."""
    key = resolve_backbone_name(backbone)
    info: Dict[str, object] = {
        "backbone": key,
        "input_size": input_size_for_backbone(key),
        "num_mask_layers": num_mask_layers_for(key),
    }
    try:
        spec = get_backbone_spec(key)  # type: ignore[name-defined]
        info.update(spec.as_dict() if hasattr(spec, "as_dict") else dict(spec))
    except Exception:  # pragma: no cover - defensive
        pass
    expected = EXPECTED_PARAMETERS.get(key) if _MASK_GENERATOR_AVAILABLE else None  # type: ignore[name-defined]
    if expected is not None:
        info["expected_mask_parameters"] = int(expected)
    return info


_extend(["build_models", "describe_backbone", "input_size_for_backbone",
         "num_mask_layers_for", "resolve_backbone_name"])
