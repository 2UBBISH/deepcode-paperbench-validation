"""Frozen ImageNet-1K pre-trained backbones (ResNet-18/50, ViT-B/32).

Paper reference: Section 5 "Pre-trained Models and Target Tasks" -- ResNet-18,
ResNet-50 and ViT-B/32 pre-trained on ImageNet-1K are re-purposed; every
parameter of the backbone stays frozen, only the input space is trained.

The pre-trained models are obtained from ``torchvision`` (weights
``IMAGENET1K_V1``).  ViT-B/32 is pre-trained at 224x224 but the paper feeds it
384x384 (Table 4, and the addendum transform code), so the positional embedding
is resized to the 12x12 patch grid with bicubic interpolation, the usual
practice for evaluating ViT at higher resolution.  If ``timm`` is installed the
native 384x384 checkpoint ``vit_base_patch32_384`` can be selected instead
(``arch="timm_384"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BackboneBundle",
    "build_backbone",
    "freeze_backbone",
    "resize_position_embeddings",
    "IMAGE_SIZE",
]

# Input resolution used by the paper for every pre-trained backbone.
IMAGE_SIZE = {"resnet18": 224, "resnet50": 224, "vit_b32": 384, "vit_b_32": 384, "vitb32": 384}


@dataclass
class BackboneBundle:
    """A frozen pre-trained classifier plus metadata used by the trainer."""

    name: str
    model: nn.Module
    num_pretrained_classes: int
    image_size: int
    feature_dim: int
    feature_kind: str  # "avgpool" (ResNet) or "cls_token" (ViT)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Output-layer feature of the pre-trained model (for t-SNE, Figure 6)."""
        feats = {}

        def hook(_module, _inp, out):
            feats["f"] = out

        handle = None
        try:
            if self.feature_kind == "avgpool":
                handle = self.model.avgpool.register_forward_hook(hook)
            else:  # ViT: class token before the classification head
                handle = self.model.encoder.register_forward_hook(hook)
            self.model.eval()
            with torch.no_grad():
                self.model(x)
        finally:
            if handle is not None:
                handle.remove()
        f = feats["f"]
        if self.feature_kind == "avgpool":
            return f.flatten(1)
        return f[:, 0]  # class token


def resize_position_embeddings(model: nn.Module, image_size: int, mode: str = "bicubic") -> nn.Module:
    """Resize ``model.encoder.pos_embedding`` to a new resolution (timm-style)."""
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "pos_embedding"):
        raise ValueError("the model does not expose encoder.pos_embedding")
    patch_size = int(getattr(model, "patch_size", 32))
    num_prefix = 1  # class token
    grid_new = image_size // patch_size

    pos = model.encoder.pos_embedding.data  # (1, 1 + g*g, dim)
    ntok = pos.shape[1] - num_prefix
    grid_old = int(math.sqrt(ntok))
    if grid_old * grid_old != ntok:  # pragma: no cover - defensive
        raise ValueError("unexpected number of patch tokens in the positional embedding")
    if grid_old == grid_new:
        return model

    prefix = pos[:, :num_prefix]
    tokens = pos[:, num_prefix:]
    tokens = tokens.reshape(1, grid_old, grid_old, -1).permute(0, 3, 1, 2)
    tokens = F.interpolate(tokens, size=(grid_new, grid_new), mode=mode, align_corners=False)
    tokens = tokens.permute(0, 2, 3, 1).reshape(1, grid_new * grid_new, -1)
    model.encoder.pos_embedding = nn.Parameter(torch.cat([prefix, tokens], dim=1))
    model.image_size = image_size
    return model


def _torchvision_resnet(name: str, pretrained: bool, image_size: int = 224) -> BackboneBundle:
    import torchvision

    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        feature_dim = 512
    elif name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        feature_dim = 2048
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported ResNet {name!r}")
    # ResNets are fully convolutional, so any input resolution works; the paper
    # uses 224x224.
    return BackboneBundle(name, model, 1000, int(image_size), feature_dim, "avgpool")


def _torchvision_vit(image_size: int, pretrained: bool, arch: str) -> BackboneBundle:
    import torchvision

    if arch == "timm_384":
        try:
            import timm  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "arch='timm_384' needs `timm`; use arch='torchvision' instead"
            ) from exc
        model = timm.create_model("vit_base_patch32_384", pretrained=pretrained, num_classes=1000)
        model.eval()
        return BackboneBundle("vit_b32", model, 1000, image_size, model.embed_dim, "cls_token")

    weights = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.vit_b_32(weights=weights)
    if image_size != 224:
        resize_position_embeddings(model, image_size)
    model.eval()
    return BackboneBundle("vit_b32", model, 1000, image_size, model.hidden_dim, "cls_token")


def build_backbone(
    name: str,
    pretrained: bool = True,
    image_size: Optional[int] = None,
    arch: str = "torchvision",
) -> BackboneBundle:
    """Build a frozen ImageNet-1K pre-trained classifier.

    Parameters
    ----------
    name:
        ``"resnet18"``, ``"resnet50"`` or ``"vit_b32"``.
    pretrained:
        Load the ImageNet-1K weights (downloaded on first use).
    image_size:
        Overrides the default input resolution (224 for ResNets, 384 for ViT).
    arch:
        ``"torchvision"`` (default) or ``"timm_384"`` (ViT only).
    """

    key = name.lower().replace("-", "").replace("_", "")
    if key in {"resnet18", "resnet50", "vitb32"}:
        default_size = IMAGE_SIZE[key]
    else:
        raise ValueError(f"unknown backbone {name!r}")
    size = int(image_size or default_size)

    if key == "vitb32":
        bundle = _torchvision_vit(size, pretrained, arch)
    else:
        bundle = _torchvision_resnet(key, pretrained, size)
    return freeze_backbone(bundle)


def freeze_backbone(bundle: BackboneBundle) -> BackboneBundle:
    """Freeze every parameter of the pre-trained model (VR never edits it)."""
    for p in bundle.model.parameters():
        p.requires_grad_(False)
    bundle.model.eval()
    return bundle
