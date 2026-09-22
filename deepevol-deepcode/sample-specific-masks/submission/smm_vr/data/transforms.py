"""Exact image transforms used by SMM, transcribed from the paper's addendum.

The paper (Appendix C / addendum) specifies the *same* pipeline for every target
dataset; only the resolution depends on the frozen pre-trained backbone:

* ``ViT_B32`` -> ``imgsize = 384``
* everything else (ResNet-18 / ResNet-50, optionally ViT-Large) -> ``imgsize = 224``

Addendum code (verbatim behaviour)::

    IMAGENETNORMALIZE = {
        'mean': [0.485, 0.456, 0.406],
        'std':  [0.229, 0.224, 0.225],
    }

    train_preprocess = transforms.Compose([
        transforms.Resize((imgsize + 32, imgsize + 32)),
        transforms.RandomCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])

Important details preserved here:

* ``Resize`` receives a 2-tuple, so aspect ratio is not preserved (this is what
  the paper's snippet does).
* ``RandomHorizontalFlip()`` keeps its default probability ``p = 0.5``.
* The RGB conversion happens *after* geometric augmentation and *before*
  ``ToTensor``, exactly as in the snippet, because several target datasets
  (SVHN, GTSRB, ...) ship grayscale or palette images.
* ImageNet statistics are used for every dataset, including the 32x32 ones.

Helpers used by ``data/datasets.py`` are exported as well:
``get_image_size``, ``build_train_transform``, ``build_test_transform``,
``build_transforms`` and ``inverse_normalize`` (visualisation only).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torchvision import transforms

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGENETNORMALIZE",
    "RESIZE_MARGIN",
    "VIT_IMAGE_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "IMAGE_SIZE_BY_BACKBONE",
    "get_image_size",
    "rgb_lambda",
    "train_transform",
    "test_transform",
    "build_train_transform",
    "build_test_transform",
    "build_transforms",
    "inverse_normalize",
    "transform_signature",
]


# ---------------------------------------------------------------------------
# Constants (verbatim from the addendum)
# ---------------------------------------------------------------------------
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
IMAGENETNORMALIZE: Dict[str, List[float]] = {
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
}

#: ``imgsize + 32`` resize before the random crop (addendum ``Resize((imgsize+32, ...))``).
RESIZE_MARGIN: int = 32

#: Input resolution of the pre-trained ViT-B32 classifier (§5, addendum).
VIT_IMAGE_SIZE: int = 384
#: Input resolution of the pre-trained ResNet-18 / ResNet-50 classifiers.
DEFAULT_IMAGE_SIZE: int = 224

#: Backbone -> input resolution.  ViT-Large in Appendix E.1 also uses 384.
IMAGE_SIZE_BY_BACKBONE: Dict[str, int] = {
    "resnet18": DEFAULT_IMAGE_SIZE,
    "resnet50": DEFAULT_IMAGE_SIZE,
    "resnet101": DEFAULT_IMAGE_SIZE,
    "vit_b32": VIT_IMAGE_SIZE,
    "vit_b_32": VIT_IMAGE_SIZE,
    "vit-large": VIT_IMAGE_SIZE,
    "vit_large": VIT_IMAGE_SIZE,
    "vit_l_16": VIT_IMAGE_SIZE,
    "vit_l32": VIT_IMAGE_SIZE,
}


def get_image_size(model: Optional[str] = None, imgsize: Optional[int] = None) -> int:
    """Return the reprogrammed input size for a pre-trained model name.

    Mirrors the addendum branch ``imgsize = 384 if model == "ViT_B32" else 224``
    while tolerating common capitalisation variants.

    Args:
        model: backbone identifier, e.g. ``"ResNet18"`` or ``"ViT_B32"``.
        imgsize: explicit override; when given it wins over ``model``.

    Returns:
        The square input resolution in pixels.
    """
    if imgsize is not None:
        return int(imgsize)
    if model is None:
        return DEFAULT_IMAGE_SIZE
    key = str(model).strip().lower().replace("-", "_")
    if key in IMAGE_SIZE_BY_BACKBONE:
        return IMAGE_SIZE_BY_BACKBONE[key]
    # Fallback: every ViT in this paper uses 384, everything else 224.
    return VIT_IMAGE_SIZE if "vit" in key else DEFAULT_IMAGE_SIZE


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def rgb_lambda(x):
    """``lambda x: x.convert('RGB') if hasattr(x, 'convert') else x`` (verbatim)."""
    return x.convert("RGB") if hasattr(x, "convert") else x


def train_transform(imgsize: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Training transform, verbatim from the addendum.

    ``Resize((imgsize+32, imgsize+32))`` -> ``RandomCrop(imgsize)`` ->
    ``RandomHorizontalFlip()`` -> RGB -> ``ToTensor`` -> ImageNet ``Normalize``.
    """
    return transforms.Compose(
        [
            transforms.Resize((imgsize + RESIZE_MARGIN, imgsize + RESIZE_MARGIN)),
            transforms.RandomCrop(imgsize),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(rgb_lambda),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]),
        ]
    )


def test_transform(imgsize: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Test transform, verbatim from the addendum.

    ``Resize((imgsize, imgsize))`` -> RGB -> ``ToTensor`` -> ImageNet ``Normalize``.
    """
    return transforms.Compose(
        [
            transforms.Resize((imgsize, imgsize)),
            transforms.Lambda(rgb_lambda),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]),
        ]
    )


def build_train_transform(
    model: Optional[str] = None,
    imgsize: Optional[int] = None,
) -> transforms.Compose:
    """Training transform for the given backbone (or explicit resolution)."""
    return train_transform(get_image_size(model, imgsize))


def build_test_transform(
    model: Optional[str] = None,
    imgsize: Optional[int] = None,
) -> transforms.Compose:
    """Test transform for the given backbone (or explicit resolution)."""
    return test_transform(get_image_size(model, imgsize))


def build_transforms(
    model: Optional[str] = None,
    imgsize: Optional[int] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_preprocess, test_preprocess)`` exactly as in the addendum."""
    size = get_image_size(model, imgsize)
    return train_transform(size), test_transform(size)


def inverse_normalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Undo ImageNet normalisation (used for figure / feature visualisation only).

    Accepts ``(C, H, W)`` or ``(B, C, H, W)`` tensors and returns a detached
    tensor clipped to ``[0, 1]``.
    """
    if tensor.dim() not in (3, 4):
        raise ValueError(f"expected a 3D or 4D tensor, got shape {tuple(tensor.shape)}")
    m = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor.detach() * s + m).clamp_(0.0, 1.0)


def transform_signature(
    model: Optional[str] = None, imgsize: Optional[int] = None
) -> Dict[str, object]:
    """Human-readable description of the transforms (logging / config dumps)."""
    size = get_image_size(model, imgsize)
    return {
        "model": model,
        "imgsize": size,
        "train": [
            f"Resize(({size + RESIZE_MARGIN}, {size + RESIZE_MARGIN}))",
            f"RandomCrop({size})",
            "RandomHorizontalFlip()",
            "convert('RGB')",
            "ToTensor()",
            "Normalize(imagenet)",
        ],
        "test": [
            f"Resize(({size}, {size}))",
            "convert('RGB')",
            "ToTensor()",
            "Normalize(imagenet)",
        ],
        "imagenet_normalize": IMAGENETNORMALIZE,
    }
