"""ImageNet data pipeline for the coupled stochastic-interpolant experiments.

This package provides the dataset / dataloader plumbing used by the two tasks in
*Stochastic Interpolants with Data-Dependent Couplings*:

* in-painting (Section 4.1) -- ImageNet images at 256x256 / 512x512, plus the
  missingness mask ``xi`` built by :mod:`si.couplings.inpainting`;
* super-resolution (Section 4.2) -- ImageNet images at 256x256 / 512x512 together
  with the low-resolution view ``D(x1)`` (64x64 / 256x256).

The public surface re-exports the HuggingFace ImageNet loader
(:mod:`si.data.imagenet`) and the preprocessing transforms
(:mod:`si.data.transforms`), both of which keep tensors in the repository-wide
``[-1, 1]`` convention.

The heavy sub-modules are resolved lazily through a module-level ``__getattr__``
(PEP 562) so that ``import si.data`` stays cheap and does not immediately pull in
``torch``/``datasets``/``numpy``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__: List[str] = [
    # ---- imagenet.py -------------------------------------------------------
    "ImageNetDataset",
    "ImageNetIterableDataset",
    "SyntheticImageNetDataset",
    "CollateFn",
    "load_imagenet_hf",
    "imagenet_split_name",
    "default_collate",
    "build_dataset",
    "build_dataloader",
    "build_dataloaders",
    "IMAGENET_NUM_CLASSES",
    "IMAGENET_HF_NAME",
    "PAPER_RESOLUTIONS",
    "PAPER_LOW_RESOLUTIONS",
    "DATASETS_AVAILABLE",
    # ---- transforms.py -----------------------------------------------------
    "TransformConfig",
    "ImageNetTransforms",
    "to_tensor",
    "to_neg_one_one",
    "to_zero_one",
    "normalize",
    "denormalize",
    "to_display",
    "from_display",
    "image_range",
    "clamp_unit",
    "resize",
    "resize_shorter_side",
    "center_crop",
    "random_crop",
    "random_hflip",
    "pad_to_square",
    "imagenet_transform",
    "make_transform",
    "get_transforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGENET_MEAN_TENSOR",
    "IMAGENET_STD_TENSOR",
]

# ---------------------------------------------------------------------------
# name -> owning sub-module routing table
# ---------------------------------------------------------------------------
_IMAGENET_NAMES = frozenset(
    {
        "ImageNetDataset",
        "ImageNetIterableDataset",
        "SyntheticImageNetDataset",
        "CollateFn",
        "load_imagenet_hf",
        "imagenet_split_name",
        "default_collate",
        "build_dataset",
        "build_dataloader",
        "build_dataloaders",
        "IMAGENET_NUM_CLASSES",
        "IMAGENET_HF_NAME",
        "DEFAULT_HF_CACHE",
        "PAPER_RESOLUTIONS",
        "PAPER_LOW_RESOLUTIONS",
        "IMAGE_KEYS",
        "LABEL_KEYS",
        "DATASETS_AVAILABLE",
    }
)

_TRANSFORM_NAMES = frozenset(
    {
        "TransformConfig",
        "ImageNetTransforms",
        "to_tensor",
        "to_neg_one_one",
        "to_zero_one",
        "normalize",
        "denormalize",
        "to_display",
        "from_display",
        "image_range",
        "clamp_unit",
        "normalize_size",
        "resize",
        "resize_shorter_side",
        "center_crop",
        "random_crop",
        "random_hflip",
        "pad_to_square",
        "imagenet_transform",
        "make_transform",
        "get_transforms",
        "IMAGENET_MEAN",
        "IMAGENET_STD",
        "IMAGENET_MEAN_TENSOR",
        "IMAGENET_STD_TENSOR",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily resolve symbols from the sibling data sub-modules."""
    if name in _IMAGENET_NAMES:
        from . import imagenet as _imagenet

        try:
            return getattr(_imagenet, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module 'si.data.imagenet' has no attribute {name!r}"
            ) from exc

    if name in _TRANSFORM_NAMES:
        from . import transforms as _transforms

        try:
            return getattr(_transforms, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module 'si.data.transforms' has no attribute {name!r}"
            ) from exc

    raise AttributeError(f"module 'si.data' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


def describe() -> Dict[str, Optional[str]]:
    """Short descriptions of the modules shipped by this package."""
    return {
        "imagenet": "HuggingFace ImageNet-1k loader, synthetic fallback, collate + dataloaders",
        "transforms": "[-1, 1] image conversion, resize/crop, ImageNet preprocessing, SR low-res views",
    }
