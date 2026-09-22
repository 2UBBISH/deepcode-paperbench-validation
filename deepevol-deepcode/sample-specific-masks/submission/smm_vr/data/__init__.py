"""Data utilities for SMM (Sample-specific Multi-channel Masks, ICML 2024).

This package exposes the target-task datasets used in the paper's main tables
(11 datasets, see Table 6), the exact input transforms transcribed from the
addendum, the split policies, and lightweight dataset statistics helpers.

Public API
----------
Transforms (``smm_vr.data.transforms``) ::

    get_image_size, build_transforms, build_train_transform, build_test_transform,
    train_transform, test_transform, inverse_normalize, transform_signature,
    IMAGENET_MEAN, IMAGENET_STD, IMAGENETNORMALIZE, DEFAULT_IMAGE_SIZE, VIT_IMAGE_SIZE

Datasets (``smm_vr.data.datasets``) ::

    build_dataloaders, build_datasets, build_train_dataset, build_test_dataset,
    get_dataset_spec, list_datasets, num_classes, dataset_image_size,
    resolve_dataset_root, targets_of, describe_dataset,
    DatasetSpec, DATASET_SPECS, MAIN_DATASETS, ALL_DATASETS, DEFAULT_BATCH_SIZES

Splits / stats (``smm_vr.data.splits``, ``smm_vr.data.dataset_stats``) are
re-exported as well, but the imports are guarded so that this package remains
importable while those optional modules are unavailable.
"""

from __future__ import annotations

from .transforms import (  # noqa: F401
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGENETNORMALIZE,
    RESIZE_MARGIN,
    VIT_IMAGE_SIZE,
    build_test_transform,
    build_train_transform,
    build_transforms,
    get_image_size,
    inverse_normalize,
    rgb_lambda,
    test_transform,
    train_transform,
    transform_signature,
)

from .datasets import (  # noqa: F401
    ALL_DATASETS,
    DATASET_ALIASES,
    DATASET_SPECS,
    DEFAULT_BATCH_SIZES,
    MAIN_DATASETS,
    ConcatDatasetWithTargets,
    DatasetSpec,
    SubsetWithTargets,
    build_dataloaders,
    build_datasets,
    build_test_dataset,
    build_train_dataset,
    canonical_name,
    dataset_image_size,
    describe_dataset,
    get_dataset_spec,
    list_datasets,
    num_classes,
    resolve_dataset_root,
    targets_of,
)

__all__ = [
    # transforms
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGENETNORMALIZE",
    "RESIZE_MARGIN",
    "VIT_IMAGE_SIZE",
    "build_test_transform",
    "build_train_transform",
    "build_transforms",
    "get_image_size",
    "inverse_normalize",
    "rgb_lambda",
    "test_transform",
    "train_transform",
    "transform_signature",
    # datasets
    "ALL_DATASETS",
    "DATASET_ALIASES",
    "DATASET_SPECS",
    "DEFAULT_BATCH_SIZES",
    "MAIN_DATASETS",
    "ConcatDatasetWithTargets",
    "DatasetSpec",
    "SubsetWithTargets",
    "build_dataloaders",
    "build_datasets",
    "build_test_dataset",
    "build_train_dataset",
    "canonical_name",
    "dataset_image_size",
    "describe_dataset",
    "get_dataset_spec",
    "list_datasets",
    "num_classes",
    "resolve_dataset_root",
    "targets_of",
]

# ---------------------------------------------------------------------------
# Optional modules: implemented on top of ``datasets.py``.
# Guarded so that ``import smm_vr.data`` never fails if they are absent.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional
    from .splits import (  # noqa: F401
        class_balanced_split,
        make_split,
        split_indices,
    )

    __all__ += ["class_balanced_split", "make_split", "split_indices"]
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - optional
    from .dataset_stats import (  # noqa: F401
        dataset_statistics,
        summarise_datasets,
    )

    __all__ += ["dataset_statistics", "summarise_datasets"]
except ImportError:  # pragma: no cover
    pass
