"""Data layer for the LBCS reproduction.

This package bundles everything needed to build the benchmarks used by
Refined Coreset Selection (RCS) / Lexicographic Bilevel Coreset Selection
(LBCS):

* :mod:`lbcs_repro.data.datasets` -- dataset registry (MNIST, F-MNIST,
  SVHN, CIFAR-10) with the paper's per-benchmark normalization, deterministic
  DataLoader factories, index-aware subsetting for coreset bookkeeping and
  label statistics helpers.
* :mod:`lbcs_repro.data.mnist_s` -- MNIST-S, the 1,000-example random MNIST
  training subset used for Section 5.1 / Figure 1 (evaluation on the full
  10,000-example MNIST test split).
* :mod:`lbcs_repro.data.robustness` -- imperfect-supervision transforms for
  Section 5.3 / Appendix E.2: symmetric label noise (30% / 50%) and
  exponential class imbalance with ratio 0.01.  Both act on the training set
  only -- the test set is never modified.

All datasets are loaded through ``torchvision`` (no Kaggle, no credentials);
a deterministic synthetic fallback is available for offline smoke tests.

Scope notes
-----------
ImageNet-1k (Section 5.4), continual learning (Appendix E.5) and streaming
(Appendix E.6) are explicitly out of scope for this reproduction.
"""

from __future__ import annotations

# --- Dataset registry / loaders ------------------------------------------------
from .datasets import (
    DATASET_SPECS,
    CACHE,
    DatasetSpec,
    IndexedDataset,
    TransformSubset,
    WithTransform,
    build_transform,
    class_balanced_indices,
    class_counts,
    dataset_meta,
    dataset_names,
    default_root,
    get_dataset,
    get_loaders,
    get_targets,
    input_shape,
    make_loader,
    normalize_name,
    num_classes,
    split_train_val,
    stratified_indices,
    subset_dataset,
)

# --- MNIST-S (Section 5.1 / Figure 1) -----------------------------------------
from .mnist_s import (
    MNIST_S_SEED,
    MNIST_S_SIZE,
    MNIST_S_SPEC,
    MNIST_TEST_SIZE,
    MNIST_TRAIN_SIZE,
    build_mnist_s,
    get_mnist_s,
    get_mnist_s_loaders,
    mnist_s_dataset,
    mnist_s_indices,
    mnist_s_meta,
    mnist_s_targets,
)

# --- Robustness transforms (Section 5.3 / Appendix E.2) ----------------------
from .robustness import (
    DEFAULT_IMBALANCE_RATIO,
    DEFAULT_NOISE_RATES,
    EXPONENTIAL_IMBALANCE_RATIO,
    NOISE_RATE_30,
    NOISE_RATE_50,
    SYMMETRIC_NOISE_RATES,
    ImbalancedDataset,
    NoisyDataset,
    SymmetricNoiseDataset,
    apply_noisy_labels,
    exponential_imbalance_indices,
    flip_labels_symmetric,
    get_imbalanced_loaders,
    get_noisy_loaders,
    imbalance_ratios,
    inject_symmetric_noise,
    make_imbalanced_dataset,
    make_noisy_dataset,
    symmetric_noise_mask,
)

__all__ = [
    # datasets.py
    "DATASET_SPECS",
    "CACHE",
    "DatasetSpec",
    "IndexedDataset",
    "TransformSubset",
    "WithTransform",
    "build_transform",
    "class_balanced_indices",
    "class_counts",
    "dataset_meta",
    "dataset_names",
    "default_root",
    "get_dataset",
    "get_loaders",
    "get_targets",
    "input_shape",
    "make_loader",
    "normalize_name",
    "num_classes",
    "split_train_val",
    "stratified_indices",
    "subset_dataset",
    # mnist_s.py
    "MNIST_S_SEED",
    "MNIST_S_SIZE",
    "MNIST_S_SPEC",
    "MNIST_TRAIN_SIZE",
    "MNIST_TEST_SIZE",
    "build_mnist_s",
    "get_mnist_s",
    "get_mnist_s_loaders",
    "mnist_s_dataset",
    "mnist_s_indices",
    "mnist_s_meta",
    "mnist_s_targets",
    # robustness.py
    "DEFAULT_IMBALANCE_RATIO",
    "DEFAULT_NOISE_RATES",
    "EXPONENTIAL_IMBALANCE_RATIO",
    "NOISE_RATE_30",
    "NOISE_RATE_50",
    "SYMMETRIC_NOISE_RATES",
    "ImbalancedDataset",
    "NoisyDataset",
    "SymmetricNoiseDataset",
    "apply_noisy_labels",
    "exponential_imbalance_indices",
    "flip_labels_symmetric",
    "get_imbalanced_loaders",
    "get_noisy_loaders",
    "imbalance_ratios",
    "inject_symmetric_noise",
    "make_imbalanced_dataset",
    "make_noisy_dataset",
    "symmetric_noise_mask",
]
