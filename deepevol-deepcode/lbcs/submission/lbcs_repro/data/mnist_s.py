"""MNIST-S: the small MNIST variant introduced for Section 5.1 / Figure 1.

The paper states (Section 5.1, verbatim):

    "We conduct experiments on MNIST-S which is constructed by random sampling
     1,000 examples from original MNIST (LeCun et al., 1998)."

Figure 1 uses "an arbitrarily random subset of MNIST" together with the
``ConvNet`` of Zhou et al. (2022).  This module therefore provides:

* :func:`mnist_s_indices` -- the deterministic random sample of ``size``
  indices out of the 60,000 MNIST training examples;
* :func:`get_mnist_s` -- the MNIST-S dataset (a :class:`TransformSubset`
  view of MNIST's training split) plus, optionally, the sampled indices;
* :func:`get_mnist_s_loaders` -- train loader over MNIST-S and test loader
  over the *full* 10,000-example MNIST test split (evaluating the full-data
  objective ``f1`` of Eq. (1) uses the benchmark test set, while the
  coreset/selection pool is the 1,000-example MNIST-S).

Everything is loaded through ``torchvision`` (no Kaggle, no API keys), and a
synthetic offline fallback is available for smoke tests
(``fallback_synthetic=True``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .datasets import (
    get_dataset,
    make_loader,
    subset_dataset,
)

__all__ = [
    "MNIST_S_SIZE",
    "MNIST_S_SEED",
    "MNIST_TRAIN_SIZE",
    "MNIST_TEST_SIZE",
    "mnist_s_indices",
    "get_mnist_s",
    "mnist_s_dataset",
    "build_mnist_s",
    "get_mnist_s_loaders",
    "mnist_s_meta",
    "mnist_s_targets",
    "MNIST_S_SPEC",
]

# --------------------------------------------------------------------------- #
# Constants (paper: 1,000 randomly sampled MNIST examples)
# --------------------------------------------------------------------------- #
MNIST_S_SIZE: int = 1000          # Section 5.1: "random sampling 1,000 examples"
MNIST_S_SEED: int = 0             # suggested default (paper does not state a seed)
MNIST_TRAIN_SIZE: int = 60000     # torchvision MNIST train split
MNIST_TEST_SIZE: int = 10000      # torchvision MNIST test split

#: Metadata for MNIST-S, mirrors :class:`lbcs_repro.data.datasets.DatasetSpec`.
MNIST_S_SPEC: Dict[str, Any] = {
    "name": "MNIST-S",
    "alias": "mnist_s",
    "num_classes": 10,
    "in_channels": 1,
    "input_shape": (1, 28, 28),
    "mean": 0.1307,
    "std": 0.3081,
    "root_subdir": "MNIST",
    "size": MNIST_S_SIZE,
    "derived_from": "MNIST",
    "notes": (
        "1000 randomly sampled examples from the original MNIST training split "
        "(Section 5.1); the test split is the full MNIST test set."
    ),
}

#: Cache of sampled indices keyed by ``(pool_size, size, seed)`` so repeated
#: calls inside the 20-repeat protocol (Section 5.1) are cheap and consistent.
_INDICES_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


# --------------------------------------------------------------------------- #
# Index sampling
# --------------------------------------------------------------------------- #
def mnist_s_indices(
    size: int = MNIST_S_SIZE,
    seed: int = MNIST_S_SEED,
    n: int = MNIST_TRAIN_SIZE,
    shuffle: bool = True,
    sort: bool = True,
) -> np.ndarray:
    """Randomly sample ``size`` examples from ``n`` MNIST training examples.

    Sampling is without replacement using :class:`numpy.random.default_rng`
    seeded with ``seed`` so that every repeat of the 20-repeat Section 5.1
    protocol uses a fully reproducible MNIST-S instance.

    Parameters
    ----------
    size:
        Number of examples to keep (paper: 1,000).
    seed:
        Random seed of the sampling RNG.
    n:
        Size of the pool the sample is drawn from (60,000 for MNIST train).
    shuffle:
        Draw a random permutation; ``False`` takes the first ``size`` indices.
    sort:
        Return ascending indices (keeps ``Subset`` index maps canonical).

    Returns
    -------
    numpy.ndarray
        Integer index array of length ``min(size, n)``.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"pool size n must be positive, got {n}")
    size = int(size)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    size = min(size, n)
    rng = np.random.default_rng(int(seed))
    if shuffle:
        idx = rng.permutation(n)
    else:
        idx = np.arange(n)
    idx = np.asarray(idx[:size], dtype=np.int64)
    if sort:
        idx = np.sort(idx)
    return idx


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def get_mnist_s(
    size: int = MNIST_S_SIZE,
    seed: int = MNIST_S_SEED,
    root: Optional[str] = None,
    download: bool = True,
    transform: Any = "default",
    normalize: bool = True,
    augment: bool = False,
    train: bool = True,
    split: Optional[str] = None,
    return_indices: bool = False,
    fallback_synthetic: bool = False,
    cache: bool = True,
    verbose: bool = False,
    **dataset_kwargs: Any,
) -> Any:
    """Build the MNIST-S dataset used by Section 5.1 and Figure 1.

    For ``train=True`` this returns a lazy view over ``size`` randomly sampled
    MNIST *training* examples (the coreset selection pool, ``n = 1000``).
    For ``train=False`` it returns the full 10,000-example MNIST *test* split,
    which is the benchmark on which the full-data objective ``f1(m)`` of
    Eq. (1) is measured.

    Arguments mirror :func:`lbcs_repro.data.datasets.get_dataset` so that the
    dataset registry can route ``get_dataset("MNIST-S", ...)`` here.  The extra
    keyword ``split`` is accepted as an alias for ``train`` (``"train"`` /
    ``"test"``) because some drivers pass the split name.

    Returns
    -------
    dataset or (dataset, indices)
        ``return_indices=True`` additionally yields the sampled index array.
    """
    if split is not None:
        train = str(split).lower() in ("train", "training")

    base = get_dataset(
        "MNIST",
        train=bool(train),
        transform=transform,
        root=root,
        download=download,
        augment=augment,
        normalize=normalize,
        verbose=verbose,
        fallback_synthetic=fallback_synthetic,
        **dataset_kwargs,
    )

    pool = len(base) if hasattr(base, "__len__") else (
        MNIST_TRAIN_SIZE if train else MNIST_TEST_SIZE
    )

    if not train:
        # Standard protocol: the evaluation set is the complete MNIST test split.
        indices = np.arange(int(pool), dtype=np.int64)
        return (base, indices) if return_indices else base

    key = (int(pool), int(size), int(seed))
    if cache and key in _INDICES_CACHE:
        indices = _INDICES_CACHE[key]
    else:
        indices = mnist_s_indices(size=size, seed=seed, n=int(pool))
        if cache:
            _INDICES_CACHE[key] = indices

    subset = subset_dataset(base, indices, transform=None)
    return (subset, indices) if return_indices else subset


#: Readability aliases.
mnist_s_dataset = get_mnist_s
build_mnist_s = get_mnist_s


def mnist_s_targets(dataset: Any) -> np.ndarray:
    """Integer labels of an MNIST-S (sub)dataset as ``int64`` array."""
    if isinstance(dataset, tuple):  # (dataset, indices) tuple
        dataset = dataset[0]
    targets = getattr(dataset, "targets", None)
    if targets is None:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]
        return np.asarray(labels, dtype=np.int64)
    return np.asarray(targets, dtype=np.int64).reshape(-1)


def get_mnist_s_loaders(
    batch_size: int = 128,
    size: int = MNIST_S_SIZE,
    seed: int = MNIST_S_SEED,
    root: Optional[str] = None,
    download: bool = True,
    normalize: bool = True,
    num_workers: int = 0,
    return_index: bool = False,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    shuffle_train: bool = True,
    fallback_synthetic: bool = False,
    verbose: bool = False,
    **dataset_kwargs: Any,
) -> Tuple[Any, Any]:
    """Return ``(train_loader, test_loader)`` for MNIST-S.

    * ``train_loader`` iterates over the 1,000-example MNIST-S selection pool.
    * ``test_loader`` iterates over the full 10,000-example MNIST test split in
      a deterministic (unshuffled) order, so accuracies are comparable across
      the 20 repeats reported in Table 1.

    With ``return_index=True`` the training loader yields ``(x, y, index)``
    triples so that a global mask ``m`` can be mapped onto the selected
    examples (used by the inner coreset loss ``L(m, theta)``).
    """
    train_ds = get_mnist_s(
        size=size,
        seed=seed,
        root=root,
        download=download,
        normalize=normalize,
        train=True,
        return_indices=return_index,
        fallback_synthetic=fallback_synthetic,
        verbose=verbose,
        **dataset_kwargs,
    )
    if return_index:
        train_ds, _ = train_ds

    test_ds = get_mnist_s(
        size=size,
        seed=seed,
        root=root,
        download=download,
        normalize=normalize,
        train=False,
        fallback_synthetic=fallback_synthetic,
        verbose=verbose,
        **dataset_kwargs,
    )
    if return_index:
        test_ds = test_ds[0] if isinstance(test_ds, tuple) else test_ds

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=bool(shuffle_train),
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


def mnist_s_meta() -> Dict[str, Any]:
    """Return the MNIST-S metadata dictionary (copy of :data:`MNIST_S_SPEC`)."""
    return dict(MNIST_S_SPEC)


# --------------------------------------------------------------------------- #
# Self-test (offline friendly)
# --------------------------------------------------------------------------- #
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Unit checks for index sampling that need no downloads."""
    report: Dict[str, Any] = {}

    idx = mnist_s_indices(size=MNIST_S_SIZE, seed=0)
    report["size"] = int(idx.shape[0])
    report["unique"] = int(np.unique(idx).shape[0])
    report["in_range"] = bool(idx.min() >= 0 and idx.max() < MNIST_TRAIN_SIZE)
    report["sorted"] = bool(np.all(np.diff(idx) > 0))

    idx2 = mnist_s_indices(size=MNIST_S_SIZE, seed=0)
    report["deterministic"] = bool(np.array_equal(idx, idx2))

    idx3 = mnist_s_indices(size=MNIST_S_SIZE, seed=1)
    report["seed_sensitive"] = bool(not np.array_equal(idx, idx3))

    idx_big = mnist_s_indices(size=100000, n=500)
    report["clamped_to_pool"] = int(idx_big.shape[0]) == 500

    assert report["size"] == MNIST_S_SIZE
    assert report["unique"] == MNIST_S_SIZE
    assert report["in_range"] and report["sorted"] and report["deterministic"]
    assert report["seed_sensitive"] and report["clamped_to_pool"]

    if verbose:
        print("[mnist_s] self-test passed:", report)
    return report


if __name__ == "__main__":  # pragma: no cover
    _selftest()
