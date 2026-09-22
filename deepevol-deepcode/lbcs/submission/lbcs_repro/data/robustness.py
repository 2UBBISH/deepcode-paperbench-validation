"""Imperfect-supervision data transforms for the LBCS reproduction (Section 5.3).

Two families of corruption are implemented, both affecting **only the training
split** (the addendum for Section 5.3 states explicitly that *"the imbalance is
just injected into the training set, which does not include the test set"*):

1. **Symmetric label noise** (Section 5.3, Appendix E.2)
   *"We inject 30% symmetric label noise into the original clean F-MNIST to
   generate the noisy version of F-MNIST. Namely, the labels of 30% training
   data are flipped."*
   The paper also evaluates a higher noise level, 50% (Appendix E.2).  Both are
   supported here: :data:`NOISE_RATE_30` / :data:`NOISE_RATE_50`.

   Symmetric noise means the corrupted label is drawn uniformly at random over
   the classes.  Following the plan ("flipping a fraction ... uniformly at
   random over classes") the default draws the replacement uniformly over *all*
   classes; ``exclude_original=True`` forces the replacement to differ from the
   original label (guaranteed corruption) for sensitivity studies.

2. **Exponential-type class imbalance** (Section 5.3)
   *"The exponential type of class imbalance (Cao et al., 2019) is used.  The
   imbalanced ratio is set to 0.01."*  The addendum notes the authors *"leveraged
   a class-imbalanced sampler, adjusting the code from
   imbalanced-semi-self/blob/master/dataset/imbalance_cifar.py to work with
   F-MNIST"*.  We reproduce that subsampling scheme exactly: with ``C`` classes
   and per-class capacity ``img_max``, class ``c`` keeps

       n_c = int(img_max * ratio ** (c / (C - 1)))

   examples, so that the most frequent class keeps ``img_max`` examples and the
   rarest keeps ``ratio * img_max`` (``ratio = 0.01`` -> 100x drop).

Everything is implemented with NumPy so the label algebra is unit-testable
without PyTorch; torch is a soft dependency used only for the dataset/loader
wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - soft dependency
    import torch
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (Section 5.3 / Appendix E.2)
# ---------------------------------------------------------------------------

#: Noise level used by the main robustness experiment of Section 5.3.
NOISE_RATE_30 = 0.30
#: Higher noise level evaluated in Appendix E.2.
NOISE_RATE_50 = 0.50
#: Noise rates swept when reproducing Figure 2(a) / Appendix E.2.
SYMMETRIC_NOISE_RATES: Tuple[float, ...] = (NOISE_RATE_30, NOISE_RATE_50)
#: Alias kept for the package facade.
DEFAULT_NOISE_RATES: Tuple[float, ...] = SYMMETRIC_NOISE_RATES
#: Fallback noise rate when none is given.
DEFAULT_NOISE_RATE = NOISE_RATE_30

#: Exponential imbalance ratio of Section 5.3 (minority / majority).
EXPONENTIAL_IMBALANCE_RATIO = 0.01
#: Alias used by the data package facade.
DEFAULT_IMBALANCE_RATIO = EXPONENTIAL_IMBALANCE_RATIO
#: Imbalance type used by the paper ("exponential type of class imbalance").
DEFAULT_IMB_TYPE = "exp"

__all__ = [
    "NOISE_RATE_30",
    "NOISE_RATE_50",
    "SYMMETRIC_NOISE_RATES",
    "DEFAULT_NOISE_RATES",
    "DEFAULT_NOISE_RATE",
    "EXPONENTIAL_IMBALANCE_RATIO",
    "DEFAULT_IMBALANCE_RATIO",
    "DEFAULT_IMB_TYPE",
    # label noise
    "symmetric_noise_mask",
    "flip_labels_symmetric",
    "inject_symmetric_noise",
    "apply_noisy_labels",
    "NoisyDataset",
    "SymmetricNoiseDataset",
    "make_noisy_dataset",
    "get_noisy_loaders",
    # class imbalance
    "exponential_class_counts",
    "exponential_imbalance_indices",
    "imbalance_ratios",
    "ImbalancedDataset",
    "make_imbalanced_dataset",
    "get_imbalanced_loaders",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _as_targets(dataset_or_targets: Any) -> np.ndarray:
    """Return integer labels for a dataset (or pass through an array)."""
    if isinstance(dataset_or_targets, (np.ndarray, list, tuple)) and not _has_targets(
        dataset_or_targets
    ):
        return np.asarray(dataset_or_targets, dtype=np.int64)
    try:
        from .datasets import get_targets  # local import: avoid import cycles

        return np.asarray(get_targets(dataset_or_targets), dtype=np.int64)
    except Exception:
        return np.asarray(dataset_or_targets.targets, dtype=np.int64)


def _has_targets(obj: Any) -> bool:
    return hasattr(obj, "targets") or hasattr(obj, "labels")


def _get_rng(seed: Optional[int], rng: Optional[np.random.Generator]) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(0 if seed is None else int(seed))


def _infer_num_classes(targets: np.ndarray, num_classes: Optional[int]) -> int:
    if num_classes is not None:
        return int(num_classes)
    return int(targets.max()) + 1 if targets.size else 1


def _class_counts(targets: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes)[
        :num_classes
    ]


# ---------------------------------------------------------------------------
# 1) Symmetric label noise  (Section 5.3, Appendix E.2)
# ---------------------------------------------------------------------------


def symmetric_noise_mask(
    targets: Any,
    noise_rate: float = DEFAULT_NOISE_RATE,
    seed: Optional[int] = 0,
    num_classes: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Boolean mask marking which training examples get a corrupted label.

    Exactly ``round(noise_rate * n)`` indices are selected uniformly at random
    without replacement, implementing *"the labels of 30% training data are
    flipped"* (Section 5.3).
    """
    y = _as_targets(targets)
    n = int(y.shape[0])
    rate = float(noise_rate)
    if rate <= 0.0 or n == 0:
        return np.zeros(n, dtype=bool)
    if rate >= 1.0:
        return np.ones(n, dtype=bool)
    gen = _get_rng(seed, rng)
    n_flip = int(round(rate * n))
    n_flip = max(0, min(n, n_flip))
    mask = np.zeros(n, dtype=bool)
    if n_flip:
        idx = gen.choice(n, size=n_flip, replace=False)
        mask[np.asarray(idx, dtype=np.int64)] = True
    return mask


def flip_labels_symmetric(
    targets: Any,
    noise_rate: float = DEFAULT_NOISE_RATE,
    num_classes: Optional[int] = None,
    seed: Optional[int] = 0,
    exclude_original: bool = False,
    rng: Optional[np.random.Generator] = None,
    return_mask: bool = False,
    corruption_mask: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Flip a fraction ``noise_rate`` of ``targets`` uniformly at random.

    Symmetric (uniform) noise: the corrupted label is sampled uniformly over the
    ``num_classes`` classes.  With ``exclude_original=True`` the replacement is
    re-drawn so it always differs from the original label.

    Returns the corrupted labels, or ``(labels, corruption_mask)`` when
    ``return_mask=True``.
    """
    y = _as_targets(targets)
    n = int(y.shape[0])
    C = _infer_num_classes(y, num_classes)
    gen = _get_rng(seed, rng)
    if corruption_mask is None:
        mask = symmetric_noise_mask(y, noise_rate, num_classes=C, rng=gen)
    else:
        mask = np.asarray(corruption_mask, dtype=bool).copy()
    new_y = y.copy()
    idx = np.flatnonzero(mask)
    if idx.size:
        if not exclude_original or C <= 1:
            new_y[idx] = gen.integers(0, C, size=idx.size).astype(np.int64)
        else:
            # Draw over the C-1 other classes and skip the original label.
            offsets = gen.integers(0, C - 1, size=idx.size).astype(np.int64)
            new_y[idx] = (y[idx] + 1 + offsets) % C
    if return_mask:
        return new_y, mask
    return new_y


def inject_symmetric_noise(
    dataset_or_targets: Any,
    noise_rate: float = DEFAULT_NOISE_RATE,
    num_classes: Optional[int] = None,
    seed: Optional[int] = 0,
    exclude_original: bool = False,
    return_mask: bool = False,
    return_labels: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Corrupt labels with symmetric noise and return the new label vector.

    Thin, explicitly named entry point used by the experiment drivers and by
    :class:`SymmetricNoiseDataset`.
    """
    return flip_labels_symmetric(
        dataset_or_targets,
        noise_rate=noise_rate,
        num_classes=num_classes,
        seed=seed,
        exclude_original=exclude_original,
        return_mask=return_mask,
    )


class NoisyDataset(Dataset):  # type: ignore[misc]
    """Dataset wrapper that replaces the labels of an arbitrary dataset.

    The wrapped dataset is untouched (so the clean labels remain recoverable
    through :attr:`clean_targets`), and the *test* split is never wrapped.
    """

    def __init__(
        self,
        dataset: Any,
        labels: Optional[Sequence[int]] = None,
        noise_rate: Optional[float] = None,
        num_classes: Optional[int] = None,
        seed: int = 0,
        exclude_original: bool = False,
        name: Optional[str] = None,
    ) -> None:
        self.dataset = dataset
        self.name = name
        self.num_classes = num_classes
        self.noise_rate = noise_rate
        self.seed = seed
        self.exclude_original = exclude_original
        self.clean_targets = np.asarray(_as_targets(dataset), dtype=np.int64)

        if labels is None:
            rate = DEFAULT_NOISE_RATE if noise_rate is None else float(noise_rate)
            labels, mask = flip_labels_symmetric(
                self.clean_targets,
                noise_rate=rate,
                num_classes=num_classes,
                seed=seed,
                exclude_original=exclude_original,
                return_mask=True,
            )
            self.corruption_mask = mask
            self.noise_rate = rate
        else:
            labels = np.asarray(labels, dtype=np.int64)
            self.corruption_mask = labels != self.clean_targets
            self.noise_rate = float(self.corruption_mask.mean())
        self.labels_arr = np.asarray(labels, dtype=np.int64)
        self.targets = self.labels_arr

    # -- dataset protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            rest = tuple(item[2:])
            return (item[0], int(self.labels_arr[index])) + rest
        return item

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        if item in {"dataset", "labels_arr", "clean_targets", "corruption_mask"}:
            raise AttributeError(item)
        return getattr(self.__dict__["dataset"], item)

    # -- helpers ----------------------------------------------------------
    @property
    def classes(self) -> Any:
        return getattr(self.dataset, "classes", None)

    @property
    def num_corrupted(self) -> int:
        return int(np.count_nonzero(self.corruption_mask))

    def corruption_mask_array(self) -> np.ndarray:
        return self.corruption_mask.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "noise_rate": self.noise_rate,
            "num_examples": len(self),
            "num_corrupted": self.num_corrupted,
            "realized_rate": (self.num_corrupted / len(self)) if len(self) else 0.0,
            "seed": self.seed,
            "exclude_original": self.exclude_original,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NoisyDataset({self.name!r}, n={len(self)}, "
            f"noise_rate={self.noise_rate}, corrupted={self.num_corrupted})"
        )


class SymmetricNoiseDataset(NoisyDataset):
    """F-MNIST/SVHN/CIFAR-10 training set with symmetric label noise.

    Matches Section 5.3: *"the labels of 30% training data are flipped"*.
    """

    def __init__(
        self,
        dataset: Any,
        noise_rate: float = DEFAULT_NOISE_RATE,
        num_classes: Optional[int] = None,
        seed: int = 0,
        exclude_original: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            dataset,
            labels=None,
            noise_rate=noise_rate,
            num_classes=num_classes,
            seed=seed,
            exclude_original=exclude_original,
            name=name,
        )


def make_noisy_dataset(
    dataset: Any,
    noise_rate: float = DEFAULT_NOISE_RATE,
    num_classes: Optional[int] = None,
    seed: int = 0,
    exclude_original: bool = False,
    name: Optional[str] = None,
) -> SymmetricNoiseDataset:
    """Wrap ``dataset`` with symmetric label noise at ``noise_rate``."""
    return SymmetricNoiseDataset(
        dataset,
        noise_rate=noise_rate,
        num_classes=num_classes,
        seed=seed,
        exclude_original=exclude_original,
        name=name,
    )


def apply_noisy_labels(
    dataset: Any,
    labels: Optional[Sequence[int]] = None,
    noise_rate: Optional[float] = None,
    num_classes: Optional[int] = None,
    seed: int = 0,
    exclude_original: bool = False,
    return_mask: bool = False,
) -> Union[NoisyDataset, Tuple[NoisyDataset, np.ndarray]]:
    """Attach (optionally precomputed) noisy labels to ``dataset``."""
    wrapped = NoisyDataset(
        dataset,
        labels=labels,
        noise_rate=noise_rate,
        num_classes=num_classes,
        seed=seed,
        exclude_original=exclude_original,
    )
    if return_mask:
        return wrapped, wrapped.corruption_mask_array()
    return wrapped


def get_noisy_loaders(
    name: str = "F-MNIST",
    noise_rate: float = DEFAULT_NOISE_RATE,
    batch_size: int = 128,
    root: Optional[str] = None,
    download: bool = True,
    num_workers: int = 0,
    shuffle_train: bool = True,
    normalize: bool = True,
    seed: Optional[int] = 0,
    num_classes: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    return_index: bool = False,
    exclude_original: bool = False,
    augment_train: bool = False,
    **dataset_kwargs: Any,
) -> Tuple[Any, Any]:
    """Build ``(train_loader, test_loader)`` for the noisy-label experiment.

    The corruption is applied to the **training set only**; the test loader
    always iterates the clean test split (addendum for Section 5.3).
    """
    from .datasets import dataset_meta, get_dataset, make_loader

    spec = dataset_meta(name)
    nc = int(num_classes) if num_classes is not None else int(spec.num_classes)

    train_ds = get_dataset(
        name,
        train=True,
        root=root,
        download=download,
        normalize=normalize,
        augment=augment_train,
        **dataset_kwargs,
    )
    test_ds = get_dataset(
        name,
        train=False,
        root=root,
        download=download,
        normalize=normalize,
        augment=False,
        **dataset_kwargs,
    )

    noisy_train = make_noisy_dataset(
        train_ds,
        noise_rate=noise_rate,
        num_classes=nc,
        seed=0 if seed is None else int(seed),
        exclude_original=exclude_original,
        name=name,
    )

    train_loader = make_loader(
        noisy_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
        drop_last=drop_last,
        return_index=return_index,
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


# ---------------------------------------------------------------------------
# 2) Exponential class imbalance  (Section 5.3)
# ---------------------------------------------------------------------------


def exponential_class_counts(
    num_classes: int,
    ratio: float = DEFAULT_IMBALANCE_RATIO,
    max_per_class: Optional[int] = None,
    min_per_class: int = 1,
    imb_type: str = DEFAULT_IMB_TYPE,
) -> np.ndarray:
    """Per-class sample counts for the exponential imbalance of Cao et al. (2019).

    Reproduces ``imbalance_cifar.py``'s ``get_img_num_per_cls``:

        ``num_c = int(img_max * ratio ** (c / (C - 1)))``

    so that class 0 keeps ``img_max`` examples and class ``C-1`` keeps
    ``ratio * img_max``; ``ratio`` is therefore the minority/majority ratio.
    Class 0 is the *majority* class, exactly as in the reference implementation.
    """
    C = int(num_classes)
    if C <= 0:
        raise ValueError("num_classes must be positive")
    if imb_type not in ("exp", "exponential"):
        raise ValueError("only the exponential imbalance type ('exp') is supported")
    img_max = 1 if max_per_class is None else int(max_per_class)
    ratio = float(ratio)
    if C == 1:
        return np.asarray([max(img_max, int(min_per_class))], dtype=np.int64)
    counts = []
    for c in range(C):
        num = img_max * (ratio ** (c / (C - 1.0)))
        counts.append(max(int(num), int(min_per_class)))
    return np.asarray(counts, dtype=np.int64)


def exponential_imbalance_indices(
    dataset_or_targets: Any,
    ratio: float = DEFAULT_IMBALANCE_RATIO,
    num_classes: Optional[int] = None,
    seed: Optional[int] = 0,
    shuffle: bool = True,
    sort: bool = True,
    max_per_class: Optional[int] = None,
    imb_type: str = DEFAULT_IMB_TYPE,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Indices of the training examples kept under exponential imbalance.

    The scheme mirrors ``imbalance_cifar.py``: the majority class keeps
    ``img_max = n / C`` examples (unless ``max_per_class`` is given), class ``c``
    keeps ``int(img_max * ratio ** (c / (C - 1)))``, and the selected indices are
    taken after an independent random permutation *within each class*.  Only the
    indices are returned, so the caller decides whether to build a subset of the
    training split (never of the test split).
    """
    y = _as_targets(dataset_or_targets)
    n = int(y.shape[0])
    C = _infer_num_classes(y, num_classes)
    if max_per_class is None:
        max_per_class = int(n / C) if C else n
    counts = exponential_class_counts(
        C,
        ratio=ratio,
        max_per_class=max_per_class,
        min_per_class=1 if n else 0,
        imb_type=imb_type,
    )
    gen = _get_rng(seed, rng)
    keep: List[np.ndarray] = []
    for c in range(C):
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        if shuffle:
            idx = gen.permutation(idx)
        take = int(min(counts[c], idx.size))
        keep.append(np.sort(idx[:take]) if sort else idx[:take])
    if not keep:
        return np.zeros(0, dtype=np.int64)
    selected = np.concatenate(keep) if sort else np.concatenate(keep)
    if sort:
        selected = np.sort(selected)
    return selected.astype(np.int64)


def imbalance_ratios(
    dataset_or_targets: Any, num_classes: Optional[int] = None
) -> Dict[str, Any]:
    """Class counts and the realized minority/majority ratio."""
    y = _as_targets(dataset_or_targets)
    C = _infer_num_classes(y, num_classes)
    counts = _class_counts(y, C)
    pos = counts[counts > 0]
    mn = int(pos.min()) if pos.size else 0
    mx = int(pos.max()) if pos.size else 0
    return {
        "num_classes": int(C),
        "counts": counts.tolist(),
        "max": mx,
        "min": mn,
        "ratio": (float(mn) / float(mx)) if mx else 0.0,
        "num_examples": int(y.shape[0]),
    }


class ImbalancedDataset(Dataset):  # type: ignore[misc]
    """Training-set view with exponential class imbalance (Section 5.3).

    ``indices`` are positions into the wrapped dataset; the imbalance therefore
    only ever re-subsets the split that was passed in (the training split).
    """

    def __init__(
        self,
        dataset: Any,
        indices: Optional[Sequence[int]] = None,
        ratio: float = DEFAULT_IMBALANCE_RATIO,
        num_classes: Optional[int] = None,
        seed: int = 0,
        imb_type: str = DEFAULT_IMB_TYPE,
        name: Optional[str] = None,
    ) -> None:
        self.dataset = dataset
        self.name = name
        self.imb_type = imb_type
        self.seed = int(seed)
        self.num_classes = num_classes
        self.ratio = float(ratio)
        self.clean_targets = np.asarray(_as_targets(dataset), dtype=np.int64)
        if indices is None:
            indices = exponential_imbalance_indices(
                self.clean_targets,
                ratio=ratio,
                num_classes=num_classes,
                seed=seed,
                imb_type=imb_type,
            )
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = self.clean_targets[self.indices]
        self.imbalance_stats = imbalance_ratios(self.targets, num_classes=num_classes)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[index])]

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        if item in {"dataset", "indices", "clean_targets", "imbalance_stats"}:
            raise AttributeError(item)
        return getattr(self.__dict__["dataset"], item)

    @property
    def classes(self) -> Any:
        return getattr(self.dataset, "classes", None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ratio": self.ratio,
            "imb_type": self.imb_type,
            "num_examples": len(self),
            "realized_ratio": self.imbalance_stats["ratio"],
            "counts": self.imbalance_stats["counts"],
            "seed": self.seed,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ImbalancedDataset({self.name!r}, n={len(self)}, "
            f"ratio={self.ratio}, realized={self.imbalance_stats['ratio']:.4f})"
        )


def make_imbalanced_dataset(
    dataset: Any,
    ratio: float = DEFAULT_IMBALANCE_RATIO,
    num_classes: Optional[int] = None,
    seed: int = 0,
    indices: Optional[Sequence[int]] = None,
    imb_type: str = DEFAULT_IMB_TYPE,
    name: Optional[str] = None,
) -> ImbalancedDataset:
    """Wrap ``dataset`` with exponential class imbalance at ``ratio``."""
    return ImbalancedDataset(
        dataset,
        indices=indices,
        ratio=ratio,
        num_classes=num_classes,
        seed=seed,
        imb_type=imb_type,
        name=name,
    )


def get_imbalanced_loaders(
    name: str = "F-MNIST",
    ratio: float = DEFAULT_IMBALANCE_RATIO,
    batch_size: int = 128,
    root: Optional[str] = None,
    download: bool = True,
    num_workers: int = 0,
    shuffle_train: bool = True,
    normalize: bool = True,
    seed: Optional[int] = 0,
    num_classes: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    return_index: bool = False,
    imb_type: str = DEFAULT_IMB_TYPE,
    augment_train: bool = False,
    **dataset_kwargs: Any,
) -> Tuple[Any, Any]:
    """Build ``(train_loader, test_loader)`` for the class-imbalanced experiment.

    Only the training split is re-subsampled; the test split stays balanced and
    is loaded verbatim (Section 5.3 addendum).
    """
    from .datasets import dataset_meta, get_dataset, make_loader

    spec = dataset_meta(name)
    nc = int(num_classes) if num_classes is not None else int(spec.num_classes)

    train_ds = get_dataset(
        name,
        train=True,
        root=root,
        download=download,
        normalize=normalize,
        augment=augment_train,
        **dataset_kwargs,
    )
    test_ds = get_dataset(
        name,
        train=False,
        root=root,
        download=download,
        normalize=normalize,
        augment=False,
        **dataset_kwargs,
    )

    imb_train = make_imbalanced_dataset(
        train_ds,
        ratio=ratio,
        num_classes=nc,
        seed=0 if seed is None else int(seed),
        imb_type=imb_type,
        name=name,
    )

    train_loader = make_loader(
        imb_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
        drop_last=drop_last,
        return_index=return_index,
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


# ---------------------------------------------------------------------------
# Offline self-test (no torch / no downloads required)
# ---------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Exercise the label algebra offline and return a report dict."""
    report: Dict[str, Any] = {}
    y = np.repeat(np.arange(10), 600)  # 6000 balanced examples, 10 classes

    # -- symmetric noise ------------------------------------------------
    for rate in SYMMETRIC_NOISE_RATES:
        noisy, mask = flip_labels_symmetric(y, rate, 10, seed=1, return_mask=True)
        assert mask.sum() == int(round(rate * y.size)), (rate, mask.sum())
        changed = noisy != y
        report[f"noise_{rate}"] = {
            "flipped": int(mask.sum()),
            "changed": int(changed.sum()),
            "rate": float(mask.mean()),
        }
        assert noisy[mask].min() >= 0 and noisy[mask].max() < 10
        # no corruption outside of the flagged positions
        assert np.array_equal(noisy[~mask], y[~mask])

    # exclude_original guarantees an actual label change
    noisy2, mask2 = flip_labels_symmetric(
        y, NOISE_RATE_30, 10, seed=2, exclude_original=True, return_mask=True
    )
    assert np.all(noisy2[mask2] != y[mask2])
    report["exclude_original_all_changed"] = True

    # deterministic under a fixed seed
    a = flip_labels_symmetric(y, 0.3, 10, seed=7)
    b = flip_labels_symmetric(y, 0.3, 10, seed=7)
    assert np.array_equal(a, b)
    report["deterministic"] = True

    # -- exponential imbalance ------------------------------------------
    idx = exponential_imbalance_indices(y, ratio=EXPONENTIAL_IMBALANCE_RATIO, num_classes=10, seed=0)
    sub = y[idx]
    stats = imbalance_ratios(sub, num_classes=10)
    counts = stats["counts"]
    assert len(idx) == len(set(idx.tolist()))
    assert counts[0] == 600, counts[0]
    assert counts[-1] == max(int(600 * EXPONENTIAL_IMBALANCE_RATIO), 1), counts[-1]
    assert abs(stats["ratio"] - EXPONENTIAL_IMBALANCE_RATIO) < 5e-3, stats["ratio"]
    for c in range(9):
        assert counts[c] >= counts[c + 1], counts
    report["imbalance"] = {
        "ratio": EXPONENTIAL_IMBALANCE_RATIO,
        "realized": stats["ratio"],
        "counts": counts,
        "num_examples": int(len(idx)),
    }

    if verbose:  # pragma: no cover - cosmetic
        print("[robustness] selftest OK:", report)
    return report


if __name__ == "__main__":  # pragma: no cover
    _selftest()
