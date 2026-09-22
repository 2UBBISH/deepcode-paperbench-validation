"""Train/test split utilities for the SMM (Sample-specific Multi-channel Masks) paper.

The paper states: *"We follow Chen et al. (2023) to split the datasets"* and reports the
resulting split sizes in Appendix C, Table 6.  We therefore implement two complementary
split policies:

``native``
    torchvision ships an official train/test split for the dataset (CIFAR10, CIFAR100,
    SVHN, GTSRB, UCF101, Food101, StanfordCars, ...).  We simply use it; these splits are
    the ones Chen et al. (2023) / torchvision use and match Table 6 closely.

``ratio``
    torchvision does **not** provide a train/test split (Flowers102 is train/val/test,
    DTD is train/val/test, SUN397 has a fixed 50/50 list, EuroSAT is a single folder,
    OxfordPets is train/val).  For these we build a deterministic, *class balanced*
    split (every class contributes the same fraction of its samples) with the ratio that
    best reproduces the Table 6 train/test sizes.

All randomness is seeded with :func:`random.Random` (a plain Python RNG) so the splits are
**identical across the paper's three training seeds** -- only the model/mask initialization
varies.  Reference sizes from Table 6 are stored in :data:`REFERENCE_SPLIT_SIZES` and can be
used to sanity-check a materialised dataset.

Public API (mirrored by ``smm_vr/data/__init__.py``):
    ``class_balanced_split(targets, train_ratio, seed=0, ...)``
    ``make_split(name, ...)``
    ``split_indices(dataset, ...)``
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # split policies
    "NATIVE_SPLIT_DATASETS",
    "RATIO_SPLIT_DATASETS",
    "DEFAULT_TRAIN_RATIOS",
    "REFERENCE_SPLIT_SIZES",
    "REFERENCE_TRAIN_SIZES",
    "REFERENCE_TEST_SIZES",
    # helpers
    "is_native_split",
    "default_split_policy",
    "default_train_ratio",
    "reference_split_size",
    "extract_targets",
    "split_indices",
    "class_balanced_split",
    "class_balanced_split_from_targets",
    "proportional_split",
    "make_split",
    "make_split_indices",
    "subsample_indices",
    "describe_split",
    "verify_split_size",
]


# --------------------------------------------------------------------------------------
# Table 6 (Appendix C): "Detailed Dataset Information"
#   Dataset | Original Image Size | Training Set Size | Testing Set Size | #Classes
# --------------------------------------------------------------------------------------
REFERENCE_SPLIT_SIZES: Dict[str, Tuple[int, int]] = {
    "cifar10": (50000, 10000),
    "cifar100": (50000, 10000),
    "svhn": (73257, 26032),
    "gtsrb": (39209, 12630),
    "flowers102": (4093, 2463),
    "dtd": (2820, 1692),
    "ucf101": (7639, 3783),
    "food101": (50500, 30300),
    "sun397": (15888, 19850),
    "eurosat": (13500, 8100),
    "oxfordpets": (2944, 3669),
    "stanfordcars": (8144, 8041),
}

REFERENCE_TRAIN_SIZES: Dict[str, int] = {k: v[0] for k, v in REFERENCE_SPLIT_SIZES.items()}
REFERENCE_TEST_SIZES: Dict[str, int] = {k: v[1] for k, v in REFERENCE_SPLIT_SIZES.items()}


#: Datasets for which torchvision's native split matches (or closely matches) Table 6.
NATIVE_SPLIT_DATASETS: Tuple[str, ...] = (
    "cifar10",
    "cifar100",
    "svhn",
    "gtsrb",
    "ucf101",
    "food101",
    "stanfordcars",
)

#: Datasets for which we construct a deterministic class-balanced split.
RATIO_SPLIT_DATASETS: Tuple[str, ...] = (
    "flowers102",
    "dtd",
    "sun397",
    "eurosat",
    "oxfordpets",
)

#: Train fraction used for the ``ratio`` policy (tuned to approach Table 6 sizes).
#  Flowers102: 1024/6149 = 0.1666   -> Table 6: 4093/2463
#  DTD:        1888/5640 = 0.3348   -> Table 6: 2820/1692
#  SUN397:     15888/39734 ~ 0.40   -> Table 6: 15888/19850 (official list is 50/50)
#  EuroSAT:    13500/21600 = 0.625  -> Table 6: 13500/8100
#  OxfordPets: 2944/3669 (train/val) -> Table 6: 2944/3669
DEFAULT_TRAIN_RATIOS: Dict[str, float] = {
    "flowers102": 4093 / 6556,          # 6149/102 available examples + val
    "dtd": 2820 / 4512,                 # 1888 train + 1888 val (+ test fold unused)
    "sun397": 0.40,
    "eurosat": 13500 / 21600,
    "oxfordpets": 2944 / 6613,          # 2944 train + 3669 val
}

#: Dataset aliases -> canonical name (kept in sync with ``data/datasets.DATASET_ALIASES``).
_ALIASES: Dict[str, str] = {
    "cifar-10": "cifar10",
    "cifar_10": "cifar10",
    "cifar-100": "cifar100",
    "cifar_100": "cifar100",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "flowers": "flowers102",
    "flowers-102": "flowers102",
    "flowers_102": "flowers102",
    "oxford_flowers102": "flowers102",
    "dtd": "dtd",
    "describable_textures": "dtd",
    "ucf": "ucf101",
    "ucf-101": "ucf101",
    "ucf_101": "ucf101",
    "food": "food101",
    "food-101": "food101",
    "food_101": "food101",
    "sun": "sun397",
    "sun-397": "sun397",
    "sun_397": "sun397",
    "eurosat": "eurosat",
    "euro_sat": "eurosat",
    "oxford-iiit-pet": "oxfordpets",
    "oxford_pets": "oxfordpets",
    "oxford-iiit-pets": "oxfordpets",
    "pets": "oxfordpets",
    "stanford-cars": "stanfordcars",
    "stanford_cars": "stanfordcars",
    "cars": "stanfordcars",
}


def _canonical(name: str) -> str:
    """Normalise a dataset name/alias to the canonical key used by the registry."""
    if name is None:
        raise ValueError("dataset name must not be None")
    key = str(name).strip().lower().replace(" ", "")
    if key in REFERENCE_SPLIT_SIZES or key in RATIO_SPLIT_DATASETS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    # last resort: strip digits/punctuation separators and retry
    squashed = key.replace("-", "").replace("_", "").replace(".", "")
    for candidate in REFERENCE_SPLIT_SIZES:
        if candidate.replace("-", "").replace("_", "") == squashed:
            return candidate
    return key


# --------------------------------------------------------------------------------------
# Policy resolution
# --------------------------------------------------------------------------------------
def is_native_split(name: str) -> bool:
    """Return ``True`` if torchvision's native split is used for ``name``."""
    return _canonical(name) in NATIVE_SPLIT_DATASETS


def default_split_policy(name: str) -> str:
    """Return ``"native"`` or ``"ratio"`` for a dataset (see module docstring)."""
    return "native" if is_native_split(name) else "ratio"


def default_train_ratio(name: str) -> float:
    """Return the default train fraction used by the ``ratio`` policy."""
    return float(DEFAULT_TRAIN_RATIOS.get(_canonical(name), 0.5))


def reference_split_size(name: str) -> Tuple[int, int]:
    """Plain-English helper: ``(train_size, test_size)`` from Table 6."""
    return REFERENCE_SPLIT_SIZES[_canonical(name)]


def _resolve_policy(name: str, split_policy: Optional[str]) -> str:
    if split_policy in (None, "default", "auto"):
        return default_split_policy(name)
    policy = str(split_policy).strip().lower()
    if policy in ("native", "official", "provided"):
        return "native"
    if policy in ("ratio", "class_balanced", "balanced", "custom", "split"):
        return "ratio"
    if policy in ("none", "train", "all"):
        return "native"
    raise ValueError(f"unknown split_policy {split_policy!r} for dataset {name!r}")


# --------------------------------------------------------------------------------------
# Target extraction (works with torchvision datasets, Subset/ConcatDataset wrappers)
# --------------------------------------------------------------------------------------
def extract_targets(dataset: Any) -> Optional[List[int]]:
    """Best-effort extraction of a flat per-sample label list.

    Supports torchvision datasets exposing ``targets`` / ``labels`` / ``_labels``,
    :class:`torch.utils.data.Subset` (recursing into the base dataset) and
    ``ConcatDataset``.  Returns ``None`` when labels cannot be recovered.
    """
    if dataset is None:
        return None

    for attr in ("targets", "labels", "_labels", "ys", "y", "target"):
        value = getattr(dataset, attr, None)
        if value is None:
            continue
        try:
            seq = list(value)
        except TypeError:
            continue
        # ``targets`` may store 1-D labels or per-sample one-hot/dicts.
        flat: List[int] = []
        ok = True
        for item in seq:
            if isinstance(item, (int,)) or (hasattr(item, "item") and getattr(item, "ndim", 1) == 0):
                flat.append(int(item))
            elif hasattr(item, "argmax"):  # one-hot row
                flat.append(int(item.argmax()))
            else:
                ok = False
                break
        if ok and flat:
            return flat

    # Subset: map child indices to child labels
    indices = getattr(dataset, "indices", None)
    base = getattr(dataset, "dataset", None)
    if indices is not None and base is not None:
        child = extract_targets(base)
        if child is not None:
            return [child[int(i)] for i in indices]

    # ConcatDataset
    children = getattr(dataset, "datasets", None)
    if children:
        merged: List[int] = []
        for child_ds in children:
            child = extract_targets(child_ds)
            if child is None:
                return None
            merged.extend(child)
        return merged

    return None


# --------------------------------------------------------------------------------------
# Core splitters
# --------------------------------------------------------------------------------------
def _group_by_class(targets: Sequence[int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = {}
    for idx, label in enumerate(targets):
        groups.setdefault(int(label), []).append(int(idx))
    return groups


def class_balanced_split(
    targets: Sequence[int],
    train_ratio: float = 0.8,
    seed: int = 0,
    *,
    min_train_per_class: int = 1,
    min_test_per_class: int = 1,
    shuffle: bool = True,
    max_train_per_class: Optional[int] = None,
    max_test_per_class: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Deterministic **class-balanced** train/test split.

    Every class contributes ``round(train_ratio * n_c)`` samples to the training split
    (clipped to ``[min_train_per_class, n_c - min_test_per_class]``), which is what makes
    the split balanced even for the highly imbalanced target datasets (DTD, SUN397,
    Flowers102).  Indices within a class are shuffled with ``random.Random(seed)`` so the
    split is reproducible and independent of the training seed.

    Args:
        targets: per-sample integer labels of the full dataset.
        train_ratio: fraction of each class used for training.
        seed: RNG seed (fixed across the paper's three training seeds).
        min_train_per_class: lower bound on train samples per class.
        min_test_per_class: lower bound on test samples per class.
        shuffle: shuffle class indices before splitting (else a contiguous cut is used).
        max_train_per_class: optional cap on train samples per class (debugging aid).
        max_test_per_class: optional cap on test samples per class (debugging aid).

    Returns:
        ``(train_indices, test_indices)`` as sorted lists of dataset indices.
    """
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f"train_ratio must lie in (0, 1), got {train_ratio!r}")
    if len(targets) == 0:
        raise ValueError("cannot split an empty target list")

    rng = random.Random(int(seed))
    groups = _group_by_class(targets)

    train_indices: List[int] = []
    test_indices: List[int] = []

    for label in sorted(groups):
        members = list(groups[label])
        if shuffle:
            rng.shuffle(members)
        n = len(members)
        if n <= 1:
            # A single sample can only be used for training (test set stays non-empty
            # overall because other classes contribute test samples).
            train_indices.extend(members)
            continue

        n_train = int(round(float(train_ratio) * n))
        upper = n - int(min_test_per_class)
        lower = int(min_train_per_class)
        if upper < lower:
            upper = max(lower, n - 1)
        n_train = max(lower, min(n_train, upper, n))
        if max_train_per_class is not None:
            n_train = min(n_train, int(max_train_per_class))

        class_train = members[:n_train]
        class_test = members[n_train:]
        if max_test_per_class is not None:
            class_test = class_test[: int(max_test_per_class)]

        train_indices.extend(class_train)
        test_indices.extend(class_test)

    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices


#: Alias kept for symmetry with :func:`extract_targets` call sites.
class_balanced_split_from_targets = class_balanced_split


def proportional_split(
    targets: Sequence[int],
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Plain (non class-balanced) random split, kept for debugging/comparison."""
    rng = random.Random(int(seed))
    indices = list(range(len(targets)))
    rng.shuffle(indices)
    n_train = int(round(float(train_ratio) * len(indices)))
    train = sorted(indices[:n_train])
    test = sorted(indices[n_train:])
    return train, test


def split_indices(
    dataset: Any,
    *,
    train_ratio: Optional[float] = None,
    split_seed: int = 0,
    policy: str = "ratio",
    class_balanced: bool = True,
    train_fraction: Optional[float] = None,
    max_train_per_class: Optional[int] = None,
    max_test_per_class: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Split an already-materialised dataset into ``(train_indices, test_indices)``.

    ``dataset`` may be a torchvision dataset, a ``Subset`` or a ``ConcatDataset``; labels are
    recovered with :func:`extract_targets`.  When ``policy == "native"`` and the dataset
    exposes explicit train/test parts (two children), the child boundaries are honoured.

    ``train_fraction`` deterministically subsamples the *training* split (class balanced)
    and is intended only for fast debugging runs.
    """
    policy = str(policy).strip().lower()
    if policy in ("native", "official", "provided"):
        children = getattr(dataset, "datasets", None)
        if children is not None and len(children) == 2:
            n_first = len(children[0])
            train = list(range(n_first))
            test = list(range(n_first, n_first + len(children[1])))
            if train_fraction is not None:
                train = subsample_indices(
                    train, extract_targets(children[0]), train_fraction, seed=split_seed
                )
            return train, test

    targets = extract_targets(dataset)
    if targets is None:
        # Fall back to a contiguous 80/20 split when labels are unavailable.
        n = len(dataset)
        n_train = int(round(0.8 * n))
        return list(range(n_train)), list(range(n_train, n))

    ratio = float(train_ratio) if train_ratio is not None else 0.8
    if class_balanced:
        train, test = class_balanced_split(
            targets,
            ratio,
            seed=split_seed,
            max_train_per_class=max_train_per_class,
            max_test_per_class=max_test_per_class,
        )
    else:
        train, test = proportional_split(targets, ratio, seed=split_seed)

    if train_fraction is not None:
        train = subsample_indices(train, targets, train_fraction, seed=split_seed)

    return train, test


def subsample_indices(
    indices: Sequence[int],
    targets: Optional[Sequence[int]],
    fraction: float,
    seed: int = 0,
) -> List[int]:
    """Deterministically (class-balanced when possible) keep ``fraction`` of ``indices``.

    Used for the ``train_fraction`` debugging knob; with ``fraction >= 1`` the input is
    returned unchanged.
    """
    if fraction is None or float(fraction) >= 1.0:
        return sorted(int(i) for i in indices)
    fraction = max(0.0, float(fraction))
    rng = random.Random(int(seed))
    indices = [int(i) for i in indices]

    if targets is None or fraction == 0.0:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        return sorted(shuffled[: max(1, int(round(fraction * len(indices))))])

    groups: Dict[int, List[int]] = {}
    for i in indices:
        groups.setdefault(int(targets[i]), []).append(i)

    kept: List[int] = []
    for label in sorted(groups):
        members = list(groups[label])
        rng.shuffle(members)
        n_keep = max(1, int(round(fraction * len(members))))
        kept.extend(members[:n_keep])
    kept.sort()
    return kept


# --------------------------------------------------------------------------------------
# Dataset-level convenience wrapper
# --------------------------------------------------------------------------------------
def make_split(
    name: str,
    dataset: Any = None,
    *,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    split_policy: Optional[str] = None,
    train_ratio: Optional[float] = None,
    split_seed: int = 0,
    train_fraction: Optional[float] = None,
    download: bool = False,
    backbone: Optional[str] = None,
    imgsize: Optional[int] = None,
    ucf101_fold: int = 1,
    annotations_path: Optional[str] = None,
    return_datasets: bool = False,
) -> Any:
    """Return the train/test split for a target dataset.

    By default this returns ``(train_indices, test_indices)``.  With
    ``return_datasets=True`` it returns ``(train_dataset, test_dataset)`` materialised via
    :func:`smm_vr.data.datasets.build_datasets` (imported lazily to avoid import cycles).

    The split is deterministic (seeded by ``split_seed``, which the training code fixes to
    a constant so that all three reported seeds share the same data split), and follows the
    Table 6 sizes as closely as the underlying torchvision data allows.
    """
    name = _canonical(name)
    policy = _resolve_policy(name, split_policy)
    ratio = float(train_ratio) if train_ratio is not None else default_train_ratio(name)

    if return_datasets:
        # Lazy import: ``datasets`` does not import ``splits``, so this is cycle-free.
        from .datasets import build_datasets  # noqa: WPS433 (local import by design)

        train_ds, test_ds, spec = build_datasets(
            name,
            backbone=backbone,
            root=root,
            data_root=data_root,
            imgsize=imgsize,
            download=download,
            split_policy=policy,
            train_ratio=ratio,
            split_seed=split_seed,
            train_fraction=train_fraction,
            ucf101_fold=ucf101_fold,
            annotations_path=annotations_path,
        )
        return train_ds, test_ds

    if dataset is None:
        from .datasets import _make_native_dataset  # type: ignore[attr-defined]

        dataset = _make_native_dataset(
            name,
            root=resolve_root(name, root=root, data_root=data_root),
            transform=None,
            download=download,
            ucf101_fold=ucf101_fold,
            annotations_path=annotations_path,
        )

    return split_indices(
        dataset,
        train_ratio=ratio,
        split_seed=split_seed,
        policy=policy,
        class_balanced=(policy == "ratio"),
        train_fraction=train_fraction,
    )


#: Alias with an explicit "indices" suffix, for readability at call sites.
make_split_indices = make_split


def resolve_root(
    name: str,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
) -> str:
    """Resolve the on-disk dataset directory (mirrors ``datasets.resolve_dataset_root``)."""
    from .datasets import resolve_dataset_root  # local import to avoid cycles

    return resolve_dataset_root(_canonical(name), root=root, data_root=data_root)


# --------------------------------------------------------------------------------------
# Reporting / verification helpers
# --------------------------------------------------------------------------------------
def describe_split(
    name: str,
    train_indices: Optional[Sequence[int]] = None,
    test_indices: Optional[Sequence[int]] = None,
    targets: Optional[Sequence[int]] = None,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """Summarise a split: sizes, number of classes and comparison to Table 6."""
    name = _canonical(name)
    ref_train, ref_test = reference_split_size(name)
    info: Dict[str, Any] = {
        "dataset": name,
        "split_seed": int(split_seed),
        "reference_train_size": ref_train,
        "reference_test_size": ref_test,
        "policy": default_split_policy(name),
    }
    if train_indices is not None:
        info["train_size"] = len(train_indices)
        info["test_size"] = len(test_indices) if test_indices is not None else None
        info["train_size_matches_table6"] = (
            len(train_indices) == ref_train if train_indices is not None else None
        )
        if test_indices is not None:
            info["test_size_matches_table6"] = len(test_indices) == ref_test
    if targets is not None and train_indices is not None:
        classes = sorted({int(targets[i]) for i in train_indices})
        info["num_classes_seen"] = len(classes)
        if test_indices is not None:
            classes_test = {int(targets[i]) for i in test_indices}
            missing = sorted(set(range(max(classes) + 1)) - classes_test)
            info["classes_missing_from_test"] = missing[:20]
    return info


def verify_split_size(
    name: str,
    train_indices: Optional[Sequence[int]] = None,
    test_indices: Optional[Sequence[int]] = None,
    *,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    tolerance: float = 0.02,
    raise_on_mismatch: bool = False,
) -> Dict[str, Any]:
    """Check a split against the Table 6 reference sizes.

    Exact equality is reported when available; otherwise the relative deviation is compared
    against ``tolerance``.  Returns a report dict (never raises unless asked to).
    """
    name = _canonical(name)
    ref_train, ref_test = reference_split_size(name)
    if train_size is None and train_indices is not None:
        train_size = len(train_indices)
    if test_size is None and test_indices is not None:
        test_size = len(test_indices)

    def _dev(size: Optional[int], ref: int) -> Optional[float]:
        if size is None:
            return None
        return abs(size - ref) / max(1, ref)

    report: Dict[str, Any] = {
        "dataset": name,
        "reference_train_size": ref_train,
        "reference_test_size": ref_test,
        "train_size": train_size,
        "test_size": test_size,
        "train_relative_error": _dev(train_size, ref_train),
        "test_relative_error": _dev(test_size, ref_test),
    }
    ok = True
    for key in ("train", "test"):
        err = report[f"{key}_relative_error"]
        if err is not None and err > float(tolerance):
            ok = False
    report["within_tolerance"] = ok
    if raise_on_mismatch and not ok:
        raise AssertionError(
            f"split for {name!r} deviates from Table 6 beyond tolerance {tolerance}: {report}"
        )
    return report


#: Backwards-compatible alias used by some analysis scripts.
class_balanced_split_indices = class_balanced_split


def iter_reference_splits() -> Iterable[Tuple[str, int, int]]:
    """Iterate ``(dataset, train_size, test_size)`` in Table 6 order."""
    for key, (train, test) in REFERENCE_SPLIT_SIZES.items():
        yield key, train, test


def data_root_default() -> str:
    """Default dataset root: ``$SMM_DATA_ROOT`` if set, else ``./data``."""
    return os.environ.get("SMM_DATA_ROOT", os.path.join(os.getcwd(), "data"))
