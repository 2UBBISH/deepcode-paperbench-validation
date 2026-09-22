"""Dataset loading utilities for the LBCS reproduction.

Paper references
----------------
* §5.1: "We conduct experiments on MNIST-S which is constructed by random
  sampling 1,000 examples from original MNIST (LeCun et al., 1998)."
* §5.2: "We employ FashionMNIST (abbreviated as F-MNIST) (Xiao et al., 2017),
  SVHN (Netzer et al., 2011), and CIFAR-10 (Krizhevsky et al., 2009) to
  evaluate our method."  All benchmarks are obtained through standard public
  routes (``torchvision.datasets``) so that no Kaggle access and no API keys
  are required (Addendum "General useful details").

The module provides:

``get_dataset``          build a (train or test) dataset, normalized per benchmark
``get_loaders``          ``(train_loader, test_loader)`` convenience helper
``make_loader``          generic DataLoader factory (deterministic worker seeding)
``subset_dataset``       index-subset view used to materialize a coreset
``split_train_val``      deterministic train/validation index split
``get_targets``          integer label array for any supported dataset
``dataset_meta``         num classes / channels / input shape / normalization
``IndexedDataset``       yields ``(x, y, index)`` triples (coreset bookkeeping)

Because a coreset is a *subset of training examples*, the loaders expose a
``return_index`` option: the training batches then carry the original dataset
index of every example, which is what the inner-loop trainer and the
full-data evaluator use to map optimized masks back to examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch / torchvision are required for the real experiments
    import torch
    import torch.nn as nn  # noqa: F401  (re-exported type hints)
    from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - allows mask-only unit tests
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    Subset = None  # type: ignore
    ConcatDataset = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    from torchvision import datasets as tv_datasets, transforms as T  # type: ignore

    _TORCHVISION_AVAILABLE = True
except Exception:  # pragma: no cover
    tv_datasets = None  # type: ignore
    T = None  # type: ignore
    _TORCHVISION_AVAILABLE = False


__all__ = [
    "DATASET_SPECS",
    "DatasetSpec",
    "get_dataset",
    "get_loaders",
    "make_loader",
    "subset_dataset",
    "split_train_val",
    "get_targets",
    "dataset_meta",
    "dataset_names",
    "num_classes",
    "input_shape",
    "IndexedDataset",
    "WithTransform",
    "TransformSubset",
    "stratified_indices",
    "class_counts",
    "class_balanced_indices",
    "default_root",
    "normalize_name",
    "build_transform",
    "CACHE",
]


# ---------------------------------------------------------------------------
# Benchmark specifications (normalization + shapes)
# ---------------------------------------------------------------------------
#
# The normalization statistics are the standard ones used with each benchmark:
# MNIST/F-MNIST channel statistics, and the well known per-channel statistics
# of SVHN and CIFAR-10.  They are exposed here so that every network in
# ``lbcs_repro.models`` receives inputs normalized exactly the same way.

@dataclass
class DatasetSpec:
    """Static description of a benchmark used in the paper."""

    name: str                       # canonical name, e.g. "F-MNIST"
    alias: Tuple[str, ...]          # accepted aliases / config spellings
    num_classes: int
    in_channels: int
    input_shape: Tuple[int, int, int]      # (C, H, W)
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    root_subdir: str                        # torchvision ``root`` sub-folder
    default_root: str = "./data"
    supports_augmentation: bool = True
    train_size: Optional[int] = None        # standard split size (informational)
    test_size: Optional[int] = None
    notes: str = ""

    def transform(self, train: bool = True, normalize: bool = True,
                  augment: bool = False) -> Any:
        """Build the torchvision transform pipeline for this benchmark."""
        return build_transform(self, train=train, normalize=normalize,
                               augment=augment)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "MNIST": DatasetSpec(
        name="MNIST",
        alias=("mnist", "MNIST", "mnist-s", "mnist_s", "MNIST-S", "MNIST_S"),
        num_classes=10,
        in_channels=1,
        input_shape=(1, 28, 28),
        mean=(0.1307,),
        std=(0.3081,),
        root_subdir="MNIST",
        train_size=60000,
        test_size=10000,
        notes="Source of MNIST-S (1000 randomly sampled examples, §5.1).",
    ),
    "F-MNIST": DatasetSpec(
        name="F-MNIST",
        alias=("f-mnist", "fmnist", "fashion-mnist", "FashionMNIST",
               "fashion_mnist", "F-MNIST", "f_mnist"),
        num_classes=10,
        in_channels=1,
        input_shape=(1, 28, 28),
        mean=(0.2860,),
        std=(0.3530,),
        root_subdir="FashionMNIST",
        train_size=60000,
        test_size=10000,
        notes="§5.2/§5.3 benchmark; LeNet is used for both selection and "
              "post-selection training.",
    ),
    "SVHN": DatasetSpec(
        name="SVHN",
        alias=("svhn", "SVHN", "svhn-cropped", "SVHN-Cropped"),
        num_classes=10,
        in_channels=3,
        input_shape=(3, 32, 32),
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
        root_subdir="SVHN",
        train_size=73257,
        test_size=26032,
        notes="§5.2/§6 benchmark (SVHN cropped 32x32 'train'/'test' splits).",
    ),
    "CIFAR-10": DatasetSpec(
        name="CIFAR-10",
        alias=("cifar10", "cifar-10", "CIFAR10", "CIFAR-10", "cifar_10"),
        num_classes=10,
        in_channels=3,
        input_shape=(3, 32, 32),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        root_subdir="CIFAR10",
        train_size=50000,
        test_size=10000,
        notes="§5.2 benchmark; ResNet-18 target model, SGD lr=0.1 + cosine, "
              "200 epochs.",
    ),
}

_ALIAS_TO_NAME: Dict[str, str] = {}
for _nm, _spec in DATASET_SPECS.items():
    _ALIAS_TO_NAME[_nm.lower()] = _nm
    for _a in _spec.alias:
        _ALIAS_TO_NAME[_a.lower()] = _nm


def normalize_name(name: str) -> str:
    """Map any paper/config spelling of a benchmark to its canonical name."""
    key = str(name).strip().lower()
    if key in _ALIAS_TO_NAME:
        return _ALIAS_TO_NAME[key]
    # tolerate underscores / spaces (e.g. "fashion mnist", "mnist_s")
    collapsed = key.replace("_", "-").replace(" ", "-")
    if collapsed in _ALIAS_TO_NAME:
        return _ALIAS_TO_NAME[collapsed]
    raise KeyError(
        f"Unknown dataset {name!r}. Known benchmarks: "
        f"{sorted(DATASET_SPECS)} (aliases: {sorted(_ALIAS_TO_NAME)})"
    )


def dataset_names() -> List[str]:
    """Canonical benchmark names supported by this module."""
    return list(DATASET_SPECS.keys())


def dataset_meta(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` of ``name``."""
    return DATASET_SPECS[normalize_name(name)]


def num_classes(name: str) -> int:
    """Number of classes of benchmark ``name`` (all in-scope ones have 10)."""
    return dataset_meta(name).num_classes


def input_shape(name: str) -> Tuple[int, int, int]:
    """Input tensor shape ``(C, H, W)`` of benchmark ``name``."""
    return dataset_meta(name).input_shape


def default_root(root: Optional[str] = None) -> str:
    """Root directory where torchvision downloads the benchmarks."""
    if root is not None:
        return root
    return os.environ.get("LBCS_DATA_ROOT", "./data")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transform(spec: DatasetSpec, train: bool = True,
                    normalize: bool = True, augment: bool = False) -> Any:
    """Compose the torchvision transform pipeline for ``spec``.

    ``augment`` is only enabled for the *target-model* training stage (the
    paper trains post-selection target models with standard recipes); coreset
    selection itself always uses deterministic inputs, matching Figure 1 /
    Algorithm 1 which evaluate a fixed network on fixed data.
    """
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        return None
    ops: List[Any] = []
    if augment and spec.in_channels == 3:
        ops.append(T.RandomCrop(32, padding=4))
        ops.append(T.RandomHorizontalFlip())
    ops.append(T.ToTensor())
    if normalize:
        ops.append(T.Normalize(list(spec.mean), list(spec.std)))
    if len(ops) == 1:
        return ops[0]
    return T.Compose(ops)


# ---------------------------------------------------------------------------
# Wrappers: transform application and index exposure
# ---------------------------------------------------------------------------

class WithTransform(Dataset if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Apply ``transform`` lazily to a cached base dataset.

    torchvision stores ``transform`` at construction time, so caching a raw
    dataset requires this thin wrapper to allow several transform pipelines
    over the same decoded data (e.g. train vs. test, or augmented targets).
    """

    def __init__(self, dataset: Any, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    # provenance used by ``get_targets`` / ``subset_dataset``
    @property
    def targets(self) -> Any:
        return getattr(self.dataset, "targets", None)

    @property
    def labels(self) -> Any:
        return getattr(self.dataset, "labels", None)

    @property
    def classes(self) -> Any:
        return getattr(self.dataset, "classes", None)

    @property
    def base(self) -> Any:
        return getattr(self.dataset, "dataset", self.dataset)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"WithTransform(base={type(self.dataset).__name__}, " \
               f"transform={self.transform})"


class IndexedDataset(Dataset if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Dataset wrapper yielding ``(x, y, index)`` triples.

    ``index`` is the index *within the wrapped dataset* (i.e. within the
    coreset subset when this wraps a ``Subset``), while ``source_index`` maps
    it back to the original training-set index.  ``lbcs.objectives`` unpacks
    3-tuples transparently, so training loops can carry either of them.
    """

    def __init__(self, dataset: Any, return_source: bool = False,
                 transform: Optional[Callable] = None):
        self.dataset = dataset
        self.return_source = return_source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            x, y = sample[0], sample[1]
        else:  # pragma: no cover - unexpected dataset format
            raise TypeError(f"Unsupported sample type: {type(sample)}")
        if self.transform is not None:
            x = self.transform(x)
        if self.return_source and isinstance(self.dataset, Subset):
            src = int(self.dataset.indices[index]) if _TORCH_AVAILABLE else index
            return x, y, src
        return x, y, int(index)

    @property
    def targets(self) -> Any:
        return get_targets(self.dataset)


class TransformSubset(Subset if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """``Subset`` with an extra per-item transform (used for coreset views)."""

    def __init__(self, dataset: Any, indices: Sequence[int],
                 transform: Optional[Callable] = None,
                 return_index: bool = False):
        if _TORCH_AVAILABLE:
            super().__init__(dataset, list(int(i) for i in indices))
        else:  # pragma: no cover
            self.dataset = dataset
            self.indices = list(indices)
        self.transform = transform
        self.return_index = return_index

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        if self.transform is not None:
            x = self.transform(x)
        if self.return_index:
            return x, y, int(self.indices[index])
        return x, y


# ---------------------------------------------------------------------------
# Raw datasets (cached) and public factory
# ---------------------------------------------------------------------------

#: cache of raw (un-transformed) datasets keyed by (name, train, root)
CACHE: Dict[Tuple[str, bool, str], Any] = {}


def _tv_constructor(name: str):
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ImportError("torchvision is required to load the benchmarks.")
    if name == "MNIST":
        return tv_datasets.MNIST
    if name == "F-MNIST":
        return tv_datasets.FashionMNIST
    if name == "SVHN":
        return tv_datasets.SVHN
    if name == "CIFAR-10":
        return tv_datasets.CIFAR10
    raise KeyError(name)  # pragma: no cover


def _load_base(name: str, train: bool, root: str, download: bool = True) -> Any:
    """Load (and cache) the *raw* torchvision dataset without transforms."""
    key = (name, bool(train), os.path.abspath(root))
    if key in CACHE:
        return CACHE[key]
    cls = _tv_constructor(name)
    if name == "SVHN":
        base = cls(root=root, split="train" if train else "test",
                   download=download, transform=None, target_transform=None)
    else:
        base = cls(root=root, train=bool(train), download=download,
                   transform=None, target_transform=None)
    CACHE[key] = base
    return base


def _maybe_synthetic(name: str, train: bool, num_examples: int,
                     seed: int = 0) -> Any:
    """Deterministic synthetic stand-in used when a download is impossible.

    This keeps smoke tests and the unit-validation harness runnable in fully
    offline environments; the real experiments always use the true benchmarks.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("torch is required for the synthetic fallback.")
    spec = dataset_meta(name)
    gen = torch.Generator().manual_seed(int(seed) + (1 if train else 0))
    x = torch.randn((num_examples, *spec.input_shape), generator=gen)
    y = torch.randint(0, spec.num_classes, (num_examples,), generator=gen)

    class _Synthetic(Dataset):  # type: ignore[misc]
        def __init__(self):
            self.data = x
            self.targets = y.tolist()
            self.classes = [str(i) for i in range(spec.num_classes)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], int(self.targets[index])

    return _Synthetic()


def get_dataset(name: str, train: bool = True, transform: Any = "default",
                root: Optional[str] = None, download: bool = True,
                augment: bool = False, normalize: bool = True,
                verbose: bool = False, fallback_synthetic: bool = False,
                source: str = "train") -> Any:
    """Return a normalized dataset for benchmark ``name``.

    Parameters
    ----------
    name:
        Benchmark spelling; ``"MNIST-S"`` / ``"mnist_s"`` are accepted and
        resolved to the MNIST-S construction of :mod:`lbcs_repro.data.mnist_s`
        (§5.1: 1000 examples randomly sampled from MNIST).
    train:
        ``True`` for the training split, ``False`` for the test split.
    transform:
        ``"default"`` (normalized, deterministic), ``None`` (raw tensors), or
        an explicit callable.
    source:
        For SVHN, torchvision exposes ``"train"`` / ``"test"`` splits; kept for
        explicitness and future extensions.
    """
    spec = dataset_meta(name)
    root = default_root(root)

    if spec.name == "MNIST" and str(name).strip().lower() in (
        "mnist-s", "mnist_s", "MNIST-S", "MNIST_S"
    ):
        # MNIST-S is only defined for the training stage of §5.1.
        from .mnist_s import get_mnist_s  # local import: avoids a cycle

        return get_mnist_s(root=root, download=download, train=train,
                           transform=transform, augment=augment)

    if transform == "default":
        transform = build_transform(spec, train=train, normalize=normalize,
                                    augment=augment)

    try:
        base = _load_base(spec.name, train=train, root=root, download=download)
    except Exception as exc:  # pragma: no cover - offline / no network
        if not fallback_synthetic:
            raise
        if verbose:
            print(f"[data] falling back to synthetic {spec.name}: {exc}")
        base = _maybe_synthetic(spec.name, train,
                                spec.train_size or 1000 if train else spec.test_size or 200)
    return WithTransform(base, transform)


def get_loaders(name: str, batch_size: int = 128, root: Optional[str] = None,
                download: bool = True, num_workers: int = 0,
                shuffle_train: bool = True, normalize: bool = True,
                augment_test: bool = False, seed: Optional[int] = None,
                pin_memory: Optional[bool] = None, drop_last: bool = False,
                return_index: bool = False, train_subset: Optional[Sequence[int]] = None,
                **dataset_kwargs) -> Tuple[Any, Any]:
    """Build ``(train_loader, test_loader)`` for a benchmark.

    The test-loader batch order is deterministic (``shuffle=False``) so that
    repeated accuracy measurements are reproducible, which matters for the
    10/20-repeat protocol of §5.1-§5.3.
    """
    train_ds = get_dataset(name, train=True, root=root, download=download,
                           normalize=normalize, **dataset_kwargs)
    test_ds = get_dataset(name, train=False, root=root, download=download,
                          normalize=normalize, augment=augment_test,
                          **dataset_kwargs)
    if train_subset is not None:
        train_ds = subset_dataset(train_ds, train_subset, return_index=False)
    train_loader = make_loader(train_ds, batch_size=batch_size,
                               shuffle=shuffle_train, num_workers=num_workers,
                               seed=seed, pin_memory=pin_memory,
                               drop_last=drop_last, return_index=return_index)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, seed=seed,
                              pin_memory=pin_memory, drop_last=False,
                              return_index=False)
    return train_loader, test_loader


def make_loader(dataset: Any, batch_size: int = 128, shuffle: bool = False,
                num_workers: int = 0, seed: Optional[int] = None,
                pin_memory: Optional[bool] = None, drop_last: bool = False,
                return_index: bool = False,
                generator: Any = None, **kwargs) -> Any:
    """Generic DataLoader factory with deterministic worker seeding."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("torch is required to build DataLoaders.")
    ds = dataset
    if return_index and not isinstance(dataset, IndexedDataset):
        ds = IndexedDataset(dataset, return_source=True)
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available()) if torch is not None else False
    if generator is None and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(num_workers),
        pin_memory=pin_memory, drop_last=bool(drop_last),
    )
    if generator is not None:
        loader_kwargs["generator"] = generator
    loader_kwargs.update(kwargs)
    return DataLoader(ds, **loader_kwargs)


# ---------------------------------------------------------------------------
# Subsets, splits and label statistics
# ---------------------------------------------------------------------------

def _base_of(dataset: Any) -> Any:
    """Peel wrappers to reach the dataset carrying ``targets``."""
    seen = 0
    while seen < 8 and hasattr(dataset, "dataset"):
        if isinstance(dataset, (WithTransform, IndexedDataset, TransformSubset)) or \
                type(dataset).__name__ in ("Subset", "TransformSubset", "WithTransform",
                                           "IndexedDataset"):
            dataset = dataset.dataset
            seen += 1
        else:
            break
    return dataset


def get_targets(dataset: Any) -> np.ndarray:
    """Integer label array of ``dataset`` (torchvision ``targets``/``labels``).

    Supports :class:`Subset`/:class:`TransformSubset` by gathering labels of
    the selected indices, which is what the noisy/imbalanced transforms of
    :mod:`lbcs_repro.data.robustness` need in order to flip labels.
    """
    if isinstance(dataset, IndexedDataset):
        return get_targets(dataset.dataset)
    if _TORCH_AVAILABLE and isinstance(dataset, Subset):
        parent = get_targets(dataset.dataset)
        return np.asarray(parent, dtype=np.int64)[np.asarray(dataset.indices, dtype=np.int64)]
    raw = getattr(dataset, "targets", None)
    if raw is None:
        raw = getattr(dataset, "labels", None)
    if raw is None:
        raw = getattr(_base_of(dataset), "targets", None)
    if raw is None:
        raw = getattr(_base_of(dataset), "labels", None)
    if raw is None:
        # last resort: iterate (slow, but only used for exotic datasets)
        out = [int(dataset[i][1]) for i in range(len(dataset))]
        return np.asarray(out, dtype=np.int64)
    if _TORCH_AVAILABLE and torch is not None and isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    return np.asarray(raw, dtype=np.int64).reshape(-1)


def subset_dataset(dataset: Any, indices: Sequence[int],
                   transform: Optional[Callable] = None,
                   return_index: bool = False) -> Any:
    """Index view of ``dataset`` (a coreset, a split, or a debug subset)."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return TransformSubset(dataset, idx, transform=transform,
                           return_index=return_index)


def split_train_val(n: int, val_frac: float = 0.1, seed: int = 0,
                    shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/validation index split of ``n`` examples."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1).")
    idx = np.arange(int(n), dtype=np.int64)
    if shuffle:
        idx = np.random.default_rng(int(seed)).permutation(idx)
    n_val = int(round(float(val_frac) * n))
    if val_frac > 0 and n_val == 0:
        n_val = 1
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx


def stratified_indices(targets: Sequence[int], per_class: int,
                       seed: int = 0, replace: bool = False) -> np.ndarray:
    """Balanced subset: ``per_class`` examples drawn from every class."""
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    picked: List[np.ndarray] = []
    for c in np.unique(y):
        cls_idx = np.flatnonzero(y == c)
        take = min(int(per_class), cls_idx.size) if not replace else int(per_class)
        if take <= 0:
            continue
        picked.append(rng.choice(cls_idx, size=take, replace=bool(replace)))
    if not picked:  # pragma: no cover
        return np.zeros((0,), dtype=np.int64)
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out.astype(np.int64)


def class_counts(targets: Sequence[int], num_classes: Optional[int] = None) -> np.ndarray:
    """Per-class example counts."""
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size else 0
    return np.bincount(y, minlength=int(num_classes)).astype(np.int64)


def class_balanced_indices(targets: Sequence[int], ratio: float = 0.01,
                           num_classes: Optional[int] = None,
                           num_examples: Optional[int] = None,
                           seed: int = 0) -> np.ndarray:
    """Index subset with exponential class imbalance (see ``data/robustness``).

    Provided here so the dataset module is self-sufficient; the canonical
    implementation with the paper's ratio ``0.01`` lives in
    :func:`lbcs_repro.data.robustness.make_imbalanced`.
    """
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if num_classes is None:
        num_classes = int(y.max()) + 1
    rng = np.random.default_rng(int(seed))
    counts = np.array([float(ratio) ** (c / max(num_classes - 1, 1))
                       for c in range(num_classes)])
    counts = counts / counts.sum()
    if num_examples is None:
        num_examples = int(np.sum(class_counts(y, num_classes) * counts))
    total_full = int(y.size)
    base = int(np.floor(total_full * counts.min()))
    per_class = np.maximum(np.round(counts * num_examples).astype(int),
                           min(base, 1) if ratio > 0 else 1)
    per_class = per_class[::-1]  # paper's convention: class 0 most frequent
    picked: List[np.ndarray] = []
    for c in range(num_classes):
        cls_idx = np.flatnonzero(y == c)
        take = int(min(per_class[c], cls_idx.size))
        if take > 0:
            picked.append(rng.choice(cls_idx, size=take, replace=False))
    out = np.concatenate(picked) if picked else np.zeros((0,), dtype=np.int64)
    rng.shuffle(out)
    return out.astype(np.int64)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> Dict[str, Any]:  # pragma: no cover
    """Offline self-test of the registry, splits and label bookkeeping."""
    info: Dict[str, Any] = {"torch": _TORCH_AVAILABLE,
                            "torchvision": _TORCHVISION_AVAILABLE}
    assert normalize_name("fashion-mnist") == "F-MNIST"
    assert normalize_name("cifar10") == "CIFAR-10"
    assert normalize_name("svhn") == "SVHN"
    assert num_classes("F-MNIST") == 10
    assert input_shape("SVHN") == (3, 32, 32)
    tr, va = split_train_val(100, 0.1, seed=0)
    assert tr.size == 90 and va.size == 10 and set(tr).isdisjoint(set(va))
    info["split_ok"] = True

    if _TORCH_AVAILABLE:
        ds = _maybe_synthetic("CIFAR-10", True, 64, seed=1)
        assert len(ds) == 64
        y = get_targets(ds)
        assert y.shape == (64,)
        sub = subset_dataset(ds, [0, 5, 7], return_index=True)
        x, yy, idx = sub[1]
        assert int(idx) == 5 and x.shape == input_shape("CIFAR-10")
        counts = class_counts(y, 10)
        assert counts.sum() == 64
        bal = stratified_indices(y, per_class=2, seed=0)
        assert bal.size <= 20
        info["synthetic_pipeline_ok"] = True
        loader = make_loader(ds, batch_size=8, shuffle=False, return_index=True,
                             num_workers=0)
        xb, yb, ib = next(iter(loader))
        assert xb.shape[0] == 8 and ib.shape[0] == 8
        info["loader_triplet_ok"] = True
    if verbose:
        print("[data.datasets] selftest:", info)
    return info


if __name__ == "__main__":  # pragma: no cover
    _selftest()
