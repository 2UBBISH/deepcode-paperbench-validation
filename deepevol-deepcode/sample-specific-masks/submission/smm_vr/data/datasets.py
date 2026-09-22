"""Target-task dataset loading for SMM (Sample-specific Multi-channel Masks for VR).

This module implements the data side of the reproduction plan (component 2):

* a metadata registry transcribed from **Appendix C, Table 6** of the paper
  (original image size, training-set size, testing-set size, number of classes),
* builders that instantiate the 11 main target datasets (plus ``stanfordcars``,
  the fine-grained failure case of Table 12) through ``torchvision.datasets``,
* the *exact* addendum preprocessing pipelines
  (:mod:`smm_vr.data.transforms`): ``imgsize = 384`` for ``ViT_B32`` else ``224``,
  ``Resize((imgsize + 32, imgsize + 32))`` + ``RandomCrop(imgsize)`` +
  ``RandomHorizontalFlip`` + RGB + ``ToTensor`` + ImageNet normalization for
  training, and ``Resize((imgsize, imgsize))`` + RGB + ``ToTensor`` + ImageNet
  normalization for testing,
* dataloader construction with the per-dataset batch sizes of Table 9
  (``b = 256`` everywhere except ``DTD`` and ``OxfordPets`` where ``b = 64``).

Split policy
------------
The paper states: "We follow Chen et al. (2023) to split the datasets."
Chen et al. (2023) re-sampled several of the original splits, so a few of the
Table 6 counts differ slightly from what plain ``torchvision`` returns.  We
therefore expose a ``split_policy`` per dataset:

``"native"``
    use whatever split the torchvision constructor produces (e.g.
    ``Flowers102(split="train"/"test")``, ``SVHN(split="train"/"test")``).
``"ratio"``
    the dataset ships without a train/test separation (EuroSAT, SUN397,
    StanfordCars via ImageFolder-style roots); we perform a deterministic,
    class-balanced split (``ratio=0.5`` by default, matching Table 6's
    ``13500 / 8100`` EuroSAT and ``15888 / 19850`` SUN397 counts).

All splits are deterministic (fixed seed) so the three experiment seeds of the
paper only vary model initialization / mask-generator init.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets

from .transforms import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_test_transform,
    build_train_transform,
    get_image_size,
)

__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "MAIN_DATASETS",
    "ALL_DATASETS",
    "DATASET_ALIASES",
    "DEFAULT_BATCH_SIZES",
    "list_datasets",
    "canonical_name",
    "get_dataset_spec",
    "dataset_image_size",
    "num_classes",
    "resolve_dataset_root",
    "build_train_dataset",
    "build_test_dataset",
    "build_datasets",
    "build_dataloaders",
    "targets_of",
    "describe_dataset",
]


# ---------------------------------------------------------------------------
# Metadata: Appendix C, Table 6 (plus StanfordCars for Appendix D.4 / Table 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a target task (Appendix C, Table 6)."""

    name: str
    num_classes: int
    original_size: int
    """Original (native) image side length in pixels (32 or 128 for these tasks)."""
    reference_train_size: int
    """Training-set size reported in Table 6 (for validation / logging)."""
    reference_test_size: int
    """Testing-set size reported in Table 6 (for validation / logging)."""
    torchvision_name: str
    """Fully qualified torchvision dataset class name used to materialise it."""
    split_policy: str = "native"
    """``"native"`` (constructor provides the split) or ``"ratio"``."""
    domain_hint: str = ""
    notes: str = ""

    @property
    def total_size(self) -> int:
        return self.reference_train_size + self.reference_test_size


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        name="cifar10",
        num_classes=10,
        original_size=32,
        reference_train_size=50000,
        reference_test_size=10000,
        torchvision_name="CIFAR10",
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        num_classes=100,
        original_size=32,
        reference_train_size=50000,
        reference_test_size=10000,
        torchvision_name="CIFAR100",
    ),
    "svhn": DatasetSpec(
        name="svhn",
        num_classes=10,
        original_size=32,
        reference_train_size=73257,
        reference_test_size=26032,
        torchvision_name="SVHN",
        notes="torchvision 'train' split (73257) is larger than the 'extra' one.",
    ),
    "gtsrb": DatasetSpec(
        name="gtsrb",
        num_classes=43,
        original_size=32,
        reference_train_size=39209,
        reference_test_size=12630,
        torchvision_name="GTSRB",
    ),
    "flowers102": DatasetSpec(
        name="flowers102",
        num_classes=102,
        original_size=128,
        reference_train_size=4093,
        reference_test_size=2463,
        torchvision_name="Flowers102",
        notes="splits: train=1020, val=1020, test=6149; train+val=2040 images. "
        "Table 6 count (4093) corresponds to re-sampled 80/20 split of all 6149.",
        split_policy="ratio",
    ),
    "dtd": DatasetSpec(
        name="dtd",
        num_classes=47,
        original_size=128,
        reference_train_size=2820,
        reference_test_size=1692,
        torchvision_name="DTD",
        notes="splits: train=val=test=1880. Table 6 count (2820) corresponds to "
        "re-sampled 62.5/37.5 split of all 5640 images.",
        split_policy="ratio",
    ),
    "ucf101": DatasetSpec(
        name="ucf101",
        num_classes=101,
        original_size=128,
        reference_train_size=7639,
        reference_test_size=3783,
        torchvision_name="UCF101",
        notes="requires annotation files (ucfTrainTestlist); split 1 by default.",
    ),
    "food101": DatasetSpec(
        name="food101",
        num_classes=101,
        original_size=128,
        reference_train_size=50500,
        reference_test_size=30300,
        torchvision_name="Food101",
    ),
    "sun397": DatasetSpec(
        name="sun397",
        num_classes=397,
        original_size=128,
        reference_train_size=15888,
        reference_test_size=19850,
        torchvision_name="SUN397",
        split_policy="ratio",
        notes="torchvision SUN397 exposes Train.txt/Test.txt via _split_list; we "
        "use it when available and fall back to a deterministic 50/50 split.",
    ),
    "eurosat": DatasetSpec(
        name="eurosat",
        num_classes=10,
        original_size=128,
        reference_train_size=13500,
        reference_test_size=8100,
        torchvision_name="EuroSAT",
        split_policy="ratio",
        notes="single 27000-image set; deterministic class-balanced 50/50 split.",
    ),
    "oxfordpets": DatasetSpec(
        name="oxfordpets",
        num_classes=37,
        original_size=128,
        reference_train_size=2944,
        reference_test_size=3669,
        torchvision_name="OxfordIIITPet",
        split_policy="ratio",
        notes="trainval=3680, test=3669. Table 6 train count (2944) = 80% of trainval.",
    ),
    # --- Appendix D.4 / Table 12: the fine-grained failure case -------------
    "stanfordcars": DatasetSpec(
        name="stanfordcars",
        num_classes=196,
        original_size=128,
        reference_train_size=8144,
        reference_test_size=8041,
        torchvision_name="StanfordCars",
        domain_hint="fine-grained failure case of VR (Appendix D.4, Table 12)",
    ),
}

MAIN_DATASETS: Tuple[str, ...] = (
    "cifar10",
    "cifar100",
    "svhn",
    "gtsrb",
    "flowers102",
    "dtd",
    "ucf101",
    "food101",
    "sun397",
    "eurosat",
    "oxfordpets",
)

ALL_DATASETS: Tuple[str, ...] = MAIN_DATASETS + ("stanfordcars",)

#: Accept the many spellings used across the paper / torchvision.
DATASET_ALIASES: Dict[str, str] = {
    "cifar10": "cifar10",
    "cifar_10": "cifar10",
    "cifar-10": "cifar10",
    "cifar100": "cifar100",
    "cifar_100": "cifar100",
    "cifar-100": "cifar100",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "flowers102": "flowers102",
    "flowers_102": "flowers102",
    "oxfordflowers102": "flowers102",
    "dtd": "dtd",
    "describabletextures": "dtd",
    "ucf101": "ucf101",
    "ucf-101": "ucf101",
    "food101": "food101",
    "food-101": "food101",
    "sun397": "sun397",
    "sun-397": "sun397",
    "eurosat": "eurosat",
    "oxfordpets": "oxfordpets",
    "oxford_pets": "oxfordpets",
    "oxford-iiit-pet": "oxfordpets",
    "oxfordiiitpet": "oxfordpets",
    "pets": "oxfordpets",
    "stanfordcars": "stanfordcars",
    "stanford_cars": "stanfordcars",
    "cars": "stanfordcars",
}

#: Batch sizes of Appendix C, Table 9 (``b`` column).
DEFAULT_BATCH_SIZES: Dict[str, int] = {name: 256 for name in ALL_DATASETS}
DEFAULT_BATCH_SIZES["dtd"] = 64
DEFAULT_BATCH_SIZES["oxfordpets"] = 64


def canonical_name(name: str) -> str:
    """Normalise a dataset name (upper case, spaces, dashes) to the registry key."""
    if name is None:
        raise ValueError("dataset name must not be None")
    key = str(name).strip().lower().replace(" ", "").replace("_", "")
    if key in DATASET_SPECS:
        return key
    for alias, target in DATASET_ALIASES.items():
        if alias.replace("_", "").replace("-", "") == key:
            return target
    if name in DATASET_SPECS:
        return name
    raise KeyError(
        f"Unknown dataset {name!r}. Known datasets: {sorted(DATASET_SPECS)}"
    )


def list_datasets(include_extra: bool = False) -> List[str]:
    """Return the 11 main target tasks (plus StanfordCars when requested)."""
    return list(MAIN_DATASETS) + (["stanfordcars"] if include_extra else [])


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` (Table 6 metadata) for ``name``."""
    return DATASET_SPECS[canonical_name(name)]


def dataset_image_size(name: str) -> int:
    """Native image side length of the target dataset (32 or 128, Table 6)."""
    return get_dataset_spec(name).original_size


def num_classes(name: str) -> int:
    """Number of target classes (Table 6)."""
    return get_dataset_spec(name).num_classes


def resolve_dataset_root(name: str, root: str, data_root: Optional[str] = None) -> str:
    """Resolve ``<data_root or root>/<name>`` as the on-disk dataset directory."""
    base = data_root if data_root else root
    if base is None:
        base = os.environ.get("SMM_DATA_ROOT", os.path.join(os.getcwd(), "data"))
    return os.path.join(base, canonical_name(name))


# ---------------------------------------------------------------------------
# Deterministic subsetting utilities
# ---------------------------------------------------------------------------


class SubsetWithTargets(Dataset):
    """``torch.utils.data.Subset`` that keeps ``targets``/``labels`` accessible.

    Downstream code (label mapping, t-SNE analysis, per-class reporting) needs
    the target labels of a subset; the stock ``Subset`` hides them.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        parent_targets = targets_of(dataset, allow_none=True)
        if parent_targets is not None:
            self.targets = [parent_targets[i] for i in self.indices]
            self.labels = self.targets
        self.classes = getattr(dataset, "classes", None)
        self.transform = getattr(dataset, "transform", None)
        self.target_transform = getattr(dataset, "target_transform", None)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __getattr__(self, item: str) -> Any:
        # Delegate unknown attributes (e.g. ``root``, ``_split_list``) to parent.
        try:
            dataset = self.__dict__["dataset"]
        except KeyError:  # pragma: no cover - during unpickling
            raise AttributeError(item)
        return getattr(dataset, item)


def targets_of(dataset: Dataset, allow_none: bool = False) -> Optional[List[int]]:
    """Best-effort extraction of integer target labels from a dataset."""
    for attr in ("targets", "labels", "_labels"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return [int(v) for v in value]
    if isinstance(dataset, Subset):
        return targets_of(dataset.dataset, allow_none=True)
    if hasattr(dataset, "samples"):
        try:
            return [int(s[1]) for s in dataset.samples]
        except (TypeError, ValueError, IndexError):
            pass
    if hasattr(dataset, "imgs"):
        try:
            return [int(s[1]) for s in dataset.imgs]
        except (TypeError, ValueError, IndexError):
            pass
    if allow_none:
        return None
    raise ValueError(
        f"Cannot determine targets for {type(dataset).__name__}; "
        "wrap it or expose a `.targets` attribute."
    )


def _class_balanced_split(
    labels: Sequence[int],
    train_ratio: float,
    seed: int,
    num_classes: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Deterministic, per-class shuffled split producing an exact train ratio."""
    import random

    rng = random.Random(seed)
    n_classes = num_classes if num_classes else (max(labels) + 1 if labels else 0)
    by_class: Dict[int, List[int]] = {c: [] for c in range(n_classes)}
    for idx, lab in enumerate(labels):
        by_class.setdefault(int(lab), []).append(idx)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for class_id in sorted(by_class):
        members = list(by_class[class_id])
        rng.shuffle(members)
        n_train = int(round(len(members) * float(train_ratio)))
        if members:
            n_train = min(max(n_train, 0), len(members))
        train_idx.extend(members[:n_train])
        test_idx.extend(members[n_train:])
    train_idx.sort()
    test_idx.sort()
    return train_idx, test_idx


def _split_indices_from_list(
    filenames: Iterable[str], split_list: Dict[str, Iterable[str]]
) -> Optional[Tuple[List[int], List[int]]]:
    """Map a ``SUN397``-style ``_split_list`` onto dataset indices."""
    filenames = [os.path.basename(str(f)) for f in filenames]
    lookup: Dict[str, str] = {}
    for split_name, entries in split_list.items():
        for entry in entries:
            lookup[os.path.basename(str(entry))] = str(split_name)
    if not lookup:
        return None
    train_idx, test_idx = [], []
    for idx, fname in enumerate(filenames):
        split_name = lookup.get(fname)
        if split_name is None:
            continue
        if split_name.lower().startswith("train"):
            train_idx.append(idx)
        else:
            test_idx.append(idx)
    if not train_idx or not test_idx:
        return None
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# torchvision constructors
# ---------------------------------------------------------------------------


def _torchvision_class(spec: DatasetSpec) -> Callable[..., Dataset]:
    cls = getattr(tv_datasets, spec.torchvision_name, None)
    if cls is None:  # pragma: no cover - depends on torchvision version
        raise ImportError(
            f"torchvision.datasets.{spec.torchvision_name} is not available in this "
            "torchvision version; upgrade torchvision or provide a custom root."
        )
    return cls


def _make_native_dataset(
    spec: DatasetSpec,
    split: str,
    root: str,
    transform: Callable,
    download: bool,
    ucf101_fold: int = 1,
    annotations_path: Optional[str] = None,
) -> Dataset:
    """Instantiate one native split of a target dataset via torchvision."""
    cls = _torchvision_class(spec)
    name = spec.name

    if name in ("cifar10", "cifar100"):
        return cls(root=root, train=(split == "train"), transform=transform, download=download)

    if name == "svhn":
        # torchvision only offers "train" (73257) and "test" (26032).
        return cls(root=root, split=split, transform=transform, download=download)

    if name == "gtsrb":
        return cls(root=root, split=split, transform=transform, download=download)

    if name == "flowers102":
        # torchvision splits: "train" (1020), "val" (1020), "test" (6149).
        # We keep the native naming available; the default ratio policy below
        # re-samples the full 6149 images when split_policy == "ratio".
        return cls(root=root, split=split, transform=transform, download=download)

    if name == "dtd":
        return cls(root=root, split=split, transform=transform, download=download)

    if name == "ucf101":
        ann = annotations_path or os.path.join(root, "ucfTrainTestlist")
        return cls(
            root=root,
            annotation_path=ann,
            frames_per_clip=1,
            step_between_clips=1,
            fold=ucf101_fold,
            train=(split == "train"),
            transform=transform,
            download=download,
        )

    if name == "food101":
        return cls(root=root, split=split, transform=transform, download=download)

    if name == "sun397":
        return cls(root=root, transform=transform, download=download)

    if name == "eurosat":
        return cls(root=root, transform=transform, download=download)

    if name == "oxfordpets":
        tv_split = "trainval" if split == "train" else "test"
        return cls(
            root=root,
            split=tv_split,
            target_types="category",
            transform=transform,
            download=download,
        )

    if name == "stanfordcars":
        return cls(root=root, split=split, transform=transform, download=download)

    raise KeyError(f"No torchvision builder registered for dataset {name!r}")


def _build_split_datasets(
    name: str,
    backbone: Optional[str],
    root: str,
    imgsize: Optional[int],
    download: bool,
    split_policy: Optional[str],
    train_ratio: Optional[float],
    split_seed: int,
    train_fraction: Optional[float],
    ucf101_fold: int,
    annotations_path: Optional[str],
) -> Tuple[Dataset, Dataset, DatasetSpec]:
    """Materialise ``(train_dataset, test_dataset, spec)`` with exact transforms."""
    spec = get_dataset_spec(name)
    imgsize = get_image_size(backbone, imgsize)
    train_tf = build_train_transform(imgsize=imgsize)
    test_tf = build_test_transform(imgsize=imgsize)
    policy = split_policy or spec.split_policy

    if policy == "ratio":
        # A single torchvision object (or native train+val) is split afterwards.
        if spec.name == "sun397":
            full = _make_native_dataset(
                spec, "train", root, None, download, annotations_path=annotations_path
            )
            files = [s[0] for s in getattr(full, "samples", [])]
            split_list = getattr(full, "_split_list", None)
            base = full.dataset if isinstance(full, Subset) else full
            if split_list is None:
                split_list = getattr(base, "_split_list", None)
            auto = (
                _split_indices_from_list(files, split_list)
                if (split_list and files)
                else None
            )
            if auto is not None:
                train_ds = SubsetWithTargets(full, auto[0])
                test_ds = SubsetWithTargets(full, auto[1])
            else:
                labels = targets_of(full)
                tr, te = _class_balanced_split(
                    labels, train_ratio if train_ratio else 0.5, split_seed, spec.num_classes
                )
                train_ds = SubsetWithTargets(full, tr)
                test_ds = SubsetWithTargets(full, te)
        else:
            if spec.name in ("eurosat",):
                full = _make_native_dataset(spec, "train", root, None, download)
            elif spec.name == "flowers102":
                full = _make_native_dataset(spec, "train", root, None, download)
                extra = _make_native_dataset(spec, "test", root, None, download)
                full = ConcatDatasetWithTargets([full, extra])
            elif spec.name == "dtd":
                full = _make_native_dataset(spec, "train", root, None, download)
                extra = _make_native_dataset(spec, "test", root, None, download)
                extra2 = _make_native_dataset(spec, "val", root, None, download)
                full = ConcatDatasetWithTargets([full, extra, extra2])
            elif spec.name == "oxfordpets":
                full = _make_native_dataset(spec, "train", root, None, download)
            else:
                full = _make_native_dataset(spec, "train", root, None, download)
            labels = targets_of(full)
            ratio = train_ratio if train_ratio else 0.5
            tr, te = _class_balanced_split(labels, ratio, split_seed, spec.num_classes)
            train_ds = SubsetWithTargets(full, tr)
            test_ds = SubsetWithTargets(full, te)
    else:
        train_ds = _make_native_dataset(
            spec, "train", root, None, download,
            ucf101_fold=ucf101_fold, annotations_path=annotations_path,
        )
        test_ds = _make_native_dataset(
            spec, "test", root, None, download,
            ucf101_fold=ucf101_fold, annotations_path=annotations_path,
        )

    # Stamp the exact addendum transforms.
    _set_transform(train_ds, train_tf)
    _set_transform(test_ds, test_tf)

    if train_fraction is not None and 0.0 < float(train_fraction) < 1.0:
        labels = targets_of(train_ds)
        tr, _ = _class_balanced_split(
            labels, float(train_fraction), split_seed, spec.num_classes
        )
        train_ds = SubsetWithTargets(train_ds, tr)

    return train_ds, test_ds, spec


class ConcatDatasetWithTargets(Dataset):
    """``ConcatDataset`` that propagates ``targets``/``classes`` from children."""

    def __init__(self, datasets: Sequence[Dataset]):
        from torch.utils.data import ConcatDataset

        self.datasets = list(datasets)
        self._concat = ConcatDataset(self.datasets)
        self.classes = getattr(self.datasets[0], "classes", None)
        self.targets: List[int] = []
        for ds in self.datasets:
            self.targets.extend(targets_of(ds))
        self.labels = self.targets
        self.transform = getattr(self.datasets[0], "transform", None)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._concat)

    def __getitem__(self, index: int):
        return self._concat[index]


def _set_transform(dataset: Dataset, transform: Callable) -> None:
    """Recursively assign ``transform`` (the addendum pipeline) to a dataset."""
    if isinstance(dataset, (Subset, SubsetWithTargets)):
        _set_transform(dataset.dataset, transform)
        dataset.transform = transform
        return
    if isinstance(dataset, Dataset):
        if hasattr(dataset, "transform"):
            dataset.transform = transform
    for attr in ("datasets", "dataset"):
        child = getattr(dataset, attr, None)
        if child is not None:
            _set_transform(child, transform)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_train_dataset(
    name: str,
    backbone: Optional[str] = None,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    imgsize: Optional[int] = None,
    transform: Optional[Callable] = None,
    download: bool = True,
    split_policy: Optional[str] = None,
    train_ratio: Optional[float] = None,
    split_seed: int = 0,
    train_fraction: Optional[float] = None,
    ucf101_fold: int = 1,
    annotations_path: Optional[str] = None,
) -> Tuple[Dataset, DatasetSpec]:
    """Build the training split of ``name`` with the addendum train transform."""
    data_dir = resolve_dataset_root(name, root or "", data_root)
    spec = get_dataset_spec(name)
    if transform is not None:
        # Caller supplied a custom pipeline (kept for debugging); still split.
        train_ds, _, spec = _build_split_datasets(
            name, backbone, data_dir, imgsize, download, split_policy,
            train_ratio, split_seed, train_fraction, ucf101_fold, annotations_path,
        )
        _set_transform(train_ds, transform)
        return train_ds, spec
    train_ds, _, spec = _build_split_datasets(
        name, backbone, data_dir, imgsize, download, split_policy,
        train_ratio, split_seed, train_fraction, ucf101_fold, annotations_path,
    )
    return train_ds, spec


def build_test_dataset(
    name: str,
    backbone: Optional[str] = None,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    imgsize: Optional[int] = None,
    transform: Optional[Callable] = None,
    download: bool = True,
    split_policy: Optional[str] = None,
    train_ratio: Optional[float] = None,
    split_seed: int = 0,
    ucf101_fold: int = 1,
    annotations_path: Optional[str] = None,
) -> Tuple[Dataset, DatasetSpec]:
    """Build the test split of ``name`` with the addendum test transform."""
    data_dir = resolve_dataset_root(name, root or "", data_root)
    spec = get_dataset_spec(name)
    _, test_ds, spec = _build_split_datasets(
        name, backbone, data_dir, imgsize, download, split_policy,
        train_ratio, split_seed, None, ucf101_fold, annotations_path,
    )
    if transform is not None:
        _set_transform(test_ds, transform)
    return test_ds, spec


def build_datasets(
    name: str,
    backbone: Optional[str] = None,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    imgsize: Optional[int] = None,
    download: bool = True,
    split_policy: Optional[str] = None,
    train_ratio: Optional[float] = None,
    split_seed: int = 0,
    train_fraction: Optional[float] = None,
    ucf101_fold: int = 1,
    annotations_path: Optional[str] = None,
) -> Tuple[Dataset, Dataset, DatasetSpec]:
    """Build ``(train_dataset, test_dataset, spec)`` for a target task.

    ``backbone`` selects the resolution of the addendum transforms
    (``ViT_B32`` -> 384, otherwise 224) and ``root``/``data_root`` the dataset
    directory.  ``train_fraction`` optionally subsamples the training split in a
    class-balanced, deterministic way (useful for quick debugging runs).
    """
    data_dir = resolve_dataset_root(name, root or "", data_root)
    return _build_split_datasets(
        name, backbone, data_dir, imgsize, download, split_policy,
        train_ratio, split_seed, train_fraction, ucf101_fold, annotations_path,
    )


def build_dataloaders(
    name: str,
    backbone: Optional[str] = None,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    imgsize: Optional[int] = None,
    batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
    shuffle_train: bool = True,
    split_policy: Optional[str] = None,
    train_ratio: Optional[float] = None,
    split_seed: int = 0,
    train_fraction: Optional[float] = None,
    ucf101_fold: int = 1,
    annotations_path: Optional[str] = None,
    drop_last: bool = False,
    device: Optional[str] = None,
    generator_seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DatasetSpec]:
    """Build train/test :class:`~torch.utils.data.DataLoader` for a target task.

    Default batch sizes follow Appendix C, Table 9: ``256`` for every dataset
    except ``DTD`` and ``OxfordPets``, which use ``64``.
    """
    spec = get_dataset_spec(name)
    train_ds, test_ds, spec = build_datasets(
        name,
        backbone=backbone,
        root=root,
        data_root=data_root,
        imgsize=imgsize,
        download=download,
        split_policy=split_policy,
        train_ratio=train_ratio,
        split_seed=split_seed,
        train_fraction=train_fraction,
        ucf101_fold=ucf101_fold,
        annotations_path=annotations_path,
    )
    bs = int(batch_size) if batch_size else DEFAULT_BATCH_SIZES[spec.name]
    tbs = int(test_batch_size) if test_batch_size else bs

    if device is None:
        use_pin = pin_memory and torch.cuda.is_available()
    else:
        use_pin = pin_memory and str(device).startswith("cuda")

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=bool(num_workers > 0),
    )
    if generator_seed is not None:
        g = torch.Generator()
        g.manual_seed(int(generator_seed))
        loader_kwargs["generator"] = g

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=shuffle_train,
        drop_last=drop_last,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tbs,
        shuffle=False,
        drop_last=False,
        **{**loader_kwargs, "persistent_workers": False},
    )
    return train_loader, test_loader, spec


def describe_dataset(name: str) -> Dict[str, Any]:
    """Return the Table 6 metadata of ``name`` as a plain dict (for logging)."""
    spec = get_dataset_spec(name)
    return {
        "name": spec.name,
        "num_classes": spec.num_classes,
        "original_image_size": f"{spec.original_size}x{spec.original_size}",
        "train_size": spec.reference_train_size,
        "test_size": spec.reference_test_size,
        "torchvision_class": spec.torchvision_name,
        "split_policy": spec.split_policy,
        "notes": spec.notes,
    }
