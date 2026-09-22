"""The 11 target datasets of the paper, with the paper's splits and transforms.

Paper reference: Section 5 and Appendix C (Table 6 = dataset sizes, Table 9 =
optimisation settings).  The train/test transforms are taken verbatim from the
paper's addendum:

.. code-block:: python

    train_preprocess = Compose([Resize((imgsize + 32, imgsize + 32)),
                                RandomCrop(imgsize), RandomHorizontalFlip(),
                                Lambda(to_RGB), ToTensor(),
                                Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    test_preprocess  = Compose([Resize((imgsize, imgsize)), Lambda(to_RGB),
                                ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])

with ``imgsize = 384`` for ViT-B/32 and ``224`` otherwise.

Splits
------
Table 6 gives the training/testing set size of every dataset.  Those sizes come
from the benchmark splits used by Chen et al. (2023); when the corresponding
split files are available they are used verbatim.  The expected format of a
split file (``split_dir`` argument, files ``<dataset>_train.txt`` /
``<dataset>_test.txt``) is one *pool index* per line, which is also the format
produced by this module when it caches the splits it built.  Otherwise the
splits are rebuilt deterministically: images are grouped by class and each class
contributes a proportional share of the requested train and test sizes (largest
remainder, fixed seed 0).  The realised sizes are then within a few samples of
Table 6 -- ``scripts/make_splits.py`` prints them and caches the result under
``data/splits``.

``gtsrb`` is the one dataset whose *official* split is used directly (39,209
training / 12,630 testing images, exactly Table 6).  torchvision ships a
reduced GTSRB training set (26,640 images); this module parses the official
archive when it is present and otherwise falls back to torchvision with a
warning.
"""

from __future__ import annotations

import csv
import os
import pathlib
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DATASET_SPECS",
    "DatasetSpec",
    "build_transforms",
    "build_dataset",
    "split_sizes",
    "deterministic_split",
]

# ImageNet statistics used by the paper (addendum).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target task specifications, Table 6 of the paper.
@dataclass
class DatasetSpec:
    name: str
    num_classes: int
    resolution: int
    train_size: int
    test_size: int
    batch_size: int = 256
    lr: float = 0.01
    notes: str = ""


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec("cifar10", 10, 32, 50000, 10000, 256, 0.01),
    "cifar100": DatasetSpec("cifar100", 100, 32, 50000, 10000, 256, 0.01),
    "svhn": DatasetSpec("svhn", 10, 32, 73257, 26032, 256, 0.01),
    "gtsrb": DatasetSpec("gtsrb", 43, 32, 39209, 12630, 256, 0.01, "official split"),
    "flowers102": DatasetSpec("flowers102", 102, 128, 4093, 2463, 256, 0.01),
    "dtd": DatasetSpec("dtd", 47, 128, 2820, 1692, 64, 0.01),
    "ucf101": DatasetSpec("ucf101", 101, 128, 7639, 3783, 256, 0.01),
    "food101": DatasetSpec("food101", 101, 128, 50500, 30300, 256, 0.01),
    "sun397": DatasetSpec("sun397", 397, 128, 15888, 19850, 256, 0.01),
    "eurosat": DatasetSpec("eurosat", 10, 128, 13500, 8100, 256, 0.01),
    "oxfordpets": DatasetSpec("oxfordpets", 37, 128, 2944, 3669, 64, 0.01),
}


def _to_rgb(x):
    """``Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x)``."""
    return x.convert("RGB") if hasattr(x, "convert") else x


def build_transforms(image_size: int, train: bool) -> Callable:
    """Train/test transforms of the paper (see module docstring)."""
    image_size = int(image_size)
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(_to_rgb),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(_to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Generic index-based pooling / subsetting of torchvision datasets
# --------------------------------------------------------------------------- #
class IndexedPool(Dataset):
    """A dataset defined by ``(base_dataset, index)`` references."""

    def __init__(self, refs: Sequence[Tuple[Dataset, int]], targets: Sequence[int],
                 transform: Optional[Callable] = None, name: str = ""):
        self.refs = list(refs)
        self.targets = list(targets)
        self.transform = transform
        self.name = name

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        base, idx = self.refs[index]
        item = base[idx]
        if isinstance(item, (tuple, list)):
            image, label = item[0], int(item[1])
        else:  # pragma: no cover - defensive
            image, label = item, int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _targets_of(base: Dataset) -> List[int]:
    """Best-effort extraction of the labels of a torchvision dataset."""
    for attr in ("targets", "labels", "_labels"):
        value = getattr(base, attr, None)
        if value is not None and len(value):
            return [int(v) for v in value]
    samples = getattr(base, "_samples", None)
    if samples:
        return [int(s[1]) for s in samples]
    raise ValueError(f"cannot extract labels from {type(base).__name__}")


def _concat(datasets: Sequence[Dataset]) -> Tuple[List[Tuple[Dataset, int]], List[int]]:
    refs: List[Tuple[Dataset, int]] = []
    targets: List[int] = []
    for ds in datasets:
        tg = _targets_of(ds)
        refs.extend((ds, i) for i in range(len(ds)))
        targets.extend(tg)
    return refs, targets


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def deterministic_split(
    targets: Sequence[int],
    num_train: int,
    num_test: int,
    seed: int = 0,
    num_classes: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Class-balanced deterministic train/test index split.

    Each class contributes ``num_train * n_c / N`` training images and
    ``num_test * n_c / N`` testing images (largest remainder rounding), which
    mirrors the class-balance of the benchmark splits while matching the total
    sizes of Table 6.
    """
    targets = [int(t) for t in targets]
    n = len(targets)
    if num_classes is None:
        num_classes = max(targets) + 1
    if num_train + num_test > n:
        raise ValueError("requested more images than available")

    by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for i, t in enumerate(targets):
        by_class.setdefault(t, []).append(i)

    sizes = {c: len(v) for c, v in by_class.items()}

    def allocate(total: int) -> Dict[int, int]:
        ideal = {c: total * sizes[c] / n for c in sizes}
        alloc = {c: int(ideal[c]) for c in ideal}
        remainder = total - sum(alloc.values())
        order = sorted(ideal, key=lambda c: (-(ideal[c] - alloc[c]), c))
        for c in order[:remainder]:
            alloc[c] += 1
        return alloc

    alloc_train = allocate(num_train)
    alloc_test = allocate(num_test)

    g = torch.Generator().manual_seed(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for c in sorted(by_class):
        idx = by_class[c]
        perm = torch.randperm(len(idx), generator=g).tolist()
        shuffled = [idx[k] for k in perm]
        n_tr = min(alloc_train.get(c, 0), len(shuffled))
        n_te = min(alloc_test.get(c, 0), len(shuffled) - n_tr)
        train_idx.extend(shuffled[:n_tr])
        test_idx.extend(shuffled[n_tr:n_tr + n_te])
    return train_idx, test_idx


def split_sizes(name: str) -> Tuple[int, int, int]:
    """``(train_size, test_size, num_classes)`` as reported in Table 6."""
    spec = DATASET_SPECS[name]
    return spec.train_size, spec.test_size, spec.num_classes


# --------------------------------------------------------------------------- #
# Raw dataset construction
# --------------------------------------------------------------------------- #
def _raw_pool(name: str, root: str, download: bool) -> Tuple[List[Tuple[Dataset, int]], List[int]]:
    """Return the full labelled pool of a dataset as ``(refs, targets)``."""
    import torchvision.datasets as tvd

    root = os.path.expanduser(root)

    if name in {"cifar10", "cifar100"}:
        cls = tvd.CIFAR10 if name == "cifar10" else tvd.CIFAR100
        train = cls(root, train=True, download=download)
        test = cls(root, train=False, download=download)
        return _concat([train, test])

    if name == "svhn":
        train = tvd.SVHN(root, split="train", download=download)
        test = tvd.SVHN(root, split="test", download=download)
        return _concat([train, test])

    if name == "gtsrb":
        return _gtsrb_pool(root, download)

    if name == "flowers102":
        parts = [tvd.Flowers102(root, split=s, download=download) for s in ("train", "val", "test")]
        return _concat(parts)

    if name == "dtd":
        parts = [tvd.DTD(root, split=s, download=download) for s in ("train", "val", "test")]
        return _concat(parts)

    if name == "food101":
        parts = [tvd.Food101(root, split=s, download=download) for s in ("train", "test")]
        return _concat(parts)

    if name == "sun397":
        parts = [tvd.SUN397(root, download=download)]
        return _concat(parts)

    if name == "eurosat":
        parts = [tvd.EuroSAT(root, download=download)]
        return _concat(parts)

    if name == "oxfordpets":
        parts = [tvd.OxfordIIITPet(root, split=s, download=download) for s in ("trainval", "test")]
        return _concat(parts)

    if name == "ucf101":
        return _ucf101_pool(root, download)

    raise ValueError(f"unknown dataset {name!r}")


def _gtsrb_pool(root: str, download: bool) -> Tuple[List[Tuple[Dataset, int]], List[int]]:
    """GTSRB: official 39,209/12,630 split when the archive is available."""
    import torchvision.datasets as tvd

    base = pathlib.Path(root) / "gtsrb" / "GTSRB"
    train_images = base / "Final_Training" / "Images"

    if train_images.is_dir():
        refs: List[Tuple[Dataset, int]] = []
        targets: List[int] = []
        files = sorted(train_images.glob("*/GT-*.csv"))
        for gt in files:
            with open(gt, newline="") as fh:
                reader = csv.reader(fh, delimiter=";")
                next(reader, None)
                for row in reader:
                    if len(row) < 8:
                        continue
                    img = gt.parent / row[0]
                    refs.append((_ImagePathDataset(img), 0))
                    targets.append(int(row[7]))
        if refs:
            test_refs, test_targets = _gtsrb_official_test(base)
            refs.extend(test_refs)
            targets.extend(test_targets)
            if len(refs) < 40000:
                warnings.warn(
                    f"GTSRB: found {len(refs)} official images; torchvision's reduced "
                    "archive is in use. Download the official GTSRB archives to match "
                    "Table 6 (39,209 train / 12,630 test).",
                    UserWarning,
                )
            return refs, targets

    warnings.warn(
        "GTSRB: official archive not found, falling back to torchvision's reduced "
        "training split (26,640 images instead of 39,209).",
        UserWarning,
    )
    train = tvd.GTSRB(root, split="train", download=download)
    test = tvd.GTSRB(root, split="test", download=download)
    return _concat([train, test])


def _gtsrb_official_test(base: pathlib.Path) -> Tuple[List[Tuple[Dataset, int]], List[int]]:
    test_dir = base / "Final_Test" / "Images"
    gt = base / "GT-final_test.csv"
    alt = base / "Final_Test" / "GT-final_test.csv"
    refs: List[Tuple[Dataset, int]] = []
    targets: List[int] = []
    csv_path = gt if gt.is_file() else alt
    if test_dir.is_dir() and csv_path.is_file():
        with open(csv_path, newline="", encoding="latin-1") as fh:
            reader = csv.reader(fh, delimiter=";")
            next(reader, None)
            for row in reader:
                if len(row) < 8:
                    continue
                refs.append((_ImagePathDataset(test_dir / row[0]), 0))
                targets.append(int(row[7]))
    return refs, targets


class _ImagePathDataset(Dataset):
    """Single-image holder so that the pool machinery can index it."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.path) as img:
            return img.copy(), 0


def _ucf101_pool(root: str, download: bool) -> Tuple[List[Tuple[Dataset, int]], List[int]]:
    """UCF101 treated as an image dataset (one centre frame per video, fold 1)."""
    import torchvision.datasets as tvd

    annotation = pathlib.Path(root) / "ucf101" / "ucfTrainTestlist"
    try:
        base = tvd.UCF101(
            root=root,
            annotation_path=str(annotation),
            frames_per_clip=1,
            step_between_clips=1,
            fold=1,
            train=True,
            download=download,
        )
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "UCF101 requires the dataset (videos + ucfTrainTestlist) under "
            f"{root}/ucf101 and a video decoder (PyAV): {exc}"
        ) from exc
    center = CenterFrameDataset(base)
    refs = [(center, i) for i in range(len(center))]
    targets = [int(t) for t in _targets_of(base)]
    return refs, targets


class CenterFrameDataset(Dataset):
    """Expose one (centre) frame of every UCF101 clip as a PIL image."""

    def __init__(self, video_dataset: Dataset):
        self.base = video_dataset
        self.targets = _targets_of(video_dataset)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        from PIL import Image

        item = self.base[index]
        video = item[0]
        label = int(item[-1])
        if isinstance(video, torch.Tensor):
            frames = video
            if frames.dim() == 4 and frames.shape[0] > 0:  # (T, H, W, C)
                frame = frames[frames.shape[0] // 2]
            else:  # pragma: no cover - defensive
                frame = frames
            array = frame.numpy()
            if array.dtype != "uint8":
                array = (array.clip(0, 1) * 255).astype("uint8")
            return Image.fromarray(array), label
        return video, label  # pragma: no cover - defensive


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_dataset(
    name: str,
    split: str,
    image_size: int,
    root: str = "data",
    download: bool = True,
    split_dir: Optional[str] = None,
    seed: int = 0,
    gtsrb_official: bool = True,
) -> Dataset:
    """Build one of the paper's 11 target tasks.

    Parameters
    ----------
    name:
        Dataset key (see :data:`DATASET_SPECS`).
    split:
        ``"train"`` or ``"test"``.
    image_size:
        224 for the ResNets, 384 for ViT-B/32 (the paper's ``imgsize``).
    root:
        Data directory (torchvision layout).
    download:
        Allow torchvision to download the data.
    split_dir:
        Optional directory with benchmark split files (see module docstring).
    """
    name = name.lower()
    if name not in DATASET_SPECS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASET_SPECS)}")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    refs, targets = _raw_pool(name, root, download)
    spec = DATASET_SPECS[name]

    # Datasets whose official split already matches Table 6 use it as is.
    exact_official = name in {"cifar10", "cifar100", "svhn"}
    if name == "gtsrb" and len(targets) >= 50000 and gtsrb_official:
        exact_official = True
    if exact_official:
        n_train = spec.train_size
        if split == "train":
            indices = list(range(min(n_train, len(refs))))
        else:
            indices = list(range(n_train, len(refs)))
        return IndexedPool([refs[i] for i in indices],
                           [targets[i] for i in indices],
                           build_transforms(image_size, split == "train"), name)

    # Otherwise rebuild the benchmark split deterministically.
    cache = pathlib.Path(split_dir) if split_dir else pathlib.Path(root) / "splits"
    cached = _load_cached_split(cache, name, targets)
    if cached is None:
        n_train, n_test = spec.train_size, spec.test_size
        if n_train + n_test > len(refs):
            # e.g. GTSRB without the official archive: keep the train/test ratio
            # but use everything that is available.
            warnings.warn(
                f"{name}: only {len(refs)} images available, fewer than the "
                f"{n_train + n_test} of Table 6; falling back to a proportional split.",
                UserWarning,
            )
            total = len(refs)
            n_train = int(round(total * spec.train_size / (spec.train_size + spec.test_size)))
            n_test = total - n_train
        train_idx, test_idx = deterministic_split(
            targets, n_train, n_test, seed=seed, num_classes=spec.num_classes
        )
        _save_cached_split(cache, name, targets, train_idx, test_idx)
    else:
        train_idx, test_idx = cached

    indices = train_idx if split == "train" else test_idx
    return IndexedPool([refs[i] for i in indices],
                       [targets[i] for i in indices],
                       build_transforms(image_size, split == "train"), name)


def _load_cached_split(cache: pathlib.Path, name: str, targets: Sequence[int]):
    train_file = cache / f"{name}_train.txt"
    test_file = cache / f"{name}_test.txt"
    if not (train_file.is_file() and test_file.is_file()):
        return None
    try:
        train_idx = [int(x) for x in train_file.read_text().split()]
        test_idx = [int(x) for x in test_file.read_text().split()]
    except ValueError:
        return None
    if max(train_idx + test_idx, default=-1) >= len(targets):
        return None
    return train_idx, test_idx


def _save_cached_split(cache: pathlib.Path, name: str, targets: Sequence[int],
                       train_idx: Sequence[int], test_idx: Sequence[int]) -> None:
    try:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{name}_train.txt").write_text("\n".join(str(i) for i in train_idx))
        (cache / f"{name}_test.txt").write_text("\n".join(str(i) for i in test_idx))
        (cache / f"{name}_meta.txt").write_text(
            f"# dataset={name} pool={len(targets)} train={len(train_idx)} test={len(test_idx)}\n"
        )
    except OSError as exc:  # pragma: no cover - read-only filesystems
        warnings.warn(f"could not cache the split for {name}: {exc}", UserWarning)
