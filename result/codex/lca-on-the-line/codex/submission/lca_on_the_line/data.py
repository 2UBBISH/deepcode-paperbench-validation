"""Dataset loading and label harmonisation for the LCA benchmark.

The benchmark uses ImageNet as the in-distribution (ID) anchor and five OOD
datasets with severe natural distribution shifts.  Every dataset is mapped onto
the same 0..999 ImageNet class index so that a single WordNet hierarchy applies
to all of them.

Dataset sources (see the addendum):

===============  =========================================================
ImageNet-1k      HuggingFace ``imagenet-1k`` (``trust_remote_code=True``)
ImageNet-v2      https://imagenetv2.org (MatchedFrequency split only,
                 commit ``d626240`` of ``vaishaal/ImageNetV2``)
ImageNet-Sketch https://huggingface.co/datasets/songweig/imagenet_sketch
ImageNet-R       https://github.com/hendrycks/imagenet-r
ImageNet-A       https://github.com/hendrycks/natural-adv-examples
ObjectNet        https://objectnet.dev
===============  =========================================================
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .hierarchy import DEFAULT_CLASS_INDEX_JSON, load_imagenet_class_index

IMAGENET_SYNSETS: List[str] = load_imagenet_class_index(DEFAULT_CLASS_INDEX_JSON)[0]
WnidToIndex: Dict[str, int] = {s: i for i, s in enumerate(IMAGENET_SYNSETS)}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".webp")


# --------------------------------------------------------------------------- #
# generic indexed image dataset
# --------------------------------------------------------------------------- #
@dataclass
class IndexedImageDataset:
    """A tiny ``(path, target)`` dataset that mirrors ``torchvision.datasets``."""

    samples: List[Tuple[str, int]]
    transform: Optional[Callable] = None
    name: str = "dataset"

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def targets(self) -> List[int]:
        return [t for _, t in self.samples]

    def __getitem__(self, index: int):
        from PIL import Image

        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def images(self, limit: Optional[int] = None) -> List:
        from PIL import Image

        samples = self.samples[:limit]
        return [Image.open(p).convert("RGB") for p, _ in samples]


def _iter_image_files(folder: str):
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.endswith(IMAGE_EXTENSIONS):
            yield path


def folder_to_samples(
    root: str,
    folder_to_index: Callable[[str], Optional[int]],
    skip_unknown: bool = True,
) -> List[Tuple[str, int]]:
    """Walk a class-folder dataset and map folder names to ImageNet indices."""
    samples: List[Tuple[str, int]] = []
    unknown: List[str] = []
    for entry in sorted(os.listdir(root)):
        class_dir = os.path.join(root, entry)
        if not os.path.isdir(class_dir):
            continue
        index = folder_to_index(entry)
        if index is None:
            unknown.append(entry)
            if not skip_unknown:
                raise KeyError("cannot map class folder %r to an ImageNet index" % entry)
            continue
        for path in _iter_image_files(class_dir):
            samples.append((path, index))
    if unknown:
        print("[data] skipped %d unmapped class folders (e.g. %s)"
              % (len(unknown), ", ".join(unknown[:5])))
    return samples


# --------------------------------------------------------------------------- #
# folder-name -> ImageNet index resolvers
# --------------------------------------------------------------------------- #
def wnid_from_folder(folder: str) -> str:
    """ImageNet-S/R/A style folders start with the WordNet id (``n01440764``)."""
    return folder.split("_")[0]


def resolve_wnid(folder: str) -> Optional[int]:
    wnid = wnid_from_folder(folder)
    return WnidToIndex.get(wnid)


def resolve_index_folder(folder: str) -> Optional[int]:
    """ImageNet-v2 ``*-format-val`` folders are the class index itself."""
    try:
        idx = int(folder)
    except ValueError:
        return resolve_wnid(folder)
    return idx if 0 <= idx < 1000 else None


def load_objectnet_mapping(csv_path: str) -> Dict[str, int]:
    """Parse ObjectNet's ``folder_name -> ImageNet class`` mapping file.

    ObjectNet ships a mapping table (``mappings.csv`` / ``folder_to_imagenet``)
    whose exact column names have changed over time, so we accept a few common
    layouts and fall back to matching the human-readable ImageNet class name.
    """
    names = load_imagenet_class_index(DEFAULT_CLASS_INDEX_JSON)[1]
    name_to_index = {n.lower().replace("_", " "): i for i, n in enumerate(names)}
    mapping: Dict[str, int] = {}
    with open(csv_path, "r") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return mapping
    header = [h.strip().lower() for h in rows[0]]

    def column(*candidates):
        for c in candidates:
            if c in header:
                return header.index(c)
        return None

    folder_col = column("objectnet_class", "folder", "name", "objectnet", "wnid")
    imgnet_col = column("imagenet_class", "imagenet", "imagenet_name", "synset", "class")
    for row in rows[1:]:
        if folder_col is None or imgnet_col is None or len(row) <= max(folder_col, imgnet_col):
            continue
        folder = row[folder_col].strip()
        target = row[imgnet_col].strip()
        if target in WnidToIndex:
            mapping[folder] = WnidToIndex[target]
        else:
            key = target.lower().replace("_", " ")
            if key in name_to_index:
                mapping[folder] = name_to_index[key]
    return mapping


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def load_imagenet_1k(
    split: str = "validation",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> IndexedImageDataset:
    """ImageNet-1k from HuggingFace (the addendum's recommended route)."""
    from datasets import load_dataset

    dataset = load_dataset(
        "imagenet-1k", split=split, cache_dir=cache_dir, trust_remote_code=True
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return HFImageDataset(dataset, name="imagenet-%s" % split)


@dataclass
class HFImageDataset:
    dataset: object
    name: str = "hf"
    transform: Optional[Callable] = None
    image_key: str = "image"
    label_key: str = "label"

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def targets(self) -> List[int]:
        return [int(x) for x in self.dataset[self.label_key]]

    def __getitem__(self, index: int):
        row = self.dataset[index]
        image = row[self.image_key].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row[self.label_key])

    def images(self, limit: Optional[int] = None) -> List:
        n = len(self) if limit is None else min(limit, len(self))
        return [self[i][0] for i in range(n)]


def load_imagenet_v2(root: str, split: str = "matched-frequency",
                     transform=None) -> IndexedImageDataset:
    """ImageNet-v2; only the ``MatchedFrequency`` split is used by the paper."""
    candidates = [
        os.path.join(root, "imagenetv2-%s-format-val-5000" % split),
        os.path.join(root, "imagenetv2-%s-format-val" % split),
        root,
    ]
    folder = next((c for c in candidates if os.path.isdir(c)), root)
    samples = folder_to_samples(folder, resolve_index_folder)
    return IndexedImageDataset(samples, transform=transform, name="imagenet-v2")


def load_imagenet_sketch(root: str, transform=None) -> IndexedImageDataset:
    sub = os.path.join(root, "sketch")
    folder = sub if os.path.isdir(sub) else root
    samples = folder_to_samples(folder, resolve_wnid)
    return IndexedImageDataset(samples, transform=transform, name="imagenet-s")


def load_imagenet_r(root: str, transform=None) -> IndexedImageDataset:
    sub = os.path.join(root, "imagenet-r")
    folder = sub if os.path.isdir(sub) else root
    samples = folder_to_samples(folder, resolve_wnid)
    return IndexedImageDataset(samples, transform=transform, name="imagenet-r")


def load_imagenet_a(root: str, transform=None) -> IndexedImageDataset:
    sub = os.path.join(root, "imagenet-a")
    folder = sub if os.path.isdir(sub) else root
    samples = folder_to_samples(folder, resolve_wnid)
    return IndexedImageDataset(samples, transform=transform, name="imagenet-a")


def load_objectnet(root: str, mapping_csv: Optional[str] = None,
                   transform=None) -> IndexedImageDataset:
    mapping_csv = mapping_csv or os.path.join(root, "mappings.csv")
    mapping = load_objectnet_mapping(mapping_csv) if os.path.exists(mapping_csv) else {}
    samples = folder_to_samples(
        root, lambda folder: mapping.get(folder), skip_unknown=True
    )
    return IndexedImageDataset(samples, transform=transform, name="objectnet")


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def default_dataset_paths(base_dir: str) -> Dict[str, str]:
    return {
        "imagenet": os.path.join(base_dir, "imagenet"),
        "imagenet_v2": os.path.join(base_dir, "imagenetv2"),
        "imagenet_s": os.path.join(base_dir, "imagenet-sketch"),
        "imagenet_r": os.path.join(base_dir, "imagenet-r"),
        "imagenet_a": os.path.join(base_dir, "imagenet-a"),
        "objectnet": os.path.join(base_dir, "objectnet"),
    }


LOADERS = {
    "imagenet": lambda root, **kw: load_imagenet_1k(cache_dir=root, **kw),
    "imagenet_v2": lambda root, **kw: load_imagenet_v2(root, **kw),
    "imagenet_s": lambda root, **kw: load_imagenet_sketch(root, **kw),
    "imagenet_r": lambda root, **kw: load_imagenet_r(root, **kw),
    "imagenet_a": lambda root, **kw: load_imagenet_a(root, **kw),
    "objectnet": lambda root, **kw: load_objectnet(root, **kw),
}

OOD_DATASETS = ("imagenet_v2", "imagenet_s", "imagenet_r", "imagenet_a", "objectnet")


def load_dataset_by_name(name: str, root: str, transform=None, **kwargs):
    if name not in LOADERS:
        raise KeyError("unknown dataset %r (available: %s)" % (name, list(LOADERS)))
    return LOADERS[name](root, transform=transform, **kwargs)


def dataset_class_index(dataset) -> Optional[List[int]]:
    """Ground-truth labels if the dataset carries them."""
    return getattr(dataset, "targets", None)
