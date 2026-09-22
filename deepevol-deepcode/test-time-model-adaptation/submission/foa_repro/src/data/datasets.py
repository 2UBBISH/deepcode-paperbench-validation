"""Dataset and online-stream loaders for the FOA reproduction.

Implements the four OOD benchmarks used in the paper (Section 4 "Datasets and
Models", Appendix B.1):

  * ImageNet-C   -- 15 corruptions x 5 severities, 50k validation images
                    (severity level 5 is the evaluation setting of the paper)
  * ImageNet-R   -- 30k artistic renditions of 200 ImageNet classes
  * ImageNet-V2  -- 10k images; the *Matched-Frequency* subset is used
  * ImageNet-Sketch -- 50,899 black-and-white sketches of 1000 classes

Source data is loaded either from the local ImageNet-style directory layout or,
for ImageNet-1K itself, via HuggingFace ``load_dataset("imagenet-1k",
trust_remote_code=True)`` as instructed by the paper's addendum.

All streams produced here are *ordered, single-pass online* streams
(``shuffle=False``) of batches of ``BS=64`` with the standard ViT 224x224
pre-processing/normalisation, exactly the input protocol of Algorithm 1.

No gradients are ever involved; this module is pure data plumbing.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Standard ImageNet normalisation used by timm ViT-Base augreg checkpoints.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Paper Section 4 / Appendix B.1: the 15 corruption types of ImageNet-C.
IMAGENET_C_CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

#: The four corruption groups mentioned in Appendix B.1.
IMAGENET_C_CORRUPTION_GROUPS: Dict[str, List[str]] = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4

#: Number of classes evaluated on each benchmark (paper Appendix B.1).
DATASET_NUM_CLASSES: Dict[str, int] = {
    "imagenet-1k": 1000,
    "imagenet": 1000,
    "imagenet-c": 1000,
    "imagenet-r": 200,
    "imagenet-v2": 1000,
    "imagenet-sketch": 1000,
    "sketch": 1000,
}

_WNID_RE = re.compile(r"^(n\d{8})")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    train: bool = False,
    resize_mode: str = "resize_center_crop",
):
    """Standard ViT evaluation pre-processing (resize + center-crop + normalise).

    Args:
        image_size: spatial size fed to the ViT (224 for ViT-Base/16).
        mean, std: ImageNet channel statistics.
        train: when ``True`` uses a random-resized-crop augmentation (unused by
            FOA, which only ever evaluates, but handy for debugging).
        resize_mode: ``"resize_center_crop"`` (paper protocol, default) or
            ``"direct"`` (plain resize to ``image_size``).

    Returns:
        A ``torchvision.transforms.Compose`` object accepting PIL images.
    """
    from torchvision import transforms as T

    ops: List[Any] = [T.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img)]
    if train:
        ops += [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
        ]
    elif resize_mode == "direct":
        ops.append(T.Resize((image_size, image_size)))
    else:
        # timm/DeiT evaluation protocol: resize the short side to 256 then
        # center-crop 224 (200/224 ratio for patch16 is 0.888.. ==  0.888888).
        ops += [
            T.Resize(int(image_size / 0.888888), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ]
    ops += [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]
    return T.Compose(ops)


def build_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    """Alias of :func:`build_transform` with the inference protocol."""
    return build_transform(image_size=image_size, train=False)


# --------------------------------------------------------------------------- #
# ImageNet-1K synset (wnid) bookkeeping
# --------------------------------------------------------------------------- #
def _class_index_from_json(path: str) -> Optional[List[str]]:
    """Read an ``imagenet_class_index.json`` mapping (index -> [wnid, name])."""
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except Exception:
        return None
    names: List[Optional[str]] = [None] * len(data)
    for key, value in data.items():
        try:
            idx = int(key)
        except ValueError:
            continue
        if isinstance(value, (list, tuple)) and value:
            names[idx] = value[0]
        elif isinstance(value, str):
            names[idx] = value
    if any(name is None for name in names):
        return None
    return [str(name) for name in names]


def _wnids_from_dirs(root: str) -> Optional[List[str]]:
    """Derive the wnid ordering from ImageNet-style class directories.

    The folder names of ImageNet-C/-R/-Sketch are the ImageNet-1K synsets; the
    canonical (timm/torchvision) label index is the lexicographically sorted
    wnid order.
    """
    if not root or not os.path.isdir(root):
        return None
    wnids = sorted(
        entry
        for entry in os.listdir(root)
        if os.path.isdir(os.path.join(root, entry)) and _WNID_RE.match(entry)
    )
    if len(wnids) < 100:
        return None
    return wnids


def load_imagenet_wnids(
    path: Optional[str] = None,
    search_roots: Optional[Sequence[str]] = None,
    cache_path: str = os.path.join("data", "imagenet_wnids.json"),
    allow_hf: bool = True,
) -> List[str]:
    """Return the 1000 ImageNet-1K wnids in canonical label-index order.

    Resolution order:
      1. an explicit JSON path (``imagenet_class_index.json`` or a plain list),
      2. a cached wnid list,
      3. class directories found under ``search_roots``,
      4. HuggingFace ``imagenet-1k`` feature names (addendum-sanctioned source).
    """
    if path and os.path.exists(path):
        if path.endswith(".json"):
            wnids = _class_index_from_json(path)
        else:
            with open(path, "r") as fh:
                wnids = [line.strip() for line in fh if line.strip()]
        if wnids and len(wnids) >= 1000:
            return list(wnids[:1000])

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as fh:
                wnids = json.load(fh)
            if isinstance(wnids, list) and len(wnids) >= 1000:
                return list(wnids[:1000])
        except Exception:
            pass

    for root in search_roots or []:
        for candidate in (root, os.path.join(root, "imagenet-1k"), os.path.join(root, "val")):
            wnids = _wnids_from_dirs(candidate)
            if wnids:
                _cache_wnids(wnids, cache_path)
                return wnids

    if allow_hf:
        try:  # pragma: no cover - network dependent
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
            feature = ds.features["label"]
            names = list(getattr(feature, "names", []) or [])
            wnids = [n for n in names if _WNID_RE.match(str(n))]
            if len(wnids) >= 1000:
                wnids = sorted(wnids)
                _cache_wnids(wnids, cache_path)
                return wnids
        except Exception:
            pass

    raise RuntimeError(
        "Could not determine the ImageNet-1K wnid ordering needed to map class "
        "directories / file prefixes to label indices. Provide "
        "`data.class_index_json` (an imagenet_class_index.json), place the data "
        "under a root whose sub-folders are wnids, or make the HuggingFace "
        "`imagenet-1k` dataset available."
    )


def _cache_wnids(wnids: Sequence[str], cache_path: Optional[str]) -> None:
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(list(wnids), fh)
    except Exception:
        pass


def wnid_to_index(wnids: Sequence[str]) -> Dict[str, int]:
    """wnid -> canonical 0..999 label index map."""
    return {str(w): i for i, w in enumerate(wnids)}


# --------------------------------------------------------------------------- #
# Core dataset classes
# --------------------------------------------------------------------------- #
class ImagePathDataset(Dataset):
    """A list of ``(path, label)`` pairs with a transform.

    Attributes:
        samples: list of ``(path, label)`` tuples (online stream order).
        transform: callable applied to the PIL image.
        class_subset: sorted list of the (source) label indices present in the
            dataset; used for ImageNet-R's 200-class evaluation.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        transform: Optional[Callable] = None,
        class_subset: Optional[Sequence[int]] = None,
    ) -> None:
        self.samples: List[Tuple[str, int]] = [(str(p), int(y)) for p, y in samples]
        self.transform = transform
        if class_subset is None:
            class_subset = sorted({int(y) for _, y in self.samples})
        self.class_subset: List[int] = list(class_subset)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, target = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img, target

    # ------------------------------------------------------------------ #
    def targets(self) -> List[int]:
        return [int(y) for _, y in self.samples]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(n={len(self.samples)}, "
            f"classes={len(self.class_subset)})"
        )


class HFImageDataset(Dataset):
    """Thin wrapper turning a HuggingFace ``datasets`` split into a torch Dataset.

    The ImageNet-1K validation split is loaded through this class (addendum:
    ``load_dataset("imagenet-1k", trust_remote_code=True)``).
    """

    def __init__(
        self,
        hf_dataset,
        image_key: str = "image",
        label_key: str = "label",
        transform: Optional[Callable] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.hf_dataset = hf_dataset
        columns = set(getattr(hf_dataset, "column_names", []) or [])
        if image_key not in columns:
            for candidate in ("image", "img", "pixel_values"):
                if candidate in columns:
                    image_key = candidate
                    break
        if label_key not in columns:
            for candidate in ("label", "labels", "target", "fine_label"):
                if candidate in columns:
                    label_key = candidate
                    break
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform
        self.indices = list(indices) if indices is not None else None
        self.class_subset: Optional[List[int]] = None

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.hf_dataset)

    def __getitem__(self, index: int):
        real = self.indices[index] if self.indices is not None else index
        item = self.hf_dataset[int(real)]
        image = item[self.image_key]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = item.get(self.label_key, 0)
        return image, int(target)


class ClassFolderDataset(ImagePathDataset):
    """ImageFolder-style dataset whose folders are ImageNet wnids."""

    def __init__(
        self,
        root: str,
        wnids: Sequence[str],
        transform: Optional[Callable] = None,
        class_filter: Optional[Sequence[str]] = None,
        limit_per_class: Optional[int] = None,
        label_space: str = "source",
    ) -> None:
        mapping = wnid_to_index(wnids)
        samples: List[Tuple[str, int]] = []
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Class-folder dataset root not found: {root}")

        folders = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if class_filter is not None:
            allowed = {str(c) for c in class_filter}
            folders = [d for d in folders if d in allowed]

        for folder in folders:
            wnid = _WNID_RE.match(folder)
            key = wnid.group(1) if wnid else folder
            if key not in mapping:
                continue
            label = mapping[key]
            folder_path = os.path.join(root, folder)
            files = sorted(
                f for f in os.listdir(folder_path) if f.endswith(_IMAGE_EXTENSIONS)
            )
            if limit_per_class:
                files = files[:limit_per_class]
            for name in files:
                samples.append((os.path.join(folder_path, name), label))

        if not samples:
            raise RuntimeError(
                f"No images with resolvable wnid class folders under {root!r}. "
                "Expected directories named like 'n01440764'."
            )
        super().__init__(samples, transform=transform)


class FlatImageDataset(ImagePathDataset):
    """Flat directory whose file names start with an ImageNet wnid.

    This is the layout produced by the official ImageNet-C archives, e.g.
    ``<root>/<corruption>/5/n01440764_10026.png``.
    """

    def __init__(
        self,
        root: str,
        wnids: Sequence[str],
        transform: Optional[Callable] = None,
        labels_file: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        mapping = wnid_to_index(wnids)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Flat image directory not found: {root}")

        files = sorted(f for f in os.listdir(root) if f.endswith(_IMAGE_EXTENSIONS))
        if limit:
            files = files[:limit]

        label_lookup: Dict[str, int] = {}
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        label_lookup[parts[0]] = int(parts[1])

        samples: List[Tuple[str, int]] = []
        for name in files:
            wnid = _WNID_RE.match(name)
            if wnid and wnid.group(1) in mapping:
                label = mapping[wnid.group(1)]
            elif name in label_lookup:
                label = label_lookup[name]
            else:
                continue
            samples.append((os.path.join(root, name), label))

        if not samples:
            # last resort: alphabetical wnid-prefix ordering (ImageNet-C copies
            # that use contiguous file names) -- documented fallback.
            prefix_map: Dict[str, int] = {}
            for name in files:
                prefix = name.split("_")[0]
                if prefix not in prefix_map:
                    prefix_map[prefix] = len(prefix_map)
            samples = [(os.path.join(root, n), prefix_map[n.split("_")[0]]) for n in files]

        super().__init__(samples, transform=transform)


# --------------------------------------------------------------------------- #
# Benchmark-specific builders
# --------------------------------------------------------------------------- #
def _find_first(dirs: Sequence[str]) -> Optional[str]:
    for d in dirs:
        if d and os.path.isdir(d):
            return d
    return None


def build_imagenet_c(
    root: str,
    corruption: str,
    severity: int = 5,
    transform: Optional[Callable] = None,
    wnids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Dataset:
    """ImageNet-C stream for one corruption / severity (default severity 5)."""
    transform = transform if transform is not None else build_eval_transform()
    candidates = [
        os.path.join(root, corruption, str(severity)),
        os.path.join(root, "imagenet-c", corruption, str(severity)),
        os.path.join(root, "ImageNet-C", corruption, str(severity)),
        os.path.join(root, "images", corruption, str(severity)),
        os.path.join(root, corruption, f"severity{severity}"),
    ]
    path = _find_first(candidates)
    if path is None:
        raise FileNotFoundError(
            f"ImageNet-C corruption {corruption!r} severity {severity} not found. "
            f"Searched: {candidates}"
        )

    subdirs = [
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    ]
    if subdirs and all(_WNID_RE.match(d) for d in subdirs):
        # <root>/<corruption>/<severity>/<wnid>/<img>
        wnids = list(wnids) if wnids is not None else load_imagenet_wnids(
            search_roots=[root]
        )
        return ClassFolderDataset(path, wnids, transform=transform)

    wnids = list(wnids) if wnids is not None else load_imagenet_wnids(search_roots=[root])
    return FlatImageDataset(path, wnids, transform=transform, limit=limit)


def build_imagenet_r(
    root: str,
    transform: Optional[Callable] = None,
    wnids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Dataset:
    """ImageNet-R: artistic renditions of 200 ImageNet classes."""
    transform = transform if transform is not None else build_eval_transform()
    path = _find_first(
        [
            root,
            os.path.join(root, "imagenet-r"),
            os.path.join(root, "imagenetr"),
            os.path.join(root, "ImageNet-R"),
        ]
    )
    if path is None:
        raise FileNotFoundError(f"ImageNet-R root not found under {root!r}")
    wnids = list(wnids) if wnids is not None else load_imagenet_wnids(search_roots=[root])
    dataset = ClassFolderDataset(path, wnids, transform=transform, limit_per_class=limit)
    return dataset


def build_imagenet_sketch(
    root: str,
    transform: Optional[Callable] = None,
    wnids: Optional[Sequence[str]] = None,
) -> Dataset:
    """ImageNet-Sketch: 50,899 black-and-white sketches, 1000 classes."""
    transform = transform if transform is not None else build_eval_transform()
    path = _find_first(
        [
            os.path.join(root, "imagenet-sketch"),
            os.path.join(root, "sketch"),
            os.path.join(root, "ImageNet-Sketch"),
            root,
        ]
    )
    if path is None:
        raise FileNotFoundError(f"ImageNet-Sketch root not found under {root!r}")
    wnids = list(wnids) if wnids is not None else load_imagenet_wnids(search_roots=[root])
    return ClassFolderDataset(path, wnids, transform=transform)


def build_imagenet_v2(
    root: str,
    subset: str = "matched-frequency",
    transform: Optional[Callable] = None,
    wnids: Optional[Sequence[str]] = None,
) -> Dataset:
    """ImageNet-V2; per Appendix B.1 the Matched-Frequency subset is used."""
    transform = transform if transform is not None else build_eval_transform()
    subset_key = (subset or "matched-frequency").lower().replace("_", "-")
    candidates = [
        os.path.join(root, f"imagenetv2-{subset_key}-format-val"),
        os.path.join(root, "imagenet-v2", f"imagenetv2-{subset_key}-format-val"),
        os.path.join(root, "ImageNetV2", f"imagenetv2-{subset_key}-format-val"),
        os.path.join(root, "imagenetv2"),
        os.path.join(root, f"imagenet-v2-{subset_key}"),
        os.path.join(root, "imagenet-v2"),
    ]
    path = _find_first(candidates)
    if path is None:
        raise FileNotFoundError(
            f"ImageNet-V2 ({subset_key}) not found. Searched: {candidates}"
        )
    wnids = list(wnids) if wnids is not None else load_imagenet_wnids(search_roots=[root])
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if subdirs and all(_WNID_RE.match(d) for d in subdirs):
        return ClassFolderDataset(path, wnids, transform=transform)
    return FlatImageDataset(path, wnids, transform=transform)


def build_hf_imagenet(
    split: str = "validation",
    transform: Optional[Callable] = None,
    num_samples: Optional[int] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """ImageNet-1K through HuggingFace (``trust_remote_code=True``, addendum)."""
    from datasets import load_dataset  # type: ignore

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("imagenet-1k", split=split, **kwargs)
    dataset = HFImageDataset(ds, transform=transform or build_eval_transform())
    if num_samples is not None and num_samples < len(dataset):
        rng = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(len(dataset), generator=rng)[: int(num_samples)].tolist()
        idx = sorted(int(i) for i in idx)  # keep the stream ordered/deterministic
        dataset = Subset(dataset, idx)
    return dataset


# --------------------------------------------------------------------------- #
# Online stream wrapper (single pass, ordered)
# --------------------------------------------------------------------------- #
class OnlineStream:
    """Ordered, single-pass batch iterator over a torch ``Dataset``.

    Yields dicts with keys ``image`` (float tensor ``[B,3,224,224]``) and
    ``label`` (int64 tensor ``[B]``), which is what the FOA runners consume.

    Attributes:
        dataset: the wrapped dataset.
        batch_size: number of samples per yielded batch.
        dataset_name: benchmark identifier.
        class_subset: source label indices present (200 for ImageNet-R).
        num_classes_eval: number of evaluated classes.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        device: Optional[Any] = None,
        drop_last: bool = False,
        dataset_name: str = "unknown",
        class_subset: Optional[Sequence[int]] = None,
        num_classes_eval: Optional[int] = None,
        pin_memory: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device = device
        self.drop_last = bool(drop_last)
        self.dataset_name = dataset_name
        if class_subset is None:
            class_subset = getattr(dataset, "class_subset", None)
            if class_subset is None and isinstance(dataset, Subset):
                class_subset = getattr(dataset.dataset, "class_subset", None)
        self.class_subset: Optional[List[int]] = (
            [int(c) for c in class_subset] if class_subset is not None else None
        )
        self.num_classes_eval = int(
            num_classes_eval
            if num_classes_eval is not None
            else (len(self.class_subset) if self.class_subset else DATASET_NUM_CLASSES.get(dataset_name, 1000))
        )
        self.pin_memory = bool(pin_memory)
        self.collate_fn = collate_fn

    # ------------------------------------------------------------------ #
    def _loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,  # FOA requires an ordered, single-pass stream
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self._loader():
            images, targets = _split_batch(batch)
            if self.device is not None and torch.is_tensor(images):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            yield {"image": images, "images": images, "x": images,
                   "label": targets, "labels": targets, "y": targets}

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def image_batches(self) -> Iterator[torch.Tensor]:
        """Yield only the image tensors (used for source-statistics sampling)."""
        for batch in self:
            yield batch["image"]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OnlineStream(dataset={self.dataset_name}, n={len(self.dataset)}, "
            f"bs={self.batch_size}, batches={len(self)})"
        )


def _split_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalise a collated batch into ``(images, targets)`` tensors."""
    if isinstance(batch, dict):
        images = None
        for key in ("image", "images", "x", "img", "pixel_values"):
            if key in batch:
                images = batch[key]
                break
        targets = None
        for key in ("label", "labels", "y", "target", "targets"):
            if key in batch:
                targets = batch[key]
                break
        if images is None:
            raise KeyError(f"Batch dict has no image field: {list(batch)}")
        if targets is None:
            targets = torch.zeros(len(images), dtype=torch.long)
        return images, targets.long()
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1].long()
    raise TypeError(f"Unsupported batch type: {type(batch)}")


# --------------------------------------------------------------------------- #
# Config-driven entry points (consumed by scripts/*)
# --------------------------------------------------------------------------- #
def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path from a dict/`Config`/None object."""
    if cfg is None:
        return default
    node = cfg
    for key in path.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(key, None)
        else:
            node = getattr(node, key, None)
        if node is None:
            return default
    return node


def normalize_dataset_name(name: Optional[str]) -> str:
    value = (name or "imagenet-c").strip().lower()
    aliases = {
        "c": "imagenet-c",
        "imagenetc": "imagenet-c",
        "imagenet_c": "imagenet-c",
        "r": "imagenet-r",
        "imagenetr": "imagenet-r",
        "imagenet_r": "imagenet-r",
        "v2": "imagenet-v2",
        "imagenetv2": "imagenet-v2",
        "imagenet_v2": "imagenet-v2",
        "sketch": "imagenet-sketch",
        "imagenetsketch": "imagenet-sketch",
        "imagenet_sketch": "imagenet-sketch",
        "in1k": "imagenet-1k",
        "imagenet": "imagenet-1k",
        "imagenet-1k": "imagenet-1k",
    }
    return aliases.get(value, value)


def build_dataset(
    dataset: str = "imagenet-c",
    root: str = "./data",
    corruption: Optional[str] = None,
    severity: int = 5,
    subset: Optional[str] = "matched-frequency",
    image_size: int = DEFAULT_IMAGE_SIZE,
    transform: Optional[Callable] = None,
    wnids: Optional[Sequence[str]] = None,
    class_index_json: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> Dataset:
    """Instantiate one of the four paper benchmarks (torch ``Dataset``)."""
    name = normalize_dataset_name(dataset)
    transform = transform if transform is not None else build_eval_transform(image_size)

    if wnids is None and name != "imagenet-1k":
        try:
            wnids = load_imagenet_wnids(
                path=class_index_json,
                search_roots=[root, os.path.join(root, name)],
                cache_path=os.path.join(root, "imagenet_wnids.json"),
            )
        except Exception:
            wnids = None  # class folders can still be resolved positionally

    if name == "imagenet-c":
        if corruption is None:
            raise ValueError("ImageNet-C requires a `corruption` name (15 available).")
        return build_imagenet_c(root, corruption, severity, transform, wnids)

    if name == "imagenet-r":
        return build_imagenet_r(root, transform, wnids)

    if name == "imagenet-v2":
        return build_imagenet_v2(root, subset or "matched-frequency", transform, wnids)

    if name == "imagenet-sketch":
        return build_imagenet_sketch(root, transform, wnids)

    if name == "imagenet-1k":
        return build_hf_imagenet(
            split=kwargs.pop("split", "validation"),
            transform=transform,
            num_samples=num_samples,
            seed=seed,
            cache_dir=hf_cache_dir or kwargs.pop("cache_dir", None),
        )

    raise ValueError(
        f"Unknown dataset {dataset!r}. Supported: imagenet-c, imagenet-r, "
        "imagenet-v2, imagenet-sketch, imagenet-1k."
    )


def build_online_stream(
    dataset: str = "imagenet-c",
    root: str = "./data",
    corruption: Optional[str] = None,
    severity: int = 5,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    subset: Optional[str] = "matched-frequency",
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: Optional[Any] = None,
    shuffle: bool = False,
    drop_last: bool = False,
    class_index_json: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> OnlineStream:
    """Build an ordered, single-pass online stream over a benchmark.

    ``shuffle`` is accepted for interface compatibility but the online TTA
    protocol of Algorithm 1 always uses ``shuffle=False``; a warning-worthy
    deviation only happens if the caller explicitly passes ``shuffle=True``.
    """
    ds = build_dataset(
        dataset=dataset,
        root=root,
        corruption=corruption,
        severity=severity,
        subset=subset,
        image_size=image_size,
        class_index_json=class_index_json,
        hf_cache_dir=hf_cache_dir,
        num_samples=num_samples,
        seed=seed,
        **kwargs,
    )
    name = normalize_dataset_name(dataset)
    return OnlineStream(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        drop_last=drop_last,
        dataset_name=name,
        num_classes_eval=DATASET_NUM_CLASSES.get(name, 1000),
        pin_memory=False,
    )


def build_dataset_loader(
    cfg: Any = None,
    dataset: Optional[str] = None,
    root: Optional[str] = None,
    corruption: Optional[str] = None,
    severity: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    subset: Optional[str] = None,
    image_size: Optional[int] = None,
    device: Optional[Any] = None,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    return_images_only: bool = False,
    **kwargs: Any,
):
    """Config-driven stream builder used by every runner script.

    Accepts either a full config object (``cfg=cfg``) plus overrides, or plain
    keyword arguments. Returns an :class:`OnlineStream` (or an image-only
    generator when ``return_images_only=True``, used for source statistics).
    """
    dataset = dataset or _cfg_get(cfg, "data.dataset", "imagenet-c")
    root = root or _cfg_get(cfg, "data.root", "./data")
    if corruption is None:
        corruption = _cfg_get(cfg, "data.corruption", None)
    if severity is None:
        severity = int(_cfg_get(cfg, "data.severity", 5) or 5)
    if batch_size is None:
        batch_size = int(_cfg_get(cfg, "data.batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
    if num_workers is None:
        num_workers = int(_cfg_get(cfg, "data.num_workers", DEFAULT_NUM_WORKERS) or 0)
    if subset is None:
        subset = _cfg_get(cfg, "data.subset", "matched-frequency")
    if image_size is None:
        image_size = int(_cfg_get(cfg, "data.image_size", DEFAULT_IMAGE_SIZE) or DEFAULT_IMAGE_SIZE)
    if seed is None:
        seed = int(_cfg_get(cfg, "seed", 0) or 0)

    stream = build_online_stream(
        dataset=dataset,
        root=root,
        corruption=corruption,
        severity=severity,
        batch_size=batch_size,
        num_workers=num_workers,
        subset=subset,
        image_size=image_size,
        device=device,
        class_index_json=_cfg_get(cfg, "data.class_index_json", None),
        hf_cache_dir=_cfg_get(cfg, "data.hf_cache_dir", None),
        num_samples=num_samples,
        seed=seed,
        **kwargs,
    )
    if return_images_only:
        return stream.image_batches()
    return stream


def build_tta_loader(cfg: Any = None, **kwargs: Any):
    """Alias of :func:`build_dataset_loader` (name probed by the FOA runner)."""
    return build_dataset_loader(cfg, **kwargs)


def build_loader(cfg: Any = None, **kwargs: Any):
    """Alias of :func:`build_dataset_loader`."""
    return build_dataset_loader(cfg, **kwargs)


def build_test_loader(cfg: Any = None, **kwargs: Any):
    """Alias of :func:`build_dataset_loader`."""
    return build_dataset_loader(cfg, **kwargs)


def build_stream(cfg: Any = None, **kwargs: Any):
    """Alias returning the raw :class:`OnlineStream`."""
    kwargs.pop("return_images_only", None)
    return build_dataset_loader(cfg, **kwargs)


# --------------------------------------------------------------------------- #
# Source (ImageNet-1K) streams
# --------------------------------------------------------------------------- #
def build_source_stream(
    cfg: Any = None,
    num_samples: int = 32,
    seed: int = 0,
    device: Optional[Any] = None,
    batch_size: Optional[int] = None,
    root: Optional[str] = None,
    images_dir: Optional[str] = None,
    image_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    **kwargs: Any,
) -> Iterator[torch.Tensor]:
    """Yield image batches from ImageNet-1K itself (no labels used).

    Used by ``scripts/compute_source_stats.py`` with ``num_samples=Q=32``
    (Appendix B.2). Source statistics are collected **without** prompt
    injection, hence the raw RGB batches returned here.

    Args:
        cfg: optional config (``data.id_root`` / ``data.root``, image size ...).
        num_samples: Q, the number of unlabeled ID images to stream.
        seed: deterministic subset selection seed.
        images_dir: explicit local ImageFolder root (highest priority).
        root: data root searched for ``imagenet-1k`` / ``val`` / HF cache.

    Yields:
        Float tensors of shape ``[B, 3, 224, 224]``.
    """
    batch_size = int(batch_size or _cfg_get(cfg, "data.batch_size", DEFAULT_BATCH_SIZE))
    batch_size = min(batch_size, max(1, int(num_samples)))
    image_size = int(image_size or _cfg_get(cfg, "data.image_size", DEFAULT_IMAGE_SIZE))
    num_workers = num_workers if num_workers is not None else int(
        _cfg_get(cfg, "data.num_workers", 0) or 0
    )
    transform = build_eval_transform(image_size)

    if images_dir and os.path.isdir(images_dir):
        from torchvision.datasets import ImageFolder

        try:
            ds = ImageFolder(images_dir, transform=transform)
        except Exception:
            ds = ImagePathDataset(_scan_image_files(images_dir), transform=transform)
        idx = list(range(min(int(num_samples), len(ds))))
        loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
        for images, _ in loader:
            if device is not None:
                images = images.to(device)
            yield images
        return

    root = root or _cfg_get(cfg, "source_stats.id_root", None) or _cfg_get(
        cfg, "data.id_root", None
    ) or _cfg_get(cfg, "data.root", "./data")

    # 1) local ImageNet-1K validation class folders
    local_val = _find_first(
        [
            os.path.join(root, "imagenet-1k", "val"),
            os.path.join(root, "imagenet-1k", "validation"),
            os.path.join(root, "imagenet", "val"),
            os.path.join(root, "ILSVRC2012_img_val"),
            os.path.join(root, "val"),
        ]
    )
    if local_val:
        try:
            wnids = load_imagenet_wnids(
                search_roots=[root, os.path.join(root, "imagenet-1k")],
                cache_path=os.path.join(root, "imagenet_wnids.json"),
            )
            ds: Dataset = ClassFolderDataset(local_val, wnids, transform=transform)
        except Exception:
            ds = ImagePathDataset(_scan_image_files(local_val), transform=transform)
        idx = _ordered_indices(len(ds), num_samples, seed)
        loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
        for images, _ in loader:
            if device is not None:
                images = images.to(device)
            yield images
        return

    # 2) HuggingFace ImageNet-1K validation split (paper addendum)
    try:
        ds = build_hf_imagenet(
            split="validation",
            transform=transform,
            num_samples=int(num_samples),
            seed=int(seed),
            cache_dir=_cfg_get(cfg, "data.hf_cache_dir", None),
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise FileNotFoundError(
            "Could not obtain ImageNet-1K source images. Either pass "
            "`images_dir=` pointing to a local ImageNet validation folder, place "
            "the classes under <root>/imagenet-1k/val, or make the HuggingFace "
            f"`imagenet-1k` dataset available. Underlying error: {exc}"
        ) from exc

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for images, _ in loader:
        if device is not None:
            images = images.to(device)
        yield images


def _ordered_indices(num_available: int, num_samples: int, seed: int) -> List[int]:
    """First ``num_samples`` indices in stream order (deterministic, seeded)."""
    num_available = int(num_available)
    num_samples = int(min(num_samples, num_available))
    if num_samples <= 0:
        return []
    if num_samples >= num_available:
        return list(range(num_available))
    rng = torch.Generator().manual_seed(int(seed))
    picked = torch.randperm(num_available, generator=rng)[:num_samples].tolist()
    return sorted(int(i) for i in picked)


def _scan_image_files(root: str) -> List[Tuple[str, int]]:
    """Recursively collect image files, using the parent folder as the class."""
    classes: Dict[str, int] = {}
    samples: List[Tuple[str, int]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.endswith(_IMAGE_EXTENSIONS):
                continue
            parent = os.path.basename(dirpath)
            if parent not in classes:
                classes[parent] = len(classes)
            samples.append((os.path.join(dirpath, name), classes[parent]))
    return samples


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #
def stream_for_corruption(cfg: Any, corruption: str, **kwargs: Any) -> OnlineStream:
    """Build the online stream for one ImageNet-C corruption (Table 2 loop)."""
    return build_online_stream(
        dataset=_cfg_get(cfg, "data.dataset", "imagenet-c"),
        root=_cfg_get(cfg, "data.root", "./data"),
        corruption=corruption,
        severity=int(_cfg_get(cfg, "data.severity", 5) or 5),
        batch_size=int(_cfg_get(cfg, "data.batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE),
        num_workers=int(_cfg_get(cfg, "data.num_workers", DEFAULT_NUM_WORKERS) or 0),
        subset=_cfg_get(cfg, "data.subset", "matched-frequency"),
        image_size=int(_cfg_get(cfg, "data.image_size", DEFAULT_IMAGE_SIZE) or DEFAULT_IMAGE_SIZE),
        class_index_json=_cfg_get(cfg, "data.class_index_json", None),
        hf_cache_dir=_cfg_get(cfg, "data.hf_cache_dir", None),
        **kwargs,
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGENET_C_CORRUPTIONS",
    "IMAGENET_C_CORRUPTION_GROUPS",
    "DATASET_NUM_CLASSES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "build_transform",
    "build_eval_transform",
    "load_imagenet_wnids",
    "wnid_to_index",
    "ImagePathDataset",
    "HFImageDataset",
    "ClassFolderDataset",
    "FlatImageDataset",
    "OnlineStream",
    "normalize_dataset_name",
    "build_dataset",
    "build_online_stream",
    "build_dataset_loader",
    "build_tta_loader",
    "build_test_loader",
    "build_loader",
    "build_stream",
    "build_source_stream",
    "stream_for_corruption",
    "build_imagenet_c",
    "build_imagenet_r",
    "build_imagenet_sketch",
    "build_imagenet_v2",
    "build_hf_imagenet",
]
