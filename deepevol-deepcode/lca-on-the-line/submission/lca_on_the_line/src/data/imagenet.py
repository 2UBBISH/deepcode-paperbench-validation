"""ImageNet-1k (in-distribution) data loading utilities.

The paper (SS4 "Dataset Setup") uses ImageNet-1k as the source in-distribution
dataset, with 75 pretrained models evaluated on it. The plan specifies loading
ImageNet-1k through HuggingFace::

    datasets.load_dataset("imagenet-1k", trust_remote_code=True)

This module provides:

* deterministic, CLIP/torchvision-compatible transforms
  (resize :math:`\\rightarrow` center-crop :math:`\\rightarrow` bicubic
  :math:`\\rightarrow` ToTensor :math:`\\rightarrow` Normalize);
* class-index alignment to the canonical 1000-class WordNet (wnid) ordering used
  by the ``imagenet_fiveai.csv`` hierarchy from ``jvlmdr/hiercls``;
* a ``torch.utils.data.Dataset`` wrapper that works with either the HuggingFace
  ``imagenet-1k`` dataset or a local ImageFolder-style ImageNet copy;
* deterministic subsetting (useful for smoke tests and feature caching).

Everything degrades gracefully: if neither the HuggingFace dataset nor a local
copy is available, helpers raise an informative :class:`ImageNetUnavailable`
unless ``allow_synthetic=True`` is requested, in which case a tiny deterministic
synthetic dataset is produced for unit tests / smoke runs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is a hard requirement in practice
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore
    Subset = object  # type: ignore
    _HAS_TORCH = False

from ..hierarchy.wordnet import IMAGENET_NUM_CLASSES, get_imagenet_wnids

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGENET_NUM_CLASSES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_RESOLUTION",
    "ImageNetUnavailable",
    "LabelMapping",
    "build_transform",
    "build_label_mapping",
    "load_imagenet_wnids",
    "load_imagenet_class_names",
    "ImageNetIDDataset",
    "load_imagenet_hf",
    "load_imagenet_local",
    "build_imagenet_dataset",
    "build_imagenet_loader",
    "subset_dataset",
    "synthetic_imagenet_like",
]

# --------------------------------------------------------------------------------------
# Constants (ImageNet statistics)
# --------------------------------------------------------------------------------------

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_RESOLUTION: int = 224
DEFAULT_CROP_PCT: float = 0.875
DEFAULT_SPLIT: str = "validation"
DEFAULT_HF_NAME: str = "imagenet-1k"


class ImageNetUnavailable(RuntimeError):
    """Raised when ImageNet-1k cannot be located (no HF dataset / no local copy)."""


# --------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------


def build_transform(
    resolution: int = DEFAULT_RESOLUTION,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    interpolation: str = "bicubic",
    crop_pct: float = DEFAULT_CROP_PCT,
    train: bool = False,
    normalize: bool = True,
    color_jitter: float = 0.0,
    flip: bool = False,
) -> Callable[[Any], Any]:
    """Build a deterministic ImageNet preprocessing pipeline.

    The default evaluation pipeline follows the standard ImageNet protocol used
    by essentially all 75 models in the paper: resize the shorter side to
    ``round(resolution / crop_pct)``, center crop ``resolution``, convert to
    tensor and normalize with ImageNet statistics.  Using the exact same
    transform for every model keeps the ID/OOD accuracy comparison (Table 1)
    fair across the VM/VLM zoo.

    Parameters
    ----------
    resolution:
        Output spatial resolution (e.g. 224, or 336 for ViT-L-14-336px).
    interpolation:
        One of ``"bilinear"``, ``"bicubic"``, ``"lanczos"`` or a string handled
        by ``torchvision.transforms.InterpolationMode``.
    crop_pct:
        Portion of the resized image retained by the center crop.
    train:
        If ``True`` applies a RandomResizedCrop instead of the deterministic
        resize+centercrop (only used if somebody trains on ImageNet).
    """
    try:
        from torchvision import transforms as T
    except Exception as exc:  # pragma: no cover
        raise ImportError("torchvision is required for build_transform") from exc

    interp = getattr(T.InterpolationMode, interpolation.upper(), T.InterpolationMode.BICUBIC)

    ops: List[Any] = []
    if train:
        ops.append(
            T.RandomResizedCrop(resolution, scale=(0.08, 1.0), interpolation=interp, antialias=True)
        )
        if color_jitter > 0:
            ops.append(T.ColorJitter(color_jitter, color_jitter, color_jitter))
        if flip:
            ops.append(T.RandomHorizontalFlip())
    else:
        resize_size = int(round(resolution / crop_pct))
        ops.append(T.Resize(resize_size, interpolation=interp, antialias=True))
        ops.append(T.CenterCrop(resolution))

    ops.append(T.ToTensor() if not hasattr(T, "v2") else T.ToTensor())
    if normalize:
        ops.append(T.Normalize(mean=list(mean), std=list(std)))
    return T.Compose(ops)


# --------------------------------------------------------------------------------------
# Label mapping / class-index alignment
# --------------------------------------------------------------------------------------


@dataclass
class LabelMapping:
    """Maps between dataset labels, canonical ImageNet indices and class names.

    Attributes
    ----------
    index_to_wnid:
        Canonical ImageNet index (:math:`0..999`) to WordNet synset id.
    index_to_name:
        Canonical index to human readable class name.
    name_to_index:
        Reverse map of ``index_to_name`` (first occurrence wins).
    local_to_index:
        Optional mapping from a *dataset-specific* integer label (e.g. the
        ordering used by ImageNet-A) to the canonical ImageNet index.
    """

    index_to_wnid: List[str] = field(default_factory=list)
    index_to_name: List[str] = field(default_factory=list)
    name_to_index: Dict[str, int] = field(default_factory=dict)
    local_to_index: Dict[int, int] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.index_to_wnid) or len(self.index_to_name) or IMAGENET_NUM_CLASSES

    def remap(self, label: int) -> int:
        """Translate a dataset-local label into the canonical ImageNet index."""
        if not self.local_to_index:
            return int(label)
        return int(self.local_to_index.get(int(label), int(label)))

    def remap_many(self, labels: Sequence[int]) -> List[int]:
        return [self.remap(l) for l in labels]

    def name(self, index: int) -> str:
        if 0 <= int(index) < len(self.index_to_name):
            return self.index_to_name[int(index)]
        return str(index)

    def index_of_name(self, name: str) -> int:
        return int(self.name_to_index.get(name, -1))


def load_imagenet_wnids(root: Optional[str] = None) -> List[str]:
    """Return the 1000 canonical ImageNet synset ids in label-index order.

    Tries, in order: (1) a cached ``wnids.txt`` under ``root``, (2) the
    HuggingFace ``imagenet-1k`` dataset metadata, (3) a plain ``wnids.txt`` in
    the package directory.  Returns an empty list when none is available.
    """
    candidates: List[str] = []
    if root:
        candidates.append(os.path.join(root, "wnids.txt"))
        candidates.append(os.path.join(root, "imagenet_wnids.txt"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "wnids.txt"))
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                wnids = [line.strip() for line in handle if line.strip()]
            if wnids:
                logger.info("Loaded %d wnids from %s", len(wnids), path)
                return wnids

    try:
        wnids = list(get_imagenet_wnids())
        if wnids:
            if root:
                try:
                    os.makedirs(root, exist_ok=True)
                    with open(os.path.join(root, "wnids.txt"), "w", encoding="utf-8") as fh:
                        fh.write("\n".join(wnids) + "\n")
                except OSError:  # pragma: no cover - read-only filesystem
                    pass
            return wnids
    except Exception as exc:  # pragma: no cover - offline
        logger.debug("Could not obtain ImageNet wnids: %s", exc)
    return []


def load_imagenet_class_names(root: Optional[str] = None) -> List[str]:
    """Return the 1000 ImageNet class names in canonical label-index order."""
    candidates: List[str] = []
    if root:
        candidates += [
            os.path.join(root, "class_names.txt"),
            os.path.join(root, "imagenet_classes.txt"),
            os.path.join(root, "LOC_synset_mapping.txt"),
        ]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "class_names.txt"))
    for path in candidates:
        if os.path.isfile(path):
            names: List[str] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    # LOC_synset_mapping.txt is "n01234567 name, name"
                    parts = line.split(" ", 1)
                    names.append(parts[1] if len(parts) == 2 and parts[0].startswith("n") else line)
            if names:
                return names

    # HuggingFace fallback (metadata only, no images downloaded).
    try:
        from datasets import load_dataset

        builder = load_dataset(DEFAULT_HF_NAME, split=DEFAULT_SPLIT, streaming=True, trust_remote_code=True)
        feature = builder.features.get("label") if hasattr(builder, "features") else None
        if feature is not None and getattr(feature, "names", None):
            return list(feature.names)
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not obtain ImageNet class names from HF: %s", exc)
    return []


def build_label_mapping(
    root: Optional[str] = None,
    local_class_names: Optional[Sequence[str]] = None,
    local_wnids: Optional[Sequence[str]] = None,
) -> LabelMapping:
    """Construct the canonical ImageNet label mapping.

    When ``local_class_names`` (and/or ``local_wnids``) are given, a
    ``local_to_index`` remapping is built so that datasets whose label ordering
    differs from the canonical one (ImageNet-A, ImageNet-R, ObjectNet) can be
    aligned with the 1000-class order.
    """
    index_to_name = load_imagenet_class_names(root)
    index_to_wnid = load_imagenet_wnids(root)

    if not index_to_wnid and index_to_name:
        index_to_wnid = ["" for _ in index_to_name]
    if not index_to_name and index_to_wnid:
        index_to_name = [w for w in index_to_wnid]

    if not index_to_name and not index_to_wnid:
        logger.warning(
            "No ImageNet class metadata found; falling back to synthetic "
            "'class_%%03d' names. Accuracy remains valid as long as every model "
            "uses the same canonical index order."
        )
        index_to_name = [f"class_{i:03d}" for i in range(IMAGENET_NUM_CLASSES)]
        index_to_wnid = ["" for _ in range(IMAGENET_NUM_CLASSES)]

    name_to_index: Dict[str, int] = {}
    for idx, name in enumerate(index_to_name):
        key = _normalise_name(name)
        if key and key not in name_to_index:
            name_to_index[key] = idx

    mapping = LabelMapping(
        index_to_wnid=list(index_to_wnid),
        index_to_name=list(index_to_name),
        name_to_index=name_to_index,
    )

    # Build the dataset-local -> canonical index remap when possible.
    if local_wnids:
        wnid_to_index = {w: i for i, w in enumerate(mapping.index_to_wnid)}
        if wnid_to_index:
            mapping.local_to_index = {
                i: wnid_to_index[w] for i, w in enumerate(local_wnids) if w in wnid_to_index
            }
    elif local_class_names:
        mapping.local_to_index = {}
        for i, name in enumerate(local_class_names):
            key = _normalise_name(name)
            if key in name_to_index:
                mapping.local_to_index[i] = name_to_index[key]
    return mapping


def _normalise_name(name: str) -> str:
    name = str(name).strip().lower()
    if "," in name:
        name = name.split(",", 1)[0]
    return name.replace("_", " ").strip()


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class ImageNetIDDataset(Dataset):
    """Wrap an indexable image collection with transforms + label remapping.

    ``source`` may be:

    * a HuggingFace ``datasets`` split (columns ``image`` and ``label``);
    * a ``torchvision.datasets.ImageFolder``-like object returning
      ``(image, label)``;
    * a list of ``(image, label)`` tuples.

    Labels are always returned as the *canonical* ImageNet index so that logits
    from every model in the zoo line up with the hierarchy's class ids.
    """

    def __init__(
        self,
        source: Any,
        transform: Optional[Callable[[Any], Any]] = None,
        label_mapping: Optional[LabelMapping] = None,
        image_column: str = "image",
        label_column: str = "label",
    ) -> None:
        self.source = source
        self.transform = transform
        self.label_mapping = label_mapping
        self.image_column = image_column
        self.label_column = label_column
        self._is_hf = hasattr(source, "column_names") and image_column in getattr(source, "column_names", [])

    def __len__(self) -> int:
        return len(self.source)  # type: ignore[arg-type]

    def _raw(self, index: int) -> Tuple[Any, int]:
        if self._is_hf:
            row = self.source[index]
            return row[self.image_column], int(row[self.label_column])
        item = self.source[index]
        if isinstance(item, dict):
            return item.get(self.image_column, item.get("img")), int(
                item.get(self.label_column, item.get("label", -1))
            )
        image, label = item[0], item[1]
        return image, int(label)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image, label = self._raw(index)
        image = _to_rgb(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.label_mapping is not None:
            label = self.label_mapping.remap(label)
        return image, int(label)


def _to_rgb(image: Any) -> Any:
    if image is None:
        return image
    try:
        from PIL import Image

        if isinstance(image, Image.Image) and image.mode != "RGB":
            return image.convert("RGB")
    except Exception:  # pragma: no cover
        pass
    return image


def load_imagenet_hf(
    split: str = DEFAULT_SPLIT,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    streaming: bool = False,
) -> Any:
    """Load ImageNet-1k from HuggingFace (``imagenet-1k``)."""
    from datasets import load_dataset

    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if streaming:
        kwargs["streaming"] = True
    logger.info("Loading HuggingFace ImageNet-1k split=%s ...", split)
    return load_dataset(DEFAULT_HF_NAME, split=split, **kwargs)


def load_imagenet_local(root: str, split: str = "val") -> Any:
    """Load a local ImageFolder-style ImageNet copy (``root/split/class/*.JPEG``)."""
    from torchvision.datasets import ImageFolder

    candidates = [os.path.join(root, split), os.path.join(root, "validation"), root]
    for path in candidates:
        if os.path.isdir(path):
            logger.info("Loading local ImageNet from %s", path)
            return ImageFolder(path)
    raise ImageNetUnavailable(f"No ImageFolder-compatible directory found under {root!r}")


def build_imagenet_dataset(
    root: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
    resolution: int = DEFAULT_RESOLUTION,
    transform: Optional[Callable[[Any], Any]] = None,
    use_hf: bool = True,
    cache_dir: Optional[str] = None,
    label_mapping: Optional[LabelMapping] = None,
    max_samples: Optional[int] = None,
    allow_synthetic: bool = False,
    seed: int = 0,
) -> ImageNetIDDataset:
    """Build the ImageNet-1k ID dataset, trying HF then a local copy.

    ``root`` may point at either an ImageFolder root or a HuggingFace cache.
    Returns an :class:`ImageNetIDDataset` with canonical label indices.
    """
    if transform is None:
        transform = build_transform(resolution=resolution)
    if label_mapping is None:
        label_mapping = build_label_mapping(root)

    source: Optional[Any] = None
    # 1) explicit local path that is an ImageFolder
    if root and os.path.isdir(root) and any(
        os.path.isdir(os.path.join(root, s)) for s in ("val", "validation", "train")
    ):
        try:
            source = load_imagenet_local(root, split=split)
        except ImageNetUnavailable:
            source = None
    # 2) HuggingFace
    if source is None and use_hf:
        try:
            source = load_imagenet_hf(split=split, cache_dir=cache_dir)
        except Exception as exc:  # pragma: no cover - offline / gated dataset
            logger.warning("HuggingFace ImageNet-1k unavailable (%s)", exc)
            source = None
    # 3) plain local directory
    if source is None and root and os.path.isdir(root):
        try:
            source = load_imagenet_local(root, split=split)
        except ImageNetUnavailable:
            source = None

    if source is None:
        if not allow_synthetic:
            raise ImageNetUnavailable(
                "ImageNet-1k could not be loaded. Provide `data.imagenet_root` in the config "
                "or run `huggingface-cli login` for the gated `imagenet-1k` dataset. "
                "Pass allow_synthetic=True for smoke tests."
            )
        logger.warning("Falling back to synthetic ImageNet-like data (SMOKE TEST ONLY).")
        return synthetic_imagenet_like(
            transform=transform, label_mapping=label_mapping, num_samples=max_samples or 32, seed=seed
        )

    dataset = ImageNetIDDataset(source, transform=transform, label_mapping=label_mapping)
    if max_samples is not None and max_samples > 0:
        dataset = subset_dataset(dataset, max_samples, seed=seed)  # type: ignore[assignment]
    return dataset


def subset_dataset(dataset: Any, num_samples: int, seed: int = 0, contiguous: bool = False) -> Any:
    """Deterministically subset a dataset (for smoke tests / caching runs)."""
    if num_samples >= len(dataset):  # type: ignore[arg-type]
        return dataset
    if contiguous:
        indices = list(range(num_samples))
    else:
        import random

        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset)), num_samples))  # type: ignore[arg-type]
    if _HAS_TORCH:
        return Subset(dataset, indices)
    return [dataset[i] for i in indices]  # pragma: no cover


def build_imagenet_loader(
    dataset: Any,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Any:
    """Build a ``torch.utils.data.DataLoader`` for ImageNet evaluation."""
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("torch is required for build_imagenet_loader")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


# --------------------------------------------------------------------------------------
# Synthetic fallback (smoke tests only)
# --------------------------------------------------------------------------------------


def synthetic_imagenet_like(
    transform: Optional[Callable[[Any], Any]] = None,
    label_mapping: Optional[LabelMapping] = None,
    num_samples: int = 32,
    num_classes: int = IMAGENET_NUM_CLASSES,
    resolution: int = DEFAULT_RESOLUTION,
    seed: int = 0,
) -> ImageNetIDDataset:
    """Create a deterministic synthetic dataset with the ImageNet interface.

    Each sample is a constant-colored image whose class is ``index % num_classes``
    so that accuracy is deterministic and reproducible.  Used only for smoke
    tests where the real ImageNet data is unavailable.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError("numpy and pillow are required for synthetic data") from exc

    rng = np.random.RandomState(seed)
    if transform is None:
        transform = build_transform(resolution=resolution)

    items: List[Tuple[Any, int]] = []
    for i in range(num_samples):
        color = tuple(int(c) for c in rng.randint(0, 255, size=3))
        image = Image.new("RGB", (resolution, resolution), color)
        items.append((image, i % num_classes))
    if label_mapping is None:
        label_mapping = build_label_mapping()
    return ImageNetIDDataset(items, transform=transform, label_mapping=label_mapping)
