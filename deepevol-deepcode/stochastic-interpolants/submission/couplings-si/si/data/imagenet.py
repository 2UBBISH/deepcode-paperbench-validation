"""ImageNet-1k data pipeline for the data-dependent-coupling experiments.

Paper mapping
-------------
* Section 4.1 (in-painting): ``rho_1(x_1)`` corresponds to ImageNet at
  resolution 256 **or** 512; the model additionally receives the class label.
* Section 4.2 (super-resolution): ``rho_1(x_1)`` is ImageNet at 256 or 512 and
  the data pipeline must additionally expose the low-resolution view
  ``D(x_1)`` (64x64 for the 64 -> 256 task) from which the conditioning
  ``xi = U(D(x_1))`` and the coupled base ``x_0 = U(D(x_1)) + sigma zeta`` are
  built.
* Addendum: ImageNet is downloaded through HuggingFace with

      from datasets import load_dataset
      dataset = load_dataset("imagenet-1k", trust_remote_code=True)

  ``load_dataset`` is called with ``trust_remote_code=True`` so the download
  never blocks waiting on stdin.

Everything produced here follows the repository-wide convention that image
tensors live in ``[-1, 1]`` (see :mod:`si.data.transforms`).  Batches are plain
``dict`` objects so that ``train.py`` can consume them without knowing whether
in-painting or super-resolution is being run:

    {"x1":    (B, C, H, W) float32 in [-1, 1],   # target / ground truth
     "label": (B,) int64,                        # ImageNet class label
     # super-resolution only:
     "low":   (B, C, H_low, W_low) float32}      # D(x1), native low resolution

The module degrades gracefully: if ``datasets``/HuggingFace is unavailable (or
the download fails), :func:`build_dataset` can fall back to a synthetic
random-tensor dataset so smoke tests, gradient checks and shape validations run
without the ~150GB ImageNet download.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .transforms import (
    ImageNetTransforms,
    TransformConfig,
    to_tensor,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the optional dependency exists
    from datasets import load_dataset as _hf_load_dataset

    DATASETS_AVAILABLE = True
except Exception:  # pragma: no cover
    _hf_load_dataset = None  # type: ignore[assignment]
    DATASETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_NUM_CLASSES = 1000
IMAGENET_HF_NAME = "imagenet-1k"
DEFAULT_HF_CACHE = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))

#: resolutions used by the paper's experiments (Section 4.1 / 4.2)
PAPER_RESOLUTIONS: Tuple[int, ...] = (256, 512)

#: low-resolution sizes for super-resolution (64 -> 256 and 256 -> 512)
PAPER_LOW_RESOLUTIONS: Tuple[int, ...] = (64, 256)

#: key aliases accepted when reading a raw dataset example / batch dict
IMAGE_KEYS: Tuple[str, ...] = ("image", "img", "x1", "pixel_values")
LABEL_KEYS: Tuple[str, ...] = ("label", "labels", "class_label", "fine_label", "y")


__all__ = [
    "ImageNetDataset",
    "ImageNetIterableDataset",
    "SyntheticImageNetDataset",
    "CollateFn",
    "build_dataset",
    "build_dataloader",
    "build_dataloaders",
    "load_imagenet_hf",
    "imagenet_split_name",
    "default_collate",
    "IMAGENET_NUM_CLASSES",
    "IMAGENET_HF_NAME",
    "PAPER_RESOLUTIONS",
    "PAPER_LOW_RESOLUTIONS",
    "DATASETS_AVAILABLE",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def imagenet_split_name(split: str) -> str:
    """Map common split aliases onto HuggingFace's ``imagenet-1k`` splits.

    HuggingFace exposes ``"train"`` and ``"validation"`` for ``imagenet-1k``
    (the paper's "Valid" column in Table 3).  This helper tolerates the usual
    aliases used across the repo/scripts.
    """
    key = str(split).lower()
    aliases = {
        "train": "train",
        "training": "train",
        "valid": "validation",
        "val": "validation",
        "validation": "validation",
        "test": "validation",  # ImageNet-1k has no public test labels
    }
    if key not in aliases:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(set(aliases))}")
    return aliases[key]


def _first_present(example: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"none of the keys {tuple(keys)} found in example with keys {sorted(example)}")


def _maybe_to_rgb(image: Any) -> Any:
    """Convert single-channel images to 3 channels (ImageNet is RGB)."""
    try:  # PIL image path
        if getattr(image, "mode", None) == "L":
            return image.convert("RGB")
    except Exception:  # pragma: no cover - defensive
        pass
    return image


def load_imagenet_hf(
    split: str = "train",
    *,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    trust_remote_code: bool = True,
    name: str = IMAGENET_HF_NAME,
    **kwargs: Any,
) -> Any:
    """Download/load ImageNet-1k through HuggingFace ``datasets``.

    Mirrors the Addendum snippet, additionally forwarding ``cache_dir`` and
    ``streaming``.  ``trust_remote_code=True`` is the default so the scripted
    loader runs without prompting on stdin.
    """
    if not DATASETS_AVAILABLE:  # pragma: no cover - depends on environment
        raise ImportError(
            "the `datasets` package is required to load ImageNet-1k "
            "(pip install datasets). Alternatively pass `synthetic=True` to "
            "build_dataset() to run smoke tests without ImageNet."
        )
    split = imagenet_split_name(split)
    kwargs.setdefault("trust_remote_code", trust_remote_code)
    if cache_dir is not None:
        kwargs.setdefault("cache_dir", cache_dir)
    logger.info("loading HuggingFace dataset %s [%s] (streaming=%s)", name, split, streaming)
    return _hf_load_dataset(name, split=split, streaming=streaming, **kwargs)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ImageNetDataset(Dataset):
    """Map-style ImageNet-1k dataset returning ``[-1, 1]`` tensors + labels.

    Parameters
    ----------
    dataset:
        A HuggingFace dataset (anything supporting ``len``/``__getitem__``), or
        any sequence of dict-like examples.  If ``None``, the dataset is loaded
        with :func:`load_imagenet_hf` using the remaining arguments.
    transform:
        Callable applied to the raw image.  Defaults to
        :class:`si.data.transforms.ImageNetTransforms` built from
        ``resolution``/``low_resolution``.
    resolution:
        Target ImageNet resolution (256 or 512 in the paper).
    low_resolution:
        When set, the dataset additionally returns ``low`` = ``D(x1)`` at the
        native low resolution (e.g. 64 for the 64 -> 256 super-resolution task).
    return_dict:
        ``True`` (default) yields dicts; ``False`` yields ``(x1, label)`` tuples.
    """

    def __init__(
        self,
        dataset: Any = None,
        *,
        split: str = "train",
        resolution: int = 256,
        low_resolution: Optional[int] = None,
        train: Optional[bool] = None,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        return_dict: bool = True,
        return_low_res: Optional[bool] = None,
        trust_remote_code: bool = True,
        max_samples: Optional[int] = None,
        hf_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.split = imagenet_split_name(split)
        self.resolution = int(resolution)
        self.low_resolution = None if low_resolution is None else int(low_resolution)
        self.train = (self.split == "train") if train is None else bool(train)
        self.return_dict = bool(return_dict)
        self.return_low_res = (self.low_resolution is not None) if return_low_res is None else bool(return_low_res)
        if self.return_low_res and self.low_resolution is None:
            raise ValueError("return_low_res=True requires low_resolution to be set")
        self.max_samples = max_samples

        if dataset is None:
            dataset = load_imagenet_hf(
                self.split,
                cache_dir=cache_dir,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
                **(hf_kwargs or {}),
            )
        self.dataset = dataset

        if transform is None:
            transform = ImageNetTransforms(
                resolution=self.resolution,
                low_resolution=self.low_resolution,
                train=self.train,
            )
        self.transform = transform

    # -- helpers ----------------------------------------------------------
    def __len__(self) -> int:
        if self.max_samples is not None:
            return min(self.max_samples, len(self.dataset))
        return len(self.dataset)

    def _raw(self, index: int) -> Dict[str, Any]:
        example = self.dataset[int(index)]
        if not isinstance(example, dict):
            example = {"image": example, "label": 0}
        return example

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._raw(index)
        image = _maybe_to_rgb(_first_present(example, IMAGE_KEYS))
        try:
            label = int(_first_present(example, LABEL_KEYS))
        except KeyError:
            label = 0
        x1 = to_tensor(image, dtype=torch.float32)
        x1 = self.transform(x1)

        if self.low_resolution is not None:
            low = self.transform.low_res_dataset(x1)
            if self.return_dict:
                return {"x1": x1, "label": label, "low": low, "low_res": low}
            return x1, low, label

        if self.return_dict:
            return {"x1": x1, "label": label}
        return x1, label

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"{self.__class__.__name__}(split={self.split!r}, resolution={self.resolution}, "
            f"low_resolution={self.low_resolution}, n={len(self)})"
        )


class ImageNetIterableDataset(IterableDataset):
    """Streaming variant of :class:`ImageNetDataset` (HuggingFace ``streaming=True``).

    Useful when ImageNet-1k cannot be materialised on disk.  Each yielded item
    follows the same dict layout as :class:`ImageNetDataset`.
    """

    def __init__(
        self,
        dataset: Any = None,
        *,
        split: str = "train",
        resolution: int = 256,
        low_resolution: Optional[int] = None,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        cache_dir: Optional[str] = None,
        return_dict: bool = True,
        trust_remote_code: bool = True,
        max_samples: Optional[int] = None,
        hf_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.split = imagenet_split_name(split)
        self.resolution = int(resolution)
        self.low_resolution = None if low_resolution is None else int(low_resolution)
        self.return_dict = bool(return_dict)
        self.max_samples = max_samples
        if dataset is None:
            dataset = load_imagenet_hf(
                self.split,
                cache_dir=cache_dir,
                streaming=True,
                trust_remote_code=trust_remote_code,
                **(hf_kwargs or {}),
            )
        self.dataset = dataset
        self.transform = transform or ImageNetTransforms(
            resolution=self.resolution,
            low_resolution=self.low_resolution,
            train=(self.split == "train"),
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i, example in enumerate(self.dataset):
            if self.max_samples is not None and i >= self.max_samples:
                return
            if not isinstance(example, dict):
                example = {"image": example, "label": 0}
            image = _maybe_to_rgb(_first_present(example, IMAGE_KEYS))
            try:
                label = int(_first_present(example, LABEL_KEYS))
            except KeyError:
                label = 0
            x1 = self.transform(to_tensor(image, dtype=torch.float32))
            if self.low_resolution is not None:
                low = self.transform.low_res_dataset(x1)
                yield {"x1": x1, "label": label, "low": low, "low_res": low}
            else:
                yield {"x1": x1, "label": label}


class SyntheticImageNetDataset(Dataset):
    """Random-tensor stand-in for ImageNet-1k used for smoke tests.

    Produces ``x1 ~ U(-1, 1)`` (optionally correlated across neighbouring
    pixels so that a *too weak* model is visible), together with random class
    labels and, when ``low_resolution`` is given, the low-resolution view.
    No download, no HuggingFace dependency.
    """

    def __init__(
        self,
        length: int = 64,
        *,
        resolution: int = 256,
        low_resolution: Optional[int] = None,
        channels: int = 3,
        num_classes: int = IMAGENET_NUM_CLASSES,
        transform: Optional[Callable[[Any], torch.Tensor]] = None,
        smooth: bool = True,
        seed: int = 0,
        return_dict: bool = True,
    ) -> None:
        self.length = int(length)
        self.resolution = int(resolution)
        self.low_resolution = None if low_resolution is None else int(low_resolution)
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.smooth = bool(smooth)
        self.seed = int(seed)
        self.return_dict = bool(return_dict)
        self.transform = transform or ImageNetTransforms(
            resolution=self.resolution, low_resolution=self.low_resolution, train=True
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(index))
        x = torch.rand(self.channels, self.resolution, self.resolution, generator=g)
        if self.smooth:  # cheap low-frequency structure
            x = torch.nn.functional.avg_pool2d(x.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
        x1 = x * 2.0 - 1.0
        label = int(torch.randint(0, self.num_classes, (1,), generator=g).item())
        if self.low_resolution is not None:
            low = self.transform.low_res_dataset(x1)
            if not self.return_dict:
                return x1, low, label
            return {"x1": x1, "label": label, "low": low, "low_res": low}
        if not self.return_dict:
            return x1, label
        return {"x1": x1, "label": label}


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


class CollateFn:
    """Collate dataset dicts/tuples into the batch layout used by ``train.py``.

    Always yields a ``dict`` with

    * ``"x1"``    : ``(B, C, H, W)`` in ``[-1, 1]``
    * ``"label"`` : ``(B,)`` int64
    * ``"low"``   : ``(B, C, H_low, W_low)`` (only when the dataset provides it)

    plus the aliases ``"image"``/``"images"`` and ``"y"`` so that loosely-typed
    consumers (e.g. ``eval/fid.py``) can find the images either way.
    """

    def __init__(self, include_aliases: bool = True, include_low_res: bool = True) -> None:
        self.include_aliases = bool(include_aliases)
        self.include_low_res = bool(include_low_res)

    def __call__(self, batch: Sequence[Any]) -> Dict[str, Any]:
        if isinstance(batch, dict):  # already collated
            return batch
        if len(batch) == 0:
            return {}

        first = batch[0]
        out: Dict[str, Any] = {}

        if isinstance(first, dict):
            x1 = torch.stack([torch.as_tensor(b.get("x1", b.get("image"))) for b in batch], dim=0)
            labels = [int(b.get("label", b.get("y", 0))) for b in batch]
            lows = [b.get("low", b.get("low_res")) for b in batch]
        elif isinstance(first, (tuple, list)):
            x1 = torch.stack([torch.as_tensor(b[0]) for b in batch], dim=0)
            if len(first) == 2:
                labels = [int(b[1]) for b in batch]
                lows = [None] * len(batch)
            else:  # (x1, low, label)
                lows = [b[1] for b in batch]
                labels = [int(b[2]) for b in batch]
        else:  # bare images
            x1 = torch.stack([torch.as_tensor(b) for b in batch], dim=0)
            labels = [0] * len(batch)
            lows = [None] * len(batch)

        out["x1"] = x1
        out["label"] = torch.as_tensor(labels, dtype=torch.long)
        if lows[0] is not None and self.include_low_res:
            out["low"] = torch.stack([torch.as_tensor(l) for l in lows], dim=0)
            out["low_res"] = out["low"]

        if self.include_aliases:
            # `x1` is the ground truth for in-painting, but for super-resolution
            # the *conditioning* is the low-res image; `x1` remains the target.
            out["image"] = out["x1"]
            out["images"] = out["x1"]
            out["y"] = out["label"]
        return out


def default_collate(batch: Sequence[Any]) -> Dict[str, Any]:
    """Module-level :class:`CollateFn` instance for use as a DataLoader callable."""
    return CollateFn()(batch)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_dataset(
    split: str = "train",
    *,
    resolution: int = 256,
    low_resolution: Optional[int] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    synthetic: bool = False,
    synthetic_length: int = 256,
    max_samples: Optional[int] = None,
    transform: Optional[Callable[[Any], torch.Tensor]] = None,
    return_dict: bool = True,
    trust_remote_code: bool = True,
    dataset: Any = None,
    **hf_kwargs: Any,
) -> Dataset:
    """Build an ImageNet dataset for one of the paper's tasks.

    Parameters mirror the experimental configurations:

    * in-painting 256: ``resolution=256``
    * in-painting 512: ``resolution=512``
    * super-resolution 64 -> 256: ``resolution=256, low_resolution=64``
    * super-resolution 256 -> 512: ``resolution=512, low_resolution=256``

    ``synthetic=True`` bypasses ImageNet entirely (random tensors) which keeps
    the training/sampling/eval pipeline runnable end-to-end on machines without
    the dataset or without ``datasets`` installed.
    """
    if synthetic:
        return SyntheticImageNetDataset(
            length=synthetic_length if max_samples is None else max_samples,
            resolution=resolution,
            low_resolution=low_resolution,
            transform=transform,
            return_dict=return_dict,
        )

    if dataset is None and not DATASETS_AVAILABLE:
        logger.warning(
            "`datasets` is not installed; falling back to SyntheticImageNetDataset. "
            "Install with `pip install datasets` and enable HF_HUB/download for real ImageNet-1k."
        )
        return SyntheticImageNetDataset(
            length=synthetic_length if max_samples is None else max_samples,
            resolution=resolution,
            low_resolution=low_resolution,
            transform=transform,
            return_dict=return_dict,
        )

    if streaming:
        return ImageNetIterableDataset(
            dataset,
            split=split,
            resolution=resolution,
            low_resolution=low_resolution,
            transform=transform,
            cache_dir=cache_dir,
            return_dict=return_dict,
            trust_remote_code=trust_remote_code,
            max_samples=max_samples,
            hf_kwargs=hf_kwargs or None,
        )

    return ImageNetDataset(
        dataset,
        split=split,
        resolution=resolution,
        low_resolution=low_resolution,
        transform=transform,
        cache_dir=cache_dir,
        streaming=False,
        return_dict=return_dict,
        trust_remote_code=trust_remote_code,
        max_samples=max_samples,
        hf_kwargs=hf_kwargs or None,
    )


def build_dataloader(
    split: str = "train",
    *,
    batch_size: int = 32,
    resolution: int = 256,
    low_resolution: Optional[int] = None,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    synthetic: bool = False,
    synthetic_length: int = 256,
    max_samples: Optional[int] = None,
    collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
    transform: Optional[Callable[[Any], torch.Tensor]] = None,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    return_dict: bool = True,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Build an ImageNet :class:`~torch.utils.data.DataLoader`.

    The paper's experiments all use ``batch_size=32`` (Addendum); a batch of 32
    with resolution 256/512 fits the pixel-space U-Net from Appendix B on a
    24GB+ GPU.  For the training split ``shuffle`` defaults to ``True`` and
    ``drop_last`` to ``True`` (so every batch has exactly the configured size);
    the validation split defaults to no shuffling and no dropping.
    """
    is_train = imagenet_split_name(split) == "train"
    if shuffle is None:
        shuffle = is_train
    if drop_last is None:
        drop_last = is_train
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    dataset = build_dataset(
        split,
        resolution=resolution,
        low_resolution=low_resolution,
        cache_dir=cache_dir,
        streaming=streaming,
        synthetic=synthetic,
        synthetic_length=synthetic_length,
        max_samples=max_samples,
        transform=transform,
        return_dict=return_dict,
        **dataset_kwargs,
    )

    if isinstance(dataset, IterableDataset):
        # Iterable datasets cannot be shuffled/dropped by the DataLoader the
        # usual way; keep the defaults minimal and let the sampler handle it.
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn or default_collate,
            pin_memory=pin_memory,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn or default_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def build_dataloaders(
    *,
    resolution: int = 256,
    low_resolution: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    synthetic: bool = False,
    synthetic_length: int = 256,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience helper returning ``(train_loader, valid_loader)`` for a task."""
    train_loader = build_dataloader(
        "train",
        batch_size=batch_size,
        resolution=resolution,
        low_resolution=low_resolution,
        num_workers=num_workers,
        synthetic=synthetic,
        synthetic_length=synthetic_length,
        **kwargs,
    )
    valid_loader = build_dataloader(
        "validation",
        batch_size=batch_size,
        resolution=resolution,
        low_resolution=low_resolution,
        num_workers=num_workers,
        synthetic=synthetic,
        synthetic_length=max(1, synthetic_length // 4),
        **kwargs,
    )
    return train_loader, valid_loader


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)

    # Synthetic in-painting pipeline (256 and 512).
    for res in (256, 512):
        ds = build_dataset("train", resolution=res, synthetic=True, synthetic_length=4)
        item = ds[0]
        assert item["x1"].shape == (3, res, res), item["x1"].shape
        assert item["x1"].min() >= -1.0 and item["x1"].max() <= 1.0
        assert 0 <= item["label"] < IMAGENET_NUM_CLASSES

    # Synthetic super-resolution pipeline (64 -> 256 and 256 -> 512).
    for res, low in ((256, 64), (512, 256)):
        ds = build_dataset(
            "train", resolution=res, low_resolution=low, synthetic=True, synthetic_length=4
        )
        item = ds[0]
        assert item["x1"].shape == (3, res, res)
        assert item["low"].shape == (3, low, low), item["low"].shape

    # Collation produces the canonical training batch layout.
    loader = build_dataloader(
        "train",
        batch_size=2,
        resolution=256,
        low_resolution=64,
        synthetic=True,
        synthetic_length=4,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["x1"].shape == (2, 3, 256, 256)
    assert batch["label"].shape == (2,) and batch["label"].dtype == torch.long
    assert batch["low"].shape == (2, 3, 64, 64)
    assert batch["image"].shape == batch["x1"].shape

    # Tuple mode.
    ds = build_dataset("train", resolution=256, synthetic=True, synthetic_length=2, return_dict=False)
    x1, label = ds[0]
    assert x1.shape == (3, 256, 256) and isinstance(label, int)

    # Split aliasing.
    assert imagenet_split_name("valid") == "validation"
    assert imagenet_split_name("test") == "validation"

    print("si/data/imagenet.py self-test OK")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _self_test()
