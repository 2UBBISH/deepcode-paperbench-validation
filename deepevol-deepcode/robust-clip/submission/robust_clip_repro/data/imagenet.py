"""ImageNet data layer for the Robust CLIP reproduction.

Addendum requirements implemented here
--------------------------------------
* ImageNet must be downloaded/loaded with HuggingFace ``datasets``::

      from datasets import load_dataset
      dataset = load_dataset("imagenet-1k", trust_remote_code=True)

  ``trust_remote_code=True`` is passed explicitly so that the loading script does
  **not** block waiting for stdin (Addendum, verbatim example code).
* Deterministic preprocessing is provided (resize-shortest-side + center crop to
  224px, the standard CLIP zero-shot evaluation recipe).
* A *pixel-space access path* is exposed: every sample carries raw,
  **non-normalized** pixels in ``[0, 1]`` (``image_to_pixels`` /
  ``normalize_imagenet_sample``), while ``normalize_pixels`` applies the model
  normalization separately.  This is required because the Addendum states that
  the PGD l_inf ball is computed *around non-normalized inputs*.

Nothing that the Addendum leaves unspecified (eps / alpha / iterations, batch
size, number of samples, ...) is invented here; those are exposed as arguments
and the values used are logged with provenance markers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("robust_clip_repro.data.imagenet")

# ---------------------------------------------------------------------------
# Provenance markers
# ---------------------------------------------------------------------------
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET = "ImageNet"
HF_DATASET_ID = "imagenet-1k"  # Addendum: load_dataset("imagenet-1k", trust_remote_code=True)
DEFAULT_SPLIT = "validation"
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "train": "train",
    "training": "train",
    "test": "test",
}

DEFAULT_IMAGE_SIZE = 224  # CLIP ViT-L/14 @ 224 (Addendum for LLaVA's vision encoder)
DEFAULT_RESIZE = 224
DEFAULT_INTERPOLATION = "bicubic"

# OpenAI CLIP normalization constants (used only by :func:`normalize_pixels`).
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# Keys tolerated when extracting a label / image out of an arbitrary record.
LABEL_KEYS: Tuple[str, ...] = (
    "label",
    "labels",
    "target",
    "targets",
    "class_id",
    "class",
    "fine_label",
    "wnid",
)
IMAGE_KEYS: Tuple[str, ...] = ("image", "img", "pixels", "pixel_values", "x", "input")

# Values that are not attributes of the paper but are needed to run the harness.
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "num_samples": UNSPECIFIED,          # how many val images to evaluate
    "batch_size": 1,                     # externally supplied
    "num_workers": 0,                    # externally supplied
    "seed": 0,                           # externally supplied
    "eps": UNSPECIFIED,                  # attack budget: not in the Addendum
    "alpha": UNSPECIFIED,                # PGD step size: not in the Addendum
    "iterations": UNSPECIFIED,           # PGD iterations: not in the Addendum
    "restarts": UNSPECIFIED,             # APGD restarts: not in the Addendum
    "norm": UNSPECIFIED,                 # l_inf / l_2 budgets come from the paper body
    "resize": DEFAULT_RESIZE,
    "image_size": DEFAULT_IMAGE_SIZE,
    "interpolation": DEFAULT_INTERPOLATION,
}


# ---------------------------------------------------------------------------
# Sample schema
# ---------------------------------------------------------------------------
@dataclass
class ImageNetSample:
    """Canonical ImageNet sample.

    ``pixels`` is always the raw (non-normalized) ``(1, 3, H, W)`` float tensor in
    ``[0, 1]`` -- the space in which the Addendum's l_inf ball is defined.
    """

    sample_id: str
    index: int
    label: int
    pixels: Any = None
    image: Any = None
    image_path: Optional[str] = None
    label_name: Optional[str] = None
    dataset: str = "imagenet-1k"
    split: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- aliases so metrics/classification.py can duck-type this object --------
    @property
    def labels(self) -> int:
        return self.label

    @property
    def target(self) -> int:
        return self.label

    def to_dict(self, include_image: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "index": self.index,
            "label": self.label,
            "labels": self.label,
            "target": self.label,
            "label_name": self.label_name,
            "dataset": self.dataset,
            "split": self.split,
            "image_path": self.image_path,
            "metadata": dict(self.metadata),
        }
        if self.pixels is not None:
            out["pixels"] = self.pixels
        if include_image and self.image is not None:
            out["image"] = self.image
        return out


# ---------------------------------------------------------------------------
# Label / image extraction helpers (duck-typed records)
# ---------------------------------------------------------------------------
def extract_label(example: Any, default: Optional[int] = None) -> Optional[int]:
    """Pull an integer class index out of a record."""
    if example is None:
        return default
    if isinstance(example, ImageNetSample):
        return example.label
    if isinstance(example, dict):
        for key in LABEL_KEYS:
            if key in example and example[key] is not None:
                return _as_int(example[key], default)
        return default
    for key in LABEL_KEYS:
        if hasattr(example, key):
            value = getattr(example, key)
            if value is not None:
                return _as_int(value, default)
    return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_image(example: Any) -> Any:
    """Pull the PIL/np image out of a record."""
    if isinstance(example, ImageNetSample):
        return example.image
    if isinstance(example, dict):
        for key in IMAGE_KEYS:
            if key in example and example[key] is not None:
                return example[key]
        return None
    for key in IMAGE_KEYS:
        if hasattr(example, key):
            value = getattr(example, key)
            if value is not None:
                return value
    return None


# ---------------------------------------------------------------------------
# Deterministic preprocessing / pixel-space access
# ---------------------------------------------------------------------------
def _to_pil(image: Any) -> Any:
    """Convert an arbitrary image-ish object to a PIL RGB image."""
    from PIL import Image  # lazy

    if image is None:
        raise ValueError("cannot convert None to a PIL image")
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as handle:
            return handle.convert("RGB")
    # numpy array / torch tensor
    try:
        import numpy as np

        if hasattr(image, "detach"):
            array = image.detach().cpu().float().numpy()
        else:
            array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] != array.shape[-1]:
            array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8) if array.max() <= 1.0 + 1e-6 else np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"unsupported image type {type(image)!r}: {exc}") from exc


def _resize_short_side(image: Any, size: int, interpolation: str = DEFAULT_INTERPOLATION) -> Any:
    """Resize so that the shortest side equals ``size`` (deterministic)."""
    from PIL import Image  # lazy

    width, height = image.size
    short, long_ = (width, height) if width <= height else (height, width)
    if short == size:
        return image
    scale = size / float(short)
    new_short = size
    new_long = int(round(long_ * scale))
    if width <= height:
        new_size = (new_short, new_long)
    else:
        new_size = (new_long, new_short)
    resample = getattr(Image, _interp_name(interpolation), Image.BICUBIC)
    return image.resize(new_size, resample=resample)


def _interp_name(interpolation: str) -> str:
    return {
        "bicubic": "BICUBIC",
        "bilinear": "BILINEAR",
        "nearest": "NEAREST",
        "lanczos": "LANCZOS",
    }.get(str(interpolation).lower(), "BICUBIC")


def deterministic_preprocess(
    image: Any,
    *,
    resize: int = DEFAULT_RESIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    interpolation: str = DEFAULT_INTERPOLATION,
) -> Any:
    """Deterministic CLIP-style preprocessing: resize shortest side + center crop."""
    from PIL import Image  # lazy

    pil = _to_pil(image)
    pil = _resize_short_side(pil, resize, interpolation)
    width, height = pil.size
    if width < image_size or height < image_size:
        # Pad (centre) if the image is smaller than the crop, keeps determinism.
        pad_w = max(0, image_size - width)
        pad_h = max(0, image_size - height)
        padded = Image.new("RGB", (width + pad_w, height + pad_h), (0, 0, 0))
        padded.paste(pil, (pad_w // 2, pad_h // 2))
        pil = padded
        width, height = pil.size
    left = (width - image_size) // 2
    top = (height - image_size) // 2
    return pil.crop((left, top, left + image_size, top + image_size))


def image_to_pixels(
    image: Any,
    *,
    resolution: int = DEFAULT_IMAGE_SIZE,
    resize: Optional[int] = DEFAULT_RESIZE,
    interpolation: str = DEFAULT_INTERPOLATION,
    device: Any = None,
    dtype: Any = None,
    in01: bool = True,
    preprocess: bool = True,
) -> Any:
    """Raw (non-normalized) ``(1, 3, H, W)`` float tensor in ``[0, 1]``.

    This is the pixel space in which the Addendum requires PGD to project the
    l_inf ball.  ``normalize_pixels`` applies model normalization *afterwards*.
    """
    import numpy as np  # lazy
    import torch  # lazy

    pil = _to_pil(image)
    if preprocess:
        pil = deterministic_preprocess(
            pil, resize=resize if resize is not None else resolution,
            image_size=resolution, interpolation=interpolation,
        )
    elif pil.size != (resolution, resolution):
        pil = _resize_short_side(pil, resolution, interpolation)
        width, height = pil.size
        left = max(0, (width - resolution) // 2)
        top = max(0, (height - resolution) // 2)
        pil = pil.crop((left, top, left + resolution, top + resolution))

    array = np.asarray(pil, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
    if in01:
        array = array / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def normalize_pixels(
    pixels: Any,
    mean: Sequence[float] = CLIP_MEAN,
    std: Sequence[float] = CLIP_STD,
) -> Any:
    """Apply CLIP channel normalization **without** mutating the raw pixels.

    The attack must keep operating on the raw tensor returned by
    :func:`image_to_pixels`; only the victim model sees the normalized copy.
    """
    import torch  # lazy

    if not torch.is_tensor(pixels):
        pixels = torch.as_tensor(pixels)
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)

    def _expand(values: Sequence[float]) -> Any:
        tensor = torch.as_tensor(list(values), dtype=pixels.dtype, device=pixels.device)
        return tensor.view(1, -1, 1, 1)

    return (pixels - _expand(mean)) / _expand(std)


def denormalize_pixels(
    pixels: Any,
    mean: Sequence[float] = CLIP_MEAN,
    std: Sequence[float] = CLIP_STD,
) -> Any:
    """Inverse of :func:`normalize_pixels` (useful for logging visualisations)."""
    import torch  # lazy

    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)
    mean_t = torch.as_tensor(list(mean), dtype=pixels.dtype, device=pixels.device).view(1, -1, 1, 1)
    std_t = torch.as_tensor(list(std), dtype=pixels.dtype, device=pixels.device).view(1, -1, 1, 1)
    return pixels * std_t + mean_t


# ---------------------------------------------------------------------------
# Sample normalization
# ---------------------------------------------------------------------------
def normalize_imagenet_sample(
    example: Any,
    index: int = 0,
    *,
    split: Optional[str] = None,
    resolution: int = DEFAULT_IMAGE_SIZE,
    resize: Optional[int] = DEFAULT_RESIZE,
    device: Any = None,
    dtype: Any = None,
    label_names: Optional[Sequence[str]] = None,
    with_pixels: bool = True,
    **_: Any,
) -> ImageNetSample:
    """Convert an arbitrary ImageNet record into an :class:`ImageNetSample`."""
    image = extract_image(example)
    label = extract_label(example, default=-1)
    label_name = None
    if label_names is not None and label is not None and 0 <= label < len(label_names):
        label_name = str(label_names[label])

    pixels = None
    if with_pixels and image is not None:
        pixels = image_to_pixels(
            image, resolution=resolution, resize=resize, device=device, dtype=dtype
        )

    image_id = None
    if isinstance(example, dict):
        image_id = example.get("image_id") or example.get("id") or example.get("file_name")
    return ImageNetSample(
        sample_id=str(image_id) if image_id is not None else f"{split or DEFAULT_SPLIT}-{index:06d}",
        index=index,
        label=int(label if label is not None else -1),
        pixels=pixels,
        image=image,
        image_path=None,
        label_name=label_name,
        dataset=HF_DATASET_ID,
        split=split,
        metadata={"provenance": {"pixels": "raw/non-normalized [0,1] (Addendum: l_inf ball)"}},
    )


def normalize_split(dataset: Any, **kwargs: Any) -> List[ImageNetSample]:
    """Normalize a whole HuggingFace split (iterable, non-streaming or streaming)."""
    samples: List[ImageNetSample] = []
    label_names = get_label_names(dataset)
    for index, example in enumerate(dataset):
        samples.append(normalize_imagenet_sample(example, index, label_names=label_names, **kwargs))
    return samples


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def resolve_split_name(split: Optional[str]) -> str:
    if split is None:
        return DEFAULT_SPLIT
    key = str(split).strip().lower()
    return SPLIT_ALIASES.get(key, str(split))


def get_label_names(dataset: Any) -> Optional[List[str]]:
    """Extract ImageNet class names from HuggingFace features when available."""
    try:
        features = getattr(dataset, "features", None)
        if features and "label" in features:
            names = getattr(features["label"], "names", None)
            if names:
                return list(names)
    except Exception:  # pragma: no cover - defensive
        pass
    return None


def load_hf_split(
    *,
    dataset_id: str = HF_DATASET_ID,
    split: str = DEFAULT_SPLIT,
    trust_remote_code: bool = True,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    hf_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """``load_dataset("imagenet-1k", trust_remote_code=True)`` (Addendum).

    ``trust_remote_code=True`` is the Addendum's explicit instruction; it avoids
    the loader script waiting for stdin.
    """
    from datasets import load_dataset  # lazy

    kwargs: Dict[str, Any] = dict(hf_kwargs or {})
    kwargs.setdefault("split", split)
    if cache_dir:
        kwargs.setdefault("cache_dir", cache_dir)
    if streaming:
        kwargs.setdefault("streaming", True)
    # Addendum mandates trust_remote_code=True (avoids waiting for stdin).
    kwargs["trust_remote_code"] = True if trust_remote_code is None else bool(trust_remote_code)
    LOGGER.info("load_dataset(%r, split=%r, trust_remote_code=True)", dataset_id, split)
    return load_dataset(dataset_id, **kwargs)


def _maybe_shuffle(dataset: Any, shuffle: bool, seed: int) -> Any:
    if not shuffle:
        return dataset
    try:
        return dataset.shuffle(seed=seed)
    except Exception:  # pragma: no cover - streaming datasets lack shuffle()
        return dataset


def load_imagenet_dataset(
    dataset_name: str = IMAGENET,
    *,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    dataset_id: str = HF_DATASET_ID,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    resolution: int = DEFAULT_IMAGE_SIZE,
    resize: Optional[int] = DEFAULT_RESIZE,
    device: Any = None,
    dtype: Any = None,
    with_pixels: bool = True,
    trust_remote_code: bool = True,
    hf_kwargs: Optional[Dict[str, Any]] = None,
    allow_empty: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> List[ImageNetSample]:
    """Load/normalize ImageNet samples for clean & robust zero-shot evaluation.

    Returns samples whose ``pixels`` are **raw, non-normalized** tensors in
    ``[0, 1]`` so that the Addendum's PGD can project in that space.
    """
    split = resolve_split_name(split)
    if num_samples is not None and num_samples <= 0:
        num_samples = None

    try:
        dataset = load_hf_split(
            dataset_id=dataset_id,
            split=split,
            trust_remote_code=trust_remote_code,
            streaming=streaming,
            cache_dir=cache_dir,
            hf_kwargs=hf_kwargs,
        )
    except Exception as exc:
        if not allow_empty:
            raise
        LOGGER.warning(
            "could not load %r (split=%s): %s -- returning an empty sample list "
            "(pass allow_empty=False to raise)",
            dataset_id, split, exc,
        )
        return []

    dataset = _maybe_shuffle(dataset, shuffle, seed)
    label_names = get_label_names(dataset)

    limit = len(dataset) if num_samples is None and hasattr(dataset, "__len__") else num_samples
    samples: List[ImageNetSample] = []
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        samples.append(
            normalize_imagenet_sample(
                example,
                index,
                split=split,
                resolution=resolution,
                resize=resize,
                device=device,
                dtype=dtype,
                label_names=label_names,
                with_pixels=with_pixels,
            )
        )

    if verbose:
        LOGGER.info(
            "loaded %d ImageNet samples (split=%s, dataset_id=%s, trust_remote_code=True)",
            len(samples), split, dataset_id,
        )
    return samples


def load_imagenet(**kwargs: Any) -> List[ImageNetSample]:
    """Alias of :func:`load_imagenet_dataset` (dataset-name-first convention)."""
    return load_imagenet_dataset(**kwargs)


# ---------------------------------------------------------------------------
# Iteration / statistics helpers
# ---------------------------------------------------------------------------
def iter_batches(
    samples: Sequence[ImageNetSample],
    batch_size: int = 1,
) -> Iterable[List[ImageNetSample]]:
    """Deterministic batching; when used with attacks batch_size=1 is typical."""
    batch: List[ImageNetSample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def dataset_statistics(samples: Sequence[Any], *, top: int = 10) -> Dict[str, Any]:
    """Summary statistics for reporting/logging (no attack hyperparameters)."""
    labels: List[int] = []
    for sample in samples:
        label = extract_label(sample, default=None)
        if label is not None:
            labels.append(int(label))
    counts: Dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "num_samples": len(samples),
        "num_labeled": len(labels),
        "num_classes": len(counts),
        "most_frequent": [{"label": int(k), "count": int(v)} for k, v in ordered[:top]],
        "has_pixels": sum(1 for s in samples if extract_image(s) is not None or _pixels_of(s) is not None),
    }


def _pixels_of(sample: Any) -> Any:
    if isinstance(sample, dict):
        return sample.get("pixels")
    return getattr(sample, "pixels", None)


# ---------------------------------------------------------------------------
# Synthetic fallback (offline smoke tests, never used for reported numbers)
# ---------------------------------------------------------------------------
def synthetic_samples(
    n: int = 8,
    *,
    num_classes: int = 1000,
    resolution: int = DEFAULT_IMAGE_SIZE,
    device: Any = None,
    dtype: Any = None,
    seed: int = 0,
) -> List[ImageNetSample]:
    """Deterministic toy ImageNet samples (raw pixels) for smoke tests."""
    import torch  # lazy

    generator = torch.Generator().manual_seed(int(seed))
    samples: List[ImageNetSample] = []
    for index in range(int(n)):
        pixels = torch.rand((1, 3, resolution, resolution), generator=generator)
        if dtype is not None:
            pixels = pixels.to(dtype=dtype)
        if device is not None:
            pixels = pixels.to(device=device)
        samples.append(
            ImageNetSample(
                sample_id=f"synthetic-{index:06d}",
                index=index,
                label=int(torch.randint(0, num_classes, (1,), generator=generator).item()),
                pixels=pixels,
                image=None,
                label_name=None,
                dataset="synthetic",
                split="synthetic",
                metadata={"synthetic": True},
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Self test / CLI
# ---------------------------------------------------------------------------
def _self_test(verbose: bool = True) -> bool:
    """Offline checks of the Addendum requirements (no network access)."""
    import inspect

    ok = True

    def check(name: str, condition: bool) -> None:
        nonlocal ok
        ok = ok and bool(condition)
        if verbose:
            print(f"[{'ok' if condition else 'FAIL'}] {name}")

    source = inspect.getsource(load_hf_split)
    check('uses load_dataset("imagenet-1k")', HF_DATASET_ID == "imagenet-1k" and "load_dataset" in source)
    check("trust_remote_code defaults to True", inspect.signature(load_hf_split).parameters["trust_remote_code"].default is True)
    check("trust_remote_code=True passed to load_dataset", "trust_remote_code" in source)

    # Pixel-space path: raw pixels must be in [0, 1] and un-normalized.
    pixels = image_to_pixels(_solid_image(8, 8, (255, 0, 0)), resolution=8, resize=8)
    check("raw pixels shape (1,3,H,W)", tuple(pixels.shape) == (1, 3, 8, 8))
    check("raw pixels in [0,1]", float(pixels.min()) >= 0.0 and float(pixels.max()) <= 1.0)
    normalized = normalize_pixels(pixels)
    check("normalize_pixels does not mutate raw pixels", float(pixels.max()) <= 1.0 and abs(float(normalized.max())) > 1.0)
    check("denormalize round-trips", bool((denormalize_pixels(normalized) - pixels).abs().max() < 1e-5))

    # Deterministic preprocessing.
    image = _solid_image(300, 200, (10, 20, 30))
    a = image_to_pixels(image, resolution=16, resize=16)
    b = image_to_pixels(image, resolution=16, resize=16)
    check("preprocessing is deterministic", bool((a - b).abs().max() == 0))
    check("deterministic_preprocess output size", deterministic_preprocess(image, resize=16, image_size=16).size == (16, 16))

    # Sample normalization + label extraction.
    sample = normalize_imagenet_sample({"image": _solid_image(64, 64, (1, 2, 3)), "label": 7}, 3,
                                       resolution=16, resize=16, label_names=[f"c{i}" for i in range(10)])
    check("label extracted", sample.label == 7 and sample.label_name == "c7")
    check("sample pixels raw", float(sample.pixels.max()) <= 1.0)
    check("metrics key aliases", extract_label(sample.to_dict()) == 7)

    # Synthetic samples / statistics.
    samples = synthetic_samples(4, resolution=8)
    stats = dataset_statistics(samples)
    check("dataset_statistics counts", stats["num_samples"] == 4 and stats["num_labeled"] == 4)
    check("iter_batches", len(list(iter_batches(samples, 2))) == 2)
    check("split aliases", resolve_split_name("val") == "validation")
    check("external defaults flagged unspecified", EXTERNAL_DEFAULTS["eps"] == UNSPECIFIED)
    if verbose:
        print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def _solid_image(width: int, height: int, color: Tuple[int, int, int]) -> Any:
    from PIL import Image  # lazy

    return Image.new("RGB", (width, height), color)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ImageNet loader / statistics for Robust CLIP")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--dataset-id", default=HF_DATASET_ID)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resolution", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--stats", action="store_true", help="print dataset statistics")
    parser.add_argument("--self-test", action="store_true", help="run offline checks")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    if args.self_test:
        return 0 if _self_test(verbose=not args.quiet) else 1
    samples = load_imagenet_dataset(
        split=args.split,
        num_samples=args.num_samples,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
        resolution=args.resolution,
        verbose=not args.quiet,
    )
    if args.stats and not args.quiet:
        print(json.dumps(dataset_statistics(samples), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
