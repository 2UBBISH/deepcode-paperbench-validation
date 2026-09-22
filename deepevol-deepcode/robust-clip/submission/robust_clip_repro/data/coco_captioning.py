"""COCO (and Flickr30k) image + reference-caption loading for captioning robustness.

This module is the data layer of the captioning evaluation
(``robust_clip_repro/eval_captioning.py``).  It provides

* ``load_captioning_dataset`` -- the single entry point used by the captioning
  harness, returning *canonical* sample dictionaries,
* ``load_coco`` / ``load_flickr30k`` -- dataset specific convenience wrappers,
* raw (non-normalized) pixel tensors so the attack engines can project the
  l_inf ball around the **non-normalized** image, as the Addendum requires
  (the model normalization is applied later, inside the victim model).

Canonical sample schema (matches ``eval_captioning.normalize_sample``)::

    {
        "sample_id": int,
        "id": <dataset specific id>,
        "image": PIL.Image,             # optional, kept when available
        "pixels": torch.Tensor,         # (1, 3, H, W) in [0, 1], raw/unnormalized
        "ground_truths": [str, ...],    # reference captions (up to num_ground_truths)
        "captions": [str, ...],         # alias of ground_truths
        "references": [str, ...],       # alias of ground_truths
        "dataset": "COCO",
        "split": "test",
    }

Nothing in this module invents paper hyperparameters: it only knows about
dataset identifiers, splits and file names.  Attack budgets, thresholds and the
number of attacked ground truths come from ``configs/captioning.yaml``.
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("robust_clip_repro.data.coco_captioning")

# --------------------------------------------------------------------------- #
# Constants (dataset identifiers only -- no paper hyperparameters here)
# --------------------------------------------------------------------------- #

COCO = "COCO"
FLICKR30K = "Flickr30k"
DATASET_NAMES: Tuple[str, ...] = (COCO, FLICKR30K)

#: Case/format insensitive aliases.
DATASET_ALIASES: Dict[str, str] = {
    "coco": COCO,
    "mscoco": COCO,
    "ms-coco": COCO,
    "coco2014": COCO,
    "coco_captions": COCO,
    "coco-caption": COCO,
    "karpathy": COCO,
    "caption_coco": COCO,
    "flickr30k": FLICKR30K,
    "flickr30k_entities": FLICKR30K,
    "flickr": FLICKR30K,
}

#: Preferred HuggingFace dataset ids per benchmark, tried in order.
#: ``None`` means "default configuration of that dataset id".
HF_DATASET_IDS: Dict[str, Tuple[Tuple[str, Optional[str]], ...]] = {
    COCO: (
        ("HuggingFaceM4/COCO", "2014captions"),
        ("HuggingFaceM4/COCO", None),
        ("nlphuji/mscoco_2014_5k_test_image_text_retrieval", None),
    ),
    FLICKR30K: (
        ("nlphuji/flickr30k", None),
        ("nlphuji/flickr30k", "default"),
    ),
}

#: Default splits (COCO captions are most commonly evaluated on the Karpathy
#: test split; the HF mirror exposes it through "test"/"validation").
DEFAULT_SPLITS: Dict[str, str] = {
    COCO: "test",
    FLICKR30K: "test",
}

#: Local LLaVA/COCO style assets (optional, used when ``annotations_file`` is set).
KARPATHY_ANNOTATION_FILES: Tuple[str, ...] = (
    "dataset_coco.json",
    "coco_karpathy_test.json",
    "karpathy_test.json",
    "captions_val2014.json",
)
DEFAULT_IMAGE_ROOT = "train2014"
DEFAULT_MAX_GROUND_TRUTHS = 5  # paper body: 5 ground truths are attacked

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

#: Candidate keys holding the reference captions in HF/local records.
CAPTION_KEYS: Tuple[str, ...] = (
    "captions",
    "caption",
    "references",
    "sentences",
    "annotations",
    "alt_texts",
    "conversations",
    "text",
    "labels",
)

IMAGE_KEYS: Tuple[str, ...] = ("image", "img", "image_path", "file_name", "filename", "coco_url")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def normalize_dataset_name(name: Optional[str]) -> str:
    """Map an alias / case variation onto a canonical dataset name."""
    if name is None:
        return COCO
    key = str(name).strip()
    if key in DATASET_NAMES:
        return key
    low = key.lower().replace(" ", "").replace("_", "").replace("/", "").replace("-", "")
    for alias, canonical in DATASET_ALIASES.items():
        if alias.replace("_", "").replace("-", "") == low:
            return canonical
    for canonical in DATASET_NAMES:
        if canonical.lower().replace("-", "") == low:
            return canonical
    raise ValueError(
        f"Unknown captioning dataset {name!r}; supported: {DATASET_NAMES} (aliases: {sorted(DATASET_ALIASES)})"
    )


def _clean_captions(values: Iterable[Any], max_captions: Optional[int] = None) -> List[str]:
    """Flatten/flatten-and-deduplicate raw caption values into strings."""
    out: List[str] = []
    seen = set()
    for value in values or ():
        text: Optional[str] = None
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            for key in ("raw", "caption", "sentences", "text", "value", "content", "answer"):
                if key in value and isinstance(value[key], str):
                    text = value[key]
                    break
            if text is None:
                # e.g. {"sentences": [{"raw": "..."}]}
                for key in ("sentences", "captions", "annotations"):
                    if key in value and isinstance(value[key], (list, tuple)) and value[key]:
                        inner = _clean_captions(value[key], None)
                        for item in inner:
                            if item and item not in seen:
                                seen.add(item)
                                out.append(item)
                continue
        else:
            text = str(value)
        text = (text or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if max_captions is not None and len(out) >= max_captions:
            break
    if max_captions is not None:
        out = out[:max_captions]
    return out


def extract_captions(example: Any, dataset_name: str = COCO, *, num_ground_truths: Optional[int] = None) -> List[str]:
    """Extract reference captions from an HF/local record (duck-typed)."""
    if example is None:
        return []
    get = example.get if hasattr(example, "get") else (lambda k, d=None: getattr(example, k, d))
    for key in CAPTION_KEYS:
        value = get(key, None)
        if value is None:
            continue
        caps = _clean_captions(value if isinstance(value, (list, tuple)) else [value], num_ground_truths)
        if caps:
            return caps
    # COCO annotation style: {"annotations": [{"caption": ...}]} / {"sentences": [...]}
    for key in ("annotations", "sentences", "captions"):
        value = get(key, None)
        if isinstance(value, (list, tuple)) and value:
            caps = _clean_captions(value, num_ground_truths)
            if caps:
                return caps
    return []


def image_to_pil(image: Any) -> Any:
    """Best-effort conversion of a record's image field into a ``PIL.Image``."""
    if image is None:
        return None
    if not isinstance(image, str):
        try:
            from PIL import Image as PILImage  # lazy

            if isinstance(image, PILImage.Image):
                return image.convert("RGB")
        except Exception:  # pragma: no cover - PIL always present in practice
            pass
        try:  # numpy / torch tensor
            import numpy as np

            if hasattr(image, "detach"):
                image = image.detach().cpu().numpy()
            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0 + 1e-6:
                    arr = (arr * 255.0).clip(0, 255).astype("uint8")
                else:
                    arr = arr.clip(0, 255).astype("uint8")
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = arr.transpose(1, 2, 0)
            from PIL import Image as PILImage

            return PILImage.fromarray(arr).convert("RGB")
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Could not convert image field of type %s: %s", type(image), exc)
            return None
    return image  # a path / filename string, resolved by the caller


def image_to_pixels(
    image: Any,
    *,
    resolution: int = 224,
    device: Any = None,
    dtype: Any = None,
    in01: bool = True,
) -> Any:
    """Convert an image (PIL / ndarray / tensor) to a ``(1, 3, H, W)`` tensor.

    Returns the **raw, non-normalized** pixel tensor so that PGD/APGD can project
    the perturbation ball around the original pixels (Addendum requirement).
    """
    import torch  # lazy

    if isinstance(image, str):
        image = _open_path(image)
    if image is None:
        return None
    if hasattr(image, "detach"):  # already a tensor
        tensor = image
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dtype != torch.float32 and dtype is None:
            tensor = tensor.float()
    else:
        pil = image_to_pil(image)
        if pil is None:
            return None
        if resolution is not None:
            pil = _resize(pil, resolution)
        arr = _pil_to_numpy(pil)
        tensor = torch.from_numpy(arr)  # (H, W, 3) uint8
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    if not in01 and tensor.max() <= 1.0 + 1e-6:
        tensor = tensor * 255.0
    return tensor


def _pil_to_numpy(pil: Any) -> Any:
    import numpy as np

    return np.asarray(pil.convert("RGB"))


def _resize(pil: Any, resolution: Any) -> Any:
    """Resize preserving aspect ratio-space convention (square resize)."""
    try:
        from PIL import Image as PILImage

        if isinstance(resolution, (tuple, list)):
            size = (int(resolution[0]), int(resolution[1]))
        else:
            size = (int(resolution), int(resolution))
        resample = getattr(PILImage, "Resampling", PILImage).BICUBIC
        return pil.resize(size, resample)
    except Exception:  # pragma: no cover
        return pil


def _open_path(path: str) -> Any:
    from PIL import Image as PILImage  # lazy

    with PILImage.open(path) as handle:
        return handle.convert("RGB")


def extract_image(example: Any, *, image_root: Optional[str] = None) -> Optional[Any]:
    """Return the image field of a record, resolving local file names if possible."""
    if example is None:
        return None
    get = example.get if hasattr(example, "get") else (lambda k, d=None: getattr(example, k, d))
    image = None
    for key in IMAGE_KEYS:
        value = get(key, None)
        if value is not None:
            image = value
            break
    if image is None:
        return None
    pil = image_to_pil(image)
    if pil is not None:
        return pil
    # ``image`` is a string: try resolving against image_root / dataset dirs.
    candidate = Path(str(image))
    roots = [Path(image_root)] if image_root else []
    roots.append(candidate.parent)
    for root in roots:
        try:
            for name in (candidate.name, str(candidate)):
                path = root / name
                if path.is_file():
                    return _open_path(str(path))
        except Exception:
            continue
    if candidate.is_file():
        return _open_path(str(candidate))
    LOGGER.debug("Could not resolve image path %r (image_root=%r)", image, image_root)
    return None


# --------------------------------------------------------------------------- #
# Canonical sample construction
# --------------------------------------------------------------------------- #
def build_sample(
    *,
    sample_id: int,
    dataset: str,
    image_id: Any = None,
    image: Any = None,
    pixels: Any = None,
    captions: Sequence[str] = (),
    split: Optional[str] = None,
    resolution: Optional[int] = None,
    device: Any = None,
    dtype: Any = None,
    num_ground_truths: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical captioning sample dictionary."""
    caps = _clean_captions(captions, num_ground_truths)
    if pixels is None and image is not None and resolution is not None:
        pixels = image_to_pixels(image, resolution=resolution, device=device, dtype=dtype)
    sample: Dict[str, Any] = {
        "sample_id": int(sample_id),
        "id": image_id if image_id is not None else int(sample_id),
        "image": image,
        "pixels": pixels,
        "ground_truths": caps,
        "captions": caps,
        "references": caps,
        "dataset": dataset,
        "split": split,
    }
    if extra:
        for key, value in extra.items():
            sample.setdefault(key, value)
    return sample


def normalize_captioning_sample(
    example: Any,
    dataset_name: str = COCO,
    index: int = 0,
    *,
    split: Optional[str] = None,
    image_root: Optional[str] = None,
    num_ground_truths: Optional[int] = None,
    resolution: Optional[int] = None,
    device: Any = None,
    dtype: Any = None,
) -> Dict[str, Any]:
    """Normalize an arbitrary record (HF ``datasets`` row or dict) into the schema."""
    dataset_name = normalize_dataset_name(dataset_name)
    get = example.get if hasattr(example, "get") else (lambda k, d=None: getattr(example, k, d))
    captions = extract_captions(example, dataset_name, num_ground_truths=num_ground_truths)
    image = extract_image(example, image_root=image_root)
    image_id = None
    for key in ("image_id", "cocoid", "coco_id", "img_id", "id", "filename", "file_name", "imgid"):
        value = get(key, None)
        if value is not None and not isinstance(value, (list, tuple, dict)):
            image_id = value
            break
    return build_sample(
        sample_id=index,
        dataset=dataset_name,
        image_id=image_id,
        image=image,
        captions=captions,
        split=split,
        resolution=resolution,
        device=device,
        dtype=dtype,
        num_ground_truths=num_ground_truths,
    )


# --------------------------------------------------------------------------- #
# Local Karpathy-style annotation loading
# --------------------------------------------------------------------------- #
def load_karpathy_split(
    annotations_file: str,
    *,
    image_root: Optional[str] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
    num_ground_truths: Optional[int] = None,
    resolution: Optional[int] = None,
    device: Any = None,
    dtype: Any = None,
    seed: int = 0,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Load a COCO-Karpathy-style JSON file (``dataset_coco.json``) if present."""
    path = Path(annotations_file)
    if not path.is_file():
        raise FileNotFoundError(f"Annotations file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images", payload if isinstance(payload, list) else [])
    wanted = None if split in (None, "all") else str(split).lower()
    image_root = image_root or str(path.parent)
    samples: List[Dict[str, Any]] = []
    for record in images:
        if wanted is not None:
            record_split = str(record.get("split", "")).lower()
            if record_split and record_split != wanted:
                continue
        captions = _clean_captions(
            [s.get("raw", s.get("caption", "")) if isinstance(s, dict) else s
             for s in (record.get("sentences") or record.get("captions") or [])],
            num_ground_truths,
        )
        if not captions:
            continue
        filename = record.get("filename") or record.get("file_name") or record.get("image")
        image = None
        if filename:
            candidate = Path(image_root) / str(filename)
            if candidate.is_file():
                image = _open_path(str(candidate))
        samples.append(
            build_sample(
                sample_id=len(samples),
                dataset=COCO,
                image_id=record.get("cocoid", record.get("imgid", record.get("id"))),
                image=image,
                captions=captions,
                split=split,
                resolution=resolution,
                device=device,
                dtype=dtype,
                num_ground_truths=num_ground_truths,
            )
        )
        if num_samples is not None and len(samples) >= num_samples:
            break
    if verbose:
        LOGGER.info("Loaded %d samples from %s (split=%s)", len(samples), path, split)
    return samples


def find_local_annotations(cache_dir: Optional[str], dataset_name: str = COCO) -> Optional[str]:
    """Search common locations for a local annotation file."""
    roots = [Path(cache_dir)] if cache_dir else []
    roots.append(Path.cwd())
    for root in roots:
        try:
            for name in KARPATHY_ANNOTATION_FILES:
                candidate = root / name
                if candidate.is_file():
                    return str(candidate)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# HuggingFace loading
# --------------------------------------------------------------------------- #
def _hf_load(
    dataset_id: str,
    config_name: Optional[str],
    split: str,
    *,
    cache_dir: Optional[str],
    trust_remote_code: bool,
    hf_kwargs: Optional[Dict[str, Any]],
) -> Any:
    from datasets import load_dataset  # lazy

    kwargs: Dict[str, Any] = dict(hf_kwargs or {})
    kwargs.setdefault("trust_remote_code", trust_remote_code)
    if cache_dir:
        kwargs.setdefault("cache_dir", cache_dir)
    LOGGER.info("load_dataset(%r%s, split=%r, trust_remote_code=%s)",
                dataset_id, f", {config_name!r}" if config_name else "", split, trust_remote_code)
    try:
        if config_name:
            return load_dataset(dataset_id, config_name, split=split, **kwargs)
        return load_dataset(dataset_id, split=split, **kwargs)
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("load_dataset(%r, %r) failed: %s", dataset_id, config_name, exc)
        return None


def _resolve_split_name(dataset: Any, split: Optional[str]) -> Optional[str]:
    """Pick an available split if the requested one does not exist."""
    if split is None:
        return None
    try:
        keys = list(getattr(dataset, "keys", lambda: [])())
    except Exception:
        keys = []
    if not keys:
        return split
    if split in keys:
        return split
    aliases = {
        "test": ("test", "validation", "val", "train"),
        "validation": ("validation", "valid", "val", "test"),
        "val": ("validation", "valid", "val", "test"),
        "train": ("train", "validation", "test"),
    }
    for candidate in aliases.get(split, (split,)):
        if candidate in keys:
            LOGGER.warning("Split %r unavailable; using %r (available: %s)", split, candidate, keys)
            return candidate
    LOGGER.warning("Split %r unavailable and no fallback found (available: %s)", split, keys)
    return split


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def load_captioning_dataset(
    dataset_name: str = COCO,
    *,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    num_ground_truths: Optional[int] = None,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    dataset_id: Optional[str] = None,
    hf_kwargs: Optional[Dict[str, Any]] = None,
    trust_remote_code: bool = True,
    streaming: bool = False,
    shuffle: bool = True,
    image_root: Optional[str] = None,
    annotations_file: Optional[str] = None,
    resolution: Optional[int] = 224,
    device: Any = None,
    dtype: Any = None,
    verbose: bool = True,
    allow_empty: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load reference captions for a captioning benchmark.

    Resolution order

    1. ``annotations_file`` (local COCO-Karpathy JSON) when provided/found,
    2. HuggingFace ``datasets.load_dataset`` with ``trust_remote_code=True``
       (candidate ids in :data:`HF_DATASET_IDS`),
    3. an empty list when ``allow_empty`` (the harness then falls back to its
       synthetic samples).

    Returns canonical sample dicts in the schema documented at module level.
    """
    dataset_name = normalize_dataset_name(dataset_name)
    split = split or DEFAULT_SPLITS.get(dataset_name)
    num_ground_truths = num_ground_truths or DEFAULT_MAX_GROUND_TRUTHS
    if kwargs:
        LOGGER.debug("load_captioning_dataset ignoring extra kwargs: %s", sorted(kwargs))

    # (1) local annotations -------------------------------------------------
    local = annotations_file or find_local_annotations(cache_dir, dataset_name)
    if local:
        try:
            samples = load_karpathy_split(
                local,
                image_root=image_root,
                split=split or "test",
                num_samples=num_samples,
                num_ground_truths=num_ground_truths,
                resolution=resolution,
                device=device,
                dtype=dtype,
                seed=seed,
                verbose=verbose,
            )
            if samples:
                return samples
        except Exception as exc:
            LOGGER.warning("Local annotation loading failed (%s); falling back to HuggingFace", exc)

    # (2) HuggingFace -------------------------------------------------------
    candidates: Tuple[Tuple[str, Optional[str]], ...]
    if dataset_id:
        candidates = ((dataset_id, None),)
    else:
        candidates = HF_DATASET_IDS.get(dataset_name, ())
    samples: List[Dict[str, Any]] = []
    for cand_id, config_name in candidates:
        try:
            dataset = _hf_load(
                cand_id,
                config_name,
                split,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                hf_kwargs=hf_kwargs,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("HuggingFace load of %r failed: %s", cand_id, exc)
            dataset = None
        if dataset is None:
            continue
        if isinstance(dataset, dict) or hasattr(dataset, "keys"):
            tried = _resolve_split_name(dataset, split)
            try:
                dataset = dataset[tried]
            except Exception:
                pass
        try:
            if streaming:
                dataset = dataset  # already an IterableDataset when streaming=True
            elif shuffle:
                try:
                    dataset = dataset.shuffle(seed=seed)
                except Exception:
                    pass
        except Exception:
            pass
        samples = _samples_from_hf(dataset, dataset_name, split=split,
                                   num_samples=num_samples,
                                   num_ground_truths=num_ground_truths,
                                   resolution=resolution, image_root=image_root,
                                   device=device, dtype=dtype, verbose=verbose)
        if samples:
            if verbose:
                LOGGER.info("Loaded %d %s captioning samples from HuggingFace %s",
                            len(samples), dataset_name, cand_id)
            return samples
        LOGGER.warning("HuggingFace dataset %r produced no usable captioning samples", cand_id)

    if not samples:
        message = (
            f"Could not load {dataset_name} captioning data (tried "
            f"{[c for c, _ in candidates] or 'no ids'}; local annotations: {local}). "
            "Install `datasets`, provide `annotations_file`/`image_root`, or set `allow_empty`."
        )
        if allow_empty:
            LOGGER.warning(message)
            return []
        raise RuntimeError(message)
    return samples


def _samples_from_hf(
    dataset: Any,
    dataset_name: str,
    *,
    split: Optional[str],
    num_samples: Optional[int],
    num_ground_truths: int,
    resolution: Optional[int],
    image_root: Optional[str],
    device: Any,
    dtype: Any,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    try:
        iterator = iter(dataset)
    except TypeError:  # pragma: no cover
        return samples
    for index, example in enumerate(iterator):
        try:
            sample = normalize_captioning_sample(
                example,
                dataset_name,
                index,
                split=split,
                image_root=image_root,
                num_ground_truths=num_ground_truths,
                resolution=resolution,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Skipping record %d (%s)", index, exc)
            continue
        if not sample.get("ground_truths"):
            continue
        samples.append(sample)
        if num_samples is not None and len(samples) >= num_samples:
            break
    return samples


def load_coco(**kwargs: Any) -> List[Dict[str, Any]]:
    """Load COCO reference captions (see :func:`load_captioning_dataset`)."""
    kwargs.setdefault("dataset_name", COCO)
    return load_captioning_dataset(**kwargs)


def load_flickr30k(**kwargs: Any) -> List[Dict[str, Any]]:
    """Load Flickr30k reference captions (see :func:`load_captioning_dataset`)."""
    kwargs.setdefault("dataset_name", FLICKR30K)
    return load_captioning_dataset(**kwargs)


LOADERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    COCO: load_coco,
    FLICKR30K: load_flickr30k,
}


def dataset_statistics(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary statistics used for logging / guarding against empty loads."""
    counts = [len(s.get("ground_truths") or ()) for s in samples]
    with_pixels = sum(1 for s in samples if s.get("pixels") is not None)
    stats = {
        "num_samples": len(samples),
        "mean_ground_truths": (sum(counts) / len(counts)) if counts else 0.0,
        "min_ground_truths": min(counts) if counts else 0,
        "max_ground_truths": max(counts) if counts else 0,
        "num_with_pixels": with_pixels,
        "datasets": sorted({s.get("dataset") for s in samples if s.get("dataset")}),
    }
    return stats


def synthetic_samples(
    n: int = 300,
    *,
    dataset_name: str = COCO,
    num_ground_truths: int = DEFAULT_MAX_GROUND_TRUTHS,
    resolution: Optional[int] = 224,
    device: Any = None,
    dtype: Any = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Deterministic toy samples (smoke tests when no dataset is available)."""
    import torch  # lazy

    rng = random.Random(seed)
    vocab = ("a dog", "a cat", "a man", "a woman", "a beach", "a street", "a plate of food",
             "a bicycle", "a train", "a bird", "a boat", "a clock")
    samples: List[Dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)
    for i in range(n):
        caps = [
            f"{rng.choice(vocab)} in the {rng.choice(('morning', 'evening', 'sun', 'rain'))} ({j})"
            for j in range(num_ground_truths)
        ]
        pixels = None
        if resolution:
            shape = (1, 3, int(resolution), int(resolution))
            pixels = torch.rand(shape, generator=generator, dtype=torch.float32)
            if dtype is not None:
                pixels = pixels.to(dtype)
            if device is not None:
                pixels = pixels.to(device)
        samples.append(
            build_sample(sample_id=i, dataset=dataset_name, image_id=i, captions=caps,
                         split="synthetic", pixels=pixels, num_ground_truths=num_ground_truths)
        )
    return samples


# --------------------------------------------------------------------------- #
# Self test / CLI
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    logging.basicConfig(level=logging.INFO)
    assert normalize_dataset_name("coco") == COCO
    assert normalize_dataset_name("Flickr30k") == FLICKR30K
    assert normalize_dataset_name("ms-coco") == COCO

    record = {
        "image_id": 42,
        "sentences": [{"raw": f"caption {i}"} for i in range(7)],
    }
    caps = extract_captions(record, COCO, num_ground_truths=5)
    assert caps == [f"caption {i}" for i in range(5)], caps

    samples = synthetic_samples(4, num_ground_truths=5)
    assert len(samples) == 4
    assert all(len(s["ground_truths"]) == 5 for s in samples)
    assert all(s["ground_truths"] == s["captions"] == s["references"] for s in samples)
    assert all(s["pixels"].shape == (1, 3, 224, 224) for s in samples)
    stats = dataset_statistics(samples)
    assert stats["mean_ground_truths"] == 5.0, stats

    example = {"caption": "a single caption", "image": None}
    norm = normalize_captioning_sample(example, COCO, 0)
    assert norm["ground_truths"] == ["a single caption"], norm

    # subset of ground truths honoured
    assert len(extract_captions(record, COCO, num_ground_truths=3)) == 3

    print("coco_captioning self-test OK")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COCO/Flickr30k captioning data loader")
    parser.add_argument("--dataset", default=COCO)
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-ground-truths", type=int, default=DEFAULT_MAX_GROUND_TRUTHS)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        return _self_test()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    samples = load_captioning_dataset(
        args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        num_ground_truths=args.num_ground_truths,
        cache_dir=args.cache_dir,
        annotations_file=args.annotations_file,
        image_root=args.image_root,
        verbose=args.verbose or True,
    )
    print(json.dumps(dataset_statistics(samples), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
