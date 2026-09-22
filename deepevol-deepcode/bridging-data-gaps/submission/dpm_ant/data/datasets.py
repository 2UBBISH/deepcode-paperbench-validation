"""Data pipeline for DPMs-ANT (Section 5.2 *Datasets* / *Evaluation Metrics*).

The paper uses:

* Source datasets: ``FFHQ`` (Karras et al., 2020) and ``LSUN Church`` (Yu et al., 2015),
  following Ojha et al. (2021) (CDC).
* 10-shot target datasets with FFHQ as source: Babies, Sunglasses, Raphael Peale
  (a.k.a. "Raphael's paintings"), Sketches and "face paintings by Amedeo Modigliani".
* 10-shot target datasets with LSUN Church as source: Haunted Houses and Landscape
  drawings.
* Larger target datasets used *only* for FID: Sunglasses (2.5k) and Babies (2.7k)
  images, "Following DDPM-PA (Zhu et al., 2022)".

All pixel data is resized/cropped to ``256 x 256`` for the DDPM backbone.  For the LDM
backbone the same 256x256 images are provided and (optionally) encoded on the fly by the
frozen 64x64 autoencoder (``f = 4``); see :func:`to_latents`.

This module is deliberately dependency-light (``PIL``/``numpy``/``torch`` only) so that
it can be imported in environments without the full evaluation stack.
"""

from __future__ import annotations

import glob
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:  # pragma: no cover - PIL is a hard dependency in practice
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

LOGGER = logging.getLogger(__name__)

__all__ = [
    "TARGET_DATASETS",
    "SOURCE_DATASETS",
    "FID_TARGET_SETS",
    "DATASET_ALIASES",
    "ImageFolderDataset",
    "FewShotDataset",
    "TensorDataset",
    "LoopedDataset",
    "build_dataset",
    "build_target_dataset",
    "build_source_dataset",
    "build_fid_dataset",
    "build_target_loader",
    "infinite_loader",
    "sample_source_target_batch",
    "load_image",
    "list_images",
    "normalize_images",
    "denormalize_images",
    "to_latents",
    "resolve_split",
    "dataset_size",
]


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

#: Source domains (Section 5.2 "Datasets").
SOURCE_DATASETS: Dict[str, Dict[str, Any]] = {
    "ffhq": {"name": "ffhq", "dir_keys": ["ffhq"], "size": 70000},
    "lsun_church": {"name": "lsun_church", "dir_keys": ["lsun_church", "church", "lsun"], "size": 126227},
}

#: The seven 10-shot target datasets of the paper.  ``shot`` is the number of training
#: images that were *actually used* for adaptation (10-shot setting).
TARGET_DATASETS: Dict[str, Dict[str, Any]] = {
    "babies": {"name": "babies", "source": "ffhq", "shot": 10, "fid_size": 2700},
    "sunglasses": {"name": "sunglasses", "source": "ffhq", "shot": 10, "fid_size": 2500},
    "raphael": {
        "name": "raphael",
        "source": "ffhq",
        "shot": 10,
        "aliases": ["raphael_peale", "raphael_paintings", "raphaels_paintings"],
    },
    "sketches": {"name": "sketches", "source": "ffhq", "shot": 10},
    "amedeo": {
        "name": "amedeo",
        "source": "ffhq",
        "shot": 10,
        "aliases": ["amedeo_modigliani", "modigliani", "amedeos_paintings"],
    },
    "haunted_houses": {
        "name": "haunted_houses",
        "source": "lsun_church",
        "shot": 10,
        "aliases": ["haunted_houses", "haunted"],
    },
    "landscape_drawings": {
        "name": "landscape_drawings",
        "source": "lsun_church",
        "shot": 10,
        "aliases": ["landscape_drawings", "landscapes"],
    },
}

#: Larger target collections used for the FID evaluations of Table 2 only ("2.5k" for
#: Sunglasses and "2.7k" for Babies).
FID_TARGET_SETS: Dict[str, Dict[str, Any]] = {
    "babies": {"name": "babies", "dir_keys": ["babies_fid", "babies_2.7k", "babies_2700"], "size": 2700},
    "sunglasses": {
        "name": "sunglasses",
        "dir_keys": ["sunglasses_fid", "sunglasses_2.5k", "sunglasses_2500"],
        "size": 2500,
    },
}

#: Alias -> canonical name lookup (used by configs and CLI arguments).
DATASET_ALIASES: Dict[str, str] = {}
for _key, _spec in TARGET_DATASETS.items():
    DATASET_ALIASES.setdefault(_key, _key)
    for _alias in _spec.get("aliases", []):
        DATASET_ALIASES[_alias] = _key
for _key, _spec in SOURCE_DATASETS.items():
    DATASET_ALIASES.setdefault(_key, _key)
    for _alias in _spec.get("dir_keys", []):
        DATASET_ALIASES.setdefault(_alias, _key)


def resolve_split(name: str) -> str:
    """Canonicalise a dataset/split name (``"raphael_paintings"`` -> ``"raphael"``)."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return DATASET_ALIASES.get(key, key)


def dataset_size(name: str) -> int:
    """Number of images available for a registered dataset (best effort)."""
    key = resolve_split(name)
    if key in TARGET_DATASETS:
        spec = TARGET_DATASETS[key]
        return int(spec.get("size", spec.get("fid_size", spec.get("shot", 10))))
    if key in SOURCE_DATASETS:
        return int(SOURCE_DATASETS[key].get("size", 0))
    if key in FID_TARGET_SETS:
        return int(FID_TARGET_SETS[key].get("size", 0))
    return 0


# --------------------------------------------------------------------------------------
# Image IO helpers
# --------------------------------------------------------------------------------------

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".npy", ".pt", ".pth")


def list_images(root: str, recursive: bool = True, exts: Sequence[str] = _IMG_EXTS) -> List[str]:
    """Return a sorted list of image/tensor files below ``root``."""
    if not root:
        return []
    if os.path.isfile(root):
        return [root]
    if not os.path.isdir(root):
        return []
    files: List[str] = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(tuple(exts)):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and fn.lower().endswith(tuple(exts)):
                files.append(p)
    return sorted(files)


def load_image(path: str, size: Optional[int] = 256, resize: bool = True) -> torch.Tensor:
    """Load one image as a float tensor in ``[-1, 1]`` with shape ``(3, H, W)``."""
    if path.endswith(".npy"):
        arr = np.load(path)
        tensor = torch.from_numpy(np.asarray(arr)).float()
        if tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim == 4:  # stacked latents -> take first
            tensor = tensor[0]
        if size is not None and resize and tensor.ndim == 3 and tensor.shape[-2:] != (size, size):
            tensor = _resize_tensor(tensor, size)
        return _to_minus_one_one(tensor)
    if path.endswith((".pt", ".pth")):
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = tensor.get("image", next(iter(tensor.values())))
        tensor = torch.as_tensor(tensor).float()
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3) and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return _to_minus_one_one(tensor)

    if Image is None:  # pragma: no cover
        raise RuntimeError("PIL is required to load image files")
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = _resize_pil(img, size)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize_pil(img, size: int):
    """Resize preserving aspect ratio then center-crop to ``size x size`` (as in CDC/DDPM-PA)."""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _resize_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    import torch.nn.functional as F

    if x.shape[-2] == x.shape[-1]:
        out = F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
    else:
        scale = size / min(x.shape[-2], x.shape[-1])
        h, w = int(round(x.shape[-2] * scale)), int(round(x.shape[-1] * scale))
        out = F.interpolate(x.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
        top, left = (h - size) // 2, (w - size) // 2
        out = out[:, :, top : top + size, left : left + size]
    return out.squeeze(0)


def _to_minus_one_one(tensor: torch.Tensor) -> torch.Tensor:
    """Map an arbitrary-range image tensor to ``[-1, 1]``."""
    if tensor.numel() == 0:
        return tensor
    tmin, tmax = float(tensor.min()), float(tensor.max())
    if tmin >= -1.01 and tmax <= 1.01:
        return tensor.clamp(-1.0, 1.0)
    if tmin >= -0.01 and tmax <= 255.0:
        return (tensor / 127.5 - 1.0).clamp(-1.0, 1.0)
    if tmin >= -0.01 and tmax <= 1.01:
        return (tensor * 2.0 - 1.0).clamp(-1.0, 1.0)
    scale = max(abs(tmin), abs(tmax)) or 1.0
    return (tensor / scale).clamp(-1.0, 1.0)


def normalize_images(x: torch.Tensor) -> torch.Tensor:
    """``[0, 1] -> [-1, 1]``."""
    return x * 2.0 - 1.0


def denormalize_images(x: torch.Tensor) -> torch.Tensor:
    """``[-1, 1] -> [0, 1]``."""
    return (x + 1.0) * 0.5


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


class ImageFolderDataset(Dataset):
    """A flat folder (or nested tree) of images.

    Parameters
    ----------
    root:
        Directory, single file, or explicit list of file paths.
    size:
        Spatial size to resize/crop to (``256`` for DDPM, ``64`` for LDM latents, or
        ``None`` to keep the native resolution).
    num_images:
        Optional cap on the number of images (used for the few-shot sets and for
        subsampling the 2.5k/2.7k FID sets).
    """

    def __init__(
        self,
        root: Any,
        size: Optional[int] = 256,
        num_images: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        recursive: bool = True,
        seed: int = 0,
        resize: bool = True,
    ) -> None:
        self.root = root
        if isinstance(root, (list, tuple)):
            self.paths = [p for p in root if os.path.exists(p)]
        else:
            self.paths = list_images(str(root), recursive=recursive)
        if num_images is not None and num_images > 0:
            if len(self.paths) > num_images:
                rng = random.Random(seed)
                self.paths = sorted(rng.sample(self.paths, num_images))
            else:
                LOGGER.warning(
                    "Requested %d images from %s but only found %d", num_images, root, len(self.paths)
                )
        self.size = size
        self.transform = transform
        self.resize = resize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = load_image(self.paths[index], size=self.size, resize=self.resize)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(n={len(self)}, size={self.size})"


class FewShotDataset(ImageFolderDataset):
    """The ``k``-shot target set (``k = 10`` unless stated otherwise).

    Handles both the "full folder with a subset of shots" layout and a folder that
    literally contains ``k`` images.
    """

    def __init__(
        self,
        root: Any,
        shots: int = 10,
        size: Optional[int] = 256,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        repeat: int = 1,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, size=size, num_images=shots, transform=transform, seed=seed, **kwargs)
        self.shots = shots
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths) * self.repeat

    def __getitem__(self, index: int) -> torch.Tensor:  # type: ignore[override]
        return super().__getitem__(index % max(1, len(self.paths)))


class TensorDataset(Dataset):
    """In-memory dataset over a ``(N, C, H, W)`` tensor or a list of tensors."""

    def __init__(self, data: torch.Tensor, repeat: int = 1) -> None:
        self.data = data
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return int(self.data.shape[0]) * self.repeat

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index % int(self.data.shape[0])]


class LoopedDataset(Dataset):
    """Infinite-length view over a (small) dataset; used for the 10-shot training loop."""

    def __init__(self, dataset: Dataset, length: int) -> None:
        self.dataset = dataset
        self.length = max(1, int(length))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % max(1, len(self.dataset))]


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def _cfg_get(cfg: Optional[Dict[str, Any]], *paths: str, default: Any = None) -> Any:
    """Look up the first existing dotted path in a nested config dict."""
    if cfg is None:
        return default
    for path in paths:
        node: Any = cfg
        ok = True
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok:
            return node
    return default


def _resolve_dir(cfg: Optional[Dict[str, Any]], name: str, source: Optional[str] = None) -> str:
    """Resolve the on-disk directory of a dataset from the config ``data`` block."""
    candidates = [
        f"data.target_dirs.{name}",
        f"data.{name}",
        f"data.dirs.{name}",
        f"data.source_dirs.{name}",
        f"data.targets.{name}",
        f"data.fid_target_dirs.{name}",
        f"data.{name}_dir",
    ]
    for key in candidates:
        val = _cfg_get(cfg, key)
        if val:
            return str(val)
    if source:
        for key in (f"data.source_dirs.{source}", f"data.{source}_dir", f"data.{source}"):
            val = _cfg_get(cfg, key)
            if val and name in str(val):
                return str(val)
    return ""


def build_dataset(
    root: Any,
    size: Optional[int] = None,
    shots: Optional[int] = None,
    repeat: int = 1,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    data: Optional[torch.Tensor] = None,
    seed: int = 0,
    **kwargs: Any,
) -> Dataset:
    """Generic factory: tensor -> :class:`TensorDataset`, else folder dataset."""
    if data is not None:
        return TensorDataset(data, repeat=repeat)
    if shots is not None and shots > 0:
        return FewShotDataset(root, shots=shots, size=size, transform=transform, repeat=repeat, seed=seed, **kwargs)
    return ImageFolderDataset(root, size=size, num_images=shots, transform=transform, seed=seed, **kwargs)


def _ldm_latent_size(cfg: Optional[Dict[str, Any]]) -> int:
    return int(
        _cfg_get(
            cfg,
            "data.ldm_latent_size",
            "models.ldm.latent_size",
            "data.latent_size",
            default=64,
        )
    )


def build_target_dataset(
    cfg: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    shots: Optional[int] = None,
    size: Optional[int] = None,
    backbone: str = "ddpm",
    repeat: int = 1,
    seed: int = 0,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    data: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Dataset:
    """Build the few-shot target dataset used for adaptation / Eq. (8).

    ``shots`` defaults to 10 (the paper's 10-shot setting); the classifier ablation of
    Table 3 uses ``shots=100``.  For ``backbone='ldm'`` the images are returned at the
    LDM latent resolution (64x64) unless an autoencoder is applied later by
    :func:`to_latents`.
    """
    if data is not None:
        return TensorDataset(data, repeat=repeat)
    shots = int(shots if shots is not None else _cfg_get(cfg, "classifier.target_pool", default=10) or 10)
    if size is None:
        size = _ldm_latent_size(cfg) if str(backbone).lower() == "ldm" else int(
            _cfg_get(cfg, "data.image_size", default=256) or 256
        )
    root = _cfg_get(cfg, f"data.target_dirs.{target}", f"data.{target}", f"data.targets.{target}", default="")
    if not root:
        raise FileNotFoundError(
            f"No directory configured for target dataset '{target}'. "
            "Set data.target_dirs.<name> in the config or pass --data-root."
        )
    return FewShotDataset(root, shots=shots, size=size, repeat=max(1, repeat), seed=seed, transform=transform, **kwargs)


def build_source_dataset(
    cfg: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    num_images: Optional[int] = None,
    size: Optional[int] = None,
    backbone: str = "ddpm",
    seed: int = 0,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    **kwargs: Any,
) -> Dataset:
    """Build a view over the source dataset (FFHQ or LSUN Church)."""
    source = resolve_split(source or "ffhq")
    size = size if size is not None else int(_cfg_get(cfg, "data.image_size", default=256) or 256)
    root = _resolve_dir(cfg, source) or _cfg_get(cfg, f"data.source_dirs.{source}", default="")
    if not root:
        raise FileNotFoundError(
            f"No directory configured for source dataset '{source}'. "
            "Set data.source_dirs.<name> in the config or pass --data-root."
        )
    return ImageFolderDataset(root, size=size, num_images=num_images, transform=transform, **kwargs)


def build_fid_dataset(
    cfg: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    size: Optional[int] = None,
    backbone: str = "ddpm",
    **kwargs: Any,
) -> Dataset:
    """Build the *larger* target set used exclusively for FID (Babies 2.7k / Sunglasses 2.5k)."""
    key = resolve_split(target or "")
    spec = FID_TARGET_SETS.get(key, {})
    root = ""
    for dir_key in spec.get("dir_keys", []):
        root = _cfg_get(cfg, f"data.fid_target_dirs.{dir_key}", f"data.fid_target_dirs.{key}", f"data.{dir_key}", default="")
        if root:
            break
    if not root:
        # Fall back to the 10-shot directory (FID then uses whatever is available).
        root = _cfg_get(cfg, f"data.target_dirs.{key}", default="")
    if not root:
        raise FileNotFoundError(f"No directory configured for the FID target set '{target}'.")
    size = size if size is not None else int(_cfg_get(cfg, "data.image_size", default=256) or 256)
    num_images = int(spec.get("size", 0)) or None
    return ImageFolderDataset(root, size=size, num_images=num_images, **kwargs)


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------


def build_target_loader(
    dataset: Dataset,
    batch_size: int = 40,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = True,
    infinite: bool = False,
    length: Optional[int] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Iterable[torch.Tensor]:
    """Wrap a target dataset in a ``DataLoader`` (optionally infinite).

    The default ``batch_size`` of 40 matches the batch size used for ANT training in
    Section 5.2 ("approximately 300 iterations and a batch size of 40").  For the
    10-shot datasets ``shuffle=True`` cycles through the samples, and ``infinite=True``
    keeps yielding batches for the full training loop.
    """
    if infinite:
        return infinite_loader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            device=device,
            length=length,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    return loader


def infinite_loader(
    dataset: Dataset,
    batch_size: int = 40,
    num_workers: int = 0,
    seed: int = 0,
    device: Optional[torch.device] = None,
    length: Optional[int] = None,
):
    """Yield an endless stream of shuffled batches from ``dataset``."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    while True:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            generator=generator,
        )
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            if device is not None:
                batch = batch.to(device)
            yield batch


def sample_source_target_batch(
    source_dataset: Dataset,
    target_dataset: Dataset,
    batch_size: int,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a matched-size batch from the source and target datasets.

    Used by the classifier fine-tuning stage, which (per the addendum "Classifier
    Training (Section 5.2)") is trained on noised *source* and *target* images.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    src_idx = rng.randint(0, max(1, len(source_dataset)), size=batch_size)
    tgt_idx = rng.randint(0, max(1, len(target_dataset)), size=batch_size)
    src = torch.stack([source_dataset[int(i)] for i in src_idx])
    tgt = torch.stack([target_dataset[int(i)] for i in tgt_idx])
    if device is not None:
        src, tgt = src.to(device), tgt.to(device)
    return src, tgt


# --------------------------------------------------------------------------------------
# LDM latent helpers
# --------------------------------------------------------------------------------------


def to_latents(
    images: torch.Tensor,
    autoencoder: Optional[torch.nn.Module] = None,
    scale_factor: float = 0.18215,
    mode: str = "mode",
) -> torch.Tensor:
    """Encode ``[-1, 1]`` pixel images to LDM latents with a frozen autoencoder.

    Falls back to spatial downsampling (``f = 4``) when no autoencoder is available, so
    the LDM pipeline remains runnable in a reduced environment.
    """
    if autoencoder is None:
        import torch.nn.functional as F

        return F.interpolate(images, scale_factor=1.0 / 4.0, mode="bilinear", align_corners=False)
    for name in ("encode", "encode_first_stage"):
        fn = getattr(autoencoder, name, None)
        if callable(fn):
            out = fn(images)
            if not torch.is_tensor(out):  # FirstStage object with .mode()
                out = out.mode() if mode == "mode" and hasattr(out, "mode") else out
            out = torch.as_tensor(out)
            if hasattr(autoencoder, "scale_factor"):
                out = out * float(getattr(autoencoder, "scale_factor"))
            else:
                out = out * scale_factor
            return out
    return images
