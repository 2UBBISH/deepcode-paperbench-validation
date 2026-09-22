"""Dataset preparation utilities for DPMs-ANT (few-shot image generation).

This module turns *raw* downloads of the source/target corpora into the exact
directory layout that :mod:`dpm_ant.data.datasets` expects, and pre-processes
every image the way the paper describes in Section 5.2 ("Datasets" /
"Evaluation Metrics"):

    * **Source domain** -- FFHQ (Karras et al., 2020b) and LSUN Church
      (Yu et al., 2015), used to adapt *from*.
    * **Target domain (10-shot)** -- Babies, Sunglasses, Raphael Peale,
      Sketches, face paintings by Amedeo Modigliani (FFHQ source) and
      Haunted Houses, Landscape drawings (LSUN Church source).
    * **FID target sets** -- larger target corpora of 2.5k (Sunglasses) and
      2.7k (Babies) images, following DDPM-PA (Zhu et al., 2022), because FID
      on 10-shot sets is unstable.  (Section 5.2, "Evaluation Metrics".)

Pre-processing rules implemented here:

    * images are resized (aspect preserving) and center-cropped to
      ``256 x 256`` for the DDPM backbone;
    * the LDM backbone consumes the *same* 256x256 images and either encodes
      them on the fly or uses the cached 64x64 latents produced by
      :func:`prepare_latents` (frozen autoencoder, scaling factor 0.18215);
    * everything is stored on disk in a flat image folder per dataset, which
      :class:`~dpm_ant.data.datasets.ImageFolderDataset` reads back.

Nothing here is required for the toy 2-D experiment; it exists so that the
main experiments (Tables 1-4, Figures 3/5) can be reproduced end to end.

Usage
-----
::

    # 1) show the directory layout this module expects / creates
    python -m dpm_ant.data.prepare_data --print-layout

    # 2) materialise the 10-shot target folders from an already-extracted raw copy
    python -m dpm_ant.data.prepare_data --raw-root /data/raw --root /data/dpm_ant \
        --prepare-all --overwrite

    # 3) same, but unzip a downloaded archive first
    python -m dpm_ant.data.prepare_data --task ffhq_sunglasses \
        --archive ~/Downloads/sunglasses.zip --download

    # 4) cache LDM latents for a prepared folder (requires a checkpoint)
    python -m dpm_ant.data.prepare_data --latents targets/sunglasses \
        --ldm-ckpt models/ldm/ldm_256.ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import tarfile
import zipfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .datasets import (
    DATASET_ALIASES,
    FID_TARGET_SETS,
    SOURCE_DATASETS,
    TARGET_DATASETS,
    resolve_split,
)

LOGGER = logging.getLogger("dpm_ant.data.prepare_data")

__all__ = [
    # layout
    "DEFAULT_DATA_ROOT",
    "DEFAULT_LAYOUT",
    "SOURCE_URLS",
    "TARGET_URLS",
    "FID_URLS",
    "default_layout",
    "ensure_layout",
    "print_layout",
    "resolve_data_dirs",
    "apply_config_layout",
    "write_data_config",
    # download / archive
    "download_file",
    "extract_archive",
    "download_and_extract",
    # image processing
    "resize_and_center_crop",
    "save_image_tensor",
    "prepare_image_folder",
    "sample_few_shot",
    "sample_fid_subset",
    "copy_or_link",
    # LDM
    "prepare_latents",
    # orchestration
    "prepare_task",
    "prepare_all",
    "prepare_from_archives",
    "verify_task_data",
    "summarize_layout",
    "main",
]

# --------------------------------------------------------------------------------------
# Layout / registries
# --------------------------------------------------------------------------------------

#: Environment variable that can override the dataset root.
DATA_ROOT_ENV = "DPM_ANT_DATA"

#: Default dataset root used when neither config nor CLI specifies one.
DEFAULT_DATA_ROOT = os.environ.get(
    DATA_ROOT_ENV, os.path.join(os.getcwd(), "data")
)

#: Canonical on-disk layout, relative to the dataset root.  Keys are the
#: canonical dataset names from :mod:`dpm_ant.data.datasets`.
DEFAULT_LAYOUT: Dict[str, Dict[str, str]] = {
    "source": {
        "ffhq": "source/ffhq",
        "lsun_church": "source/lsun_church",
    },
    "target": {
        "babies": "targets/babies",
        "sunglasses": "targets/sunglasses",
        "raphael": "targets/raphael",
        "sketches": "targets/sketches",
        "amedeo": "targets/amedeo",
        "haunted_houses": "targets/haunted_houses",
        "landscape_drawings": "targets/landscape_drawings",
    },
    "fid": {
        "babies": "fid/babies",
        "sunglasses": "fid/sunglasses",
    },
    "raw": "raw",
    "latents": "latents",
}

#: Optional download URLs.  The authors of CDC / DDPM-PA / DPMs-ANT released
#: their few-shot corpora through project pages rather than a stable CDN, so we
#: deliberately leave these empty and let the user supply ``--url`` (or a
#: ``data.urls.<name>`` config key).  See ``baselines/README.md`` for pointers.
SOURCE_URLS: Dict[str, Optional[str]] = {"ffhq": None, "lsun_church": None}

TARGET_URLS: Dict[str, Optional[str]] = {
    "babies": None,
    "sunglasses": None,
    "raphael": None,
    "sketches": None,
    "amedeo": None,
    "haunted_houses": None,
    "landscape_drawings": None,
}

FID_URLS: Dict[str, Optional[str]] = {"babies": None, "sunglasses": None}

#: Number of images in the larger FID evaluation sets (Section 5.2).
FID_SET_SIZE: Dict[str, int] = {
    "babies": int(FID_TARGET_SETS.get("babies", {}).get("size", 2700)),
    "sunglasses": int(FID_TARGET_SETS.get("sunglasses", {}).get("size", 2500)),
}

#: Default number of shots for the 10-shot tasks.
DEFAULT_SHOTS = 10

_IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".npy", ".pt", ".pth")


def default_layout(root: str = DEFAULT_DATA_ROOT) -> Dict[str, Dict[str, str]]:
    """Return the canonical layout with every relative path joined to ``root``.

    The returned dict mirrors :data:`DEFAULT_LAYOUT` but holds absolute paths
    and keeps the ``raw`` / ``latents`` helper entries.
    """
    layout: Dict[str, Dict[str, str]] = {}
    for group, entries in DEFAULT_LAYOUT.items():
        if isinstance(entries, dict):
            layout[group] = {k: os.path.join(root, v) for k, v in entries.items()}
        else:  # pragma: no cover - defensive, DEFAULT_LAYOUT is all dicts
            layout[group] = {str(entries): os.path.join(root, str(entries))}
    return layout


def ensure_layout(root: str = DEFAULT_DATA_ROOT, verbose: bool = True) -> Dict[str, Dict[str, str]]:
    """Create every directory of the canonical layout and return it."""
    layout = default_layout(root)
    for group, entries in layout.items():
        for _name, path in entries.items():
            os.makedirs(path, exist_ok=True)
    if verbose:
        LOGGER.info("dataset root ready: %s", os.path.abspath(root))
    return layout


def print_layout(root: str = DEFAULT_DATA_ROOT) -> str:
    """Human readable listing of the expected directory tree (no side effects)."""
    layout = default_layout(root)
    lines = [f"dataset root: {os.path.abspath(root)}"]
    for group in ("source", "target", "fid"):
        lines.append(f"  {group}/")
        for name, path in sorted(layout[group].items()):
            rel = os.path.relpath(path, root)
            meta = ""
            if group == "target":
                spec = TARGET_DATASETS.get(name, {})
                meta = f"   ({spec.get('shot', DEFAULT_SHOTS)}-shot, source={spec.get('source', '?')})"
            elif group == "fid":
                meta = f"   ({FID_SET_SIZE.get(name, '?')} images)"
            elif group == "source":
                meta = f"   (source dataset)"
            lines.append(f"      {rel}{meta}")
    lines.append(f"  raw/          (untouched downloads: archives or image dumps)")
    lines.append(f"  latents/      (optional cached 64x64 LDM latents)")
    text = "\n".join(lines)
    print(text)
    return text


# --------------------------------------------------------------------------------------
# Config plumbing -- keep prepare_data and datasets.py in sync
# --------------------------------------------------------------------------------------

_DEFAULT_CONFIG_PATHS = (
    os.path.join("configs", "default.yaml"),
    os.path.join("configs", "per_task.yaml"),
)


def _maybe_load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.isfile(path):
        LOGGER.warning("config file not found: %s", path)
        return {}
    try:
        import yaml  # local import: keeps this module importable without pyyaml
    except Exception:  # pragma: no cover - pyyaml is in requirements
        LOGGER.warning("pyyaml unavailable; cannot read %s", path)
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (other or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project config, merging ``default.yaml`` with ``per_task.yaml``.

    ``path`` may point at a single YAML file; otherwise the two default configs
    are merged (per-task overrides win).
    """
    if path:
        return _maybe_load_yaml(path)
    cfg: Dict[str, Any] = {}
    for candidate in _DEFAULT_CONFIG_PATHS:
        _deep_update(cfg, _maybe_load_yaml(candidate))
    return cfg


def resolve_data_dirs(
    cfg: Optional[Dict[str, Any]] = None,
    root: Optional[str] = None,
    create: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Resolve the on-disk directory of every dataset.

    Resolution order (first hit wins) mirrors
    :mod:`dpm_ant.data.datasets` so that ``prepare_data`` writes exactly where
    the loaders read:

    * source: ``data.source_dirs.<name>`` -> ``data.dirs.<name>`` -> ``data.<name>``
      -> ``sources.<name>`` -> canonical layout;
    * target: ``data.target_dirs.<name>`` -> ``data.dirs.<name>`` -> ``data.<name>``
      -> ``targets.<name>`` -> canonical layout;
    * FID:    ``data.fid_target_dirs.<name>`` -> ``data.target_dirs.<name>``
      -> ``data.fid_dirs.<name>`` -> ``fid.<name>`` -> canonical layout.
    """
    cfg = cfg or {}
    data_cfg = cfg.get("data", cfg) or {}
    root = root or data_cfg.get("root") or cfg.get("data_root") or DEFAULT_DATA_ROOT
    layout = ensure_layout(root, verbose=False) if create else default_layout(root)

    def _lookup(keys: Sequence[Tuple[str, str]], group: str, name: str) -> str:
        for block, key in keys:
            container = cfg.get(block)
            if isinstance(container, dict):
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value
        # also accept flat keys such as data.target_dirs: {babies: path}
        for block in ("data", ""):
            container = data_cfg if not block else cfg.get(block)
            if not isinstance(container, dict):
                continue
            for sub in ("source_dirs", "target_dirs", "fid_target_dirs", "fid_dirs", "dirs"):
                sub_container = container.get(sub)
                if isinstance(sub_container, dict) and isinstance(sub_container.get(name), str):
                    return sub_container[name]
        return layout[group][name]

    source_dirs: Dict[str, str] = {}
    for name in SOURCE_DATASETS:
        source_dirs[name] = _lookup(
            (("data", name), ("sources", name), ("dirs", name)), "source", name
        )
    target_dirs: Dict[str, str] = {}
    for name in TARGET_DATASETS:
        target_dirs[name] = _lookup(
            (("data", name), ("targets", name), ("dirs", name)), "target", name
        )
    fid_dirs: Dict[str, str] = {}
    for name in FID_TARGET_SETS:
        fid_dirs[name] = _lookup(
            (
                ("data", name),
                ("targets", name),
                ("fid", name),
                ("fid_targets", name),
            ),
            "fid",
            name,
        )
    return {
        "root": root,
        "source_dirs": source_dirs,
        "target_dirs": target_dirs,
        "fid_target_dirs": fid_dirs,
        "raw_dir": layout["raw"],
        "latents_dir": layout["latents"],
    }


def apply_config_layout(
    cfg: Optional[Dict[str, Any]] = None,
    root: Optional[str] = None,
    create: bool = True,
) -> Dict[str, Any]:
    """Return ``cfg`` with ``data.source_dirs`` / ``data.target_dirs`` /
    ``data.fid_target_dirs`` / ``data.root`` filled in from the resolved layout.

    This is the canonical way for entry points to obtain a config that both
    ``prepare_data`` and ``datasets`` agree on.
    """
    import copy

    cfg = copy.deepcopy(cfg or {})
    dirs = resolve_data_dirs(cfg, root=root, create=create)
    data_cfg = cfg.setdefault("data", {})
    if not isinstance(data_cfg, dict):  # pragma: no cover - malformed config
        data_cfg = {}
        cfg["data"] = data_cfg
    data_cfg.setdefault("root", dirs["root"])
    data_cfg.setdefault("source_dirs", {}).update(dirs["source_dirs"])
    data_cfg.setdefault("target_dirs", {}).update(dirs["target_dirs"])
    data_cfg.setdefault("fid_target_dirs", {}).update(dirs["fid_target_dirs"])
    data_cfg.setdefault("raw_dir", dirs["raw_dir"])
    data_cfg.setdefault("latents_dir", dirs["latents_dir"])
    return cfg


def write_data_config(
    path: str = os.path.join("configs", "data.yaml"),
    root: str = DEFAULT_DATA_ROOT,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Dump the resolved data layout to a YAML fragment for inclusion in a config."""
    dirs = resolve_data_dirs(cfg, root=root)
    payload = {
        "data": {
            "root": dirs["root"],
            "source_dirs": dirs["source_dirs"],
            "target_dirs": dirs["target_dirs"],
            "fid_target_dirs": dirs["fid_target_dirs"],
            "raw_dir": dirs["raw_dir"],
            "latents_dir": dirs["latents_dir"],
        }
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        import yaml

        with open(path, "w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
    except Exception:  # pragma: no cover - pyyaml missing
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
    LOGGER.info("wrote data config -> %s", path)
    return path


# --------------------------------------------------------------------------------------
# Download / archive handling
# --------------------------------------------------------------------------------------


def download_file(
    url: str,
    dest: str,
    overwrite: bool = False,
    timeout: int = 60,
    progress: bool = True,
) -> str:
    """Download ``url`` to ``dest`` (stdlib only, resumable is *not* attempted)."""
    if os.path.isfile(dest) and not overwrite:
        LOGGER.info("already downloaded: %s", dest)
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)

    import urllib.request

    tmp = dest + ".part"
    LOGGER.info("downloading %s -> %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            seen = 0
            next_report = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                seen += len(chunk)
                if progress and total and seen >= next_report:
                    pct = 100.0 * seen / total
                    LOGGER.info("  %5.1f%% (%.1f/%.1f MB)", pct, seen / 2**20, total / 2**20)
                    next_report = seen + max(total // 10, 1 << 20)
    except Exception as exc:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    os.replace(tmp, dest)
    return dest


def extract_archive(path: str, dest: str, overwrite: bool = False, strip: bool = False) -> str:
    """Extract a ``.zip`` / ``.tar[.gz|.bz2|.xz]`` archive into ``dest``.

    Returns the directory that most likely holds the images (the single
    top-level folder when the archive has exactly one, otherwise ``dest``).
    """
    os.makedirs(dest, exist_ok=True)
    LOGGER.info("extracting %s -> %s", path, dest)
    lower = path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            members = zf.namelist()
            if not overwrite:
                members = [m for m in members if not os.path.exists(os.path.join(dest, m))]
            zf.extractall(dest, members=members)
            names = zf.namelist()
    elif any(lower.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        with tarfile.open(path) as tf:
            tf.extractall(dest)
            names = tf.getnames()
    else:
        raise ValueError(f"unsupported archive type: {path}")

    tops = {n.split("/")[0].strip("/") for n in names if n.strip("/")}
    nested = os.path.join(dest, next(iter(tops))) if len(tops) == 1 else dest
    if strip and len(tops) == 1:
        LOGGER.info("archive has a single top-level dir; using it directly: %s", nested)
    return nested if os.path.isdir(nested) else dest


def download_and_extract(
    url: str,
    raw_dir: str,
    name: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Download ``url`` into ``raw_dir`` and extract it; returns the image root."""
    name = name or os.path.basename(url.split("?")[0]) or "archive"
    archive_path = os.path.join(raw_dir, name)
    download_file(url, archive_path, overwrite=overwrite)
    stem = os.path.splitext(name)[0]
    if stem.endswith(".tar"):  # pragma: no cover - .tar.gz case
        stem = os.path.splitext(stem)[0]
    return extract_archive(archive_path, os.path.join(raw_dir, stem), overwrite=overwrite)


# --------------------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------------------


def _resize_center_crop(img, size: int):
    """Aspect-preserving resize + center crop, matching the DDPM pipeline.

    Mirrors :func:`dpm_ant.data.datasets.load_image`: the shorter edge is scaled
    to ``size`` and the longer edge is center-cropped.  Accepts a PIL image and
    returns a PIL image.
    """
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    width, height = img.size
    if width == size and height == size:
        return img
    scale = size / min(width, height)
    new_w, new_h = max(size, int(round(width * scale))), max(size, int(round(height * scale)))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    img = img.resize((new_w, new_h), resample)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def save_image_tensor(
    image: Any,
    path: str,
    size: Optional[int] = None,
    jpg_quality: int = 95,
) -> str:
    """Save a tensor / PIL image / numpy array as an image file.

    Tensors are expected in ``[-1, 1]`` (DDPM convention) but ``[0, 1]`` and
    ``[0, 255]`` inputs are auto-detected, matching
    :func:`dpm_ant.data.datasets.denormalize_images`.
    """
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if torch.is_tensor(image):
        tensor = image.detach().float().cpu().squeeze()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        lo, hi = float(tensor.min()), float(tensor.max())
        if lo < -0.01:  # [-1, 1] -> [0, 1]
            tensor = (tensor + 1.0) / 2.0
        elif hi > 1.5:  # [0, 255] -> [0, 1]
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0)
        array = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
        img = Image.fromarray(array.squeeze() if array.shape[-1] == 1 else array, mode=None if array.shape[-1] != 1 else "L")
        if array.shape[-1] == 3:
            img = Image.fromarray(array, "RGB")
    elif isinstance(image, Image.Image):
        img = image
    else:  # numpy array
        import numpy as np

        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.max() <= 1.0 + 1e-6:
                array = (array * 255.0 + 0.5).astype("uint8")
            else:
                array = array.astype("uint8")
        img = Image.fromarray(array)

    if size is not None:
        img = _resize_center_crop(img, int(size))

    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        img.convert("RGB").save(path, quality=int(jpg_quality))
    else:
        img.save(path)
    return path


def _list_raw_images(raw: str, recursive: bool = True) -> List[str]:
    exts = _IMG_EXTS
    found: List[str] = []
    if recursive:
        for dirpath, _dirs, files in os.walk(raw):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in exts:
                    found.append(os.path.join(dirpath, fname))
    else:
        for fname in os.listdir(raw):
            if os.path.splitext(fname)[1].lower() in exts:
                found.append(os.path.join(raw, fname))
    found.sort()
    return found


def prepare_image_folder(
    src: str,
    dest: str,
    size: Optional[int] = 256,
    limit: Optional[int] = None,
    recursive: bool = True,
    overwrite: bool = False,
    start_index: int = 0,
    name_prefix: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Resize/crop every image of ``src`` into the flat folder ``dest``.

    Returns the number of images written.  Inputs may be PIL-loadable files or
    ``.npy`` / ``.pt`` tensors (useful for pre-computed LDM latents).
    """
    if not os.path.isdir(src):
        raise FileNotFoundError(f"raw image directory not found: {src}")
    paths = _list_raw_images(src, recursive=recursive)
    if start_index:
        paths = paths[int(start_index) :]
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        LOGGER.warning("no images found under %s", src)
        return 0

    os.makedirs(dest, exist_ok=True)
    prefix = name_prefix if name_prefix is not None else os.path.basename(os.path.normpath(dest))
    written = 0
    for i, path in enumerate(paths):
        out = os.path.join(dest, f"{prefix}_{i:06d}.png")
        if os.path.isfile(out) and not overwrite:
            written += 1
            continue
        try:
            if path.lower().endswith((".npy", ".pt", ".pth")):
                tensor = torch.load(path, map_location="cpu") if path.lower().endswith((".pt", ".pth")) else torch.from_numpy(__import__("numpy").load(path))
                if isinstance(tensor, dict):
                    tensor = tensor.get("latent", tensor.get("image", next(iter(tensor.values()))))
                save_image_tensor(tensor, out, size=size if tensor.dim() == 3 and tensor.shape[0] in (1, 3) else None)
            else:
                from PIL import Image

                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if size is not None:
                        img = _resize_center_crop(img, int(size))
                    img.save(out)
            written += 1
        except Exception as exc:  # pragma: no cover - corrupt file tolerance
            LOGGER.warning("skipping %s (%s)", path, exc)
        if verbose and (i + 1) % 500 == 0:
            LOGGER.info("  %s: %d/%d", dest, i + 1, len(paths))
    LOGGER.info("prepared %d images -> %s", written, dest)
    return written


def sample_few_shot(
    src: str,
    dest: str,
    shots: int = DEFAULT_SHOTS,
    size: Optional[int] = 256,
    seed: int = 0,
    recursive: bool = True,
    offset: int = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> int:
    """Select ``shots`` images from ``src`` (seeded, deterministic) into ``dest``.

    Selection mirrors :class:`dpm_ant.data.datasets.FewShotDataset`, which draws
    a seeded random subset from the sorted file listing; using the same RNG
    recipe keeps the prepared folder reproducible.
    """
    if not os.path.isdir(src):
        raise FileNotFoundError(f"raw image directory not found: {src}")
    paths = _list_raw_images(src, recursive=recursive)
    if not paths:
        LOGGER.warning("no images found under %s", src)
        return 0
    rng = random.Random(seed)
    k = min(int(shots), len(paths))
    chosen = sorted(rng.sample(range(len(paths)), k))
    if offset:
        chosen = [(i + int(offset)) % len(paths) for i in chosen]
    os.makedirs(dest, exist_ok=True)

    written = 0
    for j, idx in enumerate(chosen):
        out = os.path.join(dest, f"shot_{j:02d}.png")
        if os.path.isfile(out) and not overwrite:
            written += 1
            continue
        path = paths[idx]
        try:
            from PIL import Image

            with Image.open(path) as img:
                img = img.convert("RGB")
                if size is not None:
                    img = _resize_center_crop(img, int(size))
                img.save(out)
            written += 1
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("skipping %s (%s)", path, exc)
    if verbose:
        LOGGER.info("sampled %d-shot target set -> %s", written, dest)
    return written


def sample_fid_subset(
    src: str,
    dest: str,
    num_images: int = 2500,
    size: Optional[int] = 256,
    seed: int = 0,
    recursive: bool = True,
    shuffle: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> int:
    """Build a larger FID evaluation set (Sunglasses 2.5k / Babies 2.7k).

    By default the *first* ``num_images`` files of the sorted listing are copied,
    which matches the deterministic ``num_images`` subsampling used by
    :class:`dpm_ant.data.datasets.ImageFolderDataset`.  Set ``shuffle=True`` for
    a seeded random subset instead.
    """
    if not os.path.isdir(src):
        raise FileNotFoundError(f"raw image directory not found: {src}")
    paths = _list_raw_images(src, recursive=recursive)
    if not paths:
        LOGGER.warning("no images found under %s", src)
        return 0
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)
    paths = paths[: int(num_images)]
    os.makedirs(dest, exist_ok=True)
    written = 0
    for i, path in enumerate(paths):
        out = os.path.join(dest, f"img_{i:06d}.png")
        if os.path.isfile(out) and not overwrite:
            written += 1
            continue
        try:
            from PIL import Image

            with Image.open(path) as img:
                img = img.convert("RGB")
                if size is not None:
                    img = _resize_center_crop(img, int(size))
                img.save(out)
            written += 1
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("skipping %s (%s)", path, exc)
    if verbose:
        LOGGER.info("prepared %d-image FID set -> %s", written, dest)
    return written


def copy_or_link(src: str, dst: str, mode: str = "copy", overwrite: bool = False) -> str:
    """Place ``src`` at ``dst`` using ``copy`` (default), ``symlink`` or ``hardlink``.

    Symlinking is handy for the large FID/source corpora; only the 10-shot
    folders are ever written to during preparation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    if os.path.exists(dst) or os.path.islink(dst):
        if not overwrite:
            return dst
        os.remove(dst)
    mode = (mode or "copy").lower()
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


# --------------------------------------------------------------------------------------
# LDM latent preprocessing
# --------------------------------------------------------------------------------------


def prepare_latents(
    image_folder: str,
    out_dir: str,
    autoencoder: Any = None,
    ckpt: Optional[str] = None,
    size: int = 256,
    scale_factor: float = 0.18215,
    batch_size: int = 8,
    device: Optional[str] = None,
    overwrite: bool = False,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Cache 64x64 LDM latents for ``image_folder`` with the frozen autoencoder.

    Writes one ``.pt`` per image plus a ``manifest.json`` recording the scale
    factor and shape so the training code can validate compatibility.  The LDM
    backbone consumes the same 256x256 images as the DDPM one (Section 5.2).
    """
    from .datasets import load_image, to_latents

    if autoencoder is None and ckpt:
        from ..models.ldm_loader import load_autoencoder  # local import: torch heavy

        autoencoder, _cfg = load_autoencoder(ckpt, device=device or "cpu"), None
    if autoencoder is None:
        raise ValueError(
            "prepare_latents requires either an `autoencoder` object or a `ckpt` path "
            "to the frozen LDM autoencoder."
        )

    paths = _list_images_for_latents(image_folder)
    if limit:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"no images found under {image_folder}")

    os.makedirs(out_dir, exist_ok=True)
    latents: List[str] = []
    for start in range(0, len(paths), max(1, int(batch_size))):
        chunk = paths[start : start + max(1, int(batch_size))]
        batch = torch.stack([load_image(p, size=size) for p in chunk], dim=0)
        with torch.no_grad():
            z = to_latents(batch, autoencoder=autoencoder, scale_factor=scale_factor, mode="mode")
        z = z.detach().cpu()
        for offset, path in enumerate(chunk):
            name = os.path.splitext(os.path.basename(path))[0] + ".pt"
            out_path = os.path.join(out_dir, name)
            if os.path.isfile(out_path) and not overwrite:
                latents.append(out_path)
                continue
            torch.save(z[offset].clone(), out_path)
            latents.append(out_path)
        if verbose:
            LOGGER.info("  latents %d/%d", min(start + len(chunk), len(paths)), len(paths))

    sample = torch.load(latents[0], map_location="cpu")
    manifest = {
        "source": os.path.abspath(image_folder),
        "count": len(latents),
        "latent_shape": list(sample.shape),
        "scale_factor": float(scale_factor),
        "image_size": int(size),
        "latent_size": int(sample.shape[-1]),
        "autoencoder": getattr(autoencoder, "name", None) or type(autoencoder).__name__,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    LOGGER.info("cached %d latents -> %s", len(latents), out_dir)
    return manifest


def _list_images_for_latents(folder: str) -> List[str]:
    found = _list_raw_images(folder, recursive=True)
    return [p for p in found if not p.endswith("manifest.json")]


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _task_spec(cfg: Optional[Dict[str, Any]], task: str) -> Dict[str, Any]:
    """Resolve ``tasks.<task>`` into ``{source, target, backbone, shots}``."""
    cfg = cfg or {}
    tasks = cfg.get("tasks") or {}
    spec = dict(tasks.get(task) or {})
    source = spec.get("source")
    target = spec.get("target")

    if not source or not target:
        # fall back to parsing the task name, e.g. "ffhq_sunglasses_ldm"
        name = str(task)
        backbone = "ldm" if name.endswith("_ldm") else spec.get("backbone", "ddpm")
        stem = name[:-4] if name.endswith("_ldm") else name
        for src_name in sorted(SOURCE_DATASETS, key=len, reverse=True):
            if stem.startswith(src_name):
                source = source or src_name
                stem = stem[len(src_name) :].lstrip("_")
                break
        if stem:
            try:
                target = target or resolve_split(stem)
            except Exception:  # pragma: no cover - unknown target
                target = target or stem
        spec.setdefault("backbone", backbone)

    if target and not spec.get("source"):
        spec["source"] = source or TARGET_DATASETS.get(target, {}).get("source")
    spec.setdefault("backbone", "ldm" if str(task).endswith("_ldm") else "ddpm")
    spec.setdefault("shots", DEFAULT_SHOTS)
    return spec


def prepare_task(
    cfg: Optional[Dict[str, Any]] = None,
    task: str = "ffhq_sunglasses",
    root: Optional[str] = None,
    raw_root: Optional[str] = None,
    urls: Optional[Dict[str, str]] = None,
    download: bool = False,
    overwrite: bool = False,
    mode: str = "copy",
    size: int = 256,
    shots: Optional[int] = None,
    fid_sets: Optional[Sequence[str]] = None,
    make_latents: bool = False,
    autoencoder: Any = None,
    latent_scale: float = 0.18215,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Materialise the 10-shot target folder (and optional FID set) for ``task``.

    ``raw_root`` is expected to contain per-dataset subfolders (``raw/babies``,
    ``raw/sunglasses``, ...).  When ``download`` is set, ``urls[<dataset>]`` (or
    the ``data.urls`` config block) is fetched and extracted into ``raw_root``
    first.  Returns a report dict describing what was prepared/available.
    """
    cfg = cfg or {}
    spec = _task_spec(cfg, task)
    dirs = resolve_data_dirs(cfg, root=root, create=not dry_run)
    raw_root = raw_root or cfg.get("data", {}).get("raw_dir") or dirs["raw_dir"]
    urls = dict(urls or (cfg.get("data", {}) or {}).get("urls") or {})
    shots = int(shots or spec.get("shots") or DEFAULT_SHOTS)

    report: Dict[str, Any] = {
        "task": task,
        "source": spec.get("source"),
        "target": spec.get("target"),
        "backbone": spec.get("backbone"),
        "shots": shots,
        "written": {},
        "skipped": {},
        "raw": {},
    }
    target = spec.get("target")
    if not target:
        raise ValueError(f"could not determine target dataset for task {task!r}")

    # 1) fetch/extract archives when asked
    if download:
        url = urls.get(target) or TARGET_URLS.get(target)
        if url:
            extracted = download_and_extract(url, raw_root, name=f"{target}.zip", overwrite=overwrite)
            report["raw"][target] = extracted
        else:
            LOGGER.warning(
                "no download URL for target %r; place the images under %s and re-run",
                target,
                os.path.join(raw_root, target),
            )

    raw_target = report["raw"].get(target) or os.path.join(raw_root, target)
    dest_target = dirs["target_dirs"][target]
    if dry_run:
        report["skipped"][target] = "dry-run"
    elif os.path.isdir(raw_target):
        report["written"][dest_target] = sample_few_shot(
            raw_target,
            dest_target,
            shots=shots,
            size=size,
            seed=int(cfg.get("seed", 0)),
            overwrite=overwrite,
            verbose=verbose,
        )
    else:
        LOGGER.warning(
            "raw target images not found at %s; create that folder with the %d-shot images "
            "or pass --download/--raw-root",
            raw_target,
            shots,
        )
        report["skipped"][target] = "missing raw dir"

    # 2) optional larger FID sets (Sunglasses 2.5k / Babies 2.7k)
    for fid_name in (fid_sets or []):
        fid_name = resolve_split(str(fid_name)) if str(fid_name) in DATASET_ALIASES else str(fid_name)
        if fid_name not in FID_TARGET_SETS:
            LOGGER.warning("skipping unknown FID set %r", fid_name)
            continue
        if download and (urls.get(f"fid_{fid_name}") or FID_URLS.get(fid_name)):
            url = urls.get(f"fid_{fid_name}") or FID_URLS[fid_name]
            report["raw"][f"fid_{fid_name}"] = download_and_extract(
                url, raw_root, name=f"fid_{fid_name}.zip", overwrite=overwrite
            )
        raw_fid = report["raw"].get(f"fid_{fid_name}") or os.path.join(raw_root, f"fid_{fid_name}")
        if not os.path.isdir(raw_fid):
            raw_fid = os.path.join(raw_root, fid_name)
        dest_fid = dirs["fid_target_dirs"][fid_name]
        if dry_run:
            report["skipped"][f"fid_{fid_name}"] = "dry-run"
        elif os.path.isdir(raw_fid):
            report["written"][dest_fid] = sample_fid_subset(
                raw_fid,
                dest_fid,
                num_images=FID_SET_SIZE.get(fid_name, 2500),
                size=size,
                seed=int(cfg.get("seed", 0)),
                overwrite=overwrite,
                verbose=verbose,
            )
        else:
            LOGGER.warning("raw FID images not found at %s", raw_fid)
            report["skipped"][f"fid_{fid_name}"] = "missing raw dir"

    # 3) optional cached latents for the LDM backbone
    if make_latents and not dry_run and os.path.isdir(dest_target):
        latent_dir = os.path.join(dirs["latents_dir"], target)
        report["latents"] = prepare_latents(
            dest_target,
            latent_dir,
            autoencoder=autoencoder,
            scale_factor=latent_scale,
            verbose=verbose,
            overwrite=overwrite,
        )
    return report


def prepare_all(
    cfg: Optional[Dict[str, Any]] = None,
    root: Optional[str] = None,
    raw_root: Optional[str] = None,
    tasks: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run :func:`prepare_task` for every (or the listed) task of the config."""
    cfg = cfg or {}
    if tasks is None:
        tasks = list((cfg.get("tasks") or {}).keys()) or [
            f"{src}_{tgt}" for tgt, meta in TARGET_DATASETS.items() for src in [meta.get("source")]
        ]
    reports: Dict[str, Any] = {}
    for task in tasks:
        LOGGER.info("=== preparing task %s ===", task)
        reports[str(task)] = prepare_task(cfg, str(task), root=root, raw_root=raw_root, **kwargs)
    return reports


def prepare_from_archives(
    archives: Dict[str, str],
    cfg: Optional[Dict[str, Any]] = None,
    root: Optional[str] = None,
    raw_root: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Extract already-downloaded archives (``{dataset_name: path}``) then prepare.

    Convenience for offline setups where the corpora were fetched by hand.
    """
    cfg = cfg or {}
    dirs = resolve_data_dirs(cfg, root=root, create=True)
    raw_root = raw_root or dirs["raw_dir"]
    extracted: Dict[str, str] = {}
    for name, path in archives.items():
        dest = os.path.join(raw_root, name)
        if path and os.path.isfile(path):
            extracted[name] = extract_archive(path, dest, overwrite=kwargs.get("overwrite", False))
        elif path and os.path.isdir(path):
            extracted[name] = path
        else:
            LOGGER.warning("archive for %s not found: %s", name, path)
    reports: Dict[str, Any] = {}
    tasks = kwargs.pop("tasks", None) or list((cfg.get("tasks") or {}).keys())
    for task in tasks:
        spec = _task_spec(cfg, str(task))
        raw = extracted.get(spec.get("target"), os.path.join(raw_root, str(spec.get("target"))))
        reports[str(task)] = prepare_task(
            cfg, str(task), root=root, raw_root=os.path.dirname(raw) or raw_root, **kwargs
        )
    return reports


def verify_task_data(
    cfg: Optional[Dict[str, Any]] = None,
    task: str = "ffhq_sunglasses",
    root: Optional[str] = None,
    shots: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check that a task's target folder exists and holds at least ``shots`` images."""
    cfg = cfg or {}
    spec = _task_spec(cfg, task)
    dirs = resolve_data_dirs(cfg, root=root)
    target = spec.get("target")
    dest = dirs["target_dirs"].get(target, "")
    n = _count_images(dest)
    need = int(shots or spec.get("shots") or DEFAULT_SHOTS)
    ok = n >= need
    if verbose:
        LOGGER.info(
            "%s: target=%s dir=%s images=%d (need %d) -> %s",
            task,
            target,
            dest,
            n,
            need,
            "OK" if ok else "MISSING",
        )
    return {"task": task, "target": target, "dir": dest, "num_images": n, "required": need, "ok": ok}


def summarize_layout(
    cfg: Optional[Dict[str, Any]] = None, root: Optional[str] = None
) -> Dict[str, Dict[str, int]]:
    """Count the images available in every registered dataset directory."""
    dirs = resolve_data_dirs(cfg, root=root)
    summary: Dict[str, Dict[str, int]] = {"source": {}, "target": {}, "fid": {}}
    for name, path in dirs["source_dirs"].items():
        summary["source"][name] = _count_images(path)
    for name, path in dirs["target_dirs"].items():
        summary["target"][name] = _count_images(path)
    for name, path in dirs["fid_target_dirs"].items():
        summary["fid"][name] = _count_images(path)
    return summary


def _count_images(folder: str) -> int:
    if not folder or not os.path.isdir(folder):
        return 0
    return len(_list_raw_images(folder, recursive=True))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dpm_ant.data.prepare_data",
        description="Prepare source / 10-shot target / FID datasets for DPMs-ANT.",
    )
    parser.add_argument("--config", default=None, help="YAML config (defaults merged when omitted)")
    parser.add_argument("--root", default=None, help=f"dataset root (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--raw-root", default=None, help="folder holding untouched downloads")
    parser.add_argument("--task", default=None, help="single task key, e.g. ffhq_sunglasses")
    parser.add_argument("--tasks", nargs="*", default=None, help="explicit list of task keys")
    parser.add_argument("--prepare-all", action="store_true", help="prepare every task in the config")
    parser.add_argument("--archive", default=None, help="archive to extract before preparing")
    parser.add_argument("--archive-name", default=None, help="dataset name the archive belongs to")
    parser.add_argument("--url", default=None, help="download URL for the target dataset")
    parser.add_argument("--download", action="store_true", help="download/extract missing raw data")
    parser.add_argument("--fid-sets", nargs="*", default=None, help="build these larger FID sets")
    parser.add_argument("--shots", type=int, default=None, help=f"few-shot size (default {DEFAULT_SHOTS})")
    parser.add_argument("--size", type=int, default=256, help="image size (default 256)")
    parser.add_argument("--mode", default="copy", choices=["copy", "symlink", "hardlink"])
    parser.add_argument("--latents", default=None, help="cache LDM latents for this image folder")
    parser.add_argument("--ldm-ckpt", default=None, help="LDM autoencoder checkpoint for --latents")
    parser.add_argument("--latent-out", default=None, help="output dir for --latents")
    parser.add_argument("--print-layout", action="store_true", help="print the expected layout and exit")
    parser.add_argument("--summary", action="store_true", help="print dataset counts and exit")
    parser.add_argument("--verify", action="store_true", help="verify a task's target folder")
    parser.add_argument("--write-config", default=None, help="dump resolved data config to this YAML")
    parser.add_argument("--overwrite", action="store_true", help="re-create files that already exist")
    parser.add_argument("--dry-run", action="store_true", help="report planned work only")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    root = args.root or cfg.get("data", {}).get("root") or DEFAULT_DATA_ROOT

    if args.print_layout:
        print_layout(root)
        return 0

    if args.latents:
        manifest = prepare_latents(
            args.latents,
            args.latent_out or os.path.join(default_layout(root)["latents_dir"], os.path.basename(os.path.normpath(args.latents))),
            ckpt=args.ldm_ckpt,
            size=args.size,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.archive:
        name = args.archive_name or os.path.splitext(os.path.basename(args.archive))[0]
        extracted = extract_archive(
            args.archive, os.path.join(args.raw_root or default_layout(root)["raw_dir"], name),
            overwrite=args.overwrite,
        )
        LOGGER.info("extracted to %s", extracted)

    urls = {}
    if args.url and (args.task or args.tasks):
        spec_target = _task_spec(cfg, args.task or args.tasks[0]).get("target")
        if spec_target:
            urls[spec_target] = args.url

    if args.summary:
        print(json.dumps(summarize_layout(cfg, root=root), indent=2))
        if not (args.task or args.tasks or args.prepare_all or args.verify):
            return 0

    if args.verify:
        task = args.task or (args.tasks[0] if args.tasks else "ffhq_sunglasses")
        report = verify_task_data(cfg, task, root=root, shots=args.shots)
        print(json.dumps(report, indent=2))
        if args.write_config:
            write_data_config(args.write_config, root=root, cfg=cfg)
        return 0 if report["ok"] else 1

    common = dict(
        root=root,
        raw_root=args.raw_root,
        urls=urls,
        download=args.download,
        overwrite=args.overwrite,
        mode=args.mode,
        size=args.size,
        shots=args.shots,
        fid_sets=args.fid_sets,
        dry_run=args.dry_run,
    )
    if args.prepare_all:
        reports = prepare_all(cfg, tasks=args.tasks, **common)
    elif args.task or args.tasks:
        tasks = args.tasks or [args.task]
        reports = {t: prepare_task(cfg, t, **common) for t in tasks}
    else:
        ensure_layout(root)
        print_layout(root)
        LOGGER.info("nothing to do: pass --task <name> or --prepare-all (see --help)")
        return 0

    print(json.dumps(reports, indent=2, default=str))
    if args.write_config:
        write_data_config(args.write_config, root=root, cfg=cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
