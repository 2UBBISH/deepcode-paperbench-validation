"""Small shared helpers (seeding, device selection, checkpoint download, IO)."""

from __future__ import annotations

import os
import random
import sys
import time
import urllib.request
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (used for every experiment script)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: Optional[str] = None) -> torch.device:
    """Return the torch device to use.

    ``--device`` accepts ``auto`` (default), ``cpu``, ``cuda`` or ``mps``.
    ``auto`` picks CUDA when it is available and otherwise CPU; Apple's MPS
    backend must be requested explicitly because some of the diffusion code
    paths are slower or unsupported there.  The paper's own experiments were
    run on GPUs; the reproduction environment has none.
    """
    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_rate(model: nn.Module) -> float:
    """Proportion of fine-tuned parameters w.r.t. the pre-trained model.

    This is the "Parameter Rate" column of Table 1 (1.3% for DDPM-ANT and
    1.6% for LDM-ANT).
    """
    total = count_parameters(model)
    if total == 0:
        return 0.0
    adaptor_params = sum(
        p.numel() for name, p in model.named_parameters() if ".adaptor." in name and p.requires_grad
    )
    return adaptor_params / float(total)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def download_file(url: str, destination: str, chunk_size: int = 1 << 20) -> str:
    """Download ``url`` to ``destination`` unless it already exists."""
    if os.path.exists(destination):
        return destination
    ensure_dir(os.path.dirname(os.path.abspath(destination)))
    tmp = destination + ".part"
    print(f"[download] {url} -> {destination}", flush=True)
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp, destination)
    return destination


class Logger:
    """Minimal stdout/file logger (no external dependency)."""

    def __init__(self, path: Optional[str] = None, also_stdout: bool = True):
        self.path = path
        self.also_stdout = also_stdout
        if path is not None:
            ensure_dir(os.path.dirname(os.path.abspath(path)))
            self._handle = open(path, "a")
        else:
            self._handle = None
        self._start = time.time()

    def log(self, message: str) -> None:
        line = f"[{time.time() - self._start:8.1f}s] {message}"
        if self.also_stdout:
            print(line, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()

    def __call__(self, message: str) -> None:
        self.log(message)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def to_device(batch, device):
    """Recursively move tensors in a (nested) batch to ``device``."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


def progbar(iterable: Iterable, total: Optional[int] = None, desc: str = ""):
    """Progress bar that stays silent when stdout is not a terminal."""
    import os

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is in requirements
        return iterable
    forced = os.environ.get("DPMS_ANT_PROGRESS", "").lower() in {"1", "true", "yes"}
    disable = not (forced or sys.stdout.isatty())
    return tqdm(iterable, total=total, desc=desc, file=sys.stdout, leave=False, disable=disable)


def quantize_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convert a float tensor in [-1, 1] to a uint8 numpy array [N,H,W,C]."""
    images = images.detach().cpu().clamp(-1, 1)
    images = ((images + 1) / 2 * 255).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).numpy()


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4, padding: int = 2) -> str:
    """Save a grid of images (tensor in [-1, 1], shape [N,3,H,W])."""
    from PIL import Image

    arr = quantize_to_uint8(images)
    n, h, w, c = arr.shape
    nrow = max(1, min(nrow, n))
    ncol = int(np.ceil(n / nrow))
    canvas = np.full(
        (ncol * (h + padding) + padding, nrow * (w + padding) + padding, c), 255, dtype=np.uint8
    )
    for index in range(n):
        row, col = divmod(index, nrow)
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        canvas[y : y + h, x : x + w] = arr[index]
    image = Image.fromarray(canvas)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    image.save(path)
    return path


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, tensor.ndim)))


def freeze_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def as_tuple(values: Sequence[int]) -> tuple:
    return tuple(int(v) for v in values)


class BatchIterator:
    """Uniform interface over the several ways a caller may supply batches.

    Accepts * an iterator/generator of batches, * a callable returning either a
    single batch or a fresh iterator of batches, * a plain sequence of batches.
    Used by the classifier and the ANT trainers so that callers can always just
    write ``next(batch_iterator)``.
    """

    def __init__(self, source):
        self.source = source
        if not (hasattr(source, "__next__") or callable(source)):
            self.source = iter(source)

    def __iter__(self) -> "BatchIterator":
        return self

    def __next__(self):
        if hasattr(self.source, "__next__"):
            result = next(self.source)
        else:
            result = self.source() if callable(self.source) else self.source
            if hasattr(result, "__next__"):
                # the callable returned a fresh iterator: keep using it
                self.source = result
                result = next(result)
        return self._normalise(result)

    @staticmethod
    def _normalise(result):
        if torch.is_tensor(result):
            return result
        if isinstance(result, (tuple, list)) and result and torch.is_tensor(result[0]):
            if len(result) == 1:  # e.g. a TensorDataset wrapped in a DataLoader
                return result[0]
            return result
        raise TypeError(f"could not obtain a batch from {type(result)!r}")
