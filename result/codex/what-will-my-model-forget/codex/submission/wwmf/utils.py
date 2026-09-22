"""Small shared helpers: seeding, device selection, JSON IO and caches."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


# --------------------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (torch is imported lazily)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        pass


def resolve_device(device: str = "auto") -> str:
    """Return a torch device string.

    ``"auto"`` selects CUDA when available and otherwise falls back to CPU.
    Apple's MPS backend is *not* selected automatically: with the
    torch/transformers versions used here, seq2seq generation on MPS returns
    degenerate text, so it has to be requested explicitly (``device="mps"``).
    """
    if device != "auto":
        return device
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


def write_jsonl(path: str | os.PathLike, rows: Iterable[Any]) -> int:
    ensure_dir(Path(path).parent)
    n = 0
    with open(path, "w", encoding="utf8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike) -> List[Any]:
    rows = []
    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cache_key(*parts: Any) -> str:
    """Stable short hash used for naming cache files."""
    raw = "||".join(
        p if isinstance(p, str) else json.dumps(p, sort_keys=True, ensure_ascii=False, default=str)
        for p in parts
    )
    return hashlib.md5(raw.encode("utf8")).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# small functional helpers
# --------------------------------------------------------------------------------------
def chunks(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def flatten(nested: Iterable[Iterable[T]]) -> List[T]:
    return [x for sub in nested for x in sub]


class AverageMeter:
    """Running mean, used for the loss curves printed during training."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")
