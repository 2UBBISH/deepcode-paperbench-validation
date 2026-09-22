"""Small helpers used across the code base."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (used before every training / eval run)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "robust_clip", log_file: Optional[str] = None, level: int = logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


class AverageMeter:
    """Tracks the (weighted) mean of a scalar."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.name}: {self.avg:.4f}" if self.name else f"{self.avg:.4f}"


class Timer:
    """Wall-clock timer, e.g. to report the runtime reduction of App. B.7."""

    def __init__(self) -> None:
        self.start = time.time()

    def reset(self) -> None:
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def __str__(self) -> str:  # pragma: no cover - trivial
        secs = int(self.elapsed)
        return f"{secs // 3600:d}h {secs % 3600 // 60:d}m {secs % 60:d}s"


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def parse_eps(eps) -> float:
    """Parse a perturbation strength.

    The paper always specifies the l_inf radius as a fraction of the pixel range,
    e.g. ``'2/255'``, ``'4/255'``, ``'16/255'`` or ``'0.0078'``.  This helper
    accepts all of these spellings and returns a float in ``[0, 1]``.
    """
    if isinstance(eps, (int, float)):
        return float(eps)
    eps = str(eps).strip()
    if eps == "0":
        return 0.0
    if "/" in eps:
        num, denom = eps.split("/")
        return float(num) / float(denom)
    return float(eps)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_readable(num: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"  # pragma: no cover - unreachable for our models
