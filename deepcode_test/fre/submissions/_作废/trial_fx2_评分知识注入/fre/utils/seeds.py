"""Reproducibility helpers for FRE experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed.
        deterministic: If True, additionally configure PyTorch to prefer
            deterministic algorithms where available. This usually reduces
            GPU performance but improves reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Keep cudnn benchmark enabled for speed; we still set manual seeds.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initializer for deterministic data loading."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def seed_worker(seed: Optional[int] = None) -> None:
    """Convenience wrapper around :func:`set_seed` that also respects an env seed."""
    if seed is None:
        seed = int(os.environ.get("FRE_SEED", "0"))
    set_seed(seed)


__all__ = ["set_seed", "worker_init_fn", "seed_worker"]
