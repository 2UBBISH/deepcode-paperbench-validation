"""Deterministic seeding helpers shared by every entry point."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:  # torch is required for the project; keep the import defensive for tests.
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_seed(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and torch.  Returns the seed for convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


def make_generator(seed: Optional[int] = None):
    """Return a numpy ``Generator`` unless a legacy global seed is requested."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def seed_env(env, seed: Optional[int]) -> None:
    """Seed a gym/gymnasium style environment if it supports seeding."""
    if seed is None:
        return
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
            return
        except Exception:  # pragma: no cover - gymnasium fallback
            pass
    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
