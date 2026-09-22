"""Reproducible seeding utilities.

The paper reports results over 3 random seeds (mean +/- std) for the fidelity
experiment (Section 4.2).  We seed numpy, torch, python's ``random`` and gym's
action/observation spaces so that a run is fully reproducible.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seeds(seed: Optional[int]) -> None:
    """Seed every RNG the code base touches.

    Parameters
    ----------
    seed:
        The seed.  ``None`` disables seeding (fully random run).
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))

    try:  # torch is optional at import time
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        # keep deterministic-ish but do not crash when a kernel has no
        # deterministic implementation
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover - cpu only / non-cuda builds
            pass
    except ImportError:  # pragma: no cover
        pass


def seed_env(env, seed: Optional[int]) -> None:
    """Seed a gym(-nasium) environment and its action/observation spaces."""
    if seed is None or env is None:
        return
    try:
        env.reset(seed=int(seed))
    except TypeError:
        # older gym API
        env.seed(int(seed))
        env.reset()
    except Exception:  # pragma: no cover - defensive
        pass

    if hasattr(env, "action_space") and env.action_space is not None:
        try:
            env.action_space.seed(int(seed))
        except Exception:
            pass
    if hasattr(env, "observation_space") and env.observation_space is not None:
        try:
            env.observation_space.seed(int(seed))
        except Exception:
            pass


class RNG:
    """A thin wrapper around :class:`numpy.random.Generator` with helpers."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._gen = np.random.default_rng(seed)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """RAND(low, high) oracle used by Algorithm 2."""
        return float(self._gen.uniform(low, high))

    def bernoulli(self, p: float) -> bool:
        """Bernoulli(p) => True with probability ``p``."""
        return bool(self._gen.random() < p)

    def choice(self, a, p=None):
        return self._gen.choice(a, p=p)

    def integers(self, low, high=None, size=None):
        return self._gen.integers(low, high, size=size)

    def __getattr__(self, item):  # delegate the rest to the generator
        return getattr(self._gen, item)
