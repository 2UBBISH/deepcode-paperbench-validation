"""Deterministic seeding utilities for the FOA reproduction.

FOA is fully backpropagation-free, but reproducibility still requires a single
global seed controlling (a) the CMA-ES sampling stream, (b) the prompt
initialisation, (c) the random selection of the Q source images used to build the
source statistics bank and (d) the random ordering of the non-i.i.d. streams.

All helpers are pure glue: they contain no paper-specified math.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any, Dict, Optional

import numpy as np

try:  # torch is required by the project; guard only for doc builds
    import torch
except Exception:  # pragma: no cover - torch is always available in practice
    torch = None  # type: ignore


DEFAULT_SEED = 0

__all__ = [
    "DEFAULT_SEED",
    "seed_everything",
    "set_seed",
    "get_seed",
    "seed_worker",
    "worker_init_fn",
    "temporary_seed",
    "make_generator",
    "cma_seed",
]


def _numpy_seed(seed: int) -> None:
    np.random.seed(seed % (2 ** 32))


def _python_seed(seed: int) -> None:
    random.seed(seed)


def _torch_seed(seed: int, deterministic: bool = False) -> None:
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Best-effort determinism for cuDNN; FOA does not train so this is cheap.
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover
            pass


def seed_everything(
    seed: int = DEFAULT_SEED,
    *,
    deterministic: bool = False,
    set_cuda: bool = True,
) -> int:
    """Seed python, numpy and torch RNGs from a single integer.

    Args:
        seed: the global seed (paper default: 0).
        deterministic: also switch cuDNN into deterministic mode.
        set_cuda: whether to touch CUDA RNGs when a GPU is present.

    Returns:
        The seed that was applied (useful for logging).
    """
    seed = int(seed)
    _python_seed(seed)
    _numpy_seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if set_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            _torch_seed(seed, deterministic=True)
    # Keep PYTHONHASHSEED informative for subprocesses (no effect in-process).
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed


# Common aliases used across the runner scripts.
set_seed = seed_everything


def get_seed() -> Optional[int]:
    """Return the current global seed if it can be inferred, else ``None``.

    The value is recovered from ``numpy``'s global RNG state, which is the RNG
    the CMA-ES ``cmaes`` backend uses when no explicit seed is supplied.
    """
    try:
        state: Dict[str, Any] = np.random.get_state()
        return int(state["state"][0]) % (2 ** 32)
    except Exception:  # pragma: no cover
        return None


def cma_seed(seed: Optional[int] = DEFAULT_SEED, offset: int = 0) -> Optional[int]:
    """Derive a dedicated CMA-ES seed so prompt sampling cannot collide.

    The paper fixes a single global random seed (ambiguity (g) in the plan); we
    keep the CMA stream on that seed but allow a deterministic ``offset`` so
    independent runs (e.g. per corruption) remain reproducible without reuse.
    """
    if seed is None:
        return None
    return int(seed) + int(offset)


def seed_worker(worker_id: int) -> None:  # pragma: no cover - trivial
    """``DataLoader`` worker init that keeps dataloading deterministic."""
    worker_seed = (torch.initial_seed() if torch is not None else 0) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Standard name expected by ``torch.utils.data.DataLoader(worker_init_fn=...)``.
worker_init_fn = seed_worker


def make_generator(seed: int = DEFAULT_SEED) -> "Any":
    """Create a deterministic ``torch.Generator`` (CPU) for data/CMA streams."""
    if torch is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


@contextlib.contextmanager
def temporary_seed(seed: int = DEFAULT_SEED):
    """Context manager restoring the previous RNG state afterwards.

    Used by the ablation/sensitivity runners so that evaluating one variant does
    not perturb the sampling stream of the next.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_state = None
    if torch is not None and torch.cuda.is_available():
        try:
            cuda_state = torch.cuda.get_rng_state_all()
        except Exception:  # pragma: no cover
            cuda_state = None
    try:
        seed_everything(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                try:
                    torch.cuda.set_rng_state_all(cuda_state)
                except Exception:  # pragma: no cover
                    pass
