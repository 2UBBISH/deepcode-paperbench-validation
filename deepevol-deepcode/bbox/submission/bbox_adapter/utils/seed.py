"""Deterministic seeding helpers for BBox-Adapter.

Source: not specified in the paper -- chosen convenience defaults.

The paper never specifies its random-seed handling, so this module provides a
single reproducible entry point used by every experiment script.  It seeds the
standard library ``random`` module, NumPy (when available) and PyTorch (when
available, CPU + all CUDA devices), and it can derive per-component seeds so
that the black-box sampling, buffer SEL stochasticity and adapter
initialisation are deterministic yet independent.

Public API
----------
- :func:`set_seed` -- seed everything, returns the seed used.
- :func:`spawn_seeds` -- derive ``n`` independent sub-seeds from a base seed.
- :func:`rng_from_seed` -- build a ``random.Random`` instance.
- :func:`numpy_rng` / :func:`torch_generator` -- component RNGs.
- :func:`seed_worker` -- ``DataLoader`` worker initializer.
- :class:`SeedContext` -- context manager that restores the previous states.
- :func:`describe_seed_state` / :func:`capture_state` / :func:`restore_state`

Nothing here touches the black-box LLM; seeding only affects the small local
adapter, sampling of proposals and buffer bookkeeping.
"""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "DEFAULT_SEED",
    "set_seed",
    "seed_everything",
    "spawn_seeds",
    "rng_from_seed",
    "numpy_rng",
    "torch_generator",
    "seed_worker",
    "SeedContext",
    "capture_state",
    "restore_state",
    "describe_seed_state",
    "derive_component_seeds",
    "COMPONENT_NAMES",
]

#: Default seed used across the reproduction (paper silent; fixed for
#: reproducibility of Tables 2-7 and Figures 3-4).
DEFAULT_SEED = 0

#: Names of the pipelines that receive independent, reproducible sub-seeds.
COMPONENT_NAMES: Sequence[str] = (
    "python",
    "numpy",
    "torch",
    "adapter_init",
    "blackbox_sampling",
    "buffer",
    "dataloader",
    "train_shuffle",
    "eval",
)


# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
def _numpy():
    try:  # pragma: no cover - depends on environment
        import numpy as _np  # noqa: WPS433

        return _np
    except Exception:  # pragma: no cover
        return None


def _torch():
    try:  # pragma: no cover - depends on environment
        import torch as _torch  # noqa: WPS433

        return _torch
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Core seeding
# ---------------------------------------------------------------------------
def set_seed(seed: Optional[int] = DEFAULT_SEED, *, deterministic: bool = False,
             seed_cuda: bool = True, seed_env: bool = True) -> int:
    """Seed Python, NumPy and PyTorch.

    Parameters
    ----------
    seed:
        The seed to use.  ``None`` draws a fresh, time-based seed.
    deterministic:
        When ``True`` also asks PyTorch for deterministic cuDNN kernels
        (slower; only used for debugging numerics).
    seed_cuda:
        Seed every visible CUDA device.
    seed_env:
        Export ``PYTHONHASHSEED`` for child processes.

    Returns
    -------
    int
        The effective seed (useful when ``seed=None`` was passed).
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")

    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    if seed_env:
        os.environ["PYTHONHASHSEED"] = str(int(seed))

    random.seed(seed)

    np = _numpy()
    if np is not None:
        np.random.seed(seed % (2 ** 32))

    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        if seed_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:  # pragma: no cover
                pass
    return int(seed)


# Friendly alias matching the plan's naming.
seed_everything = set_seed


def spawn_seeds(base_seed: int, n: int) -> List[int]:
    """Derive ``n`` independent, reproducible sub-seeds from ``base_seed``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(base_seed)
    return [rng.randrange(2 ** 31 - 1) for _ in range(n)]


def derive_component_seeds(base_seed: int = DEFAULT_SEED,
                           names: Sequence[str] = COMPONENT_NAMES) -> Dict[str, int]:
    """Map each named pipeline component to its own deterministic seed."""
    seeds = spawn_seeds(base_seed, len(names))
    return {name: s for name, s in zip(names, seeds)}


def rng_from_seed(seed: Optional[int] = None) -> random.Random:
    """Return a private :class:`random.Random` instance (does not touch globals)."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return random.Random(int(seed))


def numpy_rng(seed: Optional[int] = None):
    """Return ``numpy.random.RandomState`` or ``None`` when NumPy is absent."""
    np = _numpy()
    if np is None:
        return None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return np.random.RandomState(int(seed) % (2 ** 32))


def torch_generator(seed: Optional[int] = None, device: str = "cpu"):
    """Return a ``torch.Generator`` or ``None`` when PyTorch is absent."""
    torch = _torch()
    if torch is None:
        return None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:  # pragma: no cover - torch dataloader hook
    """``DataLoader(worker_init_fn=seed_worker)`` initializer.

    NumPy and Python seeds are offset by the worker id so that workers do not
    emit identical random streams.
    """
    worker_seed = (torch_initial_seed() + int(worker_id)) % (2 ** 32)
    random.seed(worker_seed)
    np = _numpy()
    if np is not None:
        np.random.seed(worker_seed)


_TORCH_INITIAL_SEED = [DEFAULT_SEED]


def torch_initial_seed() -> int:
    """Seed recorded by the most recent :func:`set_seed` call."""
    return int(_TORCH_INITIAL_SEED[0])


def _register_initial_seed(seed: int) -> None:
    _TORCH_INITIAL_SEED[0] = int(seed)


# ---------------------------------------------------------------------------
# State capture / restore
# ---------------------------------------------------------------------------
def capture_state() -> Dict[str, Any]:
    """Snapshot global RNG states (Python / NumPy / CUDA)."""
    state: Dict[str, Any] = {"python": random.getstate()}
    np = _numpy()
    if np is not None:
        state["numpy"] = np.random.get_state()
    torch = _torch()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            try:
                state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception:  # pragma: no cover
                pass
    return state


def restore_state(state: Dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`capture_state`."""
    if "python" in state:
        random.setstate(state["python"])
    np = _numpy()
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    torch = _torch()
    if torch is not None:
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(state["cuda"])
            except Exception:  # pragma: no cover
                pass


@dataclass
class SeedContext:
    """Context manager that seeds on enter and restores state on exit."""

    seed: Optional[int] = DEFAULT_SEED
    deterministic: bool = False

    def __enter__(self) -> int:
        self._previous = capture_state()
        self._seed = set_seed(self.seed, deterministic=self.deterministic)
        return self._seed

    def __exit__(self, exc_type, exc, tb) -> bool:
        restore_state(getattr(self, "_previous", {}))
        return False


def describe_seed_state() -> Dict[str, Any]:
    """Human-readable summary used by scripts when logging a run header."""
    info: Dict[str, Any] = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "initial_seed": torch_initial_seed(),
    }
    np = _numpy()
    info["numpy_available"] = np is not None
    torch = _torch()
    info["torch_available"] = torch is not None
    if torch is not None:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_devices"] = torch.cuda.device_count()
    return info


@contextlib.contextmanager
def temporary_seed(seed: Optional[int] = None):
    """Convenience generator version of :class:`SeedContext`."""
    with SeedContext(seed):
        yield
