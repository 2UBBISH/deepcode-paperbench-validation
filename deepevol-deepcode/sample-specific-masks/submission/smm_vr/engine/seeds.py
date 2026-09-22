"""Reproducibility / seeding utilities for SMM experiments.

The SMM paper (ICML 2024) states in Section 5 ("Baselines"):

    "Experiments are run with three seeds on a single A100 GPU and the
     averaged test accuracy is reported."

This module centralises everything related to that three-seed protocol:

* :data:`SEEDS` -- the seed set used for every table in the paper.
* :func:`set_seed` / :func:`seed_everything` -- seed ``random``, ``numpy``,
  ``torch`` (CPU + all CUDA devices) and, optionally, switch cuDNN into a
  deterministic mode.
* :func:`make_worker_init_fn` / :func:`seed_worker` -- deterministic
  ``DataLoader`` worker seeding, because the paper's training transform applies
  ``RandomCrop(imgsize)`` and ``RandomHorizontalFlip()`` on every sample.
* :func:`make_generator` -- a ``torch.Generator`` for shuffling that is
  independent of the global RNG, so data ordering is reproducible even when
  the model/mask initialisation seed changes.
* :func:`temporary_seed` -- context manager to run a short block under a
  different seed and restore the previous RNG state afterwards.

Nothing here is learnable; the module only makes the paper's averaged
"mean +- std" numbers reproducible.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "SEEDS",
    "DEFAULT_SEEDS",
    "N_SEEDS",
    "get_seed",
    "set_seed",
    "seed_everything",
    "seed_all",
    "make_generator",
    "seed_worker",
    "make_worker_init_fn",
    "dataloader_seed_kwargs",
    "worker_kwargs",
    "temporary_seed",
    "capture_rng_state",
    "restore_rng_state",
    "enumerate_seeds",
    "resolve_seeds",
    "DEFAULT_DETERMINISTIC",
    "DEFAULT_CUDNN_BENCHMARK",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Seeds used for the main tables (Table 1, 2, 3, 10, ...).  The reproduction
#: plan lists ``seeds {0, 1, 2}``.
SEEDS: Tuple[int, ...] = (0, 1, 2)

#: Alternate name for :data:`SEEDS`.
DEFAULT_SEEDS: Tuple[int, ...] = SEEDS

#: Number of independent runs aggregated as "mean +- std" in the paper.
N_SEEDS: int = len(SEEDS)

#: Deterministic cuDNN is slower but reproducible.  SMM trains only a tiny
#: mask generator on top of a *frozen* backbone, so the cost is acceptable and
#: the paper reports averaged accuracies with std.
DEFAULT_DETERMINISTIC: bool = True

#: cuDNN benchmark mode; ignored when ``deterministic`` is requested.
DEFAULT_CUDNN_BENCHMARK: bool = False


# ---------------------------------------------------------------------------
# Optional numpy / torch imports
# ---------------------------------------------------------------------------

def _import_numpy():
    """Return the ``numpy`` module or ``None`` when unavailable."""
    try:  # pragma: no cover - environment dependent
        import numpy as np  # type: ignore

        return np
    except Exception:  # pragma: no cover
        return None


def _import_torch():
    """Return the ``torch`` module or ``None`` when unavailable."""
    try:  # pragma: no cover - environment dependent
        import torch  # type: ignore

        return torch
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Seed plumbing
# ---------------------------------------------------------------------------

def get_seed(seed_env: str = "SMM_SEED") -> int:
    """Return the process seed, honouring the ``SMM_SEED`` env var.

    The environment variable makes it easy to launch the same command for the
    paper's three seeds (``SMM_SEED=0``, ``1``, ``2``) without editing configs
    or passing CLI flags.

    Args:
        seed_env: Name of the environment variable to consult.

    Returns:
        The integer seed (``SEEDS[0]`` when the variable is unset or invalid).
    """
    raw = os.environ.get(seed_env)
    if raw is None:
        return SEEDS[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return SEEDS[0]


def set_seed(
    seed: int,
    *,
    deterministic: bool = DEFAULT_DETERMINISTIC,
    cudnn_benchmark: bool = DEFAULT_CUDNN_BENCHMARK,
    seed_cuda: bool = True,
) -> int:
    """Seed every RNG used by the SMM pipeline and return ``seed``.

    Seeds the standard library ``random`` module, ``numpy`` (when installed),
    ``torch`` and all available CUDA devices.  When ``deterministic`` is set,
    cuDNN is configured for reproducible convolutions (no autotuner, no
    non-deterministic algorithms).  ``CUBLAS_WORKSPACE_CONFIG`` is exported
    because ``torch.use_deterministic_algorithms`` requires it on CUDA >= 10.2.

    Args:
        seed: The run seed (one of :data:`SEEDS` for the paper's results).
        deterministic: Request deterministic cuDNN behaviour.
        cudnn_benchmark: Enable cuDNN autotuning (only when not deterministic).
        seed_cuda: Also seed ``torch.cuda`` when a GPU is available.

    Returns:
        The seed, so callers can log ``seed = set_seed(seed)``.
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)

    np = _import_numpy()
    if np is not None:
        np.random.seed(seed)

    torch = _import_torch()
    if torch is None:  # pragma: no cover - torch is a hard dependency in practice
        return seed

    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required by torch.use_deterministic_algorithms on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, AttributeError):  # pragma: no cover - older torch
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    return seed


#: Convenient aliases matching common naming conventions across codebases.
seed_everything = set_seed
seed_all = set_seed


# ---------------------------------------------------------------------------
# DataLoader reproducibility
# ---------------------------------------------------------------------------

def make_generator(seed: int, device: str = "cpu"):
    """Build a ``torch.Generator`` seeded with ``seed``.

    Useful for ``DataLoader(shuffle=True, generator=generator)`` and for
    ``sampler=RandomSampler(..., generator=generator)`` so that the epoch
    shuffling order is reproducible and independent of the global RNG used for
    model / mask initialisation.

    Args:
        seed: Seed for the generator.
        device: Device for the generator (``"cpu"`` by default; shuffle
            generators must be CPU generators).

    Returns:
        A ``torch.Generator`` instance, or ``None`` if torch is unavailable.
    """
    torch = _import_torch()
    if torch is None:  # pragma: no cover
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` that re-seeds each worker.

    Every worker derives its own seed from ``torch.initial_seed()`` (which the
    ``DataLoader`` already offsets per worker) plus ``worker_id``, so the
    random augmentations inside the SMM training transform
    (``RandomCrop``, ``RandomHorizontalFlip``) differ per worker yet remain
    reproducible for a given base seed.

    Reference: PyTorch "Reproducibility" recipe.
    """
    torch = _import_torch()
    if torch is None:  # pragma: no cover
        return
    worker_seed = (int(torch.initial_seed()) + int(worker_id)) % (2 ** 32)
    np = _import_numpy()
    if np is not None:
        np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_worker_init_fn(seed: Optional[int] = None, *, spread: int = 1000):
    """Return a ``worker_init_fn`` closing over ``seed``.

    The base seed is offset by ``spread`` so that the main-process and worker
    seed ranges are disjoint.

    Args:
        seed: Base seed; ``None`` uses :func:`get_seed`.
        spread: Offset applied to the base seed.

    Returns:
        Callable suitable for ``DataLoader(worker_init_fn=...)``.
    """
    base = get_seed() if seed is None else int(seed)
    offset = int(spread)

    def _worker_init_fn(worker_id: int) -> None:  # pragma: no cover - trivial
        # Re-derive from our explicit base so the result does not depend on
        # who created the DataLoader.
        torch = _import_torch()
        derived = (base + offset + int(worker_id)) % (2 ** 32)
        if torch is not None:
            torch.manual_seed(derived)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(derived)
        np = _import_numpy()
        if np is not None:
            np.random.seed(derived)
        random.seed(derived)

    _worker_init_fn.__name__ = "smm_worker_init_fn"
    return _worker_init_fn


#: Alias for callers wanting the factory under a shorter name.
worker_kwargs = make_worker_init_fn


def dataloader_seed_kwargs(
    seed: Optional[int] = None,
    *,
    generator: bool = True,
    worker_init: bool = True,
) -> Dict[str, object]:
    """Return ``**kwargs`` making a ``DataLoader`` reproducible for ``seed``.

    Intended usage::

        kwargs = dataloader_seed_kwargs(seed)
        loader = DataLoader(dataset, shuffle=True, **kwargs)

    Args:
        seed: Base seed; ``None`` uses :func:`get_seed`.
        generator: Include a seeded ``torch.Generator`` for shuffling.
        worker_init: Include a seeded ``worker_init_fn``.

    Returns:
        Dict with the relevant DataLoader kwargs (possibly empty).
    """
    base = get_seed() if seed is None else int(seed)
    kwargs: Dict[str, object] = {}
    if generator:
        gen = make_generator(base)
        if gen is not None:
            kwargs["generator"] = gen
    if worker_init:
        kwargs["worker_init_fn"] = make_worker_init_fn(base)
    return kwargs


# ---------------------------------------------------------------------------
# RNG state capture / restore and scoped seeding
# ---------------------------------------------------------------------------

def capture_rng_state() -> Dict[str, object]:
    """Capture the current RNG states (``random``, numpy, torch, CUDA).

    Returns:
        Dict with ``python``, ``numpy`` (optional), ``torch`` and ``cuda``
        (optional) entries, suitable for :func:`restore_rng_state`.
    """
    state: Dict[str, object] = {"python": random.getstate()}

    np = _import_numpy()
    if np is not None:
        state["numpy"] = np.random.get_state()

    torch = _import_torch()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    """Restore RNG states previously captured by :func:`capture_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])  # type: ignore[arg-type]

    np = _import_numpy()
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]

    torch = _import_torch()
    if torch is not None:
        if "torch" in state:
            torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


@contextlib.contextmanager
def temporary_seed(
    seed: int,
    *,
    deterministic: Optional[bool] = None,
) -> Iterator[int]:
    """Run a block under ``seed`` and restore the previous RNG state after.

    Typical use: re-materialise a data subset or re-run a debugging epoch
    without perturbing the main run's RNG stream.

    Args:
        seed: Temporary seed.
        deterministic: When given, override the cuDNN determinism flag inside
            the block (restored on exit).

    Yields:
        The seed.
    """
    torch = _import_torch()
    state = capture_rng_state()
    prev_deterministic = None
    if torch is not None and deterministic is not None:
        prev_deterministic = bool(torch.backends.cudnn.deterministic)

    if deterministic is None:
        local = set_seed(seed, deterministic=DEFAULT_DETERMINISTIC)
    else:
        local = set_seed(seed, deterministic=bool(deterministic))
    try:
        yield local
    finally:
        if torch is not None and prev_deterministic is not None:
            torch.backends.cudnn.deterministic = prev_deterministic
        restore_rng_state(state)


# ---------------------------------------------------------------------------
# Seed-set helpers
# ---------------------------------------------------------------------------

def resolve_seeds(
    seeds: Optional[Sequence[int]] = None,
    n_seeds: Optional[int] = None,
) -> List[int]:
    """Normalise a user-supplied seed specification into a list of seeds.

    Args:
        seeds: Explicit seeds; ``None`` -> :data:`SEEDS`.
        n_seeds: When given and ``seeds`` is ``None``, use ``range(n_seeds)``.
            When ``seeds`` is given it is truncated to the first ``n_seeds``
            entries (``n_seeds <= 0`` means "keep everything").

    Returns:
        List of integer seeds.
    """
    if seeds is None:
        base: List[int] = list(range(int(n_seeds))) if n_seeds is not None else list(SEEDS)
    else:
        if isinstance(seeds, int):
            base = [int(seeds)]
        else:
            base = [int(s) for s in seeds]
    if n_seeds is not None and n_seeds > 0:
        base = base[: int(n_seeds)]
    return base


def enumerate_seeds(
    seeds: Optional[Sequence[int]] = None,
    n_seeds: Optional[int] = None,
) -> Iterable[Tuple[int, int]]:
    """Yield ``(index, seed)`` pairs for the paper's multi-seed runs.

    Example::

        for i, seed in enumerate_seeds():          # (0, 0), (1, 1), (2, 2)
            results.append(run_once(seed=seed))
    """
    for idx, seed in enumerate(resolve_seeds(seeds, n_seeds)):
        yield idx, seed
