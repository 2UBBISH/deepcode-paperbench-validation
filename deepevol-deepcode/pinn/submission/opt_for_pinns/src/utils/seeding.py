"""Deterministic seed control for the PINN loss-landscape reproduction.

This module is *glue*: it contains no algorithm from the paper.  Its only job
is to make every experiment in the reproduction plan reproducible, so that the
"winner-selection" process described in the plan (smallest L2RE over
``(adam_lr, seed, width)`` per PDE) is itself reproducible.

Design constraints (driven by the rest of the code base):

* ``experiments/run_spectral_density.py`` and
  ``experiments/run_optimizer_comparison.py`` import ``set_seed`` from this
  module inside an ``try/except ImportError`` block, therefore ``set_seed`` is
  the canonical public name and must accept at least a positional / keyword
  ``seed`` argument.
* The pipeline uses CPU ``float64`` by default (second-order optimizers such as
  NNCG / Nyström-PCG need double precision), but the paper's environment is a
  single NVIDIA GPU, so CUDA seeding is implemented too.
* ``numpy`` is an optional dependency in some environments (the runners guard
  their own numpy imports), so every third-party import here is guarded.

Public API
----------
``set_seed`` / ``seed_everything``
    Seed Python ``random``, NumPy and PyTorch (CPU + all CUDA devices).
``get_seed``
    Return the seed that was last set (or ``None``).
``make_generator``
    Build a ``torch.Generator`` on a given device for reproducible sampling.
``spawn_seeds``
    Deterministically derive ``n`` independent child seeds from a base seed
    (used for the ``{345, 456, 567}`` seed sweeps without hard-coding them).
``SeedContext``
    Context manager that seeds on entry and restores the previous seed on exit.
``seed_from_config``
    Best-effort extraction of a seed out of a nested experiment config dict.

All functions are import-safe: importing this module never touches RNG state.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Optional third-party imports (guarded: the module must be importable even in
# a minimal environment, e.g. while only running the analytic PDE tests).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - trivial import guard
    import torch as _torch
except Exception:  # pragma: no cover
    _torch = None  # type: ignore[assignment]


__all__ = [
    "DEFAULT_SEED",
    "set_seed",
    "seed_everything",
    "get_seed",
    "make_generator",
    "spawn_seeds",
    "SeedContext",
    "seed_from_config",
    "describe_rng_state",
]

#: Default seed used when the caller does not provide one.  The paper reports
#: seeds ``{345, 456, 567, 678, 789}`` for the 5-seed sweeps; 345 is the first
#: of those, so it makes a sensible fallback.
DEFAULT_SEED: int = 345

#: Maximum value accepted by numpy's legacy seeding API (2**32 - 1).
_NUMPY_SEED_MAX = 2 ** 32 - 1

#: Track the most recently applied seed so runners can report it.
_LAST_SEED: Optional[int] = None


def _normalize_seed(seed: Optional[int]) -> int:
    """Coerce ``seed`` into a valid non-negative Python ``int``.

    Accepts ``None`` (-> :data:`DEFAULT_SEED`), numpy/torch scalar integers and
    floats that are integral.  Seed ``0`` is allowed and meaningful.
    """
    if seed is None:
        return int(DEFAULT_SEED)
    if isinstance(seed, bool):  # bool is an int subclass; reject explicitly
        return int(seed)
    if isinstance(seed, float):
        if not float(seed).is_integer():
            raise ValueError(f"seed must be integral, got {seed!r}")
        return int(seed)
    # numpy / torch integer scalars expose __index__
    try:
        return int(seed)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"could not interpret seed {seed!r}") from exc


def set_seed(
    seed: Optional[int] = None,
    *,
    deterministic: bool = False,
    benchmark: Optional[bool] = None,
    torch: bool = True,
    numpy: bool = True,
    random_: bool = True,
    cuda: bool = True,
    num_threads: Optional[int] = None,
    warn_only: bool = True,
) -> int:
    """Seed every RNG the reproduction pipeline may use.

    Parameters
    ----------
    seed:
        The seed.  ``None`` falls back to :data:`DEFAULT_SEED`.
    deterministic:
        When ``True``, request deterministic algorithms from PyTorch
        (``torch.use_deterministic_algorithms``) and disable cuDNN
        benchmarking/autotuning.  This can raise for some ops, hence
        ``warn_only``.
    benchmark:
        Explicit value for ``torch.backends.cudnn.benchmark``.  Defaults to
        ``not deterministic`` when ``None``.
    torch, numpy, random_, cuda:
        Toggle individual RNG back-ends (useful when one of them is absent).
    num_threads:
        Optional ``torch.set_num_threads`` override.  Reproducibility across
        machines improves when the thread count is pinned.
    warn_only:
        Swallow exceptions raised while enabling fully deterministic mode
        instead of propagating them.

    Returns
    -------
    int
        The normalized seed that was applied.
    """
    seed = _normalize_seed(seed)

    if random_:
        random.seed(seed)

    if numpy and _np is not None:
        try:
            _np.random.seed(seed % _NUMPY_SEED_MAX)
        except Exception:  # pragma: no cover - defensive
            pass

    if torch and _torch is not None:
        _torch.manual_seed(seed)
        if cuda:
            try:
                if _torch.cuda.is_available():
                    _torch.cuda.manual_seed(seed)
                    _torch.cuda.manual_seed_all(seed)
            except Exception:  # pragma: no cover - no CUDA build
                pass
        if num_threads is not None:
            try:
                _torch.set_num_threads(int(num_threads))
            except Exception:  # pragma: no cover - defensive
                pass

        if deterministic or benchmark is not None:
            try:
                cudnn = _torch.backends.cudnn
                cudnn.deterministic = bool(deterministic)
                cudnn.benchmark = bool(
                    benchmark if benchmark is not None else (not deterministic)
                )
            except Exception:  # pragma: no cover
                pass
        if deterministic:
            try:
                _torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except Exception:  # pragma: no cover - unsupported build/op
                if not warn_only:
                    raise

    global _LAST_SEED
    _LAST_SEED = seed
    # Keep any child process spawned with ``PYTHONHASHSEED`` consistent.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return seed


def seed_everything(seed: Optional[int] = None, **kwargs: Any) -> int:
    """Alias of :func:`set_seed` (PL-style naming used in the runners)."""
    return set_seed(seed, **kwargs)


def get_seed() -> Optional[int]:
    """Return the most recently applied seed, or ``None`` if never set."""
    return _LAST_SEED


def make_generator(
    seed: Optional[int] = None,
    device: Union[str, "Any"] = "cpu",
) -> "Any":
    """Create a ``torch.Generator`` seeded with ``seed`` on ``device``.

    Useful for the residual-collocation sampling of §2.2, which must be
    identical across optimizers being compared (same points -> same loss
    landscape).  Raises ``RuntimeError`` if PyTorch is unavailable.
    """
    if _torch is None:  # pragma: no cover - torch is a hard pipeline dep
        raise RuntimeError("PyTorch is required to build a Generator")
    seed = _normalize_seed(seed)
    try:
        generator = _torch.Generator(device=device)
    except Exception:
        # ``device`` may be a torch.device-like object without a backend.
        generator = _torch.Generator()
    generator.manual_seed(seed)
    return generator


def spawn_seeds(base_seed: Optional[int], n: int) -> List[int]:
    """Derive ``n`` deterministic, well-separated child seeds from ``base_seed``.

    Uses a seeded ``random.Random`` so the derived sequence is reproducible on
    any Python version without relying on hash randomization.  This is how the
    per-run seeds (``seed``, ``seed + 1``, ... / ``{345, 456, 567}``) are
    generated during the width/lr sweeps.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    base_seed = _normalize_seed(base_seed)
    rng = random.Random(base_seed)
    seeds: List[int] = []
    while len(seeds) < n:
        candidate = rng.randrange(0, _NUMPY_SEED_MAX)
        if candidate not in seeds:
            seeds.append(candidate)
    return seeds


class SeedContext(contextlib.AbstractContextManager):
    """Context manager seeding all RNGs on enter and restoring on exit.

    Example
    -------
    >>> with SeedContext(345):          # doctest: +SKIP
    ...     model = make_pinn(width=200)
    """

    def __init__(self, seed: Optional[int] = None, **kwargs: Any) -> None:
        self.seed = _normalize_seed(seed)
        self.kwargs = kwargs
        self._previous: Optional[int] = None

    def __enter__(self) -> int:
        self._previous = _LAST_SEED
        return set_seed(self.seed, **self.kwargs)

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: D105
        # Do not fight the caller: just re-apply whatever seed was active.
        if self._previous is not None:
            set_seed(self._previous, **self.kwargs)
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"SeedContext(seed={self.seed})"


def seed_from_config(
    cfg: Optional[Dict[str, Any]],
    default: Optional[int] = None,
    *,
    keys: Sequence[str] = ("seed", "seeds", "random_seed", "manual_seed"),
) -> int:
    """Extract a single seed from a (possibly nested) config dictionary.

    Searches ``cfg`` top-down for the first key in ``keys``.  A list value
    yields its first element (the runners iterate over the full list
    themselves).  Falls back to ``default`` (or :data:`DEFAULT_SEED`).
    """
    if isinstance(cfg, dict):
        for key in keys:
            if key in cfg:
                value = cfg[key]
                if isinstance(value, (list, tuple, set)):
                    value = next(iter(value)) if value else None
                if value is not None:
                    return _normalize_seed(value)
        # Recurse into obvious nested config namespaces.
        for parent in ("experiment", "runtime", "optimizer", "nncg", "sampling"):
            sub = cfg.get(parent)
            if isinstance(sub, dict) and any(k in sub for k in keys):
                return seed_from_config(sub, default=default, keys=keys)
    return _normalize_seed(default)


def describe_rng_state() -> Dict[str, Any]:
    """Return a small JSON-serializable snapshot of the seeding state.

    Experiment runners embed this in ``summary.json`` metadata so a reported
    result can be traced back to the exact RNG configuration that produced it.
    """
    info: Dict[str, Any] = {
        "seed": _LAST_SEED,
        "python_random_seeded": True,
        "numpy_available": _np is not None,
        "torch_available": _torch is not None,
        "cuda_available": False,
        "deterministic_algorithms": None,
    }
    if _torch is not None:
        try:
            info["cuda_available"] = bool(_torch.cuda.is_available())
        except Exception:  # pragma: no cover
            pass
        try:
            info["deterministic_algorithms"] = bool(
                _torch.are_deterministic_algorithms_enabled()
            )
        except Exception:  # pragma: no cover
            pass
        try:
            info["num_threads"] = int(_torch.get_num_threads())
        except Exception:  # pragma: no cover
            pass
    return info


# ---------------------------------------------------------------------------
# Self-test: ``python -m src.utils.seeding``
# ---------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual smoke test
    seeds = spawn_seeds(DEFAULT_SEED, 5)
    assert len(seeds) == 5 and len(set(seeds)) == 5, seeds
    assert seeds == spawn_seeds(DEFAULT_SEED, 5), "spawn_seeds must be deterministic"
    assert set_seed(1234) == 1234 and get_seed() == 1234
    assert set_seed(None) == DEFAULT_SEED

    with SeedContext(7) as applied:
        assert applied == 7

    # Reproducibility of the torch path (skip silently if torch is absent).
    if _torch is not None:
        set_seed(11)
        a = _torch.randn(3)
        set_seed(11)
        b = _torch.randn(3)
        assert _torch.allclose(a, b)
        gen = make_generator(5)
        assert isinstance(gen, _torch.Generator)

    if _np is not None:
        set_seed(3)
        x = _np.random.rand(3)
        set_seed(3)
        y = _np.random.rand(3)
        assert (x == y).all()

    cfg = {"experiment": {"seeds": [345, 456, 567]}}
    assert seed_from_config(cfg) == 345
    assert seed_from_config({}, default=99) == 99
    print("seeding self-test OK:", describe_rng_state())


if __name__ == "__main__":  # pragma: no cover
    _self_test()
