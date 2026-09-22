"""Deterministic seeding utilities for the LBCS reproduction.

Glue layer: no paper-specific formula. Provides:

* :func:`set_seed` -- seed python/numpy/torch (and CUDA) deterministically.
* :func:`resolve_seed` -- derive a deterministic per-repeat seed.
* :func:`seed_everything` -- alias of :func:`set_seed`.
* :func:`make_generator` -- seeded numpy/torch generators for loaders.
* :func:`worker_init_fn` -- ``DataLoader`` worker seeding callback.
* :func:`seed_scope` -- context manager restoring global RNG state afterwards.

The paper's statistical protocol uses fixed seeds per repeat:

* 20 repeats for Section 5.1 (Table 1),
* 10 repeats for Sections 5.2/5.3 (Tables 2/3, Figure 2, Table 8) and Section 6.

Seeds used for every repeat are logged (see :func:`describe_seed`) so the
reporting requirement "log them" is satisfied.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
from typing import Any, Dict, Iterator, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SEED",
    "PAPER_REPEATS",
    "set_seed",
    "seed_everything",
    "resolve_seed",
    "repeat_seeds",
    "make_generator",
    "worker_init_fn",
    "seed_scope",
    "torch_available",
    "describe_seed",
    "set_deterministic",
]

#: Base seed used when a caller does not provide one.
DEFAULT_SEED = 0

#: Repeats per paper section (statistical reporting protocol).
PAPER_REPEATS: Dict[str, int] = {
    "section5.1": 20,
    "section5.2": 10,
    "section5.3": 10,
    "section6": 10,
}

_TORCH_AVAILABLE: Optional[bool] = None


def torch_available() -> bool:
    """Return ``True`` when PyTorch can be imported (cached)."""

    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:  # pragma: no cover - environment dependent
            import torch  # noqa: F401

            _TORCH_AVAILABLE = True
        except Exception:  # pragma: no cover - torch is a soft dependency
            _TORCH_AVAILABLE = False
    return bool(_TORCH_AVAILABLE)


def _as_int_seed(seed: Any) -> int:
    """Coerce an arbitrary seed value into a non-negative python int."""

    if seed is None:
        seed = DEFAULT_SEED
    if isinstance(seed, (np.integer,)):
        return int(seed)
    if isinstance(seed, (int,)):
        return int(seed)
    if isinstance(seed, float):
        return int(seed)
    if isinstance(seed, str):
        return int.from_bytes(seed.encode("utf-8")[:8] or b"\x00", "little")
    try:
        return int(seed)  # type: ignore[arg-type]
    except Exception:
        return DEFAULT_SEED


def set_seed(seed: Optional[int] = None) -> int:
    """Seed python, NumPy and (optionally) PyTorch/CUDA deterministically.

    Parameters
    ----------
    seed:
        Seed value; ``None`` falls back to :data:`DEFAULT_SEED`.

    Returns
    -------
    int
        The seed that was actually applied.
    """

    value = _as_int_seed(seed)

    random.seed(value)
    np.random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)

    if torch_available():
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover - backend dependent
            pass

    LOGGER.debug("set_seed(%s)", value)
    return value


#: Alias kept for readability in drivers.
seed_everything = set_seed


def set_deterministic(enabled: bool = True, seed: Optional[int] = None) -> int:
    """Enable cuDNN determinism (and optionally seed) for reproducible runs."""

    value = _as_int_seed(seed)
    if torch_available():
        import torch

        try:
            torch.backends.cudnn.deterministic = bool(enabled)
            torch.backends.cudnn.benchmark = not bool(enabled)
            if enabled:
                # torch >= 1.8 provides use_deterministic_algorithms
                if hasattr(torch, "use_deterministic_algorithms"):
                    try:
                        torch.use_deterministic_algorithms(True, warn_only=True)
                    except TypeError:  # pragma: no cover - older signature
                        torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover - backend dependent
            pass
    if seed is not None:
        set_seed(seed)
    return value


def resolve_seed(seed: Optional[int] = None, repeat: int = 0, base: Optional[int] = None) -> int:
    """Derive a deterministic seed for a given repeat index.

    ``resolve_seed(7, 0) == 7`` and subsequent repeats get well-separated,
    order-stable values (a simple linear congruential mix), which keeps each
    repeat independent yet fully reproducible.
    """

    base_value = _as_int_seed(seed if seed is not None else base)
    rep = max(0, int(repeat))
    if rep == 0:
        return base_value
    mixed = (base_value + rep * 7919) % (2 ** 31 - 1)
    return int(mixed)


def repeat_seeds(seed: Optional[int] = None, repeats: int = 10) -> list:
    """Return the list of seeds used for ``repeats`` repetitions."""

    base = _as_int_seed(seed)
    return [resolve_seed(base, r) for r in range(max(1, int(repeats)))]


def make_generator(seed: Optional[int] = None):
    """Return a seeded generator for numpy or torch loaders.

    Returns a ``torch.Generator`` when PyTorch is importable (so it can be fed
    to ``DataLoader(generator=...)``), otherwise a ``numpy`` ``Generator``.
    """

    value = _as_int_seed(seed)
    if torch_available():
        import torch

        gen = torch.Generator()
        gen.manual_seed(value)
        return gen
    return np.random.default_rng(value)


def worker_init_fn(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` keeping workers reproducible.

    Uses the current torch seed so every worker gets a distinct but
    deterministic stream.
    """

    info = None
    if torch_available():
        import torch

        info = torch.utils.data.get_worker_info()
    if info is None:
        seed = _as_int_seed(None) + int(worker_id)
    else:
        seed = int(info.seed) % (2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    if torch_available():
        import torch

        torch.manual_seed(seed)


@contextlib.contextmanager
def seed_scope(seed: Optional[int] = None) -> Iterator[int]:
    """Context manager casing a :func:`set_seed` call.

    Global RNG states (python/numpy/torch) are saved and restored so a scoped
    seed does not leak into the surrounding experiment loop.
    """

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_states = None
    if torch_available():
        import torch

        torch_states = (torch.get_rng_state(),)
        if torch.cuda.is_available():
            torch_states = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    value = set_seed(seed)
    try:
        yield value
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_states is not None:
            import torch

            torch.set_rng_state(torch_states[0])
            if torch.cuda.is_available() and len(torch_states) > 1:
                try:
                    torch.cuda.set_rng_state_all(torch_states[1])
                except Exception:  # pragma: no cover - GPU dependent
                    pass


def describe_seed(seed: Optional[int], repeat: Optional[int] = None, section: Optional[str] = None) -> Dict[str, Any]:
    """Return a small dict describing the seed used for logging."""

    resolved = _as_int_seed(seed)
    info: Dict[str, Any] = {"seed": resolved}
    if repeat is not None:
        info["repeat"] = int(repeat)
        info["repeat_seed"] = resolve_seed(resolved, int(repeat))
    if section is not None:
        info["section"] = str(section)
        info["repeats"] = PAPER_REPEATS.get(str(section).lower(), None)
    return info


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline determinism checks (no GPU / dataset required)."""

    info: Dict[str, Any] = {}

    info["set_seed"] = set_seed(123)
    a = np.random.rand(4)
    set_seed(123)
    b = np.random.rand(4)
    assert np.allclose(a, b), "numpy seeding is not deterministic"
    info["numpy_deterministic"] = True

    set_seed(7)
    x = random.random()
    set_seed(7)
    assert x == random.random()
    info["python_deterministic"] = True

    info["resolve_seed"] = [resolve_seed(7, r) for r in range(4)]
    assert info["resolve_seed"][0] == 7
    assert len(set(info["resolve_seed"])) == 4, "repeat seeds must be distinct"
    assert resolve_seed(7, 3) == resolve_seed(7, 3), "repeat seeds must be stable"

    info["repeat_seeds"] = repeat_seeds(0, 3)

    with seed_scope(11):
        np.random.rand(2)
    with seed_scope(11):
        after = np.random.rand(2)
    with seed_scope(11):
        after2 = np.random.rand(2)
    assert np.allclose(after, after2), "seed_scope must be reproducible"
    info["seed_scope"] = True

    info["torch_available"] = torch_available()
    if torch_available():
        import torch

        set_seed(5)
        t1 = torch.rand(4)
        set_seed(5)
        t2 = torch.rand(4)
        assert torch.allclose(t1, t2), "torch seeding is not deterministic"
        gen = make_generator(3)
        assert hasattr(gen, "manual_seed")
        info["torch_deterministic"] = True

    info["describe_seed"] = describe_seed(0, repeat=4, section="section5.2")
    assert info["describe_seed"]["repeat_seed"] == resolve_seed(0, 4)

    # restore a neutral state for subsequent imports
    set_seed(DEFAULT_SEED)

    if verbose:
        print("seed selftest:", info)
    return info


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest()
