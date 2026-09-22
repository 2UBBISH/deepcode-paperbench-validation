"""Utility sub-package for RICE.

Currently hosts reproducibility helpers (seeding) and shared small containers
(buffers, logging, plotting, normalization) that are used across the
``rice.algorithms``, ``rice.environments``, ``rice.evaluation`` and
``rice.baselines`` layers.

Only the light-weight :mod:`rice.utils.seeding` module is imported eagerly
(no torch/gym requirement).  Heavier helpers are exposed lazily through a
PEP 562 ``__getattr__`` hook so that ``import rice.utils`` stays cheap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .seeding import RNG, seed_env, set_global_seeds

__all__: List[str] = [
    "RNG",
    "seed_env",
    "set_global_seeds",
    "ReplayBuffer",
    "AverageMeter",
    "RunningMeanStd",
    "configure_logging",
    "get_logger",
    "moving_average",
    "save_json",
    "load_json",
    "format_table",
]


# --------------------------------------------------------------------------- #
# Lazy registry: symbol name -> (module, attribute name)
# --------------------------------------------------------------------------- #
_LAZY: Dict[str, Tuple[str, str]] = {
    "ReplayBuffer": ("buffers", "ReplayBuffer"),
    "AverageMeter": ("buffers", "AverageMeter"),
    "RunningMeanStd": ("normalization", "RunningMeanStd"),
    "configure_logging": ("logging_utils", "configure_logging"),
    "get_logger": ("logging_utils", "get_logger"),
    "moving_average": ("plotting", "moving_average"),
    "save_json": ("io_utils", "save_json"),
    "load_json": ("io_utils", "load_json"),
    "format_table": ("io_utils", "format_table"),
}


def _import_relative(module: str) -> Any:
    """Import a sibling utility module tolerating a few sys.path layouts."""
    import importlib

    candidates = (
        f"rice.utils.{module}",
        f"rice.rice.utils.{module}",
        f"{__name__}.{module}",
        module,
    )
    last_error: Exception = ImportError(f"cannot import {module}")
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    raise last_error


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for optional utility helpers."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = _import_relative(module_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(__all__)))
