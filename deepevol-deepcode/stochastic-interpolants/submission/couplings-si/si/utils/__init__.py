"""Utility helpers for the coupled stochastic-interpolant library.

This package collects the engineering glue that is *not* specified by the paper:

* :mod:`si.utils.config`      -- YAML/dataclass configuration loading.
* :mod:`si.utils.distributed` -- Lightning-Fabric helpers (setup, barriers,
  sharded dataloaders, rank-aware logging).
* :mod:`si.utils.ema`         -- exponential moving average of model weights
  (optional; the paper does not mention EMA).

The re-exports are intentionally *lazy* (PEP 562 module ``__getattr__``) so that
importing :mod:`si.utils` never drags in ``torch``/``fabric``/``yaml`` when only a
tiny helper (e.g. a path utility) is needed.  Any missing optional dependency is
gracefully skipped: lookups for symbols of an unavailable sub-module raise a
descriptive :class:`AttributeError` instead of an import-time crash.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__: List[str] = [
    # config
    "load_config",
    "save_config",
    "Config",
    "config_to_dict",
    "get_nested",
    "set_nested",
    "merge_configs",
    "DEFAULTS",
    # distributed
    "setup",
    "is_distributed",
    "get_world_size",
    "get_rank",
    "barrier",
    "all_reduce_mean",
    "distributed_loader",
    "gather_tensor",
    "broadcast_tensor",
    "is_main_process",
    "rank_zero_print",
    "FabricLike",
    # ema
    "EMA",
    "update_ema",
    "copy_params",
]

# Names routed to their defining sub-module.
_CONFIG_NAMES = frozenset(
    {
        "load_config",
        "save_config",
        "Config",
        "config_to_dict",
        "get_nested",
        "set_nested",
        "merge_configs",
        "DEFAULTS",
    }
)
_DISTRIBUTED_NAMES = frozenset(
    {
        "setup",
        "is_distributed",
        "get_world_size",
        "get_rank",
        "barrier",
        "all_reduce_mean",
        "distributed_loader",
        "gather_tensor",
        "broadcast_tensor",
        "is_main_process",
        "rank_zero_print",
        "FabricLike",
    }
)
_EMA_NAMES = frozenset({"EMA", "update_ema", "copy_params"})


def __getattr__(name: str) -> Any:  # pragma: no cover - thin dispatcher
    """Lazily resolve utility symbols from the sibling sub-modules."""
    try:
        if name in _CONFIG_NAMES:
            from . import config as _config

            return getattr(_config, name)
        if name in _DISTRIBUTED_NAMES:
            from . import distributed as _distributed

            return getattr(_distributed, name)
        if name in _EMA_NAMES:
            from . import ema as _ema

            return getattr(_ema, name)
    except ImportError as exc:  # optional dependency missing
        raise AttributeError(
            f"si.utils.{name} is unavailable because an optional dependency is "
            f"missing ({exc})."
        ) from exc

    raise AttributeError(f"module 'si.utils' has no attribute {name!r}")


def __dir__() -> List[str]:
    """Introspection helper: module globals plus the declared public names."""
    return sorted(set(list(globals().keys()) + list(__all__)))


def describe() -> Dict[str, Optional[str]]:
    """Short descriptions of the utility modules (metadata/introspection only)."""
    return {
        "config": "YAML/dict configuration loading and merging helpers.",
        "distributed": "Lightning-Fabric setup and rank-aware utilities.",
        "ema": "Exponential moving average of model parameters (optional).",
    }
