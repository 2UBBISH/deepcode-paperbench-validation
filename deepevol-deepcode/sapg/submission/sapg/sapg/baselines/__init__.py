"""SAPG baselines package.

Exposes the comparison methods used in the SAPG paper (Section 5.2):

* ``ppo_baseline`` -- vanilla single-policy PPO at ``N = 24576`` envs, used to
  demonstrate the policy-gradient batch-size saturation (Fig. 2 concept).
* ``pql``          -- Parallel Q-Learning (PQL), a parallelised off-policy
  actor-critic (DDPG-style) baseline with mixed exploration noise.
* ``dexpbt``       -- DexPBT, PPO plus population based training with ``M = 6``
  groups whose worst members are replaced by the best weights with mutated
  hyperparameters.

The public surface is intentionally *lazy* (PEP 562 module ``__getattr__``) so
that ``import sapg.baselines`` stays cheap and does not drag in ``torch``,
``numpy`` or IsaacGym unless a baseline is actually requested.

Name resolution order
---------------------
1. a set of hand-written convenience aliases (``"pbt"`` -> ``train_dexpbt``),
2. the canonical name in the owning submodule.

If an optional baseline module is missing (e.g. a stripped-down checkout) the
attribute access raises an informative :class:`AttributeError` instead of an
opaque :class:`ImportError`, but the package itself always imports cleanly.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__all__: List[str] = [
    # method tags used by the trainer dispatch
    "PPO_METHODS",
    "BASELINE_METHODS",
    # PPO baseline (Section 5.2)
    "PPOBaselineTrainer",
    "PPOBaselineResult",
    "train_ppo_baseline",
    "run_ppo_seeds",
    "aggregate_seed_histories",
    "paper_standard_error",
    "ppo_saturation_sweep",
    "saturation_summary",
    "make_ppo_config",
    "make_ppo_policy",
    "make_ppo_env",
    # PQL baseline (Section 5.2)
    "PQLTrainer",
    "PQLConfig",
    "train_pql",
    # DexPBT baseline (Section 5.2)
    "DexPBTTrainer",
    "DexPBTConfig",
    "train_dexpbt",
    # dispatch helper
    "make_baseline",
    "train_baseline",
]


# ---------------------------------------------------------------------------
# canonical name -> owning submodule
# ---------------------------------------------------------------------------
_LAZY_EXPORTS: Dict[str, str] = {
    # ---- PPO baseline -----------------------------------------------------
    "PPOBaselineTrainer": "ppo_baseline",
    "PPOBaselineResult": "ppo_baseline",
    "train_ppo_baseline": "ppo_baseline",
    "run_ppo_seeds": "ppo_baseline",
    "aggregate_seed_histories": "ppo_baseline",
    "paper_standard_error": "ppo_baseline",
    "ppo_saturation_sweep": "ppo_baseline",
    "saturation_summary": "ppo_baseline",
    "make_ppo_config": "ppo_baseline",
    "make_ppo_policy": "ppo_baseline",
    "make_ppo_env": "ppo_baseline",
    # ---- PQL --------------------------------------------------------------
    "PQLTrainer": "pql",
    "PQLConfig": "pql",
    "train_pql": "pql",
    # ---- DexPBT -----------------------------------------------------------
    "DexPBTTrainer": "dexpbt",
    "DexPBTConfig": "dexpbt",
    "train_dexpbt": "dexpbt",
}


# ---------------------------------------------------------------------------
# method tags (Section 5.2) -- keep in sync with ``sapg.__init__.train``
# ---------------------------------------------------------------------------
PPO_METHODS: Tuple[str, ...] = (
    "ppo",
    "vanilla_ppo",
    "ppo_baseline",
    "ppo-baseline",
)

BASELINE_METHODS: Tuple[str, ...] = (
    "sapg",
    "ppo",
    "vanilla_ppo",
    "ppo_baseline",
    "ppo-baseline",
    "pql",
    "dexpbt",
    "pbt",
    "expbt",
)


# aliases for the PQL / DexPBT families: alternate spellings -> canonical name.
_ALIASES: Dict[str, str] = {
    # PQL
    "parallel_q_learning": "train_pql",
    "parallelqlearning": "train_pql",
    "train_parallel_q_learning": "train_pql",
    "apql": "train_pql",
    # DexPBT / PBT
    "pbt": "train_dexpbt",
    "expbt": "train_dexpbt",
    "dexpbt_trainer": "DexPBTTrainer",
    "population_based_training": "train_dexpbt",
    "train_pbt": "train_dexpbt",
    # PPO
    "vanilla_ppo": "train_ppo_baseline",
    "ppo": "train_ppo_baseline",
}


def _module_for(name: str) -> Optional[str]:
    """Return the owning submodule for ``name`` (resolving aliases)."""

    canonical = _ALIASES.get(name, name)
    return _LAZY_EXPORTS.get(canonical)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for the public baseline surface."""

    if name.startswith("__"):  # pragma: no cover - dunder passthrough
        raise AttributeError(name)

    # module-level constants are computed eagerly
    if name == "PPO_METHODS":
        return PPO_METHODS
    if name == "BASELINE_METHODS":
        return BASELINE_METHODS

    canonical = _ALIASES.get(name, name)
    module_name = _LAZY_EXPORTS.get(canonical)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            f"Available names: {sorted(set(__all__))}"
        )

    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}: failed to import "
            f"optional baseline submodule {module_name!r} ({exc!r})"
        ) from exc

    try:
        value = getattr(module, canonical)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}: "
            f"{module_name!r} does not define {canonical!r}"
        ) from exc

    # cache on the package so repeated lookups are cheap
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(__all__))


# ---------------------------------------------------------------------------
# convenience factories
# ---------------------------------------------------------------------------
def train_baseline(config: Any = None, method: Optional[str] = None, **kwargs: Any) -> Any:
    """Dispatch to the baseline trainer selected by ``method``/``config.method``.

    Parameters
    ----------
    config:
        A :class:`sapg.utils.config.SAPGConfig` (or any object/dict exposing a
        ``method`` field).  ``None`` falls back to the default PPO baseline.
    method:
        Explicit method override (one of :data:`BASELINE_METHODS`).
    **kwargs:
        Forwarded verbatim to the selected trainer entry point.

    Returns
    -------
    The ``(trainer, history)`` tuple produced by the selected entry point.
    """

    name = method
    if name is None and config is not None:
        if isinstance(config, dict):
            name = config.get("method")
        else:
            name = getattr(config, "method", None)
    name = "ppo" if name is None else str(name).lower().replace("-", "_")

    if name in ("ppo", "vanilla_ppo", "ppo_baseline"):
        return train_ppo_baseline(config=config, **kwargs)
    if name in ("pql", "parallel_q_learning", "apql"):
        return train_pql(config=config, **kwargs)
    if name in ("dexpbt", "pbt", "expbt"):
        return train_dexpbt(config=config, **kwargs)

    raise ValueError(
        f"unknown baseline method {name!r}; expected one of {sorted(BASELINE_METHODS)}"
    )


def make_baseline(config: Any = None, method: Optional[str] = None, **kwargs: Any) -> Any:
    """Construct (without training) the baseline trainer for ``method``."""

    name = method
    if name is None and config is not None:
        if isinstance(config, dict):
            name = config.get("method")
        else:
            name = getattr(config, "method", None)
    name = "ppo" if name is None else str(name).lower().replace("-", "_")

    if name in ("ppo", "vanilla_ppo", "ppo_baseline"):
        return PPOBaselineTrainer(config=config, **kwargs)  # noqa: F821 (lazy)
    if name in ("pql", "parallel_q_learning", "apql"):
        return PQLTrainer(config=config, **kwargs)  # noqa: F821 (lazy)
    if name in ("dexpbt", "pbt", "expbt"):
        return DexPBTTrainer(config=config, **kwargs)  # noqa: F821 (lazy)

    raise ValueError(
        f"unknown baseline method {name!r}; expected one of {sorted(BASELINE_METHODS)}"
    )


__version__ = "0.1.0"
