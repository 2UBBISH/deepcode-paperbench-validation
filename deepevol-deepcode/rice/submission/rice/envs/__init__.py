"""Environment layer for RICE (Cheng et al., ICML 2024, PMLR 235).

This package aggregates the environment-facing sub-modules used by both stages of
the RICE pipeline:

* :mod:`rice.envs.make_env` -- environment factory/registry covering the dense and
  sparse MuJoCo applications (``Hopper-v3``, ``Walker2d-v3``, ``Reacher-v2``,
  ``HalfCheetah-v3``, ``SparseHopper``, ``SparseHalfCheetah``) plus the application
  adapters for Selfish Mining (``Adopt/Reveal/Mine``), CAGE Challenge 2 (blue-agent
  action set with ``Restore`` penalty ``-1`` and trail lengths ``{30, 50, 100}``,
  final reward = sum of the three average rewards) and MetaDrive Macro-v1
  (2-D continuous action mapping to steering/acceleration/brake).
  Malware Mutation is deliberately out of scope.
* :mod:`rice.envs.sparse_reward` -- sparse-reward variants of Hopper/Walker2d/
  HalfCheetah where the reward is the forward x-position only once a threshold is
  crossed (``hopper/walker2d: 0.6``, ``halfcheetah: 5.0``, Appendix C.2).
* :mod:`rice.envs.reset_wrapper` -- Go-Explore-style state restore used by
  Algorithm 2 to jump to a stored critical state (the mixed initial state
  distribution ``mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s)``).
* :mod:`rice.envs.normalizer` -- observation normalization, applied where the paper
  does (Walker2d, HalfCheetah; Appendix C.2) and a no-op elsewhere.

Every sub-module is imported inside its own ``try/except`` so that ``import
rice.envs`` keeps working on installs without optional heavy dependencies
(``gym``/``mujoco``/``stable-baselines3``); use :func:`available` and
:func:`require` to probe availability explicitly.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List

__all__: List[str] = []

_LOGGER = logging.getLogger("rice.envs")

_SUBMODULES = (
    "make_env",
    "sparse_reward",
    "reset_wrapper",
    "normalizer",
)

# ---------------------------------------------------------------------------
# make_env: factory + registry + application helpers
# ---------------------------------------------------------------------------
_HAS_MAKE_ENV = False
try:  # pragma: no cover - availability depends on optional deps
    from .make_env import (  # noqa: F401
        ENV_SPECS,
        RESTORE_PENALTY,
        SPARSE_THRESHOLDS,
        TRAIL_LENGTHS,
        EnvSpec,
        available_envs,
        cage2_final_reward,
        d_max_for,
        env_backend,
        env_metadata,
        is_application_env,
        is_sparse_env,
        list_envs,
        make_application_env,
        make_env,
        make_sparse_env,
        make_vec_env,
        register_envs,
        resolve_env_spec,
        restore_action_penalty,
        summarize_envs,
    )

    __all__ += [
        "ENV_SPECS",
        "RESTORE_PENALTY",
        "SPARSE_THRESHOLDS",
        "TRAIL_LENGTHS",
        "EnvSpec",
        "available_envs",
        "cage2_final_reward",
        "d_max_for",
        "env_backend",
        "env_metadata",
        "is_application_env",
        "is_sparse_env",
        "list_envs",
        "make_application_env",
        "make_env",
        "make_sparse_env",
        "make_vec_env",
        "register_envs",
        "resolve_env_spec",
        "restore_action_penalty",
        "summarize_envs",
    ]
    _HAS_MAKE_ENV = True
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("rice.envs.make_env unavailable: %s", _exc)

# ---------------------------------------------------------------------------
# sparse_reward: sparse-reward wrappers for the Mazoure et al. variants
# ---------------------------------------------------------------------------
_HAS_SPARSE_REWARD = False
try:  # pragma: no cover
    from .sparse_reward import (  # noqa: F401
        SPARSE_THRESHOLDS as SPARSE_REWARD_THRESHOLDS,
        SparseReward,
        SparseRewardWrapper,
        build_sparse_reward_fn,
        forward_position,
        is_sparse_env_id,
        make_sparse_reward_env,
        sparse_reward_from_x,
        threshold_for,
        wrap_sparse_reward,
    )

    __all__ += [
        "SPARSE_REWARD_THRESHOLDS",
        "SparseReward",
        "SparseRewardWrapper",
        "build_sparse_reward_fn",
        "forward_position",
        "is_sparse_env_id",
        "make_sparse_reward_env",
        "sparse_reward_from_x",
        "threshold_for",
        "wrap_sparse_reward",
    ]
    _HAS_SPARSE_REWARD = True
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("rice.envs.sparse_reward unavailable: %s", _exc)

# ---------------------------------------------------------------------------
# reset_wrapper: Go-Explore-style state restore (Algorithm 2 critical resets)
# ---------------------------------------------------------------------------
_HAS_RESET_WRAPPER = False
try:  # pragma: no cover
    from .reset_wrapper import (  # noqa: F401
        ReplayResetter,
        ResetWrapper,
        StateSnapshot,
        env_supports_direct_state,
        get_env_state,
        make_reset_env,
        set_env_state,
        wrap_reset,
    )

    __all__ += [
        "ReplayResetter",
        "ResetWrapper",
        "StateSnapshot",
        "env_supports_direct_state",
        "get_env_state",
        "make_reset_env",
        "set_env_state",
        "wrap_reset",
    ]
    _HAS_RESET_WRAPPER = True
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("rice.envs.reset_wrapper unavailable: %s", _exc)

# ---------------------------------------------------------------------------
# normalizer: observation normalization (Walker2d, HalfCheetah only)
# ---------------------------------------------------------------------------
_HAS_NORMALIZER = False
try:  # pragma: no cover
    from .normalizer import (  # noqa: F401
        NORMALIZED_ENVS,
        NormalizeObservation,
        ObservationNormalizer,
        make_normalizer,
        should_normalize,
    )

    __all__ += [
        "NORMALIZED_ENVS",
        "NormalizeObservation",
        "ObservationNormalizer",
        "make_normalizer",
        "should_normalize",
    ]
    _HAS_NORMALIZER = True
except Exception as _exc:  # pragma: no cover
    _LOGGER.debug("rice.envs.normalizer unavailable: %s", _exc)


_AVAILABILITY: Dict[str, bool] = {
    "make_env": _HAS_MAKE_ENV,
    "sparse_reward": _HAS_SPARSE_REWARD,
    "reset_wrapper": _HAS_RESET_WRAPPER,
    "normalizer": _HAS_NORMALIZER,
}


def available(include_unavailable: bool = False) -> Dict[str, bool]:
    """Return availability flags for the environment sub-modules.

    Args:
        include_unavailable: When ``False`` (default) only successfully imported
            sub-modules are reported. When ``True`` every known sub-module name is
            reported together with its import flag.

    Returns:
        Mapping of sub-module name to availability boolean.
    """
    if include_unavailable:
        return dict(_AVAILABILITY)
    return {name: ok for name, ok in _AVAILABILITY.items() if ok}


def require(submodule: str) -> Any:
    """Import and return one environment sub-module by name.

    Args:
        submodule: One of ``"make_env"``, ``"sparse_reward"``,
            ``"reset_wrapper"`` or ``"normalizer"``. Case and ``-``/``_``
            separators are normalized.

    Returns:
        The imported sub-module object.

    Raises:
        KeyError: If ``submodule`` is not a known env sub-module.
        RuntimeError: If the sub-module exists but cannot be imported (for
            example because an optional dependency is missing).
    """
    name = str(submodule).strip().lower().replace("-", "_")
    if name not in _SUBMODULES:
        raise KeyError(
            f"Unknown rice.envs sub-module {submodule!r}; expected one of {list(_SUBMODULES)}"
        )
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"rice.envs.{name} could not be imported ({exc}). Install the optional "
            "dependency (e.g. gym/mujoco/stable-baselines3) to use this sub-module."
        ) from exc


def describe() -> str:
    """Return a one-line human-readable summary of sub-module availability."""
    ok = [n for n, flag in _AVAILABILITY.items() if flag]
    missing = [n for n, flag in _AVAILABILITY.items() if not flag]
    parts = [f"rice.envs: available={ok}"]
    if missing:
        parts.append(f"missing={missing}")
    return " | ".join(parts)


__all__ += ["available", "require", "describe"]
