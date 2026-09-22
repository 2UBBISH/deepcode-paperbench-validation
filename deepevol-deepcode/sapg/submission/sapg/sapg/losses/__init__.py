"""Loss functions for SAPG.

This package groups the three loss families used by the SAPG trainer and its
baselines:

* :mod:`sapg.losses.ppo_loss` -- the on-policy clipped surrogate objective
  (paper Eq. 2) plus the entropy regulariser of Eq. 10 and the action-bound
  regulariser (coefficient ``1e-4``).
* :mod:`sapg.losses.off_policy_loss` -- the importance-sampled off-policy PPO
  loss used by the leader (paper Eq. 3) and its combination with the on-policy
  term (paper Eq. 4, ``L = L_on + lambda * L_off`` with ``lambda = 1``).
* :mod:`sapg.losses.critic_loss` -- the critic targets and value losses for
  both the 3-step on-policy target (Eq. 5) and the 1-step off-policy target
  (Eq. 7), combined with the critic coefficient ``lambda' = 4.0`` (Eqs. 8-9).

The module uses PEP 562 lazy attribute resolution so that ``import
sapg.losses`` stays cheap and does not eagerly construct torch modules.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__: List[str] = [
    # ---- on-policy (Eq. 2 / Eq. 10) -------------------------------------
    "PPOLoss",
    "compute_ppo_loss",
    "ppo_loss",
    "on_policy_loss",
    "compute_entropy_bonus",
    "entropy_bonus",
    "entropy_coefficient_for",
    "combine_policy_and_entropy",
    "compute_bounds_loss",
    "importance_ratio",
    # ---- off-policy (Eq. 3 / Eq. 4) -------------------------------------
    "OffPolicyLoss",
    "off_policy_loss",
    "off_policy_surrogate",
    "mu_from_logprobs",
    "combine_on_off_loss",
    "combined_objective",
    # ---- critic targets / losses (Eqs. 5-9) -----------------------------
    "CriticLoss",
    "compute_critic_loss",
    "critic_loss",
    "compute_on_policy_targets",
    "compute_off_policy_targets",
    "compute_n_step_targets",
    "compute_one_step_targets",
    "compute_value_loss",
    "combined_critic_loss",
    "lambda_prime",
    # ---- meta -----------------------------------------------------------
    "loss_names",
    "make_loss",
]


# Name -> submodule that defines it.  ``critic_loss`` is imported lazily so a
# partially-populated checkout (e.g. while porting the paper) still works for
# the on-/off-policy loss families.
_LAZY_EXPORTS: Dict[str, str] = {}


def _register(module_name: str, names: List[str]) -> None:
    for name in names:
        _LAZY_EXPORTS[name] = module_name


_register(
    "ppo_loss",
    [
        "PPOLoss",
        "compute_ppo_loss",
        "ppo_loss",
        "on_policy_loss",
        "compute_entropy_bonus",
        "entropy_bonus",
        "entropy_coefficient_for",
        "combine_policy_and_entropy",
        "compute_bounds_loss",
        "importance_ratio",
    ],
)

_register(
    "off_policy_loss",
    [
        "OffPolicyLoss",
        "off_policy_loss",
        "off_policy_surrogate",
        "mu_from_logprobs",
        "combine_on_off_loss",
        "combined_objective",
    ],
)

_register(
    "critic_loss",
    [
        "CriticLoss",
        "compute_critic_loss",
        "critic_loss",
        "compute_on_policy_targets",
        "compute_off_policy_targets",
        "compute_n_step_targets",
        "compute_one_step_targets",
        "compute_value_loss",
        "combined_critic_loss",
        "lambda_prime",
    ],
)

# A few symbols are deliberately resolvable from either family; the second
# definition wins only when the primary module is unavailable (handled below).
_ALIASES: Dict[str, str] = {
    "on_policy_loss": "ppo_loss",
}


def _resolve(name: str) -> Any:
    """Import and return the attribute ``name`` from its owning submodule."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            "module {!r} has no attribute {!r} (available: {})".format(
                __name__, name, ", ".join(sorted(__all__))
            )
        )

    try:
        module = importlib.import_module("." + module_name, __name__)
    except ImportError as exc:  # pragma: no cover - optional dependency path
        # Fall back to the alternative family when the primary module is not
        # importable (e.g. critic_loss relies on utils/returns).
        for fallback in ("ppo_loss", "off_policy_loss", "critic_loss"):
            if fallback == module_name:
                continue
            try:
                module = importlib.import_module("." + fallback, __name__)
            except ImportError:
                continue
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(
            "could not import {!r} for sapg.losses.{!r}: {}".format(
                module_name, name, exc
            )
        ) from exc

    if not hasattr(module, name):
        alias = _ALIASES.get(name, name)
        if hasattr(module, alias):
            return getattr(module, alias)
        raise AttributeError(
            "module {!r} has no attribute {!r}".format(module.__name__, name)
        )
    return getattr(module, name)


def __getattr__(name: str) -> Any:  # PEP 562
    try:
        value = _resolve(name)
    except AttributeError:
        raise
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> List[str]:
    return sorted(set(__all__))


def lambda_prime() -> float:
    """Return the critic coefficient ``lambda' = 4.0`` from the paper (Sec. 4.1)."""

    return 4.0


def loss_names() -> List[str]:
    """Return the public loss-family names, useful for introspection/CLI help."""

    return sorted(set(__all__))


def make_loss(name: str, config: Any = None, **kwargs: Any) -> Any:
    """Construct one of the loss modules by name.

    Parameters
    ----------
    name:
        One of ``"ppo"``/``"on_policy"``, ``"off_policy"`` or ``"critic"`` (case
        and separators are normalised).
    config:
        Optional :class:`~sapg.utils.config.SAPGConfig` supplying hyperparameter
        defaults (``clip_epsilon``, ``entropy_coefficient``, ``critic_coefficient``
        ...).  Explicit ``kwargs`` take precedence.
    """

    key = str(name).lower().replace("-", "_").replace(" ", "_")
    if key in ("ppo", "on_policy", "onpolicy", "ppo_loss"):
        ppo_loss_cls = _resolve("PPOLoss")
        return ppo_loss_cls(config=config, **kwargs) if config is not None else ppo_loss_cls(**kwargs)
    if key in ("off_policy", "offpolicy", "off_policy_loss"):
        off_cls = _resolve("OffPolicyLoss")
        return off_cls(config=config, **kwargs) if config is not None else off_cls(**kwargs)
    if key in ("critic", "critic_loss", "value"):
        critic_cls = _resolve("CriticLoss")
        return critic_cls(config=config, **kwargs) if config is not None else critic_cls(**kwargs)
    raise ValueError(
        "unknown loss {!r}; expected one of 'ppo', 'off_policy', 'critic'".format(name)
    )
