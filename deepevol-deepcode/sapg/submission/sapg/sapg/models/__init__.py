"""SAPG model building blocks.

This package implements the two shared backbones of SAPG (Section 4.4):

* ``B_theta`` -- the actor backbone, shared across all M policies and
  conditioned on the per-policy latent ``phi_j``.
* ``C_psi`` -- the critic backbone, conditioned on the same ``phi_j``.

The public surface is resolved lazily (PEP 562) so that ``import
sapg.models`` stays cheap and does not require ``torch`` until one of the
concrete classes is actually requested.

Exports
-------
Actor / policy
    ``Actor``, ``ActorCritic``, ``Policy``, ``squash_action``,
    ``tanh_log_prob``
Critic
    ``Critic``
Networks
    ``MLP``, ``LSTM``, ``RecurrentBackbone``, ``LearnableSigma``,
    ``make_backbone``, ``mlp_units_for_task``, ``init_weights``,
    ``build_activation``, ``sigma_to_log_std``, ``log_std_to_sigma``,
    ``ACTIVATIONS``
Factories
    ``make_actor``, ``make_critic``, ``make_policy``
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__: List[str] = [
    # actor / policy
    "Actor",
    "ActorCritic",
    "Policy",
    "squash_action",
    "tanh_log_prob",
    # critic
    "Critic",
    # networks
    "MLP",
    "LSTM",
    "RecurrentBackbone",
    "LearnableSigma",
    "make_backbone",
    "mlp_units_for_task",
    "init_weights",
    "build_activation",
    "sigma_to_log_std",
    "log_std_to_sigma",
    "ACTIVATIONS",
    # factories
    "make_actor",
    "make_critic",
    "make_policy",
]

# Canonical symbol -> owning submodule.
_LAZY_EXPORTS: Dict[str, str] = {
    # sapg.models.actor
    "Actor": "actor",
    "ActorCritic": "actor",
    "Policy": "actor",
    "squash_action": "actor",
    "tanh_log_prob": "actor",
    # sapg.models.critic
    "Critic": "critic",
    # sapg.models.networks
    "MLP": "networks",
    "LSTM": "networks",
    "RecurrentBackbone": "networks",
    "LearnableSigma": "networks",
    "make_backbone": "networks",
    "mlp_units_for_task": "networks",
    "init_weights": "networks",
    "build_activation": "networks",
    "sigma_to_log_std": "networks",
    "log_std_to_sigma": "networks",
    "ACTIVATIONS": "networks",
    # factories live in the corresponding module too
    "make_actor": "actor",
    "make_critic": "critic",
    "make_policy": "actor",
}

# Symbols that are defined here rather than re-exported.
_LAZY_EXPORTS = {k: v for k, v in _LAZY_EXPORTS.items() if k in __all__ and k not in ()}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution."""
    if name in _LAZY_EXPORTS:
        module_name = _LAZY_EXPORTS[name]
        module = importlib.import_module("." + module_name, __name__)
        try:
            value = getattr(module, name)
        except AttributeError:
            # Factory helpers may only exist in some modules: build sensible
            # wrappers so the public surface stays stable.
            value = _make_factory(name)
            if value is None:
                raise AttributeError(
                    f"module {__name__!r} has no attribute {name!r} "
                    f"(not found in {module_name!r})"
                )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _make_factory(name: str) -> Any:
    """Fallback factory for the ``make_*`` helpers."""
    if name == "make_actor":

        def make_actor(config: Any = None, **kwargs: Any) -> Any:
            from .actor import Actor

            return Actor(config=config, **kwargs)

        return make_actor
    if name == "make_critic":

        def make_critic(config: Any = None, **kwargs: Any) -> Any:
            from .critic import Critic

            return Critic(config=config, **kwargs)

        return make_critic
    if name == "make_policy":

        def make_policy(config: Any = None, **kwargs: Any) -> Any:
            from .actor import ActorCritic

            return ActorCritic(config=config, **kwargs)

        return make_policy
    return None


def __dir__() -> List[str]:
    return sorted(set(__all__))
