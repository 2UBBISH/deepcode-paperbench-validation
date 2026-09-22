"""SAPG training algorithms.

This package exposes the two trainers used in the paper:

* :class:`~sapg.algorithms.ppo.PPOTrainer` -- the on-policy PPO base trainer
  used both as SAPG's inner optimization primitive and as the standalone PPO
  baseline (Section 3 Eq. 2, Section 5.2).
* :class:`~sapg.algorithms.sapg.SAPGTrainer` -- the full *Split and Aggregate
  Policy Gradients* loop (Algorithm 1, Section 4.6): split ``N`` massively
  parallel environments into ``M`` blocks, roll out one policy per block, then
  fuse the blocks' data into the leader via the importance-sampled off-policy
  PPO update.

Rollout collection helpers (:class:`~sapg.algorithms.rollout.BlockManager`,
:class:`~sapg.algorithms.rollout.RolloutCollector`, ``collect_data``,
``collect_on_policy``) live in :mod:`sapg.algorithms.rollout`.

The convenience entry points ``train_ppo_baseline`` and ``train_sapg`` are
re-exported so that ``sapg.train(...)`` and the ``scripts/`` entry points can
import everything from a single place.

Imports are performed lazily via ``__getattr__`` (PEP 562) so that
``import sapg.algorithms`` stays cheap and does not pull in ``torch`` unless a
trainer is actually requested.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__: List[str] = [
    # trainers
    "PPOTrainer",
    "PPOUpdateStats",
    "SAPGTrainer",
    "SAPGUpdateStats",
    # loss helpers re-exported for convenience
    "compute_ppo_loss",
    "compute_value_loss",
    "compute_bounds_loss",
    "compute_entropy_bonus",
    # rollout
    "BlockManager",
    "RolloutCollector",
    "collect_data",
    "collect_on_policy",
    "split_blocks",
    # aggregation schemes (defined in sapg.algorithms.sapg)
    "LeaderFollowerScheme",
    "SymmetricScheme",
    "NoAggregationScheme",
    "make_scheme",
    # entry points
    "train_ppo_baseline",
    "train_sapg",
]

# Mapping of exported name -> (submodule, attribute).  Resolved lazily.
_LAZY_EXPORTS: Dict[str, str] = {
    # sapg.algorithms.ppo
    "PPOTrainer": "ppo",
    "PPOUpdateStats": "ppo",
    "compute_ppo_loss": "ppo",
    "compute_value_loss": "ppo",
    "compute_bounds_loss": "ppo",
    "compute_entropy_bonus": "ppo",
    "train_ppo_baseline": "ppo",
    # sapg.algorithms.rollout
    "BlockManager": "rollout",
    "RolloutCollector": "rollout",
    "collect_data": "rollout",
    "collect_on_policy": "rollout",
    "split_blocks": "rollout",
    # sapg.algorithms.sapg
    "SAPGTrainer": "sapg",
    "SAPGUpdateStats": "sapg",
    "LeaderFollowerScheme": "sapg",
    "SymmetricScheme": "sapg",
    "NoAggregationScheme": "sapg",
    "make_scheme": "sapg",
    "train_sapg": "sapg",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatch
    """Lazily import the requested public attribute (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"{name!r} is not available in {module.__name__!r}: {exc}"
        ) from exc


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    return sorted(__all__)
