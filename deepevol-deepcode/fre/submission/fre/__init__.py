"""Functional Reward Encodings (FRE) reference implementation.

Zero-Shot Reinforcement Learning via Functional Reward Encodings.

This package implements:

* ``fre.models``      -- the permutation-invariant transformer VAE over reward
  functions (encoder + reward decoder) together with the z-conditioned IQL
  agent used in the RL phase.
* ``fre.priors``      -- the mixture of random unsupervised reward priors
  (goal-reaching / random linear / random MLP) used to train the encoder and
  the §5.4 "hint" superset priors.
* ``fre.data``        -- offline dataset loaders (D4RL AntMaze, D4RL Kitchen,
  ExORL RND walker/cheetah) and domain-specific preprocessing.
* ``fre.envs``        -- the zero-shot evaluation task suites for AntMaze,
  ExORL (walker/cheetah) and Kitchen, plus reward-injection wrappers.
* ``fre.evaluation``  -- the zero-shot evaluation harness (encode K=32
  ``(s, eta(s))`` samples -> ``z`` -> rollout) and the return metrics used for
  Table 1 / Table 4.
* ``fre.baselines``   -- GC-IQL / GC-BC / OPAL baselines and a thin runner for
  the external goal-conditioned FB / SF code (``facebookresearch/controllable_agent``).
* ``fre.run_fre``     -- the strided two-phase training schedule (Algorithm 1).
* ``fre.main``        -- the CLI entry point (``train`` / ``eval`` /
  ``reproduce-tables``).

Only ``fre.models`` and ``fre.priors`` (which themselves defer heavy imports)
are imported eagerly here; everything else is exposed lazily so that
``import fre`` stays cheap and importable without torch/gym/D4RL installed.
"""

from __future__ import annotations

import importlib
from typing import Any, Tuple

__version__ = "0.1.0"

# Submodules that may be pulled in on demand.  They are deliberately *not*
# imported eagerly because several of them require optional third-party
# packages (torch, gym, d4rl, dm_control) that are only needed for training or
# evaluation.
_LAZY_SUBMODULES: Tuple[str, ...] = (
    "models",
    "priors",
    "data",
    "envs",
    "evaluation",
    "baselines",
    "run_fre",
    "main",
)

__all__ = [
    "__version__",
    # lazy submodules
    "models",
    "priors",
    "data",
    "envs",
    "evaluation",
    "baselines",
    "run_fre",
    "main",
    # convenience re-exports (resolved lazily via __getattr__)
    "FREModel",
    "FREEncoder",
    "FREDecoder",
    "RewardEmbedding",
    "RewardPrior",
    "make_reward_prior",
    "IQL",
    "FRETrainConfig",
    "FRERunner",
    "run_fre",
    "run",
    "evaluate_fre",
    "evaluate_task",
    "build_tasks",
]

# Names that live in a lazily imported submodule, mapped to that submodule.
_LAZY_ATTRS = {
    "FREModel": "fre.models",
    "FREEncoder": "fre.models",
    "FREDecoder": "fre.models",
    "RewardEmbedding": "fre.models",
    "IQL": "fre.models",
    "RewardPrior": "fre.priors",
    "make_reward_prior": "fre.priors",
    "FRETrainConfig": "fre.run_fre",
    "FRERunner": "fre.run_fre",
    "run": "fre.run_fre",
    "evaluate_fre": "fre.evaluation",
    "evaluate_task": "fre.evaluation",
    "build_tasks": "fre.envs",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access.

    Supports both ``fre.<submodule>`` and a small set of convenience
    re-exports (``fre.FREModel``, ``fre.run_fre`` ...).  Imported objects are
    cached in the module globals so subsequent access is free.
    """
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"fre.{name}")
        globals()[name] = module
        return module

    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'fre' has no attribute {name!r}")


def __dir__() -> Tuple[str, ...]:
    return tuple(sorted(set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_ATTRS)))
