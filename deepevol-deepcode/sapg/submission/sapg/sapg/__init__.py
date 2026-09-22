"""SAPG: Split and Aggregate Policy Gradients.

Top-level package initializer.

What lives here
---------------
``sapg`` implements the SAPG algorithm of the paper *"SAPG: Split and Aggregate
Policy Gradients"*.  The package is organised as:

* :mod:`sapg.utils`       -- configuration, GAE / return utilities, logging
* :mod:`sapg.models`      -- shared actor / critic backbones conditioned on ``phi_j``
* :mod:`sapg.envs`        -- IsaacGym parallel environments (AllegroKuka / hands)
* :mod:`sapg.buffers`     -- per-policy rollout storage ``D_1..D_M``
* :mod:`sapg.losses`      -- on-policy PPO loss, off-policy IS loss, critic losses
* :mod:`sapg.aggregation` -- leader/follower (§4.3) and symmetric (§4.2) schemes
* :mod:`sapg.algorithms`  -- :class:`~sapg.algorithms.ppo.PPOTrainer` and
  :class:`~sapg.algorithms.sapg.SAPGTrainer` (Algorithm 1)
* :mod:`sapg.baselines`   -- PPO / PQL / DexPBT comparisons of §5.2
* :mod:`sapg.analysis`    -- diversity metrics of §6.4 (Figs. 7-8)

The package is intentionally import-light: sub-packages are imported lazily by
their consumers so that the core algorithm can be exercised without IsaacGym
installed (unit tests use lightweight environment stubs).
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "__version__",
    "N",
    "M",
    "get_version",
    "get_config",
    "make_policy",
    "make_env",
    "train",
]

__version__ = "0.1.0"

# Paper-level constants (N environments, M policies).  Kept in sync with
# ``sapg.utils.config`` but duplicated here so that ``import sapg`` alone gives
# access to the headline numbers without pulling in PyYAML/torch heavy paths.
N: int = 24576  # number of massively-parallel IsaacGym environments (Sec. 5.2)
M: int = 6      # number of policies == number of environment blocks (Sec. 4.3)


def get_version() -> str:
    """Return the package version string."""
    return __version__


def get_config(task: str = "regrasping", **overrides: Any) -> Any:
    """Build a :class:`sapg.utils.config.SAPGConfig` for ``task``.

    Thin re-export of :func:`sapg.utils.config.build_config` so that
    ``sapg.get_config("throw", num_policies=4)`` works after ``import sapg``.
    """
    from .utils.config import build_config

    return build_config(task, **overrides)


def make_env(config: Any = None, **kwargs: Any) -> Any:
    """Instantiate the IsaacGym task suite described by ``config``.

    Delegates to :func:`sapg.envs.make_env`.  IsaacGym is only imported at this
    point, which keeps ``import sapg`` usable on machines without the simulator.
    """
    from .envs import make_env as _make_env

    return _make_env(config=config, **kwargs)


def make_policy(config: Any, **kwargs: Any) -> Any:
    """Instantiate the shared actor/critic policy ``pi_theta(a|s, phi_j)``.

    Delegates to :class:`sapg.models.actor.ActorCritic`, which owns the shared
    actor backbone ``B_theta``, the shared critic backbone ``C_psi`` and the
    per-policy latents ``phi_j`` (Sec. 4.4).
    """
    from .models.actor import ActorCritic

    return ActorCritic(config=config, **kwargs)


def train(config: Any = None, **kwargs: Any) -> Any:
    """Run SAPG (Algorithm 1) or a configured baseline.

    The ``config.method`` field selects the trainer:

    * ``"sapg"``   -> :func:`sapg.algorithms.sapg.train_sapg`
    * ``"ppo"``    -> :func:`sapg.baselines.ppo_baseline.train_ppo_baseline`
    * ``"pql"``    -> :func:`sapg.baselines.pql.train_pql`
    * ``"dexpbt"`` -> :func:`sapg.baselines.dexpbt.train_dexpbt`
    """
    method = getattr(config, "method", "sapg")
    method = str(method).lower()

    if method == "sapg":
        from .algorithms.sapg import train_sapg

        return train_sapg(config, **kwargs)
    if method in ("ppo", "vanilla_ppo", "ppo_baseline"):
        from .baselines.ppo_baseline import train_ppo_baseline

        return train_ppo_baseline(config, **kwargs)
    if method == "pql":
        from .baselines.pql import train_pql

        return train_pql(config, **kwargs)
    if method in ("dexpbt", "pbt", "expbt"):
        from .baselines.dexpbt import train_dexpbt

        return train_dexpbt(config, **kwargs)

    raise ValueError(
        f"Unknown method {method!r}; expected one of "
        "'sapg', 'ppo', 'pql', 'dexpbt'."
    )
