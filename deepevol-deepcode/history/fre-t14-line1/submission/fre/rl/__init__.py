"""``fre.rl`` — offline RL components used by FRE (IQL, networks, replay buffer).

This package initializer re-exports the z-conditioned IQL agent, the
``[512, 512, 512]`` MLP/Q/V/policy networks (Table 3 / Appendix A), and the
numpy-backed offline dataset container.

Import policy
-------------
``fre.rl.iql`` and ``fre.rl.networks`` require ``torch``; ``fre.rl.replay_buffer``
is numpy-only.  To keep lightweight consumers (reporting scripts, ``--dry-run``
plans) working without ``torch`` installed, all heavy symbols are resolved
lazily through PEP 562 ``__getattr__`` and cached on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    # replay buffer (numpy only)
    "ReplayBuffer",
    "Episode",
    "DatasetStats",
    "stack_episodes",
    "make_replay_buffer",
    # networks (torch)
    "MLP",
    "make_activation",
    "QNetwork",
    "ValueNetwork",
    "GaussianPolicy",
    "TanhNormalPolicy",
    "IQLEstimator",
    "LOG_STD_MIN",
    "LOG_STD_MAX",
    # IQL (torch)
    "IQL",
    "IQLLearner",
    "expectile_loss",
    "expectile_loss_elementwise",
    "expectile",
    "awr_weights",
    "polyak_update",
    "encode_latent",
    "make_iql",
]

# ---------------------------------------------------------------------------
# Lazy attribute table: public symbol -> (submodule, attribute)
# ---------------------------------------------------------------------------
_LAZY_ATTRS = {
    # fre/rl/replay_buffer.py
    "ReplayBuffer": ("replay_buffer", "ReplayBuffer"),
    "Episode": ("replay_buffer", "Episode"),
    "DatasetStats": ("replay_buffer", "DatasetStats"),
    "stack_episodes": ("replay_buffer", "stack_episodes"),
    "make_replay_buffer": ("replay_buffer", "make_replay_buffer"),
    # fre/rl/networks.py
    "MLP": ("networks", "MLP"),
    "make_activation": ("networks", "make_activation"),
    "QNetwork": ("networks", "QNetwork"),
    "ValueNetwork": ("networks", "ValueNetwork"),
    "GaussianPolicy": ("networks", "GaussianPolicy"),
    "TanhNormalPolicy": ("networks", "TanhNormalPolicy"),
    "IQLEstimator": ("networks", "IQLEstimator"),
    "LOG_STD_MIN": ("networks", "LOG_STD_MIN"),
    "LOG_STD_MAX": ("networks", "LOG_STD_MAX"),
    # fre/rl/iql.py
    "IQL": ("iql", "IQL"),
    "IQLLearner": ("iql", "IQLLearner"),
    "expectile_loss": ("iql", "expectile_loss"),
    "expectile_loss_elementwise": ("iql", "expectile_loss_elementwise"),
    "expectile": ("iql", "expectile"),
    "awr_weights": ("iql", "awr_weights"),
    "polyak_update": ("iql", "polyak_update"),
    "encode_latent": ("iql", "encode_latent"),
    "make_iql": ("iql", "make_iql"),
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from fre.rl.iql import (  # noqa: F401
        IQL,
        IQLLearner,
        awr_weights,
        encode_latent,
        expectile,
        expectile_loss,
        expectile_loss_elementwise,
        make_iql,
        polyak_update,
    )
    from fre.rl.networks import (  # noqa: F401
        LOG_STD_MAX,
        LOG_STD_MIN,
        GaussianPolicy,
        IQLEstimator,
        MLP,
        QNetwork,
        TanhNormalPolicy,
        ValueNetwork,
        make_activation,
    )
    from fre.rl.replay_buffer import (  # noqa: F401
        DatasetStats,
        Episode,
        ReplayBuffer,
        make_replay_buffer,
        stack_episodes,
    )


def __getattr__(name: str):
    """Resolve a public symbol by lazily importing its defining submodule."""
    if name in _LAZY_ATTRS:
        from importlib import import_module

        submodule, attr = _LAZY_ATTRS[name]
        module = import_module(f"{__name__}.{submodule}")
        value = getattr(module, attr)
        globals()[name] = value  # cache so subsequent lookups are cheap
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
