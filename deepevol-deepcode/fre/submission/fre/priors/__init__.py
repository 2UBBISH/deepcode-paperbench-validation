"""Reward prior distributions for Functional Reward Encodings (FRE).

This package implements the unsupervised reward-function prior distribution
described in §4.2 / Appendix B of the FRE paper: a uniform ``1/3`` mixture of

* **goal-reaching** singleton reward functions (``goal_functions``), with goals
  sampled from the HER distribution (0.2 current / 0.5 future / 0.3 random),
* **random linear** reward functions with a ``p = 0.9`` per-dimension Bernoulli
  sparsity mask (``linear_functions``),
* **random MLP** reward functions of size ``(state_dim, 32, 1)`` with tanh
  activations and outputs clipped to ``[-1, 1]`` (``mlp_functions``),

together with the per-domain "hint" superset priors used for the
domain-knowledge study of §5.4.

``reward_prior`` dispatches over these families and produces batched
``(state, reward)`` training samples for the FRE encoder/decoder (K=32 encoder
pairs, K'=8 decoder pairs) as well as evaluation-time batches.
"""

from __future__ import annotations

from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Goal-reaching family (Appendix B, §4.2)
# ---------------------------------------------------------------------------
from fre.priors.goal_functions import (  # noqa: F401
    DEFAULT_GOAL_THRESHOLD,
    DEFAULT_UNIT_DIRECTIONS,
    HER_P_CURRENT,
    HER_P_FUTURE,
    HER_P_RANDOM,
    DirectionalRewardFunction,
    GoalRewardFunction,
    HERGoalSampler,
    RewardFunction,
    VelocityRewardFunction,
    extract_trajectory_bounds,
    goal_distances,
    goal_reached_mask,
    insert_goal_state,
    make_directional_reward_functions,
    make_velocity_hint_functions,
    sample_goal_reward_functions,
    sample_her_goal_indices,
    sample_her_goals,
    sample_random_goals,
)

# ---------------------------------------------------------------------------
# Random linear family (Appendix B)
# ---------------------------------------------------------------------------
from fre.priors.linear_functions import (  # noqa: F401
    DEFAULT_MASK_PROB,
    DEFAULT_WEIGHT_RANGE,
    LinearRewardFunction,
    antmaze_exclude_dims,
    sample_linear_reward_functions,
    sample_linear_weights,
)

# ---------------------------------------------------------------------------
# Random MLP family (Appendix B)
# ---------------------------------------------------------------------------
from fre.priors.mlp_functions import (  # noqa: F401
    DEFAULT_CLIP_VALUE,
    DEFAULT_HIDDEN_ACTIVATION,
    DEFAULT_HIDDEN_DIM,
    MLPRewardFunction,
    sample_mlp_parameters,
    sample_mlp_reward_functions,
)

# ---------------------------------------------------------------------------
# Mixture dispatcher (§4.2, §5.3, §5.4)
# ---------------------------------------------------------------------------
from fre.priors.reward_prior import (  # noqa: F401
    ANTMAZE_DEFAULT_NUM_DIRECTIONS,
    CHEETAH_HINT_VELOCITIES,
    DOMAIN_PRIOR_CONFIG,
    FAMILIES,
    FAMILY_GOAL,
    FAMILY_HINT,
    FAMILY_LINEAR,
    FAMILY_MLP,
    MixtureRewardFunction,
    PRESETS,
    RewardPrior,
    WALKER_HINT_VELOCITIES,
    build_hint_functions,
    domain_config,
    evaluate_mlp_parameters,
    hint_directions,
    hint_velocity_index,
    list_presets,
    make_reward_prior,
    resolve_distance_dims,
    resolve_family_weights,
    stack_mlp_parameters,
)

__all__ = [
    # goal family
    "RewardFunction",
    "GoalRewardFunction",
    "DirectionalRewardFunction",
    "VelocityRewardFunction",
    "HERGoalSampler",
    "goal_distances",
    "goal_reached_mask",
    "make_directional_reward_functions",
    "make_velocity_hint_functions",
    "sample_goal_reward_functions",
    "sample_her_goal_indices",
    "sample_her_goals",
    "sample_random_goals",
    "insert_goal_state",
    "extract_trajectory_bounds",
    "DEFAULT_GOAL_THRESHOLD",
    "DEFAULT_UNIT_DIRECTIONS",
    "HER_P_CURRENT",
    "HER_P_FUTURE",
    "HER_P_RANDOM",
    # linear family
    "LinearRewardFunction",
    "sample_linear_weights",
    "sample_linear_reward_functions",
    "antmaze_exclude_dims",
    "DEFAULT_MASK_PROB",
    "DEFAULT_WEIGHT_RANGE",
    # MLP family
    "MLPRewardFunction",
    "sample_mlp_parameters",
    "sample_mlp_reward_functions",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_CLIP_VALUE",
    "DEFAULT_HIDDEN_ACTIVATION",
    # mixture dispatcher
    "RewardPrior",
    "MixtureRewardFunction",
    "make_reward_prior",
    "resolve_family_weights",
    "list_presets",
    "domain_config",
    "resolve_distance_dims",
    "build_hint_functions",
    "hint_directions",
    "hint_velocity_index",
    "stack_mlp_parameters",
    "evaluate_mlp_parameters",
    "FAMILY_GOAL",
    "FAMILY_LINEAR",
    "FAMILY_MLP",
    "FAMILY_HINT",
    "FAMILIES",
    "PRESETS",
    "DOMAIN_PRIOR_CONFIG",
    "ANTMAZE_DEFAULT_NUM_DIRECTIONS",
    "WALKER_HINT_VELOCITIES",
    "CHEETAH_HINT_VELOCITIES",
    # lazy submodules
    "goal_functions",
    "linear_functions",
    "mlp_functions",
    "reward_prior",
]

_LAZY_SUBMODULES: Tuple[str, ...] = (
    "goal_functions",
    "linear_functions",
    "mlp_functions",
    "reward_prior",
)


def __getattr__(name: str) -> Any:
    """Lazily expose the prior submodules as attributes of the package."""
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"fre.priors.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Tuple[str, ...]:
    return tuple(sorted(set(list(globals().keys()) + list(_LAZY_SUBMODULES))))
