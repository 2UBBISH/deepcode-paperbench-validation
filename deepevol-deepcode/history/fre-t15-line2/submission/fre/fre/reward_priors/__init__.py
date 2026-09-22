"""Prior reward distributions ``p(eta)`` for Functional Reward Encodings (FRE).

This package implements the "prior reward distribution" of Section 4.2 of
*Zero-Shot Reinforcement Learning via Functional Reward Encodings*: a uniform
mixture of

* random singleton functions ("goal reaching" rewards) -- ``goal_reaching.py``
* random linear functions -- ``linear.py``
* random MLP functions (two linear layers) -- ``random_mlp.py``

together with the mixture itself and the named prior variants (``FRE-all``,
``FRE-goals``, ``FRE-lin``, ``FRE-mlp``, ``FRE-lin-mlp``, ``FRE-goal-mlp``,
``FRE-goal-lin`` and ``FRE-hint``) used by the prior-scaling study (Table 4)
and by Section 5.4 -- ``mixture.py``.

Reference implementation: https://github.com/kvfrans/fre
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    # Shared abstractions (goal_reaching.py)
    "RewardFunction",
    "RewardPrior",
    "as_numpy_2d",
    "euclidean_distance",
    # Goal-reaching ("singleton") prior
    "GoalReachingReward",
    "GoalReachingPrior",
    "make_goal_reaching_prior",
    "P_GOAL_CURRENT",
    "P_GOAL_FUTURE",
    "P_GOAL_RANDOM",
    # Linear prior
    "LinearRewardFunction",
    "LinearRewardPrior",
    "make_linear_prior",
    "ANTMAZE_POSITION_DIMS",
    "LINEAR_ZERO_PROB",
    # Random MLP prior
    "MLPParameters",
    "RandomMLPReward",
    "RandomMLPPrior",
    "make_mlp_prior",
    "sample_mlp_parameters",
    "MLP_HIDDEN_DIM",
    # Mixture + named variants + FRE-hint
    "MixtureRewardPrior",
    "HintMixturePrior",
    "DirectionalReward",
    "DirectionalPrior",
    "VelocityReward",
    "VelocityPrior",
    "make_mixture_prior",
    "make_prior_from_variant",
    "make_hint_prior",
    "build_component_prior",
    "variant_ratios",
    "COMPONENT_NAMES",
    "DEFAULT_PRIOR_RATIOS",
    "PRIOR_VARIANTS",
    "HINT_VARIANTS",
    "ANTMAZE_VELOCITY_DIMS",
    "HINT_VELOCITY_DIMS",
    "HINT_VELOCITY_RANGES",
    "HINT_DOMAINS",
    # Convenience factories
    "make_prior",
]

# name -> defining submodule (relative to this package).
_LAZY_ATTRS: Dict[str, str] = {
    # ---- goal_reaching.py -------------------------------------------------
    "RewardFunction": "goal_reaching",
    "RewardPrior": "goal_reaching",
    "as_numpy_2d": "goal_reaching",
    "euclidean_distance": "goal_reaching",
    "GoalReachingReward": "goal_reaching",
    "GoalReachingPrior": "goal_reaching",
    "make_goal_reaching_prior": "goal_reaching",
    "P_GOAL_CURRENT": "goal_reaching",
    "P_GOAL_FUTURE": "goal_reaching",
    "P_GOAL_RANDOM": "goal_reaching",
    # ---- linear.py --------------------------------------------------------
    "LinearRewardFunction": "linear",
    "LinearRewardPrior": "linear",
    "make_linear_prior": "linear",
    "ANTMAZE_POSITION_DIMS": "linear",
    "LINEAR_ZERO_PROB": "linear",
    # ---- random_mlp.py ----------------------------------------------------
    "MLPParameters": "random_mlp",
    "RandomMLPReward": "random_mlp",
    "RandomMLPPrior": "random_mlp",
    "make_mlp_prior": "random_mlp",
    "sample_mlp_parameters": "random_mlp",
    "MLP_HIDDEN_DIM": "random_mlp",
    # ---- mixture.py -------------------------------------------------------
    "MixtureRewardPrior": "mixture",
    "HintMixturePrior": "mixture",
    "DirectionalReward": "mixture",
    "DirectionalPrior": "mixture",
    "VelocityReward": "mixture",
    "VelocityPrior": "mixture",
    "make_mixture_prior": "mixture",
    "make_prior_from_variant": "mixture",
    "make_hint_prior": "mixture",
    "build_component_prior": "mixture",
    "variant_ratios": "mixture",
    "COMPONENT_NAMES": "mixture",
    "DEFAULT_PRIOR_RATIOS": "mixture",
    "PRIOR_VARIANTS": "mixture",
    "HINT_VARIANTS": "mixture",
    "ANTMAZE_VELOCITY_DIMS": "mixture",
    "HINT_VELOCITY_DIMS": "mixture",
    "HINT_VELOCITY_RANGES": "mixture",
    "HINT_DOMAINS": "mixture",
}

# Notable per-class probabilities / sizes mirrored for convenience so config
# code can read the paper's numbers (Appendix A Table 3 and Appendix B)
# without importing the sampler modules.
PRIOR_RATIOS: Dict[str, float] = {"goal": 0.33, "linear": 0.33, "mlp": 0.33}
"""The uniform 0.33/0.33/0.33 mixture of Section 4.2 / Table 3."""


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for the prior-reward public API.

    Keeps ``import fre.reward_priors`` cheap (and importable while individual
    sampler modules are still being built out) and avoids importing numpy at
    package import time.
    """
    if name == "make_prior":
        # Convenience wrapper around :func:`make_prior_from_variant`.
        from importlib import import_module

        mixture = import_module(f"{__name__}.mixture")
        return mixture.make_prior_from_variant

    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(f"{__name__}.{submodule}")
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"module {module.__name__!r} has no attribute {name!r}"
        ) from exc

    # Memoise so subsequent lookups bypass this hook.
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """Expose lazy names to ``dir()`` / tab-completion."""
    return sorted(set(globals()) | set(__all__))
