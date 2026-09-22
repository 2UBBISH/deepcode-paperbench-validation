"""Reward-function prior ``p(eta)`` and ground-truth evaluation rewards for FRE.

This package exposes:

* The three unsupervised reward families that make up the FRE prior
  (Section 4.2 / Appendix B):

  - :class:`~fre.rewards.goal_reaching.GoalReachingPrior` -- goal-reaching
    singletons sampled with HER-style relabelling.
  - :class:`~fre.rewards.linear.LinearPrior` -- inner products with uniformly
    sampled weight vectors, with a 0.9 Bernoulli sparsity mask.
  - :class:`~fre.rewards.mlp.MLPPrior` -- random ``(state_dim, 32, 1)`` tanh MLPs.

* The 0.33/0.33/0.33 mixture :class:`~fre.rewards.prior.MixturePrior`, plus the
  Table 4 ablation subsets and the Figure 6 "hint" prior.

* The ground-truth zero-shot evaluation rewards for AntMaze / ExORL / Kitchen
  in :mod:`fre.rewards.eval_rewards`.

All reward functions share the same ``eta: S -> [-1, 1]`` interface defined by
:class:`fre.rewards.base.RewardFunction`, so the encoder, decoder, IQL policy and
evaluation harness treat prior-sampled and target rewards identically.

Everything is re-exported defensively so that a partial install (e.g. missing
``torch``) still loads whichever components are available.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []


def _try_import(module: str, names: List[str]) -> None:
    """Best-effort re-export of ``names`` from ``module``.

    Import failures are swallowed so that a partial installation (or a missing
    optional dependency such as ``torch``) does not prevent the remaining
    components from loading.
    """
    try:
        mod = __import__(module, fromlist=list(names))
    except Exception:  # pragma: no cover - defensive
        return
    for name in names:
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
            if name not in __all__:
                __all__.append(name)


# ---------------------------------------------------------------------------
# Base interface (RewardFunction / RewardFunctionPrior + dataset helpers)
# ---------------------------------------------------------------------------
_try_import(
    "fre.rewards.base",
    [
        "RewardFunction",
        "RewardFunctionPrior",
        "is_torch_tensor",
        "to_numpy",
        "to_torch",
        "get_observations",
        "get_terminals",
        "episode_boundaries",
        "get_rng",
    ],
)

# ---------------------------------------------------------------------------
# Goal-reaching family (sparse singletons + HER sampling)
# ---------------------------------------------------------------------------
_try_import(
    "fre.rewards.goal_reaching",
    [
        "GOAL_REACHING_FAMILY",
        "DEFAULT_P_CURRENT",
        "DEFAULT_P_FUTURE",
        "DEFAULT_P_RANDOM",
        "DEFAULT_GOAL_THRESHOLD",
        "DEFAULT_GOAL_REWARD",
        "DEFAULT_FAILURE_REWARD",
        "GoalReachingReward",
        "GoalReachingPrior",
        "GoalReachingRewardPrior",
        "goal_distance",
        "goal_success",
        "episode_index_of",
        "future_indices",
        "her_goal_indices",
        "sample_reward_context",
        "sample_context_and_decoder_pairs",
        "make_goal_reaching_reward",
        "make_goal_reaching_prior",
    ],
)

# ---------------------------------------------------------------------------
# Linear family (uniform weights + 0.9 sparsity mask)
# ---------------------------------------------------------------------------
_try_import(
    "fre.rewards.linear",
    [
        "LINEAR_FAMILY",
        "DEFAULT_MASK_PROB",
        "DEFAULT_KEEP_PROB",
        "DEFAULT_WEIGHT_RANGE",
        "ANTMAZE_XY_DIMS",
        "LinearReward",
        "LinearPrior",
        "LinearRewardPrior",
        "make_linear_reward",
        "make_linear_prior",
        "antmaze_linear_prior",
        "sample_linear_context",
    ],
)

# ---------------------------------------------------------------------------
# Random-MLP family
# ---------------------------------------------------------------------------
_try_import(
    "fre.rewards.mlp",
    [
        "MLP_FAMILY",
        "DEFAULT_HIDDEN_SIZES",
        "DEFAULT_INIT_SCALE",
        "DEFAULT_ACTIVATION",
        "MLPReward",
        "MLPPrior",
        "MLPRewardPrior",
        "get_activation",
        "layer_init_scale",
        "init_mlp_weights",
        "forward_mlp",
        "make_mlp_reward",
        "make_mlp_prior",
        "sample_mlp_context",
    ],
)

# ---------------------------------------------------------------------------
# Mixture prior p(eta) = 0.33 goal / 0.33 linear / 0.33 MLP
# ---------------------------------------------------------------------------
_try_import(
    "fre.rewards.prior",
    [
        "DEFAULT_FAMILIES",
        "FAMILY_MIXTURE_RATIOS",
        "CONTEXT_SIZE",
        "DECODER_SIZE",
        "DEFAULT_BATCH_SIZE",
        "ABLATION_FAMILIES",
        "ABLATION_NAMES",
        "MIXTURE_FAMILY",
        "FAMILY_GOAL",
        "FAMILY_LINEAR",
        "FAMILY_MLP",
        "MixturePrior",
        "FREPrior",
        "RewardPrior",
        "make_mixture_prior",
        "make_fre_prior",
        "make_prior",
        "ablation_prior",
        "hint_prior",
    ],
)

# ---------------------------------------------------------------------------
# Ground-truth evaluation rewards (AntMaze / ExORL / Kitchen) + scoring
# ---------------------------------------------------------------------------
_try_import(
    "fre.rewards.eval_rewards",
    [
        "ANTMAZE_DATASET",
        "ANTMAZE_OBS_DIM",
        "ANTMAZE_XY_DIMS",
        "ANTMAZE_VEL_XY_DIMS",
        "ANTMAZE_MAX_STEPS",
        "ANTMAZE_XY_BINS",
        "ANTMAZE_XY_EXTENT",
        "ANTMAZE_GOAL_THRESHOLD",
        "ANTMAZE_GOAL_TASKS",
        "ANTMAZE_DIRECTIONAL_TASKS",
        "ANTMAZE_SIMPLEX_SEEDS",
        "ANTMAZE_PATH_TASKS",
        "ANTMAZE_PATH_ROUTES",
        "EXORL_DOMAINS",
        "EXORL_MAX_STEPS",
        "EXORL_GOAL_THRESHOLD",
        "EXORL_NUM_GOAL_STATES",
        "EXORL_VELOCITY_THRESHOLDS",
        "EXORL_PHYSICS_DIMS",
        "EXORL_RAW_OBS_DIM",
        "KITCHEN_DATASET",
        "KITCHEN_OBS_DIM",
        "KITCHEN_NUM_SUBTASKS",
        "KITCHEN_TASKS",
        "KITCHEN_FLAG_OFFSET",
        "SCORE_SUCCESS_RATE",
        "SCORE_MEAN_REWARD",
        "SCORE_NORMALIZED_RETURN",
        "TASK_GROUPS",
        "AntMazeGoalReward",
        "AntMazeDirectionalReward",
        "AntMazeSimplexReward",
        "AntMazePathReward",
        "ExORLGoalReward",
        "ExORLVelocityReward",
        "KitchenSubtaskReward",
        "TaskScoring",
        "EvalTask",
        "antmaze_eval_tasks",
        "exorl_eval_tasks",
        "kitchen_eval_tasks",
        "all_eval_tasks",
        "task_groups",
        "tasks_for_group",
        "normalized_return",
        "success_rate_score",
        "mean_reward_score",
        "score_episodes",
        "aggregate_scores",
        "antmaze_xy",
        "antmaze_velocity",
        "discretize_xy",
        "bin_to_xy",
        "bin_distance",
        "polylines",
    ],
)

# Convenience aliases if the sub-modules did not define them.
if "CONTEXT_SIZE" not in globals():
    CONTEXT_SIZE = 32
    __all__.append("CONTEXT_SIZE")
if "DECODER_SIZE" not in globals():
    DECODER_SIZE = 8
    __all__.append("DECODER_SIZE")


def __getattr__(name: str):  # pragma: no cover - informative error
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available reward exports include: {', '.join(sorted(__all__)[:20])}..."
    )
