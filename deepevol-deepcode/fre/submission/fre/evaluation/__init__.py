"""FRE evaluation package.

Aggregates the zero-shot evaluation harness (``fre.evaluation.evaluate``) and the
return-processing / aggregation pipeline (``fre.evaluation.metrics``) behind a
single import surface.

The evaluation harness implements Section 5.2 / Appendix C of the paper:

* encode ``K = 32`` reward-annotated ``(s, eta(s))`` samples of a test task into a
  128-d latent ``z`` (posterior mean),
* zero-shot rollout of the ``z``-conditioned policy (no training),
* ``20`` episodes per seed, ``5`` seeds, returns normalised to ``[0, 100]`` with
  the standard deviation reported across seeds.

Heavy third-party dependencies (torch, gym, d4rl, dm_control) are imported lazily
inside :mod:`fre.evaluation.evaluate`, so ``import fre.evaluation`` stays cheap.
"""

from __future__ import annotations

from typing import Any, Tuple

# ---------------------------------------------------------------------------
# metrics.py -- dependency light (numpy only) return processing / aggregation
# ---------------------------------------------------------------------------
from fre.evaluation.metrics import (  # noqa: F401
    EpisodeResult,
    SeedResult,
    TaskResult,
    NUM_EVAL_EPISODES,
    NUM_SEEDS,
    NORMALIZED_MIN,
    NORMALIZED_MAX,
    SUCCESS_THRESHOLD,
    aggregate_episodes,
    compute_episode_metrics,
    normalize_return,
    normalize_returns,
    denormalize_returns,
    d4rl_normalize,
    aggregate_seeds,
    aggregate_tasks,
    relative_normalize,
    aggregate_task_sets,
    compute_scores,
    build_score_table,
    format_metric,
    format_table,
    rewards_to_returns,
    first_success_step,
    episode_success,
)

# ---------------------------------------------------------------------------
# evaluate.py -- zero-shot eval harness (torch/gym imported lazily inside)
# ---------------------------------------------------------------------------
from fre.evaluation.evaluate import (  # noqa: F401
    EvalConfig,
    RolloutResult,
    FRE_NUM_ENCODING_SAMPLES,
    FRE_NUM_EVAL_EPISODES,
    FRE_NUM_EVAL_SEEDS,
    FRE_MAX_EPISODE_STEPS,
    ENCODER_REWARD_MIN,
    ENCODER_REWARD_MAX,
    resolve_domain,
    build_evaluation_tasks,
    task_names_for,
    dataset_states_array,
    encoder_states_for_domain,
    task_reward_values,
    select_encoding_samples,
    encoding_samples_from_dataset,
    encode_task_latent,
    make_action_fn,
    rollout_episode,
    resolve_return_range,
    normalize_episode_return,
    evaluate_task,
    evaluate_fre,
    evaluate_policy,
    summarize_results,
    format_result_table,
    save_results,
    relative_normalized_scores,
)

__all__ = [
    # ---- metrics: dataclasses ----
    "EpisodeResult",
    "SeedResult",
    "TaskResult",
    # ---- metrics: constants ----
    "NUM_EVAL_EPISODES",
    "NUM_SEEDS",
    "NORMALIZED_MIN",
    "NORMALIZED_MAX",
    "SUCCESS_THRESHOLD",
    # ---- metrics: aggregation / normalization ----
    "aggregate_episodes",
    "compute_episode_metrics",
    "normalize_return",
    "normalize_returns",
    "denormalize_returns",
    "d4rl_normalize",
    "aggregate_seeds",
    "aggregate_tasks",
    "relative_normalize",
    "aggregate_task_sets",
    "compute_scores",
    "build_score_table",
    "format_metric",
    "format_table",
    "rewards_to_returns",
    "first_success_step",
    "episode_success",
    # ---- evaluate: configuration / results ----
    "EvalConfig",
    "RolloutResult",
    # ---- evaluate: constants ----
    "FRE_NUM_ENCODING_SAMPLES",
    "FRE_NUM_EVAL_EPISODES",
    "FRE_NUM_EVAL_SEEDS",
    "FRE_MAX_EPISODE_STEPS",
    "ENCODER_REWARD_MIN",
    "ENCODER_REWARD_MAX",
    # ---- evaluate: task / sample preparation ----
    "resolve_domain",
    "build_evaluation_tasks",
    "task_names_for",
    "dataset_states_array",
    "encoder_states_for_domain",
    "task_reward_values",
    "select_encoding_samples",
    "encoding_samples_from_dataset",
    # ---- evaluate: encoding + rollouts ----
    "encode_task_latent",
    "make_action_fn",
    "rollout_episode",
    "resolve_return_range",
    "normalize_episode_return",
    # ---- evaluate: top level drivers ----
    "evaluate_task",
    "evaluate_fre",
    "evaluate_policy",
    "summarize_results",
    "format_result_table",
    "save_results",
    "relative_normalized_scores",
    # ---- submodules exposed lazily ----
    "evaluate",
    "metrics",
]

_LAZY_SUBMODULES: Tuple[str, ...] = ("evaluate", "metrics")


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy loader
    """Expose submodules lazily (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Tuple[str, ...]:  # pragma: no cover - introspection helper
    return tuple(sorted(set(list(globals().keys()) + list(_LAZY_SUBMODULES))))
