"""
Evaluation module for the FRE (Functional Reward Encodings) framework.

Provides:
- FREEvaluator: Zero-shot evaluator for trained FRE agents
- EvaluationTask, EvaluationResult: Task definitions and result aggregation
- Domain-specific task builders (AntMaze, ExORL, Kitchen)
- Normalization utilities for computing paper-style metrics
"""

from .metrics import (
    normalize_returns,
    compute_normalized_score,
    get_domain_normalization,
    make_antmaze_goal_reaching_reward,
    make_antmaze_directional_reward,
    make_antmaze_random_simplex_reward,
    make_antmaze_path_reward,
    make_exorl_goal_reaching_reward,
    make_exorl_velocity_reward,
    make_kitchen_subtask_reward,
    compute_episode_return,
    aggregate_seed_results,
    EvaluationTask,
    EvaluationResult,
    DOMAIN_NORMALIZATION,
)

from .evaluator import (
    FREEvaluator,
    build_antmaze_tasks,
    build_exorl_walker_tasks,
    build_exorl_cheetah_tasks,
    build_kitchen_tasks,
    build_tasks_for_domain,
    run_multi_seed_evaluation,
    evaluate_from_trainer,
)

__all__ = [
    # Metrics
    "normalize_returns",
    "compute_normalized_score",
    "get_domain_normalization",
    "make_antmaze_goal_reaching_reward",
    "make_antmaze_directional_reward",
    "make_antmaze_random_simplex_reward",
    "make_antmaze_path_reward",
    "make_exorl_goal_reaching_reward",
    "make_exorl_velocity_reward",
    "make_kitchen_subtask_reward",
    "compute_episode_return",
    "aggregate_seed_results",
    "EvaluationTask",
    "EvaluationResult",
    "DOMAIN_NORMALIZATION",
    # Evaluator
    "FREEvaluator",
    "build_antmaze_tasks",
    "build_exorl_walker_tasks",
    "build_exorl_cheetah_tasks",
    "build_kitchen_tasks",
    "build_tasks_for_domain",
    "run_multi_seed_evaluation",
    "evaluate_from_trainer",
]