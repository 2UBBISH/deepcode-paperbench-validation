"""Environment and evaluation-task package for the FRE reproduction.

This package bundles everything that touches the *environment* side of the FRE
pipeline described in the paper (Section 5 and Appendix C):

* :mod:`fre.envs.d4rl_loader` -- offline dataset loading (AntMaze
  ``antmaze-large-diverse-v2``, ExORL RND walker/cheetah, D4RL Kitchen) plus the
  ExORL physics-augmentation helpers from Appendix C.2.
* :mod:`fre.envs.antmaze_eval` -- AntMaze zero-shot evaluation suite:
  goal-reaching, directional, random-simplex and path tasks (Appendix C.1).
* :mod:`fre.envs.exorl_eval` -- ExORL velocity + goal-reaching tasks on
  walker/cheetah (Appendix C.2).
* :mod:`fre.envs.kitchen_eval` -- the 7 standard D4RL Kitchen subtasks with
  native sparse rewards (Appendix C.3).

The module intentionally uses PEP 562 lazy attribute resolution (``__getattr__``)
so that ``import fre.envs`` does not pull in ``gym`` / ``d4rl`` / ``mujoco`` /
``torch``.  Only the submodule that is actually needed gets imported, which lets
reporting / dry-run code (e.g. ``python fre/main.py --dry-run``) run in a bare
environment.

Where two submodules define a symbol with the same name (``encode_task_latent``
and ``sample_task_encoder_pairs`` exist in both :mod:`fre.envs.exorl_eval` and
:mod:`fre.envs.kitchen_eval`), we expose the ExORL variant under the plain name
and the Kitchen variant under an explicit ``kitchen_``-prefixed alias.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover - static typing only
    # Freely-typed imports are delegated to the lazy table at runtime.
    ...

__all__ = [
    # ---- d4rl_loader -----------------------------------------------------
    "load_offline_dataset",
    "load_antmaze",
    "load_exorl",
    "load_kitchen",
    "make_synthetic_dataset",
    "load_local_dataset_file",
    "resolve_dataset_path",
    "available_local_datasets",
    "walker_physics",
    "cheetah_physics",
    "compute_physics",
    "append_physics",
    "physics_dim",
    "state_dim_for",
    "action_dim_for",
    "dataset_std",
    "normalized_goal_distance",
    "D4RL_ENV_IDS",
    "EXORL_DATASETS",
    "PHYSICS_FIELDS",
    # ---- antmaze_eval ----------------------------------------------------
    "AntMazeEvalWrapper",
    "AntMazeTask",
    "GoalReachingTask",
    "DirectionalTask",
    "RandomSimplexTask",
    "PathTask",
    "EpisodeResult",
    "make_antmaze_env",
    "make_antmaze_task_suite",
    "make_goal_reaching_tasks",
    "make_directional_tasks",
    "make_random_simplex_tasks",
    "make_path_tasks",
    "get_task_suite",
    "rollout_episode",
    "evaluate_task",
    "evaluate_suite",
    "evaluate_antmaze_suite",
    "make_iql_policy_fn",
    "discretize_antmaze_observations",
    "antmaze_discretize_xy",
    "extract_antmaze_xy",
    "extract_antmaze_velocity",
    "ANTMAZE_GOAL_LOCATIONS",
    "ANTMAZE_DIRECTION_VECTORS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_PATH_KINDS",
    "ANTMAZE_GOAL_THRESHOLD",
    "ANTMAZE_DISCRETIZE_BINS",
    "ANTMAZE_MAX_EPISODE_STEPS",
    "ANTMAZE_ENCODER_SAMPLES",
    # ---- exorl_eval ------------------------------------------------------
    "ExoRLEvalWrapper",
    "ExoRLTask",
    "VelocityTask",
    "ExoRLEpisodeResult",
    "SyntheticExoRLEnv",
    "make_exorl_env",
    "make_exorl_task_suite",
    "make_velocity_tasks",
    "make_goal_tasks",
    "evaluate_exorl_suite",
    "make_exorl_policy_fn",
    "encode_task_latent",
    "sample_task_encoder_pairs",
    "physics_features",
    "augment_observations",
    "select_goal_states",
    "EXORL_DOMAINS",
    "EXORL_MAX_EPISODE_STEPS",
    "EXORL_ENCODER_SAMPLES",
    "EXORL_GOAL_THRESHOLD",
    "EXORL_NUM_GOALS",
    "EXORL_WALKER_VELOCITY_THRESHOLDS",
    "EXORL_CHEETAH_RUN_THRESHOLD",
    "EXORL_CHEETAH_WALK_THRESHOLD",
    "EXORL_WALKER_PHYSICS",
    "EXORL_CHEETAH_PHYSICS",
    "EXORL_TABLE1_REFERENCE",
    # ---- kitchen_eval ----------------------------------------------------
    "KitchenEvalWrapper",
    "KitchenTask",
    "SubtaskTask",
    "CombinedKitchenTask",
    "KitchenEpisodeResult",
    "SyntheticKitchenEnv",
    "make_kitchen_env",
    "make_subtask_tasks",
    "make_kitchen_task_suite",
    "evaluate_kitchen_suite",
    "make_kitchen_policy_fn",
    "kitchen_encode_task_latent",
    "kitchen_sample_task_encoder_pairs",
    "subtask_flag",
    "KITCHEN_ENV_ID",
    "KITCHEN_MAX_EPISODE_STEPS",
    "KITCHEN_ENCODER_SAMPLES",
    "KITCHEN_SUBTASKS",
    "KITCHEN_SUBTASK_NAMES",
    "KITCHEN_TABLE1_REFERENCE",
]


# Mapping public symbol -> (submodule, attribute).  Names that are ambiguous
# across submodules are aliased explicitly below.
_LAZY_ATTRS: Dict[str, Any] = {
    # ---- d4rl_loader -----------------------------------------------------
    "load_offline_dataset": ("d4rl_loader", "load_offline_dataset"),
    "load_antmaze": ("d4rl_loader", "load_antmaze"),
    "load_exorl": ("d4rl_loader", "load_exorl"),
    "load_kitchen": ("d4rl_loader", "load_kitchen"),
    "make_synthetic_dataset": ("d4rl_loader", "make_synthetic_dataset"),
    "load_local_dataset_file": ("d4rl_loader", "load_local_dataset_file"),
    "resolve_dataset_path": ("d4rl_loader", "resolve_dataset_path"),
    "available_local_datasets": ("d4rl_loader", "available_local_datasets"),
    "walker_physics": ("d4rl_loader", "walker_physics"),
    "cheetah_physics": ("d4rl_loader", "cheetah_physics"),
    "compute_physics": ("d4rl_loader", "compute_physics"),
    "append_physics": ("d4rl_loader", "append_physics"),
    "physics_dim": ("d4rl_loader", "physics_dim"),
    "state_dim_for": ("d4rl_loader", "state_dim_for"),
    "action_dim_for": ("d4rl_loader", "action_dim_for"),
    "dataset_std": ("d4rl_loader", "dataset_std"),
    "normalized_goal_distance": ("d4rl_loader", "normalized_goal_distance"),
    "D4RL_ENV_IDS": ("d4rl_loader", "D4RL_ENV_IDS"),
    "EXORL_DATASETS": ("d4rl_loader", "EXORL_DATASETS"),
    "PHYSICS_FIELDS": ("d4rl_loader", "PHYSICS_FIELDS"),
    # ---- antmaze_eval ----------------------------------------------------
    "AntMazeEvalWrapper": ("antmaze_eval", "AntMazeEvalWrapper"),
    "AntMazeTask": ("antmaze_eval", "AntMazeTask"),
    "GoalReachingTask": ("antmaze_eval", "GoalReachingTask"),
    "DirectionalTask": ("antmaze_eval", "DirectionalTask"),
    "RandomSimplexTask": ("antmaze_eval", "RandomSimplexTask"),
    "PathTask": ("antmaze_eval", "PathTask"),
    "EpisodeResult": ("antmaze_eval", "EpisodeResult"),
    "make_antmaze_env": ("antmaze_eval", "make_antmaze_env"),
    "make_antmaze_task_suite": ("antmaze_eval", "make_antmaze_task_suite"),
    "make_goal_reaching_tasks": ("antmaze_eval", "make_goal_reaching_tasks"),
    "make_directional_tasks": ("antmaze_eval", "make_directional_tasks"),
    "make_random_simplex_tasks": ("antmaze_eval", "make_random_simplex_tasks"),
    "make_path_tasks": ("antmaze_eval", "make_path_tasks"),
    "get_task_suite": ("antmaze_eval", "get_task_suite"),
    "rollout_episode": ("antmaze_eval", "rollout_episode"),
    "evaluate_task": ("antmaze_eval", "evaluate_task"),
    "evaluate_suite": ("antmaze_eval", "evaluate_suite"),
    "evaluate_antmaze_suite": ("antmaze_eval", "evaluate_antmaze_suite"),
    "make_iql_policy_fn": ("antmaze_eval", "make_iql_policy_fn"),
    "discretize_antmaze_observations": ("antmaze_eval", "discretize_antmaze_observations"),
    "antmaze_discretize_xy": ("antmaze_eval", "antmaze_discretize_xy"),
    "extract_antmaze_xy": ("antmaze_eval", "extract_antmaze_xy"),
    "extract_antmaze_velocity": ("antmaze_eval", "extract_antmaze_velocity"),
    "ANTMAZE_GOAL_LOCATIONS": ("antmaze_eval", "ANTMAZE_GOAL_LOCATIONS"),
    "ANTMAZE_DIRECTION_VECTORS": ("antmaze_eval", "ANTMAZE_DIRECTION_VECTORS"),
    "ANTMAZE_SIMPLEX_SEEDS": ("antmaze_eval", "ANTMAZE_SIMPLEX_SEEDS"),
    "ANTMAZE_PATH_KINDS": ("antmaze_eval", "ANTMAZE_PATH_KINDS"),
    "ANTMAZE_GOAL_THRESHOLD": ("antmaze_eval", "ANTMAZE_GOAL_THRESHOLD"),
    "ANTMAZE_DISCRETIZE_BINS": ("antmaze_eval", "ANTMAZE_DISCRETIZE_BINS"),
    "ANTMAZE_MAX_EPISODE_STEPS": ("antmaze_eval", "ANTMAZE_MAX_EPISODE_STEPS"),
    "ANTMAZE_ENCODER_SAMPLES": ("antmaze_eval", "ANTMAZE_ENCODER_SAMPLES"),
    # ---- exorl_eval ------------------------------------------------------
    "ExoRLEvalWrapper": ("exorl_eval", "ExoRLEvalWrapper"),
    "ExoRLTask": ("exorl_eval", "ExoRLTask"),
    "VelocityTask": ("exorl_eval", "VelocityTask"),
    "ExoRLEpisodeResult": ("exorl_eval", "ExoRLEpisodeResult"),
    "SyntheticExoRLEnv": ("exorl_eval", "SyntheticExoRLEnv"),
    "make_exorl_env": ("exorl_eval", "make_exorl_env"),
    "make_exorl_task_suite": ("exorl_eval", "make_exorl_task_suite"),
    "make_velocity_tasks": ("exorl_eval", "make_velocity_tasks"),
    "make_goal_tasks": ("exorl_eval", "make_goal_tasks"),
    "evaluate_exorl_suite": ("exorl_eval", "evaluate_exorl_suite"),
    "make_exorl_policy_fn": ("exorl_eval", "make_exorl_policy_fn"),
    "encode_task_latent": ("exorl_eval", "encode_task_latent"),
    "sample_task_encoder_pairs": ("exorl_eval", "sample_task_encoder_pairs"),
    "physics_features": ("exorl_eval", "physics_features"),
    "augment_observations": ("exorl_eval", "augment_observations"),
    "select_goal_states": ("exorl_eval", "select_goal_states"),
    "EXORL_DOMAINS": ("exorl_eval", "EXORL_DOMAINS"),
    "EXORL_MAX_EPISODE_STEPS": ("exorl_eval", "EXORL_MAX_EPISODE_STEPS"),
    "EXORL_ENCODER_SAMPLES": ("exorl_eval", "EXORL_ENCODER_SAMPLES"),
    "EXORL_GOAL_THRESHOLD": ("exorl_eval", "EXORL_GOAL_THRESHOLD"),
    "EXORL_NUM_GOALS": ("exorl_eval", "EXORL_NUM_GOALS"),
    "EXORL_WALKER_VELOCITY_THRESHOLDS": ("exorl_eval", "EXORL_WALKER_VELOCITY_THRESHOLDS"),
    "EXORL_CHEETAH_RUN_THRESHOLD": ("exorl_eval", "EXORL_CHEETAH_RUN_THRESHOLD"),
    "EXORL_CHEETAH_WALK_THRESHOLD": ("exorl_eval", "EXORL_CHEETAH_WALK_THRESHOLD"),
    "EXORL_WALKER_PHYSICS": ("exorl_eval", "EXORL_WALKER_PHYSICS"),
    "EXORL_CHEETAH_PHYSICS": ("exorl_eval", "EXORL_CHEETAH_PHYSICS"),
    "EXORL_TABLE1_REFERENCE": ("exorl_eval", "EXORL_TABLE1_REFERENCE"),
    # ---- kitchen_eval ----------------------------------------------------
    "KitchenEvalWrapper": ("kitchen_eval", "KitchenEvalWrapper"),
    "KitchenTask": ("kitchen_eval", "KitchenTask"),
    "SubtaskTask": ("kitchen_eval", "SubtaskTask"),
    "CombinedKitchenTask": ("kitchen_eval", "CombinedKitchenTask"),
    "KitchenEpisodeResult": ("kitchen_eval", "KitchenEpisodeResult"),
    "SyntheticKitchenEnv": ("kitchen_eval", "SyntheticKitchenEnv"),
    "make_kitchen_env": ("kitchen_eval", "make_kitchen_env"),
    "make_subtask_tasks": ("kitchen_eval", "make_subtask_tasks"),
    "make_kitchen_task_suite": ("kitchen_eval", "make_kitchen_task_suite"),
    "evaluate_kitchen_suite": ("kitchen_eval", "evaluate_kitchen_suite"),
    "make_kitchen_policy_fn": ("kitchen_eval", "make_kitchen_policy_fn"),
    # Ambiguous helpers get explicit, disambiguated aliases.
    "kitchen_encode_task_latent": ("kitchen_eval", "encode_task_latent"),
    "kitchen_sample_task_encoder_pairs": ("kitchen_eval", "sample_task_encoder_pairs"),
    "subtask_flag": ("kitchen_eval", "subtask_flag"),
    "KITCHEN_ENV_ID": ("kitchen_eval", "KITCHEN_ENV_ID"),
    "KITCHEN_MAX_EPISODE_STEPS": ("kitchen_eval", "KITCHEN_MAX_EPISODE_STEPS"),
    "KITCHEN_ENCODER_SAMPLES": ("kitchen_eval", "KITCHEN_ENCODER_SAMPLES"),
    "KITCHEN_SUBTASKS": ("kitchen_eval", "KITCHEN_SUBTASKS"),
    "KITCHEN_SUBTASK_NAMES": ("kitchen_eval", "KITCHEN_SUBTASK_NAMES"),
    "KITCHEN_TABLE1_REFERENCE": ("kitchen_eval", "KITCHEN_TABLE1_REFERENCE"),
}

# Submodules that can be accessed attribute-style, e.g. ``fre.envs.antmaze_eval``.
_SUBMODULES = ("d4rl_loader", "antmaze_eval", "exorl_eval", "kitchen_eval")


def __getattr__(name: str) -> Any:
    """Resolve a public symbol lazily from its defining submodule (PEP 562)."""
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    submodule, attribute = target
    module = importlib.import_module(f"{__name__}.{submodule}")
    try:
        value = getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - guards index drift
        raise AttributeError(
            f"module {__name__}.{submodule} has no attribute {attribute!r} "
            f"(requested as {name!r})"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
