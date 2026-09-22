"""Baseline agents for the FRE benchmark.

This package aggregates the comparison methods used in the paper's Table 1:

* :mod:`fre.baselines.gc_iql`      -- Goal-conditioned IQL (GC-IQL).
* :mod:`fre.baselines.gc_bc`       -- Goal-conditioned behavioral cloning (GC-BC).
* :mod:`fre.baselines.opal`        -- OPAL skill discovery re-implemented with the
  FRE transformer, evaluated under the privileged protocol.
* :mod:`fre.baselines.fb_sf_runner` -- thin orchestration layer around the external
  ``facebookresearch/controllable_agent`` codebase for the FB / SF baselines.

Heavy optional dependencies (``torch``, ``d4rl``, ``gym``, ``dm_control``) are
imported lazily inside the submodules so that ``import fre.baselines`` stays
cheap and the pure-data utilities remain importable in minimal environments.

The public API mirrors the pattern used by the other FRE packages: a curated set
of eager re-exports plus PEP 562 ``__getattr__`` lazy submodule resolution.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Tuple

# ---------------------------------------------------------------------------
# Eager re-exports (these modules are dependency-light in their import guards).
# ---------------------------------------------------------------------------
from fre.baselines.gc_iql import (  # noqa: F401
    DEFAULT_GC_BATCH_SIZE,
    DEFAULT_GC_DISCOUNT,
    DEFAULT_GC_HIDDEN_DIMS,
    DEFAULT_GC_LEARNING_RATE,
    DEFAULT_GC_TRAIN_STEPS,
    DEFAULT_P_CURRENT_GOAL,
    DEFAULT_P_FUTURE_GOAL,
    DEFAULT_P_RANDOM_GOAL,
    DEFAULT_GEOMETRIC_P,
    GCIQLAgent,
    GCIQLConfig,
    GoalConditionedAgent,
    GoalSampler,
    build_gc_iql,
    evaluate_gc_agent,
    evaluate_gc_iql,
    gc_iql_action_fn,
    goal_conditioned_reward,
    goal_from_task,
    sample_gc_batch,
    sample_goal_indices,
    sample_hindsight_goals,
    train_gc_iql,
)
from fre.baselines.gc_bc import (  # noqa: F401
    DEFAULT_GCBC_BATCH_SIZE,
    DEFAULT_GCBC_GEOMETRIC_P,
    DEFAULT_GCBC_HIDDEN_DIMS,
    DEFAULT_GCBC_LEARNING_RATE,
    DEFAULT_GCBC_LOG_STD_MIN,
    DEFAULT_GCBC_TRAIN_STEPS,
    GCBCAgent,
    GCBCConfig,
    GaussianBCPolicy,
    bc_action_fn,
    build_gc_bc,
    compose_inputs,
    evaluate_gc_bc,
    evaluate_gc_bc_agent,
    gcbc_action_fn,
    sample_gcbc_batch,
    sample_geometric_goal_indices,
    sample_geometric_goals,
    train_gc_bc,
)
from fre.baselines.opal import (  # noqa: F401
    DEFAULT_OPAL_BATCH_SIZE,
    DEFAULT_OPAL_HIDDEN_DIMS,
    DEFAULT_OPAL_LEARNING_RATE,
    DEFAULT_OPAL_NUM_EPISODES,
    DEFAULT_OPAL_NUM_SEEDS,
    DEFAULT_OPAL_NUM_SKILLS,
    DEFAULT_OPAL_SEGMENT_LENGTH,
    DEFAULT_OPAL_SKILL_DIM,
    DEFAULT_OPAL_TRAIN_STEPS,
    OPALAgent,
    OPALConfig,
    OPALEncoder,
    SkillPolicy,
    TrajectoryDecoder,
    build_opal,
    evaluate_opal,
    evaluate_opal_privileged,
    info_nce_loss,
    opal_action_fn,
    reconstruction_loss,
    sample_trajectory_segments,
    skill_action_fn,
    train_opal,
    trajectory_iae_loss,
)
from fre.baselines.fb_sf_runner import (  # noqa: F401
    CONTROLLABLE_AGENT_REPO,
    CONTROLLABLE_AGENT_URL,
    DATASET_IDS,
    FBSF_DEFAULT_REWARD_SAMPLES,
    FBSF_DEFAULT_SF_FEATURES,
    FBSF_METHODS,
    FBSFConfig,
    FBSFRunner,
    build_commands,
    build_eval_command,
    build_replay_command,
    build_train_command,
    collect_results,
    controllable_agent_python,
    dataset_id_for,
    export_reward_samples,
    export_task_reward_samples,
    find_controllable_agent,
    install_custom_reward,
    make_custom_reward_fn,
    reward_sample_array,
)

#: Baseline methods reported in Table 1 (in-house reproductions).
IN_HOUSE_METHODS: Tuple[str, ...] = ("gc_iql", "gc_bc", "opal")

#: Baseline methods run through the external ``controllable_agent`` repo.
EXTERNAL_METHODS: Tuple[str, ...] = ("fb", "sf")

#: All baseline method keys handled by this package.
ALL_METHODS: Tuple[str, ...] = IN_HOUSE_METHODS + EXTERNAL_METHODS

#: Mapping from method key to the module that exposes ``main(argv)`` for the CLI.
METHOD_MODULES = {
    "gc_iql": "fre.baselines.gc_iql",
    "gc_bc": "fre.baselines.gc_bc",
    "opal": "fre.baselines.opal",
    "fb": "fre.baselines.fb_sf_runner",
    "sf": "fre.baselines.fb_sf_runner",
}

_LAZY_SUBMODULES: Tuple[str, ...] = (
    "gc_iql",
    "gc_bc",
    "opal",
    "fb_sf_runner",
)


def resolve_method(method: str) -> str:
    """Return the module path that implements ``method`` (``main(argv)`` CLI)."""
    key = str(method).strip().lower().replace("-", "_")
    if key in METHOD_MODULES:
        return METHOD_MODULES[key]
    return "fre.baselines." + key


def run_method(method: str, argv: Any = None) -> int:
    """Import and dispatch to one baseline's ``main(argv)`` entry point.

    Args:
        method: Baseline key (e.g. ``"gc_iql"``, ``"gc-bc"``, ``"fb"``).
        argv: Optional argument vector forwarded to the module ``main``.

    Returns:
        The integer exit code returned by the dispatched ``main`` (0 if the
        module does not define one).
    """
    module = import_module(resolve_method(method))
    main = getattr(module, "main", None)
    if main is None:
        return 0
    return int(main(argv) or 0)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy submodule resolver (caches the imported module)."""
    if name in _LAZY_SUBMODULES:
        module = import_module(f"fre.baselines.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Tuple[str, ...]:
    return tuple(sorted(set(list(globals()) + list(_LAZY_SUBMODULES))))


__all__ = [
    # method registries / dispatch helpers
    "IN_HOUSE_METHODS",
    "EXTERNAL_METHODS",
    "ALL_METHODS",
    "METHOD_MODULES",
    "resolve_method",
    "run_method",
    # GC-IQL
    "GCIQLAgent",
    "GoalConditionedAgent",
    "GCIQLConfig",
    "GoalSampler",
    "build_gc_iql",
    "train_gc_iql",
    "evaluate_gc_agent",
    "evaluate_gc_iql",
    "gc_iql_action_fn",
    "sample_gc_batch",
    "sample_goal_indices",
    "sample_hindsight_goals",
    "goal_conditioned_reward",
    "goal_from_task",
    "DEFAULT_P_CURRENT_GOAL",
    "DEFAULT_P_FUTURE_GOAL",
    "DEFAULT_P_RANDOM_GOAL",
    "DEFAULT_GEOMETRIC_P",
    "DEFAULT_GC_HIDDEN_DIMS",
    "DEFAULT_GC_LEARNING_RATE",
    "DEFAULT_GC_DISCOUNT",
    "DEFAULT_GC_BATCH_SIZE",
    "DEFAULT_GC_TRAIN_STEPS",
    # GC-BC
    "GCBCAgent",
    "GCBCConfig",
    "GaussianBCPolicy",
    "build_gc_bc",
    "train_gc_bc",
    "evaluate_gc_bc",
    "evaluate_gc_bc_agent",
    "bc_action_fn",
    "gcbc_action_fn",
    "sample_gcbc_batch",
    "sample_geometric_goal_indices",
    "sample_geometric_goals",
    "compose_inputs",
    "DEFAULT_GCBC_HIDDEN_DIMS",
    "DEFAULT_GCBC_LOG_STD_MIN",
    "DEFAULT_GCBC_LEARNING_RATE",
    "DEFAULT_GCBC_BATCH_SIZE",
    "DEFAULT_GCBC_TRAIN_STEPS",
    "DEFAULT_GCBC_GEOMETRIC_P",
    # OPAL
    "OPALAgent",
    "OPALConfig",
    "OPALEncoder",
    "SkillPolicy",
    "TrajectoryDecoder",
    "build_opal",
    "train_opal",
    "evaluate_opal",
    "evaluate_opal_privileged",
    "skill_action_fn",
    "opal_action_fn",
    "info_nce_loss",
    "trajectory_iae_loss",
    "reconstruction_loss",
    "sample_trajectory_segments",
    "DEFAULT_OPAL_SKILL_DIM",
    "DEFAULT_OPAL_HIDDEN_DIMS",
    "DEFAULT_OPAL_LEARNING_RATE",
    "DEFAULT_OPAL_BATCH_SIZE",
    "DEFAULT_OPAL_SEGMENT_LENGTH",
    "DEFAULT_OPAL_TRAIN_STEPS",
    "DEFAULT_OPAL_NUM_SKILLS",
    "DEFAULT_OPAL_NUM_EPISODES",
    "DEFAULT_OPAL_NUM_SEEDS",
    # FB / SF (external controllable_agent runner)
    "FBSFConfig",
    "FBSFRunner",
    "FBSF_METHODS",
    "FBSF_DEFAULT_REWARD_SAMPLES",
    "FBSF_DEFAULT_SF_FEATURES",
    "CONTROLLABLE_AGENT_REPO",
    "CONTROLLABLE_AGENT_URL",
    "DATASET_IDS",
    "find_controllable_agent",
    "controllable_agent_python",
    "dataset_id_for",
    "build_replay_command",
    "build_train_command",
    "build_eval_command",
    "build_commands",
    "make_custom_reward_fn",
    "install_custom_reward",
    "reward_sample_array",
    "export_reward_samples",
    "export_task_reward_samples",
    "collect_results",
    # lazy submodules
    "gc_iql",
    "gc_bc",
    "opal",
    "fb_sf_runner",
]
