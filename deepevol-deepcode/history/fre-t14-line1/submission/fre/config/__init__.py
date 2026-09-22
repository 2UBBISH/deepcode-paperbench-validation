"""Configuration package for the FRE reproduction.

Exposes the central hyperparameter container (:class:`Config`) together with the
per-domain environment configurations / evaluation-task suites
(:mod:`fre.config.envs`).

Both submodules are import-light (no torch / gym at import time), so importing
this package is always safe.  ``fre.config.envs`` imports :class:`Config` lazily
inside :func:`fre.config.envs.make_config`, therefore the eager import below does
not create a circular dependency.
"""

from fre.config.default import Config
from fre.config.envs import (
    AGGREGATE_ROWS,
    ANTMAZE,
    ANTMAZE_DIRECTIONS,
    ANTMAZE_GOALS,
    ANTMAZE_PATHS,
    ANTMAZE_SIMPLEX_SEEDS,
    DOMAINS,
    EVAL_TASK_SUITES,
    EXORL_CHEETAH,
    EXORL_CHEETAH_PHYSICS,
    EXORL_CHEETAH_RUN_THRESHOLD,
    EXORL_CHEETAH_WALK_THRESHOLD,
    EXORL_GOAL_THRESHOLD,
    EXORL_NUM_GOALS,
    EXORL_WALKER,
    EXORL_WALKER_PHYSICS,
    EXORL_WALKER_VELOCITY_THRESHOLDS,
    KITCHEN,
    KITCHEN_SUBTASKS,
    TABLE1_FRE_PER_TASK,
    TABLE1_REFERENCE,
    DomainConfig,
    EvalTask,
    antmaze_discretize_xy,
    default_domain_overrides,
    domain_names,
    get_domain_config,
    get_env_config,
    get_eval_tasks,
    get_task,
    make_config,
)

__all__ = [
    # default.py
    "Config",
    # envs.py -- dataclasses
    "EvalTask",
    "DomainConfig",
    # envs.py -- lookups
    "domain_names",
    "get_domain_config",
    "get_env_config",
    "get_eval_tasks",
    "get_task",
    "default_domain_overrides",
    "make_config",
    "antmaze_discretize_xy",
    # envs.py -- constants
    "ANTMAZE_GOALS",
    "ANTMAZE_DIRECTIONS",
    "ANTMAZE_SIMPLEX_SEEDS",
    "ANTMAZE_PATHS",
    "EXORL_WALKER_VELOCITY_THRESHOLDS",
    "EXORL_CHEETAH_RUN_THRESHOLD",
    "EXORL_CHEETAH_WALK_THRESHOLD",
    "EXORL_GOAL_THRESHOLD",
    "EXORL_NUM_GOALS",
    "EXORL_WALKER_PHYSICS",
    "EXORL_CHEETAH_PHYSICS",
    "KITCHEN_SUBTASKS",
    "ANTMAZE",
    "EXORL_WALKER",
    "EXORL_CHEETAH",
    "KITCHEN",
    "DOMAINS",
    "EVAL_TASK_SUITES",
    "AGGREGATE_ROWS",
    "TABLE1_REFERENCE",
    "TABLE1_FRE_PER_TASK",
]
