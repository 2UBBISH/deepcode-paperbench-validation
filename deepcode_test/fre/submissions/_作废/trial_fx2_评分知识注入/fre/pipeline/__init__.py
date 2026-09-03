"""Pipeline package for FRE training, evaluation, baselines, and visualization.

This module re-exports the public entry points from each pipeline component so
callers can use ``from fre.pipeline import pretrain_encoder`` style imports
while keeping the individual submodules as the source of truth.
"""

from __future__ import annotations

# Pretraining (Phase 1)
from fre.pipeline.pretrain_encoder import (
    build_parser as pretrain_build_parser,
    main as pretrain_main,
    pretrain_encoder,
)

# FRE-conditioned offline RL training (Phase 2)
from fre.pipeline.train_agent import (
    build_parser as train_build_parser,
    main as train_main,
    train_agent,
)

# Zero-shot downstream FRE evaluation
from fre.pipeline.evaluate import (
    TaskReward,
    encode_task_latent,
    evaluate_all_tasks,
    evaluate_task,
    make_eval_env,
    make_task_reward,
    rollout_task,
)

# Unified baseline evaluation
from fre.pipeline.evaluate_baselines import (
    evaluate_all_baselines,
    evaluate_baseline,
    evaluate_fb_agent,
    evaluate_gc_agent,
    evaluate_opal_agent,
    evaluate_regression_agent,
    evaluate_sf_agent,
)

# Visualization helpers
from fre.pipeline.visualize import (
    plot_bar_comparison,
    plot_grouped_bar_comparison,
    visualize_antmaze_task,
)

__all__ = [
    # Pretrain
    "pretrain_encoder",
    "pretrain_main",
    "pretrain_build_parser",
    # Train
    "train_agent",
    "train_main",
    "train_build_parser",
    # FRE evaluation
    "TaskReward",
    "encode_task_latent",
    "evaluate_all_tasks",
    "evaluate_task",
    "make_eval_env",
    "make_task_reward",
    "rollout_task",
    # Baselines
    "evaluate_all_baselines",
    "evaluate_baseline",
    "evaluate_fb_agent",
    "evaluate_gc_agent",
    "evaluate_opal_agent",
    "evaluate_regression_agent",
    "evaluate_sf_agent",
    # Visualization
    "plot_bar_comparison",
    "plot_grouped_bar_comparison",
    "visualize_antmaze_task",
]
