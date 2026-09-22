"""Per-domain glue: dataset loading, prior construction, task suites, envs.

This module is the single place where the paper's domain-specific choices live
(Appendix C), so that the training / evaluation entry points stay generic.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from fre.configs import TrainConfig
from fre.datasets import OfflineDataset, load_d4rl_dataset, load_exorl_dataset
from fre.reward_functions import RewardPrior, build_prior
from fre.tasks.base import TaskSuite
from fre.tasks.antmaze import (
    ANTMAZE_OBS_DIM,
    make_antmaze_suites,
    xy_from_obs,
)
from fre.tasks.exorl import (
    AUX_DIMS,
    build_antmaze_hint_priors,
    build_exorl_hint_priors,
    make_exorl_suites,
    sample_fixed_goals,
)
from fre.tasks.kitchen import make_kitchen_suite


SUPPORTED_DOMAINS: Tuple[str, ...] = ("antmaze", "walker", "cheetah", "kitchen")


def default_config(domain: str, **overrides) -> TrainConfig:
    """Return the default :class:`TrainConfig` for a domain."""
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if domain in ("walker", "cheetah", "kitchen"):
        # Appendix A: ExORL and Kitchen use 1M encoder steps and 1M policy steps.
        overrides.setdefault("encoder_steps", 1_000_000)
        overrides.setdefault("policy_steps", 1_000_000)
    config = TrainConfig(domain=domain, **overrides)
    if domain == "antmaze":
        config.env_name = "antmaze-large-diverse-v2"
        # Appendix B: linear rewards exclude the (x, y) position dimensions on
        # AntMaze (their scale caused instability), and the goal-reaching prior
        # uses the maze position with the same distance threshold (2) as the
        # goal-reaching evaluation tasks.
        config.linear_excluded_dims = [0, 1]
        config.goal_dims = [0, 1]
        config.goal_threshold = 2.0
        config.discretize_xy = True
    elif domain in ("walker", "cheetah"):
        config.env_name = domain
        # ExORL goal-reaching uses the Euclidean distance in the normalised
        # observation space with a threshold of 0.1.
        config.goal_threshold = 0.1
        config.discretize_xy = False
    elif domain == "kitchen":
        config.env_name = "kitchen-complete-v0"
        config.discretize_xy = False
    else:
        raise ValueError(f"Unsupported domain '{domain}'. Options: {SUPPORTED_DOMAINS}")
    return config


def build_dataset(config: TrainConfig) -> OfflineDataset:
    """Load the offline dataset for a domain."""
    if config.domain == "antmaze":
        return load_d4rl_dataset("antmaze-large-diverse-v2")
    if config.domain in ("walker", "cheetah"):
        return load_exorl_dataset(
            config.domain,
            algo=config.exorl_algo,
            data_dir=config.exorl_data_dir,
        )
    if config.domain == "kitchen":
        return load_d4rl_dataset("kitchen-complete-v0")
    raise ValueError(f"Unsupported domain '{config.domain}'")


def build_domain_prior(config: TrainConfig, dataset: OfflineDataset) -> RewardPrior:
    """Construct the prior reward distribution for a domain (Sections 4.2-5.4)."""
    goal_sampler = dataset.goal_sampler(np.random.default_rng(config.seed))
    goal_scale = None
    if config.domain in ("walker", "cheetah"):
        # Goal distance is computed on the non-augmented observation space,
        # normalised by the standard deviation of each dimension.
        goal_scale = np.asarray(dataset.observations.std(axis=0), dtype=np.float32)
    hint_priors = None
    hint_ratios = None
    if config.use_hint_priors:
        if config.domain == "antmaze":
            hint_priors = build_antmaze_hint_priors()
        elif config.domain in ("walker", "cheetah"):
            hint_priors = build_exorl_hint_priors(config.domain)
        if hint_priors is not None:
            # ``FRE-hint`` mixes the domain-specific families (e.g. unit-direction
            # or specific-velocity rewards) together with the generic FRE-all
            # families, controlled by ``hint_ratio`` (Section 5.4 / Figure 6).
            hint_ratios = {
                **{name: config.hint_ratio / len(hint_priors) for name in hint_priors},
                "goal": (1.0 - config.hint_ratio) / 3.0,
                "lin": (1.0 - config.hint_ratio) / 3.0,
                "mlp": (1.0 - config.hint_ratio) / 3.0,
            }
    return build_prior(
        config.prior_name,
        dataset.encoder_obs_dim,
        goal_sampler,
        goal_threshold=config.goal_threshold,
        goal_dims=config.goal_dims,
        goal_scale=goal_scale,
        linear_excluded_dims=config.linear_excluded_dims,
        ratios=config.prior_ratios,
        hint_priors=hint_priors,
        hint_ratios=hint_ratios,
    )


def build_suites(
    config: TrainConfig,
    dataset: OfflineDataset,
    use_opensimplex: bool = True,
) -> List[TaskSuite]:
    """Build the evaluation task suites that make up Table 1."""
    if config.domain == "antmaze":
        return make_antmaze_suites(max_episode_steps=2000, use_opensimplex=use_opensimplex)
    if config.domain in ("walker", "cheetah"):
        goals = sample_fixed_goals(dataset, num_goals=5, seed=config.seed)
        obs_std = np.asarray(dataset.observations.std(axis=0), dtype=np.float32)
        return make_exorl_suites(
            config.domain, goals=goals, obs_std=obs_std, max_episode_steps=1000
        )
    if config.domain == "kitchen":
        return [make_kitchen_suite()]
    raise ValueError(f"Unsupported domain '{config.domain}'")


def make_env_factory(config: TrainConfig) -> Callable:
    """Return the environment factory used to roll out the trained agent."""
    from fre.envs import (
        make_antmaze_env_factory,
        make_exorl_env_factory,
        make_kitchen_env_factory,
    )

    if config.domain == "antmaze":
        return make_antmaze_env_factory(config.env_name, max_episode_steps=2000)
    if config.domain in ("walker", "cheetah"):
        return make_exorl_env_factory(config.domain, max_episode_steps=1000)
    if config.domain == "kitchen":
        return make_kitchen_env_factory(config.env_name, max_episode_steps=280)
    raise ValueError(f"Unsupported domain '{config.domain}'")


def context_states(
    dataset: OfflineDataset,
    task,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the ``(state, reward)`` context set used to encode a downstream task.

    Section 5.2 / Table 1: FRE uses only 32 ``(state, reward)`` pairs at
    evaluation time.
    """
    idx = rng.integers(0, len(dataset), size=num_samples)
    return np.asarray(dataset.encoder_observations[idx], dtype=np.float32)
