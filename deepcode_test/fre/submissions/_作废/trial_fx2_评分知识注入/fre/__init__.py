"""Functional Reward Encodings (FRE) for Zero-Shot Offline RL.

This package reproduces the paper:
    "Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning"

It provides:
- Offline dataset loaders for AntMaze, ExORL, and Kitchen.
- A prior distribution over unsupervised reward functions.
- A permutation-invariant transformer VAE that encodes arbitrary reward
  functions from a small set of state-reward examples.
- An implicit Q-learning (IQL) agent conditioned on the resulting latent code.
- Baseline implementations (GC-IQL, GC-BC, OPAL, FB, SF).
- Training, evaluation, and visualization pipelines.
"""

__version__ = "0.1.0"

from fre.config import (
    Config,
    ExperimentConfig,
    DataConfig,
    RewardSamplerConfig,
    FREConfig,
    IQLConfig,
    BaselineConfig,
    EvalConfig,
    get_config,
    resolve_device,
    ANTMAZE_TASKS,
    EXORL_TASKS,
    KITCHEN_TASKS,
    ALL_TASKS,
)

__all__ = [
    "Config",
    "ExperimentConfig",
    "DataConfig",
    "RewardSamplerConfig",
    "FREConfig",
    "IQLConfig",
    "BaselineConfig",
    "EvalConfig",
    "get_config",
    "resolve_device",
    "ANTMAZE_TASKS",
    "EXORL_TASKS",
    "KITCHEN_TASKS",
    "ALL_TASKS",
]
