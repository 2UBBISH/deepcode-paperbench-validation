"""Central configuration for Functional Reward Encodings (FRE) reproduction.

This module defines all default hyperparameters, experiment settings, and
domain/task specifications used throughout the codebase.  Configuration can be
instantiated programmatically or loaded from YAML/dictionaries.

The defaults follow the reproduction plan:

    K = 32, K' = 1024, M = 128, d_z = 64,
    transformer layers = 2, heads = 4, d_model = 128,
    encoder lr = 1e-4, RL lr = 3e-4, beta = 3.0, tau = 0.7, gamma = 0.99.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Domain / task registry
# ---------------------------------------------------------------------------

ANTMAZE_TASKS: Tuple[str, ...] = (
    "ant-goal-reaching",
    "ant-directional",
    "ant-random-simplex",
    "ant-path-loop",
    "ant-path-edges",
    "ant-path-center",
)

EXORL_TASKS: Tuple[str, ...] = (
    "walker-goal-reaching",
    "cheetah-goal-reaching",
    "walker-forward-velocity",
    "walker-backward-velocity",
    "cheetah-forward-velocity",
    "cheetah-backward-velocity",
)

KITCHEN_TASKS: Tuple[str, ...] = (
    "microwave",
    "kettle",
    "slide_cabinet",
    "hinge_cabinet",
    "bottom_burner",
    "top_burner",
    "light_switch",
)

ALL_TASKS: Tuple[str, ...] = ANTMAZE_TASKS + EXORL_TASKS + KITCHEN_TASKS


@dataclass
class DataConfig:
    """Dataset and environment configuration."""

    # Core domains
    env_name: str = "antmaze-large-diverse-v2"
    dataset_name: str = "antmaze-large-diverse-v2"

    # Data directories
    d4rl_data_path: str = field(
        default_factory=lambda: os.environ.get("D4RL_DATA_PATH", "~/.d4rl/datasets")
    )
    exorl_data_path: str = field(
        default_factory=lambda: os.environ.get("EXORL_DATA_PATH", "~/.exorl/datasets")
    )

    # State normalization
    normalize_states: bool = True
    normalize_rewards: bool = True

    # Batch sizes
    encoder_batch_size: int = 256      # number of reward functions per encoder step
    decoder_batch_size: int = 1024     # number of decoder states per reward function
    rl_batch_size: int = 256
    eval_episodes: int = 20

    # AntMaze specifics (D4RL convention)
    antmaze_reward_scale: float = 1.0
    antmaze_terminate_on_goal: bool = False

    # Kitchen specifics
    kitchen_env_name: str = "kitchen-complete-v0"
    kitchen_dataset_name: str = "kitchen-complete-v0"

    def __post_init__(self) -> None:
        self.d4rl_data_path = os.path.expanduser(self.d4rl_data_path)
        self.exorl_data_path = os.path.expanduser(self.exorl_data_path)


@dataclass
class RewardSamplerConfig:
    """Prior reward-function mixture configuration."""

    # Number of encoder context states K
    num_encoder_states: int = 32
    # Number of decoder states K'
    num_decoder_states: int = 1024

    # Singleton goal-reaching
    singleton: bool = True
    singleton_threshold: float = 1.0
    singleton_reward_inside: float = 0.0
    singleton_reward_outside: float = -1.0

    # Linear rewards
    linear: bool = True
    linear_active_fraction: float = 0.3
    linear_sparse_mask: bool = True

    # Random MLP rewards
    mlp: bool = True
    mlp_hidden_size: int = 64
    mlp_num_layers: int = 2
    mlp_activation: str = "relu"

    # Sampling probabilities (normalized automatically)
    singleton_weight: float = 1.0
    linear_weight: float = 1.0
    mlp_weight: float = 1.0

    # Optional domain-prior augmentation
    antmaze_xy_only_weight: float = 0.0
    exorl_velocity_only_weight: float = 0.0


@dataclass
class FREConfig:
    """FRE variational autoencoder configuration."""

    # Reward discretization
    num_reward_bins: int = 128
    reward_min: float = -1.0
    reward_max: float = 1.0
    use_learned_projection: bool = False  # fallback: linear projection from scalar

    # Embedding / transformer
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    state_embed_hidden: int = 128

    # Latent space
    d_z: int = 64
    beta: float = 1.0  # KL weight

    # Decoder
    decoder_hidden: int = 256
    decoder_num_layers: int = 2

    # Optimization
    encoder_lr: float = 1e-4
    encoder_weight_decay: float = 0.0
    encoder_pretrain_steps: int = 200_000
    encoder_log_interval: int = 1_000
    encoder_checkpoint_interval: int = 20_000

    # Evaluation encoding
    eval_num_encoder_states: int = 32


@dataclass
class IQLConfig:
    """Implicit Q-learning configuration."""

    hidden_size: int = 256
    num_layers: int = 2
    activation: str = "relu"

    gamma: float = 0.99
    tau: float = 0.7           # expectile for value update
    tau_goal: float = 0.9      # expectile used for goal-reaching if requested
    temperature: float = 3.0   # beta in the paper / advantage temperature
    target_tau: float = 0.005   # soft target update rate
    rl_lr: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 10.0

    # Training
    rl_train_steps: int = 1_000_000
    rl_log_interval: int = 1_000
    rl_eval_interval: int = 50_000
    rl_checkpoint_interval: int = 100_000

    # Policy
    policy_log_std_min: float = -5.0
    policy_log_std_max: float = 2.0

    # Deterministic evaluation
    deterministic_eval: bool = True


@dataclass
class BaselineConfig:
    """Baseline hyperparameters."""

    # Goal-conditioned baselines
    gc_hidden_size: int = 256
    gc_num_layers: int = 2
    gc_lr: float = 3e-4
    gc_hindsight_relabel: bool = True
    gc_hindsight_prob: float = 0.8
    gc_train_steps: int = 1_000_000

    # OPAL skill discovery
    opal_skill_dim: int = 8
    opal_action_chunk: int = 8
    opal_hidden_size: int = 256
    opal_lr: float = 3e-4
    opal_train_steps: int = 1_000_000
    opal_eval_skills: int = 10
    opal_prior_learned: bool = True

    # FB / SF
    fb_num_reward_samples: int = 5120
    sf_num_reward_samples: int = 5120
    fb_hidden_size: int = 256
    sf_hidden_size: int = 256
    fb_train_steps: int = 1_000_000
    sf_train_steps: int = 1_000_000
    fb_lr: float = 3e-4
    sf_lr: float = 3e-4


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    num_episodes: int = 20
    num_reward_samples: int = 32          # FRE evaluation context size
    fb_sf_num_reward_samples: int = 5120
    opal_num_skills: int = 10
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    max_episode_steps: int = 1000
    normalize_score_0_100: bool = True
    save_visualizations: bool = True
    results_dir: str = "results"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "fre"
    domain: str = "antmaze"           # antmaze | exorl | kitchen
    task: Optional[str] = None
    seed: int = 0
    device: str = "auto"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    wandb_project: str = "fre-reproduction"

    data: DataConfig = field(default_factory=DataConfig)
    reward_sampler: RewardSamplerConfig = field(default_factory=RewardSamplerConfig)
    fre: FREConfig = field(default_factory=FREConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _dataclass_from_dict(cls, data: Dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dictionary, ignoring unknown keys."""
    if data is None:
        return cls()
    if not is_dataclass(cls):
        return data

    valid_fields = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in valid_fields:
            continue
        field_type = cls.__dataclass_fields__[key].type
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _dataclass_from_dict(field_type, value)
        elif (
            isinstance(field_type, type)
            and issubclass(field_type, tuple)
            and isinstance(value, list)
        ):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Recursively convert dataclasses to plain dictionaries."""
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


class Config(ExperimentConfig):
    """Config with convenient loading/saving helpers.

    Inherits from :class:`ExperimentConfig` so existing field access patterns
    (``config.fre.d_z``) remain available, while adding YAML/dict conversion.
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        base = _dataclass_from_dict(ExperimentConfig, data)
        cfg = cls(
            name=base.name,
            domain=base.domain,
            task=base.task,
            seed=base.seed,
            device=base.device,
            log_dir=base.log_dir,
            checkpoint_dir=base.checkpoint_dir,
            use_wandb=base.use_wandb,
            wandb_project=base.wandb_project,
            data=base.data,
            reward_sampler=base.reward_sampler,
            fre=base.fre,
            iql=base.iql,
            baseline=base.baseline,
            eval=base.eval,
        )
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def default(cls, **overrides: Any) -> "Config":
        return cls.from_dict(overrides)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)

    def to_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    def clone(self) -> "Config":
        return Config.from_dict(self.to_dict())


def get_config(name: str = "default", **overrides: Any) -> Config:
    """Return a configuration preset.

    Args:
        name: One of ``default``, ``antmaze``, ``exorl``, ``kitchen`` or a path
            to a YAML file.
        **overrides: Additional overrides applied to the final config.
    """
    if name in ("default", "antmaze", "exorl", "kitchen") and os.path.exists(
        os.path.join("configs", f"{name}.yaml")
    ):
        cfg = Config.from_yaml(os.path.join("configs", f"{name}.yaml"))
    elif os.path.exists(name):
        cfg = Config.from_yaml(name)
    else:
        cfg = Config.default()

    # Apply flat overrides of the form section__field=value
    for key, value in overrides.items():
        if "__" in key:
            section, field = key.split("__", 1)
            if hasattr(cfg, section):
                setattr(getattr(cfg, section), field, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def resolve_device(device: str = "auto") -> str:
    """Resolve torch device string."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


__all__ = [
    "Config",
    "ExperimentConfig",
    "DataConfig",
    "RewardSamplerConfig",
    "FREConfig",
    "IQLConfig",
    "BaselineConfig",
    "EvalConfig",
    "ANTMAZE_TASKS",
    "EXORL_TASKS",
    "KITCHEN_TASKS",
    "ALL_TASKS",
    "get_config",
    "resolve_device",
]
