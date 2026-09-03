"""Configuration dataclasses and hyperparameter defaults for FRE reproduction.

This module centralizes all hyperparameters used by the FRE encoder/decoder
(phase 1), the FRE-conditioned IQL agent (phase 2), zero-shot evaluation,
baselines, and the experiment scripts.  Defaults follow the paper and the
reproduction plan, and can be overridden per-domain through the helper
``get_domain_defaults`` or by constructing the dataclasses directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core FRE VAE (encoder/decoder) configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardEmbeddingConfig:
    """Reward discretization and learned token embedding settings."""

    state_proj_dim: int = 192
    reward_bins: int = 64
    embedding_dim: int = 64
    token_dim: int = 256


@dataclass
class FREVAEConfig:
    """Hyperparameters for the permutation-invariant transformer VAE."""

    # Reward tokenizer / embedding
    reward_embedding: RewardEmbeddingConfig = field(default_factory=RewardEmbeddingConfig)

    # Transformer encoder
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"

    # Latent space
    latent_dim: int = 128

    # Decoder MLP
    decoder_hidden: Tuple[int, ...] = (256, 256)
    decoder_activation: str = "relu"

    # VAE objective
    beta: float = 1.0

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.0

    # Context sizes for training
    encoder_states: int = 32      # K
    decoder_states: int = 256     # K'
    reward_fn_batch_size: int = 16

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["reward_embedding"] = asdict(self.reward_embedding)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FREVAEConfig":
        emb_data = data.pop("reward_embedding", {})
        config = cls(**data)
        config.reward_embedding = RewardEmbeddingConfig(**emb_data)
        return config


# ---------------------------------------------------------------------------
# Random reward prior configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardPriorConfig:
    """Random reward function prior distribution settings."""

    # Singleton goal rewards
    goal_epsilon: float = 1.0

    # Random linear rewards
    p_mask: float = 0.75

    # Random MLP rewards
    mlp_hidden: int = 256

    # Uniform mixture weights over (singleton, linear, mlp).
    mixture_weights: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    # All sampled rewards are clipped to [-1, 1].
    clip_min: float = -1.0
    clip_max: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# IQL configuration
# ---------------------------------------------------------------------------
@dataclass
class IQLConfig:
    """Implicit Q-Learning network and loss hyperparameters."""

    # Network widths for Q, V, and policy
    q_hidden: Tuple[int, ...] = (256, 256)
    v_hidden: Tuple[int, ...] = (256, 256)
    policy_hidden: Tuple[int, ...] = (256, 256)

    # IQL loss hyperparameters
    gamma: float = 0.99
    expectile: float = 0.9
    awr_temperature: float = 3.0
    target_tau: float = 0.005
    advantage_clip: Tuple[float, float] = (-5.0, 2.0)

    # Policy output bounds
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# FRE-conditioned IQL agent configuration
# ---------------------------------------------------------------------------
@dataclass
class FREAgentConfig:
    """Combined FRE + IQL agent settings for phase-2 RL training."""

    state_dim: int = 17
    action_dim: int = 8
    latent_dim: int = 128

    vae: FREVAEConfig = field(default_factory=FREVAEConfig)
    reward_prior: RewardPriorConfig = field(default_factory=RewardPriorConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)

    freeze_vae: bool = True
    encoder_states: int = 32

    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "vae": self.vae.to_dict(),
            "reward_prior": self.reward_prior.to_dict(),
            "iql": self.iql.to_dict(),
            "freeze_vae": self.freeze_vae,
            "encoder_states": self.encoder_states,
            "device": self.device,
        }


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Experiment-wide training loop settings."""

    seed: int = 0
    device: str = "auto"

    # Phase 1: FRE encoder/decoder training
    vae_steps: int = 100000
    vae_batch_size: int = 16
    vae_log_interval: int = 1000
    vae_save_interval: int = 10000
    vae_checkpoint: Optional[str] = None

    # Phase 2: IQL training with frozen encoder
    rl_steps: int = 1000000
    rl_batch_size: int = 256
    rl_log_interval: int = 1000
    rl_save_interval: int = 50000
    rl_checkpoint: Optional[str] = None

    # Dataset / domain
    domain: str = "antmaze"
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    state_pool_size: Optional[int] = None
    normalize_states: bool = True

    # Checkpoint / logging
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_tensorboard: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Zero-shot downstream task evaluation settings."""

    num_examples: int = 32        # state-reward examples for FRE encoding
    num_episodes: int = 20
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    stochastic: bool = False      # sample z vs. use posterior mean
    device: str = "auto"
    output_dir: str = "eval_results"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Baseline configurations
# ---------------------------------------------------------------------------
@dataclass
class FBBaselineConfig:
    """Forward-Backward representation baseline."""

    repr_dim: int = 256
    hidden_dims: Tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    eval_reward_samples: int = 5120
    ridge_coeff: float = 1e-3


@dataclass
class SFBaselineConfig:
    """Successor Features baseline with ICM features."""

    feature_dim: int = 256
    hidden_dims: Tuple[int, ...] = (256, 256)
    icm_lr: float = 1e-4
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    eval_reward_samples: int = 5120


@dataclass
class GCIQLBaselineConfig:
    """Goal-conditioned IQL baseline."""

    hidden_dims: Tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    expectile: float = 0.9
    awr_temperature: float = 3.0
    target_tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 256
    hindsight_prob: float = 0.5


@dataclass
class GCBCBaselineConfig:
    """Goal-conditioned behavioral cloning baseline."""

    hidden_dims: Tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    batch_size: int = 256
    relabel_fraction: float = 0.5


@dataclass
class OPALBaselineConfig:
    """OPAL unsupervised skill-discovery baseline."""

    skill_dim: int = 16
    encoder_hidden: Tuple[int, ...] = (256, 256)
    decoder_hidden: Tuple[int, ...] = (256, 256)
    policy_hidden: Tuple[int, ...] = (256, 256)
    prior_hidden: Tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    beta: float = 1.0
    batch_size: int = 256
    num_skills_eval: int = 10


@dataclass
class BaselineConfigs:
    """Container for all baseline hyperparameter configurations."""

    fb: FBBaselineConfig = field(default_factory=FBBaselineConfig)
    sf: SFBaselineConfig = field(default_factory=SFBaselineConfig)
    gc_iql: GCIQLBaselineConfig = field(default_factory=GCIQLBaselineConfig)
    gc_bc: GCBCBaselineConfig = field(default_factory=GCBCBaselineConfig)
    opal: OPALBaselineConfig = field(default_factory=OPALBaselineConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Domain-specific defaults
# ---------------------------------------------------------------------------
_ANTMAZE_STATE_DIM = 29  # D4RL antmaze-large qpos(15) + qvel(14)
_ANTMAZE_ACTION_DIM = 8
_KITCHEN_STATE_DIM = 60  # D4RL kitchen-complete flattened observation
_KITCHEN_ACTION_DIM = 9
_EXORL_WALKER_STATE_DIM = 24
_EXORL_WALKER_ACTION_DIM = 6
_EXORL_CHEETAH_STATE_DIM = 18
_EXORL_CHEETAH_ACTION_DIM = 6


def get_domain_dims(domain: str) -> Tuple[int, int]:
    """Return ``(state_dim, action_dim)`` for a supported benchmark domain."""
    domain = domain.lower()
    if domain in ("antmaze", "antmaze-large-diverse-v2"):
        return _ANTMAZE_STATE_DIM, _ANTMAZE_ACTION_DIM
    if domain in ("kitchen", "kitchen-complete-v0"):
        return _KITCHEN_STATE_DIM, _KITCHEN_ACTION_DIM
    if domain in ("walker", "exorl-walker"):
        return _EXORL_WALKER_STATE_DIM, _EXORL_WALKER_ACTION_DIM
    if domain in ("cheetah", "exorl-cheetah"):
        return _EXORL_CHEETAH_STATE_DIM, _EXORL_CHEETAH_ACTION_DIM
    raise ValueError(f"Unsupported domain: {domain}")


def get_domain_defaults(domain: str, dataset_name: Optional[str] = None) -> TrainingConfig:
    """Return a :class:`TrainingConfig` populated with sensible per-domain defaults.

    Recommended phase-2 gradient steps follow the reproduction plan:
        AntMaze: 1e6
        ExORL:   5e5
        Kitchen: 1e6
    """
    domain = domain.lower()
    config = TrainingConfig()
    config.domain = domain

    if domain in ("antmaze", "antmaze-large-diverse-v2"):
        config.domain = "antmaze"
        config.dataset_name = dataset_name or "antmaze-large-diverse-v2"
        config.rl_steps = 1000000
        config.vae_steps = 100000
    elif domain in ("kitchen", "kitchen-complete-v0"):
        config.domain = "kitchen"
        config.dataset_name = dataset_name or "kitchen-complete-v0"
        config.rl_steps = 1000000
        config.vae_steps = 100000
    elif domain in ("walker", "exorl-walker"):
        config.domain = "walker"
        config.dataset_name = dataset_name or "walker"
        config.rl_steps = 500000
        config.vae_steps = 100000
    elif domain in ("cheetah", "exorl-cheetah"):
        config.domain = "cheetah"
        config.dataset_name = dataset_name or "cheetah"
        config.rl_steps = 500000
        config.vae_steps = 100000
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    return config


def default_configs_for_domain(domain: str) -> Dict[str, Any]:
    """Convenience helper returning a dict of all configs for a domain."""
    state_dim, action_dim = get_domain_dims(domain)
    fre_agent = FREAgentConfig(state_dim=state_dim, action_dim=action_dim)
    training = get_domain_defaults(domain)
    eval_config = EvalConfig()
    return {
        "domain": domain,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "fre_agent": fre_agent.to_dict(),
        "training": training.to_dict(),
        "eval": eval_config.to_dict(),
    }
