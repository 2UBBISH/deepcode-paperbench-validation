"""Functional Reward Encodings (FRE) for Zero-Shot Offline RL.

This package implements the core FRE method described in
"Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning".

Modules:
    reward_prior: random reward-function priors (goals, linear, MLP).
    reward_embedding: reward discretization and learned token embeddings.
    encoder: permutation-invariant transformer VAE encoder.
    decoder: reward decoder MLP.
    fre_vae: full VAE combining encoder/decoder with training helpers.
    iql: Implicit Q-Learning networks and losses.
    agent: FRE-conditioned IQL agent.
    dataset: offline dataset wrappers, normalization, and sampling.
    config: dataclass configuration and hyperparameter defaults.
    utils: shared utilities (seeding, normalization, logging, checkpoints).
"""

from .reward_prior import RewardPrior, make_default_reward_prior
from .reward_embedding import RewardEmbedding, reward_to_bins
from .encoder import FREEncoder
from .decoder import RewardDecoder
from .fre_vae import FREVAE
from .iql import (
    IQLNetworks,
    IQLQNetwork,
    IQLValueNetwork,
    SquashedGaussianPolicy,
    expectile_loss,
    soft_update,
    hard_update,
)
from .agent import FREAgent
from .dataset import (
    OfflineDataset,
    TorchOfflineDataset,
    load_offline_dataset,
    load_d4rl_antmaze,
    load_d4rl_kitchen,
    load_exorl,
    build_state_pool,
    make_synthetic_dataset,
)
from .config import (
    RewardEmbeddingConfig,
    FREVAEConfig,
    RewardPriorConfig,
    IQLConfig,
    FREAgentConfig,
    TrainingConfig,
    EvalConfig,
    FBBaselineConfig,
    SFBaselineConfig,
    GCIQLBaselineConfig,
    GCBCBaselineConfig,
    OPALBaselineConfig,
    BaselineConfigs,
    get_domain_dims,
    get_domain_defaults,
    default_configs_for_domain,
)
from .utils import (
    set_seed,
    resolve_device,
    to_numpy,
    to_torch,
    ensure_numpy_array,
    RunningMeanStd,
    StateNormalizer,
    compute_state_normalization,
    normalize_states,
    unnormalize_states,
    configure_logging,
    get_logger,
    save_json,
    load_json,
    save_checkpoint,
    load_checkpoint,
    freeze_module,
    unfreeze_module,
    count_parameters,
    soft_update_from_dict,
    average_dicts,
    std_dicts,
    Timer,
    sample_indices,
    stable_log_prob,
)

__version__ = "0.1.0"

__all__ = [
    # Reward prior
    "RewardPrior",
    "make_default_reward_prior",
    # Reward embedding
    "RewardEmbedding",
    "reward_to_bins",
    # Encoder / decoder / VAE
    "FREEncoder",
    "RewardDecoder",
    "FREVAE",
    # IQL
    "IQLNetworks",
    "IQLQNetwork",
    "IQLValueNetwork",
    "SquashedGaussianPolicy",
    "expectile_loss",
    "soft_update",
    "hard_update",
    # Agent
    "FREAgent",
    # Dataset
    "OfflineDataset",
    "TorchOfflineDataset",
    "load_offline_dataset",
    "load_d4rl_antmaze",
    "load_d4rl_kitchen",
    "load_exorl",
    "build_state_pool",
    "make_synthetic_dataset",
    # Config
    "RewardEmbeddingConfig",
    "FREVAEConfig",
    "RewardPriorConfig",
    "IQLConfig",
    "FREAgentConfig",
    "TrainingConfig",
    "EvalConfig",
    "FBBaselineConfig",
    "SFBaselineConfig",
    "GCIQLBaselineConfig",
    "GCBCBaselineConfig",
    "OPALBaselineConfig",
    "BaselineConfigs",
    "get_domain_dims",
    "get_domain_defaults",
    "default_configs_for_domain",
    # Utils
    "set_seed",
    "resolve_device",
    "to_numpy",
    "to_torch",
    "ensure_numpy_array",
    "RunningMeanStd",
    "StateNormalizer",
    "compute_state_normalization",
    "normalize_states",
    "unnormalize_states",
    "configure_logging",
    "get_logger",
    "save_json",
    "load_json",
    "save_checkpoint",
    "load_checkpoint",
    "freeze_module",
    "unfreeze_module",
    "count_parameters",
    "soft_update_from_dict",
    "average_dicts",
    "std_dicts",
    "Timer",
    "sample_indices",
    "stable_log_prob",
]
