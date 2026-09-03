"""
FRE Training Package

This package contains the training modules for the Functional Reward Encodings (FRE)
framework, including:

- Phase 1: Encoder+Decoder VAE training (train_encoder.py)
- Phase 2: IQL agent training with frozen encoder (train_iql.py)
- Shared utilities: logging, loss functions, checkpointing, data sampling (utils.py)
"""

from fre.training.train_encoder import (
    train_encoder,
    evaluate_encoder,
    load_pretrained_encoder,
)

from fre.training.train_iql import (
    train_iql,
    load_pretrained_iql,
    evaluate_iql_training,
)

from fre.training.utils import (
    Logger,
    create_prior_reward_function,
    sample_reward_function,
    sample_encoder_batch,
    sample_iql_batch,
    compute_vae_loss,
    expectile_loss,
    compute_iql_value_loss,
    compute_iql_q_loss,
    compute_iql_policy_loss,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_device,
    count_parameters,
    print_model_info,
)

__all__ = [
    # Encoder training
    "train_encoder",
    "evaluate_encoder",
    "load_pretrained_encoder",
    # IQL training
    "train_iql",
    "load_pretrained_iql",
    "evaluate_iql_training",
    # Utilities
    "Logger",
    "create_prior_reward_function",
    "sample_reward_function",
    "sample_encoder_batch",
    "sample_iql_batch",
    "compute_vae_loss",
    "expectile_loss",
    "compute_iql_value_loss",
    "compute_iql_q_loss",
    "compute_iql_policy_loss",
    "save_checkpoint",
    "load_checkpoint",
    "set_seed",
    "get_device",
    "count_parameters",
    "print_model_info",
]