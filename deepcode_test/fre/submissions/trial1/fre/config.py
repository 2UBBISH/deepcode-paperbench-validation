"""
Configuration and hyperparameters for Functional Reward Encodings (FRE).

All hyperparameters are centralized here for easy modification and reproducibility.
"""

import torch


class Config:
    """Central configuration for FRE training and evaluation."""

    # ============================================================
    # General
    # ============================================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ============================================================
    # Data
    # ============================================================
    # AntMaze
    antmaze_dataset: str = "antmaze-large-diverse-v2"
    # Kitchen
    kitchen_dataset: str = "kitchen-complete-v0"
    # ExORL domains
    exorl_walker: str = "walker"
    exorl_cheetah: str = "cheetah"

    # ============================================================
    # Encoder (Transformer VAE)
    # ============================================================
    # Number of encoder states K
    K_encoder: int = 32
    # Number of decoder states K'
    K_decoder: int = 32
    # Embedding dimension for reward discretization
    d_embed: int = 128
    # Transformer model dimension
    d_model: int = 256
    # Number of transformer encoder layers
    num_layers: int = 2
    # Number of attention heads
    num_heads: int = 4
    # Feedforward dimension in transformer
    dim_feedforward: int = 512
    # Dropout rate in transformer
    dropout: float = 0.1
    # Latent dimension z
    d_latent: int = 64
    # Number of reward discretization bins
    num_reward_bins: int = 64
    # Maximum absolute reward value for clipping/discretization
    R_max: float = 10.0
    # KL divergence weight (beta in beta-VAE)
    beta_kl: float = 0.1

    # ============================================================
    # Decoder
    # ============================================================
    decoder_hidden_layers: list = [256, 256]
    decoder_activation: str = "relu"

    # ============================================================
    # IQL Networks
    # ============================================================
    iql_hidden_layers: list = [256, 256, 256]
    iql_activation: str = "relu"
    # Expectile for value function regression
    tau: float = 0.7
    # Temperature for advantage-weighted policy update
    beta: float = 3.0
    # Discount factor
    gamma: float = 0.99
    # Soft target update rate
    target_update_rate: float = 0.005
    # Maximum weight for advantage-weighted regression clipping
    max_advantage_weight: float = 100.0

    # ============================================================
    # Training
    # ============================================================
    # Learning rate for encoder/decoder
    lr_encoder: float = 1e-4
    # Learning rate for IQL networks
    lr_iql: float = 3e-4
    # Batch size for encoder training
    batch_size_encoder: int = 256
    # Batch size for IQL training
    batch_size_iql: int = 256
    # Number of encoder training steps (Phase 1)
    encoder_steps: int = 200000
    # Number of IQL training steps (Phase 2)
    iql_steps: int = 1000000
    # Gradient clipping value
    grad_clip: float = 1.0
    # Log interval (steps)
    log_interval: int = 1000
    # Save interval (steps)
    save_interval: int = 10000
    # Evaluation interval during IQL training
    eval_interval: int = 50000

    # ============================================================
    # Evaluation
    # ============================================================
    # Number of states for zero-shot encoding at evaluation
    K_eval: int = 32
    # Number of evaluation episodes per task
    num_episodes: int = 20
    # Number of random seeds for evaluation
    num_seeds: int = 5
    # Maximum episode length
    max_episode_steps: int = 1000

    # ============================================================
    # Reward Functions
    # ============================================================
    # Singleton goal-reaching epsilon thresholds
    antmaze_epsilon: float = 0.5
    exorl_epsilon: float = 0.1
    kitchen_epsilon: float = 0.3
    # Linear reward sparsity (fraction of zeroed-out weights)
    linear_sparsity: float = 0.8
    # MLP hidden size
    mlp_hidden_size: int = 256
    # Mixture probabilities [singleton, linear, mlp]
    mixture_probs: list = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    # ============================================================
    # Domain-specific
    # ============================================================
    # AntMaze state dimension
    antmaze_state_dim: int = 29
    antmaze_action_dim: int = 8
    # Kitchen state dimension
    kitchen_state_dim: int = 60
    kitchen_action_dim: int = 9
    # ExORL Walker
    walker_state_dim: int = 24
    walker_action_dim: int = 6
    # ExORL Cheetah
    cheetah_state_dim: int = 17
    cheetah_action_dim: int = 6

    def __init__(self, **kwargs):
        """Override defaults with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    def to_dict(self):
        """Return all non-private attributes as a dictionary."""
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith('_') and not callable(v)}


# Default configuration instance
config = Config()