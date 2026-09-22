"""Hyperparameters for the NetHack experiments (Table 1, Appendix B.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..retention.base import RetentionConfig


@dataclass
class NetHackConfig:
    """Defaults follow Table 1 (analogous to Table 6 of Petrenko et al., 2020)."""

    activation_function: str = "relu"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-7
    adam_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    appo_clip_policy: float = 0.1
    appo_clip_baseline: float = 1.0
    baseline_cost: float = 1.0
    discounting: float = 0.999999
    entropy_cost: float = 0.001
    grad_norm_clipping: float = 4.0
    hidden_dim: int = 1738
    batch_size: int = 128
    penalty_step: float = 0.0
    penalty_time: float = 0.0
    reward_clip: float = 10.0
    reward_scale: float = 1.0
    unroll_length: int = 32

    # ---- environment ----
    character: str = "mon-hum-neu-mal"      # Human Monk
    num_actions: int = 120                   # NLE action space
    num_envs: int = 32
    num_workers: int = 8                     # APPO rollout workers
    obs_screen_shape: tuple = (21, 79)       # glyph/char grid used by the encoder
    num_chars: int = 256
    num_colors: int = 64
    char_embed_dim: int = 32
    color_embed_dim: int = 16
    message_length: int = 256
    blstats_length: int = 25
    resnet_blocks: int = 3
    resnet_channels: int = 64
    mlp_hidden: int = 128

    # ---- pre-training ----
    freeze_encoders: bool = True     # "we froze the encoders during training"
    bc_head_only_steps: int = 500_000_000  # baseline head pre-training steps
    pretrained_checkpoint: Optional[str] = None

    # ---- dataset ----
    dataset_path: Optional[str] = None   # path to the unpacked NLD-AA
    dataset_name: str = "nld-aa-v0"
    num_games: int = 8000                # ~8000 Human Monk games
    fisher_batches: int = 10000          # 10000 batches for the Fisher matrix
    fisher_batch_size: int = 128
    bc_buffer_size: int = 1_000_000

    # ---- retention ----
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    # ---- training ----
    total_steps: int = 500_000_000
    report_interval: int = 25_000_000    # Figure 5 evaluates every 25M steps
    eval_episodes: int = 200
    eval_every_steps: int = 25_000_000
    device: str = "cpu"
    seed: int = 0
    log_dir: Optional[str] = None
