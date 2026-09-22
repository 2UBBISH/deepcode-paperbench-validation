"""Hyperparameters for the RoboticSequence experiments (Table 3, Appendix B.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..retention.base import RetentionConfig


# The order of stages in the main-text RoboticSequence.
MAIN_SEQUENCE: List[str] = ["hammer", "push", "peg-unplug-side", "push-wall"]
# The two stages the pre-trained policy solves (FAR for the main sequence).
PRETRAINED_STAGES: List[str] = ["peg-unplug-side", "push-wall"]
# Alternative sequences from Appendix F.
ALTERNATIVE_SEQUENCE: List[str] = ["shelf-place", "push-back", "window-close", "door-close"]
TWO_STAGE_SEQUENCE: List[str] = ["drawer-open", "pick-place"]


@dataclass
class MetaworldConfig:
    """Defaults follow Appendix B.3."""

    stages: List[str] = field(default_factory=lambda: list(MAIN_SEQUENCE))
    pretrained_stages: List[str] = field(default_factory=lambda: list(PRETRAINED_STAGES))

    hidden_dim: int = 256
    num_layers: int = 4           # 4 hidden layers, 256 neurons each
    layer_norm_first: bool = True  # layer norm after the first layer
    batch_size: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005            # polyak averaging for the target networks
    replay_buffer_size: int = 100_000
    memory_size: int = 10_000     # 10% of the replay buffer (EM / BC buffer)
    time_limit: int = 200         # T in Algorithm 1
    success_reward_beta: float = 1.5  # beta in the augmented success reward

    num_seeds: int = 20           # "at least 20 seeds" (Appendix B.3)
    num_train_steps: int = 1_000_000
    num_eval_episodes: int = 10
    eval_every_steps: int = 10_000

    retention: RetentionConfig = field(default_factory=RetentionConfig)
    device: str = "cpu"
    seed: int = 0
    log_dir: Optional[str] = None

    # ---- analysis options (Appendix F) ----
    reset_last_layer: bool = False        # Figure 21
    observation_translation: float = 0.0  # Figure 22 (task-difference experiment)
    prefix_tasks: Optional[List[str]] = None  # Table 6
