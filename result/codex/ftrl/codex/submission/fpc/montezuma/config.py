"""Hyperparameters for the Montezuma's Revenge experiments (Table 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..retention.base import RetentionConfig


@dataclass
class MontezumaConfig:
    """Defaults follow Table 2 of the paper (based on Burda et al., 2018)."""

    env_id: str = "MontezumaRevengeNoFrameskip-v4"
    max_steps_per_episode: int = 4500          # MaxStepPerEpisode
    ext_coef: float = 2.0                      # ExtCoef
    learning_rate: float = 1e-4                # LearningRate
    num_env: int = 128                         # NumEnv
    num_step: int = 128                        # NumStep
    gamma: float = 0.999                       # Gamma
    int_gamma: float = 0.99                    # IntGamma
    gae_lambda: float = 0.95                   # Lambda
    stable_eps: float = 1e-8                   # StableEps
    state_stack_size: int = 4                  # StateStackSize
    preproc_height: int = 84                   # PreProcHeight
    preproc_width: int = 84                    # ProProcWidth
    use_gae: bool = True                       # UseGAE
    use_gpu: bool = False                      # UseGPU (no GPU in the repro env)
    use_norm: bool = False                     # UseNorm
    use_noisy_net: bool = False                # UseNoisyNet
    clip_grad_norm: float = 0.5                # ClipGradNorm
    entropy: float = 0.001                     # Entropy
    epoch: int = 4                             # Epoch
    mini_batch: int = 4                        # MiniBatch
    ppo_eps: float = 0.1                       # PPOEps
    int_coef: float = 1.0                      # IntCoef
    sticky_action: bool = True                 # StickyAction
    action_prob: float = 0.25                  # ActionProb
    update_proportion: float = 0.25            # UpdateProportion
    life_done: bool = False                    # LifeDone
    obs_norm_step: int = 50                    # ObsNormStep

    # ----------------------------- pre-training -----------------------------
    pretrain_room: int = 7
    pretrain_target_return: float = 7000.0
    pretrain_max_steps: int = 100_000_000

    # ----------------------------- bc buffer --------------------------------
    bc_trajectories: int = 500
    bc_buffer_size: int = 1_000_000
    bc_kl_coef: float = 0.01     # Figure 13 sweeps this value

    # ----------------------------- retention --------------------------------
    retention: RetentionConfig = field(default_factory=RetentionConfig)

    seed: int = 0
    total_steps: int = 50_000_000
    device: str = "cpu"
    log_dir: Optional[str] = None
    save_dir: Optional[str] = None

    # The Room-7 success rate is evaluated every 5M steps (addendum, Figure 6).
    eval_every_steps: int = 5_000_000
    eval_episodes: int = 100
