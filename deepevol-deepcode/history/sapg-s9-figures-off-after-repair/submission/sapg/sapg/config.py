"""Hyperparameter configuration for SAPG and baselines.

This module encodes the hyperparameters reported in the SAPG paper
(Tables 2-4) for each task family:

  * AllegroKuka tasks (Regrasping, Throw, Reorientation) -- 23 DoF arm+hand
  * ShadowHand (24 DoF in-hand reorientation)
  * AllegroHand (16 DoF in-hand reorientation)

The dataclasses here are consumed by ``networks.py``, ``losses.py``,
``returns.py``, ``rollout.py`` and ``algorithm.py``.

Paper references:
  - Sec 4.4 / Addendum: shared actor backbone B_theta and critic backbone
    C_psi conditioned on per-policy latents phi_j.
  - Sec 4.5: follower entropy regularization sigma * (i - 1) * H.
  - Sec 4.6: N = 24576 envs split into M = 6 blocks of N/M = 4096 envs.
  - Tables 2-4: per-task hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------------
@dataclass
class SAPGConfig:
    """Base SAPG hyperparameters shared across tasks.

    Task-specific subclasses override the defaults below.
    """

    # ---- Task identity -------------------------------------------------
    task: str = "regrasping"
    task_family: str = "allegrokuka"  # one of: allegrokuka, shadowhand, allegrohand

    # ---- Environment splitting (Sec 4, 4.6) ----------------------------
    num_envs: int = 24576          # N total parallel environments
    num_blocks: int = 6            # M blocks (one per policy)
    horizon: int = 16              # steps per env instance per update

    # ---- Policy / value network (Sec 4.4) ------------------------------
    latent_dim: int = 32           # dim of phi_j (32 AllegroKuka, 16 hands)
    actor_hidden_dims: Tuple[int, ...] = (768, 512, 256)
    critic_hidden_dims: Tuple[int, ...] = (768, 512, 256)
    use_lstm: bool = True          # AllegroKuka uses LSTM, hands use MLP
    lstm_hidden_size: int = 768
    lstm_num_layers: int = 1
    lstm_seq_len: int = 16
    activation: str = "elu"

    # ---- PPO / optimization (Sec 4.6, Tables 2-4) ----------------------
    learning_rate: float = 1e-4
    num_mini_epochs: int = 2
    minibatch_size_factor: int = 4   # minibatch = num_envs * factor
    clip_eps: float = 0.1            # PPO clip epsilon
    gamma: float = 0.99              # discount
    gae_lambda: float = 0.95         # GAE tau
    grad_norm_clip: float = 1.0
    kl_threshold: float = 0.016      # adaptive LR threshold
    adaptive_lr: bool = True

    # ---- SAPG aggregation (Sec 4.2-4.3) --------------------------------
    aggregation: str = "leader_follower"  # leader_follower | symmetric |
                                          # high_off_policy_ratio | no_off_policy
    off_policy_coef: float = 1.0          # lambda in Eq.4 / Eq.9
    critic_coef: float = 4.0              # lambda' applied to total critic loss
    subsample_off_policy: bool = True     # |D_1'| = |D_1|

    # ---- Entropy regularization (Sec 4.5) ------------------------------
    entropy_coef: float = 0.0        # sigma; leader has NO entropy term
    learnable_entropy_coef: bool = False  # per-block learnable sigma vector

    # ---- Value targets (Eq.5-6) ----------------------------------------
    on_policy_target_steps: int = 3  # 3-step on-policy target
    off_policy_target_steps: int = 1  # 1-step off-policy target

    # ---- Training schedule ---------------------------------------------
    num_iterations: int = 100000
    total_transitions: float = 2e10
    num_seeds: int = 5
    save_interval: int = 100
    log_interval: int = 1
    pbt_interval: int = 100          # DexPBT rank/exploit interval
    device: str = "cuda"
    seed: int = 0

    # ---- Derived helpers -----------------------------------------------
    @property
    def envs_per_block(self) -> int:
        assert self.num_envs % self.num_blocks == 0, (
            f"num_envs ({self.num_envs}) must be divisible by "
            f"num_blocks ({self.num_blocks})"
        )
        return self.num_envs // self.num_blocks

    @property
    def minibatch_size(self) -> int:
        return self.num_envs * self.minibatch_size_factor

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuples -> lists for json friendliness
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


# ---------------------------------------------------------------------------
# AllegroKuka tasks (Regrasping / Throw / Reorientation)
# ---------------------------------------------------------------------------
@dataclass
class AllegroKukaConfig(SAPGConfig):
    task_family: str = "allegrokuka"
    task: str = "regrasping"

    # 23 DoF: obs o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]
    obs_dim: int = 0  # filled in by env wrapper at runtime
    action_dim: int = 23

    num_envs: int = 24576
    num_blocks: int = 6
    horizon: int = 16

    latent_dim: int = 32
    actor_hidden_dims: Tuple[int, ...] = (768, 512, 256)
    critic_hidden_dims: Tuple[int, ...] = (768, 512, 256)
    use_lstm: bool = True
    lstm_hidden_size: int = 768
    lstm_num_layers: int = 1
    lstm_seq_len: int = 16

    learning_rate: float = 1e-4
    num_mini_epochs: int = 2
    clip_eps: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Regrasping-specific curriculum (Appendix A)
    regrasp_hold_steps: int = 30
    regrasp_tol_start: float = 0.075   # 7.5 cm
    regrasp_tol_end: float = 0.01      # 1 cm
    regrasp_tol_decay: float = 0.10    # -10% when avg success > 3


@dataclass
class RegraspingConfig(AllegroKukaConfig):
    task: str = "regrasping"
    entropy_coef: float = 0.0


@dataclass
class ThrowConfig(AllegroKukaConfig):
    task: str = "throw"
    entropy_coef: float = 0.0


@dataclass
class ReorientationConfig(AllegroKukaConfig):
    task: str = "reorientation"
    entropy_coef: float = 0.005


# ---------------------------------------------------------------------------
# ShadowHand (24 DoF in-hand reorientation)
# ---------------------------------------------------------------------------
@dataclass
class ShadowHandConfig(SAPGConfig):
    task_family: str = "shadowhand"
    task: str = "shadowhand"

    obs_dim: int = 0
    action_dim: int = 24

    num_envs: int = 24576
    num_blocks: int = 6
    horizon: int = 8

    latent_dim: int = 16
    actor_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 512, 256, 128)
    use_lstm: bool = False
    lstm_hidden_size: int = 0
    lstm_num_layers: int = 1
    lstm_seq_len: int = 8

    learning_rate: float = 1e-4
    num_mini_epochs: int = 5
    clip_eps: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.0


# ---------------------------------------------------------------------------
# AllegroHand (16 DoF in-hand reorientation)
# ---------------------------------------------------------------------------
@dataclass
class AllegroHandConfig(SAPGConfig):
    task_family: str = "allegrohand"
    task: str = "allegrohand"

    obs_dim: int = 0
    action_dim: int = 16

    num_envs: int = 24576
    num_blocks: int = 6
    horizon: int = 8

    latent_dim: int = 16
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    use_lstm: bool = False
    lstm_hidden_size: int = 0
    lstm_num_layers: int = 1
    lstm_seq_len: int = 8

    learning_rate: float = 1e-4
    num_mini_epochs: int = 5
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.0


# ---------------------------------------------------------------------------
# Baselines (Table 1)
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig(SAPGConfig):
    """Vanilla PPO baseline (one policy over all environments)."""

    aggregation: str = "no_off_policy"
    num_blocks: int = 1
    latent_dim: int = 0
    off_policy_coef: float = 0.0
    entropy_coef: float = 0.0


@dataclass
class PQLConfig(SAPGConfig):
    """Parallel Q-Learning baseline (one policy, off-policy replay)."""

    aggregation: str = "no_off_policy"
    num_blocks: int = 1
    latent_dim: int = 0
    off_policy_coef: float = 0.0
    entropy_coef: float = 0.0


@dataclass
class DexPBTConfig(SAPGConfig):
    """Population-based training + PPO (Petrenko et al. 2023)."""

    aggregation: str = "no_off_policy"
    num_blocks: int = 1
    latent_dim: int = 0
    off_policy_coef: float = 0.0
    population_size: int = 8
    pbt_interval: int = 100
    perturb_factor: float = 1.2


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------
TASK_CONFIGS = {
    "regrasping": RegraspingConfig,
    "throw": ThrowConfig,
    "reorientation": ReorientationConfig,
    "shadowhand": ShadowHandConfig,
    "allegrohand": AllegroHandConfig,
}

BASELINE_CONFIGS = {
    "ppo": PPOConfig,
    "pql": PQLConfig,
    "dexpbt": DexPBTConfig,
}


def get_config(task: str, algorithm: str = "sapg", **overrides) -> SAPGConfig:
    """Build a config for ``task`` and ``algorithm``.

    Args:
        task: one of ``TASK_CONFIGS`` keys.
        algorithm: ``sapg`` (default), ``ppo``, ``pql`` or ``dexpbt``.
        **overrides: any dataclass field overrides.

    Returns:
        A populated config dataclass instance.
    """
    if task not in TASK_CONFIGS:
        raise KeyError(
            f"Unknown task '{task}'. Available: {sorted(TASK_CONFIGS)}"
        )
    base_cls = TASK_CONFIGS[task]

    if algorithm == "sapg":
        cfg = base_cls()
    elif algorithm in BASELINE_CONFIGS:
        # start from the task config, then apply baseline defaults
        cfg = base_cls()
        baseline = BASELINE_CONFIGS[algorithm]()
        for f in baseline.__dataclass_fields__:
            setattr(cfg, f, getattr(baseline, f))
        # keep task-specific network/optimizer settings
        task_defaults = base_cls()
        for f in ("actor_hidden_dims", "critic_hidden_dims", "use_lstm",
                  "lstm_hidden_size", "lstm_num_layers", "lstm_seq_len",
                  "learning_rate", "num_mini_epochs", "clip_eps",
                  "horizon", "action_dim", "task_family"):
            setattr(cfg, f, getattr(task_defaults, f))
    else:
        raise KeyError(
            f"Unknown algorithm '{algorithm}'. Available: sapg, "
            f"{sorted(BASELINE_CONFIGS)}"
        )

    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Config has no field '{k}'")
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Expected results (Table 1) -- used by experiments/plot.py for validation
# ---------------------------------------------------------------------------
# metric: successes/episode for AllegroKuka, episode reward for hands
EXPECTED_RESULTS = {
    "sapg_coef0": {
        "allegrohand": (1.23e4, 3.29e2),
        "shadowhand": (1.17e4, 2.64e2),
        "regrasping": (35.7, 1.46),
        "throw": (23.7, 0.74),
        "reorientation": (33.2, 4.20),
    },
    "sapg_coef0005": {
        "allegrohand": (9.14e3, 8.38e2),
        "shadowhand": (1.28e4, 2.80e2),
        "regrasping": (33.4, 2.25),
        "throw": (18.7, 0.43),
        "reorientation": (38.6, 0.63),
    },
    "ppo": {
        "regrasping": (1.25, 1.15),
        "throw": (16.8, 0.48),
        "reorientation": (2.85, 0.05),
    },
    "dexpbt": {
        "regrasping": (31.9, 2.26),
        "throw": (19.2, 1.07),
        "reorientation": (23.2, 4.86),
    },
    "pql": {
        "regrasping": (2.73, 0.02),
        "throw": (2.62, 0.08),
        "reorientation": (1.66, 0.11),
    },
}
