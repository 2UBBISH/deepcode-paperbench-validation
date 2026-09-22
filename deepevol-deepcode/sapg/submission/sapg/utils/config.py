"""Configuration utilities for SAPG.

Hyperparameter groups follow Appendix B (Tables 2-4) and Section 5.2 of the paper
"SAPG: Split and Aggregate Policy Gradients".  Where the paper is silent we use
sensible defaults (Adam betas 0.9/0.999, eps 1e-8) as noted in the reproduction plan.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import yaml


# --------------------------------------------------------------------------------------
# Paper constants (Section 5.2)
# --------------------------------------------------------------------------------------
TOTAL_ENVS = 24576          # N
NUM_POLICIES = 6            # M
TOTAL_TRANSITIONS = int(2e10)  # training budget quoted in Section 5.2


def _allegro_kuka_defaults() -> Dict[str, Any]:
    """Table 2: AllegroKuka tasks (Regrasping / Throw / Reorientation)."""
    return dict(
        task_group="allegro_kuka",
        gamma=0.99,
        tau=0.95,               # interpreted as the GAE lambda (paper never defines tau)
        learning_rate=1e-4,
        kl_threshold=0.016,     # KL threshold for LR update
        grad_norm=1.0,
        entropy_coefficient=0.0,
        clip_epsilon=0.1,
        minibatch_size_multiplier=4,   # mini-batch size = num_envs * 4
        critic_coefficient=4.0,        # lambda'
        horizon_length=16,             # stated in Section 5.2 for all tasks
        lstm_sequence_length=16,
        bounds_loss_coefficient=1e-4,
        mini_epochs=2,
        # architecture (Appendix B.1)
        actor_mlp_units=(768, 512, 256),
        actor_activation="elu",
        use_lstm=True,
        lstm_hidden_size=768,
        lstm_num_layers=1,
        # Section 4.4 / 5.2: phi_j in R^32 for the complex environments
        phi_dim=32,
        # Appendix B.3 note: in entropy experiments each block has its own learnable sigma
        per_block_sigma=False,
        recurrent=True,
    )


def _shadow_hand_defaults() -> Dict[str, Any]:
    """Table 3: Shadow Hand."""
    base = _allegro_kuka_defaults()
    base.update(
        dict(
            task_group="shadow_hand",
            learning_rate=5e-4,
            horizon_length=8,
            mini_epochs=5,
            actor_mlp_units=(512, 512, 256, 128),
            use_lstm=False,
            lstm_hidden_size=0,
            lstm_num_layers=0,
            phi_dim=16,   # Section 5.2: phi_j in R^16 for ShadowHand and AllegroHand
            recurrent=False,
        )
    )
    return base


def _allegro_hand_defaults() -> Dict[str, Any]:
    """Table 4: Allegro Hand."""
    base = _allegro_kuka_defaults()
    base.update(
        dict(
            task_group="allegro_hand",
            learning_rate=5e-4,
            clip_epsilon=0.2,
            horizon_length=8,
            mini_epochs=5,
            actor_mlp_units=(512, 256, 128),
            use_lstm=False,
            lstm_hidden_size=0,
            lstm_num_layers=0,
            phi_dim=16,
            recurrent=False,
        )
    )
    return base


TASK_DEFAULTS = {
    "allegro_kuka": _allegro_kuka_defaults,
    "shadow_hand": _shadow_hand_defaults,
    "allegro_hand": _allegro_hand_defaults,
}

# Per-task overrides that are not part of the hyperparameter tables
TASK_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Curriculum / reward / observation specification, Appendix A
    "regrasping": dict(task_group="allegro_kuka", env_name="regrasping",
                       goal_dim=3, success_tolerance=0.075, tolerance_min=0.01,
                       success_hold_steps=30, use_curriculum=True,
                       obs_dim=23 + 23 + 7 + 3 + 3 + 3 + 1),
    "regrasp": dict(task_group="allegro_kuka", env_name="regrasping",
                    goal_dim=3, success_tolerance=0.075, tolerance_min=0.01,
                    success_hold_steps=30, use_curriculum=True,
                    obs_dim=23 + 23 + 7 + 3 + 3 + 3 + 1),
    "throw": dict(task_group="allegro_kuka", env_name="throw",
                  goal_dim=3, success_tolerance=0.075, tolerance_min=0.01,
                  success_hold_steps=30, use_curriculum=True,
                  obs_dim=23 + 23 + 7 + 3 + 3 + 3 + 1),
    "reorientation": dict(task_group="allegro_kuka", env_name="reorientation",
                          goal_dim=7, success_tolerance=0.075, tolerance_min=0.01,
                          success_hold_steps=30, use_curriculum=True,
                          obs_dim=23 + 23 + 7 + 3 + 3 + 7 + 1),
    "shadow_hand": dict(task_group="shadow_hand", env_name="shadow_hand",
                        goal_dim=4, obs_dim=24 + 24 + 7 + 3 + 3 + 4,
                        use_curriculum=False),
    "allegro_hand": dict(task_group="allegro_hand", env_name="allegro_hand",
                         goal_dim=7, obs_dim=16 + 16 + 7 + 3 + 3 + 7,
                         use_curriculum=False),
}


@dataclass
class SAPGConfig:
    """Complete configuration object.

    Attributes mirror the paper's hyperparameter tables and Algoritm-1 loop.
    """

    # ---- task -------------------------------------------------------------
    task: str = "regrasping"
    task_group: str = "allegro_kuka"
    env_name: str = "regrasping"

    # ---- algorithm --------------------------------------------------------
    method: str = "sapg"            # sapg | ppo | pql | dexpbt
    num_envs: int = TOTAL_ENVS      # N
    num_policies: int = NUM_POLICIES  # M
    leader_index: int = 1           # Section 4.3: leader is policy i = 1
    aggregation: str = "leader_follower"   # leader_follower | symmetric
    off_policy_weight: float = 1.0  # lambda (Eq. 10 of the plan / Eq. 2 of section 4.1)
    subsample_off_policy: bool = True      # |D'_1| = |D_1| (Sections 4.2, 4.3)
    entropy_coefficient: float = 0.0       # sigma (Section 4.5, Eq. 10)
    normalize_advantage: bool = True
    normalize_value: bool = True
    learn_sigma: bool = True
    target_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ---- optimisation -----------------------------------------------------
    learning_rate: float = 1e-4
    critic_learning_rate: Optional[float] = None
    optimizer: str = "adam"
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    kl_threshold: float = 0.016
    grad_norm: float = 1.0
    lr_schedule: str = "adaptive"   # adaptive KL-based schedule (Appendix B)
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    bounds_loss_coefficient: float = 1e-4
    critic_coefficient: float = 4.0     # lambda'

    # ---- PPO / data -------------------------------------------------------
    gamma: float = 0.99
    tau: float = 0.95
    clip_epsilon: float = 0.1
    horizon_length: int = 16
    mini_epochs: int = 2
    minibatch_size_multiplier: int = 4
    lstm_sequence_length: int = 16
    e_clip: float = 0.2
    max_epochs: Optional[int] = None
    total_transitions: int = TOTAL_TRANSITIONS

    # ---- architecture -----------------------------------------------------
    actor_mlp_units: tuple = (768, 512, 256)
    critic_mlp_units: tuple = (768, 512, 256)
    actor_activation: str = "elu"
    use_lstm: bool = True
    lstm_hidden_size: int = 768
    lstm_num_layers: int = 1
    phi_dim: int = 32
    obs_dim: int = 60
    action_dim: int = 23
    action_scale: float = 1.0
    per_block_sigma: bool = False
    recurrent: bool = True
    random_phi: bool = False      # diversity analysis: untrained/random policy

    # ---- logging ----------------------------------------------------------
    seed: int = 0
    log_dir: str = "runs"
    log_interval: int = 1
    save_interval: int = 100
    device: str = "cuda:0"
    num_threads: int = 1

    # ----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SAPGConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in d.items() if k in fields}
        cfg = cls(**clean)
        cfg._apply_derived()
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "SAPGConfig":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=True)

    def _apply_derived(self) -> None:
        if self.critic_learning_rate is None:
            self.critic_learning_rate = self.learning_rate


def build_config(task: str = "regrasping", **overrides) -> SAPGConfig:
    """Build a configuration for ``task`` augmented with per-task defaults."""
    key = task.lower()
    if key in TASK_OVERRIDES:
        base = TASK_DEFAULTS[TASK_OVERRIDES[key]["task_group"]]()
        base.update(TASK_OVERRIDES[key])
        base["task"] = key
    elif key in TASK_DEFAULTS:
        base = TASK_DEFAULTS[key]()
        base["task"] = key
        base.setdefault("env_name", key)
    else:
        raise ValueError(f"Unknown task {task!r}. Expected one of "
                         f"{sorted(set(list(TASK_DEFAULTS) + list(TASK_OVERRIDES)))}")
    base.update(overrides)
    cfg = SAPGConfig.from_dict(base)
    cfg._apply_derived()
    return cfg


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
