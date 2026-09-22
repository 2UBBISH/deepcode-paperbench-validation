"""PPO + Random Network Distillation (RND) for Montezuma's Revenge (Appendix B.2).

This module implements the exploration-augmented PPO learner that is used to obtain
the *pre-trained* agent ``M1`` (episode cumulative reward ~7000) and, later on, to
fine-tune ``M2`` on the whole game (possibly with an auxiliary behavioral-cloning
loss, see ``src/montezuma/m2_bc.py`` and ``src/montezuma/train_montezuma.py``).

Paper specification (Section 3 / Appendix B.2, Table 2)
------------------------------------------------------
* PPO + RND (Burda et al., 2018), reference PyTorch implementation by ``jcwleo``.
* Two networks: a *randomly initialized* **target** network and a trainable
  **prediction** network.  Both map an observation to a **512-dimensional** vector.
  The predictor is trained to regress the target outputs; the prediction error is
  used as an intrinsic reward signal (novel states -> large error -> exploration).
* Table 2 hyperparameters: ``MaxStepPerEpisode=4500``, ``ExtCoef=2.0``,
  ``LearningRate=1e-4``, ``NumEnv=128``, ``NumStep=128``, ``Gamma=0.999``,
  ``IntGamma=0.99``, ``Lambda=0.95``, ``StableEps=1e-8``, ``StateStackSize=4``,
  ``PreProcHeight=84``, ``ProProcWidth=84``, ``UseGAE=True``, ``UseNorm=False``,
  ``UseNoisyNet=False``, ``ClipGradNorm=0.5``, ``Entropy=0.001``, ``Epoch=4``,
  ``MiniBatch=4``, ``PPOEps=0.1``, ``IntCoef=1.0``, ``StickyAction=True``,
  ``ActionProb=0.25``, ``UpdateProportion=0.25``, ``LifeDone=False``,
  ``ObsNormStep=50``.
* Room completion (used for the Figure 6 / Figure 17-19 analyses): earning a coin,
  acquiring a new item, or exiting through a different passage.  Room 7 is the FAR
  boundary of the main-text experiments.

The learner is deliberately written so that the *entire* M1 -> M2 pipeline shares
one code path: ``PPORNDAgent`` is environment-agnostic (it only requires a
``MontezumaVecEnv``-like object exposing ``reset``/``step``) and accepts an
arbitrary differentiable ``aux_loss_fn(policy)`` which the fine-tuning experiments
use to add the behavioral-cloning retention term.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.montezuma.env import (
    ACTION_PROB,
    FRAME_STACK,
    MAX_STEPS_PER_EPISODE,
    NUM_ACTIONS,
    OBS_SHAPE,
    ROOM7,
    MontezumaVecEnv,
    make_env,
)
from src.montezuma.model import (
    FEATURE_DIM,
    RND_FEATURE_DIM,
    TABLE2_DEFAULTS,
    IntrinsicRewardNormalizer,
    MontezumaModelConfig,
    ObsNormalizer,
    PolicyNetwork,
    RNDModel,
    RunningMeanStd,
    build_models,
    load_model_state,
    save_model_state,
)

try:  # torch is required at runtime, optional at import time (docs / unit tests)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required to use src.montezuma.ppo_rnd")


__all__ = [
    "PPORNDConfig",
    "RolloutStorage",
    "PPORNDAgent",
    "PPORNDTrainer",
    "compute_gae",
    "train_ppo_rnd",
    "DEFAULT_TOTAL_STEPS",
    "ROOM7_EVERY",
]

#: Training budget target used for M1 (paper: "trained until ~7000 episode return").
DEFAULT_TOTAL_STEPS = 200_000_000
#: Room-7 success rate is logged every 5M steps (validation protocol of the plan).
ROOM7_EVERY = 5_000_000


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class PPORNDConfig:
    """All Table 2 hyperparameters plus pipeline knobs.

    The defaults are exactly the Table 2 (Appendix B.2) values.  ``from_config``
    accepts the project's ``Config``/YAML mapping and overlays the values found in
    the ``ppo`` / ``rnd`` / ``env`` / ``finetune`` blocks.
    """

    # --- Table 2 ---------------------------------------------------------------
    max_steps_per_episode: int = 4500            # MaxStepPerEpisode
    ext_coef: float = 2.0                        # ExtCoef
    learning_rate: float = 1e-4                  # LearningRate
    num_env: int = 128                           # NumEnv
    num_step: int = 128                          # NumStep
    gamma: float = 0.999                         # Gamma
    int_gamma: float = 0.99                      # IntGamma
    lam: float = 0.95                            # Lambda
    int_lam: Optional[float] = None              # defaults to lam
    stable_eps: float = 1e-8                     # StableEps
    state_stack_size: int = 4                    # StateStackSize
    preproc_height: int = 84                     # PreProcHeight
    preproc_width: int = 84                      # ProProcWidth
    use_gae: bool = True                         # UseGAE
    use_norm: bool = False                       # UseNorm
    use_noisy_net: bool = False                  # UseNoisyNet
    clip_grad_norm: float = 0.5                  # ClipGradNorm
    entropy: float = 0.001                       # Entropy
    epoch: int = 4                               # Epoch
    mini_batch: int = 4                          # MiniBatch (# of mini-batches)
    ppo_eps: float = 0.1                         # PPOEps
    int_coef: float = 1.0                        # IntCoef
    sticky_action: bool = True                   # StickyAction
    action_prob: float = 0.25                    # ActionProb
    update_proportion: float = 0.25              # UpdateProportion
    life_done: bool = False                      # LifeDone
    obs_norm_step: int = 50                      # ObsNormStep

    # --- model / RND ----------------------------------------------------------
    feature_dim: int = FEATURE_DIM
    rnd_feature_dim: int = RND_FEATURE_DIM
    hidden_dim: Optional[int] = None
    value_loss_coef: float = 0.5
    normalize_advantages: bool = True
    normalize_intrinsic_rewards: Optional[bool] = None   # None -> follow ``use_norm``
    normalize_ext_returns: bool = False
    obs_normalization: bool = False
    max_grad_norm: Optional[float] = None                # alias of clip_grad_norm

    # --- pipeline -------------------------------------------------------------
    num_updates: int = 0
    total_steps: int = DEFAULT_TOTAL_STEPS
    eval_every: int = 5_000_000
    room7_every: int = ROOM7_EVERY
    eval_episodes: int = 100
    save_every: int = 25_000_000
    log_every: int = 10_000
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    method: str = "none"                         # scratch | none | bc | ewc
    kl_weight: float = 1.0                       # Figure 13 sweep coefficient
    bc_dataset: Optional[str] = None
    seed: int = 0
    device: str = "cpu"
    stub: bool = False
    deterministic: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.int_lam is None:
            self.int_lam = self.lam
        if self.max_grad_norm is not None:
            self.clip_grad_norm = float(self.max_grad_norm)
        if self.normalize_intrinsic_rewards is None:
            self.normalize_intrinsic_rewards = bool(self.use_norm)

    # -- helpers ---------------------------------------------------------------
    def with_overrides(self, **overrides: Any) -> "PPORNDConfig":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None or k in self.__dataclass_fields__})

    @property
    def batch_size(self) -> int:
        """PPO batch size = NumStep * NumEnv (128 * 128 = 16384 by default)."""
        return int(self.num_step) * int(self.num_env)

    @property
    def minibatch_size(self) -> int:
        """MiniBatch in Table 2 counts *mini-batches* (4 -> 4096 samples each)."""
        n = max(1, int(self.mini_batch))
        return max(1, self.batch_size // n)

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return (int(self.state_stack_size), int(self.preproc_height), int(self.preproc_width))

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "PPORNDConfig":
        """Build a config from ``configs/montezuma.yaml``-style objects or a dict."""
        cfg = cfg if isinstance(cfg, Mapping) else getattr(cfg, "to_dict", lambda: {})()
        if not isinstance(cfg, Mapping):
            cfg = {}

        def _get(block: str, *names: str, default: Any = None) -> Any:
            sub = cfg.get(block) or {}
            if not isinstance(sub, Mapping):
                sub = {}
            for name in names:
                if name in sub:
                    return sub[name]
                if name in cfg:
                    return cfg[name]
            return default

        ppo = dict(cfg.get("ppo") or {}) if isinstance(cfg.get("ppo"), Mapping) else {}
        rnd = dict(cfg.get("rnd") or {}) if isinstance(cfg.get("rnd"), Mapping) else {}
        env = dict(cfg.get("env") or {}) if isinstance(cfg.get("env"), Mapping) else {}
        model = dict(cfg.get("model") or {}) if isinstance(cfg.get("model"), Mapping) else {}
        finetune = dict(cfg.get("finetune") or {}) if isinstance(cfg.get("finetune"), Mapping) else {}

        def pick(*sources: Mapping, names: Sequence[str], default: Any = None) -> Any:
            for src in sources:
                if not isinstance(src, Mapping):
                    continue
                for name in names:
                    if name in src and src[name] is not None:
                        return src[name]
            return default

        kwargs: Dict[str, Any] = dict(
            max_steps_per_episode=pick(env, ppo, names=("max_steps_per_episode", "max_episode_steps", "max_steps"), default=cls.max_steps_per_episode),
            ext_coef=pick(rnd, ppo, names=("ext_coef", "extrinsic_coef"), default=cls.ext_coef),
            int_coef=pick(rnd, ppo, names=("int_coef", "intrinsic_coef"), default=cls.int_coef),
            learning_rate=pick(ppo, names=("learning_rate", "lr"), default=cls.learning_rate),
            num_env=pick(env, ppo, names=("num_env", "num_envs", "n_envs"), default=cls.num_env),
            num_step=pick(ppo, names=("num_step", "num_steps", "rollout_length", "n_steps"), default=cls.num_step),
            gamma=pick(ppo, names=("gamma", "discount"), default=cls.gamma),
            int_gamma=pick(rnd, ppo, names=("int_gamma", "intrinsic_gamma"), default=cls.int_gamma),
            lam=pick(ppo, names=("lam", "lambda", "gae_lambda"), default=cls.lam),
            int_lam=pick(rnd, ppo, names=("int_lam",), default=None),
            stable_eps=pick(ppo, names=("stable_eps", "eps"), default=cls.stable_eps),
            use_gae=bool(pick(ppo, names=("use_gae",), default=cls.use_gae)),
            use_norm=bool(pick(ppo, rnd, names=("use_norm",), default=cls.use_norm)),
            use_noisy_net=bool(pick(ppo, names=("use_noisy_net",), default=cls.use_noisy_net)),
            clip_grad_norm=pick(ppo, names=("clip_grad_norm", "max_grad_norm", "grad_clip"), default=cls.clip_grad_norm),
            entropy=pick(ppo, names=("entropy", "entropy_coef", "entropy_cost"), default=cls.entropy),
            epoch=pick(ppo, names=("epoch", "epochs", "num_epochs"), default=cls.epoch),
            mini_batch=pick(ppo, names=("mini_batch", "minibatch", "num_minibatches"), default=cls.mini_batch),
            ppo_eps=pick(ppo, names=("ppo_eps", "clip_ratio", "clip"), default=cls.ppo_eps),
            sticky_action=bool(pick(env, ppo, names=("sticky_action", "sticky"), default=cls.sticky_action)),
            action_prob=pick(env, ppo, names=("action_prob",), default=cls.action_prob),
            update_proportion=pick(rnd, ppo, names=("update_proportion",), default=cls.update_proportion),
            life_done=bool(pick(env, names=("life_done",), default=cls.life_done)),
            obs_norm_step=pick(model, ppo, names=("obs_norm_step",), default=cls.obs_norm_step),
            feature_dim=pick(model, names=("feature_dim",), default=cls.feature_dim),
            rnd_feature_dim=pick(rnd, model, names=("feature_dim", "rnd_feature_dim", "output_size"), default=cls.rnd_feature_dim),
            hidden_dim=pick(model, rnd, names=("hidden_dim",), default=cls.hidden_dim),
            value_loss_coef=pick(ppo, names=("value_loss_coef", "vf_coef"), default=cls.value_loss_coef),
            normalize_advantages=bool(pick(ppo, names=("normalize_advantages",), default=cls.normalize_advantages)),
            normalize_intrinsic_rewards=pick(rnd, names=("normalize_intrinsic_rewards", "normalize_intrinsic"), default=cls.normalize_intrinsic_rewards),
            normalize_ext_returns=bool(pick(ppo, names=("normalize_ext_returns",), default=cls.normalize_ext_returns)),
            obs_normalization=bool(pick(model, names=("obs_normalization", "normalize_obs"), default=cls.obs_normalization)),
            total_steps=pick(finetune, cfg, names=("total_steps", "num_steps_total", "max_env_steps"), default=cls.total_steps),
            eval_every=pick(cfg, names=("eval_every",), default=cls.eval_every),
            room7_every=pick(cfg, names=("room7_every",), default=cls.room7_every),
            eval_episodes=pick(cfg, names=("eval_episodes", "num_eval_episodes"), default=cls.eval_episodes),
            save_every=pick(cfg, names=("save_every",), default=cls.save_every),
            log_every=pick(cfg, names=("log_every",), default=cls.log_every),
            output_dir=pick(cfg, names=("output_dir", "log_dir", "results_dir"), default=cls.output_dir),
            checkpoint=pick(cfg, cfg.get("pretrain") if isinstance(cfg.get("pretrain"), Mapping) else {}, names=("checkpoint",), default=cls.checkpoint),
            method=pick(cfg, names=("method", "retention_method"), default=cls.method),
            kl_weight=pick(cfg, names=("kl_weight", "bc_kl_weight"), default=cls.kl_weight),
            bc_dataset=pick(cfg, cfg.get("bc") if isinstance(cfg.get("bc"), Mapping) else {}, names=("dataset", "bc_dataset", "trajectories"), default=cls.bc_dataset),
            seed=pick(cfg, names=("seed",), default=cls.seed),
            device=pick(cfg, names=("device",), default=cls.device),
            stub=bool(pick(cfg, names=("stub",), default=cls.stub)),
            deterministic=bool(pick(cfg, names=("deterministic",), default=cls.deterministic)),
        )
        # explicit overrides (constructor kwargs) win
        for key, value in overrides.items():
            if value is None:
                continue
            if key in cls.__dataclass_fields__:
                kwargs[key] = value
            elif key == "lr":
                kwargs["learning_rate"] = value
            elif key in ("num_envs", "n_envs"):
                kwargs["num_env"] = value
            elif key in ("rollout_length", "n_steps", "num_steps"):
                kwargs["num_step"] = value
        return cls(**kwargs)


# --------------------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------------------
def compute_gae(
    rewards,
    values,
    masks,
    gamma: float,
    lam: float,
    last_value,
    use_gae: bool = True,
    normalize: bool = False,
    eps: float = 1e-8,
):
    """Generalized Advantage Estimation (as in the RND reference implementation).

    ``rewards``/``values``/``masks`` are expected to be ``(num_step, num_env)``
    tensors; ``last_value`` is ``(num_env,)``.  Returns ``(advantages, returns)``.
    """
    _require_torch()
    num_step, num_env = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_env, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(num_step)):
        if t == num_step - 1:
            next_value = last_value
            next_mask = masks[t]
        else:
            next_value = values[t + 1]
            next_mask = masks[t + 1]
        delta = rewards[t] + gamma * next_value * next_mask - values[t]
        if use_gae:
            gae = delta + gamma * lam * next_mask * gae
        else:
            gae = delta
        advantages[t] = gae
    returns = advantages + values
    if normalize:
        std = advantages.std() + eps
        advantages = advantages / std
    return advantages, returns


# --------------------------------------------------------------------------------------
# Rollout storage
# --------------------------------------------------------------------------------------
class RolloutStorage:
    """Fixed-horizon rollout buffer storing extrinsic *and* intrinsic streams.

    Shapes follow the RND reference implementation: a buffer of ``num_step + 1``
    timesteps (the extra slot holds ``obs[t+1]`` for bootstrapping).
    """

    def __init__(
        self,
        num_step: int,
        num_env: int,
        obs_shape: Sequence[int] = OBS_SHAPE,
        device: str = "cpu",
        num_actions: int = NUM_ACTIONS,
        use_gae: bool = True,
        gamma: float = 0.999,
        lam: float = 0.95,
        int_gamma: float = 0.99,
        int_lam: Optional[float] = None,
        normalize_advantages: bool = True,
        eps: float = 1e-8,
    ) -> None:
        _require_torch()
        self.num_step = int(num_step)
        self.num_env = int(num_env)
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.device = device
        self.num_actions = int(num_actions)
        self.use_gae = bool(use_gae)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.int_gamma = float(int_gamma)
        self.int_lam = float(lam if int_lam is None else int_lam)
        self.normalize_advantages = bool(normalize_advantages)
        self.eps = float(eps)

        f = torch.float32
        self.obs = torch.zeros(self.num_step + 1, self.num_env, *self.obs_shape, dtype=f, device=device)
        self.ext_rewards = torch.zeros(self.num_step, self.num_env, dtype=f, device=device)
        self.int_rewards = torch.zeros(self.num_step, self.num_env, dtype=f, device=device)
        self.int_raw = torch.zeros(self.num_step, self.num_env, dtype=f, device=device)
        self.masks = torch.ones(self.num_step + 1, self.num_env, dtype=f, device=device)
        self.actions = torch.zeros(self.num_step, self.num_env, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(self.num_step, self.num_env, dtype=f, device=device)
        self.values = torch.zeros(self.num_step + 1, self.num_env, dtype=f, device=device)
        self.entropies = torch.zeros(self.num_step, self.num_env, dtype=f, device=device)
        self.ext_returns = torch.zeros(self.num_step + 1, self.num_env, dtype=f, device=device)
        self.int_returns = torch.zeros(self.num_step + 1, self.num_env, dtype=f, device=device)
        self.advantages = torch.zeros(self.num_step, self.num_env, dtype=f, device=device)
        self.int_advantages = torch.zeros(self.num_step, self.num_env, dtype=f, device=device)
        self.step = 0
        self.computed = False

    # -- insertion -------------------------------------------------------------
    def reset(self, obs) -> None:
        self.obs[0] = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        self.step = 0
        self.computed = False

    def to(self, device: str) -> "RolloutStorage":
        self.device = device
        for name in (
            "obs", "ext_rewards", "int_rewards", "int_raw", "masks", "actions",
            "log_probs", "values", "entropies", "ext_returns", "int_returns",
            "advantages", "int_advantages",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def insert(
        self,
        obs,
        actions,
        log_probs,
        values,
        rewards,
        masks,
        entropies=None,
        int_rewards=None,
        int_raw=None,
    ) -> None:
        _require_torch()
        t = self.step
        self.obs[t + 1].copy_(obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32))
        self.actions[t].copy_(actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions, dtype=torch.long))
        self.log_probs[t].copy_(log_probs if isinstance(log_probs, torch.Tensor) else torch.as_tensor(log_probs, dtype=torch.float32))
        self.values[t].copy_(values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float32))
        self.ext_rewards[t].copy_(rewards if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards, dtype=torch.float32))
        if int_rewards is not None:
            self.int_rewards[t].copy_(int_rewards if isinstance(int_rewards, torch.Tensor) else torch.as_tensor(int_rewards, dtype=torch.float32))
        if int_raw is not None:
            self.int_raw[t].copy_(int_raw if isinstance(int_raw, torch.Tensor) else torch.as_tensor(int_raw, dtype=torch.float32))
        self.masks[t].copy_(masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks, dtype=torch.float32))
        if entropies is not None:
            self.entropies[t].copy_(entropies if isinstance(entropies, torch.Tensor) else torch.as_tensor(entropies, dtype=torch.float32))
        self.step += 1

    def set_last_value(self, value) -> None:
        if value is not None:
            self.values[self.num_step].copy_(
                value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
            )

    # -- advantage computation -------------------------------------------------
    def compute_returns(self, last_value=None, last_int_value=None, normalize_ext_returns: bool = False):
        _require_torch()
        self.set_last_value(last_value)
        if last_int_value is not None:
            int_last = last_int_value
        else:
            int_last = torch.zeros(self.num_env, dtype=torch.float32, device=self.device)

        # extrinsic stream (Gamma = 0.999)
        adv, ret = compute_gae(
            self.ext_rewards,
            self.values[: self.num_step],
            self.masks[: self.num_step],
            self.gamma,
            self.lam,
            last_value if last_value is not None else self.values[self.num_step],
            use_gae=self.use_gae,
            normalize=normalize_ext_returns,
            eps=self.eps,
        )
        self.advantages = adv
        self.ext_returns[: self.num_step] = ret

        # intrinsic stream (IntGamma = 0.99, same Lambda = 0.95)
        zero_values = torch.zeros_like(self.values[: self.num_step])
        int_adv, int_ret = compute_gae(
            self.int_rewards,
            zero_values,
            self.masks[: self.num_step],
            self.int_gamma,
            self.int_lam,
            int_last,
            use_gae=self.use_gae,
            normalize=False,
            eps=self.eps,
        )
        self.int_advantages = int_adv
        self.int_returns[: self.num_step] = int_ret
        self.computed = True
        return self.advantages, self.ext_returns

    def combined_advantages(self, ext_coef: float = 2.0, int_coef: float = 1.0) -> Any:
        """``ExtCoef * normalized ext. advantage + IntCoef * normalized int. advantage``."""
        _require_torch()
        ext_adv = self.advantages
        int_adv = self.int_advantages
        if self.normalize_advantages:
            ext_adv = ext_adv / (ext_adv.std() + self.eps)
            int_adv = int_adv / (int_adv.std() + self.eps)
        return float(ext_coef) * ext_adv + float(int_coef) * int_adv

    # -- iteration -------------------------------------------------------------
    def flatten(self, num_minibatches: int, shuffle: bool = True, generator=None):
        """Yield mini-batches of flattened rollout tensors."""
        _require_torch()
        batch = self.num_step * self.num_env
        obs = self.obs[: self.num_step].reshape(batch, *self.obs_shape)
        actions = self.actions.reshape(batch)
        log_probs = self.log_probs.reshape(batch)
        values = self.values[: self.num_step].reshape(batch)
        entropies = self.entropies.reshape(batch)
        returns = self.ext_returns[: self.num_step].reshape(batch)
        ext_adv = self.advantages.reshape(batch)
        int_adv = self.int_advantages.reshape(batch)
        int_raw = self.int_raw.reshape(batch)

        order = torch.randperm(batch, generator=generator) if shuffle else torch.arange(batch)
        size = max(1, batch // max(1, int(num_minibatches)))
        for start in range(0, batch, size):
            idx = order[start : start + size]
            yield {
                "obs": obs[idx],
                "actions": actions[idx],
                "old_log_probs": log_probs[idx],
                "old_values": values[idx],
                "old_entropies": entropies[idx],
                "returns": returns[idx],
                "ext_advantages": ext_adv[idx],
                "int_advantages": int_adv[idx],
                "int_raw": int_raw[idx],
            }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "obs": self.obs, "actions": self.actions, "log_probs": self.log_probs,
            "values": self.values, "ext_rewards": self.ext_rewards, "int_rewards": self.int_rewards,
            "masks": self.masks, "step": self.step,
            "num_step": self.num_step, "num_env": self.num_env, "obs_shape": self.obs_shape,
        }


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
class PPORNDAgent:
    """PPO learner with RND exploration bonus.

    Parameters
    ----------
    config : PPORNDConfig
        Table 2 hyperparameters.
    policy, rnd : optional
        Pre-built networks (otherwise ``build_models`` is used).
    aux_loss_fn : callable, optional
        ``aux_loss_fn(policy=..., obs=..., ...) -> Tensor`` added to the PPO actor
        loss.  Used by the M2 fine-tuning experiments for the behavioral-cloning
        retention term (Figure 13 sweeps its weight, ``config.kl_weight``).
    """

    def __init__(
        self,
        config: Optional[PPORNDConfig] = None,
        policy: Optional[Any] = None,
        rnd: Optional[Any] = None,
        obs_shape: Optional[Sequence[int]] = None,
        num_actions: Optional[int] = None,
        device: Optional[str] = None,
        aux_loss_fn: Optional[Callable[..., Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        _require_torch()
        self.config = config or PPORNDConfig()
        self.device = device or self.config.device
        self.obs_shape = tuple(int(x) for x in (obs_shape or self.config.observation_shape))
        self.num_actions = int(num_actions or NUM_ACTIONS)
        self.seed = int(self.config.seed if seed is None else seed)
        self.aux_loss_fn = aux_loss_fn

        models = build_models(
            cfg=None,
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            device=self.device,
            feature_dim=self.config.feature_dim,
            rnd_feature_dim=self.config.rnd_feature_dim,
            hidden_dim=self.config.hidden_dim,
            shared_backbone=True,
            normalize_obs=False,
            update_proportion=self.config.update_proportion,
        )
        self.policy: PolicyNetwork = policy if policy is not None else models["policy"]
        self.rnd: RNDModel = rnd if rnd is not None else models["rnd"]
        self.policy.to(self.device)
        self.rnd.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.rnd.parameters()),
            lr=float(self.config.learning_rate),
            eps=float(self.config.stable_eps),
        )
        # RND intrinsic-reward normalization (RewardForwardFilter + running std)
        self.int_normalizer = IntrinsicRewardNormalizer(
            num_env=int(self.config.num_env),
            int_gamma=float(self.config.int_gamma),
            enabled=bool(self.config.normalize_intrinsic_rewards),
        )
        self.return_rms = RunningMeanStd(shape=(), epsilon=float(self.config.stable_eps))
        self.obs_normalizer = ObsNormalizer(shape=self.obs_shape, enabled=bool(self.config.obs_normalization))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

        self.num_updates = 0
        self.num_env_steps = 0
        self.history: List[Dict[str, Any]] = []

    # -- rollout helpers -------------------------------------------------------
    def _to_tensor(self, obs) -> Any:
        """Convert env observations to a ``(num_env, C, H, W)`` float tensor."""
        if isinstance(obs, torch.Tensor):
            t = obs.to(self.device)
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.dtype == torch.uint8:
                t = t.float().div(255.0)
            elif t.dtype != torch.float32:
                t = t.float()
            if float(t.max()) > 1.5:
                t = t / 255.0
            return t
        arr = np.asarray(obs)
        if arr.ndim == 3:
            arr = arr[None]
        t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        if arr.dtype == np.uint8 or float(arr.max() if arr.size else 0.0) > 1.5:
            t = t / 255.0
        if t.ndim == 4 and t.shape[-1] in (1, 3, 4) and t.shape[1] not in (1, 3, 4):
            # HWC -> CHW
            t = t.permute(0, 3, 1, 2).contiguous()
        return t

    def intrinsic_reward(self, obs_tensor, normalize: Optional[bool] = None) -> Tuple[Any, Any]:
        """Raw RND prediction error and (optionally) its normalized version."""
        with torch.no_grad():
            raw = self.rnd.intrinsic_reward(obs_tensor)
        raw = raw.reshape(-1)
        do_norm = self.config.normalize_intrinsic_rewards if normalize is None else bool(normalize)
        normalized = self.int_normalizer.update(raw) if do_norm else raw
        return raw, normalized

    def act(self, obs_tensor, deterministic: bool = False):
        with torch.no_grad():
            logits, value = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            value = value.reshape(-1)
        return action, log_prob, entropy, value

    # -- update ----------------------------------------------------------------
    def compute_losses(self, batch: Mapping[str, Any], ext_coef: Optional[float] = None, int_coef: Optional[float] = None):
        """PPO surrogate loss + value loss + entropy + optional auxiliary (BC) loss."""
        cfg = self.config
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        old_values = batch["old_values"]

        log_probs, values, entropy = self.policy.pi_value(obs, actions, detach=False)
        values = values.reshape(-1)
        entropy = entropy.reshape(-1)

        adv = batch.get("combined_advantages")
        if adv is None:
            ext_adv = batch["ext_advantages"]
            int_adv = batch["int_advantages"]
            if cfg.normalize_advantages:
                ext_adv = ext_adv / (ext_adv.std() + cfg.stable_eps)
                int_adv = int_adv / (int_adv.std() + cfg.stable_eps)
            adv = float(cfg.ext_coef if ext_coef is None else ext_coef) * ext_adv + float(
                cfg.int_coef if int_coef is None else int_coef
            ) * int_adv

        ratio = torch.exp(log_probs - old_log_probs)
        eps = float(cfg.ppo_eps)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        if cfg.normalize_ext_returns and self.return_rms.count > 1:
            returns_norm = (returns - self.return_rms.mean) / (self.return_rms.std + cfg.stable_eps)
        else:
            returns_norm = returns
        value_clip = old_values + torch.clamp(values - old_values, -eps, eps)
        value_loss = 0.5 * torch.max(
            (returns_norm - values).pow(2), (returns_norm - value_clip).pow(2)
        ).mean()

        entropy_loss = -entropy.mean()

        # RND predictor loss (only a fraction ``UpdateProportion`` of the batch)
        mask = self.rnd.update_proportion_mask(obs.shape[0], cfg.update_proportion, device=self.device)
        rnd_loss = self.rnd.loss(obs, weights=mask, reduction="mean")

        aux_loss = None
        if self.aux_loss_fn is not None:
            aux_loss = self.aux_loss_fn(policy=self.policy, obs=obs, step=self.num_env_steps)
            if aux_loss is None:
                aux_loss = torch.zeros((), device=self.device)

        total = (
            policy_loss
            + float(cfg.value_loss_coef) * value_loss
            + float(cfg.entropy) * entropy_loss
            + rnd_loss
        )
        if aux_loss is not None:
            total = total + aux_loss

        info = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.mean().detach()),
            "rnd_loss": float(rnd_loss.detach()),
            "ratio_mean": float(ratio.mean().detach()),
            "clip_fraction": float(((ratio - 1.0).abs() > eps).float().mean().detach()),
        }
        if aux_loss is not None:
            info["aux_loss"] = float(aux_loss.detach())
        return total, info

    def update(self, storage: RolloutStorage, epochs: Optional[int] = None) -> Dict[str, float]:
        """Run ``Epoch`` PPO epochs over ``MiniBatch`` mini-batches (Table 2)."""
        cfg = self.config
        epochs = int(cfg.epoch if epochs is None else epochs)
        self.policy.train()
        self.rnd.train()
        agg: Dict[str, List[float]] = {}
        for _ in range(epochs):
            for batch in storage.flatten(int(cfg.mini_batch), shuffle=True, generator=self.generator):
                batch["combined_advantages"] = self._combined_for(batch)
                loss, info = self.compute_losses(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.rnd.parameters()),
                        float(cfg.clip_grad_norm),
                    )
                self.optimizer.step()
                for key, value in info.items():
                    agg.setdefault(key, []).append(value)
        self.num_updates += 1
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def _combined_for(self, batch: Mapping[str, Any]):
        cfg = self.config
        ext_adv = batch["ext_advantages"]
        int_adv = batch["int_advantages"]
        if cfg.normalize_advantages:
            ext_adv = ext_adv / (ext_adv.std() + cfg.stable_eps)
            int_adv = int_adv / (int_adv.std() + cfg.stable_eps)
        return float(cfg.ext_coef) * ext_adv + float(cfg.int_coef) * int_adv

    # -- persistence -----------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "rnd": self.rnd.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_updates": self.num_updates,
            "num_env_steps": self.num_env_steps,
            "config": self.config.to_dict(),
            "history": list(self.history),
        }

    def load_state_dict(self, state: Mapping[str, Any], load_optimizer: bool = True) -> "PPORNDAgent":
        policy = state.get("policy") or state.get("model_state") or state
        rnd = state.get("rnd")
        try:
            self.policy.load_state_dict(policy, strict=False)
        except Exception:
            pass
        if rnd is not None:
            try:
                self.rnd.load_state_dict(rnd, strict=False)
            except Exception:
                pass
        if load_optimizer and state.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        self.num_updates = int(state.get("num_updates", self.num_updates) or 0)
        self.num_env_steps = int(state.get("num_env_steps", self.num_env_steps) or 0)
        if state.get("history"):
            self.history = list(state["history"])
        return self

    def save(self, path: str) -> str:
        return save_model_state(path, policy=self.policy, rnd=self.rnd, extra={
            "optimizer": self.optimizer.state_dict(),
            "num_updates": self.num_updates,
            "num_env_steps": self.num_env_steps,
            "config": self.config.to_dict(),
            "history": list(self.history),
        })

    def load(self, path: str) -> "PPORNDAgent":
        state = load_model_state(path, policy=None, rnd=None, map_location=self.device)
        return self.load_state_dict(state)


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class PPORNDTrainer:
    """Drives PPO + RND training: rollout collection, updates, evaluation, logging.

    Works both for M1 (``method="scratch"``, purely extrinsic+intrinsic PPO) and for
    the M2 fine-tuning runs (``method`` in ``{"none", "bc", "ewc", "scratch"}`` with an
    optional auxiliary loss function).  The first half of the trainer mirrors the
    RND reference implementation (Burda et al., 2018 / jcwleo).
    """

    def __init__(
        self,
        config: Optional[PPORNDConfig] = None,
        env: Optional[Any] = None,
        agent: Optional[PPORNDAgent] = None,
        aux_loss_fn: Optional[Callable[..., Any]] = None,
        logger: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        _require_torch()
        self.config = config or PPORNDConfig()
        self.output_dir = output_dir or self.config.output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.env = env if env is not None else MontezumaVecEnv(
            num_envs=int(self.config.num_env), base_seed=int(self.config.seed), stub=bool(self.config.stub)
        )
        self.agent = agent or PPORNDAgent(
            config=self.config, aux_loss_fn=aux_loss_fn, device=self.config.device, seed=self.config.seed
        )
        if aux_loss_fn is not None:
            self.agent.aux_loss_fn = aux_loss_fn
        self.logger = logger
        self.history: List[Dict[str, Any]] = []
        self.episode_returns: List[float] = []
        self._ep_returns = np.zeros(int(self.config.num_env), dtype=np.float64)
        self._ep_lengths = np.zeros(int(self.config.num_env), dtype=np.int64)
        self._last_obs = None
        self._recent: List[float] = []

    # -- helpers ---------------------------------------------------------------
    def log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        elif self.config.verbose:
            print(message, flush=True)

    def build_storage(self) -> RolloutStorage:
        return RolloutStorage(
            num_step=int(self.config.num_step),
            num_env=int(self.config.num_env),
            obs_shape=self.agent.obs_shape,
            device=self.agent.device,
            num_actions=self.agent.num_actions,
            use_gae=bool(self.config.use_gae),
            gamma=float(self.config.gamma),
            lam=float(self.config.lam),
            int_gamma=float(self.config.int_gamma),
            int_lam=self.config.int_lam,
            normalize_advantages=bool(self.config.normalize_advantages),
            eps=float(self.config.stable_eps),
        )

    # -- training --------------------------------------------------------------
    def collect_rollout(self, storage: Optional[RolloutStorage] = None) -> Tuple[RolloutStorage, Dict[str, float]]:
        """Collect ``NumStep`` steps from ``NumEnv`` parallel environments."""
        cfg = self.config
        storage = storage or self.build_storage()
        obs = self._last_obs if self._last_obs is not None else self.env.reset()
        obs_t = self.agent._to_tensor(obs)
        storage.reset(obs_t)

        extrinsic_sum = 0.0
        intrinsic_sum = 0.0
        for _ in range(int(cfg.num_step)):
            action, log_prob, entropy, value = self.agent.act(obs_t, deterministic=False)
            raw_int, int_reward = self.agent.intrinsic_reward(obs_t)
            next_obs, rewards, dones, infos = self.env.step(action.detach().cpu().numpy())
            next_obs_t = self.agent._to_tensor(next_obs)

            mask = 1.0 - np.asarray(dones, dtype=np.float32)
            storage.insert(
                next_obs_t,
                action,
                log_prob,
                value,
                torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.agent.device),
                torch.as_tensor(mask, device=self.agent.device),
                entropies=entropy,
                int_rewards=int_reward,
                int_raw=raw_int,
            )
            extrinsic_sum += float(np.sum(np.asarray(rewards, dtype=np.float64)))
            intrinsic_sum += float(torch.sum(int_reward).item())
            self._track_episodes(np.asarray(rewards, dtype=np.float64), np.asarray(dones, dtype=bool), infos)
            obs_t = next_obs_t
            self.agent.num_env_steps += int(cfg.num_env)

        self._last_obs = obs_t.detach().cpu().numpy() if isinstance(obs_t, torch.Tensor) else obs_t
        with torch.no_grad():
            _, last_value = self.agent.policy(obs_t)
        last_value = last_value.reshape(-1)
        storage.compute_returns(last_value=last_value, normalize_ext_returns=bool(cfg.normalize_ext_returns))
        stats = {
            "ext_reward_mean": extrinsic_sum / max(1, cfg.num_step * cfg.num_env),
            "int_reward_mean": intrinsic_sum / max(1, cfg.num_step * cfg.num_env),
        }
        return storage, stats

    def _track_episodes(self, rewards, dones, infos) -> None:
        n = len(rewards)
        if len(self._ep_returns) < n:
            self._ep_returns = np.zeros(n, dtype=np.float64)
            self._ep_lengths = np.zeros(n, dtype=np.int64)
        self._ep_returns += rewards
        self._ep_lengths += 1
        for i in range(n):
            if dones[i]:
                self.episode_returns.append(float(self._ep_returns[i]))
                self._recent.append(float(self._ep_returns[i]))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0

    def train(self, total_steps: Optional[int] = None, log_every: Optional[int] = None) -> Dict[str, Any]:
        """Main loop: collect -> update, with periodic room-7 evaluation & saving."""
        cfg = self.config
        total_steps = int(total_steps or cfg.total_steps)
        log_every = int(log_every or cfg.log_every)
        start_time = time.time()
        last_log = 0

        while self.agent.num_env_steps < total_steps:
            storage, stats = self.collect_rollout()
            update_info = self.agent.update(storage)

            done_steps = self.agent.num_env_steps
            if done_steps - last_log >= log_every:
                last_log = done_steps
                mean_return = float(np.mean(self._recent[-100:])) if self._recent else float("nan")
                self.log(
                    "[ppo-rnd] steps={:.2e} updates={} ext_r={:.4f} int_r={:.4f} "
                    "policy_loss={:.4f} value_loss={:.4f} rnd_loss={:.4f} entropy={:.4f} return={:.3f} ({:.0f}s)".format(
                        done_steps,
                        self.agent.num_updates,
                        stats["ext_reward_mean"],
                        stats["int_reward_mean"],
                        update_info.get("policy_loss", float("nan")),
                        update_info.get("value_loss", float("nan")),
                        update_info.get("rnd_loss", float("nan")),
                        update_info.get("entropy", float("nan")),
                        mean_return,
                        time.time() - start_time,
                    )
                )
                self.history.append({"step": done_steps, "return": mean_return, **stats, **update_info})

            if done_steps % max(1, int(cfg.room7_every)) < self.agent.num_env_steps % max(1, int(cfg.room7_every)) or (
                done_steps >= cfg.room7_every and (done_steps // cfg.room7_every) > ((done_steps - cfg.num_step * cfg.num_env) // cfg.room7_every)
            ):
                self.evaluate_and_log(done_steps)

            if self.output_dir and cfg.save_every and done_steps % int(cfg.save_every) < cfg.num_step * cfg.num_env:
                path = os.path.join(self.output_dir, f"ppo_rnd_step_{done_steps}.pt")
                self.agent.save(path)
                self.log(f"[ppo-rnd] checkpoint saved to {path}")

        summary = {
            "method": cfg.method,
            "steps": self.agent.num_env_steps,
            "updates": self.agent.num_updates,
            "history": self.history,
            "episode_returns": self.episode_returns[-1000:],
            "mean_return": float(np.mean(self._recent[-100:])) if self._recent else float("nan"),
        }
        if self.output_dir:
            with open(os.path.join(self.output_dir, "summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
        return summary

    # -- evaluation ------------------------------------------------------------
    def evaluate(self, num_episodes: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the current policy (return + Room-7 success rate)."""
        from src.montezuma.env import evaluate_policy, room_visitation

        episodes = int(num_episodes or self.config.eval_episodes)
        env = make_env(seed=int(self.config.seed if seed is None else seed), stub=bool(self.config.stub))
        try:
            metrics = evaluate_policy(
                self.agent.policy,
                env=env,
                num_episodes=episodes,
                seed=int(self.config.seed if seed is None else seed),
                deterministic=not self.config.deterministic,
                stub=bool(self.config.stub),
                far_room=ROOM7,
            )
            metrics["visitation"] = room_visitation(
                self.agent.policy, env=env, num_episodes=1, seed=int(self.config.seed), stub=bool(self.config.stub)
            ).tolist()
        finally:
            try:
                env.close()
            except Exception:
                pass
        return metrics

    def evaluate_and_log(self, step: int) -> Dict[str, float]:
        metrics = self.evaluate()
        metrics["step"] = int(step)
        self.history.append(dict(metrics))
        self.log(
            "[eval] steps={:.2e} return={:.2f} room7_success={:.3f} max_room={}".format(
                step, metrics.get("return_mean", float("nan")), metrics.get("room7_success_rate", float("nan")), metrics.get("max_room", "?")
            )
        )
        if self.output_dir:
            path = os.path.join(self.output_dir, "room7_success.json")
            payload = {
                "steps": [h.get("step") for h in self.history if "room7_success_rate" in h],
                "room7_success_rate": [h.get("room7_success_rate") for h in self.history if "room7_success_rate" in h],
            }
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2)
        return metrics


# --------------------------------------------------------------------------------------
# Functional API
# --------------------------------------------------------------------------------------
def train_ppo_rnd(
    config: Optional[PPORNDConfig] = None,
    env: Optional[Any] = None,
    total_steps: Optional[int] = None,
    aux_loss_fn: Optional[Callable[..., Any]] = None,
    output_dir: Optional[str] = None,
    logger: Optional[Any] = None,
    init_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Train (or fine-tune) a PPO + RND agent and return the summary dict.

    ``method`` semantics follow the paper's variants:
    ``"scratch"`` (never sees BC data), ``"none"`` (vanilla fine-tuning from ``M2``),
    ``"bc"`` (fine-tuning + behavioral cloning retention, weight ``kl_weight``) and
    ``"ewc"`` (fine-tuning + EWC).
    """
    cfg = config or PPORNDConfig()
    trainer = PPORNDTrainer(config=cfg, env=env, aux_loss_fn=aux_loss_fn, logger=logger, output_dir=output_dir)
    if init_checkpoint:
        trainer.agent.load(init_checkpoint)
        trainer.log(f"[ppo-rnd] initialised from {init_checkpoint}")
    elif cfg.checkpoint and cfg.method not in ("scratch",):
        trainer.agent.load(cfg.checkpoint)
        trainer.log(f"[ppo-rnd] initialised from {cfg.checkpoint}")
    return trainer.train(total_steps=total_steps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO + RND training / fine-tuning for Montezuma's Revenge")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--method", type=str, default=None, choices=["scratch", "none", "bc", "ewc"])
    parser.add_argument("--kl-weight", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="tiny run for CPU validation")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PPORNDConfig()
    if args.config:
        from src.common.config import load_config

        raw = load_config(args.config, overrides=args.overrides or None)
        cfg = PPORNDConfig.from_config(raw)
    cfg = cfg.with_overrides(
        total_steps=args.total_steps,
        method=args.method,
        kl_weight=args.kl_weight,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        seed=args.seed,
        stub=True if args.stub else None,
    )
    if args.smoke_test:
        cfg = cfg.with_overrides(num_env=4, num_step=16, total_steps=64 * 4, eval_episodes=2, room7_every=1000,
                                 log_every=100, save_every=0, stub=True, device="cpu")
    summary = train_ppo_rnd(config=cfg, output_dir=cfg.output_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
