"""PPO baseline (single policy) for SAPG comparison.

This module implements a standard single-policy PPO trainer used as a baseline
against SAPG (Section 6, Table 1 / Figure 5). It reuses the shared actor-critic
backbone from :mod:`sapg.models` (with ``num_policies=1``) and the loss / GAE
utilities from :mod:`sapg.losses` and :mod:`sapg.gae`.

Key hyperparameters (Appendix B, Tables 2-4):
    - Adam optimizer, LR = 1e-4 (AllegroKuka), 5e-4 (hands)
    - KL-adaptive LR with threshold 0.016
    - Grad norm clip 1.0
    - gamma = 0.99, tau = 0.95
    - critic coef = 4.0
    - mini-batch size = num_envs * 4
    - mini-epochs = 2 (AllegroKuka), 5 (hands)
    - clip eps = 0.1 (AllegroKuka, ShadowHand), 0.2 (AllegroHand)
    - bounds loss coefficient 1e-4
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .gae import compute_advantages, compute_n_step_returns
from .losses import bounds_loss, critic_loss, on_policy_policy_loss
from .models import MultiPolicyActorCritic, build_actor_critic
from .rollout import RolloutBuffer, collect_rollout
from .utils import (
    AverageMeter,
    clip_grad_norm_,
    explained_variance,
    get_logger,
    kl_adaptive_learning_rate,
)

__all__ = ["PPOConfig", "PPO", "PPOTrainState"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    """Hyperparameters for the single-policy PPO baseline."""

    # Environment / rollout
    num_envs: int = 24576
    horizon: int = 16
    task: str = "allegrokuka"

    # Optimization
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.95
    n_step: int = 3
    clip_eps: float = 0.1
    critic_coef: float = 4.0
    entropy_coef: float = 0.0
    bounds_coef: float = 1e-4
    max_grad_norm: float = 1.0

    # Mini-batching
    mini_epochs: int = 2
    num_mini_batches: int = 4  # mini-batch size = num_envs * num_mini_batches

    # KL-adaptive LR
    use_kl_adaptive_lr: bool = True
    kl_threshold: float = 0.016
    kl_adaptive_factor: float = 1.5

    # Misc
    normalize_advantage: bool = True
    seed: int = 0
    device: str = "cuda"
    max_iterations: int = 100000
    log_interval: int = 1

    # Model architecture overrides
    use_lstm: Optional[bool] = None
    phi_dim: Optional[int] = None
    actor_hidden_dims: Optional[List[int]] = None
    critic_hidden_dims: Optional[List[int]] = None


@dataclass
class PPOTrainState:
    """Mutable training state for the PPO baseline."""

    iteration: int = 0
    total_samples: int = 0
    learning_rate: float = 1e-4
    obs: Optional[torch.Tensor] = None
    lstm_state: Optional[object] = None
    history: List[Dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------
class PPO:
    """Single-policy PPO trainer.

    Parameters
    ----------
    env:
        Vectorized environment exposing ``num_envs``, ``obs_dim``, ``act_dim``,
        ``reset()`` and ``step(actions)``.
    actor_critic:
        Optional pre-built :class:`MultiPolicyActorCritic` with a single policy.
        If ``None``, one is built from ``config``.
    config:
        :class:`PPOConfig` hyperparameters.
    """

    def __init__(
        self,
        env,
        actor_critic: Optional[MultiPolicyActorCritic] = None,
        config: Optional[PPOConfig] = None,
    ) -> None:
        self.env = env
        self.cfg = config or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.logger = get_logger("sapg.ppo")

        obs_dim = getattr(env, "obs_dim", None)
        act_dim = getattr(env, "act_dim", None)
        if obs_dim is None or act_dim is None:
            raise ValueError("env must expose `obs_dim` and `act_dim` attributes")

        if actor_critic is None:
            overrides = {}
            if self.cfg.use_lstm is not None:
                overrides["use_lstm"] = self.cfg.use_lstm
            if self.cfg.phi_dim is not None:
                overrides["phi_dim"] = self.cfg.phi_dim
            if self.cfg.actor_hidden_dims is not None:
                overrides["actor_hidden_dims"] = self.cfg.actor_hidden_dims
            if self.cfg.critic_hidden_dims is not None:
                overrides["critic_hidden_dims"] = self.cfg.critic_hidden_dims
            actor_critic = build_actor_critic(
                task=self.cfg.task,
                obs_dim=obs_dim,
                act_dim=act_dim,
                num_policies=1,
                **overrides,
            )
        self.ac = actor_critic.to(self.device)

        # Single optimizer over all trainable parameters.
        params = list(self.ac.actor_parameters()) + list(self.ac.critic_parameters())
        params += list(self.ac.phi_parameters())
        entropy_params = getattr(self.ac, "entropy_parameters", None)
        if callable(entropy_params):
            params += list(entropy_params())
        # De-duplicate while preserving order.
        seen = set()
        unique_params = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)
        self.optimizer = torch.optim.Adam(unique_params, lr=self.cfg.learning_rate)

        self.state = PPOTrainState(learning_rate=self.cfg.learning_rate)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset the environment and internal state."""
        obs = self.env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        self.state.obs = obs
        self.state.lstm_state = None
        return obs

    def collect(self) -> RolloutBuffer:
        """Collect one rollout of ``horizon`` steps with the single policy."""
        buffer, next_obs, next_lstm = collect_rollout(
            env=self.env,
            actor_critic=self.ac,
            policy_index=0,
            num_envs_block=self.env.num_envs,
            horizon=self.cfg.horizon,
            obs=self.state.obs,
            lstm_state=self.state.lstm_state,
            device=self.device,
        )
        self.state.obs = next_obs
        self.state.lstm_state = next_lstm
        return buffer

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _flatten(self, buffer: RolloutBuffer) -> Dict[str, torch.Tensor]:
        """Flatten a time-major buffer into a batch of transitions."""
        T, B = buffer.horizon, buffer.num_envs
        flat = {
            "obs": buffer.obs.reshape(T * B, *buffer.obs.shape[2:]),
            "actions": buffer.actions.reshape(T * B, *buffer.actions.shape[2:]),
            "log_probs": buffer.log_probs.reshape(T * B),
            "rewards": buffer.rewards.reshape(T * B),
            "values": buffer.values.reshape(T * B),
            "dones": buffer.dones.reshape(T * B),
        }
        if buffer.next_obs is not None:
            flat["next_obs"] = buffer.next_obs.reshape(T * B, *buffer.next_obs.shape[2:])
        if buffer.next_values is not None:
            flat["next_values"] = buffer.next_values.reshape(T * B)
        if buffer.lstm_states is not None:
            flat["lstm_states"] = buffer.lstm_states
        return flat

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO mini-epoch updates on the collected rollout."""
        cfg = self.cfg
        T, B = buffer.horizon, buffer.num_envs

        # --- Advantage estimation (GAE, tau=0.95) ---
        next_value = None
        if buffer.next_values is not None:
            next_value = buffer.next_values[-1]
        advantages, returns = compute_advantages(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            next_value=next_value,
            gamma=cfg.gamma,
            tau=cfg.tau,
            normalize=cfg.normalize_advantage,
        )

        # --- n-step critic targets (Eq. 5) ---
        n_step_targets = compute_n_step_returns(
            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,
            gamma=cfg.gamma,
            n=cfg.n_step,
        )

        flat = self._flatten(buffer)
        flat_adv = advantages.reshape(T * B)
        flat_ret = returns.reshape(T * B)
        flat_nstep = n_step_targets.reshape(T * B)

        total = T * B
        mini_batch_size = max(1, cfg.num_mini_batches * B)
        num_mini_batches = max(1, total // mini_batch_size)

        info: Dict[str, float] = {}
        policy_meter = AverageMeter("policy_loss")
        value_meter = AverageMeter("value_loss")
        entropy_meter = AverageMeter("entropy")
        kl_meter = AverageMeter("approx_kl")
        clip_meter = AverageMeter("clip_fraction")

        for _ in range(cfg.mini_epochs):
            perm = torch.randperm(total, device=self.device)
            for mb in range(num_mini_batches):
                idx = perm[mb * mini_batch_size : (mb + 1) * mini_batch_size]
                if idx.numel() == 0:
                    continue

                mb_obs = flat["obs"][idx]
                mb_actions = flat["actions"][idx]
                mb_old_log_probs = flat["log_probs"][idx]
                mb_adv = flat_adv[idx]
                mb_nstep = flat_nstep[idx]

                mb_lstm = None
                if "lstm_states" in flat:
                    mb_lstm = self._slice_lstm(flat["lstm_states"], idx, T, B)

                log_probs, entropy, values = self.ac.evaluate_actions(
                    mb_obs, 0, mb_actions, mb_lstm
                )

                # Policy loss (Eq. 2) -- returned as loss-to-maximize.
                policy_loss, p_info = on_policy_policy_loss(
                    log_probs=log_probs,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_adv,
                    clip_eps=cfg.clip_eps,
                )

                # Critic loss (Eq. 7) against n-step targets.
                value_loss = critic_loss(values, mb_nstep)

                # Bounds regularization on the Gaussian mean.
                b_loss = bounds_loss(self.ac.actor.mean(mb_obs, self.ac.get_phi(0, mb_obs.shape[0])), cfg.bounds_coef) \
                    if hasattr(self.ac.actor, "mean") else torch.zeros((), device=self.device)

                loss = (
                    -policy_loss
                    + cfg.critic_coef * value_loss
                    - cfg.entropy_coef * entropy.mean()
                    + b_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.optimizer.param_groups[0]["params"], cfg.max_grad_norm)
                self.optimizer.step()

                policy_meter.update(float(policy_loss.detach()))
                value_meter.update(float(value_loss.detach()))
                entropy_meter.update(float(entropy.mean().detach()))
                kl_meter.update(float(p_info.get("approx_kl", 0.0)))
                clip_meter.update(float(p_info.get("clip_fraction", 0.0)))

        # --- KL-adaptive learning rate ---
        approx_kl = kl_meter.mean
        if cfg.use_kl_adaptive_lr:
            new_lr = kl_adaptive_learning_rate(
                self.state.learning_rate,
                approx_kl,
                kl_threshold=cfg.kl_threshold,
                factor=cfg.kl_adaptive_factor,
            )
            self.state.learning_rate = new_lr
            for g in self.optimizer.param_groups:
                g["lr"] = new_lr

        info.update(
            {
                "policy_loss": policy_meter.mean,
                "value_loss": value_meter.mean,
                "entropy": entropy_meter.mean,
                "approx_kl": approx_kl,
                "clip_fraction": clip_meter.mean,
                "learning_rate": self.state.learning_rate,
                "explained_variance": explained_variance(
                    buffer.values.reshape(-1).detach().cpu().numpy(),
                    returns.reshape(-1).detach().cpu().numpy(),
                ),
            }
        )
        return info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_lstm(lstm_states, idx: torch.Tensor, T: int, B: int):
        """Slice LSTM states for a mini-batch of flattened (T*B) transitions."""
        if lstm_states is None:
            return None
        # lstm_states is expected as (T, B, ...) or a tuple of such tensors.
        if isinstance(lstm_states, (tuple, list)):
            out = []
            for s in lstm_states:
                if s is None:
                    out.append(None)
                    continue
                flat = s.reshape(T * B, *s.shape[2:])
                out.append(flat[idx])
            return tuple(out)
        flat = lstm_states.reshape(T * B, *lstm_states.shape[2:])
        return flat[idx]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, max_iterations: Optional[int] = None) -> List[Dict[str, float]]:
        """Run the full PPO training loop."""
        cfg = self.cfg
        max_iters = max_iterations if max_iterations is not None else cfg.max_iterations
        obs = self.reset()
        self.state.obs = obs

        for it in range(max_iters):
            t0 = time.time()
            buffer = self.collect()
            info = self.update(buffer)
            self.state.iteration += 1
            self.state.total_samples += cfg.horizon * self.env.num_envs
            info["iteration"] = self.state.iteration
            info["total_samples"] = self.state.total_samples
            info["time"] = time.time() - t0
            self.state.history.append(info)

            if self.state.iteration % cfg.log_interval == 0:
                self.logger.info(
                    "iter=%d samples=%.3e policy_loss=%.4f value_loss=%.4f "
                    "entropy=%.4f kl=%.5f lr=%.2e",
                    self.state.iteration,
                    self.state.total_samples,
                    info["policy_loss"],
                    info["value_loss"],
                    info["entropy"],
                    info["approx_kl"],
                    info["learning_rate"],
                )
        return self.state.history
