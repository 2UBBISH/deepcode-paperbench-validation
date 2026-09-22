"""Vanilla PPO baseline for SAPG.

This module implements the standard on-policy PPO baseline used in the SAPG
paper (Sec 5, Fig. 2).  It reuses the shared actor/critic backbones from
``sapg.networks`` (with ``latent_dim == 0`` so there is no per-policy
conditioning) and the on-policy loss / return machinery from ``sapg.losses``
and ``sapg.returns``.

Key points from the paper:
  * PPO is trained with a *scaled batch size* (number of parallel envs) to
    demonstrate the saturation effect in Fig. 2.  The batch size is simply
    ``num_envs`` (all envs belong to a single policy).
  * On-policy clipped surrogate (Eq. 2) with GAE (tau = 0.95).
  * Adam optimizer, adaptive LR via KL threshold 0.016, grad-norm clip 1.0.
  * Mini-epochs: 2 (AllegroKuka), 5 (hands); minibatch = num_envs * 4.

The trainer exposes the same ``train(num_iterations)`` / ``state_dict()`` /
``load_state_dict()`` interface as ``SAPGTrainer`` so that ``main.py`` and the
experiment scripts can treat all algorithms uniformly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..sapg.config import SAPGConfig
from ..sapg.losses import on_policy_loss, critic_loss
from ..sapg.networks import SAPGNetworks, build_networks
from ..sapg.returns import compute_advantages_and_targets, flatten_time_major
from ..sapg.rollout import RolloutBuffer, _unpack_step


__all__ = ["PPOTrainer", "PPOUpdateStats", "build_ppo_trainer"]


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------
@dataclass
class PPOUpdateStats:
    """Per-iteration statistics for the PPO baseline."""

    iteration: int = 0
    total_transitions: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    mean_reward: float = 0.0
    mean_episode_reward: float = 0.0
    successes: float = 0.0
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "iteration": self.iteration,
            "total_transitions": self.total_transitions,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "mean_reward": self.mean_reward,
            "mean_episode_reward": self.mean_episode_reward,
            "successes": self.successes,
            "wall_time": self.wall_time,
        }


# ---------------------------------------------------------------------------
# Adaptive learning rate (KL based) -- mirrors SAPGTrainer.AdaptiveLR
# ---------------------------------------------------------------------------
class AdaptiveLR:
    """KL-based adaptive learning rate scheduler.

    If the measured approximate KL exceeds ``2 * kl_threshold`` the LR is
    reduced by ``factor``; if it falls below ``0.5 * kl_threshold`` the LR is
    increased by ``factor``.  This matches the schedule used throughout the
    SAPG paper (KL threshold 0.016).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        kl_threshold: float = 0.016,
        min_lr: float = 1e-6,
        max_lr: Optional[float] = None,
        factor: float = 1.5,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.kl_threshold = float(kl_threshold)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr) if max_lr is not None else float(base_lr) * 10.0
        self.factor = float(factor)
        self.current_lr = float(base_lr)

    def step(self, approx_kl: float) -> float:
        if approx_kl > 2.0 * self.kl_threshold:
            self.current_lr = max(self.min_lr, self.current_lr / self.factor)
        elif approx_kl < 0.5 * self.kl_threshold:
            self.current_lr = min(self.max_lr, self.current_lr * self.factor)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr
        return self.current_lr


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """Vanilla PPO trainer operating on a single (large) batch of envs.

    Parameters
    ----------
    config:
        A :class:`SAPGConfig` describing the task and optimization settings.
        ``config.num_blocks`` is ignored (PPO uses a single policy over all
        envs); ``config.latent_dim`` should be 0.
    env:
        A vectorized environment exposing ``reset()`` / ``step(actions)``.
    obs_dim, action_dim:
        Observation and action dimensionalities.
    device:
        Torch device; defaults to ``config.device``.
    """

    def __init__(
        self,
        config: SAPGConfig,
        env: Any,
        obs_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.env = env
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device or config.device)

        # PPO uses a single policy -> no latent conditioning.
        self.networks: SAPGNetworks = build_networks(
            config, self.obs_dim, self.action_dim
        ).to(self.device)

        # Single optimizer over all parameters.
        self.optimizer = torch.optim.Adam(
            self.networks.parameters(), lr=config.learning_rate
        )
        self.lr_scheduler = AdaptiveLR(
            self.optimizer,
            base_lr=config.learning_rate,
            kl_threshold=config.kl_threshold,
        )

        self.num_envs = int(config.num_envs)
        self.horizon = int(config.horizon)
        self.minibatch_size = int(config.minibatch_size)
        self.num_mini_epochs = int(config.num_mini_epochs)
        self.clip_eps = float(config.clip_eps)
        self.gamma = float(config.gamma)
        self.gae_lambda = float(config.gae_lambda)
        self.grad_norm_clip = float(config.grad_norm_clip)
        self.on_policy_target_steps = int(config.on_policy_target_steps)

        # Rollout buffer (single block = all envs).
        self.buffer = RolloutBuffer(
            horizon=self.horizon,
            num_envs=self.num_envs,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        # LSTM state bookkeeping.
        self._lstm_state = None
        self._obs = None
        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._episode_successes = torch.zeros(self.num_envs, device=self.device)
        self._completed_episode_rewards: List[float] = []
        self._completed_episode_successes: List[float] = []

        self.total_transitions = 0
        self.iteration = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def _init_lstm_state(self) -> None:
        actor = self.networks.actor
        if getattr(actor, "use_lstm", False):
            self._lstm_state = (
                torch.zeros(
                    actor.lstm_num_layers,
                    self.num_envs,
                    actor.lstm_hidden_size,
                    device=self.device,
                ),
                torch.zeros(
                    actor.lstm_num_layers,
                    self.num_envs,
                    actor.lstm_hidden_size,
                    device=self.device,
                ),
            )
        else:
            self._lstm_state = None

    def _reset_env(self) -> None:
        self._obs = self.env.reset()
        if not torch.is_tensor(self._obs):
            self._obs = torch.as_tensor(self._obs, dtype=torch.float32)
        self._obs = self._obs.to(self.device).float()
        self._init_lstm_state()

    @torch.no_grad()
    def collect_rollout(self) -> RolloutBuffer:
        """Collect ``horizon`` steps from all envs into the buffer."""
        if self._obs is None:
            self._reset_env()

        self.buffer.reset()
        latent = None  # PPO: no latent conditioning

        for _ in range(self.horizon):
            action, log_prob, self._lstm_state = self.networks.actor_act(
                self._obs, latent, self._lstm_state
            )
            value, _ = self.networks.critic_value(self._obs, latent)

            step_out = self.env.step(action)
            next_obs, rewards, dones, infos = _unpack_step(step_out)

            if not torch.is_tensor(rewards):
                rewards = torch.as_tensor(rewards, dtype=torch.float32)
            if not torch.is_tensor(dones):
                dones = torch.as_tensor(dones, dtype=torch.float32)
            rewards = rewards.to(self.device).float().view(-1)
            dones = dones.to(self.device).float().view(-1)
            if not torch.is_tensor(next_obs):
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
            next_obs = next_obs.to(self.device).float()

            self.buffer.add(
                obs=self._obs,
                actions=action,
                log_probs=log_prob,
                rewards=rewards,
                dones=dones,
                values=value,
                lstm_state=self._lstm_state,
            )

            # Track episode statistics.
            self._episode_rewards += rewards
            if isinstance(infos, dict) and "successes" in infos:
                succ = infos["successes"]
                if not torch.is_tensor(succ):
                    succ = torch.as_tensor(succ, dtype=torch.float32)
                self._episode_successes += succ.to(self.device).float().view(-1)
            done_mask = dones > 0.5
            if done_mask.any():
                self._completed_episode_rewards.extend(
                    self._episode_rewards[done_mask].tolist()
                )
                self._completed_episode_successes.extend(
                    self._episode_successes[done_mask].tolist()
                )
                self._episode_rewards[done_mask] = 0.0
                self._episode_successes[done_mask] = 0.0

            self._obs = next_obs
            self.total_transitions += self.num_envs

        # Bootstrap value for the final observation.
        next_value, _ = self.networks.critic_value(self._obs, latent)
        self.buffer.compute_returns(
            next_value=next_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            on_policy_target_steps=self.on_policy_target_steps,
        )
        return self.buffer

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """Run ``num_mini_epochs`` of minibatch PPO updates."""
        buf = self.buffer
        T, B = buf.horizon, buf.num_envs
        latent = None

        # Flatten time-major buffers to [T*B, ...].
        obs = flatten_time_major(buf.obs)
        actions = flatten_time_major(buf.actions)
        old_log_probs = flatten_time_major(buf.log_probs)
        advantages = flatten_time_major(buf.advantages)
        value_targets = flatten_time_major(buf.value_targets)
        old_values = flatten_time_major(buf.values)

        n = obs.shape[0]
        mb_size = min(self.minibatch_size, n)

        agg = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        num_updates = 0

        for _ in range(self.num_mini_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_targets = value_targets[idx]
                mb_old_values = old_values[idx]

                new_log_probs, _ = self.networks.actor_log_prob(
                    mb_obs, mb_actions, latent
                )
                entropy, _ = self.networks.actor_entropy(mb_obs, latent)
                values, _ = self.networks.critic_value(mb_obs, latent)

                pol_out = on_policy_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_adv,
                    clip_eps=self.clip_eps,
                )
                crit_out = critic_loss(
                    values=values,
                    on_policy_targets=mb_targets,
                    old_values=mb_old_values,
                    value_coef=self.config.critic_coef,
                )

                loss = pol_out.loss + crit_out.loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.networks.parameters(), self.grad_norm_clip
                )
                self.optimizer.step()

                agg["policy_loss"] += float(pol_out.policy_loss.detach())
                agg["value_loss"] += float(crit_out.value_loss.detach())
                agg["entropy"] += float(pol_out.entropy.detach().mean())
                agg["approx_kl"] += float(pol_out.approx_kl.detach())
                agg["clip_fraction"] += float(pol_out.clip_fraction.detach())
                agg["grad_norm"] += float(grad_norm)
                num_updates += 1

        if num_updates > 0:
            for k in agg:
                agg[k] /= num_updates

        # Adaptive LR on the mean KL of this iteration.
        lr = self.lr_scheduler.step(agg["approx_kl"])
        agg["learning_rate"] = lr
        return agg

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train_iteration(self) -> PPOUpdateStats:
        t0 = time.time()
        self.collect_rollout()
        agg = self.update()

        mean_ep_reward = (
            sum(self._completed_episode_rewards) / len(self._completed_episode_rewards)
            if self._completed_episode_rewards
            else 0.0
        )
        mean_ep_success = (
            sum(self._completed_episode_successes)
            / len(self._completed_episode_successes)
            if self._completed_episode_successes
            else 0.0
        )
        # Keep only a rolling window to bound memory.
        self._completed_episode_rewards = self._completed_episode_rewards[-10000:]
        self._completed_episode_successes = self._completed_episode_successes[-10000:]

        self.iteration += 1
        return PPOUpdateStats(
            iteration=self.iteration,
            total_transitions=self.total_transitions,
            policy_loss=agg["policy_loss"],
            value_loss=agg["value_loss"],
            entropy=agg["entropy"],
            approx_kl=agg["approx_kl"],
            clip_fraction=agg["clip_fraction"],
            learning_rate=agg["learning_rate"],
            grad_norm=agg["grad_norm"],
            mean_reward=mean_ep_reward,
            mean_episode_reward=mean_ep_reward,
            successes=mean_ep_success,
            wall_time=time.time() - t0,
        )

    def train(self, num_iterations: Optional[int] = None) -> List[Dict[str, float]]:
        """Run the full training loop, returning a list of per-iteration stats."""
        if num_iterations is None:
            num_iterations = int(self.config.num_iterations)
        history: List[Dict[str, float]] = []
        for _ in range(int(num_iterations)):
            stats = self.train_iteration()
            history.append(stats.to_dict())
        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "networks": self.networks.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "total_transitions": self.total_transitions,
            "lr": self.lr_scheduler.current_lr,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.networks.load_state_dict(state["networks"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration = state.get("iteration", 0)
        self.total_transitions = state.get("total_transitions", 0)
        if "lr" in state:
            self.lr_scheduler.current_lr = state["lr"]
            for group in self.optimizer.param_groups:
                group["lr"] = state["lr"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_ppo_trainer(
    config: SAPGConfig,
    env: Any,
    obs_dim: int,
    action_dim: int,
    device: Optional[torch.device] = None,
) -> PPOTrainer:
    """Build a :class:`PPOTrainer` from a config and environment."""
    return PPOTrainer(config, env, obs_dim, action_dim, device=device)
