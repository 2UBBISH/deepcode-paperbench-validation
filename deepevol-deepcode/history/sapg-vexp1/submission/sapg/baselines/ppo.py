"""Vanilla PPO baseline with a batch size scaled to the full environment count.

This implements the PPO baseline used in the SAPG paper (Table 1, Figure 2).
The key property being probed is *batch-size saturation*: as the number of
parallel environments (and hence the PPO batch) grows, PPO's performance
saturates and eventually degrades, whereas SAPG continues to improve.

The implementation follows the standard PPO recipe:
  * clipped surrogate objective (Eq. 2 of the paper),
  * GAE(lambda) advantage estimation with tau = 0.95,
  * KL-adaptive learning rate (threshold 0.016),
  * gradient norm clipping at 1.0,
  * Adam with PyTorch defaults,
  * mini-batch size = num_envs * 4, mini-epochs 2 (AllegroKuka) / 5 (easy tasks).

The trainer is deliberately simulator-agnostic: it receives an ``env_factory``
callable returning a vectorized environment exposing ``reset()``,
``step(actions)``, ``num_envs``, ``obs_dim`` and ``action_dim``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..networks import build_actor, build_critic
from ..sapg.losses import compute_gae, ppo_surrogate_loss, value_loss
from ..utils.kl_lr import KLAdaptiveLR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Hyperparameters for the vanilla PPO baseline."""

    task: str = "allegrokuka_regrasping"
    num_envs: int = 24576
    horizon: int = 16
    gamma: float = 0.99
    tau: float = 0.95
    clip_eps: float = 0.1
    critic_coef: float = 4.0
    entropy_coef: float = 0.0
    learning_rate: float = 3.0e-4
    kl_threshold: float = 0.016
    kl_min_lr: float = 1.0e-5
    kl_max_lr: float = 1.0e-2
    mini_epochs: int = 2
    num_mini_batches: int = 4
    grad_norm_clip: float = 1.0
    normalize_advantage: bool = True
    value_clip: Optional[float] = None
    phi_dim: int = 0  # PPO has no per-policy latent
    recurrent: bool = False
    lstm_hidden: int = 768
    obs_dim: int = 44
    action_dim: int = 23
    device: str = "cuda"
    seed: int = 0
    target_transitions: float = 2.0e10
    output_dir: str = "runs/ppo"
    log_interval: int = 1
    save_interval: int = 100
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class PPORollout:
    """Simple dense rollout buffer for a single policy."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        recurrent: bool = False,
        lstm_hidden: int = 0,
    ) -> None:
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.recurrent = bool(recurrent)
        self.lstm_hidden = int(lstm_hidden)

        self.obs = torch.zeros(self.horizon, self.num_envs, self.obs_dim, device=device)
        self.actions = torch.zeros(self.horizon, self.num_envs, self.action_dim, device=device)
        self.log_probs = torch.zeros(self.horizon, self.num_envs, device=device)
        self.rewards = torch.zeros(self.horizon, self.num_envs, device=device)
        self.dones = torch.zeros(self.horizon, self.num_envs, device=device)
        self.values = torch.zeros(self.horizon, self.num_envs, device=device)
        if self.recurrent:
            self.lstm_h = torch.zeros(self.horizon, self.num_envs, self.lstm_hidden, device=device)
            self.lstm_c = torch.zeros(self.horizon, self.num_envs, self.lstm_hidden, device=device)

        self.advantages = torch.zeros(self.horizon, self.num_envs, device=device)
        self.returns = torch.zeros(self.horizon, self.num_envs, device=device)
        self._ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        t = self._ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        if self.recurrent and lstm_state is not None:
            self.lstm_h[t] = lstm_state[0]
            self.lstm_c[t] = lstm_state[1]
        self._ptr += 1

    def compute_advantages(self, last_values: torch.Tensor, gamma: float = 0.99, tau: float = 0.95) -> None:
        adv, ret = compute_gae(self.rewards, self.values, self.dones, gamma=gamma, tau=tau)
        self.advantages = adv
        self.returns = ret

    def flat(self) -> Dict[str, torch.Tensor]:
        out = {
            "obs": self.obs.reshape(-1, self.obs_dim),
            "actions": self.actions.reshape(-1, self.action_dim),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }
        if self.recurrent:
            out["lstm_h"] = self.lstm_h.reshape(-1, self.lstm_hidden)
            out["lstm_c"] = self.lstm_c.reshape(-1, self.lstm_hidden)
        return out

    def size(self) -> int:
        return self.horizon * self.num_envs

    def reset(self) -> None:
        self._ptr = 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """Vanilla PPO trainer (single policy, full batch)."""

    def __init__(
        self,
        config: PPOConfig,
        env_factory: Callable[[int, int], Any],
        logger: Any = None,
    ) -> None:
        self.config = config
        self.env_factory = env_factory
        self.logger = logger

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.env = env_factory(config.num_envs, config.seed)
        self.num_envs = int(getattr(self.env, "num_envs", config.num_envs))
        self.obs_dim = int(getattr(self.env, "obs_dim", config.obs_dim))
        self.action_dim = int(getattr(self.env, "action_dim", config.action_dim))

        self.actor = build_actor(
            config.task, self.obs_dim, self.action_dim, phi_dim=0
        ).to(self.device)
        self.critic = build_critic(config.task, self.obs_dim, phi_dim=0).to(self.device)
        self.recurrent = bool(getattr(self.actor, "is_recurrent", config.recurrent))

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        self.scheduler = KLAdaptiveLR(
            self.optimizer,
            threshold=config.kl_threshold,
            min_lr=config.kl_min_lr,
            max_lr=config.kl_max_lr,
        )

        self.rollout = PPORollout(
            horizon=config.horizon,
            num_envs=self.num_envs,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            recurrent=self.recurrent,
            lstm_hidden=config.lstm_hidden,
        )

        self.obs = self._to_tensor(self.env.reset())
        self.lstm_state = self._init_lstm_state()
        self.iteration = 0
        self.transitions = 0
        self._episode_returns: List[float] = []
        self._episode_successes: List[float] = []

    # -- helpers ----------------------------------------------------------

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device).float()
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _init_lstm_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.recurrent:
            return None
        h = torch.zeros(1, self.num_envs, self.config.lstm_hidden, device=self.device)
        c = torch.zeros(1, self.num_envs, self.config.lstm_hidden, device=self.device)
        return (h, c)

    def _reset_lstm(self, mask: torch.Tensor) -> None:
        if self.lstm_state is None:
            return
        h, c = self.lstm_state
        m = mask.view(1, -1, 1).float()
        self.lstm_state = (h * (1.0 - m), c * (1.0 - m))

    # -- rollout ----------------------------------------------------------

    @torch.no_grad()
    def collect_rollouts(self) -> Dict[str, float]:
        self.rollout.reset()
        ep_returns: List[float] = []
        ep_successes: List[float] = []

        for _ in range(self.config.horizon):
            dist = self.actor.distribution(self.obs, None, self.lstm_state) if self.recurrent \
                else self.actor.distribution(self.obs, None)
            if self.recurrent:
                dist, new_state = dist
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(-1)
            values = self.critic(self.obs, None, self.lstm_state)[0] if self.recurrent \
                else self.critic(self.obs, None)

            self.rollout.add(
                self.obs, actions, log_probs,
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                values,
                self.lstm_state,
            )

            obs_np, rew, done, info = self.env.step(actions.cpu().numpy())
            rew_t = self._to_tensor(rew)
            done_t = self._to_tensor(done)

            self.rollout.rewards[self.rollout._ptr - 1] = rew_t
            self.rollout.dones[self.rollout._ptr - 1] = done_t

            if isinstance(info, dict):
                if "episode_return" in info:
                    ep_returns.extend(np.asarray(info["episode_return"]).reshape(-1).tolist())
                if "episode_success" in info:
                    ep_successes.extend(np.asarray(info["episode_success"]).reshape(-1).tolist())

            self.obs = self._to_tensor(obs_np)
            if self.recurrent:
                self.lstm_state = new_state
                self._reset_lstm(done_t)

        with torch.no_grad():
            if self.recurrent:
                last_values = self.critic(self.obs, None, self.lstm_state)[0]
            else:
                last_values = self.critic(self.obs, None)
        self.rollout.compute_advantages(last_values, gamma=self.config.gamma, tau=self.config.tau)

        self.transitions += self.rollout.size()
        self._episode_returns.extend(ep_returns)
        self._episode_successes.extend(ep_successes)

        stats = {
            "rollout/reward_mean": float(self.rollout.rewards.mean().item()),
            "rollout/value_mean": float(self.rollout.values.mean().item()),
        }
        if ep_returns:
            stats["rollout/episode_return"] = float(np.mean(ep_returns))
        if ep_successes:
            stats["rollout/episode_success"] = float(np.mean(ep_successes))
        return stats

    # -- update -----------------------------------------------------------

    def update(self) -> Dict[str, float]:
        batch = self.rollout.flat()
        n = batch["obs"].shape[0]
        mini_batch_size = max(1, (self.num_envs * 4) // max(1, self.config.num_mini_batches))

        info: Dict[str, float] = {}
        approx_kls: List[float] = []

        for _ in range(self.config.mini_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                mb = {k: v[idx] for k, v in batch.items()}

                if self.config.normalize_advantage and mb["advantages"].numel() > 1:
                    adv = (mb["advantages"] - mb["advantages"].mean()) / (
                        mb["advantages"].std() + 1e-8
                    )
                else:
                    adv = mb["advantages"]

                lstm_state = None
                if self.recurrent:
                    lstm_state = (mb["lstm_h"].unsqueeze(0), mb["lstm_c"].unsqueeze(0))

                if self.recurrent:
                    dist, _ = self.actor.distribution(mb["obs"], None, lstm_state)
                    values, _ = self.critic(mb["obs"], None, lstm_state)
                else:
                    dist = self.actor.distribution(mb["obs"], None)
                    values = self.critic(mb["obs"], None)

                log_probs = dist.log_prob(mb["actions"]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                policy_loss = ppo_surrogate_loss(
                    log_probs, mb["log_probs"], adv, clip_eps=self.config.clip_eps
                )
                v_loss = value_loss(
                    values, mb["returns"], clip=self.config.value_clip, old_values=mb["values"]
                )
                loss = policy_loss + self.config.critic_coef * v_loss
                if self.config.entropy_coef:
                    loss = loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.grad_norm_clip,
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_probs - mb["log_probs"]
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                approx_kls.append(approx_kl)

                info["loss/policy"] = float(policy_loss.item())
                info["loss/value"] = float(v_loss.item())
                info["loss/total"] = float(loss.item())
                info["policy/entropy"] = float(entropy.item())

        mean_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
        self.scheduler.update(mean_kl)
        info["policy/approx_kl"] = mean_kl
        info["policy/learning_rate"] = self.scheduler.current_lr()
        return info

    # -- training loop ----------------------------------------------------

    def train(self) -> "PPOTrainer":
        cfg = self.config
        start = time.time()
        while self.transitions < cfg.target_transitions:
            self.iteration += 1
            rollout_stats = self.collect_rollouts()
            update_stats = self.update()

            if self.logger is not None and self.iteration % max(1, cfg.log_interval) == 0:
                metrics = {"iteration": self.iteration, "transitions": self.transitions}
                metrics.update(rollout_stats)
                metrics.update(update_stats)
                metrics["time/elapsed"] = time.time() - start
                self.logger.log(metrics, step=self.transitions)
            elif cfg.verbose and self.iteration % max(1, cfg.log_interval) == 0:
                print(
                    f"[PPO] iter={self.iteration} transitions={self.transitions} "
                    f"reward={rollout_stats.get('rollout/reward_mean', 0.0):.4f} "
                    f"kl={update_stats.get('policy/approx_kl', 0.0):.4f}"
                )

            if cfg.save_interval and self.iteration % cfg.save_interval == 0:
                self.save(os.path.join(cfg.output_dir, f"checkpoint_{self.iteration}.pt"))

        self.save(os.path.join(cfg.output_dir, "checkpoint_final.pt"))
        return self

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "iteration": self.iteration,
                "transitions": self.transitions,
                "config": self.config.__dict__,
            },
            path,
        )
        return path

    def load(self, path: str) -> "PPOTrainer":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.iteration = int(ckpt.get("iteration", 0))
        self.transitions = int(ckpt.get("transitions", 0))
        return self


def train_ppo(
    config: PPOConfig,
    env_factory: Callable[[int, int], Any],
    logger: Any = None,
) -> PPOTrainer:
    """Instantiate and run a vanilla PPO trainer."""
    trainer = PPOTrainer(config, env_factory, logger=logger)
    return trainer.train()


__all__ = ["PPOConfig", "PPORollout", "PPOTrainer", "train_ppo"]
