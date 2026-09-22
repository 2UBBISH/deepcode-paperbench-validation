"""Parallel Q-Learning baseline (Li et al., 2023).

"A parallelized version of DDPG with different mixed exploration i.e. varying
exploration noise across environments to further aid exploration." (Sec. 5.2)

Implementation notes
--------------------
* one shared replay buffer filled by all environments,
* a deterministic actor ``mu(s)`` and a Q critic ``Q(s, a)`` with target
  networks and Polyak averaging (DDPG),
* *mixed exploration*: every environment ``e`` gets an exploration noise scale
  that is spread logarithmically between ``min_noise`` and ``max_noise`` at
  reset time and re-drawn whenever its episode ends, so that the parallel
  environments cover a wide range of exploration levels instead of all using
  the same one.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..envs.base import EpisodeTracker
from .actor_critic import ACTIVATIONS, MLP
from .base import TrainerBase


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, units: List[int], activation: str = "elu") -> None:
        super().__init__()
        self.trunk = MLP(obs_dim, units, activation)
        self.head = nn.Linear(self.trunk.out_dim, action_dim)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.trunk(obs)))


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, units: List[int], activation: str = "elu") -> None:
        super().__init__()
        self.trunk = MLP(obs_dim + action_dim, units, activation)
        self.head = nn.Linear(self.trunk.out_dim, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat([obs, actions], dim=-1))).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, actions, rewards, dones) -> None:
        n = obs.shape[0]
        idx = torch.arange(self.ptr, self.ptr + n, device=self.device) % self.capacity
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None):
        idx = torch.randint(0, self.size, (batch_size,), generator=generator, device=self.device)
        return (
            self.obs[idx],
            self.next_obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size


class PQLTrainer(TrainerBase):
    USES_POLICY_GRADIENT = False
    name = "pql"

    def __init__(self, cfg, env, device: str = "cpu", logdir: Optional[str] = None) -> None:
        super().__init__(cfg, env, device=device, logdir=logdir)
        units = list(self.model_cfg.actor_units)
        activation = self.model_cfg.activation if self.model_cfg.activation in ACTIVATIONS else "elu"
        self.actor = DeterministicActor(env.obs_dim, env.action_dim, units, activation).to(self.device)
        self.q_critic = QCritic(env.obs_dim, env.action_dim, units, activation).to(self.device)
        self.target_actor = DeterministicActor(env.obs_dim, env.action_dim, units, activation).to(self.device)
        self.target_q_critic = QCritic(env.obs_dim, env.action_dim, units, activation).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)
        self.replay = ReplayBuffer(
            int(self.a_get("replay_size", 1_000_000)), env.obs_dim, env.action_dim, self.device
        )
        self.batch_size = int(self.a_get("batch_size", 4096))
        self.updates_per_iteration = int(self.a_get("updates_per_iteration", 1))
        self.learning_starts = int(self.a_get("learning_starts", 100))
        self.polyak = float(self.a_get("polyak", 0.005))
        self.discount = self.gamma

        exploration = self.a_get("exploration", {}) or {}
        self.min_noise = float(exploration.get("min_noise", 0.05))
        self.max_noise = float(exploration.get("max_noise", 1.0))
        self.noise_scale = self._sample_noise_scales(env.num_envs)
        self.obs: Optional[torch.Tensor] = None
        self.prev_dones: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    def _sample_noise_scales(self, num_envs: int) -> torch.Tensor:
        # "different mixed exploration": a logarithmic spread of exploration
        # levels across the parallel environments.
        low, high = np.log(self.min_noise), np.log(self.max_noise)
        scale = np.exp(np.linspace(low, high, num_envs))
        return torch.as_tensor(np.random.permutation(scale), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ #
    def collect(self) -> None:
        if self.obs is None:
            self.obs = self.env.reset()
            self.prev_dones = torch.zeros(self.env.num_envs, device=self.device)
        for _ in range(self.horizon):
            with torch.no_grad():
                actions = self.actor(self.obs)
                noise = torch.randn_like(actions) * self.noise_scale[:, None]
                noisy_actions = torch.clamp(actions + noise, -1.0, 1.0)
            step = self.env.step(noisy_actions)
            self.replay.add(
                self.obs, step.obs, noisy_actions, step.rewards, (step.dones + step.timeouts).clamp(max=1.0)
            )
            self.obs = step.obs
            # environments that finished get a new exploration level
            finished = (step.dones + step.timeouts).clamp(max=1.0).bool()
            if bool(finished.any()):
                idx = finished.nonzero(as_tuple=False).squeeze(-1)
                self.noise_scale[idx] = self._sample_noise_scales(idx.numel())

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if len(self.replay) < max(self.learning_starts, self.batch_size):
            return metrics
        q_losses, actor_losses = [], []
        for _ in range(self.updates_per_iteration):
            obs, next_obs, actions, rewards, dones = self.replay.sample(self.batch_size, self.generator)
            with torch.no_grad():
                target_actions = self.target_actor(next_obs)
                target_q = self.target_q_critic(next_obs, target_actions)
                targets = rewards + self.discount * (1.0 - dones) * target_q
            q_values = self.q_critic(obs, actions)
            q_loss = ((q_values - targets) ** 2).mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            actor_loss = -self.q_critic(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self._soft_update(self.target_q_critic, self.q_critic)
            self._soft_update(self.target_actor, self.actor)
            q_losses.append(float(q_loss))
            actor_losses.append(float(actor_loss))
        metrics["pql/q_loss"] = float(np.mean(q_losses))
        metrics["pql/actor_loss"] = float(np.mean(actor_losses))
        metrics["pql/replay_size"] = float(len(self.replay))
        return metrics

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1.0 - self.polyak).add_(self.polyak * param)

    def train_iteration(self) -> Dict[str, float]:
        metrics = super().train_iteration()
        if self.obs is not None:
            metrics["pql/mean_noise"] = float(self.noise_scale.mean())
        return metrics

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate_policy(self, policy_id: int = 0, num_episodes: int = 32, max_steps: int = 512, deterministic: bool = True):
        obs = self.env.reset()
        tracker = EpisodeTracker(self.env.num_envs)
        for _ in range(max_steps):
            actions = self.actor(obs)
            step = self.env.step(actions)
            obs = step.obs
            tracker.step(
                step.rewards.detach().cpu().numpy(),
                step.infos.get("successes", step.rewards * 0).detach().cpu().numpy(),
                (step.dones + step.timeouts).clamp(max=1.0).detach().cpu().numpy().astype(bool),
            )
            if len(tracker.finished_returns) >= num_episodes:
                break
        stats = tracker.summary()
        self.obs = None  # restart the training rollout
        return stats

    # ------------------------------------------------------------------ #
    def save(self, tag: str = "final") -> str:
        import json
        import os

        ckpt_dir = os.path.join(self.logger.logdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q_critic": self.q_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        with open(os.path.join(ckpt_dir, f"{tag}.meta.json"), "w") as handle:
            json.dump({"iteration": self.iteration, "env_steps": self.env_steps}, handle)
        return path

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.q_critic.load_state_dict(payload["q_critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.iteration = int(payload.get("iteration", 0))
        self.env_steps = int(payload.get("env_steps", 0))
