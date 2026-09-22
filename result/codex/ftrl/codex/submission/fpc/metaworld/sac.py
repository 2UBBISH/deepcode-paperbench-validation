"""Soft Actor-Critic for RoboticSequence (Haarnoja et al., 2018a; Appendix B.3).

Architecture (Appendix B.3):

* 4 hidden layers with 256 neurons and Leaky-ReLU activations, layer
  normalisation after the first layer, for both the policy and the Q-functions,
* a **separate output head per stage**; the stage ID selects the head,
* automatic entropy tuning (Haarnoja et al., 2018b),
* Adam with learning rate ``1e-3`` and batch size ``128``.

Knowledge retention is applied to the actor only (Appendix C.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from ..retention.base import RetentionConfig
from ..retention.distillation import kl_divergence
from ..retention.episodic_memory import ProtectedReplayBuffer
from ..retention.ewc import EWC
from .config import MetaworldConfig


LOG_STD_MIN, LOG_STD_MAX = -20, 2


class MLP(nn.Module):
    """4-layer MLP with Leaky-ReLU and layer norm after the first layer."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4, out_dim: int = 1,
                 layer_norm_first: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for i in range(num_layers):
            linear = nn.Linear(last, hidden_dim)
            layers.append(linear)
            if i == 0 and layer_norm_first:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadActor(nn.Module):
    """Gaussian actor with one output head per stage."""

    def __init__(self, obs_dim: int, action_dim: int, num_stages: int,
                 hidden_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        self.trunk = MLP(obs_dim, hidden_dim, num_layers, out_dim=hidden_dim)
        self.mu_heads = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(num_stages)])
        self.log_std_heads = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(num_stages)])

    def forward(self, obs: Tensor, stage: Tensor, with_logprob: bool = True):
        stage = stage.reshape(-1).long()
        if int(stage.min()) < 0 or int(stage.max()) >= len(self.mu_heads):
            raise ValueError(
                f"stage ids must be in [0, {len(self.mu_heads)}); got "
                f"[{int(stage.min())}, {int(stage.max())}]"
            )
        h = self.trunk(obs)
        mu = torch.empty(obs.shape[0], self.mu_heads[0].out_features, device=obs.device)
        log_std = torch.empty_like(mu)
        for s in range(len(self.mu_heads)):
            mask = stage == s
            if mask.any():
                mu[mask] = self.mu_heads[s](h[mask])
                log_std[mask] = self.log_std_heads[s](h[mask])
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        if not with_logprob:
            return action, None, mu, std
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mu, std

    def distribution(self, obs: Tensor, stage: Tensor) -> Normal:
        with torch.no_grad():
            _, _, mu, std = self.forward(obs, stage, with_logprob=False)
        return Normal(mu, std)


class MultiHeadCritic(nn.Module):
    """Twin Q-functions with one head per stage."""

    def __init__(self, obs_dim: int, action_dim: int, num_stages: int,
                 hidden_dim: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, hidden_dim, num_layers, out_dim=num_stages)
        self.q2 = MLP(obs_dim + action_dim, hidden_dim, num_layers, out_dim=num_stages)

    def forward(self, obs: Tensor, action: Tensor, stage: Tensor) -> Tuple[Tensor, Tensor]:
        stage = stage.reshape(-1).long()
        if int(stage.min()) < 0 or int(stage.max()) >= self.q1.net[-1].out_features:
            raise ValueError("stage ids out of range for the critic heads")
        x = torch.cat([obs, action], dim=-1)
        q1_all, q2_all = self.q1(x), self.q2(x)
        idx = stage.unsqueeze(-1)
        return q1_all.gather(1, idx), q2_all.gather(1, idx)


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: float
    stage: int
    next_stage: int


class ReplayBuffer:
    """Simple uniform replay buffer (optionally wrapping a protected region)."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int,
                 protected: Optional[ProtectedReplayBuffer] = None) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.stages = np.zeros((capacity, 1), dtype=np.int64)
        self.next_stages = np.zeros((capacity, 1), dtype=np.int64)
        self.protected = protected
        self.ptr = protected.protected_size if protected is not None else 0
        self.size = 0

    # ------------------------------------------------------------------
    def add(self, transition: Transition) -> None:
        idx = self.ptr
        self.obs[idx] = transition.obs
        self.next_obs[idx] = transition.next_obs
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.dones[idx] = transition.done
        self.stages[idx] = transition.stage
        self.next_stages[idx] = transition.next_stage
        protected = self.protected.protected_size if self.protected is not None else 0
        self.ptr = protected + (self.ptr - protected + 1) % max(self.capacity - protected, 1)
        self.size = min(self.size + 1, self.capacity)

    def load_protected(self, data: Dict[str, np.ndarray]) -> None:
        n = min(len(data["obs"]), self.protected.protected_size)
        for i in range(n):
            self.obs[i] = data["obs"][i]
            self.next_obs[i] = data["next_obs"][i]
            self.actions[i] = data["actions"][i]
            self.rewards[i] = data["rewards"][i]
            self.dones[i] = data["dones"][i]
            self.stages[i] = data["stages"][i]
            self.next_stages[i] = data["next_stages"][i]
        self.size = max(self.size, n)

    @property
    def valid_size(self) -> int:
        protected = self.protected.protected_size if self.protected is not None else 0
        return max(self.size, protected)

    def sample(self, batch_size: int, device: str = "cpu"):
        idx = np.random.randint(0, self.valid_size, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], dtype=torch.float32, device=device)  # noqa: E731
        return {
            "obs": to(self.obs),
            "next_obs": to(self.next_obs),
            "actions": to(self.actions),
            "rewards": to(self.rewards),
            "dones": to(self.dones),
            "stages": torch.as_tensor(self.stages[idx], dtype=torch.long, device=device),
            "next_stages": torch.as_tensor(self.next_stages[idx], dtype=torch.long, device=device),
        }


class SAC:
    """Soft Actor-Critic with optional knowledge retention (EWC / BC / EM)."""

    def __init__(
        self,
        config: MetaworldConfig,
        obs_dim: int,
        action_dim: int,
        num_stages: int,
        device: str = "cpu",
        teacher: Optional[MultiHeadActor] = None,
        bc_dataset: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.action_dim = action_dim
        self.num_stages = num_stages

        self.actor = MultiHeadActor(obs_dim, action_dim, num_stages, config.hidden_dim, config.num_layers).to(device)
        self.critic = MultiHeadCritic(obs_dim, action_dim, num_stages, config.hidden_dim, config.num_layers).to(device)
        self.critic_target = MultiHeadCritic(obs_dim, action_dim, num_stages, config.hidden_dim, config.num_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        # Automatic entropy tuning (Haarnoja et al., 2018b).
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

        # ``teacher`` may be either a :class:`MultiHeadActor` or a full
        # :class:`SAC` agent (whose ``.actor`` is used as the frozen teacher).
        if teacher is not None and isinstance(teacher, SAC):
            teacher = teacher.actor
        self.teacher = teacher.to(device) if teacher is not None else None
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.bc_dataset = bc_dataset
        self.retention = self._build_retention()

    # ------------------------------------------------------------------
    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def _build_retention(self):
        cfg: RetentionConfig = self.config.retention
        if cfg.method in ("none", "em"):
            return None
        if cfg.method == "ewc":
            method = EWC(cfg)
            method.register_anchor(self.actor)
            return method
        if cfg.method == "bc":
            from ..retention.distillation import BehavioralCloning

            return BehavioralCloning(cfg)
        return None

    def _retention_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        if self.retention is None or self.teacher is None:
            return torch.zeros((), device=self.device)
        if isinstance(self.retention, EWC):
            return self.retention.aux_loss(self.actor)
        # BC: KL between the frozen teacher and the current actor on states from
        # the pre-training distribution.
        n = min(self.config.batch_size, len(self.bc_dataset["obs"]))
        idx = np.random.randint(0, len(self.bc_dataset["obs"]), size=n)
        obs = torch.as_tensor(self.bc_dataset["obs"][idx], dtype=torch.float32, device=self.device)
        stage = torch.as_tensor(self.bc_dataset["stages"][idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            teacher_dist = self.teacher.distribution(obs, stage)
            teacher_mu = teacher_dist.loc
            teacher_std = teacher_dist.scale
        _, _, mu, std = self.actor(obs, stage, with_logprob=False)
        # Gaussian KL between diagonal normals, then tanh squashing correction.
        kl = (
            torch.log(std / teacher_std)
            + (teacher_std ** 2 + (teacher_mu - mu) ** 2) / (2 * std ** 2)
            - 0.5
        ).sum(dim=-1).mean()
        return kl

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, stage: int, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        stage_t = torch.as_tensor([stage], dtype=torch.long, device=self.device)
        action, _, mu, std = self.actor(obs_t, stage_t, with_logprob=False)
        if deterministic:
            action = torch.tanh(mu)
        return action.squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer) -> Dict[str, float]:
        cfg = self.config
        batch = replay.sample(cfg.batch_size, self.device)
        obs, actions = batch["obs"], batch["actions"]
        rewards, dones = batch["rewards"], batch["dones"]
        next_obs, stages, next_stages = batch["next_obs"], batch["stages"], batch["next_stages"]

        with torch.no_grad():
            next_actions, next_log_prob, _, _ = self.actor(next_obs, next_stages)
            q1_next, q2_next = self.critic_target(next_obs, next_actions, next_stages)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rewards + cfg.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(obs, actions, stages)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        new_actions, log_prob, _, _ = self.actor(obs, stages)
        q1_new, q2_new = self.critic(obs, new_actions, stages)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        if self.retention is not None:
            retention_loss = self._retention_loss(batch)
            actor_loss = actor_loss + self.retention.coefficient * retention_loss
            self.retention.on_train_step()

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()

        # Polyak averaging of the target critic.
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    # ------------------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                **(extra or {}),
            },
            path,
        )
