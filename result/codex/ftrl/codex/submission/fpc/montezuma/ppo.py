"""PPO with Random Network Distillation and optional knowledge retention.

This is the training loop used for Montezuma's Revenge (Section 3, Appendix B.2).
It follows Burda et al. (2018) with the hyperparameters of Table 2 and adds the
knowledge-retention hooks studied in the paper:

* ``bc``  -- behavioral cloning on the trajectories collected with the
  pre-trained agent (the states are FAR states of the downstream task),
* ``ewc`` -- Elastic Weight Consolidation regularising the actor only,
* ``ks``  -- kickstarting (reported to fail on this environment, but implemented
  for completeness).

The trainer is independent of the concrete vectorised environment so that it can
be smoke-tested with :mod:`fpc.testing.mock_envs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..retention.base import RetentionConfig
from ..retention.distillation import kl_divergence
from ..retention.ewc import EWC
from .config import MontezumaConfig
from .model import AtariActorCritic
from .rnd import RND


@dataclass
class RolloutBatch:
    """A flat batch of transitions produced by the rollout workers."""

    observations: Tensor
    actions: Tensor
    log_probs: Tensor
    values: Tensor
    rewards: Tensor
    extrinsic_rewards: Tensor
    terminals: Tensor


class RolloutBuffer:
    """Storage for ``num_steps`` x ``num_env`` rollouts."""

    def __init__(self, num_steps: int, num_envs: int, obs_shape, device: str = "cpu") -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.observations = torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.extrinsic_rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.terminals = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.step = 0

    def add(self, **kwargs) -> None:
        for key, value in kwargs.items():
            getattr(self, key)[self.step] = value
        self.step += 1

    def reset(self) -> None:
        self.step = 0

    def flatten(self, start: int = 0, end: Optional[int] = None) -> RolloutBatch:
        end = end or self.num_steps
        sl = slice(start, end)
        return RolloutBatch(
            observations=self.observations[sl].reshape(-1, *self.observations.shape[2:]),
            actions=self.actions[sl].reshape(-1),
            log_probs=self.log_probs[sl].reshape(-1),
            values=self.values[sl].reshape(-1),
            rewards=self.rewards[sl].reshape(-1),
            extrinsic_rewards=self.extrinsic_rewards[sl].reshape(-1),
            terminals=self.terminals[sl].reshape(-1),
        )


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminals: Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: Tensor,
) -> Tensor:
    """Generalised Advantage Estimation over ``(num_steps, num_envs)`` tensors."""

    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = last_value
            next_non_terminal = 1.0 - terminals[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - terminals[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages


class BehavioralCloningBuffer:
    """Stores ``(observation, expert action)`` pairs collected from ``pi_*``.

    The paper collects 500 trajectories with the pre-trained PPO+RND agent
    (Appendix B.2) and applies a KL loss with a tunable weight (Figure 13).
    """

    def __init__(self, observations: np.ndarray, actions: np.ndarray, device: str = "cpu") -> None:
        self.observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(actions, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        idx = torch.randint(0, len(self), (batch_size,), device=self.observations.device)
        return self.observations[idx], self.actions[idx]


class PPORNDTrainer:
    """PPO + RND trainer with knowledge-retention hooks."""

    def __init__(
        self,
        config: MontezumaConfig,
        venv,
        device: str = "cpu",
        teacher: Optional[nn.Module] = None,
        bc_buffer: Optional[BehavioralCloningBuffer] = None,
    ) -> None:
        self.config = config
        self.venv = venv
        self.device = device
        self.model = AtariActorCritic(venv.num_actions).to(device)
        self.rnd = RND(config).to(device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=config.stable_eps)
        self.buffer = RolloutBuffer(config.num_step, venv.num_envs, venv.observation_shape, device)

        self.teacher = teacher.to(device) if teacher is not None else None
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.bc_buffer = bc_buffer
        self.retention = self._build_retention()

        self.obs = None
        self.global_step = 0

    # ------------------------------------------------------------------
    def _build_retention(self):
        cfg: RetentionConfig = self.config.retention
        if cfg.method in ("none", "em"):
            return None
        if cfg.method == "ewc":
            method = EWC(cfg)
            method.register_anchor(self.model, param_filter=lambda name: name.startswith("policy_head") or "trunk" in name or "fc" in name)
            return method
        if cfg.method in ("bc", "ks"):
            from ..retention.distillation import BehavioralCloning, Kickstarting

            return BehavioralCloning(cfg) if cfg.method == "bc" else Kickstarting(cfg)
        raise ValueError(f"Unsupported retention method {cfg.method!r}")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        obs, _ = self.venv.reset()
        self.obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def compute_fisher(self, prefix_batches: int = 10) -> None:
        """Estimate the diagonal Fisher from the pre-training distribution.

        The Fisher is computed on the *pre-training* states (the behavioral
        cloning buffer when available) using the pre-trained policy ``pi_*``.
        """

        from ..retention.fisher import DiagonalFisher

        if self.retention is None or not isinstance(self.retention, EWC):
            return
        if self.bc_buffer is None:
            raise RuntimeError("EWC requires a buffer of pre-training states to estimate the Fisher.")

        def log_prob_fn(observations, **__):
            logits, _ = self.model(observations)
            return torch.distributions.Categorical(logits=logits).log_prob(
                torch.distributions.Categorical(logits=logits).sample()
            )

        batches = [{"observations": self.bc_buffer.sample(128)[0]} for _ in range(prefix_batches)]
        fisher = DiagonalFisher.estimate(self.model, log_prob_fn, batches, num_batches=prefix_batches)
        self.retention.set_fisher(fisher)

    def _retention_loss(self, batch: RolloutBatch) -> Tensor:
        if self.retention is None:
            return torch.zeros((), device=self.device)
        if isinstance(self.retention, EWC):
            if self.retention.fisher is None:
                self.compute_fisher()
            return self.retention.aux_loss(self.model)
        if self.bc_buffer is None or self.teacher is None:
            return torch.zeros((), device=self.device)
        if self.config.retention.method == "bc":
            obs, _ = self.bc_buffer.sample(min(256, len(self.bc_buffer)))
        else:  # kickstarting -- use the states visited by the online policy
            idx = torch.randint(0, batch.observations.shape[0], (min(256, batch.observations.shape[0]),), device=self.device)
            obs = batch.observations[idx]
        with torch.no_grad():
            teacher_logits, _ = self.teacher(obs)
        student_logits, _ = self.model(obs)
        return kl_divergence(teacher_logits, student_logits, direction="forward").mean()

    # ------------------------------------------------------------------
    def collect_rollout(self) -> Dict[str, float]:
        self.buffer.reset()
        episode_returns: List[float] = []
        current_returns = np.zeros(self.venv.num_envs)

        with torch.no_grad():
            for _ in range(self.config.num_step):
                logits, values = self.model(self.obs)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                next_obs, rewards, dones = self.venv.step(actions.cpu().numpy())
                next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

                intrinsic, _ = self.rnd(next_obs_t)
                total_rewards = self.config.ext_coef * rewards_t + self.config.int_coef * intrinsic.squeeze(-1)

                current_returns += rewards
                for i, done in enumerate(dones):
                    if done:
                        episode_returns.append(float(current_returns[i]))
                        current_returns[i] = 0.0

                self.buffer.add(
                    observations=self.obs,
                    actions=actions,
                    log_probs=log_probs,
                    values=values.squeeze(-1),
                    rewards=total_rewards,
                    extrinsic_rewards=rewards_t,
                    terminals=dones_t,
                )
                self.obs = next_obs_t
                self.global_step += self.venv.num_envs

        with torch.no_grad():
            _, last_values = self.model(self.obs)
        return {
            "mean_extrinsic_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
            "num_episodes": len(episode_returns),
            "last_values": last_values.squeeze(-1),
        }

    # ------------------------------------------------------------------
    def update(self, last_values: Tensor) -> Dict[str, float]:
        cfg = self.config
        advantages = compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.terminals, cfg.gamma, cfg.gae_lambda, last_values
        )
        returns = advantages + self.buffer.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps, num_envs = cfg.num_step, self.venv.num_envs
        batch_size = num_steps * num_envs
        minibatch_size = max(batch_size // max(cfg.mini_batch, 1), 1)
        flat = self.buffer.flatten()
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        stats: Dict[str, float] = {"policy_loss": 0.0, "value_loss": 0.0, "ratio": 1.0}
        num_updates = 0
        for _ in range(cfg.epoch):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]
                obs = flat.observations[idx]
                actions = flat.actions[idx]
                old_log_probs = flat.log_probs[idx]
                adv = flat_advantages[idx]
                ret = flat_returns[idx]

                new_log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_eps, 1.0 + cfg.ppo_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(values, ret)
                entropy_loss = -cfg.entropy * entropy.mean()

                loss = policy_loss + 0.5 * value_loss + entropy_loss
                retention_loss = self._retention_loss(flat)
                if self.retention is not None:
                    loss = loss + self.retention.coefficient * retention_loss

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
                self.optimiser.step()

                if self.retention is not None:
                    self.retention.on_train_step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["ratio"] += float(ratio.mean().item())
                num_updates += 1

        # RND predictor: update on a sample of the collected observations.
        if cfg.update_proportion > 0:
            n = int(cfg.update_proportion * batch_size)
            idx = torch.randint(0, batch_size, (max(n, 1),), device=self.device)
            self.rnd.update(flat.observations[idx])

        for key in stats:
            stats[key] /= max(num_updates, 1)
        return stats

    # ------------------------------------------------------------------
    def train(self, total_steps: Optional[int] = None, callback: Optional[Callable] = None) -> Dict[str, list]:
        total_steps = total_steps or self.config.total_steps
        self.reset()
        history: Dict[str, list] = {"steps": [], "mean_extrinsic_return": [], "policy_loss": [], "value_loss": []}
        while self.global_step < total_steps:
            rollout = self.collect_rollout()
            stats = self.update(rollout["last_values"])
            history["steps"].append(self.global_step)
            history["mean_extrinsic_return"].append(rollout["mean_extrinsic_return"])
            history["policy_loss"].append(stats["policy_loss"])
            history["value_loss"].append(stats["value_loss"])
            if callback is not None:
                callback(self)
        return history
