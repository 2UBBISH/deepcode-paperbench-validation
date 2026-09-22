"""Parallel Q-Learning (PQL) baseline for SAPG.

Implements the PQL baseline from Li et al. (2023) "Efficient RL for In-Hand
Manipulation with a Large Number of Parallel Environments", used as a
comparison method in the SAPG paper (Sec. 5, Table 1).

PQL is an *off-policy* actor-critic method that:
  * collects data from a large number of parallel environments,
  * stores transitions in a large replay buffer,
  * performs off-policy Q-learning style updates (TD(0) critic targets),
  * uses a single policy (no per-policy latent conditioning).

The trainer mirrors the interface of :class:`sapg.sapg.algorithm.SAPGTrainer`
and :class:`sapg.baselines.ppo.PPOTrainer` so that ``main.py`` can dispatch to
it uniformly (``train`` / ``state_dict`` / ``load_state_dict``).

Reference results (Table 1, after 2e10 samples):
    Regrasping   2.73 +/- 0.02
    Throw        2.62 +/- 0.08
    Reorientation 1.66 +/- 0.11
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..sapg.config import SAPGConfig
from ..sapg.networks import SAPGNetworks, build_networks
from ..sapg.rollout import _unpack_step

__all__ = ["PQLTrainer", "PQLUpdateStats", "ReplayBuffer", "build_pql_trainer"]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity circular replay buffer of transitions.

    Stores ``(obs, action, reward, next_obs, done)`` tuples on the target
    device.  Sampling returns uniformly random minibatches, matching the
    off-policy Q-learning update used by PQL.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device if device is not None else torch.device("cpu")

        self.obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.capacity, self.action_dim, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.next_obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self.dones = torch.zeros(self.capacity, device=self.device)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add a batch of transitions (flattened over envs/time)."""
        obs = obs.reshape(-1, self.obs_dim)
        actions = actions.reshape(-1, self.action_dim)
        rewards = rewards.reshape(-1)
        next_obs = next_obs.reshape(-1, self.obs_dim)
        dones = dones.reshape(-1).float()

        n = obs.shape[0]
        if n == 0:
            return

        # Handle wrap-around in one or two chunks.
        end = self.ptr + n
        if end <= self.capacity:
            self._write(self.ptr, end, obs, actions, rewards, next_obs, dones)
        else:
            first = self.capacity - self.ptr
            self._write(self.ptr, self.capacity, obs[:first], actions[:first],
                        rewards[:first], next_obs[:first], dones[:first])
            rest = n - first
            self._write(0, rest, obs[first:], actions[first:], rewards[first:],
                        next_obs[first:], dones[first:])
        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def _write(self, start, end, obs, actions, rewards, next_obs, dones) -> None:
        self.obs[start:end] = obs
        self.actions[start:end] = actions
        self.rewards[start:end] = rewards
        self.next_obs[start:end] = next_obs
        self.dones[start:end] = dones

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None):
        """Sample a uniform random minibatch of transitions."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device,
                            generator=generator)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@dataclass
class PQLUpdateStats:
    """Per-iteration statistics for the PQL baseline."""

    iteration: int = 0
    total_transitions: int = 0
    q_loss: float = 0.0
    policy_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    mean_reward: float = 0.0
    mean_episode_reward: float = 0.0
    successes: float = 0.0
    buffer_size: int = 0
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "iteration": self.iteration,
            "total_transitions": self.total_transitions,
            "q_loss": self.q_loss,
            "policy_loss": self.policy_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "mean_reward": self.mean_reward,
            "mean_episode_reward": self.mean_episode_reward,
            "successes": self.successes,
            "buffer_size": self.buffer_size,
            "wall_time": self.wall_time,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PQLTrainer:
    """Parallel Q-Learning baseline trainer.

    Uses a single policy (no latent conditioning) and an off-policy
    Q-learning update with a large replay buffer.  The critic is trained with
    a TD(0) target ``r + gamma * (1 - done) * V(s')`` and the actor is updated
    with a deterministic-policy-gradient / DDPG-style objective using the
    critic's value estimate as the advantage proxy.
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
        self.device = device if device is not None else torch.device(
            getattr(config, "device", "cpu")
        )

        # Single policy: latent_dim = 0 (no per-policy conditioning).
        self.networks = build_networks(config, self.obs_dim, self.action_dim)
        self.networks.to(self.device)

        # Replay buffer (capacity from config, default 1e6 transitions).
        capacity = int(getattr(config, "replay_buffer_size", 1_000_000))
        self.buffer = ReplayBuffer(capacity, self.obs_dim, self.action_dim,
                                   device=self.device)

        # Optimizers: separate LR for Q-function (q_lr) and policy (lr).
        q_lr = float(getattr(config, "q_lr", config.learning_rate))
        self.q_optimizer = torch.optim.Adam(
            self.networks.critic.parameters(), lr=q_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.networks.actor.parameters(), lr=config.learning_rate
        )

        self.gamma = float(config.gamma)
        self.grad_norm_clip = float(getattr(config, "grad_norm_clip", 1.0))
        self.minibatch_size = int(config.minibatch_size)
        self.num_mini_epochs = int(config.num_mini_epochs)
        self.updates_per_iteration = int(
            getattr(config, "pql_updates_per_iteration", 1)
        )

        # Rollout state.
        self.obs: Optional[torch.Tensor] = None
        self.lstm_state = None
        self.total_transitions = 0
        self.iteration = 0

        # Episode statistics (rolling window).
        self._episode_rewards: List[float] = []
        self._episode_successes: List[float] = []
        self._current_ep_reward = torch.zeros(
            config.num_envs, device=self.device
        )
        self._current_ep_success = torch.zeros(
            config.num_envs, device=self.device
        )
        self._window = 10000

    # -- rollout -----------------------------------------------------------
    def collect_rollout(self) -> Dict[str, torch.Tensor]:
        """Collect one horizon of transitions from all parallel envs."""
        cfg = self.config
        horizon = int(cfg.horizon)
        num_envs = int(cfg.num_envs)

        if self.obs is None:
            self.obs = self.env.reset()
            self.obs = self._to_tensor(self.obs)

        obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []

        for _ in range(horizon):
            with torch.no_grad():
                actions, _, self.lstm_state = self.networks.actor_act(
                    self.obs, latent=None, lstm_state=self.lstm_state,
                    deterministic=False,
                )
            step_out = self.env.step(actions)
            next_obs, rewards, dones, infos = _unpack_step(step_out)
            next_obs = self._to_tensor(next_obs)
            rewards = self._to_tensor(rewards).reshape(-1)
            dones = self._to_tensor(dones).reshape(-1)

            obs_list.append(self.obs)
            act_list.append(actions)
            rew_list.append(rewards)
            next_list.append(next_obs)
            done_list.append(dones)

            # Track episode statistics.
            self._current_ep_reward += rewards
            if "successes" in infos:
                succ = self._to_tensor(infos["successes"]).reshape(-1)
                self._current_ep_success += succ
            done_mask = dones > 0.5
            if done_mask.any():
                self._episode_rewards.extend(
                    self._current_ep_reward[done_mask].tolist()
                )
                self._episode_successes.extend(
                    self._current_ep_success[done_mask].tolist()
                )
                self._current_ep_reward[done_mask] = 0.0
                self._current_ep_success[done_mask] = 0.0
                if len(self._episode_rewards) > self._window:
                    self._episode_rewards = self._episode_rewards[-self._window:]
                    self._episode_successes = self._episode_successes[-self._window:]

            self.obs = next_obs

        batch = {
            "obs": torch.stack(obs_list, dim=0),
            "actions": torch.stack(act_list, dim=0),
            "rewards": torch.stack(rew_list, dim=0),
            "next_obs": torch.stack(next_list, dim=0),
            "dones": torch.stack(done_list, dim=0),
        }
        self.total_transitions += horizon * num_envs
        return batch

    # -- update ------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """Run off-policy Q-learning + policy updates on replay minibatches."""
        if len(self.buffer) < self.minibatch_size:
            return {"q_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0,
                    "approx_kl": 0.0, "grad_norm": 0.0}

        q_losses, p_losses, entropies, kls, grad_norms = [], [], [], [], []

        for _ in range(self.updates_per_iteration):
            for _ in range(self.num_mini_epochs):
                batch = self.buffer.sample(self.minibatch_size)
                obs = batch["obs"]
                actions = batch["actions"]
                rewards = batch["rewards"]
                next_obs = batch["next_obs"]
                dones = batch["dones"]

                # --- Critic (Q-function) update: TD(0) target ---
                with torch.no_grad():
                    next_values, _ = self.networks.critic_value(
                        next_obs, latent=None
                    )
                    next_values = next_values.reshape(-1)
                    targets = rewards + self.gamma * (1.0 - dones) * next_values

                values, _ = self.networks.critic_value(obs, latent=None)
                values = values.reshape(-1)
                q_loss = 0.5 * ((values - targets) ** 2).mean()

                self.q_optimizer.zero_grad(set_to_none=True)
                q_loss.backward()
                q_grad_norm = nn.utils.clip_grad_norm_(
                    self.networks.critic.parameters(), self.grad_norm_clip
                )
                self.q_optimizer.step()

                # --- Actor update: maximize Q(s, pi(s)) ---
                dist, _ = self.networks.actor_distribution(obs, latent=None)
                new_actions = dist.rsample()
                new_log_probs = dist.log_prob(new_actions).sum(dim=-1)
                q_vals, _ = self.networks.critic_value(obs, latent=None)
                q_vals = q_vals.reshape(-1)

                # Advantage proxy: Q(s, pi(s)) - V(s) baseline (detached).
                advantage = (q_vals - q_vals.mean()).detach()
                policy_loss = -(new_log_probs * advantage).mean()
                entropy = dist.entropy().sum(dim=-1).mean()

                self.policy_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                p_grad_norm = nn.utils.clip_grad_norm_(
                    self.networks.actor.parameters(), self.grad_norm_clip
                )
                self.policy_optimizer.step()

                # Approximate KL between old and new policy (diagnostic).
                with torch.no_grad():
                    old_log_probs = new_log_probs.detach()
                    approx_kl = (old_log_probs - new_log_probs).mean().abs()

                q_losses.append(float(q_loss.detach()))
                p_losses.append(float(policy_loss.detach()))
                entropies.append(float(entropy.detach()))
                kls.append(float(approx_kl))
                grad_norms.append(float(q_grad_norm) + float(p_grad_norm))

        return {
            "q_loss": _mean(q_losses),
            "policy_loss": _mean(p_losses),
            "entropy": _mean(entropies),
            "approx_kl": _mean(kls),
            "grad_norm": _mean(grad_norms),
        }

    # -- training loop -----------------------------------------------------
    def train_iteration(self) -> PQLUpdateStats:
        t0 = time.time()
        batch = self.collect_rollout()

        # Store transitions in the replay buffer.
        self.buffer.add(
            batch["obs"], batch["actions"], batch["rewards"],
            batch["next_obs"], batch["dones"],
        )

        metrics = self.update()

        self.iteration += 1
        stats = PQLUpdateStats(
            iteration=self.iteration,
            total_transitions=self.total_transitions,
            q_loss=metrics["q_loss"],
            policy_loss=metrics["policy_loss"],
            entropy=metrics["entropy"],
            approx_kl=metrics["approx_kl"],
            learning_rate=self.config.learning_rate,
            grad_norm=metrics["grad_norm"],
            mean_reward=float(batch["rewards"].mean()),
            mean_episode_reward=_mean(self._episode_rewards),
            successes=_mean(self._episode_successes),
            buffer_size=len(self.buffer),
            wall_time=time.time() - t0,
        )
        return stats

    def train(self, num_iterations: int) -> List[PQLUpdateStats]:
        history: List[PQLUpdateStats] = []
        for _ in range(int(num_iterations)):
            history.append(self.train_iteration())
        return history

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "networks": self.networks.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "buffer": {
                "obs": self.buffer.obs,
                "actions": self.buffer.actions,
                "rewards": self.buffer.rewards,
                "next_obs": self.buffer.next_obs,
                "dones": self.buffer.dones,
                "ptr": self.buffer.ptr,
                "size": self.buffer.size,
            },
            "total_transitions": self.total_transitions,
            "iteration": self.iteration,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.networks.load_state_dict(state["networks"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        buf = state.get("buffer")
        if buf is not None:
            self.buffer.obs.copy_(buf["obs"])
            self.buffer.actions.copy_(buf["actions"])
            self.buffer.rewards.copy_(buf["rewards"])
            self.buffer.next_obs.copy_(buf["next_obs"])
            self.buffer.dones.copy_(buf["dones"])
            self.buffer.ptr = buf["ptr"]
            self.buffer.size = buf["size"]
        self.total_transitions = state.get("total_transitions", 0)
        self.iteration = state.get("iteration", 0)

    # -- helpers -----------------------------------------------------------
    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device).float()
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_pql_trainer(
    config: SAPGConfig,
    env: Any,
    obs_dim: int,
    action_dim: int,
    device: Optional[torch.device] = None,
) -> PQLTrainer:
    """Factory for :class:`PQLTrainer`."""
    return PQLTrainer(config, env, obs_dim, action_dim, device=device)
