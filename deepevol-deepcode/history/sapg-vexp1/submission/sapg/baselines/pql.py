"""Parallel Q-Learning (PQL) baseline.

Reference: Li et al., 2023 ("Parallel Q-Learning: Scaling Off-policy Reinforcement
Learning under Massively Parallel Simulation").  PQL is an off-policy actor-critic
(DDPG-style) algorithm designed to exploit thousands of parallel environments.
Its key ingredients are:

  * A deterministic actor mu(s) and a Q-critic Q(s, a).
  * Target networks for both actor and critic (Polyak averaging with ``tau``).
  * A large replay buffer filled by all parallel envs.
  * *Mixed exploration*: each parallel environment is assigned a different
    exploration strategy (different noise scale / noise type), so the replay
    buffer is filled with diverse behaviour without hurting any single env's
    exploitation.  Concretely, env ``i`` uses noise scale
    ``sigma_i = sigma_min * (sigma_max / sigma_min) ** (i / (N - 1))`` and
    alternates between Gaussian and Ornstein-Uhlenbeck style temporally
    correlated noise.

The implementation below is simulator-agnostic: it consumes an ``env_factory``
returning a vectorized environment exposing ``reset()``, ``step(actions)``,
``num_envs``, ``obs_dim`` and ``action_dim`` (see ``envs/isaacgym_wrapper.py``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..networks import build_actor, build_critic


__all__ = ["PQLConfig", "ReplayBuffer", "PQLTrainer", "train_pql"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PQLConfig:
    """Hyper-parameters for the Parallel Q-Learning baseline."""

    task: str = "allegrokuka_regrasping"
    num_envs: int = 24576
    horizon: int = 16

    # RL / DDPG hyper-parameters
    gamma: float = 0.99
    tau: float = 0.005                 # Polyak averaging coefficient
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    learning_rate: float = 3.0e-4      # alias used by generic config plumbing
    grad_norm_clip: float = 1.0
    batch_size: int = 4096
    updates_per_iter: int = 1
    warmup_transitions: int = 100_000

    # Replay buffer
    buffer_size: int = 5_000_000
    buffer_device: str = "cpu"

    # Mixed exploration
    sigma_min: float = 0.05
    sigma_max: float = 0.5
    ou_theta: float = 0.15
    ou_dt: float = 0.01
    action_clip: float = 1.0

    # Architecture
    phi_dim: int = 0                   # PQL has no per-policy latent
    recurrent: bool = False
    lstm_hidden: int = 768
    obs_dim: int = 44
    action_dim: int = 23

    # Training budget
    target_transitions: float = 2.0e10
    max_iterations: int = 100_000
    device: str = "cuda"
    output_dir: str = "runs/pql"
    seed: int = 0
    log_interval: int = 1
    save_interval: int = 100
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity circular replay buffer stored on a single device."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

        self.obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self.next_obs = torch.zeros(self.capacity, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.capacity, self.action_dim, device=self.device)
        self.rewards = torch.zeros(self.capacity, 1, device=self.device)
        self.dones = torch.zeros(self.capacity, 1, device=self.device)

        self.ptr = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add a batch of transitions (all tensors have leading dim = batch)."""
        obs = obs.detach().to(self.device).reshape(-1, self.obs_dim)
        actions = actions.detach().to(self.device).reshape(-1, self.action_dim)
        rewards = rewards.detach().to(self.device).reshape(-1, 1)
        next_obs = next_obs.detach().to(self.device).reshape(-1, self.obs_dim)
        dones = dones.detach().to(self.device).reshape(-1, 1)

        n = obs.shape[0]
        if n == 0:
            return

        # Handle wrap-around in at most two chunks.
        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr:end] = obs
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
            self.next_obs[self.ptr:end] = next_obs
            self.dones[self.ptr:end] = dones
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            self.next_obs[self.ptr:] = next_obs[:first]
            self.dones[self.ptr:] = dones[:first]

            rest = n - first
            self.obs[:rest] = obs[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.dones[:rest] = dones[first:]

        self.ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = self._rng.integers(0, self.size, size=int(batch_size))
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        return {
            "obs": self.obs[idx_t],
            "actions": self.actions[idx_t],
            "rewards": self.rewards[idx_t],
            "next_obs": self.next_obs[idx_t],
            "dones": self.dones[idx_t],
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ptr": self.ptr,
            "size": self.size,
            "obs": self.obs.cpu(),
            "next_obs": self.next_obs.cpu(),
            "actions": self.actions.cpu(),
            "rewards": self.rewards.cpu(),
            "dones": self.dones.cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ptr = int(state["ptr"])
        self.size = int(state["size"])
        self.obs = state["obs"].to(self.device)
        self.next_obs = state["next_obs"].to(self.device)
        self.actions = state["actions"].to(self.device)
        self.rewards = state["rewards"].to(self.device)
        self.dones = state["dones"].to(self.device)


# ---------------------------------------------------------------------------
# Mixed exploration noise
# ---------------------------------------------------------------------------
class MixedExploration:
    """Per-environment mixed exploration noise (Gaussian + OU).

    Each of the ``num_envs`` parallel environments receives a distinct noise
    scale, log-spaced between ``sigma_min`` and ``sigma_max``.  Half of the
    environments use temporally-uncorrelated Gaussian noise while the other
    half use Ornstein-Uhlenbeck noise, yielding a diverse replay buffer.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        theta: float = 0.15,
        dt: float = 0.01,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.theta = float(theta)
        self.dt = float(dt)

        if self.num_envs > 1:
            frac = torch.linspace(0.0, 1.0, self.num_envs, device=self.device)
            sigmas = float(sigma_min) * (
                (float(sigma_max) / float(sigma_min)) ** frac
            )
        else:
            sigmas = torch.full((1,), float(sigma_max), device=self.device)
        self.sigmas = sigmas.view(-1, 1)

        # Alternate noise type per environment.
        self.use_ou = torch.zeros(self.num_envs, 1, device=self.device)
        self.use_ou[1::2] = 1.0

        self.ou_state = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            self.ou_state.zero_()
        else:
            mask = mask.view(-1, 1).to(self.device)
            self.ou_state = self.ou_state * (1.0 - mask)

    def sample(self) -> torch.Tensor:
        gaussian = torch.randn(
            self.num_envs,
            self.action_dim,
            device=self.device,
            generator=self._gen,
        )
        # OU update: dx = theta * (-x) * dt + sigma * sqrt(dt) * N(0, 1)
        noise = torch.randn(
            self.num_envs,
            self.action_dim,
            device=self.device,
            generator=self._gen,
        )
        self.ou_state = (
            self.ou_state
            + self.theta * (-self.ou_state) * self.dt
            + self.sigmas * float(np.sqrt(self.dt)) * noise
        )
        mixed = torch.where(self.use_ou.bool(), self.ou_state, gaussian * self.sigmas)
        return mixed


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PQLTrainer:
    """Parallel Q-Learning trainer (DDPG + mixed exploration)."""

    def __init__(
        self,
        config: PQLConfig,
        env_factory: Callable[[int, int], Any],
        logger: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.device = torch.device(
            config.device if torch.cuda.is_available() or "cpu" in str(config.device)
            else "cpu"
        )

        self.env = env_factory(config.num_envs, config.seed)
        self.num_envs = int(getattr(self.env, "num_envs", config.num_envs))
        self.obs_dim = int(getattr(self.env, "obs_dim", config.obs_dim))
        self.action_dim = int(getattr(self.env, "action_dim", config.action_dim))

        # Deterministic actor + Q critic (no phi conditioning for PQL).
        self.actor = build_actor(
            config.task, self.obs_dim, self.action_dim, phi_dim=0
        ).to(self.device)
        self.critic = build_critic(config.task, self.obs_dim, phi_dim=0).to(self.device)

        # Q head: critic backbone output -> scalar Q(s, a).
        critic_hidden = self._critic_hidden_dim()
        self.q_head = nn.Sequential(
            nn.Linear(critic_hidden + self.action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ).to(self.device)

        # Target networks.
        self.actor_target = build_actor(
            config.task, self.obs_dim, self.action_dim, phi_dim=0
        ).to(self.device)
        self.critic_target = build_critic(
            config.task, self.obs_dim, phi_dim=0
        ).to(self.device)
        self.q_head_target = nn.Sequential(
            nn.Linear(critic_hidden + self.action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.q_head_target.load_state_dict(self.q_head.state_dict())

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optim = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.q_head.parameters()),
            lr=config.critic_lr,
        )

        self.buffer = ReplayBuffer(
            capacity=config.buffer_size,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=config.buffer_device,
            seed=config.seed,
        )
        self.exploration = MixedExploration(
            num_envs=self.num_envs,
            action_dim=self.action_dim,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            theta=config.ou_theta,
            dt=config.ou_dt,
            device=self.device,
            seed=config.seed,
        )

        self.iteration = 0
        self.transitions = 0
        self._obs: Optional[torch.Tensor] = None
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._recent_returns: list = []
        self._recent_successes: list = []

    # -- helpers ----------------------------------------------------------
    def _critic_hidden_dim(self) -> int:
        """Infer the output width of the critic backbone."""
        for module in reversed(list(self.critic.modules())):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
        return 256

    def _to_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device).float()
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)

    def _actor_mean(self, obs: torch.Tensor, actor: Optional[nn.Module] = None) -> torch.Tensor:
        actor = actor if actor is not None else self.actor
        out = actor(obs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _q_value(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        critic: Optional[nn.Module] = None,
        q_head: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        critic = critic if critic is not None else self.critic
        q_head = q_head if q_head is not None else self.q_head
        feat = critic(obs)
        if isinstance(feat, tuple):
            feat = feat[0]
        return q_head(torch.cat([feat, actions], dim=-1))

    # -- rollout ----------------------------------------------------------
    def collect_rollouts(self) -> Dict[str, float]:
        """Collect ``horizon`` steps of experience from all parallel envs."""
        if self._obs is None:
            self._obs = self._to_tensor(self.env.reset())

        total_reward = 0.0
        num_success = 0
        num_done = 0

        for _ in range(int(self.config.horizon)):
            with torch.no_grad():
                mean = self._actor_mean(self._obs)
                noise = self.exploration.sample()
                actions = torch.clamp(
                    mean + noise,
                    -self.config.action_clip,
                    self.config.action_clip,
                )

            obs_np = self._obs.detach().cpu().numpy()
            act_np = actions.detach().cpu().numpy()
            next_obs_np, rewards, dones, infos = self.env.step(act_np)

            next_obs = self._to_tensor(next_obs_np)
            rewards_t = self._to_tensor(rewards).reshape(-1, 1)
            dones_t = self._to_tensor(dones).reshape(-1, 1)

            self.buffer.add(self._obs, actions, rewards_t, next_obs, dones_t)

            total_reward += float(rewards_t.sum().item())
            self._episode_returns += rewards_t.detach().cpu().numpy().reshape(-1)
            self._episode_lengths += 1

            done_mask = dones_t.detach().cpu().numpy().reshape(-1) > 0.5
            if done_mask.any():
                num_done += int(done_mask.sum())
                for i in np.nonzero(done_mask)[0]:
                    self._recent_returns.append(float(self._episode_returns[i]))
                    self._episode_returns[i] = 0.0
                    self._episode_lengths[i] = 0
                self.exploration.reset(dones_t)

            if isinstance(infos, dict):
                succ = infos.get("success", None)
                if succ is not None:
                    succ_arr = np.asarray(succ).reshape(-1)
                    num_success += int(np.sum(succ_arr > 0.5))
                    self._recent_successes.extend(
                        [float(s) for s in succ_arr[succ_arr > 0.5]]
                    )

            self._obs = next_obs
            self.transitions += self.num_envs

        if len(self._recent_returns) > 5000:
            self._recent_returns = self._recent_returns[-5000:]
        if len(self._recent_successes) > 5000:
            self._recent_successes = self._recent_successes[-5000:]

        return {
            "rollout/reward": total_reward / max(1, self.num_envs * self.config.horizon),
            "rollout/episodes": float(num_done),
            "rollout/successes": float(num_success),
            "buffer/size": float(len(self.buffer)),
        }

    # -- update -----------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """Run ``updates_per_iter`` DDPG gradient steps."""
        if len(self.buffer) < max(self.config.batch_size, self.config.warmup_transitions):
            return {"update/skipped": 1.0}

        metrics: Dict[str, float] = {}
        for _ in range(int(self.config.updates_per_iter)):
            batch = self.buffer.sample(self.config.batch_size)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_obs"]
            dones = batch["dones"]

            # --- critic update -------------------------------------------
            with torch.no_grad():
                next_actions = self._actor_mean(next_obs, actor=self.actor_target)
                next_actions = torch.clamp(
                    next_actions, -self.config.action_clip, self.config.action_clip
                )
                q_next = self._q_value(
                    next_obs,
                    next_actions,
                    critic=self.critic_target,
                    q_head=self.q_head_target,
                )
                target = rewards + self.config.gamma * (1.0 - dones) * q_next

            q_pred = self._q_value(obs, actions)
            critic_loss = nn.functional.mse_loss(q_pred, target)

            self.critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.config.grad_norm_clip:
                nn.utils.clip_grad_norm_(
                    list(self.critic.parameters()) + list(self.q_head.parameters()),
                    self.config.grad_norm_clip,
                )
            self.critic_optim.step()

            # --- actor update --------------------------------------------
            pred_actions = self._actor_mean(obs)
            pred_actions = torch.clamp(
                pred_actions, -self.config.action_clip, self.config.action_clip
            )
            actor_loss = -self._q_value(obs, pred_actions).mean()

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.config.grad_norm_clip:
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.grad_norm_clip
                )
            self.actor_optim.step()

            # --- Polyak target updates -----------------------------------
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.q_head, self.q_head_target)

            metrics = {
                "loss/critic": float(critic_loss.item()),
                "loss/actor": float(actor_loss.item()),
                "q/mean": float(q_pred.mean().item()),
                "q/target_mean": float(target.mean().item()),
            }
        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = float(self.config.tau)
        with torch.no_grad():
            for sp, tp in zip(source.parameters(), target.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * sp.data)
            for sb, tb in zip(source.buffers(), target.buffers()):
                tb.data.copy_(sb.data)

    # -- training loop ----------------------------------------------------
    def train(self) -> "PQLTrainer":
        start = time.time()
        target_transitions = float(self.config.target_transitions)
        while (
            self.transitions < target_transitions
            and self.iteration < int(self.config.max_iterations)
        ):
            rollout_metrics = self.collect_rollouts()
            update_metrics = self.update()
            self.iteration += 1

            if self.iteration % max(1, int(self.config.log_interval)) == 0:
                metrics = {
                    "iteration": float(self.iteration),
                    "transitions": float(self.transitions),
                    "time/elapsed": time.time() - start,
                }
                metrics.update(rollout_metrics)
                metrics.update(update_metrics)
                if self._recent_returns:
                    metrics["train/episode_return"] = float(
                        np.mean(self._recent_returns[-100:])
                    )
                if self._recent_successes:
                    metrics["train/success_rate"] = float(
                        np.mean(self._recent_successes[-100:])
                    )
                if self.logger is not None:
                    self.logger.log(metrics, step=self.iteration)

            if (
                self.config.save_interval
                and self.iteration % int(self.config.save_interval) == 0
            ):
                try:
                    self.save(
                        os.path.join(
                            self.config.output_dir, f"checkpoint_{self.iteration}.pt"
                        )
                    )
                except Exception:
                    pass

        try:
            self.save(os.path.join(self.config.output_dir, "checkpoint_final.pt"))
        except Exception:
            pass
        return self

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "iteration": self.iteration,
                "transitions": self.transitions,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "q_head": self.q_head.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "q_head_target": self.q_head_target.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )
        return path

    def load(self, path: str) -> "PQLTrainer":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.q_head.load_state_dict(ckpt["q_head"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.q_head_target.load_state_dict(ckpt["q_head_target"])
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.iteration = int(ckpt.get("iteration", 0))
        self.transitions = int(ckpt.get("transitions", 0))
        return self


def train_pql(
    config: PQLConfig,
    env_factory: Callable[[int, int], Any],
    logger: Optional[Any] = None,
) -> PQLTrainer:
    """Instantiate and run a PQL trainer to completion."""
    trainer = PQLTrainer(config, env_factory, logger=logger)
    return trainer.train()
