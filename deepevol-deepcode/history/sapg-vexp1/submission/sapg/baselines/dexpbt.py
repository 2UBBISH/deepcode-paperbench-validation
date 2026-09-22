"""DexPBT baseline: Population-Based Training for dexterous manipulation.

Reference: Petrenko et al., 2023 ("DexPBT: Scaling up Dexterous Manipulation
for Hand-Arm Systems with Population Based Training").

DexPBT maintains a population of M independent PPO agents (one per env block).
Periodically, the population is evaluated; under-performing agents *exploit*
the weights of the best agent and *explore* by mutating their hyperparameters
(e.g. learning rate, entropy coefficient, reward shaping weights).  This is the
main baseline SAPG is compared against on the hard tasks (Table 1).

The implementation here is deliberately self-contained and simulator-agnostic:
it consumes the same ``env_factory`` contract as the other trainers and reuses
the shared network builders and PPO loss primitives so that the only difference
from vanilla PPO is the population / mutation machinery.
"""

from __future__ import annotations

import copy
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


__all__ = ["DexPBTConfig", "DexPBTAgent", "DexPBTTrainer", "train_dexpbt"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DexPBTConfig:
    """Hyperparameters for the DexPBT baseline."""

    task: str = "allegrokuka_regrasping"
    num_policies: int = 6          # population size M
    num_envs: int = 24576          # total parallel envs (split across M)
    horizon: int = 16
    gamma: float = 0.99
    tau: float = 0.95
    clip_eps: float = 0.1
    critic_coef: float = 4.0
    entropy_coef: float = 0.0
    learning_rate: float = 3e-4
    kl_threshold: float = 0.016
    kl_min_lr: float = 1e-5
    kl_max_lr: float = 1e-2
    mini_epochs: int = 2
    num_mini_batches: int = 4
    grad_norm_clip: float = 1.0
    normalize_advantage: bool = True
    recurrent: bool = False
    lstm_hidden: int = 768
    obs_dim: int = 44
    action_dim: int = 23
    device: str = "cuda"

    # --- PBT specific -------------------------------------------------------
    pbt_interval: int = 50          # iterations between exploit/explore steps
    pbt_quantile: float = 0.25      # bottom fraction that exploits the best
    mutation_factor: float = 1.2    # multiplicative mutation magnitude
    mutation_prob: float = 0.5      # probability of mutating each hyperparam
    lr_bounds: Tuple[float, float] = (1e-5, 1e-2)
    entropy_bounds: Tuple[float, float] = (0.0, 0.01)

    # --- training budget ----------------------------------------------------
    target_transitions: float = 2e10
    output_dir: str = "runs/dexpbt"
    save_interval: int = 100
    log_interval: int = 1
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Single population member
# ---------------------------------------------------------------------------
class DexPBTAgent:
    """One member of the DexPBT population: actor + critic + optimizer."""

    def __init__(self, config: DexPBTConfig, index: int, device: torch.device):
        self.config = config
        self.index = index
        self.device = device

        self.actor = build_actor(
            config.task, config.obs_dim, config.action_dim, phi_dim=0
        ).to(device)
        self.critic = build_critic(config.task, config.obs_dim, phi_dim=0).to(device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        self.scheduler = KLAdaptiveLR(
            self.optimizer,
            threshold=config.kl_threshold,
            min_lr=config.kl_min_lr,
            max_lr=config.kl_max_lr,
        )

        # Mutable hyperparameters (subject to PBT mutation).
        self.learning_rate = config.learning_rate
        self.entropy_coef = config.entropy_coef
        self.clip_eps = config.clip_eps

        # Fitness tracking (mean episode return over the last evaluation window).
        self.fitness: float = -np.inf
        self.fitness_history: List[float] = []

    # -- inference ----------------------------------------------------------
    def act(self, obs: torch.Tensor, lstm_state=None):
        if getattr(self.actor, "is_recurrent", False):
            mean, new_state = self.actor(obs, lstm_state=lstm_state)
            std = self.actor.std(mean)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob, new_state
        dist = self.actor.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, None

    def value(self, obs: torch.Tensor, lstm_state=None):
        if getattr(self.critic, "is_recurrent", False):
            value, new_state = self.critic(obs, lstm_state=lstm_state)
            return value, new_state
        return self.critic(obs), None

    # -- PBT operations -----------------------------------------------------
    def clone_from(self, other: "DexPBTAgent") -> None:
        """Exploit: copy weights from a better-performing agent."""
        self.actor.load_state_dict(copy.deepcopy(other.actor.state_dict()))
        self.critic.load_state_dict(copy.deepcopy(other.critic.state_dict()))
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
        )
        self.scheduler = KLAdaptiveLR(
            self.optimizer,
            threshold=self.config.kl_threshold,
            min_lr=self.config.kl_min_lr,
            max_lr=self.config.kl_max_lr,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Explore: perturb hyperparameters multiplicatively."""
        cfg = self.config
        if rng.random() < cfg.mutation_prob:
            factor = cfg.mutation_factor if rng.random() < 0.5 else 1.0 / cfg.mutation_factor
            self.learning_rate = float(
                np.clip(self.learning_rate * factor, *cfg.lr_bounds)
            )
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        if rng.random() < cfg.mutation_prob:
            factor = cfg.mutation_factor if rng.random() < 0.5 else 1.0 / cfg.mutation_factor
            self.entropy_coef = float(
                np.clip(self.entropy_coef * factor, *cfg.entropy_bounds)
            )
        if rng.random() < cfg.mutation_prob:
            factor = cfg.mutation_factor if rng.random() < 0.5 else 1.0 / cfg.mutation_factor
            self.clip_eps = float(np.clip(self.clip_eps * factor, 0.01, 0.5))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "entropy_coef": self.entropy_coef,
            "clip_eps": self.clip_eps,
            "fitness": self.fitness,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.entropy_coef = state.get("entropy_coef", self.entropy_coef)
        self.clip_eps = state.get("clip_eps", self.clip_eps)
        self.fitness = state.get("fitness", self.fitness)


# ---------------------------------------------------------------------------
# Rollout buffer (per agent)
# ---------------------------------------------------------------------------
class _Rollout:
    def __init__(self, horizon, num_envs, obs_dim, action_dim, device, recurrent=False,
                 lstm_hidden=0):
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device
        self.recurrent = recurrent
        self.obs = torch.zeros(horizon, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.advantages = torch.zeros(horizon, num_envs, device=device)
        self.returns = torch.zeros(horizon, num_envs, device=device)
        self.lstm_states = None
        if recurrent:
            self.lstm_states = (
                torch.zeros(horizon, num_envs, lstm_hidden, device=device),
                torch.zeros(horizon, num_envs, lstm_hidden, device=device),
            )
        self.ptr = 0

    def add(self, obs, actions, log_probs, rewards, dones, values, lstm_state=None):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        if self.recurrent and lstm_state is not None:
            self.lstm_states[0][t] = lstm_state[0]
            self.lstm_states[1][t] = lstm_state[1]
        self.ptr += 1

    def compute_advantages(self, last_values, gamma, tau):
        adv, ret = compute_gae(
            self.rewards, self.values, self.dones, gamma=gamma, tau=tau,
            last_values=last_values,
        )
        self.advantages = adv
        self.returns = ret

    def flat(self) -> Dict[str, torch.Tensor]:
        out = {
            "obs": self.obs.reshape(-1, self.obs.shape[-1]),
            "actions": self.actions.reshape(-1, self.actions.shape[-1]),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }
        if self.recurrent:
            out["lstm_h"] = self.lstm_states[0].reshape(-1, self.lstm_states[0].shape[-1])
            out["lstm_c"] = self.lstm_states[1].reshape(-1, self.lstm_states[1].shape[-1])
        return out

    def size(self) -> int:
        return self.horizon * self.num_envs

    def reset(self):
        self.ptr = 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class DexPBTTrainer:
    """Population-based training driver for the DexPBT baseline."""

    def __init__(self, config: DexPBTConfig,
                 env_factory: Callable[[int, int], Any], logger=None):
        self.config = config
        self.env_factory = env_factory
        self.logger = logger
        self.device = torch.device(
            config.device if torch.cuda.is_available() or "cuda" not in config.device
            else "cpu"
        )
        self.rng = np.random.default_rng(config.seed)

        M = config.num_policies
        assert config.num_envs % M == 0, "num_envs must be divisible by num_policies"
        self.num_envs_per_policy = config.num_envs // M

        # One env per population member (each member owns its own env block).
        self.envs = [
            env_factory(self.num_envs_per_policy, config.seed + j) for j in range(M)
        ]
        self.obs_dim = getattr(self.envs[0], "obs_dim", config.obs_dim)
        self.action_dim = getattr(self.envs[0], "action_dim", config.action_dim)
        config.obs_dim = self.obs_dim
        config.action_dim = self.action_dim

        self.agents: List[DexPBTAgent] = [
            DexPBTAgent(config, j, self.device) for j in range(M)
        ]
        self.buffers = [
            _Rollout(
                config.horizon, self.num_envs_per_policy, self.obs_dim,
                self.action_dim, self.device, recurrent=config.recurrent,
                lstm_hidden=config.lstm_hidden,
            )
            for _ in range(M)
        ]

        self.obs = [self._to_tensor(e.reset()) for e in self.envs]
        self.lstm_states = [
            self._init_lstm(self.num_envs_per_policy) for _ in range(M)
        ]

        self.iteration = 0
        self.transitions = 0
        self.episode_returns = [[] for _ in range(M)]
        self.episode_successes = [[] for _ in range(M)]
        self._running_return = [np.zeros(self.num_envs_per_policy) for _ in range(M)]

    # -- helpers ------------------------------------------------------------
    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device).float()
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _init_lstm(self, n):
        if not self.config.recurrent:
            return None
        h = self.config.lstm_hidden
        return (
            torch.zeros(n, h, device=self.device),
            torch.zeros(n, h, device=self.device),
        )

    # -- rollout ------------------------------------------------------------
    def collect_rollouts(self) -> Dict[str, float]:
        cfg = self.config
        for buf in self.buffers:
            buf.reset()

        for _ in range(cfg.horizon):
            for j, agent in enumerate(self.agents):
                obs = self.obs[j]
                with torch.no_grad():
                    action, log_prob, new_state = agent.act(obs, self.lstm_states[j])
                    value, _ = agent.value(obs, self.lstm_states[j])
                act_np = action.cpu().numpy()
                obs_next, rew, done, info = self.envs[j].step(act_np)
                rew_t = self._to_tensor(rew)
                done_t = self._to_tensor(done)
                self.buffers[j].add(
                    obs, action, log_prob, rew_t, done_t, value,
                    lstm_state=self.lstm_states[j],
                )
                self._running_return[j] += rew_t.cpu().numpy()
                done_np = done_t.cpu().numpy().astype(bool)
                if done_np.any():
                    for idx in np.where(done_np)[0]:
                        self.episode_returns[j].append(float(self._running_return[j][idx]))
                        self._running_return[j][idx] = 0.0
                    if isinstance(info, dict) and "episode_success" in info:
                        es = np.asarray(info["episode_success"])
                        for idx in np.where(done_np)[0]:
                            if idx < len(es):
                                self.episode_successes[j].append(float(es[idx]))
                self.obs[j] = self._to_tensor(obs_next)
                if self.config.recurrent and new_state is not None:
                    h, c = new_state
                    mask = (1.0 - done_t).unsqueeze(-1)
                    self.lstm_states[j] = (h * mask, c * mask)
                self.transitions += self.num_envs_per_policy

        # bootstrap values
        for j, agent in enumerate(self.agents):
            with torch.no_grad():
                last_value, _ = agent.value(self.obs[j], self.lstm_states[j])
            self.buffers[j].compute_advantages(last_value, cfg.gamma, cfg.tau)

        stats = {}
        for j in range(len(self.agents)):
            if self.episode_returns[j]:
                stats[f"agent{j}/episode_return"] = float(np.mean(self.episode_returns[j][-50:]))
                self.agents[j].fitness = float(np.mean(self.episode_returns[j][-50:]))
                self.agents[j].fitness_history.append(self.agents[j].fitness)
        return stats

    # -- update -------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        cfg = self.config
        info: Dict[str, float] = {}
        for j, agent in enumerate(self.agents):
            batch = self.buffers[j].flat()
            n = batch["obs"].shape[0]
            mini_batch_size = max(1, (self.num_envs_per_policy * 4) // cfg.num_mini_batches)
            idx = torch.randperm(n, device=self.device)
            kl_accum, count = 0.0, 0
            for _ in range(cfg.mini_epochs):
                for start in range(0, n, mini_batch_size):
                    mb = idx[start:start + mini_batch_size]
                    obs = batch["obs"][mb]
                    actions = batch["actions"][mb]
                    old_log_probs = batch["log_probs"][mb]
                    adv = batch["advantages"][mb]
                    ret = batch["returns"][mb]
                    if cfg.normalize_advantage and adv.numel() > 1:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    lstm_state = None
                    if cfg.recurrent:
                        lstm_state = (batch["lstm_h"][mb], batch["lstm_c"][mb])

                    if getattr(agent.actor, "is_recurrent", False):
                        mean, _ = agent.actor(obs, lstm_state=lstm_state)
                        std = agent.actor.std(mean)
                        dist = torch.distributions.Normal(mean, std)
                        log_probs = dist.log_prob(actions).sum(-1)
                        entropy = dist.entropy().sum(-1).mean()
                    else:
                        dist = agent.actor.distribution(obs)
                        log_probs = dist.log_prob(actions).sum(-1)
                        entropy = dist.entropy().sum(-1).mean()

                    if getattr(agent.critic, "is_recurrent", False):
                        values, _ = agent.critic(obs, lstm_state=lstm_state)
                    else:
                        values = agent.critic(obs)

                    policy_loss = ppo_surrogate_loss(
                        log_probs, old_log_probs, adv, clip_eps=agent.clip_eps
                    )
                    v_loss = value_loss(values, ret)
                    loss = policy_loss + cfg.critic_coef * v_loss - agent.entropy_coef * entropy

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(agent.actor.parameters()) + list(agent.critic.parameters()),
                        cfg.grad_norm_clip,
                    )
                    agent.optimizer.step()

                    with torch.no_grad():
                        log_ratio = log_probs - old_log_probs
                        kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                    kl_accum += kl
                    count += 1

            mean_kl = kl_accum / max(1, count)
            agent.scheduler.update(mean_kl)
            info[f"agent{j}/kl"] = mean_kl
            info[f"agent{j}/lr"] = agent.scheduler.current_lr()
            info[f"agent{j}/policy_loss"] = float(policy_loss.item())
            info[f"agent{j}/value_loss"] = float(v_loss.item())
        return info

    # -- PBT exploit / explore ---------------------------------------------
    def pbt_step(self) -> Dict[str, float]:
        cfg = self.config
        fitness = np.array([a.fitness for a in self.agents], dtype=np.float64)
        finite = np.isfinite(fitness)
        if not finite.any():
            return {}
        best_idx = int(np.nanargmax(np.where(finite, fitness, -np.inf)))
        order = np.argsort(-np.where(finite, fitness, -np.inf))
        n_exploit = max(1, int(round(cfg.pbt_quantile * len(self.agents))))
        worst = order[-n_exploit:]

        info = {"pbt/best_fitness": float(fitness[best_idx])}
        for idx in worst:
            if idx == best_idx:
                continue
            self.agents[idx].clone_from(self.agents[best_idx])
            self.agents[idx].mutate(self.rng)
            info[f"pbt/agent{idx}_lr"] = self.agents[idx].learning_rate
            info[f"pbt/agent{idx}_entropy"] = self.agents[idx].entropy_coef
        return info

    # -- main loop ----------------------------------------------------------
    def train(self) -> "DexPBTTrainer":
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        start = time.time()
        while self.transitions < cfg.target_transitions:
            self.collect_rollouts()
            info = self.update()
            self.iteration += 1

            if self.iteration % cfg.pbt_interval == 0:
                info.update(self.pbt_step())

            info["iteration"] = self.iteration
            info["transitions"] = self.transitions
            info["time"] = time.time() - start
            if self.logger is not None:
                self.logger.log(info, step=self.transitions)
            if self.iteration % cfg.save_interval == 0:
                self.save(os.path.join(cfg.output_dir, f"checkpoint_{self.iteration}.pt"))
        return self

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "agents": [a.state_dict() for a in self.agents],
                "iteration": self.iteration,
                "transitions": self.transitions,
            },
            path,
        )
        return path

    def load(self, path: str) -> "DexPBTTrainer":
        payload = torch.load(path, map_location=self.device, weights_only=False)
        for agent, state in zip(self.agents, payload["agents"]):
            agent.load_state_dict(state)
        self.iteration = payload.get("iteration", 0)
        self.transitions = payload.get("transitions", 0)
        return self


def train_dexpbt(config: DexPBTConfig,
                 env_factory: Callable[[int, int], Any],
                 logger=None) -> DexPBTTrainer:
    """Instantiate and run a DexPBT trainer to completion."""
    trainer = DexPBTTrainer(config, env_factory, logger=logger)
    return trainer.train()
