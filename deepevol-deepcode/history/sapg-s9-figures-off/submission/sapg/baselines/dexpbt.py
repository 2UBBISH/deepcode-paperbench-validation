"""DexPBT baseline: Population-Based Training with PPO (Petrenko et al., 2023).

DexPBT ("Dexterous Hand Population-Based Training") is the strongest prior
baseline for the hard AllegroKuka tasks in the SAPG paper (Sec. 5, Table 1).
It maintains a *population* of PPO agents that are trained in parallel on
disjoint slices of the parallel environments.  Periodically the population is
evaluated and the worst-performing members are replaced by copies of the best
members, with their hyperparameters (learning rate, entropy coefficient,
discount factor, ...) perturbed.  This is the classic Population-Based
Training (PBT) scheme of Jaderberg et al. (2017) applied to PPO.

Reference:
    Petrenko et al. (2023). "DexPBT: Scaling up Dexterous Manipulation for
    Hand-Arm Systems with Population Based Training."

This module mirrors the ``SAPGTrainer`` / ``PPOTrainer`` / ``PQLTrainer``
interface (``train`` / ``state_dict`` / ``load_state_dict``) so that
``main.py`` can dispatch to it uniformly.

Key differences from vanilla PPO:
    * ``population_size`` independent PPO agents, each owning a contiguous
      slice of the ``num_envs`` parallel environments.
    * Each agent has its own optimizer, learning rate and entropy coefficient.
    * Every ``pbt_interval`` iterations the population is ranked by a fitness
      score (mean episode reward / successes over the last interval) and the
      bottom ``exploit_fraction`` of members are replaced by copies of the top
      members, with hyperparameters multiplied by a random factor in
      ``[1 - perturb_factor, 1 + perturb_factor]``.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..sapg.config import SAPGConfig
from ..sapg.losses import critic_loss, on_policy_loss
from ..sapg.networks import SAPGNetworks, build_networks
from ..sapg.returns import compute_advantages_and_targets, flatten_time_major
from ..sapg.rollout import RolloutBuffer, _unpack_step


__all__ = [
    "DexPBTTrainer",
    "DexPBTUpdateStats",
    "PopulationMember",
    "build_dexpbt_trainer",
]


# ---------------------------------------------------------------------------
# Adaptive learning-rate scheduler (shared convention with PPO/SAPG trainers)
# ---------------------------------------------------------------------------
class AdaptiveLR:
    """KL-based adaptive learning-rate scheduler.

    Mirrors the scheduler used by the SAPG / PPO trainers so that all methods
    share the same optimisation convention (KL threshold 0.016).
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
        self.lr = float(base_lr)

    def step(self, approx_kl: float) -> float:
        """Adjust the learning rate based on the measured approximate KL."""
        if approx_kl > 2.0 * self.kl_threshold:
            self.lr = max(self.lr / self.factor, self.min_lr)
        elif approx_kl < 0.5 * self.kl_threshold:
            self.lr = min(self.lr * self.factor, self.max_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        return self.lr


# ---------------------------------------------------------------------------
# Population member
# ---------------------------------------------------------------------------
class PopulationMember:
    """A single PPO agent inside the DexPBT population.

    Each member owns:
        * its own shared actor/critic networks (no latent conditioning),
        * its own Adam optimizer and adaptive LR scheduler,
        * its own hyperparameters (learning rate, entropy coefficient, gamma),
        * a contiguous slice of the parallel environments.
    """

    def __init__(
        self,
        member_id: int,
        config: SAPGConfig,
        obs_dim: int,
        action_dim: int,
        env_slice: slice,
        device: torch.device,
        learning_rate: Optional[float] = None,
        entropy_coef: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> None:
        self.member_id = int(member_id)
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.env_slice = env_slice
        self.device = device

        # --- hyperparameters (mutable by PBT) -----------------------------
        self.learning_rate = float(
            learning_rate if learning_rate is not None else config.learning_rate
        )
        self.entropy_coef = float(
            entropy_coef if entropy_coef is not None else getattr(config, "entropy_coef", 0.0)
        )
        self.gamma = float(gamma if gamma is not None else config.gamma)

        # --- networks -----------------------------------------------------
        self.networks = build_networks(config, obs_dim, action_dim).to(device)
        # DexPBT uses a single policy per member -> no latent conditioning.
        self.networks.eval()

        # --- optimiser ----------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.networks.parameters(), lr=self.learning_rate
        )
        self.lr_scheduler = AdaptiveLR(
            self.optimizer,
            base_lr=self.learning_rate,
            kl_threshold=getattr(config, "kl_threshold", 0.016),
        )

        # --- rollout bookkeeping -----------------------------------------
        self.num_envs = env_slice.stop - env_slice.start
        self.horizon = int(config.horizon)
        self.buffer = RolloutBuffer(
            horizon=self.horizon,
            num_envs=self.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        self.lstm_state = None

        # --- fitness tracking --------------------------------------------
        self.episode_rewards: List[float] = []
        self.episode_successes: List[float] = []
        self.fitness: float = float("-inf")
        self.total_transitions: int = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_rollout(self, env) -> Dict[str, float]:
        """Collect ``horizon`` steps from this member's env slice."""
        self.buffer.reset()
        self.networks.eval()

        obs = env.obs[self.env_slice] if hasattr(env, "obs") else None
        if obs is None:
            # Fall back to a full reset if the env does not expose observations.
            obs = env.reset()[self.env_slice]
        obs = obs.to(self.device)

        if getattr(self.networks.actor, "use_lstm", False):
            self.lstm_state = self.networks.actor.init_lstm_state(
                self.num_envs, self.device
            )

        ep_rewards = torch.zeros(self.num_envs, device=self.device)
        ep_successes = torch.zeros(self.num_envs, device=self.device)
        step_rewards: List[float] = []
        step_successes: List[float] = []

        for _ in range(self.horizon):
            action, log_prob, self.lstm_state = self.networks.actor_act(
                obs, latent=None, lstm_state=self.lstm_state
            )
            value, _ = self.networks.critic_value(obs, latent=None)

            full_action = torch.zeros(
                env.num_envs, self.action_dim, device=self.device
            )
            full_action[self.env_slice] = action
            next_obs, reward, done, info = _unpack_step(env.step(full_action))
            next_obs = next_obs.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)

            block_reward = reward[self.env_slice]
            block_done = done[self.env_slice]

            self.buffer.add(
                obs=obs,
                actions=action,
                log_probs=log_prob,
                rewards=block_reward,
                dones=block_done,
                values=value,
                lstm_state=self.lstm_state,
            )

            ep_rewards += block_reward
            if isinstance(info, dict) and "successes" in info:
                succ = info["successes"]
                if torch.is_tensor(succ):
                    succ = succ.to(self.device)[self.env_slice]
                else:
                    succ = torch.as_tensor(succ, device=self.device)[self.env_slice]
                ep_successes += succ
                step_successes.append(float(succ.mean().item()))

            step_rewards.append(float(block_reward.mean().item()))

            # Record finished episodes.
            if block_done.any():
                finished = block_done.bool()
                self.episode_rewards.extend(ep_rewards[finished].tolist())
                self.episode_successes.extend(ep_successes[finished].tolist())
                ep_rewards[finished] = 0.0
                ep_successes[finished] = 0.0

            obs = next_obs

        self.total_transitions += self.horizon * self.num_envs

        # Bootstrap value for GAE.
        with torch.no_grad():
            next_value, _ = self.networks.critic_value(obs, latent=None)

        self.buffer.compute_returns(
            next_value=next_value,
            gamma=self.gamma,
            gae_lambda=getattr(self.config, "gae_lambda", 0.95),
            on_policy_target_steps=getattr(self.config, "on_policy_target_steps", 3),
        )

        return {
            "mean_reward": float(sum(step_rewards) / max(len(step_rewards), 1)),
            "mean_success": float(sum(step_successes) / max(len(step_successes), 1)),
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """Run PPO mini-epochs over the collected buffer."""
        cfg = self.config
        clip_eps = getattr(cfg, "clip_eps", 0.1)
        mini_epochs = int(getattr(cfg, "num_mini_epochs", 2))
        minibatch_size = int(getattr(cfg, "minibatch_size", 4096))
        grad_clip = float(getattr(cfg, "grad_norm_clip", 1.0))

        obs = flatten_time_major(self.buffer.obs)
        actions = flatten_time_major(self.buffer.actions)
        old_log_probs = flatten_time_major(self.buffer.log_probs)
        advantages = flatten_time_major(self.buffer.advantages)
        value_targets = flatten_time_major(self.buffer.value_targets)
        old_values = flatten_time_major(self.buffer.values)

        num_samples = obs.shape[0]
        minibatch_size = min(minibatch_size, num_samples)

        stats: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        num_updates = 0

        self.networks.train()
        for _ in range(mini_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                idx = perm[start : start + minibatch_size]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_targets = value_targets[idx]
                mb_old_values = old_values[idx]

                new_log_probs = self.networks.actor_log_prob(
                    mb_obs, mb_actions, latent=None
                )
                entropy = self.networks.actor_entropy(mb_obs, latent=None)
                values, _ = self.networks.critic_value(mb_obs, latent=None)

                policy_out = on_policy_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    clip_eps=clip_eps,
                )
                critic_out = critic_loss(
                    values=values,
                    on_policy_targets=mb_targets,
                    old_values=mb_old_values,
                    clip_eps=clip_eps,
                    value_coef=float(getattr(cfg, "critic_coef", 1.0)),
                )

                loss = policy_out.loss + critic_out.loss
                if self.entropy_coef > 0.0:
                    loss = loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.networks.parameters(), grad_clip
                )
                self.optimizer.step()

                stats["policy_loss"] += float(policy_out.policy_loss.item())
                stats["value_loss"] += float(critic_out.value_loss.item())
                stats["entropy"] += float(entropy.mean().item())
                stats["approx_kl"] += float(policy_out.approx_kl.item())
                stats["clip_fraction"] += float(policy_out.clip_fraction.item())
                stats["grad_norm"] += float(grad_norm.item())
                num_updates += 1

        self.networks.eval()
        denom = max(num_updates, 1)
        for key in stats:
            stats[key] /= denom

        stats["learning_rate"] = self.lr_scheduler.step(stats["approx_kl"])
        return stats

    # ------------------------------------------------------------------
    # PBT operations
    # ------------------------------------------------------------------
    def compute_fitness(self) -> float:
        """Fitness = mean episode reward over the last evaluation window."""
        if self.episode_rewards:
            window = self.episode_rewards[-100:]
            self.fitness = float(sum(window) / len(window))
        elif self.episode_successes:
            window = self.episode_successes[-100:]
            self.fitness = float(sum(window) / len(window))
        else:
            self.fitness = float("-inf")
        return self.fitness

    def clone_from(self, other: "PopulationMember") -> None:
        """Copy another member's weights and hyperparameters into this one."""
        self.networks.load_state_dict(copy.deepcopy(other.networks.state_dict()))
        self.learning_rate = other.learning_rate
        self.entropy_coef = other.entropy_coef
        self.gamma = other.gamma
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self.lr_scheduler.base_lr = self.learning_rate
        self.lr_scheduler.lr = self.learning_rate
        # Reset fitness history so the clone is evaluated on its own merits.
        self.episode_rewards = []
        self.episode_successes = []
        self.fitness = float("-inf")

    def perturb(self, factor: float, rng: torch.Generator) -> None:
        """Perturb hyperparameters by a random factor in [1-f, 1+f]."""
        def _jitter(value: float, lo: float, hi: float) -> float:
            u = float(torch.rand(1, generator=rng).item())
            mult = 1.0 + factor * (2.0 * u - 1.0)
            return float(min(max(value * mult, lo), hi))

        self.learning_rate = _jitter(self.learning_rate, 1e-5, 1e-2)
        self.entropy_coef = _jitter(self.entropy_coef, 0.0, 0.05)
        self.gamma = _jitter(self.gamma, 0.9, 0.999)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self.lr_scheduler.base_lr = self.learning_rate
        self.lr_scheduler.lr = self.learning_rate

    def state_dict(self) -> Dict[str, Any]:
        return {
            "member_id": self.member_id,
            "networks": self.networks.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "entropy_coef": self.entropy_coef,
            "gamma": self.gamma,
            "fitness": self.fitness,
            "total_transitions": self.total_transitions,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.networks.load_state_dict(state["networks"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.entropy_coef = state.get("entropy_coef", self.entropy_coef)
        self.gamma = state.get("gamma", self.gamma)
        self.fitness = state.get("fitness", float("-inf"))
        self.total_transitions = state.get("total_transitions", 0)


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------
@dataclass
class DexPBTUpdateStats:
    """Per-iteration statistics for a DexPBT run."""

    iteration: int
    total_transitions: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    learning_rate: float
    grad_norm: float
    mean_reward: float
    mean_episode_reward: float
    successes: float
    best_fitness: float
    mean_fitness: float
    num_exploits: int
    wall_time: float
    per_member_fitness: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        out = {
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
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "num_exploits": self.num_exploits,
            "wall_time": self.wall_time,
        }
        for i, f in enumerate(self.per_member_fitness):
            out[f"fitness_member_{i}"] = f
        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class DexPBTTrainer:
    """Population-Based Training with PPO (DexPBT baseline).

    The population is trained in parallel on disjoint slices of the parallel
    environments.  Every ``pbt_interval`` iterations the population is ranked
    by fitness and the bottom fraction is replaced by perturbed copies of the
    top performers.
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
        self.device = torch.device(
            device if device is not None else getattr(config, "device", "cpu")
        )

        self.population_size = int(getattr(config, "population_size", 6))
        self.pbt_interval = int(getattr(config, "pbt_interval", 20))
        self.perturb_factor = float(getattr(config, "perturb_factor", 0.2))
        self.exploit_fraction = float(getattr(config, "exploit_fraction", 0.25))

        num_envs = int(getattr(env, "num_envs", getattr(config, "num_envs", 24576)))
        self.num_envs = num_envs

        # Split the parallel envs evenly across the population.
        block = num_envs // self.population_size
        self.members: List[PopulationMember] = []
        for i in range(self.population_size):
            start = i * block
            stop = num_envs if i == self.population_size - 1 else (i + 1) * block
            self.members.append(
                PopulationMember(
                    member_id=i,
                    config=config,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    env_slice=slice(start, stop),
                    device=self.device,
                )
            )

        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(getattr(config, "seed", 0)) + 1234)

        self.iteration = 0
        self.total_transitions = 0
        self.history: List[DexPBTUpdateStats] = []

    # ------------------------------------------------------------------
    def train_iteration(self) -> DexPBTUpdateStats:
        t0 = time.time()
        self.iteration += 1

        # 1. Collect rollouts for every member.
        rollout_stats = [m.collect_rollout(self.env) for m in self.members]

        # 2. PPO update for every member.
        update_stats = [m.update() for m in self.members]

        # 3. Periodic PBT exploit/explore step.
        num_exploits = 0
        if self.iteration % self.pbt_interval == 0:
            num_exploits = self._pbt_step()

        # 4. Aggregate statistics.
        def _mean(key: str) -> float:
            vals = [s[key] for s in update_stats if key in s]
            return float(sum(vals) / len(vals)) if vals else 0.0

        all_ep_rewards: List[float] = []
        all_ep_successes: List[float] = []
        for m in self.members:
            all_ep_rewards.extend(m.episode_rewards[-100:])
            all_ep_successes.extend(m.episode_successes[-100:])

        mean_ep_reward = (
            float(sum(all_ep_rewards) / len(all_ep_rewards)) if all_ep_rewards else 0.0
        )
        mean_success = (
            float(sum(all_ep_successes) / len(all_ep_successes))
            if all_ep_successes
            else 0.0
        )

        fitnesses = [m.compute_fitness() for m in self.members]
        finite = [f for f in fitnesses if f != float("-inf")]
        best_fitness = max(finite) if finite else 0.0
        mean_fitness = float(sum(finite) / len(finite)) if finite else 0.0

        self.total_transitions += sum(
            self.config.horizon * m.num_envs for m in self.members
        )

        stats = DexPBTUpdateStats(
            iteration=self.iteration,
            total_transitions=self.total_transitions,
            policy_loss=_mean("policy_loss"),
            value_loss=_mean("value_loss"),
            entropy=_mean("entropy"),
            approx_kl=_mean("approx_kl"),
            clip_fraction=_mean("clip_fraction"),
            learning_rate=_mean("learning_rate"),
            grad_norm=_mean("grad_norm"),
            mean_reward=float(
                sum(s["mean_reward"] for s in rollout_stats) / len(rollout_stats)
            ),
            mean_episode_reward=mean_ep_reward,
            successes=mean_success,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            num_exploits=num_exploits,
            wall_time=time.time() - t0,
            per_member_fitness=fitnesses,
        )
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------
    def _pbt_step(self) -> int:
        """Rank the population and exploit/explore the bottom members."""
        fitnesses = [m.compute_fitness() for m in self.members]
        order = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)

        num_exploit = max(1, int(round(self.exploit_fraction * self.population_size)))
        top = order[: max(1, self.population_size - num_exploit)]
        bottom = order[-num_exploit:]

        for dst_idx in bottom:
            src_idx = int(top[torch.randint(len(top), (1,), generator=self.rng).item()])
            if src_idx == dst_idx:
                continue
            self.members[dst_idx].clone_from(self.members[src_idx])
            self.members[dst_idx].perturb(self.perturb_factor, self.rng)
        return len(bottom)

    # ------------------------------------------------------------------
    def train(self, num_iterations: int) -> List[DexPBTUpdateStats]:
        for _ in range(int(num_iterations)):
            self.train_iteration()
        return self.history

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_transitions": self.total_transitions,
            "members": [m.state_dict() for m in self.members],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.iteration = state.get("iteration", 0)
        self.total_transitions = state.get("total_transitions", 0)
        for m, ms in zip(self.members, state.get("members", [])):
            m.load_state_dict(ms)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_dexpbt_trainer(
    config: SAPGConfig,
    env: Any,
    obs_dim: int,
    action_dim: int,
    device: Optional[torch.device] = None,
) -> DexPBTTrainer:
    """Build a :class:`DexPBTTrainer` from a config and environment."""
    return DexPBTTrainer(
        config=config,
        env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
