"""DexPBT baseline (Petrenko et al., 2023) -- population-based PPO.

"N Environments are divided into M groups, each containing N/M environments.
M separate policies are trained using PPO in each group of environments with
different hyperparameters. At regular intervals, the worst-performing policies
are replaced with the weights of best-performing policies and their
hyperparameters are mutated randomly." (Sec. 5.2 of the paper)

Contrary to SAPG, the population members do not share a backbone and never
exchange data: the only information flowing between members is the
weight-copy / hyperparameter mutation step.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .actor_critic import ActorCritic
from .base import AdaptiveLR, TrainerBase
from .losses import OnPolicyLoss
from .runner import MPolicyRunner
from .storage import RolloutStorage


class PolicyEnsemble(nn.Module):
    """Presents ``M`` independent actor-critics behind the single-policy API.

    The rollout runner selects one member per environment block, so the same
    collection code can be reused.
    """

    def __init__(self, models: List[ActorCritic]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def member(self, policy_id: int) -> ActorCritic:
        return self.models[int(policy_id)]

    def _selected(self, policy_ids) -> ActorCritic:
        ids = torch.unique(policy_ids)
        if ids.numel() != 1:
            raise ValueError("A rollout block must be controlled by exactly one population member")
        return self.member(int(ids.item()))

    # --- API used by MPolicyRunner -------------------------------------
    def act(self, obs, policy_ids=None, recurrent_states=None, dones=None, deterministic=False):
        return self._selected(policy_ids).act(
            obs,
            policy_ids=None,
            recurrent_states=recurrent_states,
            dones=dones,
            deterministic=deterministic,
        )

    def get_value(self, obs, policy_ids=None, recurrent_states=None, dones=None):
        return self._selected(policy_ids).get_value(
            obs, policy_ids=None, recurrent_states=recurrent_states, dones=dones
        )


class DexPBTTrainer(TrainerBase):
    name = "pbt"

    def __init__(self, cfg, env, device: str = "cpu", logdir: Optional[str] = None) -> None:
        super().__init__(cfg, env, device=device, logdir=logdir)
        pbt_cfg = self.a_get("pbt", {}) or {}
        self.interval = int(pbt_cfg.get("interval", 25))
        self.keep_best_fraction = float(pbt_cfg.get("keep_best_fraction", 0.5))
        self.hyperparameter_space: Dict[str, list] = dict(pbt_cfg.get("hyperparameter_space", {}))
        if not self.hyperparameter_space:
            self.hyperparameter_space = {
                "learning_rate": [self.learning_rate],
                "entropy_coef": [0.0],
                "clip_epsilon": [float(self.a_get("clip_epsilon", 0.1))],
            }

        # M independent PPO policies (no shared backbone, no aggregation)
        self.population: List[ActorCritic] = [
            ActorCritic(
                self.model_cfg,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                num_policies=1,
                normalizer=self.normalizer,
            ).to(self.device)
            for _ in range(self.num_policies)
        ]
        self.model = PolicyEnsemble(self.population)
        self.optimizers = [
            torch.optim.Adam(member.trainable_parameter_groups(self.learning_rate))
            for member in self.population
        ]
        self.hyperparameters: List[Dict[str, float]] = [
            self._sample_hyperparameters() for _ in range(self.num_policies)
        ]
        self.lr_schedulers = [
            AdaptiveLR(opt, self.kl_threshold, self.min_learning_rate) for opt in self.optimizers
        ]
        self.storage = RolloutStorage(
            num_policies=self.num_policies,
            num_envs_per_policy=self.num_envs_per_policy,
            horizon=self.horizon,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            device=self.device,
            recurrent=self.recurrent,
        )
        self.runner = MPolicyRunner(
            env, self.model, self.num_policies, self.device, recurrent=self.recurrent
        )
        self.on_policy_loss = OnPolicyLoss(cfg.get("algo", {}))
        self.fitness = [float("-inf")] * self.num_policies

    # ------------------------------------------------------------------ #
    def _sample_hyperparameters(self) -> Dict[str, float]:
        return {key: float(values[torch.randint(len(values), (1,)).item()]) for key, values in self.hyperparameter_space.items()}

    def _mutate(self, hyperparameters: Dict[str, float]) -> Dict[str, float]:
        mutated = {}
        for key, value in hyperparameters.items():
            choices = self.hyperparameter_space.get(key, [value])
            if len(choices) > 1 and torch.rand(()) < 0.5:
                mutated[key] = float(choices[torch.randint(len(choices), (1,)).item()])
            else:
                mutated[key] = value
        return mutated

    # ------------------------------------------------------------------ #
    def update(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        kl_values = []
        for policy_id, member in enumerate(self.population):
            optimizer = self.optimizers[policy_id]
            hyperparameters = self.hyperparameters[policy_id]
            clip_epsilon = float(hyperparameters.get("clip_epsilon", self.a_get("clip_epsilon", 0.1)))
            entropy_coef = float(hyperparameters.get("entropy_coef", 0.0))
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(hyperparameters.get("learning_rate", self.learning_rate))
            for _epoch in range(self.mini_epochs):
                for chunk in self.storage.on_policy_chunks(
                    policy_id, self.num_minibatches, self.minibatch_size, self.generator
                ):
                    chunk = dict(chunk)
                    chunk["policy_ids"] = torch.zeros_like(chunk["policy_ids"])
                    losses = self.on_policy_loss(member, chunk, entropy_coef=entropy_coef)
                    optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(member.parameters(), self.max_grad_norm)
                    optimizer.step()
                    metrics[f"policy{policy_id}/policy_loss"] = float(losses["policy_loss"])
                    metrics[f"policy{policy_id}/value_loss"] = float(losses["value_loss"])
                    kl_values.append(float(losses["approx_kl"]))
        if kl_values:
            metrics["approx_kl"] = float(sum(kl_values) / len(kl_values))
        return metrics

    # ------------------------------------------------------------------ #
    def train_iteration(self) -> Dict[str, float]:
        metrics = super().train_iteration()
        if self.interval and self.iteration % self.interval == 0:
            metrics.update(self.exploit_and_explore())
        return metrics

    @torch.no_grad()
    def exploit_and_explore(self) -> Dict[str, float]:
        """Replace the worst members with copies of the best and mutate them."""
        eval_cfg = self.cfg.get("eval", {}) or {}
        num_episodes = int((self.a_get("pbt", {}) or {}).get("eval_episodes", 8))
        max_steps = int(eval_cfg.get("max_steps", 512))
        for policy_id in range(self.num_policies):
            stats = self.evaluate_policy(
                policy_id, num_episodes=num_episodes, max_steps=max_steps, deterministic=True
            )
            self.fitness[policy_id] = float(stats.get("successes", 0.0)) + float(
                stats.get("episode_reward", 0.0)
            )
        order = sorted(range(self.num_policies), key=lambda i: self.fitness[i], reverse=True)
        num_keep = max(1, int(round(self.keep_best_fraction * self.num_policies)))
        best = order[:num_keep]
        worst = order[num_keep:]
        for rank, target in enumerate(worst):
            source = best[rank % len(best)]
            self.population[target].load_state_dict(copy.deepcopy(self.population[source].state_dict()))
            self.hyperparameters[target] = self._mutate(dict(self.hyperparameters[source]))
            self.fitness[target] = float("-inf")
        metrics = {
            "pbt/best_fitness": float(self.fitness[order[0]]) if self.fitness[order[0]] != float("-inf") else 0.0,
            "pbt/replaced": float(len(worst)),
        }
        for key, value in self.hyperparameters[best[0]].items():
            metrics[f"pbt/best_{key}"] = float(value)
        return metrics

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate_policy(self, policy_id: int = 0, num_episodes: int = 32, max_steps: int = 512, deterministic: bool = True):
        from ..envs.base import EpisodeTracker

        member = self.population[int(policy_id)]
        obs = self.env.reset()
        tracker = EpisodeTracker(self.env.num_envs)
        states = None
        prev_dones = None
        for _ in range(max_steps):
            actions, _, _, states = member.act(
                obs, policy_ids=None, recurrent_states=states, dones=prev_dones, deterministic=deterministic
            )
            step = self.env.step(actions)
            obs = step.obs
            prev_dones = step.dones if self.recurrent else None
            tracker.step(
                step.rewards.detach().cpu().numpy(),
                step.infos.get("successes", step.rewards * 0).detach().cpu().numpy(),
                (step.dones + step.timeouts).clamp(max=1.0).detach().cpu().numpy().astype(bool),
            )
            if len(tracker.finished_returns) >= num_episodes:
                break
        stats = tracker.summary()
        self.runner.reset()
        return stats
