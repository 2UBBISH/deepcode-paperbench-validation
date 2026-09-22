"""SAPG training loop (Algorithm 1).

This module orchestrates the Split-and-Aggregate Policy Gradient algorithm:

    1. Split N parallel environments into M blocks, one per policy.
    2. Each policy j (shared actor backbone B_theta / critic backbone C_psi
       conditioned on per-policy latent phi_j) rolls out its block, producing
       buffers D_1 .. D_M.
    3. Sample |D_1| transitions from union_{j=2}^{M} D_j -> D_1' (off-policy).
    4. Build the combined objective:
           L = OffPolicyLoss(D_1') + OnPolicyLoss(D_1)
               + sum_{j=2}^{M} OnPolicyLoss(D_j) [+ entropy terms]
    5. Update theta, psi via gradient descent; update phi_j with its own
       objective only.

The shared backbones are updated by gradients from *all* objectives, while each
per-policy latent phi_j is updated only by policy j's objective (enforced via
``SAPGNetworks.zero_other_latent_grads``).

Optimizer: Adam with per-task lr, adaptive LR via KL threshold (0.016),
grad-norm clip 1.0. Mini-epochs: 2 (AllegroKuka), 5 (hands).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .aggregation import (
    PolicyAggregationPlan,
    _flatten_buffer,
    build_aggregation_plans,
    gather_off_policy_batch,
)
from .config import SAPGConfig
from .entropy import EntropyScheduler, build_entropy_coefs
from .losses import (
    CriticLossOutput,
    PolicyLossOutput,
    combined_policy_loss,
    critic_loss,
    off_policy_loss,
    on_policy_loss,
)
from .networks import SAPGNetworks, build_networks
from .rollout import RolloutBuffer, RolloutCollector, build_rollout_collector


# ---------------------------------------------------------------------------
# Statistics containers
# ---------------------------------------------------------------------------
@dataclass
class UpdateStats:
    """Aggregated statistics from a single outer iteration."""

    iteration: int = 0
    total_transitions: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    learning_rate: float = 0.0
    mean_reward: float = 0.0
    episode_reward: float = 0.0
    successes: float = 0.0
    success_rate: float = 0.0
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "iteration": int(self.iteration),
            "total_transitions": int(self.total_transitions),
            "policy_loss": float(self.policy_loss),
            "value_loss": float(self.value_loss),
            "entropy": float(self.entropy),
            "approx_kl": float(self.approx_kl),
            "clip_fraction": float(self.clip_fraction),
            "learning_rate": float(self.learning_rate),
            "mean_reward": float(self.mean_reward),
            "episode_reward": float(self.episode_reward),
            "successes": float(self.successes),
            "success_rate": float(self.success_rate),
            "wall_time": float(self.wall_time),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SAPGTrainer:
    """Trainer implementing Algorithm 1 (split + aggregate policy gradients)."""

    def __init__(
        self,
        config: SAPGConfig,
        env,
        obs_dim: int,
        action_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.env = env
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        if device is None:
            device = torch.device(getattr(config, "device", "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        self.device = device

        self.networks: SAPGNetworks = build_networks(config, self.obs_dim, self.action_dim)
        self.networks.to(self.device)

        self.collector: RolloutCollector = build_rollout_collector(
            env, self.networks, config, self.obs_dim, self.action_dim, device=self.device
        )

        block_size = config.horizon * config.envs_per_block
        self.plans: List[PolicyAggregationPlan] = build_aggregation_plans(
            num_blocks=config.num_blocks,
            aggregation=config.aggregation,
            entropy_coef=config.entropy_coef,
            subsample_off_policy=config.subsample_off_policy,
            block_size=block_size,
        )

        self.entropy: EntropyScheduler = build_entropy_coefs(config, device=self.device)

        params = list(self.networks.parameters())
        params.extend(self.entropy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

        self._iteration = 0
        self._lr = float(config.learning_rate)

    # -- state -------------------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        return {
            "networks": self.networks.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "entropy": self.entropy.state_dict(),
            "iteration": self._iteration,
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        if "networks" in state_dict:
            self.networks.load_state_dict(state_dict["networks"])  # type: ignore[arg-type]
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])  # type: ignore[arg-type]
        if "entropy" in state_dict:
            self.entropy.load_state_dict(state_dict["entropy"])  # type: ignore[arg-type]
        self._iteration = int(state_dict.get("iteration", 0))  # type: ignore[arg-type]

    # -- one iteration -----------------------------------------------------
    def train_iteration(self) -> UpdateStats:
        cfg = self.config
        self.collector.reset()
        buffers: List[RolloutBuffer] = self.collector.collect()

        total_transitions = sum(b.size() for b in buffers)
        mean_reward = float(
            torch.stack([b.rewards.mean() for b in buffers]).mean().item()
        )

        stats = UpdateStats(
            iteration=self._iteration,
            total_transitions=total_transitions,
            mean_reward=mean_reward,
            episode_reward=mean_reward,
            learning_rate=self._lr,
        )

        num_mini_epochs = max(1, int(cfg.num_mini_epochs))
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        kls: List[float] = []
        clip_fracs: List[float] = []

        for _epoch in range(num_mini_epochs):
            for plan in self.plans:
                loss_out = self._policy_update(plan, buffers)
                if loss_out is not None:
                    policy_losses.append(float(loss_out.loss.detach().item()))
                    entropies.append(float(loss_out.entropy.detach().item()))
                    if loss_out.approx_kl is not None:
                        kls.append(float(loss_out.approx_kl.detach().item()))
                    if loss_out.clip_fraction is not None:
                        clip_fracs.append(float(loss_out.clip_fraction.detach().item()))

            vloss = self._critic_update(buffers)
            if vloss is not None:
                value_losses.append(float(vloss.loss.detach().item()))

        if policy_losses:
            stats.policy_loss = sum(policy_losses) / len(policy_losses)
        if value_losses:
            stats.value_loss = sum(value_losses) / len(value_losses)
        if entropies:
            stats.entropy = sum(entropies) / len(entropies)
        if kls:
            stats.approx_kl = sum(kls) / len(kls)
        if clip_fracs:
            stats.clip_fraction = sum(clip_fracs) / len(clip_fracs)

        # Adaptive learning rate (KL threshold, Sec 4.6 / Table 4).
        if cfg.adaptive_lr and stats.approx_kl > 0.0:
            if stats.approx_kl > 2.0 * cfg.kl_threshold:
                self._lr = max(self._lr / 1.5, 1e-6)
            elif stats.approx_kl < 0.5 * cfg.kl_threshold:
                self._lr = min(self._lr * 1.5, 1e-2)
            for group in self.optimizer.param_groups:
                group["lr"] = self._lr
        stats.learning_rate = self._lr

        self._iteration += 1
        return stats

    # -- policy update for one plan ---------------------------------------
    def _policy_update(
        self, plan: PolicyAggregationPlan, buffers: List[RolloutBuffer]
    ) -> Optional[PolicyLossOutput]:
        cfg = self.config
        on_buf = buffers[plan.on_policy_block]
        on_flat = _flatten_buffer(on_buf)

        obs = on_flat["obs"].to(self.device)
        actions = on_flat["actions"].to(self.device)
        old_log_probs = on_flat["log_probs"].to(self.device)
        advantages = on_flat.get("advantages")
        if advantages is None:
            return None
        advantages = advantages.to(self.device)

        new_log_probs = self.networks.actor_log_prob(obs, actions, plan.policy_idx)
        on_out = on_policy_loss(
            new_log_probs, old_log_probs, advantages, clip_eps=cfg.clip_eps
        )

        entropy = self.networks.actor_entropy(obs, plan.policy_idx).mean()
        coef = self.entropy.coef_for(plan.policy_idx, iteration=self._iteration)
        if isinstance(coef, torch.Tensor):
            coef = float(coef.detach().item())

        off_out = None
        if plan.has_off_policy:
            off_batch = gather_off_policy_batch(
                plan, buffers, device=self.device
            )
            if off_batch is not None:
                off_obs = off_batch["obs"].to(self.device)
                off_actions = off_batch["actions"].to(self.device)
                off_old = off_batch["log_probs"].to(self.device)
                off_behavior = off_batch["behavior_log_probs"].to(self.device)
                off_adv = off_batch.get("advantages")
                if off_adv is not None:
                    off_adv = off_adv.to(self.device)
                    off_new = self.networks.actor_log_prob(
                        off_obs, off_actions, plan.policy_idx
                    )
                    off_out = off_policy_loss(
                        off_new,
                        off_old,
                        off_behavior,
                        off_adv,
                        clip_eps=cfg.clip_eps,
                    )

        combined = combined_policy_loss(
            on_policy=on_out,
            off_policy=off_out,
            entropy=entropy,
            entropy_coef=coef,
            off_policy_coef=cfg.off_policy_coef,
        )

        self.optimizer.zero_grad(set_to_none=True)
        combined.loss.backward()
        # phi_j is updated ONLY by policy j's objective.
        self.networks.zero_other_latent_grads(plan.policy_idx)
        nn.utils.clip_grad_norm_(self.networks.parameters(), cfg.grad_norm_clip)
        self.optimizer.step()
        return combined

    # -- critic update -----------------------------------------------------
    def _critic_update(self, buffers: List[RolloutBuffer]) -> Optional[CriticLossOutput]:
        cfg = self.config
        total_loss = None
        clip_frac = torch.zeros((), device=self.device)

        for plan in self.plans:
            on_flat = _flatten_buffer(buffers[plan.on_policy_block])
            obs = on_flat["obs"].to(self.device)
            targets = on_flat.get("value_targets")
            if targets is None:
                continue
            targets = targets.to(self.device)
            old_values = on_flat.get("values")
            old_values = old_values.to(self.device) if old_values is not None else None

            values, _ = self.networks.critic_value(obs, plan.policy_idx)
            values = values.reshape(-1)

            off_targets = None
            off_values = None
            if plan.has_off_policy:
                off_batch = gather_off_policy_batch(plan, buffers, device=self.device)
                if off_batch is not None and off_batch.get("value_targets") is not None:
                    off_obs = off_batch["obs"].to(self.device)
                    off_targets = off_batch["value_targets"].to(self.device)
                    off_values, _ = self.networks.critic_value(off_obs, plan.policy_idx)
                    off_values = off_values.reshape(-1)

            out = critic_loss(
                values=values,
                on_policy_targets=targets.reshape(-1),
                off_policy_targets=(off_targets.reshape(-1) if off_targets is not None else None),
                old_values=(old_values.reshape(-1) if old_values is not None else None),
                clip_eps=None,
                off_policy_coef=cfg.off_policy_coef,
                value_coef=cfg.critic_coef,
            )
            total_loss = out.loss if total_loss is None else total_loss + out.loss
            clip_frac = clip_frac + out.value_clip_fraction

        if total_loss is None:
            return None

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.networks.parameters(), cfg.grad_norm_clip)
        self.optimizer.step()

        num = max(1, len(self.plans))
        return CriticLossOutput(
            loss=total_loss.detach(),
            value_loss=total_loss.detach(),
            value_clip_fraction=clip_frac / num,
        )

    # -- full run ----------------------------------------------------------
    def train(self, num_iterations: Optional[int] = None) -> List[Dict[str, float]]:
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        num_iterations = int(num_iterations)

        history: List[Dict[str, float]] = []
        start = time.time()
        cumulative = 0
        for it in range(num_iterations):
            t0 = time.time()
            stats = self.train_iteration()
            stats.wall_time = time.time() - t0
            cumulative += stats.total_transitions
            record = stats.to_dict()
            record["total_transitions"] = cumulative
            history.append(record)

            if self.config.log_interval and (it + 1) % self.config.log_interval == 0:
                print(
                    f"[sapg] iter {it + 1}/{num_iterations} "
                    f"transitions={cumulative} "
                    f"policy_loss={stats.policy_loss:.4f} "
                    f"value_loss={stats.value_loss:.4f} "
                    f"entropy={stats.entropy:.4f} kl={stats.approx_kl:.5f} "
                    f"lr={stats.learning_rate:.2e}",
                    flush=True,
                )
        _ = time.time() - start
        return history


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_trainer(
    config: SAPGConfig,
    env,
    obs_dim: int,
    action_dim: int,
    device: Optional[torch.device] = None,
) -> SAPGTrainer:
    """Build a :class:`SAPGTrainer` from a config and environment."""
    return SAPGTrainer(config, env, obs_dim, action_dim, device=device)


__all__ = ["SAPGTrainer", "UpdateStats", "build_trainer"]
