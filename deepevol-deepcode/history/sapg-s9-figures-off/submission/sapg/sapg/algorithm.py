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
    on_policy_loss: float = 0.0
    off_policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    per_policy_loss: List[float] = field(default_factory=list)
    per_policy_entropy: List[float] = field(default_factory=list)
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "iteration": self.iteration,
            "total_transitions": self.total_transitions,
            "policy_loss": self.policy_loss,
            "on_policy_loss": self.on_policy_loss,
            "off_policy_loss": self.off_policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "wall_time": self.wall_time,
        }


# ---------------------------------------------------------------------------
# Adaptive learning rate (KL-based)
# ---------------------------------------------------------------------------
class AdaptiveLR:
    """Adjusts the learning rate based on the measured approximate KL.

    If the KL divergence between the old and new policy exceeds
    ``kl_threshold`` the learning rate is reduced; if it is well below the
    threshold it is increased.  This mirrors the adaptive schedule used in the
    paper (threshold 0.016).
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
        self.base_lr = base_lr
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr if max_lr is not None else base_lr * 10.0
        self.factor = factor
        self.current_lr = base_lr

    def step(self, approx_kl: float) -> float:
        """Update the LR given the latest approximate KL and return it."""
        if approx_kl > 2.0 * self.kl_threshold:
            self.current_lr = max(self.min_lr, self.current_lr / self.factor)
        elif approx_kl < 0.5 * self.kl_threshold:
            self.current_lr = min(self.max_lr, self.current_lr * self.factor)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr
        return self.current_lr


# ---------------------------------------------------------------------------
# SAPG trainer
# ---------------------------------------------------------------------------
class SAPGTrainer:
    """Implements Algorithm 1 of the SAPG paper.

    Parameters
    ----------
    config:
        Task/algorithm configuration (see :mod:`sapg.config`).
    env:
        Vectorized environment exposing ``reset()`` and ``step(actions)``.
    obs_dim, action_dim:
        Observation and action dimensionality of the task.
    device:
        Torch device used for the networks and buffers.
    """

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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device or torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # --- Networks: shared backbones + per-policy latents ---------------
        self.networks: SAPGNetworks = build_networks(
            config, obs_dim, action_dim
        ).to(self.device)

        # --- Optimizer: single Adam over all parameters --------------------
        self.optimizer = torch.optim.Adam(
            self.networks.parameters(), lr=config.learning_rate
        )
        self.adaptive_lr = AdaptiveLR(
            self.optimizer,
            base_lr=config.learning_rate,
            kl_threshold=config.kl_threshold,
        )

        # --- Entropy coefficients (Sec 4.5) --------------------------------
        self.entropy_scheduler: EntropyScheduler = build_entropy_coefs(
            config, device=self.device
        )
        # Register learnable entropy parameters with the optimizer if present.
        entropy_params = list(self.entropy_scheduler.parameters())
        if entropy_params:
            self.optimizer.add_param_group({"params": entropy_params})

        # --- Rollout collection --------------------------------------------
        self.collector: RolloutCollector = build_rollout_collector(
            env, self.networks, config, obs_dim, action_dim, device=self.device
        )

        # --- Aggregation plans (Sec 4.2-4.3) -------------------------------
        self.plans: List[PolicyAggregationPlan] = build_aggregation_plans(
            num_blocks=config.num_blocks,
            aggregation=config.aggregation,
            entropy_coef=config.entropy_coef,
            subsample_off_policy=config.subsample_off_policy,
            block_size=config.envs_per_block,
        )

        # --- Bookkeeping ---------------------------------------------------
        self.iteration = 0
        self.total_transitions = 0
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, num_iterations: Optional[int] = None) -> List[UpdateStats]:
        """Run the full training loop and return per-iteration statistics."""
        num_iterations = num_iterations or self.config.num_iterations
        history: List[UpdateStats] = []
        for _ in range(num_iterations):
            stats = self.train_iteration()
            history.append(stats)
            self.iteration += 1
        return history

    def train_iteration(self) -> UpdateStats:
        """Execute one outer iteration of Algorithm 1."""
        t_start = time.time()
        cfg = self.config

        # Step 2: collect D_1 .. D_M --------------------------------------
        buffers: List[RolloutBuffer] = self.collector.collect()
        block_size = buffers[0].size()
        self.total_transitions += block_size * cfg.num_blocks

        # Step 3-4: build aggregation plans and gather off-policy batches --
        off_policy_batches: List[Optional[Dict[str, torch.Tensor]]] = []
        for plan in self.plans:
            batch = gather_off_policy_batch(
                plan, buffers, generator=self._rng, device=self.device
            )
            off_policy_batches.append(batch)

        # Step 5: mini-epoch updates --------------------------------------
        stats = self._update(buffers, off_policy_batches)
        stats.iteration = self.iteration
        stats.total_transitions = self.total_transitions
        stats.wall_time = time.time() - t_start
        return stats

    # ------------------------------------------------------------------
    # Update internals
    # ------------------------------------------------------------------
    def _update(
        self,
        buffers: List[RolloutBuffer],
        off_policy_batches: List[Optional[Dict[str, torch.Tensor]]],
    ) -> UpdateStats:
        cfg = self.config
        num_mini_epochs = cfg.num_mini_epochs
        minibatch_size = cfg.minibatch_size

        agg = UpdateStats()
        agg.per_policy_loss = [0.0] * cfg.num_blocks
        agg.per_policy_entropy = [0.0] * cfg.num_blocks

        n_updates = 0
        for _ in range(num_mini_epochs):
            for plan, off_batch in zip(self.plans, off_policy_batches):
                i = plan.policy_idx
                buf = buffers[plan.on_policy_block]

                # ---- On-policy minibatch sampling -----------------------
                on_batch = self._sample_on_policy(buf, minibatch_size)
                if on_batch is None:
                    continue

                # ---- Forward pass: new log-probs / values ---------------
                latent = self.networks.latent(i)
                new_log_probs = self.networks.actor_log_prob(
                    on_batch["obs"], on_batch["actions"], latent
                )
                values = self.networks.critic_value(on_batch["obs"], latent)

                # ---- On-policy policy loss (Eq. 2) ----------------------
                on_out: PolicyLossOutput = on_policy_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=on_batch["log_probs"],
                    advantages=on_batch["advantages"],
                    clip_eps=cfg.clip_eps,
                )

                # ---- Off-policy policy loss (Eq. 3) ---------------------
                off_out: Optional[PolicyLossOutput] = None
                off_values = None
                if off_batch is not None and off_batch["obs"].shape[0] > 0:
                    off_new_log_probs = self.networks.actor_log_prob(
                        off_batch["obs"], off_batch["actions"], latent
                    )
                    off_out = off_policy_loss(
                        new_log_probs=off_new_log_probs,
                        old_log_probs=off_batch["old_log_probs"],
                        behavior_log_probs=off_batch["behavior_log_probs"],
                        advantages=off_batch["advantages"],
                        clip_eps=cfg.clip_eps,
                    )
                    off_values = self.networks.critic_value(
                        off_batch["obs"], latent
                    )

                # ---- Entropy (Sec 4.5) ----------------------------------
                entropy = self.networks.actor_entropy(on_batch["obs"], latent)
                entropy_coef = self.entropy_scheduler.coef_for(
                    i, iteration=self.iteration
                )

                # ---- Combined policy loss (Eq. 4) -----------------------
                policy_out = combined_policy_loss(
                    on_policy=on_out,
                    off_policy=off_out,
                    entropy=entropy,
                    entropy_coef=entropy_coef,
                    off_policy_coef=cfg.off_policy_coef,
                )

                # ---- Critic loss (Eq. 7-9) ------------------------------
                critic_out: CriticLossOutput = critic_loss(
                    values=values,
                    on_policy_targets=on_batch["value_targets"],
                    off_policy_targets=(
                        off_batch["value_targets"] if off_batch is not None else None
                    ),
                    old_values=on_batch["values"],
                    clip_eps=None,
                    off_policy_coef=cfg.off_policy_coef,
                    value_coef=cfg.critic_coef,
                )
                if off_values is not None and off_batch is not None:
                    # Off-policy critic term uses its own forward values.
                    off_critic = critic_loss(
                        values=off_values,
                        off_policy_targets=off_batch["value_targets"],
                        off_policy_coef=cfg.off_policy_coef,
                        value_coef=cfg.critic_coef,
                    )
                    critic_out = CriticLossOutput(
                        loss=critic_out.loss + off_critic.loss,
                        value_loss=critic_out.value_loss + off_critic.value_loss,
                        value_clip_fraction=critic_out.value_clip_fraction,
                    )

                total_loss = policy_out.loss + critic_out.loss

                # ---- Backward: phi_j updated only by its own objective --
                self.optimizer.zero_grad(set_to_none=True)
                self.networks.zero_other_latent_grads(i)
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.networks.parameters(), cfg.grad_norm_clip
                )
                self.optimizer.step()

                # ---- Accumulate statistics ------------------------------
                agg.policy_loss += float(policy_out.loss.detach())
                agg.on_policy_loss += float(on_out.loss.detach())
                if off_out is not None:
                    agg.off_policy_loss += float(off_out.loss.detach())
                agg.value_loss += float(critic_out.value_loss.detach())
                agg.entropy += float(entropy.detach().mean())
                agg.approx_kl += float(on_out.approx_kl.detach())
                agg.clip_fraction += float(on_out.clip_fraction.detach())
                agg.grad_norm += float(grad_norm)
                agg.per_policy_loss[i] += float(policy_out.loss.detach())
                agg.per_policy_entropy[i] += float(entropy.detach().mean())
                n_updates += 1

        if n_updates > 0:
            for key in (
                "policy_loss",
                "on_policy_loss",
                "off_policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
            ):
                setattr(agg, key, getattr(agg, key) / n_updates)
            agg.per_policy_loss = [v / n_updates for v in agg.per_policy_loss]
            agg.per_policy_entropy = [
                v / n_updates for v in agg.per_policy_entropy
            ]

        # Adaptive LR based on the average approximate KL.
        if cfg.adaptive_lr:
            agg.learning_rate = self.adaptive_lr.step(agg.approx_kl)
        else:
            agg.learning_rate = cfg.learning_rate
        return agg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_on_policy(
        self, buf: RolloutBuffer, minibatch_size: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Flatten a buffer and sample a minibatch of transitions."""
        flat = buf.flat_all()
        n = flat["obs"].shape[0]
        if n == 0:
            return None
        size = min(minibatch_size, n)
        idx = torch.randint(0, n, (size,), generator=self._rng).to(self.device)
        return {k: v[idx] for k, v in flat.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "networks": self.networks.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "entropy": self.entropy_scheduler.state_dict(),
            "iteration": self.iteration,
            "total_transitions": self.total_transitions,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.networks.load_state_dict(state["networks"])
        self.optimizer.load_state_dict(state["optimizer"])
        if "entropy" in state:
            self.entropy_scheduler.load_state_dict(state["entropy"])
        self.iteration = state.get("iteration", 0)
        self.total_transitions = state.get("total_transitions", 0)


# ---------------------------------------------------------------------------
# Convenience factory
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


__all__ = ["SAPGTrainer", "UpdateStats", "AdaptiveLR", "build_trainer"]
