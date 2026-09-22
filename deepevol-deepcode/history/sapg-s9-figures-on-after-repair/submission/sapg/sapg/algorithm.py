"""SAPG core algorithm: Split and Aggregate Policy Gradients.

This module implements Algorithm 1 from the paper "SAPG: Split and Aggregate
Policy Gradients".

The algorithm splits ``N`` parallel environments into ``M`` blocks.  Each block
``j`` is controlled by its own policy ``pi_j`` (a shared backbone ``B_theta``
conditioned on a per-policy latent vector ``phi_j``).  Policy 1 is the *leader*;
policies ``2..M`` are *followers*.

Each iteration:

1. Collect ``H`` steps of on-policy data ``D_j`` for every block ``j``.
2. Uniformly subsample ``|D_1|`` transitions from ``union_{j=2}^{M} D_j`` to
   form the off-policy batch ``D_1'``.
3. The leader's objective combines its on-policy loss with an
   importance-sampled off-policy loss over ``D_1'`` (Eq. 4).
4. Followers optimise their own on-policy loss (plus optional entropy bonus).
5. A single gradient step updates the shared parameters ``theta`` (actor),
   ``psi`` (critic) and the per-policy latents ``phi_j``.

Gradient routing
----------------
``theta`` and ``psi`` receive gradients from *all* objectives.  ``phi_j`` only
receives gradients from policy ``j``'s own objective: whenever policy ``i``
evaluates data collected by policy ``j != i`` we detach ``phi_j`` (and vice
versa).  This is enforced by using :meth:`MultiPolicyActorCritic.get_phi` for
the "own" policy and :meth:`MultiPolicyActorCritic.get_phi_detached` for the
"behaviour" policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .gae import compute_advantages, compute_n_step_returns, compute_off_policy_targets
from .losses import (
    bounds_loss,
    critic_loss,
    entropy_bonus,
    off_policy_policy_loss,
    on_policy_policy_loss,
)
from .models import MultiPolicyActorCritic
from .rollout import (
    RolloutBuffer,
    collect_all_blocks,
    partition_blocks,
    subsample_off_policy,
)

__all__ = ["SAPGConfig", "SAPG", "SAPGTrainState"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SAPGConfig:
    """Hyper-parameters for the SAPG algorithm.

    Defaults follow the paper's Appendix B (Tables 2-4).  Task specific values
    are supplied by the YAML configs in ``configs/``.
    """

    # --- environment / parallelism -------------------------------------
    num_envs: int = 24576
    num_policies: int = 6
    horizon: int = 16

    # --- optimisation ---------------------------------------------------
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.95  # GAE lambda
    n_step: int = 3  # n-step critic target (Eq. 5)
    clip_eps: float = 0.1
    critic_coef: float = 4.0
    lambda_off: float = 1.0  # weight of off-policy loss (Eq. 4)
    bounds_coef: float = 1e-4
    max_grad_norm: float = 1.0

    # --- mini-batching --------------------------------------------------
    mini_epochs: int = 2
    num_mini_batches: int = 4  # mini-batch size = num_envs * num_mini_batches

    # --- entropy regularisation (followers only) ------------------------
    entropy_coef: float = 0.0  # sigma in Eq. (Sec 4.5)
    learnable_entropy_coef: bool = False

    # --- KL-adaptive learning rate --------------------------------------
    use_kl_adaptive_lr: bool = True
    kl_threshold: float = 0.016
    kl_adaptive_factor: float = 1.5

    # --- misc -----------------------------------------------------------
    normalize_advantage: bool = True
    seed: int = 0
    device: str = "cuda"
    max_iterations: int = 100000
    log_interval: int = 1


@dataclass
class SAPGTrainState:
    """Mutable training state carried across iterations."""

    iteration: int = 0
    total_samples: int = 0
    obs: Optional[torch.Tensor] = None
    lstm_states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    learning_rate: float = 1e-4
    history: List[Dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SAPG algorithm
# ---------------------------------------------------------------------------
class SAPG:
    """Split and Aggregate Policy Gradients (Algorithm 1).

    Parameters
    ----------
    env:
        Vectorised environment exposing ``num_envs``, ``obs_dim``, ``act_dim``,
        ``reset()`` and ``step(actions)``.  See ``envs/isaacgym_wrapper.py``.
    actor_critic:
        A :class:`MultiPolicyActorCritic` with ``num_policies`` latents.
    config:
        :class:`SAPGConfig` instance.
    """

    def __init__(
        self,
        env,
        actor_critic: MultiPolicyActorCritic,
        config: SAPGConfig,
    ) -> None:
        self.env = env
        self.ac = actor_critic
        self.cfg = config

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.ac.to(self.device)

        # Blocks: contiguous partition of the N environments into M blocks.
        self.blocks = partition_blocks(config.num_envs, config.num_policies)
        self.block_sizes = [end - start for start, end in self.blocks]

        # Optimiser over shared params + per-policy latents.
        params = list(self.ac.actor_parameters()) + list(self.ac.critic_parameters())
        params += list(self.ac.phi_parameters())
        if config.learnable_entropy_coef:
            params += list(self.ac.entropy_parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

        self.state = SAPGTrainState(learning_rate=config.learning_rate)

    # ------------------------------------------------------------------
    # Initialisation / reset
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset the environment and per-policy LSTM states."""
        obs = self.env.reset()
        obs = obs.to(self.device)
        self.state.obs = obs
        self.state.lstm_states = [None] * self.cfg.num_policies
        return obs

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect(self) -> List[RolloutBuffer]:
        """Collect one rollout per block (Algorithm 1, line 3)."""
        buffers, obs, lstm_states = collect_all_blocks(
            env=self.env,
            actor_critic=self.ac,
            num_policies=self.cfg.num_policies,
            horizon=self.cfg.horizon,
            obs=self.state.obs,
            lstm_states=self.state.lstm_states,
            device=self.device,
        )
        self.state.obs = obs
        self.state.lstm_states = lstm_states
        return buffers

    # ------------------------------------------------------------------
    # Loss assembly
    # ------------------------------------------------------------------
    def _prepare_on_policy(
        self, buffer: RolloutBuffer, policy_index: int
    ) -> Dict[str, torch.Tensor]:
        """Compute GAE advantages and n-step critic targets for a buffer."""
        rewards = buffer.rewards
        values = buffer.values
        dones = buffer.dones

        next_value = buffer.next_values
        if next_value is not None:
            next_value = next_value[-1] if next_value.dim() == 3 else next_value

        advantages, returns = compute_advantages(
            rewards,
            values,
            dones,
            next_value=next_value,
            gamma=self.cfg.gamma,
            tau=self.cfg.tau,
            normalize=self.cfg.normalize_advantage,
        )
        n_step_targets = compute_n_step_returns(
            rewards, values, dones, gamma=self.cfg.gamma, n=self.cfg.n_step
        )
        return {
            "advantages": advantages,
            "returns": returns,
            "n_step_targets": n_step_targets,
        }

    def _leader_off_policy_batch(
        self, follower_buffers: List[RolloutBuffer]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Uniformly subsample |D_1| transitions from union of follower data."""
        if not follower_buffers:
            return None
        target_size = follower_buffers[0].size
        return subsample_off_policy(follower_buffers, target_size)

    # ------------------------------------------------------------------
    # Single optimisation step
    # ------------------------------------------------------------------
    def update(self, buffers: List[RolloutBuffer]) -> Dict[str, float]:
        """Run mini-batch PPO updates over all policies (Algorithm 1, 5-9)."""
        cfg = self.cfg
        num_policies = cfg.num_policies

        # --- pre-compute advantages / targets per policy ---------------
        on_policy_data = [
            self._prepare_on_policy(buffers[j], j) for j in range(num_policies)
        ]

        # --- off-policy batch for the leader ---------------------------
        off_batch = self._leader_off_policy_batch(buffers[1:])

        # --- flatten buffers for mini-batching -------------------------
        flat = [self._flatten_buffer(buffers[j], on_policy_data[j]) for j in range(num_policies)]
        if off_batch is not None:
            off_batch = self._flatten_off_policy(off_batch)

        # Mini-batch size = num_envs * num_mini_batches (paper Appendix B).
        mini_batch_size = max(1, cfg.num_envs * cfg.num_mini_batches)
        num_samples = flat[0]["obs"].shape[0]
        num_mini_batches = max(1, num_samples // mini_batch_size)

        metrics: Dict[str, float] = {}
        for _ in range(cfg.mini_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for mb in range(num_mini_batches):
                idx = perm[mb * mini_batch_size : (mb + 1) * mini_batch_size]
                if idx.numel() == 0:
                    continue
                mb_metrics = self._mini_batch_step(flat, off_batch, idx)
                for k, v in mb_metrics.items():
                    metrics[k] = metrics.get(k, 0.0) + v

        # Average metrics over mini-batches.
        n_updates = max(1, cfg.mini_epochs * num_mini_batches)
        metrics = {k: v / n_updates for k, v in metrics.items()}

        # --- KL-adaptive learning rate ---------------------------------
        if cfg.use_kl_adaptive_lr and "approx_kl" in metrics:
            self._adapt_learning_rate(metrics["approx_kl"])

        return metrics

    def _mini_batch_step(
        self,
        flat: List[Dict[str, torch.Tensor]],
        off_batch: Optional[Dict[str, torch.Tensor]],
        idx: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute losses for one mini-batch and take a gradient step."""
        cfg = self.cfg
        num_policies = cfg.num_policies

        total_policy_loss = torch.zeros((), device=self.device)
        total_critic_loss = torch.zeros((), device=self.device)
        total_entropy = torch.zeros((), device=self.device)
        total_bounds = torch.zeros((), device=self.device)
        info: Dict[str, float] = {}
        kl_accum = 0.0

        for j in range(num_policies):
            data = flat[j]
            obs = data["obs"][idx]
            actions = data["actions"][idx]
            old_log_probs = data["log_probs"][idx]
            advantages = data["advantages"][idx]
            n_step_targets = data["n_step_targets"][idx]
            lstm_states = data.get("lstm_states")

            # Own policy: phi_j receives gradients here.
            phi_j = self.ac.get_phi(j, obs.shape[0])
            dist, values = self.ac.evaluate_actions(
                obs, j, actions, lstm_state=self._slice_lstm(lstm_states, idx)
            )
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().sum(-1)

            # --- on-policy policy loss (Eq. 2) -------------------------
            on_loss, on_info = on_policy_policy_loss(
                log_probs, old_log_probs, advantages, clip_eps=cfg.clip_eps
            )
            policy_loss = on_loss
            kl_accum += on_info["approx_kl"]

            # --- leader off-policy loss (Eq. 3) ------------------------
            if j == 0 and off_batch is not None:
                off_loss, off_info = self._leader_off_policy_loss(off_batch, idx)
                policy_loss = policy_loss + cfg.lambda_off * off_loss
                info["off_policy_loss"] = info.get("off_policy_loss", 0.0) + float(
                    off_loss.detach()
                )
                info["off_clip_fraction"] = info.get("off_clip_fraction", 0.0) + off_info[
                    "clip_fraction"
                ]

            # --- follower entropy bonus (Sec 4.5) ----------------------
            if j > 0 and cfg.entropy_coef != 0.0:
                ent = entropy_bonus(entropy, j, cfg.entropy_coef)
                policy_loss = policy_loss + ent
                total_entropy = total_entropy + ent.detach()

            # --- bounds regularisation ---------------------------------
            mean = dist.mean
            b_loss = bounds_loss(mean, coef=cfg.bounds_coef)
            total_bounds = total_bounds + b_loss.detach()

            total_policy_loss = total_policy_loss + policy_loss

            # --- critic loss (Eq. 7-9) ---------------------------------
            c_loss = critic_loss(values, n_step_targets, reduction="mean")
            total_critic_loss = total_critic_loss + c_loss

            info["on_policy_loss"] = info.get("on_policy_loss", 0.0) + float(
                on_loss.detach()
            )
            info["clip_fraction"] = info.get("clip_fraction", 0.0) + on_info[
                "clip_fraction"
            ]

        # --- assemble total loss ---------------------------------------
        loss = (
            -total_policy_loss
            + cfg.critic_coef * total_critic_loss
            + total_bounds
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            cfg.max_grad_norm,
        )
        self.optimizer.step()

        info["approx_kl"] = kl_accum / max(1, num_policies)
        info["policy_loss"] = float(total_policy_loss.detach())
        info["critic_loss"] = float(total_critic_loss.detach())
        info["entropy"] = float(total_entropy)
        info["grad_norm"] = float(grad_norm)
        return info

    def _leader_off_policy_loss(
        self, off_batch: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Importance-sampled off-policy loss for the leader (Eq. 3, 6)."""
        cfg = self.cfg
        obs = off_batch["obs"][idx]
        actions = off_batch["actions"][idx]
        old_log_probs_i = off_batch["log_probs"][idx]  # pi_i,old (leader, old)
        behaviour_log_probs = off_batch["behaviour_log_probs"][idx]  # pi_j
        advantages = off_batch["advantages"][idx]
        off_targets = off_batch["off_targets"][idx]

        # Leader's current policy pi_i (phi_1 receives gradients).
        dist, values = self.ac.evaluate_actions(obs, 0, actions)
        log_probs_i = dist.log_prob(actions)

        off_loss, off_info = off_policy_policy_loss(
            log_probs_i,
            behaviour_log_probs,
            old_log_probs_i,
            advantages,
            clip_eps=cfg.clip_eps,
        )

        # Off-policy critic loss (Eq. 8).
        c_loss = critic_loss(values, off_targets, reduction="mean")
        # Fold the off-policy critic loss into the returned loss via a
        # detached-free path: we add it scaled by critic_coef.
        off_loss = off_loss - cfg.critic_coef * c_loss
        return off_loss, off_info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flatten_buffer(
        self, buffer: RolloutBuffer, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Flatten a time-major buffer into (T*B, ...) tensors."""
        T, B = buffer.rewards.shape
        flat = {
            "obs": buffer.obs.reshape(T * B, *buffer.obs.shape[2:]),
            "actions": buffer.actions.reshape(T * B, *buffer.actions.shape[2:]),
            "log_probs": buffer.log_probs.reshape(T * B, *buffer.log_probs.shape[2:]),
            "advantages": data["advantages"].reshape(T * B, *data["advantages"].shape[2:]),
            "n_step_targets": data["n_step_targets"].reshape(
                T * B, *data["n_step_targets"].shape[2:]
            ),
        }
        if buffer.lstm_states is not None:
            flat["lstm_states"] = buffer.lstm_states
        return flat

    def _flatten_off_policy(
        self, off_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Flatten the off-policy batch and compute advantages/targets."""
        rewards = off_batch["rewards"]
        values = off_batch["values"]
        dones = off_batch["dones"]
        next_values = off_batch.get("next_values")

        # Off-policy advantages use GAE with the behaviour values.
        advantages, _ = compute_advantages(
            rewards,
            values,
            dones,
            next_value=next_values,
            gamma=self.cfg.gamma,
            tau=self.cfg.tau,
            normalize=self.cfg.normalize_advantage,
        )
        off_targets = compute_off_policy_targets(
            rewards, next_values, dones, gamma=self.cfg.gamma
        )

        flat = {
            "obs": off_batch["obs"],
            "actions": off_batch["actions"],
            "log_probs": off_batch["log_probs"],
            "behaviour_log_probs": off_batch["log_probs"],
            "advantages": advantages,
            "off_targets": off_targets,
        }
        return flat

    @staticmethod
    def _slice_lstm(
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]],
        idx: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if lstm_states is None:
            return None
        h, c = lstm_states
        return h[:, idx], c[:, idx]

    def _adapt_learning_rate(self, approx_kl: float) -> None:
        """KL-adaptive learning rate (Appendix B)."""
        cfg = self.cfg
        lr = self.state.learning_rate
        if approx_kl > cfg.kl_threshold * 2.0:
            lr = max(lr / cfg.kl_adaptive_factor, 1e-6)
        elif approx_kl < cfg.kl_threshold / 2.0:
            lr = min(lr * cfg.kl_adaptive_factor, 1e-2)
        if lr != self.state.learning_rate:
            self.state.learning_rate = lr
            for g in self.optimizer.param_groups:
                g["lr"] = lr

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self, max_iterations: Optional[int] = None) -> List[Dict[str, float]]:
        """Run the full Algorithm 1 training loop."""
        cfg = self.cfg
        max_iterations = max_iterations or cfg.max_iterations
        self.reset()

        for it in range(max_iterations):
            t0 = time.time()
            buffers = self.collect()
            metrics = self.update(buffers)

            samples = sum(b.size for b in buffers)
            self.state.total_samples += samples
            self.state.iteration += 1

            metrics["iteration"] = self.state.iteration
            metrics["total_samples"] = self.state.total_samples
            metrics["learning_rate"] = self.state.learning_rate
            metrics["time"] = time.time() - t0
            self.state.history.append(metrics)

            if self.state.iteration % cfg.log_interval == 0:
                print(
                    f"[SAPG] iter={self.state.iteration} "
                    f"samples={self.state.total_samples} "
                    f"policy_loss={metrics.get('policy_loss', 0.0):.4f} "
                    f"critic_loss={metrics.get('critic_loss', 0.0):.4f} "
                    f"kl={metrics.get('approx_kl', 0.0):.5f} "
                    f"lr={self.state.learning_rate:.2e} "
                    f"time={metrics['time']:.2f}s"
                )

        return self.state.history
