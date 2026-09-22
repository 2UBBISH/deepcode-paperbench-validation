"""SAPG: Split and Aggregate Policy Gradients.

Core algorithm implementation.

SAPG splits ``N`` parallel environments into ``B`` blocks.  Each block ``b``
runs its own *follower* policy ``pi_b`` (parameters ``phi_b``) and collects
transitions.  All transitions from all blocks are aggregated into a single
buffer which is *off-policy* with respect to the *leader* policy.  The leader
is then updated with a PPO-style clipped surrogate objective using the
importance ratio

    r_t = pi_leader(a_t | s_t) / pi_behavior(a_t | s_t)

which lets the leader latch onto high-reward off-policy trajectories while
retaining PPO's stability guarantees.  Followers are updated on-policy on
their own block data.

Two aggregation modes are supported:

* ``"leader"``     -- one designated leader is updated on the union of all
                      blocks' data (the default SAPG behaviour).
* ``"symmetric"``  -- no designated leader; every worker is updated with all
                      *other* workers' off-policy data (ablation, Fig. 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .policy import GaussianPolicy
from .rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SAPGConfig:
    """Hyper-parameters for the SAPG algorithm.

    Defaults follow the paper (Tables 2/3/4) and the reproduction plan.
    """

    # --- environment splitting -------------------------------------------
    num_blocks: int = 8                 # B: number of follower blocks
    num_envs_per_block: int = 1         # environments simulated per block

    # --- optimisation -----------------------------------------------------
    gamma: float = 0.99                 # discount factor
    tau: float = 0.95                   # GAE lambda
    clip_eps: float = 0.2               # PPO clipping epsilon
    value_loss_coef: float = 1.0        # lambda' in the paper's loss
    bounds_loss_coef: float = 0.001     # coefficient on the bounds loss
    entropy_coef: float = 0.0           # entropy bonus (0 in default configs)
    max_grad_norm: float = 1.0          # gradient clipping

    # --- learning-rate adaptation (KL threshold) --------------------------
    lr: float = 3e-4
    kl_threshold: float = 0.016
    lr_adapt: bool = True

    # --- mini-batching ----------------------------------------------------
    horizon: int = 16                   # rollout horizon (16 Kuka, 8 hands)
    num_mini_epochs: int = 2            # 2 (AllegroKuka), 5 (hands)
    mini_batch_multiplier: int = 4      # mini-batch size = num_envs * 4

    # --- aggregation ------------------------------------------------------
    aggregation: str = "leader"         # "leader" | "symmetric"
    leader_id: int = 0                  # index of the designated leader

    # --- misc -------------------------------------------------------------
    device: str = "cpu"
    normalize_advantages: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flatten_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Flatten (B, T, ...) tensors to (B*T, ...) for the network forward pass."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.dim() >= 2:
            out[key] = value.reshape(-1, *value.shape[2:])
        else:
            out[key] = value.reshape(-1)
    return out


def _compute_bounds_loss(policy: GaussianPolicy) -> torch.Tensor:
    """Soft bounds loss on the log-std to keep exploration in a sane range.

    Mirrors the ``bounds_loss`` term used by the paper's implementation:
    a soft penalty pushing ``log_std`` into ``[LOG_STD_MIN, LOG_STD_MAX]``.
    """
    log_std = policy.log_std
    soft_lower, soft_upper = -4.0, 1.0
    lower = torch.clamp(log_std - soft_lower, min=0.0)
    upper = torch.clamp(soft_upper - log_std, min=0.0)
    return (lower + upper).sum()


def _approx_kl(old_logprobs: torch.Tensor, new_logprobs: torch.Tensor) -> torch.Tensor:
    """Approximate KL divergence (Schulman's k3 estimator)."""
    log_ratio = new_logprobs - old_logprobs
    return torch.mean(torch.exp(log_ratio) - log_ratio - 1.0)


# ---------------------------------------------------------------------------
# SAPG algorithm
# ---------------------------------------------------------------------------
class SAPG:
    """Split and Aggregate Policy Gradients.

    Parameters
    ----------
    policy:
        The shared conditioned policy network ``B_theta`` wrapped in a
        :class:`GaussianPolicy`.  A single network is shared across all
        workers and conditioned on per-worker embeddings ``phi_j``.
    value_net:
        The shared conditioned critic ``C_psi``.
    config:
        :class:`SAPGConfig` instance.
    """

    def __init__(
        self,
        policy: GaussianPolicy,
        value_net: nn.Module,
        config: Optional[SAPGConfig] = None,
    ) -> None:
        self.config = config or SAPGConfig()
        self.device = torch.device(self.config.device)

        self.policy = policy.to(self.device)
        self.value_net = value_net.to(self.device)

        # Single optimizer over the shared networks (all phi_j embeddings live
        # inside the shared networks).
        params = list(self.policy.parameters()) + list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)

        self.current_lr = float(self.config.lr)

        # Bookkeeping for logging.
        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Learning-rate adaptation
    # ------------------------------------------------------------------
    def _adapt_lr(self, approx_kl: float) -> None:
        """KL-threshold based learning-rate schedule.

        If the approximate KL exceeds ``1.5 * kl_threshold`` the learning rate
        is halved; if it drops below ``kl_threshold / 1.5`` it is doubled.
        """
        if not self.config.lr_adapt:
            return
        threshold = self.config.kl_threshold
        if approx_kl > 1.5 * threshold:
            self.current_lr = max(self.current_lr / 1.5, 1e-6)
        elif approx_kl < threshold / 1.5:
            self.current_lr = min(self.current_lr * 1.5, 1e-2)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def _update_on_buffer(
        self,
        buffer: RolloutBuffer,
        worker_ids: torch.Tensor,
        num_mini_epochs: Optional[int] = None,
        tag: str = "update",
    ) -> Dict[str, float]:
        """Run PPO-style clipped surrogate updates on ``buffer``.

        ``worker_ids`` is a 1-D tensor of length ``B*T`` giving the worker
        index used to condition the shared networks for each transition.
        """
        cfg = self.config
        num_mini_epochs = num_mini_epochs or cfg.num_mini_epochs
        mini_batch_size = buffer.mini_batch_size(cfg.mini_batch_multiplier)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "num_updates": 0,
        }

        for _ in range(num_mini_epochs):
            for batch in buffer.get_mini_batches(
                mini_batch_size=mini_batch_size, num_mini_epochs=1
            ):
                flat = _flatten_batch(batch)
                obs = flat["obs"]
                actions = flat["actions"]
                old_logprobs = flat["old_logprobs"]
                old_values = flat["old_values"]
                advantages = flat["advantages"]
                returns = flat["returns"]

                # Worker ids for this mini-batch (conditioning of B_theta/C_psi).
                if "worker_ids" in flat:
                    mb_worker_ids = flat["worker_ids"].long()
                else:
                    mb_worker_ids = worker_ids[: obs.shape[0]].long()

                # --- evaluate actions under the current (leader) policy ----
                new_logprobs, entropy, _ = self.policy.evaluate_actions(
                    obs, actions, mb_worker_ids
                )
                new_values = self.value_net(obs, mb_worker_ids).squeeze(-1)

                # --- importance ratio -------------------------------------
                log_ratio = new_logprobs - old_logprobs
                ratio = torch.exp(log_ratio)

                # --- clipped surrogate objective --------------------------
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value loss -------------------------------------------
                value_loss = 0.5 * (new_values - returns).pow(2).mean()

                # --- bounds loss ------------------------------------------
                bounds_loss = _compute_bounds_loss(self.policy)

                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    + cfg.bounds_loss_coef * bounds_loss
                    - cfg.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                # --- logging ----------------------------------------------
                with torch.no_grad():
                    approx_kl = _approx_kl(old_logprobs, new_logprobs).item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean().item()
                    )
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += approx_kl
                stats["clip_fraction"] += clip_frac
                stats["num_updates"] += 1

        n = max(stats["num_updates"], 1)
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            stats[key] /= n

        self._adapt_lr(stats["approx_kl"])
        stats["learning_rate"] = self.current_lr
        self.last_stats = stats
        return stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        buffer: RolloutBuffer,
        num_mini_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Perform one SAPG iteration.

        Steps
        -----
        1. Compute GAE advantages/returns for every block.
        2. Update each follower on its own block data (on-policy PPO).
        3. Aggregate all blocks' transitions and update the leader with the
           clipped surrogate objective (off-policy w.r.t. the leader).

        For ``aggregation == "symmetric"`` step 3 is replaced by updating every
        worker on the union of all *other* workers' data.
        """
        cfg = self.config
        buffer.normalize_advantages() if cfg.normalize_advantages else None

        all_stats: Dict[str, float] = {}

        # --- 1. follower updates (on-policy, per block) -------------------
        follower_stats: List[Dict[str, float]] = []
        for b in range(cfg.num_blocks):
            block_buf = buffer.select_blocks([b])
            worker_ids = torch.full(
                (block_buf.num_blocks * block_buf.horizon,),
                b,
                dtype=torch.long,
                device=self.device,
            )
            follower_stats.append(
                self._update_on_buffer(
                    block_buf, worker_ids, num_mini_epochs=num_mini_epochs, tag=f"follower_{b}"
                )
            )
        all_stats["follower/policy_loss"] = sum(s["policy_loss"] for s in follower_stats) / len(follower_stats)
        all_stats["follower/value_loss"] = sum(s["value_loss"] for s in follower_stats) / len(follower_stats)
        all_stats["follower/approx_kl"] = sum(s["approx_kl"] for s in follower_stats) / len(follower_stats)

        # --- 2. aggregation ------------------------------------------------
        if cfg.aggregation == "leader":
            agg_buf = buffer.aggregate()
            worker_ids = agg_buf.worker_ids.reshape(-1).long().to(self.device)
            leader_stats = self._update_on_buffer(
                agg_buf, worker_ids, num_mini_epochs=num_mini_epochs, tag="leader"
            )
            all_stats["leader/policy_loss"] = leader_stats["policy_loss"]
            all_stats["leader/value_loss"] = leader_stats["value_loss"]
            all_stats["leader/approx_kl"] = leader_stats["approx_kl"]
            all_stats["leader/clip_fraction"] = leader_stats["clip_fraction"]
        elif cfg.aggregation == "symmetric":
            sym_stats: List[Dict[str, float]] = []
            for b in range(cfg.num_blocks):
                agg_buf = buffer.aggregate(exclude=b)
                worker_ids = torch.full(
                    (agg_buf.num_blocks * agg_buf.horizon,),
                    b,
                    dtype=torch.long,
                    device=self.device,
                )
                sym_stats.append(
                    self._update_on_buffer(
                        agg_buf, worker_ids, num_mini_epochs=num_mini_epochs, tag=f"sym_{b}"
                    )
                )
            all_stats["symmetric/policy_loss"] = sum(s["policy_loss"] for s in sym_stats) / len(sym_stats)
            all_stats["symmetric/approx_kl"] = sum(s["approx_kl"] for s in sym_stats) / len(sym_stats)
        else:
            raise ValueError(f"Unknown aggregation mode: {cfg.aggregation}")

        all_stats["learning_rate"] = self.current_lr
        self.last_stats = all_stats
        return all_stats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
        worker_ids: torch.Tensor,
        hidden=None,
        deterministic: bool = False,
    ):
        """Sample actions from the shared policy for the given workers."""
        return self.policy.act(obs, worker_ids, hidden, deterministic=deterministic)

    def value(self, obs: torch.Tensor, worker_ids: torch.Tensor) -> torch.Tensor:
        """Estimate state values with the shared critic."""
        return self.value_net(obs, worker_ids).squeeze(-1)

    def state_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.current_lr = state.get("current_lr", self.config.lr)
