"""Follower policy update for SAPG.

In SAPG the environment population is split into ``num_blocks`` blocks.  Each
block ``j`` owns a *follower* policy that is parameterised by the shared
networks ``B_theta`` (actor) and ``C_psi`` (critic) conditioned on a
per-follower parameter vector ``phi_j``.

A follower is updated with the standard PPO clipped surrogate objective on the
rollout that *it* collected (on-policy data).  The follower therefore behaves
exactly like a small independent PPO learner; the diversity between followers
comes from (a) different ``phi_j`` initialisations, (b) different environment
blocks (different object/goal initialisations) and (c) different random seeds.

The leader (see :mod:`sapg.sapg.leader`) is trained separately on the union of
all followers' transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .buffer import RolloutBuffer
from .networks import GaussianPolicy, ValueNetwork
from .ppo import (
    KLScheduler,
    PPOHyperParams,
    clip_grad_norm_,
    compute_ppo_loss,
    explained_variance,
    normalize_advantages,
)


@dataclass
class FollowerStats:
    """Aggregated diagnostics for one follower update."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    bounds_loss: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    total_loss: float = 0.0
    explained_variance: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    num_updates: int = 0

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        out = {}
        for k, v in self.__dict__.items():
            out[f"{prefix}{k}"] = v
        return out


class Follower:
    """A single follower policy operating on one block of environments.

    Parameters
    ----------
    block_id:
        Index of the environment block this follower owns.
    policy, value:
        *Shared* networks (``B_theta`` / ``C_psi``).  All followers share the
        same weights; they are distinguished by ``phi`` (and by the data they
        collect).
    optimizer:
        Adam optimiser over the shared network parameters.
    phi:
        Per-follower conditioning vector of shape ``(phi_dim,)``.  If ``None``
        the follower is unconditioned (``phi_dim == 0``).
    hparams:
        :class:`~sapg.sapg.ppo.PPOHyperParams` instance.
    """

    def __init__(
        self,
        block_id: int,
        policy: GaussianPolicy,
        value: ValueNetwork,
        optimizer: torch.optim.Optimizer,
        phi: Optional[torch.Tensor] = None,
        hparams: Optional[PPOHyperParams] = None,
        kl_scheduler: Optional[KLScheduler] = None,
        device: str = "cpu",
    ) -> None:
        self.block_id = int(block_id)
        self.policy = policy
        self.value = value
        self.optimizer = optimizer
        self.hparams = hparams if hparams is not None else PPOHyperParams()
        self.kl_scheduler = kl_scheduler
        self.device = torch.device(device)

        if phi is None:
            phi = torch.zeros(0, device=self.device)
        self.phi = phi.to(self.device).float().view(1, -1)

        # Running statistics used by the curriculum / logging.
        self.last_stats = FollowerStats()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def phi_dim(self) -> int:
        return int(self.phi.shape[-1])

    def _phi_for(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.phi_dim == 0:
            return None
        return self.phi.expand(batch_size, -1)

    def _block_ids_for(self, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size,), self.block_id, dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------ #
    # rollout collection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ):
        """Sample an action for the follower's block.

        Returns ``(action, log_prob, value, new_lstm_state)``.
        """
        obs = obs.to(self.device).float()
        batch = obs.shape[0]
        phi = self._phi_for(batch)
        block_ids = self._block_ids_for(batch)

        action, log_prob, new_state = self.policy.act(
            obs, phi, block_ids, lstm_state, deterministic=deterministic
        )
        value, _ = self.value(obs, phi, lstm_state)
        return action, log_prob, value, new_state

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #
    def update(
        self,
        buffer: RolloutBuffer,
        last_values: torch.Tensor,
        last_dones: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> FollowerStats:
        """Run the PPO update for this follower on its own rollout.

        Parameters
        ----------
        buffer:
            The follower's :class:`~sapg.sapg.buffer.RolloutBuffer` (already
            filled with a rollout).
        last_values:
            Bootstrap values for the final observation, shape ``(num_envs,)``.
        """
        hp = self.hparams

        # ---- advantage estimation (on-policy) ------------------------- #
        advantages, returns = buffer.compute_advantages(
            last_values, last_dones, gamma=hp.gamma, tau=hp.tau
        )
        advantages = normalize_advantages(advantages)

        num_transitions = buffer.num_transitions
        # mini-batch size = num_envs * 4 (paper), expressed as number of
        # mini-batches over the flattened (T * N) transitions.
        mini_batch_size = max(1, buffer.num_envs * 4)
        num_minibatches = max(1, num_transitions // mini_batch_size)

        stats = FollowerStats()
        n_updates = 0

        for _ in range(hp.mini_epochs):
            for mb in buffer.get_minibatches(
                num_minibatches, device=self.device, shuffle=True, generator=generator
            ):
                obs = mb["obs"]
                actions = mb["actions"]
                old_log_probs = mb["log_probs"]
                mb_adv = mb["advantages"]
                mb_returns = mb["returns"]
                lstm_states = mb.get("lstm_states", None)

                batch = obs.shape[0]
                phi = self._phi_for(batch)
                block_ids = self._block_ids_for(batch)

                new_log_probs, entropy = self.policy.evaluate_actions(
                    obs, actions, phi, block_ids, lstm_states
                )
                values, _ = self.value(obs, phi, lstm_states)

                loss, info = compute_ppo_loss(
                    new_log_probs=new_log_probs,
                    old_log_probs=old_log_probs,
                    values=values,
                    returns=mb_returns,
                    advantages=mb_adv,
                    entropy=entropy,
                    actions=actions,
                    clip_eps=hp.clip_eps,
                    entropy_coeff=hp.entropy_coeff,
                    critic_coeff=hp.critic_coeff,
                    bounds_loss_coeff=hp.bounds_loss_coeff,
                    action_bound=hp.action_bound,
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(
                    self.policy.parameters(), hp.grad_norm_clip
                )
                # also clip the critic parameters
                clip_grad_norm_(self.value.parameters(), hp.grad_norm_clip)
                self.optimizer.step()

                # ---- adaptive learning rate (KL based) ---------------- #
                if self.kl_scheduler is not None:
                    self.kl_scheduler.step(info["approx_kl"])
                    lr = self.kl_scheduler.lr
                else:
                    lr = self.optimizer.param_groups[0]["lr"]

                # ---- accumulate diagnostics --------------------------- #
                stats.policy_loss += info["policy_loss"]
                stats.value_loss += info["value_loss"]
                stats.entropy += info["entropy"]
                stats.bounds_loss += info["bounds_loss"]
                stats.approx_kl += info["approx_kl"]
                stats.clip_fraction += info["clip_fraction"]
                stats.total_loss += info["total_loss"]
                stats.grad_norm += grad_norm
                stats.learning_rate = lr
                n_updates += 1

        if n_updates > 0:
            for key in (
                "policy_loss",
                "value_loss",
                "entropy",
                "bounds_loss",
                "approx_kl",
                "clip_fraction",
                "total_loss",
                "grad_norm",
            ):
                setattr(stats, key, getattr(stats, key) / n_updates)
        stats.num_updates = n_updates
        stats.explained_variance = explained_variance(
            buffer.values.reshape(-1), returns.reshape(-1)
        )

        self.last_stats = stats
        return stats


# ---------------------------------------------------------------------- #
# factory
# ---------------------------------------------------------------------- #
def build_followers(
    num_blocks: int,
    policy: GaussianPolicy,
    value: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    phi_dim: int = 0,
    hparams: Optional[PPOHyperParams] = None,
    kl_scheduler: Optional[KLScheduler] = None,
    device: str = "cpu",
    seed: int = 0,
) -> List[Follower]:
    """Create ``num_blocks`` followers sharing the same networks.

    Each follower receives a distinct random ``phi_j`` (when ``phi_dim > 0``),
    which is what makes the followers explore differently.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    followers: List[Follower] = []
    for j in range(num_blocks):
        if phi_dim > 0:
            phi = torch.randn(phi_dim, generator=gen) * 0.1
        else:
            phi = None
        followers.append(
            Follower(
                block_id=j,
                policy=policy,
                value=value,
                optimizer=optimizer,
                phi=phi,
                hparams=hparams,
                kl_scheduler=kl_scheduler,
                device=device,
            )
        )
    return followers
