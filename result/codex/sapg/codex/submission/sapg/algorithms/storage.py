"""Rollout storage: one buffer per policy, GAE / n-step returns, minibatching.

Layout
------
Every policy ``j`` of SAPG/DexPBT owns a block of ``N / M`` environments and
collects ``horizon`` steps of experience into its own buffer, so buffers are
stored with shape ``[horizon, num_envs_per_policy, ...]``.

Two kinds of targets are used (Sec. 4.1):

* on-policy.  ``n``-step returns with ``n = 3`` for the AllegroKuka tasks::

      V_on^target(s_t) = sum_{k=t}^{t+2} gamma^{k-t} r_k + gamma^3 V_old(s_{t+3})

  and GAE(``tau``) advantages for the policy loss (``tau = 0.95``).
* off-policy.  Because trajectory continuation is not available for data
  produced by *another* policy, an off-policy transition is treated as a
  1-step return::

      V_off^target(s_t) = r_t + gamma * V_old(s_{t+1})

  and ``A = V_off^target - V_old(s_t)``.

For recurrent policies a minibatch is a set of *whole* sequences (all
``horizon`` steps of a subset of environments, the ``seq_len = horizon``
convention from Table 2); that also keeps the off-policy sub-sampling (Sec.
4.3) sequence-aligned.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

import torch


def _to_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Generalised advantage estimation over ``[T, B]`` tensors.

    ``dones[t] = 1`` ends the episode at step ``t`` (no bootstrap through the
    terminal transition).
    """
    values_ext = torch.cat([values, last_value.unsqueeze(0)], dim=0)
    deltas = rewards + gamma * values_ext[1:] * (1.0 - dones) - values
    advantages = torch.zeros_like(values)
    running = torch.zeros_like(values_ext[0])
    for t in reversed(range(rewards.shape[0])):
        running = deltas[t] + gamma * gae_lambda * (1.0 - dones[t]) * running
        advantages[t] = running
    return advantages


def compute_nstep_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    nstep: int,
) -> torch.Tensor:
    """``n``-step return target of Eq. 5 (``n = 3`` in the paper's experiments)."""
    horizon = rewards.shape[0]
    values_ext = torch.cat([values, last_value.unsqueeze(0)], dim=0)
    targets = torch.zeros_like(values)
    for t in range(horizon):
        acc = torch.zeros_like(values_ext[0])
        discount = 1.0
        live = torch.ones_like(values_ext[0])
        for k in range(nstep):
            idx = t + k
            if idx >= horizon:
                break
            acc = acc + discount * live * rewards[idx]
            discount = discount * gamma
            live = live * (1.0 - dones[idx])
        idx = min(t + nstep, horizon)
        acc = acc + discount * live * values_ext[idx]
        targets[t] = acc
    return targets


class PolicyBuffer:
    """Per-policy rollout buffer of shape ``[T, B, dim]``."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        policy_id: int = 0,
    ) -> None:
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device
        self.policy_id = policy_id

        shape_obs = (horizon, num_envs, obs_dim)
        self.obs = torch.zeros(shape_obs, device=device)
        self.next_obs = torch.zeros(shape_obs, device=device)
        self.actions = torch.zeros((horizon, num_envs, action_dim), device=device)
        self.logprobs = torch.zeros((horizon, num_envs), device=device)
        self.values = torch.zeros((horizon, num_envs), device=device)
        self.rewards = torch.zeros((horizon, num_envs), device=device)
        self.dones = torch.zeros((horizon, num_envs), device=device)
        self.timeouts = torch.zeros((horizon, num_envs), device=device)
        self.advantages = torch.zeros((horizon, num_envs), device=device)
        self.value_targets = torch.zeros((horizon, num_envs), device=device)
        self.last_value = torch.zeros(num_envs, device=device)

    # ------------------------------------------------------------------ #
    def insert(self, t: int, obs, actions, logprobs, values, rewards, dones, timeouts, next_obs) -> None:
        self.obs[t] = _to_tensor(obs, self.device)
        self.actions[t] = _to_tensor(actions, self.device)
        self.logprobs[t] = _to_tensor(logprobs, self.device)
        self.values[t] = _to_tensor(values, self.device)
        self.rewards[t] = _to_tensor(rewards, self.device)
        self.dones[t] = _to_tensor(dones, self.device)
        self.timeouts[t] = _to_tensor(timeouts, self.device)
        self.next_obs[t] = _to_tensor(next_obs, self.device)

    # ------------------------------------------------------------------ #
    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        nstep: int = 3,
        critic_target: str = "nstep",
    ) -> None:
        """Fill ``advantages`` and ``value_targets``.

        ``last_value`` is ``V_old(s_T)`` produced by the critic at the end of
        the rollout (used to bootstrap truncated episodes and the final step).
        """
        self.last_value = last_value.detach()
        # ---- GAE(tau) ---------------------------------------------------
        advantages = compute_gae(
            self.rewards, self.values, self.dones, self.last_value, gamma, gae_lambda
        )
        self.advantages = advantages

        # ---- critic target ----------------------------------------------
        if critic_target == "nstep":
            self.value_targets = compute_nstep_returns(
                self.rewards, self.values, self.dones, self.last_value, gamma, nstep
            )
        elif critic_target in ("gae", "returns"):
            self.value_targets = advantages + self.values
        else:
            raise ValueError(f"Unknown critic_target '{critic_target}'")

    # ------------------------------------------------------------------ #
    def num_transitions(self) -> int:
        return self.horizon * self.num_envs

    def flatten(self, keys: Sequence[str]) -> Dict[str, torch.Tensor]:
        out = {}
        for key in keys:
            value = getattr(self, key)
            out[key] = value.reshape(-1, *value.shape[2:])
        return out


class OffPolicyData:
    """Sequence-aligned off-policy mini-dataset (leader <- followers)."""

    def __init__(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        behavior_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        mu: torch.Tensor,
        advantages: torch.Tensor,
        value_targets: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        # all tensors are [T, n_seq, ...]
        self.obs = obs
        self.actions = actions
        self.behavior_logprobs = behavior_logprobs
        self.old_logprobs = old_logprobs
        self.mu = mu
        self.advantages = advantages
        self.value_targets = value_targets
        self.dones = dones
        self.horizon = obs.shape[0]
        self.num_sequences = obs.shape[1]
        self.source_policies: Optional[torch.Tensor] = None  # [n_seq] follower ids

    def num_transitions(self) -> int:
        return self.horizon * self.num_sequences

    def chunks(self, num_chunks: int, generator: Optional[torch.Generator] = None) -> List[Dict[str, torch.Tensor]]:
        order = torch.randperm(self.num_sequences, generator=generator)
        pieces = torch.chunk(order, num_chunks)
        batches = []
        for piece in pieces:
            batches.append(
                {
                    "obs": self.obs[:, piece],
                    "actions": self.actions[:, piece],
                    "behavior_logprobs": self.behavior_logprobs[:, piece],
                    "old_logprobs": self.old_logprobs[:, piece],
                    "mu": self.mu[:, piece],
                    "advantages": self.advantages[:, piece],
                    "value_targets": self.value_targets[:, piece],
                    "dones": self.dones[:, piece],
                    "policy_ids": (
                        self.source_policies[piece] if self.source_policies is not None else None
                    ),
                }
            )
        return batches


class RolloutStorage:
    """Collection of per-policy :class:`PolicyBuffer` objects."""

    def __init__(
        self,
        num_policies: int,
        num_envs_per_policy: int,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        recurrent: bool = False,
    ) -> None:
        self.num_policies = num_policies
        self.num_envs_per_policy = num_envs_per_policy
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.recurrent = recurrent
        self.buffers: List[PolicyBuffer] = [
            PolicyBuffer(horizon, num_envs_per_policy, obs_dim, action_dim, device, policy_id=i)
            for i in range(num_policies)
        ]

    # ------------------------------------------------------------------ #
    def buffer(self, policy_id: int) -> PolicyBuffer:
        return self.buffers[policy_id]

    def policy_ids(self, policy_id: int, batch_size: int) -> torch.Tensor:
        return torch.full((batch_size,), policy_id, dtype=torch.long, device=self.device)

    def compute_returns(
        self,
        last_values: Dict[int, torch.Tensor],
        gamma: float,
        gae_lambda: float,
        nstep: int = 3,
        critic_target: str = "nstep",
    ) -> None:
        for policy_id, buffer in enumerate(self.buffers):
            buffer.compute_returns(
                last_values[policy_id],
                gamma=gamma,
                gae_lambda=gae_lambda,
                nstep=nstep,
                critic_target=critic_target,
            )

    # ------------------------------------------------------------------ #
    # minibatching
    # ------------------------------------------------------------------ #
    def sequence_chunks(
        self,
        policy_id: int,
        num_chunks: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Return ``num_chunks`` minibatches of whole sequences."""
        buffer = self.buffers[policy_id]
        order = torch.randperm(buffer.num_envs, generator=generator)
        pieces = torch.chunk(order, num_chunks)
        batches = []
        for piece in pieces:
            batches.append(
                {
                    "obs": buffer.obs[:, piece],
                    "actions": buffer.actions[:, piece],
                    "old_logprobs": buffer.logprobs[:, piece],
                    "values": buffer.values[:, piece],
                    "advantages": buffer.advantages[:, piece],
                    "value_targets": buffer.value_targets[:, piece],
                    "dones": buffer.dones[:, piece],
                    "policy_ids": self.policy_ids(policy_id, piece.numel()),
                }
            )
        return batches

    def flat_chunks(
        self,
        policy_id: int,
        minibatch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = self.buffers[policy_id]
        total = buffer.num_transitions()
        perm = torch.randperm(total, generator=generator)
        for start in range(0, total, minibatch_size):
            idx = perm[start : start + minibatch_size]
            yield {
                "obs": buffer.obs.reshape(-1, buffer.obs.shape[-1])[idx],
                "actions": buffer.actions.reshape(-1, buffer.actions.shape[-1])[idx],
                "old_logprobs": buffer.logprobs.reshape(-1)[idx],
                "values": buffer.values.reshape(-1)[idx],
                "advantages": buffer.advantages.reshape(-1)[idx],
                "value_targets": buffer.value_targets.reshape(-1)[idx],
                "policy_ids": self.policy_ids(policy_id, idx.numel()),
            }

    def on_policy_chunks(
        self,
        policy_id: int,
        num_minibatches: int,
        minibatch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        if self.recurrent:
            return self.sequence_chunks(policy_id, num_minibatches, generator)
        return list(self.flat_chunks(policy_id, minibatch_size, generator))
