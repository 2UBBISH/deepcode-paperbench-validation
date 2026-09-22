"""Advantage estimation for SAPG.

This module implements the advantage / return estimators used by SAPG and the
PPO baseline:

* Generalized Advantage Estimation (GAE) with ``tau = 0.95`` (Schulman et al.,
  2016).  Used for the on-policy PPO update of every policy.
* n-step bootstrapped value targets with ``n = 3`` (Eq. 5 in the paper).  Used
  as the critic regression target for the on-policy data.
* 1-step off-policy critic targets (Eq. 6 in the paper).  Used for the
  importance-sampled off-policy aggregation performed by the leader.

All functions operate on time-major tensors of shape ``(T, B)`` where ``T`` is
the rollout horizon and ``B`` is the number of parallel environments handled by
a single policy block.  ``dones`` is expected to be a float tensor (1.0 marks a
terminal transition) so that bootstrapping is masked correctly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "compute_gae",
    "compute_n_step_returns",
    "compute_off_policy_targets",
    "compute_advantages",
]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    tau: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Args:
        rewards: Reward tensor of shape ``(T, B)``.
        values: Value estimates ``V(s_t)`` of shape ``(T, B)``.
        dones: Float tensor of shape ``(T, B)``; ``1.0`` marks a terminal
            transition (episode boundary).
        next_value: Bootstrap value ``V(s_{T})`` of shape ``(B,)`` for the state
            following the last stored transition.  Defaults to zeros.
        gamma: Discount factor (paper: ``0.99``).
        tau: GAE lambda parameter (paper: ``0.95``).

    Returns:
        ``(advantages, returns)`` both of shape ``(T, B)``.  ``returns`` are the
        GAE(lambda) value targets ``advantages + values``.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be (T, B), got shape {tuple(rewards.shape)}")
    T, B = rewards.shape

    if next_value is None:
        next_value = torch.zeros(B, dtype=values.dtype, device=values.device)
    else:
        next_value = next_value.reshape(B)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * tau * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_n_step_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    n: int = 3,
) -> torch.Tensor:
    """n-step bootstrapped value targets (Eq. 5).

    ``V_on_target(s_t) = sum_{k=t}^{t+n-1} gamma^{k-t} r_k + gamma^n V_old(s_{t+n})``

    The bootstrap term is masked whenever an episode terminates inside the
    n-step window, in which case the accumulated (discounted) reward up to the
    terminal transition is used instead.

    Args:
        rewards: Reward tensor of shape ``(T, B)``.
        values: Value estimates ``V_old(s_t)`` of shape ``(T, B)``.
        dones: Float tensor of shape ``(T, B)``; ``1.0`` marks a terminal
            transition.
        gamma: Discount factor (paper: ``0.99``).
        n: Number of steps (paper: ``3``).

    Returns:
        Value targets of shape ``(T, B)``.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be (T, B), got shape {tuple(rewards.shape)}")
    T, B = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    targets = torch.zeros_like(rewards)

    for t in range(T):
        acc = torch.zeros(B, dtype=dtype, device=device)
        discount = 1.0
        bootstrapped = False

        for k in range(n):
            idx = t + k
            if idx >= T:
                # Ran past the end of the rollout: bootstrap from the last
                # available value estimate.
                acc = acc + discount * values[T - 1]
                bootstrapped = True
                break

            acc = acc + discount * rewards[idx]
            discount = discount * gamma

            if dones[idx] > 0.5:
                # Episode terminated inside the window -> no bootstrap.
                bootstrapped = True
                break

        if not bootstrapped:
            idx = t + n
            if idx < T:
                acc = acc + discount * values[idx]
            else:
                acc = acc + discount * values[T - 1]

        targets[t] = acc

    return targets


def compute_off_policy_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """1-step off-policy critic targets (Eq. 6).

    ``V_off_target(s'_t) = r_t + gamma * V_old(s'_{t+1})``

    Args:
        rewards: Reward tensor of shape ``(T, B)`` (or ``(B,)``).
        next_values: Value estimates ``V_old(s'_{t+1})`` of matching shape.
        dones: Float tensor of matching shape; ``1.0`` marks a terminal
            transition (bootstrap masked).
        gamma: Discount factor (paper: ``0.99``).

    Returns:
        Value targets with the same shape as ``rewards``.
    """
    next_values = next_values.reshape_as(rewards)
    dones = dones.reshape_as(rewards)
    return rewards + gamma * next_values * (1.0 - dones)


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    tau: float = 0.95,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper around :func:`compute_gae` with optional normalization.

    Args:
        rewards: Reward tensor of shape ``(T, B)``.
        values: Value estimates of shape ``(T, B)``.
        dones: Float tensor of shape ``(T, B)``.
        next_value: Bootstrap value of shape ``(B,)``.
        gamma: Discount factor.
        tau: GAE lambda.
        normalize: If ``True``, standardize advantages across the whole batch.
        eps: Numerical stability constant for normalization.

    Returns:
        ``(advantages, returns)``.
    """
    advantages, returns = compute_gae(
        rewards, values, dones, next_value=next_value, gamma=gamma, tau=tau
    )
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
    return advantages, returns
