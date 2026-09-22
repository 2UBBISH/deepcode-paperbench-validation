"""Return / advantage computation for SAPG.

This module implements:

* Generalized Advantage Estimation (GAE) with ``tau`` (lambda) = 0.95, used to
  build the advantages ``A_t^{pi_old}`` consumed by the on-policy PPO surrogate
  (Eq. 2) and the off-policy importance-sampled surrogate (Eq. 3).
* The 3-step on-policy value target (Eq. 5)::

      V_on^target(s_t) = sum_{k=t}^{t+2} gamma^{k-t} r_k + gamma^3 V_{pi_j,old}(s_{t+3})

* The 1-step off-policy value target (Eq. 6)::

      V_off^target(s_t') = r_t + gamma * V_{pi_j,old}(s_{t+1}')

All functions operate on tensors shaped ``[T, B, ...]`` (time-major) or on flat
``[N, ...]`` tensors, depending on the helper.  The training loop in
``algorithm.py`` stores per-block buffers in time-major layout ``[T, B]`` where
``T`` is the rollout horizon and ``B`` is the number of environments in the
block.

Design notes
------------
* ``gamma`` defaults to 0.99 (standard for the tasks in the paper).
* ``gae_lambda`` (``tau`` in the paper) defaults to 0.95.
* The on-policy target uses exactly 3 bootstrapping steps (``on_policy_target_steps``
  in the config), the off-policy target uses exactly 1 step
  (``off_policy_target_steps``).
* Bootstrapping values are the *old* critic values ``V_{pi_j,old}`` evaluated
  under the behaviour policy ``pi_j`` that collected the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class GAEOutput:
    """Container for GAE results.

    Attributes:
        advantages: Advantage estimates ``A_t`` with the same shape as rewards.
        returns: Value targets ``A_t + V(s_t)`` (used for the on-policy critic
            loss when no explicit n-step target is requested).
    """

    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass
class TargetOutput:
    """Container for n-step value targets.

    Attributes:
        targets: The n-step bootstrapped value targets.
        valid: Boolean mask indicating which entries have a valid target
            (entries near the end of the rollout may not have enough future
            steps and are masked out).
    """

    targets: torch.Tensor
    valid: torch.Tensor


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> GAEOutput:
    """Compute Generalized Advantage Estimation (Schulman et al., 2016).

    Args:
        rewards: Rewards ``r_t`` shaped ``[T, B]`` (time-major).
        values: Critic values ``V(s_t)`` shaped ``[T, B]``.
        dones: Episode-termination flags ``d_t`` shaped ``[T, B]`` (1.0 if the
            transition ends an episode, else 0.0).
        next_value: Bootstrap value ``V(s_{T})`` shaped ``[B]``.  If ``None``,
            zeros are used (i.e. the rollout is assumed to end at ``T``).
        gamma: Discount factor.
        gae_lambda: GAE trace decay parameter (``tau`` in the paper).

    Returns:
        :class:`GAEOutput` with ``advantages`` and ``returns`` both shaped
        ``[T, B]``.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be [T, B], got shape {tuple(rewards.shape)}")
    T, B = rewards.shape

    if next_value is None:
        next_value = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
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
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return GAEOutput(advantages=advantages, returns=returns)


# ---------------------------------------------------------------------------
# On-policy n-step targets (Eq. 5)
# ---------------------------------------------------------------------------
def compute_on_policy_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    num_steps: int = 3,
) -> TargetOutput:
    """Compute the on-policy n-step value targets (Eq. 5, ``num_steps=3``).

    ``V_on^target(s_t) = sum_{k=t}^{t+n-1} gamma^{k-t} r_k + gamma^n V(s_{t+n})``

    The bootstrap value ``V(s_{t+n})`` is the *old* critic value under the
    behaviour policy ``pi_j``.  If an episode terminates before ``n`` steps,
    the sum is truncated at the terminal transition and no bootstrap is added.

    Args:
        rewards: Rewards ``r_t`` shaped ``[T, B]``.
        values: Old critic values ``V_{pi_j,old}(s_t)`` shaped ``[T, B]``.
        dones: Termination flags shaped ``[T, B]``.
        next_value: Bootstrap value ``V(s_T)`` shaped ``[B]`` for the final
            step; defaults to zeros.
        gamma: Discount factor.
        num_steps: Number of steps ``n`` (3 for the on-policy target).

    Returns:
        :class:`TargetOutput` with ``targets`` shaped ``[T, B]`` and a boolean
        ``valid`` mask of the same shape.  Entries where the episode terminated
        before ``num_steps`` are still valid (they use the truncated return);
        only entries that would require bootstrapping beyond the buffer are
        marked invalid.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be [T, B], got shape {tuple(rewards.shape)}")
    T, B = rewards.shape

    if next_value is None:
        next_value = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    else:
        next_value = next_value.reshape(B)

    targets = torch.zeros_like(rewards)
    valid = torch.zeros_like(rewards, dtype=torch.bool)

    for t in range(T):
        target = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
        discount = 1.0
        bootstrapped = False
        terminated = False

        for k in range(num_steps):
            idx = t + k
            if idx >= T:
                # Need to bootstrap from the value passed in for the final step.
                if k == 0:
                    # No reward available at all -> invalid.
                    break
                target = target + discount * next_value
                bootstrapped = True
                break

            target = target + discount * rewards[idx]
            if dones[idx] > 0.5:
                # Episode ended; no bootstrap beyond this point.
                terminated = True
                break
            discount = discount * gamma

        if not terminated and not bootstrapped:
            # Add the bootstrap term gamma^n * V(s_{t+n}).
            if t + num_steps < T:
                target = target + discount * values[t + num_steps]
                bootstrapped = True
            else:
                target = target + discount * next_value
                bootstrapped = True

        targets[t] = target
        valid[t] = True

    return TargetOutput(targets=targets, valid=valid)


# ---------------------------------------------------------------------------
# Off-policy 1-step targets (Eq. 6)
# ---------------------------------------------------------------------------
def compute_off_policy_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> TargetOutput:
    """Compute the off-policy 1-step value targets (Eq. 6).

    ``V_off^target(s_t') = r_t + gamma * V_{pi_j,old}(s_{t+1}')``

    Args:
        rewards: Rewards ``r_t`` shaped ``[N]`` or ``[T, B]``.
        next_values: Bootstrap values ``V_{pi_j,old}(s_{t+1}')`` with the same
            shape as ``rewards``.
        dones: Termination flags with the same shape as ``rewards``.
        gamma: Discount factor.

    Returns:
        :class:`TargetOutput` with ``targets`` and ``valid`` matching the input
        shape.
    """
    if rewards.shape != next_values.shape:
        raise ValueError(
            "rewards and next_values must have the same shape, got "
            f"{tuple(rewards.shape)} vs {tuple(next_values.shape)}"
        )
    targets = rewards + gamma * next_values * (1.0 - dones)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    return TargetOutput(targets=targets, valid=valid)


# ---------------------------------------------------------------------------
# Convenience wrappers used by the training loop
# ---------------------------------------------------------------------------
def compute_advantages_and_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    on_policy_target_steps: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute advantages (GAE) and on-policy n-step value targets together.

    Returns:
        Tuple ``(advantages, value_targets, valid_mask)`` each shaped ``[T, B]``.
    """
    gae = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    targets = compute_on_policy_targets(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=gamma,
        num_steps=on_policy_target_steps,
    )
    return gae.advantages, targets.targets, targets.valid


def flatten_time_major(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten a time-major ``[T, B, ...]`` tensor to ``[T*B, ...]``."""
    return tensor.reshape(-1, *tensor.shape[2:])


def unflatten_time_major(tensor: torch.Tensor, T: int, B: int) -> torch.Tensor:
    """Reshape a flat ``[T*B, ...]`` tensor back to time-major ``[T, B, ...]``."""
    return tensor.reshape(T, B, *tensor.shape[1:])
