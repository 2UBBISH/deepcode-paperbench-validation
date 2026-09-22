"""Loss functions for SAPG (Split and Aggregate Policy Gradients).

This module implements the on-policy and off-policy PPO-style losses described
in the paper:

    * On-policy policy loss (Eq. 2):
        L_on(pi_i) = E_{(s,a) ~ pi_i,old}[
            min(r_t, clip(r_t, 1-eps, 1+eps)) * A_t ]
        with r_t = pi_i(a|s) / pi_i,old(a|s).

    * Off-policy policy loss (Eq. 3):
        L_off(pi_i; X) = (1/|X|) sum_{j in X} E_{(s,a) ~ pi_j}[
            min(r_pi_i, clip(r_pi_i, mu(1-eps), mu(1+eps))) * A^{pi_i,old} ]
        with r_pi_i(s,a) = pi_i(s,a) / pi_j(s,a) and
             mu = pi_i,old(s,a) / pi_j(s,a).

    * Combined objective (Eq. 4):
        L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X)

    * On-policy critic target (n-step, Eq. 5):
        V_on_target(s_t) = sum_{k=t}^{t+n-1} gamma^{k-t} r_k
                           + gamma^n V_old(s_{t+n})

    * Off-policy critic target (1-step, Eq. 6):
        V_off_target(s'_t) = r_t + gamma V_old(s'_{t+1})

    * Critic losses (Eq. 7-9):
        L_on_critic  = E[(V(s) - V_on_target(s))^2]
        L_off_critic = (1/|X|) sum_j E[(V(s) - V_off_target(s))^2]

All functions operate on flattened (batch) tensors.  Sequence handling (LSTM)
is performed by the caller, which passes the appropriate ``lstm_state``.

Gradient routing note
---------------------
The off-policy loss for policy ``i`` is computed against data collected by
policy ``j``.  The importance ratio ``pi_i / pi_j`` must be differentiable with
respect to ``theta`` (the shared actor backbone) and ``phi_i`` (the target
policy's latent), but NOT with respect to ``phi_j`` (the behaviour policy's
latent).  Callers should therefore pass ``phi_i`` obtained via
``get_phi(i)`` and ``phi_j`` obtained via ``get_phi_detached(j)``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reduce_loss(loss: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Apply the requested reduction to a per-sample loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction}")


def _safe_ratio(numer: torch.Tensor, denom: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Numerically stable ratio ``numer / denom``."""
    return numer / (denom + eps)


# ---------------------------------------------------------------------------
# On-policy PPO policy loss (Eq. 2)
# ---------------------------------------------------------------------------
def on_policy_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.1,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Clipped surrogate on-policy policy loss (Eq. 2).

    Args:
        log_probs: ``log pi_i(a|s)`` under the current parameters.
        old_log_probs: ``log pi_i,old(a|s)`` under the behaviour parameters.
        advantages: advantage estimates ``A_t`` (typically GAE).
        clip_eps: PPO clipping parameter ``eps``.
        reduction: one of ``"mean"``, ``"sum"``, ``"none"``.

    Returns:
        ``(loss, info)`` where ``loss`` is the (reduced) surrogate loss to be
        *maximised* (the caller negates it for gradient descent) and ``info``
        contains diagnostic tensors (``ratio``, ``clip_fraction``, ``kl``).
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = torch.min(surr1, surr2)

    loss = _reduce_loss(surrogate, reduction)

    with torch.no_grad():
        clip_fraction = (torch.abs(ratio - 1.0) > clip_eps).float().mean()
        approx_kl = (old_log_probs - log_probs).mean()

    info = {
        "ratio": ratio.detach(),
        "clip_fraction": clip_fraction.detach(),
        "approx_kl": approx_kl.detach(),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Off-policy importance-sampled policy loss (Eq. 3)
# ---------------------------------------------------------------------------
def off_policy_policy_loss(
    log_probs_i: torch.Tensor,
    log_probs_j: torch.Tensor,
    old_log_probs_i: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.1,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Importance-sampled off-policy policy loss (Eq. 3).

    Computes, for a batch of transitions collected by behaviour policy ``j``:

        r_pi_i(s,a) = pi_i(a|s) / pi_j(a|s)
        mu          = pi_i,old(a|s) / pi_j(a|s)
        L_off       = min(r_pi_i, clip(r_pi_i, mu(1-eps), mu(1+eps))) * A

    Args:
        log_probs_i: ``log pi_i(a|s)`` under current parameters (differentiable
            w.r.t. ``theta`` and ``phi_i``).
        log_probs_j: ``log pi_j(a|s)`` under the behaviour policy (must be
            detached w.r.t. ``phi_j`` by the caller).
        old_log_probs_i: ``log pi_i,old(a|s)`` (the target policy's parameters
            before the update; used to build the ``mu`` correction).
        advantages: advantage estimates ``A^{pi_i,old}``.
        clip_eps: PPO clipping parameter ``eps``.
        reduction: one of ``"mean"``, ``"sum"``, ``"none"``.

    Returns:
        ``(loss, info)``.  ``loss`` is the surrogate to be *maximised*.
    """
    # r_pi_i = pi_i / pi_j  (differentiable w.r.t. theta, phi_i)
    log_ratio_i_j = log_probs_i - log_probs_j
    ratio_i = torch.exp(log_ratio_i_j)

    # mu = pi_i,old / pi_j  (no gradient: both terms are old/behaviour)
    with torch.no_grad():
        mu = torch.exp(old_log_probs_i - log_probs_j)

    lower = mu * (1.0 - clip_eps)
    upper = mu * (1.0 + clip_eps)
    surr1 = ratio_i * advantages
    surr2 = torch.clamp(ratio_i, lower, upper) * advantages
    surrogate = torch.min(surr1, surr2)

    loss = _reduce_loss(surrogate, reduction)

    with torch.no_grad():
        # Fraction of samples where the ratio was clipped relative to mu.
        clipped = (ratio_i < lower) | (ratio_i > upper)
        clip_fraction = clipped.float().mean()
        approx_kl = (old_log_probs_i - log_probs_i).mean()

    info = {
        "ratio": ratio_i.detach(),
        "mu": mu.detach(),
        "clip_fraction": clip_fraction.detach(),
        "approx_kl": approx_kl.detach(),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Critic targets
# ---------------------------------------------------------------------------
def compute_n_step_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    n: int = 3,
) -> torch.Tensor:
    """Compute n-step bootstrapped value targets (Eq. 5).

    ``V_on_target(s_t) = sum_{k=t}^{t+n-1} gamma^{k-t} r_k + gamma^n V_old(s_{t+n})``

    Args:
        rewards: ``(T, B)`` rewards for a rollout of length ``T``.
        values: ``(T, B)`` value estimates ``V_old(s_t)``.
        dones: ``(T, B)`` episode-termination flags (1.0 if ``s_{t+1}`` is
            terminal, i.e. bootstrap should be masked).
        gamma: discount factor.
        n: number of steps ``n``.

    Returns:
        ``(T, B)`` tensor of n-step targets.
    """
    T, B = rewards.shape
    device = rewards.device
    targets = torch.zeros_like(rewards)

    # Precompute discounted reward accumulation for each step.
    for t in range(T):
        acc = torch.zeros(B, device=device)
        discount = 1.0
        boot_idx = min(t + n, T - 1)
        # Sum discounted rewards r_t .. r_{t+n-1}
        for k in range(t, min(t + n, T)):
            acc = acc + discount * rewards[k]
            # If an episode ended at step k, stop accumulating and do not
            # bootstrap (the terminal value is 0).
            if dones[k].any():
                # Zero-out contributions for terminated trajectories.
                acc = torch.where(dones[k] > 0.5, acc, acc)
            discount = discount * gamma
        # Bootstrap with V_old(s_{t+n}) unless the trajectory terminated.
        if t + n < T:
            bootstrap = values[t + n]
            # Mask bootstrap where any done occurred within the n-step window.
            window_done = torch.zeros(B, device=device)
            for k in range(t, min(t + n, T)):
                window_done = torch.clamp(window_done + dones[k], max=1.0)
            bootstrap = bootstrap * (1.0 - window_done)
            acc = acc + (gamma ** n) * bootstrap
        targets[t] = acc

    return targets


def compute_off_policy_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute 1-step off-policy critic targets (Eq. 6).

    ``V_off_target(s'_t) = r_t + gamma V_old(s'_{t+1})``

    Args:
        rewards: ``(B,)`` rewards from the off-policy transitions.
        next_values: ``(B,)`` ``V_old(s'_{t+1})``.
        dones: ``(B,)`` termination flags.
        gamma: discount factor.

    Returns:
        ``(B,)`` off-policy targets.
    """
    return rewards + gamma * next_values * (1.0 - dones)


# ---------------------------------------------------------------------------
# Critic losses (Eq. 7-9)
# ---------------------------------------------------------------------------
def critic_loss(
    values: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared-error critic loss (Eq. 7 / Eq. 8).

    Args:
        values: predicted values ``V(s)``.
        targets: bootstrapped targets (on- or off-policy).
        reduction: one of ``"mean"``, ``"sum"``, ``"none"``.

    Returns:
        The (reduced) squared-error loss.
    """
    loss = (values - targets.detach()) ** 2
    return _reduce_loss(loss, reduction)


# ---------------------------------------------------------------------------
# Entropy bonus (Sec. 4.5)
# ---------------------------------------------------------------------------
def entropy_bonus(
    dist_entropy: torch.Tensor,
    policy_index: int,
    sigma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Follower entropy regularisation term ``sigma * (i-1) * H(pi)``.

    The leader (``policy_index == 1``) receives no entropy bonus.  Followers
    receive a bonus that grows with their index ``i`` (1-based).

    Args:
        dist_entropy: per-sample entropy ``H(pi(a|s))``.
        policy_index: 1-based policy index ``i``.
        sigma: base entropy coefficient ``sigma``.
        reduction: reduction mode.

    Returns:
        The (reduced) entropy bonus to be *added* to the objective (i.e. the
        caller adds ``+ entropy_bonus`` to the loss-to-maximise).
    """
    if policy_index <= 1 or sigma == 0.0:
        return torch.zeros((), device=dist_entropy.device,
                           dtype=dist_entropy.dtype)
    coef = sigma * (policy_index - 1)
    return coef * _reduce_loss(dist_entropy, reduction)


# ---------------------------------------------------------------------------
# Combined SAPG objective (Eq. 4)
# ---------------------------------------------------------------------------
def sapg_policy_loss(
    on_policy: Dict[str, torch.Tensor],
    off_policy: Optional[Dict[str, torch.Tensor]] = None,
    lambda_off: float = 1.0,
    entropy: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combined SAPG policy objective (Eq. 4).

    ``L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X) + entropy_bonus``

    Args:
        on_policy: dict with keys ``log_probs``, ``old_log_probs``,
            ``advantages`` (and optionally ``clip_eps``).
        off_policy: optional dict with keys ``log_probs_i``, ``log_probs_j``,
            ``old_log_probs_i``, ``advantages`` (and optionally ``clip_eps``).
        lambda_off: off-policy loss weight ``lambda``.
        entropy: optional pre-computed entropy bonus (already scaled).

    Returns:
        ``(total_loss, info)`` where ``total_loss`` is the objective to be
        *maximised*.
    """
    clip_eps = on_policy.get("clip_eps", 0.1)
    loss_on, info_on = on_policy_policy_loss(
        log_probs=on_policy["log_probs"],
        old_log_probs=on_policy["old_log_probs"],
        advantages=on_policy["advantages"],
        clip_eps=clip_eps,
    )

    total = loss_on
    info = {"loss_on": loss_on.detach()}
    for k, v in info_on.items():
        info[f"on/{k}"] = v

    if off_policy is not None:
        off_clip_eps = off_policy.get("clip_eps", clip_eps)
        loss_off, info_off = off_policy_policy_loss(
            log_probs_i=off_policy["log_probs_i"],
            log_probs_j=off_policy["log_probs_j"],
            old_log_probs_i=off_policy["old_log_probs_i"],
            advantages=off_policy["advantages"],
            clip_eps=off_clip_eps,
        )
        total = total + lambda_off * loss_off
        info["loss_off"] = loss_off.detach()
        for k, v in info_off.items():
            info[f"off/{k}"] = v

    if entropy is not None:
        total = total + entropy
        info["entropy"] = entropy.detach()

    info["loss_total"] = total.detach()
    return total, info


def sapg_critic_loss(
    on_policy_values: torch.Tensor,
    on_policy_targets: torch.Tensor,
    off_policy_values: Optional[torch.Tensor] = None,
    off_policy_targets: Optional[torch.Tensor] = None,
    lambda_off: float = 1.0,
    critic_coef: float = 4.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combined SAPG critic loss (Eq. 7-9).

    ``L_critic = critic_coef * (L_on_critic + lambda * L_off_critic)``

    Args:
        on_policy_values: ``V(s)`` for on-policy transitions.
        on_policy_targets: n-step targets for on-policy transitions.
        off_policy_values: optional ``V(s')`` for off-policy transitions.
        off_policy_targets: optional 1-step targets for off-policy transitions.
        lambda_off: off-policy weight ``lambda``.
        critic_coef: critic loss coefficient ``lambda'`` (default 4.0).

    Returns:
        ``(total_loss, info)`` where ``total_loss`` is minimised.
    """
    loss_on = critic_loss(on_policy_values, on_policy_targets)
    total = loss_on
    info = {"critic_on": loss_on.detach()}

    if off_policy_values is not None and off_policy_targets is not None:
        loss_off = critic_loss(off_policy_values, off_policy_targets)
        total = total + lambda_off * loss_off
        info["critic_off"] = loss_off.detach()

    total = critic_coef * total
    info["critic_total"] = total.detach()
    return total, info


# ---------------------------------------------------------------------------
# Bounds loss (regularisation on the action distribution bounds)
# ---------------------------------------------------------------------------
def bounds_loss(mean: torch.Tensor, coef: float = 1e-4) -> torch.Tensor:
    """L2 regularisation on the Gaussian mean (bounds loss).

    Penalises large action means to keep the policy within reasonable bounds.

    Args:
        mean: Gaussian mean ``mu(s)``.
        coef: coefficient (default ``1e-4``).

    Returns:
        Scalar loss.
    """
    return coef * (mean ** 2).mean()


__all__ = [
    "on_policy_policy_loss",
    "off_policy_policy_loss",
    "compute_n_step_returns",
    "compute_off_policy_targets",
    "critic_loss",
    "entropy_bonus",
    "sapg_policy_loss",
    "sapg_critic_loss",
    "bounds_loss",
]
