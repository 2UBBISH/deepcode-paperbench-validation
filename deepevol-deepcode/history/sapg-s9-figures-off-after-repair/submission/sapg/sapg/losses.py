"""Loss functions for SAPG (Split and Aggregate Policy Gradients).

This module implements the policy and critic losses described in the paper:

    On-policy PPO loss (Eq. 2):
        L_on(pi_theta) = E_{pi_old}[ min( r_t, clip(r_t, 1-eps, 1+eps) ) * A_t^{pi_old} ]
        r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)

    Off-policy importance-sampled loss (Eq. 3):
        L_off(pi_i; X) = (1/|X|) sum_{j in X} E_{(s,a)~pi_j}[
            min( r_{pi_i}, clip(r_{pi_i}, mu(1-eps), mu(1+eps)) ) * A^{pi_{i,old}} ]
        r_{pi_i}(s,a) = pi_i(s,a) / pi_j(s,a)      (importance ratio)
        mu            = pi_{i,old}(s,a) / pi_j(s,a) (off-policy correction term)

    Combined policy loss (Eq. 4):
        L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X)

    Critic losses (Eq. 7-9):
        L_on^critic  = E[ (V_on^target(s_t)  - V_theta(s_t))^2 ]
        L_off^critic = E[ (V_off^target(s_t') - V_theta(s_t'))^2 ]
        L^critic     = L_on^critic + lambda * L_off^critic

All losses are returned as *negatives* of the objectives above so that they can
be minimized with gradient descent (the paper writes them as maximization
objectives).

Consistency property (verified in tests): when i == j, pi_j == pi_{i,old}, so
mu == 1 and r_{pi_i} == r_t, and Eq. 3 reduces exactly to Eq. 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class PolicyLossOutput:
    """Container for the outputs of a policy loss computation."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor


@dataclass
class CriticLossOutput:
    """Container for the outputs of a critic loss computation."""

    loss: torch.Tensor
    value_loss: torch.Tensor
    value_clip_fraction: torch.Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_ratio(numer_log_prob: torch.Tensor, denom_log_prob: torch.Tensor) -> torch.Tensor:
    """Compute an importance ratio exp(log pi - log pi_old) in a numerically
    stable way (subtract the max before exponentiating is unnecessary here
    because the difference is small in practice, but we clamp the exponent).
    """
    log_ratio = numer_log_prob - denom_log_prob
    # Clamp to avoid inf/nan from pathological ratios.
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    return torch.exp(log_ratio)


def _clip_fraction(ratio: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Fraction of ratios that were clipped (outside [low, high])."""
    return ((ratio < low) | (ratio > high)).float().mean()


# ---------------------------------------------------------------------------
# On-policy PPO loss (Eq. 2)
# ---------------------------------------------------------------------------
def on_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.1,
    normalize_advantage: bool = True,
    advantage_eps: float = 1e-8,
) -> PolicyLossOutput:
    """Compute the on-policy clipped PPO surrogate loss (Eq. 2).

    Args:
        new_log_probs: log pi_theta(a_t | s_t), shape [B].
        old_log_probs: log pi_old(a_t | s_t), shape [B].
        advantages:   A_t^{pi_old}, shape [B].
        clip_eps:     PPO clipping epsilon.
        normalize_advantage: whether to standardize advantages over the batch.

    Returns:
        PolicyLossOutput where ``loss`` is the *negative* surrogate (to minimize).
    """
    if normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + advantage_eps)

    ratio = _safe_ratio(new_log_probs, old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = torch.min(surr1, surr2)

    # Approximate KL (Schulman's k3 estimator): E[ratio - 1 - log(ratio)]
    approx_kl = (ratio - 1.0 - (new_log_probs - old_log_probs)).mean()
    clip_frac = _clip_fraction(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    policy_loss = -surrogate.mean()
    return PolicyLossOutput(
        loss=policy_loss,
        policy_loss=policy_loss,
        entropy=torch.zeros((), device=new_log_probs.device),
        approx_kl=approx_kl.detach(),
        clip_fraction=clip_frac.detach(),
        ratio_mean=ratio.mean().detach(),
    )


# ---------------------------------------------------------------------------
# Off-policy importance-sampled loss (Eq. 3)
# ---------------------------------------------------------------------------
def off_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.1,
    normalize_advantage: bool = True,
    advantage_eps: float = 1e-8,
) -> PolicyLossOutput:
    """Compute the off-policy importance-sampled loss (Eq. 3).

    The data (s, a) was collected by a *behavior* policy pi_j (j != i). We want
    to update the target policy pi_i. Following the paper:

        r_{pi_i}(s,a) = pi_i(s,a) / pi_j(s,a)          (importance ratio)
        mu            = pi_{i,old}(s,a) / pi_j(s,a)     (off-policy correction)
        L_off = min( r_{pi_i}, clip(r_{pi_i}, mu(1-eps), mu(1+eps)) ) * A^{pi_{i,old}}

    Args:
        new_log_probs:      log pi_i(a|s)      (current target policy).
        old_log_probs:      log pi_{i,old}(a|s) (target policy at rollout time).
        behavior_log_probs: log pi_j(a|s)      (behavior policy that collected data).
        advantages:         A^{pi_{i,old}}.
        clip_eps:           PPO clipping epsilon.

    Returns:
        PolicyLossOutput where ``loss`` is the *negative* objective (to minimize).

    Note:
        When i == j, behavior_log_probs == old_log_probs, so mu == 1 and
        r_{pi_i} == ratio, and this reduces exactly to :func:`on_policy_loss`.
    """
    if normalize_advantage and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + advantage_eps)

    # Importance ratio r_{pi_i} = pi_i / pi_j
    ratio = _safe_ratio(new_log_probs, behavior_log_probs)
    # Off-policy correction term mu = pi_{i,old} / pi_j
    mu = _safe_ratio(old_log_probs, behavior_log_probs)

    low = mu * (1.0 - clip_eps)
    high = mu * (1.0 + clip_eps)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, min=low, max=high) * advantages
    surrogate = torch.min(surr1, surr2)

    # Approximate KL between pi_i and pi_{i,old} (the on-policy KL we care about).
    approx_kl = (ratio - 1.0 - (new_log_probs - behavior_log_probs)).mean()
    clip_frac = _clip_fraction(ratio, low, high)

    policy_loss = -surrogate.mean()
    return PolicyLossOutput(
        loss=policy_loss,
        policy_loss=policy_loss,
        entropy=torch.zeros((), device=new_log_probs.device),
        approx_kl=approx_kl.detach(),
        clip_fraction=clip_frac.detach(),
        ratio_mean=ratio.mean().detach(),
    )


# ---------------------------------------------------------------------------
# Combined policy loss (Eq. 4) with entropy regularization (Sec 4.5)
# ---------------------------------------------------------------------------
def combined_policy_loss(
    on_policy: Optional[PolicyLossOutput] = None,
    off_policy: Optional[PolicyLossOutput] = None,
    entropy: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.0,
    off_policy_coef: float = 1.0,
) -> PolicyLossOutput:
    """Combine on-policy and off-policy policy losses (Eq. 4) plus entropy.

        L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X) + sigma * (i-1) * H

    The entropy term is *added* to the loss (i.e. we minimize -H scaled by the
    coefficient), matching the paper's follower entropy regularization. The
    caller is responsible for scaling ``entropy_coef`` by ``(i-1)`` for
    followers and setting it to 0 for the leader.

    Args:
        on_policy:      output of :func:`on_policy_loss` (or None).
        off_policy:     output of :func:`off_policy_loss` (or None).
        entropy:        mean entropy H(pi(a|s)) of the current policy.
        entropy_coef:   effective entropy coefficient (already scaled by i-1).
        off_policy_coef: lambda in Eq. 4.

    Returns:
        PolicyLossOutput with the combined loss.
    """
    device = None
    total = None
    policy_loss = None
    approx_kl = None
    clip_fraction = None
    ratio_mean = None

    if on_policy is not None:
        total = on_policy.loss
        policy_loss = on_policy.policy_loss
        approx_kl = on_policy.approx_kl
        clip_fraction = on_policy.clip_fraction
        ratio_mean = on_policy.ratio_mean
        device = on_policy.loss.device

    if off_policy is not None:
        term = off_policy_coef * off_policy.loss
        total = term if total is None else total + term
        policy_loss = off_policy.policy_loss if policy_loss is None else policy_loss + off_policy.policy_loss
        approx_kl = off_policy.approx_kl if approx_kl is None else approx_kl
        clip_fraction = off_policy.clip_fraction if clip_fraction is None else clip_fraction
        ratio_mean = off_policy.ratio_mean if ratio_mean is None else ratio_mean
        device = off_policy.loss.device

    if total is None:
        raise ValueError("At least one of on_policy / off_policy must be provided.")

    entropy_term = torch.zeros((), device=device)
    if entropy is not None and entropy_coef != 0.0:
        entropy_term = entropy
        total = total - entropy_coef * entropy

    return PolicyLossOutput(
        loss=total,
        policy_loss=policy_loss,
        entropy=entropy_term,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
    )


# ---------------------------------------------------------------------------
# Critic losses (Eq. 7-9)
# ---------------------------------------------------------------------------
def critic_loss(
    values: torch.Tensor,
    on_policy_targets: Optional[torch.Tensor] = None,
    off_policy_targets: Optional[torch.Tensor] = None,
    old_values: Optional[torch.Tensor] = None,
    clip_eps: Optional[float] = None,
    off_policy_coef: float = 1.0,
    value_coef: float = 1.0,
) -> CriticLossOutput:
    """Compute the critic loss (Eq. 7-9).

        L_on^critic  = E[ (V_on^target  - V_theta)^2 ]           (Eq. 7)
        L_off^critic = E[ (V_off^target - V_theta)^2 ]           (Eq. 8)
        L^critic     = L_on^critic + lambda * L_off^critic       (Eq. 9)

    Optionally applies value clipping (as in standard PPO) when ``old_values``
    and ``clip_eps`` are provided.

    Args:
        values:            V_theta(s) for the current batch.
        on_policy_targets: V_on^target (Eq. 5) for the on-policy batch.
        off_policy_targets: V_off^target (Eq. 6) for the off-policy batch.
        old_values:        V at rollout time (for value clipping).
        clip_eps:          value clipping epsilon (None disables clipping).
        off_policy_coef:   lambda in Eq. 9.
        value_coef:        overall critic coefficient (lambda' = 4.0 in paper).

    Returns:
        CriticLossOutput where ``loss`` is the (positive) squared-error loss.
    """
    total = None
    clip_frac = torch.zeros((), device=values.device)

    def _value_error(v: torch.Tensor, target: torch.Tensor, old_v: Optional[torch.Tensor]) -> torch.Tensor:
        if clip_eps is not None and old_v is not None:
            v_clipped = old_v + torch.clamp(v - old_v, -clip_eps, clip_eps)
            err = torch.max((v - target) ** 2, (v_clipped - target) ** 2)
            return err.mean()
        return ((v - target) ** 2).mean()

    if on_policy_targets is not None:
        on_err = _value_error(values, on_policy_targets, old_values)
        total = on_err

    if off_policy_targets is not None:
        off_err = _value_error(values, off_policy_targets, old_values)
        term = off_policy_coef * off_err
        total = term if total is None else total + term

    if total is None:
        raise ValueError("At least one of on_policy_targets / off_policy_targets must be provided.")

    total = value_coef * total
    return CriticLossOutput(
        loss=total,
        value_loss=total.detach(),
        value_clip_fraction=clip_frac,
    )


# ---------------------------------------------------------------------------
# Convenience: full SAPG loss for a single policy
# ---------------------------------------------------------------------------
def sapg_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.1,
    behavior_log_probs: Optional[torch.Tensor] = None,
    off_policy_advantages: Optional[torch.Tensor] = None,
    off_policy_old_log_probs: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.0,
    off_policy_coef: float = 1.0,
    normalize_advantage: bool = True,
) -> PolicyLossOutput:
    """Compute the full SAPG policy loss for one policy (Eq. 4).

    This is a convenience wrapper that builds the on-policy term and, when
    off-policy data is supplied, the off-policy term, then combines them.

    Args:
        new_log_probs:          log pi_i(a|s) on the policy's own data D_i.
        old_log_probs:          log pi_{i,old}(a|s) on D_i.
        advantages:             A^{pi_{i,old}} on D_i.
        clip_eps:               PPO clip epsilon.
        behavior_log_probs:     log pi_j(a|s) for off-policy data D_1' (j != i).
        off_policy_advantages:  A^{pi_{i,old}} for off-policy data.
        off_policy_old_log_probs: log pi_{i,old}(a|s) for off-policy data.
        entropy:                mean entropy of pi_i.
        entropy_coef:           effective entropy coefficient (scaled by i-1).
        off_policy_coef:        lambda in Eq. 4.
    """
    on = on_policy_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        clip_eps=clip_eps,
        normalize_advantage=normalize_advantage,
    )

    off = None
    if behavior_log_probs is not None:
        off = off_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=off_policy_old_log_probs if off_policy_old_log_probs is not None else old_log_probs,
            behavior_log_probs=behavior_log_probs,
            advantages=off_policy_advantages if off_policy_advantages is not None else advantages,
            clip_eps=clip_eps,
            normalize_advantage=normalize_advantage,
        )

    return combined_policy_loss(
        on_policy=on,
        off_policy=off,
        entropy=entropy,
        entropy_coef=entropy_coef,
        off_policy_coef=off_policy_coef,
    )


__all__ = [
    "PolicyLossOutput",
    "CriticLossOutput",
    "on_policy_loss",
    "off_policy_loss",
    "combined_policy_loss",
    "critic_loss",
    "sapg_policy_loss",
]
