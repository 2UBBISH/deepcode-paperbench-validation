"""On-policy and off-policy PPO losses used by SAPG (Sec. 4.1).

On-policy (Eq. 2 of the paper)::

    L_on(pi_theta) = E_{pi_old}[ min( r_t, clip(r_t, 1-eps, 1+eps) ) A_t^{pi_old} ]
    r_t = pi_theta(a|s) / pi_old(a|s)

Off-policy (Eq. 3 of the paper, importance-sampled PPO of Meng et al. 2023)::

    L_off(pi_i; X) = 1/|X| sum_{j in X} E_{(s,a)~pi_j}[
        min( r_{pi_i}(s,a), clip(r_{pi_i}(s,a), mu(1-eps), mu(1+eps)) ) A^{pi_{i,old}}(s,a) ]

    r_{pi_i}(s,a) = pi_i(s,a) / pi_j(s,a)
    mu            = pi_{i,old}(s,a) / pi_j(s,a)

and the total objective of policy ``i`` is ``L(pi_i) = L_on(pi_i) + lambda *
L_off(pi_i; X)`` with ``lambda = 1`` (Sec. 4.3).

Note that ``pi_j`` (the behaviour policy that produced the data) is a constant
here: gradients only flow through ``pi_i``.  Only the numerator of ``r_{pi_i}``
and the critic of policy ``i`` are differentiated.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .actor_critic import ActorCritic


def _bounds_loss(mu: torch.Tensor, bound: Optional[float]) -> torch.Tensor:
    if bound is None:
        return torch.zeros((), device=mu.device)
    return torch.mean(torch.relu(torch.abs(mu) - bound) ** 2)


def compute_on_policy_loss(
    model: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    value_targets: torch.Tensor,
    policy_ids: Optional[torch.Tensor],
    clip_epsilon: float,
    critic_coef: float = 4.0,
    entropy_coef: float = 0.0,
    bounds_loss_coef: float = 0.0,
    action_bound: Optional[float] = None,
    dones: Optional[torch.Tensor] = None,
    normalise_advantages: bool = False,
) -> Dict[str, torch.Tensor]:
    """PPO clipped surrogate + value regression for one minibatch."""
    log_prob, entropy, value, _, action_mean = model.evaluate_actions(
        obs, actions, policy_ids=policy_ids, dones=dones
    )
    if normalise_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = torch.exp(log_prob - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = ((value - value_targets) ** 2).mean()
    entropy_bonus = entropy.mean()

    b_loss = _bounds_loss(action_mean, action_bound if bounds_loss_coef > 0 else None)

    with torch.no_grad():
        approx_kl = (old_logprobs - log_prob).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
        # standard deviation of the sampled log-probs (for logging)
        std_ratio = ratio.std()

    total = policy_loss + critic_coef * value_loss + bounds_loss_coef * b_loss
    if entropy_coef != 0.0:
        total = total - entropy_coef * entropy_bonus

    return {
        "loss": total,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy_bonus.detach(),
        "bounds_loss": b_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "ratio_std": std_ratio.detach(),
        "ratio_mean": ratio.detach().mean(),
    }


def compute_off_policy_loss(
    model: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    mu_weights: torch.Tensor,
    advantages: torch.Tensor,
    value_targets: torch.Tensor,
    policy_ids: Optional[torch.Tensor],
    clip_epsilon: float,
    old_logprobs: Optional[torch.Tensor] = None,
    critic_coef: float = 4.0,
    bounds_loss_coef: float = 0.0,
    action_bound: Optional[float] = None,
    dones: Optional[torch.Tensor] = None,
    normalise_advantages: bool = False,
) -> Dict[str, torch.Tensor]:
    """Importance-sampled (off-policy) PPO loss for policy ``i``.

    ``behavior_logprobs`` holds ``log pi_j(a|s)`` (the data-collecting
    follower), ``mu_weights`` holds ``mu = pi_{i,old}(s,a) / pi_j(s,a)``
    computed once at the beginning of the update.
    """
    log_prob, entropy, value, _, action_mean = model.evaluate_actions(
        obs, actions, policy_ids=policy_ids, dones=dones
    )
    if normalise_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = torch.exp(log_prob - behavior_logprobs)          # r_{pi_i} = pi_i / pi_j
    lower = mu_weights * (1.0 - clip_epsilon)
    upper = mu_weights * (1.0 + clip_epsilon)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, lower, upper) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = ((value - value_targets) ** 2).mean()

    b_loss = _bounds_loss(action_mean, action_bound if bounds_loss_coef > 0 else None)

    with torch.no_grad():
        # KL(pi_{i,old} || pi_i) estimate, exactly as in the on-policy loss
        approx_kl = (
            (old_logprobs - log_prob).mean() if old_logprobs is not None else torch.zeros((), device=log_prob.device)
        )
        # effective sample size of the importance weights (diagnostic)
        ess = (mu_weights.sum() ** 2) / (mu_weights.pow(2).sum() + 1e-8)

    total = policy_loss + critic_coef * value_loss + bounds_loss_coef * b_loss
    return {
        "loss": total,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.mean().detach(),
        "bounds_loss": b_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "ess": ess.detach(),
        "ratio_mean": ratio.detach().mean(),
        "mu_mean": mu_weights.detach().mean(),
    }


class OnPolicyLoss:
    """Callable wrapper mirroring the paper's ``ONPOLICYLOSS`` operator."""

    def __init__(self, cfg) -> None:
        self.clip_epsilon = float(cfg.get("clip_epsilon", 0.1))
        self.critic_coef = float(cfg.get("critic_coef", 4.0))
        self.bounds_loss_coef = float(cfg.get("bounds_loss_coef", 0.0001))
        self.action_bound = cfg.get("action_bound", None)
        self.normalise_advantages = bool(cfg.get("normalise_advantages", False))

    def __call__(self, model, batch, entropy_coef: float = 0.0) -> Dict[str, torch.Tensor]:
        return compute_on_policy_loss(
            model,
            obs=batch["obs"],
            actions=batch["actions"],
            old_logprobs=batch["old_logprobs"],
            advantages=batch["advantages"],
            value_targets=batch["value_targets"],
            policy_ids=batch.get("policy_ids"),
            clip_epsilon=self.clip_epsilon,
            critic_coef=self.critic_coef,
            entropy_coef=entropy_coef,
            bounds_loss_coef=self.bounds_loss_coef,
            action_bound=self.action_bound,
            dones=batch.get("dones"),
            normalise_advantages=self.normalise_advantages,
        )


class OffPolicyLoss:
    """Callable wrapper mirroring the paper's ``OFFPOLICYLOSS`` operator."""

    def __init__(self, cfg) -> None:
        self.clip_epsilon = float(cfg.get("clip_epsilon", 0.1))
        self.critic_coef = float(cfg.get("critic_coef", 4.0))
        self.bounds_loss_coef = float(cfg.get("bounds_loss_coef", 0.0001))
        self.action_bound = cfg.get("action_bound", None)
        self.normalise_advantages = bool(cfg.get("normalise_advantages", False))

    def __call__(self, model, batch) -> Dict[str, torch.Tensor]:
        return compute_off_policy_loss(
            model,
            obs=batch["obs"],
            actions=batch["actions"],
            behavior_logprobs=batch["behavior_logprobs"],
            mu_weights=batch["mu"],
            advantages=batch["advantages"],
            value_targets=batch["value_targets"],
            policy_ids=batch.get("policy_ids"),
            clip_epsilon=self.clip_epsilon,
            old_logprobs=batch.get("old_logprobs"),
            critic_coef=self.critic_coef,
            bounds_loss_coef=self.bounds_loss_coef,
            action_bound=self.action_bound,
            dones=batch.get("dones"),
            normalise_advantages=self.normalise_advantages,
        )
