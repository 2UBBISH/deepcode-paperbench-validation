"""Loss functions for SAPG (Split and Aggregate Policy Gradients).

Implements:
  * COMPONENT 2 -- PPO on-policy clipped surrogate loss (Eq. 2).
  * COMPONENT 3 -- Importance-sampled off-policy loss (Eq. 3) with the
    ``mu`` correction term, and the combined objective (Eq. 4).
  * COMPONENT 5 -- Entropy regularization (Eq. 10).
  * COMPONENT 6 -- Critic targets: on-policy n-step returns (Eq. 5-6) and
    off-policy 1-step returns (Eq. 7-9).

All functions operate on flat tensors of shape ``[T, ...]`` where ``T`` is the
number of transitions in a mini-batch.  Recurrent policies (AllegroKuka) are
handled by passing the LSTM hidden state through the ``lstm_state`` argument;
for feed-forward policies this argument is ignored.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .actor_critic import SharedActorCritic


# ---------------------------------------------------------------------------
# On-policy PPO loss (Eq. 2)
# ---------------------------------------------------------------------------
def ppo_surrogate_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped surrogate objective (Eq. 2).

    L_on(pi) = E[ min(r_t, clip(r_t, 1-eps, 1+eps)) * A_t ],
    with r_t = pi_theta(a|s) / pi_old(a|s).

    Returns the *negative* objective (a loss to be minimized).
    """
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


# ---------------------------------------------------------------------------
# Off-policy importance-sampled loss (Eq. 3)
# ---------------------------------------------------------------------------
def off_policy_surrogate_loss(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Importance-sampled off-policy objective (Eq. 3).

    L_off(pi_i; X) = (1/|X|) sum_{j in X} E_{(s,a)~pi_j}[
        min(r_pi_i, clip(r_pi_i, mu(1-eps), mu(1+eps))) * A^{pi_i, old} ]

    where
        r_pi_i = pi_i(s,a) / pi_j(s,a)          (target / behavior)
        mu     = pi_{i,old}(s,a) / pi_j(s,a)    (old target / behavior)

    ``log_probs``          -> log pi_i(s, a)          (current target policy)
    ``behavior_log_probs`` -> log pi_j(s, a)          (data-collecting policy)
    ``old_log_probs``      -> log pi_{i, old}(s, a)   (target policy at update start)

    Returns the *negative* objective (a loss to be minimized).
    """
    # r_pi_i = pi_i / pi_j
    ratio = torch.exp(log_probs - behavior_log_probs)
    # mu = pi_{i,old} / pi_j
    mu = torch.exp(old_log_probs - behavior_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio, mu * (1.0 - clip_eps), mu * (1.0 + clip_eps)
    ) * advantages
    return -torch.min(unclipped, clipped).mean()


def combined_policy_loss(
    on_policy_loss: torch.Tensor,
    off_policy_loss: Optional[torch.Tensor] = None,
    lam: float = 1.0,
) -> torch.Tensor:
    """Combined objective L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X) (Eq. 4)."""
    if off_policy_loss is None:
        return on_policy_loss
    return on_policy_loss + lam * off_policy_loss


# ---------------------------------------------------------------------------
# Entropy regularization (Eq. 10)
# ---------------------------------------------------------------------------
def entropy_bonus(
    entropy: torch.Tensor,
    sigma: float,
    policy_index: int,
) -> torch.Tensor:
    """Entropy regularization term for follower policies (Eq. 10).

    Follower loss: L(pi_i) = L_on(pi_i) + sigma * (i - 1) * H(pi(a|s)).

    ``policy_index`` is the 1-based index ``i`` of the policy.  The leader
    (i == 1) receives no entropy bonus.  Returns a *loss* contribution
    (negative entropy scaled by the coefficient).
    """
    if sigma <= 0.0 or policy_index <= 1:
        return torch.zeros((), dtype=entropy.dtype, device=entropy.device)
    return -sigma * float(policy_index - 1) * entropy.mean()


# ---------------------------------------------------------------------------
# Critic targets (Eq. 5-9)
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    tau: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards: [T, N] rewards.
        values:  [T, N] value estimates V(s_t).
        dones:   [T, N] episode-termination flags (1.0 if terminal).
        gamma:   discount factor.
        tau:     GAE lambda (paper uses tau = 0.95).

    Returns:
        (advantages, returns) each of shape [T, N].
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros_like(values[0])
        else:
            next_value = values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * tau * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def n_step_value_target(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    n: int = 3,
    gamma: float = 0.99,
) -> torch.Tensor:
    """On-policy n-step value target (Eq. 5-6).

    V_target(s_t) = sum_{k=t}^{t+n-1} gamma^{k-t} r_k + gamma^n V_old(s_{t+n}).

    Args:
        rewards: [T, N] rewards.
        values:  [T, N] value estimates V_old(s_t).
        dones:   [T, N] termination flags.
        n:       number of steps (paper uses n = 3).
        gamma:   discount factor.

    Returns:
        [T, N] value targets.
    """
    T = rewards.shape[0]
    targets = torch.zeros_like(rewards)
    for t in range(T):
        acc = torch.zeros_like(rewards[t])
        discount = 1.0
        bootstrap = torch.ones_like(rewards[t])
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            acc = acc + discount * rewards[idx] * bootstrap
            bootstrap = bootstrap * (1.0 - dones[idx])
            discount = discount * gamma
        # bootstrap value at s_{t+n}
        idx = t + n
        if idx < T:
            acc = acc + discount * values[idx] * bootstrap
        targets[t] = acc
    return targets


def one_step_value_target(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Off-policy 1-step value target (Eq. 7-9).

    V_target(s'_t) = r_t + gamma * V_old(s'_{t+1}).

    Args:
        rewards:     [T, N] rewards from the off-policy data.
        next_values: [T, N] V_old(s'_{t+1}).
        dones:       [T, N] termination flags.
        gamma:       discount factor.

    Returns:
        [T, N] value targets.
    """
    return rewards + gamma * next_values * (1.0 - dones)


def value_loss(
    values: torch.Tensor,
    targets: torch.Tensor,
    clip: Optional[float] = None,
    old_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Squared-error value loss, optionally with PPO-style value clipping."""
    if clip is not None and old_values is not None:
        clipped = old_values + torch.clamp(values - old_values, -clip, clip)
        loss_unclipped = (values - targets) ** 2
        loss_clipped = (clipped - targets) ** 2
        return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
    return 0.5 * ((values - targets) ** 2).mean()


def combined_critic_loss(
    on_policy_loss: torch.Tensor,
    off_policy_loss: Optional[torch.Tensor] = None,
    lam: float = 1.0,
) -> torch.Tensor:
    """L_critic = L_critic_on + lambda * L_critic_off (Sec 4.1)."""
    if off_policy_loss is None:
        return on_policy_loss
    return on_policy_loss + lam * off_policy_loss


# ---------------------------------------------------------------------------
# High-level helpers operating on SharedActorCritic policies
# ---------------------------------------------------------------------------
def evaluate_policy(
    policy: SharedActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute log-prob, entropy and value for a batch under ``policy``."""
    log_prob, new_state = policy.log_prob(obs, actions, lstm_state=lstm_state)
    entropy, _ = policy.entropy(obs, lstm_state=lstm_state)
    value, _ = policy.value(obs, lstm_state=lstm_state)
    return {
        "log_prob": log_prob,
        "entropy": entropy,
        "value": value,
        "lstm_state": new_state,
    }


def compute_policy_loss(
    policy: SharedActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    behavior_log_probs: Optional[torch.Tensor] = None,
    off_policy: bool = False,
    lam: float = 1.0,
    entropy_coef: float = 0.0,
    policy_index: int = 1,
    lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the (possibly combined) policy loss for one policy.

    When ``off_policy`` is True the batch is assumed to come from a different
    (behavior) policy, so ``behavior_log_probs`` must be supplied and the
    importance-sampled objective (Eq. 3) is used.  Otherwise the standard PPO
    objective (Eq. 2) is used.
    """
    log_prob, _ = policy.log_prob(obs, actions, lstm_state=lstm_state)

    if off_policy:
        assert behavior_log_probs is not None, (
            "behavior_log_probs required for off-policy loss"
        )
        loss = off_policy_surrogate_loss(
            log_prob, behavior_log_probs, old_log_probs, advantages, clip_eps
        )
    else:
        loss = ppo_surrogate_loss(
            log_prob, old_log_probs, advantages, clip_eps
        )

    info = {"policy_loss": float(loss.detach().item())}

    if entropy_coef > 0.0 and policy_index > 1:
        entropy, _ = policy.entropy(obs, lstm_state=lstm_state)
        ent_term = entropy_bonus(entropy, entropy_coef, policy_index)
        loss = loss + ent_term
        info["entropy"] = float(entropy.detach().mean().item())

    return loss, info
