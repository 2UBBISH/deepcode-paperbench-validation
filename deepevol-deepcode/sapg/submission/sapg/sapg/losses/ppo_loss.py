"""On-policy PPO policy loss (Eq. 2) and entropy regularization (Eq. 10).

This module implements the on-policy objective used by both the vanilla PPO
baseline and the SAPG follower/leader updates.

From Section 3 (Preliminaries), the PPO objective is the clipped surrogate

    L_on(pi_theta) = E_{pi_old} [ min( r_t(pi_theta),
                                      clip(r_t(pi_theta), 1-eps, 1+eps) ) A_t^{pi_old} ]

    with   r_t(pi_theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t).            (Eq. 2)

From Section 4.5 (Enforcing diversity through entropy regularization), each
follower additionally receives an entropy bonus scaled by its index:

    L(pi_i) = L_on(pi_i) + sigma * (i - 1) * H(pi(a | s))                     (Eq. 10)

The leader (i = 1) receives no entropy term, and ``sigma`` is a hyperparameter.
Note that the entropy term enters the objective with a *positive* sign, i.e. it
*rewards* entropy (the loss that is minimised subtracts the entropy bonus).

All functions accept either explicit tensors or a batch-like object (dict,
``OffPolicyBatch``, or ``RolloutBuffer``) so that the trainers can call them
uniformly.  Sign convention: losses returned by the ``*_loss`` helpers are
*minimised* (i.e. the objective is negated).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "compute_ppo_loss",
    "ppo_loss",
    "on_policy_loss",
    "compute_entropy_bonus",
    "entropy_bonus",
    "entropy_coefficient_for",
    "compute_bounds_loss",
    "combine_policy_and_entropy",
    "importance_ratio",
    "PPOLoss",
]


# ---------------------------------------------------------------------------
# batch access helpers
# ---------------------------------------------------------------------------
_NEW_LOGPROB_KEYS: Sequence[str] = (
    "new_logprobs",
    "logprobs",
    "log_prob",
    "logprob",
    "new_log_prob",
    "cur_logprobs",
)
_OLD_LOGPROB_KEYS: Sequence[str] = (
    "old_logprobs",
    "logprobs_old",
    "old_log_prob",
    "behaviour_logprobs",
    "behavior_logprobs",
)
_ADVANTAGE_KEYS: Sequence[str] = (
    "advantages",
    "adv",
    "advantages_norm",
    "normalized_advantages",
)
_ENTROPY_KEYS: Sequence[str] = ("entropy", "entropies", "entropy_bonus")
_VALUE_KEYS: Sequence[str] = ("values", "value", "new_values", "vpreds")
_CLIP_FRACTION_KEYS: Sequence[str] = ("clip_fraction", "clip_frac")


def _get(batch: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Fetch the first available key from a batch-like object."""
    if batch is None:
        return default
    if isinstance(batch, dict):
        for k in keys:
            if k in batch:
                return batch[k]
        return default
    # objects with `.get`
    get = getattr(batch, "get", None)
    if callable(get):
        for k in keys:
            try:
                value = get(k)
            except TypeError:
                break
            if value is not None:
                return value
    # attribute access / nested `.data` / `.storage`
    for container in (batch, getattr(batch, "data", None), getattr(batch, "storage", None)):
        if container is None:
            continue
        if isinstance(container, dict):
            for k in keys:
                if k in container:
                    return container[k]
            continue
        for k in keys:
            if hasattr(container, k):
                return getattr(container, k)
    return default


def _as_tensor(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


# ---------------------------------------------------------------------------
# importance ratio / clipped surrogate
# ---------------------------------------------------------------------------
def importance_ratio(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    *,
    clamp: Optional[float] = 20.0,
) -> torch.Tensor:
    """``r_t = pi_theta(a|s) / pi_old(a|s)`` computed in log space."""
    log_ratio = new_logprobs - old_logprobs
    if clamp is not None:
        log_ratio = torch.clamp(log_ratio, -float(clamp), float(clamp))
    return torch.exp(log_ratio)


def _clipped_surrogate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> Dict[str, torch.Tensor]:
    """Unclipped and clipped surrogate terms of Eq. 2."""
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratio * advantages
    surrogate = torch.min(unclipped, clipped)

    with torch.no_grad():
        # Fraction of samples for which the clip is active and reduced the
        # surrogate (the standard rl_games "clip fraction" diagnostic).
        clip_frac = (unclipped > clipped).float().mean() if advantages.numel() else ratio.new_zeros(())
        if advantages.numel() == 0:
            clip_frac = ratio.new_zeros(())
    return {
        "surrogate": surrogate,
        "unclipped": unclipped,
        "clipped": clipped,
        "clip_frac": clip_frac,
    }


# ---------------------------------------------------------------------------
# Eq. 2
# ---------------------------------------------------------------------------
def compute_ppo_loss(
    ratio: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    clip_epsilon: float = 0.1,
    old_logprobs: Optional[torch.Tensor] = None,
    new_logprobs: Optional[torch.Tensor] = None,
    *,
    batch: Any = None,
    maximize_objective: bool = False,
    clamp_ratio: Optional[float] = 20.0,
    return_stats: bool = True,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """PPO clipped surrogate loss (Eq. 2).

    Args:
        ratio: importance ratio ``r_t``.  Computed from ``new_logprobs`` /
            ``old_logprobs`` when omitted.
        advantages: ``A_t^{pi_old}`` advantage estimates.
        clip_epsilon: the PPO clipping hyperparameter ``epsilon``.
        old_logprobs / new_logprobs: log-probabilities of the behaviour policy
            and the current policy respectively.
        batch: optional batch-like object to resolve missing inputs from.
        maximize_objective: when True the *objective* (maximised) is returned
            instead of the negated loss (minimised).

    Returns a dict with at least ``policy_loss`` (minimised) plus diagnostics
    ``surrogate``, ``ratio_mean``, ``ratio_max``, ``clip_frac`` (unless
    ``return_stats`` is False).
    """
    if advantages is None:
        advantages = _get(batch, _ADVANTAGE_KEYS)
    advantages = _as_tensor(advantages)

    if ratio is None:
        if new_logprobs is None:
            new_logprobs = _get(batch, _NEW_LOGPROB_KEYS)
        if old_logprobs is None:
            old_logprobs = _get(batch, _OLD_LOGPROB_KEYS)
        new_logprobs = _as_tensor(new_logprobs)
        old_logprobs = _as_tensor(old_logprobs)
        if new_logprobs is not None and old_logprobs is not None:
            ratio = importance_ratio(new_logprobs, old_logprobs, clamp=clamp_ratio)
        elif new_logprobs is not None and old_logprobs is None:
            # No behaviour log-probs available: treat the data as on-policy.
            ratio = torch.ones_like(new_logprobs)
    ratio = _as_tensor(ratio)

    if ratio is None or advantages is None:
        raise ValueError(
            "compute_ppo_loss requires either `ratio`, or `new_logprobs` and "
            "`old_logprobs`, together with `advantages`."
        )

    terms = _clipped_surrogate(ratio, advantages, clip_epsilon)
    surrogate = terms["surrogate"]

    # The objective is to be maximised; the loss minimised is its negation.
    objective = surrogate.mean()
    policy_loss = objective if maximize_objective else -objective

    out: Dict[str, torch.Tensor] = {
        "policy_loss": policy_loss,
        "objective": objective,
        "surrogate": surrogate,
    }
    if return_stats:
        with torch.no_grad():
            out["ratio_mean"] = ratio.mean()
            out["ratio_max"] = ratio.max() if ratio.numel() else ratio.new_zeros(())
            out["ratio_min"] = ratio.min() if ratio.numel() else ratio.new_zeros(())
            out["clip_frac"] = terms["clip_frac"]
        if new_logprobs is not None and old_logprobs is not None:
            with torch.no_grad():
                out["kl"] = (old_logprobs - new_logprobs).mean()
    return out


# Convenience aliases -------------------------------------------------------
def ppo_loss(
    new_logprobs: Optional[torch.Tensor] = None,
    old_logprobs: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    clip_epsilon: float = 0.1,
    *,
    batch: Any = None,
    weight: float = 1.0,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Functional form of Eq. 2 taking log-probabilities directly."""
    out = compute_ppo_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_epsilon=clip_epsilon,
        batch=batch,
        **kwargs,
    )
    if weight != 1.0:
        out["policy_loss"] = out["policy_loss"] * weight
    return out


# Alias used by the trainer / aggregation code.
on_policy_loss = ppo_loss


# ---------------------------------------------------------------------------
# Eq. 10  (entropy regularization)
# ---------------------------------------------------------------------------
def compute_entropy_bonus(
    entropy: torch.Tensor,
    coefficient: float,
    *,
    maximize_objective: bool = False,
) -> torch.Tensor:
    """Entropy bonus term ``- coefficient * H(pi)`` (added to the loss).

    Eq. 10 adds ``sigma * (i-1) * H(pi(a|s))`` to the *objective* for follower
    ``i``; since we minimise the loss, the returned term is negated.
    """
    term = -float(coefficient) * entropy.mean()
    return term if not maximize_objective else -term


# Alias.
entropy_bonus = compute_entropy_bonus


def entropy_coefficient_for(
    policy_index: int,
    sigma: float = 0.0,
    *,
    leader_index: int = 1,
    num_policies: Optional[int] = None,
    maximize_entropy: Optional[bool] = None,
) -> float:
    """Entropy coefficient ``sigma * (i - 1)`` for (1-based) policy ``policy_index``.

    The leader (``i == leader_index``, default 1) always receives coefficient 0
    ("The leader doesn't have any entropy loss.", Sec. 4.5).
    """
    i = int(policy_index)
    if i == int(leader_index):
        return 0.0
    return float(sigma) * float(i - 1)


def combine_policy_and_entropy(
    policy_loss: torch.Tensor,
    entropy: Optional[torch.Tensor] = None,
    coefficient: float = 0.0,
) -> torch.Tensor:
    """``L_on + (- coefficient * H)`` (minimised form of Eq. 10)."""
    if entropy is None or coefficient == 0.0:
        return policy_loss
    return policy_loss + compute_entropy_bonus(entropy, coefficient)


# ---------------------------------------------------------------------------
# action-bound regularization
# ---------------------------------------------------------------------------
def compute_bounds_loss(
    action_mean: torch.Tensor,
    coefficient: float = 1e-4,
    *,
    bound: float = 1.0,
) -> torch.Tensor:
    """Action-bound regularizer on the pre-tanh output.

    The paper only states a "bounds loss coefficient" of ``1e-4``; following the
    rl_games / DexPBT reference implementation we penalise the pre-tanh output
    whenever it exceeds the tanh saturation region ``bound``:

        b(x) = sum(max(|x| - bound, 0)^2)
    """
    if action_mean is None:
        raise ValueError("compute_bounds_loss requires the pre-tanh action mean")
    exceed = torch.clamp(torch.abs(action_mean) - float(bound), min=0.0)
    return float(coefficient) * (exceed * exceed).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------
class PPOLoss(torch.nn.Module):
    """``nn.Module`` wrapper around Eq. 2 + optional Eq. 10 entropy term."""

    def __init__(
        self,
        clip_epsilon: float = 0.1,
        entropy_coefficient: float = 0.0,
        leader_index: int = 1,
        bounds_loss_coefficient: float = 0.0,
        normalize_advantages: bool = False,
        config: Any = None,
    ) -> None:
        super().__init__()
        if config is not None:
            gen = getattr(config, "general", None) or config
            clip_epsilon = getattr(gen, "clip_epsilon", None) or clip_epsilon
            entropy_coefficient = getattr(
                gen, "entropy_coefficient", entropy_coefficient
            )
            if hasattr(gen, "leader_index"):
                leader_index = gen.leader_index
            bounds_loss_coefficient = getattr(
                gen, "bounds_loss_coefficient", bounds_loss_coefficient
            )
            normalize_advantages = getattr(
                gen, "normalize_advantage", normalize_advantages
            )
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coefficient = float(entropy_coefficient)
        self.leader_index = int(leader_index)
        self.bounds_loss_coefficient = float(bounds_loss_coefficient)
        self.normalize_advantages = bool(normalize_advantages)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _maybe_normalize(advantages: torch.Tensor) -> torch.Tensor:
        if advantages.numel() <= 1:
            return advantages
        std = advantages.std(unbiased=False)
        return (advantages - advantages.mean()) / (std + 1e-8)

    def forward(
        self,
        batch: Any = None,
        *,
        new_logprobs: Optional[torch.Tensor] = None,
        old_logprobs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        action_mean: Optional[torch.Tensor] = None,
        policy_index: Optional[int] = None,
        coefficient: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if advantages is None:
            advantages = _get(batch, _ADVANTAGE_KEYS)
        advantages = _as_tensor(advantages)
        if self.normalize_advantages and advantages is not None:
            advantages = self._maybe_normalize(advantages)

        entropy = _as_tensor(entropy if entropy is not None else _get(batch, _ENTROPY_KEYS))
        if action_mean is None:
            action_mean = _get(batch, ("action_mean", "mu", "pre_tanh")) 

        out = compute_ppo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
            batch=batch,
            clamp_ratio=20.0,
        )

        if coefficient is None:
            coefficient = (
                entropy_coefficient_for(
                    policy_index if policy_index is not None else self.leader_index,
                    self.entropy_coefficient,
                    leader_index=self.leader_index,
                )
                if policy_index is not None
                else self.entropy_coefficient
            )

        total = out["policy_loss"]
        if entropy is not None and coefficient:
            ent_term = compute_entropy_bonus(entropy, coefficient)
            total = total + ent_term
            out["entropy_loss"] = ent_term
            out["entropy_coefficient"] = torch.as_tensor(float(coefficient), dtype=total.dtype)
        if (
            action_mean is not None
            and getattr(self, "bounds_loss_coefficient", 0.0)
        ):
            bounds = compute_bounds_loss(action_mean, self.bounds_loss_coefficient)
            total = total + bounds
            out["bounds_loss"] = bounds

        out["total_loss"] = total
        return out

    def loss(self, batch: Any = None, **kwargs: Any) -> torch.Tensor:
        return self.forward(batch, **kwargs)["total_loss"]

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"clip_epsilon={self.clip_epsilon}, "
            f"entropy_coefficient={self.entropy_coefficient}, "
            f"leader_index={self.leader_index}"
        )
