"""Off-policy, importance-sampled PPO loss (SAPG, Section 4.1, Eq. 3 + Eq. 4).

To update policy ``pi_i`` with data sampled by a *different* policy
``pi_j`` (``j`` in the aggregation set ``X``) SAPG uses an
importance-sampled clipped surrogate::

    L_off(pi_i; X) = 1/|X| sum_{j in X} E_{(s,a) ~ pi_j}[
                        min( r_{pi_i}(s,a),
                             clip(r_{pi_i}(s,a), mu(1-eps), mu(1+eps)) )
                        * A^{pi_i,old}(s,a) ]                   (Eq. 3)

    with   r_{pi_i}(s, a) = pi_i(s, a) / pi_j(s, a)
           mu             = pi_{i,old}(s, a) / pi_j(s, a)        (Eq. 3, 2nd line)

and the total objective combines it with the on-policy term::

    L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X)                (Eq. 4)

with ``lambda = 1`` for the leader/follower variant that works best
(Section 4.3).  Because the clipping bounds are rescaled by
``mu = pi_{i,old} / pi_j``, when ``i == j`` we have ``pi_j == pi_{i,old}``
so ``mu == 1`` and ``r_{pi_i} = pi_i / pi_{i,old}``, i.e. Eq. 3 reduces
*exactly* to the on-policy PPO surrogate of Eq. 2 (Section 4.1: "Note that
when i = j, then pi_j = pi_{i,old} and this reduces to the on-policy update
as expected").

Everything is expressed in log-probability space for numerical stability:
``r = exp(log pi_i - log pi_j)`` and ``mu = exp(log pi_{i,old} - log pi_j)``.

Sign convention
---------------
As in :mod:`sapg.algorithms.ppo`, the returned ``off_policy_loss`` is the
quantity that is **minimised** (i.e. the negated surrogate mean), matching the
standard PPO/rl_games convention.  Set ``maximize_objective=True`` to obtain
the raw (maximised) surrogate instead.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "importance_ratio",
    "mu_from_logprobs",
    "off_policy_surrogate",
    "off_policy_loss",
    "combine_on_off_loss",
    "combined_objective",
    "OffPolicyLoss",
]

# Names under which the different quantities may appear inside an off-policy
# batch (``OffPolicyBatch`` / plain dict).
_NEW_LOGPROB_KEYS = ("new_logprobs", "logprobs", "log_prob", "logprob", "new_log_prob")
_SOURCE_LOGPROB_KEYS = (
    "behaviour_logprobs",
    "behavior_logprobs",
    "source_logprobs",
    "source_log_prob",
    "old_logprobs_source",
    "behaviour_log_prob",
    "behavior_log_prob",
)
_TARGET_OLD_LOGPROB_KEYS = (
    "target_old_logprobs",
    "target_logprobs",
    "leader_old_logprobs",
    "target_old_log_prob",
    "target_log_prob",
    "mu_logprobs",
)
_ADVANTAGE_KEYS = ("advantages", "adv", "advantage")
_MU_KEYS = ("mu", "off_policy_mu", "correction")
_SOURCE_POLICY_KEYS = ("source_policy", "source_policy_index", "policy_index", "source")

_LOG_RATIO_CLAMP = 20.0  # exp(20) ~ 4.8e8, plenty for clipping bounds


# --------------------------------------------------------------------------- #
# basic building blocks
# --------------------------------------------------------------------------- #
def importance_ratio(
    new_logprobs: torch.Tensor,
    behaviour_logprobs: torch.Tensor,
    clamp: Optional[float] = _LOG_RATIO_CLAMP,
) -> torch.Tensor:
    """Importance ratio ``r_{pi_i}(s,a) = pi_i(s,a) / pi_j(s,a)``.

    ``pi_i`` is the policy being updated (``new_logprobs``) and ``pi_j`` is the
    behaviour/source policy that generated the data
    (``behaviour_logprobs``).  When ``i == j`` the two log-probabilities are
    identical and the ratio is exactly ``1``.
    """
    log_ratio = new_logprobs - behaviour_logprobs
    if clamp is not None:
        log_ratio = torch.clamp(log_ratio, -clamp, clamp)
    return torch.exp(log_ratio)


def mu_from_logprobs(
    target_old_logprobs: torch.Tensor,
    behaviour_logprobs: torch.Tensor,
    clamp: Optional[float] = _LOG_RATIO_CLAMP,
) -> torch.Tensor:
    """Off-policy correction ``mu = pi_{i,old}(s,a) / pi_j(s,a)`` (Eq. 3).

    ``target_old_logprobs`` are the log-probabilities of the update target
    policy *before* the update (``pi_{i,old}``) evaluated on the off-policy
    samples, and ``behaviour_logprobs`` are those of the sampling policy
    ``pi_j``.  For on-policy data (``i == j``) this returns ``1``.
    """
    log_mu = target_old_logprobs - behaviour_logprobs
    if clamp is not None:
        log_mu = torch.clamp(log_mu, -clamp, clamp)
    return torch.exp(log_mu)


def _reduce_per_source(
    values: torch.Tensor,
    source_policy: Optional[torch.Tensor],
    reduction: str = "source",
) -> torch.Tensor:
    """Average ``values`` following Eq. 3's ``1/|X| sum_{j in X} E_{pi_j}[.]``.

    ``reduction="source"`` computes the mean separately for every source
    policy ``j`` and then averages those means (this is exactly
    ``1/|X| sum_j E_{(s,a)~pi_j}[...]`` when each ``j`` supplies the same
    number of samples).  ``reduction="mean"`` performs a single flat mean.
    """
    if source_policy is None or reduction != "source":
        return values.mean()

    unique = torch.unique(source_policy)
    if unique.numel() <= 1:
        return values.mean()

    per_policy = []
    for j in unique.tolist():
        mask = source_policy == j
        if mask.any():
            per_policy.append(values[mask].mean())
    if not per_policy:
        return values.mean()
    return torch.stack(per_policy).mean()


def _extract(
    data: Any,
    keys: Sequence[str],
    default: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Fetch the first available tensor among ``keys`` from a batch object."""
    if data is None:
        return default
    for key in keys:
        if isinstance(data, dict):
            if key in data:
                return data[key]
        elif hasattr(data, key):
            value = getattr(data, key)
            if value is not None:
                return value
        elif hasattr(data, "get"):
            try:
                value = data.get(key, None)
            except Exception:  # pragma: no cover - defensive
                value = None
            if value is not None:
                return value
    return default


# --------------------------------------------------------------------------- #
# surrogate / loss
# --------------------------------------------------------------------------- #
def off_policy_surrogate(
    new_logprobs: torch.Tensor,
    behaviour_logprobs: torch.Tensor,
    target_old_logprobs: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    clip_epsilon: float = 0.1,
    clip_mu: Optional[float] = None,
    source_policy: Optional[torch.Tensor] = None,
    reduction: str = "source",
    return_stats: bool = False,
):
    """Clipped importance-sampled surrogate of Eq. 3 (positive = maximised).

    Computes::

        min(r, clip(r, mu (1-eps), mu (1+eps))) * A

    Returns the (possibly normalised) per-sample surrogate, or, when
    ``return_stats`` is true, a tuple ``(surrogate, stats)`` where ``stats``
    holds ``ratio``, ``mu``, ``clip_frac`` and ``ratio_mean``.
    """
    ratio = importance_ratio(new_logprobs, behaviour_logprobs)

    if mu is None:
        if target_old_logprobs is not None:
            mu = mu_from_logprobs(target_old_logprobs, behaviour_logprobs)
        else:
            # i == j (on-policy data): mu = 1, the surrogate degenerates to Eq. 2
            mu = torch.ones_like(ratio)
    mu = mu.to(ratio.dtype).reshape(-1) if mu.dim() > 1 else mu
    if mu.numel() == 1 and ratio.numel() != 1:
        mu = mu.expand_as(ratio)
    if clip_mu is not None:
        mu = mu.clamp(max=float(clip_mu))

    lower = mu * (1.0 - clip_epsilon)
    upper = mu * (1.0 + clip_epsilon)
    clipped = torch.clamp(ratio, lower, upper)

    if advantages is None:
        advantages = torch.ones_like(ratio)
    advantages = advantages.reshape(-1) if advantages.dim() > 1 else advantages
    if advantages.numel() == 1 and ratio.numel() != 1:
        advantages = advantages.expand_as(ratio)

    unclipped_obj = ratio * advantages
    clipped_obj = clipped * advantages
    surrogate = torch.min(unclipped_obj, clipped_obj)

    # mean over source policies j in X (Eq. 3's 1/|X| sum_j)
    surrogate_mean = _reduce_per_source(surrogate, source_policy, reduction)

    if not return_stats:
        return surrogate_mean

    with torch.no_grad():
        active = torch.clamp(ratio - lower, min=0.0) + torch.clamp(upper - ratio, min=0.0)
        clip_frac = (active > 0).float().mean()
        stats = {
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "mu_mean": mu.mean(),
            "clip_frac": clip_frac,
        }
    return surrogate_mean, stats


def off_policy_loss(
    new_logprobs: Optional[torch.Tensor] = None,
    behaviour_logprobs: Optional[torch.Tensor] = None,
    target_old_logprobs: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    clip_epsilon: float = 0.1,
    source_policy: Optional[torch.Tensor] = None,
    reduction: str = "source",
    weight: float = 1.0,
    lam: Optional[float] = None,
    clip_mu: Optional[float] = None,
    batch: Any = None,
    normalize_advantages: bool = False,
    maximize_objective: bool = False,
    eps: float = 1e-8,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Eq. 3 loss for one off-policy batch (data collected by policies in ``X``).

    The function accepts the quantities either positionally / by keyword, or
    bundled in ``batch`` (a :class:`sapg.buffers.rollout_buffer.OffPolicyBatch`,
    a ``dict`` of tensors, or any object exposing those attributes).  Keyword
    aliases are resolved so the trainer can pass its dict straight through
    (``logprobs``/``behaviour_logprobs``/``old_logprobs`` ...).

    Returns
    -------
    dict with
        ``off_policy_loss``   scalar tensor to be minimised (negated surrogate
                              unless ``maximize_objective=True``), scaled by
                              ``weight``/``lam`` when given,
        ``surrogate``         the positive (maximised) surrogate value,
        ``ratio_mean``, ``ratio_max``, ``mu_mean``, ``clip_frac``,
        ``policy_loss``       alias of ``off_policy_loss``.
    """
    # --- resolve inputs, allowing aliases / a batch object ----------------- #
    if batch is not None:
        if new_logprobs is None:
            new_logprobs = _extract(batch, _NEW_LOGPROB_KEYS)
        if behaviour_logprobs is None:
            behaviour_logprobs = _extract(batch, _SOURCE_LOGPROB_KEYS)
        if target_old_logprobs is None:
            target_old_logprobs = _extract(batch, _TARGET_OLD_LOGPROB_KEYS)
        if advantages is None:
            advantages = _extract(batch, _ADVANTAGE_KEYS)
        if mu is None:
            mu = _extract(batch, _MU_KEYS)
        if source_policy is None:
            source_policy = _extract(batch, _SOURCE_POLICY_KEYS)

    if new_logprobs is None:
        new_logprobs = kwargs.pop("new_log_prob", None) or kwargs.pop("logprob", None)
    if behaviour_logprobs is None:
        behaviour_logprobs = kwargs.pop("behaviour_log_prob", None)
    if target_old_logprobs is None:
        target_old_logprobs = kwargs.pop("target_old_log_prob", None)

    if new_logprobs is None:
        raise ValueError("off_policy_loss requires `new_logprobs` (log pi_i(s,a))")
    if behaviour_logprobs is None:
        # Without the source-policy log-probs the data is treated as on-policy
        # (pi_j == pi_i,old), i.e. r reduces to 1.
        behaviour_logprobs = new_logprobs

    new_logprobs = new_logprobs.reshape(-1)
    behaviour_logprobs = behaviour_logprobs.reshape(-1)
    if target_old_logprobs is not None:
        target_old_logprobs = target_old_logprobs.reshape(-1)
    if advantages is not None:
        advantages = advantages.reshape(-1)
    if mu is not None:
        mu = mu.reshape(-1)
    if source_policy is not None:
        source_policy = source_policy.reshape(-1)

    if normalize_advantages and advantages is not None and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

    surrogate, stats = off_policy_surrogate(
        new_logprobs=new_logprobs,
        behaviour_logprobs=behaviour_logprobs,
        target_old_logprobs=target_old_logprobs,
        advantages=advantages,
        mu=mu,
        clip_epsilon=clip_epsilon,
        clip_mu=clip_mu,
        source_policy=source_policy,
        reduction=reduction,
        return_stats=True,
    )

    scale = float(lam) if lam is not None else float(weight)
    objective = surrogate if maximize_objective else -surrogate
    loss = objective * scale

    return {
        "off_policy_loss": loss,
        "policy_loss": loss,
        "surrogate": surrogate,
        "off_policy_objective": surrogate,
        "ratio_mean": stats["ratio_mean"].detach(),
        "ratio_max": stats["ratio_max"].detach(),
        "mu_mean": stats["mu_mean"].detach(),
        "clip_frac": stats["clip_frac"].detach(),
    }


def combine_on_off_loss(
    on_policy_loss: torch.Tensor,
    off_policy_terms: Optional[Iterable[torch.Tensor]] = None,
    lam: float = 1.0,
) -> torch.Tensor:
    """Eq. 4: ``L(pi_i) = L_on(pi_i) + lambda * L_off(pi_i; X)``."""
    total = on_policy_loss
    if off_policy_terms is None:
        return total
    weight = 1.0 / max(1, len(list(off_policy_terms))) if isinstance(off_policy_terms, Sequence) else 1.0
    terms = list(off_policy_terms) if off_policy_terms is not None else []
    if not terms:
        return total
    combined = torch.stack([t for t in terms]).mean() * float(lam)
    return total + combined


def combined_objective(
    on_policy_loss: torch.Tensor,
    off_policy_loss_value: Optional[torch.Tensor] = None,
    lam: float = 1.0,
) -> torch.Tensor:
    """Convenience wrapper for Eq. 4 (scalar in, scalar out)."""
    if off_policy_loss_value is None:
        return on_policy_loss
    return on_policy_loss + float(lam) * off_policy_loss_value


# --------------------------------------------------------------------------- #
# nn.Module wrapper
# --------------------------------------------------------------------------- #
class OffPolicyLoss(nn.Module):
    """Eq. 3 as a module (used by ``SAPGTrainer`` for the leader's update).

    Parameters
    ----------
    clip_epsilon:
        PPO clip range ``eps`` of Eq. 2/3 (0.1 for AllegroKuka, 0.2 for the
        Shadow/Allegro hand task groups, cf. Appendix B.1-B.3).
    lam:
        Weight ``lambda`` multiplying the off-policy term in Eq. 4
        (``lambda = 1`` in the best leader/follower variant, Section 4.3).
    reduction:
        ``"source"`` averages the per-source-policy expectations exactly as
        ``1/|X| sum_{j in X} E_{pi_j}[.]``; ``"mean"`` takes a flat mean.
    """

    def __init__(
        self,
        clip_epsilon: float = 0.1,
        lam: float = 1.0,
        reduction: str = "source",
        normalize_advantages: bool = False,
        clip_mu: Optional[float] = None,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            clip_epsilon = float(getattr(config, "clip_epsilon", clip_epsilon))
            lam = float(getattr(config, "off_policy_weight", lam))
        self.clip_epsilon = float(clip_epsilon)
        self.lam = float(lam)
        self.reduction = reduction
        self.normalize_advantages = bool(normalize_advantages)
        self.clip_mu = clip_mu

    # -- helpers ----------------------------------------------------------- #
    def forward(self, batch: Any = None, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Compute the Eq. 3 loss for ``batch`` (see :func:`off_policy_loss`)."""
        out = off_policy_loss(
            batch=batch,
            clip_epsilon=self.clip_epsilon,
            reduction=self.reduction,
            normalize_advantages=self.normalize_advantages,
            clip_mu=self.clip_mu,
            weight=1.0,  # lambda is applied by the caller via Eq. 4
            **kwargs,
        )
        return out

    def loss(self, batch: Any = None, **kwargs: Any) -> torch.Tensor:
        """Scalar Eq. 3 loss (already lambda-scaled)."""
        out = self.forward(batch, **kwargs)
        return out["off_policy_loss"] * self.lam

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"clip_epsilon={self.clip_epsilon}, lam={self.lam}, reduction={self.reduction}"
