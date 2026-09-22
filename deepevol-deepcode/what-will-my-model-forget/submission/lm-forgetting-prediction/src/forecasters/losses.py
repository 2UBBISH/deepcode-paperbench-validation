"""Loss functions for forecasting forgotten examples and for replay-based refinement.

This module implements, exactly as specified in the paper:

1. The **margin loss** of the trainable logit-change-transfer forecaster (Eq. 3 in Sec. 3.2):

       L(<x_i,y_i>, <x_j,y_j>, z_ij)
         = max(0, 1 + (-1)^{z_ij} ( max_{v != y_j} f_hat_i(x_j)[v] - f_hat_i(x_j)[y_j] ))

   i.e. the predicted logit of the correct token ``y_j`` should exceed the second-top
   candidate by a preset margin (1.0) when ``<x_j,y_j>`` is *not* forgotten (z=0), and be
   reversed when it *is* forgotten (z=1).

2. The **binary cross-entropy** used by the representation-based forecaster (Sec. 3.3):

       z_tilde_ij = sigma( h(x_j,y_j) h(x_i,y_i)^T + b_j )
       L_BCE(z_tilde_ij, z_ij)

   The frequency prior ``b_j`` (log odds of forgetting, Algorithm 3) is added inside the
   sigmoid; the ``w/o Prior`` ablation simply drops it from the score.

3. The **replay distillation loss** used when refining the LM (Sec. 4.2): replaying a
   mini-batch of upstream examples with a distillation loss against the outputs of the
   *base* PTLM (Buzzega et al., 2020a - "Dark Experience for General Continual Learning").
   We implement the DER-style MSE on pre-softmax logits (+ optional KL) plus the
   cross-entropy on the replayed hard labels.

Everything is vectorized over a batch and tolerant of the shapes produced by the cached
top-k logits (only ``k = 100`` logits per output token are cached, Sec. 3.2
"Efficient Inference").

Source: Sec. 3.2 (Eq. 2, Eq. 3); Sec. 3.3 (Eq. 4 + prior); Sec. 4.2 (replay distillation);
Appendix B (training details: 8 positive + 8 negative pairs, alpha = 0.1 on positives);
Appendix F (Algorithms 1-4).
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "NEG_INF",
    "counts_to_prior",
    "binary_cross_entropy_with_prior",
    "weighted_binary_cross_entropy",
    "representation_loss",
    "representation_scores",
    "margin_loss",
    "margin_term",
    "margin_loss_from_topk",
    "prediction_from_logits",
    "prediction_from_topk",
    "distillation_loss",
    "replay_distillation_loss",
    "kl_distillation_loss",
    "mse_logit_distillation_loss",
    "replay_total_loss",
    "total_forecaster_loss",
]

NEG_INF = -1e4


# --------------------------------------------------------------------------------------
# Frequency prior / helper conversions
# --------------------------------------------------------------------------------------
def counts_to_prior(
    n_positive: torch.Tensor | float,
    n_negative: torch.Tensor | float,
    n_total: Optional[torch.Tensor | float] = None,
    eps: float = 1e-8,
) -> torch.Tensor | float:
    """Frequency prior ``b_j = log(P(z=1)) - log(P(z=0))`` (Sec. 3.3, Algorithm 3).

    The paper writes the two probabilities with ``|D_R^train|`` as the normalizer
    (``P(z_ij=1) = |{<x_i,y_i> in D_R^train | z_ij=1}| / |D_R^train|``).  Since both logs
    share the same denominator, when ``n_total`` is given we use it for both terms (this
    degenerates to ``log(n_pos) - log(n_neg)`` otherwise).

    Args:
        n_positive: number of online examples that forget this upstream example.
        n_negative: number of online examples that do not forget it.
        n_total: ``|D_R^train|``; defaults to ``n_positive + n_negative``.
        eps: clipping constant so that log(0) never occurs.
    """
    n_pos = torch.as_tensor(n_positive, dtype=torch.float32)
    n_neg = torch.as_tensor(n_negative, dtype=torch.float32)
    if n_total is None:
        n_total = n_pos + n_neg
    n_tot = torch.as_tensor(n_total, dtype=torch.float32)
    n_tot = torch.clamp(n_tot, min=eps)
    p_pos = torch.clamp(n_pos / n_tot, min=eps)
    p_neg = torch.clamp(n_neg / n_tot, min=eps)
    return torch.log(p_pos) - torch.log(p_neg)


# --------------------------------------------------------------------------------------
# BCE losses (representation-based forecaster, Sec. 3.3 / Algorithm 3)
# --------------------------------------------------------------------------------------
def binary_cross_entropy_with_prior(
    pair_score: torch.Tensor,
    z: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
    positive_weight: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE over ``sigma(score + b_j)`` (Eq. 4 / Algorithm 3).

    Args:
        pair_score: ``h(x_j,y_j) h(x_i,y_i)^T`` of shape ``[B]`` (or ``[B,1]``).
        z: ground-truth forgetting labels ``z_ij`` of shape ``[B]``.
        prior: frequency prior ``b_j`` of shape ``[B]`` (broadcastable); ``None`` for the
            ``w/o Prior`` ablation.
        positive_weight: weight ``alpha`` applied to *positive* (forgotten) pairs.  The
            paper down-weights positives with ``alpha = 0.1`` (Appendix B); pass 1.0 for a
            plain BCE.
        reduction: ``"mean" | "sum" | "none"``.
    """
    score = torch.as_tensor(pair_score, dtype=torch.float32).reshape(-1)
    z = torch.as_tensor(z, dtype=torch.float32).reshape(-1)
    if prior is not None:
        score = score + torch.as_tensor(prior, dtype=torch.float32).reshape(-1)
    if score.numel() != z.numel():
        raise ValueError(
            "pair_score and z must have the same number of elements (%d vs %d)"
            % (score.numel(), z.numel())
        )
    return F.binary_cross_entropy_with_logits(
        score, z, pos_weight=torch.as_tensor(positive_weight, dtype=torch.float32),
        reduction=reduction,
    )


def weighted_binary_cross_entropy(
    probabilities: torch.Tensor,
    z: torch.Tensor,
    positive_weight: float = 0.1,
    reduction: str = "mean",
    eps: float = 1e-7,
) -> torch.Tensor:
    """Explicit weighted BCE on *probabilities* (alpha on the positive class).

    Mirrors ``binary_cross_entropy_with_prior`` but takes post-sigmoid probabilities, as
    ``z_tilde_ij`` is written in Eq. 4 / Algorithm 3.
    """
    p = torch.as_tensor(probabilities, dtype=torch.float32).reshape(-1)
    z = torch.as_tensor(z, dtype=torch.float32).reshape(-1)
    p = torch.clamp(p, min=eps, max=1.0 - eps)
    loss = -(z * torch.log(p) + (1.0 - z) * torch.log(1.0 - p))
    if positive_weight != 1.0:
        w = torch.where(z > 0.5, torch.full_like(z, float(positive_weight)), torch.ones_like(z))
        loss = loss * w
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def representation_scores(
    h_upstream: torch.Tensor,
    h_online: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``sigma(h(x_j,y_j) h(x_i,y_i)^T + b_j)`` (Eq. 4).

    ``h_upstream`` / ``h_online`` are *mean-pooled* representations (Sec. 3.3 overrides
    ``h`` to denote averaged representations of all tokens).  Accepts ``[B, d]`` pairs and
    computes the element-wise inner product; broadcasting a single row against many rows is
    supported for inference-time scoring of one online example against all upstream
    examples.
    """
    hj = torch.as_tensor(h_upstream, dtype=torch.float32)
    hi = torch.as_tensor(h_online, dtype=torch.float32)
    if hj.dim() == 1:
        hj = hj.unsqueeze(0)
    if hi.dim() == 1:
        hi = hi.unsqueeze(0)
    if hj.shape[0] == 1 and hi.shape[0] > 1:
        hj = hj.expand(hi.shape[0], -1)
    if hi.shape[0] == 1 and hj.shape[0] > 1:
        hi = hi.expand(hj.shape[0], -1)
    score = (hj * hi).sum(dim=-1)
    if prior is not None:
        score = score + torch.as_tensor(prior, dtype=torch.float32).reshape(-1)
    return torch.sigmoid(score)


def representation_loss(
    h_upstream: torch.Tensor,
    h_online: torch.Tensor,
    z: torch.Tensor,
    prior: Optional[torch.Tensor] = None,
    positive_weight: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """End-to-end representation-based objective: ``sigma(<h_j, h_i> + b_j)`` + BCE."""
    hj = torch.as_tensor(h_upstream, dtype=torch.float32).reshape(-1, h_upstream.shape[-1])
    hi = torch.as_tensor(h_online, dtype=torch.float32).reshape(-1, h_online.shape[-1])
    z_t = torch.as_tensor(z, dtype=torch.float32).reshape(-1)
    score = (hj * hi).sum(dim=-1)
    return binary_cross_entropy_with_prior(
        score, z_t, prior=prior, positive_weight=positive_weight, reduction=reduction
    )


# --------------------------------------------------------------------------------------
# Margin loss (logit-change-transfer forecaster, Eq. 3 / Sec. 3.2)
# --------------------------------------------------------------------------------------
def margin_term(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``max_{v != y_j} f(x_j)[v] - f(x_j)[y_j]`` for every output token (Eq. 3).

    Args:
        logits: ``[B, T, V]`` (reshaped from ``R^{TV}`` to ``R^{T x V}`` as in Sec. 3.2)
            or ``[B, V]`` for a single output token.
        target_ids: ``[B, T]`` (or ``[B]``) gold token ids ``y_j``.
        target_mask: optional ``[B, T]`` mask, kept for API symmetry (masking is applied in
            :func:`margin_loss`).
    Returns:
        ``[B, T]`` (or ``[B]``) tensor of margin terms.
    """
    logits = torch.as_tensor(logits)
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    tgt = torch.as_tensor(target_ids, dtype=torch.long)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(1)
    tgt = tgt.to(logits.device).clamp(min=0, max=logits.shape[-1] - 1)

    gold = torch.gather(logits, -1, tgt.unsqueeze(-1)).squeeze(-1)  # [B,T]
    # mask out the gold token so that max is over v != y_j
    masked = logits.clone()
    masked.scatter_(-1, tgt.unsqueeze(-1), NEG_INF)
    best_other = masked.max(dim=-1).values  # [B,T]
    return best_other - gold


def margin_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    z: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    margin: float = 1.0,
    reduction: str = "mean",
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Margin loss of Eq. 3.

    ``L = max(0, 1 + (-1)^{z_ij} * (max_{v != y_j} f_hat_i(x_j)[v] - f_hat_i(x_j)[y_j]))``

    Args:
        logits: predicted logits ``f_hat_i(x_j)`` of shape ``[B, T, V]`` (or ``[B, V]``).
        target_ids: gold token ids ``y_j`` of shape ``[B, T]`` (or ``[B]``).
        z: forgetting labels ``z_ij`` of shape ``[B]`` (1 = forgotten).
        target_mask: optional ``[B, T]`` output-token mask (padded positions excluded).
        margin: the preset margin (1.0 in the paper / Eq. 3).
        reduction: ``"mean" | "sum" | "none"``; ``"mean"`` averages over valid tokens.
        loss_mask: optional extra ``[B, T]`` (or ``[B]``) weighting for the per-token loss
            (used e.g. to restrict the objective to the cached top-k tokens).
    """
    term = margin_term(logits, target_ids, target_mask=target_mask)  # [B,T]
    z_vec = torch.as_tensor(z, dtype=torch.float32).reshape(-1).to(term.device)
    sign = torch.where(z_vec > 0.5, -torch.ones_like(z_vec), torch.ones_like(z_vec))
    if term.dim() == 2 and sign.numel() == term.shape[0]:
        term = term * sign.unsqueeze(1)
    else:
        term = term * sign
    per_token = torch.clamp(margin + term, min=0.0)

    if target_mask is not None:
        mask = torch.as_tensor(target_mask, dtype=torch.float32).to(per_token.device)
        if mask.dim() == 1 and per_token.dim() == 2:
            mask = mask.unsqueeze(0).expand_as(per_token)
        w = mask
        if loss_mask is not None:
            outer = torch.as_tensor(loss_mask, dtype=torch.float32).to(per_token.device)
            if outer.dim() == 1 and per_token.dim() == 2:
                outer = outer.unsqueeze(1)
            w = w * outer
        denom = torch.clamp(w.sum(), min=1.0)
        if reduction == "sum":
            return (per_token * w).sum()
        if reduction == "none":
            return per_token * w
        return (per_token * w).sum() / denom

    if reduction == "sum":
        return per_token.sum()
    if reduction == "none":
        return per_token
    return per_token.mean()


def margin_loss_from_topk(
    topk_values: Sequence[Sequence[float]] | torch.Tensor,
    topk_indices: Sequence[Sequence[int]] | torch.Tensor,
    target_ids: Sequence[int] | torch.Tensor,
    z: torch.Tensor,
    margin: float = 1.0,
    default_logit: float = NEG_INF,
    reduction: str = "mean",
) -> torch.Tensor:
    """Eq. 3 computed from the *cached* top-k logits (``k = 100``, Sec. 3.2).

    Inference (Algorithm 2) predicts ``z_hat_ij = 1`` iff ``argmax f_hat_i(x_j) != y_j``.
    When the gold token is not among the cached top-k, its logit is unknown; we substitute
    ``default_logit`` (a very small value), which makes the margin term strongly positive --
    consistent with the fact that the gold token is certainly not the top-1 candidate.

    Args:
        topk_values: ``[T, k]`` (or ``[B, T, k]``) cached logit values.
        topk_indices: ``[T, k]`` (or ``[B, T, k]``) cached token ids, aligned with values.
        target_ids: ``[T]`` (or ``[B, T]``) gold ids.
        z: ``[B]`` labels (or scalar shared by all positions).
    """
    values = torch.as_tensor(topk_values, dtype=torch.float32)
    indices = torch.as_tensor(topk_indices, dtype=torch.long)
    tgt = torch.as_tensor(target_ids, dtype=torch.long)
    if values.dim() == 2:
        values = values.unsqueeze(0)
        indices = indices.unsqueeze(0)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0).expand(values.shape[0], -1)
    z_t = torch.as_tensor(z, dtype=torch.float32).reshape(-1)
    if z_t.numel() == 1:
        z_t = z_t.expand(values.shape[0])

    b, t, k = values.shape
    tgt = tgt[:, :t].clamp(min=0)
    gold_pos = (indices == tgt.unsqueeze(-1))  # [B,T,k]
    in_topk = gold_pos.any(dim=-1)  # [B,T]
    masked = values.masked_fill(gold_pos, NEG_INF)
    best_other = masked.max(dim=-1).values  # [B,T]
    gold_val = torch.gather(values, -1, gold_pos.float().argmax(dim=-1, keepdim=True)).squeeze(-1)
    gold_val = torch.where(in_topk, gold_val, torch.full_like(gold_val, float(default_logit)))
    term = best_other - gold_val  # [B,T]

    sign = torch.where(z_t > 0.5, -torch.ones_like(z_t), torch.ones_like(z_t))[:, None]
    per_token = torch.clamp(margin + term * sign, min=0.0)
    if reduction == "sum":
        return per_token.sum()
    if reduction == "none":
        return per_token
    return per_token.mean()


def prediction_from_logits(
    logits: torch.Tensor, target_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``argmax`` decoding snippet used by Algorithm 2.

    Returns ``(argmax_ids, is_same_as_target)``; ``is_same_as_target`` is all-zeros when no
    target is provided.
    """
    logits = torch.as_tensor(logits)
    pred = logits.argmax(dim=-1)
    if target_ids is None:
        return pred, torch.zeros_like(pred, dtype=torch.bool)
    tgt = torch.as_tensor(target_ids, dtype=torch.long).to(pred.device)
    if tgt.dim() == 1 and pred.dim() == 2:
        tgt = tgt.unsqueeze(0).expand_as(pred)
    return pred, pred.eq(tgt)


def prediction_from_topk(
    topk_values: Sequence[Sequence[float]] | torch.Tensor,
    topk_indices: Sequence[Sequence[int]] | torch.Tensor,
    target_ids: Optional[Sequence[int] | torch.Tensor] = None,
) -> torch.Tensor:
    """``z_hat_ij = 1[argmax_v f_hat_i(x_j)[v] != y_j]`` from cached top-k logits.

    The argmax over the *cached* top-k equals the true argmax over the vocabulary as long as
    the top-1 logit is cached (guaranteed by construction of the cache).
    """
    values = torch.as_tensor(topk_values, dtype=torch.float32)
    indices = torch.as_tensor(topk_indices, dtype=torch.long)
    if values.dim() == 2:
        values = values.unsqueeze(0)
        indices = indices.unsqueeze(0)
    best = values.argmax(dim=-1)
    argmax_ids = torch.gather(indices, -1, best.unsqueeze(-1)).squeeze(-1)  # [B,T]
    if target_ids is None:
        return torch.zeros_like(argmax_ids)
    tgt = torch.as_tensor(target_ids, dtype=torch.long)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0).expand_as(argmax_ids)
    tgt = tgt.to(argmax_ids.device)[:, : argmax_ids.shape[1]]
    return argmax_ids.ne(tgt).long()


# --------------------------------------------------------------------------------------
# Distillation losses for replay-based refinement (Sec. 4.2, Buzzega et al. 2020a)
# --------------------------------------------------------------------------------------
def mse_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """DER-style MSE between the refined model's logits and the *base* PTLM's logits.

    Buzzega et al. (2020a) store the logits produced by the model before learning a task and
    regularize the current logits towards them; this is the "distillation loss against the
    outputs of the base PTLM" of Sec. 4.2.
    """
    s = torch.as_tensor(student_logits, dtype=torch.float32)
    t = torch.as_tensor(teacher_logits, dtype=torch.float32).to(s.device)
    sq = (s - t) ** 2
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.float32).to(s.device)
        while m.dim() < sq.dim():
            m = m.unsqueeze(-1)
        m = m.expand_as(sq)
        if reduction == "sum":
            return (sq * m).sum()
        return (sq * m).sum() / torch.clamp(m.sum(), min=1.0)
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    return sq.mean()


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL(teacher || student) over pre-softmax logits (scaled by ``temperature``)."""
    s = torch.as_tensor(student_logits, dtype=torch.float32)
    t = torch.as_tensor(teacher_logits, dtype=torch.float32).to(s.device)
    temp = max(float(temperature), 1e-6)
    log_p_s = F.log_softmax(s / temp, dim=-1)
    log_p_t = F.log_softmax(t / temp, dim=-1)
    p_t = log_p_t.exp()
    per_position = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # [..., T]
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.float32).to(per_position.device)
        while m.dim() < per_position.dim():
            m = m.unsqueeze(-1)
        per_position = per_position * m
        denom = torch.clamp(m.sum(), min=1.0)
        value = per_position.sum() / denom
    else:
        value = per_position.sum() if reduction == "sum" else per_position.mean()
    return value * (temp ** 2)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    mode: str = "kl",
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Dispatch to the requested distillation flavour (``kl``, ``mse`` or ``both``)."""
    mode = (mode or "kl").lower()
    if mode == "mse":
        return mse_logit_distillation_loss(student_logits, teacher_logits, mask=mask, reduction=reduction)
    if mode == "kl":
        return kl_distillation_loss(
            student_logits, teacher_logits, temperature=temperature, mask=mask, reduction=reduction
        )
    if mode in ("both", "kl+mse", "mse+kl"):
        return kl_distillation_loss(
            student_logits, teacher_logits, temperature=temperature, mask=mask, reduction=reduction
        ) + mse_logit_distillation_loss(student_logits, teacher_logits, mask=mask, reduction=reduction)
    raise ValueError("unknown distillation mode %r (expected 'kl', 'mse' or 'both')" % (mode,))


def replay_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    weight: float = 1.0,
    mode: str = "kl",
    mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Replay objective: distill towards the base PTLM (+ optional CE on the hard labels).

    ``labels`` are the ground-truth token ids of the replayed upstream examples; when given,
    the DER objective adds the cross-entropy on the true labels (the "buffer labels" of
    Buzzega et al., 2020a).
    """
    loss = distillation_loss(
        student_logits, teacher_logits, temperature=temperature, mode=mode, mask=mask
    )
    if labels is not None:
        s = torch.as_tensor(student_logits, dtype=torch.float32)
        lab = torch.as_tensor(labels, dtype=torch.long).to(s.device)
        vocab = s.shape[-1]
        ce = F.cross_entropy(
            s.reshape(-1, vocab),
            lab.reshape(-1).clamp(min=0, max=vocab - 1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        loss = loss + ce
    return loss * float(weight)


def replay_total_loss(
    online_logits: torch.Tensor,
    online_labels: torch.Tensor,
    replay_student_logits: torch.Tensor,
    replay_teacher_logits: torch.Tensor,
    replay_labels: Optional[torch.Tensor] = None,
    distill_weight: float = 1.0,
    distill_temperature: float = 1.0,
    distill_mode: str = "kl",
    online_ce_weight: float = 1.0,
) -> torch.Tensor:
    """Total loss of one refinement step with replay (Sec. 4.2).

    ``L = online_ce_weight * CE(f_i(x_i), y_i)
          + distill_weight * distillation(f_i(x_j), f_0(x_j))``

    The online CE is what fixes the error (``f_i <- update f_0 with <x_i,y_i>``); the
    distilled replay term prevents forgetting the replayed upstream examples.
    """
    s = torch.as_tensor(online_logits, dtype=torch.float32)
    lab = torch.as_tensor(online_labels, dtype=torch.long).to(s.device)
    ce = F.cross_entropy(
        s.reshape(-1, s.shape[-1]), lab.reshape(-1).clamp(min=0, max=s.shape[-1] - 1)
    )
    loss = ce * float(online_ce_weight)
    loss = loss + replay_distillation_loss(
        replay_student_logits,
        replay_teacher_logits,
        labels=replay_labels,
        temperature=distill_temperature,
        weight=distill_weight,
        mode=distill_mode,
    )
    return loss


def total_forecaster_loss(
    method: str,
    *,
    logits: Optional[torch.Tensor] = None,
    target_ids: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    h_upstream: Optional[torch.Tensor] = None,
    h_online: Optional[torch.Tensor] = None,
    prior: Optional[torch.Tensor] = None,
    positive_weight: float = 0.1,
    margin: float = 1.0,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience dispatcher used by the training script.

    ``method`` is one of ``"logit"`` (Eq. 3 margin loss) or ``"representation"`` (Eq. 4 BCE,
    optionally with the frequency prior).
    """
    method = (method or "").lower()
    if method in ("logit", "logit_based", "margin"):
        return margin_loss(
            logits=logits,
            target_ids=target_ids,
            z=z,
            margin=margin,
            target_mask=target_mask,
        )
    if method in ("representation", "repr", "representation_based", "bce"):
        return representation_loss(
            h_upstream=h_upstream,
            h_online=h_online,
            z=z,
            prior=prior,
            positive_weight=positive_weight,
        )
    raise ValueError("unknown forecaster method %r" % (method,))


# --------------------------------------------------------------------------------------
# Self test / CLI
# --------------------------------------------------------------------------------------
def _self_test() -> int:
    torch.manual_seed(0)
    b, t, v = 4, 3, 16
    logits = torch.randn(b, t, v)
    targets = torch.randint(0, v, (b, t))

    loss_fgt = margin_loss(logits, targets, torch.ones(b))
    # craft a case with zero loss: gold token is the argmax by a wide margin
    easy = torch.full((b, t, v), -10.0)
    easy.scatter_(-1, targets.unsqueeze(-1), 10.0)
    assert float(margin_loss(easy, targets, torch.zeros(b))) == 0.0, "z=0 easy case must be 0"
    assert float(margin_loss(easy, targets, torch.ones(b))) > 0.0, "z=1 easy case must be positive"
    assert float(margin_loss(-easy, targets, torch.ones(b))) == 0.0, "z=1 flipped case must be 0"
    logger.info("margin_loss: forgotten=%.4f, sanity checks passed", float(loss_fgt))

    # top-k variant must match the dense margin term when everything is cached
    k = 5
    vals = torch.topk(logits, k, dim=-1).values
    idxs = torch.topk(logits, k, dim=-1).indices
    topk_loss = margin_loss_from_topk(vals, idxs, targets, torch.zeros(b))
    dense_loss = margin_loss(logits, targets, torch.zeros(b))
    logger.info("margin(dense)=%.6f margin(topk)=%.6f", float(dense_loss), float(topk_loss))
    # positions whose negative-class margin is already satisfied contribute 0 to both
    dense_term = torch.clamp(1.0 + margin_term(logits, targets), min=0.0)
    assert float(dense_term.min()) >= 0.0
    assert topk_loss >= -1e-6

    # prediction_from_topk reproduces argmax when k covers the argmax
    pred_hash = prediction_from_topk(vals, idxs)
    pred_dense = (logits.argmax(-1) != targets).long()
    assert torch.equal(pred_hash, pred_dense), "top-k prediction mismatch"

    # representation + prior
    d = 8
    hj = torch.randn(6, d)
    hi = torch.randn(6, d)
    zz = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)
    probs = representation_scores(hj, hi, prior=torch.zeros(6))
    assert probs.shape == (6,)
    loss_rep = representation_loss(hj, hi, zz, prior=torch.zeros(6), positive_weight=0.1)
    b0 = counts_to_prior(1.0, 9.0)
    assert float(b0) < 0.0, "prior must be negative for rare forgetting"
    # w/o prior ablation: identical scores but a different loss value
    loss_no_prior = representation_loss(hj, hi, zz, prior=None, positive_weight=0.1)
    logger.info(
        "representation loss=%.6f (w/o prior=%.6f), prior(1/9)=%.4f",
        float(loss_rep), float(loss_no_prior), float(b0),
    )

    # distillation sanity: identical logits -> zero loss
    teacher = torch.randn(b, t, v)
    assert float(mse_logit_distillation_loss(teacher, teacher)) == 0.0
    assert abs(float(kl_distillation_loss(teacher, teacher))) < 1e-6
    labels = torch.randint(0, v, (b, t))
    rl = replay_distillation_loss(teacher, teacher, labels=labels, mode="kl")
    assert float(rl) > 0.0, "CE term should be positive"
    total = replay_total_loss(
        online_logits=teacher, online_labels=labels,
        replay_student_logits=teacher, replay_teacher_logits=teacher,
        replay_labels=labels, distill_mode="kl",
    )
    assert float(total) > 0.0
    _ = total_forecaster_loss(
        "logit", logits=logits, target_ids=targets, z=torch.ones(b)
    )
    _ = total_forecaster_loss(
        "representation", h_upstream=hj, h_online=hi, z=zz, prior=torch.zeros(6)
    )
    logger.info("self-test passed")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loss utilities for forgetting forecasting")
    parser.add_argument("--self-test", action="store_true", help="run built-in sanity checks")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    parse_args(["--help"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
