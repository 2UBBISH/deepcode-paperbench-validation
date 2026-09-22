"""Core Classifier-Free Guidance (CFG) logit combination for language models.

Implements the *paper's* central operation, Equation 7 of "Stay on Topic with
Classifier-Free Guidance" (Sanchez et al., 2023)::

    log P_hat( w_i | w_{j<i}, c )
        = log P_theta( w_i | w_{j<i} )
          + gamma * ( log P_theta( w_i | w_{j<i}, c ) - log P_theta( w_i | w_{j<i} ) )   (7)

Read as a vector operation on the *logits* of the next-token prediction (the paper
explicitly applies CFG "to the logits of next-token predictions", Section 2.2), this is

    guided = uncond + gamma * (cond - uncond)

with the standard limiting behaviour

    gamma = 1  ->  guided == cond        (vanilla conditional generation)
    gamma = 0  ->  guided == uncond      (unconditional generation)

The *negative prompting* extension of Section 2.1, Equation 5, replaces the
unconditional term ``log P_theta(w_i | w_{j<i})`` by a pass conditioned on a negative
prompt ``c_bar``::

    log P_hat( w_i | w_{j<i}, c, c_bar )
        = log P_theta( w_i | w_{j<i}, c_bar )
          + gamma * ( log P_theta( w_i | w_{j<i}, c ) - log P_theta( w_i | w_{j<i}, c_bar ) )  (5)

i.e. exactly the same linear combination, only the "unconditional" anchor changes.  This
module is therefore prompt-agnostic: it only ever sees the two logit tensors.

Everything here is *training-free* and operates purely on pre-softmax logits, so it is
architecture agnostic and can be dropped into any HuggingFace ``AutoModelForCausalLM``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

try:  # torch is required for the model wrapper / generation paths, but not for the math
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only in torch-free environments
    torch = None  # type: ignore
    _HAS_TORCH = False


ArrayLike = Union["torch.Tensor", np.ndarray]


def _is_torch(x) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def cfg_combine(logits_uncond: ArrayLike, logits_cond: ArrayLike, gamma: float) -> ArrayLike:
    """Apply Equation 7 / Equation 5 of the paper to a pair of next-token logits.

    Parameters
    ----------
    logits_uncond:
        ``log P_theta(w_i | w_{j<i})`` (prefix-dropped pass) -- or, with negative
        prompting (Eq. 5), the negative-prompt pass ``log P_theta(w_i | w_{j<i}, c_bar)``.
        Shape ``[..., vocab]``.
    logits_cond:
        ``log P_theta(w_i | w_{j<i}, c)`` -- the conditional pass with the prompt ``c``.
        Same shape as ``logits_uncond``.
    gamma:
        Guidance strength. Broadcast over the leading (batch/beam) dimensions.
        ``gamma == 1`` returns the conditional logits unchanged; ``gamma == 0``
        returns the unconditional logits unchanged.

    Returns
    -------
    The guided logits, delta-free and in the same space (raw pre-softmax logits),
    identical in dtype/shape to the inputs.
    """
    if _is_torch(logits_cond) or _is_torch(logits_uncond):
        if not (_is_torch(logits_cond) and _is_torch(logits_uncond)):
            raise TypeError("cfg_combine: both logits must be the same backend (torch or numpy)")
        if logits_cond.shape != logits_uncond.shape:
            raise ValueError(
                "cfg_combine: shape mismatch %s vs %s"
                % (tuple(logits_cond.shape), tuple(logits_uncond.shape))
            )
        g = torch.as_tensor(gamma, dtype=logits_cond.dtype, device=logits_cond.device)
        # Broadcast scalar (or [batch] / [batch, 1]) gamma over the [batch, vocab] tensor.
        return logits_uncond + g * (logits_cond - logits_uncond)

    cond = np.asarray(logits_cond)
    uncond = np.asarray(logits_uncond)
    if cond.shape != uncond.shape:
        raise ValueError(
            "cfg_combine: shape mismatch %s vs %s" % (cond.shape, uncond.shape)
        )
    g = np.asarray(gamma, dtype=cond.dtype)
    return uncond + g * (cond - uncond)


def guided_logits(
    logits_cond: ArrayLike,
    logits_uncond: Optional[ArrayLike] = None,
    gamma: float = 1.0,
) -> ArrayLike:
    """Convenience wrapper around :func:`cfg_combine`.

    ``logits_uncond=None`` means "no CFG" (a plain conditional forward pass) and the
    conditional logits are returned as-is, mirroring the identity that ``gamma == 1``
    recovers ordinary sampling.
    """
    if logits_uncond is None or gamma == 1.0:
        return logits_cond
    return cfg_combine(logits_uncond, logits_cond, gamma)


def negative_prompt_logits(
    logits_cond: ArrayLike,
    logits_negative: ArrayLike,
    gamma: float = 1.0,
) -> ArrayLike:
    """Equation 5: CFG whose "unconditional" anchor is a negative prompt ``c_bar``.

    Semantically distinct from :func:`cfg_combine` (in our codebase ``logits_uncond``
    already carries whichever anchor we chose), but kept as an explicit, named entry
    point so the negative-prompting experiments of Section 3.4 read like the paper.
    """
    return cfg_combine(logits_negative, logits_cond, gamma)


def log_softmax(x: ArrayLike, axis: int = -1) -> ArrayLike:
    """Numerically-stable log-softmax used by the entropy/overlap analyses.

    CFG is applied in logit space *before* any normalisation, so this helper is only
    used to turn post-CFG logits into ``log P`` when we need probabilities.
    """
    if _is_torch(x):
        return torch.log_softmax(x, dim=axis)
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def top_p_filter(logits: ArrayLike, top_p: float) -> ArrayLike:
    """Nucleus (top-p) filtering applied to *logits*, exactly as the HF sampler does.

    Tokens whose cumulative probability mass exceeds ``top_p`` are set to ``-inf``.
    Sorting is descending by probability, keeping the smallest prefix of tokens whose
    cumulative mass is ``>= top_p`` (the first token beyond the nucleus is removed as
    well); ties are broken by index, matching HuggingFace's implementation.
    """
    if top_p is None or top_p >= 1.0:
        return logits

    if _is_torch(logits):
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold; keep the
        # first token that crosses top_p (shift by one like HF transformers).
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        filtered = sorted_logits.masked_fill(mask, float("-inf"))
        out = torch.empty_like(filtered).scatter_(-1, sorted_idx, filtered)
        return out

    logits = np.asarray(logits, dtype=np.float64)
    idx = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, idx, axis=-1)
    shifted = sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True)
    p = np.exp(shifted)
    p = p / np.sum(p, axis=-1, keepdims=True)
    cumulative = np.cumsum(p, axis=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].copy()
    mask[..., 0] = False
    sorted_logits = np.where(mask, -np.inf, sorted_logits)
    out = np.empty_like(sorted_logits)
    np.put_along_axis(out, idx, sorted_logits, axis=-1)
    return out


def softmax(logits: ArrayLike, axis: int = -1) -> ArrayLike:
    """Plain softmax over the vocabulary dimension."""
    if _is_torch(logits):
        return torch.softmax(logits, dim=axis)
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
