"""Core classifier-free guidance math for autoregressive language models.

The paper applies CFG in *logit space*:

    log P_hat(w_i | w_{<i}, c) = log P(w_i | w_{<i})
        + gamma * ( log P(w_i | w_<i, c) - log P(w_i | w_<i) )      (Eq. 7)

Because ``log P(w_i | .) = logits - logsumexp(logits)`` and the
``logsumexp`` term is a constant with respect to the vocabulary index at
a given decoding step, Equation 7 is equivalent to the logit mixture

    logits_cfg = (1 - gamma) * logits_uncond + gamma * logits_cond

which is what is implemented here.  ``softmax(logits_cfg)`` is the
gamma-reweighted next-token distribution ``P_hat(w_i | w_{<i}, c)``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def guidance_weight(gamma: float) -> float:
    """Validate and return the guidance strength ``gamma``.

    ``gamma = 1`` recovers vanilla (conditional) sampling, ``gamma = 0``
    recovers the unconditional distribution, and ``gamma > 1``
    over-emphasises the conditioning, exactly as in text-to-image CFG.
    """
    if gamma is None or (isinstance(gamma, float) and math.isnan(gamma)):
        raise ValueError("gamma must be a finite number")
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    return float(gamma)


def cfg_combine_logits(
    logits_cond: torch.Tensor,
    logits_uncond: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Combine conditional/unconditional logits with classifier-free guidance.

    Implements ``(1 - gamma) * logits_uncond + gamma * logits_cond``, i.e.
    a step of size ``gamma`` away from the unconditional vector in the
    direction of the conditional vector (Eq. 4/7 of the paper).

    Args:
        logits_cond: ``[..., vocab]`` conditional (prompted) logits.
        logits_uncond: ``[..., vocab]`` unconditional (or negatively
            prompted) logits.  Must broadcast against ``logits_cond``.
        gamma: guidance strength.

    Returns:
        A tensor with the same shape as ``logits_cond`` in ``float32``.
    """
    gamma = guidance_weight(gamma)
    cond = logits_cond.float()
    uncond = logits_uncond.float()
    return uncond + gamma * (cond - uncond)


def cfg_combine_logprobs(
    logprobs_cond: torch.Tensor,
    logprobs_uncond: torch.Tensor,
    gamma: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Combine *log-probabilities* with CFG (literal Eq. 7).

    This is the log-space form of :func:`cfg_combine_logits`; the two are
    equivalent after a softmax because they differ only by a per-step
    additive constant.  The equivalence is asserted in the unit tests.

    Args:
        logprobs_cond: ``[..., vocab]`` conditional log-probabilities
            (i.e. ``log_softmax(logits_cond)``).
        logprobs_uncond: ``[..., vocab]`` unconditional log-probabilities.
        gamma: guidance strength.
        normalize: if ``True`` (default) re-normalise with a log-softmax
            so that the result is a valid log-probability vector.
    """
    gamma = guidance_weight(gamma)
    cond = logprobs_cond.float()
    uncond = logprobs_uncond.float()
    out = uncond + gamma * (cond - uncond)
    if normalize:
        out = F.log_softmax(out, dim=-1)
    return out


def cfg_logits_from_logprobs(
    logprobs_cond: torch.Tensor,
    logprobs_uncond: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Recover CFG *logits* from conditional/unconditional log-probs."""
    return cfg_combine_logprobs(logprobs_cond, logprobs_uncond, gamma, normalize=False)


def token_cfg_logprobs(
    logits_cond: torch.Tensor,
    logits_uncond: torch.Tensor,
    gamma: float,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token CFG log-probabilities.

    Convenience wrapper used by the likelihood-based scoring code: given
    the raw logits of the conditional and unconditional branches for the
    same set of continuation tokens, return ``log P_hat`` either for the
    whole vocabulary or (if ``targets`` is given) for the specific target
    token ids.

    Args:
        logits_cond: ``[n_tokens, vocab]`` conditional logits.
        logits_uncond: ``[n_tokens, vocab]`` unconditional logits.
        gamma: guidance strength.
        targets: optional ``[n_tokens]`` tensor of target token ids.
    """
    logits = cfg_combine_logits(logits_cond, logits_uncond, gamma)
    logprobs = F.log_softmax(logits, dim=-1)
    if targets is None:
        return logprobs
    idx = targets.reshape(-1, 1)
    return logprobs.gather(-1, idx).squeeze(-1)


def effective_temperature(temperature: float, gamma: float) -> float:
    """Temperature that would (approximately) match a CFG step.

    Provided for the discussion in Section 5.1 of the paper: CFG is *not*
    equivalent to temperature scaling because it re-orders the top tokens,
    but this gives a useful reference point.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return float(temperature) / max(gamma, 1e-8)


def interpolate_contexts(
    contexts: Sequence[str],
) -> str:  # pragma: no cover - documentary helper
    """Placeholder documenting the multi-context extension.

    Section 3.2 notes that one could upweight ``w_p, w_cot`` jointly; in
    CFG terms this corresponds to choosing a richer conditional context
    ``c`` (or, in future work, a combination of several CFG passes).
    """
    return "\n".join(contexts)
