"""Statistical utilities used across the paper's analyses.

* :func:`entropy_from_logits` -- Section 5.1 (mean sampling entropy).
  This is ``H(p) = -sum_k p_k log p_k`` in nats, i.e. exactly what the
  paper's ``scipy.stats.entropy`` call computes (the vectorised form here
  avoids a Python loop over tokens).
* :func:`top_p_overlap` -- Section 5.2 / Appendix E (top-p = 90 % overlap).
* :func:`spearman_correlation` -- Section 5.2 (``r_s > .7`` correlations).
* :func:`estimate_pass_at_k` -- Section 3.3.1 (unbiased pass@k estimator of
  Chen et al., 2021).
* :func:`ancova` -- Section 4.1 / Appendix C.2 (ANCOVA p-values).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency, present in the repro env
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


# ----------------------------------------------------------------------
# Entropy (Section 5.1)
# ----------------------------------------------------------------------
def entropy_from_logits(logits: torch.Tensor, normalized_probs: bool = False) -> torch.Tensor:
    """Shannon entropy ``H(p) = -sum_k p_k log p_k`` of a logit distribution.

    Args:
        logits: ``[..., vocab]`` logits (or probabilities when
            ``normalized_probs`` is ``True``).
        normalized_probs: interpret the input as probabilities rather than
            logits.

    Returns:
        ``[...]`` tensor of entropies in nats.
    """
    logits = logits.float()
    if normalized_probs:
        probs = logits
    else:
        probs = F.softmax(logits, dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    return -(probs * logp).sum(dim=-1)


def mean_entropy_per_token(logprobs: torch.Tensor) -> float:
    """Mean entropy over completion tokens (as defined in the addendum).

    ``logprobs`` is ``[n_tokens, vocab]`` of log-probabilities; the returned
    value is ``(1/n) sum_i H(p(x_i | x_<i))``.
    """
    if logprobs.numel() == 0:
        return float("nan")
    return float(entropy_from_logits(logprobs, normalized_probs=False).mean())


# ----------------------------------------------------------------------
# Top-p overlap (Section 5.2 / Appendix E)
# ----------------------------------------------------------------------
def top_p_token_set(logprobs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Indices of the smallest set of tokens covering ``p`` probability mass.

    The paper's addendum notes this is the number of tokens one would sample
    from when drawing from the top-``p`` truncation of the distribution.
    """
    probs = logprobs.exp() if float(logprobs.max()) <= 0 else F.softmax(logprobs, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    mask = cumulative >= p
    cut = torch.argmax(mask.to(torch.int32), dim=-1)
    keep = torch.zeros_like(sorted_probs, dtype=torch.bool)
    keep[..., : cut + 1] = True
    return sorted_idx[keep]


def top_p_overlap(logprobs_a: torch.Tensor, logprobs_b: torch.Tensor, p: float = 0.9) -> int:
    """Size of the intersection of the two top-``p`` token sets (Appendix E)."""
    set_a = set(top_p_token_set(logprobs_a, p).tolist())
    set_b = set(top_p_token_set(logprobs_b, p).tolist())
    return len(set_a & set_b)


def top_p_overlap_fraction(
    logprobs_a: torch.Tensor, logprobs_b: torch.Tensor, p: float = 0.9
) -> float:
    """Jaccard-style overlap of the two top-``p`` sets, in ``[0, 1]``."""
    set_a = set(top_p_token_set(logprobs_a, p).tolist())
    set_b = set(top_p_token_set(logprobs_b, p).tolist())
    union = len(set_a | set_b)
    return float(len(set_a & set_b) / union) if union else 0.0


def top_k_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int = 10) -> int:
    """Number of shared tokens between the two top-``k`` sets."""
    top_a = set(torch.topk(logits_a.float(), k, dim=-1).indices.tolist())
    top_b = set(torch.topk(logits_b.float(), k, dim=-1).indices.tolist())
    return len(top_a & top_b)


# ----------------------------------------------------------------------
# Correlations (Section 5.2)
# ----------------------------------------------------------------------
def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Spearman rank correlation and its two-sided p-value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if scipy_stats is not None:
        res = scipy_stats.spearmanr(x, y)
        return float(res.statistic), float(res.pvalue)
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties share the mean rank)."""
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # handle ties by averaging
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1
    return ranks


# ----------------------------------------------------------------------
# pass@k (Section 3.3.1)
# ----------------------------------------------------------------------
def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator of Chen et al. (2021).

    ``pass@k = 1 - C(n - c, k) / C(n, k)`` where ``n`` samples were drawn
    and ``c`` of them are correct.
    """
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k(n: int, c: int, ks: Iterable[int]) -> Dict[int, float]:
    """Return ``{k: pass@k}`` for one problem."""
    return {k: estimate_pass_at_k(n, c, k) for k in ks}


def aggregate_pass_at_k(
    n_samples: Sequence[int], n_correct: Sequence[int], ks: Iterable[int]
) -> Dict[int, float]:
    """Average :func:`estimate_pass_at_k` over a benchmark's problems."""
    out: Dict[int, float] = {}
    for k in ks:
        vals = [estimate_pass_at_k(n, c, k) for n, c in zip(n_samples, n_correct)]
        out[k] = float(np.mean(vals))
    return out


# ----------------------------------------------------------------------
# ANCOVA (Section 4.1 / Appendix C.2)
# ----------------------------------------------------------------------
def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = X.shape[0] - X.shape[1]
    sigma2 = rss / dof if dof > 0 else float("nan")
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:  # pragma: no cover
        cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return beta, se, rss, dof


def ancova(
    x: Sequence[float],
    y: Sequence[float],
    group: Sequence[int],
    interaction: bool = False,
) -> Dict[str, float]:
    """ANCOVA of ``y`` on a covariate ``x`` with a two-level factor.

    The paper compares the regression lines of the vanilla (``group=0``) and
    CFG (``group=1``) groups when both are expressed as a function of
    inference FLOPs per token, with the covariate log-transformed and
    significance set at ``p = .01`` (Table 6).

    With ``interaction=False`` (the default) we fit

        y = b0 + b1 * x + b2 * group + e

    and report the p-value of ``b2`` -- the classic ANCOVA test of an
    adjusted group difference.  With ``interaction=True`` the model becomes

        y = b0 + b1 * x + b2 * group + b3 * (x * group) + e

    and we additionally report an F-test of the nested comparison against
    the no-interaction model, testing whether the two regression *lines*
    differ.

    Returns a dict with ``p_value`` (group effect), ``coef_group``,
    ``t_stat``, an optional ``p_value_interaction`` / ``f_stat`` and the
    residual degrees of freedom.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(group, dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("ANCOVA needs at least 4 observations")

    def design(cols):
        return np.column_stack([np.ones(n)] + cols)

    X_base = design([x, g])
    beta, se, rss_base, dof = _ols(X_base, y)
    t_stat = beta[2] / se[2] if se[2] > 0 else float("nan")
    if scipy_stats is not None and math.isfinite(t_stat):
        p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), dof)))
    else:  # pragma: no cover
        p_value = float("nan")

    out: Dict[str, float] = {
        "p_value": p_value,
        "t_stat": float(t_stat),
        "coef_group": float(beta[2]),
        "coef_covariate": float(beta[1]),
        "intercept": float(beta[0]),
        "resid_dof": float(dof),
        "rss": rss_base,
    }

    if interaction:
        X_full = design([x, g, x * g])
        beta_f, se_f, rss_full, dof_f = _ols(X_full, y)
        # nested F-test: base vs full (2 extra params are group + interaction
        # in the full model relative to the covariate-only model)
        X_cov = design([x])
        _, _, rss_cov, _ = _ols(X_cov, y)
        df_num = 2
        df_den = dof_f
        if df_den > 0 and rss_full > 0:
            f_stat = max(((rss_cov - rss_full) / df_num) / (rss_full / df_den), 0.0)
        else:  # pragma: no cover
            f_stat = float("nan")
        if scipy_stats is not None and math.isfinite(f_stat):
            p_int = float(1 - scipy_stats.f.cdf(f_stat, df_num, df_den))
        else:  # pragma: no cover
            p_int = float("nan")
        out.update(
            {
                "p_value_interaction": p_int,
                "f_stat": float(f_stat),
                "coef_interaction": float(beta_f[3]),
            }
        )
    return out


def significance_label(p_value: float, cutoff: float = 0.01) -> str:
    """Map a p-value to the paper's Table 6 convention."""
    return "p>.01" if p_value > cutoff else "significant"
