"""Metrics package for the BaM reproduction.

Aggregates the metric helpers used to evaluate BaM and the baselines against the
synthetic targets (Gaussian / sinh-arcsinh) and the real PosteriorDB targets.

Two metric families are provided:

* :mod:`bam_repro.metrics.kl_metrics` -- forward ``KL(p; q)`` and reverse
  ``KL(q; p)`` divergences for the synthetic targets (closed form for Gaussian
  targets, Monte-Carlo otherwise) plus helpers that turn a variational trace
  into the ``(gradient evaluations, KL)`` curves of Figures 5.1/5.2 (E.3/E.4).
* :mod:`bam_repro.metrics.posteriordb_metrics` -- the relative mean error
  ``|| (mu - mu_hat) / sigma ||_2`` and relative standard-deviation error
  ``|| (sigma - sigma_hat) / sigma ||_2`` against HMC reference samples used in
  Section 5.2 / Figures 5.3 and E.6, plus reconstruction error helpers.

The module re-exports the whole public surface so consumers can simply write
``from bam_repro.metrics import kl_divergence, relative_mean_error``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# KL metrics (always available)
# ---------------------------------------------------------------------------
from .kl_metrics import (
    DEFAULT_KL_SAMPLES,
    KLResult,
    empirical_log_ratio,
    forward_kl,
    gaussian_forward_kl,
    gaussian_reverse_kl,
    has_closed_form_kl,
    is_gaussian_target,
    kl_curve,
    kl_divergence,
    kl_from_fit,
    kl_pair,
    mc_forward_kl,
    mc_reverse_kl,
    reverse_kl,
)

__all__ = [
    # ---- kl_metrics -------------------------------------------------------
    "DEFAULT_KL_SAMPLES",
    "KLResult",
    "empirical_log_ratio",
    "forward_kl",
    "gaussian_forward_kl",
    "gaussian_reverse_kl",
    "has_closed_form_kl",
    "is_gaussian_target",
    "kl_curve",
    "kl_divergence",
    "kl_from_fit",
    "kl_pair",
    "mc_forward_kl",
    "mc_reverse_kl",
    "reverse_kl",
]

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# PosteriorDB metrics (optional import: the module needs reference samples and
# is not required for the synthetic experiments)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised indirectly when the module exists
    from .posteriordb_metrics import (  # type: ignore
        PAPER_POSTERIORDB_MODELS,
        error_curve,
        relative_mean_error,
        relative_sd_error,
        relative_errors,
        reconstruction_mse,
        summarize_errors,
    )
except Exception:  # pragma: no cover - defensive: keep package importable
    pass
else:  # pragma: no cover - only when the module is present
    __all__ += [
        "PAPER_POSTERIORDB_MODELS",
        "error_curve",
        "relative_mean_error",
        "relative_sd_error",
        "relative_errors",
        "reconstruction_mse",
        "summarize_errors",
    ]
