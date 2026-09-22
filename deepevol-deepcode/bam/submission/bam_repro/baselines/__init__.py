"""Baseline variational inference algorithms used to compare against BaM.

This package aggregates the four gradient-based / score-based baselines that the
paper compares against Batch and Match (BaM):

* ``ADVI``        -- Algorithm 2 from the paper: reparameterized full-covariance
  Gaussian variational inference minimizing the negative ELBO with ADAM.
* ``ScoreADVI``   -- the same family + optimizer, but with the empirical
  score-based (weighted Fisher) divergence of eqs. (93)-(98) as the loss.
* ``FisherADVI``  -- the same family + optimizer, but minimizing the unweighted
  Fisher divergence ``E_q[||grad log q - grad log p||^2]`` (M = I).
* ``GSM``         -- Algorithm 3 from the paper: Gaussian Score Matching
  (Modi et al., 2023), a per-sample score-based update.

The shared machinery (full-covariance Gaussian variational family, reparameterized
sampling, ADAM optimizer, result container) lives in :mod:`bam_repro.baselines.advi`
and is reused by the two ADVI variants.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared gradient-based VI engine + ADVI (Algorithm 2)
# ---------------------------------------------------------------------------
from .advi import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    ADVI,
    LOSSES,
    ADVIResult,
    GradientVI,
    GradientVIResult,
    advi_fit,
)

# ---------------------------------------------------------------------------
# Score-ADVI (score-based divergence loss)
# ---------------------------------------------------------------------------
from .score_advi import (
    DEFAULT_SCORE_LR,
    PAPER_GAUSSIAN_SCORE_LR,
    ScoreADVI,
    ScoreADVIResult,
    paper_score_learning_rate,
    score_advi_fit,
    score_divergence_batch_form,
    score_divergence_estimate,
    score_divergence_from_stats,
    score_loss,
)

# ---------------------------------------------------------------------------
# Fisher-ADVI (unweighted Fisher divergence loss)
# ---------------------------------------------------------------------------
from .fisher_advi import (
    DEFAULT_FISHER_LR,
    FISHER_LR_GRID,
    PAPER_GAUSSIAN_FISHER_LR,
    PAPER_NONGaussian_FISHER_LR,
    FisherADVI,
    FisherADVIResult,
    fisher_advi_fit,
    fisher_divergence_batch_form,
    fisher_divergence_estimate,
    fisher_divergence_from_stats,
    fisher_divergence_mc,
    fisher_loss,
    gaussian_fisher_divergence,
    grid_search_fisher,
    make_fisher_advi,
    paper_fisher_learning_rate,
)

# ---------------------------------------------------------------------------
# GSM (Algorithm 3)
# ---------------------------------------------------------------------------
from .gsm import (
    GSM,
    GSMResult,
    gsm_batch_update,
    gsm_divergence,
    gsm_fit,
    gsm_sample_update,
    gsm_step,
    solve_rho,
)

__all__ = [
    # shared engine
    "GradientVI",
    "GradientVIResult",
    "LOSSES",
    "ADAM_BETA1",
    "ADAM_BETA2",
    "ADAM_EPS",
    # ADVI
    "ADVI",
    "ADVIResult",
    "advi_fit",
    # Score-ADVI
    "ScoreADVI",
    "ScoreADVIResult",
    "score_advi_fit",
    "score_divergence_estimate",
    "score_divergence_from_stats",
    "score_divergence_batch_form",
    "score_loss",
    "paper_score_learning_rate",
    "PAPER_GAUSSIAN_SCORE_LR",
    "DEFAULT_SCORE_LR",
    # Fisher-ADVI
    "FisherADVI",
    "FisherADVIResult",
    "fisher_advi_fit",
    "make_fisher_advi",
    "fisher_divergence_estimate",
    "fisher_divergence_from_stats",
    "fisher_divergence_batch_form",
    "fisher_divergence_mc",
    "fisher_loss",
    "gaussian_fisher_divergence",
    "grid_search_fisher",
    "paper_fisher_learning_rate",
    "FISHER_LR_GRID",
    "PAPER_GAUSSIAN_FISHER_LR",
    "PAPER_NONGaussian_FISHER_LR",
    "DEFAULT_FISHER_LR",
    # GSM
    "GSM",
    "GSMResult",
    "gsm_fit",
    "gsm_step",
    "gsm_batch_update",
    "gsm_sample_update",
    "gsm_divergence",
    "solve_rho",
]

__version__ = "0.1.0"
