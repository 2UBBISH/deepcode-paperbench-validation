"""BaM: Batch and Match black-box variational inference.

This sub-package contains the load-bearing numerical core of the paper
"Batch and Match: Black-Box Variational Inference with a Score-Based
Divergence":

* :mod:`bam.matrix_equations` -- closed-form solvers of the quadratic matrix
  equation ``X U X + X = V`` (Appendix B, Lemmas B.1--B.3).
* :mod:`bam.vi_base` -- the shared full-covariance Gaussian variational state
  (Section 2.2).
* :mod:`bam.bam` -- Algorithm 1 (batch step + match step).
* :mod:`bam.learning_rate` -- schedules for the inverse regularization
  ``lambda_t`` (Section 3.1, 5.1).
"""

from __future__ import annotations

from .matrix_equations import (
    ensure_spd,
    inverse_spd,
    matrix_sqrt,
    matrix_sqrt_inv,
    residual,
    solve_quadratic_matrix_equation,
    solve_quadratic_matrix_equation_dense,
    solve_quadratic_matrix_equation_low_rank,
    symmetrize,
)

from .vi_base import (
    GaussianVariational,
    batch_covariance,
    batch_mean,
    gaussian_entropy,
    gaussian_kl,
    gaussian_log_density,
    gaussian_score,
    init_gaussian_state,
    project_spd,
    reparameterize,
    standard_normal,
)

from .bam import (
    BaM,
    BaMResult,
    BatchStatistics,
    MatchStepResult,
    bam_fit,
    bam_match_step,
    batch_statistics,
    empirical_divergence,
    low_rank_factor,
    match_step,
    resolve_lambda,
)

from .learning_rate import (
    DEFAULT_GAUSSIAN_SCHEDULE,
    DEFAULT_NON_GAUSSIAN_SCHEDULE,
    Schedule,
    b_over_t_schedule,
    b_schedule,
    bd_over_sqrt_t_schedule,
    bd_over_t_schedule,
    bd_schedule,
    constant_schedule,
    decay_schedule,
    make_schedule,
    resolve_schedule,
    schedule_values,
)

__all__ = [
    # matrix_equations
    "ensure_spd",
    "inverse_spd",
    "matrix_sqrt",
    "matrix_sqrt_inv",
    "residual",
    "solve_quadratic_matrix_equation",
    "solve_quadratic_matrix_equation_dense",
    "solve_quadratic_matrix_equation_low_rank",
    "symmetrize",
    # vi_base
    "GaussianVariational",
    "batch_covariance",
    "batch_mean",
    "gaussian_entropy",
    "gaussian_kl",
    "gaussian_log_density",
    "gaussian_score",
    "init_gaussian_state",
    "project_spd",
    "reparameterize",
    "standard_normal",
    # bam
    "BaM",
    "BaMResult",
    "BatchStatistics",
    "MatchStepResult",
    "bam_fit",
    "bam_match_step",
    "batch_statistics",
    "empirical_divergence",
    "low_rank_factor",
    "match_step",
    "resolve_lambda",
    # learning_rate
    "DEFAULT_GAUSSIAN_SCHEDULE",
    "DEFAULT_NON_GAUSSIAN_SCHEDULE",
    "Schedule",
    "b_over_t_schedule",
    "b_schedule",
    "bd_over_sqrt_t_schedule",
    "bd_over_t_schedule",
    "bd_schedule",
    "constant_schedule",
    "decay_schedule",
    "make_schedule",
    "resolve_schedule",
    "schedule_values",
]

__version__ = "0.1.0"
