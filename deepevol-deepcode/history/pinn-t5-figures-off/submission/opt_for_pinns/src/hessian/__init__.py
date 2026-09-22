"""Hessian utilities for PINN loss-landscape analysis.

Exposes:
    - hvp: matrix-free Hessian-vector products (Pearlmutter trick)
    - spectral_density: SLQ-based spectral density estimation
    - lbfgs_unroll: L-BFGS preconditioner unrolling (Algorithm 2)
    - precond_matvec: preconditioned Hessian matvec (Algorithm 3)
"""

from .hvp import hvp, hvp_from_grad, HVPOperator, flat_grad
from .spectral_density import (
    spectral_density,
    SpectralDensityEstimator,
    slq_spectral_density,
    lanczos_tridiag,
    gaussian_smoothing,
)
from .lbfgs_unroll import (
    LBFGSHistory,
    unroll_lbfgs,
    build_preconditioner_columns,
    lbfgs_two_loop,
    apply_lbfgs_preconditioner,
)
from .precond_matvec import (
    PreconditionedHessianOperator,
    preconditioned_hessian_matvec,
    build_preconditioned_matvec,
)

__all__ = [
    "hvp",
    "hvp_from_grad",
    "HVPOperator",
    "flat_grad",
    "spectral_density",
    "SpectralDensityEstimator",
    "slq_spectral_density",
    "lanczos_tridiag",
    "gaussian_smoothing",
    "LBFGSHistory",
    "unroll_lbfgs",
    "build_preconditioner_columns",
    "lbfgs_two_loop",
    "apply_lbfgs_preconditioner",
    "PreconditionedHessianOperator",
    "preconditioned_hessian_matvec",
    "build_preconditioned_matvec",
]
