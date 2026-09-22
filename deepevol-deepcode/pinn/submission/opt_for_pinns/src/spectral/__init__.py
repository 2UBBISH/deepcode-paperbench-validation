"""Spectral analysis subpackage for the PINN loss-landscape reproduction.

This package groups the matrix-free second-order / spectral machinery used to
study the Hessian of the PINN loss ``L(w)`` (Eq. (2) of the paper) and its
L-BFGS-preconditioned counterpart ``Htilde_k^T H_L(w) Htilde_k``:

* :mod:`src.spectral.hvp` -- Pearlmutter-style Hessian-vector products
  (``H_L(w) v``) plus flat-parameter helpers and cheap spectral diagnostics.
* :mod:`src.spectral.lbfgs_unroll` -- Algorithm 2 (Appendix C.2): unroll the
  L-BFGS history ``{y_i, s_i, rho_i}`` into the explicit factors
  ``Ytilde``, ``Vtilde``, ``Stilde`` such that ``H_k = Htilde_k Htilde_k^T``.
* :mod:`src.spectral.preconditioned_mvp` -- Algorithm 3 (Appendix C.2):
  matrix-vector products with ``Htilde_k``, so that stochastic Lanczos
  quadrature can estimate the spectrum of ``Htilde_k^T H_L Htilde_k``.
* :mod:`src.spectral.spectral_density` -- stochastic Lanczos quadrature (SLQ)
  density estimation for ``H_L`` and for the preconditioned operator,
  optionally per loss component (residual / initial / boundary).

The four modules have no circular dependencies: ``hvp`` is self-contained,
``lbfgs_unroll`` depends on it only through flat-vector helpers, and
``preconditioned_mvp`` / ``spectral_density`` build on both.

All heavy symbols are re-exported here with a relative-import ``try``/``except``
fallback to the ``src.spectral.*`` absolute path (so the code works whether the
project root or the package directory is on ``sys.path``).  The optional
sub-modules are imported defensively: if one of them has not been written yet
(or a third-party dependency for it is missing) the package still imports and
the remaining symbols stay available.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hessian-vector products and flat-parameter utilities (always available).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import shim
    from .hvp import (
        DEFAULT_DTYPE,
        LossClosure,
        FlatLossClosure,
        ParamLike,
        VecLike,
        params_of,
        num_parameters,
        param_shapes,
        flatten_tensors,
        flatten_params,
        unflatten,
        unflatten_like,
        set_flat_params,
        cast_model_dtype,
        grad_flat,
        loss_and_grad,
        hessian_vector_product,
        hvp,
        HVP,
        loss_grad_hvp,
        hvp_batched,
        hessian_matrix,
        hessian_diagonal_hutchinson,
        trace_hutchinson,
        LinearOperator,
        HessianOperator,
        HessianVectorProduct,
        LossHessianOracle,
        as_operator,
        power_iteration,
        spectral_norm,
        top_eigenvalues,
    )
except ImportError:  # pragma: no cover - fallback for absolute imports
    from src.spectral.hvp import (  # type: ignore[no-redef]
        DEFAULT_DTYPE,
        LossClosure,
        FlatLossClosure,
        ParamLike,
        VecLike,
        params_of,
        num_parameters,
        param_shapes,
        flatten_tensors,
        flatten_params,
        unflatten,
        unflatten_like,
        set_flat_params,
        cast_model_dtype,
        grad_flat,
        loss_and_grad,
        hessian_vector_product,
        hvp,
        HVP,
        loss_grad_hvp,
        hvp_batched,
        hessian_matrix,
        hessian_diagonal_hutchinson,
        trace_hutchinson,
        LinearOperator,
        HessianOperator,
        HessianVectorProduct,
        LossHessianOracle,
        as_operator,
        power_iteration,
        spectral_norm,
        top_eigenvalues,
    )

__all__ = [
    # hvp.py
    "DEFAULT_DTYPE",
    "LossClosure",
    "FlatLossClosure",
    "ParamLike",
    "VecLike",
    "params_of",
    "num_parameters",
    "param_shapes",
    "flatten_tensors",
    "flatten_params",
    "unflatten",
    "unflatten_like",
    "set_flat_params",
    "cast_model_dtype",
    "grad_flat",
    "loss_and_grad",
    "hessian_vector_product",
    "hvp",
    "HVP",
    "loss_grad_hvp",
    "hvp_batched",
    "hessian_matrix",
    "hessian_diagonal_hutchinson",
    "trace_hutchinson",
    "LinearOperator",
    "HessianOperator",
    "HessianVectorProduct",
    "LossHessianOracle",
    "as_operator",
    "power_iteration",
    "spectral_norm",
    "top_eigenvalues",
]


# ---------------------------------------------------------------------------
# Algorithm 2: unrolled L-BFGS factors.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional until the module is written
    try:
        from .lbfgs_unroll import (  # type: ignore[import-not-found]
            LBFGSUnroll,
            UnrolledLBFGS,
            unroll_lbfgs,
            lbfgs_factors,
            LBFGSFactors,
        )
    except ImportError:
        from src.spectral.lbfgs_unroll import (  # type: ignore[no-redef]
            LBFGSUnroll,
            UnrolledLBFGS,
            unroll_lbfgs,
            lbfgs_factors,
            LBFGSFactors,
        )

    __all__ += [
        "LBFGSUnroll",
        "UnrolledLBFGS",
        "unroll_lbfgs",
        "lbfgs_factors",
        "LBFGSFactors",
    ]
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Algorithm 3: preconditioned matrix-vector products.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional until the module is written
    try:
        from .preconditioned_mvp import (  # type: ignore[import-not-found]
            PreconditionedHessian,
            preconditioned_mvp,
            preconditioned_matvec,
            PreconditionedMVP,
        )
    except ImportError:
        from src.spectral.preconditioned_mvp import (  # type: ignore[no-redef]
            PreconditionedHessian,
            preconditioned_mvp,
            preconditioned_matvec,
            PreconditionedMVP,
        )

    __all__ += [
        "PreconditionedHessian",
        "preconditioned_mvp",
        "preconditioned_matvec",
        "PreconditionedMVP",
    ]
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Spectral density via stochastic Lanczos quadrature (SLQ).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional until the module is written
    try:
        from .spectral_density import (  # type: ignore[import-not-found]
            lanczos_tridiag,
            slq_density,
            SpectralDensityEstimator,
            spectral_density,
            estimate_condition_number,
        )
    except ImportError:
        from src.spectral.spectral_density import (  # type: ignore[no-redef]
            lanczos_tridiag,
            slq_density,
            SpectralDensityEstimator,
            spectral_density,
            estimate_condition_number,
        )

    __all__ += [
        "lanczos_tridiag",
        "slq_density",
        "SpectralDensityEstimator",
        "spectral_density",
        "estimate_condition_number",
    ]
except ImportError:  # pragma: no cover
    pass


def available_modules() -> dict:
    """Report which optional spectral sub-modules were importable.

    Returns
    -------
    dict
        Mapping ``module_name -> bool`` used by experiment runners to skip
        analyses whose implementation is unavailable.
    """
    import importlib

    names = {
        "hvp": "src.spectral.hvp",
        "lbfgs_unroll": "src.spectral.lbfgs_unroll",
        "preconditioned_mvp": "src.spectral.preconditioned_mvp",
        "spectral_density": "src.spectral.spectral_density",
    }
    status = {}
    for short, dotted in names.items():
        try:
            importlib.import_module(dotted)
            status[short] = True
        except Exception:
            try:
                importlib.import_module("." + short, package=__name__)
                status[short] = True
            except Exception:
                status[short] = False
    return status


__all__ += ["available_modules"]
