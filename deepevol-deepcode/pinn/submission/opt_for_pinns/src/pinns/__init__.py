"""PINN problem infrastructure for "Challenges in Training PINNs".

This subpackage bundles the pieces of the physics-informed neural network
(PINN) problem definition that are shared by every experiment:

* :mod:`~opt_for_pinns.src.pinns.model`     -- MLP ansatz ``u(x; w)`` (tanh, 3 hidden layers).
* :mod:`~opt_for_pinns.src.pinns.problems`  -- the differential operator ``D``, the boundary
  operator ``B`` and the analytical solution of the convection / reaction / wave PDEs.
* :mod:`~opt_for_pinns.src.pinns.sampling`  -- the fixed sampling protocol of §2.2
  (10,000 residual points from a 255x100 grid, 257 IC points, 101 BC points per boundary).
* :mod:`~opt_for_pinns.src.pinns.loss`      -- the PINN loss ``L(w)`` of Eq. (2) together with
  its additive residual / initial / boundary components.
* :mod:`~opt_for_pinns.src.pinns.metrics`   -- the L2 relative error of Eq. (3) evaluated on the
  full grid.

The heavy subpackages (``optimizers``, ``spectral``) deliberately do *not* depend on this
``__init__``; they import the concrete modules directly so that importing :mod:`pinns` stays
cheap and free of circular imports.
"""

from __future__ import annotations

# --- model -------------------------------------------------------------------------------
from .model import MLP, make_pinn, xavier_normal_init_

# --- PDE problems ------------------------------------------------------------------------
from .problems import (
    PROBLEMS,
    BoundaryCondition,
    Convection,
    PDEProblem,
    Reaction,
    Wave,
    apply_condition,
    first_grad,
    get_problem,
    second_grad,
)

# --- sampling ----------------------------------------------------------------------------
from .sampling import (
    AXIS_T,
    AXIS_X,
    PINNSampler,
    SamplingConfig,
    build_sampler,
    condition_point_counts,
    condition_points,
    condition_points_flat,
    full_evaluation_points,
    interior_evaluation_grid,
    sample_residual_points,
    sampler_from_config,
    spatial_time_grid,
)

# --- loss --------------------------------------------------------------------------------
from .loss import (
    BOUNDARY,
    INITIAL,
    LossBreakdown,
    PINNLoss,
    classify_condition,
    condition_values,
    loss_breakdown,
    make_loss_fn,
    pinn_loss,
    residual_values,
)

# --- metrics -----------------------------------------------------------------------------
from .metrics import (
    REGION_BOUNDARY,
    REGION_INITIAL,
    REGION_INTERIOR,
    L2REReport,
    compute_l2re,
    evaluate,
    l2re,
    predict,
    solution_error,
)

__all__ = [
    # model
    "MLP",
    "make_pinn",
    "xavier_normal_init_",
    # problems
    "PROBLEMS",
    "BoundaryCondition",
    "Convection",
    "PDEProblem",
    "Reaction",
    "Wave",
    "apply_condition",
    "first_grad",
    "get_problem",
    "second_grad",
    # sampling
    "AXIS_T",
    "AXIS_X",
    "PINNSampler",
    "SamplingConfig",
    "build_sampler",
    "condition_point_counts",
    "condition_points",
    "condition_points_flat",
    "full_evaluation_points",
    "interior_evaluation_grid",
    "sample_residual_points",
    "sampler_from_config",
    "spatial_time_grid",
    # loss
    "BOUNDARY",
    "INITIAL",
    "LossBreakdown",
    "PINNLoss",
    "classify_condition",
    "condition_values",
    "loss_breakdown",
    "make_loss_fn",
    "pinn_loss",
    "residual_values",
    # metrics
    "REGION_BOUNDARY",
    "REGION_INITIAL",
    "REGION_INTERIOR",
    "L2REReport",
    "compute_l2re",
    "evaluate",
    "l2re",
    "predict",
    "solution_error",
]
