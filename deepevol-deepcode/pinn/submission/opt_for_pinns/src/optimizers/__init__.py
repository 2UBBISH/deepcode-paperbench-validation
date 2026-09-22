"""Optimizers subpackage for the "Challenges in Training PINNs" reproduction.

Aggregates the optimizers used across the paper's experiments:

* :mod:`~src.optimizers.first_order` -- Adam (with the paper's LR grid) and a
  plain gradient-descent control used in the NNCG fine-tuning study.
* :mod:`~src.optimizers.lbfgs_wrapper` -- L-BFGS (``lr=1.0``, ``memory=100``,
  strong-Wolfe line search) with recording of the curvature history
  ``(s_k, y_k, rho_k)`` consumed by the spectral-density pipeline.
* :mod:`~src.optimizers.combined` -- Adam+L-BFGS with a configurable switch
  iteration (1k / 11k / 31k) and a 41 000-iteration budget.
* :mod:`~src.optimizers.armijo` -- Armijo backtracking line search
  (Algorithm 7, Appendix E.2).
* :mod:`~src.optimizers.nystrom` -- Randomized Nyström approximation
  (Algorithm 5) and Nyström-preconditioned CG (Algorithm 6).
* :mod:`~src.optimizers.nncg` -- NysNewton-CG (Algorithm 4) and its
  ``mu`` tuning helper.

The imports are intentionally eager but guarded with a ``try/except`` shim so
that the package can be imported both as ``src.optimizers`` (from the project
root) and via a relative path.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# First-order optimizers
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import path shim
    from .first_order import (
        ADAM_LR_GRID,
        AdamOptimizer,
        AdamSweepResult,
        FirstOrderOptimizer,
        GradientDescent,
        TrainingHistory,
        adam_lr_grid,
        run_first_order,
        sweep_adam_lr,
    )
except ImportError:  # pragma: no cover
    from src.optimizers.first_order import (  # type: ignore
        ADAM_LR_GRID,
        AdamOptimizer,
        AdamSweepResult,
        FirstOrderOptimizer,
        GradientDescent,
        TrainingHistory,
        adam_lr_grid,
        run_first_order,
        sweep_adam_lr,
    )

# ---------------------------------------------------------------------------
# L-BFGS
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .lbfgs_wrapper import (
        LBFGS_DEFAULTS,
        LBFGSHistory,
        LBFGSOptimizer,
        clear_recording_state,
        lbfgs_recording_state,
        run_lbfgs,
    )
except ImportError:  # pragma: no cover
    from src.optimizers.lbfgs_wrapper import (  # type: ignore
        LBFGS_DEFAULTS,
        LBFGSHistory,
        LBFGSOptimizer,
        clear_recording_state,
        lbfgs_recording_state,
        run_lbfgs,
    )

# ---------------------------------------------------------------------------
# Adam + L-BFGS
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .combined import (
        COMBINED_TOTAL_ITERATIONS,
        DEFAULT_SWITCH_POINT,
        SWITCH_POINTS,
        AdamLBFGS,
        CombinedOptimizer,
        CombinedResult,
        parameter_distance,
        run_adam_lbfgs,
        run_combined,
        sweep_switch_points,
    )
except ImportError:  # pragma: no cover
    from src.optimizers.combined import (  # type: ignore
        COMBINED_TOTAL_ITERATIONS,
        DEFAULT_SWITCH_POINT,
        SWITCH_POINTS,
        AdamLBFGS,
        CombinedOptimizer,
        CombinedResult,
        parameter_distance,
        run_adam_lbfgs,
        run_combined,
        sweep_switch_points,
    )

# ---------------------------------------------------------------------------
# Armijo line search
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .armijo import (
        Armijo,
        ArmijoConfig,
        ArmijoLineSearch,
        ArmijoResult,
        armijo,
        armijo_backtracking,
        flat_grad,
        flatten_params,
        flatten_tensors,
        make_line_search_oracle,
        set_flat_params,
    )
except ImportError:  # pragma: no cover
    from src.optimizers.armijo import (  # type: ignore
        Armijo,
        ArmijoConfig,
        ArmijoLineSearch,
        ArmijoResult,
        armijo,
        armijo_backtracking,
        flat_grad,
        flatten_params,
        flatten_tensors,
        make_line_search_oracle,
        set_flat_params,
    )

# ---------------------------------------------------------------------------
# Nyström linear algebra (Algorithms 5 & 6)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .nystrom import (
        NystromPCG,
        NystromPCGInfo,
        NystromPreconditioner,
        RandomizedNystromApproximation,
        RandomizedNystromInfo,
        as_matvec,
        matvec_apply,
        nystrom_pcg,
        randomized_nystrom_approximation,
    )
except ImportError:  # pragma: no cover
    from src.optimizers.nystrom import (  # type: ignore
        NystromPCG,
        NystromPCGInfo,
        NystromPreconditioner,
        RandomizedNystromApproximation,
        RandomizedNystromInfo,
        as_matvec,
        matvec_apply,
        nystrom_pcg,
        randomized_nystrom_approximation,
    )

# ---------------------------------------------------------------------------
# NysNewton-CG (Algorithm 4)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .nncg import (
        NNCG,
        NNCG_DEFAULTS,
        NNCG_MU_GRID,
        NNCGConfig,
        NNCGHistory,
        NNCGOptimizer,
        NNCGResult,
        MUTuningResult,
        NysNewtonCG,
        run_nncg,
        tune_mu,
    )
except ImportError:  # pragma: no cover
    from src.optimizers.nncg import (  # type: ignore
        NNCG,
        NNCG_DEFAULTS,
        NNCG_MU_GRID,
        NNCGConfig,
        NNCGHistory,
        NNCGOptimizer,
        NNCGResult,
        MUTuningResult,
        NysNewtonCG,
        run_nncg,
        tune_mu,
    )

__all__ = [
    # first order
    "ADAM_LR_GRID",
    "adam_lr_grid",
    "FirstOrderOptimizer",
    "AdamOptimizer",
    "GradientDescent",
    "TrainingHistory",
    "run_first_order",
    "AdamSweepResult",
    "sweep_adam_lr",
    # L-BFGS
    "LBFGS_DEFAULTS",
    "LBFGSHistory",
    "LBFGSOptimizer",
    "lbfgs_recording_state",
    "clear_recording_state",
    "run_lbfgs",
    # Adam + L-BFGS
    "SWITCH_POINTS",
    "DEFAULT_SWITCH_POINT",
    "COMBINED_TOTAL_ITERATIONS",
    "CombinedOptimizer",
    "AdamLBFGS",
    "CombinedResult",
    "run_adam_lbfgs",
    "run_combined",
    "sweep_switch_points",
    "parameter_distance",
    # Armijo
    "ArmijoConfig",
    "ArmijoResult",
    "ArmijoLineSearch",
    "armijo_backtracking",
    "armijo",
    "Armijo",
    "make_line_search_oracle",
    "flatten_tensors",
    "flatten_params",
    "flat_grad",
    "set_flat_params",
    # Nyström
    "randomized_nystrom_approximation",
    "RandomizedNystromApproximation",
    "RandomizedNystromInfo",
    "NystromPreconditioner",
    "nystrom_pcg",
    "NystromPCG",
    "NystromPCGInfo",
    "as_matvec",
    "matvec_apply",
    # NNCG
    "NNCG_MU_GRID",
    "NNCG_DEFAULTS",
    "NNCGConfig",
    "NNCGHistory",
    "NNCGResult",
    "MUTuningResult",
    "NNCGOptimizer",
    "NNCG",
    "NysNewtonCG",
    "run_nncg",
    "tune_mu",
]
