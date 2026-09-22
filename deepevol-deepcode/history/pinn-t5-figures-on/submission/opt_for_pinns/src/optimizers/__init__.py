"""opt_for_pinns.src.optimizers

Second-order optimizer components for PINN training (Section 7.2, Appendix E.2).

- ``armijo``  : Algorithm 7 -- Armijo backtracking line search.
- ``nystrom`` : Algorithm 5 -- RandomizedNystromApproximation.
- ``pcg``     : Algorithm 6 -- NystromPCG.
- ``nncg``    : Algorithm 4 -- NysNewton-CG (NNCG).
"""

from __future__ import annotations

from .armijo import armijo_line_search, make_armijo_objective
from .nystrom import NystromApproximation, randomized_nystrom_approximation
from .pcg import PCGResult, apply_preconditioner, nystrom_pcg
from .nncg import NNCG, flatten_params, unflatten_params

__all__ = [
    "armijo_line_search",
    "make_armijo_objective",
    "NystromApproximation",
    "randomized_nystrom_approximation",
    "PCGResult",
    "apply_preconditioner",
    "nystrom_pcg",
    "NNCG",
    "flatten_params",
    "unflatten_params",
]
