"""Optimizers for PINN training.

Includes:
- Adam+L-BFGS switching wrapper (Section 5)
- NysNewton-CG (NNCG, Algorithm 4)
- RandomizedNystromApproximation (Algorithm 5)
- NystromPCG (Algorithm 6)
- Armijo line search (Algorithm 7)
"""

from .adam_lbfgs import AdamLBFGS, train_adam_lbfgs
from .armijo import armijo_line_search
from .nystrom import RandomizedNystromApproximation, randomized_nystrom_approximation
from .pcg import NystromPCG, nystrom_pcg
from .nncg import NysNewtonCG, nncg_minimize

__all__ = [
    "AdamLBFGS",
    "train_adam_lbfgs",
    "armijo_line_search",
    "RandomizedNystromApproximation",
    "randomized_nystrom_approximation",
    "NystromPCG",
    "nystrom_pcg",
    "NysNewtonCG",
    "nncg_minimize",
]
