"""Batch-and-Match (BaM) score-based black-box variational inference.

This package implements the algorithms and experiments described in
"Batch and Match: Score-Based Black-Box Variational Inference".
"""

from . import quadratic_solver
from . import batch_stats
from . import divergences
from . import match_update
from . import algorithm
from . import baselines
from . import targets
from . import utils

__all__ = [
    "quadratic_solver",
    "batch_stats",
    "divergences",
    "match_update",
    "algorithm",
    "baselines",
    "targets",
    "utils",
]
