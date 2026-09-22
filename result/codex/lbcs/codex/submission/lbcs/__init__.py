"""Lexicographic Bilevel Coreset Selection (LBCS).

Reference implementation for the paper
"Refined Coreset Selection: Towards Minimal Coreset Size under Model
Performance Constraints" (Xia et al., ICML 2024).

The package is organised as follows:

* :mod:`lbcs.models`       -- network architectures (Table 7 of the paper).
* :mod:`lbcs.data`         -- datasets and corrupted / imbalanced variants.
* :mod:`lbcs.lexiflow`     -- black-box lexicographic optimiser (Algorithm 2).
* :mod:`lbcs.objectives`   -- the two RCS objectives ``f1`` and ``f2``.
* :mod:`lbcs.lbcs`         -- the proposed method (Algorithm 1).
* :mod:`lbcs.baselines`    -- the compared coreset selection methods.
"""

from .lexiflow import LexiFlow, LexiFlowConfig, practical_less, practical_eq
from .objectives import BilevelObjective
from .lbcs import LBCS, LBCSConfig, random_mask

__all__ = [
    "LexiFlow",
    "LexiFlowConfig",
    "practical_less",
    "practical_eq",
    "BilevelObjective",
    "LBCS",
    "LBCSConfig",
    "random_mask",
]

__version__ = "1.0.0"
