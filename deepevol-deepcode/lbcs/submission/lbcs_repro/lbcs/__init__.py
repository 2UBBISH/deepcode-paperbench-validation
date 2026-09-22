"""LBCS: Lexicographic Bilevel Coreset Selection.

Refined Coreset Selection (RCS) minimises the coreset size subject to a
model-performance constraint, with a lexicographic priority
``performance > size``.  This package implements:

* the objective formulations ``f1`` (full-data loss) and ``f2`` (L0 size),
* the practical lexicographic relations and the ``F_H`` threshold tracker,
* Algorithm 1 (``bilevel.py``): the LBCS outer/inner loop,
* Algorithm 2 (``lexiflow.py``): the LexiFlow randomized direct search,
* mask representation / discretization / acceleration utilities.
"""

from .masks import (
    Grouping,
    continuous_to_binary,
    init_binary_mask,
    init_continuous_mask,
    l0_norm,
    num_selected,
)
from .discretize import clamp_mask, discretize_mask, project_to_binary

__all__ = [
    "Grouping",
    "continuous_to_binary",
    "init_binary_mask",
    "init_continuous_mask",
    "l0_norm",
    "num_selected",
    "clamp_mask",
    "discretize_mask",
    "project_to_binary",
]
