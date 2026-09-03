"""Baseline implementations for zero-shot offline RL.

This package contains the comparison methods used in the FRE paper:

- Forward-Backward representations (FB)
- Successor Features with ICM features (SF)
- Goal-Conditioned IQL (GC-IQL)
- Goal-Conditioned Behavioral Cloning (GC-BC)
- OPAL unsupervised skill discovery
"""

from .fb import FB
from .sf import SF, ICMFeatureNet
from .gc_iql import GCIQL
from .gc_bc import GCBC
from .opal import OPAL, TrajectoryEncoder, TrajectoryDecoder

__all__ = [
    "FB",
    "SF",
    "ICMFeatureNet",
    "GCIQL",
    "GCBC",
    "OPAL",
    "TrajectoryEncoder",
    "TrajectoryDecoder",
]
