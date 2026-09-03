"""
Baseline implementations for zero-shot RL comparison.

Includes:
- GC-BC: Goal-Conditioned Behavioral Cloning
- GC-IQL: Goal-Conditioned Implicit Q-Learning
- OPAL: Offline Primitive Discovery
- FB: Forward-Backward method (via controllable_agent)
- SF: Successor Features (via controllable_agent with ICM features)
"""

from .gc_bc import GoalConditionedBC
from .gc_iql import GCIQL
from .opal import OPAL
from .fb_sf import FBWrapper, SFWrapper, FBConfig, MethodType

__all__ = [
    "GoalConditionedBC",
    "GCIQL",
    "OPAL",
    "FBWrapper",
    "SFWrapper",
    "FBConfig",
    "MethodType",
]