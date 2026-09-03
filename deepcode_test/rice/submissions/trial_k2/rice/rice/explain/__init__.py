"""Explanation-method submodules for RICE.

This package provides lightweight stubs / drop-in replacements for the
alternative explanation methods used in the ablation study
(Experiment F).  Each module exposes a callable that maps a target policy
and an observation to an importance score, so the refining pipeline can
swap explanation sources without changing its interface.
"""

from rice.explain.random_explanation import RandomExplanation, random_explanation
from rice.explain.integrated_gradients import IntegratedGradients, integrated_gradients
from rice.explain.airs_stub import AIRSStub, airs_explanation

__all__ = [
    "RandomExplanation",
    "random_explanation",
    "IntegratedGradients",
    "integrated_gradients",
    "AIRSStub",
    "airs_explanation",
]
