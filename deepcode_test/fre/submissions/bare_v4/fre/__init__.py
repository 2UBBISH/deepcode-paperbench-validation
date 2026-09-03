"""
Functional Reward Encoding (FRE) — Core Package

Implements the method from:
"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"
Frans, Park, Abbeel, Levine (ICML 2024)
"""

from .encoder import FREModel, FREEncoder, FREDecoder
from .iql import FREIQLAgent
from .prior import MixedRewardPrior, GoalReachingReward, RandomLinearReward, RandomMLPReward
from .training import FREPipeline
from .evaluation import FREZeroShotEvaluator, EvaluationTask

__all__ = [
    "FREModel",
    "FREEncoder",
    "FREDecoder",
    "FREIQLAgent",
    "MixedRewardPrior",
    "GoalReachingReward",
    "RandomLinearReward",
    "RandomMLPReward",
    "FREPipeline",
    "FREZeroShotEvaluator",
    "EvaluationTask",
]