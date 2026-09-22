"""Online adaptation framework (Section 3.4, Algorithm 1)."""

from .bank import SampleBank, SampleEntry, OutcomeSupervision
from .feedback import AIFeedback, CombinedFeedback, GroundTruthFeedback, build_feedback
from .online_adaptation import OnlineAdaptation

__all__ = [
    "SampleBank",
    "SampleEntry",
    "OutcomeSupervision",
    "AIFeedback",
    "CombinedFeedback",
    "GroundTruthFeedback",
    "build_feedback",
    "OnlineAdaptation",
]
