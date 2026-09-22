"""Knowledge-retention methods used to mitigate forgetting of pre-trained
capabilities (FPC).

The paper considers four families of methods (Section 2 and Appendix C):

* regularization-based: Elastic Weight Consolidation (EWC),
* distillation-based: behavioral cloning (BC) and kickstarting (KS),
* replay-based: episodic memory (EM),
* parameter-isolation methods (discussed, not used in the experiments).

Following Wolczyk et al. (2022) the retention loss is *never* applied to the
critic -- only the actor/policy parameters are regularised.
"""

from .base import RetentionMethod, RetentionConfig
from .ewc import EWC, DiagonalFisher
from .distillation import BehavioralCloning, Kickstarting, kl_divergence
from .episodic_memory import EpisodicMemory, ProtectedReplayBuffer
from .schedules import ConstantSchedule, ExponentialDecaySchedule, make_schedule

__all__ = [
    "RetentionMethod",
    "RetentionConfig",
    "EWC",
    "DiagonalFisher",
    "BehavioralCloning",
    "Kickstarting",
    "kl_divergence",
    "EpisodicMemory",
    "ProtectedReplayBuffer",
    "ConstantSchedule",
    "ExponentialDecaySchedule",
    "make_schedule",
]
