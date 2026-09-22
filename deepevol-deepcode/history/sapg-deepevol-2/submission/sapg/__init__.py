"""SAPG: Split and Aggregate Policy Gradients.

A new class of on-policy RL algorithms that scale to tens of thousands of
parallel environments by splitting environments into blocks (each with its own
"follower" policy) and aggregating their data into a single "leader" policy via
an off-policy (importance-weighted, clipped) update.
"""

from .networks import ActorNetwork, CriticNetwork, build_actor, build_critic
from .policy import GaussianPolicy
from .rollout_buffer import RolloutBuffer
from .ppo import PPO
from .sapg_algorithm import SAPG
from .aggregation import AggregationMode

__all__ = [
    "ActorNetwork",
    "CriticNetwork",
    "build_actor",
    "build_critic",
    "GaussianPolicy",
    "RolloutBuffer",
    "PPO",
    "SAPG",
    "AggregationMode",
]

__version__ = "0.1.0"
