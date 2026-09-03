"""RICE: Refining Reinforcement Learning Agents via Critical Explanations.

Top-level package exports and version metadata.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "RICE Reproduction Team"
__description__ = (
    "Refining Reinforcement Learning Agents via Critical Explanations (RICE)"
)

# Re-export the most commonly used public API so users can write
#   import rice
#   rice.train_target(...)
#   rice.train_mask(...)
#   rice.refine_agent(...)

try:
    from rice.training.train_target import train_target_agent as train_target
except Exception:  # pragma: no cover - optional external deps may fail
    train_target = None  # type: ignore

try:
    from rice.training.train_mask import train_mask
except Exception:  # pragma: no cover
    train_mask = None  # type: ignore

try:
    from rice.training.refine_agent import refine_agent
except Exception:  # pragma: no cover
    refine_agent = None  # type: ignore

try:
    from rice.agents.target_agent import TargetAgent, TargetAgentConfig
except Exception:  # pragma: no cover
    TargetAgent = None  # type: ignore
    TargetAgentConfig = None  # type: ignore

try:
    from rice.agents.mask_network import MaskNetwork, MaskTrainingConfig
except Exception:  # pragma: no cover
    MaskNetwork = None  # type: ignore
    MaskTrainingConfig = None  # type: ignore

try:
    from rice.agents.rnd_network import RNDModule
except Exception:  # pragma: no cover
    RNDModule = None  # type: ignore

try:
    from rice.envs.resettable_env import CriticalStateBuffer, ResettableEnv
except Exception:  # pragma: no cover
    CriticalStateBuffer = None  # type: ignore
    ResettableEnv = None  # type: ignore

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "train_target",
    "train_mask",
    "refine_agent",
    "TargetAgent",
    "TargetAgentConfig",
    "MaskNetwork",
    "MaskTrainingConfig",
    "RNDModule",
    "CriticalStateBuffer",
    "ResettableEnv",
]
