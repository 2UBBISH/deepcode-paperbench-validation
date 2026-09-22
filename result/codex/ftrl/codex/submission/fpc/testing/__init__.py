"""Lightweight mock environments used for smoke tests.

These let the three training loops run end-to-end on a CPU in a few seconds,
without the heavy optional dependencies (NLE, ALE, MuJoCo/Meta-World).
"""

from .mock_envs import (
    MockAtariVecEnv,
    MockNetHackEnv,
    MockRoboticSequence,
)

__all__ = ["MockAtariVecEnv", "MockNetHackEnv", "MockRoboticSequence"]
