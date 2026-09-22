"""Vectorized environments for SAPG.

This package provides the task environments used by SAPG and the baselines:

- ``allegrokuka``: 23-DoF Allegro hand + Kuka arm tasks (Regrasping, Throw,
  Reorientation) following Petrenko et al. 2023 (DexPBT).
- ``shadowhand``: 24-DoF ShadowHand in-hand reorientation following
  Li et al. 2023 (PQL).
- ``allegrohand``: 16-DoF AllegroHand in-hand reorientation following
  Li et al. 2023 (PQL).
- ``isaacgym_wrapper``: simulator-agnostic vectorized wrapper exposing a
  uniform ``reset``/``step``/``block_slice`` API over either real IsaacGym
  tasks (N=24576 envs split into M=6 blocks) or a pure-PyTorch
  ``DummyVectorEnv`` fallback used for smoke tests.

The public factory :func:`make_env` builds the correct environment from a
:class:`sapg.config.SAPGConfig`.
"""

from .isaacgym_wrapper import (
    HAS_ISAACGYM,
    TASK_ENV_REGISTRY,
    DummyVectorEnv,
    IsaacGymVectorEnv,
    make_env,
)

__all__ = [
    "HAS_ISAACGYM",
    "TASK_ENV_REGISTRY",
    "DummyVectorEnv",
    "IsaacGymVectorEnv",
    "make_env",
]
