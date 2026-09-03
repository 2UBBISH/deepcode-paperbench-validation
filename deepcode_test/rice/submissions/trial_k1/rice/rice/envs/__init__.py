"""RICE environment adapters and wrappers.

This package exposes:

* ``ResettableEnv`` / ``CriticalStateBuffer`` / ``make_resettable`` — mixed
  initial-state distribution used during refining.
* ``make_mujoco_env`` and the sparse/dense MuJoCo reward wrappers.
* Optional domain adapters for selfish mining, CAGE Challenge 2, MetaDrive and
  malware mutation.  Adapters are imported lazily so missing external
  dependencies do not break the package import.
"""

from __future__ import annotations

from rice.envs.resettable_env import (
    CriticalStateBuffer,
    ResettableEnv,
    make_resettable,
)
from rice.envs.mujoco_wrappers import (
    DenseRewardWrapper,
    NormalizeObservationWrapper,
    SparseHalfCheetahWrapper,
    SparseHopperWrapper,
    SparseRewardWrapper,
    SparseWalker2dWrapper,
    TerminationWrapper,
    ReacherWrapper,
    make_mujoco_env,
    is_sparse_env,
    get_x_position,
)

# Optional domain adapters: keep imports soft so that missing external repos
# (pto-selfish-mining, CybORG, DI-drive/MetaDrive, malware_rl) do not break
# ``import rice.envs`` for users working only on MuJoCo.
try:
    from rice.envs.selfish_mining_env import SelfishMiningEnvAdapter, make_selfish_mining_env
except Exception:  # pragma: no cover - external dependency may be absent
    SelfishMiningEnvAdapter = None  # type: ignore
    make_selfish_mining_env = None  # type: ignore

try:
    from rice.envs.cage_env import CageChallenge2Adapter, make_cage_env
except Exception:  # pragma: no cover - external dependency may be absent
    CageChallenge2Adapter = None  # type: ignore
    make_cage_env = None  # type: ignore

try:
    from rice.envs.metadrive_env import MetaDriveMacroAdapter, make_metadrive_env
except Exception:  # pragma: no cover - external dependency may be absent
    MetaDriveMacroAdapter = None  # type: ignore
    make_metadrive_env = None  # type: ignore

try:
    from rice.envs.malware_env import MalConvMalwareEnv, make_malware_env
except Exception:  # pragma: no cover - external dependency may be absent
    MalConvMalwareEnv = None  # type: ignore
    make_malware_env = None  # type: ignore

__all__ = [
    # Resettable / refining
    "CriticalStateBuffer",
    "ResettableEnv",
    "make_resettable",
    # MuJoCo
    "DenseRewardWrapper",
    "NormalizeObservationWrapper",
    "SparseHalfCheetahWrapper",
    "SparseHopperWrapper",
    "SparseRewardWrapper",
    "SparseWalker2dWrapper",
    "TerminationWrapper",
    "ReacherWrapper",
    "make_mujoco_env",
    "is_sparse_env",
    "get_x_position",
    # Optional domain adapters
    "SelfishMiningEnvAdapter",
    "make_selfish_mining_env",
    "CageChallenge2Adapter",
    "make_cage_env",
    "MetaDriveMacroAdapter",
    "make_metadrive_env",
    "MalConvMalwareEnv",
    "make_malware_env",
]
