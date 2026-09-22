"""Environment package for SAPG.

Exposes a unified registry of the five manipulation tasks used in the paper:

    * allegrokuka_regrasping  (23 DoF, recurrent LSTM policy)
    * allegrokuka_throw       (23 DoF, recurrent LSTM policy)
    * allegrokuka_reorientation (23 DoF, recurrent LSTM policy)
    * shadowhand              (24 DoF, MLP policy)
    * allegrohand             (16 DoF, MLP policy)

The public entry point is :func:`make_env`, which returns a vectorized
environment implementing the :class:`~envs.isaacgym_wrapper.VectorEnv`
interface (``reset``/``step``/``close``).  When IsaacGym is unavailable the
factory transparently falls back to the NumPy ``DummyVectorEnv`` so that the
full training pipeline remains runnable for smoke tests.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .isaacgym_wrapper import (
    TASK_ALIASES,
    TASK_SPECS,
    DummyVectorEnv,
    IsaacGymVectorEnv,
    VectorEnv,
    isaacgym_available,
    make_vector_env,
    resolve_task_name,
    task_spec,
)

__all__ = [
    "VectorEnv",
    "DummyVectorEnv",
    "IsaacGymVectorEnv",
    "make_vector_env",
    "make_env",
    "resolve_task_name",
    "task_spec",
    "TASK_SPECS",
    "TASK_ALIASES",
    "TASK_NAMES",
    "isaacgym_available",
    "list_tasks",
]

#: Canonical task names in the order used by the paper's tables/figures.
TASK_NAMES: List[str] = [
    "allegrokuka_regrasping",
    "allegrokuka_throw",
    "allegrokuka_reorientation",
    "shadowhand",
    "allegrohand",
]


def list_tasks() -> List[str]:
    """Return the list of canonical task names."""
    return list(TASK_NAMES)


def make_env(
    task: str,
    num_envs: int = 64,
    seed: int = 0,
    device: str = "cuda:0",
    force_dummy: bool = False,
    **kwargs: Any,
) -> VectorEnv:
    """Create a vectorized environment for ``task``.

    Thin wrapper around :func:`envs.isaacgym_wrapper.make_vector_env` that also
    normalizes task aliases and validates the task name.

    Args:
        task: Task name or alias (e.g. ``"regrasping"``, ``"shadow"``).
        num_envs: Number of parallel environments.
        seed: Random seed.
        device: Torch device string for the simulator.
        force_dummy: Force the NumPy fallback backend.
        **kwargs: Forwarded to the underlying task factory.

    Returns:
        A :class:`VectorEnv` instance.
    """
    canonical = resolve_task_name(task)
    if canonical not in TASK_SPECS:
        raise ValueError(
            f"Unknown task '{task}' (resolved to '{canonical}'). "
            f"Available tasks: {TASK_NAMES}"
        )
    return make_vector_env(
        canonical,
        num_envs=num_envs,
        seed=seed,
        device=device,
        force_dummy=force_dummy,
        **kwargs,
    )


def env_spec(task: str) -> Dict[str, Any]:
    """Return the observation/action spec for ``task`` (alias of ``task_spec``)."""
    return task_spec(resolve_task_name(task))
