"""Environment package for SAPG.

Exposes the vectorized task environments used in the paper:

* ``allegrokuka``  -- Regrasping / Throw / Reorientation (Allegro hand + Kuka arm)
* ``shadow_hand``  -- 24-DoF ShadowHand in-hand reorientation
* ``allegro_hand`` -- 16-DoF AllegroHand in-hand reorientation

All environments follow the same minimal vectorized contract consumed by the
SAPG rollout / algorithm modules::

    env.num_envs   -> int
    env.obs_dim    -> int
    env.act_dim    -> int
    obs            = env.reset()                 # (num_envs, obs_dim)
    obs, rew, done, infos = env.step(actions)    # actions: (num_envs, act_dim)
    env.close()

The heavy IsaacGym dependency is optional: every task falls back to a
pure-PyTorch analytic dynamics model when IsaacGym is unavailable, which keeps
the full training pipeline runnable on CPU for debugging / CI.
"""

from __future__ import annotations

from typing import Any, Dict

from .isaacgym_wrapper import (
    HAS_ISAACGYM,
    IsaacGymEnvWrapper,
    MockVectorEnv,
    make_env,
)

__all__ = [
    "HAS_ISAACGYM",
    "IsaacGymEnvWrapper",
    "MockVectorEnv",
    "make_env",
    "make_task_env",
    "TASK_NAMES",
]


# Canonical task identifiers used throughout the paper (Table 1 / Figure 5).
TASK_NAMES = (
    "allegrohand",
    "shadowhand",
    "regrasping",
    "throw",
    "reorientation",
)


def make_task_env(
    task: str,
    num_envs: int = 24576,
    device: str = "cuda:0",
    headless: bool = True,
    seed: int = 0,
    force_mock: bool = False,
    **task_kwargs: Any,
):
    """Create a vectorized environment for one of the paper's five tasks.

    This is a thin, task-name aware dispatcher on top of
    :func:`sapg.envs.isaacgym_wrapper.make_env`.  It maps the paper's task
    names onto the concrete task classes and forwards any extra keyword
    arguments (e.g. ``episode_length``, ``use_curriculum``) to the task.

    Args:
        task: One of ``TASK_NAMES``.
        num_envs: Number of parallel environments (N).  The paper uses 24576.
        device: Torch device string.
        headless: Whether to run IsaacGym headless.
        seed: RNG seed.
        force_mock: Force the dependency-free mock environment.
        **task_kwargs: Extra kwargs forwarded to the task constructor.

    Returns:
        A vectorized environment exposing ``reset`` / ``step`` / ``close``.
    """
    task = str(task).lower()

    # Normalise aliases so callers can use either the paper's task names or the
    # underlying environment family names.  ``allegrokuka`` is the paper's name
    # for the Allegro-hand + Kuka-arm *family*; its headline (and default)
    # concrete task is ``regrasping``.
    aliases: Dict[str, str] = {
        "allegro": "allegrohand",
        "allegro_hand": "allegrohand",
        "shadow": "shadowhand",
        "shadow_hand": "shadowhand",
        "regrasp": "regrasping",
        "throw": "throw",
        "reorient": "reorientation",
        "allegrokuka": "regrasping",
        "allegro_kuka": "regrasping",
    }
    task = aliases.get(task, task)

    if task not in TASK_NAMES:
        raise ValueError(
            f"Unknown task '{task}'. Expected one of {TASK_NAMES}."
        )

    return make_env(
        task=task,
        num_envs=num_envs,
        device=device,
        headless=headless,
        seed=seed,
        force_mock=force_mock,
        **task_kwargs,
    )
