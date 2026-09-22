"""SAPG environment suite (Appendix A, Sec. 5.1).

This package exposes the vectorised environment interface used by the rollout
loop (``reset()`` / ``step(actions)``) together with the task-specific wrappers
for the three simulator families used in the paper:

* ``allegro_kuka``  -- Allegro hand + Kuka arm: ``regrasping``, ``throw`` and
  ``reorientation`` (Appendix A.1 - A.3).
* ``shadow_hand``   -- Shadow Hand in-hand reorientation (Appendix A.4).
* ``allegro_hand``  -- Allegro Hand in-hand reorientation (Appendix A.5).

All real implementations live in the sibling modules; this initializer only
re-exports them lazily (PEP 562) so that ``import sapg.envs`` stays cheap and
does not require IsaacGym (or even torch) at import time.

The single entry point used by the training scripts is :func:`make_env`, which
dispatches on the task name and returns an :class:`~sapg.envs.isaac_env.IsaacEnv`
facade (real IsaacGym backend when available, dependency-free surrogate
otherwise).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

__all__: List[str] = [
    # Core interface ------------------------------------------------------
    "IsaacEnv",
    "IsaacGymParallelEnv",
    "SurrogateVectorEnv",
    "EnvConfig",
    "make_env",
    "make_isaac_env",
    # Task metadata / helpers --------------------------------------------
    "TASK_REGISTRY",
    "HAS_ISAACGYM",
    "task_group_of",
    "joint_dim_for",
    "goal_dim_for",
    "action_dim_for",
    "obs_dim_for",
    "default_reward_weights",
    "default_isaac_task_cfg",
    # Quaternion utilities ------------------------------------------------
    "quat_normalize",
    "quat_mul",
    "quat_conjugate",
    "quat_angle_error",
    "quat_random",
    # Allegro-Kuka tasks (Appendix A.1 - A.3) ----------------------------
    "AllegroKukaEnv",
    "RegraspingEnv",
    "ThrowEnv",
    "ReorientationEnv",
    "make_allegro_kuka_env",
    "make_regrasping_env",
    "make_throw_env",
    "make_reorientation_env",
    # Shadow Hand (Appendix A.4) -----------------------------------------
    "ShadowHandEnv",
    "make_shadow_hand_env",
    # Allegro Hand (Appendix A.5) ----------------------------------------
    "AllegroHandEnv",
    "make_allegro_hand_env",
    # Curriculum (Sec. 5.1 / Appendix A.1) -------------------------------
    "SuccessCurriculum",
    "make_curriculum",
    "CURRICULUM_TASKS",
    "TASKS",
]

#: Names exported from :mod:`sapg.envs.isaac_env` (loaded eagerly on first use).
_ISAAC_NAMES = frozenset(
    {
        "IsaacEnv",
        "IsaacGymParallelEnv",
        "SurrogateVectorEnv",
        "EnvConfig",
        "make_isaac_env",
        "TASK_REGISTRY",
        "HAS_ISAACGYM",
        "task_group_of",
        "joint_dim_for",
        "goal_dim_for",
        "action_dim_for",
        "obs_dim_for",
        "default_reward_weights",
        "default_isaac_task_cfg",
        "quat_normalize",
        "quat_mul",
        "quat_conjugate",
        "quat_angle_error",
        "quat_random",
    }
)

#: Mapping public name -> defining submodule (sibling module of this package).
_LAZY_EXPORTS: Dict[str, str] = {
    **{name: "isaac_env" for name in _ISAAC_NAMES},
    # Allegro-Kuka tasks (Appendix A.1 - A.3)
    "AllegroKukaEnv": "allegro_kuka",
    "RegraspingEnv": "allegro_kuka",
    "ThrowEnv": "allegro_kuka",
    "ReorientationEnv": "allegro_kuka",
    "make_allegro_kuka_env": "allegro_kuka",
    "make_regrasping_env": "allegro_kuka",
    "make_throw_env": "allegro_kuka",
    "make_reorientation_env": "allegro_kuka",
    # Shadow Hand (Appendix A.4)
    "ShadowHandEnv": "shadow_hand",
    "make_shadow_hand_env": "shadow_hand",
    # Allegro Hand (Appendix A.5)
    "AllegroHandEnv": "allegro_hand",
    "make_allegro_hand_env": "allegro_hand",
    # Curriculum (Sec. 5.1)
    "SuccessCurriculum": "curriculum",
    "make_curriculum": "curriculum",
    "CURRICULUM_TASKS": "curriculum",
}

#: Canonical list of the five tasks from Sec. 5.1 (order as in the paper).
TASKS: List[str] = [
    "regrasping",
    "throw",
    "reorientation",
    "shadow_hand",
    "allegro_hand",
]

#: Tasks whose success criterion uses the 7.5cm -> 1cm tolerance curriculum.
CURRICULUM_TASKS: List[str] = ["regrasping"]

#: All public names that are (lazily) resolvable from this package.
_LAZY_EXPORTS["make_env"] = "__self__"


def __getattr__(name: str) -> Any:  # pragma: no cover - thin dispatch
    """PEP 562 lazy attribute resolution for :mod:`sapg.envs`."""
    if name == "make_env":
        return make_env

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; "
            f"available names: {sorted(__all__)}"
        )

    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise AttributeError(
            f"cannot resolve {name!r}: failed to import {__name__}.{module_name} ({exc})"
        ) from exc

    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"module {__name__}.{module_name} has no attribute {name!r}"
        ) from exc


def __dir__() -> List[str]:
    return sorted(set(__all__))


def make_env(
    task: str = "regrasping",
    config: Optional[Any] = None,
    num_envs: Optional[int] = None,
    **overrides: Any,
) -> Any:
    """Create the vectorised environment for ``task`` (Sec. 5.1).

    Parameters
    ----------
    task:
        One of :data:`TASKS` (``"regrasping"``, ``"throw"``, ``"reorientation"``,
        ``"shadow_hand"``, ``"allegro_hand"``).
    config:
        Optional :class:`sapg.utils.config.SAPGConfig` (or plain mapping) whose
        fields seed the environment configuration.
    num_envs:
        Number of parallel environments ``N`` (defaults to the config value or
        ``24576`` per Sec. 5.2).
    **overrides:
        Additional :class:`~sapg.envs.isaac_env.EnvConfig` overrides.

    Returns
    -------
    IsaacEnv
        The unified environment facade used by the rollout collectors.
    """
    task_name = str(task).strip().lower().replace("-", "_")
    if task_name in ("allegro_kuka",):
        task_name = "regrasping"
    if task_name not in TASKS:
        raise ValueError(
            f"unknown task {task!r}; expected one of {TASKS} "
            "(aliases: \"regrasp\", \"shadow\", \"allegro\")"
        )
    alias = {
        "regrasp": "regrasping",
        "in_hand_reorientation": "shadow_hand",
        "shadow": "shadow_hand",
        "allegro": "allegro_hand",
    }.get(task_name, task_name)

    if num_envs is not None:
        overrides.setdefault("num_envs", num_envs)

    # Task-specific wrappers add observation/reward details on top of the core
    # facade; fall back to the plain facade when a wrapper is unavailable.
    module_name = {
        "regrasping": "allegro_kuka",
        "throw": "allegro_kuka",
        "reorientation": "allegro_kuka",
        "shadow_hand": "shadow_hand",
        "allegro_hand": "allegro_hand",
    }[alias]

    try:
        module = importlib.import_module(f".{module_name}", __name__)
        factory = getattr(module, "make_env", None)
        if factory is not None:
            return factory(task=alias, config=config, **overrides)
    except ImportError:  # pragma: no cover - wrapper modules are optional
        pass

    from .isaac_env import make_isaac_env

    return make_isaac_env(task=alias, config=config, **overrides)
