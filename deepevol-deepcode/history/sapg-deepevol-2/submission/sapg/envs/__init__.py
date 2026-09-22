"""Environment package for SAPG.

Provides GPU-parallel manipulation environments used in the paper:

* :class:`AllegroKukaEnv`  -- Allegro 16-DoF hand + Kuka 7-DoF arm (23 joints).
  Tasks: ``regrasping``, ``throw``, ``reorientation``.
* :class:`ShadowHandEnv`   -- 24-DoF in-hand cube reorientation.
* :class:`AllegroHandEnv`  -- 16-DoF in-hand cube reorientation.

A lightweight, dependency-free (numpy) vectorized simulation is used so that the
full training pipeline can be exercised on CPU.  When IsaacGym is available the
same interface can be backed by the GPU simulator (see ``backend`` argument).

Also exposes the success-tolerance curriculum from :mod:`envs.curriculum`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .curriculum import SuccessToleranceCurriculum, build_curriculum

__all__ = [
    "AllegroKukaEnv",
    "ShadowHandEnv",
    "AllegroHandEnv",
    "SuccessToleranceCurriculum",
    "build_curriculum",
    "make_env",
    "ENV_REGISTRY",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy import helper
    # Lazy imports avoid a hard dependency on torch/numpy at package import time
    # and prevent circular imports between the env modules.
    if name == "AllegroKukaEnv":
        from .allegro_kuka import AllegroKukaEnv

        return AllegroKukaEnv
    if name == "ShadowHandEnv":
        from .shadow_hand import ShadowHandEnv

        return ShadowHandEnv
    if name == "AllegroHandEnv":
        from .allegro_hand import AllegroHandEnv

        return AllegroHandEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _registry() -> Dict[str, Any]:
    from .allegro_hand import AllegroHandEnv
    from .allegro_kuka import AllegroKukaEnv
    from .shadow_hand import ShadowHandEnv

    return {
        "allegro_kuka": AllegroKukaEnv,
        "allegro_hand": AllegroHandEnv,
        "shadow_hand": ShadowHandEnv,
    }


class _LazyRegistry(dict):
    """Dict-like registry that populates itself on first access."""

    def _ensure(self) -> None:
        if not dict.__len__(self):
            dict.update(self, _registry())

    def __getitem__(self, key: str) -> Any:
        self._ensure()
        return dict.__getitem__(self, key)

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return dict.__contains__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure()
        return dict.get(self, key, default)

    def keys(self):  # type: ignore[override]
        self._ensure()
        return dict.keys(self)

    def items(self):  # type: ignore[override]
        self._ensure()
        return dict.items(self)


ENV_REGISTRY: Dict[str, Any] = _LazyRegistry()


def make_env(
    env_name: str = "allegro_kuka",
    num_envs: int = 1,
    task: str = "regrasping",
    cfg: Optional[Any] = None,
    device: Optional[Any] = None,
    seed: int = 0,
    **kwargs: Any,
):
    """Factory creating a vectorized environment by name.

    Parameters
    ----------
    env_name:
        One of ``"allegro_kuka"``, ``"shadow_hand"``, ``"allegro_hand"``.
    num_envs:
        Number of parallel environments.
    task:
        Task name (only meaningful for ``allegro_kuka``).
    cfg:
        Optional config dict/object providing ``episode_length``,
        ``control_freq`` and curriculum settings.
    device:
        Torch device (or string) the environment tensors live on.
    seed:
        Random seed for the environment RNG.
    """
    registry = _registry()
    key = str(env_name).lower()
    if key not in registry:
        raise ValueError(
            f"Unknown env_name {env_name!r}. Available: {sorted(registry.keys())}"
        )
    env_cls = registry[key]
    return env_cls(
        num_envs=num_envs,
        task=task,
        cfg=cfg,
        device=device,
        seed=seed,
        **kwargs,
    )
