"""Environment package for SAPG.

Provides massively-parallel manipulation environments used in the paper:

* :class:`AllegroKukaEnv`  -- Allegro 16-DoF hand + Kuka 7-DoF arm (23 joints).
  Tasks: ``regrasping``, ``throw``, ``reorientation``.
* :class:`ShadowHandEnv`   -- 24-DoF in-hand cube reorientation.
* :class:`AllegroHandEnv`  -- 16-DoF in-hand cube reorientation.

The environments are written against a small, dependency-light vectorised
interface so that they can run either on top of IsaacGym (when available) or on
a pure-PyTorch analytic fallback (used for CPU-only reproduction / unit tests).
"""

from __future__ import annotations

from .reward import (
    RewardConfig,
    compute_allegro_kuka_reward,
    compute_reorientation_reward,
    is_success,
    r_lift,
    r_orientation,
    r_reach,
    r_success,
    r_target,
)

__all__ = [
    "RewardConfig",
    "compute_allegro_kuka_reward",
    "compute_reorientation_reward",
    "is_success",
    "r_reach",
    "r_lift",
    "r_target",
    "r_success",
    "r_orientation",
    "AllegroKukaEnv",
    "ShadowHandEnv",
    "AllegroHandEnv",
    "make_env",
    "ENV_REGISTRY",
]


def __getattr__(name):  # pragma: no cover - lazy import to avoid heavy deps
    if name == "AllegroKukaEnv":
        from .allegro_kuka import AllegroKukaEnv

        return AllegroKukaEnv
    if name == "ShadowHandEnv":
        from .shadow_hand import ShadowHandEnv

        return ShadowHandEnv
    if name == "AllegroHandEnv":
        from .allegro_hand import AllegroHandEnv

        return AllegroHandEnv
    if name in ("make_env", "ENV_REGISTRY"):
        from .allegro_kuka import AllegroKukaEnv
        from .allegro_hand import AllegroHandEnv
        from .shadow_hand import ShadowHandEnv

        registry = {
            "allegro_kuka": AllegroKukaEnv,
            "shadow_hand": ShadowHandEnv,
            "allegro_hand": AllegroHandEnv,
        }
        if name == "ENV_REGISTRY":
            return registry

        def make_env(name, **kwargs):
            if name not in registry:
                raise KeyError(
                    f"Unknown environment '{name}'. Available: {sorted(registry)}"
                )
            return registry[name](**kwargs)

        return make_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
