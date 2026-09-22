"""IsaacGym tasks of the paper (Sec. 5.1 / App. A).

These modules import IsaacGym lazily: everything except a full-scale training
run works without a GPU-driven simulator installed (see the CPU toy suite in
``sapg.envs.toy``).
"""

from __future__ import annotations

from ..base import VecEnv

TASK_REGISTRY = {
    "regrasping": ("allegro_kuka", "AllegroKukaRegrasping"),
    "throw": ("allegro_kuka", "AllegroKukaThrow"),
    "reorientation": ("allegro_kuka", "AllegroKukaReorientation"),
    "shadow_hand": ("in_hand", "ShadowHandReorientation"),
    "allegro_hand": ("in_hand", "AllegroHandReorientation"),
}


def make_isaacgym_env(cfg, device: str = "cuda:0", for_eval: bool = False) -> VecEnv:
    """Instantiate one of the five benchmark tasks."""
    name = str(cfg.env.name)
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown IsaacGym task '{name}'. Known: {sorted(TASK_REGISTRY)}")
    module_name, class_name = TASK_REGISTRY[name]

    try:  # pragma: no cover - requires a GPU + IsaacGym installation
        import isaacgym  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "IsaacGym is not importable.  The five benchmark tasks of the paper "
            "(AllegroKuka Regrasping / Throw / Reorientation, ShadowHand and "
            "AllegroHand reorientation) require IsaacGym and a GPU.  Use the CPU "
            "toy suite (env.name=multimodal_collect) to smoke-test the algorithms "
            "and see the README for the full reproduction workflow."
        ) from exc

    module = __import__(f"sapg.envs.isaacgym.{module_name}", fromlist=[class_name])
    task_cls = getattr(module, class_name)
    return task_cls(cfg, device=device, for_eval=for_eval)


__all__ = ["make_isaacgym_env", "TASK_REGISTRY"]
