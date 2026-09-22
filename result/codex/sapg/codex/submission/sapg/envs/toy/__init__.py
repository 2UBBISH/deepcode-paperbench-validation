from __future__ import annotations

from .multimodal_collect import MultiModalCollectEnv

TOY_ENVS = {
    "multimodal_collect": MultiModalCollectEnv,
}


def make_toy_env(cfg, device: str = "cpu") -> MultiModalCollectEnv:
    name = str(cfg.env.name)
    if name not in TOY_ENVS:
        raise KeyError(f"Unknown toy env '{name}', available: {sorted(TOY_ENVS)}")
    env_cfg = cfg.get("env", {}) or {}
    return TOY_ENVS[name](
        num_envs=int(env_cfg.get("num_envs", 1024)),
        seed=int(cfg.get("seed", 0)),
        num_landmarks=int(env_cfg.get("num_landmarks", 6)),
        max_steps=int(env_cfg.get("max_steps", 64)),
        capture_radius=float(env_cfg.get("capture_radius", 0.12)),
        step_size=float(env_cfg.get("step_size", 0.15)),
        device=device,
    )


__all__ = ["MultiModalCollectEnv", "make_toy_env", "TOY_ENVS"]
