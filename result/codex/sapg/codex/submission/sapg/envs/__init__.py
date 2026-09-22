from __future__ import annotations

from .base import VecEnv, StepResult, EpisodeTracker

# Names of the five benchmark tasks of the paper (Sec. 5.1).
ALLEGRO_KUKA_TASKS = ("regrasping", "throw", "reorientation")
IN_HAND_TASKS = ("shadow_hand", "allegro_hand")
BENCHMARK_TASKS = ALLEGRO_KUKA_TASKS + IN_HAND_TASKS

# CPU-only suite used for smoke tests / qualitative checks of the paper's
# claims without a GPU-driven simulator (see README, "toy suite").
TOY_TASKS = ("multimodal_collect",)


def make_env(cfg, device: str = "cpu", for_eval: bool = False) -> VecEnv:
    """Build the environment named by ``cfg.env.name``.

    ``isaacgym.*`` names lazily import the IsaacGym task modules so that the
    rest of the repository (algorithms, analysis, toy suite) works without a
    GPU-driven simulator installed.
    """
    name = str(cfg.env.name)
    if name in BENCHMARK_TASKS:
        from .isaacgym import make_isaacgym_env

        return make_isaacgym_env(cfg, device=device, for_eval=for_eval)
    if name in TOY_TASKS:
        from .toy import make_toy_env

        return make_toy_env(cfg, device=device)
    raise KeyError(
        f"Unknown env '{name}'. Benchmark tasks: {BENCHMARK_TASKS}; toy tasks: {TOY_TASKS}"
    )


__all__ = [
    "VecEnv",
    "StepResult",
    "EpisodeTracker",
    "make_env",
    "ALLEGRO_KUKA_TASKS",
    "IN_HAND_TASKS",
    "BENCHMARK_TASKS",
    "TOY_TASKS",
]
