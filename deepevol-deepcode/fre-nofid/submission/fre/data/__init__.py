"""Data loading subpackage for the FRE reproduction.

Aggregates the D4RL (AntMaze, Kitchen) and ExORL (RND walker/cheetah) dataset
loaders behind a single import surface.  Uses best-effort re-exports so that a
partial installation (e.g. missing ``d4rl``) still exposes whatever loaders are
importable.
"""

from __future__ import annotations

from typing import Any, List

__all__: List[str] = []


def _try_import(module: str, names: List[str]) -> None:
    """Re-export ``names`` from ``module``; silently skip on failure."""
    try:
        mod = __import__(module, fromlist=list(names))
    except Exception:  # pragma: no cover - optional dependency
        return
    for name in names:
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
            __all__.append(name)


# --- D4RL loader (AntMaze + Kitchen) ---------------------------------------
_try_import(
    __name__ + ".d4rl_loader",
    [
        "ANTMAZE_DATASET",
        "KITCHEN_DATASET",
        "KITCHEN_DATASETS",
        "KITCHEN_TASKS",
        "ANTMAZE_DATASETS",
        "d4rl_available",
        "make_env",
        "load_d4rl_dataset",
        "load_antmaze",
        "load_kitchen",
        "load_kitchen_multitask",
        "kitchen_subtask_rewards",
        "dataset_info",
        "dataset_observation_stats",
        "split_train_validation",
        "to_replay_buffer",
        "load_antmaze_buffer",
        "load_kitchen_buffer",
        "find_dataset_root",
    ],
)

# --- ExORL loader (RND walker/cheetah) -------------------------------------
_try_import(
    __name__ + ".exorl_loader",
    [
        "MAX_EPISODE_STEPS",
        "GOAL_DISTANCE_THRESHOLD",
        "NUM_GOAL_STATES",
        "DEFAULT_DATASET_KIND",
        "WALKER_PHYSICS",
        "CHEETAH_PHYSICS",
        "VELOCITY_THRESHOLDS",
        "RAW_OBS_DIM",
        "EXORL_DOMAINS",
        "load_exorl_dataset",
        "load_exorl",
        "load_exorl_walker",
        "load_exorl_cheetah",
        "load_exorl_multitask",
        "compute_physics",
        "append_physics",
        "normalize_observations",
        "observation_stats",
        "normalize_physics",
        "select_goal_states",
        "goal_reward",
        "goal_done",
        "velocity_reward",
        "velocity_done",
        "make_goal_reward_fn",
        "make_velocity_reward_fn",
        "evaluation_tasks",
        "to_replay_buffer",
        "save_exorl_dataset",
        "dataset_info",
        "physics_names",
        "physics_dim",
        "find_exorl_files",
        "find_exorl_root",
        "exorl_available",
    ],
)

# Disambiguate: both loaders export ``to_replay_buffer``/``dataset_info``; the
# ExORL variants win last, which is the more common ExORL usage.  Keep the
# D4RL-specific convenience helpers clearly named so callers can pick either.
try:  # re-expose the D4RL variants under explicit aliases
    from . import d4rl_loader as _d4rl_loader  # type: ignore

    globals()["d4rl_to_replay_buffer"] = getattr(_d4rl_loader, "to_replay_buffer")
    globals()["load_d4rl_buffer"] = getattr(_d4rl_loader, "to_replay_buffer")
    for _alias in ("d4rl_to_replay_buffer", "load_d4rl_buffer"):
        if _alias not in __all__:
            __all__.append(_alias)
except Exception:  # pragma: no cover - optional dependency
    pass


def load_dataset(domain: str, dataset: Any = None, **kwargs: Any) -> Any:
    """Load a canonical transition dict for ``domain`` in {"antmaze","kitchen","exorl"}.

    Thin dispatch helper used by the training/eval drivers so they do not need to
    know which loader to import.  Returns whatever the underlying loader returns
    (a canonical numpy dict), keeping the FRE convention of
    ``observations/actions/rewards/next_observations/terminals``.
    """
    domain = str(domain).lower()
    if domain.startswith("ant"):
        return load_antmaze(dataset or ANTMAZE_DATASET, **kwargs)  # type: ignore[name-defined]
    if domain.startswith("kitchen"):
        return load_kitchen(dataset or KITCHEN_DATASET, **kwargs)  # type: ignore[name-defined]
    if domain.startswith("exorl"):
        # ``dataset`` may name the domain ("walker"/"cheetah").
        return load_exorl_dataset(dataset or "walker", **kwargs)  # type: ignore[name-defined]
    # Fall back to treating ``domain`` as a raw domain name for ExORL.
    return load_exorl_dataset(domain, **kwargs)  # type: ignore[name-defined]


if "load_dataset" not in __all__:
    __all__.append("load_dataset")


def __getattr__(name: str) -> Any:  # pragma: no cover - error helper
    available = sorted(set(__all__))
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available data exports: {available}"
    )
