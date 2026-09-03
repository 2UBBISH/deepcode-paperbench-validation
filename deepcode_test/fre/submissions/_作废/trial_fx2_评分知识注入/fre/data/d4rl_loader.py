"""D4RL offline dataset loading utilities for AntMaze and Kitchen.

This module wraps the official D4RL package and converts its dataset
dictionaries into the :class:`~fre.data.dataset.OfflineDataset` abstraction
used by the rest of the FRE codebase. State normalization is handled by
``OfflineDataset`` according to the supplied :class:`~fre.config.DataConfig`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from fre.config import DataConfig
from fre.data.dataset import OfflineDataset


def _set_d4rl_paths(cfg: DataConfig) -> None:
    """Point D4RL to a custom data directory if one is configured."""
    path = getattr(cfg, "d4rl_data_path", None)
    if path is not None and str(path).strip():
        os.environ["D4RL_DATASET_DIR"] = str(path)
        os.environ["D4RL_DATA_PATH"] = str(path)


def load_d4rl_env(env_name: str, **kwargs: Any):
    """Create a Gym environment from a D4RL task name.

    The ``d4rl`` import is intentionally lazy so that this module can be
    imported even when D4RL is not installed, as long as no loading function
    is called.
    """
    import gym

    import d4rl  # noqa: F401  (registers D4RL envs)

    return gym.make(env_name, **kwargs)


def _extract_d4rl_dataset(env) -> Dict[str, np.ndarray]:
    """Convert a D4RL environment/dataset into the OfflineDataset key space.

    D4RL exposes ``observations``/``next_observations``; FRE internally uses
    ``states``/``next_states``. This helper performs that renaming and fills
    any missing ``timeouts`` field with zeros.
    """
    import d4rl

    # qlearning_dataset works for AntMaze and Kitchen. Fall back to get_dataset
    # for envs where qlearning_dataset is unavailable.
    try:
        raw = d4rl.qlearning_dataset(env)
    except Exception:
        raw = d4rl.get_dataset(env)

    data: Dict[str, np.ndarray] = {
        "states": np.asarray(raw["observations"], dtype=np.float32),
        "actions": np.asarray(raw["actions"], dtype=np.float32),
        "next_states": np.asarray(raw["next_observations"], dtype=np.float32),
        "rewards": np.asarray(raw["rewards"], dtype=np.float32),
        "terminals": np.asarray(raw["terminals"], dtype=np.float32),
    }

    if "timeouts" in raw:
        data["timeouts"] = np.asarray(raw["timeouts"], dtype=np.float32)
    else:
        data["timeouts"] = np.zeros_like(data["terminals"], dtype=np.float32)

    # Some D4RL versions include episode identifiers. Preserve them when
    # available because OfflineDataset can use them for trajectory inference.
    if "episode_ids" in raw:
        data["episode_ids"] = np.asarray(raw["episode_ids"])

    return data


def load_d4rl_dataset(
    cfg: DataConfig,
    env_name: Optional[str] = None,
    device: str = "cpu",
) -> OfflineDataset:
    """Load a D4RL dataset as an :class:`OfflineDataset`.

    Args:
        cfg: FRE data configuration. ``cfg.env_name`` (or ``cfg.dataset_name``)
            is used when ``env_name`` is not supplied.
        env_name: Explicit Gym/D4RL environment name, e.g.
            ``"antmaze-large-diverse-v2"`` or ``"kitchen-complete-v0"``.
        device: Default device for tensors stored in the dataset.

    Returns:
        An OfflineDataset with normalized/raw states and episode boundaries
        inferred from terminals/timeouts.
    """
    _set_d4rl_paths(cfg)

    if env_name is None:
        env_name = getattr(cfg, "env_name", None) or getattr(cfg, "dataset_name", None)
    if env_name is None or not str(env_name):
        raise ValueError("No D4RL environment name specified in config or arguments.")

    env = load_d4rl_env(str(env_name))
    try:
        data = _extract_d4rl_dataset(env)
    finally:
        try:
            env.close()
        except Exception:
            pass

    return OfflineDataset(data=data, cfg=cfg, device=device)


def load_d4rl_dataset_and_env(
    cfg: DataConfig,
    env_name: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[OfflineDataset, Any]:
    """Like :func:`load_d4rl_dataset` but also returns the live Gym environment.

    The returned environment is useful for online evaluation rollouts.
    """
    _set_d4rl_paths(cfg)

    if env_name is None:
        env_name = getattr(cfg, "env_name", None) or getattr(cfg, "dataset_name", None)
    if env_name is None or not str(env_name):
        raise ValueError("No D4RL environment name specified in config or arguments.")

    env = load_d4rl_env(str(env_name))
    data = _extract_d4rl_dataset(env)
    dataset = OfflineDataset(data=data, cfg=cfg, device=device)
    return dataset, env


def load_antmaze_dataset(
    cfg: DataConfig,
    device: str = "cpu",
) -> OfflineDataset:
    """Load the default AntMaze dataset.

    Defaults to ``antmaze-large-diverse-v2`` but honors
    ``cfg.antmaze_env_name`` when present.
    """
    env_name = getattr(cfg, "antmaze_env_name", None) or "antmaze-large-diverse-v2"
    return load_d4rl_dataset(cfg, env_name=str(env_name), device=device)


def load_kitchen_dataset(
    cfg: DataConfig,
    device: str = "cpu",
) -> OfflineDataset:
    """Load the default Kitchen dataset.

    Defaults to ``kitchen-complete-v0`` but honors ``cfg.kitchen_env_name``
    when present.
    """
    env_name = getattr(cfg, "kitchen_env_name", None) or "kitchen-complete-v0"
    return load_d4rl_dataset(cfg, env_name=str(env_name), device=device)
