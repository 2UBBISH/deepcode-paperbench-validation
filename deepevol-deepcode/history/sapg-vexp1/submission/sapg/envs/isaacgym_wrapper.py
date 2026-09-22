"""Vectorized environment wrapper for SAPG.

This module provides a thin, simulator-agnostic interface over NVIDIA IsaacGym
(Makoviychuk et al., 2021) that the SAPG trainer and baselines consume.  The
trainer only relies on the following contract::

    env = make_vector_env(task, num_envs, seed=0)
    obs = env.reset()                       # -> np.ndarray [num_envs, obs_dim]
    obs, rew, done, info = env.step(acts)   # acts: [num_envs, action_dim]

Because IsaacGym is distributed separately from PyPI (it must be installed
manually from NVIDIA), this module degrades gracefully: if ``isaacgym`` is not
importable we fall back to :class:`DummyVectorEnv`, a lightweight NumPy
implementation that mimics the observation/reward structure of the real tasks.
This keeps the whole codebase runnable (and unit-testable) on machines without
a GPU while still using the real simulator when it is available.

Task registry
-------------
``allegrokuka_regrasping``, ``allegrokuka_throw``, ``allegrokuka_reorientation``,
``shadowhand``, ``allegrohand``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - depends on the host machine
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

#: Canonical task name -> (obs_dim, action_dim, is_recurrent)
TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "allegrokuka_regrasping": {"obs_dim": 40, "action_dim": 23, "recurrent": True},
    "allegrokuka_throw": {"obs_dim": 40, "action_dim": 23, "recurrent": True},
    "allegrokuka_reorientation": {"obs_dim": 44, "action_dim": 23, "recurrent": True},
    "shadowhand": {"obs_dim": 60, "action_dim": 24, "recurrent": False},
    "allegrohand": {"obs_dim": 44, "action_dim": 16, "recurrent": False},
}

#: Aliases accepted on the command line / in configs.
TASK_ALIASES: Dict[str, str] = {
    "regrasping": "allegrokuka_regrasping",
    "allegrokuka": "allegrokuka_regrasping",
    "throw": "allegrokuka_throw",
    "reorientation": "allegrokuka_reorientation",
    "shadow": "shadowhand",
    "shadow_hand": "shadowhand",
    "allegro": "allegrohand",
    "allegro_hand": "allegrohand",
}


def resolve_task_name(task: str) -> str:
    """Normalize a task alias to its canonical registry name."""
    key = str(task).strip().lower().replace("-", "_")
    if key in TASK_SPECS:
        return key
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    raise KeyError(
        f"Unknown task '{task}'. Known tasks: {sorted(TASK_SPECS)} "
        f"(aliases: {sorted(TASK_ALIASES)})"
    )


def task_spec(task: str) -> Dict[str, Any]:
    """Return the observation/action dimensionality for ``task``."""
    return TASK_SPECS[resolve_task_name(task)]


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class VectorEnv:
    """Minimal vectorized environment interface used across the codebase."""

    num_envs: int
    obs_dim: int
    action_dim: int
    task: str

    def reset(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, actions: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass

    # -- convenience -----------------------------------------------------
    @property
    def num_actions(self) -> int:
        return self.action_dim

    def __len__(self) -> int:
        return self.num_envs


# ---------------------------------------------------------------------------
# Dummy (NumPy) fallback environment
# ---------------------------------------------------------------------------


class DummyVectorEnv(VectorEnv):
    """A cheap stand-in for the IsaacGym tasks.

    The dynamics are a simple linear system with Gaussian noise.  Rewards are
    shaped so that a policy can make measurable progress, which lets the full
    SAPG/PPO/DexPBT/PQL pipelines be exercised end-to-end without a GPU.
    """

    def __init__(
        self,
        task: str = "allegrokuka_regrasping",
        num_envs: int = 64,
        seed: int = 0,
        horizon: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.task = resolve_task_name(task)
        spec = TASK_SPECS[self.task]
        self.num_envs = int(num_envs)
        self.obs_dim = int(spec["obs_dim"])
        self.action_dim = int(spec["action_dim"])
        self.horizon = int(horizon)
        self.device = device

        self._rng = np.random.RandomState(seed)
        self._obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._goal = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._t = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._successes = np.zeros(self.num_envs, dtype=np.float64)

    # -- helpers ---------------------------------------------------------
    def _sample_goal(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=(self.num_envs, self.obs_dim)).astype(
            np.float32
        )

    def _sample_obs(self) -> np.ndarray:
        return self._rng.uniform(-0.5, 0.5, size=(self.num_envs, self.obs_dim)).astype(
            np.float32
        )

    # -- interface -------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._obs = self._sample_obs()
        self._goal = self._sample_goal()
        self._t[:] = 0
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        self._successes[:] = 0.0
        return self._obs.copy()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        assert actions.shape == (self.num_envs, self.action_dim), (
            f"expected actions of shape {(self.num_envs, self.action_dim)}, "
            f"got {actions.shape}"
        )

        # Simple controllable dynamics: move a slice of the observation toward
        # the goal, with noise.
        act = np.tanh(actions)
        delta = np.zeros_like(self._obs)
        delta[:, : self.action_dim] = 0.1 * act
        self._obs = self._obs + delta + 0.01 * self._rng.randn(*self._obs.shape).astype(
            np.float32
        )

        # Reward: negative distance to goal plus a success bonus.
        dist = np.linalg.norm(self._obs - self._goal, axis=-1)
        reward = -dist.astype(np.float32)
        success = (dist < 0.5).astype(np.float32)
        reward = reward + 5.0 * success

        self._t += 1
        self._episode_returns += reward
        self._episode_lengths += 1
        self._successes += success

        done = (self._t >= self.horizon).astype(np.float32)
        info = {
            "success": success,
            "episode_return": self._episode_returns.copy(),
            "episode_length": self._episode_lengths.copy(),
            "episode_success": self._successes.copy(),
        }

        # Auto-reset finished environments.
        if np.any(done > 0):
            idx = np.where(done > 0)[0]
            self._obs[idx] = self._sample_obs()[idx]
            self._goal[idx] = self._sample_goal()[idx]
            self._t[idx] = 0
            self._episode_returns[idx] = 0.0
            self._episode_lengths[idx] = 0
            self._successes[idx] = 0.0

        return self._obs.copy(), reward, done, info

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# IsaacGym-backed environment
# ---------------------------------------------------------------------------


class IsaacGymVectorEnv(VectorEnv):
    """Wrapper around the task-specific IsaacGym environments.

    The heavy lifting (asset loading, tensor allocation, physics stepping) is
    delegated to the task classes in :mod:`envs.allegrokuka_tasks`,
    :mod:`envs.shadowhand_task` and :mod:`envs.allegrohand_task`.  This class
    only normalizes the interface and handles device placement.
    """

    def __init__(
        self,
        task: str,
        num_envs: int,
        seed: int = 0,
        device: str = "cuda:0",
        headless: bool = True,
        **kwargs: Any,
    ) -> None:
        self.task = resolve_task_name(task)
        spec = TASK_SPECS[self.task]
        self.num_envs = int(num_envs)
        self.obs_dim = int(spec["obs_dim"])
        self.action_dim = int(spec["action_dim"])
        self.device = device

        self._task = self._build_task(seed=seed, headless=headless, **kwargs)

    def _build_task(self, seed: int, headless: bool, **kwargs: Any):
        if self.task.startswith("allegrokuka"):
            from .allegrokuka_tasks import make_allegrokuka_task

            variant = self.task.split("_", 1)[1]
            return make_allegrokuka_task(
                variant=variant,
                num_envs=self.num_envs,
                seed=seed,
                device=self.device,
                headless=headless,
                **kwargs,
            )
        if self.task == "shadowhand":
            from .shadowhand_task import make_shadowhand_task

            return make_shadowhand_task(
                num_envs=self.num_envs,
                seed=seed,
                device=self.device,
                headless=headless,
                **kwargs,
            )
        if self.task == "allegrohand":
            from .allegrohand_task import make_allegrohand_task

            return make_allegrohand_task(
                num_envs=self.num_envs,
                seed=seed,
                device=self.device,
                headless=headless,
                **kwargs,
            )
        raise KeyError(f"No IsaacGym task registered for '{self.task}'")

    # -- interface -------------------------------------------------------
    def reset(self) -> np.ndarray:
        obs = self._task.reset()
        return _to_numpy(obs)

    def step(self, actions: np.ndarray):
        obs, rew, done, info = self._task.step(actions)
        return _to_numpy(obs), _to_numpy(rew), _to_numpy(done), info

    def close(self) -> None:
        close = getattr(self._task, "close", None)
        if callable(close):
            close()


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors (or arrays) to CPU NumPy arrays."""
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def isaacgym_available() -> bool:
    """Return True when the IsaacGym python package can be imported."""
    try:  # pragma: no cover - environment dependent
        import isaacgym  # noqa: F401

        return True
    except Exception:
        return False


def make_vector_env(
    task: str,
    num_envs: int,
    seed: int = 0,
    device: str = "cuda:0",
    force_dummy: bool = False,
    **kwargs: Any,
) -> VectorEnv:
    """Create a vectorized environment for ``task``.

    Uses IsaacGym when available (and not explicitly disabled), otherwise falls
    back to :class:`DummyVectorEnv`.
    """
    task_name = resolve_task_name(task)
    use_dummy = force_dummy or os.environ.get("SAPG_FORCE_DUMMY_ENV", "0") == "1"
    if not use_dummy and isaacgym_available():
        try:
            return IsaacGymVectorEnv(
                task=task_name, num_envs=num_envs, seed=seed, device=device, **kwargs
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            print(
                f"[envs] IsaacGym task '{task_name}' failed to initialize "
                f"({exc!r}); falling back to DummyVectorEnv."
            )
    return DummyVectorEnv(task=task_name, num_envs=num_envs, seed=seed, device=device)


__all__ = [
    "VectorEnv",
    "DummyVectorEnv",
    "IsaacGymVectorEnv",
    "make_vector_env",
    "resolve_task_name",
    "task_spec",
    "isaacgym_available",
    "TASK_SPECS",
    "TASK_ALIASES",
]
