"""IsaacGym environment wrapper for SAPG.

This module provides a thin, dependency-tolerant wrapper around NVIDIA IsaacGym
vectorised environments.  IsaacGym is *not* pip-installable and requires an
NVIDIA GPU + CUDA 11.x, so the wrapper is written defensively:

* If IsaacGym is available it is used to create the underlying simulation.
* If it is not available (e.g. CPU-only debugging / CI), a lightweight
  :class:`MockVectorEnv` is substituted so that the rest of the SAPG code base
  (rollout collection, losses, algorithm loop) can still be exercised end to end.

The wrapper exposes the minimal interface consumed by ``sapg.sapg.rollout`` and
``sapg.sapg.algorithm``::

    env.num_envs      -> int
    env.obs_dim       -> int
    env.act_dim       -> int
    env.reset()       -> torch.Tensor  (num_envs, obs_dim)
    env.step(actions) -> (obs, rewards, dones, infos)

Observation construction ``o_t`` follows the paper (Section 4.2 / Addendum):

    o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]

where ``q, q_dot`` are the robot joint positions/velocities, ``x_t, v_t,
omega_t`` the object pose/linear/angular velocity, ``g_t`` the goal and ``z_t``
optional auxiliary features (e.g. previous action, phase).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # pragma: no cover - depends on the host machine
    import isaacgym  # noqa: F401
    from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

    _HAS_ISAACGYM = True
except Exception:  # pragma: no cover
    _HAS_ISAACGYM = False


__all__ = [
    "IsaacGymEnvWrapper",
    "MockVectorEnv",
    "make_env",
    "HAS_ISAACGYM",
]


HAS_ISAACGYM = _HAS_ISAACGYM


# ---------------------------------------------------------------------------
# Mock environment (CPU fallback / debugging)
# ---------------------------------------------------------------------------
class MockVectorEnv:
    """A deterministic, dependency-free stand-in for an IsaacGym task.

    The dynamics are intentionally simple (a damped random walk) but the
    interface, tensor shapes and dtypes exactly match what the real IsaacGym
    wrapper produces.  This lets the full SAPG pipeline be unit-tested without
    a GPU.
    """

    def __init__(
        self,
        num_envs: int = 1024,
        obs_dim: int = 40,
        act_dim: int = 23,
        horizon: int = 16,
        device: str = "cpu",
        seed: int = 0,
        task: str = "mock",
    ) -> None:
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.device = torch.device(device)
        self.task = task

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed))

        self._state = torch.zeros(self.num_envs, self.obs_dim, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_lengths: List[int] = []

    # -- core API ----------------------------------------------------------
    def reset(self) -> torch.Tensor:
        self._state = 0.1 * torch.randn(
            self.num_envs, self.obs_dim, generator=self._generator
        ).to(self.device)
        self._step_count.zero_()
        self._episode_return.zero_()
        return self._state.clone()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions = actions.to(self.device)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        noise = 0.05 * torch.randn(
            self.num_envs, self.obs_dim, generator=self._generator
        ).to(self.device)
        # Project the action into the observation space (first act_dim dims).
        act_effect = torch.zeros_like(self._state)
        act_effect[:, : self.act_dim] = actions[:, : self.act_dim]
        self._state = 0.95 * self._state + 0.1 * act_effect + noise

        rewards = -torch.sum(self._state ** 2, dim=-1)
        self._episode_return += rewards
        self._step_count += 1

        dones = (self._step_count >= self.horizon).float()
        if dones.any():
            self._episode_lengths.extend(
                [int(self.horizon)] * int(dones.sum().item())
            )
            reset_ids = dones.bool()
            self._state[reset_ids] = 0.1 * torch.randn(
                int(reset_ids.sum()), self.obs_dim, generator=self._generator
            ).to(self.device)
            self._step_count[reset_ids] = 0
            self._episode_return[reset_ids] = 0.0

        infos: Dict[str, Any] = {
            "episode": {
                "r": self._episode_return.clone(),
                "l": self._step_count.clone(),
            }
        }
        return self._state.clone(), rewards, dones, infos

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


# ---------------------------------------------------------------------------
# Real IsaacGym wrapper
# ---------------------------------------------------------------------------
class IsaacGymEnvWrapper:
    """Wrapper around an IsaacGym task exposing the SAPG env interface.

    Parameters
    ----------
    task:
        One of ``allegrokuka``, ``shadowhand``, ``allegrohand`` (or a custom
        registered task name).
    num_envs:
        Number of parallel environments ``N`` (paper uses 24576).
    device:
        Torch device string, e.g. ``"cuda:0"``.
    headless:
        Run IsaacGym without a viewer.
    task_kwargs:
        Extra keyword arguments forwarded to the task constructor.
    """

    def __init__(
        self,
        task: str = "allegrokuka",
        num_envs: int = 24576,
        device: str = "cuda:0",
        headless: bool = True,
        seed: int = 0,
        task_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task = task
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.headless = headless
        self.seed = int(seed)
        self.task_kwargs = dict(task_kwargs or {})

        self._env = None
        self._obs_dim: Optional[int] = None
        self._act_dim: Optional[int] = None

        if HAS_ISAACGYM:
            self._build_isaacgym_env()
        else:
            # Fall back to the mock env so downstream code still runs.
            self._fallback = MockVectorEnv(
                num_envs=self.num_envs,
                obs_dim=self.task_kwargs.get("obs_dim", 40),
                act_dim=self.task_kwargs.get("act_dim", 23),
                horizon=self.task_kwargs.get("horizon", 16),
                device=str(self.device),
                seed=self.seed,
                task=task,
            )
            self._obs_dim = self._fallback.obs_dim
            self._act_dim = self._fallback.act_dim

    # -- construction ------------------------------------------------------
    def _build_isaacgym_env(self) -> None:  # pragma: no cover - needs GPU
        from isaacgym import gymapi  # noqa: F401

        # Task-specific constructors live in the sibling modules.
        if self.task == "allegrokuka":
            from .allegrokuka import AllegroKukaTask

            self._env = AllegroKukaTask(
                num_envs=self.num_envs,
                device=self.device,
                headless=self.headless,
                seed=self.seed,
                **self.task_kwargs,
            )
        elif self.task == "shadowhand":
            from .shadow_hand import ShadowHandTask

            self._env = ShadowHandTask(
                num_envs=self.num_envs,
                device=self.device,
                headless=self.headless,
                seed=self.seed,
                **self.task_kwargs,
            )
        elif self.task == "allegrohand":
            from .allegro_hand import AllegroHandTask

            self._env = AllegroHandTask(
                num_envs=self.num_envs,
                device=self.device,
                headless=self.headless,
                seed=self.seed,
                **self.task_kwargs,
            )
        else:
            raise ValueError(f"Unknown IsaacGym task: {self.task}")

        self._obs_dim = int(self._env.obs_dim)
        self._act_dim = int(self._env.act_dim)

    # -- properties --------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        assert self._obs_dim is not None
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        assert self._act_dim is not None
        return self._act_dim

    # -- core API ----------------------------------------------------------
    def reset(self) -> torch.Tensor:
        if self._env is not None:  # pragma: no cover - needs GPU
            obs = self._env.reset()
            return self._to_tensor(obs)
        return self._fallback.reset()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self._env is not None:  # pragma: no cover - needs GPU
            obs, rewards, dones, infos = self._env.step(actions)
            return (
                self._to_tensor(obs),
                self._to_tensor(rewards),
                self._to_tensor(dones),
                infos,
            )
        return self._fallback.step(actions)

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):  # pragma: no cover
            self._env.close()

    # -- helpers -----------------------------------------------------------
    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, device=self.device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_env(
    task: str,
    num_envs: int = 24576,
    device: str = "cuda:0",
    headless: bool = True,
    seed: int = 0,
    force_mock: bool = False,
    **task_kwargs: Any,
) -> Any:
    """Create an environment for ``task``.

    Parameters
    ----------
    force_mock:
        If ``True`` always return a :class:`MockVectorEnv` (useful for tests).
    """
    if force_mock or not HAS_ISAACGYM:
        return MockVectorEnv(
            num_envs=num_envs,
            obs_dim=task_kwargs.get("obs_dim", 40),
            act_dim=task_kwargs.get("act_dim", 23),
            horizon=task_kwargs.get("horizon", 16),
            device=device if torch.cuda.is_available() else "cpu",
            seed=seed,
            task=task,
        )
    return IsaacGymEnvWrapper(
        task=task,
        num_envs=num_envs,
        device=device,
        headless=headless,
        seed=seed,
        task_kwargs=task_kwargs,
    )
