"""IsaacGym vectorized environment wrapper for SAPG.

The paper (SAPG, Sec 4.6) uses N = 24576 parallel environments on a single GPU,
split into M = 6 blocks of N/M = 4096 envs each.  Each block is rolled out by a
different policy (shared actor backbone conditioned on a per-policy latent).

This module provides:

* ``IsaacGymVectorEnv`` -- a thin wrapper around an IsaacGym task exposing the
  minimal vectorized API used by ``sapg.rollout.RolloutCollector``:
  ``reset() -> obs`` and ``step(actions) -> (obs, rewards, dones, infos)``.
* ``DummyVectorEnv`` -- a pure-PyTorch fallback used for smoke tests / dry runs
  when IsaacGym is not installed.  It implements a simple continuous-control
  task so the full SAPG training loop can be exercised without a GPU simulator.
* ``make_env`` -- factory that builds the correct environment for a task.

The wrapper is deliberately simulator-agnostic at the interface level so that
``rollout.py`` never needs to know whether IsaacGym is present.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:  # pragma: no cover - IsaacGym is optional / GPU only
    import isaacgym  # noqa: F401
    from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401

    _HAS_ISAACGYM = True
except Exception:  # pragma: no cover
    _HAS_ISAACGYM = False


__all__ = [
    "IsaacGymVectorEnv",
    "DummyVectorEnv",
    "make_env",
    "HAS_ISAACGYM",
    "TASK_ENV_REGISTRY",
]

HAS_ISAACGYM = _HAS_ISAACGYM


# ---------------------------------------------------------------------------
# Task registry: maps task name -> (module, class) for the real IsaacGym tasks.
# Imported lazily so that missing IsaacGym does not break module import.
# ---------------------------------------------------------------------------
TASK_ENV_REGISTRY: Dict[str, Tuple[str, str]] = {
    "regrasping": ("envs.allegrokuka", "AllegroKukaRegrasping"),
    "throw": ("envs.allegrokuka", "AllegroKukaThrow"),
    "reorientation": ("envs.allegrokuka", "AllegroKukaReorientation"),
    "shadowhand": ("envs.shadowhand", "ShadowHandReorientation"),
    "allegrohand": ("envs.allegrohand", "AllegroHandReorientation"),
}


# ---------------------------------------------------------------------------
# Real IsaacGym wrapper
# ---------------------------------------------------------------------------
class IsaacGymVectorEnv:
    """Vectorized IsaacGym environment exposing the SAPG rollout API.

    Parameters
    ----------
    task:
        Name of the task (key into ``TASK_ENV_REGISTRY``).
    num_envs:
        Total number of parallel environments N (default 24576).
    num_blocks:
        Number of blocks M the envs are split into (default 6).
    device:
        Torch device used for the simulator tensors.
    seed:
        Optional RNG seed.
    task_kwargs:
        Extra keyword arguments forwarded to the underlying task class.
    """

    def __init__(
        self,
        task: str,
        num_envs: int = 24576,
        num_blocks: int = 6,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        **task_kwargs: Any,
    ) -> None:
        if not HAS_ISAACGYM:
            raise ImportError(
                "IsaacGym is not available. Use DummyVectorEnv for dry runs or "
                "install isaacgym to run the real tasks."
            )
        if task not in TASK_ENV_REGISTRY:
            raise ValueError(
                f"Unknown task '{task}'. Available: {sorted(TASK_ENV_REGISTRY)}"
            )
        if num_envs % num_blocks != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_blocks ({num_blocks})"
            )

        self.task = task
        self.num_envs = int(num_envs)
        self.num_blocks = int(num_blocks)
        self.envs_per_block = self.num_envs // self.num_blocks
        self.device = device if device is not None else torch.device("cuda:0")
        self.seed = seed

        module_name, class_name = TASK_ENV_REGISTRY[task]
        module = __import__(module_name, fromlist=[class_name])
        task_cls = getattr(module, class_name)

        self._task = task_cls(
            num_envs=self.num_envs,
            device=self.device,
            seed=seed,
            **task_kwargs,
        )

        self.obs_dim = int(self._task.obs_dim)
        self.action_dim = int(self._task.action_dim)

    # -- core API ----------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset all envs and return the initial observations ``[N, obs_dim]``."""
        obs = self._task.reset()
        return self._as_tensor(obs)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step all envs.

        Returns
        -------
        obs: ``[N, obs_dim]``
        rewards: ``[N]``
        dones: ``[N]`` (bool)
        infos: dict, may contain ``"successes"`` for the AllegroKuka tasks.
        """
        actions = self._as_tensor(actions)
        obs, rewards, dones, infos = self._task.step(actions)
        return (
            self._as_tensor(obs),
            self._as_tensor(rewards).float(),
            self._as_tensor(dones).bool(),
            infos if infos is not None else {},
        )

    # -- helpers -----------------------------------------------------------
    def _as_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(np.asarray(x), device=self.device)

    def block_slice(self, block_idx: int) -> slice:
        """Return the contiguous env-index slice for ``block_idx``."""
        start = block_idx * self.envs_per_block
        return slice(start, start + self.envs_per_block)

    def close(self) -> None:
        if hasattr(self._task, "close"):
            self._task.close()

    def __len__(self) -> int:
        return self.num_envs


# ---------------------------------------------------------------------------
# Dummy fallback environment (pure PyTorch, no simulator)
# ---------------------------------------------------------------------------
class DummyVectorEnv:
    """A lightweight vectorized continuous-control env used for smoke tests.

    Implements a smooth quadratic goal-reaching task: the agent must drive a
    ``action_dim``-dimensional state to a fixed random goal.  Rewards are
    ``-||state - goal||^2``, which yields a well-behaved learning signal so the
    full SAPG loop (rollout -> aggregation -> losses -> update) can be validated
    without IsaacGym.

    The interface matches :class:`IsaacGymVectorEnv` exactly.
    """

    def __init__(
        self,
        task: str = "regrasping",
        num_envs: int = 24576,
        num_blocks: int = 6,
        obs_dim: int = 64,
        action_dim: int = 23,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        horizon: int = 16,
        **_: Any,
    ) -> None:
        if num_envs % num_blocks != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_blocks ({num_blocks})"
            )
        self.task = task
        self.num_envs = int(num_envs)
        self.num_blocks = int(num_blocks)
        self.envs_per_block = self.num_envs // self.num_blocks
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = device if device is not None else torch.device("cpu")
        self.horizon = int(horizon)
        self.seed = seed

        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))

        self._state = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._goal = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._successes = torch.zeros(self.num_envs, device=self.device)
        self._sample_goals()

    # -- internals ---------------------------------------------------------
    def _sample_goals(self) -> None:
        self._goal = torch.randn(
            self.num_envs, self.action_dim, generator=self._generator
        ).to(self.device)

    def _obs(self) -> torch.Tensor:
        """Observation = [state, goal, state-goal] padded/truncated to obs_dim."""
        diff = self._state - self._goal
        raw = torch.cat([self._state, self._goal, diff], dim=-1)
        if raw.shape[-1] >= self.obs_dim:
            return raw[..., : self.obs_dim]
        pad = torch.zeros(
            self.num_envs, self.obs_dim - raw.shape[-1], device=self.device
        )
        return torch.cat([raw, pad], dim=-1)

    # -- core API ----------------------------------------------------------
    def reset(self) -> torch.Tensor:
        self._state = 0.1 * torch.randn(
            self.num_envs, self.action_dim, generator=self._generator
        ).to(self.device)
        self._step_count.zero_()
        self._successes.zero_()
        self._sample_goals()
        return self._obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        actions = actions.to(self.device).float()
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got {actions.shape[-1]}"
            )
        # Clip actions to a bounded range (like a normalized torque command).
        actions = torch.clamp(actions, -1.0, 1.0)
        self._state = 0.9 * self._state + 0.1 * actions

        dist_sq = ((self._state - self._goal) ** 2).sum(dim=-1)
        rewards = -dist_sq
        success = (dist_sq < 0.05).float()
        self._successes += success

        self._step_count += 1
        dones = self._step_count >= self.horizon
        # Auto-reset finished envs.
        if dones.any():
            idx = dones.nonzero(as_tuple=False).squeeze(-1)
            self._state[idx] = 0.1 * torch.randn(
                idx.numel(), self.action_dim, generator=self._generator
            ).to(self.device)
            new_goals = torch.randn(
                idx.numel(), self.action_dim, generator=self._generator
            ).to(self.device)
            self._goal[idx] = new_goals
            self._step_count[idx] = 0

        infos = {"successes": self._successes.clone()}
        return self._obs(), rewards, dones, infos

    def block_slice(self, block_idx: int) -> slice:
        start = block_idx * self.envs_per_block
        return slice(start, start + self.envs_per_block)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass

    def __len__(self) -> int:
        return self.num_envs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_env(
    config: Any,
    dry_run: bool = False,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> Any:
    """Build the vectorized environment for ``config``.

    Parameters
    ----------
    config:
        A ``SAPGConfig`` (or subclass) instance.  Uses ``task``, ``num_envs``,
        ``num_blocks``, ``horizon`` and ``seed``.
    dry_run:
        If True, always return a :class:`DummyVectorEnv` (no IsaacGym needed).
    device:
        Torch device for the env tensors.

    Returns
    -------
    An environment object exposing ``reset()`` / ``step()`` / ``block_slice()``.
    """
    task = getattr(config, "task", "regrasping")
    num_envs = int(getattr(config, "num_envs", 24576))
    num_blocks = int(getattr(config, "num_blocks", 6))
    horizon = int(getattr(config, "horizon", 16))
    seed = getattr(config, "seed", None)
    obs_dim = int(getattr(config, "obs_dim", 64))
    action_dim = int(getattr(config, "action_dim", 23))

    if device is None:
        device = torch.device(getattr(config, "device", "cpu"))

    if dry_run or not HAS_ISAACGYM:
        return DummyVectorEnv(
            task=task,
            num_envs=num_envs,
            num_blocks=num_blocks,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            seed=seed,
            horizon=horizon,
            **kwargs,
        )

    return IsaacGymVectorEnv(
        task=task,
        num_envs=num_envs,
        num_blocks=num_blocks,
        device=device,
        seed=seed,
        **kwargs,
    )
