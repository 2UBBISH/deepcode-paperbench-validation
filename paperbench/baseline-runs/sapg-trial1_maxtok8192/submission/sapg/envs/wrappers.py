"""Environment wrappers for SAPG.

This module provides lightweight, framework-agnostic wrappers used by the SAPG
training loop:

* :class:`EnvWrapper` -- a thin base class exposing a Gym-like API
  (``reset``/``step``/``close``) plus the extra bookkeeping SAPG needs
  (``num_envs``, ``obs_dim``, ``action_dim``, ``successes``).
* :class:`NormalizeObsWrapper` -- running mean/std observation normalization.
* :class:`BlockAssignmentWrapper` -- assigns each parallel environment to a
  follower *block* (``block_id``) so that rollouts can be split per follower.
* :class:`CurriculumWrapper` -- hooks the success-tolerance curriculum
  (:class:`sapg.utils.curriculum.SuccessToleranceCurriculum`) into the env so
  that the task tolerance ``delta`` shrinks as the agent succeeds.
* :func:`make_env` -- generic factory that stacks the wrappers above.

The wrappers are intentionally written so that they work both with a real
vectorized simulator (IsaacGym / MuJoCo) and with the lightweight analytic
environments shipped in this repository (used for CPU-only reproduction and
unit tests).  A "vectorized" environment is expected to expose batched
``reset()``/``step(actions)`` returning arrays of shape ``(num_envs, ...)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.curriculum import CurriculumConfig, SuccessToleranceCurriculum

__all__ = [
    "EnvWrapper",
    "NormalizeObsWrapper",
    "BlockAssignmentWrapper",
    "CurriculumWrapper",
    "make_env",
]


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------
class EnvWrapper:
    """Base class for SAPG environment wrappers.

    Parameters
    ----------
    env:
        The wrapped environment.  It must expose ``reset()`` and
        ``step(actions)`` and the attributes ``num_envs``, ``obs_dim`` and
        ``action_dim``.  If those attributes are missing they are inferred
        from the first ``reset()`` call.
    """

    def __init__(self, env: Any):
        self.env = env
        self._num_envs: Optional[int] = getattr(env, "num_envs", None)
        self._obs_dim: Optional[int] = getattr(env, "obs_dim", None)
        self._action_dim: Optional[int] = getattr(env, "action_dim", None)

    # -- attribute passthrough ------------------------------------------------
    @property
    def num_envs(self) -> int:
        if self._num_envs is None:
            self._num_envs = getattr(self.env, "num_envs", 1)
        return int(self._num_envs)

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            self._obs_dim = getattr(self.env, "obs_dim", None)
        return int(self._obs_dim)

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
            self._action_dim = getattr(self.env, "action_dim", None)
        return int(self._action_dim)

    @property
    def successes(self) -> np.ndarray:
        """Number of successes accumulated in the current episode per env."""
        if hasattr(self.env, "successes"):
            return np.asarray(self.env.successes)
        return np.zeros(self.num_envs, dtype=np.float32)

    @property
    def delta(self) -> float:
        """Current task tolerance (curriculum)."""
        return float(getattr(self.env, "delta", 0.0))

    # -- gym-like API ---------------------------------------------------------
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def __getattr__(self, item):
        # Delegate unknown attributes to the wrapped env (careful with dunder).
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return getattr(self.env, item)


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------
class NormalizeObsWrapper(EnvWrapper):
    """Running mean/std observation normalization.

    Uses Welford's online algorithm so that statistics can be updated with
    batched observations.  Normalization is applied to the observations
    returned by ``reset`` and ``step``.
    """

    def __init__(
        self,
        env: Any,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        update_stats: bool = True,
    ):
        super().__init__(env)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.update_stats = bool(update_stats)

        obs_dim = self.obs_dim
        self._count = 0.0
        self._mean = np.zeros(obs_dim, dtype=np.float64)
        self._var = np.ones(obs_dim, dtype=np.float64)

    # -- statistics -----------------------------------------------------------
    def _update_stats(self, obs: np.ndarray) -> None:
        if not self.update_stats:
            return
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs[None, :]
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self._mean
        tot_count = self._count + batch_count
        new_mean = self._mean + delta * batch_count / tot_count
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self._count * batch_count / tot_count
        new_var = m2 / tot_count

        self._mean = new_mean
        self._var = new_var
        self._count = tot_count

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        std = np.sqrt(self._var + self.epsilon).astype(np.float32)
        mean = self._mean.astype(np.float32)
        normed = (obs - mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    # -- API ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._update_stats(obs)
        return self._normalize(obs)

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        self._update_stats(obs)
        return self._normalize(obs), reward, done, info

    def state_dict(self) -> Dict[str, Any]:
        return {
            "count": self._count,
            "mean": self._mean.copy(),
            "var": self._var.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._count = float(state["count"])
        self._mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self._var = np.asarray(state["var"], dtype=np.float64).copy()


# ---------------------------------------------------------------------------
# Block assignment
# ---------------------------------------------------------------------------
class BlockAssignmentWrapper(EnvWrapper):
    """Assigns each parallel environment to a follower block.

    SAPG splits the ``num_envs`` parallel environments into ``num_blocks``
    blocks; each block is optimized by its own follower policy.  This wrapper
    exposes the per-environment ``block_ids`` array (shape ``(num_envs,)``) and
    the ``num_blocks`` attribute consumed by the training loop.

    Parameters
    ----------
    env:
        Wrapped environment.
    num_blocks:
        Number of follower blocks.  Must divide ``num_envs`` evenly.
    block_ids:
        Optional explicit assignment.  If ``None``, environments are split
        contiguously into equal-sized blocks.
    """

    def __init__(
        self,
        env: Any,
        num_blocks: int = 1,
        block_ids: Optional[Sequence[int]] = None,
    ):
        super().__init__(env)
        self.num_blocks = int(num_blocks)
        num_envs = self.num_envs

        if block_ids is not None:
            block_ids = np.asarray(block_ids, dtype=np.int64)
            if block_ids.shape[0] != num_envs:
                raise ValueError(
                    f"block_ids length {block_ids.shape[0]} != num_envs {num_envs}"
                )
        else:
            if num_envs % self.num_blocks != 0:
                raise ValueError(
                    f"num_envs ({num_envs}) must be divisible by "
                    f"num_blocks ({self.num_blocks})"
                )
            block_ids = np.repeat(
                np.arange(self.num_blocks), num_envs // self.num_blocks
            )
        self.block_ids = block_ids

    @property
    def num_envs_per_block(self) -> int:
        return self.num_envs // self.num_blocks

    def block_indices(self, block_id: int) -> np.ndarray:
        """Indices of the environments belonging to ``block_id``."""
        return np.nonzero(self.block_ids == block_id)[0]

    def block_slices(self) -> List[np.ndarray]:
        return [self.block_indices(b) for b in range(self.num_blocks)]


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
class CurriculumWrapper(EnvWrapper):
    """Wires the success-tolerance curriculum into the environment.

    The wrapper tracks per-episode successes reported by the environment and,
    whenever the average number of successes per episode exceeds the
    configured threshold (default ``3``), decreases the task tolerance
    ``delta`` by 10% (down to a floor of 1cm).

    The environment is expected to expose a mutable ``delta`` attribute and a
    ``successes`` array (number of successes in the current episode per env).
    """

    def __init__(
        self,
        env: Any,
        config: Optional[CurriculumConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(env)
        if config is None:
            config = CurriculumConfig(**kwargs)
        self.curriculum = SuccessToleranceCurriculum(config)
        # Push the initial tolerance into the env.
        self._apply_delta()

    # -- helpers --------------------------------------------------------------
    def _apply_delta(self) -> None:
        if hasattr(self.env, "delta"):
            try:
                self.env.delta = self.curriculum.delta
            except Exception:
                pass

    def _observe(self, done: np.ndarray, info: Optional[Dict[str, Any]] = None) -> None:
        """Update the curriculum from finished episodes."""
        done = np.asarray(done).astype(bool)
        if not done.any():
            return

        successes = self.successes
        # Prefer explicit per-episode success counts from info when available.
        if info is not None and "episode_successes" in info:
            ep_successes = np.asarray(info["episode_successes"], dtype=np.float32)
        else:
            ep_successes = np.asarray(successes, dtype=np.float32)

        finished = ep_successes[done]
        if finished.size == 0:
            return
        changed = self.curriculum.update_batch(finished.tolist())
        if changed:
            self._apply_delta()

    # -- API ------------------------------------------------------------------
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        self._observe(done, info)
        return obs, reward, done, info

    @property
    def delta(self) -> float:
        return float(self.curriculum.delta)

    def state_dict(self) -> Dict[str, Any]:
        return {"curriculum": self.curriculum.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "curriculum" in state:
            self.curriculum.load_state_dict(state["curriculum"])
            self._apply_delta()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_env(
    env: Any,
    num_blocks: int = 1,
    normalize_obs: bool = True,
    use_curriculum: bool = False,
    curriculum_config: Optional[CurriculumConfig] = None,
    block_ids: Optional[Sequence[int]] = None,
    normalize_clip: float = 10.0,
) -> EnvWrapper:
    """Stack SAPG wrappers around ``env``.

    Order (innermost -> outermost):
    ``env`` -> ``BlockAssignmentWrapper`` -> ``CurriculumWrapper`` ->
    ``NormalizeObsWrapper``.

    Parameters
    ----------
    env:
        Base (vectorized) environment.
    num_blocks:
        Number of follower blocks.
    normalize_obs:
        Whether to apply running observation normalization.
    use_curriculum:
        Whether to attach the success-tolerance curriculum.
    curriculum_config:
        Optional :class:`CurriculumConfig` for the curriculum.
    block_ids:
        Optional explicit block assignment.
    normalize_clip:
        Clipping value for observation normalization.
    """
    wrapped: EnvWrapper = env if isinstance(env, EnvWrapper) else EnvWrapper(env)

    if num_blocks > 1 or block_ids is not None:
        wrapped = BlockAssignmentWrapper(wrapped, num_blocks=num_blocks, block_ids=block_ids)

    if use_curriculum:
        wrapped = CurriculumWrapper(wrapped, config=curriculum_config)

    if normalize_obs:
        wrapped = NormalizeObsWrapper(wrapped, clip=normalize_clip)

    return wrapped
