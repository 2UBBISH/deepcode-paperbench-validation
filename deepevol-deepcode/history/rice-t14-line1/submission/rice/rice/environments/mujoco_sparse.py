"""Sparse-reward MuJoCo environments for RICE (Appendix C.2).

Paper specification (verbatim, §C.2):

* Hopper:  "Under the sparse reward setting (Mazoure et al., 2019), the reward
  informs the x position of the hopper only if ``x > 0.6`` in our experiments.
  The episode concludes if the Hopper becomes unhealthy."
* Walker2d: "Under the sparse reward setting (Mazoure et al., 2019), the reward
  informs the x position of the hopper only if ``x > 0.6`` in our experiments.
  The episode concludes if the walker is deemed unhealthy."
* HalfCheetah: "Under the sparse reward setting (Mazoure et al., 2019), the
  reward informs the x position of the hopper only if ``x > 5`` in our
  experiments. Episodes conclude after 1000 timesteps."
* Reacher: "It is worth noting that there is no sparse reward implementation of
  Reacher-v2 in Mazoure et al. (2019)."  -> no sparse Reacher exists.

Sparse variants are therefore ``SparseHopper``, ``SparseWalker2d`` and
``SparseHalfCheetah``.  Per the reproduction addendum only Hopper and
HalfCheetah are used as Experiment II success criteria (``SparseWalker2d``
refining is explicitly OUT OF SCOPE); the Walker2d sparse task is implemented
for completeness so the registry stays faithful to the paper's environment list.

Implementation note
-------------------
Mazoure et al. (2019) keep the *dense* reward-to-go structure but replace the
terminal task reward with a threshold indicator/position.  We therefore
implement the sparse signal as:

    r_t = (x_t - x_prev)  if x_t > threshold  else  0

i.e. the increment of the forward ``x`` position is only revealed once the
agent's position has passed ``threshold``.  ``sparse_mode="binary"`` yields the
pure indicator ``1.0`` instead of the x increment, and
``sparse_mode="position"`` reveals the absolute x coordinate.  The environment
construction (observation normalization, episode length, healthy termination,
non-finite termination) is inherited unchanged from :mod:`mujoco_dense`, which
follows §C.2 as well.

This module is import-safe: no gym / MuJoCo import happens at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ._common import EnvBase, WrapperBase, normalize_reset, normalize_step

try:  # pragma: no cover - import guard for documentation builds
    from .mujoco_dense import (
        MUJOCO_DENSE_SPECS,
        MujocoDenseEnvSpec,
        make_env as make_dense_env,
        resolve_name as resolve_dense_name,
    )
except Exception:  # pragma: no cover - very defensive, keeps module importable
    MUJOCO_DENSE_SPECS = {}  # type: ignore[assignment]

    class MujocoDenseEnvSpec:  # type: ignore[no-redef]
        """Minimal stand-in when ``mujoco_dense`` cannot be imported."""

        name = "Hopper-v3"
        gym_ids = ("Hopper-v3",)
        obs_dim = 11
        act_dim = 3
        max_episode_steps = 1000
        normalize_obs = False

    def resolve_dense_name(name: str) -> str:  # type: ignore[misc]
        return name

    def make_dense_env(name: str = "Hopper-v3", **kwargs: Any) -> Any:  # type: ignore[misc]
        raise ImportError(
            "rice.environments.mujoco_dense is unavailable; cannot build a "
            "sparse MuJoCo environment. Install MuJoCo + gym, or set "
            "RICE_ALLOW_MUJOCO_FALLBACK=1 to use the analytic stand-ins."
        )


# --------------------------------------------------------------------------------------
# Sparse task specifications (Table: §C.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MujocoSparseEnvSpec:
    """Static description of a sparse-reward MuJoCo task (§C.2).

    Attributes
    ----------
    name:
        Canonical RICE name, e.g. ``"SparseHopper"``.
    base_name:
        Dense MuJoCo counterpart used to build the underlying simulator.
    threshold:
        Sparse reward threshold (``x > threshold``); 0.6 for Hopper/Walker2d,
        5.0 for HalfCheetah.
    sparse_mode:
        ``"increment"`` (default, reward = delta x once past the threshold),
        ``"binary"`` (reward = 1.0 once past the threshold) or ``"position"``
        (reward = absolute x once past the threshold).
    in_scope:
        Whether Experiment II uses this task (SparseWalker2d is out of scope).
    """

    name: str
    base_name: str
    threshold: float
    sparse_mode: str = "increment"
    max_episode_steps: int = 1000
    terminate_when_unhealthy: bool = True
    normalize_obs: bool = False
    obs_dim: int = 11
    act_dim: int = 3
    net_arch: Tuple[int, ...] = (64, 64)
    in_scope: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_name": self.base_name,
            "threshold": self.threshold,
            "sparse_mode": self.sparse_mode,
            "max_episode_steps": self.max_episode_steps,
            "terminate_when_unhealthy": self.terminate_when_unhealthy,
            "normalize_obs": self.normalize_obs,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "net_arch": tuple(self.net_arch),
            "in_scope": self.in_scope,
            "note": self.note,
        }


SPARSE_SPECS: Dict[str, MujocoSparseEnvSpec] = {
    "SparseHopper": MujocoSparseEnvSpec(
        name="SparseHopper",
        base_name="Hopper-v3",
        threshold=0.6,
        sparse_mode="increment",
        max_episode_steps=1000,
        terminate_when_unhealthy=True,
        normalize_obs=False,
        obs_dim=11,
        act_dim=3,
        in_scope=True,
        note="§C.2: reward informs x only if x > 0.6; episode ends if unhealthy.",
    ),
    "SparseWalker2d": MujocoSparseEnvSpec(
        name="SparseWalker2d",
        base_name="Walker2d-v3",
        threshold=0.6,
        sparse_mode="increment",
        max_episode_steps=1000,
        terminate_when_unhealthy=True,
        normalize_obs=True,
        obs_dim=17,
        act_dim=6,
        in_scope=False,
        note="§C.2: reward informs x only if x > 0.6; OUT OF SCOPE for refining "
        "(addendum) but registered for completeness.",
    ),
    "SparseHalfCheetah": MujocoSparseEnvSpec(
        name="SparseHalfCheetah",
        base_name="HalfCheetah-v3",
        threshold=5.0,
        sparse_mode="increment",
        max_episode_steps=1000,
        terminate_when_unhealthy=False,
        normalize_obs=True,
        obs_dim=17,
        act_dim=6,
        in_scope=True,
        note="§C.2: reward informs x only if x > 5; episodes end after 1000 steps.",
    ),
}


_ALIASES: Dict[str, str] = {
    "sparsehopper": "SparseHopper",
    "sparsehopper-v3": "SparseHopper",
    "hopper-sparse": "SparseHopper",
    "sparsehopper": "SparseHopper",
    "sparsewalker2d": "SparseWalker2d",
    "sparsewalker2d-v3": "SparseWalker2d",
    "walker2d-sparse": "SparseWalker2d",
    "sparsewalker": "SparseWalker2d",
    "sparsehalfcheetah": "SparseHalfCheetah",
    "sparsehalfcheetah-v3": "SparseHalfCheetah",
    "halfcheetah-sparse": "SparseHalfCheetah",
    "sparsehc": "SparseHalfCheetah",
}


def resolve_name(name: str) -> str:
    """Map a canonical/alias sparse task name onto :data:`SPARSE_SPECS`."""

    if name in SPARSE_SPECS:
        return name
    key = str(name).strip().replace("_", "").replace(" ", "").lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # tolerate the paper's own naming, e.g. "Hopper-sparse".
    base = key.replace("sparse", "")
    for candidate in SPARSE_SPECS:
        if candidate.replace("sparse", "").lower() == base:
            return candidate
    raise KeyError(
        f"Unknown sparse MuJoCo environment {name!r}. "
        f"Known: {sorted(SPARSE_SPECS)}"
    )


def get_spec(name: str) -> MujocoSparseEnvSpec:
    """Return the :class:`MujocoSparseEnvSpec` for ``name``."""

    return SPARSE_SPECS[resolve_name(name)]


# --------------------------------------------------------------------------------------
# Position extraction helpers
# --------------------------------------------------------------------------------------


def _unwrap(env: Any) -> Any:
    """Return the innermost environment object (skipping gym wrappers)."""

    seen = 0
    while hasattr(env, "env") and not hasattr(env, "sim") and seen < 16:
        env = env.env
        seen += 1
    return env


def forward_position(env: Any, info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Best-effort forward ``x`` coordinate of the agent (the sparse signal).

    Strategy (first match wins):
      1. ``info["x_position"]`` (gym MuJoCo step info),
      2. ``env.sim.data.qpos[0]`` (mujoco_py based gym versions),
      3. ``env.data.qpos[0]`` (mujoco>=2.3 bindings),
      4. ``env.state[0]`` (gym MuJoCo cached state vector, or the fallback env),
      5. the first observation entry (dimension 0 is the root x position for
         Hopper / Walker2d / HalfCheetah).
    """

    if info:
        for key in ("x_position", "x_pos", "x"):
            value = info.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    inner = _unwrap(env)

    for attr in ("sim", "data"):
        obj = getattr(inner, attr, None)
        if obj is not None:
            data = getattr(obj, "data", obj)
            qpos = getattr(data, "qpos", None)
            if qpos is not None and len(np.atleast_1d(qpos)) > 0:
                try:
                    return float(np.atleast_1d(qpos)[0])
                except (TypeError, ValueError):
                    pass

    state = getattr(inner, "state", None)
    if state is not None:
        arr = np.atleast_1d(state)
        if arr.size > 0:
            try:
                return float(arr[0])
            except (TypeError, ValueError):
                pass

    for attr in ("unwrapped", "env"):
        obj = getattr(env, attr, None)
        if obj is not None and obj is not env:
            pos = forward_position(obj, None)
            if pos is not None:
                return pos

    return None


# --------------------------------------------------------------------------------------
# Sparse reward wrapper
# --------------------------------------------------------------------------------------


class SparseRewardWrapper(WrapperBase):
    """Turn a dense MuJoCo task into its Mazoure et al. (2019) sparse variant.

    The reward is only *informed* (i.e. non-zero) once the agent's forward
    position passes ``threshold`` (§C.2).  Three formulations are supported via
    ``sparse_mode``:

    ``increment`` (default)
        ``r = x_t - x_prev`` if ``x_t > threshold`` else ``0``.
    ``binary``
        ``r = 1.0`` if ``x_t > threshold`` else ``0``.
    ``position``
        ``r = x_t`` if ``x_t > threshold`` else ``0``.

    The wrapper also accumulates the *dense* reward in ``info["dense_reward"]``
    so that diagnostics (and the "No Refine" dense baselines) remain available
    without a second simulation.  ``x_prev`` is reset at every episode start so
    the first revealed reward accounts only for progress made inside the
    episode.
    """

    def __init__(
        self,
        env: Any,
        threshold: float = 0.6,
        sparse_mode: str = "increment",
        reward_scale: float = 1.0,
        keep_dense_info: bool = True,
    ) -> None:
        super().__init__(env)
        mode = str(sparse_mode).lower()
        if mode not in ("increment", "binary", "position", "delta"):
            raise ValueError(
                f"Unknown sparse_mode {sparse_mode!r}; expected one of "
                "'increment', 'binary', 'position'."
            )
        self.threshold = float(threshold)
        self.sparse_mode = "increment" if mode == "delta" else mode
        self.reward_scale = float(reward_scale)
        self.keep_dense_info = bool(keep_dense_info)
        self._x_prev: Optional[float] = None
        self._dense_return = 0.0

    # -- helpers -----------------------------------------------------------------
    def _resolve_position(self, info: Dict[str, Any]) -> Optional[float]:
        return forward_position(self.env, info)

    def _sparse_reward(self, x: Optional[float]) -> float:
        if x is None or not np.isfinite(x) or x <= self.threshold:
            # Below the threshold the reward does not inform the position.
            if x is not None and np.isfinite(x):
                self._x_prev = x
            return 0.0
        if self.sparse_mode == "binary":
            reward = 1.0
        elif self.sparse_mode == "position":
            reward = float(x)
        else:  # "increment"
            prev = self._x_prev if self._x_prev is not None else self.threshold
            reward = float(x) - float(prev)
            if reward < 0.0:
                # The agent moved backwards: no information is revealed.
                reward = 0.0
        self._x_prev = float(x)
        return float(reward) * self.reward_scale

    # -- gym API -----------------------------------------------------------------
    def reset(self, **kwargs: Any) -> Any:
        obs, info = normalize_reset(self.env.reset(**kwargs))
        self._x_prev = None
        self._dense_return = 0.0
        pos = self._resolve_position(info if isinstance(info, dict) else {})
        if pos is None:
            # Fall back to observation[0] (root x position for these tasks).
            try:
                obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
                if obs_arr.size:
                    pos = float(obs_arr[0])
            except Exception:  # pragma: no cover - exotic observation types
                pos = None
        self._x_prev = pos
        return obs, info

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        dense_reward = float(reward)
        self._dense_return += dense_reward
        pos = self._resolve_position(info if isinstance(info, dict) else {})
        if pos is None:
            try:
                obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
                pos = float(obs_arr[0]) if obs_arr.size else None
            except Exception:  # pragma: no cover
                pos = None
        sparse_reward = self._sparse_reward(pos)
        if self.keep_dense_info and isinstance(info, dict):
            info.setdefault("dense_reward", dense_reward)
            info.setdefault("dense_return", self._dense_return)
            info.setdefault("sparse_reward", sparse_reward)
            if pos is not None:
                info.setdefault("x_position", pos)
        return obs, sparse_reward, terminated, truncated, info

    def state_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "sparse_mode": self.sparse_mode,
            "reward_scale": self.reward_scale,
            "x_prev": self._x_prev,
            "dense_return": self._dense_return,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.threshold = float(state.get("threshold", self.threshold))
        self.sparse_mode = str(state.get("sparse_mode", self.sparse_mode))
        self.reward_scale = float(state.get("reward_scale", self.reward_scale))
        self._x_prev = state.get("x_prev", None)
        self._dense_return = float(state.get("dense_return", 0.0))


class SparseInfoWrapper(WrapperBase):
    """Guarantee an ``info`` dict and expose the live forward position.

    Useful for Go-Explore style state save/restore: the sparse reward depends on
    ``x_prev`` which is *episode-local* state, so ``env_reset.py`` must be able
    to persist it.  This wrapper mirrors the underlying position into
    ``info["x_position"]`` on every step and reset.
    """

    def reset(self, **kwargs: Any) -> Any:
        obs, info = normalize_reset(self.env.reset(**kwargs))
        if not isinstance(info, dict):
            info = {}
        pos = forward_position(self.env, info)
        if pos is not None:
            info["x_position"] = pos
        return obs, info

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        if not isinstance(info, dict):
            info = {}
        pos = forward_position(self.env, info)
        if pos is not None:
            info.setdefault("x_position", pos)
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def make_env(
    name: str = "SparseHopper",
    threshold: Optional[float] = None,
    sparse_mode: Optional[str] = None,
    reward_scale: float = 1.0,
    seed: Optional[int] = None,
    normalize: Optional[bool] = None,
    normalize_obs: Optional[bool] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    gym_id: Optional[str] = None,
    **env_kwargs: Any,
) -> Any:
    """Build one sparse-reward MuJoCo environment (§C.2).

    Parameters mirror :func:`rice.environments.mujoco_dense.make_env` and add
    the sparse-specific knobs ``threshold`` and ``sparse_mode``.  Values default
    to the per-task :class:`MujocoSparseEnvSpec`.

    The returned object is a *single* (non-vectorized) environment, matching
    Algorithm 1/2's sequential roll-out design.  ``rise_canonical_name`` and
    ``rise_threshold`` attributes are attached for downstream tooling.
    """

    canonical = resolve_name(name)
    spec = SPARSE_SPECS[canonical]

    thr = float(spec.threshold if threshold is None else threshold)
    mode = str(spec.sparse_mode if sparse_mode is None else sparse_mode)
    steps = int(spec.max_episode_steps if max_episode_steps is None else max_episode_steps)
    if normalize is None:
        normalize = normalize_obs
    if normalize is None:
        normalize = bool(spec.normalize_obs)

    # The dense pipeline handles -v3/-v4 fallback, healthy-reward re-injection,
    # observation normalization and episode-length enforcement (§C.2).
    base = make_dense_env(
        spec.base_name,
        normalize=normalize,
        max_episode_steps=steps,
        render_mode=render_mode,
        gym_id=gym_id,
        seed=seed,
        **env_kwargs,
    )

    env: Any = base
    env = SparseInfoWrapper(env)
    env = SparseRewardWrapper(
        env,
        threshold=thr,
        sparse_mode=mode,
        reward_scale=reward_scale,
        keep_dense_info=True,
    )

    try:
        env.rise_canonical_name = canonical
        env.rise_threshold = thr
        env.rise_sparse_mode = mode
        env.rise_base_name = spec.base_name
    except Exception:  # pragma: no cover - attribute assignment is best effort
        pass

    if seed is not None:
        try:
            from ..utils.seeding import seed_env

            seed_env(env, seed)
        except Exception:  # pragma: no cover - optional convenience only
            pass

    return env


def make_mujoco_sparse(name: str = "SparseHopper", **kwargs: Any) -> Any:
    """Registry-friendly alias of :func:`make_env`."""

    return make_env(name=name, **kwargs)


def make_sparse_env(name: str = "SparseHopper", **kwargs: Any) -> Any:
    """Alias kept for symmetry with :mod:`mujoco_dense`."""

    return make_env(name=name, **kwargs)


def make_sparse_hopper(**kwargs: Any) -> Any:
    """``SparseHopper`` (threshold 0.6) — Experiment II in scope."""

    return make_env(name="SparseHopper", **kwargs)


def make_sparse_walker2d(**kwargs: Any) -> Any:
    """``SparseWalker2d`` (threshold 0.6) — OUT OF SCOPE for refining."""

    return make_env(name="SparseWalker2d", **kwargs)


def make_sparse_halfcheetah(**kwargs: Any) -> Any:
    """``SparseHalfCheetah`` (threshold 5.0) — Experiment II in scope."""

    return make_env(name="SparseHalfCheetah", **kwargs)


def sparse_tasks() -> Tuple[str, ...]:
    """Canonical sparse task names."""

    return tuple(SPARSE_SPECS.keys())


def in_scope_tasks() -> Tuple[str, ...]:
    """Sparse tasks used as Experiment II success criteria (addendum)."""

    return tuple(name for name, spec in SPARSE_SPECS.items() if spec.in_scope)


def register_fallbacks(force: bool = False) -> bool:
    """Proxy to :func:`mujoco_dense.register_fallbacks` (CPU-only stand-ins).

    Requires ``RICE_ALLOW_MUJOCO_FALLBACK=1`` unless ``force`` is True.
    """

    try:
        from .mujoco_dense import register_fallbacks as _register
    except Exception:  # pragma: no cover
        return False
    return bool(_register(force=force))


__all__ = [
    "MujocoSparseEnvSpec",
    "SPARSE_SPECS",
    "SparseRewardWrapper",
    "SparseInfoWrapper",
    "forward_position",
    "resolve_name",
    "get_spec",
    "make_env",
    "make_mujoco_sparse",
    "make_sparse_env",
    "make_sparse_hopper",
    "make_sparse_walker2d",
    "make_sparse_halfcheetah",
    "sparse_tasks",
    "in_scope_tasks",
    "register_fallbacks",
]
