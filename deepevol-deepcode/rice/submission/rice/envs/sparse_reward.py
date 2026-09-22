"""Sparse reward variants of the MuJoCo games used by RICE.

Paper reference (Appendix C.2, "Extra Introduction to Applications"):

    "Under the sparse reward setting (Mazoure et al., 2019), the reward informs
     the x position of the hopper only if ``x > 0.6`` in our experiments."
     (Hopper and Walker2d)

    "Under the sparse reward setting (Mazoure et al., 2019), the reward informs
     the x position of the hopper only if ``x > 5`` in our experiments."
     (HalfCheetah)

So the sparse reward is the *forward position* of the agent (the x-coordinate of
the torso / root), emitted only once the forward threshold has been exceeded.
This matches the standard Mazoure et al. (2019) sparse-reward formulation used
for SparseHopper / SparseHalfCheetah (and SparseWalker2d, which the paper lists
but whose refining experiments are out of scope).

The wrapper is intentionally defensive:

* it works with both the old 4-tuple (``obs, reward, done, info``) and the new
  5-tuple (``obs, reward, terminated, truncated, info``) gym step APIs;
* it recovers the forward position from ``info`` when the simulator exposes it
  (``x_position``/``x_pos``) and otherwise falls back to the first observation
  dimension, which is the x-coordinate for Hopper / Walker2d / HalfCheetah (the
  paper notes observations start with positional values);
* the original, dense reward is always preserved in ``info["dense_reward"]`` so
  downstream code (refinement logging, fidelity evaluator) can still inspect it.

Thresholds (also mirrored in :mod:`rice.envs.make_env` as
``SPARSE_THRESHOLDS``):

    hopper       0.6
    walker2d     0.6
    halfcheetah  5.0
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - gym is an optional dependency for smoke tests
    import gym

    _GYM_BASE = gym.Wrapper
except Exception:  # pragma: no cover
    gym = None
    _GYM_BASE = object


__all__ = [
    "SPARSE_THRESHOLDS",
    "SPARSE_X_KEYS",
    "sparse_reward_from_x",
    "forward_position",
    "threshold_for",
    "SparseRewardWrapper",
    "SparseReward",
    "make_sparse_reward_env",
    "wrap_sparse_reward",
    "build_sparse_reward_fn",
    "is_sparse_env_id",
]


# ---------------------------------------------------------------------------
# Thresholds (Appendix C.2)
# ---------------------------------------------------------------------------
#: Forward-position thresholds at which the sparse reward starts being emitted.
SPARSE_THRESHOLDS: Dict[str, float] = {
    "hopper": 0.6,
    "walker2d": 0.6,
    "halfcheetah": 5.0,
}

#: Keys looked up (in order) inside ``info`` to recover the forward position.
SPARSE_X_KEYS: Tuple[str, ...] = (
    "x_position",
    "x_pos",
    "xpos",
    "forward_x",
    "torso_x",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _spec_key(env_id: str) -> str:
    """Strip ``sparse`` prefixes and ``-vN`` suffixes to get the base env key."""
    if not isinstance(env_id, str):
        return ""
    name = env_id.lower()
    for token in ("sparse-", "sparse_", "sparse"):
        if name.startswith(token):
            name = name[len(token):]
            break
    name = name.split("-v")[0]
    name = name.split("-")[0]
    return name.strip("_ ")


def is_sparse_env_id(env_id: str) -> bool:
    """Return True if ``env_id`` names one of the sparse MuJoCo games."""
    if not isinstance(env_id, str):
        return False
    name = env_id.lower()
    return name.startswith("sparse-") or name.startswith("sparse_")


def forward_position(
    obs: Sequence[float],
    info: Optional[Dict[str, Any]] = None,
    obs_index: int = 0,
) -> float:
    """Recover the agent's forward (x) position.

    Preference order:
      1. a well-known key inside ``info`` (the MuJoCo v3 environments expose
         ``x_position`` in their info dict),
      2. ``obs[obs_index]`` -- the first observation entry is the x-coordinate of
         the torso/root for Hopper, Walker2d and HalfCheetah.
    """
    if isinstance(info, dict):
        for key in SPARSE_X_KEYS:
            if key in info:
                try:
                    value = info[key]
                    if isinstance(value, np.ndarray):
                        value = float(np.asarray(value).reshape(-1)[0])
                    return float(value)
                except Exception:
                    continue
    try:
        arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return 0.0
        idx = int(obs_index) % arr.size
        return float(arr[idx])
    except Exception:
        return 0.0


def sparse_reward_from_x(x: float, threshold: float) -> float:
    """The Mazoure et al. (2019) sparse reward: forward position past threshold.

    ``r = x if x > threshold else 0``.
    """
    x = float(x)
    if x > float(threshold):
        return x
    return 0.0


def threshold_for(env_id: str, default: Optional[float] = None) -> float:
    """Return the sparse threshold configured for ``env_id``."""
    key = _spec_key(env_id)
    if key in SPARSE_THRESHOLDS:
        return SPARSE_THRESHOLDS[key]
    if default is not None:
        return float(default)
    # Unknown environment -> conservative, non-zero threshold.
    return 0.6


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class SparseRewardWrapper(_GYM_BASE):  # type: ignore[misc]
    """Replace the dense MuJoCo reward with the sparse forward-position signal.

    Parameters
    ----------
    env:
        The (dense) environment to wrap.
    threshold:
        Forward position past which the reward becomes non-zero.  Defaults to
        the value listed for the environment in :data:`SPARSE_THRESHOLDS`
        (Hopper/Walker2d ``0.6``, HalfCheetah ``5.0``).
    env_id:
        Optional environment identifier used to look up the default threshold.
    obs_index:
        Fallback index of the forward position in the observation vector.
    reward_scale:
        Multiplicative factor applied to the sparse reward (default 1.0).
    keep_dense:
        If True (default) the original dense reward is stored in
        ``info["dense_reward"]``.
    use_info:
        If True (default) prefer ``info['x_position']``-style keys over the
        observation entry when recovering the forward position.
    """

    def __init__(
        self,
        env: Any,
        threshold: Optional[float] = None,
        env_id: Optional[str] = None,
        obs_index: int = 0,
        reward_scale: float = 1.0,
        keep_dense: bool = True,
        use_info: bool = True,
    ) -> None:
        if _GYM_BASE is object:  # pragma: no cover - gym missing
            self.env = env
        else:
            super().__init__(env)
        resolved_id = env_id
        if resolved_id is None:
            spec = getattr(env, "spec", None)
            resolved_id = (
                getattr(spec, "id", None)
                or getattr(env, "rice_env_key", None)
                or getattr(env, "rice_env_id", None)
            )
        if threshold is None:
            threshold = threshold_for(resolved_id or "", default=0.6)
        self.env_id = resolved_id
        self.threshold = float(threshold)
        self.obs_index = int(obs_index)
        self.reward_scale = float(reward_scale)
        self.keep_dense = bool(keep_dense)
        self.use_info = bool(use_info)

        # Lightweight bookkeeping, handy for logging / d_max estimation.
        self.episode_sparse_return = 0.0
        self.episode_dense_return = 0.0
        self._last_x = 0.0
        self._last_dense_reward = 0.0

    # -- internals ---------------------------------------------------------
    def _resolve_x(self, obs: Any, info: Dict[str, Any]) -> float:
        if self.use_info:
            return forward_position(obs, info, obs_index=self.obs_index)
        return forward_position(obs, None, obs_index=self.obs_index)

    def _convert(self, obs: Any, reward: float, info: Any):
        info = dict(info) if isinstance(info, dict) else {}
        x = self._resolve_x(obs, info)
        sparse_r = self.reward_scale * sparse_reward_from_x(x, self.threshold)
        self._last_x = x
        self._last_dense_reward = float(reward)
        if self.keep_dense:
            info["dense_reward"] = float(reward)
        info["x_position_sparse"] = x
        info["sparse_threshold"] = self.threshold
        info["reward_sparse"] = float(sparse_r)
        self.episode_sparse_return += float(sparse_r)
        self.episode_dense_return += float(reward)
        return sparse_r, info

    # -- gym API -----------------------------------------------------------
    def reset(self, *args, **kwargs):
        self.episode_sparse_return = 0.0
        self.episode_dense_return = 0.0
        out = self.env.reset(*args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, dict(info) if isinstance(info, dict) else {}
        return out

    def step(self, action):
        out = self.env.step(action)
        if not isinstance(out, tuple):  # pragma: no cover - defensive
            return out

        if len(out) == 5:  # new gym API
            obs, reward, terminated, truncated, info = out
            sparse_r, info = self._convert(obs, reward, info)
            return obs, sparse_r, terminated, truncated, info

        if len(out) == 4:  # old gym API
            obs, reward, done, info = out
            sparse_r, info = self._convert(obs, reward, info)
            return obs, sparse_r, done, info

        return out  # pragma: no cover - unexpected API

    # -- introspection helpers --------------------------------------------
    def dense_reward(self) -> float:
        """Dense reward emitted at the most recent step."""
        return self._last_dense_reward

    def sparse_reward(self) -> float:
        """Sparse reward emitted at the most recent step."""
        return sparse_reward_from_x(self._last_x, self.threshold)

    def last_x_position(self) -> float:
        """Forward position observed at the most recent step."""
        return self._last_x

    def get_sparse_stats(self) -> Dict[str, float]:
        """Cumulative sparse/dense returns plus the last forward position."""
        return {
            "threshold": self.threshold,
            "episode_sparse_return": float(self.episode_sparse_return),
            "episode_dense_return": float(self.episode_dense_return),
            "last_x_position": float(self._last_x),
        }

    # -- passthrough for the Go-Explore style reset wrapper ----------------
    def get_state(self):
        inner = getattr(self.env, "get_state", None)
        if callable(inner):
            return inner()
        return None

    def set_state(self, state) -> bool:
        inner = getattr(self.env, "set_state", None)
        if callable(inner):
            return bool(inner(state))
        restore = getattr(self.env, "restore_state", None)
        if callable(restore):
            return bool(restore(state))
        return False


#: Convenience alias (some modules refer to the wrapper as ``SparseReward``).
SparseReward = SparseRewardWrapper


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def make_sparse_reward_env(
    env: Any,
    env_id: Optional[str] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> SparseRewardWrapper:
    """Wrap ``env`` with :class:`SparseRewardWrapper` (idempotent)."""
    if isinstance(env, SparseRewardWrapper):
        if threshold is not None:
            env.threshold = float(threshold)
        return env
    return SparseRewardWrapper(env, threshold=threshold, env_id=env_id, **kwargs)


def wrap_sparse_reward(env: Any, *args, **kwargs) -> SparseRewardWrapper:
    """Alias of :func:`make_sparse_reward_env` for readability."""
    return make_sparse_reward_env(env, *args, **kwargs)


def build_sparse_reward_fn(
    env_id: str,
    threshold: Optional[float] = None,
    obs_index: int = 0,
) -> Callable[[Any, float, Optional[Dict[str, Any]]], Tuple[float, Dict[str, Any]]]:
    """Return a stateless ``(obs, dense_reward, info) -> (r, info)`` callable.

    Useful when only the reward signal (not a full wrapper) is needed, e.g. in
    the RND / refinement loops that re-use the dense environment for
    bookkeeping.
    """
    thr = float(threshold) if threshold is not None else threshold_for(env_id)

    def _fn(obs: Any, dense_reward: float, info: Optional[Dict[str, Any]] = None):
        info = dict(info) if isinstance(info, dict) else {}
        x = forward_position(obs, info, obs_index=obs_index)
        info["dense_reward"] = float(dense_reward)
        info["x_position_sparse"] = x
        return sparse_reward_from_x(x, thr), info

    return _fn
