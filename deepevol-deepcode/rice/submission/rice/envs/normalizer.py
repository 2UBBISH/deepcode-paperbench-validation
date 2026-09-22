"""Observation normalization for the RICE environments.

Appendix C.2 of the paper (Cheng et al., ICML 2024) states explicitly, for two of
the MuJoCo games:

* "We use ``Walker2d-v3`` in our experiments and **normalize the observation** when
  training the DRL agent."
* "We use ``HalfCheetah-v3`` in our experiments and **normalize the observation**
  when training the DRL agent."

Hopper, Reacher, Selfish Mining, CAGE-2 and autonomous driving are *not* described
as normalized, so the wrapper is a no-op for them (``should_normalize`` returns
False and the factory simply does not wrap them).

Two modes are supported, mirroring common practice:

``train``  statistics are updated online from every observed transition
``eval``   normalization is performed with frozen (loaded) statistics, which is
           required for reproducible *no-refine* / *refine* reward comparisons and
           for the fidelity evaluator (Experiment I) that must not let the
           normalizer drift while a trajectory is measured.

The statistics are maintained by :class:`rice.utils.metrics.RunningMeanStd`
(streaming Welford update), and the transform is ``(x - mean) / (std + eps)`` with
the result clipped to ``+/- clip`` (default 10), see
:func:`rice.utils.metrics.normalize` and ``RunningMeanStd.__call__``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # gym is the primary dependency, but keep the module importable without it
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - gym missing
    gym = None
    spaces = None

from ..utils.io import ensure_dir, save_pickle, load_pickle
from ..utils.metrics import RunningMeanStd, normalize

__all__ = [
    "NORMALIZED_ENVS",
    "should_normalize",
    "make_normalizer",
    "ObservationNormalizer",
    "NormalizeObservation",
]


# --------------------------------------------------------------------------------------
# Which environments the paper normalizes
# --------------------------------------------------------------------------------------
#: Environment ids whose observations the paper normalizes (§C.2: Walker2d, HalfCheetah).
NORMALIZED_ENVS: Tuple[str, ...] = (
    "walker2d",
    "halfcheetah",
)


def should_normalize(env_id: str) -> bool:
    """Return ``True`` if the paper normalizes observations for ``env_id``.

    Args:
        env_id: Gym id (``"Walker2d-v3"``, ``"SparseHalfCheetah-v3"``, ...) or a
            short alias such as ``"walker2d"``.

    Returns:
        Whether :class:`ObservationNormalizer` should wrap the environment.
    """
    name = str(env_id).lower()
    # strip common prefixes used by the RICE env registry
    for prefix in ("sparse_", "sparse-", "sparse"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if "-v" in name:
        name = name.split("-v", 1)[0]
    return name in NORMALIZED_ENVS


# --------------------------------------------------------------------------------------
# Wrapper
# --------------------------------------------------------------------------------------
class ObservationNormalizer(gym.Wrapper if gym is not None else object):
    """Normalize observations with an online mean/std estimate.

    Args:
        env: Gym environment (or any object exposing ``observation_space``,
            ``reset`` and ``step``).
        epsilon: Numerical stabilizer added to the std.
        clip: Absolute value used to clip normalized observations.
        mode: ``"train"`` updates the statistics from every transition,
            ``"eval"``/``"frozen"`` never updates them.
        stats: Optional pre-computed statistics ``{"mean": ..., "var": ...,
            "count": ...}`` used to warm-start the normalizer (e.g. loaded from a
            checkpoint), so that a frozen evaluator matches the training-time
            normalization exactly.
        default_stats: Optional ``(mean, std)`` taken from the environment's
            ``observation_space`` when available; used to initialize the running
            statistics for the first observations (gym spaces usually expose only
            bounds, so the midpoint is used instead).
    """

    def __init__(
        self,
        env: Any,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        mode: str = "train",
        stats: Optional[Dict[str, Any]] = None,
        default_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    ) -> None:
        if gym is not None:
            super().__init__(env)
        else:  # pragma: no cover
            self.env = env
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.mode = str(mode).lower()
        self._frozen = self.mode in ("eval", "evaluation", "frozen", "test")

        shape = self._observation_shape(env)
        self.rms = RunningMeanStd(shape=shape, epsilon=1e-4)

        if default_stats is not None:
            mean, std = default_stats
            self.rms.mean = np.asarray(mean, dtype=np.float64).copy()
            self.rms.var = np.square(np.asarray(std, dtype=np.float64))
            self.rms.count = 1.0

        if stats:
            self.load_stats(stats)

        # keep the (unchanged) gym space to avoid breaking downstream checks
        self._raw_observation_space = getattr(env, "observation_space", None)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _observation_shape(env: Any):
        space = getattr(env, "observation_space", None)
        if space is not None and hasattr(space, "shape") and space.shape is not None:
            return tuple(space.shape)
        # fall back to a flat vector based on the env's declared dimensionality
        n = getattr(env, "observation_dim", None) or getattr(env, "obs_dim", None)
        if n is not None:
            return (int(n),)
        return ()

    def _normalize(self, obs: Any) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.size == 0 or self.rms.mean.size == 0:
            return obs
        return normalize(obs, self.rms.mean, self.rms.std, eps=self.epsilon)

    # ---------------------------------------------------------------- gym API
    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        obs, info = (out if isinstance(out, tuple) else (out, {}))
        obs = np.asarray(obs, dtype=np.float64)
        if not self._frozen:
            self.rms.update(obs)
        normed = np.clip(self._normalize(obs), -self.clip, self.clip)
        return (normed, info) if isinstance(out, tuple) else normed

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
        else:  # pragma: no cover - old gym API
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        obs = np.asarray(obs, dtype=np.float64)
        if not self._frozen:
            self.rms.update(obs)
        normed = np.clip(self._normalize(obs), -self.clip, self.clip)
        if len(out) == 5:
            return normed, reward, terminated, truncated, info
        return normed, reward, done, info  # pragma: no cover

    # ------------------------------------------------------------- statistics
    def get_stats(self) -> Dict[str, Any]:
        """Return the current normalization statistics as plain python objects."""
        return {
            "mean": np.asarray(self.rms.mean).tolist(),
            "var": np.asarray(self.rms.var).tolist(),
            "count": float(self.rms.count),
        }

    def load_stats(self, stats: Dict[str, Any]) -> None:
        """Restore statistics produced by :meth:`get_stats`."""
        self.rms.mean = np.asarray(stats["mean"], dtype=np.float64)
        self.rms.var = np.asarray(stats["var"], dtype=np.float64)
        self.rms.count = float(stats.get("count", 1.0))

    def freeze(self) -> "ObservationNormalizer":
        """Stop updating the statistics (used for evaluation)."""
        self._frozen = True
        self.mode = "eval"
        return self

    def unfreeze(self) -> "ObservationNormalizer":
        """Resume updating the statistics (used for training)."""
        self._frozen = False
        self.mode = "train"
        return self

    def save_stats(self, path: str) -> str:
        """Persist the statistics to ``path`` (pickle)."""
        directory = os.path.dirname(os.path.abspath(path))
        ensure_dir(directory)
        return save_pickle(self.get_stats(), path)

    def load_stats_from(self, path: str) -> "ObservationNormalizer":
        """Load statistics previously written by :meth:`save_stats`."""
        self.load_stats(load_pickle(path))
        return self


# SB3-compatible alias (the paper's agents are trained with Stable-Baselines3).
NormalizeObservation = ObservationNormalizer


def make_normalizer(env: Any, env_id: Optional[str] = None, **kwargs) -> Any:
    """Wrap ``env`` with :class:`ObservationNormalizer` when the paper says so.

    Args:
        env: Environment to (maybe) wrap.
        env_id: Environment identifier; when ``None`` the wrapper is applied only
            if ``env`` already exposes ``observation_space`` and the caller opts in
            via ``force=True``.
        **kwargs: Forwarded to :class:`ObservationNormalizer`. ``force=True``
            overrides :func:`should_normalize`.

    Returns:
        Either ``env`` unchanged or the wrapped environment.
    """
    force = bool(kwargs.pop("force", False))
    if not force and (env_id is None or not should_normalize(env_id)):
        return env
    return ObservationNormalizer(env, **kwargs)
