"""Shared helpers for the RICE environment wrappers.

This module centralises the small amount of *plumbing* that every environment
wrapper in :mod:`rice.environments` needs:

* a tolerant detection of ``gym`` / ``gymnasium`` (both APIs are supported, and
  the module still imports on a machine where neither is installed, so that
  ``import rice`` never hard-fails because of a missing simulator),
* minimal ``Box`` / ``Discrete`` space fallbacks,
* reset/step return-signature normalisation (4-tuple legacy gym vs. 5-tuple
  gymnasium),
* episode-length discovery,
* access to :class:`rice.utils.seeding.RNG` (with a numpy fallback),
* access to :class:`rice.algorithms.rnd.RunningMeanStd` (with a local fallback)
  used by the observation normalisation wrappers described in Appendix C.2
  ("normalize the observation when training the DRL agent").

Nothing in this file depends on the paper's algorithms other than the two reuse
helpers above; it is a pure leaf utility.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "gym",
    "spaces",
    "IS_GYMNASIUM",
    "GYM_AVAILABLE",
    "EnvBase",
    "WrapperBase",
    "Box",
    "Discrete",
    "DictSpace",
    "MultiDiscrete",
    "make_box",
    "make_discrete",
    "normalize_reset",
    "normalize_step",
    "env_max_episode_steps",
    "get_rng_class",
    "import_running_mean_std",
    "RunningMeanStdFallback",
]

# ---------------------------------------------------------------------------
# gym / gymnasium detection
# ---------------------------------------------------------------------------
gym = None  # type: ignore
spaces = None  # type: ignore
IS_GYMNASIUM = False
GYM_AVAILABLE = False

try:  # pragma: no cover - depends on the installed stack
    import gym as _gym  # type: ignore

    gym = _gym
    spaces = _gym.spaces
    GYM_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        import gymnasium as _gymnas  # type: ignore

        gym = _gymnas
        spaces = _gymnas.spaces
        IS_GYMNASIUM = True
        GYM_AVAILABLE = True
    except Exception:
        gym = None
        spaces = None
        GYM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Minimal space fallbacks (used only when gym is unavailable)
# ---------------------------------------------------------------------------
class _FallbackSpace(object):
    """Very small subset of the gym space API."""

    def __init__(self, shape=None, dtype=np.float32, n=None, low=None, high=None):
        self.shape = tuple(shape) if shape is not None else ()
        self.dtype = dtype
        self.n = n
        self.low = low
        self.high = high

    def sample(self):  # pragma: no cover - trivial
        if self.n is not None:
            return int(np.random.randint(self.n))
        return np.zeros(self.shape, dtype=self.dtype)

    def contains(self, x) -> bool:  # pragma: no cover - trivial
        return True


class _FallbackBox(_FallbackSpace):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        super().__init__(shape=low.shape if shape is None else shape, dtype=dtype, low=low, high=high)

    def sample(self):  # pragma: no cover - trivial
        return np.random.uniform(self.low, self.high).astype(np.float32)


class _FallbackDiscrete(_FallbackSpace):
    def __init__(self, n, dtype=np.int64):
        super().__init__(shape=(), dtype=dtype, n=int(n), low=0, high=int(n) - 1)

    def sample(self):  # pragma: no cover - trivial
        return int(np.random.randint(self.n))


class _FallbackDict(_FallbackSpace):
    def __init__(self, mapping):
        super().__init__()
        self.spaces = dict(mapping)

    def sample(self):  # pragma: no cover - trivial
        return {k: v.sample() for k, v in self.spaces.items()}


class _FallbackMultiDiscrete(_FallbackSpace):
    def __init__(self, nvec):
        self.nvec = np.asarray(nvec, dtype=np.int64)
        super().__init__(shape=self.nvec.shape, dtype=np.int64, n=int(self.nvec.prod()))

    def sample(self):  # pragma: no cover - trivial
        return np.array([np.random.randint(int(n)) for n in self.nvec], dtype=np.int64)


if spaces is not None:
    Box = spaces.Box
    Discrete = spaces.Discrete
    DictSpace = spaces.Dict
    MultiDiscrete = getattr(spaces, "MultiDiscrete", _FallbackMultiDiscrete)
else:  # pragma: no cover - only on a machine without gym
    Box = _FallbackBox
    Discrete = _FallbackDiscrete
    DictSpace = _FallbackDict
    MultiDiscrete = _FallbackMultiDiscrete


def make_box(low, high, shape=None, dtype=np.float32):
    """Create a Box space that works with either gym stack (or the fallback)."""
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if shape is None:
        shape = low.shape
    if low.shape != shape:
        low = np.broadcast_to(low, shape).astype(np.float32)
    if high.shape != shape:
        high = np.broadcast_to(high, shape).astype(np.float32)
    try:
        return Box(low=low, high=high, dtype=dtype)
    except TypeError:  # pragma: no cover - fallback space signature
        return _FallbackBox(low, high, shape=shape, dtype=dtype)


def make_discrete(n: int):
    """Create a Discrete space that works with either gym stack (or fallback)."""
    try:
        return Discrete(int(n))
    except Exception:  # pragma: no cover
        return _FallbackDiscrete(int(n))


# ---------------------------------------------------------------------------
# Environment / wrapper base classes
# ---------------------------------------------------------------------------
if GYM_AVAILABLE:
    class EnvBase(gym.Env):  # type: ignore[misc, valid-type]
        """``gym.Env`` subclass so that SB3 / gym utilities accept our envs."""

        metadata = {"render.modes": []}

    class WrapperBase(gym.Wrapper):  # type: ignore[misc, valid-type]
        """``gym.Wrapper`` subclass; spaces are delegated to the wrapped env."""

        def __init__(self, env):
            super().__init__(env)

else:  # pragma: no cover - only without gym
    class EnvBase(object):
        """Duck-typed stand-in for ``gym.Env``."""

        metadata = {"render.modes": []}

    class WrapperBase(object):
        """Duck-typed stand-in for ``gym.Wrapper``."""

        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            # ``self.__dict__`` lookup guard avoids infinite recursion.
            if name.startswith("__") or name in self.__dict__:
                raise AttributeError(name)
            return getattr(self.__dict__["env"], name)

        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

        def step(self, action):
            return self.env.step(action)

        def render(self, *args, **kwargs):
            return self.env.render(*args, **kwargs)

        def close(self):
            return self.env.close()


# ---------------------------------------------------------------------------
# Reset / step signature normalisation
# ---------------------------------------------------------------------------
def normalize_reset(result) -> Tuple[Any, Dict[str, Any]]:
    """Normalise the return value of ``env.reset`` to ``(obs, info)``."""
    if isinstance(result, tuple):
        if len(result) == 1:
            return result[0], {}
        obs, info = result[0], result[1]
        return obs, (info if isinstance(info, dict) else {})
    return result, {}


def normalize_step(result) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalise ``env.step`` to ``(obs, reward, terminated, truncated, info)``."""
    if not isinstance(result, tuple):  # pragma: no cover - defensive
        raise TypeError("env.step must return a tuple")
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), (info or {})
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, (info or {})
    raise ValueError("unexpected env.step return of length %d" % len(result))


def env_max_episode_steps(env, default: int = 1000) -> int:
    """Best-effort lookup of the episode length (``TimeLimit`` aware)."""
    for attr in ("max_episode_steps", "_max_episode_steps"):
        try:
            value = getattr(env, attr, None)
        except Exception:
            value = None
        if isinstance(value, (int, np.integer)) and value > 0:
            return int(value)
    try:
        spec = getattr(env, "spec", None)
        if spec is not None and getattr(spec, "max_episode_steps", None):
            return int(spec.max_episode_steps)
    except Exception:
        pass
    current = env
    for _ in range(8):
        current = getattr(current, "env", None)
        for attr in ("max_episode_steps", "_max_episode_steps"):
            try:
                value = getattr(current, attr, None)
            except Exception:
                value = None
            if isinstance(value, (int, np.integer)) and value > 0:
                return int(value)
        if current is None:
            break
    return int(default)


# ---------------------------------------------------------------------------
# rice.utils.seeding.RNG (with numpy fallback)
# ---------------------------------------------------------------------------
def _import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


class _FallbackRNG(object):
    """Numpy-only stand-in for :class:`rice.utils.seeding.RNG`."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.generator = np.random.default_rng(seed)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self.generator.uniform(low, high))

    def bernoulli(self, p: float) -> bool:
        return bool(self.generator.uniform(0.0, 1.0) < p)

    def choice(self, a, p=None):
        return self.generator.choice(a, p=p)

    def integers(self, low, high=None, size=None):
        return self.generator.integers(low, high, size)

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.generator, item)


def get_rng_class():
    """Return the canonical RNG class, falling back to a numpy shim."""
    for mod_name in ("rice.utils.seeding", "rice.rice.utils.seeding", "utils.seeding"):
        module = _import_module(mod_name)
        if module is not None and hasattr(module, "RNG"):
            return module.RNG
    return _FallbackRNG


# ---------------------------------------------------------------------------
# RunningMeanStd (reuse from rice.algorithms.rnd when available)
# ---------------------------------------------------------------------------
class RunningMeanStdFallback(object):
    """Numerically stable running mean/variance (Welford) -- local fallback."""

    def __init__(self, shape=(), epsilon: float = 1e-4, clip: Optional[float] = None):
        self.shape = tuple(shape) if not np.isscalar(shape) else (int(shape),)
        self.epsilon = float(epsilon)
        self.clip = clip
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(epsilon)

    @property
    def std(self):
        return np.sqrt(self.var)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape) and x.shape != self.shape:
            x = x.reshape(self.shape if len(self.shape) else x.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > (1 if self.shape == () else len(self.shape) - 1) else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip: Optional[float] = None, epsilon: Optional[float] = None, update: bool = False):
        x = np.asarray(x, dtype=np.float64)
        if update:
            self.update(np.atleast_2d(x) if x.ndim == 1 else x)
        eps = self.epsilon if epsilon is None else epsilon
        out = (x - self.mean) / np.sqrt(self.var + eps)
        clip_value = self.clip if clip is None else clip
        if clip_value is not None:
            out = np.clip(out, -clip_value, clip_value)
        return out.astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count, "shape": self.shape}

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
        if "shape" in state:
            self.shape = tuple(state["shape"])


def import_running_mean_std():
    """Return :class:`rice.algorithms.rnd.RunningMeanStd` if importable."""
    for mod_name in ("rice.algorithms.rnd", "rice.rice.algorithms.rnd", "algorithms.rnd", "..algorithms.rnd"):
        if mod_name.startswith(".."):
            continue
        module = _import_module(mod_name)
        if module is not None and hasattr(module, "RunningMeanStd"):
            return module.RunningMeanStd
    # last resort: relative import (works when the package name is ``rice``)
    try:  # pragma: no cover - depends on packaging layout
        from ..algorithms.rnd import RunningMeanStd  # type: ignore

        return RunningMeanStd
    except Exception:
        return RunningMeanStdFallback
