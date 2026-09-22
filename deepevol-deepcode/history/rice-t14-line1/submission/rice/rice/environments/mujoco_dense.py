"""Dense-reward MuJoCo environments used in the RICE paper (Appendix C.2).

Environments (verbatim from §C.2 Extra Introduction to Applications):

* ``Hopper-v3``      -- reward = healthy + forward + control cost; episode ends if the
                        Hopper becomes unhealthy (Erez et al., 2012).
* ``Walker2d-v3``    -- 6-dimensional action space; the reward combines a healthy reward
                        bonus, forward reward and control cost; "we use Walker2d-v3 in our
                        experiments and normalize the observation when training the DRL agent".
* ``Reacher-v2``     -- reward = "reward distance" - "reward control"; "an episode concludes
                        either after 50 timesteps with a new random target or if any state
                        space value becomes non-finite".
* ``HalfCheetah-v3`` -- reward balances positive "forward reward" with "control cost"
                        penalties; "episodes conclude after 1000 timesteps"; observation is
                        normalized when training the DRL agent.

In-scope deviations / defaults (the paper is silent on these and they are documented in the
package README):

* If only the newer Gymnasium/MuJoCo ``*-v4`` ids are installed, we fall back to them,
  re-injecting the ``healthy_reward`` term that v4 removed (and the ``terminate_when_unhealthy``
  semantics) so the dense reward keeps the paper's "healthy + forward + control" form.
* Observation normalization is implemented with a running mean/std (Welford) wrapper that is
  serializable, so refining runs can resume with the same statistics.
* On machines without MuJoCo a lightweight analytic stand-in (``FallbackLocomotionEnv``) can be
  registered under the canonical ids through ``RICE_ALLOW_MUJOCO_FALLBACK=1``; it preserves the
  observation/action dimensionalities and the reward structure and exists only so the algorithm
  logic (mask training, roll-in, restoring, fidelity evaluation) can be smoke-tested on CPU.

The module is import-safe: it never imports ``gym``/``mujoco`` eagerly through a hard dependency;
:func:`make_env` performs the actual lookup lazily and raises an informative ``ImportError`` when
a requested task truly cannot be built.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ._common import (  # noqa: F401  (re-exported plumbing)
    EnvBase,
    GYM_AVAILABLE,
    IS_GYMNASIUM,
    WrapperBase,
    env_max_episode_steps,
    import_running_mean_std,
    make_box,
    normalize_reset,
    normalize_step,
)

__all__ = [
    "MUJOCO_DENSE_SPECS",
    "MujocoDenseEnvSpec",
    "ObservationNormalizeWrapper",
    "TerminateOnNonFiniteWrapper",
    "HealthyRewardWrapper",
    "EpisodeLengthWrapper",
    "FallbackLocomotionEnv",
    "register_fallbacks",
    "make_hopper",
    "make_walker2d",
    "make_reacher",
    "make_halfcheetah",
    "make_mujoco_dense",
    "make_env",
    "resolve_name",
]


# --------------------------------------------------------------------------------------
# Task specifications (dimensionalities + §C.2 per-task settings)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MujocoDenseEnvSpec:
    """Static description of one dense-reward MuJoCo task (§C.2)."""

    name: str
    family: str
    gym_ids: Tuple[str, ...]
    obs_dim: int
    act_dim: int
    max_episode_steps: int
    normalize_obs: bool = False
    exclude_current_positions: bool = True
    terminate_on_nonfinite: bool = False
    inject_healthy_reward: bool = False
    terminate_when_unhealthy: bool = True
    net_arch: Tuple[int, ...] = (64, 64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "gym_ids": list(self.gym_ids),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "max_episode_steps": self.max_episode_steps,
            "normalize_obs": self.normalize_obs,
            "exclude_current_positions": self.exclude_current_positions,
            "terminate_on_nonfinite": self.terminate_on_nonfinite,
            "inject_healthy_reward": self.inject_healthy_reward,
            "terminate_when_unhealthy": self.terminate_when_unhealthy,
            "net_arch": list(self.net_arch),
        }


MUJOCO_DENSE_SPECS: Dict[str, MujocoDenseEnvSpec] = {
    "Hopper-v3": MujocoDenseEnvSpec(
        name="Hopper-v3",
        family="hopper",
        # v4 is used only as a fallback: v3 is what the paper uses.
        gym_ids=("Hopper-v3", "Hopper-v4"),
        obs_dim=11,
        act_dim=3,
        max_episode_steps=1000,
        normalize_obs=False,          # §C.2 normalizes only Walker2d / HalfCheetah
        exclude_current_positions=True,
        inject_healthy_reward=True,   # v4 removed healthy_reward from the return
        terminate_when_unhealthy=True,
    ),
    "Walker2d-v3": MujocoDenseEnvSpec(
        name="Walker2d-v3",
        family="walker2d",
        gym_ids=("Walker2d-v3", "Walker2d-v4"),
        obs_dim=17,
        act_dim=6,
        max_episode_steps=1000,
        normalize_obs=True,           # §C.2: "normalize the observation when training the DRL agent"
        exclude_current_positions=True,
        inject_healthy_reward=True,
        terminate_when_unhealthy=True,
    ),
    "Reacher-v2": MujocoDenseEnvSpec(
        name="Reacher-v2",
        family="reacher",
        gym_ids=("Reacher-v2", "Reacher-v4"),
        obs_dim=11,
        act_dim=2,
        max_episode_steps=50,         # §C.2: "an episode concludes either after 50 timesteps ..."
        normalize_obs=False,
        exclude_current_positions=True,
        terminate_on_nonfinite=True,  # §C.2: "... or if any state space value becomes non-finite"
        inject_healthy_reward=False,
        terminate_when_unhealthy=False,
    ),
    "HalfCheetah-v3": MujocoDenseEnvSpec(
        name="HalfCheetah-v3",
        family="halfcheetah",
        gym_ids=("HalfCheetah-v3", "HalfCheetah-v4"),
        obs_dim=17,
        act_dim=6,
        max_episode_steps=1000,       # §C.2: "Episodes conclude after 1000 timesteps"
        normalize_obs=True,           # §C.2: "normalize the observation when training the DRL agent"
        exclude_current_positions=True,
        inject_healthy_reward=False,
        terminate_when_unhealthy=False,
    ),
}


_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3",
    "hopper-v3": "Hopper-v3",
    "hopper-v4": "Hopper-v3",
    "walker2d": "Walker2d-v3",
    "walker": "Walker2d-v3",
    "walker2d-v3": "Walker2d-v3",
    "walker2d-v4": "Walker2d-v3",
    "reacher": "Reacher-v2",
    "reacher-v2": "Reacher-v2",
    "reacher-v4": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3",
    "half_cheetah": "HalfCheetah-v3",
    "halfcheetah-v3": "HalfCheetah-v3",
    "halfcheetah-v4": "HalfCheetah-v3",
}


def resolve_name(name: str) -> str:
    """Map a canonical name / friendlier alias onto a key of :data:`MUJOCO_DENSE_SPECS`."""
    key = str(name).strip()
    if key in MUJOCO_DENSE_SPECS:
        return key
    normalised = key.lower().replace(" ", "").replace("_", "")
    if normalised in _ALIASES:
        return _ALIASES[normalised]
    for alias, canonical in _ALIASES.items():
        if alias.replace("-", "").replace("_", "") == normalised:
            return canonical
    raise KeyError(
        f"Unknown MuJoCo dense task {name!r}. Known tasks: {sorted(MUJOCO_DENSE_SPECS)}"
    )


# --------------------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------------------
class ObservationNormalizeWrapper(WrapperBase):
    """Running mean/std observation normalization (§C.2 for Walker2d / HalfCheetah).

    The statistics are maintained with :class:`RunningMeanStd` from
    :mod:`rice.algorithms.rnd` (Welford, numerically stable) and are exposed through
    ``state_dict``/``load_state_dict`` so a refining run can resume training with identical
    normalization constants -- important because the RICE refining loop restores simulator
    states and replays observations through the policy.
    """

    _RMS = None  # resolved lazily on first instantiation

    def __init__(
        self,
        env,
        normalize: bool = True,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        update: bool = True,
    ) -> None:
        super().__init__(env)
        self.normalize = bool(normalize)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.update = bool(update)
        self.obs_rms = None
        if self.normalize:
            rms_cls = ObservationNormalizeWrapper._RMS or import_running_mean_std()
            ObservationNormalizeWrapper._RMS = rms_cls
            shape = self._observation_shape()
            self.obs_rms = rms_cls(shape=shape, epsilon=1e-4, clip=self.clip)
        try:
            self.observation_space = make_box(
                low=-np.inf, high=np.inf, shape=self._observation_shape(), dtype=np.float32
            )
        except Exception:  # pragma: no cover - space shims always work
            pass

    # -- helpers -----------------------------------------------------------------
    def _observation_shape(self) -> Tuple[int, ...]:
        space = getattr(self.env, "observation_space", None)
        if space is None:
            return (0,)
        shape = tuple(getattr(space, "shape", ()) or ())
        return shape if shape else (int(getattr(space, "n", 1)),)

    def _normalize(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float64)
        if self.obs_rms is None:
            return arr.astype(np.float32)
        try:
            out = self.obs_rms.normalize(arr, clip=self.clip, epsilon=self.epsilon, update=self.update)
        except TypeError:  # pragma: no cover - simpler RunningMeanStd signature
            if self.update:
                self.obs_rms.update(arr)
            out = self.obs_rms.normalize(arr, clip=self.clip)
        return np.asarray(out, dtype=np.float32)

    # -- gym API -----------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = normalize_reset(self.env.reset(**kwargs))
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        return self._normalize(obs), reward, terminated, truncated, info

    # -- statistics ---------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "normalize": self.normalize,
            "clip": self.clip,
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms is not None else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.normalize = bool(state.get("normalize", self.normalize))
        self.clip = float(state.get("clip", self.clip))
        if self.obs_rms is not None and state.get("obs_rms") is not None:
            self.obs_rms.load_state_dict(state["obs_rms"])


class TerminateOnNonFiniteWrapper(WrapperBase):
    """Reacher-v2 termination rule from §C.2: end the episode on non-finite state values."""

    def __init__(self, env, enabled: bool = True) -> None:
        super().__init__(env)
        self.enabled = bool(enabled)

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        if self.enabled:
            try:
                finite = bool(np.isfinite(np.asarray(obs, dtype=np.float64)).all())
            except Exception:
                finite = True
            if not finite:
                terminated = True
                info = dict(info or {})
                info.setdefault("non_finite_state", True)
        return obs, reward, terminated, truncated, info


class HealthyRewardWrapper(WrapperBase):
    """Re-inject the "healthy reward" term of the §C.2 dense reward decomposition.

    ``Hopper-v3``/``Walker2d-v3`` pay ``+1`` per step while the body is healthy. Gymnasium's
    ``-v4`` variants dropped that term (``healthy_reward = 0``), so when we fall back to v4 we
    add it back to keep the paper's ``reward = healthy + forward + control`` structure. When the
    environment already pays a non-zero healthy reward this wrapper is a no-op.
    """

    def __init__(self, env, healthy_reward: float = 1.0, enabled: bool = True) -> None:
        super().__init__(env)
        self.healthy_reward = float(healthy_reward)
        self.enabled = bool(enabled)

    def _healthy(self) -> bool:
        inner = getattr(self.env, "unwrapped", self.env)
        for attr in ("is_healthy", "_is_healthy"):
            func = getattr(inner, attr, None)
            if callable(func):
                try:
                    return bool(func())
                except Exception:
                    return True
        # Walker2d keeps its health criterion inline in step(); re-derive it when possible.
        try:
            data = getattr(inner, "data", None)
            if data is not None and hasattr(data, "qpos"):
                angle = float(np.asarray(data.qpos)[1])
                return abs(angle) < 1.0
        except Exception:
            pass
        return True

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        if self.enabled and not terminated:
            existing = 0.0
            inner = getattr(self.env, "unwrapped", self.env)
            try:
                existing = float(getattr(inner, "healthy_reward", 0.0) or 0.0)
            except Exception:
                existing = 0.0
            if existing == 0.0 and self._healthy():
                reward = float(reward) + self.healthy_reward
        return obs, reward, terminated, truncated, info


class EpisodeLengthWrapper(WrapperBase):
    """Enforce ``max_episode_steps`` (the ``-v3`` ids already carry a ``TimeLimit``)."""

    def __init__(self, env, max_episode_steps: Optional[int] = None) -> None:
        super().__init__(env)
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self._elapsed = 0

    def reset(self, **kwargs):
        self._elapsed = 0
        return normalize_reset(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        self._elapsed += 1
        if self.max_episode_steps is not None and self._elapsed >= self.max_episode_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


# --------------------------------------------------------------------------------------
# CPU-only stand-in (only used when MuJoCo is unavailable and explicitly allowed)
# --------------------------------------------------------------------------------------
class FallbackLocomotionEnv(EnvBase):
    """Minimal analytic locomotion stand-in preserving the paper's I/O contract.

    NOT part of the paper: it exists purely so the RICE algorithm logic can be exercised on
    machines without MuJoCo (``RICE_ALLOW_MUJOCO_FALLBACK=1``). It exposes the same observation
    and action dimensionalities, the same ``reward = healthy + forward + control`` structure and
    a raw ``state`` array so the Go-Explore state save/restore path can be unit-tested.
    """

    def __init__(
        self,
        obs_dim: int = 11,
        act_dim: int = 3,
        max_episode_steps: int = 1000,
        healthy_reward: float = 1.0,
        control_cost_weight: float = 0.05,
        healthy_angle_range: float = 1.0,
        terminate_when_unhealthy: bool = True,
        sparse_threshold: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.healthy_reward = float(healthy_reward)
        self.control_cost_weight = float(control_cost_weight)
        self.healthy_angle_range = float(healthy_angle_range)
        self.terminate_when_unhealthy = bool(terminate_when_unhealthy)
        self.sparse_threshold = sparse_threshold
        self.action_space = make_box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
        self.observation_space = make_box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.metadata = {"render.modes": []} if not hasattr(self, "metadata") else self.metadata
        self._rng = np.random.default_rng(seed)
        self._elapsed = 0
        self._state = np.zeros(self.obs_dim, dtype=np.float64)
        self.state = self._state.copy()

    # -- state handling (Go-Explore style; the real envs are handled via mujoco sim state)
    def set_state_from_observation(self, obs: np.ndarray) -> None:
        self._state = np.asarray(obs, dtype=np.float64).copy()
        self.state = self._state.copy()

    def reset(self, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._elapsed = 0
        self._state = self._rng.normal(0.0, 0.05, size=self.obs_dim)
        # index 0 is the forward position x, index 1 the "torso angle" for the health check.
        self._state[0] = 0.0
        self.state = self._state.copy()
        return self._state.astype(np.float32), {}

    def _healthy(self) -> bool:
        if self.obs_dim < 2:
            return True
        return abs(float(self._state[1])) < self.healthy_angle_range

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if action.size < self.act_dim:
            action = np.pad(action, (0, self.act_dim - action.size))
        action = action[: self.act_dim]
        prev_x = float(self._state[0])
        healthy = self._healthy()
        # crude but reproducible first-order dynamics
        self._state[0] = prev_x + 0.02 * float(action.sum()) + 0.01 * self._rng.normal()
        if self.obs_dim > 1:
            self._state[1] = 0.9 * self._state[1] + 0.05 * float(action.mean()) + 0.01 * self._rng.normal()
        if self.obs_dim > 2:
            self._state[2:] = (
                0.95 * self._state[2:]
                + 0.01 * self._rng.normal(size=self.obs_dim - 2)
                + 0.001 * action[: self.obs_dim - 2] if self.obs_dim - 2 > 0 else self._state[2:]
            )
        self._elapsed += 1
        x_velocity = (float(self._state[0]) - prev_x) / 0.02
        forward_reward = 1.0 * x_velocity
        control_cost = self.control_cost_weight * float(np.sum(action ** 2))
        healthy_reward = self.healthy_reward if healthy else 0.0
        if self.sparse_threshold is not None:
            reward = float(self._state[0]) if float(self._state[0]) > self.sparse_threshold else 0.0
        else:
            reward = healthy_reward + forward_reward - control_cost
        now_healthy = self._healthy()
        terminated = bool(self.terminate_when_unhealthy and not now_healthy)
        truncated = bool(self._elapsed >= self.max_episode_steps)
        self.state = self._state.copy()
        info = {"x_position": float(self._state[0]), "x_velocity": x_velocity}
        return self._state.astype(np.float32).copy(), float(reward), terminated, truncated, info

    def render(self, *args, **kwargs):  # pragma: no cover - nothing to render
        return None

    def close(self):  # pragma: no cover
        return None


def _fallback_allowed() -> bool:
    return str(os.environ.get("RICE_ALLOW_MUJOCO_FALLBACK", "0")).lower() in {"1", "true", "yes", "on"}


def register_fallbacks(force: bool = False) -> bool:
    """Register the analytic stand-ins in the gym registry (opt-in)."""
    if not _fallback_allowed() and not force:
        return False
    if not GYM_AVAILABLE:  # pragma: no cover - fallback needs a registry to register into
        return False
    from ._common import gym as gym_module  # local import: optional dependency

    for canonical, spec in MUJOCO_DENSE_SPECS.items():
        for gym_id in spec.gym_ids:
            entry_point = _make_fallback_factory(spec)
            try:
                gym_module.register(id=gym_id, entry_point=entry_point, max_episode_steps=None)
            except Exception:
                # id already registered -> keep the existing implementation
                pass
    return True


def _make_fallback_factory(spec: MujocoDenseEnvSpec) -> Callable[[], FallbackLocomotionEnv]:
    def _factory(**kwargs) -> FallbackLocomotionEnv:
        kwargs.setdefault("obs_dim", spec.obs_dim)
        kwargs.setdefault("act_dim", spec.act_dim)
        kwargs.setdefault("max_episode_steps", spec.max_episode_steps)
        kwargs.setdefault("terminate_when_unhealthy", spec.terminate_when_unhealthy)
        return FallbackLocomotionEnv(**kwargs)

    _factory.__name__ = f"fallback_{spec.family}"
    return _factory


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------
def _raw_gym_env(spec: MujocoDenseEnvSpec, gym_id: Optional[str], **kwargs) -> Tuple[Any, str]:
    """Instantiate the underlying environment, preferring the paper's ``-v3`` id."""
    if not GYM_AVAILABLE:
        raise ImportError(
            "gym/gymnasium is required to build MuJoCo environments. Install `gym==0.21-0.26` "
            "with MuJoCo, or set RICE_ALLOW_MUJOCO_FALLBACK=1 for the CPU stand-in."
        )
    from ._common import gym as gym_module  # local import: optional dependency

    ids = (gym_id,) if gym_id else spec.gym_ids
    last_err: Optional[Exception] = None
    for candidate in ids:
        try:
            return gym_module.make(candidate, **kwargs), candidate
        except Exception as exc:  # MuJoCo / id missing -> try the next variant
            last_err = exc
    if _fallback_allowed():
        register_fallbacks(force=True)
        for candidate in ids:
            try:
                return gym_module.make(candidate, **kwargs), f"{candidate} (fallback)"
            except Exception as exc:  # pragma: no cover
                last_err = exc
    raise ImportError(
        f"Could not create any of {list(ids)} ({last_err!r}). If MuJoCo is not installed, set "
        f"RICE_ALLOW_MUJOCO_FALLBACK=1 to use the analytic stand-in (algorithm smoke tests only)."
    )


def make_env(
    name: str = "Hopper-v3",
    normalize: Optional[bool] = None,
    normalize_obs: Optional[bool] = None,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    exclude_current_positions: Optional[bool] = None,
    terminate_on_nonfinite: Optional[bool] = None,
    inject_healthy_reward: Optional[bool] = None,
    obs_clip: float = 10.0,
    update_obs_stats: bool = True,
    gym_id: Optional[str] = None,
    **env_kwargs,
) -> Any:
    """Build one dense-reward MuJoCo environment as described in §C.2.

    Parameters mirror the paper's per-task settings and are all overridable so the sparse
    wrappers (:mod:`rice.environments.mujoco_sparse`) and the hyper-parameter sweeps can reuse
    the same construction code.

    Returns a *single* (non-vectorised) environment, matching Algorithm 1/2's sequential,
    one-``s_0``-per-iteration roll-in.
    """
    canonical = resolve_name(name)
    spec = MUJOCO_DENSE_SPECS[canonical]

    if normalize is not None and normalize_obs is None:
        normalize_obs = normalize
    normalize_obs = spec.normalize_obs if normalize_obs is None else bool(normalize_obs)
    exclude_current_positions = (
        spec.exclude_current_positions
        if exclude_current_positions is None
        else bool(exclude_current_positions)
    )
    terminate_on_nonfinite = (
        spec.terminate_on_nonfinite if terminate_on_nonfinite is None else bool(terminate_on_nonfinite)
    )
    inject_healthy = (
        spec.inject_healthy_reward if inject_healthy_reward is None else bool(inject_healthy_reward)
    )

    make_kwargs: Dict[str, Any] = dict(env_kwargs)
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    # Only Hopper/HalfCheetah expose `exclude_current_positions_from_observation`.
    if spec.family in {"hopper", "halfcheetah"}:
        make_kwargs["exclude_current_positions_from_observation"] = exclude_current_positions
    if spec.family in {"hopper", "walker2d"}:
        make_kwargs.setdefault("terminate_when_unhealthy", spec.terminate_when_unhealthy)

    env, resolved_id = _raw_gym_env(spec, gym_id, **make_kwargs)

    # Re-inject or keep the "healthy reward" term of the §C.2 dense reward decomposition.
    if inject_healthy and "fallback" not in resolved_id:
        inner = getattr(env, "unwrapped", env)
        existing = float(getattr(inner, "healthy_reward", 0.0) or 0.0)
        if existing == 0.0:
            env = HealthyRewardWrapper(env, healthy_reward=1.0, enabled=True)

    if terminate_on_nonfinite:
        env = TerminateOnNonFiniteWrapper(env, enabled=True)

    if normalize_obs:
        env = ObservationNormalizeWrapper(
            env, normalize=True, clip=obs_clip, update=update_obs_stats
        )

    # Explicit episode length override (v3 already carries a TimeLimit with the right value).
    current_limit = env_max_episode_steps(env, default=0)
    target_limit = int(max_episode_steps) if max_episode_steps else spec.max_episode_steps
    if target_limit and current_limit != target_limit:
        env = EpisodeLengthWrapper(env, max_episode_steps=target_limit)

    if seed is not None:
        try:
            env.reset(seed=int(seed))
        except TypeError:
            try:
                env.seed(int(seed))
            except Exception:
                pass
        except Exception:
            pass

    try:
        env.rise_canonical_name = canonical  # type: ignore[attr-defined]
        env.rise_gym_id = resolved_id  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass
    return env


# -- per-task factories (registry contract of `rice.environments`) ---------------------
def make_hopper(**kwargs) -> Any:
    """``Hopper-v3`` (§C.2): dense healthy + forward + control reward, ends when unhealthy."""
    return make_env("Hopper-v3", **kwargs)


def make_walker2d(**kwargs) -> Any:
    """``Walker2d-v3`` (§C.2): 6-D action space, normalized observations."""
    return make_env("Walker2d-v3", **kwargs)


def make_reacher(**kwargs) -> Any:
    """``Reacher-v2`` (§C.2): 50-step episodes, "reward distance" - "reward control"."""
    kwargs.setdefault("max_episode_steps", 50)
    return make_env("Reacher-v2", **kwargs)


def make_halfcheetah(**kwargs) -> Any:
    """``HalfCheetah-v3`` (§C.2): forward reward minus control cost, normalized observations."""
    return make_env("HalfCheetah-v3", **kwargs)


def make_mujoco_dense(name: str = "Hopper-v3", **kwargs) -> Any:
    """Alias used by the environment registry (`rice.environments.make_env`)."""
    return make_env(name, **kwargs)


def get_spec(name: str) -> MujocoDenseEnvSpec:
    """Return the (immutable) spec of a task; useful for scripts and tests."""
    return MUJOCO_DENSE_SPECS[resolve_name(name)]
