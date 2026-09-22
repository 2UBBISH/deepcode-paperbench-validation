"""Environment wrappers used to inject *custom* (prior / evaluation) reward
functions into an environment at evaluation time.

The FRE evaluation harness never uses the environment's native reward signal:
for every evaluation task we build a reward function ``eta`` (a
:class:`fre.priors.goal_functions.RewardFunction` instance or any callable that
maps a batch of states to scalar rewards) and wrap the environment so that
``env.step`` returns ``eta(s)`` instead of the environment reward.

This module is deliberately dependency-light: ``gym`` (and ``dm_control`` for the
ExORL physics features) are imported lazily / defensively so that the rest of the
code base stays importable in minimal environments.

The wrappers follow the standard ``gym.Wrapper`` protocol when ``gym`` is
available and fall back to a duck-typed wrapper that proxies every attribute to
the wrapped environment otherwise.

Conventions
-----------
* ``eta`` is evaluated on the *reward state*, which by default is the observation
  returned by the environment.  For ExORL the FRE encoder consumes physics
  features appended to the observation (Appendix C.2), so a ``state_fn`` can be
  supplied to build the reward state (e.g. basic state + physics features), and
  :class:`PhysicsObservationWrapper` appends the same features to the
  observation fed to the policy/encoder.
* ``done`` is OR-ed with the task success signal when ``success_fn`` is given,
  which is how the singleton goal-reaching tasks (Appendix C) terminate.
* Environment rewards are available under ``info["env_reward"]`` so that native
  (D4RL) score normalization can still be computed if required.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - gym is optional at import time
    import gym  # type: ignore

    _GymWrapper = gym.Wrapper
    _HAS_GYM = True
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    _GymWrapper = object  # type: ignore
    _HAS_GYM = False


__all__ = [
    "DEFAULT_GOAL_THRESHOLD",
    "DEFAULT_EXORL_GOAL_THRESHOLD",
    "DEFAULT_MAX_EPISODE_STEPS",
    "RewardWrapper",
    "RewardFunctionWrapper",
    "SuccessWrapper",
    "TimeLimitWrapper",
    "PhysicsObservationWrapper",
    "evaluate_reward_function",
    "as_reward_fn",
    "wrap_reward",
    "wrap_env",
    "inject_reward_fn",
    "make_success_done_fn",
    "make_goal_success_fn",
    "constant_reward_fn",
    "normalize_reward_fn",
    "compute_exorl_physics_features",
    "augment_observation_with_physics",
    "exorl_reward_state_fn",
    "call_env_reset",
    "call_env_step",
]

# ---------------------------------------------------------------------------
# constants (Appendix C)
# ---------------------------------------------------------------------------

#: Euclidean goal-reaching threshold used for the singleton goal tasks.
DEFAULT_GOAL_THRESHOLD = 0.1
#: ExORL goal states use a Euclidean distance threshold of 0.1 (Appendix C.2).
DEFAULT_EXORL_GOAL_THRESHOLD = 0.1

#: Maximum episode length per domain (Appendix C): AntMaze 2000, ExORL 1000.
DEFAULT_MAX_EPISODE_STEPS: Dict[str, int] = {
    "antmaze": 2000,
    "exorl": 1000,
    "kitchen": 280,
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def call_env_reset(env: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
    """Call ``env.reset`` and normalize the (obs, info) / obs return signature."""
    try:
        out = env.reset(**kwargs)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], dict(out[1]) if out[1] is not None else {}
    return out, {}


def call_env_step(env: Any, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Call ``env.step`` and normalize 4-tuple / 5-tuple return signatures."""
    out = env.step(action)
    if out is None:
        raise RuntimeError("environment step returned None")
    if len(out) == 5:  # gym >= 0.26: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        info = dict(info) if info is not None else {}
        info.setdefault("terminated", bool(terminated))
        info.setdefault("truncated", bool(truncated))
        return obs, float(reward), done, info
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), dict(info) if info is not None else {}
    raise RuntimeError(f"unexpected step return of length {len(out)}")


def _to_numpy(value: Any) -> Any:
    """Convert torch tensors to numpy (numpy inputs pass through untouched)."""
    if value is None:
        return None
    if hasattr(value, "detach"):  # torch.Tensor
        return value.detach().cpu().numpy()
    return np.asarray(value) if not isinstance(value, np.ndarray) else value


def as_reward_fn(reward_fn: Union[Callable, Any]) -> Callable[[Any], Any]:
    """Normalize a reward function into a callable ``states -> rewards``.

    Accepts

    * a plain callable,
    * any object exposing ``.reward(states)`` (the ``fre.priors`` interface,
      e.g. :class:`~fre.priors.goal_functions.GoalRewardFunction`),
    * or any object exposing ``.__call__(states)`` (handled implicitly).
    """
    if reward_fn is None:
        raise ValueError("reward_fn must not be None")
    if callable(reward_fn):
        return reward_fn
    if hasattr(reward_fn, "reward"):
        return reward_fn.reward
    raise TypeError(f"cannot interpret {type(reward_fn)} as a reward function")


def evaluate_reward_function(reward_fn: Union[Callable, Any], states: Any) -> float:
    """Evaluate ``eta(states)`` and reduce it to a python float.

    Multi-dimensional outputs (e.g. a batched :class:`MixtureRewardFunction`) are
    mean-reduced so that a scalar per-step reward is always returned.
    """
    fn = as_reward_fn(reward_fn)
    with np.errstate(all="ignore"):
        try:
            import torch  # local import: torch is optional

            if isinstance(states, torch.Tensor):
                prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
                try:
                    value = fn(states)
                finally:
                    torch.set_grad_enabled(prev)
                value = value.detach().cpu().numpy()
            else:
                value = fn(states)
        except ImportError:  # pragma: no cover
            value = fn(states)
    value = _to_numpy(value)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.mean())


def normalize_reward_fn(
    reward_fn: Union[Callable, Any],
    reward_min: Union[float, Any] = -1.0,
    reward_max: Union[float, Any] = 1.0,
    normalize: bool = True,
) -> Callable[[Any], Any]:
    """Wrap a reward function with an affine map onto ``[reward_min, reward_max]``.

    The FRE paper normalizes rewards to ``[0, 1]`` before discretization; the
    evaluation-time rollouts use the raw ``eta`` unless this wrapper is applied.
    """

    base = as_reward_fn(reward_fn)

    def fn(states: Any) -> Any:
        value = np.asarray(base(states), dtype=np.float64)
        if not normalize:
            return value
        lo = np.asarray(reward_min, dtype=np.float64)
        hi = np.asarray(reward_max, dtype=np.float64)
        return np.where(value == 0.0, lo, hi)

    return fn


def constant_reward_fn(value: float = 0.0) -> Callable[[Any], Any]:
    """A reward function ignoring the state (useful for debugging/random policy)."""

    def fn(states: Any) -> Any:
        shape = getattr(states, "shape", None)
        if shape is None:
            return value
        out = np.zeros(shape[:-1], dtype=np.float32)
        out[...] = value
        return out

    return fn


def make_goal_success_fn(
    goal: Any,
    threshold: float = DEFAULT_GOAL_THRESHOLD,
    state_fn: Optional[Callable[[Any], Any]] = None,
    distance_dims: Optional[Union[Sequence[int], slice]] = None,
    state_std: Optional[Any] = None,
    normalize_dims: bool = False,
) -> Callable[[Any], bool]:
    """Build a boolean goal-success function for a single goal state."""
    goal_arr = np.asarray(_to_numpy(goal), dtype=np.float64).reshape(-1)
    if distance_dims is not None:
        goal_arr = goal_arr[distance_dims]
    std_arr = None
    if state_std is not None:
        std_arr = np.asarray(_to_numpy(state_std), dtype=np.float64).reshape(-1)
        if distance_dims is not None:
            std_arr = std_arr[distance_dims]

    def success_fn(state: Any) -> bool:
        state_arr = np.asarray(_to_numpy(state_fn(state) if state_fn else state), dtype=np.float64)
        state_arr = state_arr.reshape(-1)
        if distance_dims is not None:
            state_arr = state_arr[distance_dims]
        diff = state_arr - goal_arr
        if std_arr is not None and normalize_dims:
            diff = diff / (std_arr + 1e-6)
        return bool(np.sqrt(np.sum(diff * diff)) < threshold)

    return success_fn


def make_success_done_fn(
    success_fn: Callable[[Any], bool],
    done_on_success: bool = True,
) -> Callable[[Any], bool]:
    """Turn a success predicate into a done predicate (possibly a no-op)."""

    if not done_on_success:
        return lambda state: False

    def fn(state: Any) -> bool:
        try:
            return bool(success_fn(state))
        except Exception:
            return False

    return fn


# ---------------------------------------------------------------------------
# wrappers
# ---------------------------------------------------------------------------


class RewardWrapper(_GymWrapper):  # type: ignore[misc]
    """Replace the environment reward with a custom reward function ``eta``.

    Parameters
    ----------
    env:
        Wrapped environment.
    reward_fn:
        Any callable / object with ``.reward(states)`` mapping a batch of states
        to scalar rewards (see :func:`as_reward_fn`).
    state_fn:
        Optional function mapping the observation to the *reward state*.  Used by
        ExORL where ``eta`` is defined on physics-augmented states.
    success_fn:
        Optional predicate on the reward state marking task success.  When it
        fires, ``done`` becomes ``True`` and ``info["success"] = True``.
    done_fn:
        Optional custom done predicate on the reward state.
    info_reward_key:
        Key under which the custom reward is stored in ``info``.
    max_episode_steps:
        Optional episode length limit enforced by the wrapper (AntMaze 2000,
        ExORL 1000, Kitchen 280 by default when built through the task modules).
    reward_clip:
        Optional ``(low, high)`` tuple clipping the per-step reward.
    accumulate:
        If ``True``, ``info[info_reward_key]`` stores the episode return instead
        of the instantaneous reward (the returned reward is unaffected).
    count_success_as_done:
        Whether successful goal-reaching terminates the episode.
    name:
        Optional label used in ``__repr__``.
    """

    def __init__(
        self,
        env: Any,
        reward_fn: Union[Callable, Any, None] = None,
        state_fn: Optional[Callable[[Any], Any]] = None,
        success_fn: Optional[Callable[[Any], bool]] = None,
        done_fn: Optional[Callable[[Any], bool]] = None,
        info_reward_key: str = "task_reward",
        max_episode_steps: Optional[int] = None,
        reward_clip: Optional[Tuple[float, float]] = None,
        accumulate: bool = False,
        count_success_as_done: bool = True,
        store_env_reward: bool = True,
        name: Optional[str] = None,
    ) -> None:
        try:
            super().__init__(env)
        except TypeError:  # fallback base class (no gym available)
            self.env = env
        self.reward_fn = reward_fn
        self.state_fn = state_fn
        self.success_fn = success_fn
        self.done_fn = done_fn
        self.info_reward_key = info_reward_key
        self.max_episode_steps = max_episode_steps
        self.reward_clip = reward_clip
        self.accumulate = bool(accumulate)
        self.count_success_as_done = bool(count_success_as_done)
        self.store_env_reward = bool(store_env_reward)
        self.name = name

        self._steps = 0
        self._episode_reward = 0.0
        self.episode_reward_buffer = 0.0

    # -- properties ---------------------------------------------------------

    @property
    def unwrapped_env(self) -> Any:
        """The innermost environment object."""
        return getattr(self.env, "unwrapped", self.env)

    def set_reward_function(self, reward_fn: Union[Callable, Any]) -> None:
        """Swap the reward function (one env can be reused across eval tasks)."""
        self.reward_fn = reward_fn

    def set_success_function(self, success_fn: Optional[Callable[[Any], bool]]) -> None:
        self.success_fn = success_fn

    # -- internals ----------------------------------------------------------

    def _reward_state(self, obs: Any) -> Any:
        if self.state_fn is None:
            return obs
        return self.state_fn(obs)

    def compute_reward(self, obs: Any) -> Tuple[float, bool, Any]:
        """Return ``(reward, success, reward_state)`` for an observation."""
        state = self._reward_state(obs)
        if self.reward_fn is None:
            reward = 0.0
        else:
            reward = evaluate_reward_function(self.reward_fn, state)
        if self.reward_clip is not None:
            reward = float(np.clip(reward, self.reward_clip[0], self.reward_clip[1]))
        success = False
        if self.success_fn is not None:
            try:
                success = bool(self.success_fn(state))
            except Exception:
                success = False
        return reward, success, state

    def is_done(self, obs: Any, env_done: bool, success: bool) -> bool:
        done = bool(env_done)
        if self.count_success_as_done and success:
            done = True
        if self.done_fn is not None:
            try:
                done = done or bool(self.done_fn(self._reward_state(obs)))
            except Exception:
                pass
        return done

    # -- gym API ------------------------------------------------------------

    def reset(self, **kwargs: Any):  # noqa: D102
        obs, info = call_env_reset(self.env, **kwargs)
        self._steps = 0
        self._episode_reward = 0.0
        self.episode_reward_buffer = 0.0
        if isinstance(info, dict):
            info.setdefault("episode_steps", self._steps)
        return obs, info

    def step(self, action: Any):  # noqa: D102
        obs, env_reward, env_done, info = call_env_step(self.env, action)
        reward, success, _ = self.compute_reward(obs)
        self._steps += 1
        self._episode_reward += reward
        self.episode_reward_buffer = self._episode_reward

        done = self.is_done(obs, env_done, success)
        if self.max_episode_steps is not None and self._steps >= int(self.max_episode_steps):
            done = True

        if not isinstance(info, dict):
            info = {}
        if self.info_reward_key:
            info[self.info_reward_key] = (
                self._episode_reward if self.accumulate else float(reward)
            )
        if self.store_env_reward:
            info["env_reward"] = float(env_reward)
        info["success"] = bool(success) or bool(info.get("success", False))
        info["episode_steps"] = self._steps
        info["episode_return"] = float(self._episode_reward)
        return obs, float(reward), bool(done), info

    # -- misc ---------------------------------------------------------------

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        parts = [f"max_steps={self.max_episode_steps}"]
        if self.name:
            parts.append(f"name={self.name!r}")
        return ", ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.env!r}, {self.extra_repr()})"


#: Backwards-compatible alias used in the plan / task modules.
RewardFunctionWrapper = RewardWrapper


class SuccessWrapper(RewardWrapper):
    """Convenience wrapper that only marks success/done with a constant reward."""

    def __init__(
        self,
        env: Any,
        success_fn: Callable[[Any], bool],
        state_fn: Optional[Callable[[Any], Any]] = None,
        reward_fn: Optional[Union[Callable, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env, reward_fn=reward_fn, state_fn=state_fn, success_fn=success_fn, **kwargs)


class TimeLimitWrapper(_GymWrapper):  # type: ignore[misc]
    """Minimal episode-length wrapper (used when the base env has no limit)."""

    def __init__(self, env: Any, max_episode_steps: int) -> None:
        try:
            super().__init__(env)
        except TypeError:  # fallback base class
            self.env = env
        self.max_episode_steps = int(max_episode_steps)
        self._steps = 0

    def reset(self, **kwargs: Any):  # noqa: D102
        self._steps = 0
        obs, info = call_env_reset(self.env, **kwargs)
        return obs, info

    def step(self, action: Any):  # noqa: D102
        obs, reward, done, info = call_env_step(self.env, action)
        self._steps += 1
        if self._steps >= self.max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        info["episode_steps"] = self._steps
        return obs, reward, done, info


# ---------------------------------------------------------------------------
# ExORL physics-augmented observations
# ---------------------------------------------------------------------------


def _flatten_physics_value(value: Any) -> np.ndarray:
    """Turn a dm_control physics result into a flat float vector."""
    arr = np.asarray(_to_numpy(value), dtype=np.float64)
    return arr.reshape(-1)


def compute_exorl_physics_features(env: Any) -> np.ndarray:
    """Compute the ExORL physics feature vector for the *current* env state.

    Walker: ``horizontal_velocity()`` (2-d), ``torso_upright()``,
    ``torso_height()``.  Cheetah: ``speed()``.  (Appendix C.2.)

    The wrapped environment (or any of its parents) must expose ``.physics``.
    Returns an empty array when no physics object is found.
    """
    physics = None
    for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
        if candidate is not None and hasattr(candidate, "physics"):
            physics = candidate.physics
            break
    if physics is None:
        return np.zeros(0, dtype=np.float64)

    features = []
    if hasattr(physics, "horizontal_velocity"):
        try:
            features.append(_flatten_physics_value(physics.horizontal_velocity()))
        except Exception:
            features.append(np.zeros(2, dtype=np.float64))
    if hasattr(physics, "torso_upright"):
        try:
            features.append(_flatten_physics_value(physics.torso_upright()))
        except Exception:
            features.append(np.zeros(1, dtype=np.float64))
    if hasattr(physics, "torso_height"):
        try:
            features.append(_flatten_physics_value(physics.torso_height()))
        except Exception:
            features.append(np.zeros(1, dtype=np.float64))
    if hasattr(physics, "speed"):
        try:
            features.append(_flatten_physics_value(physics.speed()))
        except Exception:
            features.append(np.zeros(1, dtype=np.float64))
    if hasattr(physics, "velocities"):
        try:
            features.append(_flatten_physics_value(physics.velocities()))
        except Exception:
            pass

    if not features:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(features, axis=0).astype(np.float64)


def augment_observation_with_physics(obs: Any, physics_features: Any) -> np.ndarray:
    """Concatenate raw observation with physics features (ExORL encoder input)."""
    obs_arr = np.asarray(_to_numpy(obs), dtype=np.float64).reshape(-1)
    phys_arr = np.asarray(_to_numpy(physics_features), dtype=np.float64).reshape(-1)
    if phys_arr.size == 0:
        return obs_arr.astype(np.float32)
    return np.concatenate([obs_arr, phys_arr], axis=0).astype(np.float32)


class PhysicsObservationWrapper(_GymWrapper):  # type: ignore[misc]
    """Append ExORL physics features to the observation.

    With ``augment=True`` the observation returned by ``step``/``reset`` is
    ``[obs, physics_features]`` — matching the encoder-side augmentation of
    Appendix C.2 and the evaluation-time features.  ``state_fn``-style access to
    the original (non-augmented) state is exposed as ``info["obs"]`` and through
    :meth:`base_observation` so that goal distances can be computed on the basic
    state only (augmented dims are excluded from goal distance).
    """

    def __init__(
        self,
        env: Any,
        augment: bool = True,
        physics_fn: Optional[Callable[[Any], Any]] = None,
        wrap_observation: bool = True,
    ) -> None:
        try:
            super().__init__(env)
        except TypeError:  # fallback base class
            self.env = env
        self.augment = bool(augment)
        self.physics_fn = physics_fn or compute_exorl_physics_features
        self.wrap_observation = bool(wrap_observation)
        self._last_physics: np.ndarray = np.zeros(0, dtype=np.float64)

    # -- helpers ------------------------------------------------------------

    def physics_features(self) -> np.ndarray:
        return np.asarray(self.physics_fn(self.env), dtype=np.float64).reshape(-1)

    def base_observation(self) -> Optional[np.ndarray]:
        """The last non-augmented observation, if any."""
        return getattr(self, "_last_base_obs", None)

    def _augment(self, obs: Any) -> np.ndarray:
        base = np.asarray(_to_numpy(obs), dtype=np.float64).reshape(-1)
        self._last_base_obs = base
        try:
            self._last_physics = self.physics_features()
        except Exception:
            self._last_physics = np.zeros(0, dtype=np.float64)
        if not self.augment:
            return base.astype(np.float32)
        return augment_observation_with_physics(base, self._last_physics)

    # -- gym API ------------------------------------------------------------

    def reset(self, **kwargs: Any):  # noqa: D102
        obs, info = call_env_reset(self.env, **kwargs)
        aug = self._augment(obs)
        info["obs"] = np.asarray(_to_numpy(obs), dtype=np.float64).reshape(-1)
        info["physics"] = np.array(self._last_physics, copy=True)
        return aug, info

    def step(self, action: Any):  # noqa: D102
        obs, reward, done, info = call_env_step(self.env, action)
        aug = self._augment(obs)
        info["obs"] = np.asarray(_to_numpy(obs), dtype=np.float64).reshape(-1)
        info["physics"] = np.array(self._last_physics, copy=True)
        return aug, reward, done, info


def exorl_reward_state_fn(env: Any, base_state_fn: Optional[Callable[[Any], Any]] = None):
    """Build a reward-state function for ExORL that appends physics features.

    ``eta`` for ExORL is defined on the same physics-augmented state that the
    encoder consumes (Appendix C.2), which is also used at evaluation.
    """

    def state_fn(obs: Any) -> np.ndarray:
        base = base_state_fn(obs) if base_state_fn is not None else obs
        physics = compute_exorl_physics_features(env)
        return augment_observation_with_physics(base, physics)

    return state_fn


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------


def wrap_reward(
    env: Any,
    reward_fn: Union[Callable, Any],
    state_fn: Optional[Callable[[Any], Any]] = None,
    success_fn: Optional[Callable[[Any], bool]] = None,
    done_fn: Optional[Callable[[Any], bool]] = None,
    max_episode_steps: Optional[int] = None,
    domain: Optional[str] = None,
    **kwargs: Any,
) -> RewardWrapper:
    """Wrap ``env`` so that ``eta`` replaces the native reward.

    ``domain`` selects the default episode length from
    :data:`DEFAULT_MAX_EPISODE_STEPS` when ``max_episode_steps`` is not given.
    """
    if max_episode_steps is None and domain is not None:
        max_episode_steps = DEFAULT_MAX_EPISODE_STEPS.get(domain)
    return RewardWrapper(
        env,
        reward_fn=reward_fn,
        state_fn=state_fn,
        success_fn=success_fn,
        done_fn=done_fn,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def wrap_env(
    env: Any,
    reward_fn: Optional[Union[Callable, Any]] = None,
    state_fn: Optional[Callable[[Any], Any]] = None,
    success_fn: Optional[Callable[[Any], bool]] = None,
    done_fn: Optional[Callable[[Any], bool]] = None,
    max_episode_steps: Optional[int] = None,
    domain: Optional[str] = None,
    physics_augment: bool = False,
    **kwargs: Any,
) -> Any:
    """High-level evaluation wrapper factory.

    If ``physics_augment`` is true the environment observation is first extended
    with ExORL physics features (Appendix C.2) and the reward state defaults to
    the augmented observation, matching encoder-time preprocessing.
    """
    if physics_augment:
        env = PhysicsObservationWrapper(env, augment=True)
        if state_fn is None:
            state_fn = lambda obs: obs  # noqa: E731 - augmented obs already carries physics
    return wrap_reward(
        env,
        reward_fn=reward_fn,
        state_fn=state_fn,
        success_fn=success_fn,
        done_fn=done_fn,
        max_episode_steps=max_episode_steps,
        domain=domain,
        **kwargs,
    )


def inject_reward_fn(env: Any, reward_fn: Union[Callable, Any], **kwargs: Any) -> Any:
    """Attach ``reward_fn`` to ``env`` in place (monkey-patching ``step``).

    Useful when a third-party rollout utility insists on the *original* env
    object (e.g. dm_control's ``evaluation`` helpers, or the
    ``controllable_agent`` baselines).  Returns the (same) environment.

    The native ``step`` is preserved as ``env._native_step`` so the wrapper can
    be removed with :func:`remove_reward_fn`.
    """
    if getattr(env, "_fre_reward_injected", False):
        env._fre_reward_fn = reward_fn
        return env

    env._fre_reward_fn = reward_fn
    env._fre_reward_injected = True
    env._fre_wrapper = RewardWrapper(env, reward_fn=reward_fn, **kwargs)
    native_step = env.step

    def step(action: Any):  # pragma: no cover - exercised through envs
        obs, _env_reward, _done, info = call_env_step(env, action)
        reward, success, _ = env._fre_wrapper.compute_reward(obs)
        done = env._fre_wrapper.is_done(obs, _done, success)
        info["task_reward"] = float(reward)
        info["success"] = bool(success)
        info["env_reward"] = float(_env_reward)
        return obs, float(reward), bool(done), info

    env._native_step = native_step
    env.step = step  # type: ignore[assignment]
    return env


def remove_reward_fn(env: Any) -> Any:
    """Undo :func:`inject_reward_fn`."""
    if getattr(env, "_fre_reward_injected", False) and hasattr(env, "_native_step"):
        env.step = env._native_step
        env._fre_reward_injected = False
    return env
