"""Mixed initial-state distribution wrapper for RICE refinement.

At the start of each training episode, with probability ``p`` the environment is
initialized from a sampled critical state (uniformly drawn from a
``CriticalStateBuffer``); otherwise a default ``env.reset()`` is performed.

For environments that do not expose a direct state-setting API, the wrapper
supports fallback strategies such as replaying the action sequence that led to
the critical state.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from .critical_state_buffer import CriticalState, CriticalStateBuffer


RestoreFn = Callable[[gym.Env, CriticalState], Any]
FallbackFn = Callable[[gym.Env, CriticalState, Optional[Dict[str, Any]]], Any]


def _has_method(obj: Any, name: str) -> bool:
    return hasattr(obj, name) and callable(getattr(obj, name))


def default_restore_state(env: gym.Env, critical_state: CriticalState) -> Any:
    """Best-effort direct state restoration.

    Tries, in order:
      1. ``env.set_state(state)`` / ``env.unwrapped.set_state(state)``
      2. ``env.restore_state(state)`` / ``env.unwrapped.restore_state(state)``
      3. MuJoCo-specific ``sim.set_state`` from ``qpos``/``qvel`` in ``env_state``
      4. Action-history replay (roll-out stored actions from a fresh reset)
      5. Plain ``env.reset()`` as a last resort.

    Returns the observation produced after restoration.
    """
    state = critical_state.env_state
    action_history = critical_state.action_history or []

    # 1. Generic set_state API.
    for target in (env, getattr(env, "unwrapped", env)):
        if _has_method(target, "set_state") and state is not None:
            try:
                obs = target.set_state(state)
                if obs is not None:
                    return obs
            except Exception:
                pass
        if _has_method(target, "restore_state") and state is not None:
            try:
                obs = target.restore_state(state)
                if obs is not None:
                    return obs
            except Exception:
                pass

    # 2. MuJoCo-specific restoration from qpos/qvel.
    if state is not None:
        sim = getattr(env, "sim", None) or getattr(getattr(env, "unwrapped", env), "sim", None)
        if sim is not None and hasattr(sim, "set_state"):
            try:
                qpos = state.get("qpos") if isinstance(state, dict) else None
                qvel = state.get("qvel") if isinstance(state, dict) else None
                if qpos is not None and qvel is not None:
                    sim.set_state(np.concatenate([qpos, qvel]))
                    sim.forward()
                    return env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else env.reset()
            except Exception:
                pass

    # 3. Replay action history from a fresh reset.
    if action_history:
        try:
            obs = env.reset()
            for action in action_history:
                step_result = env.step(action)
                obs = step_result[0] if isinstance(step_result, tuple) else step_result.observation
            return obs
        except Exception:
            pass

    # 4. Last resort: default reset.
    return env.reset()


def default_fallback_reset(
    env: gym.Env,
    critical_state: CriticalState,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Fallback when critical-state restoration fails: plain default reset."""
    return env.reset(**(reset_kwargs or {}))


class MixedResetEnv(gym.Wrapper):
    """Wrapper that resets from a critical-state buffer with probability ``p``.

    Parameters
    ----------
    env : gym.Env
        The base task environment.
    critical_buffer : CriticalStateBuffer
        Buffer containing critical states sampled during target-policy rollouts.
    p : float
        Probability of resetting from a critical state (vs. default reset).
    restore_fn : Optional[RestoreFn]
        Function ``(env, critical_state) -> obs`` used to restore the simulator
        to a critical state. If ``None``, ``default_restore_state`` is used.
    fallback_fn : Optional[FallbackFn]
        Function ``(env, critical_state, reset_kwargs) -> obs`` called when
        ``restore_fn`` raises an exception. If ``None``,
        ``default_fallback_reset`` is used.
    """

    def __init__(
        self,
        env: gym.Env,
        critical_buffer: CriticalStateBuffer,
        p: float = 0.5,
        restore_fn: Optional[RestoreFn] = None,
        fallback_fn: Optional[FallbackFn] = None,
    ):
        super().__init__(env)
        self.critical_buffer = critical_buffer
        self.p = float(p)
        self.restore_fn = restore_fn or default_restore_state
        self.fallback_fn = fallback_fn or default_fallback_reset
        self._last_reset_source: Optional[str] = None
        self._last_critical_state: Optional[CriticalState] = None

    @property
    def last_reset_source(self) -> Optional[str]:
        """"default" or "critical"; useful for logging."""
        return self._last_reset_source

    @property
    def last_critical_state(self) -> Optional[CriticalState]:
        return self._last_critical_state

    def reset(self, **kwargs) -> Any:
        """Reset the environment from the mixed initial-state distribution."""
        if np.random.rand() < self.p and len(self.critical_buffer) > 0:
            critical_state = self.critical_buffer.sample()
            self._last_critical_state = critical_state
            try:
                obs = self.restore_fn(self.env, critical_state)
                self._last_reset_source = "critical"
                return obs
            except Exception:
                obs = self.fallback_fn(self.env, critical_state, kwargs)
                self._last_reset_source = "default"
                return obs
        else:
            self._last_critical_state = None
            self._last_reset_source = "default"
            return self.env.reset(**kwargs)

    def set_critical_buffer(self, critical_buffer: CriticalStateBuffer) -> None:
        """Replace the critical-state buffer (e.g. after mask re-training)."""
        self.critical_buffer = critical_buffer

    def set_p(self, p: float) -> None:
        """Update the critical-state reset probability."""
        self.p = float(np.clip(p, 0.0, 1.0))


def make_mixed_reset_env(
    env: gym.Env,
    critical_buffer: Union[CriticalStateBuffer, str],
    p: float = 0.5,
    restore_fn: Optional[RestoreFn] = None,
    fallback_fn: Optional[FallbackFn] = None,
) -> MixedResetEnv:
    """Convenience factory for ``MixedResetEnv``.

    ``critical_buffer`` may be a ``CriticalStateBuffer`` instance or a path to a
    saved buffer (``.npz`` file).
    """
    if isinstance(critical_buffer, str):
        critical_buffer = CriticalStateBuffer.load(critical_buffer)
    return MixedResetEnv(
        env,
        critical_buffer,
        p=p,
        restore_fn=restore_fn,
        fallback_fn=fallback_fn,
    )
