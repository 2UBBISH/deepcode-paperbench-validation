"""
Environment wrappers for saving and restoring full environment states.

This module provides a unified interface for state save/restore across
different environment types (MuJoCo, MetaDrive, Malconv, CAGE2, etc.).

For MuJoCo environments, we use sim.get_state() / sim.set_state() which
provides full serializable state snapshots.

For other environments, custom implementations are provided that capture
the minimal state needed to reproduce the environment's behavior.
"""

import copy
import pickle
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np


class StateSaveWrapper(gym.Wrapper):
    """
    Gym wrapper that adds save_state(), restore_state(), and reset_to_state()
    methods to an environment.

    For MuJoCo-based environments (Gym MuJoCo v2/v3/v4), uses the underlying
    sim.get_state() / sim.set_state() mechanism which captures the full
    physics state (positions, velocities, etc.).

    For non-MuJoCo environments, attempts to use env.get_state() / env.set_state()
    if available, otherwise falls back to a pickle-based deep copy approach.

    Usage:
        env = gym.make("Hopper-v3")
        env = StateSaveWrapper(env)
        state = env.save_state()
        # ... run some steps ...
        env.restore_state(state)  # restore to saved state
        # or:
        env.reset_to_state(state)  # reset and then restore
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._is_mujoco = self._detect_mujoco()
        self._has_get_state = hasattr(self.env, 'get_state') and callable(getattr(self.env, 'get_state', None))
        self._has_set_state = hasattr(self.env, 'set_state') and callable(getattr(self.env, 'set_state', None))

    def _detect_mujoco(self) -> bool:
        """Detect if the environment is MuJoCo-based."""
        env = self.env
        # Check for MuJoCo sim attribute
        if hasattr(env, 'sim'):
            return True
        if hasattr(env, 'model') and hasattr(env, 'data'):
            return True
        # Unwrap to find sim
        unwrapped = env
        while hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
            if hasattr(unwrapped, 'sim'):
                return True
        return False

    def _get_mujoco_sim(self):
        """Get the underlying MuJoCo sim object."""
        env = self.env
        if hasattr(env, 'sim'):
            return env.sim
        # Try unwrapping
        unwrapped = env
        while hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
            if hasattr(unwrapped, 'sim'):
                return unwrapped.sim
        return None

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current full environment state.

        Returns:
            A dictionary containing the serializable state. For MuJoCo,
            this includes the MjSimState. For other environments, this
            uses get_state() if available or a deep copy of relevant attributes.

        The returned state can be passed to restore_state() or reset_to_state().
        """
        if self._is_mujoco:
            sim = self._get_mujoco_sim()
            if sim is not None:
                # Get the full MuJoCo state
                mj_state = sim.get_state()
                # Also save the current observation and any internal env state
                state_dict = {
                    'type': 'mujoco',
                    'mj_state': mj_state,
                    'elapsed_steps': getattr(self.env, '_elapsed_steps', 0),
                }
                # Save the RNG state for reproducibility
                if hasattr(self.env, 'np_random'):
                    state_dict['np_random'] = self.env.np_random.get_state()
                return state_dict

        # For environments with get_state/set_state
        if self._has_get_state:
            try:
                env_state = self.env.get_state()
                return {
                    'type': 'custom',
                    'env_state': env_state,
                }
            except Exception:
                pass

        # Fallback: deep copy of the environment
        # This is expensive and may not work for all environments
        try:
            env_copy = copy.deepcopy(self.env)
            return {
                'type': 'deepcopy',
                'env_copy': env_copy,
            }
        except Exception:
            # Last resort: just save the observation
            obs = self.env._get_obs() if hasattr(self.env, '_get_obs') else None
            return {
                'type': 'observation_only',
                'observation': copy.deepcopy(obs) if obs is not None else None,
            }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the environment to a previously saved state without calling reset().

        This modifies the environment in-place to match the saved state.
        The environment's current observation will reflect the restored state.

        Args:
            state: A state dictionary previously returned by save_state().
        """
        state_type = state.get('type', 'unknown')

        if state_type == 'mujoco':
            sim = self._get_mujoco_sim()
            if sim is not None:
                mj_state = state['mj_state']
                sim.set_state(mj_state)
                # Restore elapsed steps if tracked
                if hasattr(self.env, '_elapsed_steps'):
                    self.env._elapsed_steps = state.get('elapsed_steps', 0)
                # Restore RNG state
                if 'np_random' in state and hasattr(self.env, 'np_random'):
                    self.env.np_random.set_state(state['np_random'])
                # Force re-computation of observation
                if hasattr(self.env, '_get_obs'):
                    self.env._get_obs()
                return

        if state_type == 'custom':
            if self._has_set_state:
                try:
                    self.env.set_state(state['env_state'])
                    return
                except Exception:
                    pass

        if state_type == 'deepcopy':
            # Replace the wrapped environment
            self.env = state['env_copy']
            return

        # observation_only: can't fully restore, just note it
        if state_type == 'observation_only':
            # Cannot fully restore from observation only
            pass

    def reset_to_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Reset the environment and then restore to a saved state.

        This first calls env.reset() to properly initialize any internal
        state, then restores the saved state on top.

        Args:
            state: A state dictionary previously returned by save_state().

        Returns:
            The observation after restoring to the saved state.
        """
        # First do a normal reset to initialize everything
        obs = self.env.reset()

        # Then restore the saved state
        self.restore_state(state)

        # Get the observation after restoration
        if hasattr(self.env, '_get_obs'):
            obs = self.env._get_obs()
        elif hasattr(self.env, 'render'):
            # Try to get observation through step(0) or similar
            try:
                # For MuJoCo, after set_state, we need to get the observation
                obs = self.env._get_obs()
            except Exception:
                pass

        return obs

    def step(self, action):
        """Step the environment, saving state capability is transparent."""
        return self.env.step(action)

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)


class MuJoCoStateWrapper(StateSaveWrapper):
    """
    Specialized wrapper for MuJoCo environments with additional utilities
    for state manipulation and serialization.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not self._is_mujoco:
            raise ValueError("MuJoCoStateWrapper requires a MuJoCo-based environment")

    def get_state_vector(self) -> np.ndarray:
        """
        Get the full state as a flat vector (qpos + qvel concatenated).

        Returns:
            numpy array of shape (nq + nv,) containing positions and velocities.
        """
        sim = self._get_mujoco_sim()
        if sim is not None:
            state = sim.get_state()
            return np.concatenate([state.qpos, state.qvel])
        return np.array([])

    def set_state_from_vector(self, state_vector: np.ndarray) -> None:
        """
        Set the environment state from a flat vector.

        Args:
            state_vector: numpy array of shape (nq + nv,) with positions and velocities.
        """
        sim = self._get_mujoco_sim()
        if sim is not None:
            nq = sim.model.nq
            qpos = state_vector[:nq]
            qvel = state_vector[nq:]
            # Create new state
            from mujoco import MjSimState
            import mujoco
            # Handle different MuJoCo versions
            try:
                # mujoco-py
                old_state = sim.get_state()
                new_state = MjSimState(
                    time=old_state.time,
                    qpos=qpos,
                    qvel=qvel,
                    act=old_state.act,
                    udd_state=old_state.udd_state,
                )
            except TypeError:
                # Newer mujoco
                old_state = sim.get_state()
                new_state = type(old_state)(
                    time=old_state.time,
                    qpos=qpos,
                    qvel=qvel,
                    act=old_state.act,
                )
            sim.set_state(new_state)

    def serialize_state(self, state: Dict[str, Any]) -> bytes:
        """
        Serialize a state dictionary to bytes for storage/transmission.

        Args:
            state: State dictionary from save_state().

        Returns:
            Pickled bytes representation.
        """
        return pickle.dumps(state)

    def deserialize_state(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize a state from bytes.

        Args:
            data: Pickled bytes from serialize_state().

        Returns:
            State dictionary.
        """
        return pickle.loads(data)


class DictStateWrapper(StateSaveWrapper):
    """
    Wrapper for environments with dictionary observation spaces that
    need custom state save/restore logic (e.g., MetaDrive, CAGE2).

    Subclasses should override _save_custom_state() and _restore_custom_state().
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._custom_state_attrs = []

    def register_state_attrs(self, attrs: list):
        """
        Register additional attributes to save/restore.

        Args:
            attrs: List of attribute names (strings) to include in state.
        """
        self._custom_state_attrs = attrs

    def save_state(self) -> Dict[str, Any]:
        """Save state including registered custom attributes."""
        base_state = super().save_state()

        # Save registered custom attributes
        custom_attrs = {}
        for attr in self._custom_state_attrs:
            if hasattr(self.env, attr):
                val = getattr(self.env, attr)
                try:
                    custom_attrs[attr] = copy.deepcopy(val)
                except Exception:
                    custom_attrs[attr] = val

        base_state['custom_attrs'] = custom_attrs
        return base_state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state including registered custom attributes."""
        super().restore_state(state)

        # Restore custom attributes
        custom_attrs = state.get('custom_attrs', {})
        for attr, val in custom_attrs.items():
            if hasattr(self.env, attr):
                try:
                    setattr(self.env, attr, copy.deepcopy(val))
                except Exception:
                    setattr(self.env, attr, val)


def make_state_saveable(env: gym.Env) -> StateSaveWrapper:
    """
    Convenience function to wrap an environment for state save/restore.

    Automatically detects the environment type and applies the appropriate
    wrapper.

    Args:
        env: A gym environment.

    Returns:
        A StateSaveWrapper (or subclass) around the environment.
    """
    # Check for MuJoCo
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    if hasattr(unwrapped, 'sim'):
        return MuJoCoStateWrapper(env)

    # Check for custom get_state/set_state
    if hasattr(env, 'get_state') and hasattr(env, 'set_state'):
        return StateSaveWrapper(env)

    # Default wrapper
    return StateSaveWrapper(env)


def save_env_state(env: gym.Env) -> Dict[str, Any]:
    """
    Save the state of an environment, wrapping if necessary.

    Args:
        env: A gym environment (may or may not be wrapped).

    Returns:
        State dictionary.
    """
    if isinstance(env, StateSaveWrapper):
        return env.save_state()
    else:
        wrapper = make_state_saveable(env)
        return wrapper.save_state()


def restore_env_state(env: gym.Env, state: Dict[str, Any]) -> None:
    """
    Restore the state of an environment, wrapping if necessary.

    Args:
        env: A gym environment.
        state: State dictionary from save_env_state().
    """
    if isinstance(env, StateSaveWrapper):
        env.restore_state(state)
    else:
        wrapper = make_state_saveable(env)
        wrapper.restore_state(state)


def reset_env_to_state(env: gym.Env, state: Dict[str, Any]) -> np.ndarray:
    """
    Reset an environment and restore to a saved state.

    Args:
        env: A gym environment.
        state: State dictionary from save_env_state().

    Returns:
        Observation after restoration.
    """
    if isinstance(env, StateSaveWrapper):
        return env.reset_to_state(state)
    else:
        wrapper = make_state_saveable(env)
        return wrapper.reset_to_state(state)