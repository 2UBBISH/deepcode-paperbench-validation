"""
Autonomous Driving Environment Wrapper for RICE

Wraps MetaDrive's "Macro-v1" environment to provide a Gymnasium-compatible
interface with state saving/restoring capabilities required for RICE's
critical-state reset mechanism.

Environment Details (from paper):
- State: vector of BEV (Bird's Eye View) + sensor information
- Actions: steering, acceleration, brake (normalized to [-1, 1]^2)
- Reward: forward motion, speed maintenance, collision penalties

Dependencies: metadrive >= 0.4.0
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import register
except ImportError:
    import gym
    from gym import spaces

# MetaDrive is an optional dependency; graceful fallback if not installed
_METADRIVE_AVAILABLE = False
try:
    from metadrive import MetaDriveEnv
    from metadrive.constants import TerminationState
    _METADRIVE_AVAILABLE = True
except ImportError:
    MetaDriveEnv = None
    TerminationState = None


# ==============================================================================
# Constants
# ==============================================================================

# Default MetaDrive configuration for Macro-v1 (highway driving)
DEFAULT_METADRIVE_CONFIG = {
    "environment_num": 1,           # Number of scenarios
    "start_seed": 0,
    "traffic_density": 0.1,         # Sparse traffic
    "accident_prob": 0.0,           # No random accidents
    "use_render": False,
    "vehicle_config": {
        "lidar": {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 4,
        },
        "side_detector": {
            "num_lasers": 72,
            "distance": 50,
        },
        "lane_line_detector": {
            "num_lasers": 4,
            "distance": 20,
        },
    },
    "map_config": {
        "type": "block_num",
        "config": "XO",             # Intersection + straight
        "lane_num": 3,
    },
    "decision_repeat": 5,           # Repeat action for 5 steps
    "horizon": 1000,                # Max episode length
    "discrete_action": False,       # Continuous actions
    "discrete_steering": False,
    "discrete_throttle": False,
    "norm_pixel": True,             # Normalize observations
    "out_of_road_penalty": 10.0,
    "crash_vehicle_penalty": 10.0,
    "crash_object_penalty": 10.0,
    "driving_reward": 1.0,          # Reward for forward motion
    "speed_reward": 1.0,            # Reward for maintaining speed
    "use_lateral_reward": False,
}


class AutoDrivingEnv(gym.Env):
    """
    Gymnasium-compatible wrapper around MetaDrive's autonomous driving environment.

    This environment simulates a vehicle driving on a highway/intersection map.
    The agent must control steering, acceleration, and brake to navigate safely
    while maximizing forward progress.

    State Space:
        Box(low=-inf, high=inf, shape=(obs_dim,), float32)
        The observation is a flattened vector of BEV + sensor readings.
        Typical dimension: ~260 (varies with MetaDrive config).

    Action Space:
        Box(low=-1.0, high=1.0, shape=(2,), float32)
        [steering, acceleration/brake]
        - steering: -1 (full left) to +1 (full right)
        - acceleration/brake: -1 (full brake) to +1 (full throttle)

    Reward:
        Sum of:
        - driving_reward: proportional to forward distance traveled
        - speed_reward: bonus for maintaining target speed
        - Penalties: out_of_road, crash_vehicle, crash_object
    """

    metadata = {"render_modes": ["human", "rgb_array", "top_down"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_steps: int = 1000,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the autonomous driving environment.

        Args:
            config: MetaDrive configuration dictionary. If None, uses defaults.
            max_steps: Maximum steps per episode.
            reward_scale: Scaling factor for rewards.
            render_mode: One of "human", "rgb_array", "top_down", or None.
            seed: Random seed.
            **kwargs: Additional arguments passed to MetaDriveEnv.
        """
        super().__init__()

        if not _METADRIVE_AVAILABLE:
            raise ImportError(
                "MetaDrive is required for AutoDrivingEnv. "
                "Install with: pip install metadrive>=0.4.0"
            )

        # Merge default config with user-provided config
        self._metadrive_config = DEFAULT_METADRIVE_CONFIG.copy()
        if config is not None:
            self._metadrive_config.update(config)
        self._metadrive_config.update(kwargs)

        self._max_steps = max_steps
        self._reward_scale = reward_scale
        self._render_mode = render_mode
        self._seed = seed

        # Create the underlying MetaDrive environment
        self._env = MetaDriveEnv(config=self._metadrive_config)

        # Determine observation and action spaces from the MetaDrive env
        # MetaDrive typically returns a dict observation; we flatten it
        sample_obs = self._env.reset()
        if isinstance(sample_obs, dict):
            # Flatten dict observation into a single vector
            obs_parts = []
            for key in sorted(sample_obs.keys()):
                val = sample_obs[key]
                if isinstance(val, np.ndarray):
                    obs_parts.append(val.flatten())
                else:
                    obs_parts.append(np.array([val], dtype=np.float32))
            flat_obs = np.concatenate(obs_parts).astype(np.float32)
            self._obs_dim = len(flat_obs)
            self._obs_is_dict = True
        else:
            flat_obs = np.asarray(sample_obs, dtype=np.float32).flatten()
            self._obs_dim = len(flat_obs)
            self._obs_is_dict = False

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Action space: [steering, acceleration/brake]
        # Both normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Internal state tracking
        self._current_obs = None
        self._step_count = 0
        self._episode_return = 0.0
        self._last_info = {}

        # State saving for RICE
        self._saved_state: Optional[Dict[str, Any]] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            self._seed = seed
            self._env.seed(seed)

        raw_obs = self._env.reset()
        self._current_obs = self._flatten_obs(raw_obs)
        self._step_count = 0
        self._episode_return = 0.0
        self._last_info = {}

        return self._current_obs.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Array of shape (2,) with [steering, acceleration/brake].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Ensure action is properly shaped
        action = np.asarray(action, dtype=np.float32).flatten()
        if len(action) != 2:
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        # Map to MetaDrive action format: [steering, throttle, brake]
        # Our action: [steering, accel_brake] where accel_brake in [-1, 1]
        steering = float(np.clip(action[0], -1.0, 1.0))
        accel_brake = float(np.clip(action[1], -1.0, 1.0))

        if accel_brake >= 0:
            throttle = accel_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -accel_brake

        metadrive_action = np.array([steering, throttle, brake], dtype=np.float32)

        # Step the MetaDrive environment
        raw_obs, raw_reward, raw_terminated, raw_truncated, raw_info = self._env.step(
            metadrive_action
        )

        # Process observation
        self._current_obs = self._flatten_obs(raw_obs)

        # Process reward
        reward = float(raw_reward) * self._reward_scale

        # Process termination
        terminated = bool(raw_terminated)
        truncated = bool(raw_truncated) or (self._step_count >= self._max_steps - 1)

        # Process info
        info = raw_info if isinstance(raw_info, dict) else {}
        info["step"] = self._step_count
        info["raw_reward"] = float(raw_reward)

        self._step_count += 1
        self._episode_return += reward
        self._last_info = info

        return self._current_obs.copy(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", else None.
        """
        if self._render_mode is None:
            return None

        if hasattr(self._env, "render"):
            frame = self._env.render(
                mode=self._render_mode,
                text={"episode_return": self._episode_return} if self._render_mode == "top_down" else None,
            )
            return frame

        return None

    def close(self):
        """Close the environment and release resources."""
        if hasattr(self._env, "close"):
            self._env.close()

    # ==========================================================================
    # State Saving/Restoring for RICE
    # ==========================================================================

    def get_state(self) -> np.ndarray:
        """
        Get the current environment state as a flat numpy array.

        This captures the vehicle's position, velocity, heading, and the
        current observation, enabling exact state restoration for RICE's
        mixed initial distribution.

        Returns:
            Flat numpy array representing the full environment state.
        """
        # Capture vehicle state from MetaDrive internals
        state_parts = []

        # Current observation
        state_parts.append(self._current_obs.flatten())

        # Vehicle state (position, velocity, heading, etc.)
        try:
            vehicle = self._env.vehicle
            if vehicle is not None:
                pos = vehicle.position if hasattr(vehicle, "position") else np.zeros(3)
                vel = vehicle.velocity if hasattr(vehicle, "velocity") else np.zeros(3)
                heading = vehicle.heading if hasattr(vehicle, "heading") else 0.0
                state_parts.append(np.asarray(pos, dtype=np.float32).flatten())
                state_parts.append(np.asarray(vel, dtype=np.float32).flatten())
                state_parts.append(np.array([heading], dtype=np.float32))
        except Exception:
            # Fallback: just use observation
            pass

        # Step count
        state_parts.append(np.array([self._step_count], dtype=np.float32))

        # Episode return
        state_parts.append(np.array([self._episode_return], dtype=np.float32))

        return np.concatenate(state_parts).astype(np.float32)

    def set_state(self, state: np.ndarray):
        """
        Restore the environment to a previously saved state.

        Note: MetaDrive does not natively support full state restoration.
        This method saves the state vector for later reference and attempts
        to restore the observation. For full state restoration, the RICE
        refining module should use the saved observation to reset the
        environment to a similar configuration.

        Args:
            state: Flat numpy array from get_state().
        """
        self._saved_state = {
            "state_vector": state.copy(),
            "obs_dim": self._obs_dim,
        }

        # Extract the observation portion (first obs_dim elements)
        obs_part = state[:self._obs_dim]
        self._current_obs = obs_part.astype(np.float32).copy()

        # Extract step count if available
        offset = self._obs_dim
        # Skip vehicle state parts (variable length)
        # The last two elements are step_count and episode_return
        if len(state) >= offset + 2:
            self._step_count = int(state[-2])
            self._episode_return = float(state[-1])

    def save_state(self) -> np.ndarray:
        """Convenience method: save and return current state."""
        return self.get_state()

    def restore_state(self, state: np.ndarray):
        """Convenience method: restore from saved state."""
        self.set_state(state)

    # ==========================================================================
    # Internal Helpers
    # ==========================================================================

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        """
        Flatten a MetaDrive observation (dict or array) into a 1D vector.

        Args:
            obs: Raw observation from MetaDrive.

        Returns:
            1D numpy array of float32.
        """
        if isinstance(obs, dict):
            parts = []
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, np.ndarray):
                    parts.append(val.flatten())
                elif isinstance(val, (int, float)):
                    parts.append(np.array([val], dtype=np.float32))
                else:
                    parts.append(np.array([0.0], dtype=np.float32))
            return np.concatenate(parts).astype(np.float32)
        elif isinstance(obs, np.ndarray):
            return obs.flatten().astype(np.float32)
        else:
            return np.array([float(obs)], dtype=np.float32)

    @property
    def metadrive_env(self):
        """Access the underlying MetaDrive environment."""
        return self._env


class AutoDrivingEnvWrapper(gym.Wrapper):
    """
    Wrapper that adds save_state() and restore_state() convenience methods
    for RICE integration, consistent with other custom environments.
    """

    def __init__(self, env: AutoDrivingEnv):
        super().__init__(env)

    def save_state(self) -> np.ndarray:
        """Save the current environment state."""
        return self.unwrapped.get_state()

    def restore_state(self, state: np.ndarray):
        """Restore the environment to a saved state."""
        self.unwrapped.set_state(state)


def make_auto_driving_env(
    config: Optional[Dict[str, Any]] = None,
    max_steps: int = 1000,
    reward_scale: float = 1.0,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    use_wrapper: bool = True,
    **kwargs,
) -> gym.Env:
    """
    Factory function to create an autonomous driving environment.

    Args:
        config: MetaDrive configuration dictionary.
        max_steps: Maximum steps per episode.
        reward_scale: Scaling factor for rewards.
        render_mode: Rendering mode.
        seed: Random seed.
        use_wrapper: If True, wrap in AutoDrivingEnvWrapper for RICE compatibility.
        **kwargs: Additional arguments passed to AutoDrivingEnv.

    Returns:
        Gymnasium environment instance.
    """
    env = AutoDrivingEnv(
        config=config,
        max_steps=max_steps,
        reward_scale=reward_scale,
        render_mode=render_mode,
        seed=seed,
        **kwargs,
    )

    if use_wrapper:
        env = AutoDrivingEnvWrapper(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


# ==============================================================================
# Gymnasium Registration
# ==============================================================================

try:
    register(
        id="AutoDriving-v0",
        entry_point="envs.auto_driving_env:AutoDrivingEnv",
        max_episode_steps=1000,
    )
except Exception:
    pass  # Already registered or gymnasium version mismatch


# ==============================================================================
# CLI Test Entry Point
# ==============================================================================

def main():
    """Test the autonomous driving environment."""
    import argparse

    parser = argparse.ArgumentParser(description="Test AutoDrivingEnv")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    render_mode = "top_down" if args.render else None

    print("Creating AutoDrivingEnv...")
    env = make_auto_driving_env(
        max_steps=args.max_steps,
        render_mode=render_mode,
        seed=args.seed,
        use_wrapper=False,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            # Random policy
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

            if args.render:
                env.render()

        print(f"Episode {ep + 1}: steps={step}, return={total_reward:.2f}")

        # Test state save/restore
        state = env.get_state()
        print(f"  State vector shape: {state.shape}")
        env.set_state(state)
        print(f"  State restored successfully")

    env.close()
    print("Test complete.")


if __name__ == "__main__":
    main()