"""
Autonomous Driving Environment Wrapper for RICE

Implements a MetaDrive "Macro-v1" environment wrapper with state save/restore
capabilities for critical state collection and mixed initial distribution.

MetaDrive is a driving simulator that provides a gym-like interface.
This wrapper:
- Configures the MetaDrive environment with appropriate settings
- Supports saving/restoring full environment state (vehicle positions, map state, etc.)
- Provides a consistent interface for the RICE pipeline

Reference: Li et al., "MetaDrive: Composing Diverse Driving Scenarios for Generalizable RL"
Paper: RICE uses MetaDrive "Macro-v1" environment for autonomous driving experiments.
"""

import copy
import pickle
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import gym

try:
    from metadrive import MetaDriveEnv
    HAS_METADRIVE = True
except ImportError:
    HAS_METADRIVE = False
    MetaDriveEnv = None


class MetaDriveStateWrapper(gym.Wrapper):
    """
    Wrapper for MetaDrive environments that adds state save/restore functionality.
    
    MetaDrive environments have complex internal state (vehicle positions, velocities,
    map configuration, traffic vehicles, etc.). This wrapper captures the full state
    via deep copy of relevant internal attributes and provides methods to restore them.
    
    The state is saved as a dictionary containing:
    - 'vehicle_states': Dict of vehicle state information
    - 'map_state': Map/road network state
    - 'traffic_states': States of other traffic vehicles
    - 'episode_step': Current step count
    - 'seed': Random seed state (if available)
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._saved_state: Optional[Dict[str, Any]] = None
        
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current full environment state.
        
        For MetaDrive, this captures:
        - The ego vehicle's position, heading, velocity, etc.
        - Other traffic vehicles' states
        - Map/road network configuration
        - Current episode step
        
        Returns:
            Dict containing the serializable environment state
        """
        state = {}
        
        # Try to save using MetaDrive's built-in state capture if available
        env_unwrapped = self.env
        while hasattr(env_unwrapped, 'env'):
            env_unwrapped = env_unwrapped.env
        
        # Capture vehicle state
        if hasattr(env_unwrapped, 'vehicle') and env_unwrapped.vehicle is not None:
            vehicle = env_unwrapped.vehicle
            state['vehicle_state'] = {
                'position': np.array(vehicle.position, dtype=np.float64).copy() if hasattr(vehicle, 'position') else None,
                'heading': float(vehicle.heading) if hasattr(vehicle, 'heading') else None,
                'velocity': np.array(vehicle.velocity, dtype=np.float64).copy() if hasattr(vehicle, 'velocity') else None,
                'steering': float(vehicle.steering) if hasattr(vehicle, 'steering') else None,
                'throttle_brake': float(vehicle.throttle_brake) if hasattr(vehicle, 'throttle_brake') else None,
            }
        
        # Capture traffic vehicles state
        if hasattr(env_unwrapped, 'traffic_vehicles') and env_unwrapped.traffic_vehicles is not None:
            traffic_states = []
            for tv in env_unwrapped.traffic_vehicles:
                tv_state = {
                    'position': np.array(tv.position, dtype=np.float64).copy() if hasattr(tv, 'position') else None,
                    'heading': float(tv.heading) if hasattr(tv, 'heading') else None,
                    'velocity': np.array(tv.velocity, dtype=np.float64).copy() if hasattr(tv, 'velocity') else None,
                }
                traffic_states.append(tv_state)
            state['traffic_states'] = traffic_states
        
        # Capture map/engine state if available
        if hasattr(env_unwrapped, 'engine') and env_unwrapped.engine is not None:
            engine = env_unwrapped.engine
            if hasattr(engine, 'get_state'):
                try:
                    state['engine_state'] = engine.get_state()
                except Exception:
                    state['engine_state'] = None
            # Try to capture random state for reproducibility
            if hasattr(engine, 'random_seed'):
                state['random_seed'] = engine.random_seed
        
        # Capture episode step
        if hasattr(env_unwrapped, 'episode_step'):
            state['episode_step'] = env_unwrapped.episode_step
        elif hasattr(env_unwrapped, 'step_count'):
            state['episode_step'] = env_unwrapped.step_count
        
        # Capture current observation (as fallback)
        try:
            # Get the latest observation if available
            if hasattr(env_unwrapped, 'observations') and hasattr(env_unwrapped.observations, 'last_obs'):
                state['last_observation'] = np.array(env_unwrapped.observations.last_obs, dtype=np.float64).copy()
        except Exception:
            pass
        
        # Deep copy the entire state for safety
        self._saved_state = copy.deepcopy(state)
        return self._saved_state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the environment to a previously saved state.
        
        Args:
            state: State dictionary previously returned by save_state()
        """
        if state is None:
            return
        
        env_unwrapped = self.env
        while hasattr(env_unwrapped, 'env'):
            env_unwrapped = env_unwrapped.env
        
        # Restore vehicle state
        if 'vehicle_state' in state and state['vehicle_state'] is not None:
            vs = state['vehicle_state']
            if hasattr(env_unwrapped, 'vehicle') and env_unwrapped.vehicle is not None:
                vehicle = env_unwrapped.vehicle
                if vs.get('position') is not None and hasattr(vehicle, 'set_position'):
                    vehicle.set_position(vs['position'])
                elif vs.get('position') is not None and hasattr(vehicle, 'position'):
                    # Direct assignment fallback
                    try:
                        vehicle.position = vs['position'].copy()
                    except Exception:
                        pass
                if vs.get('heading') is not None and hasattr(vehicle, 'set_heading'):
                    vehicle.set_heading(vs['heading'])
                if vs.get('velocity') is not None and hasattr(vehicle, 'set_velocity'):
                    vehicle.set_velocity(vs['velocity'])
                elif vs.get('velocity') is not None and hasattr(vehicle, 'velocity'):
                    try:
                        vehicle.velocity = vs['velocity'].copy()
                    except Exception:
                        pass
        
        # Restore traffic vehicles state
        if ('traffic_states' in state and state['traffic_states'] is not None 
            and hasattr(env_unwrapped, 'traffic_vehicles')):
            for i, tv_state in enumerate(state['traffic_states']):
                if i < len(env_unwrapped.traffic_vehicles):
                    tv = env_unwrapped.traffic_vehicles[i]
                    if tv_state.get('position') is not None and hasattr(tv, 'set_position'):
                        tv.set_position(tv_state['position'])
                    if tv_state.get('heading') is not None and hasattr(tv, 'set_heading'):
                        tv.set_heading(tv_state['heading'])
        
        # Restore engine state
        if 'engine_state' in state and state['engine_state'] is not None:
            if hasattr(env_unwrapped, 'engine') and hasattr(env_unwrapped.engine, 'set_state'):
                try:
                    env_unwrapped.engine.set_state(state['engine_state'])
                except Exception:
                    pass
        
        self._saved_state = copy.deepcopy(state)
    
    def reset_to_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Reset the environment and restore to a saved state.
        
        This is the primary method used by the RICE refining pipeline to implement
        the mixed initial distribution: with probability p, the environment is
        reset to a critical state instead of the default initial state.
        
        Args:
            state: State dictionary previously returned by save_state()
            
        Returns:
            Observation after restoring to the state
        """
        # First, do a normal reset to initialize the environment
        obs = self.env.reset()
        
        # Then restore the saved state
        if state is not None:
            self.restore_state(state)
            # Get the observation after state restoration
            # MetaDrive may need a step to update observations after state change
            try:
                obs = self._get_current_observation()
            except Exception:
                pass
        
        return obs
    
    def _get_current_observation(self) -> np.ndarray:
        """Get the current observation from the environment."""
        env_unwrapped = self.env
        while hasattr(env_unwrapped, 'env'):
            env_unwrapped = env_unwrapped.env
        
        if hasattr(env_unwrapped, 'get_observation'):
            return env_unwrapped.get_observation()
        elif hasattr(env_unwrapped, '_get_obs'):
            return env_unwrapped._get_obs()
        elif hasattr(env_unwrapped, 'observations') and hasattr(env_unwrapped.observations, 'last_obs'):
            return env_unwrapped.observations.last_obs
        else:
            # Return zeros as fallback
            obs_space = self.env.observation_space
            if hasattr(obs_space, 'shape'):
                return np.zeros(obs_space.shape, dtype=np.float32)
            return np.zeros(1, dtype=np.float32)


class SparseRewardWrapper(gym.Wrapper):
    """
    Wrapper that converts dense MetaDrive rewards to sparse rewards.
    
    In sparse mode, the agent only receives a reward when it successfully
    reaches the destination (or a significant milestone). This is used
    for evaluating the agent's ability to learn from sparse feedback.
    """
    
    def __init__(self, env: gym.Env, success_reward: float = 10.0, 
                 milestone_reward: float = 1.0, milestone_distance: float = 50.0):
        super().__init__(env)
        self.success_reward = success_reward
        self.milestone_reward = milestone_reward
        self.milestone_distance = milestone_distance
        self._last_distance = None
        self._total_distance = 0.0
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_distance = None
        self._total_distance = 0.0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Convert to sparse reward
        sparse_reward = 0.0
        
        # Check for success (reached destination)
        if info.get('arrive_dest', False) or info.get('reach_destination', False):
            sparse_reward = self.success_reward
        
        # Check for milestone (traveled significant distance)
        if 'distance_travelled' in info:
            dist = info['distance_travelled']
            if self._last_distance is not None:
                segment = dist - self._last_distance
                self._total_distance += segment
                if self._total_distance >= self.milestone_distance:
                    sparse_reward += self.milestone_reward
                    self._total_distance = 0.0
            self._last_distance = dist
        
        # Override reward
        info['original_reward'] = reward
        return obs, sparse_reward, done, info


def make_metadrive_env(config: Optional[Dict[str, Any]] = None, 
                       use_sparse_reward: bool = False,
                       seed: int = 42,
                       max_steps: Optional[int] = None) -> gym.Env:
    """
    Create a MetaDrive autonomous driving environment with state save/restore wrapper.
    
    Args:
        config: MetaDrive configuration dictionary. If None, uses default Macro-v1 settings.
        use_sparse_reward: Whether to use sparse reward variant
        seed: Random seed
        max_steps: Maximum episode steps (overrides config if provided)
        
    Returns:
        Wrapped gym environment with state save/restore capability
        
    Raises:
        ImportError: If MetaDrive is not installed
    """
    if not HAS_METADRIVE:
        raise ImportError(
            "MetaDrive is not installed. Please install it with:\n"
            "  pip install metadrive-simulator\n"
            "Or follow instructions at: https://github.com/metadriverse/metadrive"
        )
    
    # Default configuration for Macro-v1 environment
    # Based on MetaDrive paper and RICE paper specifications
    default_config = {
        "environment_num": 100,  # Number of different maps for training
        "start_seed": seed,
        "traffic_density": 0.1,  # Moderate traffic
        "accident_prob": 1.0,  # Collision detection enabled
        "use_render": False,
        "vehicle_config": {
            "lidar": {
                "num_lasers": 240,
                "distance": 50,
                "num_others": 4,
            },
        },
        "map": "SSSSS",  # Map type (S = straight, C = curve, etc.)
        "random_spawn": True,
        "horizon": max_steps if max_steps is not None else 1000,
        "agent_policy": None,  # We control the agent
        "crash_done": True,  # Episode ends on collision
        "out_of_road_done": True,  # Episode ends when off road
        "crash_vehicle_penalty": 5.0,
        "crash_object_penalty": 5.0,
        "out_of_road_penalty": 5.0,
        "driving_reward": 1.0,  # Reward for forward movement
        "speed_reward": 0.1,  # Reward for maintaining speed
        "use_lateral_reward": False,
        "success_reward": 20.0,  # Reward for reaching destination
    }
    
    if config is not None:
        default_config.update(config)
    
    # Create the MetaDrive environment
    env = MetaDriveEnv(default_config)
    
    # Wrap with state save/restore
    env = MetaDriveStateWrapper(env)
    
    # Optionally wrap with sparse reward
    if use_sparse_reward:
        env = SparseRewardWrapper(env)
    
    # Set seed
    env.seed(seed)
    
    return env


def make_state_saveable_metadrive(env: gym.Env) -> MetaDriveStateWrapper:
    """
    Wrap an existing MetaDrive environment with state save/restore capability.
    
    Args:
        env: A MetaDrive gym environment
        
    Returns:
        MetaDriveStateWrapper with state save/restore
    """
    if isinstance(env, MetaDriveStateWrapper):
        return env
    return MetaDriveStateWrapper(env)


# Convenience function matching the pattern from rice/env_wrappers.py
def make_env(env_name: str = "MetaDrive-Macro-v1",
             seed: int = 42,
             max_episode_steps: Optional[int] = None,
             use_sparse_reward: bool = False,
             config: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    Create a MetaDrive environment with state save/restore wrapper.
    
    This is the main entry point for creating autonomous driving environments
    in the RICE pipeline. It matches the interface used by other domain scripts.
    
    Args:
        env_name: Environment identifier (default: "MetaDrive-Macro-v1")
        seed: Random seed
        max_episode_steps: Maximum steps per episode
        use_sparse_reward: Whether to use sparse rewards
        config: Additional MetaDrive configuration
        
    Returns:
        Wrapped gym environment
    """
    return make_metadrive_env(
        config=config,
        use_sparse_reward=use_sparse_reward,
        seed=seed,
        max_steps=max_episode_steps,
    )


# For compatibility with rice.env_wrappers.make_state_saveable
def get_state_dim(env: gym.Env) -> int:
    """Get the state/observation dimension for a MetaDrive environment."""
    obs_space = env.observation_space
    if hasattr(obs_space, 'shape'):
        return int(np.prod(obs_space.shape))
    elif hasattr(obs_space, 'n'):
        return int(obs_space.n)
    return 1


def get_action_dim(env: gym.Env) -> int:
    """Get the action dimension for a MetaDrive environment."""
    act_space = env.action_space
    if hasattr(act_space, 'shape'):
        return int(np.prod(act_space.shape))
    elif hasattr(act_space, 'n'):
        return int(act_space.n)
    return 1


def is_discrete_action(env: gym.Env) -> bool:
    """Check if the environment has discrete action space."""
    return isinstance(env.action_space, gym.spaces.Discrete)


# Test function
def test_env():
    """Quick test to verify the environment wrapper works."""
    if not HAS_METADRIVE:
        print("MetaDrive not installed. Skipping test.")
        return
    
    print("Creating MetaDrive environment...")
    env = make_metadrive_env(seed=42, max_steps=200)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test state save/restore
    print("Testing state save/restore...")
    state = env.save_state()
    print(f"Saved state keys: {list(state.keys())}")
    
    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    
    # Restore state
    obs_restored = env.reset_to_state(state)
    print(f"Restored observation shape: {obs_restored.shape}")
    
    print("Environment test passed!")
    env.close()


if __name__ == "__main__":
    test_env()