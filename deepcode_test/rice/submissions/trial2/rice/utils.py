"""
RICE Utilities Module

Provides helper functions for:
- Configuration loading and merging
- State setting and environment manipulation
- Replay buffer for critical states
- General utilities (seeding, logging, etc.)
"""

import os
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque
import pickle
import logging
from pathlib import Path


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_config(env_name: Optional[str] = None, base_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load default configuration and optionally merge with environment-specific overrides.

    Args:
        env_name: Name of the environment (e.g., 'hopper', 'walker2d'). If provided,
                  loads env_specific/{env_name}.yaml and merges with defaults.
        base_config_path: Path to default.yaml. If None, uses config/default.yaml
                          relative to project root.

    Returns:
        Merged configuration dictionary.
    """
    # Determine project root
    if base_config_path is None:
        project_root = Path(__file__).parent.parent
        base_config_path = project_root / "config" / "default.yaml"
    else:
        base_config_path = Path(base_config_path)

    # Load default config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge with environment-specific config if provided
    if env_name is not None:
        env_config_path = base_config_path.parent / "env_specific" / f"{env_name}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
            config = deep_merge(config, env_config)

    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries. Values from override take precedence.

    Args:
        base: Base dictionary.
        override: Override dictionary.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration dictionary to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# ==============================================================================
# Seeding and Reproducibility
# ==============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across numpy, random, and torch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# Critical State Buffer
# ==============================================================================

class CriticalStateBuffer:
    """
    Buffer for storing critical states identified by the mask network.

    Each entry stores:
        - state: The critical state vector
        - action: The agent's action at that state (optional)
        - next_state: The resulting next state (optional)
        - importance: The importance score I(s) = 1 - ξ(aᵉ=0|s)
        - trajectory_id: Identifier for the trajectory
        - step: Step index within the trajectory
    """

    def __init__(self, max_size: int = 10000):
        """
        Args:
            max_size: Maximum number of critical states to store.
        """
        self.max_size = max_size
        self.states: List[np.ndarray] = []
        self.actions: List[Any] = []
        self.next_states: List[Optional[np.ndarray]] = []
        self.importances: List[float] = []
        self.trajectory_ids: List[int] = []
        self.steps: List[int] = []

    def add(self,
            state: np.ndarray,
            action: Any = None,
            next_state: Optional[np.ndarray] = None,
            importance: float = 0.0,
            trajectory_id: int = 0,
            step: int = 0) -> None:
        """
        Add a critical state to the buffer. If buffer is full, removes the
        entry with the lowest importance score.

        Args:
            state: State vector.
            action: Action taken at this state.
            next_state: Resulting next state.
            importance: Importance score I(s).
            trajectory_id: Trajectory identifier.
            step: Step index within trajectory.
        """
        if len(self.states) >= self.max_size:
            # Remove entry with lowest importance
            min_idx = np.argmin(self.importances)
            self._remove_at(min_idx)

        self.states.append(np.array(state, copy=True))
        self.actions.append(action)
        self.next_states.append(
            np.array(next_state, copy=True) if next_state is not None else None
        )
        self.importances.append(importance)
        self.trajectory_ids.append(trajectory_id)
        self.steps.append(step)

    def _remove_at(self, idx: int) -> None:
        """Remove entry at given index."""
        del self.states[idx]
        del self.actions[idx]
        del self.next_states[idx]
        del self.importances[idx]
        del self.trajectory_ids[idx]
        del self.steps[idx]

    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Sample n critical states uniformly from the buffer.

        Args:
            n: Number of states to sample.

        Returns:
            List of dictionaries with keys: state, action, next_state,
            importance, trajectory_id, step.
        """
        if len(self.states) == 0:
            return []

        n = min(n, len(self.states))
        indices = np.random.choice(len(self.states), size=n, replace=False)

        samples = []
        for idx in indices:
            samples.append({
                'state': self.states[idx],
                'action': self.actions[idx],
                'next_state': self.next_states[idx],
                'importance': self.importances[idx],
                'trajectory_id': self.trajectory_ids[idx],
                'step': self.steps[idx],
            })
        return samples

    def get_top_k(self, k: int = 10) -> List[Dict[str, Any]]:
        """
        Get the k states with highest importance scores.

        Args:
            k: Number of top states to return.

        Returns:
            List of dictionaries sorted by importance (descending).
        """
        if len(self.states) == 0:
            return []

        k = min(k, len(self.states))
        sorted_indices = np.argsort(self.importances)[::-1][:k]

        results = []
        for idx in sorted_indices:
            results.append({
                'state': self.states[idx],
                'action': self.actions[idx],
                'next_state': self.next_states[idx],
                'importance': self.importances[idx],
                'trajectory_id': self.trajectory_ids[idx],
                'step': self.steps[idx],
            })
        return results

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str) -> None:
        """Save buffer to disk using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'states': self.states,
            'actions': self.actions,
            'next_states': self.next_states,
            'importances': self.importances,
            'trajectory_ids': self.trajectory_ids,
            'steps': self.steps,
            'max_size': self.max_size,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'CriticalStateBuffer':
        """Load buffer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        buffer = cls(max_size=data['max_size'])
        buffer.states = data['states']
        buffer.actions = data['actions']
        buffer.next_states = data['next_states']
        buffer.importances = data['importances']
        buffer.trajectory_ids = data['trajectory_ids']
        buffer.steps = data['steps']
        return buffer


# ==============================================================================
# State Setting Utilities
# ==============================================================================

def set_mujoco_state(env, state: np.ndarray) -> None:
    """
    Set the full state of a MuJoCo environment.

    For MuJoCo environments (via gymnasium/mujoco), this sets the internal
    simulation state using the qpos and qvel arrays.

    Args:
        env: MuJoCo Gym environment (must have `sim` or `unwrapped.sim` attribute).
        state: Full state vector (concatenation of qpos and qvel).
    """
    # Try to access the sim object
    if hasattr(env, 'sim'):
        sim = env.sim
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'sim'):
        sim = env.unwrapped.sim
    elif hasattr(env, 'env') and hasattr(env.env, 'sim'):
        sim = env.env.sim
    else:
        raise AttributeError("Cannot access MuJoCo sim object from environment")

    # Get model dimensions
    nq = sim.model.nq  # number of joint positions
    nv = sim.model.nv  # number of joint velocities

    # Split state into qpos and qvel
    qpos = state[:nq]
    qvel = state[nq:nq + nv]

    # Set the state
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel

    # Forward kinematics to update derived quantities
    sim.forward()


def get_mujoco_state(env) -> np.ndarray:
    """
    Get the full state of a MuJoCo environment.

    Args:
        env: MuJoCo Gym environment.

    Returns:
        Full state vector (concatenation of qpos and qvel).
    """
    if hasattr(env, 'sim'):
        sim = env.sim
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'sim'):
        sim = env.unwrapped.sim
    elif hasattr(env, 'env') and hasattr(env.env, 'sim'):
        sim = env.env.sim
    else:
        raise AttributeError("Cannot access MuJoCo sim object from environment")

    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    return np.concatenate([qpos, qvel])


def set_env_state(env, state: np.ndarray) -> None:
    """
    Generic state-setting function that tries multiple strategies.

    Strategies (in order):
    1. MuJoCo-style: use sim.data.qpos/qvel
    2. env.set_state(state) method if available
    3. env.reset_to_state(state) method if available
    4. Fallback: store state and use env-specific restore

    Args:
        env: Gym environment.
        state: State vector to set.
    """
    # Strategy 1: MuJoCo
    try:
        set_mujoco_state(env, state)
        return
    except (AttributeError, IndexError):
        pass

    # Strategy 2: Direct set_state method
    if hasattr(env, 'set_state'):
        env.set_state(state)
        return
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'set_state'):
        env.unwrapped.set_state(state)
        return

    # Strategy 3: reset_to_state method
    if hasattr(env, 'reset_to_state'):
        env.reset_to_state(state)
        return
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'reset_to_state'):
        env.unwrapped.reset_to_state(state)
        return

    # Strategy 4: Try env.sim.set_state (for some MuJoCo wrappers)
    try:
        if hasattr(env, 'sim'):
            env.sim.set_state(state)
            return
    except Exception:
        pass

    raise NotImplementedError(
        f"Environment {type(env).__name__} does not support set_state. "
        "Implement a custom wrapper or use environment-specific methods."
    )


# ==============================================================================
# Trajectory Collection
# ==============================================================================

def collect_trajectory(env, policy, max_steps: int = 1000,
                       deterministic: bool = False) -> Dict[str, Any]:
    """
    Collect a single trajectory using the given policy.

    Args:
        env: Gym environment.
        policy: Policy function that takes observation and returns action.
                Can be a callable or an SB3 model (with .predict() method).
        max_steps: Maximum steps per trajectory.
        deterministic: Whether to use deterministic actions.

    Returns:
        Dictionary with keys:
            - observations: List of observations
            - actions: List of actions
            - rewards: List of rewards
            - next_observations: List of next observations
            - dones: List of done flags
            - total_reward: Sum of rewards
            - length: Number of steps
    """
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    obs, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        # Get action from policy
        if hasattr(policy, 'predict'):
            action, _states = policy.predict(obs, deterministic=deterministic)
        else:
            action = policy(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        dones.append(done)

        total_reward += reward
        obs = next_obs

        if done:
            break

    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'dones': dones,
        'total_reward': total_reward,
        'length': len(observations),
    }


def collect_trajectories(env, policy, num_trajectories: int = 100,
                         max_steps: int = 1000,
                         deterministic: bool = False) -> List[Dict[str, Any]]:
    """
    Collect multiple trajectories.

    Args:
        env: Gym environment.
        policy: Policy function.
        num_trajectories: Number of trajectories to collect.
        max_steps: Maximum steps per trajectory.
        deterministic: Whether to use deterministic actions.

    Returns:
        List of trajectory dictionaries.
    """
    trajectories = []
    for i in range(num_trajectories):
        traj = collect_trajectory(env, policy, max_steps, deterministic)
        trajectories.append(traj)
    return trajectories


# ==============================================================================
# Logging and Metrics
# ==============================================================================

class Logger:
    """
    Simple logger for tracking experiment metrics.

    Stores scalar values over time and supports computing statistics.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.metrics: Dict[str, List[float]] = {}
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a scalar metric.

        Args:
            key: Metric name.
            value: Scalar value.
            step: Optional step number (stored as separate metric if provided).
        """
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

        if step is not None:
            step_key = f"{key}_step"
            if step_key not in self.metrics:
                self.metrics[step_key] = []
            self.metrics[step_key].append(step)

    def get_metric(self, key: str) -> List[float]:
        """Get all logged values for a metric."""
        return self.metrics.get(key, [])

    def get_stats(self, key: str) -> Dict[str, float]:
        """Get statistics (mean, std, min, max) for a metric."""
        values = self.get_metric(key)
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }

    def save(self, filename: str = "metrics.pkl") -> None:
        """Save metrics to disk."""
        if self.log_dir:
            path = os.path.join(self.log_dir, filename)
            with open(path, 'wb') as f:
                pickle.dump(self.metrics, f)

    def load(self, path: str) -> None:
        """Load metrics from disk."""
        with open(path, 'rb') as f:
            self.metrics = pickle.load(f)

    def to_dataframe(self):
        """Convert metrics to pandas DataFrame (requires pandas)."""
        import pandas as pd
        return pd.DataFrame(self.metrics)


# ==============================================================================
# Device Management
# ==============================================================================

def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get torch device from string or auto-detect.

    Args:
        device_str: 'cpu', 'cuda', 'cuda:0', etc. If None, auto-detects.

    Returns:
        torch.device
    """
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_str)


# ==============================================================================
# Network Initialization
# ==============================================================================

def init_weights(m: torch.nn.Module, gain: float = 1.0) -> None:
    """
    Initialize network weights using orthogonal initialization.

    Args:
        m: PyTorch module.
        gain: Gain factor for orthogonal initialization.
    """
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def build_mlp(input_dim: int,
              output_dim: int,
              hidden_sizes: List[int],
              activation: torch.nn.Module = torch.nn.ReLU,
              output_activation: Optional[torch.nn.Module] = None,
              use_layer_norm: bool = False) -> torch.nn.Sequential:
    """
    Build a Multi-Layer Perceptron.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        hidden_sizes: List of hidden layer sizes.
        activation: Activation function class.
        output_activation: Optional activation for output layer.
        use_layer_norm: Whether to use LayerNorm after each hidden layer.

    Returns:
        torch.nn.Sequential model.
    """
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(torch.nn.Linear(prev_dim, hidden_dim))
        if use_layer_norm:
            layers.append(torch.nn.LayerNorm(hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim

    layers.append(torch.nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())

    model = torch.nn.Sequential(*layers)
    model.apply(init_weights)
    return model


# ==============================================================================
# Environment Helpers
# ==============================================================================

def make_env(env_id: str, seed: int = 0, **kwargs) -> Any:
    """
    Create a Gym environment with proper seeding.

    Args:
        env_id: Gym environment ID (e.g., 'Hopper-v4').
        seed: Random seed.
        **kwargs: Additional arguments for gym.make.

    Returns:
        Gym environment.
    """
    import gymnasium as gym

    env = gym.make(env_id, **kwargs)
    env.reset(seed=seed)
    return env


def make_vec_env(env_id: str, n_envs: int = 1, seed: int = 0, **kwargs) -> Any:
    """
    Create a vectorized environment using Stable-Baselines3 utilities.

    Args:
        env_id: Gym environment ID.
        n_envs: Number of parallel environments.
        seed: Random seed.
        **kwargs: Additional arguments.

    Returns:
        VecEnv instance.
    """
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

    return sb3_make_vec_env(env_id, n_envs=n_envs, seed=seed, **kwargs)


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_policy(env, policy, n_episodes: int = 100,
                    deterministic: bool = True,
                    render: bool = False) -> Dict[str, float]:
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: Gym environment.
        policy: Policy to evaluate (SB3 model or callable).
        n_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic actions.
        render: Whether to render the environment.

    Returns:
        Dictionary with 'mean_reward', 'std_reward', 'rewards' list.
    """
    rewards = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(obs, deterministic=deterministic)
            else:
                action = policy(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if render:
                env.render()

        rewards.append(episode_reward)

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'rewards': rewards,
    }


# ==============================================================================
# Miscellaneous
# ==============================================================================

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"