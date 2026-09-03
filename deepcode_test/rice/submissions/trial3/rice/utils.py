"""
RICE Utility Functions
======================
General-purpose utilities for trajectory collection, Generalized Advantage
Estimation (GAE), discounted returns computation, and state serialization.

These utilities are used by the mask network trainer, the refining algorithm,
and experiment scripts across all domains.
"""

import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Trajectory Collection
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """
    A simple buffer for storing trajectory data during collection.

    Stores: states, actions, rewards, dones, values, log_probs, masks,
    and optionally next_states and infos.
    """

    def __init__(self, state_dim: int, action_dim: int, capacity: int,
                 discrete_action: bool = False, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.discrete_action = discrete_action
        self.device = device

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        if discrete_action:
            self.actions = np.zeros((capacity,), dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.masks = np.zeros((capacity,), dtype=np.float32)  # 1.0 = not done

        # Optional fields
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.infos: List[Dict] = []

        self.ptr = 0
        self.full = False

    def add(self, state: np.ndarray, action: Union[np.ndarray, int],
            reward: float, done: bool, value: float, log_prob: float,
            mask: float = 1.0, next_state: Optional[np.ndarray] = None,
            info: Optional[Dict] = None):
        """Add a single transition to the buffer."""
        if self.ptr >= self.capacity:
            self.full = True
            # Shift buffer left by half capacity (ring-buffer style)
            half = self.capacity // 2
            self.states[:half] = self.states[half:]
            self.actions[:half] = self.actions[half:]
            self.rewards[:half] = self.rewards[half:]
            self.dones[:half] = self.dones[half:]
            self.values[:half] = self.values[half:]
            self.log_probs[:half] = self.log_probs[half:]
            self.masks[:half] = self.masks[half:]
            self.next_states[:half] = self.next_states[half:]
            self.ptr = half

        idx = self.ptr
        self.states[idx] = state
        if self.discrete_action:
            self.actions[idx] = int(action)
        else:
            self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.masks[idx] = mask
        if next_state is not None:
            self.next_states[idx] = next_state
        if info is not None:
            if len(self.infos) <= idx:
                self.infos.append(info)
            else:
                self.infos[idx] = info
        self.ptr += 1

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return all stored data up to current pointer."""
        end = self.ptr
        return {
            "states": self.states[:end],
            "actions": self.actions[:end],
            "rewards": self.rewards[:end],
            "dones": self.dones[:end],
            "values": self.values[:end],
            "log_probs": self.log_probs[:end],
            "masks": self.masks[:end],
            "next_states": self.next_states[:end],
        }

    def clear(self):
        """Reset the buffer."""
        self.ptr = 0
        self.full = False
        self.infos.clear()

    def __len__(self) -> int:
        return self.ptr


def collect_trajectories(
    env,
    policy_fn: Callable[[np.ndarray], Tuple[Any, float, float, float]],
    num_steps: int,
    state_dim: int,
    action_dim: int,
    gamma: float = 0.99,
    discrete_action: bool = False,
    device: str = "cpu",
    verbose: bool = False,
) -> TrajectoryBuffer:
    """
    Collect trajectories by running the policy in the environment.

    Args:
        env: Gym-like environment with reset() and step().
        policy_fn: Function that takes a state and returns
            (action, log_prob, value, entropy).
        num_steps: Total number of environment steps to collect.
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        gamma: Discount factor (used for bootstrapping value at episode end).
        discrete_action: Whether action space is discrete.
        device: Device string (unused here, for API consistency).
        verbose: If True, print collection progress.

    Returns:
        TrajectoryBuffer containing collected transitions.
    """
    buffer = TrajectoryBuffer(state_dim, action_dim, num_steps,
                              discrete_action=discrete_action, device=device)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Gym 0.26+ returns (obs, info)

    episode_reward = 0.0
    episode_length = 0

    for step in range(num_steps):
        state = np.array(obs, dtype=np.float32)
        action, log_prob, value, entropy = policy_fn(state)

        # Step environment
        result = env.step(action)
        if len(result) == 4:
            next_obs, reward, done, info = result
            truncated = False
        else:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        episode_reward += reward
        episode_length += 1

        # Store transition
        mask = 0.0 if done else 1.0
        buffer.add(state, action, reward, done, value, log_prob,
                   mask=mask, next_state=np.array(next_obs, dtype=np.float32),
                   info=info)

        if done:
            if verbose:
                print(f"  Episode finished: reward={episode_reward:.2f}, "
                      f"length={episode_length}")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            episode_reward = 0.0
            episode_length = 0
        else:
            obs = next_obs

    return buffer


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation (GAE)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE) and discounted returns.

    Args:
        rewards: Array of shape (T,) with per-step rewards.
        values: Array of shape (T,) with value predictions V(s_t).
        dones: Array of shape (T,) with done flags (1.0 = terminal).
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        last_value: Value estimate for the state after the last transition
            (used for bootstrapping if the last transition is not terminal).

    Returns:
        advantages: Array of shape (T,) with GAE advantages.
        returns: Array of shape (T,) with discounted returns (advantages + values).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    last_value: float = 0.0,
) -> np.ndarray:
    """
    Compute simple discounted returns (Monte Carlo) without GAE.

    Args:
        rewards: Array of shape (T,) with per-step rewards.
        dones: Array of shape (T,) with done flags.
        gamma: Discount factor.
        last_value: Value estimate for bootstrapping.

    Returns:
        returns: Array of shape (T,) with discounted returns.
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    running_return = last_value
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return
    return returns


# ---------------------------------------------------------------------------
# State Serialization
# ---------------------------------------------------------------------------

def save_state_dict(state: Dict[str, Any], path: str) -> None:
    """
    Save a state dictionary to disk using pickle.

    Args:
        state: Dictionary containing environment state data.
        path: File path to save to.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state_dict(path: str) -> Dict[str, Any]:
    """
    Load a state dictionary from disk.

    Args:
        path: File path to load from.

    Returns:
        Dictionary containing environment state data.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# PyTorch Model Helpers
# ---------------------------------------------------------------------------

def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """
    Orthogonal weight initialization for linear layers.

    Args:
        layer: PyTorch module (Linear layer).
        gain: Scaling factor for the orthogonal matrix.
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


def to_tensor(x: Union[np.ndarray, torch.Tensor, List],
              device: str = "cpu") -> torch.Tensor:
    """
    Convert input to a PyTorch tensor on the specified device.

    Args:
        x: Input array, list, or tensor.
        device: Target device string.

    Returns:
        PyTorch tensor.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy array.

    Args:
        x: Input tensor or array.

    Returns:
        Numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Episode Runner (for evaluation)
# ---------------------------------------------------------------------------

def run_episode(
    env,
    policy_fn: Callable[[np.ndarray], Any],
    max_steps: int = 1000,
    deterministic: bool = True,
    render: bool = False,
) -> Dict[str, Any]:
    """
    Run a single episode and return statistics.

    Args:
        env: Gym-like environment.
        policy_fn: Function mapping state -> action.
        max_steps: Maximum steps before truncation.
        deterministic: Whether to use deterministic actions.
        render: Whether to render the environment.

    Returns:
        Dictionary with keys: total_reward, length, states, actions, rewards,
        dones, infos.
    """
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    total_reward = 0.0
    states = []
    actions = []
    rewards = []
    dones = []
    infos = []

    for step in range(max_steps):
        if render:
            env.render()

        state = np.array(obs, dtype=np.float32)
        action = policy_fn(state)

        result = env.step(action)
        if len(result) == 4:
            next_obs, reward, done, info = result
        else:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)

        total_reward += reward
        if done:
            break
        obs = next_obs

    return {
        "total_reward": total_reward,
        "length": len(states),
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "infos": infos,
    }


def evaluate_policy(
    env,
    policy_fn: Callable[[np.ndarray], Any],
    num_episodes: int = 10,
    max_steps: int = 1000,
    deterministic: bool = True,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: Gym-like environment.
        policy_fn: Function mapping state -> action.
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        deterministic: Whether to use deterministic actions.
        verbose: If True, print per-episode results.

    Returns:
        Dictionary with keys: mean_reward, std_reward, mean_length, std_length,
        all_rewards.
    """
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        result = run_episode(env, policy_fn, max_steps, deterministic)
        episode_rewards.append(result["total_reward"])
        episode_lengths.append(result["length"])
        if verbose:
            print(f"  Episode {ep + 1}: reward={result['total_reward']:.2f}, "
                  f"length={result['length']}")

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "all_rewards": episode_rewards,
    }


# ---------------------------------------------------------------------------
# Set Random Seeds
# ---------------------------------------------------------------------------

def set_seed(seed: int, env=None) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and environment.

    Args:
        seed: Integer seed value.
        env: Optional environment to seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)