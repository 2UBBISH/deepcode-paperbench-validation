"""Utility functions for the RICE implementation.

Includes:
- State restoration helpers for simulator-based environments
- Reward computation helpers
- Environment wrapper utilities
- Action space sampling helpers
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import torch


def make_random_action_sampler(action_dim: int, discrete: bool = False) -> Callable[[], np.ndarray]:
    """
    Create a function that samples random actions from the action space.

    Args:
        action_dim: Dimension/num of actions.
        discrete: Whether the action space is discrete.

    Returns:
        Function () -> random_action.
    """
    if discrete:
        return lambda: np.random.randint(0, action_dim)
    else:
        return lambda: np.random.uniform(-1.0, 1.0, size=(action_dim,))


def compute_cumulative_return(rewards: List[float], gamma: float = 0.99) -> float:
    """
    Compute discounted cumulative return.

    Args:
        rewards: List of per-step rewards.
        gamma: Discount factor.

    Returns:
        Discounted cumulative return.
    """
    ret = 0.0
    for r in reversed(rewards):
        ret = r + gamma * ret
    return ret


def compute_episode_stats(
    rewards: List[float],
    gamma: float = 0.99,
) -> Dict[str, float]:
    """
    Compute statistics for an episode.

    Args:
        rewards: List of per-step rewards.
        gamma: Discount factor.

    Returns:
        Dict with total_reward, discounted_return, episode_length.
    """
    return {
        "total_reward": sum(rewards),
        "discounted_return": compute_cumulative_return(rewards, gamma),
        "episode_length": len(rewards),
    }


def run_episode(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    max_steps: int = 1000,
    deterministic: bool = False,
    record_states: bool = False,
) -> Dict[str, Any]:
    """
    Run a single episode using the given policy.

    Args:
        policy_fn: Function (state) -> action.
        env_reset_fn: Function () -> initial_state.
        env_step_fn: Function (action) -> (next_state, reward, done, info).
        max_steps: Maximum steps per episode.
        deterministic: Whether to use deterministic actions.
        record_states: Whether to record all states visited.

    Returns:
        Dict with states, actions, rewards, total_reward, length, done.
    """
    states = []
    actions = []
    rewards = []

    state = env_reset_fn()
    done = False
    step = 0

    while not done and step < max_steps:
        action = policy_fn(state)
        next_state, reward, done, info = env_step_fn(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        step += 1

    result = {
        "states": np.array(states, dtype=np.float32) if record_states else None,
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "total_reward": float(sum(rewards)),
        "length": step,
        "done": done,
    }
    return result


def evaluate_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    n_episodes: int = 10,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a policy over multiple episodes.

    Args:
        policy_fn: Function (state) -> action.
        env_reset_fn: Function () -> initial_state.
        env_step_fn: Function (action) -> (next_state, reward, done, info).
        n_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        deterministic: Whether to use deterministic actions.

    Returns:
        Dict with mean_reward, std_reward, mean_length.
    """
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        result = run_episode(
            policy_fn=policy_fn,
            env_reset_fn=env_reset_fn,
            env_step_fn=env_step_fn,
            max_steps=max_steps,
            deterministic=deterministic,
        )
        episode_rewards.append(result["total_reward"])
        episode_lengths.append(result["length"])

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "n_episodes": n_episodes,
    }


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
