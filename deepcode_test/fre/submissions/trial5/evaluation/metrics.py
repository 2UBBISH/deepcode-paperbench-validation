"""
Evaluation metrics for the FRE framework.

Provides normalized return computation, task-specific reward functions
for downstream evaluation, and aggregation utilities for reporting
results as in the paper (Table 1, Figures 5-6).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any


# ============================================================
# Normalized Return Computation
# ============================================================

def normalize_returns(
    returns: np.ndarray,
    min_return: float = 0.0,
    max_return: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Normalize returns to [0, 100] scale based on reference min/max.

    Formula: normalized = 100 * (return - min_return) / (max_return - min_return)

    Args:
        returns: Array of raw undiscounted returns.
        min_return: Reference minimum return (e.g., random policy).
        max_return: Reference maximum return (e.g., expert policy).
        clip: If True, clip normalized values to [0, 100].

    Returns:
        Normalized returns in [0, 100] range.
    """
    if max_return <= min_return:
        raise ValueError(f"max_return ({max_return}) must be > min_return ({min_return})")

    normalized = 100.0 * (returns - min_return) / (max_return - min_return)
    if clip:
        normalized = np.clip(normalized, 0.0, 100.0)
    return normalized


def compute_normalized_score(
    raw_returns: List[float],
    min_return: float,
    max_return: float,
) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of normalized returns.

    Args:
        raw_returns: List of raw episode returns.
        min_return: Reference minimum return.
        max_return: Reference maximum return.

    Returns:
        Tuple of (mean_normalized, std_normalized).
    """
    returns_arr = np.array(raw_returns)
    normalized = normalize_returns(returns_arr, min_return, max_return)
    return float(np.mean(normalized)), float(np.std(normalized))


# ============================================================
# Domain-Specific Normalization Constants
# ============================================================

# These values are based on the paper's benchmarks and typical D4RL/ExORL
# reference returns. They should be calibrated per domain.

DOMAIN_NORMALIZATION = {
    # AntMaze: random ~0, expert ~1 (sparse reward: -1 per step until goal)
    # Paper reports normalized returns 0-100; we use raw returns and normalize.
    # For AntMaze, typical max return is 0 (goal reached immediately) and min is
    # -1000 (max episode length). We scale accordingly.
    "antmaze": {
        "min_return": -1000.0,   # worst case: 1000 steps at -1 each
        "max_return": 0.0,       # best case: goal reached immediately
    },
    # ExORL Walker: dense rewards, typical range varies
    "exorl_walker": {
        "min_return": 0.0,
        "max_return": 1000.0,    # approximate expert return
    },
    # ExORL Cheetah: dense rewards
    "exorl_cheetah": {
        "min_return": 0.0,
        "max_return": 1000.0,    # approximate expert return
    },
    # Kitchen: sparse reward (0/1 per subtask), max 4 completed subtasks
    "kitchen": {
        "min_return": 0.0,
        "max_return": 4.0,       # all subtasks completed
    },
}


def get_domain_normalization(domain: str) -> Tuple[float, float]:
    """
    Get (min_return, max_return) for a given domain.

    Args:
        domain: One of 'antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen'.

    Returns:
        Tuple of (min_return, max_return).
    """
    if domain not in DOMAIN_NORMALIZATION:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_NORMALIZATION.keys())}")
    norm = DOMAIN_NORMALIZATION[domain]
    return norm["min_return"], norm["max_return"]


# ============================================================
# Task-Specific Evaluation Reward Functions
# ============================================================

class EvaluationTask:
    """
    Represents a downstream evaluation task with a reward function
    and metadata.
    """

    def __init__(
        self,
        name: str,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        description: str = "",
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
    ):
        """
        Args:
            name: Task name (e.g., 'goal-reaching', 'directional').
            reward_fn: Function mapping states -> scalar rewards.
            description: Human-readable task description.
            min_return: Task-specific min return override.
            max_return: Task-specific max return override.
        """
        self.name = name
        self.reward_fn = reward_fn
        self.description = description
        self.min_return = min_return
        self.max_return = max_return

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return self.reward_fn(states)

    def __repr__(self) -> str:
        return f"EvaluationTask({self.name})"


# ============================================================
# AntMaze Task Reward Functions
# ============================================================

def make_antmaze_goal_reaching_reward(goal: np.ndarray, threshold: float = 0.5) -> Callable:
    """
    Create a goal-reaching reward for AntMaze: -1 per step until within
    threshold of goal, then 0.

    Args:
        goal: Goal state (x, y) coordinates.
        threshold: Distance threshold for goal reached.

    Returns:
        Reward function: states -> rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # states shape: (batch, state_dim), first 2 dims are (x, y)
        distances = np.linalg.norm(states[:, :2] - goal[:2], axis=1)
        rewards = np.where(distances < threshold, 0.0, -1.0)
        return rewards
    return reward_fn


def make_antmaze_directional_reward(direction: np.ndarray) -> Callable:
    """
    Create a directional reward for AntMaze: reward = dot(position, direction).

    Args:
        direction: 2D direction vector (normalized).

    Returns:
        Reward function: states -> rewards.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    def reward_fn(states: np.ndarray) -> np.ndarray:
        return np.dot(states[:, :2], direction)
    return reward_fn


def make_antmaze_random_simplex_reward(
    state_dim: int,
    num_frequencies: int = 10,
    rng: Optional[np.random.RandomState] = None,
) -> Callable:
    """
    Create a random simplex (procedural noise) reward for AntMaze.
    Uses random Fourier features.

    Args:
        state_dim: State dimensionality.
        num_frequencies: Number of random frequencies.
        rng: Random state for reproducibility.

    Returns:
        Reward function: states -> rewards.
    """
    if rng is None:
        rng = np.random.RandomState()
    # Random frequencies and phases
    frequencies = rng.randn(num_frequencies, state_dim) * 2.0
    phases = rng.uniform(0, 2 * np.pi, size=num_frequencies)

    def reward_fn(states: np.ndarray) -> np.ndarray:
        # states: (batch, state_dim)
        projections = np.dot(states, frequencies.T)  # (batch, num_freq)
        rewards = np.mean(np.cos(projections + phases), axis=1)
        return rewards
    return reward_fn


def make_antmaze_path_reward(path_points: np.ndarray, threshold: float = 0.5) -> Callable:
    """
    Create a path-following reward for AntMaze.
    Reward = -distance to nearest point on path.

    Args:
        path_points: Array of (x, y) points defining the path.
        threshold: Distance threshold for being "on path".

    Returns:
        Reward function: states -> rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # states: (batch, state_dim)
        positions = states[:, :2]  # (batch, 2)
        # Compute distances to all path points
        # positions: (batch, 2), path_points: (num_path, 2)
        diffs = positions[:, np.newaxis, :] - path_points[np.newaxis, :, :]  # (batch, num_path, 2)
        distances = np.linalg.norm(diffs, axis=2)  # (batch, num_path)
        min_distances = np.min(distances, axis=1)  # (batch,)
        rewards = -min_distances
        return rewards
    return reward_fn


# ============================================================
# ExORL Task Reward Functions
# ============================================================

def make_exorl_goal_reaching_reward(goal: np.ndarray, threshold: float = 0.5) -> Callable:
    """
    Create a goal-reaching reward for ExORL (Walker/Cheetah).
    Reward = -distance to goal.

    Args:
        goal: Goal state vector.
        threshold: Distance threshold for goal reached.

    Returns:
        Reward function: states -> rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(states - goal, axis=1)
        rewards = -distances
        return rewards
    return reward_fn


def make_exorl_velocity_reward(target_velocity: float, velocity_idx: int = 0) -> Callable:
    """
    Create a velocity reward for ExORL.
    Reward = -abs(velocity - target_velocity).

    Args:
        target_velocity: Target velocity value.
        velocity_idx: Index of velocity component in state.

    Returns:
        Reward function: states -> rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        velocities = states[:, velocity_idx]
        rewards = -np.abs(velocities - target_velocity)
        return rewards
    return reward_fn


# ============================================================
# Kitchen Task Reward Functions
# ============================================================

def make_kitchen_subtask_reward(
    subtask_idx: int,
    completion_threshold: float = 0.5,
) -> Callable:
    """
    Create a reward for a specific Kitchen subtask.
    Reward = 1 if subtask completed, 0 otherwise.

    In the Kitchen environment, the state contains object positions
    that indicate subtask completion. This is a simplified version;
    for exact reproduction, use the environment's built-in task
    completion detection.

    Args:
        subtask_idx: Index of the subtask (0-6 for 7 tasks).
        completion_threshold: Threshold for considering subtask complete.

    Returns:
        Reward function: states -> rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # Kitchen state includes object positions; subtask completion
        # is typically indicated by specific state features.
        # This is a placeholder; actual implementation depends on
        # the specific Kitchen environment state representation.
        # For D4RL kitchen-complete-v0, the state is 30-dim with
        # object positions at specific indices.
        #
        # Simplified: use a feature based on state norm in relevant dims.
        # In practice, we use the environment's task completion API.
        rewards = np.zeros(len(states))
        return rewards
    return reward_fn


# ============================================================
# Evaluation Result Aggregation
# ============================================================

class EvaluationResult:
    """
    Stores and aggregates evaluation results across tasks and seeds.
    """

    def __init__(self):
        self.task_results: Dict[str, List[float]] = {}  # task_name -> list of mean returns per seed
        self.task_raw_returns: Dict[str, List[List[float]]] = {}  # task_name -> list of episode returns per seed

    def add_result(
        self,
        task_name: str,
        episode_returns: List[float],
        seed: int = 0,
    ):
        """
        Add evaluation result for a task.

        Args:
            task_name: Name of the evaluation task.
            episode_returns: List of raw returns for each episode.
            seed: Random seed identifier.
        """
        if task_name not in self.task_results:
            self.task_results[task_name] = []
            self.task_raw_returns[task_name] = []
        self.task_results[task_name].append(float(np.mean(episode_returns)))
        self.task_raw_returns[task_name].append(episode_returns)

    def get_task_stats(self, task_name: str) -> Tuple[float, float]:
        """
        Get mean and std of mean returns across seeds for a task.

        Args:
            task_name: Task name.

        Returns:
            Tuple of (mean_of_means, std_of_means).
        """
        means = self.task_results[task_name]
        return float(np.mean(means)), float(np.std(means))

    def get_all_task_stats(self) -> Dict[str, Tuple[float, float]]:
        """
        Get stats for all tasks.

        Returns:
            Dict mapping task_name -> (mean, std).
        """
        return {name: self.get_task_stats(name) for name in self.task_results}

    def get_overall_average(self) -> Tuple[float, float]:
        """
        Compute overall average across all tasks (as in paper Table 1).

        Returns:
            Tuple of (overall_mean, overall_std).
        """
        all_means = []
        for task_name in self.task_results:
            mean_per_task = np.mean(self.task_results[task_name])
            all_means.append(mean_per_task)
        return float(np.mean(all_means)), float(np.std(all_means))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a serializable dictionary.

        Returns:
            Dict with task statistics.
        """
        result = {}
        for task_name in self.task_results:
            mean, std = self.get_task_stats(task_name)
            result[task_name] = {
                "mean": mean,
                "std": std,
                "num_seeds": len(self.task_results[task_name]),
                "num_episodes_per_seed": len(self.task_raw_returns[task_name][0])
                if self.task_raw_returns[task_name] else 0,
            }
        overall_mean, overall_std = self.get_overall_average()
        result["overall"] = {"mean": overall_mean, "std": overall_std}
        return result

    def __repr__(self) -> str:
        return f"EvaluationResult({len(self.task_results)} tasks)"


# ============================================================
# Utility: Compute Returns from Episode
# ============================================================

def compute_episode_return(
    rewards: List[float],
    discount: float = 1.0,
) -> float:
    """
    Compute (possibly discounted) return from a list of per-step rewards.

    For evaluation, the paper uses undiscounted returns (discount=1.0).

    Args:
        rewards: List of per-timestep rewards.
        discount: Discount factor (1.0 for undiscounted).

    Returns:
        Total return.
    """
    if discount == 1.0:
        return float(np.sum(rewards))
    else:
        discounted = 0.0
        for r in reversed(rewards):
            discounted = r + discount * discounted
        return float(discounted)


def aggregate_seed_results(
    all_seed_results: List[Dict[str, List[float]]],
    normalize_fn: Optional[Callable] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate results across multiple seeds.

    Args:
        all_seed_results: List of dicts mapping task_name -> list of episode returns.
        normalize_fn: Optional function to normalize returns before aggregation.

    Returns:
        Dict mapping task_name -> (mean_normalized, std_normalized).
    """
    task_means: Dict[str, List[float]] = {}

    for seed_result in all_seed_results:
        for task_name, episode_returns in seed_result.items():
            if task_name not in task_means:
                task_means[task_name] = []
            mean_return = np.mean(episode_returns)
            if normalize_fn is not None:
                mean_return = normalize_fn(mean_return)
            task_means[task_name].append(mean_return)

    result = {}
    for task_name, means in task_means.items():
        result[task_name] = (float(np.mean(means)), float(np.std(means)))

    return result