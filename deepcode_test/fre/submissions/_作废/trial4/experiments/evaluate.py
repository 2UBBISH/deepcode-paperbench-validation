"""
Zero-Shot Evaluation Script for Functional Reward Encodings (FRE).

This script evaluates a trained FRE agent on downstream tasks without any
fine-tuning. Given a checkpoint, it:
  1. Loads the trained encoder (frozen) and IQL policy.
  2. For each evaluation task, encodes the task's reward function using
     K=32 (state, reward) pairs sampled from the dataset.
  3. Rolls out the conditioned policy in the environment for N episodes.
  4. Reports mean and standard deviation of returns.

Supports domains: AntMaze, ExORL (Walker, Cheetah), Kitchen.
"""

import argparse
import yaml
import json
import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Callable, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fre.utils import (
    load_dataset,
    set_seed,
    get_device,
    StateNormalizer,
    ReplayBuffer,
)
from fre.fre_agent import FREAgent, create_fre_agent
from fre.reward_prior import RewardFunction, GoalReachingReward


# ==============================================================================
# Evaluation Task Definitions
# ==============================================================================

class EvalTask:
    """Represents a single zero-shot evaluation task.

    Attributes:
        name: Human-readable task name (e.g., "goal-reaching-0").
        env_name: Gym environment identifier.
        reward_fn: Callable that maps states (np.ndarray) to scalar rewards.
        num_episodes: Number of rollout episodes for this task.
        max_episode_steps: Maximum steps per episode.
        goal_state: Optional goal state for goal-reaching tasks (for logging).
    """

    def __init__(
        self,
        name: str,
        env_name: str,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        num_episodes: int = 20,
        max_episode_steps: int = 1000,
        goal_state: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.env_name = env_name
        self.reward_fn = reward_fn
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.goal_state = goal_state


class DirectionalReward(RewardFunction):
    """Reward based on velocity in a specified direction.

    η(s) = dot(velocity, direction)

    Args:
        direction: Unit vector (or any vector) in the desired direction.
        vel_start_idx: Start index of velocity components in the state vector.
        vel_dim: Number of velocity dimensions (typically 2 for planar).
    """

    def __init__(self, direction: np.ndarray, vel_start_idx: int = 13, vel_dim: int = 2):
        super().__init__()
        self.direction = direction / (np.linalg.norm(direction) + 1e-8)
        self.vel_start_idx = vel_start_idx
        self.vel_dim = vel_dim

    def __call__(self, states: np.ndarray) -> np.ndarray:
        vel = states[..., self.vel_start_idx:self.vel_start_idx + self.vel_dim]
        return np.dot(vel, self.direction)

    @property
    def reward_type(self) -> str:
        return "directional"


class RandomSimplexReward(RewardFunction):
    """Procedural noise reward function based on random sine/cosine features.

    η(s) = Σ_i w_i * sin(dot(freq_i, s) + phase_i)

    This approximates the "random-simplex" tasks mentioned in the paper.
    """

    def __init__(self, state_dim: int, num_features: int = 32, seed: int = 0):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.frequencies = rng.randn(num_features, state_dim) * 0.5
        self.phases = rng.uniform(0, 2 * np.pi, size=num_features)
        self.weights = rng.randn(num_features) / np.sqrt(num_features)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        # states shape: (..., state_dim)
        raw = np.dot(states, self.frequencies.T) + self.phases  # (..., num_features)
        return np.dot(np.sin(raw), self.weights)

    @property
    def reward_type(self) -> str:
        return "random_simplex"


class PathReward(RewardFunction):
    """Reward for following a sequence of subgoals (path task).

    The agent must reach subgoals in order. Reward is negative distance
    to the current subgoal. When within epsilon of the current subgoal,
    advance to the next one. Once all subgoals are reached, reward is 0.

    Args:
        subgoals: List of goal states (np.ndarray each).
        epsilon: Distance threshold for reaching a subgoal.
        state_slice: Optional slice to use for distance computation
                     (e.g., first 2 dims for XY position).
    """

    def __init__(
        self,
        subgoals: List[np.ndarray],
        epsilon: float = 0.5,
        state_slice: Optional[slice] = None,
    ):
        super().__init__()
        self.subgoals = subgoals
        self.epsilon = epsilon
        self.state_slice = state_slice if state_slice is not None else slice(None)
        # Internal state per episode (reset externally)
        self._current_idx = 0

    def reset(self):
        """Reset the path progress (call before each episode)."""
        self._current_idx = 0

    def __call__(self, states: np.ndarray) -> np.ndarray:
        # For encoding, we just compute distance to the first subgoal
        # (or average over all subgoals). For rollout, we need stateful tracking.
        # Here we provide a stateless version for encoding: distance to first subgoal.
        if self._current_idx >= len(self.subgoals):
            return np.zeros(states.shape[:-1] if states.ndim > 1 else ())
        goal = self.subgoals[self._current_idx]
        diff = states[..., self.state_slice] - goal[self.state_slice]
        dist = np.linalg.norm(diff, axis=-1)
        return -dist

    def step_update(self, state: np.ndarray) -> bool:
        """Check if current subgoal reached; if so, advance. Returns True if all done."""
        if self._current_idx >= len(self.subgoals):
            return True
        goal = self.subgoals[self._current_idx]
        diff = state[self.state_slice] - goal[self.state_slice]
        dist = np.linalg.norm(diff)
        if dist < self.epsilon:
            self._current_idx += 1
        return self._current_idx >= len(self.subgoals)

    @property
    def reward_type(self) -> str:
        return "path"


class KitchenSubtaskReward(RewardFunction):
    """Reward for a specific kitchen subtask.

    Returns 1.0 if the subtask is completed (based on state), else 0.0.

    The completion detection uses heuristic thresholds on object positions
    and joint angles from the kitchen environment state.

    Args:
        subtask_name: One of the 7 kitchen subtasks.
        state_dim: Full state dimension (used for index mapping).
    """

    # Known subtask indices and thresholds for the D4RL kitchen environment.
    # The state vector (30-dim for kitchen-complete-v0) contains:
    #   [0]: microwave x, [1]: microwave y (or z?)
    #   ... object positions, then joint angles.
    # We use approximate heuristics based on the D4RL kitchen task definitions.
    SUBTASK_CONFIGS = {
        "microwave": {"obj_idx": 0, "threshold": 0.3, "target": 0.0},
        "kettle": {"obj_idx": 4, "threshold": 0.3, "target": 0.0},
        "light": {"obj_idx": 8, "threshold": 0.3, "target": 0.0},
        "slide_cabinet": {"obj_idx": 12, "threshold": 0.3, "target": 0.0},
        "hinge_cabinet": {"obj_idx": 16, "threshold": 0.3, "target": 0.0},
        "bottom_burner": {"obj_idx": 20, "threshold": 0.3, "target": 0.0},
        "top_burner": {"obj_idx": 24, "threshold": 0.3, "target": 0.0},
    }

    def __init__(self, subtask_name: str, state_dim: int = 30):
        super().__init__()
        self.subtask_name = subtask_name
        if subtask_name not in self.SUBTASK_CONFIGS:
            raise ValueError(f"Unknown subtask: {subtask_name}. "
                             f"Choose from: {list(self.SUBTASK_CONFIGS.keys())}")
        self.config = self.SUBTASK_CONFIGS[subtask_name]

    def __call__(self, states: np.ndarray) -> np.ndarray:
        # Heuristic: check if the object's position is near the target.
        idx = self.config["obj_idx"]
        target = self.config["target"]
        threshold = self.config["threshold"]
        # Assume state has object positions at given indices
        if states.ndim == 1:
            val = states[idx]
        else:
            val = states[..., idx]
        # Simple threshold: if value is close to target, subtask completed
        completed = np.abs(val - target) < threshold
        return completed.astype(np.float32)

    @property
    def reward_type(self) -> str:
        return f"kitchen_{self.subtask_name}"


# ==============================================================================
# Task Factory Functions
# ==============================================================================

def get_antmaze_tasks(
    replay_buffer: ReplayBuffer,
    config: dict,
    num_episodes: int = 20,
) -> List[EvalTask]:
    """Build evaluation tasks for the AntMaze domain.

    Tasks:
      - Goal-reaching: N random goals sampled from dataset states.
      - Directional: 4 cardinal directions (N, S, E, W).
      - Random-simplex: 3 random procedural noise functions.
      - Path: edges, loop, center (if goal states specified in config).
    """
    tasks = []
    env_name = config.get("env_name", "antmaze-large-diverse-v2")
    max_steps = config.get("evaluation", {}).get("max_episode_steps", 1000)
    state_dim = replay_buffer.states.shape[1]
    dataset_states = replay_buffer.states

    # --- Goal-reaching tasks ---
    num_goals = config.get("evaluation", {}).get("num_goal_tasks", 10)
    goal_epsilon = config.get("evaluation", {}).get("goal_epsilon", 0.5)
    rng = np.random.RandomState(config.get("evaluation", {}).get("goal_seed", 42))
    goal_indices = rng.choice(len(dataset_states), size=num_goals, replace=False)
    for i, idx in enumerate(goal_indices):
        goal_state = dataset_states[idx].copy()
        reward_fn = GoalReachingReward(goal_state, epsilon=goal_epsilon)
        tasks.append(EvalTask(
            name=f"goal-reaching-{i}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
            goal_state=goal_state,
        ))

    # --- Directional tasks ---
    directions = {
        "north": np.array([0.0, 1.0]),
        "south": np.array([0.0, -1.0]),
        "east": np.array([1.0, 0.0]),
        "west": np.array([-1.0, 0.0]),
    }
    # Velocity indices for AntMaze (torso velocity in XY plane)
    vel_start = config.get("evaluation", {}).get("vel_start_idx", 13)
    for dir_name, dir_vec in directions.items():
        reward_fn = DirectionalReward(dir_vec, vel_start_idx=vel_start, vel_dim=2)
        tasks.append(EvalTask(
            name=f"directional-{dir_name}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
        ))

    # --- Random-simplex tasks ---
    num_simplex = config.get("evaluation", {}).get("num_simplex_tasks", 3)
    for i in range(num_simplex):
        reward_fn = RandomSimplexReward(state_dim, num_features=32, seed=100 + i)
        tasks.append(EvalTask(
            name=f"random-simplex-{i}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
        ))

    # --- Path tasks ---
    path_tasks_cfg = config.get("evaluation", {}).get("path_tasks", [])
    for path_cfg in path_tasks_cfg:
        subgoals = [np.array(g) for g in path_cfg["subgoals"]]
        epsilon = path_cfg.get("epsilon", 0.5)
        state_slice = slice(0, 2)  # XY only for AntMaze
        reward_fn = PathReward(subgoals, epsilon=epsilon, state_slice=state_slice)
        tasks.append(EvalTask(
            name=f"path-{path_cfg['name']}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
        ))

    return tasks


def get_exorl_walker_tasks(
    replay_buffer: ReplayBuffer,
    config: dict,
    num_episodes: int = 20,
) -> List[EvalTask]:
    """Build evaluation tasks for ExORL Walker domain.

    Tasks:
      - Goal-reaching: random states from dataset.
      - Velocity: forward (+x) and backward (-x).
    """
    tasks = []
    env_name = config.get("env_name", "walker-walk")  # DMControl walker
    max_steps = config.get("evaluation", {}).get("max_episode_steps", 1000)
    state_dim = replay_buffer.states.shape[1]
    dataset_states = replay_buffer.states

    # Goal-reaching
    num_goals = config.get("evaluation", {}).get("num_goal_tasks", 10)
    goal_epsilon = config.get("evaluation", {}).get("goal_epsilon", 0.5)
    rng = np.random.RandomState(config.get("evaluation", {}).get("goal_seed", 42))
    goal_indices = rng.choice(len(dataset_states), size=num_goals, replace=False)
    for i, idx in enumerate(goal_indices):
        goal_state = dataset_states[idx].copy()
        reward_fn = GoalReachingReward(goal_state, epsilon=goal_epsilon)
        tasks.append(EvalTask(
            name=f"goal-reaching-{i}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
            goal_state=goal_state,
        ))

    # Velocity tasks: forward/backward based on x-velocity
    # For walker, velocity is typically the first derivative of position.
    vel_start = config.get("evaluation", {}).get("vel_start_idx", 0)
    for dir_name, sign in [("forward", 1.0), ("backward", -1.0)]:
        direction = np.array([sign])
        reward_fn = DirectionalReward(direction, vel_start_idx=vel_start, vel_dim=1)
        tasks.append(EvalTask(
            name=f"velocity-{dir_name}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
        ))

    return tasks


def get_exorl_cheetah_tasks(
    replay_buffer: ReplayBuffer,
    config: dict,
    num_episodes: int = 20,
) -> List[EvalTask]:
    """Build evaluation tasks for ExORL Cheetah domain.

    Tasks:
      - Goal-reaching: random states from dataset.
      - Velocity: forward (+x) and backward (-x).
    """
    tasks = []
    env_name = config.get("env_name", "cheetah-run")  # DMControl cheetah
    max_steps = config.get("evaluation", {}).get("max_episode_steps", 1000)
    state_dim = replay_buffer.states.shape[1]
    dataset_states = replay_buffer.states

    # Goal-reaching
    num_goals = config.get("evaluation", {}).get("num_goal_tasks", 10)
    goal_epsilon = config.get("evaluation", {}).get("goal_epsilon", 0.5)
    rng = np.random.RandomState(config.get("evaluation", {}).get("goal_seed", 42))
    goal_indices = rng.choice(len(dataset_states), size=num_goals, replace=False)
    for i, idx in enumerate(goal_indices):
        goal_state = dataset_states[idx].copy()
        reward_fn = GoalReachingReward(goal_state, epsilon=goal_epsilon)
        tasks.append(EvalTask(
            name=f"goal-reaching-{i}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
            goal_state=goal_state,
        ))

    # Velocity tasks
    vel_start = config.get("evaluation", {}).get("vel_start_idx", 0)
    for dir_name, sign in [("forward", 1.0), ("backward", -1.0)]:
        direction = np.array([sign])
        reward_fn = DirectionalReward(direction, vel_start_idx=vel_start, vel_dim=1)
        tasks.append(EvalTask(
            name=f"velocity-{dir_name}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
        ))

    return tasks


def get_kitchen_tasks(
    replay_buffer: ReplayBuffer,
    config: dict,
    num_episodes: int = 20,
) -> List[EvalTask]:
    """Build evaluation tasks for the Kitchen domain.

    Tasks: 7 subtasks (microwave, kettle, light, slide_cabinet,
            hinge_cabinet, bottom_burner, top_burner).
    """
    tasks = []
    env_name = config.get("env_name", "kitchen-complete-v0")
    max_steps = config.get("evaluation", {}).get("max_episode_steps", 280)
    state_dim = replay_buffer.states.shape[1]

    subtask_names = [
        "microwave", "kettle", "light", "slide_cabinet",
        "hinge_cabinet", "bottom_burner", "top_burner",
    ]
    for name in subtask_names:
        reward_fn = KitchenSubtaskReward(name, state_dim=state_dim)
        tasks.append(EvalTask(
            name=f"subtask-{name}",
            env_name=env_name,
            reward_fn=reward_fn,
            num_episodes=num_episodes,
            max_episode_steps=max_steps,
        ))

    return tasks


def get_eval_tasks(
    domain: str,
    replay_buffer: ReplayBuffer,
    config: dict,
    num_episodes: int = 20,
) -> List[EvalTask]:
    """Dispatch to the appropriate task factory based on domain."""
    domain_lower = domain.lower()
    if "antmaze" in domain_lower:
        return get_antmaze_tasks(replay_buffer, config, num_episodes)
    elif "walker" in domain_lower:
        return get_exorl_walker_tasks(replay_buffer, config, num_episodes)
    elif "cheetah" in domain_lower:
        return get_exorl_cheetah_tasks(replay_buffer, config, num_episodes)
    elif "kitchen" in domain_lower:
        return get_kitchen_tasks(replay_buffer, config, num_episodes)
    else:
        raise ValueError(f"Unknown domain: {domain}. "
                         f"Supported: antmaze, walker, cheetah, kitchen")


# ==============================================================================
# Environment Helpers
# ==============================================================================

def make_env(env_name: str, **kwargs) -> Any:
    """Create a Gym environment with fallback handling.

    Attempts to create via gym.make. If the environment is not registered,
    tries to import d4rl or other packages that register the environment.
    """
    try:
        import gym
        env = gym.make(env_name, **kwargs)
        return env
    except Exception as e:
        # Try importing d4rl which registers D4RL environments
        try:
            import d4rl
            import gym
            env = gym.make(env_name, **kwargs)
            return env
        except ImportError:
            raise RuntimeError(
                f"Cannot create environment '{env_name}'. "
                f"Make sure the required package (d4rl, dm_control, etc.) is installed. "
                f"Original error: {e}"
            )


# ==============================================================================
# Evaluation Loop
# ==============================================================================

def evaluate_policy(
    env: Any,
    agent: FREAgent,
    z: torch.Tensor,
    normalizer: StateNormalizer,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Roll out the conditioned policy and collect returns.

    Args:
        env: Gym environment.
        agent: Trained FREAgent (in eval mode).
        z: Latent vector encoding the task reward function.
        normalizer: StateNormalizer used during training.
        num_episodes: Number of episodes to evaluate.
        max_episode_steps: Maximum steps per episode.
        deterministic: Whether to use deterministic policy (mean) or sample.
        render: Whether to render the environment.
        verbose: Print per-episode returns.

    Returns:
        Dict with keys: mean_return, std_return, returns (list), success_rate.
    """
    returns = []
    successes = []

    for ep in range(num_episodes):
        state = env.reset()
        # Handle gym API variations
        if isinstance(state, tuple):
            state = state[0]  # New gym API returns (obs, info)
        state = np.asarray(state, dtype=np.float32)

        done = False
        truncated = False
        ep_return = 0.0
        step = 0

        while not (done or truncated) and step < max_episode_steps:
            # Normalize state
            norm_state = normalizer.normalize(state)
            # Get action from policy
            action = agent.get_action(norm_state, z, deterministic=deterministic)
            # Step environment
            result = env.step(action)
            # Handle gym API variations
            if len(result) == 4:
                next_state, reward, done, info = result
                truncated = False
            else:
                next_state, reward, done, truncated, info = result

            next_state = np.asarray(next_state, dtype=np.float32)
            ep_return += float(reward)
            state = next_state
            step += 1

            if render:
                env.render()

        returns.append(ep_return)
        # Check success if info contains it
        if "success" in info:
            successes.append(float(info["success"]))
        elif hasattr(env, 'get_success'):
            successes.append(float(env.get_success()))
        else:
            # For goal-reaching, success = final distance < epsilon
            successes.append(0.0)

        if verbose:
            print(f"  Episode {ep + 1}/{num_episodes}: return = {ep_return:.3f}")

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    success_rate = float(np.mean(successes)) if successes else 0.0

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "returns": [float(r) for r in returns],
        "success_rate": success_rate,
    }


def evaluate_task(
    task: EvalTask,
    agent: FREAgent,
    normalizer: StateNormalizer,
    config: dict,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a single task: encode reward → rollout policy.

    Args:
        task: EvalTask with reward function and environment name.
        agent: Trained FREAgent.
        normalizer: StateNormalizer.
        config: Full configuration dict.
        device: Torch device.

    Returns:
        Dict with evaluation results.
    """
    K_encode = config.get("evaluation", {}).get("K_encode", 32)
    deterministic = config.get("evaluation", {}).get("deterministic", True)

    # Encode the reward function into latent z
    z = agent.encode_reward(
        reward_fn=task.reward_fn,
        K=K_encode,
        deterministic=deterministic,
    )
    z = z.to(device)

    # Create environment
    env = make_env(task.env_name)

    # Evaluate
    result = evaluate_policy(
        env=env,
        agent=agent,
        z=z,
        normalizer=normalizer,
        num_episodes=task.num_episodes,
        max_episode_steps=task.max_episode_steps,
        deterministic=deterministic,
        render=False,
        verbose=True,
    )

    env.close()
    return result


# ==============================================================================
# Main Entry Point
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of FRE agents on downstream tasks."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file (e.g., experiments/configs/antmaze.yaml)."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the FRE agent checkpoint (.pt file)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20,
        help="Number of rollout episodes per task."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use: 'auto', 'cuda', 'cpu'."
    )
    parser.add_argument(
        "--tasks", type=str, nargs="*", default=None,
        help="Specific task names to evaluate (default: all tasks for the domain)."
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed evaluation progress."
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation routine."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set random seed
    set_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset (needed for state normalization and encoding states)
    dataset_name = config.get("dataset", config.get("domain", "antmaze"))
    print(f"Loading dataset: {dataset_name}")
    replay_buffer = load_dataset(dataset_name)

    # Compute state normalization statistics
    state_dim = replay_buffer.states.shape[1]
    normalizer = StateNormalizer(state_dim=state_dim)
    normalizer.update(replay_buffer.states)
    normalizer.freeze()
    print(f"State normalizer computed: mean shape={normalizer.mean.shape}, "
          f"std shape={normalizer.std.shape}")

    # Determine action dimension from dataset
    action_dim = replay_buffer.actions.shape[1]

    # Create FRE agent with the same architecture as training
    # We need a dummy reward prior (not used during evaluation)
    from fre.reward_prior import RewardPrior
    dummy_prior = RewardPrior(
        dataset_states=replay_buffer.states,
        state_dim=state_dim,
    )

    # Update config with inferred dimensions
    config["state_dim"] = state_dim
    config["action_dim"] = action_dim

    agent = create_fre_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        reward_prior=dummy_prior,
        config=config,
    )
    agent.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    agent.load_checkpoint(args.checkpoint, load_optimizer=False)
    agent.eval()
    print("Checkpoint loaded successfully.")

    # Build evaluation tasks
    domain = config.get("domain", "antmaze")
    all_tasks = get_eval_tasks(
        domain=domain,
        replay_buffer=replay_buffer,
        config=config,
        num_episodes=args.num_episodes,
    )
    print(f"Generated {len(all_tasks)} evaluation tasks for domain '{domain}'.")

    # Filter tasks if specific names requested
    if args.tasks:
        task_name_set = set(args.tasks)
        tasks = [t for t in all_tasks if t.name in task_name_set]
        if len(tasks) == 0:
            print(f"Warning: No tasks matched the filter {args.tasks}. "
                  f"Available tasks: {[t.name for t in all_tasks]}")
            tasks = all_tasks
        else:
            print(f"Filtered to {len(tasks)} tasks: {[t.name for t in tasks]}")
    else:
        tasks = all_tasks

    # Evaluate each task
    results = {}
    total_start = time.time()

    for i, task in enumerate(tasks):
        print(f"\n{'=' * 60}")
        print(f"Task {i + 1}/{len(tasks)}: {task.name}")
        print(f"{'=' * 60}")

        task_start = time.time()
        try:
            eval_result = evaluate_task(
                task=task,
                agent=agent,
                normalizer=normalizer,
                config=config,
                device=device,
            )
            elapsed = time.time() - task_start
            print(f"  Mean return: {eval_result['mean_return']:.3f} "
                  f"± {eval_result['std_return']:.3f}")
            print(f"  Success rate: {eval_result['success_rate']:.3f}")
            print(f"  Time: {elapsed:.1f}s")

            results[task.name] = {
                "mean_return": eval_result["mean_return"],
                "std_return": eval_result["std_return"],
                "success_rate": eval_result["success_rate"],
                "returns": eval_result["returns"],
            }
        except Exception as e:
            print(f"  ERROR evaluating task '{task.name}': {e}")
            import traceback
            traceback.print_exc()
            results[task.name] = {
                "mean_return": None,
                "std_return": None,
                "success_rate": None,
                "error": str(e),
            }

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete. Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Compute aggregate statistics
    valid_returns = [r["mean_return"] for r in results.values()
                     if r["mean_return"] is not None]
    if valid_returns:
        overall_mean = np.mean(valid_returns)
        overall_std = np.std(valid_returns)
        print(f"Overall mean return across {len(valid_returns)} tasks: "
              f"{overall_mean:.3f} ± {overall_std:.3f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"eval_results_{domain}_seed{args.seed}.json"
    )
    output_data = {
        "domain": domain,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "config": args.config,
        "overall_mean": float(np.mean(valid_returns)) if valid_returns else None,
        "overall_std": float(np.std(valid_returns)) if valid_returns else None,
        "tasks": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")

    # Also save a summary CSV
    csv_path = os.path.join(
        args.output_dir,
        f"eval_summary_{domain}_seed{args.seed}.csv"
    )
    with open(csv_path, "w") as f:
        f.write("task,mean_return,std_return,success_rate\n")
        for task_name, task_result in results.items():
            mean_r = task_result.get("mean_return", "N/A")
            std_r = task_result.get("std_return", "N/A")
            succ = task_result.get("success_rate", "N/A")
            f.write(f"{task_name},{mean_r},{std_r},{succ}\n")
    print(f"Summary CSV saved to {csv_path}")


if __name__ == "__main__":
    main()