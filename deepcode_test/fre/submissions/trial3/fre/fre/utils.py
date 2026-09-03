"""
Utility functions for Functional Reward Encodings (FRE).

Provides:
- Logging utilities (TensorBoard, WandB, CSV)
- Evaluation helpers (running episodes, computing statistics)
- Visualization utilities (reward heatmaps, trajectory plots)
- Configuration loading/saving
- Metric tracking and aggregation
"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import torch
import torch.nn as nn

# ==============================================================================
# Logging Utilities
# ==============================================================================

class Logger:
    """
    Unified logger supporting console output, CSV logging, and optional
    TensorBoard / WandB integration.
    """
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "fre",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        verbose: bool = True,
    ):
        self.log_dir = log_dir
        self.verbose = verbose
        os.makedirs(log_dir, exist_ok=True)

        # CSV logger
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.csv_file = None
        self.csv_written_header = False

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
            except ImportError:
                logging.warning("TensorBoard not available; skipping.")

        # WandB
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=wandb_config,
                    dir=log_dir,
                )
            except ImportError:
                logging.warning("Weights & Biases not available; skipping.")

        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.step_count = 0

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a dictionary of scalar metrics."""
        if step is None:
            step = self.step_count
        self.step_count = step + 1

        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        # Console
        if self.verbose:
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"[Step {step}] {metric_str}")

        # CSV
        self._log_csv(metrics, step)

        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

        # WandB
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _log_csv(self, metrics: Dict[str, float], step: int):
        """Append metrics to CSV file."""
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w")

        if not self.csv_written_header:
            header = ["step"] + sorted(metrics.keys())
            self.csv_file.write(",".join(header) + "\n")
            self.csv_written_header = True

        row = [str(step)] + [str(metrics.get(k, "")) for k in sorted(metrics.keys())]
        self.csv_file.write(",".join(row) + "\n")
        self.csv_file.flush()

    def get_history(self, key: str) -> List[float]:
        """Get the history of a specific metric."""
        return self.metrics_history.get(key, [])

    def save_metrics_summary(self, filepath: Optional[str] = None):
        """Save a JSON summary of all metrics (mean, std, min, max)."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, "metrics_summary.json")

        summary = {}
        for key, values in self.metrics_history.items():
            if len(values) > 0:
                arr = np.array(values)
                summary[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "last": float(arr[-1]),
                    "count": len(values),
                }

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self):
        """Clean up logging resources."""
        if self.csv_file is not None:
            self.csv_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


# ==============================================================================
# Evaluation Helpers
# ==============================================================================

def evaluate_policy_on_env(
    env,
    policy_fn,
    z: np.ndarray,
    num_episodes: int = 20,
    max_steps: int = 1000,
    deterministic: bool = True,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate a z-conditioned policy on an environment.

    Args:
        env: Gym environment (will be wrapped to reset properly).
        policy_fn: Function (state, z) -> action.
        z: Latent encoding vector (np.ndarray of shape (latent_dim,)).
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        deterministic: Whether to use deterministic policy.
        render: Whether to render the environment.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: 'returns', 'lengths', 'success_rate', 'mean_return', 'std_return'.
    """
    if seed is not None:
        if hasattr(env, 'seed'):
            env.seed(seed)
        np.random.seed(seed)

    episode_returns = []
    episode_lengths = []
    successes = []

    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gymnasium API

        done = False
        truncated = False
        ep_return = 0.0
        ep_length = 0

        while not (done or truncated) and ep_length < max_steps:
            action = policy_fn(obs, z)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()

            step_result = env.step(action)
            if len(step_result) == 4:
                next_obs, reward, done, info = step_result
                truncated = False
            else:
                next_obs, reward, done, truncated, info = step_result

            obs = next_obs
            ep_return += reward
            ep_length += 1

            if render:
                env.render()

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        # Check for success if available in info
        if 'success' in info:
            successes.append(float(info['success']))
        elif hasattr(env, 'get_success_metric'):
            successes.append(float(env.get_success_metric()))

    returns = np.array(episode_returns)
    lengths = np.array(episode_lengths)

    result = {
        'returns': returns.tolist(),
        'lengths': lengths.tolist(),
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'mean_length': float(np.mean(lengths)),
    }

    if successes:
        result['success_rate'] = float(np.mean(successes))
        result['successes'] = successes

    return result


def evaluate_policy_batch(
    env,
    policy_fn,
    z_list: List[np.ndarray],
    num_episodes_per_z: int = 20,
    max_steps: int = 1000,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate policy for multiple z vectors.

    Returns:
        Dict with 'all_returns' (shape: num_z x num_episodes), 'mean_returns', 'std_returns'.
    """
    all_returns = []
    for z in z_list:
        result = evaluate_policy_on_env(
            env, policy_fn, z,
            num_episodes=num_episodes_per_z,
            max_steps=max_steps,
            deterministic=deterministic,
            seed=seed,
        )
        all_returns.append(result['returns'])

    all_returns = np.array(all_returns)
    return {
        'all_returns': all_returns,
        'mean_returns': np.mean(all_returns, axis=1),
        'std_returns': np.std(all_returns, axis=1),
        'overall_mean': float(np.mean(all_returns)),
        'overall_std': float(np.std(all_returns)),
    }


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def compute_reward_heatmap(
    reward_fn,
    state_grid: np.ndarray,
    z: Optional[np.ndarray] = None,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute reward values over a 2D grid of states for heatmap visualization.

    Args:
        reward_fn: Function that maps states -> rewards, or (states, z) -> rewards.
        state_grid: Array of shape (N, state_dim) representing grid points.
        z: Optional latent vector for decoder-based reward prediction.
        batch_size: Batch size for processing.
        device: Device for computation.

    Returns:
        Array of shape (N,) with reward values.
    """
    rewards = []
    for i in range(0, len(state_grid), batch_size):
        batch = state_grid[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)
        with torch.no_grad():
            if z is not None:
                z_tensor = torch.FloatTensor(z).to(device).unsqueeze(0).expand(len(batch_tensor), -1)
                r = reward_fn(batch_tensor, z_tensor)
            else:
                r = reward_fn(batch_tensor)
        if isinstance(r, torch.Tensor):
            r = r.cpu().numpy()
        rewards.append(r.reshape(-1))
    return np.concatenate(rewards)


def compute_value_heatmap(
    value_fn,
    state_grid: np.ndarray,
    z: np.ndarray,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute value function over a 2D grid for heatmap visualization.

    Args:
        value_fn: Function (states, z) -> values.
        state_grid: Array of shape (N, state_dim).
        z: Latent vector.
        batch_size: Batch size.
        device: Device.

    Returns:
        Array of shape (N,) with value estimates.
    """
    values = []
    z_tensor = torch.FloatTensor(z).to(device)
    for i in range(0, len(state_grid), batch_size):
        batch = state_grid[i:i + batch_size]
        batch_tensor = torch.FloatTensor(batch).to(device)
        with torch.no_grad():
            v = value_fn(batch_tensor, z_tensor.unsqueeze(0).expand(len(batch_tensor), -1))
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        values.append(v.reshape(-1))
    return np.concatenate(values)


def generate_state_grid_2d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 100,
    fixed_dims: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate a 2D grid of states for visualization (e.g., for AntMaze).

    Args:
        x_range: (min_x, max_x).
        y_range: (min_y, max_y).
        resolution: Number of points per dimension.
        fixed_dims: Values for remaining state dimensions (if state_dim > 2).

    Returns:
        Array of shape (resolution*resolution, state_dim).
    """
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)

    if fixed_dims is not None:
        n_points = grid_2d.shape[0]
        full_grid = np.zeros((n_points, 2 + len(fixed_dims)))
        full_grid[:, :2] = grid_2d
        full_grid[:, 2:] = fixed_dims
        return full_grid

    return grid_2d


def plot_reward_heatmap(
    ax,
    grid_values: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 100,
    title: str = "Reward Heatmap",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot a 2D heatmap on a given matplotlib axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("Matplotlib not available for plotting.")
        return

    img = grid_values.reshape(resolution, resolution)
    im = ax.imshow(
        img,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im


def plot_trajectory(
    ax,
    trajectory: np.ndarray,
    color: str = "red",
    linewidth: float = 2.0,
    alpha: float = 0.8,
    label: str = "Trajectory",
    scatter_start: bool = True,
    scatter_end: bool = True,
):
    """Plot a 2D trajectory on a given matplotlib axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ax.plot(
        trajectory[:, 0], trajectory[:, 1],
        color=color, linewidth=linewidth, alpha=alpha, label=label,
    )
    if scatter_start:
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color="green", s=100, marker="o", label="Start")
    if scatter_end:
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="blue", s=100, marker="*", label="End")


def plot_encoding_states(
    ax,
    states: np.ndarray,
    color: str = "white",
    edgecolor: str = "black",
    size: float = 30.0,
    marker: str = "o",
    label: str = "Encoding States",
):
    """Plot encoding states as scatter points."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ax.scatter(
        states[:, 0], states[:, 1],
        c=color, edgecolors=edgecolor, s=size, marker=marker,
        label=label, zorder=5,
    )


# ==============================================================================
# Configuration Utilities
# ==============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError:
        # Fallback: try to parse as JSON
        with open(config_path, "r") as f:
            return json.load(f)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save a configuration dictionary to a YAML file."""
    try:
        import yaml
    except ImportError:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge two config dictionaries, with override taking precedence."""
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


# ==============================================================================
# Metric Tracking
# ==============================================================================

class MetricTracker:
    """Track running statistics for scalar metrics."""

    def __init__(self):
        self.values: Dict[str, List[float]] = defaultdict(list)

    def update(self, metrics: Dict[str, float]):
        """Add new metric values."""
        for key, value in metrics.items():
            self.values[key].append(value)

    def mean(self, key: str, window: Optional[int] = None) -> float:
        """Get mean of a metric, optionally over last `window` values."""
        vals = self.values.get(key, [])
        if not vals:
            return 0.0
        if window is not None:
            vals = vals[-window:]
        return float(np.mean(vals))

    def std(self, key: str, window: Optional[int] = None) -> float:
        """Get standard deviation of a metric."""
        vals = self.values.get(key, [])
        if not vals:
            return 0.0
        if window is not None:
            vals = vals[-window:]
        return float(np.std(vals))

    def latest(self, key: str) -> float:
        """Get the most recent value."""
        vals = self.values.get(key, [])
        return vals[-1] if vals else 0.0

    def summary(self, window: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        result = {}
        for key in self.values:
            vals = self.values[key]
            if window is not None:
                vals = vals[-window:]
            arr = np.array(vals)
            result[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "latest": float(arr[-1]),
            }
        return result

    def reset(self):
        """Clear all tracked values."""
        self.values.clear()


# ==============================================================================
# Miscellaneous Utilities
# ==============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility across numpy, torch, and random."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def save_json(data: Dict, filepath: str):
    """Save a dictionary as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load a dictionary from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def create_sweep_configs(
    base_config: Dict,
    sweep_params: Dict[str, List[Any]],
    output_dir: str,
) -> List[str]:
    """
    Create configuration files for a hyperparameter sweep.

    Args:
        base_config: Base configuration dictionary.
        sweep_params: Dictionary mapping parameter names to lists of values.
        output_dir: Directory to save config files.

    Returns:
        List of paths to generated config files.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_paths = []

    # Generate all combinations
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())

    def generate_combinations(idx, current_config):
        if idx == len(keys):
            # Save config
            name_parts = []
            for k, v in current_config.items():
                if k in sweep_params:
                    name_parts.append(f"{k}={v}")
            name = "_".join(name_parts) if name_parts else "default"
            path = os.path.join(output_dir, f"{name}.yaml")
            save_config(current_config, path)
            config_paths.append(path)
            return

        key = keys[idx]
        for val in values[idx]:
            new_config = current_config.copy()
            # Handle nested keys with dot notation
            if "." in key:
                parts = key.split(".")
                d = new_config
                for p in parts[:-1]:
                    if p not in d:
                        d[p] = {}
                    d = d[p]
                d[parts[-1]] = val
            else:
                new_config[key] = val
            generate_combinations(idx + 1, new_config)

    generate_combinations(0, base_config.copy())
    return config_paths


# ==============================================================================
# Environment Helpers
# ==============================================================================

def make_env(domain: str, task: Optional[str] = None) -> Any:
    """
    Create a gym environment for a given domain.

    Args:
        domain: One of 'antmaze', 'kitchen', 'walker', 'cheetah'.
        task: Specific task name (e.g., 'umaze', 'complete').

    Returns:
        Gym environment.
    """
    import gym

    if domain == "antmaze":
        maze_name = task or "umaze"
        env_name = f"antmaze-{maze_name}-v0"
        env = gym.make(env_name)
    elif domain == "kitchen":
        kitchen_type = task or "complete"
        env_name = f"kitchen-{kitchen_type}-v0"
        env = gym.make(env_name)
    elif domain in ("walker", "cheetah"):
        # ExORL environments use dm_control
        try:
            from dm_control import suite
            if domain == "walker":
                env = suite.load("walker", "walk")
            else:
                env = suite.load("cheetah", "run")
        except ImportError:
            # Fallback: create a minimal wrapper
            env = _create_minimal_env(domain)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return env


def _create_minimal_env(domain: str):
    """Create a minimal environment wrapper for ExORL domains."""
    import gym

    if domain == "walker":
        state_dim = 24
        action_dim = 6
    elif domain == "cheetah":
        state_dim = 17
        action_dim = 6
    else:
        raise ValueError(f"Unknown domain: {domain}")

    class MinimalEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )
            self._state = np.zeros(state_dim, dtype=np.float32)
            self._step_count = 0

        def reset(self):
            self._state = np.random.randn(state_dim).astype(np.float32) * 0.1
            self._step_count = 0
            return self._state

        def step(self, action):
            self._state = self._state + 0.01 * np.array(action).flatten()
            self._step_count += 1
            reward = -np.linalg.norm(self._state)  # Dummy reward
            done = self._step_count >= 1000
            return self._state, reward, done, {}

        def seed(self, seed):
            np.random.seed(seed)

        def render(self, mode="human"):
            pass

        def close(self):
            pass

    return MinimalEnv()


# ==============================================================================
# Built-in Tests
# ==============================================================================

def test_logger():
    """Quick test of the Logger class."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = Logger(log_dir=tmpdir, verbose=False)
        logger.log_metrics({"loss": 0.5, "reward": 10.0}, step=0)
        logger.log_metrics({"loss": 0.3, "reward": 12.0}, step=1)
        logger.save_metrics_summary()
        logger.close()
        assert os.path.exists(os.path.join(tmpdir, "metrics.csv"))
        assert os.path.exists(os.path.join(tmpdir, "metrics_summary.json"))
    print("Logger test passed!")


def test_metric_tracker():
    """Quick test of MetricTracker."""
    tracker = MetricTracker()
    tracker.update({"loss": 0.5, "acc": 0.8})
    tracker.update({"loss": 0.3, "acc": 0.9})
    assert abs(tracker.mean("loss") - 0.4) < 1e-6
    assert tracker.latest("acc") == 0.9
    print("MetricTracker test passed!")


if __name__ == "__main__":
    test_logger()
    test_metric_tracker()
    print("All utils tests passed!")