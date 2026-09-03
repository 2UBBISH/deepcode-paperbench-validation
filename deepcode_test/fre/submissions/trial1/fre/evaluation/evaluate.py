"""
Zero-shot evaluation module for Functional Reward Encodings (FRE).

Evaluates a trained FRE agent (encoder + IQL networks) on downstream tasks
without any fine-tuning. Given a few reward-annotated state samples (K=32),
the encoder produces a latent z, and the IQL policy conditioned on z is
rolled out in the environment.

Supports: AntMaze (D4RL), ExORL (Walker, Cheetah), Kitchen (D4RL).
"""

import os
import time
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Any, Callable
from collections import defaultdict

from fre.config import config
from fre.data.dataset import OfflineDataset, load_dataset
from fre.models.encoder import FREEncoder
from fre.models.decoder import RewardDecoder
from fre.models.iql import IQLNetworks
from fre.reward_functions.base import RewardFunction
from fre.reward_functions.eval_rewards import (
    create_eval_reward_function,
    get_eval_tasks,
)
from fre.training.utils import set_seed, get_device


# ============================================================
# Environment Creation
# ============================================================

def make_env(domain: str, task_name: Optional[str] = None) -> Any:
    """
    Create an environment instance for evaluation.

    Args:
        domain: Domain name ('antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen')
        task_name: Optional task name for environment variants

    Returns:
        Environment object with gym-like interface (reset, step, etc.)
    """
    if domain == "antmaze":
        return _make_antmaze_env()
    elif domain == "exorl_walker":
        return _make_exorl_env("walker")
    elif domain == "exorl_cheetah":
        return _make_exorl_env("cheetah")
    elif domain == "kitchen":
        return _make_kitchen_env()
    else:
        raise ValueError(f"Unknown domain: {domain}")


def _make_antmaze_env() -> Any:
    """Create AntMaze evaluation environment."""
    try:
        import gym
        import d4rl

        # Use the large play environment for evaluation
        # The antmaze-large-play-v2 is the standard evaluation environment
        env_name = "antmaze-large-play-v2"

        # Try v2 first, fall back to v0
        try:
            env = gym.make(env_name)
        except Exception:
            env = gym.make("antmaze-large-play-v0")

        return env
    except ImportError as e:
        raise ImportError(
            "D4RL and gym are required for AntMaze evaluation. "
            "Install with: pip install d4rl gym"
        ) from e


def _make_exorl_env(agent_type: str) -> Any:
    """
    Create ExORL evaluation environment (Walker or Cheetah).

    ExORL uses DeepMind Control Suite environments. We attempt to create
    a gym-compatible wrapper.

    Args:
        agent_type: 'walker' or 'cheetah'
    """
    try:
        import gym

        # Try to use exorl's environment creation
        try:
            from exorl.envs import make_env as exorl_make_env

            if agent_type == "walker":
                env = exorl_make_env("walker", "walk", action_repeat=1)
            else:
                env = exorl_make_env("cheetah", "run", action_repeat=1)
            return env
        except ImportError:
            pass

        # Fallback: try dm_control with gym wrapper
        try:
            from dm_control import suite
            from dm_control.suite.wrappers import pixels

            if agent_type == "walker":
                dm_env = suite.load("walker", "walk")
            else:
                dm_env = suite.load("cheetah", "run")

            # Wrap as gym environment
            try:
                from dm_env_wrappers import DmControlWrapper
                return DmControlWrapper(dm_env)
            except ImportError:
                # Simple wrapper
                return _SimpleDMControlWrapper(dm_env)
        except ImportError:
            pass

        # Last resort: try gym-mujoco environments
        if agent_type == "walker":
            try:
                return gym.make("Walker2d-v4")
            except Exception:
                pass
        else:
            try:
                return gym.make("HalfCheetah-v4")
            except Exception:
                pass

        raise ImportError(
            "Could not create ExORL environment. Please install exorl or "
            "dm_control: pip install dm_control"
        )
    except ImportError as e:
        raise ImportError(
            "ExORL evaluation requires gym and dm_control/exorl. "
            "Install with: pip install gym dm_control"
        ) from e


def _make_kitchen_env() -> Any:
    """Create Kitchen evaluation environment."""
    try:
        import gym
        import d4rl

        env_name = "kitchen-complete-v0"
        env = gym.make(env_name)
        return env
    except ImportError as e:
        raise ImportError(
            "D4RL and gym are required for Kitchen evaluation. "
            "Install with: pip install d4rl gym"
        ) from e


class _SimpleDMControlWrapper:
    """
    Simple wrapper to make a dm_control environment gym-like.
    Provides reset() -> state and step(action) -> (state, reward, done, info).
    """

    def __init__(self, dm_env):
        self._env = dm_env
        self._reset_next_step = True

        # Infer observation and action specs
        obs_spec = dm_env.observation_spec()
        action_spec = dm_env.action_spec()

        # Compute observation dimension
        self.observation_space = self._infer_obs_space(obs_spec)
        self.action_space = self._infer_action_space(action_spec)

    def _infer_obs_space(self, obs_spec):
        """Infer observation dimension from dm_control spec."""
        import gym
        total_dim = 0
        for key, spec in obs_spec.items():
            if hasattr(spec, 'shape'):
                total_dim += int(np.prod(spec.shape))
            else:
                total_dim += 1
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def _infer_action_space(self, action_spec):
        """Infer action space from dm_control spec."""
        import gym
        if hasattr(action_spec, 'shape'):
            shape = action_spec.shape
            if hasattr(action_spec, 'minimum') and hasattr(action_spec, 'maximum'):
                low = action_spec.minimum
                high = action_spec.maximum
            else:
                low = -np.ones(shape)
                high = np.ones(shape)
            return gym.spaces.Box(
                low=low, high=high, shape=shape, dtype=np.float32
            )
        else:
            return gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

    def _flatten_obs(self, timestep):
        """Flatten dm_control observation into a 1D numpy array."""
        obs = timestep.observation
        flat_parts = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (np.ndarray, list)):
                flat_parts.append(np.asarray(val).flatten())
            else:
                flat_parts.append(np.array([float(val)]))
        return np.concatenate(flat_parts).astype(np.float32)

    def reset(self):
        self._reset_next_step = False
        timestep = self._env.reset()
        return self._flatten_obs(timestep)

    def step(self, action):
        if self._reset_next_step:
            return self.reset(), 0.0, False, {}
        timestep = self._env.step(action)
        obs = self._flatten_obs(timestep)
        reward = timestep.reward if timestep.reward is not None else 0.0
        done = timestep.last()
        info = {}
        if done:
            self._reset_next_step = True
        return obs, reward, done, info

    def seed(self, seed=None):
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()


# ============================================================
# Evaluation Functions
# ============================================================

@torch.no_grad()
def evaluate_agent(
    encoder: FREEncoder,
    iql_networks: IQLNetworks,
    dataset: OfflineDataset,
    eval_reward_fn: RewardFunction,
    K: int = 32,
    num_episodes: int = 20,
    device: Optional[str] = None,
    seed: int = 0,
    max_episode_steps: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a trained FRE agent on a single downstream task.

    Steps:
    1. Sample K states from the dataset, compute rewards using eval_reward_fn.
    2. Encode to get latent z (deterministic, using mu).
    3. Roll out the policy pi(a|s, z) in the environment for num_episodes.
    4. Compute average undiscounted return.

    Args:
        encoder: Trained FRE encoder (frozen).
        iql_networks: Trained IQL networks.
        dataset: Offline dataset for sampling demonstration states.
        eval_reward_fn: Evaluation reward function (callable on states).
        K: Number of demonstration state-reward pairs for encoding.
        num_episodes: Number of evaluation episodes.
        device: Device for computation.
        seed: Random seed for reproducibility.
        max_episode_steps: Maximum steps per episode.
        verbose: Whether to print per-episode returns.

    Returns:
        Dictionary with:
            - 'mean_return': Average undiscounted return across episodes.
            - 'std_return': Standard deviation of returns.
            - 'returns': List of per-episode returns.
            - 'z_norm': L2 norm of the latent vector z.
            - 'num_episodes': Number of episodes evaluated.
    """
    if device is None:
        device = config.device
    device = get_device(device)

    set_seed(seed)

    # Put models in eval mode
    encoder.eval()
    iql_networks.q1.eval()
    iql_networks.q2.eval()
    iql_networks.v.eval()
    iql_networks.policy.eval()

    # Step 1: Sample K states from dataset and compute rewards
    states = dataset.sample_states(K)  # (K, state_dim)
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states).float().to(device)
    else:
        states = states.float().to(device)

    # Compute rewards for these states
    with torch.no_grad():
        rewards = eval_reward_fn(states)  # (K,)

    # Step 2: Encode to get latent z (deterministic)
    states_enc = states.unsqueeze(0)  # (1, K, state_dim)
    rewards_enc = rewards.unsqueeze(0)  # (1, K)
    z = encoder.encode_deterministic(states_enc, rewards_enc)  # (1, d_latent)
    z_norm = torch.norm(z).item()

    # Step 3: Create environment and roll out
    # Determine domain from reward function info
    reward_info = eval_reward_fn.get_info()
    domain = _infer_domain_from_reward(reward_info)

    env = make_env(domain)
    if hasattr(env, 'seed'):
        env.seed(seed)

    returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gym's new API returning (obs, info)

        done = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0

        while not done and not truncated and episode_steps < max_episode_steps:
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(
                np.asarray(obs, dtype=np.float32)
            ).unsqueeze(0).to(device)  # (1, state_dim)

            # Get action from policy
            action = iql_networks.get_action(
                obs_tensor, z, deterministic=True
            )  # (1, action_dim)
            action_np = action.squeeze(0).cpu().numpy()

            # Step environment
            step_result = env.step(action_np)
            if len(step_result) == 5:
                # New gym API: (obs, reward, terminated, truncated, info)
                next_obs, reward, done, truncated, info = step_result
            else:
                # Old gym API: (obs, reward, done, info)
                next_obs, reward, done, info = step_result
                truncated = False

            # Compute reward using eval_reward_fn (not environment reward)
            next_obs_tensor = torch.from_numpy(
                np.asarray(next_obs, dtype=np.float32)
            ).unsqueeze(0).to(device)
            eval_reward = eval_reward_fn(next_obs_tensor).item()

            episode_return += eval_reward
            episode_steps += 1
            obs = next_obs

        returns.append(episode_return)
        episode_lengths.append(episode_steps)

        if verbose:
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"return = {episode_return:.4f}, steps = {episode_steps}")

    env.close()

    # Compute statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)

    result = {
        'mean_return': mean_return,
        'std_return': std_return,
        'returns': returns,
        'episode_lengths': episode_lengths,
        'mean_episode_length': mean_length,
        'z_norm': z_norm,
        'num_episodes': num_episodes,
    }

    return result


def _infer_domain_from_reward(reward_info: Dict[str, Any]) -> str:
    """Infer domain from reward function info dictionary."""
    reward_type = reward_info.get('type', '')

    if 'antmaze' in reward_type:
        return 'antmaze'
    elif 'exorl' in reward_type:
        if 'walker' in reward_type:
            return 'exorl_walker'
        elif 'cheetah' in reward_type:
            return 'exorl_cheetah'
        return 'exorl_walker'  # default
    elif 'kitchen' in reward_type:
        return 'kitchen'
    else:
        # Default fallback
        return 'antmaze'


@torch.no_grad()
def evaluate_all_tasks(
    encoder: FREEncoder,
    iql_networks: IQLNetworks,
    dataset: OfflineDataset,
    domain: str,
    K: int = 32,
    num_episodes: int = 20,
    device: Optional[str] = None,
    seed: int = 0,
    max_episode_steps: int = 1000,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate a trained FRE agent on all downstream tasks for a domain.

    Args:
        encoder: Trained FRE encoder.
        iql_networks: Trained IQL networks.
        dataset: Offline dataset.
        domain: Domain name ('antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen').
        K: Number of demonstration states.
        num_episodes: Number of episodes per task.
        device: Computation device.
        seed: Base random seed (incremented per task).
        max_episode_steps: Maximum steps per episode.
        verbose: Whether to print progress.

    Returns:
        Nested dictionary: task_name -> evaluation result dict.
    """
    if device is None:
        device = config.device
    device = get_device(device)

    # Get list of evaluation tasks for this domain
    task_names = get_eval_tasks(domain)
    state_dim = dataset.state_dim

    results = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating on domain: {domain}")
        print(f"Number of tasks: {len(task_names)}")
        print(f"K = {K}, num_episodes = {num_episodes}")
        print(f"{'='*60}\n")

    for i, task_name in enumerate(task_names):
        if verbose:
            print(f"\nTask {i+1}/{len(task_names)}: {task_name}")

        # Create evaluation reward function
        eval_reward_fn = create_eval_reward_function(
            task_name=task_name,
            state_dim=state_dim,
            device=device,
        )

        # Evaluate
        task_seed = seed + i * 100  # Different seed per task
        result = evaluate_agent(
            encoder=encoder,
            iql_networks=iql_networks,
            dataset=dataset,
            eval_reward_fn=eval_reward_fn,
            K=K,
            num_episodes=num_episodes,
            device=device,
            seed=task_seed,
            max_episode_steps=max_episode_steps,
            verbose=verbose,
        )

        results[task_name] = result

        if verbose:
            print(f"  Mean return: {result['mean_return']:.4f} "
                  f"± {result['std_return']:.4f}")

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY - {domain}")
        print(f"{'='*60}")
        for task_name, result in results.items():
            print(f"  {task_name:40s}: {result['mean_return']:8.4f} "
                  f"± {result['std_return']:8.4f}")
        avg_return = np.mean([r['mean_return'] for r in results.values()])
        print(f"  {'AVERAGE':40s}: {avg_return:8.4f}")
        print(f"{'='*60}\n")

    return results


def run_evaluation(
    encoder_path: str,
    iql_path: str,
    domain: str,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 0,
    K: int = 32,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline: load models, evaluate on all tasks, save results.

    Args:
        encoder_path: Path to encoder checkpoint (or directory with 'encoder.pt').
        iql_path: Path to IQL checkpoint (or directory with 'iql.pt').
        domain: Domain name.
        data_dir: Directory for offline dataset.
        device: Computation device.
        seed: Random seed.
        K: Number of demonstration states.
        num_episodes: Number of episodes per task.
        max_episode_steps: Maximum steps per episode.
        output_dir: Directory to save results (JSON).
        verbose: Whether to print progress.

    Returns:
        Dictionary with all evaluation results.
    """
    if device is None:
        device = config.device
    device = get_device(device)

    set_seed(seed)

    # Load dataset
    if verbose:
        print(f"Loading dataset for domain: {domain}")
    dataset = load_dataset(domain, data_dir=data_dir, device=device)

    state_dim = dataset.state_dim
    action_dim = dataset.action_dim

    # Load encoder
    if verbose:
        print(f"Loading encoder from: {encoder_path}")
    if os.path.isdir(encoder_path):
        encoder_file = os.path.join(encoder_path, "encoder.pt")
    else:
        encoder_file = encoder_path

    encoder = FREEncoder(
        state_dim=state_dim,
        d_embed=config.d_embed,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_latent=config.d_latent,
        num_reward_bins=config.num_reward_bins,
        r_max=config.r_max,
    ).to(device)

    decoder = RewardDecoder(
        state_dim=state_dim,
        d_latent=config.d_latent,
        hidden_dims=config.decoder_hidden_dims,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(encoder_file, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    # Load IQL networks
    if verbose:
        print(f"Loading IQL networks from: {iql_path}")
    if os.path.isdir(iql_path):
        iql_file = os.path.join(iql_path, "iql.pt")
    else:
        iql_file = iql_path

    iql_networks = IQLNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        d_latent=config.d_latent,
        hidden_dims=config.iql_hidden_dims,
    ).to(device)

    iql_checkpoint = torch.load(iql_file, map_location=device)
    iql_networks.q1.load_state_dict(iql_checkpoint['q1_state_dict'])
    iql_networks.q2.load_state_dict(iql_checkpoint['q2_state_dict'])
    iql_networks.q1_target.load_state_dict(iql_checkpoint['q1_target_state_dict'])
    iql_networks.q2_target.load_state_dict(iql_checkpoint['q2_target_state_dict'])
    iql_networks.v.load_state_dict(iql_checkpoint['v_state_dict'])
    iql_networks.policy.load_state_dict(iql_checkpoint['policy_state_dict'])
    iql_networks.eval_mode()

    # Run evaluation
    results = evaluate_all_tasks(
        encoder=encoder,
        iql_networks=iql_networks,
        dataset=dataset,
        domain=domain,
        K=K,
        num_episodes=num_episodes,
        device=device,
        seed=seed,
        max_episode_steps=max_episode_steps,
        verbose=verbose,
    )

    # Save results if output directory specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        import json

        # Convert results to serializable format
        serializable_results = {}
        for task_name, result in results.items():
            serializable_results[task_name] = {
                'mean_return': float(result['mean_return']),
                'std_return': float(result['std_return']),
                'returns': [float(r) for r in result['returns']],
                'mean_episode_length': float(result['mean_episode_length']),
                'z_norm': float(result['z_norm']),
                'num_episodes': result['num_episodes'],
            }

        output_file = os.path.join(output_dir, f"eval_results_{domain}.json")
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_file}")

    return results


# ============================================================
# Multi-Seed Evaluation
# ============================================================

def evaluate_multiple_seeds(
    encoder_path: str,
    iql_path: str,
    domain: str,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    seeds: List[int] = None,
    K: int = 32,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate across multiple random seeds and aggregate results.

    This matches the paper's evaluation protocol: 5 seeds, report mean ± std.

    Args:
        encoder_path: Path to encoder checkpoint.
        iql_path: Path to IQL checkpoint.
        domain: Domain name.
        data_dir: Dataset directory.
        device: Computation device.
        seeds: List of random seeds (default: [0, 1, 2, 3, 4]).
        K: Number of demonstration states.
        num_episodes: Number of episodes per task per seed.
        max_episode_steps: Maximum steps per episode.
        output_dir: Directory to save results.
        verbose: Whether to print progress.

    Returns:
        Nested dictionary: task_name -> {
            'mean_across_seeds': float,
            'std_across_seeds': float,
            'per_seed_means': List[float],
            'per_seed_stds': List[float],
        }
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    if device is None:
        device = config.device
    device = get_device(device)

    # Collect per-seed results
    all_seed_results = defaultdict(list)

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"SEED {seed_idx + 1}/{len(seeds)}: seed={seed}")
            print(f"{'#'*60}")

        results = run_evaluation(
            encoder_path=encoder_path,
            iql_path=iql_path,
            domain=domain,
            data_dir=data_dir,
            device=device,
            seed=seed,
            K=K,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            output_dir=None,  # Save at the end
            verbose=verbose,
        )

        for task_name, result in results.items():
            all_seed_results[task_name].append({
                'mean_return': result['mean_return'],
                'std_return': result['std_return'],
                'returns': result['returns'],
            })

    # Aggregate across seeds
    aggregated = {}
    for task_name, seed_results in all_seed_results.items():
        per_seed_means = [r['mean_return'] for r in seed_results]
        per_seed_stds = [r['std_return'] for r in seed_results]

        aggregated[task_name] = {
            'mean_across_seeds': float(np.mean(per_seed_means)),
            'std_across_seeds': float(np.std(per_seed_means)),
            'per_seed_means': per_seed_means,
            'per_seed_stds': per_seed_stds,
        }

    # Print final summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - {domain} (averaged over {len(seeds)} seeds)")
        print(f"{'='*60}")
        for task_name, agg in aggregated.items():
            print(f"  {task_name:40s}: {agg['mean_across_seeds']:8.4f} "
                  f"± {agg['std_across_seeds']:8.4f}")
        overall_mean = np.mean([a['mean_across_seeds'] for a in aggregated.values()])
        overall_std = np.std([a['mean_across_seeds'] for a in aggregated.values()])
        print(f"  {'OVERALL AVERAGE':40s}: {overall_mean:8.4f} ± {overall_std:8.4f}")
        print(f"{'='*60}\n")

    # Save aggregated results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        import json

        output_file = os.path.join(
            output_dir, f"aggregated_results_{domain}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)
        if verbose:
            print(f"Aggregated results saved to: {output_file}")

    return aggregated


# ============================================================
# Visualization Utilities
# ============================================================

@torch.no_grad()
def visualize_value_function(
    encoder: FREEncoder,
    iql_networks: IQLNetworks,
    dataset: OfflineDataset,
    eval_reward_fn: RewardFunction,
    K: int = 32,
    device: Optional[str] = None,
    grid_resolution: int = 100,
    x_range: Tuple[float, float] = (-2.0, 20.0),
    y_range: Tuple[float, float] = (-2.0, 20.0),
) -> Dict[str, np.ndarray]:
    """
    Visualize the value function and policy for 2D state spaces (AntMaze).

    Creates a grid over the xy-plane and evaluates V(s, z) and the
    decoder's predicted reward for each point.

    Args:
        encoder: Trained encoder.
        iql_networks: Trained IQL networks.
        dataset: Offline dataset.
        eval_reward_fn: Evaluation reward function.
        K: Number of demonstration states.
        device: Computation device.
        grid_resolution: Number of grid points per dimension.
        x_range: (min_x, max_x) for grid.
        y_range: (min_y, max_y) for grid.

    Returns:
        Dictionary with:
            - 'X': 2D grid of x coordinates.
            - 'Y': 2D grid of y coordinates.
            - 'V': Value function values on grid.
            - 'decoder_reward': Decoder predicted rewards on grid.
            - 'true_reward': True reward function values on grid.
    """
    if device is None:
        device = config.device
    device = get_device(device)

    encoder.eval()
    iql_networks.v.eval()

    # Encode z from K demonstration states
    states = dataset.sample_states(K)
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states).float().to(device)
    else:
        states = states.float().to(device)

    rewards = eval_reward_fn(states)
    states_enc = states.unsqueeze(0)
    rewards_enc = rewards.unsqueeze(0)
    z = encoder.encode_deterministic(states_enc, rewards_enc)  # (1, d_latent)

    # Create grid
    xs = np.linspace(x_range[0], x_range[1], grid_resolution)
    ys = np.linspace(y_range[0], y_range[1], grid_resolution)
    X, Y = np.meshgrid(xs, ys)

    # For AntMaze, the state has more dimensions than just xy.
    # We need to fill in reasonable defaults for other dimensions.
    # Use the mean of the dataset for non-xy dimensions.
    all_states = dataset.get_all_states()
    if isinstance(all_states, torch.Tensor):
        all_states_np = all_states.cpu().numpy()
    else:
        all_states_np = all_states

    mean_state = np.mean(all_states_np, axis=0)

    # Evaluate on grid
    V = np.zeros((grid_resolution, grid_resolution))
    true_reward = np.zeros((grid_resolution, grid_resolution))

    grid_points = []
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            state = mean_state.copy()
            state[0] = X[i, j]  # x position
            state[1] = Y[i, j]  # y position
            grid_points.append(state)

    grid_tensor = torch.from_numpy(
        np.array(grid_points, dtype=np.float32)
    ).to(device)

    # Compute in batches to avoid OOM
    batch_size = 1024
    v_values = []
    r_values = []

    for b in range(0, len(grid_points), batch_size):
        batch = grid_tensor[b:b + batch_size]
        z_batch = z.expand(batch.shape[0], -1)

        v_batch = iql_networks.v(batch, z_batch).squeeze(-1)
        r_batch = eval_reward_fn(batch)

        v_values.append(v_batch.cpu().numpy())
        r_values.append(r_batch.cpu().numpy())

    v_all = np.concatenate(v_values)
    r_all = np.concatenate(r_values)

    V = v_all.reshape(grid_resolution, grid_resolution)
    true_reward = r_all.reshape(grid_resolution, grid_resolution)

    return {
        'X': X,
        'Y': Y,
        'V': V,
        'true_reward': true_reward,
        'z': z.cpu().numpy(),
    }


# ============================================================
# Test Function
# ============================================================

def test_evaluation() -> bool:
    """
    Quick test of evaluation utilities (without requiring actual environments).

    Returns:
        True if all tests pass.
    """
    print("Testing evaluation module...")

    # Test domain inference
    assert _infer_domain_from_reward({'type': 'antmaze_goal'}) == 'antmaze'
    assert _infer_domain_from_reward({'type': 'exorl_walker_goal'}) == 'exorl_walker'
    assert _infer_domain_from_reward({'type': 'exorl_cheetah_velocity'}) == 'exorl_cheetah'
    assert _infer_domain_from_reward({'type': 'kitchen_subtask'}) == 'kitchen'
    print("  Domain inference: PASSED")

    # Test that get_eval_tasks returns lists
    for domain in ['antmaze', 'exorl_walker', 'exorl_cheetah', 'kitchen']:
        tasks = get_eval_tasks(domain)
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        print(f"  {domain}: {len(tasks)} tasks")

    # Test create_eval_reward_function
    try:
        r_fn = create_eval_reward_function(
            'antmaze_goal_2.0_4.0', state_dim=29, device='cpu'
        )
        assert r_fn is not None
        print("  create_eval_reward_function: PASSED")
    except Exception as e:
        print(f"  create_eval_reward_function: FAILED - {e}")
        return False

    print("Evaluation module tests: ALL PASSED")
    return True


if __name__ == "__main__":
    test_evaluation()