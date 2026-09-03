#!/usr/bin/env python3
"""
Zero-Shot Evaluation Script for Functional Reward Encodings (FRE).

This script evaluates a trained FRE agent on downstream tasks without any
additional training. It loads a checkpoint containing the FRE encoder and
IQL agent, then for each evaluation task defined in the configuration:
  1. Defines the task-specific reward function.
  2. Samples K encoding states from the dataset, computes their rewards
     under the task reward function, and encodes them to a latent vector z.
  3. Runs the z-conditioned policy in the environment for multiple episodes.
  4. Normalizes returns to [0, 100] and reports results.

Usage:
    python scripts/evaluate.py --config configs/antmaze.yaml --checkpoint checkpoints/antmaze_final.pt
    python scripts/evaluate.py --config configs/kitchen.yaml --checkpoint checkpoints/kitchen_final.pt --seed 42
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch

from typing import Dict, List, Optional, Tuple, Any, Callable

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fre.utils import (
    load_config,
    merge_configs,
    set_seed,
    get_device,
    Logger,
    MetricTracker,
    evaluate_policy_on_env,
    make_env,
    save_json,
    format_time,
    normalize_score as util_normalize_score,
)
from fre.data_utils import (
    load_dataset,
    sample_encoder_states,
    ReplayBuffer,
)
from fre.fre_model import FREModel, build_fre_model
from fre.iql import IQLAgent, build_iql_agent
from fre.reward_prior import RewardPrior


# ==============================================================================
# Task Reward Function Builders
# ==============================================================================

def build_goal_reaching_reward(target: np.ndarray, threshold: float = 0.5) -> Callable:
    """
    Build a goal-reaching reward function.
    Reward = -1 if distance > threshold, else 0.
    
    Args:
        target: Goal state vector (or relevant dimensions).
        threshold: Distance threshold for success.
    
    Returns:
        Callable: reward_fn(states) -> np.ndarray of rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # states: (N, state_dim), target: (goal_dim,)
        # Use only the first goal_dim dimensions for distance
        goal_dim = len(target)
        diff = states[:, :goal_dim] - target[np.newaxis, :]
        dist = np.linalg.norm(diff, axis=1)
        rewards = np.where(dist > threshold, -1.0, 0.0)
        return rewards
    return reward_fn


def build_directional_reward(direction: np.ndarray) -> Callable:
    """
    Build a directional reward function.
    Reward = dot product of state (first dims) with direction vector.
    
    Args:
        direction: Direction vector (e.g., [1, 0] for +x).
    
    Returns:
        Callable: reward_fn(states) -> np.ndarray of rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        dim = len(direction)
        rewards = np.dot(states[:, :dim], direction)
        return rewards
    return reward_fn


def build_random_simplex_reward(state_dim: int, seed: int = 0) -> Callable:
    """
    Build a random simplex (procedural noise) reward function.
    Uses a random linear combination of sinusoidal features.
    
    Args:
        state_dim: Dimensionality of state space.
        seed: Random seed for reproducibility.
    
    Returns:
        Callable: reward_fn(states) -> np.ndarray of rewards.
    """
    rng = np.random.RandomState(seed)
    # Generate random frequencies and phases
    num_features = 16
    frequencies = rng.randn(num_features, state_dim) * 2.0
    phases = rng.randn(num_features) * np.pi
    weights = rng.randn(num_features)
    
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # states: (N, state_dim)
        projections = np.dot(states, frequencies.T)  # (N, num_features)
        features = np.sin(projections + phases[np.newaxis, :])  # (N, num_features)
        rewards = np.dot(features, weights)  # (N,)
        return rewards
    return reward_fn


def build_path_reward(path_points: np.ndarray, threshold: float = 0.5) -> Callable:
    """
    Build a path-following reward function.
    Reward = -min_distance_to_path if > threshold, else 0.
    
    Args:
        path_points: Array of (x, y) points defining the path.
        threshold: Distance threshold for being "on path".
    
    Returns:
        Callable: reward_fn(states) -> np.ndarray of rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # states: (N, state_dim), use first 2 dims
        positions = states[:, :2]  # (N, 2)
        # Compute minimum distance to any path segment
        min_dists = np.full(len(states), np.inf)
        for i in range(len(path_points) - 1):
            a = path_points[i]
            b = path_points[i + 1]
            ab = b - a
            ab_len_sq = np.dot(ab, ab)
            if ab_len_sq < 1e-8:
                dists = np.linalg.norm(positions - a[np.newaxis, :], axis=1)
            else:
                t = np.clip(
                    np.dot(positions - a[np.newaxis, :], ab) / ab_len_sq,
                    0.0, 1.0
                )
                projections = a[np.newaxis, :] + t[:, np.newaxis] * ab[np.newaxis, :]
                dists = np.linalg.norm(positions - projections, axis=1)
            min_dists = np.minimum(min_dists, dists)
        rewards = np.where(min_dists > threshold, -1.0, 0.0)
        return rewards
    return reward_fn


def build_velocity_reward(target_velocity: float, velocity_idx: int = 0) -> Callable:
    """
    Build a velocity reward function for locomotion tasks.
    Reward = -(current_velocity - target_velocity)^2
    
    Args:
        target_velocity: Desired velocity.
        velocity_idx: Index of velocity component in state.
    
    Returns:
        Callable: reward_fn(states) -> np.ndarray of rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # states: (N, state_dim)
        vel = states[:, velocity_idx]
        rewards = -((vel - target_velocity) ** 2)
        return rewards
    return reward_fn


def build_kitchen_subtask_reward(subtask_ids: List[int]) -> Callable:
    """
    Build a Kitchen subtask reward function.
    Reward = number of completed subtasks (0 to len(subtask_ids)).
    Note: This is a simplified version; the actual reward depends on
    environment state which includes subtask completion indicators.
    
    Args:
        subtask_ids: List of subtask indices to check.
    
    Returns:
        Callable: reward_fn(states) -> np.ndarray of rewards.
    """
    def reward_fn(states: np.ndarray) -> np.ndarray:
        # For Kitchen, the state includes subtask completion indicators.
        # We assume the relevant dimensions encode subtask progress.
        # This is a placeholder; actual evaluation uses environment returns.
        rewards = np.zeros(len(states))
        for sid in subtask_ids:
            if sid < states.shape[1]:
                rewards += states[:, sid]
        return rewards
    return reward_fn


# ==============================================================================
# Task Reward Function Factory
# ==============================================================================

def create_task_reward_fn(task_config: Dict[str, Any], state_dim: int) -> Callable:
    """
    Create a reward function from a task configuration dictionary.
    
    Args:
        task_config: Dictionary with 'type' and type-specific parameters.
        state_dim: Dimensionality of the state space.
    
    Returns:
        Callable: reward_fn(states) -> np.ndarray of rewards.
    """
    task_type = task_config.get('type', 'goal_reaching')
    
    if task_type == 'goal_reaching':
        target = np.array(task_config['target'], dtype=np.float32)
        threshold = task_config.get('threshold', 0.5)
        return build_goal_reaching_reward(target, threshold)
    
    elif task_type == 'directional':
        direction = np.array(task_config['direction'], dtype=np.float32)
        return build_directional_reward(direction)
    
    elif task_type == 'random_simplex':
        seed = task_config.get('seed', 0)
        return build_random_simplex_reward(state_dim, seed)
    
    elif task_type == 'path':
        path_points = np.array(task_config['path_points'], dtype=np.float32)
        threshold = task_config.get('threshold', 0.5)
        return build_path_reward(path_points, threshold)
    
    elif task_type == 'velocity':
        target_velocity = task_config.get('target_velocity', 1.0)
        velocity_idx = task_config.get('velocity_idx', 0)
        return build_velocity_reward(target_velocity, velocity_idx)
    
    elif task_type == 'subtask':
        subtask_ids = task_config.get('subtask_ids', [task_config.get('subtask_id', 0)])
        if isinstance(subtask_ids, int):
            subtask_ids = [subtask_ids]
        return build_kitchen_subtask_reward(subtask_ids)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ==============================================================================
# Checkpoint Loading
# ==============================================================================

def load_fre_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[FREModel, IQLAgent]:
    """
    Load a trained FRE model and IQL agent from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        state_dim: State space dimension.
        action_dim: Action space dimension.
        config: Configuration dictionary with model hyperparameters.
        device: Torch device.
    
    Returns:
        Tuple of (FREModel, IQLAgent) with loaded weights.
    """
    # Build models with same architecture as training
    fre_model = build_fre_model(
        state_dim=state_dim,
        latent_dim=config.get('latent_dim', 64),
        d_model=config.get('d_model', 256),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        d_emb=config.get('d_emb', 64),
        num_bins=config.get('num_bins', 100),
        reward_min=config.get('reward_min', -10.0),
        reward_max=config.get('reward_max', 10.0),
        decoder_hidden_dims=config.get('decoder_hidden_dims', [256, 256]),
        beta=config.get('beta', 0.1),
        dropout=config.get('dropout', 0.0),
        max_num_states=config.get('max_num_states', 32),
    ).to(device)
    
    iql_agent = build_iql_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=config.get('latent_dim', 64),
        hidden_dims=config.get('iql_hidden_dims', [256, 256]),
        activation=config.get('activation', 'relu'),
        dropout=config.get('dropout', 0.0),
        expectile=config.get('expectile', 0.7),
        temperature=config.get('temperature', 3.0),
        discount=config.get('discount', 0.99),
        soft_target_update_rate=config.get('soft_target_update_rate', 0.005),
        log_std_min=config.get('log_std_min', -5.0),
        log_std_max=config.get('log_std_max', 2.0),
        device=device,
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'fre_model_state_dict' in checkpoint:
        fre_model.load_state_dict(checkpoint['fre_model_state_dict'])
    elif 'encoder_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
        fre_model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        fre_model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    elif 'model_state_dict' in checkpoint:
        fre_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading directly
        fre_model.load_state_dict(checkpoint, strict=False)
    
    if 'iql_state_dict' in checkpoint:
        iql_agent.load_state_dict(checkpoint['iql_state_dict'])
    elif 'agent_state_dict' in checkpoint:
        iql_agent.load_state_dict(checkpoint['agent_state_dict'])
    
    fre_model.eval()
    iql_agent.eval()
    
    return fre_model, iql_agent


# ==============================================================================
# Main Evaluation Function
# ==============================================================================

def evaluate_all_tasks(
    fre_model: FREModel,
    iql_agent: IQLAgent,
    replay_buffer: ReplayBuffer,
    env: Any,
    tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
    logger: Optional[Logger] = None,
) -> Dict[str, Any]:
    """
    Evaluate the FRE agent on all downstream tasks.
    
    Args:
        fre_model: Trained FRE encoder/decoder model.
        iql_agent: Trained IQL agent.
        replay_buffer: Dataset replay buffer for sampling encoding states.
        env: Gym environment (or wrapper).
        tasks: List of task configuration dictionaries.
        config: Full configuration dictionary.
        device: Torch device.
        logger: Optional logger for recording metrics.
    
    Returns:
        Dictionary mapping task names to evaluation results.
    """
    K_encoder = config.get('K_encoder', 32)
    num_episodes = config.get('evaluation', {}).get('num_episodes', 20)
    max_steps = config.get('evaluation', {}).get('max_steps', 1000)
    deterministic = config.get('evaluation', {}).get('deterministic', True)
    ref_min = config.get('ref_min_score', 0.0)
    ref_max = config.get('ref_max_score', 100.0)
    
    state_dim = replay_buffer.observations.shape[1]
    
    results = {}
    all_normalized_returns = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating {len(tasks)} downstream tasks...")
    print(f"{'='*70}\n")
    
    for task_idx, task_config in enumerate(tasks):
        task_name = task_config.get('name', f'task_{task_idx}')
        task_type = task_config.get('type', 'unknown')
        
        print(f"[{task_idx+1}/{len(tasks)}] Task: {task_name} (type={task_type})")
        
        # Create task reward function
        try:
            reward_fn = create_task_reward_fn(task_config, state_dim)
        except Exception as e:
            print(f"  ERROR creating reward function: {e}")
            results[task_name] = {
                'error': str(e),
                'normalized_return_mean': 0.0,
                'normalized_return_std': 0.0,
                'raw_return_mean': 0.0,
                'raw_return_std': 0.0,
            }
            continue
        
        # Sample encoding states and compute rewards
        encoder_states = sample_encoder_states(replay_buffer, K_encoder)
        encoder_rewards = reward_fn(encoder_states)
        
        # Encode to latent z
        with torch.no_grad():
            states_tensor = torch.FloatTensor(encoder_states).unsqueeze(0).to(device)
            rewards_tensor = torch.FloatTensor(encoder_rewards).unsqueeze(0).to(device)
            z = fre_model.encode_rewards(states_tensor, rewards_tensor)
            z_np = z.cpu().numpy().flatten()
        
        # Create policy function for evaluation
        def policy_fn(state: np.ndarray, z_vec: np.ndarray = z_np) -> np.ndarray:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                z_t = torch.FloatTensor(z_vec).unsqueeze(0).to(device)
                action = iql_agent.select_action(s, z_t, deterministic=deterministic)
                return action.cpu().numpy().flatten()
        
        # Evaluate policy
        eval_result = evaluate_policy_on_env(
            env=env,
            policy_fn=policy_fn,
            z=z_np,
            num_episodes=num_episodes,
            max_steps=max_steps,
            deterministic=deterministic,
            render=False,
        )
        
        raw_returns = eval_result['returns']
        raw_mean = float(np.mean(raw_returns))
        raw_std = float(np.std(raw_returns))
        
        # Normalize returns to [0, 100]
        normalized_returns = util_normalize_score(raw_returns, ref_min, ref_max, clip=True)
        norm_mean = float(np.mean(normalized_returns))
        norm_std = float(np.std(normalized_returns))
        
        results[task_name] = {
            'task_type': task_type,
            'raw_return_mean': raw_mean,
            'raw_return_std': raw_std,
            'normalized_return_mean': norm_mean,
            'normalized_return_std': norm_std,
            'raw_returns': raw_returns.tolist(),
            'normalized_returns': normalized_returns.tolist(),
            'episode_lengths': eval_result.get('lengths', []),
            'success_rate': eval_result.get('success_rate', 0.0),
        }
        
        all_normalized_returns.append(norm_mean)
        
        print(f"  Raw return: {raw_mean:.2f} ± {raw_std:.2f}")
        print(f"  Normalized: {norm_mean:.1f} ± {norm_std:.1f}")
        
        if logger is not None:
            logger.log_metrics({
                f'eval/{task_name}/raw_return_mean': raw_mean,
                f'eval/{task_name}/raw_return_std': raw_std,
                f'eval/{task_name}/normalized_return_mean': norm_mean,
                f'eval/{task_name}/normalized_return_std': norm_std,
            }, step=0)
    
    # Compute aggregate statistics
    if all_normalized_returns:
        overall_mean = float(np.mean(all_normalized_returns))
        overall_std = float(np.std(all_normalized_returns))
    else:
        overall_mean = 0.0
        overall_std = 0.0
    
    results['_aggregate'] = {
        'overall_normalized_mean': overall_mean,
        'overall_normalized_std': overall_std,
        'num_tasks': len(tasks),
        'num_tasks_evaluated': len(all_normalized_returns),
    }
    
    print(f"\n{'='*70}")
    print(f"Overall Results:")
    print(f"  Average normalized return: {overall_mean:.1f} ± {overall_std:.1f}")
    print(f"  Tasks evaluated: {len(all_normalized_returns)}/{len(tasks)}")
    print(f"{'='*70}\n")
    
    return results


# ==============================================================================
# Command-Line Interface
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of FRE agent on downstream tasks."
    )
    
    # Required arguments
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML configuration file (e.g., configs/antmaze.yaml).'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to training checkpoint (.pt file).'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save evaluation results (JSON). Default: auto-generated in checkpoint dir.'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for evaluation. Overrides config seed.'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cpu, cuda, cuda:0, etc.). Overrides config.'
    )
    parser.add_argument(
        '--num_episodes', type=int, default=None,
        help='Number of evaluation episodes per task. Overrides config.'
    )
    parser.add_argument(
        '--tasks', type=str, nargs='*', default=None,
        help='Specific task names to evaluate. If not provided, evaluates all tasks.'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Print detailed evaluation progress.'
    )
    parser.add_argument(
        '--quiet', action='store_true', default=False,
        help='Suppress detailed output.'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['device'] = args.device
    if args.num_episodes is not None:
        if 'evaluation' not in config:
            config['evaluation'] = {}
        config['evaluation']['num_episodes'] = args.num_episodes
    
    # Set seed and device
    seed = config.get('seed', 0)
    set_seed(seed)
    device = get_device(config.get('device', 'auto'))
    
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load dataset
    domain = config.get('domain', 'antmaze')
    task = config.get('task', None)
    data_dir = config.get('data_dir', None)
    
    print(f"\nLoading dataset: domain={domain}, task={task}")
    replay_buffer, env = load_dataset(domain=domain, task=task, data_dir=data_dir)
    
    state_dim = replay_buffer.observations.shape[1]
    action_dim = replay_buffer.actions.shape[1]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Dataset size: {len(replay_buffer)} transitions")
    
    # Load trained models
    print("\nLoading checkpoint...")
    fre_model, iql_agent = load_fre_checkpoint(
        checkpoint_path=args.checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=device,
    )
    print("Checkpoint loaded successfully.")
    
    # Get evaluation tasks
    eval_config = config.get('evaluation', {})
    all_tasks = eval_config.get('tasks', [])
    
    if not all_tasks:
        print("ERROR: No evaluation tasks defined in config.")
        sys.exit(1)
    
    # Filter tasks if specific tasks requested
    if args.tasks:
        task_names = set(args.tasks)
        tasks = [t for t in all_tasks if t.get('name') in task_names]
        if not tasks:
            print(f"ERROR: None of the requested tasks {args.tasks} found in config.")
            print(f"Available tasks: {[t.get('name') for t in all_tasks]}")
            sys.exit(1)
        print(f"Evaluating {len(tasks)}/{len(all_tasks)} specified tasks.")
    else:
        tasks = all_tasks
    
    # Setup output path
    if args.output:
        output_path = args.output
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(checkpoint_dir, 'evaluation_results.json')
    
    # Setup logger (optional)
    log_dir = os.path.join(os.path.dirname(args.checkpoint), 'eval_logs')
    logger = Logger(
        log_dir=log_dir,
        use_tensorboard=False,
        use_wandb=False,
        verbose=not args.quiet,
    )
    
    # Run evaluation
    start_time = time.time()
    results = evaluate_all_tasks(
        fre_model=fre_model,
        iql_agent=iql_agent,
        replay_buffer=replay_buffer,
        env=env,
        tasks=tasks,
        config=config,
        device=device,
        logger=logger,
    )
    elapsed = time.time() - start_time
    
    # Save results
    output_data = {
        'config_file': args.config,
        'checkpoint': args.checkpoint,
        'seed': seed,
        'device': str(device),
        'domain': domain,
        'task': task,
        'evaluation_time': format_time(elapsed),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
    }
    
    save_json(output_data, output_path)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<30} {'Type':<15} {'Raw Return':<15} {'Normalized':<15}")
    print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    
    for task_name, task_result in results.items():
        if task_name == '_aggregate':
            continue
        raw = task_result.get('raw_return_mean', 0.0)
        norm = task_result.get('normalized_return_mean', 0.0)
        ttype = task_result.get('task_type', 'unknown')
        print(f"{task_name:<30} {ttype:<15} {raw:>14.2f} {norm:>14.1f}")
    
    agg = results.get('_aggregate', {})
    print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'OVERALL AVERAGE':<30} {'':<15} {'':<15} {agg.get('overall_normalized_mean', 0.0):>14.1f}")
    print(f"{'='*70}")
    
    logger.close()
    
    return results


if __name__ == '__main__':
    main()