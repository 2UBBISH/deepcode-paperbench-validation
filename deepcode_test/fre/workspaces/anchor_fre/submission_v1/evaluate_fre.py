"""
Evaluation script for FRE agents.

Tests trained FRE agents on downstream tasks in a zero-shot manner.
"""

import argparse
import numpy as np
import torch
import gym
import d4rl
from tqdm import tqdm
import os

from fre import FREEncoder, FREIQL
from environments import (
    AntMazeGoalReaching,
    AntMazeDirectional,
    AntMazeRandomSimplex,
    AntMazePath
)


def encode_reward_function(encoder, reward_fn, states, num_samples=32, device='cuda'):
    """
    Encode a reward function into latent z using sample states.

    Args:
        encoder: FREEncoder
        reward_fn: Reward function to encode
        states: Pool of states to sample from
        num_samples: Number of (state, reward) pairs to use (K)
        device: Device

    Returns:
        z: Encoded latent vector
    """
    # Sample random states
    indices = np.random.choice(len(states), size=num_samples, replace=False)
    sample_states = states[indices]

    # Evaluate reward function
    rewards = reward_fn(sample_states)

    # Convert to tensors
    states_t = torch.FloatTensor(sample_states).unsqueeze(0).to(device)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        z, _, _ = encoder.encode(states_t, rewards_t)
        z = z.squeeze(0)  # (latent_dim,)

    return z


def evaluate_task(env, policy, z, reward_fn, num_episodes=20, max_steps=2000):
    """
    Evaluate policy on a task.

    Args:
        env: Gym environment
        policy: Policy network
        z: Task encoding
        reward_fn: Reward function for the task
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        mean_return: Average return over episodes
        std_return: Standard deviation of returns
    """
    returns = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        prev_state = state

        for step in range(max_steps):
            # Select action
            action = policy.select_action(state, z.cpu().numpy(), deterministic=True)

            # Step environment
            next_state, _, done, _ = env.step(action)

            # Compute reward using task reward function
            # Some reward functions need both current and next state
            try:
                reward = reward_fn(state, next_state)
            except:
                reward = reward_fn(state)

            episode_return += reward
            state = next_state

            if done:
                break

        returns.append(episode_return)

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    return mean_return, std_return


def normalize_score(score, task_name):
    """
    Normalize score to [0, 100] range.
    This is a simplified normalization - in practice you'd need
    task-specific normalization based on random and expert performance.
    """
    # Placeholder normalization
    # In the paper, scores are normalized per task
    # For now, just clip to reasonable range
    return np.clip(score, 0, 100)


def evaluate_antmaze(checkpoint_path, env_name='antmaze-large-diverse-v2',
                     num_eval_samples=32, device='cuda'):
    """
    Evaluate FRE agent on AntMaze tasks.

    Args:
        checkpoint_path: Path to saved checkpoint
        env_name: D4RL environment name
        num_eval_samples: Number of samples for encoding (K)
        device: Device
    """
    # Load environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load dataset for sampling states
    dataset = env.get_dataset()
    all_states = dataset['observations'].astype(np.float32)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create encoder
    encoder = FREEncoder(state_dim=state_dim).to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    # Create IQL agent
    iql_agent = FREIQL(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    iql_agent.policy.load_state_dict(checkpoint['policy'])
    iql_agent.policy.eval()

    # Evaluation tasks
    eval_tasks = {
        'ant-goal-reaching': AntMazeGoalReaching.get_all_tasks(),
        'ant-directional': AntMazeDirectional.get_all_tasks(),
        'ant-random-simplex': AntMazeRandomSimplex.get_all_tasks(),
        'ant-path-center': [AntMazePath('center')],
        'ant-path-loop': [AntMazePath('loop')],
        'ant-path-edges': [AntMazePath('edges')],
    }

    results = {}

    for task_category, tasks in eval_tasks.items():
        print(f"\nEvaluating {task_category}...")
        category_scores = []

        for task in tasks:
            # Encode task
            z = encode_reward_function(encoder, task, all_states,
                                      num_samples=num_eval_samples, device=device)

            # Evaluate
            mean_return, std_return = evaluate_task(env, iql_agent.policy, z, task)

            print(f"  {task.__class__.__name__}: {mean_return:.2f} ± {std_return:.2f}")
            category_scores.append(mean_return)

        # Average across tasks in category
        avg_score = np.mean(category_scores)
        results[task_category] = {
            'mean': avg_score,
            'std': np.std(category_scores),
            'individual': category_scores
        }

    # Print summary
    print("\n=== Summary ===")
    for task_category, result in results.items():
        print(f"{task_category}: {result['mean']:.2f} ± {result['std']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2',
                       help='Environment name')
    parser.add_argument('--num_eval_samples', type=int, default=32,
                       help='Number of samples for encoding reward function')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Evaluate
    if 'antmaze' in args.env.lower():
        results = evaluate_antmaze(
            args.checkpoint,
            args.env,
            args.num_eval_samples,
            args.device
        )
    else:
        raise NotImplementedError(f"Evaluation not implemented for {args.env}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
