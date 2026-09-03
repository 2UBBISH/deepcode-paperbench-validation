"""
Training script for baseline methods (GC-IQL and GC-BC).

Trains goal-conditioned policies using hindsight relabeling.
"""

import argparse
import numpy as np
import torch
import gym
import d4rl
from tqdm import tqdm
import os

from baselines import GCIQL, GCBC


def segment_dataset_into_trajectories(dataset):
    """
    Segment flat dataset into trajectories.

    Args:
        dataset: D4RL dataset dictionary

    Returns:
        List of trajectories, each containing states, actions, etc.
    """
    states = dataset['observations']
    actions = dataset['actions']
    next_states = dataset['next_observations']
    rewards = dataset['rewards']
    dones = dataset['terminals']

    # Handle timeouts
    if 'timeouts' in dataset:
        dones = np.logical_or(dones, dataset['timeouts'])

    trajectories = []
    current_traj = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
    }

    for i in range(len(states)):
        current_traj['states'].append(states[i])
        current_traj['actions'].append(actions[i])
        current_traj['next_states'].append(next_states[i])
        current_traj['rewards'].append(rewards[i])

        if dones[i]:
            # End of trajectory
            for key in current_traj:
                current_traj[key] = np.array(current_traj[key])
            trajectories.append(current_traj)

            # Start new trajectory
            current_traj = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
            }

    # Add last trajectory if not empty
    if len(current_traj['states']) > 0:
        for key in current_traj:
            current_traj[key] = np.array(current_traj[key])
        trajectories.append(current_traj)

    return trajectories


def train_gc_iql(env_name, args):
    """Train GC-IQL."""
    print("\n=== Training GC-IQL ===")

    # Load environment and dataset
    env = gym.make(env_name)
    dataset = env.get_dataset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Segment into trajectories
    trajectories = segment_dataset_into_trajectories(dataset)
    print(f"Dataset: {len(trajectories)} trajectories")

    # Create agent
    agent = GCIQL(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        expectile=args.expectile,
        temperature=args.temperature,
        device=args.device
    )

    # Training loop
    for step in tqdm(range(args.num_steps), desc="Training GC-IQL"):
        # Sample trajectory
        traj_idx = np.random.randint(len(trajectories))
        traj = trajectories[traj_idx]

        # Sample transitions from trajectory
        batch_indices = np.random.choice(len(traj['states']),
                                        size=min(args.batch_size, len(traj['states'])),
                                        replace=False)

        batch_states = []
        batch_actions = []
        batch_next_states = []
        batch_goals = []
        batch_rewards = []
        batch_dones = []

        for idx in batch_indices:
            state = traj['states'][idx]
            action = traj['actions'][idx]
            next_state = traj['next_states'][idx]

            # Sample goal with hindsight relabeling
            goal = agent.sample_goal(traj_idx, idx, {'states': [traj['states']]})

            # Compute goal-conditioned reward
            reward = agent.compute_goal_reward(state, goal)
            done = 1.0 if reward == 0.0 else 0.0  # Done when goal reached

            batch_states.append(state)
            batch_actions.append(action)
            batch_next_states.append(next_state)
            batch_goals.append(goal)
            batch_rewards.append(reward)
            batch_dones.append(done)

        # Convert to tensors
        batch_states = torch.FloatTensor(np.array(batch_states)).to(args.device)
        batch_actions = torch.FloatTensor(np.array(batch_actions)).to(args.device)
        batch_next_states = torch.FloatTensor(np.array(batch_next_states)).to(args.device)
        batch_goals = torch.FloatTensor(np.array(batch_goals)).to(args.device)
        batch_rewards = torch.FloatTensor(np.array(batch_rewards)).to(args.device)
        batch_dones = torch.FloatTensor(np.array(batch_dones)).to(args.device)

        # Update
        losses = agent.train_step(batch_states, batch_actions, batch_next_states,
                                 batch_goals, batch_rewards, batch_dones)

        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}: Q Loss = {losses['q_loss']:.4f}, "
                  f"V Loss = {losses['v_loss']:.4f}, "
                  f"Policy Loss = {losses['policy_loss']:.4f}")

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'gc_iql_{env_name}.pt')
    torch.save({
        'q1': agent.q1.state_dict(),
        'q2': agent.q2.state_dict(),
        'v': agent.v.state_dict(),
        'policy': agent.policy.state_dict(),
    }, save_path)

    print(f"Model saved to {save_path}")


def train_gc_bc(env_name, args):
    """Train GC-BC."""
    print("\n=== Training GC-BC ===")

    # Load environment and dataset
    env = gym.make(env_name)
    dataset = env.get_dataset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Segment into trajectories
    trajectories = segment_dataset_into_trajectories(dataset)
    print(f"Dataset: {len(trajectories)} trajectories")

    # Create agent
    agent = GCBC(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        device=args.device
    )

    # Training loop
    for step in tqdm(range(args.num_steps), desc="Training GC-BC"):
        # Sample trajectory
        traj_idx = np.random.randint(len(trajectories))
        traj = trajectories[traj_idx]

        # Sample transitions from trajectory
        batch_indices = np.random.choice(len(traj['states']),
                                        size=min(args.batch_size, len(traj['states'])),
                                        replace=False)

        batch_states = []
        batch_actions = []
        batch_goals = []

        for idx in batch_indices:
            state = traj['states'][idx]
            action = traj['actions'][idx]

            # Sample goal using geometric distribution (future states only)
            goal = agent.sample_goal_geometric(traj['states'], idx)

            batch_states.append(state)
            batch_actions.append(action)
            batch_goals.append(goal)

        # Convert to tensors
        batch_states = torch.FloatTensor(np.array(batch_states)).to(args.device)
        batch_actions = torch.FloatTensor(np.array(batch_actions)).to(args.device)
        batch_goals = torch.FloatTensor(np.array(batch_goals)).to(args.device)

        # Update
        loss = agent.train_step(batch_states, batch_actions, batch_goals)

        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}: Loss = {loss:.4f}")

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'gc_bc_{env_name}.pt')
    torch.save({
        'policy': agent.policy.state_dict(),
    }, save_path)

    print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument('--method', type=str, required=True, choices=['gc-iql', 'gc-bc'],
                       help='Baseline method to train')

    # Environment
    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2',
                       help='D4RL environment name')

    # Training
    parser.add_argument('--num_steps', type=int, default=1000000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    # IQL parameters (for GC-IQL)
    parser.add_argument('--gamma', type=float, default=0.88,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.001,
                       help='Target network update rate')
    parser.add_argument('--expectile', type=float, default=0.8,
                       help='IQL expectile')
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='AWR temperature')

    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Save directory')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Train
    if args.method == 'gc-iql':
        train_gc_iql(args.env, args)
    elif args.method == 'gc-bc':
        train_gc_bc(args.env, args)


if __name__ == '__main__':
    main()
