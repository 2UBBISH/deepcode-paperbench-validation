"""
Training script for Functional Reward Encodings (FRE).

Implements the two-phase training procedure:
1. Train encoder-decoder network
2. Freeze encoder and train IQL policy
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import gym
import d4rl
from tqdm import tqdm
import os
import json

from fre import FREEncoder, FREDecoder, RewardFunctionSampler, FREIQL


class ReplayBuffer:
    """Simple replay buffer for offline RL."""

    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.size = len(states)

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }

    def get_all_states(self):
        """Get all states in the dataset."""
        return self.states


def load_d4rl_dataset(env_name):
    """Load D4RL dataset and create replay buffer."""
    env = gym.make(env_name)
    dataset = env.get_dataset()

    states = dataset['observations'].astype(np.float32)
    actions = dataset['actions'].astype(np.float32)
    next_states = dataset['next_observations'].astype(np.float32)
    rewards = dataset['rewards'].astype(np.float32)
    dones = dataset['terminals'].astype(np.float32)

    # Handle timeouts
    if 'timeouts' in dataset:
        dones = np.logical_or(dones, dataset['timeouts']).astype(np.float32)

    print(f"Loaded dataset: {env_name}")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Dataset size: {len(states)}")

    buffer = ReplayBuffer(states, actions, rewards, next_states, dones)
    return buffer, env


def train_encoder_decoder(encoder, decoder, reward_sampler, buffer, args):
    """
    Phase 1: Train FRE encoder-decoder network.

    Args:
        encoder: FREEncoder
        decoder: FREDecoder
        reward_sampler: RewardFunctionSampler
        buffer: ReplayBuffer
        args: Training arguments
    """
    print("\n=== Phase 1: Training Encoder-Decoder ===")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr
    )

    all_states = buffer.get_all_states()
    num_states = len(all_states)

    for step in tqdm(range(args.encoder_steps), desc="Encoder training"):
        # Sample reward function
        reward_fn = reward_sampler.sample()

        # Sample K encoder states and K' decoder states
        encoder_indices = np.random.choice(num_states, size=args.K, replace=False)
        decoder_indices = np.random.choice(num_states, size=args.K_prime, replace=False)

        encoder_states = all_states[encoder_indices]
        decoder_states = all_states[decoder_indices]

        # Evaluate reward function on sampled states
        encoder_rewards = reward_fn(encoder_states)
        decoder_rewards = reward_fn(decoder_states)

        # Convert to tensors
        encoder_states_t = torch.FloatTensor(encoder_states).unsqueeze(0).to(args.device)
        encoder_rewards_t = torch.FloatTensor(encoder_rewards).unsqueeze(0).to(args.device)
        decoder_states_t = torch.FloatTensor(decoder_states).unsqueeze(0).to(args.device)
        decoder_rewards_t = torch.FloatTensor(decoder_rewards).to(args.device)

        # Encode
        z, mean, logstd = encoder.encode(encoder_states_t, encoder_rewards_t)

        # Decode
        pred_rewards = decoder(decoder_states_t.squeeze(0), z.squeeze(0))

        # Compute losses
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(pred_rewards, decoder_rewards_t)

        # KL divergence loss
        kl_loss = encoder.compute_kl_loss(mean, logstd)

        # Total loss
        total_loss = reconstruction_loss + args.beta * kl_loss

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Logging
        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}/{args.encoder_steps}: "
                  f"Recon Loss = {reconstruction_loss.item():.4f}, "
                  f"KL Loss = {kl_loss.item():.4f}")

    print("Encoder-decoder training complete!\n")


def train_policy(encoder, iql_agent, reward_sampler, buffer, args):
    """
    Phase 2: Train IQL policy with frozen encoder.

    Args:
        encoder: Frozen FREEncoder
        iql_agent: FREIQL agent
        reward_sampler: RewardFunctionSampler
        buffer: ReplayBuffer
        args: Training arguments
    """
    print("\n=== Phase 2: Training Policy with IQL ===")

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    all_states = buffer.get_all_states()
    num_states = len(all_states)

    for step in tqdm(range(args.policy_steps), desc="Policy training"):
        # Sample reward function
        reward_fn = reward_sampler.sample()

        # Sample K encoder states to encode the reward function
        encoder_indices = np.random.choice(num_states, size=args.K, replace=False)
        encoder_states = all_states[encoder_indices]
        encoder_rewards = reward_fn(encoder_states)

        # Encode reward function to get z
        with torch.no_grad():
            encoder_states_t = torch.FloatTensor(encoder_states).unsqueeze(0).to(args.device)
            encoder_rewards_t = torch.FloatTensor(encoder_rewards).unsqueeze(0).to(args.device)
            z, _, _ = encoder.encode(encoder_states_t, encoder_rewards_t)
            z = z.squeeze(0)  # (latent_dim,)

        # Sample batch of transitions from replay buffer
        batch = buffer.sample(args.batch_size)

        # Compute rewards for the transitions using the sampled reward function
        batch_rewards = reward_fn(batch['states'])

        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(args.device)
        actions = torch.FloatTensor(batch['actions']).to(args.device)
        rewards = torch.FloatTensor(batch_rewards).to(args.device)
        next_states = torch.FloatTensor(batch['next_states']).to(args.device)
        dones = torch.FloatTensor(batch['dones']).to(args.device)

        # Expand z to batch size
        z_batch = z.unsqueeze(0).expand(args.batch_size, -1)

        # Update IQL
        losses = iql_agent.train_step(states, actions, rewards, next_states, dones, z_batch)

        # Logging
        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}/{args.policy_steps}: "
                  f"Q Loss = {losses['q_loss']:.4f}, "
                  f"V Loss = {losses['v_loss']:.4f}, "
                  f"Policy Loss = {losses['policy_loss']:.4f}")

    print("Policy training complete!\n")


def main():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env', type=str, default='antmaze-large-diverse-v2',
                       help='D4RL environment name')

    # Training
    parser.add_argument('--encoder_steps', type=int, default=150000,
                       help='Number of encoder training steps')
    parser.add_argument('--policy_steps', type=int, default=850000,
                       help='Number of policy training steps')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')

    # FRE parameters
    parser.add_argument('--K', type=int, default=32,
                       help='Number of encoder state-reward pairs')
    parser.add_argument('--K_prime', type=int, default=8,
                       help='Number of decoder state-reward pairs')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension')
    parser.add_argument('--beta', type=float, default=0.01,
                       help='KL divergence weight')

    # IQL parameters
    parser.add_argument('--gamma', type=float, default=0.88,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.001,
                       help='Target network update rate')
    parser.add_argument('--expectile', type=float, default=0.8,
                       help='IQL expectile')
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='AWR temperature')

    # Reward function parameters
    parser.add_argument('--goal_ratio', type=float, default=0.33,
                       help='Ratio of goal-reaching rewards')
    parser.add_argument('--linear_ratio', type=float, default=0.33,
                       help='Ratio of linear rewards')
    parser.add_argument('--mlp_ratio', type=float, default=0.34,
                       help='Ratio of MLP rewards')

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

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    buffer, env = load_d4rl_dataset(args.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create networks
    encoder = FREEncoder(
        state_dim=state_dim,
        latent_dim=args.latent_dim,
        beta=args.beta
    ).to(args.device)

    decoder = FREDecoder(
        state_dim=state_dim,
        latent_dim=args.latent_dim
    ).to(args.device)

    # Create reward function sampler
    # For AntMaze, exclude XY positions from linear functions (first 2 dims)
    exclude_dims = [0, 1] if 'antmaze' in args.env.lower() else None

    # Create simple dataset wrapper for reward sampler
    class SimpleDataset:
        def __init__(self, states):
            self.states = states

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return {'observations': self.states[idx]}

    dataset = SimpleDataset(buffer.get_all_states())

    reward_sampler = RewardFunctionSampler(
        dataset=dataset,
        state_dim=state_dim,
        goal_ratio=args.goal_ratio,
        linear_ratio=args.linear_ratio,
        mlp_ratio=args.mlp_ratio,
        exclude_dims=exclude_dims
    )

    # Phase 1: Train encoder-decoder
    train_encoder_decoder(encoder, decoder, reward_sampler, buffer, args)

    # Save encoder-decoder
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
    }, os.path.join(args.save_dir, f'fre_encoder_{args.env}.pt'))

    # Create IQL agent
    iql_agent = FREIQL(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        expectile=args.expectile,
        temperature=args.temperature,
        device=args.device
    )

    # Phase 2: Train policy
    train_policy(encoder, iql_agent, reward_sampler, buffer, args)

    # Save full model
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'q1': iql_agent.q1.state_dict(),
        'q2': iql_agent.q2.state_dict(),
        'v': iql_agent.v.state_dict(),
        'policy': iql_agent.policy.state_dict(),
    }, os.path.join(args.save_dir, f'fre_full_{args.env}.pt'))

    print(f"Training complete! Models saved to {args.save_dir}")


if __name__ == '__main__':
    main()
