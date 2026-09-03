"""
Main training and evaluation script for Functional Reward Encoding (FRE).

Replicates the zero-shot offline RL experiments from:
"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"
Frans, Park, Abbeel, Levine (ICML 2024).

Usage:
    # Train FRE-all on AntMaze
    python train.py --env antmaze --method fre --prior fre-all

    # Train FRE-goals on AntMaze
    python train.py --env antmaze --method fre --prior fre-goals

    # Train GC-IQL baseline on AntMaze
    python train.py --env antmaze --method gc-iql

    # Evaluate a trained agent
    python train.py --env antmaze --method fre --eval --checkpoint path/to/ckpt
"""

import argparse
import os
import sys
import torch
import numpy as np
from typing import Dict, Optional, Callable

from fre import FREPipeline
from fre import MixedRewardPrior, GoalReachingReward, RandomLinearReward, RandomMLPReward
from fre import FREZeroShotEvaluator, EvaluationTask
from baselines import GoalConditionedBC, GCIQL, OPAL
from environments import OfflineDataset
from utils import discretize_xy, normalize_states, HindsightRelabeler


# ============================================================
# Training entry points per method
# ============================================================

def train_fre(
    env_name: str,
    prior_type: str = "fre-all",
    state_dim: int = None,
    action_dim: int = None,
    dataset: OfflineDataset = None,
    encoder_steps: int = 150_000,
    policy_steps: int = 850_000,
    batch_size: int = 512,
    latent_dim: int = 128,
    beta: float = 0.01,
    lr: float = 1e-4,
    K_encoder: int = 32,
    K_decoder: int = 8,
    expectile: float = 0.8,
    temperature: float = 3.0,
    discount: float = 0.88,
    target_update_rate: float = 0.001,
    device: str = "cpu",
    log_interval: int = 1000,
    save_path: str = None,
):
    """
    Full FRE training pipeline (Algorithm 1).

    Phase 1: Train FRE encoder-decoder on random reward functions.
    Phase 2: Freeze encoder, train FRE-conditioned IQL policy.
    """
    if dataset is None:
        dataset = OfflineDataset(env_name, device=device)

    S = dataset.state_dim
    A = dataset.action_dim

    # Build reward prior based on prior_type
    ratios = _get_prior_ratios(prior_type)

    # Config for AntMaze: exclude XY from linear rewards
    linear_exclude_xy = ("antmaze" in env_name.lower())

    prior = MixedRewardPrior(
        state_dim=S,
        dataset_states=dataset.all_states,
        ratios=ratios,
        linear_exclude_xy=linear_exclude_xy,
        xy_indices=(0, 1),
    )

    pipeline = FREPipeline(
        state_dim=S,
        action_dim=A,
        latent_dim=latent_dim,
        beta=beta,
        lr=lr,
        K_encoder=K_encoder,
        K_decoder=K_decoder,
        expectile=expectile,
        temperature=temperature,
        discount=discount,
        target_update_rate=target_update_rate,
        device=device,
    )

    # ========================
    # Phase 1: Train encoder
    # ========================
    print(f"[Phase 1] Training FRE encoder for {encoder_steps} steps...")
    for step in range(encoder_steps):
        # Sample one batch of reward function (each batch element is a separate η)
        reward_fn, _, _, _ = prior.sample(batch_size, device=device)

        # For each reward function in the batch, sample K encoder & K' decoder states
        enc_states = dataset.sample_encoder_states(batch_size, K_encoder).to(device)
        dec_states = dataset.sample_decoder_states(batch_size, K_decoder).to(device)

        # Evaluate reward functions on states
        enc_rewards = _eval_reward_batch(reward_fn, enc_states)
        dec_rewards = _eval_reward_batch(reward_fn, dec_states)

        # Train FRE encoder-decoder
        total_loss, mse_loss, kl_loss = pipeline.fre_model(
            enc_states, enc_rewards, dec_states, dec_rewards
        )
        pipeline.encoder_optimizer.zero_grad()
        total_loss.backward()
        pipeline.encoder_optimizer.step()

        if step % log_interval == 0 or step == encoder_steps - 1:
            print(f"  Step {step}: total={total_loss.item():.4f}, "
                  f"mse={mse_loss.item():.4f}, kl={kl_loss.item():.4f}")

    # Freeze encoder for Phase 2
    pipeline.freeze_encoder()
    print("[Phase 1] Encoder training complete. Encoder frozen.")

    # ========================
    # Phase 2: Train policy
    # ========================
    print(f"[Phase 2] Training FRE-conditioned IQL for {policy_steps} steps...")
    for step in range(policy_steps):
        # Sample transitions
        batch = dataset.sample_batch(batch_size)
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)
        rewards_transition = batch['rewards'].to(device)
        next_states = batch['next_states'].to(device)
        dones = batch['dones'].to(device)

        # Sample reward function and encode z
        reward_fn, _, _, _ = prior.sample(batch_size, device=device)
        enc_states = dataset.sample_encoder_states(batch_size, K_encoder).to(device)

        with torch.no_grad():
            enc_rewards = _eval_reward_batch(reward_fn, enc_states)
            z = pipeline.fre_model.encode(enc_states, enc_rewards)

        # Compute transition rewards from the sampled reward function
        transition_rewards = _eval_reward_single(reward_fn, states).unsqueeze(-1)

        # IQL training step
        vf_loss = pipeline.rl_agent._expectile_loss(
            pipeline.rl_agent.qf(states, actions, z).detach()
            - pipeline.rl_agent.vf(states, z)
        ).mean()

        with torch.no_grad():
            next_v = pipeline.rl_agent.target_vf(next_states, z)
            q_target = transition_rewards + discount * (1 - dones) * next_v

        q_val = pipeline.rl_agent.qf(states, actions, z)
        qf_loss = torch.nn.functional.mse_loss(q_val, q_target)

        with torch.no_grad():
            adv = q_val.detach() - pipeline.rl_agent.vf(states, z).detach()
            exp_adv = torch.exp(adv / temperature).clamp(max=100.0)

        _, log_prob, _ = pipeline.rl_agent.policy.sample(states, z)
        pi_loss = -(exp_adv * log_prob).mean()

        critic_loss = vf_loss + qf_loss
        total_rl_loss = critic_loss + pi_loss

        pipeline.rl_optimizer.zero_grad()
        total_rl_loss.backward()
        pipeline.rl_optimizer.step()

        pipeline.rl_agent.update_targets()

        if step % log_interval == 0 or step == policy_steps - 1:
            print(f"  Step {step}: vf={vf_loss.item():.4f}, "
                  f"qf={qf_loss.item():.4f}, pi={pi_loss.item():.4f}")

    print("[Phase 2] Policy training complete.")

    if save_path:
        pipeline.save(save_path)
        print(f"Saved checkpoint to {save_path}")

    return pipeline


def train_gc_bc(
    env_name: str,
    dataset: OfflineDataset = None,
    num_steps: int = 1_000_000,
    batch_size: int = 512,
    lr: float = 1e-4,
    device: str = "cpu",
    log_interval: int = 1000,
    save_path: str = None,
):
    """Train GC-BC baseline."""
    if dataset is None:
        dataset = OfflineDataset(env_name, device=device)

    S, A = dataset.state_dim, dataset.action_dim
    agent = GoalConditionedBC(state_dim=S, action_dim=A, lr=lr, device=device)

    print(f"Training GC-BC for {num_steps} steps...")
    for step in range(num_steps):
        # Sample transitions and goals
        batch = dataset.sample_batch(batch_size)
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)

        # HER goal sampling (GC-BC uses geometric only per addendum)
        goals = _sample_geometric_goals(dataset, batch_size, states)

        loss = agent.train_step(states, actions, goals)

        if step % log_interval == 0 or step == num_steps - 1:
            print(f"  Step {step}: loss={loss:.4f}")

    print("GC-BC training complete.")
    return agent


def train_gc_iql(
    env_name: str,
    dataset: OfflineDataset = None,
    num_steps: int = 1_000_000,
    batch_size: int = 512,
    lr: float = 1e-4,
    expectile: float = 0.8,
    temperature: float = 3.0,
    discount: float = 0.88,
    target_update_rate: float = 0.001,
    device: str = "cpu",
    log_interval: int = 1000,
    save_path: str = None,
):
    """Train GC-IQL baseline."""
    if dataset is None:
        dataset = OfflineDataset(env_name, device=device)

    S, A = dataset.state_dim, dataset.action_dim
    agent = GCIQL(
        state_dim=S,
        action_dim=A,
        expectile=expectile,
        temperature=temperature,
        discount=discount,
        target_update_rate=target_update_rate,
        lr=lr,
        device=device,
    )

    her = HindsightRelabeler(p_random=0.3, p_geometric=0.5, p_current=0.2)

    print(f"Training GC-IQL for {num_steps} steps...")
    for step in range(num_steps):
        batch = dataset.sample_batch(batch_size)
        states = batch['states'].to(device)
        actions = batch['actions'].to(device)
        next_states = batch['next_states'].to(device)
        dones = batch['dones'].to(device)

        # Sample HER goals
        goals, her_rewards, her_dones = _sample_her_goals(
            dataset, her, batch_size, states
        )

        losses = agent.train_step(states, actions, next_states, dones, goals, her_rewards)

        if step % log_interval == 0 or step == num_steps - 1:
            print(f"  Step {step}: vf={losses['value_loss']:.4f}, "
                  f"qf={losses['q_loss']:.4f}, pi={losses['policy_loss']:.4f}")

    print("GC-IQL training complete.")
    return agent


def train_opal(
    env_name: str,
    dataset: OfflineDataset = None,
    num_steps: int = 1_000_000,
    batch_size: int = 512,
    chunk_length: int = 10,
    lr: float = 1e-4,
    device: str = "cpu",
    log_interval: int = 1000,
    save_path: str = None,
):
    """Train OPAL baseline."""
    if dataset is None:
        dataset = OfflineDataset(env_name, device=device)

    S, A = dataset.state_dim, dataset.action_dim
    agent = OPAL(
        state_dim=S, action_dim=A, lr=lr,
        chunk_length=chunk_length, device=device,
    )

    print(f"Training OPAL for {num_steps} steps...")
    for step in range(num_steps):
        # Sample trajectory chunks
        chunk_states, chunk_actions = _sample_chunks(
            dataset, batch_size, chunk_length
        )
        chunk_states = chunk_states.to(device)
        chunk_actions = chunk_actions.to(device)

        losses = agent.train_step(chunk_states, chunk_actions)

        if step % log_interval == 0 or step == num_steps - 1:
            print(f"  Step {step}: total={losses['total_loss']:.4f}, "
                  f"nll={losses['nll']:.4f}, kl={losses['kl_loss']:.4f}")

    print("OPAL training complete.")
    return agent


# ============================================================
# Helper functions
# ============================================================

def _get_prior_ratios(prior_type: str):
    """Map prior type name to (goal_ratio, linear_ratio, mlp_ratio)."""
    mapping = {
        "fre-all":      (0.33, 0.33, 0.34),
        "fre-goals":    (1.0,  0.0,  0.0),
        "fre-lin":      (0.0,  1.0,  0.0),
        "fre-mlp":      (0.0,  0.0,  1.0),
        "fre-lin-mlp":  (0.0,  0.5,  0.5),
        "fre-goal-mlp": (0.5,  0.0,  0.5),
        "fre-goal-lin": (0.5,  0.5,  0.0),
        "fre-hint":     (0.33, 0.33, 0.34),  # same ratios, different prior distribution
    }
    return mapping.get(prior_type, (0.33, 0.33, 0.34))


def _eval_reward_batch(reward_fn, states):
    """Evaluate a batch of reward functions on batched states (B, K, D) -> (B, K)."""
    B, K, D = states.shape
    states_flat = states.view(B * K, D)
    rewards_flat = reward_fn(states_flat)
    return rewards_flat.view(B, K)


def _eval_reward_single(reward_fn, states):
    """Evaluate a batch of reward functions on single states (B, D) -> (B,)."""
    return reward_fn(states)


def _sample_geometric_goals(dataset, batch_size, current_states):
    """Sample goals using geometric distribution for GC-BC."""
    N = dataset.all_states.shape[0]
    # Simplified: sample random future states
    indices = torch.randint(0, N, (batch_size,))
    return dataset.all_states[indices].to(current_states.device)


def _sample_her_goals(dataset, her, batch_size, current_states):
    """Sample HER goals and compute goal-conditioned rewards."""
    goals, her_rewards, her_dones = her.sample_goals(
        batch_size, dataset.all_states
    )
    # For current state as goal: reward = 0, done = True
    # For other goals: reward = -1, done = False
    her_rewards = her_rewards.to(current_states.device).unsqueeze(-1)
    her_dones = her_dones.to(current_states.device).unsqueeze(-1)
    return goals.to(current_states.device), her_rewards, her_dones


def _sample_chunks(dataset, batch_size, chunk_length):
    """Sample trajectory chunks for OPAL training."""
    N = dataset.states.shape[0]
    # Simplified: contiguous chunks from random start positions
    start_indices = torch.randint(0, N - chunk_length, (batch_size,))
    indices = start_indices.unsqueeze(-1) + torch.arange(chunk_length).unsqueeze(0)
    indices = indices.clamp(0, N - 1)
    return dataset.states[indices], dataset.actions[indices]


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FRE: Functional Reward Encodings")

    parser.add_argument("--env", type=str, default="antmaze",
                        choices=["antmaze", "walker", "cheetah", "kitchen"],
                        help="Environment/domain to train on")
    parser.add_argument("--method", type=str, default="fre",
                        choices=["fre", "gc-bc", "gc-iql", "opal", "fb", "sf"],
                        help="Method to train/evaluate")
    parser.add_argument("--prior", type=str, default="fre-all",
                        choices=["fre-all", "fre-goals", "fre-lin", "fre-mlp",
                                 "fre-lin-mlp", "fre-goal-mlp", "fre-goal-lin", "fre-hint"],
                        help="Prior reward distribution (FRE only)")
    parser.add_argument("--eval", action="store_true",
                        help="Run zero-shot evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save checkpoint")
    parser.add_argument("--encoder-steps", type=int, default=150_000,
                        help="Number of encoder training steps (FRE only)")
    parser.add_argument("--policy-steps", type=int, default=850_000,
                        help="Number of policy training steps")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=1000,
                        help="Logging interval")

    args = parser.parse_args()

    print(f"=== FRE Zero-Shot RL ===")
    print(f"Environment: {args.env}")
    print(f"Method: {args.method}")
    if args.method == "fre":
        print(f"Prior: {args.prior}")

    if args.eval and args.checkpoint:
        print(f"Evaluation mode. Loading checkpoint: {args.checkpoint}")
        # Load and evaluate (placeholder — needs actual env integration)
        print("Evaluation requires environment integration (D4RL/DeepMind Control). "
              "See environments/env_wrappers.py for interfaces.")
        return

    if args.method == "fre":
        train_fre(
            env_name=args.env,
            prior_type=args.prior,
            encoder_steps=args.encoder_steps,
            policy_steps=args.policy_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            log_interval=args.log_interval,
            save_path=args.save,
        )
    elif args.method == "gc-bc":
        train_gc_bc(
            env_name=args.env,
            num_steps=args.policy_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            log_interval=args.log_interval,
            save_path=args.save,
        )
    elif args.method == "gc-iql":
        train_gc_iql(
            env_name=args.env,
            num_steps=args.policy_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            log_interval=args.log_interval,
            save_path=args.save,
        )
    elif args.method == "opal":
        train_opal(
            env_name=args.env,
            num_steps=args.policy_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            log_interval=args.log_interval,
            save_path=args.save,
        )
    elif args.method in ("fb", "sf"):
        print(f"{args.method.upper()} requires facebookresearch/controllable_agent. "
              "See baselines/fb_sf.py for wrapper interfaces.")


if __name__ == "__main__":
    main()