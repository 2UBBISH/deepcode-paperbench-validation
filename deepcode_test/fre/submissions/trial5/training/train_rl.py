"""
Phase 2: Train IQL agent with frozen FRE encoder.

This module implements the second phase of the FRE training pipeline:
- Freeze the pre-trained FRE encoder
- Sample random reward functions from the prior distribution
- Encode reward functions into latent z using the frozen encoder
- Train IQL agent (Q, V, policy) conditioned on z using offline RL

Reference: Algorithm 1 (Phase 2) from "Functional Reward Encodings for
Zero-Shot Offline Reinforcement Learning"
"""

import logging
import time
from collections import deque
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.fre_encoder import FREEncoder
from models.iql_agent import IQLAgent
from rewards.mixture import MixtureRewardDistribution
from data.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class IQLTrainer:
    """
    Trainer for Phase 2: Offline RL with IQL conditioned on frozen FRE encoder.

    This trainer:
    1. Samples a random reward function η from the prior mixture distribution
    2. Encodes η into latent z using the frozen FRE encoder
    3. Computes rewards for RL batch using the same η
    4. Updates Q, V, and policy networks using IQL losses

    The encoder is kept in eval mode and its parameters are frozen.
    """

    def __init__(
        self,
        encoder: FREEncoder,
        agent: IQLAgent,
        reward_dist: MixtureRewardDistribution,
        replay_buffer: ReplayBuffer,
        learning_rate: float = 3e-4,
        K_enc: int = 64,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the IQL trainer.

        Args:
            encoder: Pre-trained FRE encoder (will be frozen)
            agent: IQL agent (Q, V, policy networks)
            reward_dist: Mixture distribution over reward function families
            replay_buffer: Replay buffer with offline dataset
            learning_rate: Learning rate for Adam optimizer
            K_enc: Number of encoding states to sample for computing z
            batch_size: Batch size for RL updates
            device: Torch device for computation
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = encoder.to(self.device)
        self.agent = agent.to(self.device)
        self.reward_dist = reward_dist
        self.replay_buffer = replay_buffer

        self.K_enc = K_enc
        self.batch_size = batch_size

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Optimizer for IQL agent only (encoder is frozen)
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=learning_rate,
        )

        # Training state
        self.train_step = 0
        self.train_losses: Dict[str, deque] = {
            "value_loss": deque(maxlen=1000),
            "q_loss": deque(maxlen=1000),
            "policy_loss": deque(maxlen=1000),
            "total_loss": deque(maxlen=1000),
        }

        # Logging
        self.logger = logger

    def _encode_reward_function(
        self, reward_fn, rng: Optional[np.random.RandomState] = None
    ) -> torch.Tensor:
        """
        Encode a reward function into latent z using the frozen encoder.

        Args:
            reward_fn: Callable reward function η(s) -> rewards
            rng: Random number generator for reproducibility

        Returns:
            Latent vector z of shape (1, latent_dim)
        """
        # Sample encoding states from replay buffer
        enc_states = self.replay_buffer.sample_states(self.K_enc, rng=rng)
        enc_states_tensor = torch.from_numpy(enc_states).float().to(self.device)

        # Compute rewards for encoding states
        with torch.no_grad():
            enc_rewards = reward_fn(enc_states_tensor)

        # Encode to latent z (deterministic: use mu only)
        with torch.no_grad():
            z = self.encoder.encode_deterministic(enc_states_tensor, enc_rewards)

        return z  # shape: (1, latent_dim)

    def train_step(
        self, rng: Optional[np.random.RandomState] = None
    ) -> Dict[str, float]:
        """
        Perform a single training step: sample η, encode z, update IQL.

        Args:
            rng: Random number generator for reproducibility

        Returns:
            Dictionary of loss values for logging
        """
        self.agent.train()

        # 1. Sample a random reward function from the prior
        reward_fn = self.reward_dist.sample(
            dataset_states=self.replay_buffer.get_all_states()
        )

        # 2. Encode reward function to latent z
        z = self._encode_reward_function(reward_fn, rng=rng)

        # 3. Sample RL batch with rewards computed by the same reward function
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_rl_batch(
            self.batch_size, reward_fn=reward_fn, rng=rng
        )

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 4. Compute IQL losses
        value_loss = self.agent.compute_value_loss(states, actions, z)
        q_loss = self.agent.compute_q_loss(states, actions, rewards, next_states, dones, z)
        policy_loss = self.agent.compute_policy_loss(states, actions, z)

        total_loss = value_loss + q_loss + policy_loss

        # 5. Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=10.0)

        self.optimizer.step()

        # 6. Update target networks
        self.agent.update_targets()

        # 7. Log losses
        self.train_step += 1

        loss_dict = {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "total_loss": total_loss.item(),
        }

        for key, value in loss_dict.items():
            self.train_losses[key].append(value)

        return loss_dict

    def train(
        self,
        num_steps: int,
        log_interval: int = 1000,
        eval_interval: int = 10000,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the Phase 2 training loop for a specified number of steps.

        Args:
            num_steps: Total number of training steps
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            rng_seed: Random seed for reproducibility
            verbose: Whether to print progress

        Returns:
            Dictionary of training statistics
        """
        rng = np.random.RandomState(rng_seed)
        start_time = time.time()
        stats = {
            "step_losses": [],
            "eval_results": [],
        }

        self.logger.info(f"Starting IQL training for {num_steps} steps...")

        for step in range(num_steps):
            loss_dict = self.train_step(rng=rng)

            if verbose and (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                avg_losses = {
                    key: np.mean(list(vals)) if vals else 0.0
                    for key, vals in self.train_losses.items()
                }
                self.logger.info(
                    f"Step {step + 1}/{num_steps} | "
                    f"V: {avg_losses['value_loss']:.4f} | "
                    f"Q: {avg_losses['q_loss']:.4f} | "
                    f"π: {avg_losses['policy_loss']:.4f} | "
                    f"Total: {avg_losses['total_loss']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
                stats["step_losses"].append((step + 1, avg_losses))

            if (step + 1) % eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=5, rng=rng)
                stats["eval_results"].append((step + 1, eval_stats))
                if verbose:
                    self.logger.info(
                        f"Eval @ step {step + 1}: "
                        f"avg_return={eval_stats['avg_return']:.2f} ± "
                        f"{eval_stats['std_return']:.2f}"
                    )

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s ({total_time/3600:.2f}h)")

        return stats

    def evaluate(
        self,
        num_episodes: int = 10,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the current policy on a random reward function from the prior.

        Note: This is an offline evaluation using the dataset, not environment
        interaction. For environment evaluation, use evaluation/evaluator.py.

        Args:
            num_episodes: Number of evaluation episodes (reward functions)
            rng: Random number generator

        Returns:
            Dictionary with average and std of returns
        """
        self.agent.eval()

        returns = []
        for _ in range(num_episodes):
            # Sample a reward function
            reward_fn = self.reward_dist.sample(
                dataset_states=self.replay_buffer.get_all_states()
            )

            # Encode to z
            z = self._encode_reward_function(reward_fn, rng=rng)

            # Sample a batch and compute average reward under current policy
            # (offline evaluation: we can't actually roll out, so we estimate
            #  by computing Q-values on dataset states)
            states = self.replay_buffer.sample_states(1024, rng=rng)
            states_tensor = torch.from_numpy(states).float().to(self.device)

            with torch.no_grad():
                # Get actions from policy
                actions, _, _ = self.agent.policy.sample(states_tensor, z)
                # Compute Q-values as proxy for return
                q_values = self.agent.q1(states_tensor, actions, z)
                avg_q = q_values.mean().item()

            returns.append(avg_q)

        self.agent.train()

        return {
            "avg_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get the trainer state for checkpointing."""
        return {
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state from checkpoint."""
        self.agent.load_state_dict(state_dict["agent"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.train_step = state_dict.get("train_step", 0)

    def save_checkpoint(self, path: str):
        """Save trainer checkpoint to file."""
        checkpoint = {
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load trainer checkpoint from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step = checkpoint.get("train_step", 0)
        self.logger.info(f"Checkpoint loaded from {path} (step {self.train_step})")

    def to(self, device: torch.device):
        """Move all components to the specified device."""
        self.device = device
        self.encoder = self.encoder.to(device)
        self.agent = self.agent.to(device)
        self.replay_buffer = self.replay_buffer.to(device)
        return self