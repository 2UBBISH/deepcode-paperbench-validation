"""
FRE Training Loop

Implements the full two-phase training procedure from Algorithm 1:

Phase 1: Train the FRE encoder using gradients from the decoder
         (Equation 6), with RL components frozen.
Phase 2: Freeze the encoder, train the RL components (IQL) with
         sampled reward functions from the prior distribution.

Key hyperparameters (from Appendix Table 3):
- Batch size: 512
- Encoder training steps: 150,000 (1M for ExORL/Kitchen)
- Policy training steps: 850,000 (1M for ExORL/Kitchen)
- K (encoder pairs): 32
- K' (decoder pairs): 8
- β (KL weight): 0.01
- Learning rate: 1e-4 (Adam)
- Discount: 0.88
- IQL expectile: 0.8
- AWR temperature: 3.0
- Target update rate: 0.001
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
from copy import deepcopy

from .encoder import FREModel
from .iql import FREIQLAgent


class FREPipeline:
    """
    Complete FRE training and evaluation pipeline.

    Manages the two-phase training:
    1. Encoder phase: train FRE auto-encoder on random reward functions
    2. Policy phase: train FRE-conditioned IQL agent with frozen encoder
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        # FRE hyperparameters
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        num_reward_bins: int = 32,
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        encoder_mlp_dim: int = 256,
        decoder_hidden_dims: list = None,
        beta: float = 0.01,
        # IQL hyperparameters
        rl_hidden_dims: list = None,
        expectile: float = 0.8,
        temperature: float = 3.0,
        discount: float = 0.88,
        target_update_rate: float = 0.001,
        # Training hyperparameters
        lr: float = 1e-4,
        K_encoder: int = 32,
        K_decoder: int = 8,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.K_encoder = K_encoder
        self.K_decoder = K_decoder
        self.device = torch.device(device)

        # FRE model
        self.fre_model = FREModel(
            state_dim=state_dim,
            latent_dim=latent_dim,
            state_embed_dim=state_embed_dim,
            reward_embed_dim=reward_embed_dim,
            num_reward_bins=num_reward_bins,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            mlp_dim=encoder_mlp_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            beta=beta,
        ).to(self.device)

        # IQL agent
        self.rl_agent = FREIQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dims=rl_hidden_dims,
            expectile=expectile,
            temperature=temperature,
            discount=discount,
            target_update_rate=target_update_rate,
        ).to(self.device)

        # Optimizers
        self.encoder_optimizer = torch.optim.Adam(
            self.fre_model.parameters(), lr=lr
        )
        self.rl_optimizer = torch.optim.Adam(
            self.rl_agent.parameters(), lr=lr
        )

        self.encoder_trained = False

    def _sample_states_for_encoder(
        self,
        dataset_states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample K encoder states uniformly from the dataset."""
        N = dataset_states.shape[0]
        indices = torch.randint(0, N, (batch_size, self.K_encoder))
        return dataset_states[indices].to(self.device)

    def _sample_states_for_decoder(
        self,
        dataset_states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample K' decoder states uniformly from the dataset."""
        N = dataset_states.shape[0]
        indices = torch.randint(0, N, (batch_size, self.K_decoder))
        return dataset_states[indices].to(self.device)

    def _evaluate_reward_on_states(
        self,
        reward_fn: Callable,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate a batched reward function on a batch of states.

        reward_fn is expected to handle shape (B, K, D) -> (B, K).
        For simple reward functions, we vectorize over K.
        """
        B, K, D = states.shape
        states_flat = states.view(B * K, D)
        rewards_flat = reward_fn(states_flat)
        return rewards_flat.view(B, K)

    def train_encoder_step(
        self,
        reward_fn: Callable,
        dataset_states: torch.Tensor,
    ) -> dict:
        """
        Single training step for the FRE encoder-decoder (Phase 1).
        """
        B = min(512, dataset_states.shape[0])  # batch size per paper

        # Sample encoder and decoder states
        enc_states = self._sample_states_for_encoder(dataset_states, B)
        dec_states = self._sample_states_for_decoder(dataset_states, B)

        # Evaluate reward function
        enc_rewards = self._evaluate_reward_on_states(reward_fn, enc_states)
        dec_rewards = self._evaluate_reward_on_states(reward_fn, dec_states)

        # Train FRE
        self.encoder_optimizer.zero_grad()
        total_loss, mse_loss, kl_loss = self.fre_model(
            enc_states, enc_rewards, dec_states, dec_rewards
        )
        total_loss.backward()
        self.encoder_optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def train_rl_step(
        self,
        reward_fn: Callable,
        dataset: dict,
    ) -> dict:
        """
        Single training step for FRE-conditioned IQL (Phase 2).

        Args:
            reward_fn: callable that maps states -> rewards
            dataset: dict with keys:
                'states': (N, state_dim)
                'actions': (N, action_dim)
                'next_states': (N, state_dim)
                'dones': (N, 1)
                'all_states': (N_all, state_dim) — for encoder state sampling
        Returns:
            Dictionary of training losses.
        """
        B = min(512, dataset['states'].shape[0])

        # Sample transitions
        N = dataset['states'].shape[0]
        indices = torch.randint(0, N, (B,))
        states = dataset['states'][indices].to(self.device)
        actions = dataset['actions'][indices].to(self.device)
        next_states = dataset['next_states'][indices].to(self.device)
        dones = dataset['dones'][indices].to(self.device)

        # Sample encoder states and compute rewards
        enc_states = self._sample_states_for_encoder(
            dataset['all_states'], B
        )
        enc_rewards = self._evaluate_reward_on_states(reward_fn, enc_states)

        # Encode z from encoder states + rewards (frozen encoder, no grad)
        with torch.no_grad():
            z = self.fre_model.encode(enc_states, enc_rewards)

        # Compute transition rewards
        states_for_reward = states  # (B, state_dim)
        transition_rewards = reward_fn(states_for_reward).unsqueeze(-1)  # (B, 1)

        # IQL training step
        losses = self.rl_agent.train_step(
            states, actions, transition_rewards, next_states, dones, z
        )

        # Update networks
        self.rl_optimizer.zero_grad()
        (losses["critic_loss"] + losses["policy_loss"]).backward()
        self.rl_optimizer.step()

        # Update target networks
        self.rl_agent.update_targets()

        return losses

    def freeze_encoder(self):
        """Freeze encoder weights after Phase 1 convergence."""
        for param in self.fre_model.encoder.parameters():
            param.requires_grad = False
        self.encoder_trained = True

    def encode_reward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a reward function from (state, reward) samples.
        Used at evaluation time for zero-shot task inference.

        Args:
            states:  (K, state_dim) — K reward-annotated states
            rewards: (K,) — corresponding rewards
        Returns:
            z: (latent_dim,) — latent encoding
        """
        with torch.no_grad():
            # Add batch dim
            states_b = states.unsqueeze(0).to(self.device)  # (1, K, D)
            rewards_b = rewards.unsqueeze(0).to(self.device)  # (1, K)
            z = self.fre_model.encode(states_b, rewards_b)  # (1, latent_dim)
        return z.squeeze(0)

    def get_action(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic action from the policy for a given state and latent z.
        Used at evaluation time.
        """
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
            z = z.unsqueeze(0).to(self.device) if z.dim() == 1 else z.to(self.device)
            return self.rl_agent.policy.get_action(state, z).squeeze(0)

    def save(self, path: str):
        """Save full pipeline checkpoint."""
        torch.save({
            'fre_model': self.fre_model.state_dict(),
            'rl_agent': self.rl_agent.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'rl_optimizer': self.rl_optimizer.state_dict(),
            'encoder_trained': self.encoder_trained,
        }, path)

    def load(self, path: str):
        """Load full pipeline checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.fre_model.load_state_dict(ckpt['fre_model'])
        self.rl_agent.load_state_dict(ckpt['rl_agent'])
        self.encoder_optimizer.load_state_dict(ckpt['encoder_optimizer'])
        self.rl_optimizer.load_state_dict(ckpt['rl_optimizer'])
        self.encoder_trained = ckpt['encoder_trained']