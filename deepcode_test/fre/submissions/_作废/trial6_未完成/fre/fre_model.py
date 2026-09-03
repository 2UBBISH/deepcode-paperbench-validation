"""
FRE Model: Full Functional Reward Encoding VAE.

Combines the permutation-invariant Transformer encoder and feedforward decoder
into a variational autoencoder that learns a compact latent representation of
reward functions. Trained on unsupervised random reward functions from the
MixedPrior distribution.

Architecture:
    Encoder: (states, rewards) -> Transformer -> aggregation -> (mu, logvar) -> z
    Decoder: (states, z) -> MLP -> predicted rewards

Loss (Equation 6):
    L_FRE = MSE(rewards_true, rewards_pred) + beta * KL(N(mu, sigma) || N(0, I))

Training follows Algorithm 1 (encoder phase) from the paper:
    while not converged:
        Sample reward function eta ~ p(eta)
        Sample K encoder states and K' decoder states from dataset
        Compute rewards using eta
        Forward pass, compute loss, backprop
"""

from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from fre.encoder import FREEncoder, create_fre_encoder
from fre.decoder import RewardDecoder, create_reward_decoder
from fre.prior import MixedPrior, RewardFunction
from fre.dataset import OfflineDataset


class FREModel(nn.Module):
    """
    Full Functional Reward Encoding VAE model.

    Combines encoder and decoder for joint training on unsupervised
    reward functions. The encoder learns to compress reward function
    information into a latent vector z, and the decoder learns to
    reconstruct rewards from z.

    Attributes:
        encoder: FREEncoder (Transformer VAE encoder)
        decoder: RewardDecoder (feedforward reward predictor)
        beta: KL divergence weight in the VAE loss
        state_dim: Dimensionality of the state space
        d_latent: Dimensionality of the latent space
    """

    def __init__(
        self,
        encoder: FREEncoder,
        decoder: RewardDecoder,
        beta: float = 0.1,
        state_dim: Optional[int] = None,
        d_latent: Optional[int] = None,
    ):
        """
        Initialize the FRE model.

        Args:
            encoder: FREEncoder instance
            decoder: RewardDecoder instance
            beta: KL divergence weight (default 0.1)
            state_dim: State dimensionality (inferred if not provided)
            d_latent: Latent dimensionality (inferred if not provided)
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.state_dim = state_dim or encoder.state_dim
        self.d_latent = d_latent or encoder.d_latent

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode reward function, decode rewards.

        Args:
            encoder_states: (batch, K, state_dim) encoder state set
            encoder_rewards: (batch, K) rewards for encoder states
            decoder_states: (batch, K', state_dim) decoder state set

        Returns:
            z: (batch, d_latent) sampled latent vectors
            mu: (batch, d_latent) posterior mean
            logvar: (batch, d_latent) posterior log variance
            kl: (batch,) KL divergence per sample
            pred_rewards: (batch, K') predicted rewards for decoder states
        """
        # Encode: get latent distribution and sample z
        z, mu, logvar, kl = self.encoder(encoder_states, encoder_rewards)

        # Decode: predict rewards for decoder states
        pred_rewards = self.decoder(decoder_states, z)

        return z, mu, logvar, kl, pred_rewards

    def encode_reward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Encode a reward function into a latent vector z.

        Args:
            states: (batch, K, state_dim) or (K, state_dim) encoder states
            rewards: (batch, K) or (K,) rewards
            deterministic: If True, use mean (no sampling); if False, sample

        Returns:
            z: (batch, d_latent) or (d_latent,) latent vector
        """
        # Handle single sample (no batch dim)
        single_input = states.dim() == 2
        if single_input:
            states = states.unsqueeze(0)  # (1, K, state_dim)
            rewards = rewards.unsqueeze(0)  # (1, K)

        if deterministic:
            z = self.encoder.encode_deterministic(states, rewards)
        else:
            mu, logvar = self.encoder.encode(states, rewards)
            z, _ = self.encoder.reparameterize(mu, logvar)

        if single_input:
            z = z.squeeze(0)

        return z

    def decode_reward(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict rewards for states given latent vector z.

        Args:
            states: (batch, K', state_dim) or (K', state_dim) states
            z: (batch, d_latent) or (d_latent,) latent vector

        Returns:
            pred_rewards: (batch, K') or (K',) predicted rewards
        """
        single_input = states.dim() == 2
        if single_input:
            states = states.unsqueeze(0)
            z = z.unsqueeze(0)

        pred_rewards = self.decoder(states, z)

        if single_input:
            pred_rewards = pred_rewards.squeeze(0)

        return pred_rewards

    def compute_loss(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the VAE loss for a batch.

        Args:
            encoder_states: (batch, K, state_dim)
            encoder_rewards: (batch, K)
            decoder_states: (batch, K', state_dim)
            decoder_rewards: (batch, K') ground truth rewards

        Returns:
            Dictionary with keys: 'loss', 'mse', 'kl', 'mse_unweighted'
        """
        z, mu, logvar, kl, pred_rewards = self.forward(
            encoder_states, encoder_rewards, decoder_states
        )

        # MSE reconstruction loss (mean over batch and decoder states)
        mse = nn.functional.mse_loss(pred_rewards, decoder_rewards, reduction='mean')

        # KL divergence (mean over batch)
        kl_mean = kl.mean()

        # Total VAE loss
        loss = mse + self.beta * kl_mean

        return {
            'loss': loss,
            'mse': mse,
            'kl': kl_mean,
            'mse_unweighted': mse.detach(),
        }

    def train_step(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            encoder_states: (batch, K, state_dim)
            encoder_rewards: (batch, K)
            decoder_states: (batch, K', state_dim)
            decoder_rewards: (batch, K')
            optimizer: PyTorch optimizer
            clip_grad_norm: Optional gradient clipping value

        Returns:
            Dictionary of loss values (detached floats)
        """
        self.train()
        optimizer.zero_grad()

        loss_dict = self.compute_loss(
            encoder_states, encoder_rewards,
            decoder_states, decoder_rewards,
        )

        loss_dict['loss'].backward()

        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)

        optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}

    @torch.no_grad()
    def evaluate_reconstruction(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality without updating parameters.

        Args:
            encoder_states: (batch, K, state_dim)
            encoder_rewards: (batch, K)
            decoder_states: (batch, K', state_dim)
            decoder_rewards: (batch, K')

        Returns:
            Dictionary with loss values and correlation metrics
        """
        self.eval()

        z, mu, logvar, kl, pred_rewards = self.forward(
            encoder_states, encoder_rewards, decoder_states
        )

        mse = nn.functional.mse_loss(pred_rewards, decoder_rewards, reduction='mean')
        kl_mean = kl.mean()
        loss = mse + self.beta * kl_mean

        # Compute Pearson correlation between predicted and true rewards
        pred_flat = pred_rewards.reshape(-1)
        true_flat = decoder_rewards.reshape(-1)
        pred_mean = pred_flat.mean()
        true_mean = true_flat.mean()
        pred_centered = pred_flat - pred_mean
        true_centered = true_flat - true_mean
        numerator = (pred_centered * true_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (true_centered ** 2).sum())
        correlation = (numerator / (denominator + 1e-8)).item()

        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'kl': kl_mean.item(),
            'correlation': correlation,
        }

    def get_encoder_parameters(self):
        """Return encoder parameters (for separate optimization)."""
        return self.encoder.parameters()

    def get_decoder_parameters(self):
        """Return decoder parameters (for separate optimization)."""
        return self.decoder.parameters()

    def freeze_encoder(self):
        """Freeze encoder parameters (for IQL training phase)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """Unfreeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = True

    def save(self, path: str):
        """
        Save model state dict and configuration.

        Args:
            path: File path for saving
        """
        state = {
            'model_state_dict': self.state_dict(),
            'encoder_config': self.encoder.get_config(),
            'decoder_config': {
                'state_dim': self.decoder.state_dim,
                'd_latent': self.decoder.d_latent,
                'hidden_dims': self.decoder.hidden_dims,
            },
            'beta': self.beta,
            'state_dim': self.state_dim,
            'd_latent': self.d_latent,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, map_location: str = 'cpu') -> 'FREModel':
        """
        Load a saved FRE model.

        Args:
            path: File path to load from
            map_location: Device mapping

        Returns:
            FREModel instance with loaded weights
        """
        checkpoint = torch.load(path, map_location=map_location)

        # Reconstruct encoder
        encoder_config = checkpoint['encoder_config']
        encoder = create_fre_encoder(**encoder_config)

        # Reconstruct decoder
        decoder_config = checkpoint['decoder_config']
        decoder = create_reward_decoder(**decoder_config)

        # Create model
        model = cls(
            encoder=encoder,
            decoder=decoder,
            beta=checkpoint['beta'],
            state_dim=checkpoint.get('state_dim'),
            d_latent=checkpoint.get('d_latent'),
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class FRETrainer:
    """
    Standalone trainer for the FRE model (Phase 1 of strided training).

    Handles the training loop: sampling reward functions from the prior,
    sampling states from the dataset, computing rewards, and updating
    the encoder and decoder.

    This is used by the main trainer (fre/trainer.py) for the encoder
    pre-training phase.
    """

    def __init__(
        self,
        model: FREModel,
        prior: MixedPrior,
        dataset: OfflineDataset,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        K_encoder: int = 32,
        K_decoder: int = 128,
        batch_size: int = 256,
        clip_grad_norm: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        """
        Initialize the FRE trainer.

        Args:
            model: FREModel instance
            prior: MixedPrior for sampling reward functions
            dataset: OfflineDataset for sampling states
            optimizer: PyTorch optimizer
            device: Computation device
            K_encoder: Number of encoder states per reward function
            K_decoder: Number of decoder states per reward function
            batch_size: Number of reward functions per training step
            clip_grad_norm: Optional gradient clipping value
            rng: Random state for reproducibility
        """
        self.model = model
        self.prior = prior
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = device
        self.K_encoder = K_encoder
        self.K_decoder = K_decoder
        self.batch_size = batch_size
        self.clip_grad_norm = clip_grad_norm
        self.rng = rng if rng is not None else np.random.RandomState()

        self.model.to(self.device)

        # Training statistics
        self.train_step_count = 0
        self.log_history: List[Dict[str, float]] = []

    def sample_training_batch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of reward functions and states for training.

        For each of batch_size reward functions:
        - Sample K_encoder states and compute rewards (for encoder)
        - Sample K_decoder states and compute rewards (for decoder)

        Returns:
            encoder_states: (batch_size, K_encoder, state_dim)
            encoder_rewards: (batch_size, K_encoder)
            decoder_states: (batch_size, K_decoder, state_dim)
            decoder_rewards: (batch_size, K_decoder)
        """
        B = self.batch_size
        K_e = self.K_encoder
        K_d = self.K_decoder
        state_dim = self.model.state_dim

        # Sample reward functions
        reward_fns = self.prior.sample_batch(B, self.rng)

        # Sample all states at once (more efficient)
        total_states_needed = B * (K_e + K_d)
        all_states = self.dataset.sample_random_norm_states(total_states_needed)
        all_states = torch.from_numpy(all_states).float().to(self.device)

        # Split into encoder and decoder states
        encoder_states_flat = all_states[:B * K_e]
        decoder_states_flat = all_states[B * K_e:]

        # Compute rewards for each reward function
        encoder_rewards_list = []
        decoder_rewards_list = []

        for i, rf in enumerate(reward_fns):
            # Encoder states for this reward function
            e_states = encoder_states_flat[i * K_e:(i + 1) * K_e].cpu().numpy()
            e_rewards = rf(e_states)
            encoder_rewards_list.append(torch.from_numpy(e_rewards).float())

            # Decoder states for this reward function
            d_states = decoder_states_flat[i * K_d:(i + 1) * K_d].cpu().numpy()
            d_rewards = rf(d_states)
            decoder_rewards_list.append(torch.from_numpy(d_rewards).float())

        # Reshape to batch tensors
        encoder_states = encoder_states_flat.view(B, K_e, state_dim)
        encoder_rewards = torch.stack(encoder_rewards_list).to(self.device)
        decoder_states = decoder_states_flat.view(B, K_d, state_dim)
        decoder_rewards = torch.stack(decoder_rewards_list).to(self.device)

        return encoder_states, encoder_rewards, decoder_states, decoder_rewards

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step: sample batch, compute loss, update.

        Returns:
            Dictionary of loss values
        """
        encoder_states, encoder_rewards, decoder_states, decoder_rewards = \
            self.sample_training_batch()

        loss_dict = self.model.train_step(
            encoder_states=encoder_states,
            encoder_rewards=encoder_rewards,
            decoder_states=decoder_states,
            decoder_rewards=decoder_rewards,
            optimizer=self.optimizer,
            clip_grad_norm=self.clip_grad_norm,
        )

        self.train_step_count += 1
        self.log_history.append(loss_dict)

        return loss_dict

    @torch.no_grad()
    def validate(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Run validation: evaluate reconstruction on multiple batches.

        Args:
            num_batches: Number of validation batches

        Returns:
            Averaged validation metrics
        """
        self.model.eval()
        metrics_sum = {'loss': 0.0, 'mse': 0.0, 'kl': 0.0, 'correlation': 0.0}

        for _ in range(num_batches):
            encoder_states, encoder_rewards, decoder_states, decoder_rewards = \
                self.sample_training_batch()

            metrics = self.model.evaluate_reconstruction(
                encoder_states, encoder_rewards,
                decoder_states, decoder_rewards,
            )

            for k in metrics_sum:
                metrics_sum[k] += metrics[k]

        self.model.train()

        return {k: v / num_batches for k, v in metrics_sum.items()}

    def get_recent_losses(self, window: int = 100) -> Dict[str, float]:
        """Get average losses over the last `window` training steps."""
        if not self.log_history:
            return {'loss': 0.0, 'mse': 0.0, 'kl': 0.0}

        recent = self.log_history[-window:]
        avg = {}
        for key in recent[0].keys():
            avg[key] = np.mean([h[key] for h in recent])
        return avg


def create_fre_model(
    state_dim: int,
    d_model: int = 256,
    d_reward: int = 32,
    d_latent: int = 64,
    num_bins: int = 50,
    r_min: float = -1.0,
    r_max: float = 1.0,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.0,
    decoder_hidden_dims: Optional[List[int]] = None,
    beta: float = 0.1,
) -> FREModel:
    """
    Factory function to create a complete FRE model with default hyperparameters.

    Args:
        state_dim: Dimensionality of the state space
        d_model: Transformer model dimension (default 256)
        d_reward: Reward embedding dimension (default 32)
        d_latent: Latent space dimension (default 64)
        num_bins: Number of reward discretization bins (default 50)
        r_min: Minimum reward value (default -1.0)
        r_max: Maximum reward value (default 1.0)
        nhead: Number of attention heads (default 4)
        num_layers: Number of Transformer layers (default 4)
        dim_feedforward: Transformer feedforward dimension (default 1024)
        dropout: Dropout rate (default 0.0)
        decoder_hidden_dims: Decoder MLP hidden dimensions (default [256, 256])
        beta: KL divergence weight (default 0.1)

    Returns:
        FREModel instance
    """
    if decoder_hidden_dims is None:
        decoder_hidden_dims = [256, 256]

    encoder = create_fre_encoder(
        state_dim=state_dim,
        d_model=d_model,
        d_reward=d_reward,
        d_latent=d_latent,
        num_bins=num_bins,
        r_min=r_min,
        r_max=r_max,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    decoder = create_reward_decoder(
        state_dim=state_dim,
        d_latent=d_latent,
        hidden_dims=decoder_hidden_dims,
        dropout=dropout,
    )

    model = FREModel(
        encoder=encoder,
        decoder=decoder,
        beta=beta,
        state_dim=state_dim,
        d_latent=d_latent,
    )

    return model


def create_fre_trainer(
    model: FREModel,
    prior: MixedPrior,
    dataset: OfflineDataset,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device('cpu'),
    K_encoder: int = 32,
    K_decoder: int = 128,
    batch_size: int = 256,
    clip_grad_norm: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None,
) -> FRETrainer:
    """
    Factory function to create an FRETrainer with Adam optimizer.

    Args:
        model: FREModel instance
        prior: MixedPrior instance
        dataset: OfflineDataset instance
        learning_rate: Adam learning rate (default 1e-4)
        device: Computation device
        K_encoder: Encoder states per reward function (default 32)
        K_decoder: Decoder states per reward function (default 128)
        batch_size: Reward functions per training step (default 256)
        clip_grad_norm: Optional gradient clipping
        rng: Random state

    Returns:
        FRETrainer instance
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = FRETrainer(
        model=model,
        prior=prior,
        dataset=dataset,
        optimizer=optimizer,
        device=device,
        K_encoder=K_encoder,
        K_decoder=K_decoder,
        batch_size=batch_size,
        clip_grad_norm=clip_grad_norm,
        rng=rng,
    )

    return trainer