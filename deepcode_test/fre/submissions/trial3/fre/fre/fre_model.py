"""
FRE VAE Model: Joint training of encoder and decoder.

This module implements the full Functional Reward Encodings (FRE) variational
autoencoder. It combines the permutation-invariant transformer encoder and the
feedforward reward decoder, and provides the training loop for Phase 1 of the
FRE pipeline: unsupervised pre-training on random reward functions.

The loss function (Equation 6 in the paper, standard β-VAE):
    L = (1/K') Σ MSE(η(s_k^d), decoder(s_k^d, z)) + β * KL(N(μ, σ²) || N(0, I))

where z is sampled from the encoder given K encoding state-reward pairs,
and the decoder predicts rewards for K' decoding states.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .encoder import FREEncoder
from .decoder import RewardDecoder
from .reward_prior import RewardPrior


class FREModel(nn.Module):
    """
    Full FRE Variational Autoencoder model.

    Combines the permutation-invariant transformer encoder with the
    feedforward reward decoder. Trained on random unsupervised reward
    functions to learn a latent representation of arbitrary reward functions.

    Args:
        state_dim: Dimensionality of the state space.
        latent_dim: Dimensionality of the latent z vector (default: 64).
        d_model: Transformer hidden dimension (default: 256).
        num_layers: Number of transformer layers (default: 2).
        num_heads: Number of attention heads (default: 4).
        d_ff: Feedforward dimension in transformer (default: 1024).
        d_emb: Reward embedding dimension (default: 64).
        num_bins: Number of reward discretization bins (default: 100).
        reward_min: Minimum expected reward value (default: -10.0).
        reward_max: Maximum expected reward value (default: 10.0).
        decoder_hidden_dims: Hidden layer dimensions for decoder MLP
            (default: [256, 256]).
        beta: KL divergence weight (β-VAE parameter, default: 0.1).
        dropout: Dropout rate for both encoder and decoder (default: 0.0).
        max_num_states: Maximum number of encoding/decoding states (default: 32).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 1024,
        d_emb: int = 64,
        num_bins: int = 100,
        reward_min: float = -10.0,
        reward_max: float = 10.0,
        decoder_hidden_dims: Optional[list] = None,
        beta: float = 0.1,
        dropout: float = 0.0,
        max_num_states: int = 32,
    ):
        super().__init__()

        if decoder_hidden_dims is None:
            decoder_hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.max_num_states = max_num_states

        # Build encoder
        self.encoder = FREEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            d_emb=d_emb,
            num_bins=num_bins,
            reward_min=reward_min,
            reward_max=reward_max,
            dropout=dropout,
            max_num_states=max_num_states,
        )

        # Build decoder
        self.decoder = RewardDecoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation="relu",
            dropout=dropout,
        )

        # Track training statistics
        self.train_step_counter = 0

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full FRE VAE.

        Args:
            encoder_states: Tensor of shape (batch_size, K, state_dim) containing
                the encoding state-reward pairs.
            encoder_rewards: Tensor of shape (batch_size, K) containing the
                rewards for encoding states.
            decoder_states: Tensor of shape (batch_size, K', state_dim) containing
                the decoding states for reconstruction.
            encoder_mask: Optional mask of shape (batch_size, K) for variable-length
                encoding sets (1 = valid, 0 = padding).

        Returns:
            Dictionary containing:
                - 'decoder_preds': Predicted rewards for decoding states
                  (batch_size, K').
                - 'z': Sampled latent vector (batch_size, latent_dim).
                - 'mu': Mean of latent Gaussian (batch_size, latent_dim).
                - 'logvar': Log variance of latent Gaussian (batch_size, latent_dim).
                - 'mse_loss': Reconstruction MSE loss (scalar).
                - 'kl_loss': KL divergence loss (scalar).
                - 'total_loss': Total loss = mse_loss + beta * kl_loss (scalar).
        """
        batch_size = encoder_states.shape[0]

        # 1. Encode: get latent distribution parameters and sample z
        z, mu, logvar = self.encoder(
            encoder_states, encoder_rewards, mask=encoder_mask
        )

        # 2. Decode: predict rewards for decoding states
        decoder_preds = self.decoder(decoder_states, z)  # (batch_size, K')

        # 3. Compute losses
        # MSE reconstruction loss (averaged over decoding states and batch)
        mse_loss = F.mse_loss(
            decoder_preds,
            torch.zeros_like(decoder_preds),  # placeholder, actual loss computed in training_step
            reduction='none'
        ).mean()

        # KL divergence
        kl_loss = self.encoder.kl_divergence(mu, logvar)

        # Total loss
        total_loss = mse_loss + self.beta * kl_loss

        return {
            'decoder_preds': decoder_preds,
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
        }

    def compute_loss(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full FRE VAE loss.

        Args:
            encoder_states: (batch_size, K, state_dim)
            encoder_rewards: (batch_size, K)
            decoder_states: (batch_size, K', state_dim)
            decoder_rewards: (batch_size, K') - ground truth rewards for decoding states
            encoder_mask: Optional (batch_size, K)

        Returns:
            Dictionary with loss components.
        """
        batch_size = encoder_states.shape[0]
        K_prime = decoder_states.shape[1]

        # Encode
        z, mu, logvar = self.encoder(
            encoder_states, encoder_rewards, mask=encoder_mask
        )

        # Decode
        decoder_preds = self.decoder(decoder_states, z)  # (batch_size, K')

        # MSE reconstruction loss: average over decoding states and batch
        mse_loss = F.mse_loss(decoder_preds, decoder_rewards, reduction='mean')

        # KL divergence (already averaged over batch and latent dims)
        kl_loss = self.encoder.kl_divergence(mu, logvar)

        # Total loss
        total_loss = mse_loss + self.beta * kl_loss

        return {
            'decoder_preds': decoder_preds,
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'mse_loss': mse_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
        }

    def encode_rewards(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a set of state-reward pairs into a latent vector z.

        This is the primary interface used during RL training (Phase 2)
        and zero-shot evaluation.

        Args:
            states: (batch_size, K, state_dim) or (K, state_dim)
            rewards: (batch_size, K) or (K,)
            mask: Optional mask.

        Returns:
            z: (batch_size, latent_dim) or (latent_dim,)
        """
        # Handle single (non-batched) input
        single_input = states.dim() == 2
        if single_input:
            states = states.unsqueeze(0)  # (1, K, state_dim)
            rewards = rewards.unsqueeze(0)  # (1, K)
            if mask is not None:
                mask = mask.unsqueeze(0)

        z, _, _ = self.encoder(states, rewards, mask=mask)

        if single_input:
            z = z.squeeze(0)

        return z

    def decode_rewards(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent vector z to predict rewards for given states.

        Args:
            states: (batch_size, num_states, state_dim) or (num_states, state_dim)
            z: (batch_size, latent_dim) or (latent_dim,)

        Returns:
            Predicted rewards: (batch_size, num_states) or (num_states,)
        """
        return self.decoder(states, z)

    def get_encoder_parameters(self):
        """Return encoder parameters (for separate optimization if needed)."""
        return self.encoder.parameters()

    def get_decoder_parameters(self):
        """Return decoder parameters (for separate optimization if needed)."""
        return self.decoder.parameters()


class FRETrainer:
    """
    Trainer for the FRE VAE model (Phase 1: unsupervised pre-training).

    Handles the training loop: sampling reward functions from the prior,
    generating state-reward pairs, computing the VAE loss, and updating
    model parameters.

    Args:
        model: FREModel instance.
        reward_prior: RewardPrior instance for sampling reward functions.
        dataset_states: numpy array of all states from the offline dataset
            (num_total_states, state_dim). Used for sampling encoding and
            decoding states.
        learning_rate: Learning rate for Adam optimizer (default: 1e-4).
        weight_decay: Weight decay for AdamW (default: 1e-5).
        beta: KL divergence weight (default: 0.1, can override model's beta).
        K_encoder: Number of encoding states (default: 32).
        K_decoder: Number of decoding states (default: 32).
        device: Torch device to use (default: 'cpu').
        use_amp: Whether to use automatic mixed precision (default: False).
    """

    def __init__(
        self,
        model: FREModel,
        reward_prior: RewardPrior,
        dataset_states: np.ndarray,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        beta: Optional[float] = None,
        K_encoder: int = 32,
        K_decoder: int = 32,
        device: str = 'cpu',
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.reward_prior = reward_prior
        self.dataset_states = dataset_states
        self.K_encoder = K_encoder
        self.K_decoder = K_decoder
        self.device = device
        self.use_amp = use_amp

        # Override beta if provided
        if beta is not None:
            self.model.beta = beta

        # Optimizer: AdamW with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler (optional)
        self.scheduler = None

        # Scaler for automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Training statistics
        self.train_step = 0
        self.total_mse_loss = 0.0
        self.total_kl_loss = 0.0
        self.total_loss = 0.0
        self.log_interval = 100

    def sample_encoding_states(self, batch_size: int) -> np.ndarray:
        """
        Sample K_encoder states uniformly from the dataset for each batch element.

        Args:
            batch_size: Number of independent samples in the batch.

        Returns:
            Array of shape (batch_size, K_encoder, state_dim).
        """
        num_total = len(self.dataset_states)
        indices = np.random.randint(0, num_total, size=(batch_size, self.K_encoder))
        return self.dataset_states[indices]  # (batch_size, K_encoder, state_dim)

    def sample_decoding_states(
        self,
        encoder_indices: Optional[np.ndarray] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Sample K_decoder states uniformly from the dataset, disjoint from
        encoding states if encoder_indices is provided.

        Args:
            encoder_indices: Indices of encoding states to avoid overlap.
                Shape (batch_size, K_encoder) or None.
            batch_size: Number of independent samples.

        Returns:
            Array of shape (batch_size, K_decoder, state_dim).
        """
        num_total = len(self.dataset_states)
        decoder_indices = np.zeros((batch_size, self.K_decoder), dtype=np.int64)

        for b in range(batch_size):
            if encoder_indices is not None:
                # Exclude encoding state indices for this batch element
                exclude = set(encoder_indices[b].tolist())
                available = [i for i in range(num_total) if i not in exclude]
                if len(available) < self.K_decoder:
                    # If not enough states, sample with replacement from all states
                    chosen = np.random.choice(num_total, size=self.K_decoder, replace=True)
                else:
                    chosen = np.random.choice(available, size=self.K_decoder, replace=False)
            else:
                chosen = np.random.choice(num_total, size=self.K_decoder, replace=False)
            decoder_indices[b] = chosen

        return self.dataset_states[decoder_indices]

    def training_step(self, batch_size: int = 1) -> Dict[str, float]:
        """
        Perform a single training step for the FRE VAE.

        Steps:
        1. Sample a reward function η from the prior.
        2. Sample K encoding states and K' decoding states.
        3. Compute rewards for both sets using η.
        4. Forward pass through encoder and decoder.
        5. Compute loss (MSE + β * KL).
        6. Backpropagate and update parameters.

        Args:
            batch_size: Number of independent reward functions to sample
                per step (default: 1, but can be increased for efficiency).

        Returns:
            Dictionary with loss values for logging.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 1. Sample encoding states
        encoder_states_np = self.sample_encoding_states(batch_size)
        encoder_indices = None  # We don't track indices in this simple version

        # 2. Sample decoding states (disjoint from encoding)
        decoder_states_np = self.sample_decoding_states(
            encoder_indices=encoder_indices, batch_size=batch_size
        )

        # 3. Sample reward functions and compute rewards
        encoder_rewards_list = []
        decoder_rewards_list = []

        for b in range(batch_size):
            family, reward_fn = self.reward_prior.sample()

            # Compute rewards for encoding states
            enc_r = self.reward_prior.compute_rewards(
                reward_fn, encoder_states_np[b]
            )  # (K_encoder,)
            encoder_rewards_list.append(enc_r)

            # Compute rewards for decoding states
            dec_r = self.reward_prior.compute_rewards(
                reward_fn, decoder_states_np[b]
            )  # (K_decoder,)
            decoder_rewards_list.append(dec_r)

        encoder_rewards_np = np.stack(encoder_rewards_list, axis=0)  # (B, K_encoder)
        decoder_rewards_np = np.stack(decoder_rewards_list, axis=0)  # (B, K_decoder)

        # 4. Convert to tensors
        encoder_states = torch.from_numpy(encoder_states_np).float().to(self.device)
        encoder_rewards = torch.from_numpy(encoder_rewards_np).float().to(self.device)
        decoder_states = torch.from_numpy(decoder_states_np).float().to(self.device)
        decoder_rewards = torch.from_numpy(decoder_rewards_np).float().to(self.device)

        # 5. Forward pass and loss computation
        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = self.model.compute_loss(
                    encoder_states, encoder_rewards,
                    decoder_states, decoder_rewards,
                )
                loss = loss_dict['total_loss']
        else:
            loss_dict = self.model.compute_loss(
                encoder_states, encoder_rewards,
                decoder_states, decoder_rewards,
            )
            loss = loss_dict['total_loss']

        # 6. Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Update statistics
        self.train_step += 1
        mse_val = loss_dict['mse_loss'].item()
        kl_val = loss_dict['kl_loss'].item()
        total_val = loss.item()

        self.total_mse_loss += mse_val
        self.total_kl_loss += kl_val
        self.total_loss += total_val

        return {
            'mse_loss': mse_val,
            'kl_loss': kl_val,
            'total_loss': total_val,
            'train_step': self.train_step,
        }

    def train(
        self,
        num_steps: int,
        batch_size: int = 1,
        log_interval: int = 100,
        eval_interval: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Run the full FRE pre-training loop.

        Args:
            num_steps: Total number of training steps.
            batch_size: Batch size per step.
            log_interval: How often to log training statistics.
            eval_interval: How often to run evaluation.
            verbose: Whether to print progress.

        Returns:
            Dictionary containing training history (losses over time).
        """
        history = {
            'step': [],
            'mse_loss': [],
            'kl_loss': [],
            'total_loss': [],
        }

        for step in range(num_steps):
            loss_info = self.training_step(batch_size=batch_size)

            if (step + 1) % log_interval == 0 or step == 0:
                avg_mse = self.total_mse_loss / max(1, log_interval)
                avg_kl = self.total_kl_loss / max(1, log_interval)
                avg_total = self.total_loss / max(1, log_interval)

                history['step'].append(step + 1)
                history['mse_loss'].append(avg_mse)
                history['kl_loss'].append(avg_kl)
                history['total_loss'].append(avg_total)

                if verbose:
                    print(
                        f"FRE Step {step + 1:6d}/{num_steps} | "
                        f"MSE: {avg_mse:.6f} | "
                        f"KL: {avg_kl:.6f} | "
                        f"Total: {avg_total:.6f}"
                    )

                # Reset accumulators
                self.total_mse_loss = 0.0
                self.total_kl_loss = 0.0
                self.total_loss = 0.0

            # Step scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()

        return history

    def evaluate_reconstruction(
        self,
        num_samples: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate the model's reconstruction quality on held-out reward functions.

        Args:
            num_samples: Number of reward functions to evaluate.

        Returns:
            Dictionary with average MSE, KL, and total loss.
        """
        self.model.eval()
        total_mse = 0.0
        total_kl = 0.0
        total_loss = 0.0

        with torch.no_grad():
            for _ in range(num_samples):
                # Sample states
                encoder_states_np = self.sample_encoding_states(1)
                decoder_states_np = self.sample_decoding_states(batch_size=1)

                # Sample reward function
                family, reward_fn = self.reward_prior.sample()
                enc_r = self.reward_prior.compute_rewards(reward_fn, encoder_states_np[0])
                dec_r = self.reward_prior.compute_rewards(reward_fn, decoder_states_np[0])

                # Convert to tensors
                encoder_states = torch.from_numpy(encoder_states_np).float().to(self.device)
                encoder_rewards = torch.from_numpy(enc_r).float().to(self.device).unsqueeze(0)
                decoder_states = torch.from_numpy(decoder_states_np).float().to(self.device)
                decoder_rewards = torch.from_numpy(dec_r).float().to(self.device).unsqueeze(0)

                loss_dict = self.model.compute_loss(
                    encoder_states, encoder_rewards,
                    decoder_states, decoder_rewards,
                )

                total_mse += loss_dict['mse_loss'].item()
                total_kl += loss_dict['kl_loss'].item()
                total_loss += loss_dict['total_loss'].item()

        self.model.train()
        return {
            'eval_mse': total_mse / num_samples,
            'eval_kl': total_kl / num_samples,
            'eval_total': total_loss / num_samples,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state to a checkpoint file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'model_config': {
                'state_dim': self.model.state_dim,
                'latent_dim': self.model.latent_dim,
                'beta': self.model.beta,
                'K_encoder': self.K_encoder,
                'K_decoder': self.K_decoder,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = checkpoint['train_step']


def build_fre_model(
    state_dim: int,
    latent_dim: int = 64,
    d_model: int = 256,
    num_layers: int = 2,
    num_heads: int = 4,
    d_ff: int = 1024,
    d_emb: int = 64,
    num_bins: int = 100,
    reward_min: float = -10.0,
    reward_max: float = 10.0,
    decoder_hidden_dims: Optional[list] = None,
    beta: float = 0.1,
    dropout: float = 0.0,
    max_num_states: int = 32,
) -> FREModel:
    """
    Factory function to create a FREModel with default or custom settings.

    Args:
        state_dim: Dimensionality of the state space.
        latent_dim: Latent z dimension (default: 64).
        d_model: Transformer hidden dimension (default: 256).
        num_layers: Number of transformer layers (default: 2).
        num_heads: Number of attention heads (default: 4).
        d_ff: Feedforward dimension (default: 1024).
        d_emb: Reward embedding dimension (default: 64).
        num_bins: Number of reward bins (default: 100).
        reward_min: Minimum reward value (default: -10.0).
        reward_max: Maximum reward value (default: 10.0).
        decoder_hidden_dims: Decoder hidden dimensions (default: [256, 256]).
        beta: KL weight (default: 0.1).
        dropout: Dropout rate (default: 0.0).
        max_num_states: Maximum number of states (default: 32).

    Returns:
        FREModel instance.
    """
    return FREModel(
        state_dim=state_dim,
        latent_dim=latent_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        d_emb=d_emb,
        num_bins=num_bins,
        reward_min=reward_min,
        reward_max=reward_max,
        decoder_hidden_dims=decoder_hidden_dims,
        beta=beta,
        dropout=dropout,
        max_num_states=max_num_states,
    )