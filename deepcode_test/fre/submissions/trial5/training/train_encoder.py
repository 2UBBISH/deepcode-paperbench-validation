"""
Phase 1: FRE Encoder Training

Trains the transformer-based VAE encoder and feedforward decoder jointly
using the unsupervised prior reward distribution. The encoder learns to
map a set of (state, reward) pairs into a latent Gaussian distribution,
and the decoder learns to predict rewards from states conditioned on the
latent code.

Training follows Algorithm 1 (Phase 1) from the paper:
  - Sample reward function η ~ p(η) from the mixture distribution
  - Sample K encoding states and K' decoding states from the offline dataset
  - Compute rewards for both sets using η
  - Encode: z ~ pθ(· | {(s_k^e, η(s_k^e))})
  - Decode: predict η(s_k^d) from (s_k^d, z)
  - Loss = MSE(reconstructed, true) + β * KL(pθ || N(0,I))
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Tuple, List
from collections import deque
import logging
import time

from models.fre_encoder import FREEncoder
from models.fre_decoder import FREDecoder
from rewards.mixture import MixtureRewardDistribution
from data.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class FREEncoderTrainer:
    """
    Trainer for Phase 1: Learning the functional reward encoding (FRE).

    Trains the encoder and decoder jointly using the unsupervised prior
    reward distribution. After training, the encoder can produce a latent
    vector z that captures the structure of any reward function given only
    a few (state, reward) examples.

    Attributes:
        encoder: FREEncoder (transformer-based VAE)
        decoder: FREDecoder (feedforward reward predictor)
        reward_dist: MixtureRewardDistribution for sampling η
        replay_buffer: ReplayBuffer providing offline states
        optimizer: Adam optimizer for encoder + decoder parameters
        K_enc: Number of encoding states
        K_dec: Number of decoding states
        beta_kl: KL divergence penalty coefficient (β-VAE)
        device: torch device
    """

    def __init__(
        self,
        encoder: FREEncoder,
        decoder: FREDecoder,
        reward_dist: MixtureRewardDistribution,
        replay_buffer: ReplayBuffer,
        learning_rate: float = 3e-4,
        K_enc: int = 64,
        K_dec: int = 64,
        beta_kl: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the encoder trainer.

        Args:
            encoder: FREEncoder instance
            decoder: FREDecoder instance
            reward_dist: MixtureRewardDistribution for sampling reward functions
            replay_buffer: ReplayBuffer providing offline dataset states
            learning_rate: Adam learning rate (default 3e-4)
            K_enc: Number of encoding states to sample (default 64)
            K_dec: Number of decoding states to sample (default 64)
            beta_kl: KL divergence penalty coefficient (default 0.1)
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.reward_dist = reward_dist
        self.replay_buffer = replay_buffer

        self.K_enc = K_enc
        self.K_dec = K_dec
        self.beta_kl = beta_kl

        # Optimizer: train encoder and decoder jointly
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )

        # Tracking
        self.train_step = 0
        self.metrics_history: Dict[str, deque] = {
            "loss": deque(maxlen=1000),
            "mse_loss": deque(maxlen=1000),
            "kl_loss": deque(maxlen=1000),
        }

        logger.info(
            f"FREEncoderTrainer initialized: K_enc={K_enc}, K_dec={K_dec}, "
            f"beta_kl={beta_kl}, lr={learning_rate}, device={self.device}"
        )

    def train_step_fn(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
        """
        Execute a single training step for the encoder and decoder.

        Algorithm:
        1. Sample reward function η from the mixture distribution
        2. Sample K_enc encoding states and K_dec decoding states from dataset
        3. Compute rewards for both sets using η
        4. Encode: z ~ pθ(· | {(s_enc, r_enc)})
        5. Decode: predict r_dec from (s_dec, z)
        6. Compute loss = MSE(r_pred, r_true) + β * KL(pθ || N(0,I))
        7. Backpropagate and update parameters

        Args:
            rng: Optional numpy RandomState for reproducibility

        Returns:
            Dict with loss values: 'loss', 'mse_loss', 'kl_loss'
        """
        self.encoder.train()
        self.decoder.train()

        if rng is None:
            rng = np.random.RandomState()

        # 1. Sample reward function η from mixture distribution
        # Get all states for goal sampling (singleton rewards need dataset states)
        all_states = self.replay_buffer.get_all_states()
        reward_fn = self.reward_dist.sample(dataset_states=all_states)

        # 2. Sample encoding and decoding states (disjoint sets)
        # Encoding states: used as input to the encoder
        enc_states_np = self.replay_buffer.sample_states(self.K_enc, rng=rng)
        # Decoding states: used for reconstruction loss (different from encoding)
        dec_states_np = self.replay_buffer.sample_states(self.K_dec, rng=rng)

        # Convert to tensors
        enc_states = torch.from_numpy(enc_states_np).float().to(self.device)
        dec_states = torch.from_numpy(dec_states_np).float().to(self.device)

        # 3. Compute rewards for both sets using η
        with torch.no_grad():
            enc_rewards = reward_fn(enc_states)  # shape: (K_enc,)
            dec_rewards = reward_fn(dec_states)  # shape: (K_dec,)

        # 4. Encode: z ~ pθ(· | {(s_enc, r_enc)})
        z, mu, logvar = self.encoder(enc_states, enc_rewards)

        # 5. Decode: predict rewards for decoding states
        pred_rewards = self.decoder(dec_states, z)  # shape: (K_dec,)

        # 6. Compute losses
        # MSE reconstruction loss
        mse_loss = nn.functional.mse_loss(pred_rewards, dec_rewards)

        # KL divergence to standard normal prior
        kl_loss = self.encoder.kl_divergence(mu, logvar)

        # Total loss with β-VAE weighting
        total_loss = mse_loss + self.beta_kl * kl_loss

        # 7. Backpropagate and update
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        # Track metrics
        self.train_step += 1
        loss_dict = {
            "loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        for key, value in loss_dict.items():
            self.metrics_history[key].append(value)

        return loss_dict

    def train(
        self,
        num_steps: int,
        log_interval: int = 1000,
        eval_interval: int = 5000,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Run the full Phase 1 training loop.

        Args:
            num_steps: Total number of training steps
            log_interval: Steps between logging metrics
            eval_interval: Steps between evaluation (detailed loss reporting)
            rng_seed: Random seed for reproducibility
            verbose: Whether to print progress

        Returns:
            Dict mapping metric names to lists of values recorded at log intervals
        """
        rng = np.random.RandomState(rng_seed) if rng_seed is not None else np.random.RandomState()

        log_history: Dict[str, List[float]] = {
            "step": [],
            "loss": [],
            "mse_loss": [],
            "kl_loss": [],
        }

        start_time = time.time()

        for step in range(1, num_steps + 1):
            loss_dict = self.train_step_fn(rng=rng)

            if step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0

                # Compute running averages
                avg_loss = np.mean(self.metrics_history["loss"]) if self.metrics_history["loss"] else loss_dict["loss"]
                avg_mse = np.mean(self.metrics_history["mse_loss"]) if self.metrics_history["mse_loss"] else loss_dict["mse_loss"]
                avg_kl = np.mean(self.metrics_history["kl_loss"]) if self.metrics_history["kl_loss"] else loss_dict["kl_loss"]

                log_history["step"].append(step)
                log_history["loss"].append(avg_loss)
                log_history["mse_loss"].append(avg_mse)
                log_history["kl_loss"].append(avg_kl)

                if verbose:
                    logger.info(
                        f"Step {step:>8d}/{num_steps} | "
                        f"Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | "
                        f"KL: {avg_kl:.6f} | Steps/s: {steps_per_sec:.1f}"
                    )

            if step % eval_interval == 0:
                # Detailed evaluation: compute reconstruction quality on a larger batch
                eval_metrics = self.evaluate(num_samples=1024, rng=rng)
                if verbose:
                    logger.info(
                        f"  Eval @ step {step}: "
                        f"MSE={eval_metrics['mse_loss']:.6f}, "
                        f"KL={eval_metrics['kl_loss']:.6f}, "
                        f"R²={eval_metrics.get('r2_score', float('nan')):.4f}"
                    )

        total_time = time.time() - start_time
        if verbose:
            logger.info(f"Phase 1 training completed in {total_time:.1f}s ({total_time/60:.1f}min)")

        return log_history

    def evaluate(
        self,
        num_samples: int = 1024,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the encoder-decoder on a batch of held-out-like states.

        Computes reconstruction MSE, KL divergence, and optionally R² score
        to assess how well the encoder captures reward function structure.

        Args:
            num_samples: Number of decoding states to evaluate on
            rng: Random state for reproducibility

        Returns:
            Dict with 'mse_loss', 'kl_loss', and optionally 'r2_score'
        """
        self.encoder.eval()
        self.decoder.eval()

        if rng is None:
            rng = np.random.RandomState()

        all_states = self.replay_buffer.get_all_states()
        reward_fn = self.reward_dist.sample(dataset_states=all_states)

        # Encoding states
        enc_states_np = self.replay_buffer.sample_states(self.K_enc, rng=rng)
        enc_states = torch.from_numpy(enc_states_np).float().to(self.device)

        # Decoding states (larger batch for evaluation)
        dec_states_np = self.replay_buffer.sample_states(num_samples, rng=rng)
        dec_states = torch.from_numpy(dec_states_np).float().to(self.device)

        with torch.no_grad():
            enc_rewards = reward_fn(enc_states)
            dec_rewards = reward_fn(dec_states)

            # Deterministic encoding (use mu, no sampling)
            z = self.encoder.encode_deterministic(enc_states, enc_rewards)

            # Also get mu, logvar for KL computation
            _, mu, logvar = self.encoder(enc_states, enc_rewards)

            pred_rewards = self.decoder(dec_states, z)

            mse_loss = nn.functional.mse_loss(pred_rewards, dec_rewards).item()
            kl_loss = self.encoder.kl_divergence(mu, logvar).item()

            # R² score (coefficient of determination)
            ss_res = ((dec_rewards - pred_rewards) ** 2).sum().item()
            ss_tot = ((dec_rewards - dec_rewards.mean()) ** 2).sum().item()
            r2_score = 1.0 - ss_res / (ss_tot + 1e-8)

        self.encoder.train()
        self.decoder.train()

        return {
            "mse_loss": mse_loss,
            "kl_loss": kl_loss,
            "r2_score": r2_score,
        }

    def get_reward_range_estimate(
        self,
        num_functions: int = 100,
        num_states: int = 1024,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[float, float]:
        """
        Estimate the typical reward range by sampling many reward functions
        and evaluating them on random states. Useful for setting reward
        embedding bin boundaries.

        Args:
            num_functions: Number of reward functions to sample
            num_states: Number of states per function
            rng: Random state

        Returns:
            (reward_min, reward_max) tuple
        """
        if rng is None:
            rng = np.random.RandomState()

        all_states = self.replay_buffer.get_all_states()
        all_rewards = []

        for _ in range(num_functions):
            reward_fn = self.reward_dist.sample(dataset_states=all_states)
            states_np = self.replay_buffer.sample_states(num_states, rng=rng)
            states = torch.from_numpy(states_np).float().to(self.device)
            with torch.no_grad():
                rewards = reward_fn(states)
            all_rewards.append(rewards.cpu().numpy())

        all_rewards = np.concatenate(all_rewards)
        reward_min = float(np.percentile(all_rewards, 1))
        reward_max = float(np.percentile(all_rewards, 99))

        logger.info(f"Estimated reward range: [{reward_min:.4f}, {reward_max:.4f}]")
        return reward_min, reward_max

    def state_dict(self) -> Dict:
        """Get the full state dict for checkpointing."""
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint."""
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.train_step = state_dict.get("train_step", 0)

    def save_checkpoint(self, path: str):
        """Save training checkpoint to disk."""
        torch.save(self.state_dict(), path)
        logger.info(f"Encoder checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)
        logger.info(f"Encoder checkpoint loaded from {path} (step {self.train_step})")