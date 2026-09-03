"""
Main orchestrator for FRE training pipeline.

Implements Algorithm 1 from the paper with strided training scheme:
- Phase 1: Train FRE encoder/decoder on unsupervised reward functions
- Phase 2: Train IQL agent with frozen encoder on the same reward distribution

The trainer alternates between phases or runs them sequentially,
managing checkpointing, logging, and evaluation.
"""

import logging
import time
import os
from typing import Optional, Dict, Any, Tuple
from collections import deque

import numpy as np
import torch

from models.fre_encoder import FREEncoder
from models.fre_decoder import FREDecoder
from models.iql_agent import IQLAgent
from rewards.mixture import MixtureRewardDistribution
from data.replay_buffer import ReplayBuffer
from training.train_encoder import FREEncoderTrainer
from training.train_rl import IQLTrainer

logger = logging.getLogger(__name__)


class FRETrainer:
    """
    Main orchestrator for the full FRE training pipeline.

    Manages Phase 1 (encoder/decoder training) and Phase 2 (IQL agent training),
    with strided training, checkpointing, and evaluation support.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state space.
    action_dim : int
        Dimensionality of the action space.
    replay_buffer : ReplayBuffer
        Replay buffer containing the offline dataset.
    config : dict
        Configuration dictionary with all hyperparameters.
    device : torch.device, optional
        Device to run training on.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer: ReplayBuffer,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Extract configuration sections
        self.encoder_config = config.get("encoder", {})
        self.decoder_config = config.get("decoder", {})
        self.iql_config = config.get("iql", {})
        self.reward_config = config.get("reward", {})
        self.training_config = config.get("training", {})

        # Build models
        self._build_models()

        # Build reward distribution
        self._build_reward_distribution()

        # Build phase trainers
        self._build_trainers()

        # Training state
        self.current_phase = "encoder"  # 'encoder' or 'rl'
        self.global_step = 0
        self.encoder_steps_completed = 0
        self.rl_steps_completed = 0
        self.best_encoder_loss = float("inf")
        self.best_rl_loss = float("inf")

        # Logging
        self.metrics_history = {
            "encoder_loss": deque(maxlen=1000),
            "encoder_mse": deque(maxlen=1000),
            "encoder_kl": deque(maxlen=1000),
            "rl_value_loss": deque(maxlen=1000),
            "rl_q_loss": deque(maxlen=1000),
            "rl_policy_loss": deque(maxlen=1000),
        }

        logger.info(f"FRETrainer initialized on device: {self.device}")
        logger.info(
            f"Encoder params: {sum(p.numel() for p in self.encoder.parameters()):,}"
        )
        logger.info(
            f"Decoder params: {sum(p.numel() for p in self.decoder.parameters()):,}"
        )
        logger.info(
            f"IQL Agent params: {sum(p.numel() for p in self.agent.parameters()):,}"
        )

    def _build_models(self):
        """Build encoder, decoder, and IQL agent from config."""
        # FRE Encoder
        self.encoder = FREEncoder(
            state_dim=self.state_dim,
            embed_dim=self.encoder_config.get("embed_dim", 256),
            latent_dim=self.encoder_config.get("latent_dim", 64),
            num_layers=self.encoder_config.get("num_layers", 3),
            num_heads=self.encoder_config.get("num_heads", 4),
            dropout=self.encoder_config.get("dropout", 0.1),
            num_bins=self.encoder_config.get("num_bins", 64),
        ).to(self.device)

        # FRE Decoder
        self.decoder = FREDecoder(
            state_dim=self.state_dim,
            latent_dim=self.encoder_config.get("latent_dim", 64),
            hidden_dims=self.decoder_config.get("hidden_dims", [256, 256]),
        ).to(self.device)

        # IQL Agent
        self.agent = IQLAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.encoder_config.get("latent_dim", 64),
            hidden_dims=self.iql_config.get("hidden_dims", [256, 256]),
            expectile=self.iql_config.get("expectile", 0.7),
            temperature=self.iql_config.get("temperature", 3.0),
            discount=self.iql_config.get("discount", 0.99),
            target_tau=self.iql_config.get("target_tau", 0.005),
        ).to(self.device)

    def _build_reward_distribution(self):
        """Build the mixture reward distribution."""
        self.reward_dist = MixtureRewardDistribution(
            state_dim=self.state_dim,
            device=self.device,
            singleton_threshold=self.reward_config.get("singleton_threshold", 0.5),
            linear_sparsity=self.reward_config.get("linear_sparsity", 0.8),
            mlp_hidden_dim=self.reward_config.get("mlp_hidden_dim", 256),
        )

    def _build_trainers(self):
        """Build Phase 1 and Phase 2 trainers."""
        # Phase 1: Encoder trainer
        self.encoder_trainer = FREEncoderTrainer(
            encoder=self.encoder,
            decoder=self.decoder,
            reward_dist=self.reward_dist,
            replay_buffer=self.replay_buffer,
            learning_rate=self.training_config.get("encoder_lr", 3e-4),
            K_enc=self.training_config.get("K_enc", 64),
            K_dec=self.training_config.get("K_dec", 64),
            beta_kl=self.training_config.get("beta_kl", 0.1),
            device=self.device,
        )

        # Phase 2: RL trainer
        self.rl_trainer = IQLTrainer(
            encoder=self.encoder,
            agent=self.agent,
            reward_dist=self.reward_dist,
            replay_buffer=self.replay_buffer,
            learning_rate=self.training_config.get("rl_lr", 3e-4),
            K_enc=self.training_config.get("K_enc", 64),
            batch_size=self.training_config.get("batch_size", 256),
            device=self.device,
        )

    def train_encoder_phase(
        self,
        num_steps: int,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        checkpoint_dir: Optional[str] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Phase 1: Train the FRE encoder and decoder.

        Parameters
        ----------
        num_steps : int
            Number of training steps.
        log_interval : int
            Steps between logging.
        eval_interval : int
            Steps between evaluation.
        save_interval : int
            Steps between checkpoint saves.
        checkpoint_dir : str, optional
            Directory for saving checkpoints.
        rng_seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        Dict with training statistics.
        """
        logger.info(f"Starting Phase 1 (Encoder Training) for {num_steps} steps")
        self.current_phase = "encoder"

        stats = self.encoder_trainer.train(
            num_steps=num_steps,
            log_interval=log_interval,
            eval_interval=eval_interval,
            rng_seed=rng_seed,
            verbose=verbose,
        )

        self.encoder_steps_completed += num_steps
        self.global_step += num_steps

        # Save checkpoint if directory provided
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(
                checkpoint_dir, f"encoder_phase_step_{self.encoder_steps_completed}.pt"
            )
            self.encoder_trainer.save_checkpoint(path)
            logger.info(f"Encoder checkpoint saved to {path}")

        return stats

    def train_rl_phase(
        self,
        num_steps: int,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        checkpoint_dir: Optional[str] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Phase 2: Train the IQL agent with frozen encoder.

        Parameters
        ----------
        num_steps : int
            Number of training steps.
        log_interval : int
            Steps between logging.
        eval_interval : int
            Steps between evaluation.
        save_interval : int
            Steps between checkpoint saves.
        checkpoint_dir : str, optional
            Directory for saving checkpoints.
        rng_seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        Dict with training statistics.
        """
        logger.info(f"Starting Phase 2 (RL Training) for {num_steps} steps")
        self.current_phase = "rl"

        stats = self.rl_trainer.train(
            num_steps=num_steps,
            log_interval=log_interval,
            eval_interval=eval_interval,
            rng_seed=rng_seed,
            verbose=verbose,
        )

        self.rl_steps_completed += num_steps
        self.global_step += num_steps

        # Save checkpoint if directory provided
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(
                checkpoint_dir, f"rl_phase_step_{self.rl_steps_completed}.pt"
            )
            self.rl_trainer.save_checkpoint(path)
            logger.info(f"RL checkpoint saved to {path}")

        return stats

    def train_full(
        self,
        encoder_steps: int = 100000,
        rl_steps: int = 1000000,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 10000,
        checkpoint_dir: Optional[str] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline: Phase 1 then Phase 2.

        Parameters
        ----------
        encoder_steps : int
            Number of Phase 1 training steps.
        rl_steps : int
            Number of Phase 2 training steps.
        log_interval : int
            Steps between logging.
        eval_interval : int
            Steps between evaluation.
        save_interval : int
            Steps between checkpoint saves.
        checkpoint_dir : str, optional
            Directory for saving checkpoints.
        rng_seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        Dict with combined training statistics.
        """
        start_time = time.time()

        # Phase 1: Encoder training
        encoder_stats = {}
        if encoder_steps > 0:
            encoder_stats = self.train_encoder_phase(
                num_steps=encoder_steps,
                log_interval=log_interval,
                eval_interval=eval_interval,
                save_interval=save_interval,
                checkpoint_dir=checkpoint_dir,
                rng_seed=rng_seed,
                verbose=verbose,
            )

        # Phase 2: RL training
        rl_stats = {}
        if rl_steps > 0:
            rl_stats = self.train_rl_phase(
                num_steps=rl_steps,
                log_interval=log_interval,
                eval_interval=eval_interval,
                save_interval=save_interval,
                checkpoint_dir=checkpoint_dir,
                rng_seed=rng_seed,
                verbose=verbose,
            )

        total_time = time.time() - start_time

        combined_stats = {
            "encoder_stats": encoder_stats,
            "rl_stats": rl_stats,
            "total_time": total_time,
            "encoder_steps_completed": self.encoder_steps_completed,
            "rl_steps_completed": self.rl_steps_completed,
            "global_step": self.global_step,
        }

        logger.info(f"Full training completed in {total_time:.1f}s")
        logger.info(f"  Encoder steps: {self.encoder_steps_completed}")
        logger.info(f"  RL steps: {self.rl_steps_completed}")

        return combined_stats

    def train_strided(
        self,
        total_steps: int = 1100000,
        encoder_interval: int = 100000,
        rl_interval: int = 1000000,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 10000,
        checkpoint_dir: Optional[str] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run strided training: alternate between encoder and RL phases.

        This implements the strided training scheme where the encoder is
        periodically updated during RL training to prevent catastrophic
        forgetting of the reward representation.

        Parameters
        ----------
        total_steps : int
            Total number of training steps across both phases.
        encoder_interval : int
            Number of encoder steps per strided block.
        rl_interval : int
            Number of RL steps per strided block.
        log_interval : int
            Steps between logging.
        eval_interval : int
            Steps between evaluation.
        save_interval : int
            Steps between checkpoint saves.
        checkpoint_dir : str, optional
            Directory for saving checkpoints.
        rng_seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        Dict with combined training statistics.
        """
        logger.info(
            f"Starting strided training: total={total_steps}, "
            f"encoder_interval={encoder_interval}, rl_interval={rl_interval}"
        )

        start_time = time.time()
        all_encoder_stats = []
        all_rl_stats = []

        # First, train encoder for initial period
        if encoder_interval > 0:
            logger.info("Initial encoder training phase...")
            enc_stats = self.train_encoder_phase(
                num_steps=encoder_interval,
                log_interval=log_interval,
                eval_interval=eval_interval,
                save_interval=save_interval,
                checkpoint_dir=checkpoint_dir,
                rng_seed=rng_seed,
                verbose=verbose,
            )
            all_encoder_stats.append(enc_stats)

        # Then alternate RL and encoder updates
        remaining_steps = total_steps - self.global_step
        while remaining_steps > 0:
            # RL phase
            rl_steps_this = min(rl_interval, remaining_steps)
            if rl_steps_this > 0:
                logger.info(f"RL training phase ({rl_steps_this} steps)...")
                rl_stats = self.train_rl_phase(
                    num_steps=rl_steps_this,
                    log_interval=log_interval,
                    eval_interval=eval_interval,
                    save_interval=save_interval,
                    checkpoint_dir=checkpoint_dir,
                    rng_seed=rng_seed,
                    verbose=verbose,
                )
                all_rl_stats.append(rl_stats)
                remaining_steps -= rl_steps_this

            # Encoder refresh phase
            if remaining_steps > 0 and encoder_interval > 0:
                enc_steps_this = min(encoder_interval, remaining_steps)
                logger.info(f"Encoder refresh phase ({enc_steps_this} steps)...")
                enc_stats = self.train_encoder_phase(
                    num_steps=enc_steps_this,
                    log_interval=log_interval,
                    eval_interval=eval_interval,
                    save_interval=save_interval,
                    checkpoint_dir=checkpoint_dir,
                    rng_seed=rng_seed,
                    verbose=verbose,
                )
                all_encoder_stats.append(enc_stats)
                remaining_steps -= enc_steps_this

        total_time = time.time() - start_time

        combined_stats = {
            "encoder_stats": all_encoder_stats,
            "rl_stats": all_rl_stats,
            "total_time": total_time,
            "encoder_steps_completed": self.encoder_steps_completed,
            "rl_steps_completed": self.rl_steps_completed,
            "global_step": self.global_step,
        }

        logger.info(f"Strided training completed in {total_time:.1f}s")
        return combined_stats

    def save_checkpoint(
        self,
        path: str,
        save_optimizers: bool = True,
    ):
        """
        Save full training state to a checkpoint file.

        Parameters
        ----------
        path : str
            Path to save the checkpoint.
        save_optimizers : bool
            Whether to include optimizer states.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "agent_state_dict": self.agent.state_dict(),
            "encoder_trainer_state": self.encoder_trainer.state_dict(),
            "rl_trainer_state": self.rl_trainer.state_dict(),
            "global_step": self.global_step,
            "encoder_steps_completed": self.encoder_steps_completed,
            "rl_steps_completed": self.rl_steps_completed,
            "current_phase": self.current_phase,
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

        torch.save(checkpoint, path)
        logger.info(f"Full checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Load full training state from a checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.agent.load_state_dict(checkpoint["agent_state_dict"])

        if "encoder_trainer_state" in checkpoint:
            self.encoder_trainer.load_state_dict(checkpoint["encoder_trainer_state"])
        if "rl_trainer_state" in checkpoint:
            self.rl_trainer.load_state_dict(checkpoint["rl_trainer_state"])

        self.global_step = checkpoint.get("global_step", 0)
        self.encoder_steps_completed = checkpoint.get("encoder_steps_completed", 0)
        self.rl_steps_completed = checkpoint.get("rl_steps_completed", 0)
        self.current_phase = checkpoint.get("current_phase", "encoder")

        logger.info(f"Checkpoint loaded from {path} (global_step={self.global_step})")

    def get_encoder(self) -> FREEncoder:
        """Get the trained FRE encoder (for evaluation)."""
        return self.encoder

    def get_decoder(self) -> FREDecoder:
        """Get the trained FRE decoder."""
        return self.decoder

    def get_agent(self) -> IQLAgent:
        """Get the trained IQL agent (for evaluation)."""
        return self.agent

    def to(self, device: torch.device):
        """Move all models to the specified device."""
        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.agent = self.agent.to(device)
        self.reward_dist = self.reward_dist.to(device)
        self.encoder_trainer.to(device)
        self.rl_trainer.to(device)
        return self

    def eval(self):
        """Set all models to evaluation mode."""
        self.encoder.eval()
        self.decoder.eval()
        self.agent.q1.eval()
        self.agent.q2.eval()
        self.agent.v.eval()
        self.agent.policy.eval()

    def train(self):
        """Set all models to training mode."""
        self.encoder.train()
        self.decoder.train()
        self.agent.q1.train()
        self.agent.q2.train()
        self.agent.v.train()
        self.agent.policy.train()


def create_trainer_from_config(
    config: Dict[str, Any],
    replay_buffer: ReplayBuffer,
    device: Optional[torch.device] = None,
) -> FRETrainer:
    """
    Factory function to create a FRETrainer from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary (e.g., loaded from YAML).
    replay_buffer : ReplayBuffer
        Replay buffer with the offline dataset.
    device : torch.device, optional
        Device to run on.

    Returns
    -------
    FRETrainer instance.
    """
    env_config = config.get("env", {})
    state_dim = env_config.get("state_dim")
    action_dim = env_config.get("action_dim")

    if state_dim is None or action_dim is None:
        # Infer from replay buffer
        sample = replay_buffer.sample(1)
        state_dim = sample["states"].shape[-1]
        action_dim = sample["actions"].shape[-1]
        logger.info(f"Inferred state_dim={state_dim}, action_dim={action_dim}")

    return FRETrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        config=config,
        device=device,
    )