"""
Strided Training Orchestration for FRE.

This module implements the full training pipeline:
  Phase 1: Train FRE encoder+decoder (VAE) on random reward functions from the prior.
  Phase 2: Freeze encoder, train IQL agent conditioned on latent z.

The trainer coordinates dataset loading, prior sampling, FRE pre-training,
IQL policy training, checkpointing, and logging.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Tuple, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fre.dataset import OfflineDataset, DatasetNormalizer, load_dataset
from fre.prior import MixedPrior, create_mixed_prior, RewardFunction
from fre.encoder import FREEncoder, create_fre_encoder
from fre.decoder import RewardDecoder, create_reward_decoder
from fre.fre_model import FREModel, FRETrainer, create_fre_model, create_fre_trainer
from fre.policy import create_policy
from fre.iql import IQLAgent, IQLTrainer, create_iql_agent, create_iql_trainer

logger = logging.getLogger(__name__)


class StridedTrainer:
    """
    Full strided training orchestrator for FRE + IQL.

    Phase 1: Train FRE VAE (encoder + decoder) on random reward functions.
    Phase 2: Freeze encoder, train IQL agent conditioned on latent z.

    Handles dataset loading, prior creation, model instantiation,
    training loops, checkpointing, and logging.
    """

    def __init__(
        self,
        # Dataset
        dataset_name: str = "antmaze-large-diverse-v2",
        dataset_path: Optional[str] = None,
        # State info
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        # FRE hyperparameters
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
        K_encoder: int = 32,
        K_decoder: int = 128,
        # IQL hyperparameters
        iql_hidden_dims: Optional[List[int]] = None,
        expectile: float = 0.7,
        temperature: float = 3.0,
        discount: float = 0.99,
        policy_log_std_min: float = -5.0,
        policy_log_std_max: float = 2.0,
        # Prior hyperparameters
        goal_threshold: float = 0.5,
        linear_sparsity: float = 0.8,
        mlp_hidden_dim: int = 256,
        prior_weights: Optional[List[float]] = None,
        include_goal: bool = True,
        include_linear: bool = True,
        include_mlp: bool = True,
        # Training hyperparameters
        fre_steps: int = 100_000,
        iql_steps: int = 1_000_000,
        fre_batch_size: int = 256,
        iql_batch_size: int = 256,
        fre_lr: float = 1e-4,
        iql_lr: float = 3e-4,
        fre_clip_grad_norm: Optional[float] = None,
        iql_clip_grad_norm: Optional[float] = None,
        policy_update_delay: int = 1,
        # Logging & checkpointing
        log_interval: int = 1000,
        eval_interval: int = 5000,
        checkpoint_interval: int = 10000,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        use_wandb: bool = False,
        wandb_project: str = "fre",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        # Device
        device: str = "auto",
        # Seed
        seed: int = 42,
        # Discrete action space
        discrete: bool = False,
        num_actions: Optional[int] = None,
    ):
        """
        Initialize the strided trainer.

        Args:
            dataset_name: Name of dataset to load (D4RL or ExORL).
            dataset_path: Optional path to ExORL dataset files.
            state_dim: State dimension (auto-detected if None).
            action_dim: Action dimension (auto-detected if None).
            d_model: Transformer model dimension.
            d_reward: Reward embedding dimension.
            d_latent: Latent vector dimension.
            num_bins: Number of reward discretization bins.
            r_min, r_max: Reward range for discretization.
            nhead: Number of Transformer attention heads.
            num_layers: Number of Transformer layers.
            dim_feedforward: Transformer feedforward dimension.
            dropout: Dropout rate.
            decoder_hidden_dims: Hidden dimensions for decoder MLP.
            beta: VAE KL divergence weight.
            K_encoder: Number of encoder states per reward function.
            K_decoder: Number of decoder states per reward function.
            iql_hidden_dims: Hidden dimensions for IQL networks.
            expectile: IQL expectile parameter.
            temperature: IQL AWR temperature.
            discount: Discount factor.
            policy_log_std_min/max: Log std bounds for Gaussian policy.
            goal_threshold: Threshold for goal-reaching prior.
            linear_sparsity: Sparsity fraction for linear prior.
            mlp_hidden_dim: Hidden dim for MLP prior.
            prior_weights: Mixture weights for prior families.
            include_goal/linear/mlp: Which prior families to include.
            fre_steps: Number of FRE training steps.
            iql_steps: Number of IQL training steps.
            fre_batch_size: Batch size for FRE training.
            iql_batch_size: Batch size for IQL training.
            fre_lr: Learning rate for FRE.
            iql_lr: Learning rate for IQL.
            fre_clip_grad_norm: Gradient clipping for FRE.
            iql_clip_grad_norm: Gradient clipping for IQL.
            policy_update_delay: Delay policy updates (steps).
            log_interval: Steps between logging.
            eval_interval: Steps between evaluation.
            checkpoint_interval: Steps between checkpoints.
            checkpoint_dir: Directory for saving checkpoints.
            log_dir: Directory for logs.
            use_wandb: Whether to use Weights & Biases logging.
            wandb_project: W&B project name.
            wandb_entity: W&B entity.
            wandb_config: Additional W&B config.
            device: Device string ("auto", "cuda", "cpu").
            seed: Random seed.
            discrete: Whether action space is discrete.
            num_actions: Number of discrete actions (if discrete).
        """
        self.seed = seed
        self._set_seed(seed)

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, data_path=dataset_path)
        self.dataset_name = dataset_name

        # Auto-detect dimensions
        if state_dim is None:
            state_dim = self.dataset.observations.shape[1]
        if action_dim is None:
            action_dim = self.dataset.actions.shape[1]

        self.state_dim = state_dim
        self.action_dim = action_dim

        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"Dataset size: {len(self.dataset)} transitions")

        # Store hyperparameters
        self.d_model = d_model
        self.d_reward = d_reward
        self.d_latent = d_latent
        self.num_bins = num_bins
        self.r_min = r_min
        self.r_max = r_max
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.decoder_hidden_dims = decoder_hidden_dims or [256, 256]
        self.beta = beta
        self.K_encoder = K_encoder
        self.K_decoder = K_decoder
        self.iql_hidden_dims = iql_hidden_dims or [256, 256]
        self.expectile = expectile
        self.temperature = temperature
        self.discount = discount
        self.policy_log_std_min = policy_log_std_min
        self.policy_log_std_max = policy_log_std_max
        self.goal_threshold = goal_threshold
        self.linear_sparsity = linear_sparsity
        self.mlp_hidden_dim = mlp_hidden_dim
        self.prior_weights = prior_weights
        self.include_goal = include_goal
        self.include_linear = include_linear
        self.include_mlp = include_mlp
        self.fre_steps = fre_steps
        self.iql_steps = iql_steps
        self.fre_batch_size = fre_batch_size
        self.iql_batch_size = iql_batch_size
        self.fre_lr = fre_lr
        self.iql_lr = iql_lr
        self.fre_clip_grad_norm = fre_clip_grad_norm
        self.iql_clip_grad_norm = iql_clip_grad_norm
        self.policy_update_delay = policy_update_delay
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_config = wandb_config or {}
        self.discrete = discrete
        self.num_actions = num_actions

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize W&B
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config={
                        **self._get_config_dict(),
                        **self.wandb_config,
                    },
                    name=f"FRE_{dataset_name}_seed{seed}",
                )
                logger.info("W&B initialized.")
            except ImportError:
                logger.warning("wandb not installed. Skipping W&B logging.")
                self.use_wandb = False

        # Create prior
        logger.info("Creating mixed prior...")
        self.prior = create_mixed_prior(
            state_dim=state_dim,
            dataset_states=self.dataset.get_all_states(),
            goal_threshold=goal_threshold,
            linear_sparsity=linear_sparsity,
            mlp_hidden_dim=mlp_hidden_dim,
            weights=prior_weights,
            include_goal=include_goal,
            include_linear=include_linear,
            include_mlp=include_mlp,
        )
        logger.info(f"Prior families: {self.prior.family_names}")

        # Create FRE model
        logger.info("Creating FRE model...")
        self.fre_model = create_fre_model(
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
            decoder_hidden_dims=self.decoder_hidden_dims,
            beta=beta,
        ).to(self.device)

        # Create FRE trainer
        self.fre_trainer = create_fre_trainer(
            model=self.fre_model,
            prior=self.prior,
            dataset=self.dataset,
            learning_rate=fre_lr,
            device=self.device,
            K_encoder=K_encoder,
            K_decoder=K_decoder,
            batch_size=fre_batch_size,
            clip_grad_norm=fre_clip_grad_norm,
            rng=np.random.RandomState(seed),
        )

        # Create IQL agent (will be initialized after FRE training)
        self.iql_agent = None
        self.iql_trainer = None

        # Training state
        self.fre_step = 0
        self.iql_step = 0
        self.phase = "init"  # "init", "fre", "iql", "done"

        # Metrics history
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)

        logger.info("StridedTrainer initialized successfully.")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_config_dict(self) -> Dict[str, Any]:
        """Return all configuration as a dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "d_reward": self.d_reward,
            "d_latent": self.d_latent,
            "num_bins": self.num_bins,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "decoder_hidden_dims": self.decoder_hidden_dims,
            "beta": self.beta,
            "K_encoder": self.K_encoder,
            "K_decoder": self.K_decoder,
            "iql_hidden_dims": self.iql_hidden_dims,
            "expectile": self.expectile,
            "temperature": self.temperature,
            "discount": self.discount,
            "goal_threshold": self.goal_threshold,
            "linear_sparsity": self.linear_sparsity,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "prior_weights": self.prior_weights,
            "include_goal": self.include_goal,
            "include_linear": self.include_linear,
            "include_mlp": self.include_mlp,
            "fre_steps": self.fre_steps,
            "iql_steps": self.iql_steps,
            "fre_batch_size": self.fre_batch_size,
            "iql_batch_size": self.iql_batch_size,
            "fre_lr": self.fre_lr,
            "iql_lr": self.iql_lr,
            "seed": self.seed,
            "discrete": self.discrete,
            "num_actions": self.num_actions,
        }

    def train_fre_phase(self, steps: Optional[int] = None) -> Dict[str, float]:
        """
        Run Phase 1: FRE encoder-decoder training.

        Args:
            steps: Number of training steps (defaults to self.fre_steps).

        Returns:
            Dictionary of final metrics.
        """
        if steps is None:
            steps = self.fre_steps

        self.phase = "fre"
        logger.info(f"Starting FRE training phase for {steps} steps...")
        start_time = time.time()

        for step in range(steps):
            # Training step
            loss_dict = self.fre_trainer.train_step()

            self.fre_step += 1

            # Logging
            if self.fre_step % self.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.fre_step / max(elapsed, 1e-8)

                log_str = (
                    f"[FRE] Step {self.fre_step}/{steps} "
                    f"({steps_per_sec:.1f} steps/s) | "
                    f"Loss: {loss_dict.get('loss', 0):.4f} | "
                    f"MSE: {loss_dict.get('mse', 0):.4f} | "
                    f"KL: {loss_dict.get('kl', 0):.4f}"
                )
                logger.info(log_str)

                # Record metrics
                for k, v in loss_dict.items():
                    self.metrics_history[f"fre/{k}"].append(v)

                if self.use_wandb and self.wandb_run:
                    import wandb
                    wandb.log(
                        {f"fre/{k}": v for k, v in loss_dict.items()},
                        step=self.fre_step,
                    )

            # Checkpointing
            if self.fre_step % self.checkpoint_interval == 0:
                self.save_checkpoint(tag=f"fre_step{self.fre_step}")

            # Validation
            if self.fre_step % self.eval_interval == 0:
                val_metrics = self.fre_trainer.validate(num_samples=10)
                logger.info(
                    f"[FRE Val] Step {self.fre_step} | "
                    f"Val MSE: {val_metrics.get('val_mse', 0):.4f} | "
                    f"Val KL: {val_metrics.get('val_kl', 0):.4f}"
                )
                for k, v in val_metrics.items():
                    self.metrics_history[f"fre_val/{k}"].append(v)
                if self.use_wandb and self.wandb_run:
                    import wandb
                    wandb.log(
                        {f"fre_val/{k}": v for k, v in val_metrics.items()},
                        step=self.fre_step,
                    )

        # Final metrics
        recent_losses = self.fre_trainer.get_recent_losses(window=100)
        elapsed = time.time() - start_time
        logger.info(f"FRE training completed in {elapsed:.1f}s ({self.fre_step} steps)")

        return recent_losses

    def train_iql_phase(self, steps: Optional[int] = None) -> Dict[str, float]:
        """
        Run Phase 2: IQL agent training with frozen FRE encoder.

        Args:
            steps: Number of training steps (defaults to self.iql_steps).

        Returns:
            Dictionary of final metrics.
        """
        if steps is None:
            steps = self.iql_steps

        # Initialize IQL agent if not already done
        if self.iql_agent is None:
            self._init_iql()

        self.phase = "iql"
        logger.info(f"Starting IQL training phase for {steps} steps...")
        start_time = time.time()

        for step in range(steps):
            # Training step
            loss_dict = self.iql_trainer.train_step()

            self.iql_step += 1

            # Logging
            if self.iql_step % self.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.iql_step / max(elapsed, 1e-8)

                log_str = (
                    f"[IQL] Step {self.iql_step}/{steps} "
                    f"({steps_per_sec:.1f} steps/s) | "
                    f"V Loss: {loss_dict.get('v_loss', 0):.4f} | "
                    f"Q Loss: {loss_dict.get('q_loss', 0):.4f} | "
                    f"Policy Loss: {loss_dict.get('policy_loss', 0):.4f}"
                )
                logger.info(log_str)

                # Record metrics
                for k, v in loss_dict.items():
                    self.metrics_history[f"iql/{k}"].append(v)

                if self.use_wandb and self.wandb_run:
                    import wandb
                    wandb.log(
                        {f"iql/{k}": v for k, v in loss_dict.items()},
                        step=self.fre_step + self.iql_step,
                    )

            # Checkpointing
            if self.iql_step % self.checkpoint_interval == 0:
                self.save_checkpoint(tag=f"iql_step{self.iql_step}")

        # Final metrics
        recent_losses = self.iql_trainer.get_recent_losses(window=100)
        elapsed = time.time() - start_time
        logger.info(f"IQL training completed in {elapsed:.1f}s ({self.iql_step} steps)")

        self.phase = "done"
        return recent_losses

    def _init_iql(self):
        """Initialize IQL agent and trainer after FRE training."""
        logger.info("Initializing IQL agent...")

        # Freeze FRE encoder
        self.fre_model.freeze_encoder()

        # Create IQL agent
        self.iql_agent = create_iql_agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            d_latent=self.d_latent,
            hidden_dims=self.iql_hidden_dims,
            expectile=self.expectile,
            temperature=self.temperature,
            discount=self.discount,
            policy_log_std_min=self.policy_log_std_min,
            policy_log_std_max=self.policy_log_std_max,
            discrete=self.discrete,
            num_actions=self.num_actions,
            device=self.device,
        )

        # Create IQL trainer
        self.iql_trainer = create_iql_trainer(
            agent=self.iql_agent,
            fre_model=self.fre_model,
            prior=self.prior,
            dataset=self.dataset,
            device=self.device,
            K_encoder=self.K_encoder,
            batch_size=self.iql_batch_size,
            clip_grad_norm=self.iql_clip_grad_norm,
            policy_update_delay=self.policy_update_delay,
            rng=np.random.RandomState(self.seed + 1),
        )

        logger.info("IQL agent initialized.")

    def train_full(self) -> Dict[str, Any]:
        """
        Run the complete strided training pipeline (Phase 1 + Phase 2).

        Returns:
            Dictionary of final metrics from both phases.
        """
        logger.info("=" * 60)
        logger.info("Starting FULL training pipeline")
        logger.info("=" * 60)

        # Phase 1: FRE training
        fre_metrics = self.train_fre_phase()

        # Save FRE checkpoint
        self.save_checkpoint(tag="fre_final")

        # Phase 2: IQL training
        iql_metrics = self.train_iql_phase()

        # Save final checkpoint
        self.save_checkpoint(tag="final")

        logger.info("=" * 60)
        logger.info("Training pipeline COMPLETE")
        logger.info("=" * 60)

        return {
            "fre_metrics": fre_metrics,
            "iql_metrics": iql_metrics,
            "fre_steps": self.fre_step,
            "iql_steps": self.iql_step,
        }

    def encode_reward(self, states: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Encode a reward function into latent z using the trained FRE encoder.

        Args:
            states: (N, state_dim) array of states.
            rewards: (N,) array of scalar rewards.

        Returns:
            Latent vector z as numpy array of shape (d_latent,).
        """
        self.fre_model.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            z = self.fre_model.encode_reward(states_t, rewards_t, deterministic=True)
            return z.cpu().numpy()

    def get_action(self, state: np.ndarray, z: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get action from trained IQL policy conditioned on latent z.

        Args:
            state: (state_dim,) array.
            z: (d_latent,) latent vector.
            deterministic: Whether to use deterministic (mean) action.

        Returns:
            Action as numpy array of shape (action_dim,).
        """
        if self.iql_agent is None:
            raise RuntimeError("IQL agent not initialized. Run train_iql_phase() first.")

        self.iql_agent.eval()
        return self.iql_agent.get_action(state, z, deterministic=deterministic)

    def save_checkpoint(self, tag: str = "latest"):
        """
        Save a full training checkpoint.

        Args:
            tag: String tag for the checkpoint (e.g., "fre_final", "iql_step100000").
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")

        checkpoint = {
            "fre_model_state_dict": self.fre_model.state_dict(),
            "fre_step": self.fre_step,
            "iql_step": self.iql_step,
            "phase": self.phase,
            "config": self._get_config_dict(),
            "metrics_history": dict(self.metrics_history),
            "dataset_normalizer": self.dataset.normalizer.to_dict() if self.dataset.normalizer else None,
        }

        if self.iql_agent is not None:
            checkpoint["iql_agent_state_dict"] = self.iql_agent.state_dict()

        if self.iql_trainer is not None:
            checkpoint["iql_optimizer_state_dict"] = {
                "v_optimizer": self.iql_trainer.v_optimizer.state_dict(),
                "q_optimizer": self.iql_trainer.q_optimizer.state_dict(),
                "policy_optimizer": self.iql_trainer.policy_optimizer.state_dict(),
            }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, tag: str = "latest"):
        """
        Load a training checkpoint.

        Args:
            tag: String tag for the checkpoint to load.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load FRE model
        self.fre_model.load_state_dict(checkpoint["fre_model_state_dict"])
        self.fre_step = checkpoint.get("fre_step", 0)
        self.iql_step = checkpoint.get("iql_step", 0)
        self.phase = checkpoint.get("phase", "init")

        # Load metrics
        if "metrics_history" in checkpoint:
            for k, v in checkpoint["metrics_history"].items():
                self.metrics_history[k] = v

        # Load dataset normalizer if available
        if "dataset_normalizer" in checkpoint and checkpoint["dataset_normalizer"] is not None:
            self.dataset.normalizer = DatasetNormalizer.from_dict(
                checkpoint["dataset_normalizer"]
            )

        # Load IQL agent if available
        if "iql_agent_state_dict" in checkpoint:
            if self.iql_agent is None:
                self._init_iql()
            self.iql_agent.load_state_dict(checkpoint["iql_agent_state_dict"])

            # Load IQL optimizers
            if "iql_optimizer_state_dict" in checkpoint and self.iql_trainer is not None:
                opt_state = checkpoint["iql_optimizer_state_dict"]
                self.iql_trainer.v_optimizer.load_state_dict(opt_state["v_optimizer"])
                self.iql_trainer.q_optimizer.load_state_dict(opt_state["q_optimizer"])
                self.iql_trainer.policy_optimizer.load_state_dict(opt_state["policy_optimizer"])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a summary of training metrics."""
        summary = {
            "fre_step": self.fre_step,
            "iql_step": self.iql_step,
            "phase": self.phase,
            "dataset_name": self.dataset_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "d_latent": self.d_latent,
        }

        # Add recent losses
        if self.fre_trainer is not None:
            summary["fre_recent_losses"] = self.fre_trainer.get_recent_losses(window=100)

        if self.iql_trainer is not None:
            summary["iql_recent_losses"] = self.iql_trainer.get_recent_losses(window=100)

        return summary

    def export_for_evaluation(self, output_dir: str):
        """
        Export trained models for zero-shot evaluation.

        Saves:
            - FRE encoder state dict
            - IQL policy state dict
            - Dataset normalizer
            - Configuration

        Args:
            output_dir: Directory to save exported files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save FRE encoder
        torch.save(
            self.fre_model.encoder.state_dict(),
            os.path.join(output_dir, "fre_encoder.pt"),
        )

        # Save IQL policy
        if self.iql_agent is not None:
            torch.save(
                self.iql_agent.policy.state_dict(),
                os.path.join(output_dir, "iql_policy.pt"),
            )

        # Save normalizer
        if self.dataset.normalizer is not None:
            torch.save(
                self.dataset.normalizer.to_dict(),
                os.path.join(output_dir, "normalizer.pt"),
            )

        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self._get_config_dict(), f, indent=2)

        logger.info(f"Models exported to {output_dir}")


def create_strided_trainer(
    dataset_name: str = "antmaze-large-diverse-v2",
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> StridedTrainer:
    """
    Factory function to create a StridedTrainer with optional config override.

    Args:
        dataset_name: Name of the dataset to load.
        config: Optional dictionary of configuration overrides.
        **kwargs: Additional keyword arguments to override config.

    Returns:
        Configured StridedTrainer instance.
    """
    # Default configuration
    default_config = {
        "dataset_name": dataset_name,
        "d_model": 256,
        "d_reward": 32,
        "d_latent": 64,
        "num_bins": 50,
        "r_min": -1.0,
        "r_max": 1.0,
        "nhead": 4,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "decoder_hidden_dims": [256, 256],
        "beta": 0.1,
        "K_encoder": 32,
        "K_decoder": 128,
        "iql_hidden_dims": [256, 256],
        "expectile": 0.7,
        "temperature": 3.0,
        "discount": 0.99,
        "goal_threshold": 0.5,
        "linear_sparsity": 0.8,
        "mlp_hidden_dim": 256,
        "include_goal": True,
        "include_linear": True,
        "include_mlp": True,
        "fre_steps": 100_000,
        "iql_steps": 1_000_000,
        "fre_batch_size": 256,
        "iql_batch_size": 256,
        "fre_lr": 1e-4,
        "iql_lr": 3e-4,
        "log_interval": 1000,
        "eval_interval": 5000,
        "checkpoint_interval": 10000,
        "checkpoint_dir": "./checkpoints",
        "log_dir": "./logs",
        "device": "auto",
        "seed": 42,
    }

    # Apply config overrides
    if config is not None:
        default_config.update(config)

    # Apply keyword overrides
    default_config.update(kwargs)

    return StridedTrainer(**default_config)


# Domain-specific configurations
DOMAIN_CONFIGS = {
    "antmaze": {
        "expectile": 0.7,
        "temperature": 3.0,
        "discount": 0.99,
        "goal_threshold": 0.5,
        "beta": 0.1,
        "fre_steps": 100_000,
        "iql_steps": 1_000_000,
    },
    "exorl": {
        "expectile": 0.9,
        "temperature": 3.0,
        "discount": 0.99,
        "goal_threshold": 0.5,
        "beta": 0.1,
        "fre_steps": 100_000,
        "iql_steps": 1_000_000,
    },
    "kitchen": {
        "expectile": 0.7,
        "temperature": 3.0,
        "discount": 0.99,
        "goal_threshold": 0.5,
        "beta": 0.1,
        "fre_steps": 100_000,
        "iql_steps": 1_000_000,
    },
}


def create_domain_trainer(
    domain: str,
    dataset_name: str,
    **override_kwargs,
) -> StridedTrainer:
    """
    Create a StridedTrainer with domain-specific defaults.

    Args:
        domain: One of "antmaze", "exorl", "kitchen".
        dataset_name: Specific dataset name.
        **override_kwargs: Additional overrides.

    Returns:
        Configured StridedTrainer.
    """
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(DOMAIN_CONFIGS.keys())}")

    config = DOMAIN_CONFIGS[domain].copy()
    config["dataset_name"] = dataset_name
    config.update(override_kwargs)

    return StridedTrainer(**config)