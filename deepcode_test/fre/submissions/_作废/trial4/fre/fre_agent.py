"""
FRE Agent: Main training loop implementing Algorithm 1 from the paper.

Implements the strided training scheme:
  Phase 1: Train encoder+decoder (FRE) on diverse reward functions until convergence.
  Phase 2: Freeze encoder; train IQL agent conditioned on latent z.

This module orchestrates the full FRE pipeline for zero-shot offline RL.
"""

import os
import time
import copy
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from fre.encoder import RewardEncoder, vae_loss
from fre.decoder import RewardDecoder, reconstruction_loss, create_decoder
from fre.reward_prior import RewardPrior, create_reward_prior
from fre.iql import IQLAgent
from fre.utils import ReplayBuffer, set_seed, get_device


class FREAgent:
    """
    Functional Reward Encodings (FRE) agent.

    Implements Algorithm 1 from the paper:
      1. Train encoder-decoder VAE on random reward functions (Phase 1).
      2. Freeze encoder; train IQL agent conditioned on latent z (Phase 2).

    At evaluation time, given a downstream reward function, encode it to z
    using K=32 (state, reward) pairs, then roll out the policy π(a|s, z).

    Args:
        state_dim: Dimensionality of the state space.
        action_dim: Dimensionality of the action space.
        replay_buffer: ReplayBuffer containing the offline dataset.
        reward_prior: RewardPrior for sampling unsupervised reward functions.
        encoder_kwargs: Keyword arguments for RewardEncoder.
        decoder_kwargs: Keyword arguments for RewardDecoder.
        iql_kwargs: Keyword arguments for IQLAgent.
        d_z: Latent dimension (default 64).
        K_encoder: Number of (state, reward) pairs for encoder input (default 32).
        K_decoder: Number of states for decoder reconstruction (default 64).
        beta: KL divergence weight for VAE (default 1.0).
        vae_lr: Learning rate for VAE (encoder+decoder) training.
        iql_lr: Learning rate for IQL training (overrides iql_kwargs if set).
        device: torch device string or object.
        checkpoint_dir: Directory for saving model checkpoints.
        log_dir: Directory for tensorboard logs.
        use_wandb: Whether to use Weights & Biases logging.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer: ReplayBuffer,
        reward_prior: RewardPrior,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        iql_kwargs: Optional[Dict[str, Any]] = None,
        d_z: int = 64,
        K_encoder: int = 32,
        K_decoder: int = 64,
        beta: float = 1.0,
        vae_lr: float = 3e-4,
        iql_lr: float = 3e-4,
        device: Optional[Any] = None,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        use_wandb: bool = False,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_z = d_z
        self.K_encoder = K_encoder
        self.K_decoder = K_decoder
        self.beta = beta
        self.vae_lr = vae_lr
        self.iql_lr = iql_lr
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb

        # Device
        if device is None:
            self.device = get_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Data
        self.replay_buffer = replay_buffer
        self.reward_prior = reward_prior

        # Build encoder
        encoder_kwargs = encoder_kwargs or {}
        encoder_kwargs.setdefault("state_dim", state_dim)
        encoder_kwargs.setdefault("d_model", 256)
        encoder_kwargs.setdefault("nhead", 4)
        encoder_kwargs.setdefault("num_layers", 3)
        encoder_kwargs.setdefault("d_z", d_z)
        encoder_kwargs.setdefault("num_reward_bins", 50)
        encoder_kwargs.setdefault("dropout", 0.1)
        self.encoder = RewardEncoder(**encoder_kwargs).to(self.device)

        # Build decoder
        decoder_kwargs = decoder_kwargs or {}
        decoder_kwargs.setdefault("state_dim", state_dim)
        decoder_kwargs.setdefault("d_z", d_z)
        decoder_kwargs.setdefault("hidden_dims", [256, 256])
        self.decoder = create_decoder(**decoder_kwargs).to(self.device)

        # Build IQL agent
        iql_kwargs = iql_kwargs or {}
        iql_kwargs.setdefault("state_dim", state_dim)
        iql_kwargs.setdefault("action_dim", action_dim)
        iql_kwargs.setdefault("d_z", d_z)
        iql_kwargs.setdefault("hidden_dims", (256, 256))
        iql_kwargs.setdefault("lr", iql_lr)
        iql_kwargs.setdefault("device", self.device)
        self.iql_agent = IQLAgent(**iql_kwargs)

        # Optimizers
        self.vae_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=vae_lr,
        )

        # Training state
        self.phase = "init"  # "vae", "iql", or "done"
        self.vae_step = 0
        self.iql_step = 0
        self.total_vae_steps = 0
        self.total_iql_steps = 0

        # Logging
        self.writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        # Wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project="fre", reinit=True)
            except ImportError:
                print("[FREAgent] wandb not installed; skipping wandb logging.")
                self.use_wandb = False

        # Checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Metrics history
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Phase 1: VAE Training (Encoder + Decoder)
    # ------------------------------------------------------------------

    def train_vae_step(self) -> Dict[str, float]:
        """
        Perform one gradient step of VAE training (Phase 1).

        Samples a random reward function η ~ p(η), encodes K_encoder
        (state, reward) pairs to z, decodes K_decoder states, and
        minimizes reconstruction + KL loss.

        Returns:
            Dictionary of loss values for logging.
        """
        self.encoder.train()
        self.decoder.train()

        # 1. Sample reward function
        reward_fn, reward_type = self.reward_prior.sample()

        # 2. Sample encoder states (K_encoder) and compute rewards
        encoder_states_np = self.replay_buffer.sample_states(self.K_encoder)
        encoder_rewards_np = reward_fn(encoder_states_np)

        # 3. Sample decoder states (K_decoder, disjoint from encoder)
        decoder_states_np = self.replay_buffer.sample_states(self.K_decoder)
        decoder_rewards_np = reward_fn(decoder_states_np)

        # Convert to tensors
        encoder_states = torch.FloatTensor(encoder_states_np).to(self.device)  # (K, state_dim)
        encoder_rewards = torch.FloatTensor(encoder_rewards_np).to(self.device)  # (K,)
        decoder_states = torch.FloatTensor(decoder_states_np).to(self.device)  # (K', state_dim)
        decoder_rewards = torch.FloatTensor(decoder_rewards_np).to(self.device)  # (K',)

        # Add batch dimension for encoder (batch_size=1 for single reward function)
        encoder_states = encoder_states.unsqueeze(0)  # (1, K, state_dim)
        encoder_rewards = encoder_rewards.unsqueeze(0)  # (1, K)
        decoder_states = decoder_states.unsqueeze(0)  # (1, K', state_dim)
        decoder_rewards = decoder_rewards.unsqueeze(0)  # (1, K')

        # 4. Forward encoder → z
        mu, logvar = self.encoder(encoder_states, encoder_rewards)
        z = self.encoder.reparameterize(mu, logvar)  # (1, d_z)

        # 5. Forward decoder → predicted rewards
        pred_rewards = self.decoder(decoder_states, z)  # (1, K')

        # 6. Compute losses
        recon_loss = reconstruction_loss(self.decoder, decoder_states, decoder_rewards, z)
        total_loss = vae_loss(recon_loss, mu, logvar, beta=self.beta)

        # 7. Backward pass
        self.vae_optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=10.0)
        self.vae_optimizer.step()

        self.vae_step += 1
        self.total_vae_steps += 1

        # Compute KL divergence for logging
        kl_div = (total_loss - recon_loss) / self.beta if self.beta > 0 else torch.tensor(0.0)

        metrics = {
            "vae/total_loss": total_loss.item(),
            "vae/recon_loss": recon_loss.item(),
            "vae/kl_div": kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
            "vae/reward_type": reward_type,
        }

        return metrics

    def train_vae(
        self,
        num_steps: int = 100_000,
        log_interval: int = 1000,
        eval_interval: int = 5000,
        save_interval: int = 10000,
        early_stop_patience: Optional[int] = None,
        early_stop_min_delta: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Run Phase 1: VAE training for num_steps gradient steps.

        Args:
            num_steps: Number of gradient steps.
            log_interval: Steps between logging.
            eval_interval: Steps between evaluation logging.
            save_interval: Steps between checkpoint saves.
            early_stop_patience: If set, stop early when loss plateaus.
            early_stop_min_delta: Minimum change to count as improvement.

        Returns:
            Dictionary of training statistics.
        """
        self.phase = "vae"
        print(f"[FREAgent] Starting Phase 1: VAE training for {num_steps} steps...")

        best_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        for step in range(num_steps):
            metrics = self.train_vae_step()

            # Logging
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[VAE] Step {step+1}/{num_steps} | "
                    f"Total Loss: {metrics['vae/total_loss']:.4f} | "
                    f"Recon: {metrics['vae/recon_loss']:.4f} | "
                    f"KL: {metrics['vae/kl_div']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
                if self.writer:
                    self.writer.add_scalar("vae/total_loss", metrics["vae/total_loss"], self.total_vae_steps)
                    self.writer.add_scalar("vae/recon_loss", metrics["vae/recon_loss"], self.total_vae_steps)
                    self.writer.add_scalar("vae/kl_div", metrics["vae/kl_div"], self.total_vae_steps)

            # Evaluation logging (more detailed)
            if (step + 1) % eval_interval == 0:
                eval_metrics = self._evaluate_vae_reconstruction(num_samples=20)
                metrics.update(eval_metrics)
                if self.writer:
                    for k, v in eval_metrics.items():
                        self.writer.add_scalar(f"vae_eval/{k}", v, self.total_vae_steps)

            # Save checkpoint
            if (step + 1) % save_interval == 0:
                self.save_checkpoint(tag=f"vae_step{step+1}")

            # Early stopping
            if early_stop_patience is not None:
                current_loss = metrics["vae/total_loss"]
                if current_loss < best_loss - early_stop_min_delta:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"[FREAgent] Early stopping at step {step+1}")
                    break

            # Store metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.metrics_history[k].append(v)

        elapsed = time.time() - start_time
        print(f"[FREAgent] Phase 1 completed in {elapsed:.1f}s ({self.total_vae_steps} steps)")

        # Save final VAE checkpoint
        self.save_checkpoint(tag="vae_final")

        return {
            "total_steps": self.total_vae_steps,
            "elapsed_time": elapsed,
            "final_loss": metrics.get("vae/total_loss", None),
        }

    def _evaluate_vae_reconstruction(self, num_samples: int = 20) -> Dict[str, float]:
        """
        Evaluate VAE reconstruction quality on held-out reward functions.

        Args:
            num_samples: Number of reward functions to evaluate.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.encoder.eval()
        self.decoder.eval()

        total_recon = 0.0
        total_kl = 0.0

        with torch.no_grad():
            for _ in range(num_samples):
                reward_fn, _ = self.reward_prior.sample()

                encoder_states_np = self.replay_buffer.sample_states(self.K_encoder)
                encoder_rewards_np = reward_fn(encoder_states_np)

                decoder_states_np = self.replay_buffer.sample_states(self.K_decoder)
                decoder_rewards_np = reward_fn(decoder_states_np)

                encoder_states = torch.FloatTensor(encoder_states_np).unsqueeze(0).to(self.device)
                encoder_rewards = torch.FloatTensor(encoder_rewards_np).unsqueeze(0).to(self.device)
                decoder_states = torch.FloatTensor(decoder_states_np).unsqueeze(0).to(self.device)
                decoder_rewards = torch.FloatTensor(decoder_rewards_np).unsqueeze(0).to(self.device)

                mu, logvar = self.encoder(encoder_states, encoder_rewards)
                z = self.encoder.reparameterize(mu, logvar)

                recon_loss = reconstruction_loss(self.decoder, decoder_states, decoder_rewards, z)
                kl = (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=-1)).mean()

                total_recon += recon_loss.item()
                total_kl += kl.item()

        self.encoder.train()
        self.decoder.train()

        return {
            "eval_recon_loss": total_recon / num_samples,
            "eval_kl_div": total_kl / num_samples,
        }

    # ------------------------------------------------------------------
    # Phase 2: IQL Training (Frozen Encoder)
    # ------------------------------------------------------------------

    def train_iql_step(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Perform one gradient step of IQL training (Phase 2).

        Samples a random reward function η, encodes it to z using the
        frozen encoder, samples a batch of transitions from the replay
        buffer, relabels rewards with η(s), and updates IQL networks.

        Args:
            batch_size: Batch size for IQL training.

        Returns:
            Dictionary of loss values for logging.
        """
        self.encoder.eval()  # Frozen encoder
        self.iql_agent.train()

        # 1. Sample reward function and encode to z
        reward_fn, reward_type = self.reward_prior.sample()

        encoder_states_np = self.replay_buffer.sample_states(self.K_encoder)
        encoder_rewards_np = reward_fn(encoder_states_np)

        encoder_states = torch.FloatTensor(encoder_states_np).unsqueeze(0).to(self.device)
        encoder_rewards = torch.FloatTensor(encoder_rewards_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.encoder.encode_deterministic(encoder_states, encoder_rewards)
        # z shape: (1, d_z)

        # 2. Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # 3. Relabel rewards using the sampled reward function
        # Compute η(s) for the sampled states
        states_np = states.cpu().numpy()
        rewards_np = reward_fn(states_np)
        rewards = torch.FloatTensor(rewards_np).to(self.device)

        # 4. Expand z to match batch size
        z_batch = z.expand(batch_size, -1)  # (B, d_z)

        # 5. IQL training step
        iql_metrics = self.iql_agent.train_step(
            batch={
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "dones": dones,
            },
            z=z_batch,
            update_target=True,
        )

        self.iql_step += 1
        self.total_iql_steps += 1

        # Add reward type info
        iql_metrics["iql/reward_type"] = reward_type

        return iql_metrics

    def train_iql(
        self,
        num_steps: int = 1_000_000,
        batch_size: int = 256,
        log_interval: int = 1000,
        eval_interval: int = 10000,
        save_interval: int = 50000,
    ) -> Dict[str, Any]:
        """
        Run Phase 2: IQL training with frozen encoder.

        Args:
            num_steps: Number of gradient steps.
            batch_size: Batch size per step.
            log_interval: Steps between logging.
            eval_interval: Steps between evaluation logging.
            save_interval: Steps between checkpoint saves.

        Returns:
            Dictionary of training statistics.
        """
        self.phase = "iql"
        print(f"[FREAgent] Starting Phase 2: IQL training for {num_steps} steps...")

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        start_time = time.time()

        for step in range(num_steps):
            metrics = self.train_iql_step(batch_size=batch_size)

            # Logging
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[IQL] Step {step+1}/{num_steps} | "
                    f"V Loss: {metrics.get('iql/v_loss', 0):.4f} | "
                    f"Q Loss: {metrics.get('iql/q_loss', 0):.4f} | "
                    f"Policy Loss: {metrics.get('iql/policy_loss', 0):.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
                if self.writer:
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(k, v, self.total_iql_steps)

            # Save checkpoint
            if (step + 1) % save_interval == 0:
                self.save_checkpoint(tag=f"iql_step{step+1}")

            # Store metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.metrics_history[k].append(v)

        elapsed = time.time() - start_time
        print(f"[FREAgent] Phase 2 completed in {elapsed:.1f}s ({self.total_iql_steps} steps)")

        # Save final checkpoint
        self.save_checkpoint(tag="iql_final")

        self.phase = "done"

        return {
            "total_steps": self.total_iql_steps,
            "elapsed_time": elapsed,
        }

    # ------------------------------------------------------------------
    # Full Training (Algorithm 1)
    # ------------------------------------------------------------------

    def train(
        self,
        vae_steps: int = 100_000,
        iql_steps: int = 1_000_000,
        batch_size: int = 256,
        vae_log_interval: int = 1000,
        iql_log_interval: int = 1000,
        vae_save_interval: int = 10000,
        iql_save_interval: int = 50000,
        early_stop_patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full strided training scheme (Algorithm 1).

        Phase 1: Train encoder+decoder VAE.
        Phase 2: Freeze encoder, train IQL agent.

        Args:
            vae_steps: Number of VAE training steps.
            iql_steps: Number of IQL training steps.
            batch_size: Batch size for IQL training.
            vae_log_interval: Logging interval for VAE phase.
            iql_log_interval: Logging interval for IQL phase.
            vae_save_interval: Checkpoint interval for VAE phase.
            iql_save_interval: Checkpoint interval for IQL phase.
            early_stop_patience: Early stopping patience for VAE phase.

        Returns:
            Dictionary with training statistics for both phases.
        """
        print("=" * 60)
        print("[FREAgent] Starting full training (Algorithm 1)")
        print(f"  Phase 1 (VAE): {vae_steps} steps")
        print(f"  Phase 2 (IQL): {iql_steps} steps")
        print(f"  Latent dim: {self.d_z}")
        print(f"  K_encoder: {self.K_encoder}, K_decoder: {self.K_decoder}")
        print(f"  Beta: {self.beta}")
        print("=" * 60)

        # Phase 1: VAE
        vae_stats = self.train_vae(
            num_steps=vae_steps,
            log_interval=vae_log_interval,
            save_interval=vae_save_interval,
            early_stop_patience=early_stop_patience,
        )

        # Phase 2: IQL
        iql_stats = self.train_iql(
            num_steps=iql_steps,
            batch_size=batch_size,
            log_interval=iql_log_interval,
            save_interval=iql_save_interval,
        )

        print("=" * 60)
        print("[FREAgent] Training complete!")
        print("=" * 60)

        return {
            "vae_stats": vae_stats,
            "iql_stats": iql_stats,
        }

    # ------------------------------------------------------------------
    # Encoding a downstream reward function (for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_reward(
        self,
        reward_fn: Any,
        K: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Encode a downstream reward function into latent vector z.

        Samples K states from the dataset, computes η(s) for each,
        and passes through the frozen encoder.

        Args:
            reward_fn: Callable reward function η(s) → scalar or array.
            K: Number of (state, reward) pairs (default: self.K_encoder).
            deterministic: If True, use μ directly; else sample from N(μ, σ²).

        Returns:
            Latent vector z of shape (1, d_z).
        """
        if K is None:
            K = self.K_encoder

        self.encoder.eval()

        # Sample states and compute rewards
        states_np = self.replay_buffer.sample_states(K)
        rewards_np = reward_fn(states_np)

        states = torch.FloatTensor(states_np).unsqueeze(0).to(self.device)
        rewards = torch.FloatTensor(rewards_np).unsqueeze(0).to(self.device)

        if deterministic:
            z = self.encoder.encode_deterministic(states, rewards)
        else:
            mu, logvar = self.encoder(states, rewards)
            z = self.encoder.reparameterize(mu, logvar)

        return z

    @torch.no_grad()
    def get_action(
        self,
        state: np.ndarray,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get action from the IQL policy conditioned on latent z.

        Args:
            state: State array of shape (state_dim,).
            z: Latent vector of shape (1, d_z) or (d_z,).
            deterministic: If True, use mean action; else sample.

        Returns:
            Action array of shape (action_dim,).
        """
        self.iql_agent.eval()
        return self.iql_agent.get_action(state, z, deterministic=deterministic)

    @torch.no_grad()
    def get_actions(
        self,
        states: np.ndarray,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get actions for a batch of states.

        Args:
            states: States array of shape (B, state_dim).
            z: Latent vector of shape (1, d_z) or (B, d_z).
            deterministic: If True, use mean action; else sample.

        Returns:
            Actions array of shape (B, action_dim).
        """
        self.iql_agent.eval()
        return self.iql_agent.get_actions(states, z, deterministic=deterministic)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> str:
        """
        Save full agent checkpoint.

        Args:
            tag: Identifier for the checkpoint file.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "iql_state_dict": self.iql_agent.state_dict(),
            "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
            "phase": self.phase,
            "vae_step": self.vae_step,
            "iql_step": self.iql_step,
            "total_vae_steps": self.total_vae_steps,
            "total_iql_steps": self.total_iql_steps,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "d_z": self.d_z,
                "K_encoder": self.K_encoder,
                "K_decoder": self.K_decoder,
                "beta": self.beta,
            },
        }

        path = os.path.join(self.checkpoint_dir, f"fre_agent_{tag}.pt")
        torch.save(checkpoint, path)
        print(f"[FREAgent] Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: str, load_optimizer: bool = False) -> None:
        """
        Load agent checkpoint.

        Args:
            path: Path to the checkpoint file.
            load_optimizer: Whether to load optimizer state.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.iql_agent.load_state_dict(checkpoint["iql_state_dict"])

        if load_optimizer and "vae_optimizer_state_dict" in checkpoint:
            self.vae_optimizer.load_state_dict(checkpoint["vae_optimizer_state_dict"])

        self.phase = checkpoint.get("phase", "init")
        self.vae_step = checkpoint.get("vae_step", 0)
        self.iql_step = checkpoint.get("iql_step", 0)
        self.total_vae_steps = checkpoint.get("total_vae_steps", 0)
        self.total_iql_steps = checkpoint.get("total_iql_steps", 0)

        print(f"[FREAgent] Checkpoint loaded from {path} (phase={self.phase})")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def to(self, device: Any) -> "FREAgent":
        """Move all networks to the specified device."""
        self.device = device if not isinstance(device, str) else torch.device(device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.iql_agent.to(self.device)
        return self

    def train(self) -> None:
        """Set all networks to training mode."""
        self.encoder.train()
        self.decoder.train()
        self.iql_agent.train()

    def eval(self) -> None:
        """Set all networks to evaluation mode."""
        self.encoder.eval()
        self.decoder.eval()
        self.iql_agent.eval()

    def close(self) -> None:
        """Clean up logging resources."""
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()


def create_fre_agent(
    state_dim: int,
    action_dim: int,
    replay_buffer: ReplayBuffer,
    reward_prior: RewardPrior,
    config: Optional[Dict[str, Any]] = None,
) -> FREAgent:
    """
    Factory function to create an FREAgent from a configuration dictionary.

    Args:
        state_dim: State dimensionality.
        action_dim: Action dimensionality.
        replay_buffer: ReplayBuffer with offline data.
        reward_prior: RewardPrior for sampling reward functions.
        config: Configuration dictionary with keys:
            - d_z, K_encoder, K_decoder, beta
            - vae_lr, iql_lr
            - encoder_kwargs, decoder_kwargs, iql_kwargs
            - checkpoint_dir, log_dir, use_wandb
            - device

    Returns:
        Configured FREAgent instance.
    """
    config = config or {}

    return FREAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        reward_prior=reward_prior,
        d_z=config.get("d_z", 64),
        K_encoder=config.get("K_encoder", 32),
        K_decoder=config.get("K_decoder", 64),
        beta=config.get("beta", 1.0),
        vae_lr=config.get("vae_lr", 3e-4),
        iql_lr=config.get("iql_lr", 3e-4),
        encoder_kwargs=config.get("encoder_kwargs", None),
        decoder_kwargs=config.get("decoder_kwargs", None),
        iql_kwargs=config.get("iql_kwargs", None),
        device=config.get("device", None),
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
        log_dir=config.get("log_dir", "./logs"),
        use_wandb=config.get("use_wandb", False),
    )