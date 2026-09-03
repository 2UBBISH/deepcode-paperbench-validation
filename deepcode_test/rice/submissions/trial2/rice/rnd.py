"""
Random Network Distillation (RND) Module for RICE

Implements the exploration bonus mechanism used during the refining process.
RND uses two networks:
  - Target network f_target: fixed, randomly initialized MLP mapping state → embedding
  - Predictor network f_pred: trained to minimize MSE(f_pred(s), f_target(s))

The exploration bonus is: r_rnd(s) = ||f_pred(s) - f_target(s)||²

This encourages the agent to visit novel states where the predictor has high error.

Reference: Burda et al. "Exploration by Random Network Distillation" (ICLR 2019)
As used in RICE paper (Section 3.3, Algorithm 2)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RNDNetwork(nn.Module):
    """
    MLP network used for both target and predictor in RND.
    
    Architecture: input_dim → hidden layers → embedding_dim
    Uses orthogonal initialization for stability.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_sizes: List[int] = None,
        activation_fn: nn.Module = nn.ReLU,
        use_layer_norm: bool = False,
    ):
        """
        Args:
            input_dim: Dimension of the state/observation vector.
            embedding_dim: Dimension of the output embedding (default 128).
            hidden_sizes: List of hidden layer sizes. Default: [64, 64].
            activation_fn: Activation function class.
            use_layer_norm: Whether to use LayerNorm after each hidden layer.
        """
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        
        # Initialize weights orthogonally
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        return self.network(x)


class RNDModule:
    """
    Random Network Distillation module for exploration bonus.
    
    Maintains a fixed target network and a trainable predictor network.
    Computes intrinsic reward as MSE between predictor and target embeddings.
    
    Usage during refining:
        1. Initialize with state dimension.
        2. At each step, call compute_bonus(state) to get r_rnd.
        3. Periodically call update_predictor(states) to train predictor.
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_sizes: Optional[List[int]] = None,
        learning_rate: float = 1e-4,
        device: Union[str, torch.device] = "auto",
        normalize_obs: bool = True,
        obs_rms: Optional[Any] = None,
        clip_bonus: Optional[float] = None,
    ):
        """
        Args:
            input_dim: Dimension of the state/observation vector.
            embedding_dim: Dimension of the output embedding (default 128).
            hidden_sizes: List of hidden layer sizes. Default: [64, 64].
            learning_rate: Learning rate for predictor optimizer (default 1e-4).
            device: Device to place networks on ("auto", "cpu", "cuda").
            normalize_obs: Whether to normalize observations before feeding to RND.
            obs_rms: RunningMeanStd for observation normalization (optional).
            clip_bonus: If set, clip the exploration bonus to this maximum value.
        """
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.normalize_obs = normalize_obs
        self.obs_rms = obs_rms
        self.clip_bonus = clip_bonus
        
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        
        # Create target network (fixed, randomly initialized)
        self.target_network = RNDNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        
        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Create predictor network (trainable)
        self.predictor_network = RNDNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
        ).to(self.device)
        
        # Optimizer for predictor only
        self.optimizer = optim.Adam(
            self.predictor_network.parameters(),
            lr=learning_rate,
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Statistics tracking
        self.total_updates = 0
        self.running_loss = 0.0
        self.running_bonus_mean = 0.0
        self.running_bonus_std = 1.0
        
        # Bonus normalization (EMA)
        self.bonus_ema_mean = 0.0
        self.bonus_ema_var = 1.0
        self.bonus_ema_decay = 0.99
    
    def _preprocess_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess observations: convert to tensor, normalize if enabled.
        
        Args:
            obs: Observations as numpy array or torch tensor.
            
        Returns:
            Preprocessed tensor on the correct device.
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        obs = obs.to(self.device)
        
        # Ensure 2D: (batch_size, input_dim)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Normalize if enabled
        if self.normalize_obs and self.obs_rms is not None:
            # obs_rms expects numpy; convert, normalize, convert back
            obs_np = obs.cpu().numpy()
            obs_np = np.clip(
                (obs_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
                -5.0,
                5.0,
            )
            obs = torch.from_numpy(obs_np).float().to(self.device)
        
        return obs
    
    def compute_bonus(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute exploration bonus for given states.
        
        r_rnd(s) = ||f_pred(s) - f_target(s)||²
        
        Args:
            obs: State(s) to compute bonus for. Shape: (input_dim,) or (batch, input_dim).
            normalize: Whether to normalize bonus by running statistics.
            
        Returns:
            Exploration bonus as numpy array. Shape: (batch_size,) or scalar.
        """
        self.target_network.eval()
        self.predictor_network.eval()
        
        with torch.no_grad():
            obs_tensor = self._preprocess_obs(obs)
            
            target_embedding = self.target_network(obs_tensor)
            predictor_embedding = self.predictor_network(obs_tensor)
            
            # MSE per sample: mean over embedding dimension
            mse = ((predictor_embedding - target_embedding) ** 2).mean(dim=-1)
            
            bonus = mse.cpu().numpy()
            
            # Clip bonus if configured
            if self.clip_bonus is not None:
                bonus = np.clip(bonus, 0, self.clip_bonus)
            
            # Normalize by running statistics
            if normalize:
                # Update EMA statistics
                batch_mean = bonus.mean()
                batch_var = bonus.var() if len(bonus) > 1 else 0.0
                
                self.bonus_ema_mean = (
                    self.bonus_ema_decay * self.bonus_ema_mean
                    + (1 - self.bonus_ema_decay) * batch_mean
                )
                self.bonus_ema_var = (
                    self.bonus_ema_decay * self.bonus_ema_var
                    + (1 - self.bonus_ema_decay) * batch_var
                )
                
                # Normalize
                std = np.sqrt(self.bonus_ema_var + 1e-8)
                bonus = bonus / std
            
            # Squeeze if single sample
            if bonus.shape[0] == 1:
                bonus = bonus.item()
            
            return bonus
    
    def update_predictor(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        batch_size: int = 64,
        n_epochs: int = 1,
    ) -> Dict[str, float]:
        """
        Update predictor network to minimize MSE with target network.
        
        Args:
            obs: States to train on. Shape: (n_samples, input_dim).
            batch_size: Mini-batch size for training.
            n_epochs: Number of passes over the data.
            
        Returns:
            Dictionary with training statistics (loss, n_updates).
        """
        self.target_network.eval()
        self.predictor_network.train()
        
        obs_tensor = self._preprocess_obs(obs)
        n_samples = obs_tensor.shape[0]
        
        total_loss = 0.0
        n_updates = 0
        
        for epoch in range(n_epochs):
            # Shuffle indices
            indices = torch.randperm(n_samples, device=self.device)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                batch_obs = obs_tensor[batch_indices]
                
                # Forward pass
                with torch.no_grad():
                    target_embedding = self.target_network(batch_obs)
                
                predictor_embedding = self.predictor_network(batch_obs)
                
                # Compute loss
                loss = self.loss_fn(predictor_embedding, target_embedding)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
                self.total_updates += 1
        
        avg_loss = total_loss / max(n_updates, 1)
        self.running_loss = 0.9 * self.running_loss + 0.1 * avg_loss
        
        return {
            "loss": avg_loss,
            "n_updates": n_updates,
            "total_updates": self.total_updates,
            "running_loss": self.running_loss,
        }
    
    def update_predictor_batch(
        self,
        obs: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single batch update of predictor (for online training during rollout).
        
        Args:
            obs: States to train on. Shape: (batch_size, input_dim).
            
        Returns:
            Dictionary with training statistics.
        """
        self.target_network.eval()
        self.predictor_network.train()
        
        obs_tensor = self._preprocess_obs(obs)
        
        with torch.no_grad():
            target_embedding = self.target_network(obs_tensor)
        
        predictor_embedding = self.predictor_network(obs_tensor)
        loss = self.loss_fn(predictor_embedding, target_embedding)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        self.running_loss = 0.9 * self.running_loss + 0.1 * loss.item()
        
        return {
            "loss": loss.item(),
            "n_updates": 1,
            "total_updates": self.total_updates,
            "running_loss": self.running_loss,
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get current RND statistics.
        
        Returns:
            Dictionary with running statistics.
        """
        return {
            "total_updates": self.total_updates,
            "running_loss": self.running_loss,
            "bonus_ema_mean": self.bonus_ema_mean,
            "bonus_ema_std": np.sqrt(max(self.bonus_ema_var, 1e-8)),
        }
    
    def save(self, path: str) -> None:
        """
        Save RND module state.
        
        Args:
            path: File path to save to.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        state = {
            "target_network": self.target_network.state_dict(),
            "predictor_network": self.predictor_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "total_updates": self.total_updates,
            "running_loss": self.running_loss,
            "bonus_ema_mean": self.bonus_ema_mean,
            "bonus_ema_var": self.bonus_ema_var,
            "bonus_ema_decay": self.bonus_ema_decay,
            "normalize_obs": self.normalize_obs,
            "clip_bonus": self.clip_bonus,
        }
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """
        Load RND module state.
        
        Args:
            path: File path to load from.
        """
        state = torch.load(path, map_location=self.device)
        
        self.target_network.load_state_dict(state["target_network"])
        self.predictor_network.load_state_dict(state["predictor_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_updates = state.get("total_updates", 0)
        self.running_loss = state.get("running_loss", 0.0)
        self.bonus_ema_mean = state.get("bonus_ema_mean", 0.0)
        self.bonus_ema_var = state.get("bonus_ema_var", 1.0)
        self.bonus_ema_decay = state.get("bonus_ema_decay", 0.99)
        self.normalize_obs = state.get("normalize_obs", True)
        self.clip_bonus = state.get("clip_bonus", None)
    
    def to(self, device: Union[str, torch.device]) -> "RNDModule":
        """
        Move RND module to specified device.
        
        Args:
            device: Target device.
            
        Returns:
            Self for chaining.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.target_network = self.target_network.to(self.device)
        self.predictor_network = self.predictor_network.to(self.device)
        return self


class RunningMeanStd:
    """
    Running mean and standard deviation tracker for observation normalization.
    Used by RND to normalize observations before computing bonus.
    """
    
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        """
        Args:
            shape: Shape of the observations.
            epsilon: Small constant for numerical stability.
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with a batch of observations.
        
        Args:
            x: Batch of observations. Shape: (batch_size, *shape).
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Update running statistics from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count
    
    def normalize(self, x: np.ndarray, clip_range: float = 5.0) -> np.ndarray:
        """
        Normalize observations.
        
        Args:
            x: Observations to normalize.
            clip_range: Clip normalized values to [-clip_range, clip_range].
            
        Returns:
            Normalized observations.
        """
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8),
            -clip_range,
            clip_range,
        )


def create_rnd_module(
    input_dim: int,
    config: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "auto",
) -> RNDModule:
    """
    Factory function to create an RND module from configuration.
    
    Args:
        input_dim: Dimension of the state/observation vector.
        config: Configuration dictionary (from YAML). Uses 'rnd' section.
        device: Device to place networks on.
        
    Returns:
        Configured RNDModule instance.
    """
    if config is None:
        config = {}
    
    rnd_config = config.get("rnd", {})
    
    return RNDModule(
        input_dim=input_dim,
        embedding_dim=rnd_config.get("embedding_dim", 128),
        hidden_sizes=rnd_config.get("hidden_sizes", [64, 64]),
        learning_rate=rnd_config.get("learning_rate", 1e-4),
        device=device,
        normalize_obs=rnd_config.get("normalize_obs", True),
        clip_bonus=rnd_config.get("clip_bonus", None),
    )


# ==============================================================================
# CLI Entry Point (for testing/debugging)
# ==============================================================================

def main():
    """CLI for testing RND module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RND Module Test")
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--n-updates", type=int, default=100, help="Number of update iterations")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RND Module Test")
    print("=" * 60)
    
    # Create RND module
    rnd = RNDModule(
        input_dim=args.input_dim,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )
    print(f"Device: {rnd.device}")
    print(f"Input dim: {rnd.input_dim}")
    print(f"Embedding dim: {rnd.embedding_dim}")
    
    # Generate random test data
    test_states = np.random.randn(args.n_samples, args.input_dim).astype(np.float32)
    
    # Compute initial bonus
    initial_bonus = rnd.compute_bonus(test_states[:10])
    print(f"\nInitial bonus (first 10 samples): {initial_bonus}")
    print(f"Initial bonus mean: {initial_bonus.mean():.6f}")
    
    # Train predictor
    print(f"\nTraining predictor for {args.n_updates} iterations...")
    start_time = time.time()
    
    for i in range(args.n_updates):
        stats = rnd.update_predictor(test_states, batch_size=64, n_epochs=1)
        if (i + 1) % 20 == 0:
            print(f"  Update {i+1}/{args.n_updates}: loss={stats['loss']:.6f}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s")
    
    # Compute bonus after training
    final_bonus = rnd.compute_bonus(test_states[:10])
    print(f"\nFinal bonus (first 10 samples): {final_bonus}")
    print(f"Final bonus mean: {final_bonus.mean():.6f}")
    
    # Bonus should decrease after training (predictor learns to match target)
    print(f"\nBonus reduction: {initial_bonus.mean() - final_bonus.mean():.6f}")
    
    # Test on novel states
    novel_states = np.random.randn(10, args.input_dim).astype(np.float32) * 2.0
    novel_bonus = rnd.compute_bonus(novel_states)
    print(f"\nNovel states bonus: {novel_bonus}")
    print(f"Novel bonus mean: {novel_bonus.mean():.6f}")
    print("(Novel states should have higher bonus than familiar ones)")
    
    # Save and load test
    save_path = "/tmp/rnd_test.pt"
    rnd.save(save_path)
    print(f"\nSaved RND to {save_path}")
    
    rnd2 = RNDModule(input_dim=args.input_dim, embedding_dim=args.embedding_dim)
    rnd2.load(save_path)
    
    bonus_after_load = rnd2.compute_bonus(test_states[:10])
    print(f"Bonus after load: {bonus_after_load}")
    print(f"Load test {'PASSED' if np.allclose(final_bonus, bonus_after_load, atol=1e-5) else 'FAILED'}")
    
    # Cleanup
    os.remove(save_path)
    
    print("\n" + "=" * 60)
    print("RND Module Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()