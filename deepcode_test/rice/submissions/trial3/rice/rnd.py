"""
Random Network Distillation (RND) Module

Implements the exploration bonus mechanism used during the RICE refining phase.
RND uses a fixed, randomly-initialized target network and a trainable predictor
network. The prediction error serves as an intrinsic exploration bonus that
encourages the agent to visit novel states.

Reference: Burda et al. (2019) "Exploration by Random Network Distillation"
As used in RICE: "RICE: Refining via Critical State Explanation"

Components:
    - Target network f: fixed, randomly initialized MLP
    - Predictor network f̂: same architecture, trained to predict f(s)
    - Bonus: r_rnd(s) = ||f̂(s) - f(s)||² (MSE)

Architecture:
    - Input dim = state_dim
    - Hidden layers: [64, 64] (default)
    - Output dim = embedding_dim (default 64)
    - Activation: ReLU (or LeakyReLU)
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ==============================================================================
# RND Network Components
# ==============================================================================

class RNDNetwork(nn.Module):
    """
    MLP network used for both target (fixed) and predictor (trainable) in RND.

    Args:
        input_dim: Dimension of the state space.
        hidden_sizes: Tuple of hidden layer sizes. Default: (64, 64).
        output_dim: Dimension of the embedding space. Default: 64.
        activation: Activation function name. Options: "relu", "leaky_relu", "tanh".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        output_dim: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using orthogonal initialization (consistent with PPO)
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Return activation module by name."""
        if name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self):
        """Orthogonal initialization for linear layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: State tensor of shape (batch_size, input_dim).

        Returns:
            Embedding tensor of shape (batch_size, output_dim).
        """
        return self.network(x)


# ==============================================================================
# RND Module (Main Interface)
# ==============================================================================

class RNDModule:
    """
    Random Network Distillation module for computing exploration bonuses.

    Maintains a fixed target network and a trainable predictor network.
    The exploration bonus is the MSE between their outputs.

    Args:
        state_dim: Dimension of the state space.
        hidden_sizes: Hidden layer sizes for both networks. Default: (64, 64).
        embedding_dim: Output dimension of the embedding. Default: 64.
        learning_rate: Learning rate for predictor optimizer. Default: 1e-4.
        device: Device to run on ("cpu" or "cuda").
        activation: Activation function. Default: "relu".
        normalize_obs: Whether to normalize observations before computing bonus.
            Default: True (using running mean/std).
        obs_rms_decay: Decay rate for running mean/std of observations.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        embedding_dim: int = 64,
        learning_rate: float = 1e-4,
        device: str = "cpu",
        activation: str = "relu",
        normalize_obs: bool = True,
        obs_rms_decay: float = 0.99,
    ):
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.device = device
        self.normalize_obs = normalize_obs

        # Target network (fixed, randomly initialized)
        self.target_network = RNDNetwork(
            input_dim=state_dim,
            hidden_sizes=hidden_sizes,
            output_dim=embedding_dim,
            activation=activation,
        ).to(device)
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Predictor network (trainable)
        self.predictor_network = RNDNetwork(
            input_dim=state_dim,
            hidden_sizes=hidden_sizes,
            output_dim=embedding_dim,
            activation=activation,
        ).to(device)

        # Optimizer for predictor
        self.optimizer = optim.Adam(
            self.predictor_network.parameters(),
            lr=learning_rate,
        )

        # Running statistics for observation normalization
        if normalize_obs:
            self.obs_rms = RunningMeanStd(shape=(state_dim,), decay=obs_rms_decay)
        else:
            self.obs_rms = None

        # Training statistics
        self.total_updates = 0
        self.cumulative_loss = 0.0

    def compute_bonus(self, states: np.ndarray) -> np.ndarray:
        """
        Compute the RND exploration bonus for a batch of states.

        r_rnd(s) = ||f̂(s) - f(s)||²

        Args:
            states: numpy array of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            numpy array of bonus values, shape (batch_size,) or scalar.
        """
        squeeze_output = (states.ndim == 1)
        if squeeze_output:
            states = states.reshape(1, -1)

        # Normalize observations if enabled
        if self.normalize_obs and self.obs_rms is not None:
            states_normalized = self.obs_rms.normalize(states)
        else:
            states_normalized = states

        # Convert to tensor
        states_tensor = torch.FloatTensor(states_normalized).to(self.device)

        with torch.no_grad():
            target_embedding = self.target_network(states_tensor)
            predictor_embedding = self.predictor_network(states_tensor)

            # MSE per sample: mean over embedding dimension
            mse = ((predictor_embedding - target_embedding) ** 2).mean(dim=1)

        bonus = mse.cpu().numpy()

        if squeeze_output:
            return float(bonus[0])
        return bonus

    def update(self, states: np.ndarray, num_epochs: int = 1) -> Dict[str, float]:
        """
        Update the predictor network to better match the target network.

        Args:
            states: numpy array of shape (num_states, state_dim).
            num_epochs: Number of epochs to train on this batch.

        Returns:
            Dictionary with training statistics (loss, etc.).
        """
        if len(states) == 0:
            return {"rnd_loss": 0.0}

        # Update running statistics
        if self.normalize_obs and self.obs_rms is not None:
            self.obs_rms.update(states)
            states_normalized = self.obs_rms.normalize(states)
        else:
            states_normalized = states

        states_tensor = torch.FloatTensor(states_normalized).to(self.device)

        # Compute target embeddings (fixed)
        with torch.no_grad():
            target_embeddings = self.target_network(states_tensor)

        total_loss = 0.0
        num_batches = 0

        self.predictor_network.train()

        for epoch in range(num_epochs):
            # Shuffle for better training
            perm = torch.randperm(len(states_tensor), device=self.device)
            shuffled_states = states_tensor[perm]
            shuffled_targets = target_embeddings[perm]

            # Process in mini-batches of 256
            batch_size = 256
            for start in range(0, len(shuffled_states), batch_size):
                end = min(start + batch_size, len(shuffled_states))
                batch_states = shuffled_states[start:end]
                batch_targets = shuffled_targets[start:end]

                self.optimizer.zero_grad()

                predictions = self.predictor_network(batch_states)
                loss = F.mse_loss(predictions, batch_targets)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        self.total_updates += 1
        avg_loss = total_loss / max(num_batches, 1)
        self.cumulative_loss += avg_loss

        return {"rnd_loss": avg_loss}

    def update_on_trajectory(
        self,
        states: np.ndarray,
        num_epochs: int = 4,
    ) -> Dict[str, float]:
        """
        Convenience method: update RND predictor on states from a trajectory.

        This is the typical usage during refining: after collecting a trajectory,
        call this to update the predictor.

        Args:
            states: numpy array of shape (trajectory_length, state_dim).
            num_epochs: Number of training epochs on this trajectory's states.

        Returns:
            Training statistics dict.
        """
        return self.update(states, num_epochs=num_epochs)

    def get_normalized_bonus(
        self,
        states: np.ndarray,
        bonus_mean: Optional[float] = None,
        bonus_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute normalized exploration bonus (bonus divided by running std).

        This normalization helps keep the bonus scale consistent across training.

        Args:
            states: numpy array of shape (batch_size, state_dim).
            bonus_mean: Optional running mean of bonuses for normalization.
            bonus_std: Optional running std of bonuses for normalization.

        Returns:
            Normalized bonus values.
        """
        raw_bonus = self.compute_bonus(states)

        if bonus_std is not None and bonus_std > 1e-8:
            normalized = raw_bonus / bonus_std
        else:
            normalized = raw_bonus

        return normalized

    def save(self, path: str):
        """Save RND module state to disk."""
        checkpoint = {
            "predictor_state_dict": self.predictor_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
            "cumulative_loss": self.cumulative_loss,
            "state_dim": self.state_dim,
            "embedding_dim": self.embedding_dim,
        }
        if self.obs_rms is not None:
            checkpoint["obs_rms_mean"] = self.obs_rms.mean
            checkpoint["obs_rms_var"] = self.obs_rms.var
            checkpoint["obs_rms_count"] = self.obs_rms.count
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load RND module state from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.predictor_network.load_state_dict(checkpoint["predictor_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_updates = checkpoint.get("total_updates", 0)
        self.cumulative_loss = checkpoint.get("cumulative_loss", 0.0)
        if self.obs_rms is not None and "obs_rms_mean" in checkpoint:
            self.obs_rms.mean = checkpoint["obs_rms_mean"]
            self.obs_rms.var = checkpoint["obs_rms_var"]
            self.obs_rms.count = checkpoint["obs_rms_count"]

    def to(self, device: str):
        """Move networks to specified device."""
        self.device = device
        self.target_network = self.target_network.to(device)
        self.predictor_network = self.predictor_network.to(device)
        # Recreate optimizer on new device
        self.optimizer = optim.Adam(
            self.predictor_network.parameters(),
            lr=self.learning_rate,
        )


# ==============================================================================
# Running Mean and Standard Deviation (for observation normalization)
# ==============================================================================

class RunningMeanStd:
    """
    Tracks running mean and standard deviation of a data stream.

    Used for normalizing observations before feeding to RND networks,
    which improves stability of the exploration bonus.

    Based on the implementation in Stable-Baselines3.

    Args:
        shape: Shape of the data (e.g., (state_dim,)).
        decay: Decay factor for exponential moving average. Default: 0.99.
        epsilon: Small value to avoid division by zero. Default: 1e-4.
    """

    def __init__(
        self,
        shape: Tuple[int, ...] = (),
        decay: float = 0.99,
        epsilon: float = 1e-4,
    ):
        self.shape = shape
        self.decay = decay
        self.epsilon = epsilon

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """
        Update running statistics with a batch of data.

        Args:
            x: numpy array of shape (batch_size, *shape).
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
    ):
        """Update running statistics from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize data using running statistics.

        Args:
            x: numpy array of shape (batch_size, *shape).

        Returns:
            Normalized data: (x - mean) / sqrt(var + epsilon).
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)


# ==============================================================================
# Bonus Normalizer (for keeping RND bonus at consistent scale)
# ==============================================================================

class BonusNormalizer:
    """
    Maintains running statistics of RND bonuses for normalization.

    During refining, the RND bonus scale can change as the predictor improves.
    Normalizing by a running standard deviation keeps the bonus contribution
    stable relative to the environment reward.

    Args:
        decay: Decay factor for exponential moving average. Default: 0.99.
        epsilon: Small value to avoid division by zero. Default: 1e-8.
    """

    def __init__(self, decay: float = 0.99, epsilon: float = 1e-8):
        self.decay = decay
        self.epsilon = epsilon
        self.running_mean = 0.0
        self.running_std = 1.0
        self.initialized = False

    def update(self, bonuses: np.ndarray):
        """
        Update running statistics with a batch of bonus values.

        Args:
            bonuses: numpy array of bonus values.
        """
        batch_mean = np.mean(bonuses)
        batch_std = np.std(bonuses)

        if not self.initialized:
            self.running_mean = batch_mean
            self.running_std = batch_std
            self.initialized = True
        else:
            self.running_mean = (
                self.decay * self.running_mean + (1 - self.decay) * batch_mean
            )
            self.running_std = (
                self.decay * self.running_std + (1 - self.decay) * batch_std
            )

    def normalize(self, bonuses: np.ndarray) -> np.ndarray:
        """
        Normalize bonuses: (bonus - mean) / (std + epsilon).

        Args:
            bonuses: numpy array of bonus values.

        Returns:
            Normalized bonuses.
        """
        if not self.initialized:
            return bonuses
        return (bonuses - self.running_mean) / (self.running_std + self.epsilon)


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_rnd_module(
    state_dim: int,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    embedding_dim: int = 64,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    normalize_obs: bool = True,
) -> RNDModule:
    """
    Factory function to create an RND module with default settings.

    Args:
        state_dim: Dimension of the state space.
        hidden_sizes: Hidden layer sizes. Default: (64, 64).
        embedding_dim: Embedding dimension. Default: 64.
        learning_rate: Learning rate for predictor. Default: 1e-4.
        device: Device string. Default: "cpu".
        normalize_obs: Whether to normalize observations. Default: True.

    Returns:
        Configured RNDModule instance.
    """
    return RNDModule(
        state_dim=state_dim,
        hidden_sizes=hidden_sizes,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        device=device,
        normalize_obs=normalize_obs,
    )


def compute_rnd_bonus_batch(
    rnd_module: RNDModule,
    states: np.ndarray,
    bonus_normalizer: Optional[BonusNormalizer] = None,
    lambda_coef: float = 0.01,
) -> np.ndarray:
    """
    Compute the combined RND bonus for a batch of states:
        r_combined = lambda_coef * r_rnd(s)

    Optionally normalizes the bonus using a BonusNormalizer.

    Args:
        rnd_module: The RND module.
        states: numpy array of shape (batch_size, state_dim).
        bonus_normalizer: Optional BonusNormalizer for scaling.
        lambda_coef: Weight coefficient λ for the exploration bonus.

    Returns:
        numpy array of combined bonus values, shape (batch_size,).
    """
    raw_bonus = rnd_module.compute_bonus(states)

    if bonus_normalizer is not None:
        bonus_normalizer.update(raw_bonus)
        normalized_bonus = bonus_normalizer.normalize(raw_bonus)
    else:
        normalized_bonus = raw_bonus

    return lambda_coef * normalized_bonus