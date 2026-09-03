"""
Random Network Distillation (RND) exploration bonus.

Based on Burda et al. (2018) "Exploration by Random Network Distillation."

RND uses two networks:
- target network f: randomly initialized, fixed, produces a representation
- predictor network f̂: trained to predict the target's output

The prediction error ||f(s) - f̂(s)||² serves as an intrinsic exploration
bonus: it is high for novel states (where f̂ hasn't learned to predict yet)
and low for familiar states.

In RICE, RND is used as an exploration bonus added to the task reward:
    R'(s_t, a_t) = R(s_t, a_t) + λ * ||f(s_{t+1}) - f̂(s_{t+1})||²

where λ controls the trade-off between task reward and exploration.

As state coverage increases, RND bonuses decay to zero, and a performing
policy is recovered.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


class RNDNetwork(nn.Module):
    """A feedforward network used for both target and predictor in RND."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ):
        """
        Args:
            input_dim: State dimension.
            output_dim: Dimension of the embedding output.
            hidden_sizes: Hidden layer sizes.
        """
        super().__init__()
        layers = []
        prev_size = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs
        layers.append(nn.Linear(prev_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDExploration:
    """
    Random Network Distillation for intrinsic exploration rewards.

    The target network is randomly initialized and fixed.
    The predictor network is trained to match the target's output.
    Prediction error serves as the exploration bonus.
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int = 128,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: Dimension of the state space.
            output_dim: Output dimension of the RND networks.
            hidden_sizes: Hidden layer sizes.
            lr: Learning rate for predictor optimizer.
            device: Device to run on.
        """
        self.device = device
        self.output_dim = output_dim

        # Target network: randomly initialized, fixed (no gradients)
        self.target = RNDNetwork(state_dim, output_dim, hidden_sizes).to(device)
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network: trained to match target output
        self.predictor = RNDNetwork(state_dim, output_dim, hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # Running statistics for normalization
        self.register_buffer("obs_mean", torch.zeros(state_dim, device=device))
        self.register_buffer("obs_var", torch.ones(state_dim, device=device))
        self.register_buffer("rnd_mean", torch.tensor(0.0, device=device))
        self.register_buffer("rnd_std", torch.tensor(1.0, device=device))
        self._obs_count = 0
        self._rnd_values = []
        self._initialized_norm = False

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Track buffer tensors (not nn.Module buffers but plain attributes)."""
        setattr(self, name, tensor)

    def get_bonus(self, state: np.ndarray) -> float:
        """
        Compute RND exploration bonus for a single state.
        Args:
            state: numpy array of shape (state_dim,)
        Returns:
            bonus: float, normalized prediction error
        """
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_out = self.target(s)
            pred_out = self.predictor(s)
            error = ((target_out - pred_out) ** 2).sum(dim=-1)
            raw_bonus = error.item()

            # Normalize by running statistics
            if self._initialized_norm:
                bonus = (raw_bonus - self.rnd_mean.item()) / (self.rnd_std.item() + 1e-8)
            else:
                bonus = raw_bonus

        return float(bonus)

    def get_bonus_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute RND exploration bonuses for a batch of states.
        Args:
            states: Tensor of shape (batch_size, state_dim)
        Returns:
            bonuses: Tensor of shape (batch_size,)
        """
        with torch.no_grad():
            target_out = self.target(states)
            pred_out = self.predictor(states)
            raw_bonus = ((target_out - pred_out) ** 2).sum(dim=-1)

        # Normalize
        if self._initialized_norm:
            bonus = (raw_bonus - self.rnd_mean) / (self.rnd_std + 1e-8)
        else:
            bonus = raw_bonus

        return bonus

    def update(self, states: torch.Tensor) -> float:
        """
        Train the predictor network to match the target network output.
        Args:
            states: Tensor of shape (batch_size, state_dim)
        Returns:
            mse_loss: Mean squared error loss
        """
        target_out = self.target(states)
        pred_out = self.predictor(states)
        loss = nn.functional.mse_loss(pred_out, target_out)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_norm(self, bonus_value: float) -> None:
        """
        Update running statistics for normalization of RND bonuses.
        This implements the normalization described in the RND paper.
        """
        self._rnd_values.append(bonus_value)
        if len(self._rnd_values) >= 100:
            self.rnd_mean = torch.tensor(
                float(np.mean(self._rnd_values[-1000:])), device=self.device
            )
            self.rnd_std = torch.tensor(
                float(np.std(self._rnd_values[-1000:]) + 1e-8), device=self.device
            )
            self._initialized_norm = True

    def save(self, path: str) -> None:
        """Save RND predictor weights."""
        torch.save(
            {
                "predictor": self.predictor.state_dict(),
                "target": self.target.state_dict(),
                "rnd_mean": self.rnd_mean.item(),
                "rnd_std": self.rnd_std.item(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load RND predictor weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.target.load_state_dict(checkpoint["target"])
        self.rnd_mean = torch.tensor(checkpoint["rnd_mean"], device=self.device)
        self.rnd_std = torch.tensor(checkpoint["rnd_std"], device=self.device)
        self._initialized_norm = True