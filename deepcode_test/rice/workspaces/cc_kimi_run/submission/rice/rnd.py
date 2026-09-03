"""Random Network Distillation (RND) exploration bonus."""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rice.utils import get_device


class RNDNetwork(nn.Module):
    """Small fully-connected network used for both target and predictor."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), nn.ReLU()])
            prev_size = h
        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class RNDBonus:
    """Compute normalized RND intrinsic rewards.

    The bonus is ||f(s) - f_hat(s)||^2 where f is a fixed random target network
    and f_hat is a predictor network trained via gradient descent.
    """

    def __init__(
        self,
        obs_dim: int,
        lr: float = 1e-4,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        output_dim: int = 64,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
        norm_eps: float = 1e-8,
    ) -> None:
        self.device = device or get_device()
        self.gamma = gamma
        self.norm_eps = norm_eps
        self.target = RNDNetwork(obs_dim, hidden_sizes, output_dim).to(self.device)
        self.predictor = RNDNetwork(obs_dim, hidden_sizes, output_dim).to(self.device)
        # Target network is fixed.
        for param in self.target.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = norm_eps

    def _normalize(self, bonus: np.ndarray) -> np.ndarray:
        """Normalize bonuses using running mean and variance."""
        # Update running statistics.
        batch_mean = float(np.mean(bonus))
        batch_var = float(np.var(bonus))
        batch_count = bonus.size
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        self.running_mean += delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.running_var = m2 / total_count
        self.count = total_count
        return bonus / np.sqrt(self.running_var + self.norm_eps)

    def compute_bonus(self, obs: np.ndarray) -> np.ndarray:
        """Compute raw RND bonus for observations."""
        self.predictor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            target_features = self.target(obs_t)
            pred_features = self.predictor(obs_t)
            bonus = torch.sum((target_features - pred_features) ** 2, dim=-1).cpu().numpy()
        return bonus

    def update(self, obs: np.ndarray, n_epochs: int = 4, batch_size: int = 256) -> float:
        """Update the predictor network to regress to the target network."""
        self.predictor.train()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(obs_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        n_batches = 0
        for _ in range(n_epochs):
            for (batch_obs,) in loader:
                with torch.no_grad():
                    target = self.target(batch_obs)
                pred = self.predictor(batch_obs)
                loss = nn.functional.mse_loss(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def compute_and_update(
        self,
        obs: np.ndarray,
        update: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Compute normalized bonuses and optionally update the predictor."""
        bonus = self.compute_bonus(obs)
        norm_bonus = self._normalize(bonus)
        loss = 0.0
        if update:
            loss = self.update(obs)
        return norm_bonus, loss
