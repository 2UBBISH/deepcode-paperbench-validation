"""Random Network Distillation (RND) exploration bonus for RICE refinement.

The RND bonus is defined as the squared error between a fixed, randomly
initialized target network and a trainable predictor network:

    b_RND(s) = || φ_target(s) - φ_predictor(s) ||^2

The bonus is normalized by a running estimate of its mean and standard
deviation before being scaled by λ and added to the environment reward.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class RNDNetwork(nn.Module):
    """Simple MLP feature network used for both target and predictor."""

    def __init__(
        self,
        obs_dim: int,
        output_dim: int = 128,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type = nn.Tanh,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def build_rnd_networks(
    obs_dim: int,
    output_dim: int = 128,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: type = nn.Tanh,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[RNDNetwork, RNDNetwork]:
    """Create target and predictor RND networks with identical architecture.

    The target network is initialized randomly and frozen; the predictor is
    trainable and optimized to match the target outputs.
    """
    target = RNDNetwork(obs_dim, output_dim, hidden_sizes, activation).to(device)
    predictor = RNDNetwork(obs_dim, output_dim, hidden_sizes, activation).to(device)
    for param in target.parameters():
        param.requires_grad = False
    target.eval()
    return target, predictor


class RunningMeanStd:
    """Online running mean and standard deviation estimator.

    Uses Welford's algorithm so that the normalization statistics can be
    updated incrementally as new RND bonus values are observed.
    """

    def __init__(self, shape: Tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of values."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Return (x - mean) / sqrt(var + eps)."""
        return (x - self.mean) / np.sqrt(self.var + eps)


class RNDBonus:
    """Computes and normalizes the RND exploration bonus.

    The predictor is trained to minimize the MSE between its output and the
    target network output.  The per-state bonus is the squared prediction
    error, normalized by a running mean and standard deviation.
    """

    def __init__(
        self,
        obs_dim: int,
        output_dim: int = 128,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type = nn.Tanh,
        lr: float = 1e-4,
        update_proportion: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_net, self.predictor_net = build_rnd_networks(
            obs_dim, output_dim, hidden_sizes, activation, self.device
        )
        self.optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=lr)
        self.running_stats = RunningMeanStd(shape=(1,))
        self.update_proportion = update_proportion

    def _prepare_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert observation to a float tensor on the correct device."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs

    def compute_bonus(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
    ) -> np.ndarray:
        """Return the RND bonus for the given observations."""
        obs_t = self._prepare_obs(obs)
        with torch.no_grad():
            target_features = self.target_net(obs_t)
            pred_features = self.predictor_net(obs_t)
            bonus = torch.sum((target_features - pred_features) ** 2, dim=-1)
        bonus_np = bonus.cpu().numpy()
        if normalize:
            bonus_np = self.running_stats.normalize(bonus_np)
        return bonus_np

    def update(self, obs: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """Train the predictor on a batch of observations and update stats.

        Returns a dict with the raw and normalized bonus statistics.
        """
        obs_t = self._prepare_obs(obs)
        batch_size = obs_t.shape[0]
        # Randomly mask a subset of samples if update_proportion < 1, matching
        # the original RND implementation which drops some samples to reduce
        # correlation between predictor gradients.
        if self.update_proportion < 1.0:
            mask = torch.rand(batch_size, device=self.device) < self.update_proportion
            if mask.sum() == 0:
                return {"rnd_loss": 0.0, "rnd_bonus_mean": 0.0, "rnd_bonus_std": 1.0}
            obs_t = obs_t[mask]

        with torch.no_grad():
            target_features = self.target_net(obs_t)
        pred_features = self.predictor_net(obs_t)
        loss = F.mse_loss(pred_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            bonus = torch.sum((target_features - pred_features) ** 2, dim=-1)
        bonus_np = bonus.cpu().numpy()
        self.running_stats.update(bonus_np.reshape(-1, 1))
        normalized = self.running_stats.normalize(bonus_np.reshape(-1, 1))
        return {
            "rnd_loss": loss.item(),
            "rnd_bonus_mean": float(np.mean(bonus_np)),
            "rnd_bonus_std": float(np.std(bonus_np)),
            "rnd_bonus_norm_mean": float(np.mean(normalized)),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "running_mean": self.running_stats.mean.copy(),
            "running_var": self.running_stats.var.copy(),
            "running_count": self.running_stats.count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.predictor_net.load_state_dict(state_dict["predictor"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.running_stats.mean = state_dict["running_mean"].copy()
        self.running_stats.var = state_dict["running_var"].copy()
        self.running_stats.count = state_dict["running_count"]


class RNDRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper that adds a scaled RND bonus to environment rewards.

    The wrapper also stores the raw bonus and normalized bonus in the info
    dict for logging and analysis.
    """

    def __init__(
        self,
        env: gym.Env,
        rnd_bonus: RNDBonus,
        lambda_rnd: float = 0.01,
        update_every: int = 1,
    ) -> None:
        super().__init__(env)
        self.rnd_bonus = rnd_bonus
        self.lambda_rnd = lambda_rnd
        self.update_every = update_every
        self._step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        bonus = self.rnd_bonus.compute_bonus(obs, normalize=True)
        bonus_value = float(bonus.item() if bonus.size == 1 else bonus[0])
        scaled_bonus = self.lambda_rnd * bonus_value
        new_reward = reward + scaled_bonus

        if self.update_every > 0 and self._step_count % self.update_every == 0:
            stats = self.rnd_bonus.update(obs)
            info["rnd_stats"] = stats

        info["rnd_bonus"] = bonus_value
        info["rnd_scaled_bonus"] = scaled_bonus
        info["env_reward"] = reward
        return obs, new_reward, terminated, truncated, info


def make_rnd_bonus(
    observation_space: gym.Space,
    output_dim: int = 128,
    hidden_sizes: Sequence[int] = (64, 64),
    activation: type = nn.Tanh,
    lr: float = 1e-4,
    device: Union[str, torch.device] = "cpu",
) -> RNDBonus:
    """Factory that builds an ``RNDBonus`` from a Gym/Gymnasium observation space."""
    obs_dim = int(np.prod(observation_space.shape))
    return RNDBonus(
        obs_dim=obs_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        lr=lr,
        device=device,
    )
