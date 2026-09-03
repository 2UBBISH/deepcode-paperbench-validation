"""Random Network Distillation (RND) exploration bonus for RICE refining.

The RND module maintains a fixed random target network :math:`\\phi_{target}`
and a trainable predictor network :math:`\\phi_{pred}`.  For a given state
:math:`s`, the intrinsic bonus is the squared Euclidean distance between the
two network outputs:

.. math::
    b_{RND}(s) = \\|\\phi_{target}(s) - \\phi_{pred}(s)\\|_2^2

The bonus is normalised using running mean and variance statistics before being
scaled by a domain-specific coefficient :math:`\\lambda`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor


class RNDNetwork(nn.Module):
    """Small MLP used as either the RND target or predictor network.

    Parameters
    ----------
    observation_space : spaces.Space
        Observation space of the environment.
    output_dim : int
        Dimensionality of the RND embedding.
    hidden_sizes : Tuple[int, ...]
        Hidden layer sizes.
    activation : Type[nn.Module]
        Activation function class.
    normalize_inputs : bool
        If ``True``, apply running mean/variance normalization to inputs.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        output_dim: int = 64,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: type = nn.ReLU,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.normalize_inputs = normalize_inputs

        obs_dim = get_flattened_obs_dim(observation_space)
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

        # Running input normalisation statistics.
        self.register_buffer("obs_mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("obs_var", torch.ones(obs_dim, dtype=torch.float32))
        self.register_buffer("count", torch.zeros(1, dtype=torch.float32))

    def update_obs_stats(self, obs: torch.Tensor) -> None:
        """Update running mean/variance for input observations (Welford)."""
        if not self.normalize_inputs:
            return
        with torch.no_grad():
            flat = obs.reshape(-1, obs.shape[-1])
            batch_mean = flat.mean(dim=0)
            batch_var = flat.var(dim=0, unbiased=False)
            batch_count = flat.shape[0]

            delta = batch_mean - self.obs_mean
            total_count = self.count + batch_count

            self.obs_mean += delta * batch_count / total_count
            m_a = self.obs_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
            self.obs_var = M2 / total_count
            self.count = total_count

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise observations using running statistics."""
        if not self.normalize_inputs:
            return obs
        eps = 1e-8
        return (obs - self.obs_mean) / torch.sqrt(self.obs_var + eps)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the RND embedding for ``obs``."""
        flat = obs.reshape(-1, self.obs_mean.shape[0])
        if self.training or self.normalize_inputs:
            self.update_obs_stats(flat)
        norm = self.normalize_obs(flat)
        return self.network(norm)


class RNDModule(nn.Module):
    """RND target/predictor pair with bonus computation and normalisation.

    Parameters
    ----------
    observation_space : spaces.Space
        Observation space.
    output_dim : int
        Embedding dimension.
    hidden_sizes : Tuple[int, ...]
        Hidden layer sizes for both networks.
    activation : Type[nn.Module]
        Activation class.
    normalize_inputs : bool
        Whether to normalise predictor inputs.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        output_dim: int = 64,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: type = nn.ReLU,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.output_dim = output_dim

        self.target_net = RNDNetwork(
            observation_space,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            normalize_inputs=False,
        )
        self.predictor_net = RNDNetwork(
            observation_space,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            normalize_inputs=normalize_inputs,
        )

        # Freeze target network.
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Running bonus normalisation statistics.
        self.register_buffer("bonus_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("bonus_var", torch.ones(1, dtype=torch.float32))
        self.register_buffer("bonus_count", torch.zeros(1, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return target and predictor embeddings."""
        with torch.no_grad():
            target = self.target_net(obs)
        pred = self.predictor_net(obs)
        return target, pred

    def compute_bonus(self, obs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute the RND bonus for ``obs``.

        Parameters
        ----------
        obs : torch.Tensor
            Observations.
        normalize : bool
            If ``True``, normalise the bonus using running statistics.

        Returns
        -------
        torch.Tensor
            Intrinsic bonus of shape ``(batch_size,)``.
        """
        target, pred = self.forward(obs)
        bonus = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        if normalize:
            bonus = self.normalize_bonus(bonus)
        return bonus

    def update_bonus_stats(self, bonus: torch.Tensor) -> None:
        """Update running bonus statistics (Welford)."""
        with torch.no_grad():
            flat = bonus.reshape(-1)
            batch_mean = flat.mean()
            batch_var = flat.var(unbiased=False)
            batch_count = flat.shape[0]

            delta = batch_mean - self.bonus_mean
            total_count = self.bonus_count + batch_count
            self.bonus_mean += delta * batch_count / total_count
            m_a = self.bonus_var * self.bonus_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta * delta * self.bonus_count * batch_count / total_count
            self.bonus_var = M2 / total_count
            self.bonus_count = total_count

    def normalize_bonus(self, bonus: torch.Tensor) -> torch.Tensor:
        """Normalise bonus with running mean/variance."""
        eps = 1e-8
        return bonus / torch.sqrt(self.bonus_var + eps)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the predictor training loss for ``obs``."""
        target, pred = self.forward(obs)
        return F.mse_loss(pred, target)

    def save(self, path: str) -> None:
        """Save RND module state."""
        torch.save({"state_dict": self.state_dict(), "config": self.config_dict()}, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """Load RND module state."""
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["state_dict"])

    def config_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable configuration dict."""
        return {
            "output_dim": self.output_dim,
            "hidden_sizes": self.hidden_sizes,
            "normalize_inputs": self.predictor_net.normalize_inputs,
        }


class RNDRewardWrapper:
    """Mixin-style helper to augment environment rewards with an RND bonus.

    This class is intentionally lightweight: it holds an ``RNDModule`` and
    provides a single method ``augment_reward`` that can be called from a
    Gymnasium wrapper or from a training loop.
    """

    def __init__(
        self,
        rnd_module: RNDModule,
        lambda_coef: float = 0.01,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.rnd_module = rnd_module
        self.lambda_coef = lambda_coef
        self.device = device

    def bonus(self, obs: np.ndarray) -> np.ndarray:
        """Compute the normalised RND bonus for a numpy observation."""
        obs_t = obs_as_tensor(obs, self.device)
        with torch.no_grad():
            bonus = self.rnd_module.compute_bonus(obs_t, normalize=True)
        return bonus.cpu().numpy()

    def augment_reward(self, obs: np.ndarray, reward: float) -> float:
        """Return ``reward + lambda * b_RND(obs)``."""
        bonus = self.bonus(obs)
        if bonus.shape:
            bonus = float(bonus.item())
        else:
            bonus = float(bonus)
        return reward + self.lambda_coef * bonus


def make_rnd_module(
    observation_space: spaces.Space,
    output_dim: int = 64,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation: type = nn.ReLU,
    normalize_inputs: bool = True,
    device: Union[str, torch.device] = "auto",
) -> RNDModule:
    """Factory for an ``RNDModule``."""
    module = RNDModule(
        observation_space=observation_space,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        normalize_inputs=normalize_inputs,
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return module.to(device)


def default_rnd_config(domain: str = "mujoco") -> Dict[str, Any]:
    """Return default RND hyper-parameters for a domain.

    The paper uses small MLPs for the RND target/predictor networks and a
    domain-specific exploration coefficient :math:`\\lambda` (Table 3).
    """
    base = {
        "output_dim": 64,
        "hidden_sizes": (64, 64),
        "activation": "ReLU",
        "normalize_inputs": True,
    }
    lambdas = {
        "mujoco": 0.01,
        "hopper": 0.01,
        "walker2d": 0.01,
        "reacher": 0.01,
        "halfcheetah": 0.01,
        "selfish_mining": 0.001,
        "cage": 0.001,
        "metadrive": 0.01,
        "malware": 0.001,
    }
    key = domain.lower()
    base["lambda_coef"] = lambdas.get(key, 0.01)
    return base
