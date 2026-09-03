"""Frozen target-policy wrappers used by RICE.

A *target policy* is the pre-trained agent whose decisions we want to explain
and refine.  This module provides a small common interface so that downstream
components (mask training, refinement, evaluation) do not need to know whether
the underlying agent was trained with Stable-Baselines3, Tianshou, DI-drive, or
a custom PyTorch PPO loop.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _as_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a numpy array to a tensor on ``device`` without copying if possible."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class BaseTargetPolicy(abc.ABC):
    """Abstract interface for a frozen target policy."""

    def __init__(self, observation_space: Any, action_space: Any, device: Union[str, torch.device] = "auto"):
        self.observation_space = observation_space
        self.action_space = action_space
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @abc.abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Return an action (and optional info dict) for a single observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (value, log_prob, entropy) for the given observations/actions."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the estimated value V(s)."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist the policy to disk."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "BaseTargetPolicy":
        """Load a persisted policy from disk."""
        raise NotImplementedError


class MLPActorCritic(nn.Module):
    """Simple MLP actor-critic compatible with discrete and continuous actions.

    This is the default network used for non-SB3 target agents (e.g. selfish
    mining, CAGE, malware).  It follows the architecture described in the paper:
    a configurable multi-layer perceptron for both actor and critic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: type = nn.Tanh,
        discrete: bool = True,
        share_backbone: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.share_backbone = share_backbone

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), activation()])
            prev = h
        self.backbone = nn.Sequential(*layers)

        if share_backbone:
            self.actor_head = nn.Linear(prev, action_dim)
            self.critic_head = nn.Linear(prev, 1)
        else:
            actor_layers: list[nn.Module] = []
            critic_layers: list[nn.Module] = []
            for h in hidden_sizes:
                actor_layers.extend([nn.Linear(prev, h), activation()])
                critic_layers.extend([nn.Linear(prev, h), activation()])
                prev = h
            self.actor = nn.Sequential(*actor_layers, nn.Linear(prev, action_dim))
            self.critic = nn.Sequential(*critic_layers, nn.Linear(prev, 1))

        if not discrete:
            # Learned state-independent log standard deviations.
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def _actor_logits(self, obs: torch.Tensor) -> torch.Tensor:
        if self.share_backbone:
            return self.actor_head(self._features(obs))
        return self.actor(obs)

    def _value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.share_backbone:
            return self.critic_head(self._features(obs))
        return self.critic(obs)

    def forward(self, obs: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """Return action distribution and value estimate."""
        if self.share_backbone:
            features = self._features(obs)
            logits_or_mean = self.actor_head(features)
            value = self.critic_head(features)
        else:
            logits_or_mean = self._actor_logits(obs)
            value = self._value(obs)

        if self.discrete:
            dist = Categorical(logits=logits_or_mean)
        else:
            dist = Normal(logits_or_mean, self.action_log_std.exp())
        return dist, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.share_backbone:
            return self.critic_head(self._features(obs)).squeeze(-1)
        return self._value(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1) if not self.discrete else dist.log_prob(action)
        entropy = dist.entropy().sum(-1) if not self.discrete else dist.entropy()
        return action, log_prob, entropy, value


class TorchTargetPolicy(BaseTargetPolicy):
    """Frozen wrapper around a PyTorch :class:`MLPActorCritic` module."""

    def __init__(
        self,
        model: MLPActorCritic,
        observation_space: Any,
        action_space: Any,
        device: Union[str, torch.device] = "auto",
    ):
        super().__init__(observation_space, action_space, device)
        self.model = model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_t = _as_tensor(observation, self.device).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.model.forward(obs_t)
            if deterministic:
                if self.model.discrete:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.mean
            else:
                action = dist.sample()
            action = action.squeeze(0).cpu().numpy()
        return action, {}

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        with torch.no_grad():
            dist, value = self.model.forward(observations)
            if self.model.discrete:
                log_prob = dist.log_prob(actions)
                entropy = dist.entropy()
            else:
                log_prob = dist.log_prob(actions).sum(-1)
                entropy = dist.entropy().sum(-1)
        return value, log_prob, entropy

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device)
        with torch.no_grad():
            return self.model.get_value(observations)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "obs_dim": self.model.obs_dim,
                "action_dim": self.model.action_dim,
                "hidden_sizes": tuple(self.model.backbone[0].out_features for _ in range(len(self.model.backbone) // 2)),
                "discrete": self.model.discrete,
                "share_backbone": self.model.share_backbone,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "TorchTargetPolicy":
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")
        model = MLPActorCritic(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_sizes=checkpoint.get("hidden_sizes", (64, 64)),
            discrete=checkpoint.get("discrete", True),
            share_backbone=checkpoint.get("share_backbone", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, observation_space=None, action_space=None, device=kwargs.get("device", "auto"))


class SB3TargetPolicy(BaseTargetPolicy):
    """Frozen wrapper around a Stable-Baselines3 PPO model."""

    def __init__(
        self,
        model: "stable_baselines3.PPO",  # type: ignore[name-defined]
        device: Union[str, torch.device] = "auto",
    ):
        # Import here so that the module can be imported even when SB3 is not installed.
        from stable_baselines3 import PPO

        if not isinstance(model, PPO):
            raise TypeError("SB3TargetPolicy only supports stable_baselines3.PPO models")
        super().__init__(model.observation_space, model.action_space, device)
        self.sb3_model = model
        self.sb3_model.set_device(self.device)
        self.sb3_model.policy.eval()
        for p in self.sb3_model.policy.parameters():
            p.requires_grad = False

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        action, _ = self.sb3_model.predict(observation, deterministic=deterministic)
        return np.asarray(action), None

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        with torch.no_grad():
            values, log_prob, entropy = self.sb3_model.policy.evaluate_actions(observations, actions)
        return values, log_prob, entropy

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device)
        with torch.no_grad():
            return self.sb3_model.policy.predict_values(observations)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sb3_model.save(path)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "SB3TargetPolicy":
        from stable_baselines3 import PPO

        model = PPO.load(path, device=kwargs.get("device", "auto"))
        return cls(model, device=kwargs.get("device", "auto"))


def load_target_policy(
    path: Union[str, Path],
    backend: Optional[str] = None,
    **kwargs: Any,
) -> BaseTargetPolicy:
    """Load a target policy, auto-detecting the backend if not specified.

    Parameters
    ----------
    path:
        Path to the saved policy.  SB3 policies are saved as ``.zip`` files,
        PyTorch policies as ``.pt`` / ``.pth`` files.
    backend:
        Either ``"sb3"`` or ``"torch"``.  If ``None``, inferred from the file
        extension.
    """
    path = Path(path)
    if backend is None:
        if path.suffix == ".zip":
            backend = "sb3"
        else:
            backend = "torch"
    if backend == "sb3":
        return SB3TargetPolicy.load(path, **kwargs)
    if backend == "torch":
        return TorchTargetPolicy.load(path, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
