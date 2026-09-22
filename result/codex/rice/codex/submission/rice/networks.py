"""Neural networks used by RICE.

* :class:`ActorCritic` -- the policy/value network of the target agent and of
  the refining phase (PPO).  The paper relies on the default ``MlpPolicy`` of
  Stable-Baselines3 (two hidden layers of 64 units with ``tanh`` activations)
  for the MuJoCo tasks, on a 4x128 MLP for selfish mining and on a 3x64 MLP for
  the CAGE challenge.
* :class:`MaskNet` -- the *mask network* of Algorithm 1, a small MLP with a
  binary action head (blind / do not blind the target agent).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("unknown activation {!r}".format(name))


def mlp(
    sizes: Sequence[int],
    activation: str = "tanh",
    output_activation: Optional[Callable[[], nn.Module]] = None,
) -> nn.Sequential:
    layers: list = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(_activation(activation))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def init_orthogonal(module: nn.Module, gain: float = np.sqrt(2)) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.constant_(layer.bias, 0.0)


class ActorCritic(nn.Module):
    """Shared-trunk actor critic with a Gaussian (Box) or Categorical head."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Iterable[int] = (64, 64),
        discrete: bool = True,
        activation: str = "tanh",
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.discrete = bool(discrete)
        hidden = tuple(int(h) for h in hidden)

        self.shared = mlp((obs_dim,) + hidden, activation)
        self.pi = mlp((hidden[-1], act_dim), activation)
        init_orthogonal(self.pi, gain=0.01)
        self.vf = mlp((hidden[-1], 1), activation)
        init_orthogonal(self.vf, gain=1.0)
        if not self.discrete:
            self.log_std = nn.Parameter(
                torch.full((act_dim,), float(log_std_init), dtype=torch.float32)
            )

    # -------------------------------------------------------------- helpers
    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.vf(self.features(obs)).squeeze(-1)

    def dist(self, obs: torch.Tensor):
        h = self.features(obs)
        pi = self.pi(h)
        if self.discrete:
            return Categorical(logits=pi)
        mean = pi
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor):
        return self.dist(obs), self.value(obs)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        dist, value = self.forward(obs_t)
        if deterministic:
            if self.discrete:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.discrete:
            action_np = action.squeeze(-1).cpu().numpy().astype(np.int64)
        else:
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        return action_np, float(log_prob.sum().item()), float(value.squeeze(-1).item())

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions)
        if self.discrete:
            log_prob = log_prob.reshape(-1)
        else:
            log_prob = log_prob.sum(-1)
        entropy = dist.entropy()
        if not self.discrete:
            entropy = entropy.sum(-1)
        return log_prob, entropy, value

    @staticmethod
    def from_checkpoint(path: str, map_location: str = "cpu") -> "ActorCritic":
        ckpt = torch.load(path, map_location=map_location)
        model = ActorCritic(
            ckpt["obs_dim"],
            ckpt["act_dim"],
            hidden=ckpt.get("hidden", (64, 64)),
            discrete=ckpt.get("discrete", True),
            activation=ckpt.get("activation", "tanh"),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden": tuple(
                    layer.out_features
                    for layer in self.shared
                    if isinstance(layer, nn.Linear)
                ),
                "discrete": self.discrete,
            },
            path,
        )


class MaskNet(nn.Module):
    """Binary mask network: ``a^m = 1`` blinds the target agent at this step."""

    MASK_KEEP = 0  # a^m = 0 -> the target agent uses its own action
    MASK_BLIND = 1  # a^m = 1 -> a uniformly random action is executed

    def __init__(
        self,
        obs_dim: int,
        hidden: Iterable[int] = (64, 64),
        activation: str = "tanh",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        hidden = tuple(int(h) for h in hidden)
        self.net = mlp((obs_dim,) + hidden + (2,), activation)
        init_orthogonal(self.net, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    @torch.no_grad()
    def mask_prob(self, obs) -> np.ndarray:
        """Probability that the mask *blinds* the agent at ``obs``."""
        obs_t = torch.as_tensor(np.atleast_2d(np.asarray(obs, dtype=np.float32)))
        probs = self.dist(obs_t).probs[..., self.MASK_BLIND]
        return probs.cpu().numpy()

    @torch.no_grad()
    def importance(self, obs) -> np.ndarray:
        """Importance score of Algorithm 1: ``P(a^m = 0 | s)``."""
        obs_t = torch.as_tensor(np.atleast_2d(np.asarray(obs, dtype=np.float32)))
        probs = self.dist(obs_t).probs[..., self.MASK_KEEP]
        return probs.cpu().numpy()

    @staticmethod
    def from_checkpoint(path: str, map_location: str = "cpu") -> "MaskNet":
        ckpt = torch.load(path, map_location=map_location)
        model = MaskNet(ckpt["obs_dim"], hidden=ckpt.get("hidden", (64, 64)))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "hidden": tuple(
                    layer.out_features
                    for layer in self.net
                    if isinstance(layer, nn.Linear)
                )[:-1],
            },
            path,
        )
