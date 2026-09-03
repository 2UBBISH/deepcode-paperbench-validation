"""Mask network architecture for RICE.

The mask network ξ(s) outputs the probability that a decision step is critical.
It is trained to approximate Q_diff(s,a) = Q^π(s,a) - E_{a'}[Q^π(s,a')].
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskNetwork(nn.Module):
    """MLP mask network that outputs a single critical-step probability.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the flattened observation vector.
    hidden_sizes : Sequence[int]
        Hidden layer widths. By default matches the SB3 MLP default (64, 64).
    activation : Type[nn.Module]
        Activation function class applied after each hidden layer.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: type = nn.Tanh,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)

        layers: List[nn.Module] = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ξ(s) ∈ (0, 1)."""
        logits = self.net(obs)
        return torch.sigmoid(logits)

    def predict(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Run inference and return critical probabilities.

        Parameters
        ----------
        obs : np.ndarray or torch.Tensor
            Observation(s). If NumPy, converted to a float tensor.
        deterministic : bool
            Ignored for the mask network (kept for API consistency), since the
            output is already a deterministic probability.
        return_tensor : bool
            If True, return a torch.Tensor; otherwise return a NumPy array.

        Returns
        -------
        xi : np.ndarray or torch.Tensor
            Critical probabilities with shape ``(batch_size, 1)`` or ``(1,)``
            for a single observation.
        """
        del deterministic  # not stochastic at inference
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            xi = self.forward(obs)
        if return_tensor:
            return xi
        return xi.cpu().numpy()

    def get_mask_output(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias for ``forward`` used by trainers."""
        return self.forward(obs)


def build_mask_network(
    observation_space,
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: type = nn.Tanh,
) -> MaskNetwork:
    """Factory that builds a MaskNetwork from a Gym/Gymnasium observation space.

    Parameters
    ----------
    observation_space : gym.Space or gymnasium.Space
        Expected to be a ``Box`` with a 1-D shape.
    hidden_sizes : Sequence[int], optional
        Hidden layer widths. Defaults to ``(64, 64)``.
    activation : Type[nn.Module]
        Activation class. Defaults to ``nn.Tanh``.

    Returns
    -------
    MaskNetwork
        Instantiated mask network.
    """
    if hidden_sizes is None:
        hidden_sizes = (64, 64)

    obs_shape = observation_space.shape
    if len(obs_shape) != 1:
        raise ValueError(
            f"MaskNetwork expects a 1-D observation vector, got shape {obs_shape}"
        )
    obs_dim = int(obs_shape[0])
    return MaskNetwork(
        obs_dim=obs_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )


def match_target_mask_network(
    target_policy,
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: type = nn.Tanh,
) -> MaskNetwork:
    """Build a MaskNetwork whose architecture matches a target policy.

    If ``target_policy`` exposes ``observation_space`` and the hidden sizes can
    be inferred, they are used; otherwise the provided ``hidden_sizes`` (or the
    default) are used.

    Parameters
    ----------
    target_policy : rice.agents.BaseTargetPolicy or nn.Module
        The frozen target policy. Must expose ``observation_space``.
    hidden_sizes : Sequence[int], optional
        Hidden sizes to use if they cannot be inferred from the policy.
    activation : Type[nn.Module]
        Activation class.

    Returns
    -------
    MaskNetwork
        Mask network with the same MLP architecture as the target policy.
    """
    observation_space = getattr(target_policy, "observation_space", None)
    if observation_space is None:
        raise ValueError("target_policy must expose an observation_space attribute")

    # Try to infer hidden sizes from a PyTorch actor-critic backbone.
    if hidden_sizes is None:
        hidden_sizes = _infer_hidden_sizes(target_policy)

    return build_mask_network(observation_space, hidden_sizes=hidden_sizes, activation=activation)


def _infer_hidden_sizes(target_policy) -> Tuple[int, ...]:
    """Best-effort inference of hidden layer widths from a policy network."""
    model = getattr(target_policy, "model", None)
    if model is None and isinstance(target_policy, nn.Module):
        model = target_policy

    if model is None or not isinstance(model, nn.Module):
        return (64, 64)

    # Look for a Sequential backbone of Linear -> Activation layers.
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            sizes = []
            for layer in module:
                if isinstance(layer, nn.Linear):
                    out_features = layer.out_features
                    # Skip the final output layer (usually action_dim or 1).
                    sizes.append(out_features)
            if len(sizes) >= 2:
                # The last element is the output layer; drop it.
                return tuple(sizes[:-1])

    return (64, 64)
