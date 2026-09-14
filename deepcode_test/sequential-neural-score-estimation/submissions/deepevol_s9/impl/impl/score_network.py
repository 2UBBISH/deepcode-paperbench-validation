"""Conditional score network architecture.

The paper uses three fully-connected MLPs with 256 hidden units and SiLU
activations: one for the noisy parameter vector, one for the observation
vector, and one final MLP acting on the concatenation of those embeddings and a
64-dimensional sinusoidal time embedding. This module implements that
architecture directly without a normalisation-flow density model.
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor

from impl.config import TIME_EMBEDDING_DIM


def sinusoidal_time_embedding(t, dim: int = TIME_EMBEDDING_DIM) -> Tensor:
    """Return a sinusoidal time embedding of dimension ``dim``.

    The function accepts a Python float, a 0-D tensor, a 1-D batch tensor, or a
    2-D tensor whose final dimension is 1. Scalars return a 1-D embedding so
    single-sample network calls concatenate correctly. The raw time is scaled by
    1000 before generating frequencies, a common diffusion-model convention that
    makes the otherwise unit-interval time values visible to the periodic
    features.
    """

    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    else:
        t = t.float()

    scalar_input = t.ndim == 0
    if t.ndim == 2 and t.size(1) == 1:
        t = t.squeeze(-1)

    t = t * 1000.0
    half_dim = dim // 2
    frequencies = torch.arange(half_dim, dtype=torch.float32, device=t.device)
    frequencies = 1.0 / (10000.0 ** (frequencies / half_dim))

    args = t.unsqueeze(-1) * frequencies
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if scalar_input:
        embedding = embedding.squeeze(0)
    return embedding


class _MLP(nn.Module):
    """Small fully-connected MLP with SiLU activations between layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        blocks: list[nn.Module] = []
        in_features = input_dim
        for _ in range(layers - 1):
            blocks.append(nn.Linear(in_features, hidden))
            blocks.append(nn.SiLU())
            in_features = hidden
        blocks.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ScoreNetworkMLP(nn.Module):
    """Conditional score network returning a score vector of dimension ``theta_dim``.

    Parameters
    ----------
    theta_dim:
        Dimensionality of the parameters of interest.
    x_dim:
        Dimensionality of the observation summary statistics.
    hidden:
        Width of each hidden layer in the three MLPs.
    layers:
        Number of linear layers in each MLP. With the default of 3, each MLP
        has two hidden layers and one output layer.
    """

    def __init__(self, theta_dim: int, x_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        if theta_dim <= 0 or x_dim <= 0:
            raise ValueError("theta_dim and x_dim must be positive")

        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.time_dim = TIME_EMBEDDING_DIM

        theta_out = max(30, 4 * self.theta_dim)
        x_out = max(30, 4 * self.x_dim)

        self.theta_embed = _MLP(self.theta_dim, theta_out, hidden, layers)
        self.x_embed = _MLP(self.x_dim, x_out, hidden, layers)
        self.final = _MLP(theta_out + x_out + self.time_dim, self.theta_dim, hidden, layers)

    def forward(self, theta_t: Tensor, x: Tensor, t) -> Tensor:
        """Return the score ``s_psi(theta_t, x, t)``."""

        theta_embedding = self.theta_embed(theta_t)
        x_embedding = self.x_embed(x)
        time_embedding = sinusoidal_time_embedding(t, self.time_dim)

        # Match batch dimensions for single-sample and batch calls. The usual
        # path has batch tensors of shape (N, D) and time of shape (N,) or
        # (N, 1). The branch below also supports querying a single posterior
        # point where theta and x are 1-D and t is scalar.
        if theta_embedding.ndim == 2:
            if x_embedding.ndim == 1:
                x_embedding = x_embedding.unsqueeze(0)
            if time_embedding.ndim == 1:
                time_embedding = time_embedding.unsqueeze(0)
        elif theta_embedding.ndim == 1:
            if x_embedding.ndim == 2 and x_embedding.size(0) == 1:
                x_embedding = x_embedding.squeeze(0)
            if time_embedding.ndim == 2 and time_embedding.size(0) == 1:
                time_embedding = time_embedding.squeeze(0)

        concatenated = torch.cat([theta_embedding, x_embedding, time_embedding], dim=-1)
        return self.final(concatenated)
