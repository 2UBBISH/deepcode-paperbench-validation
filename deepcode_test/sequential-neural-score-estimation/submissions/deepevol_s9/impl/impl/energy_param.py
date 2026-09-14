'''Energy-based score-network parameterization.'''

from __future__ import annotations

import torch
from torch import Tensor, nn

from impl.score_network import sinusoidal_time_embedding


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        blocks = []
        in_features = int(input_dim)
        for _ in range(max(1, int(layers)) - 1):
            blocks.append(nn.Linear(in_features, hidden))
            blocks.append(nn.SiLU())
            in_features = hidden
        blocks.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EnergyScoreNetwork(nn.Module):
    '''Energy network whose negative gradient is the score.

    The network keeps the same three-MLP embedding architecture as
    :class:`impl.score_network.ScoreNetworkMLP`, but the final MLP outputs a single
    scalar energy. This allows ``s_psi = -grad_theta E_psi``.
    '''

    def __init__(self, theta_dim: int, x_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        if theta_dim <= 0 or x_dim <= 0:
            raise ValueError('theta_dim and x_dim must be positive')

        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.time_dim = 64

        theta_out = max(30, 4 * self.theta_dim)
        x_out = max(30, 4 * self.x_dim)

        self.theta_embed = _MLP(self.theta_dim, theta_out, int(hidden), int(layers))
        self.x_embed = _MLP(self.x_dim, x_out, int(hidden), int(layers))
        self.final = _MLP(
            theta_out + x_out + self.time_dim,
            1,
            int(hidden),
            int(layers),
        )

    def forward(self, theta_t: Tensor, x: Tensor, t) -> Tensor:
        theta_embedding = self.theta_embed(theta_t)
        x_embedding = self.x_embed(x)
        time_embedding = sinusoidal_time_embedding(t, self.time_dim)

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

        concat = torch.cat([theta_embedding, x_embedding, time_embedding], dim=-1)
        return self.final(concat).squeeze(-1)

    def score(self, theta_t: Tensor, x: Tensor, t) -> Tensor:
        '''Return the score vector ``-grad_theta energy``.'''

        with torch.enable_grad():
            theta_in = theta_t.detach().to(dtype=torch.float32).requires_grad_(True)
            energy = self.forward(theta_in, x, t)
            grad = torch.autograd.grad(energy.sum(), theta_in, create_graph=True)[0]
        return -grad

    def unnormalized_log_posterior(self, theta: Tensor, observation: Tensor) -> Tensor:
        '''Return ``-energy(theta, observation, t=0)`` as an approximate unnormalized
        posterior log density.
        '''

        return -self.forward(theta, observation, 0.0)
