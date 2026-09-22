"""Network builders for SAPG.

Implements MLP / LSTM backbones with ELU activations, a Gaussian policy head
with a learnable (input-independent) sigma vector, and value heads.

Reference: SAPG paper, Sec 4.4 and Addendum (architecture details).
"""

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ELU, output_activation=nn.Identity, use_layernorm=False):
    """Build a multi-layer perceptron.

    Args:
        sizes: list of layer sizes [in, h1, h2, ..., out]
        activation: activation module class for hidden layers
        output_activation: activation module class for the output layer
        use_layernorm: whether to apply LayerNorm after each hidden layer
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if use_layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class GaussianHead(nn.Module):
    """Gaussian policy head with a learnable, input-independent sigma vector.

    The mean is produced by the backbone; sigma is a free parameter (log-std),
    shared across the batch but independent of the observation (as in the paper).
    """

    def __init__(self, latent_dim, action_dim, init_log_std=-1.0):
        super().__init__()
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, mean):
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def distribution(self, mean):
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class MLPGaussianActor(nn.Module):
    """MLP actor producing a Gaussian mean, with learnable sigma."""

    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 256, 128),
                 phi_dim=0, activation=nn.ELU, init_log_std=-1.0):
        super().__init__()
        self.phi_dim = phi_dim
        in_dim = obs_dim + phi_dim
        self.backbone = mlp([in_dim] + list(hidden_sizes), activation=activation)
        self.mu_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.head = GaussianHead(hidden_sizes[-1], action_dim, init_log_std)

    def forward(self, obs, phi=None):
        x = obs
        if self.phi_dim > 0:
            assert phi is not None, "phi latent required for conditioning"
            x = torch.cat([obs, phi], dim=-1)
        h = self.backbone(x)
        mean = self.mu_head(h)
        return mean

    def distribution(self, obs, phi=None):
        mean = self.forward(obs, phi)
        return self.head.distribution(mean)

    def log_prob(self, obs, actions, phi=None):
        dist = self.distribution(obs, phi)
        return dist.log_prob(actions).sum(dim=-1)

    def entropy(self, obs, phi=None):
        dist = self.distribution(obs, phi)
        return dist.entropy().sum(dim=-1)


class LSTMPolicy(nn.Module):
    """MLP -> LSTM -> Gaussian mean actor (used for AllegroKuka).

    The LSTM operates over a sequence of length 1 per step (single-layer).
    """

    def __init__(self, obs_dim, action_dim, mlp_sizes=(768, 512, 256),
                 lstm_hidden=768, phi_dim=0, activation=nn.ELU, init_log_std=-1.0):
        super().__init__()
        self.phi_dim = phi_dim
        self.lstm_hidden = lstm_hidden
        in_dim = obs_dim + phi_dim
        self.mlp = mlp([in_dim] + list(mlp_sizes), activation=activation)
        self.lstm = nn.LSTM(mlp_sizes[-1], lstm_hidden, num_layers=1)
        self.mu_head = nn.Linear(lstm_hidden, action_dim)
        self.head = GaussianHead(lstm_hidden, action_dim, init_log_std)

    def forward(self, obs, phi=None, lstm_state=None):
        """Returns (mean, new_lstm_state).

        obs: (B, obs_dim). lstm_state: (h, c) each (1, B, lstm_hidden) or None.
        """
        x = obs
        if self.phi_dim > 0:
            assert phi is not None, "phi latent required for conditioning"
            x = torch.cat([obs, phi], dim=-1)
        h = self.mlp(x)
        h = h.unsqueeze(0)  # (1, B, feat)
        out, new_state = self.lstm(h, lstm_state)
        out = out.squeeze(0)
        mean = self.mu_head(out)
        return mean, new_state

    def distribution(self, obs, phi=None, lstm_state=None):
        mean, new_state = self.forward(obs, phi, lstm_state)
        return self.head.distribution(mean), new_state

    def log_prob(self, obs, actions, phi=None, lstm_state=None):
        dist, new_state = self.distribution(obs, phi, lstm_state)
        return dist.log_prob(actions).sum(dim=-1), new_state

    def entropy(self, obs, phi=None, lstm_state=None):
        dist, new_state = self.distribution(obs, phi, lstm_state)
        return dist.entropy().sum(dim=-1), new_state


class Critic(nn.Module):
    """Value network mirroring the actor backbone, conditioned on phi."""

    def __init__(self, obs_dim, hidden_sizes=(512, 256, 128), phi_dim=0,
                 activation=nn.ELU):
        super().__init__()
        self.phi_dim = phi_dim
        in_dim = obs_dim + phi_dim
        self.net = mlp([in_dim] + list(hidden_sizes) + [1], activation=activation)

    def forward(self, obs, phi=None):
        x = obs
        if self.phi_dim > 0:
            assert phi is not None, "phi latent required for conditioning"
            x = torch.cat([obs, phi], dim=-1)
        return self.net(x).squeeze(-1)


class LSTMCritic(nn.Module):
    """LSTM value network mirroring the AllegroKuka actor backbone."""

    def __init__(self, obs_dim, mlp_sizes=(768, 512, 256), lstm_hidden=768,
                 phi_dim=0, activation=nn.ELU):
        super().__init__()
        self.phi_dim = phi_dim
        self.lstm_hidden = lstm_hidden
        in_dim = obs_dim + phi_dim
        self.mlp = mlp([in_dim] + list(mlp_sizes), activation=activation)
        self.lstm = nn.LSTM(mlp_sizes[-1], lstm_hidden, num_layers=1)
        self.v_head = nn.Linear(lstm_hidden, 1)

    def forward(self, obs, phi=None, lstm_state=None):
        x = obs
        if self.phi_dim > 0:
            assert phi is not None, "phi latent required for conditioning"
            x = torch.cat([obs, phi], dim=-1)
        h = self.mlp(x).unsqueeze(0)
        out, new_state = self.lstm(h, lstm_state)
        v = self.v_head(out.squeeze(0)).squeeze(-1)
        return v, new_state


def build_actor(task, obs_dim, action_dim, phi_dim):
    """Factory for actor networks per task (paper Sec 4.4 / Addendum)."""
    if task == "allegrokuka":
        return LSTMPolicy(obs_dim, action_dim, mlp_sizes=(768, 512, 256),
                          lstm_hidden=768, phi_dim=phi_dim)
    elif task == "shadowhand":
        return MLPGaussianActor(obs_dim, action_dim, hidden_sizes=(512, 512, 256, 128),
                                phi_dim=phi_dim)
    elif task == "allegrohand":
        return MLPGaussianActor(obs_dim, action_dim, hidden_sizes=(512, 256, 128),
                                phi_dim=phi_dim)
    else:
        raise ValueError(f"Unknown task: {task}")


def build_critic(task, obs_dim, phi_dim):
    """Factory for critic networks per task (mirrors actor backbone)."""
    if task == "allegrokuka":
        return LSTMCritic(obs_dim, mlp_sizes=(768, 512, 256), lstm_hidden=768,
                          phi_dim=phi_dim)
    elif task == "shadowhand":
        return Critic(obs_dim, hidden_sizes=(512, 512, 256, 128), phi_dim=phi_dim)
    elif task == "allegrohand":
        return Critic(obs_dim, hidden_sizes=(512, 256, 128), phi_dim=phi_dim)
    else:
        raise ValueError(f"Unknown task: {task}")
