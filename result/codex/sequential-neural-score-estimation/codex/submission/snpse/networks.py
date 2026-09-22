"""Score network architecture (Appendix E.3.2).

The architecture is deliberately simple and is used unchanged across every
benchmark experiment and the real-world neuroscience experiment:

* a 3-layer MLP embedding network for ``theta_t`` (output dim ``max(30, 4 d)``),
* a 3-layer MLP embedding network for ``x`` (output dim ``max(30, 4 p)``),
* a 64 dimensional sinusoidal embedding of ``t``,
* the three embeddings are concatenated and fed into a final 3-layer MLP with
  ``d`` output dimensions (one per parameter).

All fully connected layers have 256 hidden units and SiLU activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int = 64, t_scale: float = 1000.0) -> torch.Tensor:
    """Sinusoidal embedding of the diffusion time (Appendix E.3.2).

    ``(t_emb)_i = sin(t_scale * t / 10000^((i-1)/31))`` for ``i <= 32`` and
    ``(t_emb)_i = cos(t_scale * t / 10000^(((i-32)-1)/31))`` for ``i > 32``.

    The paper writes the embedding with ``t_scale = 1``.  Since the diffusion
    time is normalised to ``t in [0, 1]``, that choice makes every sinusoidal
    feature almost linear in ``t`` and gives the network very little resolution
    in the noise level; score-based generative modelling codebases
    (e.g. Song et al., 2021) instead multiply the sinusoidal argument by 1000,
    and we follow that convention by default.  ``t_scale`` is exposed as a
    hyperparameter (``TrainingConfig.t_scale``) and set to 1.0 to reproduce the
    literal reading of the paper.
    """
    if dim % 2 != 0:
        raise ValueError("sinusoidal embedding dimension must be even")
    half = dim // 2
    device, dtype = t.device, t.dtype
    idx = torch.arange(half, device=device, dtype=dtype)
    denom = 10000.0 ** (idx / (half - 1))
    t = t.reshape(-1, 1) / denom.reshape(1, -1)
    t = t * t_scale
    return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


def mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    activation: str = "silu",
) -> nn.Sequential:
    """Fully connected MLP with ``num_layers`` linear layers and activations."""
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    act = {"silu": nn.SiLU, "swish": nn.SiLU, "relu": nn.ReLU}[activation.lower()]
    layers = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(act())
    return nn.Sequential(*layers)


class ScoreNetwork(nn.Module):
    """Time-dependent conditional score network ``s_psi(theta_t, x, t)``."""

    def __init__(
        self,
        dim_parameters: int,
        dim_data: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 64,
        t_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.dim_parameters = dim_parameters
        self.dim_data = dim_data
        self.time_embed_dim = time_embed_dim
        self.t_scale = t_scale

        theta_out = max(30, 4 * dim_parameters)
        x_out = max(30, 4 * dim_data)
        self.theta_embed = mlp(dim_parameters, theta_out, hidden_dim, num_layers)
        self.x_embed = mlp(dim_data, x_out, hidden_dim, num_layers)
        self.score_mlp = mlp(theta_out + x_out + time_embed_dim, dim_parameters, hidden_dim, num_layers)

        # A small initialisation scale keeps the initial predicted score small,
        # which stabilises the first few training steps.
        with torch.no_grad():
            last = [m for m in self.score_mlp if isinstance(m, nn.Linear)][-1]
            last.weight.mul_(0.1)
            last.bias.mul_(0.1)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the score of the perturbed posterior.

        Args:
            theta_t: (B, d) perturbed parameters.
            x: (B, p) observations (or (p,) to be broadcast).
            t: (B,) diffusion times in [0, 1].
        """
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[0] != theta_t.shape[0]:
            x = x.expand(theta_t.shape[0], -1)
        t = t.reshape(-1)
        if t.shape[0] != theta_t.shape[0]:
            t = t.expand(theta_t.shape[0])
        emb = torch.cat(
            [
                self.theta_embed(theta_t),
                self.x_embed(x),
                sinusoidal_embedding(t, self.time_embed_dim, self.t_scale),
            ],
            dim=-1,
        )
        return self.score_mlp(emb)


class EnergyNetwork(nn.Module):
    """Scalar energy network ``E_psi(theta_t, x, t)`` (Appendix G).

    An alternative parameterisation of the diffusion model in which the score
    is obtained as ``s_psi(theta_t, x, t) = -grad_theta E_psi(theta_t, x, t)``.
    Unlike the direct score parameterisation, this yields an unnormalised
    density ``p_t(theta_t | x) ∝ exp(-E_psi(theta_t, x, t))`` for free, which
    removes the need for the instantaneous change-of-variables formula (5) when
    computing the truncated proposal of TSNPSE.
    """

    def __init__(
        self,
        dim_parameters: int,
        dim_data: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 64,
        t_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.dim_parameters = dim_parameters
        self.dim_data = dim_data
        self.time_embed_dim = time_embed_dim
        self.t_scale = t_scale
        theta_out = max(30, 4 * dim_parameters)
        x_out = max(30, 4 * dim_data)
        self.theta_embed = mlp(dim_parameters, theta_out, hidden_dim, num_layers)
        self.x_embed = mlp(dim_data, x_out, hidden_dim, num_layers)
        self.energy_mlp = mlp(theta_out + x_out + time_embed_dim, 1, hidden_dim, num_layers)

    def energy(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[0] != theta_t.shape[0]:
            x = x.expand(theta_t.shape[0], -1)
        t = t.reshape(-1)
        if t.shape[0] != theta_t.shape[0]:
            t = t.expand(theta_t.shape[0])
        emb = torch.cat(
            [
                self.theta_embed(theta_t),
                self.x_embed(x),
                sinusoidal_embedding(t, self.time_embed_dim, self.t_scale),
            ],
            dim=-1,
        )
        return self.energy_mlp(emb).squeeze(-1)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score, obtained as the negative gradient of the energy."""
        with torch.enable_grad():
            theta_t = theta_t.detach().requires_grad_(True)
            e = self.energy(theta_t, x, t).sum()
            # `create_graph=True` keeps the score differentiable, which the
            # instantaneous change-of-variables formula (Eq. 5) requires.
            (grad,) = torch.autograd.grad(e, theta_t, create_graph=True)
        return -grad
