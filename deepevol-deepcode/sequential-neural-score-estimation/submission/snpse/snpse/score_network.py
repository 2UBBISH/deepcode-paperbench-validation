"""Score network architecture for SNPSE / NPSE / NLSE.

This module implements the conditional score network ``s_psi(theta_t, x, t)``
used throughout the paper, following Appendix E.3.2 verbatim:

* ``theta_t`` embedding network: 3-layer fully-connected MLP with 256 hidden
  units in each layer.  Input dimension ``d`` (``theta in R^d``), output
  dimension ``max(30, 4 * d)``.
* ``x`` embedding network: 3-layer fully-connected MLP with 256 hidden units in
  each layer.  Input dimension ``p`` (``x in R^p``), output dimension
  ``max(30, 4 * p)``.
* ``t`` sinusoidal embedding into 64 dimensions (Vaswani et al., 2017)::

      (t_emb)_i = sin(t / 10000 ** ((i - 1) / 31))         if i <= 32
      (t_emb)_i = cos(t / 10000 ** (((i - 32) - 1) / 31))  if i > 32

* Score network: concatenate ``[theta_emb, x_emb, t_emb]`` and feed into a
  3-layer fully-connected MLP with 256 hidden units in each layer whose output
  dimension is ``d``.
* SiLU activation functions between layers for all MLP networks.

Appendix G additionally describes the *energy-based* parameterisation
``s_psi(theta_t, x, t) = -grad_{theta_t} E_psi(theta_t, x, t)`` with a scalar
``E_psi`` (final layer with a single output), which yields an unnormalised
perturbed posterior ``p_t(theta_t | x) ~ exp(-E_psi(theta_t, x, t))``.  This is
implemented by :class:`EnergyNetwork` / :class:`EnergyScoreNetwork`.

Note on "3 layers": as in Appendix E.3.2 the three layers of the embedding and
head MLPs have 256 hidden units, hence the default module contains three
``nn.Linear`` layers (two hidden of width 256 plus the input/output
projections); this is exposed through ``n_layers``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "sinusoidal_embedding",
    "MLP",
    "ScoreNetwork",
    "EnergyNetwork",
    "EnergyScoreNetwork",
    "get_score_network",
    "embedding_dim",
]

# --------------------------------------------------------------------------- #
# Defaults from Appendix E.3.2
# --------------------------------------------------------------------------- #
HIDDEN_DIM: int = 256
N_LAYERS: int = 3
TIME_EMBED_DIM: int = 64
MIN_EMBED_DIM: int = 30
EMBED_MULTIPLIER: int = 4
T_MAX: float = 1.0


def _make_activation(name: str) -> nn.Module:
    """Build an activation module by name (SiLU is the paper's default)."""
    name = str(name).lower()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    raise ValueError(f"Unknown activation '{name}'.")


def embedding_dim(input_dim: int) -> int:
    """Output dimension of an embedding MLP: ``max(30, 4 * input_dim)``."""
    return max(MIN_EMBED_DIM, EMBED_MULTIPLIER * int(input_dim))


class MLP(nn.Module):
    """Fully-connected MLP with ``n_layers`` linear layers.

    The hidden layers have ``hidden_dim`` units (256 in the paper) and the
    activation (SiLU by default) is applied between layers.  The final linear
    layer maps to ``out_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        activation: str = "silu",
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)

        layers: list[nn.Module] = []
        current = self.in_dim
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(current, self.hidden_dim))
            layers.append(_make_activation(activation))
            current = self.hidden_dim
        layers.append(nn.Linear(current, self.out_dim))
        if final_activation:
            layers.append(_make_activation(activation))
        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Default PyTorch initialisation (plan: unspecified -> default init)."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def sinusoidal_embedding(
    t: torch.Tensor,
    dim: int = TIME_EMBED_DIM,
    scale: float = 1.0,
) -> torch.Tensor:
    r"""Sinusoidal time embedding of Appendix E.3.2 (equation for ``t_emb``).

    .. math::

        (t_{emb})_i = \sin\left(\frac{t}{10000^{(i-1)/31}}\right),
        \quad i \le 32, \qquad
        (t_{emb})_i = \cos\left(\frac{t}{10000^{((i-32)-1)/31}}\right),
        \quad i > 32 .

    The formula is written for ``dim = 64`` (two halves of 32 entries, with the
    divisor exponent normalised by ``31 = dim/2 - 1``).  It is generalised here
    to an arbitrary even ``dim`` by using ``dim/2`` frequencies and the same
    normalisation constant ``dim/2 - 1``.

    Args:
        t: Time values, any shape.  ``t in [0, T]`` with ``T = 1`` in this code.
        dim: Embedding dimension (64 in the paper).
        scale: Optional multiplier applied to ``t`` before embedding.

    Returns:
        Tensor of shape ``(*t.shape, dim)``.
    """
    if dim % 2 != 0:
        raise ValueError("Embedding dimension must be even.")
    half = dim // 2
    t = t.reshape(-1, 1).float() * float(scale)
    idx = torch.arange(half, device=t.device, dtype=t.dtype)  # 0 ... half-1
    # For i <= 32 -> exponent (i - 1) / 31 with i = 1 ... 32  =>  0 ... 31 = idx.
    denom = 10000.0 ** (idx / max(half - 1, 1))
    angles = t / denom  # (n, half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ScoreNetwork(nn.Module):
    """Vector-valued conditional score network ``s_psi(theta_t, x, t)``.

    ``[theta_emb, x_emb, t_emb]`` is fed into a 3-layer MLP with 256 hidden
    units per layer producing a vector of dimension ``d``.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        theta_emb_dim: Optional[int] = None,
        x_emb_dim: Optional[int] = None,
        time_emb_dim: int = TIME_EMBED_DIM,
        activation: str = "silu",
        t_max: float = T_MAX,
        t_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.time_emb_dim = int(time_emb_dim)
        self.theta_emb_dim = int(theta_emb_dim) if theta_emb_dim is not None else embedding_dim(self.theta_dim)
        self.x_emb_dim = int(x_emb_dim) if x_emb_dim is not None else embedding_dim(self.x_dim)
        # Paper feeds t (in [0, 1]) directly into the sinusoidal embedding.
        self.t_scale = float(t_scale) if t_scale is not None else float(t_max)
        self.activation = activation

        self.theta_net = MLP(
            self.theta_dim, self.theta_emb_dim, hidden_dim=self.hidden_dim,
            n_layers=self.n_layers, activation=activation,
        )
        self.x_net = MLP(
            self.x_dim, self.x_emb_dim, hidden_dim=self.hidden_dim,
            n_layers=self.n_layers, activation=activation,
        )
        self.score_net = MLP(
            self.theta_emb_dim + self.x_emb_dim + self.time_emb_dim,
            self.theta_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            activation=activation,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _flatten_inputs(value: torch.Tensor, name: str) -> torch.Tensor:
        """Accept ``(n, d)`` or higher-rank tensors by flattening trailing dims."""
        if value.dim() == 1:
            value = value.reshape(1, -1)
        return value.reshape(value.shape[0], -1)

    def _tensor_for_t(self, t: torch.Tensor, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=reference.dtype, device=reference.device)
        t = t.to(device=reference.device, dtype=reference.dtype)
        if t.dim() == 0:
            t = t.expand(batch_size)
        return t.reshape(-1)

    def embed(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``[theta_emb, x_emb, t_emb]`` concatenated along the last axis."""
        theta_t = self._flatten_inputs(theta_t, "theta_t")
        x = self._flatten_inputs(x, "x")
        if x.shape[0] != theta_t.shape[0]:
            if x.shape[0] == 1:
                x = x.expand(theta_t.shape[0], -1)
            elif theta_t.shape[0] == 1:
                theta_t = theta_t.expand(x.shape[0], -1)
            else:
                raise ValueError(
                    f"Batch size mismatch between theta_t ({theta_t.shape[0]}) and x ({x.shape[0]})."
                )
        t = self._tensor_for_t(t, theta_t.shape[0], theta_t)

        theta_emb = self.theta_net(theta_t)
        x_emb = self.x_net(x)
        t_emb = sinusoidal_embedding(t, dim=self.time_emb_dim, scale=self.t_scale).to(theta_emb.dtype)
        return torch.cat([theta_emb, x_emb, t_emb], dim=-1)

    # ------------------------------------------------------------------ #
    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimate ``grad_{theta_t} log p_t(theta_t | x)``."""
        return self.score_net(self.embed(theta_t, x, t))


class EnergyNetwork(nn.Module):
    """Scalar energy network ``E_psi(theta_t, x, t)`` (Appendix G)."""

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        theta_emb_dim: Optional[int] = None,
        x_emb_dim: Optional[int] = None,
        time_emb_dim: int = TIME_EMBED_DIM,
        activation: str = "silu",
        t_max: float = T_MAX,
        t_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.time_emb_dim = int(time_emb_dim)
        self.theta_emb_dim = int(theta_emb_dim) if theta_emb_dim is not None else embedding_dim(self.theta_dim)
        self.x_emb_dim = int(x_emb_dim) if x_emb_dim is not None else embedding_dim(self.x_dim)
        self.t_scale = float(t_scale) if t_scale is not None else float(t_max)

        self.theta_net = MLP(
            self.theta_dim, self.theta_emb_dim, hidden_dim=self.hidden_dim,
            n_layers=self.n_layers, activation=activation,
        )
        self.x_net = MLP(
            self.x_dim, self.x_emb_dim, hidden_dim=self.hidden_dim,
            n_layers=self.n_layers, activation=activation,
        )
        # Final layer has a single output -> scalar energy.
        self.energy_net = MLP(
            self.theta_emb_dim + self.x_emb_dim + self.time_emb_dim,
            1,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            activation=activation,
        )

        self._score_mlp = ScoreNetwork.__new__(ScoreNetwork)  # helper reuse (no params)

    # ------------------------------------------------------------------ #
    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the scalar energy of shape ``(n, 1)``."""
        theta_t = ScoreNetwork._flatten_inputs(theta_t, "theta_t")
        x = ScoreNetwork._flatten_inputs(x, "x")
        if x.shape[0] != theta_t.shape[0]:
            if x.shape[0] == 1:
                x = x.expand(theta_t.shape[0], -1)
            elif theta_t.shape[0] == 1:
                theta_t = theta_t.expand(x.shape[0], -1)
            else:
                raise ValueError("Batch size mismatch between theta_t and x.")
        t = ScoreNetwork._tensor_for_t(self, t, theta_t.shape[0], theta_t)
        theta_emb = self.theta_net(theta_t)
        x_emb = self.x_net(x)
        t_emb = sinusoidal_embedding(t, dim=self.time_emb_dim, scale=self.t_scale).to(theta_emb.dtype)
        return self.energy_net(torch.cat([theta_emb, x_emb, t_emb], dim=-1))

    def score(
        self,
        theta_t: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """``s = -grad_{theta_t} E_psi`` computed via autograd (see Appendix G)."""
        with torch.enable_grad():
            needs_grad = not theta_t.requires_grad
            if needs_grad:
                theta_t = theta_t.detach().requires_grad_(True)
            energy = self.forward(theta_t, x, t).sum()
            (grad,) = torch.autograd.grad(
                energy, theta_t, create_graph=create_graph, retain_graph=create_graph
            )
        return -grad


class EnergyScoreNetwork(nn.Module):
    """Score network wrapper with the energy-based parameterisation.

    ``forward(theta_t, x, t)`` returns ``-grad_{theta_t} E_psi(theta_t, x, t)``
    so that it is a drop-in replacement for :class:`ScoreNetwork` inside the DSM
    losses.  The (unnormalised) perturbed posterior is
    ``p_t(theta_t | x) ~ exp(-E_psi(theta_t, x, t))``, which makes HPR_eps
    estimation and truncated-proposal sampling a single forward pass.
    """

    def __init__(self, energy_net: Optional[EnergyNetwork] = None, **kwargs) -> None:
        super().__init__()
        self.energy_net = energy_net if energy_net is not None else EnergyNetwork(**kwargs)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.energy_net.score(theta_t, x, t, create_graph=True)

    @torch.no_grad()
    def energy(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Scalar energy, useful for cheap truncated-proposal acceptance tests."""
        return self.energy_net(theta_t, x, t)


def get_score_network(
    theta_dim: int,
    x_dim: int,
    parameterisation: str = "score",
    **kwargs,
) -> nn.Module:
    """Factory for the paper's score network.

    Args:
        theta_dim: Dimension ``d`` of the parameters.
        x_dim: Dimension ``p`` of the observations.
        parameterisation: ``"score"`` (default, vector-valued ``s_psi``) or
            ``"energy"`` (Appendix G, scalar ``E_psi`` with ``s = -grad E``).
        **kwargs: Additional keyword arguments (``hidden_dim``, ``n_layers``,
            ``time_emb_dim``, ``activation``, ...) forwarded to the network.
    """
    parameterisation = str(parameterisation).lower()
    if parameterisation in ("score", "vector", "direct"):
        return ScoreNetwork(theta_dim=theta_dim, x_dim=x_dim, **kwargs)
    if parameterisation in ("energy", "energy-based", "energy_based"):
        return EnergyScoreNetwork(theta_dim=theta_dim, x_dim=x_dim, **kwargs)
    raise ValueError(f"Unknown parameterisation '{parameterisation}'.")


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters (utility for sanity checks)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
