"""Score-network architectures for Neural Posterior Score Estimation.

This module implements the score-network components described in the paper:

* a 3-layer fully-connected ``theta`` embedding with 256 hidden units,
* a 3-layer fully-connected ``x`` (observation) embedding with 256 hidden units,
* a 64-dimensional sinusoidal time embedding,
* a final 3-layer fully-connected MLP with 256 hidden units that maps the
  concatenated embeddings to a score of the same dimensionality as ``theta``.

All MLPs use SiLU activations.  The network also applies the required
per-dimension standardisation of ``theta_t`` and ``x`` before the embeddings.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _hidden_size(dim: int) -> int:
    """Output dimension of theta/x embedding MLPs."""
    return max(30, 4 * dim)


class MLP(nn.Module):
    """A simple fully-connected MLP with SiLU activations.

    Parameters
    ----------
    in_dim:
        Input dimensionality.
    out_dim:
        Output dimensionality.
    hidden_dim:
        Width of every hidden layer (default 256).
    num_layers:
        Total number of linear layers.  ``num_layers=3`` corresponds to
        ``in -> hidden -> hidden -> out``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding with the paper's 64-dimensional scheme.

    For ``i = 1, ..., 32`` the embedding uses sine terms and for
    ``i = 33, ..., 64`` cosine terms.  Indexing below is 0-based so that the
    first 32 components are sine and the last 32 are cosine.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for this sinusoidal scheme")
        self.embedding_dim = embedding_dim
        self.n_freq = embedding_dim // 2
        # Frequencies used by the paper: 10000^((i-1)/31) for the 32 terms.
        exponents = torch.arange(self.n_freq, dtype=torch.float32) / float(self.n_freq - 1)
        self.register_buffer("freq", 10000.0 ** exponents, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return shape ``(*t.shape, embedding_dim)``.

        ``t`` may be a scalar, a 1D tensor, or a tensor of arbitrary leading
        shape.  Its final dimension, if present, should have size one.
        """
        t = t.float()
        leading = t.shape
        flat_t = t.reshape(-1)
        scaled = flat_t[:, None] / self.freq[None, :]  # (N, n_freq)
        sine = torch.sin(scaled)
        cosine = torch.cos(scaled)
        emb = torch.cat([sine, cosine], dim=-1)  # (N, embedding_dim)
        return emb.reshape(*leading, self.embedding_dim)


class EmbeddingMLP(MLP):
    """3-layer embedding MLP with the paper's default hidden width."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(in_dim, out_dim, hidden_dim=hidden_dim, num_layers=3)


class Standardizer(nn.Module):
    """Per-dimension affine standardisation.

    Keeps ``mean`` and ``std`` as buffers.  Standardisation is
    ``(x - mean) / std`` with ``std`` clamped to a small positive value to
    avoid division by zero for constant dimensions.
    """

    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        if mean is not None:
            self.register_buffer("mean", mean.to(torch.float32).reshape(-1))
        else:
            self.register_buffer("mean", torch.zeros(0))
        if std is not None:
            self.register_buffer("std", std.to(torch.float32).reshape(-1))
        else:
            self.register_buffer("std", torch.ones(0))

    def set(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.to(torch.float32).reshape(-1)
        std = std.to(torch.float32).reshape(-1)
        if mean.numel() != self.mean.numel() and self.mean.numel() != 0:
            raise ValueError("Standardizer dimension mismatch")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.to(mean.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean.numel() == 0:
            return x
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std.clamp_min(self.eps)


class ScoreNetwork(nn.Module):
    """Posterior score network ``s_psi(theta_t, x, t)``.

    Parameters
    ----------
    theta_dim:
        Dimensionality of the parameters ``theta``.
    x_dim:
        Dimensionality of the observations ``x``.
    hidden_dim:
        Hidden width used in all MLPs (256 in the paper).
    time_embedding_dim:
        Dimension of the sinusoidal time embedding (64 in the paper).
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        hidden_dim: int = 256,
        time_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim

        self.theta_embedder = EmbeddingMLP(theta_dim, _hidden_size(theta_dim), hidden_dim)
        self.x_embedder = EmbeddingMLP(x_dim, _hidden_size(x_dim), hidden_dim)
        self.time_embedder = SinusoidalTimeEmbedding(time_embedding_dim)

        score_in_dim = _hidden_size(theta_dim) + _hidden_size(x_dim) + time_embedding_dim
        self.score_mlp = MLP(score_in_dim, theta_dim, hidden_dim=hidden_dim, num_layers=3)

        # Per-dimension standardisation for theta_t and x.
        self.theta_standardizer = Standardizer()
        self.x_standardizer = Standardizer()

    def set_standardization(
        self,
        theta_mean: Optional[torch.Tensor] = None,
        theta_std: Optional[torch.Tensor] = None,
        x_mean: Optional[torch.Tensor] = None,
        x_std: Optional[torch.Tensor] = None,
    ) -> None:
        """Set empirical standardisation parameters."""
        if theta_mean is not None:
            self.theta_standardizer.set(theta_mean, theta_std if theta_std is not None else torch.ones_like(theta_mean))
        if x_mean is not None and self.x_dim > 0:
            self.x_standardizer.set(x_mean, x_std if x_std is not None else torch.ones_like(x_mean))

    def forward(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the score for perturbed parameters ``theta_t``.

        Parameters
        ----------
        theta:
            Perturbed parameters, shape ``(..., theta_dim)``.
        x:
            Observations, shape ``(..., x_dim)``.
        t:
            Diffusion time, shape ``(...)`` or ``(..., 1)``.

        Returns
        -------
        Score tensor with the same shape as ``theta``.
        """
        leading = theta.shape[:-1]
        theta_flat = theta.reshape(-1, self.theta_dim)
        x_flat = x.reshape(-1, self.x_dim)

        theta_std = self.theta_standardizer(theta_flat)
        x_std = self.x_standardizer(x_flat)

        theta_emb = self.theta_embedder(theta_std)
        x_emb = self.x_embedder(x_std)
        t_emb = self.time_embedder(t)  # shape (..., time_embedding_dim)
        t_flat = t_emb.reshape(-1, self.time_embedding_dim)

        h = torch.cat([theta_emb, x_emb, t_flat], dim=-1)
        out = self.score_mlp(h)
        return out.reshape(*leading, self.theta_dim)


class PriorScoreNetwork(nn.Module):
    """A time-only score network ``s_prior(theta_t, t)`` for implicit priors.

    This is used by ``src/prior.py`` when the prior score is not available in
    closed form and must be estimated by prior denoising score matching.
    """

    def __init__(
        self,
        theta_dim: int,
        hidden_dim: int = 256,
        time_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.theta_dim = theta_dim
        self.theta_embedder = EmbeddingMLP(theta_dim, _hidden_size(theta_dim), hidden_dim)
        self.time_embedder = SinusoidalTimeEmbedding(time_embedding_dim)
        in_dim = _hidden_size(theta_dim) + time_embedding_dim
        self.score_mlp = MLP(in_dim, theta_dim, hidden_dim=hidden_dim, num_layers=3)
        self.theta_standardizer = Standardizer()

    def set_standardization(
        self,
        theta_mean: Optional[torch.Tensor] = None,
        theta_std: Optional[torch.Tensor] = None,
    ) -> None:
        if theta_mean is not None:
            self.theta_standardizer.set(
                theta_mean, theta_std if theta_std is not None else torch.ones_like(theta_mean)
            )

    def forward(self, theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        leading = theta.shape[:-1]
        theta_flat = theta.reshape(-1, self.theta_dim)
        theta_std = self.theta_standardizer(theta_flat)
        theta_emb = self.theta_embedder(theta_std)
        t_flat = self.time_embedder(t).reshape(-1, self.time_embedder.embedding_dim)
        h = torch.cat([theta_emb, t_flat], dim=-1)
        out = self.score_mlp(h)
        return out.reshape(*leading, self.theta_dim)


def compute_standardization_stats(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-dimension mean and std over the first axis of ``samples``."""
    samples = samples.reshape(samples.shape[0], -1)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    return mean, std
