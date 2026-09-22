"""Small velocity networks for the Gaussian-mixture studies (Figure 2)."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class TimeFeatures(nn.Module):
    """Random Fourier features of t, as in the 2-D experiments of the paper."""

    def __init__(self, dim: int = 32, scale: float = 1.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t: Tensor) -> Tensor:
        freqs = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(freqs), torch.cos(freqs), t[:, None]], dim=-1)


class MLPVelocity(nn.Module):
    """Fully-connected velocity field b_t(x, xi) on R^dim.

    Signature-compatible with
    :class:`si_couplings.models.velocity.VelocityModel`, i.e. it can be passed
    directly to the loss functions in :mod:`si_couplings.losses`.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden: int = 128,
        depth: int = 3,
        time_dim: int = 32,
        num_classes: int = 0,
        cond_dim: int = 0,
        activation: str = "silu",
        time_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes if num_classes > 0 else None
        self.time_scale = time_scale
        self.time_features = TimeFeatures(time_dim)
        embed_dim = dim + time_dim + 1 + cond_dim
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(self.num_classes, hidden)
            embed_dim += hidden
        act = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        layers = [nn.Linear(embed_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if t.dim() > 1:
            t = t.reshape(t.shape[0])
        feats = [x, self.time_features(t * self.time_scale)]
        if self.num_classes is not None and labels is not None:
            feats.append(self.label_emb(labels))
        if cond is not None:
            feats.append(cond if cond.dim() == 2 else cond.flatten(1))
        h = torch.cat(feats, dim=-1)
        return self.net(h)


class ScoreModel(nn.Module):
    """Network g_hat_t(x, xi) approximating E[z | I_t = x] (eq. 7)."""

    def __init__(self, dim: int = 2, hidden: int = 128, depth: int = 3, time_dim: int = 32, **kwargs):
        super().__init__()
        self.dim = dim
        self.time_features = TimeFeatures(time_dim)
        layers = [nn.Linear(dim + time_dim + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor, cond=None, labels=None, mask=None) -> Tensor:
        if t.dim() > 1:
            t = t.reshape(t.shape[0])
        return self.net(torch.cat([x, self.time_features(t)], dim=-1))


__all__ = ["TimeFeatures", "MLPVelocity", "ScoreModel"]
