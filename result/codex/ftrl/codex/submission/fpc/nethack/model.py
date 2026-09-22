"""NetHack model architecture (Appendix B.1).

The model is the one introduced by Tuyls et al. (2023), scaling up the
``Chaotic Dwarven GPT-5`` solution built on Sample Factory (Petrenko et al.,
2020):

* the main dungeon screen is encoded by embedding each ``(character, colour)``
  pair with an embedding lookup table and processing the resulting grid with a
  ResNet,
* ``blstats`` (health, hunger, ...) and ``message`` (textual notifications) are
  processed by two-layer MLPs,
* the three representations are merged and fed into an LSTM whose output is
  passed to a policy head and a baseline (value) head,
* the actor and the critic share the backbone.

The default hidden size is 1738 (Table 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .config import NetHackConfig


@dataclass
class NetHackObservation:
    """Container for a batch of NLE observations."""

    glyphs: Tensor          # (B, H, W) long -- character ids
    colors: Tensor          # (B, H, W) long -- colour ids
    blstats: Tensor         # (B, 25) float
    message: Tensor         # (B, 256) long -- character ids

    def to(self, device) -> "NetHackObservation":
        return NetHackObservation(
            glyphs=self.glyphs.to(device),
            colors=self.colors.to(device),
            blstats=self.blstats.to(device),
            message=self.message.to(device),
        )


class ResidualBlock(nn.Module):
    """A standard pre-activation residual block with two 3x3 convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class ScreenEncoder(nn.Module):
    """Embed ``(character, colour)`` per cell and process the grid with a ResNet."""

    def __init__(self, config: NetHackConfig) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(config.num_chars, config.char_embed_dim)
        self.color_embedding = nn.Embedding(config.num_colors, config.color_embed_dim)
        in_channels = config.char_embed_dim + config.color_embed_dim
        self.input_conv = nn.Conv2d(in_channels, config.resnet_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(config.resnet_channels) for _ in range(config.resnet_blocks)]
        )
        self.output_dim = config.resnet_channels * 2  # (mean, max) pooling

    def forward(self, glyphs: Tensor, colors: Tensor) -> Tensor:
        char_emb = self.char_embedding(glyphs)                 # (B, H, W, C1)
        color_emb = self.color_embedding(colors)               # (B, H, W, C2)
        x = torch.cat([char_emb, color_emb], dim=-1).permute(0, 3, 1, 2)
        x = F.relu(self.input_conv(x))
        x = self.blocks(x)
        mean = x.mean(dim=(2, 3))
        maximum = x.amax(dim=(2, 3))
        return torch.cat([mean, maximum], dim=-1)


class MLPEncoder(nn.Module):
    """Two-layer MLP used for ``blstats`` and ``message`` (Appendix B.1)."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: Optional[int] = None,
                 num_embeddings: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim) if embedding_dim else None
        if embedding_dim:
            input_dim = input_dim * embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.embedding is not None:
            x = self.embedding(x.long()).flatten(start_dim=1)
        else:
            x = x.float()
        return self.net(x)


class NetHackModel(nn.Module):
    """Shared-backbone actor-critic with an LSTM core."""

    def __init__(self, config: Optional[NetHackConfig] = None) -> None:
        super().__init__()
        self.config = config or NetHackConfig()
        cfg = self.config

        self.screen_encoder = ScreenEncoder(cfg)
        self.blstats_encoder = MLPEncoder(cfg.blstats_length, cfg.mlp_hidden)
        self.message_encoder = MLPEncoder(
            cfg.message_length, cfg.mlp_hidden, embedding_dim=8, num_embeddings=cfg.num_chars
        )

        merged_dim = self.screen_encoder.output_dim + self.blstats_encoder.output_dim + self.message_encoder.output_dim
        self.merge = nn.Sequential(nn.Linear(merged_dim, cfg.hidden_dim), nn.ReLU())
        self.core = nn.LSTM(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(cfg.hidden_dim, cfg.num_actions)
        self.baseline_head = nn.Linear(cfg.hidden_dim, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.baseline_head.weight, gain=1.0)
        nn.init.zeros_(self.baseline_head.bias)

    # ------------------------------------------------------------------
    def encode(self, obs: NetHackObservation) -> Tensor:
        screen = self.screen_encoder(obs.glyphs, obs.colors)
        blstats = self.blstats_encoder(obs.blstats)
        message = self.message_encoder(obs.message)
        return self.merge(torch.cat([screen, blstats, message], dim=-1))

    def forward(self, obs: NetHackObservation, states=None) -> Tuple[Tensor, Tensor, object]:
        """Return ``(policy_logits, baseline, lstm_states)``.

        ``obs`` tensors are shaped ``(B, T, ...)``; the returned logits are
        ``(B, T, num_actions)``.
        """

        batch, timesteps = obs.glyphs.shape[0], obs.glyphs.shape[1]
        flat = NetHackObservation(
            glyphs=obs.glyphs.reshape(batch * timesteps, *obs.glyphs.shape[2:]),
            colors=obs.colors.reshape(batch * timesteps, *obs.colors.shape[2:]),
            blstats=obs.blstats.reshape(batch * timesteps, -1),
            message=obs.message.reshape(batch * timesteps, -1),
        )
        embeddings = self.encode(flat).reshape(batch, timesteps, -1)
        core_out, new_states = self.core(embeddings, states)
        return self.policy_head(core_out), self.baseline_head(core_out), new_states

    def initial_states(self, batch_size: int, device) -> Tuple[Tensor, Tensor]:
        cfg = self.config
        zeros = torch.zeros(1, batch_size, cfg.hidden_dim, device=device)
        return zeros, zeros.clone()

    def distribution(self, obs: NetHackObservation, states=None) -> Categorical:
        logits, _, _ = self.forward(obs, states)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs: NetHackObservation, states=None):
        logits, baseline, new_states = self.forward(obs, states)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), baseline, new_states

    def freeze_encoders(self) -> None:
        """Freeze the screen/blstats/message encoders (Appendix B.1)."""

        for module in (self.screen_encoder, self.blstats_encoder, self.message_encoder, self.merge):
            for p in module.parameters():
                p.requires_grad_(False)
