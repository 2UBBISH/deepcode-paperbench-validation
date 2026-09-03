"""Functional Reward Encoding variational autoencoder.

This module implements the core FRE model described in the paper:

* A permutation-invariant transformer encoder consumes a small set of
  ``(state, reward)`` context pairs and produces a latent vector ``z``.
* A Gaussian latent distribution is parameterized from the mean-pooled
  transformer output with a standard VAE reparameterization.
* A conditional reward decoder ``q(reward | state, z)`` reconstructs the
  underlying reward function for decoder states.

The loss is::

    L_FRE = E_{q(z|c)} [ -1/K' sum_k (r_pred(s_k, z) - eta(s_k))^2 ]
            - beta * KL(q(z|c) || N(0, I))

where ``c`` is the encoder context and ``s_1..s_K'`` are independently
sampled decoder states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from fre.modeling.decoder import RewardDecoder
from fre.modeling.reward_embedding import RewardEmbedding
from fre.modeling.transformer_encoder import TransformerEncoder


@dataclass
class FREOutput:
    """Outputs produced by :meth:`FREVAE.forward`."""

    loss: torch.Tensor
    reconstruction_mse: torch.Tensor
    kl: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    log_sigma: torch.Tensor
    reward_pred: torch.Tensor
    reward_target: torch.Tensor


class FREVAE(nn.Module):
    """Functional Reward Encoding variational autoencoder.

    Parameters
    ----------
    state_dim:
        Dimensionality of the state observations.
    z_dim:
        Latent dimensionality. Defaults to 64.
    d_model:
        Transformer model width. Defaults to 128.
    d_ff:
        Transformer feed-forward width. Defaults to 256.
    n_heads:
        Number of attention heads. Defaults to 4.
    n_layers:
        Number of transformer encoder layers. Defaults to 2.
    num_bins:
        Number of reward discretization bins. Defaults to 128.
    reward_min:
        Minimum scalar reward used by the reward embedding. Defaults to -1.0.
    reward_max:
        Maximum scalar reward used by the reward embedding. Defaults to 1.0.
    use_linear_reward:
        If ``True``, use a learned linear projection from scalar reward to the
        embedding dimension instead of a discrete embedding table.
    dropout:
        Transformer dropout. Defaults to 0.0.
    activation:
        Transformer feed-forward activation. Defaults to ``"relu"``.
    decoder_hidden_dim:
        Reward decoder hidden width. Defaults to 256.
    decoder_num_hidden:
        Number of reward decoder hidden layers. Defaults to 2.
    beta:
        KL weight. Defaults to 1.0.
    normalize_rewards:
        If ``True``, rewards used for both embedding and reconstruction are
        clipped into ``[reward_min, reward_max]``. If ``False``, the raw
        values are passed through (the embedding layer itself still clips).
    """

    def __init__(
        self,
        state_dim: int,
        z_dim: int = 64,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        num_bins: int = 128,
        reward_min: float = -1.0,
        reward_max: float = 1.0,
        use_linear_reward: bool = False,
        dropout: float = 0.0,
        activation: str = "relu",
        decoder_hidden_dim: int = 256,
        decoder_num_hidden: int = 2,
        beta: float = 1.0,
        normalize_rewards: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.state_dim = int(state_dim)
        self.z_dim = int(z_dim)
        self.d_model = int(d_model)
        self.num_bins = int(num_bins)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)
        self.beta = float(beta)
        self.normalize_rewards = bool(normalize_rewards)

        # State tokens are projected into the transformer embedding space.
        self.state_embedding = nn.Linear(self.state_dim, self.d_model)
        # Reward tokens: uniform-magnitude discretization + learned embedding.
        self.reward_embedding = RewardEmbedding(
            num_bins=self.num_bins,
            embedding_dim=self.d_model,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            use_linear=use_linear_reward,
        )
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            input_dim=2 * self.d_model,
        )

        # VAE latent heads.
        self.mu_head = nn.Linear(self.d_model, self.z_dim)
        self.log_sigma_head = nn.Linear(self.d_model, self.z_dim)

        # Conditional reward decoder.
        self.decoder = RewardDecoder(
            state_dim=self.state_dim,
            z_dim=self.z_dim,
            hidden_dim=decoder_hidden_dim,
            num_hidden=decoder_num_hidden,
            activation=activation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_reward_values(self, rewards: torch.Tensor) -> torch.Tensor:
        """Clip scalar rewards into the bounded embedding range."""
        if self.normalize_rewards:
            return torch.clamp(rewards, min=self.reward_min, max=self.reward_max)
        return rewards

    def encode(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        return_z: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode a set of ``(state, reward)`` context pairs.

        Parameters
        ----------
        states:
            Tensor of shape ``(batch, num_tokens, state_dim)``.
        rewards:
            Tensor of shape ``(batch, num_tokens)``.
        return_z:
            If ``True`` (default), a reparameterized latent sample is returned
            as the third element. Otherwise ``None`` is returned.

        Returns
        -------
        ``(mu, log_sigma, z)`` where ``mu`` and ``log_sigma`` have shape
        ``(batch, z_dim)`` and ``z`` has the same shape when requested.
        """
        state_emb = self.state_embedding(states)
        reward_values = self._normalize_reward_values(rewards)
        reward_emb = self.reward_embedding(reward_values)
        pooled = self.encoder(state_emb, reward_emb)

        mu = self.mu_head(pooled)
        log_sigma = self.log_sigma_head(pooled)

        z = self.reparameterize(mu, log_sigma) if return_z else None
        return mu, log_sigma, z

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Sample ``z = mu + exp(log_sigma) * epsilon`` with ``epsilon ~ N(0, I)``."""
        sigma = torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def decode(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict scalar rewards ``q(eta(s) | s, z)``."""
        return self.decoder(states, z)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: Optional[torch.Tensor] = None,
    ) -> FREOutput:
        """Compute the FRE VAE objective and associated statistics.

        Parameters
        ----------
        encoder_states:
            Shape ``(batch, K, state_dim)`` context states.
        encoder_rewards:
            Shape ``(batch, K)`` context rewards.
        decoder_states:
            Shape ``(batch, K', state_dim)`` decoder states.
        decoder_rewards:
            Optional shape ``(batch, K')`` decoder targets. When omitted,
            the decoder states are used only to compute the latent KL, which
            is useful for encoding-only passes.
        """
        mu, log_sigma, z = self.encode(
            encoder_states, encoder_rewards, return_z=True
        )

        # KL between q(z|c) and N(0, I), averaged over the batch.
        # For a diagonal Gaussian:
        #   KL = -0.5 * sum(1 + 2*log_sigma - mu^2 - sigma^2)
        kl_per_dim = -0.5 * (
            1.0 + 2.0 * log_sigma - mu.pow(2) - (2.0 * log_sigma).exp()
        )
        kl = kl_per_dim.sum(dim=-1).mean(dim=0)

        if decoder_rewards is not None:
            reward_pred = self.decode(decoder_states, z)
            reward_target = self._normalize_reward_values(decoder_rewards)
            # Average reconstruction MSE over decoder tokens and batch.
            recon_mse = ((reward_pred - reward_target) ** 2).mean()
            loss = recon_mse - self.beta * kl
        else:
            # When targets are absent, return a placeholder reconstruction
            # value of zero and keep gradients only through the KL term.
            reward_pred = self.decode(decoder_states, z.detach())
            reward_target = self._normalize_reward_values(
                torch.zeros_like(reward_pred)
            )
            recon_mse = torch.zeros((), device=states_device(encoder_states))
            loss = -self.beta * kl

        return FREOutput(
            loss=loss,
            reconstruction_mse=recon_mse,
            kl=kl,
            z=z,
            mu=mu,
            log_sigma=log_sigma,
            reward_pred=reward_pred,
            reward_target=reward_target,
        )

    def reconstruct_reward(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """Encode context and decode rewards for query states.

        This is a convenience wrapper used by evaluation and visualization.
        """
        _, _, z = self.encode(states, rewards, return_z=True)
        return self.decode(query_states, z)

    # ------------------------------------------------------------------
    # Configuration helper
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any, state_dim: Optional[int] = None) -> "FREVAE":
        """Build a :class:`FREVAE` from a project configuration object.

        ``cfg`` is expected to be a ``Config`` or a dataclass containing an
        ``fre`` section. The helper is deliberately defensive: any missing
        attributes fall back to the defaults used throughout the repository.
        """
        fre_cfg = getattr(cfg, "fre", cfg)

        def get(name: str, default: Any) -> Any:
            value = getattr(fre_cfg, name, default)
            # Nested dataclasses sometimes expose an ``asdict``-like dict.
            if isinstance(value, dict) and name in value:
                return value[name]
            return value

        # State dimension may live under the FRE section or be inferred from
        # a nested data configuration.
        resolved_state_dim = state_dim
        if resolved_state_dim is None:
            resolved_state_dim = get("state_dim", None)
        if resolved_state_dim is None:
            data_cfg = getattr(cfg, "data", None)
            resolved_state_dim = getattr(data_cfg, "state_dim", None)
        if resolved_state_dim is None:
            raise ValueError(
                "FREVAE.from_config requires state_dim either explicitly or "
                "in cfg.fre.state_dim / cfg.data.state_dim."
            )

        return cls(
            state_dim=int(resolved_state_dim),
            z_dim=int(get("z_dim", 64)),
            d_model=int(get("d_model", 128)),
            d_ff=int(get("d_ff", 256)),
            n_heads=int(get("n_heads", 4)),
            n_layers=int(get("n_layers", 2)),
            num_bins=int(get("num_bins", get("reward_num_bins", 128))),
            reward_min=float(get("reward_min", -1.0)),
            reward_max=float(get("reward_max", 1.0)),
            use_linear_reward=bool(get("use_linear_reward", False)),
            dropout=float(get("dropout", 0.0)),
            activation=str(get("activation", "relu")),
            decoder_hidden_dim=int(get("decoder_hidden_dim", 256)),
            decoder_num_hidden=int(get("decoder_num_hidden", 2)),
            beta=float(get("beta", 1.0)),
            normalize_rewards=bool(get("normalize_rewards", True)),
        )

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, z_dim={self.z_dim}, "
            f"d_model={self.d_model}, num_bins={self.num_bins}, "
            f"beta={self.beta}"
        )


def states_device(tensor: torch.Tensor) -> torch.device:
    """Return the device of a tensor, used for scalar placeholder tensors."""
    return tensor.device


__all__ = ["FREVAE", "FREOutput"]
