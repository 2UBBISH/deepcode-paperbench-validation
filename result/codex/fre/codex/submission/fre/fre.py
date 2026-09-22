"""The Functional Reward Encoding (FRE) module.

Section 4.1 formulates FRE as an information-bottleneck objective over the
chain ``L_eta^e -> Z -> L_eta^d``::

    I(L_eta^d ; Z) - beta I(L_eta^e ; Z)

whose variational lower bound (Equation 6 of the paper) is::

    E_{eta, L^e, L^d, z ~ p_theta(z | L^e)} [
        sum_{k=1}^{K'} log q_theta(eta(s_k^d) | s_k^d, z)
        - beta * KL( p_theta(z | L^e) || u(z) )
    ] + const

with ``u(z) = N(0, I)`` the uninformative prior over ``z``.  ``q_theta`` is a
Gaussian with fixed unit variance, so maximising the log-likelihood is
equivalent to minimising the mean-squared error between predicted and true
rewards -- which is how the paper describes the practical implementation.

This is a denoising-auto-encoder-like objective over reward-annotated state
sets, except the encoder is *probabilistic* with an explicit information
penalty (unlike denoising auto-encoders and neural processes, which use
deterministic encoders).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fre.decoder import RewardDecoder
from fre.encoder import TransformerEncoder, gaussian_kl


@dataclass
class FREConfig:
    """Hyperparameters of the FRE encoder/decoder (Appendix A)."""

    state_dim: int
    z_dim: int = 128
    # encoder
    state_embed_dim: int = 64
    reward_embed_dim: int = 64
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    encoder_mlp_dim: int = 256
    num_reward_bins: int = 32
    # decoder
    decoder_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    # objective
    beta: float = 0.01
    # sample counts
    num_encode_pairs: int = 32
    num_decode_pairs: int = 8


class FRE(nn.Module):
    """Joint encoder + decoder module trained with the information bottleneck."""

    def __init__(self, config: FREConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(
            state_dim=config.state_dim,
            z_dim=config.z_dim,
            state_embed_dim=config.state_embed_dim,
            reward_embed_dim=config.reward_embed_dim,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            mlp_dim=config.encoder_mlp_dim,
            num_reward_bins=config.num_reward_bins,
        )
        self.decoder = RewardDecoder(
            state_dim=config.state_dim,
            z_dim=config.z_dim,
            hidden_dims=config.decoder_hidden_dims,
        )

    # -- encoding / decoding -------------------------------------------------------
    def encode(
        self,
        states: torch.Tensor,
        reward_bins: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        return self.encoder.encode(states, reward_bins, sample=sample)

    def decode(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(states, z)

    def encode_rewards(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        sample: bool = True,
        reward_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """Convenience wrapper: discretise raw rewards then encode.

        ``reward_range`` must match the analytic output range of the reward
        family so that the bin indices line up with the embedding table entries
        that were learned during unsupervised pre-training (goal-reaching
        rewards live in ``[-1, 0]``, velocity rewards in ``[0, 1]``, and random
        linear / MLP rewards in ``[-1, 1]``).
        """
        from fre.reward_functions import discretize_reward

        bins = discretize_reward(
            rewards, self.config.num_reward_bins, r_min=reward_range[0], r_max=reward_range[1]
        )
        return self.encode(states, bins, sample=sample)

    # -- training objective --------------------------------------------------------
    def loss(
        self,
        enc_states: torch.Tensor,
        enc_reward_bins: torch.Tensor,
        dec_states: torch.Tensor,
        dec_rewards: torch.Tensor,
        beta: Optional[float] = None,
        return_parts: bool = False,
    ):
        """Compute the negative of Equation 6 (a loss to minimise).

        Args:
            enc_states: ``(B, K, state_dim)`` encoding states.
            enc_reward_bins: ``(B, K)`` discretised rewards for encoding states.
            dec_states: ``(B, K', state_dim)`` decoding states.
            dec_rewards: ``(B, K')`` true rewards for decoding states.
            beta: KL weight (defaults to ``config.beta``).

        Returns:
            The scalar loss, or ``(loss, recon, kl)`` if ``return_parts``.
        """
        beta = self.config.beta if beta is None else float(beta)
        mean, log_std = self.encoder(enc_states, enc_reward_bins)
        std = log_std.exp()
        z = mean + std * torch.randn_like(std)

        pred = self.decoder(dec_states, z)
        recon = torch.nn.functional.mse_loss(pred, dec_rewards)
        kl = gaussian_kl(mean, log_std)
        loss = recon + beta * kl
        if return_parts:
            return loss, recon.detach(), kl.detach()
        return loss

    @torch.no_grad()
    def latent_for(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        sample: bool = False,
        reward_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        """Compute ``z`` for evaluation-time reward-annotated samples.

        At test time a downstream task is specified by a small number of
        ``(s, eta(s))`` samples (32 in all of the paper's experiments).  The
        encoder maps that context set to ``z``; we use the posterior mean for
        evaluation, which is the standard deterministic choice.
        """
        return self.encode_rewards(states, rewards, sample=sample, reward_range=reward_range)
