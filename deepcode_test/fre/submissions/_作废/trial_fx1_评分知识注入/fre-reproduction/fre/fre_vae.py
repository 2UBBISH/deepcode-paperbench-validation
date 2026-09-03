"""FRE variational autoencoder: encoder/decoder pair and training helpers.

This module ties together the reward tokenizer, permutation-invariant
transformer encoder, and MLP reward decoder from the Functional Reward
Encoding paper.  It is intentionally self-contained so that phase-1 training
can be run without any RL components.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import RewardDecoder
from .encoder import FREEncoder
from .reward_embedding import RewardEmbedding


class FREVAE(nn.Module):
    """Variational autoencoder over state-reward pairs.

    Parameters
    ----------
    state_dim:
        Dimensionality of raw environment states.
    latent_dim:
        Size of the learned reward-function embedding ``z``.
    d_model:
        Transformer token/model dimension.
    nhead:
        Number of transformer attention heads.
    num_layers:
        Number of transformer encoder layers.
    reward_bins:
        Number of reward discretization bins used by :class:`RewardEmbedding`.
    embedding_dim:
        Learned reward-bin embedding size.
    decoder_hidden:
        Hidden widths of the reward decoder MLP.
    beta:
        KL-weight in the VAE objective.
    device:
        Torch device on which all modules are placed.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        reward_bins: int = 64,
        embedding_dim: int = 64,
        decoder_hidden: Tuple[int, ...] = (256, 256),
        beta: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.beta = beta
        self.device = torch.device(device)

        self.reward_embedding = RewardEmbedding(
            state_dim=state_dim,
            num_bins=reward_bins,
            embedding_dim=embedding_dim,
            state_proj_dim=192,
            token_dim=d_model,
        )
        self.encoder = FREEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
        )
        self.decoder = RewardDecoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden,
            activation="relu",
        )

        self.to(self.device)

    # ------------------------------------------------------------------
    # Encoding / decoding primitives
    # ------------------------------------------------------------------
    def encode(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode state-reward pairs into ``(mu, logvar, z)``.

        ``states`` is expected to have shape ``(..., state_dim)`` and
        ``rewards`` to have matching leading dimensions.  The returned tensors
        have shape ``(..., latent_dim)``.
        """
        tokens = self.reward_embedding(states, rewards)
        return self.encoder(tokens, mask=mask)

    def encode_reward_fn(
        self,
        reward_fn: Any,
        states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a reward function on states and encode the resulting pairs."""
        rewards = reward_fn(states)
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=states.device)
        return self.encode(states, rewards, mask=mask)

    def decode_reward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict scalar rewards for states conditioned on latent codes ``z``."""
        pred = self.decoder(states, z)
        # Decoder may keep a trailing singleton dimension; remove it when the
        # target/reward tensor does not have one.
        if pred.dim() > 1 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return pred

    # ------------------------------------------------------------------
    # VAE objective
    # ------------------------------------------------------------------
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(mu, sigma^2) || N(0, I)) averaged over the latent dimension."""
        return 0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the full VAE forward pass.

        Parameters
        ----------
        encoder_states:
            States used as the encoder context, shape ``(..., K, state_dim)``.
        encoder_rewards:
            Rewards corresponding to ``encoder_states``, shape ``(..., K)``.
        decoder_states:
            States whose rewards should be reconstructed, shape
            ``(..., K', state_dim)``.
        decoder_rewards:
            Optional ground-truth rewards for ``decoder_states``.  If provided,
            the returned dictionary contains ``loss``, ``recon_loss``, and
            ``kl_loss``.

        Returns
        -------
        Dictionary with keys ``mu``, ``logvar``, ``z``, ``decoder_rewards`` and,
        when ``decoder_rewards`` is supplied, the scalar VAE losses.
        """
        mu, logvar, z = self.encode(encoder_states, encoder_rewards, mask=mask)
        pred = self.decode_reward(decoder_states, z)

        out: Dict[str, torch.Tensor] = {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "decoder_rewards": pred,
        }

        if decoder_rewards is not None:
            if pred.shape != decoder_rewards.shape:
                pred_for_loss = pred.reshape_as(decoder_rewards)
            else:
                pred_for_loss = pred

            recon_loss = F.mse_loss(pred_for_loss, decoder_rewards)
            kl = self.kl_divergence(mu, logvar).mean()
            loss = recon_loss + self.beta * kl

            out["recon_loss"] = recon_loss
            out["kl_loss"] = kl
            out["loss"] = loss

        return out

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def configure_optimizer(self, lr: float = 1e-4) -> torch.optim.Adam:
        """Return an Adam optimizer over all VAE parameters."""
        return torch.optim.Adam(self.parameters(), lr=lr)

    def training_step(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one optimizer update and return the computed metrics."""
        optimizer.zero_grad(set_to_none=True)
        out = self(
            encoder_states=encoder_states,
            encoder_rewards=encoder_rewards,
            decoder_states=decoder_states,
            decoder_rewards=decoder_rewards,
            mask=mask,
        )
        out["loss"].backward()
        optimizer.step()
        return {k: v.detach() for k, v in out.items()}

    def encode_latent_for_task(
        self,
        reward_fn: Any,
        states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the mode (mu) of the latent posterior for a reward function.

        This is the standard inference path used by the downstream IQL agent:
        evaluate the task reward on a set of states and encode the resulting
        pairs into a single latent vector.
        """
        mu, _, z = self.encode_reward_fn(reward_fn, states, mask=mask)
        # During zero-shot evaluation the paper encodes with the frozen encoder
        # and conditions on the sampled latent.  Returning z (rather than mu)
        # preserves the stochastic VAE formulation; scripts can also call
        # ``encode_reward_fn`` to obtain ``mu`` when deterministic encoding is
        # preferred.
        return z
