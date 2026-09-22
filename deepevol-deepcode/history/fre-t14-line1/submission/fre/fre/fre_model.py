"""Functional Reward Encoding (FRE) joint model: encoder + decoder + Eq. (6) loss.

This module implements the core FRE training objective from the paper
"Zero-Shot Reinforcement Learning via Functional Reward Encodings" (ICML 2024).

Paper references (verbatim):

    Section 4.1 -- Functional Reward Encoding
    ---------------------------------------------------------------------------
    "any reward function eta can be represented as a lookup table over the set of
     state-reward pairs:
         L_eta := { (s^e, eta(s^e)) : s^e in D }                                  (1)
     ... This can be formulated as the following information bottleneck objective
     over the structure of L_eta^e -> Z -> L_eta^d:
         I(L_eta^d ; Z) - beta I(L_eta^e ; Z)                                     (2)
     ... we derive its variational lower bound as follows (Alemi et al., 2016):
         I(L_eta^d ; Z) - beta I(L_eta^e ; Z)
           >= E_{eta, L_eta^e, L_eta^d, z ~ p_theta(z | L_eta^e)}[
                  sum_{k=1}^{K'} log q_theta(eta(s_k^d) | s_k^d, z)
                  - beta D_KL(p_theta(z | L_eta^e) || u(z)) ] + (const)          (6)
     where ... u(z) is an uninformative prior over z, which we define as the unit
     Gaussian."

    Practical Implementation (Section 4.1)
    ---------------------------------------------------------------------------
    "The encoder p_theta(z | .) is implemented as a permutation-invariant
     transformer ... The decoder q_theta(eta(s) | s, z) is implemented as a
     feedforward neural network. Crucially, the states sampled for decoding are
     different than those used for encoding. ... We train both the encoder and
     decoder networks jointly, minimizing mean-squared error between the predicted
     and true rewards under the decoding states."

    Algorithm 1 (Section 4.3)
    ---------------------------------------------------------------------------
        Sample reward function eta ~ p(eta)
        Sample K states for encoder {s_k^e} ~ D
        Sample K' states for decoder {s_k^d} ~ D
        Train FRE by maximizing Equation (6)

The practical implementation maximises Eq. (6) which, for a Gaussian decoder
q_theta(eta(s^d) | s^d, z) = N(mu_theta(s^d, z), 1) with unit variance, reduces (up
to additive constants) to minimising

    L = (1/K') * sum_k ( mu_theta(s_k^d, z) - eta(s_k^d) )^2
        + beta * KL( p_theta(z | L_eta^e) || N(0, I) )

with beta = 0.01 (Appendix A, Table 3).  We therefore expose the loss as a
minimisation objective while documenting the exact equivalence to Eq. (6).
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from fre.fre.decoder import Decoder, decoder_mse_loss
from fre.fre.encoder import Encoder, diagonal_gaussian_kl

__all__ = ["FREModel", "FRELoss", "fre_loss"]


class FRELoss(NamedTuple):
    """Container for the decomposable FRE training objective.

    Attributes:
        loss: Total objective being minimised, ``reconstruction + beta * kl``.
        reconstruction: Mean-squared error between predicted and true rewards on
            the decoder states (the ``-E[sum_k log q_theta(...)]`` term of Eq. 6
            up to an additive constant).
        kl: ``D_KL(p_theta(z | L_eta^e) || N(0, I))`` term of Eq. 6.
        weighted_kl: ``beta * kl``.
        z: The latent task vector sampled from the encoder posterior.
        pred: Decoder predictions of shape ``(batch, K_prime)``.
        target: Ground-truth rewards on the decoder states, shape ``(batch, K_prime)``.
    """

    loss: Tensor
    reconstruction: Tensor
    kl: Tensor
    weighted_kl: Tensor
    z: Tensor
    pred: Tensor
    target: Tensor

    # Convenience aliases matching Eq. (6) terminology.
    @property
    def mse(self) -> Tensor:
        """Alias for :attr:`reconstruction`."""
        return self.reconstruction

    def as_dict(self) -> Dict[str, Tensor]:
        """Return the scalar components as a plain dict (useful for logging)."""
        return {
            "loss": self.loss,
            "reconstruction": self.reconstruction,
            "mse": self.reconstruction,
            "kl": self.kl,
            "weighted_kl": self.weighted_kl,
        }


class FREModel(nn.Module):
    """Joint FRE encoder + decoder trained with the Eq. (6) objective.

    The model maps a context set of ``K`` ``(state, reward)`` pairs through a
    permutation-invariant transformer encoder to a 128-dim Gaussian latent ``z``,
    and decodes the reward of ``K'`` *disjoint* states conditioned on ``z``.

    Args:
        encoder: The FRE encoder ``p_theta(z | L_eta^e)``.
        decoder: The FRE decoder ``q_theta(eta(s) | s, z)``.
        beta_kl: Compression strength ``beta`` of Eq. (6). Default 0.01
            (Appendix A, Table 3).
        free_bits: Optional lower bound on the per-sample KL (in nats) to mitigate
            posterior collapse. The paper does not mention this; default 0.0
            (disabled) so the implementation matches Eq. (6) exactly.
        use_rsample: If ``True`` (default) the latent is sampled with the
            reparameterised ``rsample()`` so gradients flow into the encoder; the
            paper trains the encoder and decoder jointly, which requires this.
        mse_reduction: Reduction used for the reconstruction term; ``"mean"``
            follows the paper ("minimizing mean-squared error between the
            predicted and true rewards").
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        beta_kl: float = 0.01,
        free_bits: float = 0.0,
        use_rsample: bool = True,
        mse_reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta_kl = float(beta_kl)
        self.free_bits = float(free_bits)
        self.use_rsample = bool(use_rsample)
        self.mse_reduction = str(mse_reduction)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config, state_dim: int, **overrides) -> "FREModel":
        """Build an :class:`FREModel` from a ``fre.config.default.Config``.

        Args:
            config: Config-like object exposing the FRE hyperparameters.
            state_dim: Dimensionality of the environment states.
            **overrides: Forwarded to the encoder/decoder ``from_config`` calls via
                the ``encoder_kwargs`` / ``decoder_kwargs`` dicts, plus ``beta_kl``.
        """
        encoder_kwargs = dict(overrides.pop("encoder_kwargs", {}) or {})
        decoder_kwargs = dict(overrides.pop("decoder_kwargs", {}) or {})
        beta_kl = overrides.pop("beta_kl", getattr(config, "beta_kl", 0.01))
        free_bits = overrides.pop("free_bits", getattr(config, "free_bits", 0.0))
        use_rsample = overrides.pop("use_rsample", True)
        mse_reduction = overrides.pop("mse_reduction", "mean")

        # Any remaining keyword arguments are treated as shared architecture
        # overrides and forwarded to both sub-networks (unknown keys ignored).
        encoder = Encoder.from_config(config, state_dim, **encoder_kwargs, **dict(overrides))
        decoder = Decoder.from_config(config, state_dim, **decoder_kwargs)
        return cls(
            encoder=encoder,
            decoder=decoder,
            beta_kl=beta_kl,
            free_bits=free_bits,
            use_rsample=use_rsample,
            mse_reduction=mse_reduction,
        )

    # ------------------------------------------------------------------
    # Shapes / properties
    # ------------------------------------------------------------------
    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent task representation ``z`` (128)."""
        return int(getattr(self.encoder, "latent_dim", 128))

    @property
    def num_encoder_samples(self) -> int:
        """``K`` -- number of ``(s, eta(s))`` pairs used to encode ``z``."""
        return int(getattr(self.encoder, "num_encoder_samples", 32))

    @property
    def num_decoder_samples(self) -> int:
        """``K'`` -- number of states used to decode/reconstruct rewards."""
        return int(getattr(self.decoder, "num_decoder_samples", 8))

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward(
        self,
        encoder_states: Tensor,
        encoder_rewards: Tensor,
        decoder_states: Tensor,
        decoder_rewards: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> FRELoss:
        """Compute the Eq. (6) objective.

        Args:
            encoder_states: ``(B, K, state_dim)`` states ``s^e`` for the encoder.
            encoder_rewards: ``(B, K)`` rewards ``eta(s^e)`` for the encoder.
            decoder_states: ``(B, K', state_dim)`` disjoint states ``s^d``.
            decoder_rewards: ``(B, K')`` true rewards ``eta(s^d)`` on the decoder
                states. Required to compute the objective.
            key_padding_mask: Optional ``(B, K)`` boolean mask where ``True`` marks
                tokens to ignore (needed when a sampled context has fewer than ``K``
                entries).

        Returns:
            :class:`FRELoss` with the total objective and its components.
        """
        return self.loss(
            encoder_states=encoder_states,
            encoder_rewards=encoder_rewards,
            decoder_states=decoder_states,
            decoder_rewards=decoder_rewards,
            key_padding_mask=key_padding_mask,
        )

    def encode(
        self,
        encoder_states: Tensor,
        encoder_rewards: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        sample: bool = True,
    ) -> Tensor:
        """Encode ``K`` reward-annotated states into a latent task vector ``z``.

        Args:
            encoder_states: ``(B, K, state_dim)`` (or ``(K, state_dim)``).
            encoder_rewards: ``(B, K)`` (or ``(K,)``).
            key_padding_mask: Optional ``(B, K)`` mask (``True`` = ignore).
            sample: If ``True`` sample from ``p_theta(z | .)``; otherwise return the
                posterior mean (deterministic encoding, typically used at eval).

        Returns:
            ``(B, latent_dim)`` latent task representation.
        """
        posterior = self.encoder(encoder_states, encoder_rewards, key_padding_mask=key_padding_mask)
        if sample:
            return posterior.rsample() if self.use_rsample else posterior.sample()
        return posterior.mean

    def decode(self, decoder_states: Tensor, z: Tensor) -> Tensor:
        """Predict rewards ``eta(s^d)`` for ``K'`` states given the task latent ``z``.

        Args:
            decoder_states: ``(B, K', state_dim)`` (or ``(K', state_dim)``).
            z: ``(B, latent_dim)`` (or ``(latent_dim,)``).

        Returns:
            ``(B, K')`` predicted rewards.
        """
        return self.decoder(decoder_states, z)

    def loss(
        self,
        encoder_states: Tensor,
        encoder_rewards: Tensor,
        decoder_states: Tensor,
        decoder_rewards: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        beta_kl: Optional[float] = None,
    ) -> FRELoss:
        """Evaluate Eq. (6): reconstruction (MSE) + ``beta`` * KL to the unit Gaussian.

        Following Section 4.1 ("minimizing mean-squared error between the predicted
        and true rewards under the decoding states"), the negative log-likelihood
        term ``-E[sum_k log q_theta(eta(s_k^d) | s_k^d, z)]`` of Eq. (6) is
        implemented as the MSE between the decoder's prediction and the true reward
        (equivalently a unit-variance Gaussian decoder).

        The KL term of Eq. (6) uses ``u(z) = N(0, I)``.
        """
        if decoder_rewards is None:
            raise ValueError("`decoder_rewards` (eta(s^d)) is required to evaluate Eq. (6).")

        beta = float(self.beta_kl if beta_kl is None else beta_kl)

        # --- Encoder: z ~ p_theta(z | s^e_1..K, eta(s^e_1..K)) -------------------
        posterior = self.encoder(encoder_states, encoder_rewards, key_padding_mask=key_padding_mask)
        z = posterior.rsample() if self.use_rsample else posterior.sample()

        # --- Decoder: q_theta(eta(s^d) | s^d, z), independent per decoding state -
        pred = self.decoder(decoder_states, z)
        reconstruction = decoder_mse_loss(pred, decoder_rewards, reduction=self.mse_reduction)

        # --- KL(p_theta(z | L_eta^e) || u(z)) with u(z) = N(0, I) ----------------
        kl = diagonal_gaussian_kl(posterior.loc, posterior.scale)
        if self.free_bits > 0.0:
            kl = torch.clamp(kl, min=self.free_bits)
        weighted_kl = beta * kl

        total = reconstruction + weighted_kl
        target = decoder_rewards.reshape(pred.shape) if decoder_rewards.shape != pred.shape else decoder_rewards
        return FRELoss(
            loss=total,
            reconstruction=reconstruction,
            kl=kl,
            weighted_kl=weighted_kl,
            z=z,
            pred=pred,
            target=target,
        )

    # ------------------------------------------------------------------
    # Extra evaluation utilities (not part of the training objective)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruction_mse(
        self,
        encoder_states: Tensor,
        encoder_rewards: Tensor,
        decoder_states: Tensor,
        decoder_rewards: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> float:
        """Held-out reward reconstruction MSE using the posterior mean for ``z``.

        This mirrors the paper's qualitative check that ``z`` is maximally
        informative about ``L_eta``: the decoder should be able to predict the
        reward of states that were *not* shown to the encoder.
        """
        self.eval()
        posterior = self.encoder(encoder_states, encoder_rewards, key_padding_mask=key_padding_mask)
        pred = self.decoder(decoder_states, posterior.mean)
        target = decoder_rewards.reshape(pred.shape) if decoder_rewards.shape != pred.shape else decoder_rewards
        return float(torch.mean((pred - target) ** 2).item())

    @torch.no_grad()
    def latent_statistics(
        self,
        encoder_states: Tensor,
        encoder_rewards: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Diagnostics on the posterior: mean KL, mean/std of ``z``, mean log-std."""
        posterior = self.encoder(encoder_states, encoder_rewards, key_padding_mask=key_padding_mask)
        kl = diagonal_gaussian_kl(posterior.loc, posterior.scale)
        z = posterior.mean
        return {
            "kl": float(kl.item()),
            "z_mean": float(z.mean().item()),
            "z_std": float(z.std().item()),
            "mean_log_std": float(posterior.loc.new_tensor(posterior.scale.log()).mean().item()),
        }

    def trainable_parameters(self):
        """Iterate over parameters with ``requires_grad`` (used for the optimiser)."""
        return (p for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, K={self.num_encoder_samples}, "
            f"K'={self.num_decoder_samples}, beta_kl={self.beta_kl}, "
            f"use_rsample={self.use_rsample}"
        )


def fre_loss(
    model: FREModel,
    encoder_states: Tensor,
    encoder_rewards: Tensor,
    decoder_states: Tensor,
    decoder_rewards: Tensor,
    key_padding_mask: Optional[Tensor] = None,
    beta_kl: Optional[float] = None,
) -> FRELoss:
    """Functional wrapper around :meth:`FREModel.loss` (Eq. 6)."""
    return model.loss(
        encoder_states=encoder_states,
        encoder_rewards=encoder_rewards,
        decoder_states=decoder_states,
        decoder_rewards=decoder_rewards,
        key_padding_mask=key_padding_mask,
        beta_kl=beta_kl,
    )
