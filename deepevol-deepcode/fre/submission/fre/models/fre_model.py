"""Functional Reward Encoding (FRE) model.

This module ties together the permutation-invariant transformer encoder
``p_theta(z | s^e_1, eta(s^e_1), ..., s^e_K, eta(s^e_K))`` and the feed-forward
decoder ``q_theta(eta(s^d) | s^d, z)`` and implements the variational lower
bound of the information bottleneck objective used to train them (Equation 6):

.. math::

    I(L^d_eta ; Z) - beta I(L^e_eta ; Z)
    >= E_{eta, L^e, L^d, z ~ p_theta(z|L^e)} [
           sum_{k=1}^{K'} log q_theta(eta(s^d_k) | s^d_k, z)
           - beta D_KL(p_theta(z | L^e) || u(z)) ] + const

with ``u(z)`` the unit Gaussian.  The decoder is a unit-variance Gaussian, so
``log q_theta`` is equivalent (up to a constant) to the negative mean-squared
error between the predicted and true decoder rewards (see §4.1 "Practical
Implementation": "We train both the encoder and decoder networks jointly,
minimizing mean-squared error between the predicted and true rewards under the
decoding states.").

Hyperparameters follow Appendix A Table 3 (batch size 512, K = 32 encoder
reward pairs, K' = 8 decoder reward pairs, beta = 0.01, Adam lr = 1e-4) and the
addendum ("Additional Details on the FRE architecture").
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.models.fre_decoder import FREDecoder, build_mlp
from fre.models.fre_encoder import FREEncoder, kl_divergence_to_unit_gaussian

__all__ = ["FREModel", "fre_elbo_loss", "DEFAULT_BETA"]

#: KL weight from Appendix A Table 3 ("beta KL Weight 0.01").
DEFAULT_BETA: float = 0.01

#: Number of (state, reward) pairs passed to the encoder / decoder (Table 3).
DEFAULT_NUM_ENCODER_PAIRS: int = 32
DEFAULT_NUM_DECODER_PAIRS: int = 8


def fre_elbo_loss(
    reconstruction_loss: torch.Tensor,
    kl: torch.Tensor,
    beta: float = DEFAULT_BETA,
) -> torch.Tensor:
    """Combine the reconstruction term and KL term of Equation 6.

    The variational lower bound is (up to a constant, and negated to obtain a
    loss):

    .. math::
        \\mathcal{L} = -\\sum_{k=1}^{K'} \\log q_\\theta(\\eta(s^d_k) | s^d_k, z)
                       + \\beta D_{KL}(p_\\theta(z|L^e) \\| u(z))

    ``reconstruction_loss`` is the *positive* reconstruction error (the mean
    squared error, which is the negative log-likelihood of a unit-variance
    Gaussian decoder up to a constant).
    """
    return reconstruction_loss + beta * kl


class FREModel(nn.Module):
    """Encoder + decoder reward-function auto-encoder trained with Eq. (6).

    Parameters
    ----------
    state_dim:
        Dimensionality of the (preprocessed) environment state.
    latent_dim:
        Dimensionality of the task latent ``z``; 128 per the addendum.
    state_embed_dim, reward_embed_dim:
        64 + 64 = 128-dimensional token fed to the transformer (addendum).
    num_layers, num_heads, mlp_dim:
        4 pre-norm transformer blocks, 4 attention heads, MLP width 256
        (addendum: residual/attention dim 128, MLP expands to 256 and back).
    decoder_hidden_dims:
        ``[512, 512, 512]`` per Table 3.
    beta:
        KL weight of the information bottleneck; 0.01 per Table 3.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        state_embed_dim: int = 64,
        reward_embed_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        num_reward_bins: int = 32,
        decoder_hidden_dims: Sequence[int] = (512, 512, 512),
        beta: float = DEFAULT_BETA,
        encoder_activation: str = "gelu",
        decoder_activation: str = "relu",
        decoder_use_layer_norm: bool = False,
        normalize_rewards: bool = True,
        dropout: float = 0.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()

        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.normalize_rewards = bool(normalize_rewards)

        self.encoder = FREEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            state_embed_dim=state_embed_dim,
            reward_embed_dim=reward_embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_reward_bins=num_reward_bins,
            dropout=dropout,
            activation=encoder_activation,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            normalize_rewards=normalize_rewards,
        )

        self.decoder = FREDecoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=decoder_activation,
            use_layer_norm=decoder_use_layer_norm,
        )

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------
    @property
    def reward_embedding(self):
        """The learned 32-bin reward embedding table (32 -> 64)."""
        return self.encoder.reward_embedding

    def encode(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample: bool = False,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        """Return the latent ``z`` (posterior mean by default)."""
        return self.encoder(
            encoder_states,
            encoder_rewards,
            mask=mask,
            sample=sample,
            already_normalized=already_normalized,
        )[0]

    def decode(
        self, decoder_states: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Predict ``eta(s^d)`` for each decoder state given the shared ``z``."""
        return self.decoder(decoder_states, z)

    @staticmethod
    def evaluate_reward_function(
        reward_fn: Union[Callable[[torch.Tensor], torch.Tensor], Any],
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a reward function object/callable on a batch of states.

        Accepts either a plain callable ``eta(states) -> rewards`` or an object
        exposing a ``reward(states)`` method (the interface used by the reward
        priors in :mod:`fre.priors`).
        """
        if callable(reward_fn):
            rewards = reward_fn(states)
        elif hasattr(reward_fn, "reward"):
            rewards = reward_fn.reward(states)
        else:  # pragma: no cover - defensive
            raise TypeError(
                "reward_fn must be callable or expose a `.reward(states)` method, "
                f"got {type(reward_fn)!r}"
            )
        if not torch.is_tensor(rewards):  # pragma: no cover - defensive
            rewards = torch.as_tensor(rewards, dtype=states.dtype, device=states.device)
        rewards = rewards.reshape(*states.shape[:-1])
        return rewards

    # ------------------------------------------------------------------
    # Equation 6
    # ------------------------------------------------------------------
    def loss(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
        beta: Optional[float] = None,
        sample: bool = True,
        already_normalized: bool = False,
        return_latent: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Compute the (negated) variational lower bound of Equation 6.

        Shapes
        ------
        encoder_states : ``(B, K, state_dim)``
        encoder_rewards : ``(B, K)``
        decoder_states : ``(B, K', state_dim)``
        decoder_rewards : ``(B, K')``
        encoder_mask, decoder_mask : ``(B, K)`` / ``(B, K')``, ``True`` = valid.

        Returns
        -------
        dict with keys ``loss`` (total objective), ``reconstruction_loss``
        (mean squared error over decoder states, i.e. the negative
        log-likelihood of the unit-variance Gaussian decoder), ``kl`` and
        ``beta``.
        """
        beta = self.beta if beta is None else float(beta)

        # ---- encoder: z ~ p_theta(z | L^e) -----------------------------
        z, mean, log_std, _ = self.encoder(
            encoder_states,
            encoder_rewards,
            mask=encoder_mask,
            sample=sample,
            already_normalized=already_normalized,
        )

        # ---- decoder: q_theta(eta(s^d) | s^d, z) ------------------------
        pred_rewards = self.decoder(decoder_states, z)
        if pred_rewards.dim() == decoder_rewards.dim() + 1:
            pred_rewards = pred_rewards.squeeze(-1)

        reconstruction_loss = self.decoder.mse_loss(
            decoder_states, z, decoder_rewards, mask=decoder_mask
        )

        # ---- KL(p_theta(z | L^e) || u(z)) with u(z) = N(0, I) ----------
        kl = kl_divergence_to_unit_gaussian(mean, log_std)
        kl_mean = kl.mean()

        total = fre_elbo_loss(reconstruction_loss, kl_mean, beta=beta)

        out: Dict[str, torch.Tensor] = {
            "loss": total,
            "reconstruction_loss": reconstruction_loss.detach(),
            "mse": reconstruction_loss.detach(),
            "kl": kl_mean.detach(),
            "beta": torch.as_tensor(beta, device=total.device),
            "pred_reward_mean": pred_rewards.detach().mean(),
            "pred_reward_std": pred_rewards.detach().std(unbiased=False)
            if pred_rewards.numel() > 1
            else torch.zeros((), device=total.device),
            "target_reward_mean": decoder_rewards.detach().mean(),
        }

        if return_latent:
            return out, z
        return out

    def forward(
        self,
        encoder_states: torch.Tensor,
        encoder_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
        decoder_rewards: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Alias for :meth:`loss` (Equation 6 objective)."""
        return self.loss(
            encoder_states,
            encoder_rewards,
            decoder_states,
            decoder_rewards,
            encoder_mask=encoder_mask,
            decoder_mask=decoder_mask,
            beta=beta,
        )

    # ------------------------------------------------------------------
    # sampling helpers (used by the training loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_encoder_states(
        self,
        dataset_states: torch.Tensor,
        batch_size: int,
        num_pairs: int = DEFAULT_NUM_ENCODER_PAIRS,
    ) -> torch.Tensor:
        """Sample ``(B, K)`` states uniformly from the offline dataset.

        §4.1: "K encoder states are sampled uniformly from the offline dataset".
        ``dataset_states`` is the flat ``(N, state_dim)`` array of dataset
        states.
        """
        n = dataset_states.shape[0]
        idx = torch.randint(0, n, (batch_size, num_pairs), device=dataset_states.device)
        return dataset_states[idx]

    @torch.no_grad()
    def sample_decoder_states(
        self,
        dataset_states: torch.Tensor,
        batch_size: int,
        num_pairs: int = DEFAULT_NUM_DECODER_PAIRS,
    ) -> torch.Tensor:
        """Sample disjoint ``(B, K')`` decoder states uniformly from ``D``."""
        return self.sample_encoder_states(dataset_states, batch_size, num_pairs)

    @torch.no_grad()
    def training_batch(
        self,
        dataset_states: torch.Tensor,
        reward_prior: Any,
        batch_size: int,
        num_encoder_pairs: int = DEFAULT_NUM_ENCODER_PAIRS,
        num_decoder_pairs: int = DEFAULT_NUM_DECODER_PAIRS,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build one training batch of Equation 6.

        1. sample one reward function ``eta ~ p(eta)`` per batch element,
        2. sample K encoder states and K' decoder states uniformly from ``D``
           (the reward prior may override the sampling, e.g. for goal-reaching
           rewards whose goal is drawn from the HER distribution),
        3. label the states with ``eta``.

        Returns a dict of tensors ready for :meth:`loss`.
        """
        if device is None:
            device = dataset_states.device
        dataset_states = dataset_states.to(device)

        batch = reward_prior.sample_batch(
            dataset_states,
            batch_size=batch_size,
            num_encoder_pairs=num_encoder_pairs,
            num_decoder_pairs=num_decoder_pairs,
            device=device,
        )

        return {
            "encoder_states": batch["encoder_states"],
            "encoder_rewards": batch["encoder_rewards"],
            "decoder_states": batch["decoder_states"],
            "decoder_rewards": batch["decoder_rewards"],
            "encoder_mask": batch.get("encoder_mask"),
            "decoder_mask": batch.get("decoder_mask"),
        }

    @torch.no_grad()
    def encode_batch(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample: bool = False,
    ) -> torch.Tensor:
        """Encode a batch of ``(s, eta(s))`` sets and return ``z``.

        Used at evaluation time: the test task's reward function is queried on
        ``K = 32`` states and the posterior mean is used as ``z``.
        """
        return self.encode(states, rewards, mask=mask, sample=sample)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"state_dim={self.state_dim}, latent_dim={self.latent_dim}, "
            f"beta={self.beta}"
        )
