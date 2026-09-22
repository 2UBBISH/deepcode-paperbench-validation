"""Info-bottleneck / VAE training objective for Functional Reward Encodings (FRE).

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings".

Objective (Eq. 6 of the paper)::

    L(theta) = E_{eta ~ p(eta)} E_{s_d ~ rho} [ (eta(s_d) - q_theta(s_d, z))^2 ]
               + beta * KL( q_theta(z | context) || N(0, I) )

where

  * ``context`` is a set of ``K = 32`` (state, reward) pairs sampled from the
    reward function ``eta`` (see :mod:`fre.rewards`),
  * ``s_d`` are ``K' = 8`` *decoder* states sampled from the offline dataset
    (disjoint from the encoder context states),
  * ``z`` is a 128-dim latent task embedding sampled with the reparameterization
    trick from the encoder posterior (see :mod:`fre.encoder`),
  * ``q_theta(s, z)`` is the decoder MLP (see :mod:`fre.decoder`),
  * the reward-prediction log-likelihood is implemented as a mean-squared error
    (Gaussian likelihood with fixed unit variance), and
  * ``beta = 0.01``.

This module deliberately keeps the loss self-contained: it only needs the
``(pred_reward, true_reward, mu, log_sigma)`` tensors produced by
:class:`fre.decoder.FREModel`, so it can be unit tested without an environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DEFAULT_BETA",
    "kl_divergence",
    "gaussian_nll",
    "reward_prediction_loss",
    "info_bottleneck_loss",
    "VAELoss",
    "LossOutput",
]


# Default from the paper's experiments (Eq. 6, Sec 4.1).
DEFAULT_BETA: float = 0.01

# Numerical guards applied to log_sigma before use.
_LOG_SIGMA_MIN: float = -10.0
_LOG_SIGMA_MAX: float = 5.0


def _match_shape(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten / broadcast ``pred`` and ``target`` to a common shape."""
    if pred.shape != target.shape:
        # Allow e.g. pred (B, 1) vs target (B,) and (B, K') vs (B, 1, K').
        target = target.reshape(pred.shape) if target.numel() == pred.numel() else target
        pred = pred.reshape(target.shape) if pred.numel() == target.numel() else pred
        if pred.shape != target.shape:
            pred, target = torch.broadcast_tensors(pred, target)
    return pred, target


def kl_divergence(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ) summed over the last (latent) dimension.

    Uses the closed form

        KL = -0.5 * sum_j ( 1 + log_sigma_j - mu_j^2 - exp(log_sigma_j) )

    Args:
        mu: ``(..., latent_dim)`` posterior means.
        log_sigma: ``(..., latent_dim)`` posterior log standard deviations.

    Returns:
        Tensor of shape ``(...)`` containing the per-sample KL divergence
        (already summed over the latent dimension).
    """
    log_sigma = torch.clamp(log_sigma, _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
    kl = -0.5 * (1.0 + log_sigma - mu.pow(2) - log_sigma.exp())
    return kl.sum(dim=-1)


def gaussian_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log-likelihood of ``target`` under ``N(pred, exp(2*log_sigma))``.

    When ``log_sigma`` is ``None`` (the setting used in the paper) this reduces
    to the mean-squared error up to a constant, which is what
    :func:`reward_prediction_loss` returns.
    """
    pred, target = _match_shape(pred, target)
    if log_sigma is None:
        return F.mse_loss(pred, target)
    log_sigma = torch.clamp(log_sigma, _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
    log_sigma = log_sigma.reshape(pred.shape) if log_sigma.numel() == pred.numel() else log_sigma
    return (0.5 * (pred - target).pow(2) * torch.exp(-2.0 * log_sigma) + log_sigma).mean()


def reward_prediction_loss(
    pred_reward: torch.Tensor,
    true_reward: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reward-prediction term of Eq. 6 (MSE, per the paper)."""
    pred_reward, true_reward = _match_shape(pred_reward, true_reward)
    return F.mse_loss(pred_reward, true_reward, reduction=reduction)


@dataclass
class LossOutput:
    """Structured result of the FRE Phase-1 objective (useful for logging)."""

    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    beta: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def __iter__(self):
        # Convenient unpacking: ``total, recon, kl, metrics = out``
        return iter((self.total, self.recon, self.kl, self.metrics))

    def as_dict(self) -> Dict[str, float]:
        d = {
            "loss": float(self.total.detach().cpu()),
            "recon_loss": float(self.recon.detach().cpu()),
            "kl_loss": float(self.kl.detach().cpu()),
            "beta": float(self.beta),
        }
        d.update(self.metrics)
        return d


def info_bottleneck_loss(
    pred_reward: torch.Tensor,
    true_reward: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    beta: float = DEFAULT_BETA,
    recon_coeff: float = 1.0,
    free_bits: float = 0.0,
    return_metrics: bool = False,
):
    """Compute Eq. 6: ``recon_coeff * MSE + beta * KL``.

    Args:
        pred_reward: ``(B, K')`` (or any shape) decoder reward predictions.
        true_reward: same shape as ``pred_reward``; ``eta(s_d)``.
        mu: ``(B, latent_dim)`` posterior mean from the encoder.
        log_sigma: ``(B, latent_dim)`` posterior log-std from the encoder.
        beta: KL weight (paper: ``0.01``).
        recon_coeff: Weight on the reward-prediction term (paper: ``1.0``).
        free_bits: Optional per-dimension KL floor (in nats per latent dim). A
            value of ``0`` disables the floor and reproduces the paper exactly.
        return_metrics: If ``True`` returns ``(loss, metrics_dict)`` instead of a
            :class:`LossOutput`.

    Returns:
        :class:`LossOutput` (or ``(loss, metrics)`` tuple when
        ``return_metrics=True``).
    """
    recon = reward_prediction_loss(pred_reward, true_reward)

    # Per-sample KL -> mean over the batch. ``mu`` may be flattened with extra
    # leading dims when multiple context samples are drawn per reward function.
    mu_flat = mu.reshape(-1, mu.shape[-1])
    log_sigma_flat = log_sigma.reshape(-1, log_sigma.shape[-1])

    per_sample_kl = kl_divergence(mu_flat, log_sigma_flat)
    if free_bits > 0.0:
        # Free-bits (Kingma et al.): clamp each *dimension* to at least free_bits.
        kl_dims = -0.5 * (
            1.0 + torch.clamp(log_sigma_flat, _LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
            - mu_flat.pow(2)
            - torch.clamp(log_sigma_flat, _LOG_SIGMA_MIN, _LOG_SIGMA_MAX).exp()
        )
        per_sample_kl = torch.clamp(kl_dims, min=free_bits).sum(dim=-1)

    kl = per_sample_kl.mean()

    total = recon_coeff * recon + beta * kl

    metrics = {
        "recon_loss": float(recon.detach().cpu()),
        "kl_loss": float(kl.detach().cpu()),
        "kl_per_dim": float(kl.detach().cpu()) / max(1, mu_flat.shape[-1]),
        "reward_rmse": float(torch.sqrt(torch.clamp(recon.detach(), min=0.0)).cpu()),
        "log_sigma_mean": float(log_sigma_flat.detach().mean().cpu()),
        "z_std_mean": float(torch.exp(log_sigma_flat.detach()).mean().cpu()),
    }

    out = LossOutput(total=total, recon=recon, kl=kl, beta=beta, metrics=metrics)
    if return_metrics:
        return total, out.as_dict()
    return out


class VAELoss(nn.Module):
    """``nn.Module`` wrapper around :func:`info_bottleneck_loss`.

    Provides optional KL warm-up / beta annealing over training steps, which is
    sometimes helpful for stability early in Phase-1 training. With
    ``beta=DEFAULT_BETA``, ``warmup_steps=0`` and ``free_bits=0`` this is exactly
    the paper's objective.
    """

    def __init__(
        self,
        beta: float = DEFAULT_BETA,
        recon_coeff: float = 1.0,
        free_bits: float = 0.0,
        warmup_steps: int = 0,
        beta_start: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.recon_coeff = float(recon_coeff)
        self.free_bits = float(free_bits)
        self.warmup_steps = int(warmup_steps)
        self.beta_start = float(beta_start)

    def current_beta(self, step: Optional[int] = None) -> float:
        """Beta value for the given optimizer step (linear warm-up if enabled)."""
        if not self.warmup_steps or step is None:
            return self.beta
        frac = min(1.0, max(0.0, float(step) / float(self.warmup_steps)))
        return self.beta_start + frac * (self.beta - self.beta_start)

    def forward(
        self,
        pred_reward: torch.Tensor,
        true_reward: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        step: Optional[int] = None,
        return_metrics: bool = False,
    ):
        return info_bottleneck_loss(
            pred_reward=pred_reward,
            true_reward=true_reward,
            mu=mu,
            log_sigma=log_sigma,
            beta=self.current_beta(step),
            recon_coeff=self.recon_coeff,
            free_bits=self.free_bits,
            return_metrics=return_metrics,
        )

    def extra_repr(self) -> str:
        return (
            f"beta={self.beta}, recon_coeff={self.recon_coeff}, "
            f"free_bits={self.free_bits}, warmup_steps={self.warmup_steps}"
        )


def fre_phase1_loss(
    encoder: nn.Module,
    decoder: nn.Module,
    context_states: torch.Tensor,
    context_rewards: torch.Tensor,
    decoder_states: torch.Tensor,
    decoder_rewards: torch.Tensor,
    beta: float = DEFAULT_BETA,
    use_mean_latent: bool = False,
    return_metrics: bool = False,
):
    """End-to-end helper: encode a context, decode decoder states, compute Eq. 6.

    Args:
        encoder: :class:`fre.encoder.FREEncoder` (or compatible module exposing
            ``encode(states, rewards, use_mean=...)``).
        decoder: :class:`fre.decoder.FREDecoder` (or compatible module exposing
            ``predict_for_context(states, z)``).
        context_states: ``(B, K, state_dim)`` ensemble of context states.
        context_rewards: ``(B, K)`` rewards for the context states.
        decoder_states: ``(B, K', state_dim)`` held-out decoder states.
        decoder_rewards: ``(B, K')`` ground-truth rewards ``eta(s_d)``.
        beta: KL weight.
        use_mean_latent: If ``True`` uses the posterior mean instead of a sample
            (deterministic; handy for evaluation / sanity tests).
        return_metrics: If ``True`` returns ``(loss, metrics)``.

    Returns:
        :class:`LossOutput` or ``(loss, metrics)``.
    """
    if use_mean_latent:
        z = encoder.encode(context_states, context_rewards, use_mean=True)
        with torch.no_grad():
            mu, log_sigma = encoder(context_states, context_rewards)
    else:
        if hasattr(encoder, "sample"):
            z, mu, log_sigma = encoder.sample(context_states, context_rewards)
        else:  # pragma: no cover - fallback for minimal encoders
            mu, log_sigma = encoder(context_states, context_rewards)
            std = torch.exp(torch.clamp(log_sigma, _LOG_SIGMA_MIN, _LOG_SIGMA_MAX))
            z = mu + std * torch.randn_like(std)

    pred_reward = decoder.predict_for_context(decoder_states, z)

    return info_bottleneck_loss(
        pred_reward=pred_reward,
        true_reward=decoder_rewards,
        mu=mu,
        log_sigma=log_sigma,
        beta=beta,
        return_metrics=return_metrics,
    )
