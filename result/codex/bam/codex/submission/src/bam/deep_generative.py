"""The deep generative model posterior of Section 5.3.

Given a pre-trained decoder ``Omega(., theta)`` and a new observation ``x'``, the
target of the variational inference problem is

    p(z' | x')  ∝  N(z'; 0, I) N(x'; Omega(z', theta), sigma^2 I),     sigma^2 = 0.1,

with ``z' in R^256`` and ``x' in R^3072``.  Only the score is needed by the VI
algorithms; reconstruction quality is evaluated by feeding the posterior mean
``E[z' | x']`` through the decoder and comparing with ``x'``.

Two variational approximations are compared in Section 5.3:

* a *full-covariance* Gaussian fitted by BaM / ADVI / GSM, and
* amortized variational inference (AVI) using the pre-trained encoder, which
  gives a factorized Gaussian ``q(z' | x')`` essentially for free.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from .metrics import reconstruction_mse
from .targets import Target
from .vae import VAEConfig, decoder_apply, encoder_apply


class DeepGenerativePosterior(Target):
    """``p(z | x_obs)`` for a fixed observation, with a fixed (pre-trained) decoder."""

    is_normalized = False
    is_samplable = False

    def __init__(self, decoder_params: Dict[str, Any], config: VAEConfig, x_obs: np.ndarray,
                 name: str = "vae_posterior"):
        self.decoder_params = decoder_params
        self.config = config
        self.x_obs = jnp.asarray(np.asarray(x_obs, dtype=np.float64).reshape(-1))
        self.dim = int(config.latent_dim)
        self.name = name

        def log_joint(z):
            x_hat = decoder_apply(decoder_params, z, config).reshape(-1)
            prior = -0.5 * jnp.sum(z**2)
            like = -0.5 * jnp.sum((self.x_obs - x_hat) ** 2) / config.sigma2
            return prior + like

        self.log_joint = log_joint
        self._score_batch = jax.jit(jax.vmap(jax.grad(log_joint)))
        self._logd_batch = jax.jit(jax.vmap(log_joint))

    def log_density(self, Z) -> np.ndarray:
        Z = np.atleast_2d(Z)
        return np.asarray(self._logd_batch(jnp.asarray(Z)))

    def score(self, Z) -> np.ndarray:
        Z = np.atleast_2d(Z)
        return np.asarray(self._score_batch(jnp.asarray(Z)))

    def log_density_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._logd_batch(jnp.atleast_2d(Z))

    def score_jax(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._score_batch(jnp.atleast_2d(Z))

    # -- evaluation ----------------------------------------------------------
    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """Apply the decoder to (a batch of) latent codes."""
        z = jnp.asarray(np.atleast_2d(z))
        return np.asarray(decoder_apply(self.decoder_params, z, self.config))

    def reconstruction_mse(self, z: np.ndarray) -> float:
        x_hat = self.reconstruct(z)[0] if np.asarray(z).ndim == 1 else self.reconstruct(z)
        x = np.asarray(self.x_obs)
        return reconstruction_mse(x, np.asarray(x_hat).reshape(-1), self.config.sigma2)


def amortized_posterior(encoder_params: Dict[str, Any], decoder_params: Dict[str, Any],
                        config: VAEConfig, x_obs: np.ndarray) -> Dict[str, np.ndarray]:
    """AVI baseline: ``q(z | x) = N(mean(x), diag(exp(logvar(x))))`` from the encoder."""
    x = jnp.asarray(np.asarray(x_obs, dtype=np.float64).reshape(1, 32, 32, 3))
    mean, logvar = encoder_apply(encoder_params, x)
    mean = np.asarray(mean)[0]
    sd = np.exp(0.5 * np.asarray(logvar)[0])
    return {"mu": mean, "Sigma": np.diag(sd**2), "sd": sd, "logvar": np.asarray(logvar)[0]}


__all__ = ["DeepGenerativePosterior", "amortized_posterior"]
