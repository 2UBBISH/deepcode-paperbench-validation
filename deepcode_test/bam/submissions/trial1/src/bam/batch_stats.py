"""Batch sample and score statistics for Batch-and-Match variational inference.

Given a batch of samples ``z_1, ..., z_B ~ q_t`` and the corresponding target
scores ``g_b = grad_z log p(z_b)``, this module computes the four empirical
quantities required by the BaM covariance and mean updates:

    zbar  = (1/B) sum_b z_b
    gbar  = (1/B) sum_b g_b
    C     = (1/B) sum_b (z_b - zbar)(z_b - zbar)^T
    Gamma = (1/B) sum_b (g_b - gbar)(g_b - gbar)^T

All quantities are computed with JAX so the operations remain differentiable
whenever the inputs (in particular the scores ``g``) are differentiable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class BatchStats(NamedTuple):
    """Container for empirical batch statistics used by BaM.

    Attributes
    ----------
    zbar:
        Sample mean of the latent batch, shape ``(D,)``.
    gbar:
        Sample mean of the target scores, shape ``(D,)``.
    C:
        Sample covariance of the latent batch, shape ``(D, D)``.
    Gamma:
        Sample covariance of the target scores, shape ``(D, D)``.
    """

    zbar: jnp.ndarray
    gbar: jnp.ndarray
    C: jnp.ndarray
    Gamma: jnp.ndarray


def compute_batch_stats(
    z: jnp.ndarray,
    g: jnp.ndarray,
) -> BatchStats:
    """Compute empirical means and covariances of samples and scores.

    Parameters
    ----------
    z:
        Batch of latent samples, shape ``(B, D)``.
    g:
        Batch of target scores ``grad_z log p(z_b)``, shape ``(B, D)``.

    Returns
    -------
    BatchStats
        A named tuple containing ``zbar``, ``gbar``, ``C``, and ``Gamma``.
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z to have shape (B, D), got {z.shape}.")
    if g.shape != z.shape:
        raise ValueError(
            f"Expected g to have the same shape as z ({z.shape}), got {g.shape}."
        )

    batch_size = z.shape[0]
    if batch_size == 0:
        raise ValueError("The batch size must be positive.")

    zbar = jnp.mean(z, axis=0)
    gbar = jnp.mean(g, axis=0)

    z_centered = z - zbar[jnp.newaxis, :]
    g_centered = g - gbar[jnp.newaxis, :]

    # Empirical covariances normalized by B, matching the paper's definitions.
    C = (z_centered.T @ z_centered) / batch_size
    Gamma = (g_centered.T @ g_centered) / batch_size

    return BatchStats(zbar=zbar, gbar=gbar, C=C, Gamma=Gamma)


def batch_stats_to_dict(stats: BatchStats) -> dict[str, jnp.ndarray]:
    """Convert a :class:`BatchStats` tuple into a plain dictionary."""
    return {
        "zbar": stats.zbar,
        "gbar": stats.gbar,
        "C": stats.C,
        "Gamma": stats.Gamma,
    }
