"""Deep generative model utilities for the CIFAR-10 experiments.

This module implements the Section 5.3 machinery of the Batch-and-Match paper:

* A downloader/loader for CIFAR-10 flattened to observation vectors in R^3072
  (internally kept as 32x32x3 images for the convolutional networks).
* A small convolutional VAE with a 256-dimensional latent space and a
  Gaussian decoder with fixed variance ``sigma^2 = 0.1``.
* A :class:`DeepGenerativeTarget` posterior wrapper exposing the unnormalized
  log posterior ``log p(z) + log p(x | z, theta)`` and its score (gradient)
  via JAX automatic differentiation.
* Amortized variational inference (AVI) encoder helpers used as a factorized
  Gaussian baseline and as an initializer for full-covariance BaM/ADVI runs.

The implementation is intentionally self-contained (apart from JAX/NumPy) so it
can be exercised without optional deep-learning frameworks.
"""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "CIFAR10_URL",
    "download_cifar10",
    "load_cifar10",
    "flatten_images",
    "init_vae_params",
    "encoder_forward",
    "decoder_forward",
    "vae_elbo",
    "train_vae",
    "encode",
    "avi_posterior",
    "initial_posterior_params",
    "initial_advi_factor",
    "reconstruction_mse",
    "DeepGenerativeTarget",
]


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# ---------------------------------------------------------------------------
# Small pytree-friendly Adam optimizer (kept local to avoid coupling with the
# variational-inference baselines module).
# ---------------------------------------------------------------------------


def _adam_init(params):
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }


def _adam_step(params, grads, state, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(
        lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * g_i ** 2, state["v"], grads
    )
    m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1.0 - beta1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1.0 - beta2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p_i, mh_i, vh_i: p_i - learning_rate * mh_i / (jnp.sqrt(vh_i) + eps),
        params,
        m_hat,
        v_hat,
    )
    return new_params, {"m": m, "v": v, "t": t}


# ---------------------------------------------------------------------------
# CIFAR-10 data loading
# ---------------------------------------------------------------------------


def download_cifar10(data_dir: Optional[str] = "~/.bam_data/cifar10") -> Path:
    """Download and extract CIFAR-10, returning the extracted directory."""
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "cifar-10-python.tar.gz"
    extracted_dir = data_dir / "cifar-10-batches-py"

    if not tar_path.exists():
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)

    if not extracted_dir.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)

    return extracted_dir


def _unpickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10(
    data_dir: Optional[str] = "~/.bam_data/cifar10",
    normalize: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load CIFAR-10 as (N, 32, 32, 3) float32 arrays in [0, 1].

    Returns ``(train_x, test_x)``.  Use :func:`flatten_images` to obtain the
    R^3072 observation representation used in the likelihood.
    """
    cifar_dir = download_cifar10(data_dir)

    train_parts = []
    for i in range(1, 6):
        batch = _unpickle(cifar_dir / f"data_batch_{i}")
        train_parts.append(batch[b"data"])

    train_x = np.concatenate(train_parts, axis=0)
    test_batch = _unpickle(cifar_dir / "test_batch")
    test_x = test_batch[b"data"]

    # CIFAR-10 stores RGB data row-major as (N, 3, 32, 32).
    train_x = train_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_x = test_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    if normalize:
        train_x /= 255.0
        test_x /= 255.0

    return jnp.asarray(train_x), jnp.asarray(test_x)


def flatten_images(x: jnp.ndarray) -> jnp.ndarray:
    """Flatten (N, H, W, C) images to (N, H*W*C) observation vectors."""
    x = jnp.asarray(x)
    return x.reshape(x.shape[0], -1)


# ---------------------------------------------------------------------------
# Convolutional encoder/decoder primitives
# ---------------------------------------------------------------------------


def _he_normal(key, shape, fan_in):
    return jax.random.normal(key, shape) * jnp.sqrt(2.0 / max(float(fan_in), 1.0))


def _conv(x, w, b):
    return (
        jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + b
    )


def _conv_transpose(x, w, b):
    return (
        jax.lax.conv_transpose(
            x,
            w,
            strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        + b
    )


def _as_batch(z: jnp.ndarray) -> jnp.ndarray:
    z = jnp.asarray(z)
    if z.ndim == 1:
        z = z[None, :]
    return z


def _as_batch_image(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x)
    if x.ndim == 3:
        x = x[None, ...]
    return x


def _init_encoder(key, latent_dim: int = 256):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "conv1": {"w": _he_normal(k1, (4, 4, 3, 32), 4 * 4 * 3), "b": jnp.zeros(32)},
        "conv2": {"w": _he_normal(k2, (4, 4, 32, 64), 4 * 4 * 32), "b": jnp.zeros(64)},
        "conv3": {
            "w": _he_normal(k3, (4, 4, 64, 128), 4 * 4 * 64),
            "b": jnp.zeros(128),
        },
        "dense1": {"w": _he_normal(k4, (2048, 512), 2048), "b": jnp.zeros(512)},
        "dense2": {"w": _he_normal(k5, (512, 2 * latent_dim), 512), "b": jnp.zeros(2 * latent_dim)},
    }


def _init_decoder(key, latent_dim: int = 256):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "dense1": {"w": _he_normal(k1, (latent_dim, 2048), latent_dim), "b": jnp.zeros(2048)},
        "convT1": {"w": _he_normal(k2, (4, 4, 64, 128), 4 * 4 * 128), "b": jnp.zeros(64)},
        "convT2": {"w": _he_normal(k3, (4, 4, 32, 64), 4 * 4 * 64), "b": jnp.zeros(32)},
        "convT3": {"w": _he_normal(k4, (4, 4, 3, 32), 4 * 4 * 32), "b": jnp.zeros(3)},
    }


def init_vae_params(key, latent_dim: int = 256):
    """Initialize encoder and decoder parameters.

    Returns a dict ``{"encoder": ..., "decoder": ...}`` where each value is a
    pytree of arrays.
    """
    k1, k2 = jax.random.split(key)
    return {
        "encoder": _init_encoder(k1, latent_dim),
        "decoder": _init_decoder(k2, latent_dim),
    }


def encoder_forward(encoder_params, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map (N, 32, 32, 3) images to Gaussian latent mean and log-variance."""
    h = jax.nn.relu(_conv(x, encoder_params["conv1"]["w"], encoder_params["conv1"]["b"]))
    h = jax.nn.relu(_conv(h, encoder_params["conv2"]["w"], encoder_params["conv2"]["b"]))
    h = jax.nn.relu(_conv(h, encoder_params["conv3"]["w"], encoder_params["conv3"]["b"]))
    h = h.reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ encoder_params["dense1"]["w"] + encoder_params["dense1"]["b"])
    out = h @ encoder_params["dense2"]["w"] + encoder_params["dense2"]["b"]
    half = out.shape[1] // 2
    return out[:, :half], out[:, half:]


def decoder_forward(decoder_params, z: jnp.ndarray) -> jnp.ndarray:
    """Map (N, latent_dim) latent vectors to (N, 32, 32, 3) mean images."""
    z = _as_batch(z)
    h = z @ decoder_params["dense1"]["w"] + decoder_params["dense1"]["b"]
    h = jax.nn.relu(h)
    h = h.reshape(h.shape[0], 4, 4, 128)
    h = jax.nn.relu(_conv_transpose(h, decoder_params["convT1"]["w"], decoder_params["convT1"]["b"]))
    h = jax.nn.relu(_conv_transpose(h, decoder_params["convT2"]["w"], decoder_params["convT2"]["b"]))
    h = _conv_transpose(h, decoder_params["convT3"]["w"], decoder_params["convT3"]["b"])
    return jax.nn.sigmoid(h)


# ---------------------------------------------------------------------------
# VAE training
# ---------------------------------------------------------------------------


def vae_elbo(params, x: jnp.ndarray, sigma2: float, key) -> jnp.ndarray:
    """Negative mean ELBO used as the VAE training loss.

    The decoder likelihood is Gaussian with fixed variance ``sigma2`` and the
    prior is a standard Gaussian on the latent space.
    """
    mu, logvar = encoder_forward(params["encoder"], x)
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, mu.shape)
    z = mu + std * eps

    x_mean = decoder_forward(params["decoder"], z)
    x_flat = x.reshape(x.shape[0], -1)
    mean_flat = x_mean.reshape(x_mean.shape[0], -1)
    obs_dim = x_flat.shape[1]

    log_px = -0.5 * jnp.sum((x_flat - mean_flat) ** 2, axis=1) / sigma2
    log_px -= 0.5 * obs_dim * jnp.log(2.0 * jnp.pi * sigma2)
    kl = -0.5 * jnp.sum(1.0 + logvar - mu ** 2 - jnp.exp(logvar), axis=1)

    elbo = jnp.mean(log_px - kl)
    return -elbo


def train_vae(
    key,
    train_x: jnp.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    latent_dim: int = 256,
    sigma2: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Train the convolutional VAE and return ``{"encoder": ..., "decoder": ...}``."""
    params = init_vae_params(key, latent_dim)
    opt_state = _adam_init(params)
    n = train_x.shape[0]
    step_key = key

    for epoch in range(epochs):
        step_key, perm_key = jax.random.split(step_key)
        perm = jax.random.permutation(perm_key, n)
        epoch_losses = []
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = train_x[idx]
            step_key, batch_key = jax.random.split(step_key)
            loss, grads = jax.value_and_grad(
                lambda p: vae_elbo(p, xb, sigma2, batch_key)
            )(params)
            params, opt_state = _adam_step(params, grads, opt_state, learning_rate)
            epoch_losses.append(float(loss))
        if verbose:
            print(
                f"[vae] epoch {epoch + 1}/{epochs} "
                f"loss {float(np.mean(epoch_losses)):.4f}"
            )

    return params


# ---------------------------------------------------------------------------
# Amortized variational inference (AVI) helpers
# ---------------------------------------------------------------------------


def encode(encoder_params, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the factorized Gaussian encoder mean and log-variance.

    Accepts either a single image ``(32, 32, 3)`` or a batch ``(N, 32, 32, 3)``.
    """
    single = jnp.asarray(x).ndim == 3
    xb = _as_batch_image(x)
    mu, logvar = encoder_forward(encoder_params, xb)
    if single:
        return mu[0], logvar[0]
    return mu, logvar


def avi_posterior(encoder_params, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(mean, diagonal_variance)`` of the factorized AVI posterior."""
    mu, logvar = encode(encoder_params, x)
    return mu, jnp.exp(logvar)


def initial_posterior_params(
    encoder_params,
    x: jnp.ndarray,
    *,
    use_encoder_cov: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initializer for full-covariance Gaussian inference.

    The mean is initialized at the amortized encoder mean.  The covariance is
    either the encoder's factorized variance (default) or the identity matrix.
    """
    mu, logvar = encode(encoder_params, x)
    if use_encoder_cov:
        sigma0 = jnp.diag(jnp.exp(logvar))
    else:
        sigma0 = jnp.eye(mu.shape[0])
    return mu, sigma0


def initial_advi_factor(
    encoder_params,
    x: jnp.ndarray,
    *,
    use_encoder_cov: bool = True,
    jitter: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(mu, L)`` with ``Sigma = L L^T`` for full-covariance ADVI."""
    mu, sigma0 = initial_posterior_params(
        encoder_params, x, use_encoder_cov=use_encoder_cov
    )
    d = sigma0.shape[0]
    L = jnp.linalg.cholesky(sigma0 + jitter * jnp.eye(d))
    return mu, L


def reconstruction_mse(decoder_params, z_mean: jnp.ndarray, x_obs: jnp.ndarray) -> jnp.ndarray:
    """Mean squared reconstruction error ``||x_obs - decode(z_mean)||^2``.

    ``z_mean`` is the posterior mean E[z|x].  ``x_obs`` may be a single image
    or a batch of images.
    """
    x_hat = decoder_forward(decoder_params, _as_batch(z_mean))
    x_obs_b = _as_batch_image(x_obs)
    return jnp.mean(jnp.sum((x_obs_b - x_hat) ** 2, axis=(1, 2, 3)))


# ---------------------------------------------------------------------------
# Posterior target for a fixed observation
# ---------------------------------------------------------------------------


class DeepGenerativeTarget:
    """Unnormalized posterior ``p(z | x) ∝ p(z) p(x | z, theta)``.

    The decoder variance is held fixed at ``sigma2`` (0.1 in the paper), and the
    prior is a standard Gaussian.  Log densities and scores are computed with
    JAX so they can be consumed directly by the BaM and baseline algorithms.
    """

    def __init__(self, x_obs: jnp.ndarray, decoder_params, sigma2: float = 0.1):
        self.x_obs = jnp.asarray(x_obs)
        self.decoder_params = decoder_params
        self.sigma2 = float(sigma2)

        flat = self.x_obs.reshape(-1)
        self.observation_dim = int(flat.shape[0])
        self.dim = int(decoder_params["dense1"]["w"].shape[0])

    def _decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return decoder_forward(self.decoder_params, _as_batch(z))

    def _log_likelihood_single(self, z: jnp.ndarray) -> jnp.ndarray:
        x_mean = self._decode(z)[0]
        diff = (self.x_obs - x_mean).reshape(-1)
        quad = jnp.sum(diff * diff)
        return (
            -0.5 * quad / self.sigma2
            - 0.5 * self.observation_dim * jnp.log(2.0 * jnp.pi * self.sigma2)
        )

    def _log_prob_single(self, z: jnp.ndarray) -> jnp.ndarray:
        log_prior = -0.5 * jnp.sum(z * z) - 0.5 * self.dim * jnp.log(2.0 * jnp.pi)
        return log_prior + self._log_likelihood_single(z)

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        """Unnormalized log posterior for a single vector or a batch."""
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self._log_prob_single(z)
        return jax.vmap(self._log_prob_single)(z)

    def log_likelihood(self, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self._log_likelihood_single(z)
        return jax.vmap(self._log_likelihood_single)(z)

    def score(self, z: jnp.ndarray) -> jnp.ndarray:
        """Gradient ``grad_z log p(z | x)`` for a single vector or a batch."""
        grad_fn = jax.grad(self._log_prob_single)
        z = jnp.asarray(z)
        if z.ndim == 1:
            return grad_fn(z)
        return jax.vmap(grad_fn)(z)

    def score_fn(self):
        """Jitted, vmapped score function accepting ``(B, D)`` batches."""
        return jax.jit(jax.vmap(jax.grad(self._log_prob_single)))

    def log_prob_fn(self):
        """Jitted, vmapped log-probability function accepting ``(B, D)`` batches."""
        return jax.jit(jax.vmap(self._log_prob_single))

    def reconstruct(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode ``z`` (single vector or batch) to observation-space means."""
        x_mean = self._decode(z)
        if jnp.asarray(z).ndim == 1:
            return x_mean[0]
        return x_mean
