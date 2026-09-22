"""Convolutional VAE used for the deep generative model of Section 5.3.

The generative model is

    z_n ~ N(0, I),      x_n | z_n ~ N(Omega(z_n, theta), sigma^2 I),   sigma^2 = 0.1,

with ``x_n in R^3072`` (32x32x3 CIFAR-10 images, modelled as continuous) and
``z_n in R^256``.  ``Omega(., theta)`` is the convolutional decoder described in
the addendum to the paper:

    Dense -> [4, 4, 2 c_hid]
    ConvTranspose(2 c_hid -> 2 c_hid, k=3, s=2) -> 8x8
    Conv(2 c_hid -> 2 c_hid, k=3, s=1)          -> 8x8
    ConvTranspose(2 c_hid -> c_hid, k=3, s=2)   -> 16x16
    Conv(c_hid -> c_hid, k=3, s=1)              -> 16x16
    ConvTranspose(c_hid -> 3, k=3, s=2)         -> 32x32
    tanh

with GELU activations in all hidden layers, no dropout, no normalization and no
explicit pooling (downsampling happens through ``stride=2`` convolutions).  The
encoder mirrors the decoder:

    Conv(3 -> c_hid, k=3, s=2) -> 16x16
    Conv(c_hid -> c_hid, k=3, s=1) -> 16x16
    Conv(c_hid -> 2 c_hid, k=3, s=2) -> 8x8
    Conv(2 c_hid -> 2 c_hid, k=3, s=1) -> 8x8
    Conv(2 c_hid -> 2 c_hid, k=3, s=2) -> 4x4
    Flatten -> Dense(latent_dim)

and outputs the mean of the factorized Gaussian ``q(z | x)`` used for amortized
variational inference.  (The addendum lists a mean head only; the encoder used
here additionally predicts a log-variance, which AVI requires.)

The decoder is pre-trained with variational expectation-maximization, i.e. by
maximizing the ELBO with a factorized Gaussian ``q(z | x)`` -- this is exactly
standard VAE training (Kingma & Welling, 2014), with one Monte-Carlo sample for
the negative ELBO and the optimizer settings from the addendum (Adam, linear
warmup from 0 to 1e-4 over 100 batches, linear decay to 1e-5 over 500 batches).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .adam import Adam


@dataclass(frozen=True)
class VAEConfig:
    latent_dim: int = 256
    c_hid: int = 64
    sigma2: float = 0.1
    image_shape: Tuple[int, int, int] = (32, 32, 3)
    # optimizer settings from the addendum
    lr_init: float = 0.0
    lr_peak: float = 1e-4
    lr_end: float = 1e-5
    warmup_steps: int = 100
    decay_steps: int = 500

    @property
    def x_dim(self) -> int:
        a, b, c = self.image_shape
        return a * b * c


# ----------------------------------------------------------------------------
# parameter initialization
# ----------------------------------------------------------------------------


def _conv_params(key, shape, bias: bool = True):
    w = jax.random.normal(key, shape) * jnp.sqrt(2.0 / (np.prod(shape[:-1]) + 1e-8)) / 1.0
    p = {"w": w}
    if bias:
        p["b"] = jnp.zeros(shape[-1])
    return p


def _dense_params(key, n_in, n_out):
    w = jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
    return {"w": w, "b": jnp.zeros(n_out)}


def init_params(key, config: VAEConfig) -> Dict[str, Any]:
    keys = jax.random.split(key, 40)
    c, c2 = config.c_hid, 2 * config.c_hid
    enc = {
        "c1": _conv_params(keys[0], (3, 3, 3, c)),
        "c2": _conv_params(keys[1], (3, 3, c, c)),
        "c3": _conv_params(keys[2], (3, 3, c, c2)),
        "c4": _conv_params(keys[3], (3, 3, c2, c2)),
        "c5": _conv_params(keys[4], (3, 3, c2, c2)),
        "d_mean": _dense_params(keys[5], 4 * 4 * c2, config.latent_dim),
        "d_logvar": _dense_params(keys[6], 4 * 4 * c2, config.latent_dim),
    }
    dec = {
        "d1": _dense_params(keys[7], config.latent_dim, 4 * 4 * c2),
        # for the transposed convolutions the JAX kernel layout ('HWIO') is
        # (height, width, input_channels, output_channels) of the transposed op
        "ct1": _conv_params(keys[8], (3, 3, c2, c2)),
        "c6": _conv_params(keys[9], (3, 3, c2, c2)),
        "ct2": _conv_params(keys[10], (3, 3, c2, c)),
        "c7": _conv_params(keys[11], (3, 3, c, c)),
        "ct3": _conv_params(keys[12], (3, 3, c, 3)),
    }
    return {"enc": enc, "dec": dec}


# ----------------------------------------------------------------------------
# forward passes
# ----------------------------------------------------------------------------


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def _conv(x, p, stride: int = 1):
    y = jax.lax.conv_general_dilated(x, p["w"], window_strides=(stride, stride), padding="SAME",
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + p["b"]


def _conv_transpose(x, p, stride: int = 2):
    y = jax.lax.conv_transpose(x, p["w"], strides=(stride, stride), padding="SAME",
                               dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + p["b"]


def encoder_apply(enc: Dict[str, Any], x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map images ``x`` (..., 32, 32, 3) to ``(mean, logvar)`` of ``q(z | x)``."""
    h = _gelu(_conv(x, enc["c1"], stride=2))   # 16x16
    h = _gelu(_conv(h, enc["c2"], stride=1))
    h = _gelu(_conv(h, enc["c3"], stride=2))   # 8x8
    h = _gelu(_conv(h, enc["c4"], stride=1))
    h = _gelu(_conv(h, enc["c5"], stride=2))   # 4x4
    h = h.reshape(*h.shape[:-3], -1)
    mean = h @ enc["d_mean"]["w"] + enc["d_mean"]["b"]
    logvar = h @ enc["d_logvar"]["w"] + enc["d_logvar"]["b"]
    return mean, jnp.clip(logvar, -8.0, 8.0)


def decoder_apply(dec: Dict[str, Any], z: jnp.ndarray, config: VAEConfig) -> jnp.ndarray:
    """The generative network ``Omega(z, theta)``; outputs lie in [-1, 1]."""
    c2 = 2 * config.c_hid
    lead = z.shape[:-1]
    z = z.reshape(-1, z.shape[-1])
    h = z @ dec["d1"]["w"] + dec["d1"]["b"]
    h = h.reshape(-1, 4, 4, c2)
    h = _gelu(_conv_transpose(h, dec["ct1"], stride=2))  # 8x8
    h = _gelu(_conv(h, dec["c6"], stride=1))
    h = _gelu(_conv_transpose(h, dec["ct2"], stride=2))  # 16x16
    h = _gelu(_conv(h, dec["c7"], stride=1))
    h = _conv_transpose(h, dec["ct3"], stride=2)         # 32x32
    h = jnp.tanh(h)
    return h.reshape(*lead, h.shape[-3], h.shape[-2], h.shape[-1])


# ----------------------------------------------------------------------------
# training (variational expectation maximization)
# ----------------------------------------------------------------------------


def negative_elbo(params, x: jnp.ndarray, key, config: VAEConfig, n_mc: int = 1):
    """``-E_q[log p(x | z)] + KL(q(z|x) || p(z))`` with ``n_mc`` samples."""
    mean, logvar = encoder_apply(params["enc"], x)
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, (n_mc,) + mean.shape)
    z = (mean[None] + eps * std[None]).reshape(-1, mean.shape[-1])
    x_hat = decoder_apply(params["dec"], z, config)
    x_exp = jnp.tile(x, (n_mc,) + (1,) * (x.ndim - 1))
    recon = jnp.mean(jnp.sum((x_hat - x_exp) ** 2, axis=tuple(range(1, x_hat.ndim)))) / (2.0 * config.sigma2)
    kl = 0.5 * jnp.sum(mean**2 + jnp.exp(logvar) - 1.0 - logvar, axis=-1)
    return jnp.mean(recon) + jnp.mean(kl)


def lr_schedule(step: int, config: VAEConfig) -> float:
    """Linear warmup to the peak, then linear decay (addendum's optimizer settings)."""
    if step < config.warmup_steps:
        frac = (step + 1) / max(config.warmup_steps, 1)
        return config.lr_init + frac * (config.lr_peak - config.lr_init)
    frac = min(1.0, (step - config.warmup_steps + 1) / max(config.decay_steps, 1))
    return config.lr_peak + frac * (config.lr_end - config.lr_peak)


def _flatten(tree) -> Tuple[np.ndarray, list]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return np.concatenate([np.asarray(l).ravel() for l in leaves]), treedef


def _unflatten(vec: np.ndarray, treedef, template):
    leaves = jax.tree_util.tree_leaves(template)
    out, i = [], 0
    for l in leaves:
        n = int(np.prod(np.shape(l)))
        out.append(vec[i:i + n].reshape(np.shape(l)))
        i += n
    return jax.tree_util.tree_unflatten(treedef, out)


def train_vae(x_train: np.ndarray, key, config: VAEConfig, n_epochs: int = 100, batch_size: int = 128,
              params: Optional[Dict[str, Any]] = None, verbose: bool = True,
              log_every: int = 50) -> Dict[str, Any]:
    """Pre-train the decoder (and encoder) by maximizing the ELBO."""
    n = x_train.shape[0]
    steps_per_epoch = max(1, n // batch_size)
    params = init_params(key, config) if params is None else params
    flat, treedef = _flatten(params)
    dtype = jnp.asarray(flat).dtype
    opt = Adam(flat, lr=config.lr_peak)
    history = []
    grad_fn = jax.jit(jax.value_and_grad(negative_elbo), static_argnums=(3,))
    step = 0
    keys = jax.random.split(key, n_epochs * steps_per_epoch + 1)
    for epoch in range(n_epochs):
        perm = np.random.default_rng(epoch).permutation(n)
        epoch_loss = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = jnp.asarray(x_train[idx], dtype=dtype)
            opt.lr = lr_schedule(step, config)
            loss, grad = grad_fn(params, xb, keys[step], config)
            g, _ = _flatten(grad)
            flat = opt.update(flat, g)
            params = _unflatten(flat, treedef, params)
            epoch_loss += float(loss)
            step += 1
        epoch_loss /= steps_per_epoch
        history.append(epoch_loss)
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch:4d}  -ELBO {epoch_loss:.2f}  lr {opt.lr:.2e}", flush=True)
    return {"params": params, "history": np.asarray(history)}


def save_params(path: str, params) -> None:
    flat, treedef = _flatten(params)
    np.savez_compressed(path, flat=np.asarray(flat), treedef=json.dumps(str(treedef)))


def load_params(path: str, config: VAEConfig, key=None):
    """Load decoder/encoder parameters; ``key`` is only used to rebuild the template."""
    template = init_params(jax.random.PRNGKey(0) if key is None else key, config)
    with np.load(path) as f:
        flat = np.asarray(f["flat"])
    treedef = jax.tree_util.tree_structure(template)
    return _unflatten(flat, treedef, template)


__all__ = [
    "VAEConfig",
    "init_params",
    "encoder_apply",
    "decoder_apply",
    "negative_elbo",
    "train_vae",
    "save_params",
    "load_params",
]
