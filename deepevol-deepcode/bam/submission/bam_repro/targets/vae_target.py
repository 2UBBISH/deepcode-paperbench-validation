"""Deep-generative-model target for Section 5.3 / Appendix E.6 of
"Batch and Match: Black-Box Variational Inference with a Score-Based Divergence".

Paper specification (Section 5.3, page 6)
-----------------------------------------
The likelihood is parameterised by the output of a neural network Omega, i.e.

    z_n ~ N(0, I)                                   (1)
    x_n | z_n ~ N(Omega(z_n, theta_hat), sigma^2 I),

with ``sigma^2 = 0.1``, images ``x_n in R^{3072}`` (CIFAR-10, Krizhevsky 2009),
latent representation ``z_n in R^{256}``, and ``Omega`` a 5-layer convolutional
network (Appendix E.6).  The neural network is pre-trained by variational
expectation-maximisation: theta_hat maximises the marginal likelihood
``p({x_n}_{n=1}^N | theta)`` where the marginalisation is performed with
amortised variational inference (a factorised Gaussian encoder + ELBO); the
optimisation runs for 100 epochs (Figure E.8).

Given a *new* observation ``x'`` we approximate the posterior ``p(z' | x')``.
This module exposes the exact target potential used by BaM / ADVI / GSM /
Score-ADVI / Fisher-ADVI:

    log p(z' | x') = log p(z') + log p(x' | z') + const,
    log p(z')      = -0.5 ||z'||^2 - (D/2) log(2 pi),
    log p(x' | z') = -0.5/sigma^2 ||x' - Omega(z')||^2 - (M/2) log(2 pi sigma^2),

and its analytic score (an implementation of the BBVI interface used by
``bam_repro.bam.bam`` and the baselines, which call ``score_fn(z)``):

    score(z') = Grad_z [log p(z') + log p(x' | z')]
              = -z' + (1/sigma^2) J_Omega(z')^T (x' - Omega(z')).

The Jacobian-transpose-vector product ``J^T r`` is obtained by the exact manual
backward pass of the NumPy convolutional decoder (no autodiff dependency).  The
implementation is deliberately backend-light (NumPy only) so that it works
everywhere; the paper's reference implementation uses JAX, and all algorithms
that consume this target are backend agnostic.

Documented defaults (paper silent)
----------------------------------
* ``c_hid = 64`` convolutional channels (the paper only says "5 layers").
* Encoder mirrors the decoder; 100 warm-up steps 0 -> 1e-4 followed by 500
  decay steps 1e-4 -> 1e-5 (cosine), then constant 1e-5; ``mc_sim = 1``.
* Mini-batch size 128 for the decoder pre-training.
* Images are rescaled from ``uint8`` to ``[-1, 1]`` to match the ``tanh``
  decoder output (plan item (e)).
* When the CIFAR-10 files are unavailable a documented synthetic surrogate
  image data set is used so that the full pipeline stays runnable.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is optional; a small polynomial approximation is used otherwise
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover - fallback path
    def _erf(x):  # Abramowitz & Stegun 7.1.26, |eps| < 3e-7
        x = np.asarray(x, dtype=np.float64)
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        y = 1.0 - (
            (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592
        ) * t * np.exp(-ax * ax)
        return sign * y


__all__ = [
    # specification constants
    "VAE_IMAGE_DIM",
    "VAE_LATENT_DIM",
    "VAE_IMAGE_SHAPE",
    "VAE_SIGMA2",
    "VAE_C_HID",
    "VAE_N_EPOCHS",
    "VAE_MC_SIM",
    "VAE_BATCH_SIZE",
    "VAE_LR_WARMUP_STEPS",
    "VAE_LR_WARMUP_START",
    "VAE_LR_PEAK",
    "VAE_LR_DECAY_STEPS",
    "VAE_LR_FINAL",
    "PAPER_VAE_PILOT_T",
    "PAPER_VAE_T",
    "PAPER_VAE_BATCH_SIZES",
    "PAPER_VAE_ADVI_LR",
    "PAPER_VAE_ADVI_LR_GRID",
    "PAPER_VAE_BAM_LAMBDAS",
    "PAPER_VAE_BAM_LAMBDA_GRIDS",
    "PAPER_VAE_GRAD_BUDGET",
    "PAPER_VAE_WALLCLOCK_TARGET_BATCH",
    # numerics / layers
    "gelu",
    "gelu_grad",
    "tanh_grad",
    "Linear",
    "Conv2D",
    "ConvTranspose2D",
    "GELU",
    "Tanh",
    "Flatten",
    "Reshape",
    "Decoder",
    "Encoder",
    "VAE",
    "Adam",
    "vae_lr_schedule",
    "train_vae",
    "save_vae",
    "load_vae",
    # target
    "VAETarget",
    "VAEPosteriorTarget",
    "vae_target_from_image",
    "score_function",
    "amortized_posterior",
    "amortized_reconstruction",
    "amortized_reconstruction_mse",
    "laplace_posterior",
    "flat_to_image",
    "image_to_flat",
    # data
    "ImageData",
    "load_cifar10",
    "find_cifar10_dir",
    "synthetic_images",
    "synthetic_image_data",
]


# ---------------------------------------------------------------------------
# Paper constants (Section 5.3 / Appendix E.6)
# ---------------------------------------------------------------------------

VAE_IMAGE_DIM: int = 3072          # x_n in R^{3072} (3 x 32 x 32 CIFAR-10)
VAE_LATENT_DIM: int = 256          # z_n in R^{256}
VAE_IMAGE_SHAPE: Tuple[int, int, int] = (3, 32, 32)
VAE_SIGMA2: float = 0.1            # sigma^2 = 0.1 (paper)

VAE_C_HID: int = 64                # documented default: 5-layer conv net width
VAE_N_EPOCHS: int = 100            # "optimization is performed over 100 epochs"
VAE_MC_SIM: int = 1                # Monte-Carlo samples for the ELBO
VAE_BATCH_SIZE: int = 128          # documented default for decoder pre-training

VAE_LR_WARMUP_STEPS: int = 100     # addendum: 0 -> 1e-4 over 100 warmup steps
VAE_LR_WARMUP_START: float = 0.0
VAE_LR_PEAK: float = 1e-4
VAE_LR_DECAY_STEPS: int = 500      # addendum: 1e-4 -> 1e-5 over 500 decay steps
VAE_LR_FINAL: float = 1e-5

# Posterior inference experiment (Section 5.3): pilot T=100, main T=1000
PAPER_VAE_PILOT_T: int = 100
PAPER_VAE_T: int = 1000
PAPER_VAE_BATCH_SIZES: Tuple[int, ...] = (10, 100, 300)

# "For ADVI, we consistently find the best learning rate to be l = 0.02
#  (after searching l = 0.001, 0.01, 0.02, 0.05)."
PAPER_VAE_ADVI_LR: float = 0.02
PAPER_VAE_ADVI_LR_GRID: Tuple[float, ...] = (0.001, 0.01, 0.02, 0.05)

# BaM: selected lambda per batch size (Appendix E.6)
PAPER_VAE_BAM_LAMBDAS: Dict[int, float] = {10: 0.1, 100: 50.0, 300: 7500.0}
PAPER_VAE_BAM_LAMBDA_GRIDS: Dict[int, Tuple[float, ...]] = {
    10: (0.01, 0.1, 0.2, 10.0),
    100: (2.0, 20.0, 50.0, 100.0, 200.0),
    300: (1000.0, 5000.0, 7500.0, 10000.0),
}
PAPER_VAE_GRAD_BUDGET: int = 3000  # 3000 gradient evaluations comparison
PAPER_VAE_WALLCLOCK_TARGET_BATCH: int = 300  # BaM B=300 fastest in wallclock

_MIN_EIG: float = 1e-12


# ---------------------------------------------------------------------------
# small image helpers
# ---------------------------------------------------------------------------

def flat_to_image(x, image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE):
    """(..., 3072) flattened CHW images -> (..., 3, 32, 32)."""
    x = np.asarray(x)
    return x.reshape(*x.shape[:-1], *image_shape)


def image_to_flat(x):
    """(..., 3, 32, 32) -> (..., 3072) flattened CHW images."""
    x = np.asarray(x)
    return x.reshape(*x.shape[:-3], int(np.prod(x.shape[-3:])))


def _pad(x, pad: int):
    if pad <= 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------

def gelu(x):
    """Exact GELU: 0.5 x (1 + erf(x / sqrt(2)))."""
    x = np.asarray(x)
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def gelu_grad(x):
    """Derivative of the exact GELU."""
    x = np.asarray(x)
    cdf = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return cdf + x * pdf


def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2


# ---------------------------------------------------------------------------
# numpy conv primitives (NCHW layout)
#
# im2col / col2im are implemented explicitly so that both the convolution and
# its transposed version (used as the decoder) have exact adjoints, which gives
# us the analytic score without any autodiff dependency.
# ---------------------------------------------------------------------------

def _im2col_padded(xp, k: int, stride: int, ho: int, wo: int):
    """(N, C, Hp, Wp) *already padded* -> (N, C*k*k, ho*wo)."""
    n, c, _, _ = xp.shape
    cols = np.empty((n, c, k, k, ho, wo), dtype=xp.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j, :, :] = xp[
                :, :, i : i + stride * ho : stride, j : j + stride * wo : stride
            ]
    return cols.reshape(n, c * k * k, ho * wo)


def _col2im_canvas(cols, hin: int, win: int, k: int, stride: int, canvas_h: int, canvas_w: int):
    """Adjoint of :func:`_im2col_padded`: (N, C*k*k, hin*win) -> (N, C, canvas_h, canvas_w)."""
    n, ck, _ = cols.shape
    c = ck // (k * k)
    colsr = cols.reshape(n, c, k, k, hin, win)
    canvas = np.zeros((n, c, canvas_h, canvas_w), dtype=cols.dtype)
    for i in range(k):
        for j in range(k):
            canvas[:, :, i : i + stride * hin : stride, j : j + stride * win : stride] += colsr[
                :, :, i, j, :, :
            ]
    return canvas


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

class _Layer:
    """Base layer: holds a ``params`` dict and, after backward, a ``grads`` dict."""

    name = "layer"

    def __init__(self):
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}

    def forward(self, x, need_cache: bool = False):
        raise NotImplementedError

    def backward(self, dout, need_param_grads: bool = True):
        """Returns ``(dx, *param_grads)`` in the order of :meth:`param_list`."""
        raise NotImplementedError


class Linear(_Layer):
    def __init__(self, fan_in: int, fan_out: int, rng=None, gain: float = 1.0, name: str = "linear"):
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        scale = gain * np.sqrt(2.0 / max(fan_in, 1))
        self.name = name
        self.fan_in, self.fan_out = fan_in, fan_out
        self.params = {
            "W": rng.normal(0.0, scale, size=(fan_out, fan_in)),
            "b": np.zeros(fan_out),
        }
        self._x = None

    @property
    def W(self):
        return self.params["W"]

    @property
    def b(self):
        return self.params["b"]

    def forward(self, x, need_cache: bool = False):
        if need_cache:
            self._x = x
        return x @ self.params["W"].T + self.params["b"]

    def backward(self, dout, need_param_grads: bool = True):
        x = self._x
        if need_param_grads:
            dW = np.einsum("ni,nj->ij", dout, x)
            db = dout.sum(axis=0)
            self.grads = {"W": dW, "b": db}
        dx = dout @ self.params["W"]
        return (dx,)


class Conv2D(_Layer):
    """Convolution (NCHW), weight shape ``(C_out, C_in, k, k)``."""

    def __init__(self, cin, cout, k, stride=1, pad=0, rng=None, gain=1.0, name="conv"):
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        scale = gain * np.sqrt(2.0 / max(cin * k * k, 1))
        self.name = name
        self.cin, self.cout, self.k = cin, cout, k
        self.stride, self.pad = int(stride), int(pad)
        self.params = {
            "W": rng.normal(0.0, scale, size=(cout, cin, k, k)),
            "b": np.zeros(cout),
        }
        self._xshape = None
        self._cols = None

    def forward(self, x, need_cache: bool = False):
        n, c, h, w = x.shape
        ho = (h + 2 * self.pad - self.k) // self.stride + 1
        wo = (w + 2 * self.pad - self.k) // self.stride + 1
        xp = _pad(x, self.pad)
        cols = _im2col_padded(xp, self.k, self.stride, ho, wo)  # (N, C*k*k, ho*wo)
        if need_cache:
            self._xshape = (n, c, h, w)
            self._cols = cols
        wflat = self.params["W"].reshape(self.cout, -1)
        out = np.einsum("oi,nil->nol", wflat, cols) + self.params["b"][None, :, None]
        return out.reshape(n, self.cout, ho, wo)

    def backward(self, dout, need_param_grads: bool = True):
        n, c, h, w = self._xshape
        cols = self._cols
        dflat = dout.reshape(n, self.cout, -1)
        if need_param_grads:
            dW = np.einsum("nol,nil->oi", dflat, cols).reshape(self.params["W"].shape)
            db = dflat.sum(axis=(0, 2))
            self.grads = {"W": dW, "b": db}
        wflat = self.params["W"].reshape(self.cout, -1)
        dcols = np.einsum("oi,nol->nil", wflat, dflat)
        dx = _col2im_canvas(
            dcols, dout.shape[2], dout.shape[3], self.k, self.stride, h + 2 * self.pad, w + 2 * self.pad
        )
        if self.pad:
            dx = dx[:, :, self.pad : h + self.pad, self.pad : w + self.pad]
        return (dx,)


class ConvTranspose2D(_Layer):
    """Transposed convolution (NCHW), weight shape ``(C_in, C_out, k, k)``.

    Forward: canvas of size ``(H-1)*stride + k`` built by scattering
    ``W^T x``, then cropped by ``pad``.  Its adjoint is the plain convolution
    used in :meth:`backward`.
    """

    def __init__(self, cin, cout, k, stride=1, pad=0, rng=None, gain=1.0, name="convT"):
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        scale = gain * np.sqrt(2.0 / max(cin * k * k, 1))
        self.name = name
        self.cin, self.cout, self.k = cin, cout, k
        self.stride, self.pad = int(stride), int(pad)
        self.params = {
            "W": rng.normal(0.0, scale, size=(cin, cout, k, k)),
            "b": np.zeros(cout),
        }
        self._cache = None

    def out_size(self, h: int, w: int) -> Tuple[int, int]:
        ho = (h - 1) * self.stride - 2 * self.pad + self.k
        wo = (w - 1) * self.stride - 2 * self.pad + self.k
        return ho, wo

    def forward(self, x, need_cache: bool = False):
        n, cin, h, w = x.shape
        ho, wo = self.out_size(h, w)
        canvas_h = (h - 1) * self.stride + self.k
        canvas_w = (w - 1) * self.stride + self.k
        wflat = self.params["W"].reshape(cin, -1)  # (C_in, C_out*k*k)
        xflat = x.reshape(n, cin, h * w)
        cols = np.einsum("co,ncl->nol", wflat, xflat)  # (N, C_out*k*k, h*w)
        if need_cache:
            self._cache = (cols, xflat, (n, cin, h, w), (canvas_h, canvas_w))
        out = _col2im_canvas(cols, h, w, self.k, self.stride, canvas_h, canvas_w)
        if self.pad:
            out = out[:, :, self.pad : self.pad + ho, self.pad : self.pad + wo]
        return out + self.params["b"][None, :, None, None]

    def backward(self, dout, need_param_grads: bool = True):
        cols, xflat, (n, cin, h, w), (canvas_h, canvas_w) = self._cache
        ho, wo = self.out_size(h, w)
        if self.pad:
            dpad = np.zeros((n, self.cout, canvas_h, canvas_w), dtype=dout.dtype)
            dpad[:, :, self.pad : self.pad + ho, self.pad : self.pad + wo] = dout
        else:
            dpad = dout
        dcols = _im2col_padded(dpad, self.k, self.stride, h, w)  # (N, C_out*k*k, h*w)
        wflat = self.params["W"].reshape(cin, -1)
        if need_param_grads:
            dWflat = np.einsum("ncl,nol->co", xflat, dcols)  # (C_in, C_out*k*k)
            self.grads = {
                "W": dWflat.reshape(self.params["W"].shape),
                "b": dout.sum(axis=(0, 2, 3)),
            }
        dx = np.einsum("co,nol->ncl", wflat, dcols).reshape(n, cin, h, w)
        return (dx,)


class GELU(_Layer):
    def __init__(self, name="gelu"):
        super().__init__()
        self.name = name
        self._x = None

    def forward(self, x, need_cache: bool = False):
        if need_cache:
            self._x = x
        return gelu(x)

    def backward(self, dout, need_param_grads: bool = True):
        return (dout * gelu_grad(self._x),)


class Tanh(_Layer):
    def __init__(self, name="tanh"):
        super().__init__()
        self.name = name
        self._out = None

    def forward(self, x, need_cache: bool = False):
        out = np.tanh(x)
        if need_cache:
            self._out = out
        return out

    def backward(self, dout, need_param_grads: bool = True):
        if self._out is None:
            raise RuntimeError("Tanh.backward called without a cached forward pass")
        return (dout * (1.0 - self._out ** 2),)


class Flatten(_Layer):
    def __init__(self, name="flatten"):
        super().__init__()
        self.name = name
        self._shape = None

    def forward(self, x, need_cache: bool = False):
        if need_cache:
            self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, need_param_grads: bool = True):
        return (dout.reshape(self._shape),)


class Reshape(_Layer):
    def __init__(self, shape: Sequence[int], name="reshape"):
        super().__init__()
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        self._shape = None

    def forward(self, x, need_cache: bool = False):
        if need_cache:
            self._shape = x.shape
        return x.reshape(x.shape[0], *self.shape)

    def backward(self, dout, need_param_grads: bool = True):
        return (dout.reshape(self._shape),)


class _Sequential(_Layer):
    """Convenience container mirroring the decoder/encoder parameter interface."""

    def __init__(self, layers: Sequence[_Layer], name: str = "seq"):
        super().__init__()
        self.layers = list(layers)
        self.name = name

    def param_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params.items():
                out[f"{prefix}{i}.{k}"] = v
        return out

    def grads_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for k, g in layer.grads.items():
                out[f"{prefix}{i}.{k}"] = g
        return out

    def num_params(self) -> int:
        return int(sum(v.size for v in self.param_dict().values()))


# ---------------------------------------------------------------------------
# Decoder Omega(. , theta_hat) and Encoder (amortised factorised Gaussian)
# ---------------------------------------------------------------------------

DECODER_SPATIAL: int = 4  # spatial size feeding the transposed-conv stack


def make_decoder(
    latent_dim: int = VAE_LATENT_DIM,
    image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE,
    c_hid: int = VAE_C_HID,
    rng=None,
    spatial: int = DECODER_SPATIAL,
) -> "_Sequential":
    """5-layer convolutional decoder ``Omega`` with GELU hidden and tanh output."""
    rng = np.random.default_rng(0) if rng is None else rng
    layers: List[_Layer] = [
        Linear(latent_dim, c_hid * spatial * spatial, rng=rng, gain=1.0, name="lin"),
        GELU(),
        Reshape((c_hid, spatial, spatial)),
        ConvTranspose2D(c_hid, c_hid, 3, stride=1, pad=1, rng=rng, name="deconv5"),
        GELU(),
        ConvTranspose2D(c_hid, c_hid, 3, stride=1, pad=1, rng=rng, name="deconv4"),
        GELU(),
        ConvTranspose2D(c_hid, c_hid, 4, stride=2, pad=1, rng=rng, name="deconv3"),
        GELU(),
        ConvTranspose2D(c_hid, c_hid, 4, stride=2, pad=1, rng=rng, name="deconv2"),
        GELU(),
        ConvTranspose2D(c_hid, image_shape[0], 4, stride=2, pad=1, rng=rng, name="deconv1"),
        Tanh(),
    ]
    return _Sequential(layers, name="decoder")


def make_encoder(
    latent_dim: int = VAE_LATENT_DIM,
    image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE,
    c_hid: int = VAE_C_HID,
    rng=None,
    spatial: int = DECODER_SPATIAL,
) -> "_Sequential":
    """5-layer convolutional encoder (no final activation -> mean & log-variance)."""
    rng = np.random.default_rng(0) if rng is None else rng
    layers: List[_Layer] = [
        Conv2D(image_shape[0], c_hid, 4, stride=2, pad=1, rng=rng, name="conv1"),
        GELU(),
        Conv2D(c_hid, c_hid, 4, stride=2, pad=1, rng=rng, name="conv2"),
        GELU(),
        Conv2D(c_hid, c_hid, 4, stride=2, pad=1, rng=rng, name="conv3"),
        GELU(),
        Conv2D(c_hid, c_hid, 3, stride=1, pad=1, rng=rng, name="conv4"),
        GELU(),
        Conv2D(c_hid, c_hid, 3, stride=1, pad=1, rng=rng, name="conv5"),
        GELU(),
        Flatten(),
        Linear(c_hid * spatial * spatial, 2 * latent_dim, rng=rng, gain=1.0, name="lin_out"),
    ]
    return _Sequential(layers, name="encoder")


class Decoder:
    """Thin wrapper around the decoder layer stack with BaM-friendly helpers.

    The wrapper only adds (i) batching conveniences, (ii) the analytic score
    term ``(1/sigma^2) J_Omega(z)^T (x - Omega(z))`` and (iii) reconstruction
    metrics.  All parameters live in the wrapped ``_Sequential`` so the same
    arrays are shared with :class:`VAE` training.
    """

    def __init__(self, net: Optional[_Sequential] = None, latent_dim: int = VAE_LATENT_DIM,
                 image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE, c_hid: int = VAE_C_HID, rng=None):
        self.net = net if net is not None else make_decoder(latent_dim, image_shape, c_hid, rng=rng)
        self.latent_dim = int(latent_dim)
        self.image_shape = tuple(image_shape)
        self.image_dim = int(np.prod(self.image_shape))

    # -- parameter plumbing -------------------------------------------------
    def param_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        return self.net.param_dict(prefix)

    def grads_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        return self.net.grads_dict(prefix)

    def num_params(self) -> int:
        return self.net.num_params()

    # -- forward / backward -------------------------------------------------
    def forward(self, z, need_cache: bool = False, image: bool = True):
        out = self.net.forward(np.atleast_2d(z), need_cache=need_cache)
        if not image:
            out = image_to_flat(out)
        return out[0] if np.ndim(z) == 1 else out

    def reconstruction(self, z):
        """Omega(z): (D,) -> (3072,) or (B, D) -> (B, 3072) flattened images."""
        return self.forward(z, need_cache=False, image=False)

    def backward(self, dout, need_param_grads: bool = False, single: bool = False):
        """Backpropagation of ``dout`` through Omega; returns ``Grad_z`` (and caches grads)."""
        d = np.atleast_2d(dout)
        for layer in reversed(self.net.layers):
            d = layer.backward(d, need_param_grads=need_param_grads)[0]
        return d[0] if single else d

    def jacobian_transpose_product(self, z, r):
        """``J_Omega(z)^T r`` with the analytic backward pass."""
        zz = np.atleast_2d(z)
        self.net.forward(zz, need_cache=True)
        dz = self.backward(r, need_param_grads=False)
        return dz[0] if np.ndim(z) == 1 else dz

    def mse(self, z, x_flat, reduction: str = "mean"):
        rec = np.atleast_2d(self.reconstruction(z))
        x = np.atleast_2d(x_flat)
        diff = rec - x
        per = np.mean(diff ** 2, axis=-1)
        if reduction == "none":
            return per
        return float(np.mean(per))


# ---------------------------------------------------------------------------
# VAE (decoder pre-training by variational EM / amortised VI)
# ---------------------------------------------------------------------------

class Adam:
    """Adam optimiser operating on a dict of *live* parameter arrays (in-place)."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = float(lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k, p in params.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)


def vae_lr_schedule(step: int, peak: float = VAE_LR_PEAK, warmup_steps: int = VAE_LR_WARMUP_STEPS,
                    decay_steps: int = VAE_LR_DECAY_STEPS, final: float = VAE_LR_FINAL,
                    start: float = VAE_LR_WARMUP_START, mode: str = "cosine") -> float:
    """Decoder pre-training learning rate schedule (Appendix E.6 / addendum).

    Linear warm-up ``start -> peak`` over ``warmup_steps`` steps, followed by a
    cosine (default; linear also available) decay ``peak -> final`` over
    ``decay_steps`` steps, then constant ``final``.
    """
    if step < warmup_steps:
        frac = (step + 1) / max(warmup_steps, 1)
        return float(start + (peak - start) * frac)
    t = step - warmup_steps
    if t >= decay_steps:
        return float(final)
    frac = t / max(decay_steps, 1)
    if mode == "cosine":
        w = 0.5 * (1.0 + np.cos(np.pi * frac))
    else:
        w = 1.0 - frac
    return float(final + (peak - final) * w)


class VAE:
    """Convolutional VAE: factorised Gaussian encoder + Gaussian decoder.

    Matches Section 5.3 / Appendix E.6::

        q(z | x) = N(mu_phi(x), diag(sigma_phi^2(x)))     (factorised Gaussian)
        p(x | z) = N(Omega(z, theta), sigma^2 I),  sigma^2 = 0.1

    The training objective is ``-ELBO`` (up to constants), i.e.

        0.5/sigma^2 ||x - Omega(z)||^2 + KL(q(z|x) || N(0, I)),
    """

    def __init__(self, encoder: Optional[_Sequential] = None, decoder: Optional[_Sequential] = None,
                 latent_dim: int = VAE_LATENT_DIM, image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE,
                 c_hid: int = VAE_C_HID, sigma2: float = VAE_SIGMA2, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        self.encoder = encoder if encoder is not None else make_encoder(latent_dim, image_shape, c_hid, rng=rng)
        self.decoder = decoder if decoder is not None else make_decoder(latent_dim, image_shape, c_hid, rng=rng)
        self.latent_dim = int(latent_dim)
        self.image_shape = tuple(image_shape)
        self.image_dim = int(np.prod(self.image_shape))
        self.sigma2 = float(sigma2)

    # -- parameter plumbing -------------------------------------------------
    def params(self) -> Dict[str, np.ndarray]:
        d = {f"enc/{k}": v for k, v in self.encoder.param_dict().items()}
        d.update({f"dec/{k}": v for k, v in self.decoder.param_dict().items()})
        return d

    def grads(self) -> Dict[str, np.ndarray]:
        d = {f"enc/{k}": v for k, v in self.encoder.grads_dict().items()}
        d.update({f"dec/{k}": v for k, v in self.decoder.grads_dict().items()})
        return d

    def num_params(self) -> int:
        return self.encoder.num_params() + self.decoder.num_params()

    # -- inference network --------------------------------------------------
    def encode(self, x_flat):
        x_img = flat_to_image(np.atleast_2d(x_flat), self.image_shape)
        out = self.encoder.forward(x_img, need_cache=False)
        mu, logvar = out[:, : self.latent_dim], out[:, self.latent_dim :]
        return mu, logvar

    # -- ELBO / gradients ---------------------------------------------------
    def loss_and_grads(self, x_flat, rng=None, mc_sim: int = VAE_MC_SIM, need_grads: bool = True):
        """Returns ``(loss, grads, diagnostics)`` with ``loss = -ELBO + const``."""
        x_flat = np.atleast_2d(x_flat)
        b = x_flat.shape[0]
        x_img = flat_to_image(x_flat, self.image_shape)
        mu, logvar = self.encode_with_cache(x_img)

        rec_sum = 0.0
        kl_sum = 0.0
        loss = 0.0
        if need_grads:
            for layer in self.decoder.layers:
                layer.grads = {}
            for layer in self.encoder.layers:
                layer.grads = {}
        for _ in range(int(mc_sim)):
            eps = rng.standard_normal(mu.shape) if rng is not None else np.random.standard_normal(mu.shape)
            sd = np.exp(0.5 * logvar)
            z = mu + sd * eps
            recon = self.decoder.forward(z, need_cache=True)
            diff = recon - x_img
            rec = 0.5 * np.sum(diff ** 2) / (self.sigma2 * b)
            kl = 0.5 * np.sum(mu ** 2 + np.exp(logvar) - logvar - 1.0) / b
            loss += rec / mc_sim + kl / mc_sim
            rec_sum += rec / mc_sim
            kl_sum += kl / mc_sim
            if need_grads:
                dz = self.decoder.backward(diff / (self.sigma2 * b * mc_sim), need_param_grads=True)
                dmu = dz + mu / (b * mc_sim)
                dlogvar = 0.5 * eps * sd * dz + 0.5 * (np.exp(logvar) - 1.0) / (b * mc_sim)
                self.encoder.backward(dmu, dlogvar, need_param_grads=True)
        grads = self.grads() if need_grads else {}
        diag = {
            "reconstruction_energy": float(rec_sum),
            "kl": float(kl_sum),
            "elbo": float(-rec_sum - kl_sum - 0.5 * self.image_dim * np.log(2.0 * np.pi * self.sigma2)),
        }
        return float(loss), grads, diag

    def encode_with_cache(self, x_img):
        out = self.encoder.forward(x_img, need_cache=True)
        return out[:, : self.latent_dim], out[:, self.latent_dim :]

    def elbo(self, x_flat, rng=None, mc_sim: int = VAE_MC_SIM) -> float:
        _, _, diag = self.loss_and_grads(x_flat, rng=rng, mc_sim=mc_sim, need_grads=False)
        return diag["elbo"]


def train_vae(
    data: "ImageData",
    latent_dim: int = VAE_LATENT_DIM,
    c_hid: int = VAE_C_HID,
    image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE,
    epochs: int = VAE_N_EPOCHS,
    batch_size: int = VAE_BATCH_SIZE,
    sigma2: float = VAE_SIGMA2,
    mc_sim: int = VAE_MC_SIM,
    seed: int = 0,
    lr_peak: float = VAE_LR_PEAK,
    lr_warmup: int = VAE_LR_WARMUP_STEPS,
    lr_decay: int = VAE_LR_DECAY_STEPS,
    lr_final: float = VAE_LR_FINAL,
    lr_start: float = VAE_LR_WARMUP_START,
    max_steps: Optional[int] = None,
    vae: Optional[VAE] = None,
    verbose: bool = True,
) -> Tuple[VAE, Dict[str, list]]:
    """Pre-train the decoder by variational EM (amortised VI), Appendix E.6.

    Optimises the ELBO over the family of factorised Gaussians for 100 epochs
    (``epochs``) with Adam and the warm-up/decay schedule of the addendum.
    Returns the trained :class:`VAE` and a history dict with ``elbo`` per epoch.
    """
    rng = np.random.default_rng(seed)
    if vae is None:
        vae = VAE(latent_dim=latent_dim, image_shape=image_shape, c_hid=c_hid, sigma2=sigma2, rng=rng)
    params = vae.params()
    opt = Adam(params, lr=lr_peak)
    history: Dict[str, list] = {"epoch": [], "elbo": [], "loss": [], "lr": [], "step": []}
    step = 0
    n_train = data.num_train
    for epoch in range(int(epochs)):
        perm = rng.permutation(n_train)
        running = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            if idx.size == 0:
                continue
            xb = data.train[idx]
            lr = vae_lr_schedule(step, peak=lr_peak, warmup_steps=lr_warmup, decay_steps=lr_decay,
                                 final=lr_final, start=lr_start)
            opt.set_lr(lr)
            loss, grads, diag = vae.loss_and_grads(xb, rng=rng, mc_sim=mc_sim, need_grads=True)
            opt.step(params, grads)
            running += diag["elbo"]
            n_batches += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        history["epoch"].append(epoch)
        history["elbo"].append(running / max(n_batches, 1))
        history["loss"].append(-running / max(n_batches, 1))
        history["lr"].append(lr)
        history["step"].append(step)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(
                f"[vae] epoch {epoch + 1:3d}/{epochs}  ELBO={history['elbo'][-1]:12.3f}  lr={lr:.2e}",
                flush=True,
            )
        if max_steps is not None and step >= max_steps:
            break
    return vae, history


def save_vae(path: str, vae: VAE) -> str:
    arrays = {k: v for k, v in vae.params().items()}
    arrays["_meta_latent_dim"] = np.array([vae.latent_dim])
    arrays["_meta_sigma2"] = np.array([vae.sigma2])
    arrays["_meta_image_shape"] = np.array(vae.image_shape)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_vae(path: str, c_hid: int = VAE_C_HID, verbose: bool = False) -> VAE:
    with np.load(path) as z:
        latent_dim = int(z["_meta_latent_dim"][0]) if "_meta_latent_dim" in z else VAE_LATENT_DIM
        sigma2 = float(z["_meta_sigma2"][0]) if "_meta_sigma2" in z else VAE_SIGMA2
        image_shape = tuple(int(s) for s in z["_meta_image_shape"]) if "_meta_image_shape" in z else VAE_IMAGE_SHAPE
        vae = VAE(latent_dim=latent_dim, image_shape=image_shape, c_hid=c_hid, sigma2=sigma2)
        params = vae.params()
        missing = [k for k in params if k not in z.files]
        for k in params:
            if k in z.files:
                params[k][...] = z[k]
        if missing and verbose:
            print(f"[vae] warning: {len(missing)} missing parameter arrays in {path}")
    return vae


# ---------------------------------------------------------------------------
# The target: p(z' | x')  (Section 5.3)
# ---------------------------------------------------------------------------

class VAETarget:
    """Target distribution ``p(z' | x') prop p(z') p(x' | z')`` for one image.

    Parameters
    ----------
    decoder : Decoder | VAE | _Sequential
        The pre-trained neural network ``Omega(. , theta_hat)``.
    x : array
        The new observation ``x'`` (flattened ``(3072,)`` or image-shaped).
    sigma2 : float
        Observation noise variance, ``0.1`` in the paper.
    """

    name = "vae"

    def __init__(self, decoder, x, sigma2: float = VAE_SIGMA2, name: str = "vae",
                 image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE):
        if isinstance(decoder, VAE):
            decoder = decoder.decoder
        if isinstance(decoder, _Sequential):
            lat = decoder.params  # pragma: no cover - defensive
            decoder = Decoder(decoder, latent_dim=self._infer_latent_dim(decoder), image_shape=image_shape)
        elif isinstance(decoder, Decoder):
            image_shape = decoder.image_shape
        else:
            raise TypeError("decoder must be a Decoder, _Sequential or VAE instance")
        self.decoder = decoder
        self.latent_dim = int(decoder.latent_dim)
        self.image_dim = int(np.prod(image_shape))
        self.image_shape = tuple(image_shape)
        self.sigma2 = float(sigma2)
        self.name = name
        x = np.asarray(x, dtype=np.float64)
        self.x_flat = x.reshape(-1) if x.ndim == 1 else x.reshape(x.shape[0], -1)[0]
        if self.x_flat.shape[0] != self.image_dim:
            raise ValueError(f"x has dimension {self.x_flat.shape[0]}, expected {self.image_dim}")
        self.x_image = flat_to_image(self.x_flat, self.image_shape)
        self._laplace_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -- basic specification -------------------------------------------------
    @staticmethod
    def _infer_latent_dim(net: _Sequential) -> int:  # pragma: no cover - defensive
        for layer in net.layers:
            if isinstance(layer, Linear):
                return layer.fan_in if layer is net.layers[0] else VAE_LATENT_DIM
        return VAE_LATENT_DIM

    @property
    def dim(self) -> int:
        return self.latent_dim

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"VAETarget(dim={self.dim}, image_dim={self.image_dim}, sigma2={self.sigma2})"

    # -- densities -----------------------------------------------------------
    def log_prior(self, z):
        z = np.atleast_2d(z)
        return -0.5 * np.sum(z ** 2, axis=-1) - 0.5 * self.dim * np.log(2.0 * np.pi)

    def log_likelihood(self, z):
        """log N(x' ; Omega(z), sigma^2 I)."""
        zz = np.atleast_2d(z)
        rec = self.decoder.reconstruction(zz)
        diff = self.x_flat[None, :] - rec
        quad = np.sum(diff ** 2, axis=-1)
        return -0.5 * quad / self.sigma2 - 0.5 * self.image_dim * np.log(2.0 * np.pi * self.sigma2)

    def log_prob(self, z):
        """Un-normalised (up to a constant) log target ``log p(z') + log p(x'|z')``."""
        out = self.log_prior(z) + self.log_likelihood(z)
        return float(out[0]) if np.ndim(z) == 1 else out

    def grad_log_prob(self, z):
        return self.score(z)

    def score(self, z):
        """Exact score ``-z + (1/sigma^2) J_Omega(z)^T (x' - Omega(z))``.

        Supports ``z`` of shape ``(D,)`` (returns ``(D,)``) and ``(B, D)``
        (returns ``(B, D)``), which is the signature expected by
        :class:`bam_repro.bam.bam.BaM` and the baseline algorithms.
        """
        single = np.asarray(z).ndim == 1
        zz = np.atleast_2d(np.asarray(z, dtype=np.float64))
        rec = self.decoder.forward(zz, need_cache=True)  # caches activations
        r = (self.x_image[None, :] - rec) / self.sigma2  # (B, 3, 32, 32)
        dz = self.decoder.backward(r, need_param_grads=False)
        out = dz - zz
        return out[0] if single else out

    # -- reconstruction metrics (Figure 5.4) ---------------------------------
    def reconstruction(self, z):
        return self.decoder.reconstruction(z)

    def reconstruction_mse(self, z) -> float:
        """MSE of ``Omega(E[z' | x'])`` against ``x'`` (Section 5.3 metric)."""
        single = np.asarray(z).ndim == 1
        zz = np.atleast_2d(np.asarray(z, dtype=np.float64))
        rec = self.decoder.reconstruction(zz)
        mse = np.mean((rec - self.x_flat[None, :]) ** 2, axis=-1)
        return float(mse[0]) if single else float(np.mean(mse))

    # -- curvature / Laplace approximation -----------------------------------
    def hessian_vector_product(self, z, v, fd_step: float = 1e-3):
        """``H v`` for ``H = I + (1/sigma^2) J^T J`` (Gauss-Newton Hessian + prior).

        ``J v`` is obtained with a central finite difference through the decoder
        (one extra forward pass) and ``J^T (J v)`` with the exact backward pass.
        """
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        jv = (self.decoder.reconstruction(z + fd_step * v) - self.decoder.reconstruction(z - fd_step * v)) / (2.0 * fd_step)
        self.decoder.forward(np.atleast_2d(z), need_cache=True)
        jtjv = self.decoder.backward(np.atleast_2d(jv), need_param_grads=False)[0]
        return v + jtjv / self.sigma2

    def hessian(self, z, fd_step: float = 1e-3) -> np.ndarray:
        d = self.dim
        H = np.empty((d, d))
        for i in range(d):
            e = np.zeros(d)
            e[i] = 1.0
            H[:, i] = self.hessian_vector_product(z, e, fd_step=fd_step)
        return 0.5 * (H + H.T)

    # -- sampling ------------------------------------------------------------
    def mode(self, n_iter: int = 300, lr: float = 0.05, tol: float = 1e-6) -> np.ndarray:
        """MAP estimate by gradient ascent on ``log p(z' | x')``."""
        z = np.zeros(self.dim)
        for _ in range(int(n_iter)):
            g = self.score(z)
            step = lr * g
            z_new = z + step
            if np.linalg.norm(step) < tol:
                z = z_new
                break
            z = z_new
        return z

    def laplace_posterior(self, n_iter: int = 300, lr: float = 0.05, cg_iters: int = 64,
                          tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """Laplace/Gaussian approximation ``N(mode, H^{-1})`` of ``p(z' | x')``.

        ``H = I + (1/sigma^2) J^T J`` is SPD; the covariance is obtained with
        preconditioned-free conjugate gradients on :meth:`hessian_vector_product`
        (matrix-free), so no ``3072 x 256`` Jacobian is ever materialised.
        """
        z = self.mode(n_iter=n_iter, lr=lr)
        d = self.dim
        # solve H X = I column-by-column with CG (d small: 256)
        Sigma = np.empty((d, d))
        for i in range(d):
            b = np.zeros(d)
            b[i] = 1.0
            Sigma[:, i] = _conjugate_gradient(lambda v: self.hessian_vector_product(z, v), b,
                                              max_iter=cg_iters, tol=tol)
        Sigma = 0.5 * (Sigma + Sigma.T)
        w, Q = np.linalg.eigh(Sigma)
        w = np.clip(w, _MIN_EIG, None)
        Sigma = (Q * w) @ Q.T
        self._laplace_cache = (z, Sigma)
        return z, Sigma

    def sample(self, n: int = 1, rng=None, dtype=None):
        """Samples from the Laplace approximation (lazily computed and cached).

        The paper samples from ``p(z' | x')``; the Laplace approximation is a
        documented stand-in that avoids running MCMC inside the target and is
        only used by utilities that need target draws (the reported metric of
        Section 5.3 is the reconstruction MSE, which needs no target samples).
        """
        if self._laplace_cache is None:
            self.laplace_posterior()
        mu, Sigma = self._laplace_cache
        rng = np.random.default_rng(0) if rng is None else rng
        eps = rng.standard_normal((int(n), self.dim))
        L = np.linalg.cholesky(Sigma + _MIN_EIG * np.eye(self.dim))
        z = mu[None, :] + eps @ L.T
        z = z.astype(dtype) if dtype is not None else z
        return z[0] if int(n) == 1 else z

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "image_dim": self.image_dim,
            "sigma2": self.sigma2,
            "n_decoder_params": int(self.decoder.num_params()),
        }


# Alias kept for symmetry with the other target modules
VAEPosteriorTarget = VAETarget


def vae_target_from_image(decoder, x, sigma2: float = VAE_SIGMA2, **kwargs) -> VAETarget:
    """Convenience constructor mirroring ``gaussian_target``-style factories."""
    return VAETarget(decoder, x, sigma2=sigma2, **kwargs)


def score_function(decoder, x, sigma2: float = VAE_SIGMA2) -> Callable[[np.ndarray], np.ndarray]:
    """Returns the bare ``score(z)`` callable expected by BaM / baselines."""
    target = VAETarget(decoder, x, sigma2=sigma2)
    return target.score


def _conjugate_gradient(matvec, b, max_iter: int = 64, tol: float = 1e-8):
    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
    rs = float(r @ r)
    if rs < tol:
        return x
    for _ in range(int(max_iter)):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if denom <= 0.0:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if rs_new < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


# ---------------------------------------------------------------------------
# Amortised variational inference (AVI) reference
# ---------------------------------------------------------------------------

def amortized_posterior(encoder, x_flat, image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE,
                        sigma2: float = VAE_SIGMA2):
    """Factorised Gaussian ``q(z | x) = N(mu, diag(var))`` from the encoder.

    Returns ``(mu, var, Sigma)`` with ``Sigma = diag(var)`` (the AVI baseline of
    Section 5.3 uses a factorised Gaussian, unlike BaM/ADVI).
    """
    if isinstance(encoder, VAE):
        encoder = encoder.encoder
    vae_shim = VAE(encoder=encoder, decoder=make_decoder(image_shape=image_shape), sigma2=sigma2)
    mu, logvar = vae_shim.encode(x_flat)
    var = np.exp(logvar)
    Sigma = np.diag(var[0])
    return mu[0], var[0], Sigma


def amortized_reconstruction(decoder, encoder, x_flat, image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE):
    """``Omega(E_q[z | x'])`` using the amortised encoder mean."""
    mu, _, _ = amortized_posterior(encoder, x_flat, image_shape=image_shape)
    dec = decoder if isinstance(decoder, Decoder) else (decoder.decoder if isinstance(decoder, VAE) else Decoder(decoder))
    return dec.reconstruction(mu)


def amortized_reconstruction_mse(decoder, encoder, x_flat, image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE) -> float:
    """AVI reconstruction MSE (used as the "AVI" reference in Figure 5.4)."""
    rec = amortized_reconstruction(decoder, encoder, x_flat, image_shape=image_shape)
    x = np.asarray(x_flat).reshape(-1)
    return float(np.mean((rec - x) ** 2))


def laplace_posterior(target: VAETarget, **kwargs):
    """Module-level alias of :meth:`VAETarget.laplace_posterior`."""
    if not isinstance(target, VAETarget):
        raise TypeError("laplace_posterior expects a VAETarget instance")
    return target.laplace_posterior(**kwargs)


# ---------------------------------------------------------------------------
# CIFAR-10 data (Section 5.3) with a documented synthetic fallback
# ---------------------------------------------------------------------------

_CIFAR10_BATCH_FILES = [f"data_batch_{i}" for i in range(1, 6)]
_CIFAR10_TEST_FILE = "test_batch"
_DEFAULT_CIFAR_DIRS = (
    "cifar-10-batches-py",
    "data/cifar-10-batches-py",
    "data/cifar10/cifar-10-batches-py",
    os.path.join("~", "cifar-10-batches-py"),
    os.path.join("~", ".keras", "datasets", "cifar-10-batches-py"),
)


def find_cifar10_dir(root: Optional[str] = None) -> Optional[str]:
    """Locate the CIFAR-10 python batches directory, or return ``None``."""
    candidates: List[str] = []
    if root is not None:
        candidates.append(root)
        candidates.append(os.path.join(root, "cifar-10-batches-py"))
    env = os.environ.get("BAM_CIFAR10_DIR")
    if env:
        candidates.append(env)
    candidates.extend(_DEFAULT_CIFAR_DIRS)
    for cand in candidates:
        if not cand:
            continue
        path = os.path.expanduser(cand)
        if os.path.isfile(os.path.join(path, _CIFAR10_TEST_FILE)):
            return path
    return None


@dataclass
class ImageData:
    """Flat image dataset with the paper's ``[-1, 1]`` scaling."""

    train: np.ndarray
    test: np.ndarray
    name: str = "cifar10"
    image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE

    def __post_init__(self):
        self.train = np.asarray(self.train, dtype=np.float64).reshape(len(self.train), -1)
        self.test = np.asarray(self.test, dtype=np.float64).reshape(len(self.test), -1)

    @property
    def num_train(self) -> int:
        return int(self.train.shape[0])

    @property
    def num_test(self) -> int:
        return int(self.test.shape[0])

    @property
    def image_dim(self) -> int:
        return int(self.train.shape[1])

    def batches(self, batch_size: int, rng=None, shuffle: bool = True) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(0) if rng is None else rng
        idx = np.arange(self.num_train)
        if shuffle:
            idx = rng.permutation(idx)
        for start in range(0, idx.size, int(batch_size)):
            chunk = idx[start : start + int(batch_size)]
            if chunk.size:
                yield self.train[chunk]

    def sample_test(self, rng=None) -> np.ndarray:
        """Draw a single test image ``x'`` (Section 5.3 protocol)."""
        rng = np.random.default_rng(0) if rng is None else rng
        i = int(rng.integers(self.num_test))
        return self.test[i]

    def reconstruction_mse(self, x: np.ndarray, x_hat: np.ndarray) -> float:
        return float(np.mean((np.asarray(x_hat).reshape(-1) - np.asarray(x).reshape(-1)) ** 2))


def synthetic_images(n: int, seed: int = 0, image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE) -> np.ndarray:
    """Documented synthetic image surrogate (smoothed low-frequency patterns).

    Used only when the CIFAR-10 files are not present, so that the pipeline
    (decoder pre-training, BaM/ADVI/GSM comparison, MSE curves) remains fully
    runnable.  Values live in ``[-1, 1]`` exactly like the real pipeline.
    """
    rng = np.random.default_rng(seed)
    c, h, w = image_shape
    low = rng.normal(0.0, 1.0, size=(int(n), c, 4, 4))
    # nearest-neighbour upsampling by 8 (4 -> 32) gives smooth, learnable images
    up = np.repeat(np.repeat(low, h // 4, axis=2), w // 4, axis=3)
    noise = rng.normal(0.0, 0.15, size=up.shape)
    x = np.tanh(up + noise)
    return x.reshape(int(n), -1)


def synthetic_image_data(n_train: int = 512, n_test: int = 64, seed: int = 0,
                         image_shape: Tuple[int, int, int] = VAE_IMAGE_SHAPE) -> ImageData:
    return ImageData(
        train=synthetic_images(n_train, seed=seed, image_shape=image_shape),
        test=synthetic_images(n_test, seed=seed + 1, image_shape=image_shape),
        name="synthetic-images",
        image_shape=image_shape,
    )


def load_cifar10(root: Optional[str] = None, normalize: bool = True, subset: Optional[int] = None,
                 allow_synthetic: bool = True, seed: int = 0) -> ImageData:
    """Load CIFAR-10 as ``(N, 3072)`` floats scaled to ``[-1, 1]``.

    The images are modelled as continuous (Section 5.3) and the ``[-1, 1]``
    scaling matches the ``tanh`` decoder output.  Falls back to a synthetic
    image surrogate when the dataset is unavailable and ``allow_synthetic``.
    """
    path = find_cifar10_dir(root)
    if path is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "CIFAR-10 python batches not found. Set BAM_CIFAR10_DIR or pass root=..., or "
                "use allow_synthetic=True."
            )
        return synthetic_image_data(seed=seed)
    train_parts, test_parts = [], []
    for name in _CIFAR10_BATCH_FILES:
        with open(os.path.join(path, name), "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        train_parts.append(d["data"].reshape(-1, 3, 32, 32))
    with open(os.path.join(path, _CIFAR10_TEST_FILE), "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    test_parts.append(d["data"].reshape(-1, 3, 32, 32))
    train = np.concatenate(train_parts, axis=0).astype(np.float64)
    test = np.concatenate(test_parts, axis=0).astype(np.float64)
    if normalize:
        train = train / 127.5 - 1.0
        test = test / 127.5 - 1.0
    if subset is not None:
        train = train[: int(subset)]
        test = test[: max(1, int(subset) // 8)]
    return ImageData(train=train, test=test, name="cifar10")


# ---------------------------------------------------------------------------
# Self-check (executed only when run as a script)
# ---------------------------------------------------------------------------

def _self_check(seed: int = 0) -> Dict[str, float]:  # pragma: no cover - diagnostic
    """Finite differences vs. analytic score, and a tiny end-to-end pre-training."""
    rng = np.random.default_rng(seed)
    target = VAETarget(make_decoder(latent_dim=16, c_hid=8, rng=rng).__class__ and Decoder(
        make_decoder(latent_dim=16, c_hid=8, rng=rng), latent_dim=16), synthetic_images(1, seed=seed))
    z = rng.normal(0.0, 0.1, size=target.dim)
    g = target.score(z)
    fd = np.zeros_like(z)
    eps = 1e-5
    for i in range(target.dim):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd[i] = (target.log_prob(zp) - target.log_prob(zm)) / (2 * eps)
    return {"max_score_error": float(np.max(np.abs(g - fd)))}


if __name__ == "__main__":  # pragma: no cover
    print(_self_check())
