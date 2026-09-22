"""Data-dependent couplings rho(x_0, x_1) between the base and the target.

Section 3.2 of the paper ("Designing data-dependent couplings") proposes

    rho(x_0, x_1) = rho_1(x_1) rho_0(x_0 | x_1),
    x_0 = m(x_1) + sigma zeta,      zeta ~ N(0, Id),  zeta _||_ m(x_1),

with m(x_1) a (possibly randomised) corrupted observation of the target and
sigma a (possibly state-dependent) scale.  The four couplings used in the
paper and implemented here are

=====================  =================================================
name                   construction
=====================  =================================================
independent            x_0 ~ N(0, Id), independent of x_1  (baseline)
data_decorruption      x_0 = x_1 + sigma zeta              (Sec. 3.3)
inpainting             x_0 = xi o x_1 + (1 - xi) o zeta   (Sec. 4.1)
super_resolution       x_0 = U(D(x_1)) + sigma zeta       (Sec. 4.2)
=====================  =================================================

Every coupling returns a :class:`CoupledBatch` holding the pair (x_0, x_1)
together with the conditioning signal xi that is fed to the velocity model
(appended to the channel dimension of x_t, as in Appendix B), and the
elementwise mask that is used to project the model output (in-painting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CoupledBatch:
    """A batch of coupled (x_0, x_1) pairs plus the model conditioning."""

    x0: Tensor
    x1: Tensor
    cond: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    mask: Optional[Tensor] = None  # 1 where the velocity output is kept
    sigma: Tensor | float = 0.0
    extra: dict = field(default_factory=dict)

    def to(self, device) -> "CoupledBatch":
        return CoupledBatch(
            x0=self.x0.to(device),
            x1=self.x1.to(device),
            cond=None if self.cond is None else self.cond.to(device),
            labels=None if self.labels is None else self.labels.to(device),
            mask=None if self.mask is None else self.mask.to(device),
            sigma=self.sigma,
            extra={k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.extra.items()},
        )


class Coupling(torch.nn.Module):
    """Base class for couplings.  Sub-classes implement :meth:`sample_base`."""

    name: str = "coupling"
    #: whether the coupling carries its own randomness (sigma zeta) so that
    #: the interpolant can be used with gamma_t = 0
    self_noised: bool = True
    #: channels appended to the model input as conditioning
    cond_channels: int = 0
    #: whether the model output should be masked (in-painting)
    masks_output: bool = False

    def sample_base(self, x1: Tensor, **kwargs) -> tuple:  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, x1: Tensor, labels: Optional[Tensor] = None, **kwargs) -> CoupledBatch:
        x0, cond, mask, sigma = self.sample_base(x1, **kwargs)
        return CoupledBatch(x0=x0, x1=x1, cond=cond, labels=labels, mask=mask, sigma=sigma)

    @property
    def info(self) -> dict:
        return {
            "name": self.name,
            "self_noised": self.self_noised,
            "cond_channels": self.cond_channels,
            "masks_output": self.masks_output,
        }


class IndependentCoupling(Coupling):
    """rho(x_0, x_1) = rho_0(x_0) rho_1(x_1) with rho_0 = N(0, Id).

    This is the standard (data-agnostic) coupling used as the baseline in
    Tables 2 and 3 and in the right panel of Figure 2.
    """

    name = "independent"
    self_noised = False
    cond_channels = 0

    def __init__(self, base_scale: float = 1.0, channels: int = 3, image_size: int = 256):
        super().__init__()
        self.base_scale = float(base_scale)
        self.channels = int(channels)
        self.image_size = int(image_size)
        self._base_like: Optional[Tensor] = None

    def _base(self, x1: Tensor) -> Tensor:
        if self._base_like is None or self._base_like.shape[0] < x1.shape[0]:
            self._base_like = torch.randn(x1.shape[0], 1, 1, 1, dtype=x1.dtype, device=x1.device)
        return torch.randn_like(x1) * self.base_scale

    def sample_base(self, x1: Tensor, **kwargs):
        x0 = torch.randn_like(x1) * self.base_scale
        return x0, None, None, 0.0


class GaussianAdaptedCoupling(Coupling):
    """x_0 = m(x_1) + sigma zeta (eq. 18).

    ``m_fn`` maps a batch of targets to their corrupted observations; it may
    be randomised (it is allowed to depend on additional conditioning
    information).  ``sigma`` is either a scalar or a tensor broadcastable
    against ``x1``.
    """

    name = "gaussian_adapted"
    self_noised = True
    cond_channels = 0

    def __init__(self, m_fn=None, sigma: float = 1.0, cond_fn=None, masks_output: bool = False):
        super().__init__()
        self.m_fn = m_fn if m_fn is not None else (lambda x: x)
        self.sigma = float(sigma)
        self.cond_fn = cond_fn
        self.masks_output = masks_output

    def sample_base(self, x1: Tensor, **kwargs):
        m = self.m_fn(x1)
        x0 = m + self.sigma * torch.randn_like(x1)
        cond = None if self.cond_fn is None else self.cond_fn(x1, m)
        return x0, cond, None, self.sigma


class DataDecorruptionCoupling(GaussianAdaptedCoupling):
    """x_0 = x_1 + sigma zeta, with C = sigma^2 Id (Sec. 3.3).

    This is the coupling for which the paper gives the explicit transport
    cost comparison of eq. (21):

        E|I_dot_t|^2 (coupled)     = d sigma^2
        E|I_dot_t|^2 (independent) = 2 E|x_1|^2 + d sigma^2

    for alpha_t = 1 - t, beta_t = t and gamma_t = 0.
    """

    name = "data_decorruption"

    def __init__(self, sigma: float = 0.5):
        super().__init__(m_fn=lambda x: x, sigma=sigma)


def tile_mask(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    n_tiles: int = 8,
    p_missing: float = 0.3,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Sample the 64-tile in-painting mask of Section 4.1.

    "During training, the mask is drawn randomly by tiling the image into 64
    tiles; each tile is selected to enter the mask with probability p = 0.3."

    To keep the notation of the paper, in which the conditioning variable
    satisfies x_0 = xi o x_1 + (1 - xi) o zeta, ``xi`` is the *observed*
    indicator: ``xi = 1`` where the image is known (x_0 = x_1) and ``xi = 0``
    on the missing tiles (which are re-drawn as noise).  ``n_tiles`` is the
    number of tiles per side, so the default ``n_tiles=8`` gives 8 x 8 = 64
    tiles.  The mask value is shared across channels at a given spatial
    location, as stated in the paper.

    Returns
    -------
    xi: Tensor of shape (B, 1, H, W) taking values in {0, 1}.
    """
    if height % n_tiles != 0 or width % n_tiles != 0:
        raise ValueError(
            f"image size {(height, width)} is not divisible into {n_tiles} x {n_tiles} tiles"
        )
    thr = torch.rand(batch_size, 1, n_tiles, n_tiles, device=device)
    missing_tiles = (thr < p_missing).to(dtype)  # 1 = tile is missing
    xi = 1.0 - missing_tiles
    tile_h, tile_w = height // n_tiles, width // n_tiles
    xi = xi.repeat_interleave(tile_h, dim=2).repeat_interleave(tile_w, dim=3)
    return xi


class InpaintingCoupling(Coupling):
    """x_0 = xi o x_1 + (1 - xi) o zeta (Sec. 4.1).

    The unmasked pixels of the interpolant are identical to those of x_1 for
    every t (because alpha_t + beta_t = 1), hence I_dot_t = 0 there and the
    velocity model output can be masked to the missing region.  This is
    implemented through ``masks_output=True`` and the ``mask`` returned by
    :meth:`sample_base` (which equals 1 - xi).
    """

    name = "inpainting"
    self_noised = True
    cond_channels = 3  # xi, repeated over the channel dimension of x_t

    def __init__(self, n_tiles: int = 8, p_missing: float = 0.3, channels: int = 3):
        super().__init__()
        self.n_tiles = int(n_tiles)
        self.p_missing = float(p_missing)
        self.channels = int(channels)
        self.cond_channels = int(channels)
        self.masks_output = True

    def sample_base(self, x1: Tensor, xi: Optional[Tensor] = None, **kwargs):
        b, c, h, w = x1.shape
        if xi is None:
            xi = tile_mask(b, c, h, w, self.n_tiles, self.p_missing, x1.device, x1.dtype)
        xi = xi.expand(b, c, h, w) if xi.shape[1] == 1 else xi
        zeta = torch.randn_like(x1)
        x0 = xi * x1 + (1.0 - xi) * zeta
        mask = 1.0 - xi  # the velocity is non-zero only in the missing region
        return x0, xi, mask, 1.0

    def conditioning(self, xi: Tensor) -> Tensor:
        """xi broadcast to the channel dimension, for appending to x_t."""
        return xi.expand(-1, self.channels, -1, -1) if xi.shape[1] == 1 else xi


class SuperResolutionCoupling(Coupling):
    """x_0 = U(D(x_1)) + sigma zeta, with xi = U(D(x_1)) (Sec. 4.2).

    ``scale`` is the down-sampling factor, e.g. ``scale=4`` maps 256x256
    targets to a 64x64 low-resolution image which is then up-sampled back to
    256x256.  The up-sampled low-resolution image xi is appended to the
    channels of x_t and is also the mean of the base density.
    """

    name = "super_resolution"
    self_noised = True
    cond_channels = 3

    def __init__(self, scale: int = 4, sigma: float = 0.5, down_mode: str = "area", up_mode: str = "bicubic"):
        super().__init__()
        self.scale = int(scale)
        self.sigma = float(sigma)
        self.down_mode = down_mode
        self.up_mode = up_mode

    # -- image operators --------------------------------------------------
    def downsample(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        return F.interpolate(
            x,
            size=(h // self.scale, w // self.scale),
            mode=self.down_mode,
            antialias=True if self.down_mode in ("bilinear", "bicubic") else None,
        )

    def upsample(self, x: Tensor, size=None) -> Tensor:
        return F.interpolate(x, size=size, scale_factor=None if size is not None else self.scale,
                             mode=self.up_mode, align_corners=False)

    def sample_base(self, x1: Tensor, xi: Optional[Tensor] = None, **kwargs):
        if xi is None:
            xi = self.upsample(self.downsample(x1), size=x1.shape[-2:])
        x0 = self.base_from_xi(xi)
        return x0, xi, None, self.sigma

    def base_from_xi(self, xi: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        """x_0 = xi + sigma zeta (the coupled base)."""
        return xi + self.sigma * torch.randn(
            xi.shape, device=xi.device, dtype=xi.dtype, generator=generator
        )


class SuperResolutionBaselineCoupling(SuperResolutionCoupling):
    """The uncoupled counterpart of the super-resolution task.

    The conditioning signal is still the up-sampled low-resolution image
    xi = U(D(x_1)) (a conditional generative model is allowed to see it), but
    the base density is the data-agnostic Gaussian rho_0 = N(0, Id) with an
    independent coupling.  Comparing with
    :class:`SuperResolutionCoupling` isolates the effect of replacing the
    Gaussian base by the coupled one.
    """

    name = "super_resolution_baseline"
    self_noised = False

    def __init__(self, scale: int = 4, sigma: float = 0.5, base_scale: float = 1.0,
                 down_mode: str = "area", up_mode: str = "bicubic"):
        super().__init__(scale=scale, sigma=sigma, down_mode=down_mode, up_mode=up_mode)
        self.base_scale = float(base_scale)

    def base_from_xi(self, xi: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        return self.base_scale * torch.randn(
            xi.shape, device=xi.device, dtype=xi.dtype, generator=generator
        )


def build_coupling(name: str, **kwargs) -> Coupling:
    """Factory used by the configuration files."""
    name = name.lower()
    table = {
        "independent": IndependentCoupling,
        "baseline": IndependentCoupling,
        "gaussian": IndependentCoupling,
        "inpainting": InpaintingCoupling,
        "super_resolution": SuperResolutionCoupling,
        "superresolution": SuperResolutionCoupling,
        "superres": SuperResolutionCoupling,
        "super_resolution_baseline": SuperResolutionBaselineCoupling,
        "superres_baseline": SuperResolutionBaselineCoupling,
        "data_decorruption": DataDecorruptionCoupling,
        "decorruption": DataDecorruptionCoupling,
        "gaussian_adapted": GaussianAdaptedCoupling,
    }
    if name not in table:
        raise ValueError(f"unknown coupling {name!r}")
    cls = table[name]
    # configurations share a single ``coupling`` block, so silently drop the
    # options that do not belong to the selected coupling
    import inspect

    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**kwargs)


__all__ = [
    "CoupledBatch",
    "Coupling",
    "IndependentCoupling",
    "GaussianAdaptedCoupling",
    "DataDecorruptionCoupling",
    "InpaintingCoupling",
    "SuperResolutionCoupling",
    "SuperResolutionBaselineCoupling",
    "tile_mask",
    "build_coupling",
]
