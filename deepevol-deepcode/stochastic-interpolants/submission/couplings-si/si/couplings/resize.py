"""Image downsampling / upsampling operators for the super-resolution coupling.

Implements the operators from Section 4.2 of *Stochastic Interpolants with
Data-Dependent Couplings*:

    D : R^{C x W x H} -> R^{C x W_low x H_low}      (downsampling)
    U : R^{C x W_low x H_low} -> R^{C x W x H}      (upsampling)

which are used to define the coupled base sample

    x_0 = U(D(x_1)) + sigma * zeta,   zeta ~ N(0, Id),  sigma > 0

and the model conditioning

    xi = U(D(x_1))

(the up-sampled low-resolution image, appended to the model input channels).

The natural pairing with the stochastic-interpolant framework is the base mean
``m(x_1) = U(D(x_1))``: with ``sigma > 0`` the base density is well defined over
the whole ambient space (a small amount of Gaussian noise smooths it off the
low-dimensional manifold obtained when ``sigma = 0``).

This module is deliberately dependency-light: all operators are implemented on
top of :mod:`torch.nn.functional` so that ``D`` and ``U`` can be applied on the
fly inside the training loop (no PIL / torchvision round-trip, exact gradients,
fully device-agnostic and batched).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Size",
    "normalize_size",
    "Downsample",
    "Upsample",
    "ResizePair",
    "downsample",
    "upsample",
    "default_superres_pair",
    "low_resolution",
]

Size = Union[int, Sequence[int]]

#: Supported interpolation modes (forwarded to ``torch.nn.functional.interpolate``).
SUPPORTED_MODES = ("nearest", "bilinear", "bicubic", "area")


def normalize_size(
    size: Size,
    x: Optional[torch.Tensor] = None,
    spatial_dims: int = 2,
) -> Tuple[int, ...]:
    """Normalize a size specification to a tuple of ``spatial_dims`` ints.

    Parameters
    ----------
    size:
        An int (interpreted as a square spatial size) or a sequence of ints.
        The sequence is interpreted as ``(..., H, W)`` in PyTorch's *spatial*
        ordering convention (i.e. last two entries are height and width), but
        since :mod:`torch.nn.functional.interpolate` accepts exactly that
        ordering, sizes are forwarded unchanged.
    x:
        Optional reference tensor. If ``size`` is ``None``, ``x.shape[-spatial_dims:]``
        is used.
    spatial_dims:
        Number of spatial dimensions (2 for images).
    """
    if size is None:
        if x is None:
            raise ValueError("Either `size` or a reference tensor `x` must be given.")
        return tuple(int(s) for s in x.shape[-spatial_dims:])

    if isinstance(size, (int, torch.Tensor)) and not isinstance(size, torch.Tensor):
        return tuple(int(size) for _ in range(spatial_dims))

    if isinstance(size, torch.Tensor):
        if size.numel() == 1:
            return tuple(int(size.item()) for _ in range(spatial_dims))
        return tuple(int(s) for s in size.reshape(-1).tolist())[-spatial_dims:]

    if isinstance(size, (list, tuple)):
        items = [int(s) for s in size]
        if len(items) == 1:
            return tuple(items[0] for _ in range(spatial_dims))
        if len(items) < spatial_dims:
            # e.g. (H, W) passed as a 2-sequence is the common case; anything
            # shorter is left-padded with the first entry.
            items = [items[0]] * (spatial_dims - len(items)) + items
        return tuple(items[-spatial_dims:])

    raise TypeError(f"Unsupported size specification: {size!r}")


def _check_input(x: torch.Tensor, name: str = "x") -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(x)}.")
    if x.dim() < 3:
        raise ValueError(
            f"`{name}` must have at least 3 dims (C, H, W), got shape {tuple(x.shape)}."
        )
    return x


# --------------------------------------------------------------------------- #
# Functional operators
# --------------------------------------------------------------------------- #
def downsample(
    x: torch.Tensor,
    size: Optional[Size] = None,
    mode: str = "area",
    antialias: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Downsample ``x`` (``D`` in Section 4.2).

    Parameters
    ----------
    x:
        Tensor of shape ``(B, C, H, W)`` (or ``(C, H, W)``).
    size:
        Target low-resolution spatial size. Either an int (square) or ``(H, W)``.
        Mutually exclusive with ``scale``.
    mode:
        One of ``"area"``, ``"bilinear"``, ``"bicubic"``, ``"nearest"``.
        ``"area"`` = average pooling (the natural low-pass downsampling and the
        default used by this implementation), matching the ``D`` used to build
        the low-resolution conditioning image.
    antialias:
        Apply anti-aliasing when using a (bi)linear/(bi)cubic filter. Ignored
        for ``"area"`` and ``"nearest"`` which are exact for their own support.
    scale:
        Whereas ``size = round(H / scale)``; convenient for integer factors.
    """
    x = _check_input(x)

    if size is None and scale is None:
        raise ValueError("One of `size` or `scale` must be provided to `downsample`.")
    if size is not None and scale is not None:
        raise ValueError("`size` and `scale` are mutually exclusive.")

    if scale is not None:
        if scale < 1:
            raise ValueError(f"Downsampling scale must be >= 1, got {scale}.")
        h, w = int(x.shape[-2]), int(x.shape[-1])
        size = (max(1, int(round(h / scale))), max(1, int(round(w / scale))))

    out_size = normalize_size(size, spatial_dims=2)
    in_size = (int(x.shape[-2]), int(x.shape[-1]))

    if out_size == in_size:
        return x

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported downsample mode {mode!r}; expected one of {SUPPORTED_MODES}.")

    if mode == "area":
        # Average pooling requires divisibility; fall back to interpolate's
        # (adaptive) area kernel otherwise so that arbitrary factor-ratios work.
        if in_size[0] % out_size[0] == 0 and in_size[1] % out_size[1] == 0:
            return F.avg_pool2d(x, kernel_size=(in_size[0] // out_size[0], in_size[1] // out_size[1]))
        return F.interpolate(x, size=out_size, mode="area")

    return F.interpolate(
        x,
        size=out_size,
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
        antialias=bool(antialias) if mode in ("bilinear", "bicubic") else False,
    )


def upsample(
    x: torch.Tensor,
    size: Optional[Size] = None,
    mode: str = "bilinear",
    antialias: bool = False,
    scale: Optional[float] = None,
    align_corners: bool = False,
) -> torch.Tensor:
    """Upsample ``x`` (``U`` in Section 4.2). Inverse-style companion of :func:`downsample`.

    Parameters
    ----------
    x:
        Low-resolution tensor of shape ``(B, C, H_low, W_low)``.
    size:
        Target high-resolution spatial size (int or ``(H, W)``).
    mode:
        Interpolation used to synthesize the full-resolution conditioning image.
        ``"bilinear"`` is the default (smooth, differentiable and, as used in the
        paper, appends the upsampled low-res image to the U-Net input channels).
    scale:
        Whereas ``size = round(H * scale)``.
    """
    x = _check_input(x)

    if size is None and scale is None:
        raise ValueError("One of `size` or `scale` must be provided to `upsample`.")
    if size is not None and scale is not None:
        raise ValueError("`size` and `scale` are mutually exclusive.")

    if scale is not None:
        if scale < 1:
            raise ValueError(f"Upsampling scale must be >= 1, got {scale}.")
        h, w = int(x.shape[-2]), int(x.shape[-1])
        size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))

    out_size = normalize_size(size, spatial_dims=2)
    in_size = (int(x.shape[-2]), int(x.shape[-1]))

    if out_size == in_size:
        return x

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported upsample mode {mode!r}; expected one of {SUPPORTED_MODES}.")

    if mode == "area":
        raise ValueError("`mode='area'` is not a valid upsampling mode; use 'bilinear'/'nearest'.")

    return F.interpolate(
        x,
        size=out_size,
        mode=mode,
        align_corners=align_corners if mode in ("bilinear", "bicubic") else None,
        antialias=bool(antialias) if mode in ("bilinear", "bicubic") else False,
    )


# --------------------------------------------------------------------------- #
# nn.Module wrappers
# --------------------------------------------------------------------------- #
class Downsample(nn.Module):
    """``nn.Module`` wrapper around :func:`downsample` implementing ``D``.

    Either a fixed ``size``/``scale`` is given, or a ``low_res`` size which is
    resolved relative to the input at call time (see :class:`ResizePair`).
    """

    def __init__(
        self,
        size: Optional[Size] = None,
        scale: Optional[float] = None,
        mode: str = "area",
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale = scale
        self.mode = mode
        self.antialias = antialias

    def forward(self, x: torch.Tensor, size: Optional[Size] = None) -> torch.Tensor:
        target = size if size is not None else self.size
        return downsample(x, size=target, mode=self.mode, antialias=self.antialias, scale=self.scale)

    def extra_repr(self) -> str:
        return f"size={self.size}, scale={self.scale}, mode={self.mode}, antialias={self.antialias}"


class Upsample(nn.Module):
    """``nn.Module`` wrapper around :func:`upsample` implementing ``U``."""

    def __init__(
        self,
        size: Optional[Size] = None,
        scale: Optional[float] = None,
        mode: str = "bilinear",
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale = scale
        self.mode = mode
        self.antialias = antialias

    def forward(self, x: torch.Tensor, size: Optional[Size] = None) -> torch.Tensor:
        target = size if size is not None else self.size
        return upsample(x, size=target, mode=self.mode, antialias=self.antialias, scale=self.scale)

    def extra_repr(self) -> str:
        return f"size={self.size}, scale={self.scale}, mode={self.mode}, antialias={self.antialias}"


class ResizePair(nn.Module):
    """The ``(D, U)`` operator pair used by the super-resolution coupling.

    Given high-resolution images ``x_1``, ``self.forward(x_1)`` returns the triple

        ``(low_res, upsampled_low_res, x_0_mean)``

    where ``low_res = D(x_1)``, ``xi = upsampled_low_res = U(D(x_1))`` is the
    conditioning image of Section 4.2, and ``x_0_mean = xi`` is the base mean
    ``m(x_1)`` (noise is added by the coupling, not here).

    Parameters
    ----------
    low_res:
        Low-resolution spatial size (int for square images or ``(H_low, W_low)``).
    down_mode / up_mode:
        Interpolation modes for ``D`` and ``U``.
    """

    def __init__(
        self,
        low_res: Size,
        down_mode: str = "area",
        up_mode: str = "bilinear",
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.low_res = normalize_size(low_res)
        self.down_mode = down_mode
        self.up_mode = up_mode
        self.antialias = antialias

    # -- individual operators -------------------------------------------------
    def D(self, x1: torch.Tensor, low_res: Optional[Size] = None) -> torch.Tensor:
        """Downsample ``x_1`` to the low-resolution size."""
        size = self.low_res if low_res is None else normalize_size(low_res)
        return downsample(x1, size=size, mode=self.down_mode, antialias=self.antialias)

    def U(self, low: torch.Tensor, high_res: Optional[Size] = None) -> torch.Tensor:
        """Upsample a low-resolution image back to the high-resolution size."""
        return upsample(low, size=high_res, mode=self.up_mode)

    def UD(self, x1: torch.Tensor, low_res: Optional[Size] = None) -> torch.Tensor:
        """``U(D(x_1))`` -- the conditioning image ``xi`` / base mean ``m(x_1)``."""
        low = self.D(x1, low_res=low_res)
        return self.U(low, high_res=(int(x1.shape[-2]), int(x1.shape[-1])))

    # -- combined ------------------------------------------------------------
    def forward(
        self, x1: torch.Tensor, low_res: Optional[Size] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(xi, low_res)`` with ``xi = U(D(x_1))``."""
        _check_input(x1, "x1")
        low = self.D(x1, low_res=low_res)
        xi = self.U(low, high_res=(int(x1.shape[-2]), int(x1.shape[-1])))
        return xi, low

    def extra_repr(self) -> str:
        return (
            f"low_res={self.low_res}, down_mode={self.down_mode}, "
            f"up_mode={self.up_mode}, antialias={self.antialias}"
        )


def low_resolution(x1: torch.Tensor, low_res: Size, mode: str = "area") -> torch.Tensor:
    """``D(x_1)``: downsample a high-resolution image (Section 4.2)."""
    return downsample(x1, size=normalize_size(low_res), mode=mode)


def default_superres_pair(low_res: Size, **kwargs) -> ResizePair:
    """Factory for the ``(D, U)`` pair with the paper's default operators.

    ``D`` = area (average-pool) downsampling, ``U`` = bilinear upsampling.
    """
    return ResizePair(low_res=low_res, **kwargs)


def _self_test() -> None:
    torch.manual_seed(0)

    # 256 -> 64 -> 256 (super-resolution 64x64 -> 256x256, Table 3)
    x1 = torch.rand(2, 3, 256, 256)
    pair = ResizePair(low_res=64)
    xi, low = pair(x1)
    assert low.shape == (2, 3, 64, 64), low.shape
    assert xi.shape == x1.shape, xi.shape
    assert pair.D(x1).shape == (2, 3, 64, 64)
    assert pair.UD(x1).shape == x1.shape
    # exact factor-4 average pooling must match the functional area path
    assert torch.allclose(pair.D(x1), F.avg_pool2d(x1, 4), atol=1e-6)

    # non-divisible spatial sizes fall back to area interpolation
    x = torch.rand(1, 3, 250, 250)
    d = downsample(x, size=64, mode="area")
    assert d.shape == (1, 3, 64, 64), d.shape
    u = upsample(d, size=250)
    assert u.shape == x.shape, u.shape

    # scale-based API
    assert downsample(x, scale=4).shape == (1, 3, 62, 62)
    assert upsample(d, scale=4).shape == (1, 3, 256, 256)

    # 256 -> 512 super-resolution (Section 4.2, Fig. 6)
    pair2 = ResizePair(low_res=256)
    xi2, low2 = pair2(torch.rand(1, 3, 512, 512))
    assert low2.shape == (1, 3, 256, 256)
    assert xi2.shape == (1, 3, 512, 512)

    # 3D (unbatched) input is supported
    assert downsample(torch.rand(3, 256, 256), size=64).shape == (3, 64, 64)

    # channel-wise modes keep values in range for nearest
    assert upsample(d, size=250, mode="nearest").shape == (1, 3, 250, 250)

    # error handling
    for bad in (lambda: downsample(x, mode="area"), lambda: upsample(d)):
        try:
            bad()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected ValueError for missing size/scale")

    print("resize.py self-test passed.")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
