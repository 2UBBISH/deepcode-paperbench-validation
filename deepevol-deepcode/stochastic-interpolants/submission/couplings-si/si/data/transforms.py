"""Image transforms for the data-dependent-coupling stochastic-interpolant experiments.

The paper (Sections 4.1 and 4.2, and the benchmark addendum) works with ImageNet-1k
images at resolution 256 or 512:

* In-painting: :math:`x_1 \\in \\mathbb{R}^{C \\times W \\times H}` is a (256 or 512)
  ImageNet image; the mask takes the same value for all channels at a given spatial
  location and noise is added *independently per channel* inside the masked region.
* Super-resolution: :math:`x_1` is the high-resolution image and the low-resolution
  observation is obtained with the downsampling / upsampling operators
  :math:`D` and :math:`U` from :mod:`si.couplings.resize`.

This module supplies the (de)normalisation convention used throughout the repository:

* model / sampler / loss tensors live in ``[-1, 1]``  (``to_tensor`` / ``to_neg_one_one``),
* display / FID code wants ``[0, 1]``          (``denormalize`` / ``to_display``).

Everything is a plain tensor operation on batched ``(B, C, H, W)`` (or unbatched
``(C, H, W)``) tensors so that the transforms can run on GPU inside the training loop.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    # core conversions
    "to_tensor",
    "to_neg_one_one",
    "to_zero_one",
    "denormalize",
    "normalize",
    "to_display",
    "from_display",
    "image_range",
    "clamp_unit",
    # geometry
    "resize",
    "center_crop",
    "random_crop",
    "random_hflip",
    "pad_to_square",
    "normalize_size",
    "resize_shorter_side",
    "imagenet_transform",
    # high level helpers
    "get_transforms",
    "make_transform",
    "ImageNetTransforms",
    "TransformConfig",
    # resnet/normalisation statistics
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGENET_MEAN_TENSOR",
    "IMAGENET_STD_TENSOR",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Directory-style mean of ImageNet RGB channels (used for the inception/FID network).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
#: Directory-style std of ImageNet RGB channels.
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_STD).view(3, 1, 1)


# --------------------------------------------------------------------------------------
# Small internal utilities
# --------------------------------------------------------------------------------------

def _as_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    """Return ``shape`` as a tuple of ints (accepts torch.Size)."""
    return tuple(int(s) for s in shape)


def _spatial_dims(n: int) -> int:
    """Number of trailing spatial dims for a tensor with ``n`` dims.

    ``(C, H, W) -> 2``, ``(B, C, H, W) -> 2``, ``(B, T, C, H, W) -> 3``.
    """
    if n == 3:
        return 2
    if n == 4:
        return 2
    if n == 5:
        return 3
    # Conservative default: everything except batch+channel is spatial.
    return max(1, n - 2)


def _reshape_param(value: Union[float, torch.Tensor], ndim: int, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a per-channel constant/tensor to ``(1, C, 1, ..., 1)``."""
    if isinstance(value, torch.Tensor):
        v = value.to(device=like.device, dtype=like.dtype)
        if v.ndim == 1:
            v = v.view(1, -1, *([1] * (ndim - 2)))
        return v
    return torch.tensor(float(value), device=like.device, dtype=like.dtype)


# --------------------------------------------------------------------------------------
# Range conversions: the repository convention is [-1, 1]
# --------------------------------------------------------------------------------------

def to_tensor(image: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert an image to a ``[-1, 1]`` floating tensor, preserving the layout.

    Accepts numpy arrays / torch tensors / PIL images / nested python sequences.

    * uint8 ``[0, 255]`` (or any integer input) is rescaled to ``[-1, 1]``,
    * float input already in ``[0, 1]`` is mapped by ``2 * x - 1``,
    * ``(H, W, C)`` uint8 arrays are transposed to ``(C, H, W)``.

    This mirrors the ``[-1, 1]`` convention expected by ``eval/fid.py`` and the
    samplers, and used by the ``[-1, 1]`` base samples in Sections 4.1/4.2.
    """
    # PIL
    if hasattr(image, "convert") and not isinstance(image, torch.Tensor):
        arr = np.asarray(image.convert("RGB"))
        return to_tensor(arr, dtype=dtype)

    if isinstance(image, torch.Tensor):
        t = image
    else:
        arr = np.asarray(image)
        if arr.dtype == np.uint8:
            t = torch.from_numpy(arr).to(torch.float32).div_(255.0)
        elif np.issubdtype(arr.dtype, np.integer):
            maxv = float(np.iinfo(arr.dtype).max) or 1.0
            t = torch.from_numpy(arr.astype(np.float32)).div_(maxv)
        else:
            t = torch.from_numpy(np.asarray(arr, dtype=np.float32))

    # HWC -> CHW for 3D inputs whose last dim looks like channels (<= 4)
    if t.ndim == 3 and t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
        t = t.permute(2, 0, 1)

    t = t.to(dtype)
    if not t.is_floating_point():
        t = t.float()
        if t.max() > 1.0 + 1e-6:
            t = t / 255.0

    # Already in [-1, 1]?  (min < 0 signals the repo convention)
    if float(t.min()) < -1e-6:
        return t.clamp(-1.0, 1.0)
    return t.clamp(0.0, 1.0) * 2.0 - 1.0


def to_neg_one_one(image: Any, **kwargs: Any) -> torch.Tensor:
    """Alias of :func:`to_tensor` documenting the target range."""
    return to_tensor(image, **kwargs)


def to_zero_one(image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Map ``[-1, 1]`` -> ``[0, 1]`` (tensor or numpy array)."""
    if isinstance(image, torch.Tensor):
        return (image.clamp(-1.0, 1.0) + 1.0) / 2.0
    arr = np.asarray(image)
    return (np.clip(arr, -1.0, 1.0) + 1.0) / 2.0


def normalize(x: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Standardise a ``[0, 1]`` tensor: ``(x - mean) / std`` (default -> ``[-1, 1]``)."""
    return (x - mean) / std


def denormalize(
    x: Union[torch.Tensor, np.ndarray],
    mean: float = 0.5,
    std: float = 0.5,
) -> Union[torch.Tensor, np.ndarray]:
    """Inverse of :func:`normalize`: map ``[-1, 1]`` back to ``[0, 1]``."""
    if isinstance(x, torch.Tensor):
        return x * std + mean
    return np.asarray(x) * std + mean


def to_display(images: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert ``[-1, 1]`` batched images to ``[0, 1]`` ``(B, H, W, C)`` numpy arrays.

    Handles both ``(B, C, H, W)`` tensors and ``(C, H, W)`` single images.  Matches the
    convention used in :mod:`eval.qualitative`.
    """
    if isinstance(images, torch.Tensor):
        t = images.detach().float().cpu()
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = to_zero_one(t)  # (B, C, H, W) in [0, 1]
        return t.permute(0, 2, 3, 1).numpy()
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return to_zero_one(arr)


def from_display(images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Inverse of :func:`to_display`: ``[0, 1]`` ``(B, H, W, C)`` -> ``[-1, 1]`` tensors."""
    if isinstance(images, torch.Tensor):
        arr = images.detach().float().cpu()
        if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
            arr = arr.permute(0, 3, 1, 2)
        return to_tensor(arr)
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 4:
        arr = np.transpose(arr, (0, 3, 1, 2))
    return to_tensor(arr)


def image_range(x: torch.Tensor) -> Tuple[float, float]:
    """Return ``(min, max)`` of a tensor as python floats (diagnostics/logging)."""
    return float(x.min()), float(x.max())


def clamp_unit(x: torch.Tensor) -> torch.Tensor:
    """Clamp model output to the valid ``[-1, 1]`` image range."""
    return x.clamp(-1.0, 1.0)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

def normalize_size(
    size: Union[int, Sequence[int], torch.Size, None],
    x: Optional[torch.Tensor] = None,
    spatial_dims: int = 2,
) -> Tuple[int, ...]:
    """Normalise an int / sequence / torch.Size spec into a tuple of spatial sizes.

    ``size=None`` falls back to ``x.shape[-spatial_dims:]``.
    """
    if size is None:
        if x is None:
            raise ValueError("normalize_size requires either `size` or `x`.")
        return tuple(int(s) for s in x.shape[-spatial_dims:])
    if isinstance(size, (int, np.integer)):
        return tuple([int(size)] * spatial_dims)
    return tuple(int(s) for s in size)


def resize(
    x: torch.Tensor,
    size: Union[int, Sequence[int]],
    mode: str = "bilinear",
    antialias: bool = True,
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize the spatial dims of a batched/unbatched tensor with ``F.interpolate``.

    ``size`` is the *target* spatial size (int or ``(H, W)``); if a single int is given
    the shorter side is scaled to it while preserving the aspect ratio.
    """
    if mode in ("area",):
        # `area` is not a valid interpolate mode for downsampling by factor; use adaptive
        # average pooling semantics via `F.interpolate(..., mode="area")` which is valid.
        pass
    ndim = x.ndim
    if ndim == 3:
        x_ = x.unsqueeze(0)
        squeeze = True
    elif ndim == 4:
        x_ = x
        squeeze = False
    else:
        raise ValueError(f"resize expects 3D or 4D input, got shape {tuple(x.shape)}")

    h, w = x_.shape[-2:]
    if isinstance(size, (int, np.integer)):
        # scale the shorter side to `size`, keeping the aspect ratio
        scale = float(size) / float(min(h, w))
        target = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    else:
        target = tuple(int(s) for s in size)
        if len(target) == 1:
            target = (target[0], target[0])

    if target == (h, w):
        return x

    kwargs: Dict[str, Any] = {}
    if mode in ("bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = align_corners
        kwargs["antialias"] = antialias
    out = F.interpolate(x_, size=target, mode=mode, **kwargs)
    return out.squeeze(0) if squeeze else out


def resize_shorter_side(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    """Scale so the *shorter* side equals ``size`` (standard ImageNet preprocessing)."""
    return resize(x, int(size), mode=mode)


def center_crop(x: torch.Tensor, size: Union[int, Sequence[int]]) -> torch.Tensor:
    """Center-crop the spatial dims to ``size``."""
    target = normalize_size(size, x=x)
    if len(target) == 1:
        target = (target[0], target[0])
    h, w = x.shape[-2:]
    th, tw = target
    if th > h or tw > w:
        raise ValueError(f"center_crop size {target} exceeds input spatial size {(h, w)}")
    top = (h - th) // 2
    left = (w - tw) // 2
    return x[..., top : top + th, left : left + tw]


def random_crop(
    x: torch.Tensor,
    size: Union[int, Sequence[int]],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly crop the spatial dims to ``size`` (per-batch random offset)."""
    target = normalize_size(size, x=x)
    if len(target) == 1:
        target = (target[0], target[0])
    h, w = x.shape[-2:]
    th, tw = target
    if th > h or tw > w:
        raise ValueError(f"random_crop size {target} exceeds input spatial size {(h, w)}")
    top = int(torch.randint(0, h - th + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - tw + 1, (1,), generator=generator).item())
    return x[..., top : top + th, left : left + tw]


def random_hflip(
    x: torch.Tensor,
    p: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Horizontally flip the whole batch with probability ``p``."""
    if p <= 0.0:
        return x
    if p >= 1.0 or float(torch.rand((), generator=generator).item()) < p:
        return torch.flip(x, dims=[-1])
    return x


def pad_to_square(
    x: torch.Tensor,
    fill: float = -1.0,
    size: Optional[Union[int, Sequence[int]]] = None,
) -> torch.Tensor:
    """Zero/constant-pad the spatial dims to a square (or explicit ``size``).

    ImageNet-1k is essentially square already; this only guards against odd-shaped
    samples so that the tiled in-painting mask / U-Net strides stay valid.
    """
    if size is not None:
        target = normalize_size(size, x=x)
        if len(target) == 1:
            target = (target[0], target[0])
    else:
        s = max(int(x.shape[-2]), int(x.shape[-1]))
        target = (s, s)
    h, w = x.shape[-2:]
    th, tw = target
    if th < h or tw < w:
        return center_crop(x, target)
    pad_h = th - h
    pad_w = tw - w
    # F.pad takes (left, right, top, bottom) for the last two dims
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    if all(v == 0 for v in pad):
        return x
    return F.pad(x, pad, mode="constant", value=float(fill))


# --------------------------------------------------------------------------------------
# Config + high level pipeline
# --------------------------------------------------------------------------------------

class TransformConfig:
    """Small dataclass-like container for the eval / train preprocessing settings.

    Parameters mirror the paper's settings: ImageNet at 256 or 512, plus the 64x64
    low-resolution view used by the super-resolution task (Section 4.2).
    """

    __slots__ = (
        "resolution",
        "low_resolution",
        "crop",
        "random_crop",
        "hflip",
        "hflip_prob",
        "pad",
        "range",
        "antialias",
    )

    def __init__(
        self,
        resolution: int = 256,
        low_resolution: Optional[int] = None,
        crop: Optional[Union[int, Sequence[int]]] = None,
        random_crop: bool = False,
        hflip: bool = True,
        hflip_prob: float = 0.5,
        pad: bool = True,
        range: str = "neg_one_one",
        antialias: bool = True,
    ) -> None:
        self.resolution = int(resolution)
        self.low_resolution = None if low_resolution is None else int(low_resolution)
        self.crop = crop
        self.random_crop = bool(random_crop)
        self.hflip = bool(hflip)
        self.hflip_prob = float(hflip_prob)
        self.pad = bool(pad)
        self.range = str(range)
        self.antialias = bool(antialias)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None, **overrides: Any) -> "TransformConfig":
        cfg = dict(d or {})
        cfg.update(overrides)
        # tolerate common aliases used in YAML configs
        if "image_size" in cfg and "resolution" not in cfg:
            cfg["resolution"] = cfg.pop("image_size")
        if "size" in cfg and "resolution" not in cfg:
            cfg["resolution"] = cfg.pop("size")
        if "low_res" in cfg and "low_resolution" not in cfg:
            cfg["low_resolution"] = cfg.pop("low_res")
        known = {k: cfg[k] for k in cls.__slots__ if k in cfg}
        return cls(**known)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        args = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__slots__)
        return f"TransformConfig({args})"


class ImageNetTransforms:
    """Callable preprocessing pipeline for ImageNet-1k images.

    Produces ``x1`` (high resolution, ``[-1, 1]``) and, for super-resolution, the
    low-resolution observation ``xi = U(D(x1))``.  Works on torch tensors, numpy
    arrays and PIL images.
    """

    def __init__(
        self,
        resolution: int = 256,
        low_resolution: Optional[int] = None,
        *,
        train: bool = False,
        crop: Optional[Union[int, Sequence[int]]] = None,
        random_crop: Optional[bool] = None,
        hflip: Optional[bool] = None,
        hflip_prob: float = 0.5,
        pad: bool = True,
        antialias: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.cfg = TransformConfig(
            resolution=resolution,
            low_resolution=low_resolution,
            crop=crop,
            random_crop=train if random_crop is None else bool(random_crop),
            hflip=train if hflip is None else bool(hflip),
            hflip_prob=hflip_prob,
            pad=pad,
            antialias=antialias,
        )
        self.generator = generator
        self.train = bool(train)

    # -- internals -------------------------------------------------------------------
    def _to_target(self, x: torch.Tensor) -> torch.Tensor:
        target = int(self.cfg.resolution)
        h, w = x.shape[-2:]
        if (h, w) != (target, target):
            x = resize(x, target, mode="bilinear", antialias=self.cfg.antialias)
        if self.cfg.pad:
            x = pad_to_square(x, size=target)
        if self.cfg.crop is not None and tuple(normalize_size(self.cfg.crop)) != (target, target):
            x = (
                random_crop(x, self.cfg.crop, generator=self.generator)
                if self.cfg.random_crop
                else center_crop(x, self.cfg.crop)
            )
        return x

    # -- API -------------------------------------------------------------------------
    def __call__(self, image: Any) -> torch.Tensor:
        x = to_tensor(image)
        if self.train and self.cfg.hflip:
            x = random_hflip(x, self.cfg.hflip_prob, generator=self.generator)
        return self._to_target(x)

    def low_resolution(self, x1: torch.Tensor) -> torch.Tensor:
        """Return ``U(D(x1))`` at the same spatial resolution as ``x1``.

        This is the conditioning ``xi`` of the super-resolution coupling (Section 4.2),
        computed with the exact ``D``/``U`` operators from :mod:`si.couplings.resize`.
        """
        if self.cfg.low_resolution is None:
            raise ValueError("TransformConfig.low_resolution is not set")
        from ..couplings.resize import ResizePair

        pair = ResizePair(self.cfg.low_resolution)
        return pair.UD(x1)

    def low_res_dataset(self, x1: torch.Tensor) -> torch.Tensor:
        """Return the *natively low-resolution* image ``D(x1)`` (64x64 for 64->256)."""
        if self.cfg.low_resolution is None:
            raise ValueError("TransformConfig.low_resolution is not set")
        from ..couplings.resize import downsample

        return downsample(x1, self.cfg.low_resolution)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ImageNetTransforms(train={self.train}, {self.cfg!r})"


def imagenet_transform(
    resolution: int = 256,
    train: bool = False,
    low_resolution: Optional[int] = None,
    **kwargs: Any,
) -> ImageNetTransforms:
    """Factory for the standard ImageNet preprocessing (paper: 256 or 512)."""
    return ImageNetTransforms(
        resolution=resolution,
        low_resolution=low_resolution,
        train=train,
        **kwargs,
    )


def make_transform(
    config: Optional[Union[TransformConfig, Dict[str, Any]]] = None,
    **overrides: Any,
) -> ImageNetTransforms:
    """Build an :class:`ImageNetTransforms` from a dict/config plus overrides."""
    train = bool(overrides.pop("train", False))
    if isinstance(config, TransformConfig):
        cfg = config.to_dict()
        cfg.update(overrides)
        return ImageNetTransforms(train=train, **cfg)
    cfg = TransformConfig.from_dict(config, **overrides)
    return ImageNetTransforms(train=train, **cfg.to_dict())


def get_transforms(
    resolution: int = 256,
    train: bool = False,
    low_resolution: Optional[int] = None,
    **kwargs: Any,
) -> ImageNetTransforms:
    """Convenience alias preferred by the data loaders."""
    return imagenet_transform(
        resolution=resolution,
        train=train,
        low_resolution=low_resolution,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)

    # (1) uint8 numpy -> [-1, 1] CHW
    arr = np.random.randint(0, 256, size=(32, 40, 3), dtype=np.uint8)
    t = to_tensor(arr)
    assert t.shape == (3, 32, 40), t.shape
    assert t.min() >= -1.0 - 1e-6 and t.max() <= 1.0 + 1e-6, (t.min(), t.max())

    # (2) round trip [-1,1] -> display -> [-1,1]
    disp = to_display(t)
    assert disp.shape == (1, 32, 40, 3)
    assert disp.min() >= -1e-6 and disp.max() <= 1.0 + 1e-6
    back = from_display(disp)
    assert torch.allclose(back, t.clamp(-1, 1), atol=1e-5)

    # (3) transform pipeline at 256
    tf = imagenet_transform(256)
    x1 = to_tensor(np.random.randint(0, 256, size=(300, 320, 3), dtype=np.uint8))
    x1 = tf(x1)
    assert x1.shape == (3, 256, 256), x1.shape
    assert x1.min() >= -1.0 - 1e-6 and x1.max() <= 1.0 + 1e-6

    # (4) batched shape preservation
    batch = torch.rand(4, 3, 256, 256) * 2 - 1
    assert resize(batch, 128).shape == (4, 3, 128, 128)
    assert center_crop(batch, 128).shape == (4, 3, 128, 128)

    # (5) super-resolution low-res views: D(x1) at 64 and U(D(x1)) at 256
    tf_sr = imagenet_transform(256, low_resolution=64)
    low = tf_sr.low_res_dataset(x1.unsqueeze(0))
    assert low.shape == (1, 3, 64, 64), low.shape
    xi = tf_sr.low_resolution(x1.unsqueeze(0))
    assert xi.shape == (1, 3, 256, 256), xi.shape

    # (6) 512 config
    tf512 = imagenet_transform(512)
    x512 = tf512(to_tensor(np.random.randint(0, 256, size=(600, 600, 3), dtype=np.uint8)))
    assert x512.shape == (3, 512, 512), x512.shape

    # (7) 64x64 low-res dataset fed to a 64->256 super-resolution transform
    tf64 = imagenet_transform(64, low_resolution=None)
    x64 = tf64(to_tensor(np.random.randint(0, 256, size=(70, 90, 3), dtype=np.uint8)))
    assert x64.shape == (3, 64, 64), x64.shape

    # (8) config round trip
    cfg = TransformConfig.from_dict({"image_size": 512, "low_res": 64, "hflip": False})
    assert cfg.resolution == 512 and cfg.low_resolution == 64 and cfg.hflip is False
    tf2 = make_transform(cfg, train=True)
    assert tf2.cfg.resolution == 512 and tf2.cfg.hflip is True

    print("si.data.transforms self-test passed")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
