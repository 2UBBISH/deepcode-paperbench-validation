"""Pixel-space vs. model-normalized input handling for the Robust CLIP reproduction.

The Addendum mandates that every ``l_inf`` ball used by :mod:`attacks.pgd` and
:mod:`attacks.apgd` is computed around **NON-normalized** inputs, i.e. raw pixel
tensors in ``[0, 1]``.  Model normalization (``(x - mean) / std``) is therefore a
purely internal step performed *after* an attack has produced adversarial raw
pixels.

This module is the single place where:

* the canonical normalization statistics live (OpenAI CLIP, ImageNet, identity),
* raw pixels are converted to / from model space,
* perturbations are projected back onto a ball around the raw image,
* perturbation deltas are converted between pixel space and model space.

Nothing the Addendum leaves unspecified is invented here: norms, epsilons and
hyper-parameters are always caller supplied, and statistics are named after the
upstream checkpoint they come from (``CLIP_MEAN``/``CLIP_STD`` are the published
OpenAI CLIP constants; nothing more).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

LOGGER = logging.getLogger("robust_clip_repro.utils.normalization")

#: Marker for values that the Addendum does not specify.
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

# ---------------------------------------------------------------------------
# Canonical statistics (upstream, published values -- not paper-specific)
# ---------------------------------------------------------------------------
# OpenAI CLIP (used for ViT-L/14@224 by LLaVA-1.5 and OpenFlamingo).
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# ImageNet / torchvision statistics (kept for completeness; not used by CLIP).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Identity normalization: model space == raw pixel space.
IDENTITY_MEAN: Tuple[float, float, float] = (0.0, 0.0, 0.0)
IDENTITY_STD: Tuple[float, float, float] = (1.0, 1.0, 1.0)

#: Raw pixel space is ``[0, 1]`` throughout the benchmark (``ToTensor``-style).
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

#: Tolerance used by the raw-vs-normalized heuristic.
DEFAULT_TOLERANCE = 1e-3

#: Supported ball norms for projection.
SUPPORTED_NORMS: Tuple[str, ...] = ("linf", "l2")

#: Normalization name -> (mean, std) registry.
NORMALIZATION_REGISTRY: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "clip": (CLIP_MEAN, CLIP_STD),
    "openai_clip": (CLIP_MEAN, CLIP_STD),
    "openclip": (CLIP_MEAN, CLIP_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "identity": (IDENTITY_MEAN, IDENTITY_STD),
    "none": (IDENTITY_MEAN, IDENTITY_STD),
    "raw": (IDENTITY_MEAN, IDENTITY_STD),
}

#: Human readable origin of each registry entry (for logging/provenance).
NORMALIZATION_SOURCES: Dict[str, str] = {
    "clip": "OpenAI CLIP ViT-L/14 published statistics (UPSTREAM value)",
    "openai_clip": "OpenAI CLIP ViT-L/14 published statistics (UPSTREAM value)",
    "openclip": "OpenAI CLIP ViT-L/14 published statistics (UPSTREAM value)",
    "imagenet": "torchvision ImageNet statistics (UPSTREAM value)",
    "identity": "no-op normalization (raw pixels == model space)",
    "none": "no-op normalization (raw pixels == model space)",
    "raw": "no-op normalization (raw pixels == model space)",
}


def _torch():
    """Import :mod:`torch` lazily so the pure-python helpers stay importable."""
    import torch  # noqa: WPS433 (local import intentional)

    return torch


def _as_triple(value: Union[float, int, Sequence[float], None], names: str) -> Tuple[float, float, float]:
    """Coerce ``value`` into a 3-tuple of floats (broadcasting scalars)."""
    if value is None:
        return (0.0, 0.0, 0.0)
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    if hasattr(value, "tolist"):  # numpy array / tensor
        value = value.tolist()
    seq = list(value)  # type: ignore[arg-type]
    if len(seq) == 3:
        return (float(seq[0]), float(seq[1]), float(seq[2]))
    if len(seq) == 1:
        return (float(seq[0]),) * 3
    raise ValueError(f"{names} must have 1 or 3 elements, got {len(seq)}")


def mean_std_tensors(
    mean: Union[Sequence[float], float] = CLIP_MEAN,
    std: Union[Sequence[float], float] = CLIP_STD,
    *,
    device: Any = None,
    dtype: Any = None,
    ndim: int = 4,
) -> Tuple[Any, Any]:
    """Return ``(mean, std)`` as broadcastable tensors shaped ``(1, 3, 1, 1)``.

    ``ndim=4`` matches ``(B, C, H, W)`` pixel tensors; pass ``ndim=3`` for
    ``(C, H, W)`` images.
    """
    torch = _torch()
    m = _as_triple(mean, "mean")
    s = _as_triple(std, "std")
    shape = [1] * max(int(ndim), 1)
    if len(shape) >= 2:
        shape[1] = 3
    m_t = torch.tensor(m, device=device, dtype=dtype).view(*shape)
    s_t = torch.tensor(s, device=device, dtype=dtype).view(*shape)
    return m_t, s_t


def normalize_pixels(
    pixels: Any,
    mean: Union[Sequence[float], float] = CLIP_MEAN,
    std: Union[Sequence[float], float] = CLIP_STD,
    *,
    inplace: bool = False,
) -> Any:
    """Convert raw ``[0, 1]`` pixels into model-normalized space.

    ``pixels`` is never mutated unless ``inplace=True``, which matters because
    the attack engines must keep operating on the raw tensor.
    """
    torch = _torch()
    m, s = mean_std_tensors(mean, std, device=pixels.device, dtype=pixels.dtype, ndim=pixels.dim())
    if inplace:
        return pixels.sub_(m).div_(s)
    return (pixels - m) / s


def denormalize_pixels(
    normalized: Any,
    mean: Union[Sequence[float], float] = CLIP_MEAN,
    std: Union[Sequence[float], float] = CLIP_STD,
    *,
    inplace: bool = False,
) -> Any:
    """Inverse of :func:`normalize_pixels`: model space -> raw ``[0, 1]`` space."""
    m, s = mean_std_tensors(mean, std, device=normalized.device, dtype=normalized.dtype, ndim=normalized.dim())
    if inplace:
        return normalized.mul_(s).add_(m)
    return normalized * s + m


def pixel_range(pixels: Any) -> Tuple[float, float]:
    """Return ``(min, max)`` of a pixel tensor as python floats."""
    return float(pixels.min().item()), float(pixels.max().item())


def is_normalized(
    pixels: Any,
    mean: Union[Sequence[float], float] = CLIP_MEAN,
    std: Union[Sequence[float], float] = CLIP_STD,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Heuristic: does ``pixels`` look like it is already model-normalized?

    Raw images live in ``[0, 1]``; model-normalized tensors leave that interval
    (CLIP statistics push values well outside it).  A tensor is treated as
    normalized when it does not lie inside ``[0 - tol, 1 + tol]``.
    """
    lo, hi = pixel_range(pixels)
    return lo < (PIXEL_MIN - tolerance) or hi > (PIXEL_MAX + tolerance)


def ensure_raw_pixels(
    pixels: Any,
    mean: Union[Sequence[float], float] = CLIP_MEAN,
    std: Union[Sequence[float], float] = CLIP_STD,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Any:
    """Return ``pixels`` guaranteed to be in raw ``[0, 1]`` space.

    If the input already looks normalized it is denormalized first, so callers
    can never accidentally build an ``l_inf`` ball around normalized inputs.
    """
    if is_normalized(pixels, mean, std, tolerance=tolerance):
        LOGGER.debug("Input detected as model-normalized; denormalizing to raw pixel space.")
        return denormalize_pixels(pixels, mean, std)
    return pixels


def clamp_pixels(
    pixels: Any,
    *,
    pixel_min: float = PIXEL_MIN,
    pixel_max: float = PIXEL_MAX,
) -> Any:
    """Clamp a pixel tensor into the valid image range."""
    torch = _torch()
    return torch.clamp(pixels, float(pixel_min), float(pixel_max))


def adversarial_pixels(
    pixels: Any,
    delta: Any,
    *,
    pixel_min: float = PIXEL_MIN,
    pixel_max: float = PIXEL_MAX,
) -> Any:
    """``clamp(x + delta)`` in raw pixel space (the space attacks live in)."""
    return clamp_pixels(pixels + delta, pixel_min=pixel_min, pixel_max=pixel_max)


def project_linf(
    pixels: Any,
    delta: Any,
    eps: float,
    *,
    pixel_min: float = PIXEL_MIN,
    pixel_max: float = PIXEL_MAX,
) -> Any:
    """Project ``delta`` onto the ``l_inf`` ball of radius ``eps`` around raw pixels.

    The projection is performed around *non-normalized* ``pixels`` (Addendum),
    then re-derived from the clamped adversarial image so that
    ``pixels + result`` is always a valid image in ``[pixel_min, pixel_max]``.
    """
    torch = _torch()
    eps = float(eps)
    delta = torch.clamp(delta, -eps, eps)
    adv = torch.clamp(pixels + delta, float(pixel_min), float(pixel_max))
    return adv - pixels


def l2_norm(delta: Any, *, keepdim: bool = True) -> Any:
    """Per-sample ``l_2`` norm of a ``(B, C, H, W)`` perturbation tensor."""
    flat = delta.reshape(delta.shape[0], -1)
    norm = flat.norm(p=2, dim=1)
    if keepdim:
        return norm.view(-1, *([1] * (delta.dim() - 1)))
    return norm


def project_l2(
    pixels: Any,
    delta: Any,
    eps: float,
    *,
    pixel_min: float = PIXEL_MIN,
    pixel_max: float = PIXEL_MAX,
) -> Any:
    """Project ``delta`` onto the ``l_2`` ball of radius ``eps`` around raw pixels."""
    torch = _torch()
    eps = float(eps)
    norm = l2_norm(delta, keepdim=True)
    factor = torch.clamp(eps / norm.clamp_min(1e-12), max=1.0)
    delta = delta * factor
    adv = torch.clamp(pixels + delta, float(pixel_min), float(pixel_max))
    delta = adv - pixels
    # Clamping toward the valid box can only shrink the norm, but re-project
    # defensively in case a caller passes a non-standard box.
    norm = l2_norm(delta, keepdim=True)
    factor = torch.clamp(eps / norm.clamp_min(1e-12), max=1.0)
    return delta * factor


def project_ball(
    pixels: Any,
    delta: Any,
    eps: float,
    *,
    norm: str = "linf",
    pixel_min: float = PIXEL_MIN,
    pixel_max: float = PIXEL_MAX,
) -> Any:
    """Dispatch to :func:`project_linf` / :func:`project_l2`."""
    key = str(norm).lower()
    if key in ("linf", "inf", "l_inf", "l-inf", "linfty"):
        return project_linf(pixels, delta, eps, pixel_min=pixel_min, pixel_max=pixel_max)
    if key in ("l2", "l_2", "l-2"):
        return project_l2(pixels, delta, eps, pixel_min=pixel_min, pixel_max=pixel_max)
    raise ValueError(f"Unsupported norm {norm!r}; expected one of {SUPPORTED_NORMS}")


def perturbation_norm(delta: Any, norm: str = "linf") -> Any:
    """Per-sample perturbation norm (``l_inf`` max-abs or ``l_2``)."""
    key = str(norm).lower()
    if key in ("linf", "inf", "l_inf", "l-inf", "linfty"):
        return delta.reshape(delta.shape[0], -1).abs().amax(dim=1)
    if key in ("l2", "l_2", "l-2"):
        return l2_norm(delta, keepdim=False)
    raise ValueError(f"Unsupported norm {norm!r}; expected one of {SUPPORTED_NORMS}")


def ball_contains(
    pixels: Any,
    adversarial: Any,
    eps: float,
    *,
    norm: str = "linf",
    atol: float = 1e-5,
) -> bool:
    """True when ``adversarial`` lies inside the eps-ball around ``pixels``."""
    delta = adversarial - pixels
    norm_value = perturbation_norm(delta, norm)
    return bool(float(norm_value.max().item()) <= float(eps) + float(atol))


def pixel_delta_from_model_delta(
    delta_model: Any,
    std: Union[Sequence[float], float] = CLIP_STD,
) -> Any:
    """Convert a delta expressed in model space to the equivalent pixel-space delta.

    ``(x_adv - x) / std`` is the model-space delta, hence ``delta_pixel = delta_model * std``.
    """
    _, s = mean_std_tensors(IDENTITY_MEAN, std, device=delta_model.device, dtype=delta_model.dtype, ndim=delta_model.dim())
    return delta_model * s


def model_delta_from_pixel_delta(
    delta_pixel: Any,
    std: Union[Sequence[float], float] = CLIP_STD,
) -> Any:
    """Inverse of :func:`pixel_delta_from_model_delta` (``delta_model = delta_pixel / std``)."""
    _, s = mean_std_tensors(IDENTITY_MEAN, std, device=delta_pixel.device, dtype=delta_pixel.dtype, ndim=delta_pixel.dim())
    return delta_pixel / s


# ---------------------------------------------------------------------------
# NormalizationSpec
# ---------------------------------------------------------------------------
@dataclass
class NormalizationSpec:
    """Named normalization statistics plus pixel-space bounds.

    Instances are hashable-by-value (dataclass) and serialisable so evaluation
    configs can record exactly which normalization accompanied an attack run.
    """

    name: str = "clip"
    mean: Tuple[float, float, float] = CLIP_MEAN
    std: Tuple[float, float, float] = CLIP_STD
    pixel_min: float = PIXEL_MIN
    pixel_max: float = PIXEL_MAX
    tolerance: float = DEFAULT_TOLERANCE
    provenance: str = field(default_factory=lambda: NORMALIZATION_SOURCES.get("clip", UNSPECIFIED))

    # -- construction -----------------------------------------------------
    def __post_init__(self) -> None:
        self.mean = _as_triple(self.mean, "mean")
        self.std = _as_triple(self.std, "std")
        if any(s == 0.0 for s in self.std):
            raise ValueError("Normalization std must be non-zero")

    @classmethod
    def clip(cls, **kwargs: Any) -> "NormalizationSpec":
        return cls(name="clip", mean=CLIP_MEAN, std=CLIP_STD, **kwargs)

    @classmethod
    def imagenet(cls, **kwargs: Any) -> "NormalizationSpec":
        return cls(name="imagenet", mean=IMAGENET_MEAN, std=IMAGENET_STD, **kwargs)

    @classmethod
    def identity(cls, **kwargs: Any) -> "NormalizationSpec":
        return cls(name="identity", mean=IDENTITY_MEAN, std=IDENTITY_STD, **kwargs)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "NormalizationSpec":
        cfg = dict(cfg or {})
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        spec = cfg.pop("normalization", None)
        if isinstance(spec, NormalizationSpec):
            base = spec
        elif isinstance(spec, (str, type(None))) and spec and spec.lower() in NORMALIZATION_REGISTRY:
            base = get_normalization(spec)
        else:
            base = cls()
        if cfg.get("mean") is not None:
            cfg["mean"] = _as_triple(cfg["mean"], "mean")
        if cfg.get("std") is not None:
            cfg["std"] = _as_triple(cfg["std"], "std")
        allowed = {"name", "mean", "std", "pixel_min", "pixel_max", "tolerance", "provenance"}
        base = replace(base, **{k: v for k, v in cfg.items() if k in allowed})
        return base

    # -- operations -------------------------------------------------------
    def normalize(self, pixels: Any, *, inplace: bool = False) -> Any:
        return normalize_pixels(pixels, self.mean, self.std, inplace=inplace)

    def denormalize(self, normalized: Any, *, inplace: bool = False) -> Any:
        return denormalize_pixels(normalized, self.mean, self.std, inplace=inplace)

    def is_normalized(self, pixels: Any) -> bool:
        return is_normalized(pixels, self.mean, self.std, tolerance=self.tolerance)

    def ensure_raw(self, pixels: Any) -> Any:
        return ensure_raw_pixels(pixels, self.mean, self.std, tolerance=self.tolerance)

    def clamp(self, pixels: Any) -> Any:
        return clamp_pixels(pixels, pixel_min=self.pixel_min, pixel_max=self.pixel_max)

    def adversarial(self, pixels: Any, delta: Any) -> Any:
        return adversarial_pixels(pixels, delta, pixel_min=self.pixel_min, pixel_max=self.pixel_max)

    def project(self, pixels: Any, delta: Any, eps: float, *, norm: str = "linf") -> Any:
        return project_ball(
            pixels, delta, eps, norm=norm, pixel_min=self.pixel_min, pixel_max=self.pixel_max
        )

    def describe(self, pixels: Optional[Any] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "name": self.name,
            "mean": list(self.mean),
            "std": list(self.std),
            "pixel_min": self.pixel_min,
            "pixel_max": self.pixel_max,
            "provenance": self.provenance,
            "projection_space": "pixel (non-normalized)",
        }
        if pixels is not None:
            info["pixel_range"] = list(pixel_range(pixels))
            info["looks_normalized"] = self.is_normalized(pixels)
        return info

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean": list(self.mean),
            "std": list(self.std),
            "pixel_min": self.pixel_min,
            "pixel_max": self.pixel_max,
            "tolerance": self.tolerance,
            "provenance": self.provenance,
        }


def get_normalization(spec: Union[str, NormalizationSpec, Dict[str, Any], None] = None) -> NormalizationSpec:
    """Resolve a normalization name / spec / dict into a :class:`NormalizationSpec`."""
    if isinstance(spec, NormalizationSpec):
        return spec
    if isinstance(spec, dict):
        return NormalizationSpec.from_dict(spec)
    if spec is None:
        return NormalizationSpec.clip()
    key = str(spec).lower()
    if key not in NORMALIZATION_REGISTRY:
        raise KeyError(f"Unknown normalization {spec!r}; known: {sorted(NORMALIZATION_REGISTRY)}")
    mean, std = NORMALIZATION_REGISTRY[key]
    return NormalizationSpec(name=key, mean=mean, std=std, provenance=NORMALIZATION_SOURCES.get(key, UNSPECIFIED))


def normalization_metadata(spec: Union[str, NormalizationSpec, None] = None) -> Dict[str, Any]:
    """JSON-friendly provenance record for logging a run."""
    resolved = get_normalization(spec)
    meta = resolved.as_dict()
    meta["specified_by_addendum"] = False
    meta["note"] = (
        "The Addendum only requires that the l_inf ball is computed around "
        "NON-normalized inputs; the concrete statistics are upstream checkpoint values."
    )
    return meta


def build_pixel_to_model_fn(spec: Union[str, NormalizationSpec, None] = None):
    """Return ``fn(raw_pixels) -> model_space_pixels`` for a resolved spec."""
    resolved = get_normalization(spec)

    def _fn(pixels: Any) -> Any:
        return resolved.normalize(resolved.ensure_raw(pixels))

    _fn.spec = resolved  # type: ignore[attr-defined]
    return _fn


def build_model_to_pixel_fn(spec: Union[str, NormalizationSpec, None] = None):
    """Return ``fn(model_space_pixels) -> raw_pixels`` for a resolved spec."""
    resolved = get_normalization(spec)

    def _fn(normalized: Any) -> Any:
        return resolved.clamp(resolved.denormalize(normalized))

    _fn.spec = resolved  # type: ignore[attr-defined]
    return _fn


__all__ = [
    "UNSPECIFIED",
    "CLIP_MEAN",
    "CLIP_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IDENTITY_MEAN",
    "IDENTITY_STD",
    "PIXEL_MIN",
    "PIXEL_MAX",
    "DEFAULT_TOLERANCE",
    "SUPPORTED_NORMS",
    "NORMALIZATION_REGISTRY",
    "NORMALIZATION_SOURCES",
    "NormalizationSpec",
    "mean_std_tensors",
    "normalize_pixels",
    "denormalize_pixels",
    "pixel_range",
    "is_normalized",
    "ensure_raw_pixels",
    "clamp_pixels",
    "adversarial_pixels",
    "project_linf",
    "project_l2",
    "project_ball",
    "perturbation_norm",
    "l2_norm",
    "ball_contains",
    "pixel_delta_from_model_delta",
    "model_delta_from_pixel_delta",
    "get_normalization",
    "normalization_metadata",
    "build_pixel_to_model_fn",
    "build_model_to_pixel_fn",
]


# ---------------------------------------------------------------------------
# Self test / CLI
# ---------------------------------------------------------------------------
def _self_test(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of the normalization helpers (no downloads, no models)."""
    torch = _torch()
    results: Dict[str, Any] = {}

    x = torch.rand(2, 3, 8, 8)

    # round-trip normalization
    x_norm = normalize_pixels(x, CLIP_MEAN, CLIP_STD)
    assert not torch.allclose(x, x_norm, atol=1e-3)
    x_back = denormalize_pixels(x_norm, CLIP_MEAN, CLIP_STD)
    assert torch.allclose(x, x_back, atol=1e-5), "normalize/denormalize must round-trip"
    results["round_trip_ok"] = True

    # raw-vs-normalized detection
    assert not is_normalized(x, CLIP_MEAN, CLIP_STD)
    assert is_normalized(x_norm, CLIP_MEAN, CLIP_STD)
    assert torch.allclose(ensure_raw_pixels(x_norm, CLIP_MEAN, CLIP_STD), x, atol=1e-5)
    results["raw_detection_ok"] = True

    # l_inf projection happens around RAW pixels
    eps = 0.05
    delta = torch.randn_like(x) * 0.5
    projected = project_linf(x, delta, eps)
    assert float(projected.abs().max().item()) <= eps + 1e-6
    adv = x + projected
    assert float(adv.min().item()) >= PIXEL_MIN - 1e-6
    assert float(adv.max().item()) <= PIXEL_MAX + 1e-6
    assert ball_contains(x, adv, eps, norm="linf")
    # A tensor that is *already* bound-satisfying must not be rescaled.
    small = torch.clamp(delta, -eps, eps)
    assert torch.allclose(project_linf(x, small, eps), small, atol=1e-6)
    results["linf_projection_ok"] = True

    # l2 projection
    l2_eps = 1.0
    projected_l2 = project_l2(x, delta, l2_eps)
    norms = perturbation_norm(projected_l2, "l2")
    assert float(norms.max().item()) <= l2_eps + 1e-5
    assert ball_contains(x, x + projected_l2, l2_eps, norm="l2")
    results["l2_projection_ok"] = True

    # per-channel delta conversion
    d_model = torch.ones(1, 3, 1, 1)
    d_pixel = pixel_delta_from_model_delta(d_model, CLIP_STD)
    assert torch.allclose(d_pixel.view(3), torch.tensor(CLIP_STD), atol=1e-6)
    d_model_back = model_delta_from_pixel_delta(d_pixel, CLIP_STD)
    assert torch.allclose(d_model_back, d_model, atol=1e-6)
    results["delta_conversion_ok"] = True

    # specs
    spec = get_normalization("clip")
    assert spec.mean == CLIP_MEAN and spec.std == CLIP_STD
    assert torch.allclose(spec.normalize(x), x_norm, atol=1e-6)
    meta = normalization_metadata(spec)
    assert meta["projection_space"] == "pixel (non-normalized)"
    results["spec_ok"] = True

    if verbose:
        print("robust_clip_repro.utils.normalization self-test: OK")
        for key, value in results.items():
            print(f"  - {key}: {value}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pixel-space / model-normalization helpers.")
    parser.add_argument("--self-test", action="store_true", help="run the offline self-test")
    parser.add_argument("--normalization", default="clip", choices=sorted(NORMALIZATION_REGISTRY))
    parser.add_argument("--provenance", action="store_true", help="print normalization provenance")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    if args.provenance:
        import json

        print(json.dumps(normalization_metadata(args.normalization), indent=2))
        return 0
    if args.self_test or not any([args.provenance]):
        _self_test(verbose=not args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
