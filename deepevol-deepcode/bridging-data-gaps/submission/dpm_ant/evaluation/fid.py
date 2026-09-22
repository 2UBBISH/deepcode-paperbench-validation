"""FID evaluation for DPMs-ANT.

Reproduces the FID protocol of Section 5.2 ("Evaluation Metrics") of
*Adapting Pretrained Diffusion Models for Few-Shot Image Generation*:

    "FID is a widely used metric for assessing the generation quality of
    generative models by calculating the distribution distances between
    generated samples and datasets. However, FID may become unstable and
    unreliable when applied to datasets with few samples, such as the 10-shot
    datasets used in this paper. Following DDPM-PA (Zhu et al., 2022), we
    provide FID evaluations using larger target datasets, such as Sunglasses
    and Babies, consisting of 2.5k and 2.7k images, respectively."

So:
  * FID is computed between *generated* images and a **larger** target set
    (Sunglasses 2,500 images, Babies 2,700 images).
  * Lower is better.
  * 10-shot FID is disabled by default (``compute_10shot_fid: false``) because
    the paper explicitly calls it unstable.

Backends (tried in order, whichever is installed first):
  1. ``clean-fid``          -> ``cleanfid.fid`` (default, matches the plan)
  2. ``pytorch-fid``        -> ``pytorch_fid.fid_score``
  3. pure-torch fallback    -> InceptionV3 pool3 features + Frechet distance
                              (implemented locally, no third-party dependency)

All tensors used by the rest of this code base live in ``[-1, 1]`` (DDPM
convention, see ``dpm_ant/data/datasets.py``); they are converted back to
uint8 images before being handed to any FID backend.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

__all__ = [
    "FIDConfig",
    "FIDResult",
    "FIDMetric",
    "compute_fid",
    "compute_fid_from_dirs",
    "frechet_distance",
    "activation_statistics",
    "extract_inception_features",
    "save_images",
    "load_fid_reference",
    "build_fid_metric",
    "TARGET_FID_SIZES",
    "PAPER_FID_REFERENCE",
]

# ---------------------------------------------------------------------------
# Paper reference numbers (Table 2 / Table 3 / Appendix B.3)
# ---------------------------------------------------------------------------

#: Number of images in the "larger target datasets" used for FID (§5.2).
TARGET_FID_SIZES: Dict[str, int] = {
    "babies": 2700,  # "2.7k" images
    "sunglasses": 2500,  # "2.5k" images
}

#: FID numbers reported in the paper, used for automated comparisons.
PAPER_FID_REFERENCE: Dict[str, Dict[str, float]] = {
    "ddpm_ant": {"babies": 46.70, "sunglasses": 20.06},
    "ablation_sunglasses_300iters": {
        # Figure 4 (300 iterations on 10-shot Sunglasses)
        "full_model_finetune": 41.88,
        "adaptor_only": 38.65,
        "ant_wo_an": 26.41,
        "full_ant": 20.66,
    },
    "sensitivity": {
        # Appendix B.3 Tables 5-7 (the best configuration reaches FID 18.13)
        "best_gamma": 5.0,
        "best_omega": 0.02,
        "best_iterations": 300,
        "best_fid": 18.13,
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FIDConfig:
    """Configuration of the FID evaluation.

    Fields
    ------
    num_samples:
        Number of generated images used for the FID computation.  ``None`` uses
        every available generated image.
    target:
        Name of the target dataset (``"sunglasses"``, ``"babies"``, ...).
    target_size:
        Number of reference images.  Defaults to the paper value
        (:data:`TARGET_FID_SIZES`) when the target is known.
    backend:
        ``"auto"`` (default), ``"clean-fid"``, ``"pytorch-fid"`` or
        ``"torch"`` (the dependency-free fallback).
    batch_size:
        Batch size used when extracting features.
    device:
        Torch device string.
    resize:
        Resize images to this size before feature extraction.  FID backends
        default to 299 for Inception; ``None`` keeps the native resolution.
    compute_10shot:
        Compute FID against the 10-shot target set too.  Disabled by default
        ("FID may become unstable and unreliable when applied to datasets with
        few samples" -- §5.2).
    feature_dim:
        Dimensionality of the fallback features (InceptionV3 ``pool3`` = 2048).
    use_inception_download:
        Allow ``torchvision`` to download InceptionV3 weights (fallback path).
    seed:
        Optional seed for reproducible subsampling.
    """

    num_samples: Optional[int] = None
    target: Optional[str] = None
    target_size: Optional[int] = None
    backend: str = "auto"
    batch_size: int = 32
    device: str = "cpu"
    resize: Optional[int] = 299
    compute_10shot: bool = False
    feature_dim: int = 2048
    use_inception_download: bool = True
    seed: Optional[int] = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.target_size is None and self.target is not None:
            key = str(self.target).lower().replace("-", "_")
            for name, size in TARGET_FID_SIZES.items():
                if name in key:
                    self.target_size = size
                    break

    @property
    def expected_reference_size(self) -> Optional[int]:
        return self.target_size

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, cfg: Optional[dict] = None, **overrides) -> "FIDConfig":
        """Build a config from a (possibly nested) YAML config dict.

        Recognised blocks: ``evaluation.fid``, ``evaluation``, ``data``,
        ``tasks.<task>`` and top level keys.
        """
        kwargs: Dict[str, object] = {}
        cfg = cfg or {}

        def _merge(src: dict) -> None:
            for key in (
                "num_samples",
                "target",
                "target_size",
                "backend",
                "batch_size",
                "device",
                "resize",
                "feature_dim",
                "use_inception_download",
                "seed",
            ):
                if isinstance(src, dict) and key in src and src[key] is not None:
                    kwargs[key] = src[key]
            if isinstance(src, dict):
                if "compute_10shot_fid" in src:
                    kwargs["compute_10shot"] = bool(src["compute_10shot_fid"])
                if "compute_10shot" in src:
                    kwargs["compute_10shot"] = bool(src["compute_10shot"])

        for block in ("evaluation", "fid"):
            _merge(cfg.get(block, {}) if isinstance(cfg, dict) else {})
        if isinstance(cfg, dict) and isinstance(cfg.get("evaluation"), dict):
            _merge(cfg["evaluation"].get("fid", {}))
        _merge(cfg)
        _merge(overrides)

        kwargs.setdefault("device", overrides.get("device", "cpu"))
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "target": self.target,
            "target_size": self.target_size,
            "backend": self.backend,
            "batch_size": self.batch_size,
            "device": self.device,
            "resize": self.resize,
            "compute_10shot": self.compute_10shot,
            "feature_dim": self.feature_dim,
        }

    def replace(self, **overrides) -> "FIDConfig":
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return FIDConfig(**data)


@dataclass
class FIDResult:
    """Container for a FID measurement."""

    fid: float
    num_generated: int
    num_reference: int
    backend: str
    target: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "fid": self.fid,
            "num_generated": self.num_generated,
            "num_reference": self.num_reference,
            "backend": self.backend,
        }
        if self.target is not None:
            out["target"] = self.target
        out.update(self.extra)
        return out

    def __float__(self) -> float:  # pragma: no cover - convenience
        return float(self.fid)


# ---------------------------------------------------------------------------
# Frechet distance (the actual FID formula)
# ---------------------------------------------------------------------------


def frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Frechet Inception Distance between two Gaussians.

    ``FID = ||mu_1 - mu_2||^2 + Tr(Sigma_1 + Sigma_2 - 2 (Sigma_1 Sigma_2)^{1/2})``

    A tiny ridge is added to the covariance diagonals for numerical stability
    (the standard scipy/``pytorch-fid`` trick).
    """
    mu1 = mu1.double().flatten()
    mu2 = mu2.double().flatten()
    sigma1 = sigma1.double()
    sigma2 = sigma2.double()

    if mu1.shape != mu2.shape:
        raise ValueError(
            f"Feature dimensionality mismatch: {mu1.shape} vs {mu2.shape}"
        )

    diff = mu1 - mu2

    eye = torch.eye(sigma1.shape[0], dtype=torch.float64, device=sigma1.device)
    sigma1 = sigma1 + eye * eps
    sigma2 = sigma2 + eye * eps

    # Product of the two covariance matrices; sqrtm via eigendecomposition
    # (stable for symmetric PSD matrices, avoids scipy dependency).
    covmean = _sqrtm_symmetrized(sigma1 @ sigma2)

    if not torch.isfinite(covmean).all():  # pragma: no cover - numeric guard
        LOGGER.warning("FID: non-finite sqrtm result, adding eps * I^2")
        covmean = _sqrtm_symmetrized(sigma1 @ sigma2 + eye * eps)

    trace = torch.trace(sigma1) + torch.trace(sigma2) - 2.0 * torch.trace(covmean)
    fid = float((diff @ diff).item() + trace.item())
    if not math.isfinite(fid):  # pragma: no cover - numeric guard
        LOGGER.warning("FID: non-finite value %s, clamping", fid)
        fid = float("inf")
    return max(fid, 0.0) if math.isfinite(fid) else fid


def _sqrtm_symmetrized(matrix: torch.Tensor) -> torch.Tensor:
    """Numerically safe matrix square root of a symmetric PSD matrix."""
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    try:
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=0.0)
        return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.transpose(-1, -2)
    except Exception:  # pragma: no cover - extremely defensive
        vals, vecs = torch.linalg.eig(matrix)
        vals = torch.clamp(vals.real, min=0.0)
        return (vecs.real * vals.sqrt().unsqueeze(0)) @ torch.linalg.pinv(vecs.real)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def activation_statistics(
    features: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mu, sigma)`` of a ``(N, D)`` feature matrix."""
    features = features.reshape(features.shape[0], -1).double()
    n = features.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to compute FID statistics")
    mu = features.mean(dim=0)
    centred = features - mu
    sigma = (centred.transpose(0, 1) @ centred) / (n - 1)
    return mu, sigma


def _to_uint8_images(images: torch.Tensor) -> torch.Tensor:
    """Convert ``[-1, 1]`` (or ``[0, 1]``) float tensors to uint8 ``[0, 255]``."""
    images = images.detach().float()
    if images.dim() == 3:
        images = images.unsqueeze(0)
    # Auto-detect the convention used by the caller.
    min_v, max_v = float(images.min()), float(images.max())
    if min_v >= -0.01 and max_v <= 1.01:
        images = images * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    return (images * 255.0).round().clamp(0, 255).to(torch.uint8)


def _to_float_images(images: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_to_uint8_images`, returned in ``[-1, 1]``."""
    images = images.float()
    if images.max() > 1.5:  # uint8 range
        images = images / 255.0
    return images * 2.0 - 1.0


def save_images(images: torch.Tensor, out_dir: str, prefix: str = "") -> List[str]:
    """Save a ``(N, 3, H, W)`` tensor (in ``[-1, 1]``) as PNG files.

    Returns the list of written paths.  Falls back to ``.pt`` files when PIL is
    unavailable so the FID backends can still be fed via a directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    images = _to_uint8_images(images).cpu()
    paths: List[str] = []
    try:  # pragma: no cover - PIL is normally available
        from PIL import Image  # type: ignore

        for i, img in enumerate(images):
            arr = img.permute(1, 2, 0).numpy()
            path = os.path.join(out_dir, f"{prefix}{i:06d}.png")
            Image.fromarray(arr).save(path)
            paths.append(path)
        return paths
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("save_images: PIL unavailable (%s); saving .pt tensors", exc)
        for i, img in enumerate(images):
            path = os.path.join(out_dir, f"{prefix}{i:06d}.pt")
            torch.save(img, path)
            paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# Dependency-free fallback feature extractor (InceptionV3 pool3)
# ---------------------------------------------------------------------------


class _InceptionFeatureExtractor(nn.Module):
    """InceptionV3 feature extractor used by the dependency-free FID path.

    By default this reproduces the standard "FID features": the 2048-d
    ``pool3`` (avg-pool) activations of InceptionV3, evaluated at the native
    Inception resolution of 299x299.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        device: Union[str, torch.device] = "cpu",
        allow_download: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.feature_dim = feature_dim
        self.allow_download = allow_download
        self.net: Optional[nn.Module] = None
        self._hook_output: Optional[torch.Tensor] = None
        self._error: Optional[str] = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        try:  # pragma: no cover - depends on torchvision availability
            import torchvision  # noqa: F401
            from torchvision.models import inception_v3

            try:
                from torchvision.models import Inception_V3_Weights  # type: ignore

                weights = (
                    Inception_V3_Weights.IMAGENET1K_V1
                    if self.allow_download
                    else None
                )
                model = inception_v3(weights=weights, aux_logits=True)
            except Exception:  # older torchvision
                model = inception_v3(
                    pretrained=self.allow_download, aux_logits=True
                )
            model.eval()
            model.to(self.device)
            model.fc = nn.Identity()  # keep pool3 features
            self.net = model
            # Register a hook on the average pooling layer (pool3 equivalent).
            pool = getattr(model, "avgpool", None)
            if isinstance(pool, nn.Module):
                pool.register_forward_hook(self._capture)
            return
        except Exception as exc:  # pragma: no cover
            self._error = str(exc)
            LOGGER.warning(
                "InceptionV3 unavailable (%s); FID fallback uses downsampled "
                "raw-pixel statistics instead of Inception features.",
                exc,
            )

    def _capture(self, _module, _inp, output) -> None:  # pragma: no cover
        self._hook_output = output

    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features for a ``[-1, 1]`` batch; returns ``(N, D)``."""
        images = images.to(self.device)
        if self.resize_to is not None and images.shape[-1] != self.resize_to:
            images = F.interpolate(
                images,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            )
        if self.net is None:
            # Degenerate but deterministic fallback: 64x64 global average
            # pooled pixels followed by a fixed random projection.
            small = F.adaptive_avg_pool2d(images, 64).flatten(1)
            return self._project(small)
        with torch.no_grad():
            self._hook_output = None
            try:
                self.net(images)
            except Exception:  # pragma: no cover - e.g. aux logits mismatch
                pass
            feats = self._hook_output
            if feats is None:  # pragma: no cover
                out = self.net(images)
                feats = out if isinstance(out, torch.Tensor) else out[0]
            if feats.dim() == 4:
                feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            else:
                feats = feats.flatten(1)
        if feats.shape[1] != self.feature_dim:
            feats = self._project(feats)
        return feats

    # ------------------------------------------------------------------
    resize_to: Optional[int] = 299

    def _project(self, feats: torch.Tensor) -> torch.Tensor:
        """Deterministic random projection to ``feature_dim`` (fallback only)."""
        d_in = feats.shape[1]
        if d_in == self.feature_dim:
            return feats
        gen = torch.Generator(device="cpu").manual_seed(0)
        proj = torch.randn(d_in, self.feature_dim, generator=gen)
        proj = proj / math.sqrt(max(d_in, 1))
        proj = proj.to(feats.device, feats.dtype)
        return feats @ proj


def extract_inception_features(
    images: torch.Tensor,
    extractor: nn.Module,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract features for a full image tensor, batch by batch."""
    feats: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            chunk = images[start : start + batch_size]
            feats.append(extractor(chunk).detach().cpu())
    return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# Main metric class
# ---------------------------------------------------------------------------


class FIDMetric:
    """Compute FID between generated samples and a (larger) target set.

    Parameters
    ----------
    config:
        :class:`FIDConfig`; when ``None`` a default config is used.
    device:
        Overrides ``config.device``.
    """

    def __init__(
        self,
        config: Optional[FIDConfig] = None,
        device: Optional[str] = None,
        **overrides,
    ) -> None:
        cfg = config or FIDConfig()
        if overrides:
            cfg = cfg.replace(**overrides)
        if device is not None:
            cfg.device = device
        self.config = cfg
        self.device = torch.device(cfg.device)
        self._extractor: Optional[nn.Module] = None
        self._backend: Optional[str] = None
        self._tmpdir: Optional[str] = None

    # ------------------------------------------------------------------
    # Backend discovery
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        if self._backend is None:
            self._backend = self._resolve_backend(self.config.backend)
        return self._backend

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        requested = (requested or "auto").lower()
        if requested in ("torch", "fallback", "native"):
            return "torch"
        if requested in ("clean-fid", "cleanfid", "clean_fid"):
            return "clean-fid"
        if requested in ("pytorch-fid", "pytorch_fid", "pytorchfid", "pt"):
            return "pytorch-fid"
        # auto: prefer clean-fid, then pytorch-fid, then the local fallback.
        try:  # pragma: no cover - optional dependency
            import cleanfid  # noqa: F401

            return "clean-fid"
        except Exception:
            pass
        try:  # pragma: no cover - optional dependency
            import pytorch_fid  # noqa: F401

            return "pytorch-fid"
        except Exception:
            return "torch"

    # ------------------------------------------------------------------
    # Directory based computation (used by clean-fid / pytorch-fid)
    # ------------------------------------------------------------------
    def _make_tmpdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="dpm_ant_fid_")
        return self._tmpdir

    def _run_dir_backend(
        self, gen_dir: str, ref_dir: str, backend: str
    ) -> Optional[float]:
        """Dispatch to clean-fid / pytorch-fid.  ``None`` if unavailable."""
        if backend == "clean-fid":
            try:  # pragma: no cover - optional dependency
                from cleanfid import fid as cleanfid_fid  # type: ignore

                return float(
                    cleanfid_fid.compute_fid(
                        gen_dir,
                        ref_dir,
                        mode="clean",
                        num_workers=0,
                        batch_size=self.config.batch_size,
                        device=self.device,
                        verbose=False,
                    )
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("clean-fid failed (%s); falling back", exc)
                return None
        if backend == "pytorch-fid":
            try:  # pragma: no cover - optional dependency
                from pytorch_fid.fid_score import (  # type: ignore
                    calculate_fid_given_paths,
                )

                return float(
                    calculate_fid_given_paths(
                        [gen_dir, ref_dir],
                        batch_size=self.config.batch_size,
                        device=self.device,
                        dims=self.config.feature_dim,
                        num_workers=0,
                    )
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("pytorch-fid failed (%s); falling back", exc)
                return None
        return None

    # ------------------------------------------------------------------
    # Local (dependency-free) computation
    # ------------------------------------------------------------------
    def _get_extractor(self) -> nn.Module:  # pragma: no cover - needs torch/vision
        if self._extractor is None:
            extractor = _InceptionFeatureExtractor(
                feature_dim=self.config.feature_dim,
                device=self.device,
                allow_download=self.config.use_inception_download,
            )
            extractor.resize_to = self.config.resize
            self._extractor = extractor
        return self._extractor

    def _features_of(self, images: torch.Tensor, use_extractor: bool = True) -> torch.Tensor:
        if not use_extractor:
            small = F.adaptive_avg_pool2d(images.to(self.device), 8).flatten(1)
            return small.cpu()
        extractor = self._get_extractor()
        return extract_inception_features(
            images, extractor, batch_size=self.config.batch_size
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(
        self,
        generated: Union[torch.Tensor, str, Sequence[torch.Tensor]],
        reference: Optional[Union[torch.Tensor, str]] = None,
        reference_dir: Optional[str] = None,
        return_details: bool = False,
    ) -> Dict[str, object]:
        """Compute FID between ``generated`` and the reference target set.

        ``generated`` may be a ``[-1, 1]`` tensor of shape ``(N, 3, H, W)`` or a
        directory containing images.  The reference may be a directory
        (``reference_dir`` / ``reference`` as a string) or an in-memory tensor.
        """
        cfg = self.config

        # -- generated images -------------------------------------------
        gen_dir: Optional[str] = None
        gen_tensor: Optional[torch.Tensor] = None
        if isinstance(generated, str):
            gen_dir = generated
        else:
            images = generated
            if isinstance(images, (list, tuple)):
                images = torch.cat([t if t.dim() == 4 else t.unsqueeze(0) for t in images], 0)
            images = images.detach().float()
            if cfg.num_samples is not None and images.shape[0] > cfg.num_samples:
                images = images[: cfg.num_samples]
            gen_tensor = images

        # -- reference images -------------------------------------------
        ref_dir = reference_dir
        ref_tensor: Optional[torch.Tensor] = None
        if isinstance(reference, str) and ref_dir is None:
            ref_dir = reference
        elif torch.is_tensor(reference):
            ref_tensor = reference.detach().float()

        backend = self.backend
        fid_value: Optional[float] = None
        used_backend = backend

        if backend in ("clean-fid", "pytorch-fid"):
            if gen_dir is None:
                gen_dir = os.path.join(self._make_tmpdir(), "generated")
                save_images(gen_tensor, gen_dir, prefix=f"{cfg.target or 'gen'}_")
            if ref_dir is None and ref_tensor is not None:
                ref_dir = os.path.join(self._make_tmpdir(), "reference")
                save_images(ref_tensor, ref_dir, prefix="ref_")
            if ref_dir is not None and os.path.isdir(gen_dir or "") and os.path.isdir(ref_dir):
                fid_value = self._run_dir_backend(gen_dir, ref_dir, backend)
            if fid_value is None:
                LOGGER.info("Falling back to the dependency-free FID backend")
                used_backend = "torch"
        else:
            used_backend = "torch"

        if fid_value is None:
            if ref_tensor is None:
                if ref_dir is None:
                    raise ValueError(
                        "FID requires either a reference directory or a "
                        "reference image tensor."
                    )
                ref_tensor = load_fid_reference(
                    ref_dir,
                    limit=cfg.target_size,
                    size=None,
                    device="cpu",
                )
            if gen_tensor is None:
                raise ValueError(
                    "The fallback FID backend needs generated images as a "
                    "tensor (directories are only supported by clean-fid / "
                    "pytorch-fid)."
                )
            gen_feats = self._features_of(gen_tensor)
            ref_feats = self._features_of(ref_tensor)
            mu1, s1 = activation_statistics(gen_feats)
            mu2, s2 = activation_statistics(ref_feats)
            fid_value = frechet_distance(mu1, s1, mu2, s2)
            used_backend = "torch"

        num_gen = (
            int(gen_tensor.shape[0])
            if gen_tensor is not None
            else len(os.listdir(gen_dir)) if gen_dir else 0
        )
        num_ref = (
            int(ref_tensor.shape[0])
            if ref_tensor is not None
            else len(os.listdir(ref_dir)) if ref_dir else 0
        )

        result = FIDResult(
            fid=float(fid_value),
            num_generated=num_gen,
            num_reference=num_ref,
            backend=used_backend,
            target=cfg.target,
            extra={
                "expected_reference_size": cfg.expected_reference_size,
                "paper_reference": (
                    PAPER_FID_REFERENCE["ddpm_ant"].get(cfg.target)
                    if cfg.target
                    else None
                ),
                "resize": cfg.resize,
            },
        )
        out = result.to_dict()
        if return_details:
            out["generated_dir"] = gen_dir
            out["reference_dir"] = ref_dir
        return out

    # Convenience alias -------------------------------------------------
    __call__ = compute

    def compute_10shot(
        self,
        generated: Union[torch.Tensor, str],
        ten_shot_ref: Union[torch.Tensor, str],
        **kwargs,
    ) -> Dict[str, object]:
        """FID against the 10-shot set (off by default; kept for completeness)."""
        if not self.config.compute_10shot:
            warnings.warn(
                "10-shot FID is disabled by default (Section 5.2: FID on 10-shot "
                "datasets is unstable). Pass compute_10shot=True to force it.",
                UserWarning,
            )
            if not self.config.compute_10shot:
                return {
                    "fid": float("nan"),
                    "skipped": True,
                    "reason": "10-shot FID disabled (unstable per Section 5.2)",
                }
        if isinstance(ten_shot_ref, str):
            return self.compute(generated, reference_dir=ten_shot_ref, **kwargs)
        return self.compute(generated, reference=ten_shot_ref, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"FIDMetric(target={self.config.target}, "
            f"target_size={self.config.target_size}, "
            f"backend={self.config.backend})"
        )

    def cleanup(self) -> None:
        """Remove temporary directories created for directory-based backends."""
        if self._tmpdir and os.path.isdir(self._tmpdir):
            import shutil

            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------


def load_fid_reference(
    ref_dir: str,
    limit: Optional[int] = None,
    size: Optional[int] = None,
    device: str = "cpu",
    recursive: bool = True,
) -> torch.Tensor:
    """Load reference images from a directory as a ``[-1, 1]`` tensor.

    Uses ``dpm_ant.data.datasets`` helpers when importable so preprocessing
    matches the generation pipeline exactly.
    """
    try:
        from ..data import datasets as _ds  # type: ignore

        paths = _ds.list_images(ref_dir, recursive=recursive)
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise FileNotFoundError(f"No images found under {ref_dir!r}")
        imgs = [_ds.load_image(p, size=size, resize=size is not None) for p in paths]
        return torch.stack(imgs, dim=0).to(device)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - standalone fallback
        LOGGER.warning("load_fid_reference: falling back to local loader (%s)", exc)

    from PIL import Image  # type: ignore
    import numpy as np

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pt", ".pth", ".npy")
    files: List[str] = []
    for root, _dirs, names in os.walk(ref_dir):
        for name in sorted(names):
            if name.lower().endswith(exts):
                files.append(os.path.join(root, name))
        if not recursive:
            break
    files = sorted(files)
    if limit is not None:
        files = files[:limit]

    out: List[torch.Tensor] = []
    for path in files:
        if path.endswith((".pt", ".pth")):
            t = torch.load(path, map_location="cpu")
            if t.dtype == torch.uint8:
                t = _to_float_images(t)
            out.append(t.float())
            continue
        if path.endswith(".npy"):
            arr = np.load(path)
            t = torch.from_numpy(arr)
            out.append(_to_float_images(t))
            continue
        img = Image.open(path).convert("RGB")
        if size is not None:
            w, h = img.size
            scale = size / min(w, h)
            img = img.resize((max(size, int(round(w * scale))), max(size, int(round(h * scale)))), Image.BICUBIC)
            w, h = img.size
            left, top = (w - size) // 2, (h - size) // 2
            img = img.crop((left, top, left + size, top + size))
        arr = np.asarray(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        out.append(t * 2.0 - 1.0)
    if not out:
        raise FileNotFoundError(f"No images found under {ref_dir!r}")
    return torch.stack(out, dim=0).to(device)


def compute_fid(
    generated: Union[torch.Tensor, str, Sequence[torch.Tensor]],
    reference: Optional[Union[torch.Tensor, str]] = None,
    reference_dir: Optional[str] = None,
    cfg: Optional[dict] = None,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_samples: Optional[int] = None,
    target: Optional[str] = None,
    target_size: Optional[int] = None,
    return_details: bool = False,
    metric: Optional[FIDMetric] = None,
    **overrides,
) -> Dict[str, object]:
    """One-shot FID computation.

    ``reference``/``reference_dir`` point at the larger target dataset
    (Sunglasses 2.5k, Babies 2.7k) per Section 5.2.  Returns a dict containing
    ``fid`` (lower is better) plus bookkeeping fields.
    """
    if metric is None:
        config = FIDConfig.from_dict(cfg, **overrides)
        if device is not None:
            config.device = device
        if backend is not None:
            config.backend = backend
        if batch_size is not None:
            config.batch_size = batch_size
        if num_samples is not None:
            config.num_samples = num_samples
        if target is not None:
            config.target = target
            if target_size is None:
                config.__post_init__()
        if target_size is not None:
            config.target_size = target_size
        metric = FIDMetric(config)
    return metric.compute(
        generated, reference=reference, reference_dir=reference_dir,
        return_details=return_details,
    )


def compute_fid_from_dirs(
    generated_dir: str,
    reference_dir: str,
    cfg: Optional[dict] = None,
    device: Optional[str] = None,
    backend: Optional[str] = None,
    target: Optional[str] = None,
    num_samples: Optional[int] = None,
    return_details: bool = False,
) -> Dict[str, object]:
    """Compute FID between two image directories (the DDPM-PA evaluation style)."""
    return compute_fid(
        generated_dir,
        reference_dir=reference_dir,
        cfg=cfg,
        device=device,
        backend=backend,
        target=target,
        num_samples=num_samples,
        return_details=return_details,
    )


def build_fid_metric(
    cfg: Optional[dict] = None,
    device: Optional[str] = None,
    target: Optional[str] = None,
    **overrides,
) -> FIDMetric:
    """Build a :class:`FIDMetric` from a (nested) config dict."""
    config = FIDConfig.from_dict(cfg, **overrides)
    if device is not None:
        config.device = device
    if target is not None:
        config.target = target
        config.__post_init__()
    return FIDMetric(config)
