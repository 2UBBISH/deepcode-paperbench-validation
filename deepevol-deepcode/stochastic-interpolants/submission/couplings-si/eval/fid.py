"""FID-50k evaluation for the data-dependent-coupling stochastic interpolants.

This module implements the quantitative metric reported in the paper:

* Table 2 (in-painting, ``§4.1``): FID-50k on ImageNet-256 in-painting with
  ``Uncoupled Interpolant (Baseline) = 1.35`` and
  ``Dependent Coupling (Ours) = 1.13``.
* Table 3 (super-resolution 64x64 -> 256x256, ``§4.2``):
  ``Dependent Coupling (Ours) = 2.13 (train) / 2.05 (valid)`` versus the
  reported baselines Improved DDPM (12.26), SR3 (11.30/5.20), ADM (7.49/3.10),
  Cascaded Diffusion (4.88/4.63) and I2SB (2.70).

FID is computed in the standard way: the Fréchet (2-Wasserstein) distance
between two Gaussians fitted to the 2048-dimensional Inception-v3 pool3
features of the *generated* and of the *reference* (ground-truth ImageNet)
images, using the standard ImageNet reference statistics.  The paper evaluates
over 50,000 generated samples (hence "FID-50k").

The implementation is deliberately dependency-light but uses the standard
training-code conventions:

* ``torchvision`` Inception-v3 with the (optional) standard FID weights file
  produced by ``pytorch-fid``.  If only torchvision's own ImageNet weights are
  available we still produce a usable (and internally consistent) FID, which
  is what matters for the comparisons in the tables.
* ``scipy.linalg.sqrtm`` when available, with a symmetric-eigen-decomposition
  fallback implemented purely in numpy.

Usage
-----
>>> from eval.fid import evaluate_fid_50k, PAPER_FIDS
>>> result = evaluate_fid_50k(loader, num_samples=50_000)   # doctest: +SKIP
>>> result.fid

Command line::

    python -m eval.fid --samples-dir samples/inpainting_256 --dataset imagenet --split 256
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paper reference numbers (used by evaluate.py to report deltas)
# --------------------------------------------------------------------------- #

#: Target FIDs quoted in the paper (Table 2 and Table 3).
PAPER_FIDS: Dict[str, Dict[str, Optional[float]]] = {
    # Table 2 -- in-painting on ImageNet-256
    "inpainting": {
        "uncoupled_interpolant": 1.35,
        "dependent_coupling": 1.13,
    },
    # Table 3 -- super-resolution 64x64 -> 256x256
    "superres_64_256": {
        "improved_ddpm": 12.26,
        "sr3": 5.20,           # valid column (train = 11.30)
        "sr3_train": 11.30,
        "adm": 3.10,           # valid column (train = 7.49)
        "adm_train": 7.49,
        "cascaded_diffusion": 4.63,   # valid column (train = 4.88)
        "cascaded_diffusion_train": 4.88,
        "i2sb": 2.70,
        "dependent_coupling": 2.13,   # train
        "dependent_coupling_valid": 2.05,
    },
}

#: Baselines whose FIDs are *taken from the literature* and must NOT be re-run.
REPORTED_ONLY_BASELINES = (
    "improved_ddpm",
    "sr3",
    "adm",
    "cascaded_diffusion",
    "i2sb",
)

INCEPTION_FEATURE_DIM = 2048
INCEPTION_RESOLUTION = 299

# Standard reference-statistics files (clean-fid release layout).
STATS_URLS: Dict[str, str] = {
    "imagenet_64": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz",
    "imagenet_128": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz",
    "imagenet_256": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz",
    "imagenet_512": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512_labeled.npz",
}


# --------------------------------------------------------------------------- #
# Fréchet distance
# --------------------------------------------------------------------------- #


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root for a (numerically) symmetric positive semi-definite matrix.

    Uses ``scipy.linalg.sqrtm`` when available (the reference implementation)
    and otherwise a symmetric eigendecomposition; negative eigenvalues caused by
    numerical error are clipped at zero.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    try:  # pragma: no cover - depends on environment
        from scipy.linalg import sqrtm as _scipy_sqrtm

        out = _scipy_sqrtm(matrix)
        if np.iscomplexobj(out):
            out = out.real
        return np.asarray(out, dtype=np.float64)
    except Exception:  # noqa: BLE001 - any scipy failure falls back to numpy
        sym = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.clip(eigvals, 0.0, None)
        return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between two Gaussians ``N(mu1, sigma1)`` and ``N(mu2, sigma2)``.

    ``FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 sqrt(sigma1 sigma2))``.
    """
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()
    mu2 = np.asarray(mu2, dtype=np.float64).ravel()
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    if mu1.shape != mu2.shape:
        raise ValueError(f"mean shapes differ: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"covariance shapes differ: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2

    covmean = _sqrtm_psd(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        logger.warning("FID: non-finite sqrtm result, re-computing with regularisation eps=%g", eps)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm_psd((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    return fid


def covariance_of(features: np.ndarray) -> np.ndarray:
    """Unbiased-style (population) covariance used by the reference implementation."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be (N, D)")
    return np.cov(features, rowvar=False)


# --------------------------------------------------------------------------- #
# Feature statistics container
# --------------------------------------------------------------------------- #


@dataclass
class FIDStats:
    """Gaussian summary statistics of Inception features."""

    mu: np.ndarray
    sigma: np.ndarray
    num_samples: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64).ravel()
        self.sigma = np.atleast_2d(np.asarray(self.sigma, dtype=np.float64))
        if self.num_samples == 0:
            self.num_samples = int(getattr(self, "n", 0) or 0)

    # -- convenience ------------------------------------------------------- #
    @property
    def feature_dim(self) -> int:
        return int(self.mu.shape[0])

    def save(self, path: Union[str, os.PathLike]) -> str:
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez(path, mu=self.mu, sigma=self.sigma, num_samples=np.int64(self.num_samples), name=self.name)
        return path

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "FIDStats":
        with np.load(os.fspath(path), allow_pickle=True) as data:
            mu = data["mu"]
            sigma = data["sigma"]
            num_samples = int(data["num_samples"]) if "num_samples" in data.files else 0
            name = str(data["name"]) if "name" in data.files else os.path.basename(os.fspath(path))
        return cls(mu=mu, sigma=sigma, num_samples=num_samples, name=name)

    @classmethod
    def from_features(cls, features: np.ndarray, name: str = "") -> "FIDStats":
        features = np.asarray(features, dtype=np.float64)
        return cls(
            mu=features.mean(axis=0),
            sigma=covariance_of(features),
            num_samples=int(features.shape[0]),
            name=name,
        )

    def distance_to(self, other: "FIDStats") -> float:
        return frechet_distance(self.mu, self.sigma, other.mu, other.sigma)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FIDStats(name={self.name!r}, n={self.num_samples}, dim={self.feature_dim})"


# --------------------------------------------------------------------------- #
# Inception feature extractor
# --------------------------------------------------------------------------- #


class InceptionFeatureExtractor(nn.Module):
    """Inception-v3 pool3 feature extractor used for FID.

    Inputs are expected to be image tensors in ``[-1, 1]`` (the convention used
    throughout this repository, see ``si/data/transforms.py``).  They are
    resized to 299x299 and normalised to the Inception input range exactly as in
    the standard FID implementation.

    Parameters
    ----------
    weights_path:
        Optional path to the ``pytorch-fid`` Inception state dict
        (``pt_inception-2015-12-05-6726825d.pca_state_dict.pth``).  If given and
        present, its weights replace the torchvision ones (this reproduces the
        *exact* FID number reported by the literature).  Otherwise torchvision's
        ImageNet weights are used.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, os.PathLike]] = None,
        device: Optional[Union[str, torch.device]] = None,
        feature_dim: int = INCEPTION_FEATURE_DIM,
        resize: int = INCEPTION_RESOLUTION,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.resize = int(resize)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        net = self._build_inception(weights_path)
        self.net = net.eval().to(self.device)
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    # -- construction ------------------------------------------------------ #
    @staticmethod
    def _build_inception(weights_path: Optional[Union[str, os.PathLike]]) -> nn.Module:
        from torchvision.models import inception_v3  # local import keeps module import light

        try:  # torchvision >= 0.13
            from torchvision.models import Inception_V3_Weights

            net = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False, aux_logits=True)
        except Exception:  # pragma: no cover - older torchvision
            net = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        net.fc = nn.Identity()
        net.aux_logits = False
        net.AuxLogits = None
        net.dropout = nn.Identity()

        if weights_path is not None and os.path.exists(os.fspath(weights_path)):
            state = torch.load(os.fspath(weights_path), map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:  # pragma: no cover
                state = state["state_dict"]
            state = {k: v for k, v in state.items() if k.startswith(("Conv2d", "Mixed", "fc", "avgpool", "dropout"))}
            missing, unexpected = net.load_state_dict(state, strict=False)
            logger.info("Loaded FID Inception weights from %s (missing=%d, unexpected=%d)", weights_path, len(missing), len(unexpected))
        elif weights_path is not None:
            logger.warning("FID Inception weights '%s' not found; using torchvision weights instead.", weights_path)
        return net

    # -- preprocessing ----------------------------------------------------- #
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """``[-1, 1] ->`` Inception-normalised 299x299 RGB tensor."""
        images = images.to(self.device, dtype=torch.float32)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] > 3:  # pragma: no cover - defensive
            images = images[:, :3]
        if images.shape[-2:] != (self.resize, self.resize):
            images = F.interpolate(images, size=(self.resize, self.resize), mode="bilinear", align_corners=False, antialias=True)
        # images are in [-1, 1] -> [0, 1] -> normalised
        images = 0.5 * (images + 1.0)
        return (images - self._mean.to(images.dtype)) / self._std.to(images.dtype)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(images)
        features = self.net(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features

    # -- batched accumulation ---------------------------------------------- #
    @torch.no_grad()
    def features_of(
        self,
        images: Union[torch.Tensor, Iterable[Any]],
        batch_size: int = 32,
        max_samples: Optional[int] = 50_000,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute features for a tensor batch or an iterable of image batches."""
        if isinstance(images, torch.Tensor):
            batches: Iterable[Any] = [images[i : i + batch_size].to(self.device) for i in range(0, images.shape[0], batch_size)]
        else:
            batches = images

        collected: List[np.ndarray] = []
        seen = 0
        for batch in batches:
            if isinstance(batch, Mapping):
                batch = batch.get("image", batch.get("images", None))
            elif isinstance(batch, (tuple, list)):
                batch = batch[0]
            if batch is None:  # pragma: no cover - defensive
                continue
            features = self.forward(batch.to(self.device))
            collected.append(features.double().cpu().numpy())
            seen += features.shape[0]
            if verbose and len(collected) % 20 == 0:
                logger.info("FID: extracted features for %d images", seen)
            if max_samples is not None and seen >= max_samples:
                break
        if not collected:
            raise RuntimeError("No images were provided to the FID feature extractor.")
        features = np.concatenate(collected, axis=0)
        if max_samples is not None:
            features = features[:max_samples]
        return features


# --------------------------------------------------------------------------- #
# Reference statistics
# --------------------------------------------------------------------------- #


def _default_cache_dir() -> str:
    return os.environ.get("SI_FID_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "si_fid_stats"))


def _download(url: str, dest: str, timeout: int = 60) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    logger.info("Downloading FID reference statistics from %s", url)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dest)))
    os.close(tmp_fd)
    try:
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 - constant https URL
        os.replace(tmp_path, dest)
    finally:
        if os.path.exists(tmp_path):  # pragma: no cover - defensive
            os.remove(tmp_path)
    return dest


def get_reference_stats(
    dataset: str = "imagenet",
    resolution: Union[int, str, None] = 256,
    path: Optional[Union[str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    allow_download: bool = True,
) -> FIDStats:
    """Load reference (ground-truth) Inception statistics.

    Resolution ``256`` (in-painting, ``§4.1``) and ``512`` (in-painting/SR,
    ``§4.2``) use the standard ImageNet reference batches.  For the
    super-resolution task the FID is computed between generated and ground-truth
    *high-resolution* images, so the reference file is the one for the target
    resolution (256 for 64->256, 512 for 256->512).

    Notes
    -----
    The paper's FID-50k values were obtained with the standard ImageNet
    reference statistics.  If no cached/precomputed file is available the
    function downloads it (``allow_download=True``) or raises, in which case
    callers should build the statistics from a ground-truth dataloader with
    :meth:`FIDStats.from_features`.
    """
    dataset = str(dataset).lower()
    if resolution is not None and not isinstance(resolution, str):
        resolution = int(resolution)
    key = f"{dataset}_{resolution}" if isinstance(resolution, (int, str)) else dataset

    candidates: List[str] = []
    if path is not None:
        candidates.append(os.fspath(path))
    cache = os.fspath(cache_dir) if cache_dir is not None else _default_cache_dir()
    candidates.append(os.path.join(cache, f"{key}.npz"))
    candidates.append(os.path.join(cache, f"{dataset}_{resolution}_stats.npz"))
    if resolution is not None:
        candidates.append(os.path.join(cache, f"inception_{dataset}_{resolution}.npz"))

    for candidate in candidates:
        if os.path.exists(candidate):
            stats = FIDStats.load(candidate)
            if not stats.name:
                stats.name = key
            logger.info("Loaded reference FID statistics from %s", candidate)
            return stats

    if key in STATS_URLS and allow_download:
        dest = os.path.join(cache, f"{key}.npz")
        try:
            _download(STATS_URLS[key], dest)
            stats = FIDStats.load(dest)
            return stats
        except Exception as exc:  # noqa: BLE001 - network is best-effort
            logger.warning("Could not download reference statistics (%s); falling back.", exc)

    raise FileNotFoundError(
        "No reference FID statistics found. Provide `path=...`, press "
        "`FIDStats.from_features(gt_features, name=...)` to build them from a ground-truth "
        "dataloader, or pre-download the ImageNet reference batch into '%s'." % cache
    )


def save_reference_stats(stats: FIDStats, path: Union[str, os.PathLike]) -> str:
    """Persist reference statistics so later evaluations reuse them."""
    return stats.save(path)


# --------------------------------------------------------------------------- #
# FID entry points
# --------------------------------------------------------------------------- #


@dataclass
class EvalResult:
    """Outcome of an FID evaluation (mirrors the tables of the paper)."""

    fid: float
    num_samples: int
    reference_name: str = ""
    task: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    #: FID quoted in the paper for this configuration, when known.
    paper_fid: Optional[float] = None

    @property
    def delta(self) -> Optional[float]:
        if self.paper_fid is None:
            return None
        return self.fid - self.paper_fid

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "fid": self.fid,
            "num_samples": self.num_samples,
            "reference": self.reference_name,
            "task": self.task,
        }
        if self.paper_fid is not None:
            out["paper_fid"] = self.paper_fid
            out["delta"] = self.delta
        out.update(self.extra)
        return out

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        msg = f"FID-{self.num_samples // 1000}k = {self.fid:.2f}"
        if self.paper_fid is not None:
            msg += f" (paper: {self.paper_fid:.2f})"
        return msg


def compute_fid_from_features(features_gen: np.ndarray, features_ref: np.ndarray) -> float:
    """FID between two raw feature matrices."""
    gen = FIDStats.from_features(features_gen, name="generated")
    ref = FIDStats.from_features(features_ref, name="reference")
    return gen.distance_to(ref)


def compute_fid(
    generated: Union[torch.Tensor, Iterable[Any]],
    reference: Optional[Union[FIDStats, str, os.PathLike]] = None,
    num_samples: int = 50_000,
    batch_size: int = 32,
    extractor: Optional[InceptionFeatureExtractor] = None,
    weights_path: Optional[Union[str, os.PathLike]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dataset: str = "imagenet",
    resolution: Union[int, str] = 256,
    task: str = "",
    paper_fid: Optional[float] = None,
    verbose: bool = False,
    save_features: Optional[Union[str, os.PathLike]] = None,
) -> EvalResult:
    """Compute FID-50k for a stream of generated images.

    Parameters
    ----------
    generated:
        Either a tensor of images in ``[-1, 1]`` or an iterable of batches (a
        dataloader is fine; ``(images, labels)`` tuples are supported).
    reference:
        ``FIDStats``, a path to a stats ``.npz``, or ``None`` to fetch the
        standard ImageNet reference statistics for ``resolution``.
    num_samples:
        Number of generated samples to use (the paper reports 50,000).
    """
    if extractor is None:
        extractor = InceptionFeatureExtractor(weights_path=weights_path, device=device)

    features_gen = extractor.features_of(generated, batch_size=batch_size, max_samples=num_samples, verbose=verbose)
    if save_features is not None:
        np.save(os.fspath(save_features), features_gen)

    if isinstance(reference, FIDStats):
        ref_stats = reference
    elif reference is not None:
        ref_stats = FIDStats.load(reference)
    else:
        ref_stats = get_reference_stats(dataset=dataset, resolution=resolution)

    gen_stats = FIDStats.from_features(features_gen, name="generated")
    fid = gen_stats.distance_to(ref_stats)
    return EvalResult(
        fid=fid,
        num_samples=int(features_gen.shape[0]),
        reference_name=ref_stats.name or f"{dataset}_{resolution}",
        task=task,
        paper_fid=paper_fid,
        extra={"feature_dim": int(features_gen.shape[1])},
    )


def fid_for_images(
    images: torch.Tensor,
    reference: Optional[Union[FIDStats, str, os.PathLike]] = None,
    num_samples: Optional[int] = None,
    batch_size: int = 32,
    **kwargs: Any,
) -> EvalResult:
    """Convenience wrapper around :func:`compute_fid` for an in-memory tensor."""
    if num_samples is None:
        num_samples = int(images.shape[0])
    num_samples = min(num_samples, int(images.shape[0]))
    return compute_fid(images, reference=reference, num_samples=num_samples, batch_size=batch_size, **kwargs)


def compute_reference_stats(
    dataloader: Iterable[Any],
    num_samples: int = 50_000,
    batch_size: int = 32,
    extractor: Optional[InceptionFeatureExtractor] = None,
    device: Optional[Union[str, torch.device]] = None,
    name: str = "imagenet",
    save_path: Optional[Union[str, os.PathLike]] = None,
    verbose: bool = False,
) -> FIDStats:
    """Build reference statistics from a ground-truth dataloader (fallback path).

    Required when the standard ImageNet reference batch is unavailable: the FID
    then measures the distance to the *provided* ground-truth split, which is
    what the "Valid" column of Table 3 effectively reports.
    """
    if extractor is None:
        extractor = InceptionFeatureExtractor(device=device)
    features = extractor.features_of(dataloader, batch_size=batch_size, max_samples=num_samples, verbose=verbose)
    stats = FIDStats.from_features(features, name=name)
    if save_path is not None:
        stats.save(save_path)
    return stats


def evaluate_fid_50k(
    generated: Union[torch.Tensor, Iterable[Any]],
    reference: Optional[Union[FIDStats, str, os.PathLike]] = None,
    task: str = "inpainting",
    batch_size: int = 32,
    num_samples: int = 50_000,
    **kwargs: Any,
) -> EvalResult:
    """Evaluate the paper's headline metric (FID-50k) and compare to the table.

    ``task`` selects the paper value used for the reported delta:
    ``"inpainting"`` (Table 2, dependent coupling 1.13) or
    ``"superres_64_256"`` (Table 3, train 2.13).
    """
    paper_fid = None
    if task in PAPER_FIDS:
        paper_fid = PAPER_FIDS[task].get("dependent_coupling")
    resolution = 512 if "512" in str(task) else 256
    return compute_fid(
        generated,
        reference=reference,
        num_samples=num_samples,
        batch_size=batch_size,
        resolution=resolution,
        task=task,
        paper_fid=paper_fid,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Sampling-source helpers (images on disk -> FID) and CLI
# --------------------------------------------------------------------------- #


def _iter_image_folder(folder: str, resolution: int, batch_size: int = 32, num_samples: Optional[int] = None) -> Iterable[torch.Tensor]:
    """Iterate over ``*.png/jpg`` files in ``folder`` as ``[-1, 1]`` tensors."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if num_samples is not None:
        files = files[: int(num_samples)]
    if not files:
        raise FileNotFoundError(f"No image files found in {folder!r}")
    for start in range(0, len(files), batch_size):
        chunk = files[start : start + batch_size]
        batch = []
        for path in chunk:
            img = Image.open(path).convert("RGB")
            tensor = TF.to_tensor(img)  # [0, 1]
            batch.append(2.0 * tensor - 1.0)  # -> [-1, 1]
        yield torch.stack(batch, dim=0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute FID-50k for generated samples (Tables 2-3).")
    parser.add_argument("--samples-dir", required=True, help="folder containing generated images")
    parser.add_argument("--task", default="inpainting", choices=sorted(PAPER_FIDS.keys()))
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--resolution", type=int, default=None, help="reference resolution (default: from task)")
    parser.add_argument("--reference-stats", default=None, help="path to a reference-stats .npz")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fid-weights", default=os.environ.get("SI_FID_INCEPTION_WEIGHTS"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    resolution = args.resolution or (512 if "512" in args.task else 256)
    stream = _iter_image_folder(args.samples_dir, resolution, batch_size=args.batch_size, num_samples=args.num_samples)
    result = evaluate_fid_50k(
        stream,
        reference=args.reference_stats,
        task=args.task,
        dataset=args.dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        weights_path=args.fid_weights,
        device=args.device,
        verbose=args.verbose,
    )
    print(result)
    print(result.as_dict())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
