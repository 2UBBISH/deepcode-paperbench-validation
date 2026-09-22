"""Evaluation metrics of Section 5.2.

    "To evaluate the diversity of generation, we use Intra-LPIPS and FID
    following CDC (Ojha et al., 2021).  For Intra-LPIPS, we generate 1,000
    images, each of which will be assigned to the training sample with the
    smallest LPIPS distance.  The Intra-LPIPS measurement is obtained by
    averaging the pairwise LPIPS distances within the same cluster and then
    averaging these results across all clusters."

Both metrics are implemented here:

* :class:`LPIPSMetric` -- the perceptual metric of Zhang et al. (2018).  The
  official ``lpips`` package is used when installed; otherwise a faithful
  AlexNet-feature re-implementation is used (with a warning, because the
  learned 1x1 calibration weights live in the official package).
* :class:`FIDMetric` -- Frechet Inception Distance using the standard
  InceptionV3 features of ``pytorch-fid`` (falls back to torchvision's
  InceptionV3 when the package is missing; FID values then differ slightly).
* :func:`intra_lpips` -- the clustered diversity metric described above.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import progbar


# ---------------------------------------------------------------------- #
# LPIPS
# ---------------------------------------------------------------------- #
class _TorchvisionLPIPS(nn.Module):
    """Minimal LPIPS (Zhang et al., 2018) on top of torchvision AlexNet.

    Only used when the official ``lpips`` package is unavailable.  The feature
    extractor (AlexNet conv1..conv5 with the standard channel slicing) and the
    spatial averaging match the reference implementation; without the official
    calibration weights the numbers are a close approximation.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            from torchvision.models import alexnet, AlexNet_Weights
        except Exception as error:  # pragma: no cover
            raise RuntimeError("torchvision is required for the LPIPS fallback") from error
        net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features
        self.slices = [net[:3], net[3:6], net[6:8], net[8:10], net[10:13]]
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.to(device)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std
        features_a, features_b = a, b
        distances = []
        for slice_ in self.slices:
            features_a = slice_(features_a)
            features_b = slice_(features_b)
            diff = (features_a - features_b) ** 2
            distances.append(diff.mean(dim=list(range(1, diff.ndim))))
        return torch.stack(distances, dim=-1).sum(-1)


class LPIPSMetric:
    """Callable ``lpips(a, b) -> [N]`` distance with the official package first."""

    def __init__(self, device: str = "cpu", net: str = "alex", warn: bool = True):
        self.device = torch.device(device)
        self.using_official = False
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
            self.using_official = True
        except Exception:
            if warn:
                warnings.warn(
                    "the `lpips` package is not installed: falling back to a "
                    "re-implementation based on torchvision AlexNet features. "
                    "Install `lpips` for values comparable to the paper.",
                    RuntimeWarning,
                )
            self.model = _TorchvisionLPIPS(self.device).eval()

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.to(self.device, dtype=torch.float32).clamp(-1, 1)
        b = b.to(self.device, dtype=torch.float32).clamp(-1, 1)
        return self.model(a, b).reshape(-1)

    @torch.no_grad()
    def pairwise(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Full ``[N, M]`` distance matrix (chunked to bound memory)."""
        outputs = []
        for index in range(a.shape[0]):
            outputs.append(self(a[index : index + 1].expand(b.shape[0], *a.shape[1:]), b))
        return torch.stack(outputs, dim=0)


def pairwise_lpips_matrix(
    images: torch.Tensor, metric: LPIPSMetric, chunk: int = 32
) -> torch.Tensor:
    """``[N, N]`` matrix of LPIPS distances inside one sample set."""
    count = images.shape[0]
    matrix = torch.zeros(count, count)
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        block = images[start:end]
        distances = metric.pairwise(block, images)
        matrix[start:end] = distances.cpu()
    return matrix


# ---------------------------------------------------------------------- #
# Intra-LPIPS (Section 5.2 / CDC)
# ---------------------------------------------------------------------- #
def intra_lpips(
    generated: torch.Tensor,
    train_images: torch.Tensor,
    metric: Optional[LPIPSMetric] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> float:
    """The diversity metric of the paper.

    Every generated image is assigned to the closest *training* image (minimum
    LPIPS), then the pairwise LPIPS distances inside each cluster are averaged
    and the cluster averages are averaged.  A model that copies the training
    images scores 0.
    """
    metric = metric or LPIPSMetric(device=device)
    generated = generated.to(device)
    train_images = train_images.to(device)

    # 1. assignment of every generated image to its nearest training image
    distances = metric.pairwise(generated, train_images)  # [N, K]
    assignment = distances.argmin(dim=1)

    # 2. pairwise LPIPS inside each cluster, averaged per cluster
    cluster_scores: List[float] = []
    iterator = range(train_images.shape[0])
    if verbose:
        iterator = progbar(iterator, desc="intra-lpips")
    for cluster in iterator:
        members = generated[assignment == cluster]
        if members.shape[0] < 2:
            # a cluster with a single generated image has no within-cluster
            # pair; the paper defines a perfect copy of the training set as
            # Intra-LPIPS = 0, so such clusters contribute 0
            cluster_scores.append(0.0)
            continue
        matrix = pairwise_lpips_matrix(members, metric)
        count = members.shape[0]
        total = matrix.sum().item() - matrix.diagonal().sum().item()
        cluster_scores.append(total / (count * (count - 1)))
    if not cluster_scores:
        return float("nan")
    return float(np.mean(cluster_scores))


# ---------------------------------------------------------------------- #
# FID
# ---------------------------------------------------------------------- #
class FIDMetric:
    """Frechet Inception Distance with the standard ``pytorch-fid`` features."""

    def __init__(self, device: str = "cpu", dims: int = 2048, warn: bool = True):
        self.device = torch.device(device)
        self.dims = dims
        self.using_official = False
        try:
            from pytorch_fid.inception import InceptionV3  # type: ignore

            block = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            self.model = InceptionV3([block]).to(self.device).eval()
            self.using_official = True
        except Exception:
            if warn:
                warnings.warn(
                    "`pytorch-fid` is not installed: falling back to torchvision "
                    "InceptionV3 features (FID values are not numerically identical).",
                    RuntimeWarning,
                )
            self.model = _TorchvisionInception(self.device).eval()

    @torch.no_grad()
    def features(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        images = images.to(self.device, dtype=torch.float32)
        outputs = []
        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].clamp(-1, 1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            if not self.using_official:
                batch = (batch + 1) / 2  # torchvision expects [0, 1]
                batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            features = self.model(batch)[0]
            if features.ndim == 4:
                features = features.mean(dim=[2, 3])
            outputs.append(features.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def compute(self, generated: torch.Tensor, real: torch.Tensor, batch_size: int = 32) -> float:
        return frechet_distance(
            self.features(real, batch_size=batch_size),
            self.features(generated, batch_size=batch_size),
        )


class _TorchvisionInception(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = nn.Identity()
        self.model = model.eval()

    def forward(self, x):
        return (self.model(x),)


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance between two Gaussian feature distributions."""
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mean_a - mean_b
    from scipy import linalg

    cov_mean, _ = linalg.sqrtm(np.dot(cov_a, cov_b), disp=False)
    if not np.isfinite(cov_mean).all():
        offset = np.eye(cov_a.shape[0]) * eps
        cov_mean = linalg.sqrtm(np.dot(cov_a + offset, cov_b + offset), disp=False)[0]
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    return float(np.dot(diff, diff) + np.trace(cov_a + cov_b - 2 * cov_mean))


def compute_fid(
    generated: torch.Tensor,
    real: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 32,
    metric: Optional[FIDMetric] = None,
) -> float:
    metric = metric or FIDMetric(device=device)
    return metric.compute(generated, real, batch_size=batch_size)


@dataclass
class MetricResults:
    intra_lpips: Optional[float] = None
    fid: Optional[float] = None

    def as_dict(self) -> dict:
        return {"intra_lpips": self.intra_lpips, "fid": self.fid}
