"""Intra-LPIPS diversity metric for DPMs-ANT (Section 5.2, Ojha et al. 2021 / CDC).

Definition reproduced from the paper
------------------------------------
"For Intra-LPIPS, we generate 1,000 images, each of which will be assigned to the
training sample with the smallest LPIPS distance. The Intra-LPIPS measurement is
obtained by averaging the pairwise LPIPS distances within the same cluster and then
averaging these results across all clusters. A model that flawlessly duplicates
training samples will have an Intra-LPIPS score of zero, which indicates a lack of
diversity. However, higher Intra-LPIPS scores imply greater generation diversity."

Implementation notes
--------------------
*   Clustering is *nearest-neighbour assignment*: each generated sample is assigned
    to the training image minimising the LPIPS distance (no k-means).
*   Within-cluster pairwise LPIPS = mean over all ordered pairs ``i != j`` in the
    cluster of ``LPIPS(x_i, x_j)``.  Singleton clusters contribute ``0`` only if
    ``include_singletons=True`` (default ``True`` so the reported value degrades to
    0 exactly when the model duplicates the training set).  Set
    ``include_singletons=False`` to average over non-singleton clusters only.
*   "Averaging these results across all clusters" can be an unweighted mean across
    clusters (paper wording) or a sample-weighted mean.  The paper says *averaging
    across clusters*, so ``cluster_average="mean"`` is the default; the
    ``"weighted"`` option performs the pooled average over all generated pairs.

The metric uses the `lpips` package (AlexNet backbone by default, matching CDC);
a small pure-torch fallback (LPIPS-style VGG/Alex feature distance) is provided so
the pipeline still runs when `lpips` is not installed (it emits a warning).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.datasets import denormalize_images, load_image

LOGGER = logging.getLogger(__name__)

__all__ = [
    "LPIPSMetric",
    "IntraLPIPS",
    "compute_intra_lpips",
    "pairwise_distances",
    "assign_to_nearest",
    "cluster_pairwise_mean",
    "build_lpips_metric",
    "load_reference_images",
    "IntraLPIPSConfig",
]


# --------------------------------------------------------------------------------------
# LPIPS backend
# --------------------------------------------------------------------------------------
class _TorchLPIPS(nn.Module):
    """Minimal LPIPS-style perceptual distance (fallback when `lpips` is missing).

    Uses the first few blocks of a torchvision AlexNet/VGG with the standard LPIPS
    channel-normalisation, freezing all weights.  It is *not* bit-exact with the
    official LPIPS release but preserves the ranking properties the metric needs.
    """

    def __init__(self, net: str = "alex", device: Union[str, torch.device] = "cpu"):
        super().__init__()
        import torchvision  # local import: only needed for the fallback

        if net.startswith("vgg"):
            try:
                feats = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features
            except Exception:  # pragma: no cover - offline
                feats = torchvision.models.vgg16(weights=None).features
            slices = [feats[:4], feats[4:9], feats[9:16], feats[16:23], feats[23:30]]
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            try:
                feats = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT).features
            except Exception:  # pragma: no cover - offline
                feats = torchvision.models.alexnet(weights=None).features
            slices = [feats[:3], feats[3:6], feats[6:10], feats[10:13], feats[13:]]
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.slices = nn.ModuleList(slices)
        self.channels = [3] + [s[0].out_channels if hasattr(s[0], "out_channels") else 64 for s in slices]
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = (x - self.mean) / self.std
        b = (y - self.mean) / self.std
        dist = x.new_zeros(x.shape[0])
        for i, sl in enumerate(self.slices):
            a = sl(a)
            b = sl(b)
            dist = dist + (a - b).pow(2).mean(dim=[1, 2, 3]) / (2 ** i)
        return dist


class LPIPSMetric:
    """Thin wrapper around the official ``lpips`` package with a torch fallback.

    Parameters
    ----------
    net: ``"alex"`` (default, matches CDC/DDPM-PA) or ``"vgg"``.
    device: torch device for the backbone.
    batch_size: how many pairs / images to push through the backbone at once.
    """

    def __init__(self, net: str = "alex", device: Union[str, torch.device] = "cpu", batch_size: int = 16):
        self.net = net
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self._backend = None
        self._backend_name = "uninitialised"
        try:  # pragma: no cover - depends on environment
            import lpips

            self._backend = lpips.LPIPS(net=net, verbose=False).to(self.device).eval()
            for p in self._backend.parameters():
                p.requires_grad_(False)
            self._backend_name = f"lpips:{net}"
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("`lpips` package unavailable (%s); using torch fallback metric.", exc)
            self._backend = _TorchLPIPS(net=net, device=self.device)
            self._backend_name = f"torch-fallback:{net}"

    # -- distances ---------------------------------------------------------------------
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Per-sample perceptual distance between aligned batches ``x`` and ``y``.

        Expects inputs in ``[-1, 1]`` (DPM convention); they are clamped for safety.
        Returns a ``(N,)`` tensor on the metric device.
        """
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch in LPIPS distance: {tuple(x.shape)} vs {tuple(y.shape)}")
        n = x.shape[0]
        outs: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                xb = x[start : start + self.batch_size].to(self.device).float()
                yb = y[start : start + self.batch_size].to(self.device).float()
                xb = xb.clamp(-1.0, 1.0)
                yb = yb.clamp(-1.0, 1.0)
                d = self._backend(xb, yb)
                if isinstance(d, (tuple, list)):
                    d = d[0]
                outs.append(d.reshape(-1))
        return torch.cat(outs, dim=0)

    def pairwise_block(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Full pairwise distance matrix between sets ``a`` (N) and ``b`` (M).

        Returns an ``(N, M)`` tensor; memory is bounded by batching over ``b``.
        """
        rows: List[torch.Tensor] = []
        n = a.shape[0]
        for i in range(n):
            ai = a[i : i + 1].expand(b.shape[0], *a.shape[1:])
            rows.append(self.distance(ai, b).unsqueeze(0))
        return torch.cat(rows, dim=0)

    def pairwise_matrix(self, a: torch.Tensor) -> torch.Tensor:
        """Symmetric ``(N, N)`` pairwise distance matrix with zero diagonal."""
        n = a.shape[0]
        rows: List[List[torch.Tensor]] = []
        for i in range(n):
            row: List[torch.Tensor] = []
            for j in range(i, n):
                d = self.distance(a[i : i + 1], a[j : j + 1])
                row.append(d.reshape(1))
            rows.append(row)
        mat = a.new_zeros(n, n)
        for i in range(n):
            for k, j in enumerate(range(i, n)):
                mat[i, j] = rows[i][k]
                mat[j, i] = rows[i][k]
        return mat

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.distance(x, y)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LPIPSMetric(net={self.net!r}, backend={self._backend_name!r}, device={self.device})"


# --------------------------------------------------------------------------------------
# Core metric
# --------------------------------------------------------------------------------------
def pairwise_distances(
    metric: Union[LPIPSMetric, nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Pairwise LPIPS matrix between two image sets (``N x M``)."""
    if isinstance(metric, LPIPSMetric):
        return metric.pairwise_block(x, y)
    # generic module(a, b) -> per-pair distance
    rows: List[torch.Tensor] = []
    for i in range(x.shape[0]):
        xi = x[i : i + 1].expand(y.shape[0], *x.shape[1:])
        with torch.no_grad():
            d = metric(xi, y)
        if isinstance(d, (tuple, list)):
            d = d[0]
        rows.append(d.reshape(-1).unsqueeze(0))
    return torch.cat(rows, dim=0).detach()


def assign_to_nearest(gen_to_train: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour cluster assignment.

    ``gen_to_train`` has shape ``(num_generated, num_train)``; returns an
    ``(num_generated,)`` long tensor of cluster indices.
    """
    return gen_to_train.argmin(dim=1)


def cluster_pairwise_mean(
    dmat: torch.Tensor,
    labels: torch.Tensor,
    num_clusters: int,
    include_singletons: bool = True,
    cluster_average: str = "mean",
) -> Tuple[float, List[float], List[int]]:
    """Mean pairwise distance within each cluster, averaged over clusters.

    Parameters
    ----------
    dmat: ``(N, N)`` symmetric distance matrix of generated samples.
    labels: ``(N,)`` cluster index per generated sample.
    num_clusters: number of training samples (clusters).
    include_singletons: if ``False``, singleton clusters are skipped instead of
        contributing ``0``.
    cluster_average: ``"mean"`` (unweighted average across clusters, paper
        wording) or ``"weighted"`` (pooled average over all unordered pairs).

    Returns
    -------
    ``(intra_lpips, per_cluster, cluster_sizes)``
    """
    per_cluster: List[float] = []
    sizes: List[int] = []
    pair_sum = 0.0
    pair_count = 0
    for c in range(num_clusters):
        idx = (labels == c).nonzero(as_tuple=False).reshape(-1)
        n = int(idx.numel())
        sizes.append(n)
        if n < 2:
            per_cluster.append(0.0 if include_singletons else float("nan"))
            continue
        sub = dmat[idx][:, idx]
        # sum of upper-triangle (unordered) pairs
        tri = torch.triu_indices(n, n, offset=1)
        vals = sub[tri[0], tri[1]]
        m = float(vals.mean().item())
        per_cluster.append(m)
        pair_sum += float(vals.sum().item())
        pair_count += int(vals.numel())

    valid = [v for v in per_cluster if v == v]  # drop NaN
    if not valid:
        return 0.0, per_cluster, sizes
    if cluster_average == "weighted":
        score = pair_sum / pair_count if pair_count > 0 else 0.0
    else:
        score = sum(valid) / len(valid)
    return float(score), per_cluster, sizes


@dataclass
class IntraLPIPSConfig:
    """Configuration for the Intra-LPIPS evaluation."""

    num_samples: int = 1000
    net: str = "alex"
    batch_size: int = 16
    device: str = "cpu"
    include_singletons: bool = True
    cluster_average: str = "mean"
    seed: Optional[int] = None
    #: number of training (cluster-centroid) images used; ``None`` = all available
    num_train_images: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: Optional[Dict] = None, **overrides) -> "IntraLPIPSConfig":
        cfg = dict(cfg or {})
        out = {}
        for block_name in ("evaluation", "intra_lpips"):
            block = cfg.get(block_name)
            if isinstance(block, dict):
                out.update(block)
        # flat keys
        for key in (
            "num_samples",
            "lpips_net",
            "net",
            "batch_size",
            "device",
            "include_singletons",
            "cluster_average",
            "seed",
            "num_train_images",
        ):
            if key in cfg and not isinstance(cfg[key], dict):
                out[key] = cfg[key]
        if "lpips_net" in out and "net" not in out:
            out["net"] = out.pop("lpips_net")
        if "batch_size" in out and "batch_size" in overrides:
            pass
        out.update({k: v for k, v in overrides.items() if v is not None})
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in out.items() if k in allowed})

    def to_dict(self) -> Dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


class IntraLPIPS:
    """Intra-LPIPS evaluator (Section 5.2).

    Example
    -------
    >>> ev = IntraLPIPS(IntraLPIPSConfig(num_samples=1000, device="cuda"))
    >>> result = ev.compute(generated, train_images)
    >>> result["intra_lpips"]
    0.613
    """

    def __init__(
        self,
        config: Optional[IntraLPIPSConfig] = None,
        metric: Optional[LPIPSMetric] = None,
        device: Optional[Union[str, torch.device]] = None,
        **overrides,
    ):
        self.config = config or IntraLPIPSConfig.from_dict(None, **overrides)
        if overrides:
            self.config = self.config.from_dict(self.config.to_dict(), **overrides)
        if device is not None:
            self.config.device = str(device)
        if metric is not None:
            self.metric = metric
        else:
            self.metric = LPIPSMetric(
                net=self.config.net, device=self.config.device, batch_size=self.config.batch_size
            )

    # -- data helpers ------------------------------------------------------------------
    def _prepare(
        self,
        generated: Union[torch.Tensor, Sequence[str]],
        train_images: Optional[Union[torch.Tensor, Sequence[str]]] = None,
        train_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gen = _to_tensor(generated, size=None)
        if train_images is not None:
            train = _to_tensor(train_images, size=None)
        elif train_dir:
            train = load_reference_images(train_dir, size=gen.shape[-1], limit=self.config.num_train_images)
        else:
            raise ValueError("Either `train_images` or `train_dir` must be provided.")
        if self.config.num_train_images is not None:
            train = train[: self.config.num_train_images]
        if self.config.num_samples is not None:
            gen = gen[: self.config.num_samples]
        if gen.shape[2:] != train.shape[2:]:
            train = F.interpolate(train, size=gen.shape[2:], mode="bilinear", align_corners=False)
        return gen, train

    # -- main entry point ---------------------------------------------------------------
    def compute(
        self,
        generated: Union[torch.Tensor, Sequence[str]],
        train_images: Optional[Union[torch.Tensor, Sequence[str]]] = None,
        train_dir: Optional[str] = None,
        return_details: bool = False,
    ) -> Dict:
        """Compute Intra-LPIPS.

        Returns a dict with ``intra_lpips`` (float, higher is better) plus, when
        ``return_details=True``, per-cluster means/sizes and the assignment histogram.
        """
        gen, train = self._prepare(generated, train_images, train_dir)
        n_gen, n_train = gen.shape[0], train.shape[0]
        LOGGER.info(
            "Intra-LPIPS: %d generated images vs %d training samples (backend=%s)",
            n_gen,
            n_train,
            getattr(self.metric, "_backend_name", "unknown"),
        )

        gen_to_train = pairwise_distances(self.metric, gen, train)  # (N, M)
        labels = assign_to_nearest(gen_to_train)
        min_dists = gen_to_train.gather(1, labels.view(-1, 1)).reshape(-1)

        dmat = self.metric.pairwise_matrix(gen)  # (N, N)
        score, per_cluster, sizes = cluster_pairwise_mean(
            dmat,
            labels,
            num_clusters=n_train,
            include_singletons=self.config.include_singletons,
            cluster_average=self.config.cluster_average,
        )

        result: Dict = {
            "intra_lpips": score,
            "num_generated": n_gen,
            "num_train": n_train,
            "mean_nearest_distance": float(min_dists.mean().item()) if n_gen else 0.0,
            "num_nonempty_clusters": int(sum(1 for s in sizes if s > 0)),
            "backend": getattr(self.metric, "_backend_name", "unknown"),
        }
        if return_details:
            result["per_cluster"] = per_cluster
            result["cluster_sizes"] = sizes
            result["labels"] = labels.cpu().tolist()
            result["nearest_distances"] = min_dists.cpu().tolist()
        return result

    __call__ = compute


# --------------------------------------------------------------------------------------
# Functional API
# --------------------------------------------------------------------------------------
def compute_intra_lpips(
    generated: Union[torch.Tensor, Sequence[str]],
    train_images: Optional[Union[torch.Tensor, Sequence[str]]] = None,
    train_dir: Optional[str] = None,
    num_samples: int = 1000,
    net: str = "alex",
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 16,
    include_singletons: bool = True,
    cluster_average: str = "mean",
    return_details: bool = False,
    metric: Optional[LPIPSMetric] = None,
) -> Dict:
    """One-shot Intra-LPIPS computation (Eq. definition in Section 5.2)."""
    cfg = IntraLPIPSConfig(
        num_samples=num_samples,
        net=net,
        batch_size=batch_size,
        device=str(device),
        include_singletons=include_singletons,
        cluster_average=cluster_average,
    )
    return IntraLPIPS(cfg, metric=metric).compute(
        generated, train_images=train_images, train_dir=train_dir, return_details=return_details
    )


def build_lpips_metric(cfg: Optional[Dict] = None, device: Optional[Union[str, torch.device]] = None, **overrides) -> LPIPSMetric:
    conf = IntraLPIPSConfig.from_dict(cfg, **overrides)
    return LPIPSMetric(net=conf.net, device=device or conf.device, batch_size=conf.batch_size)


def load_reference_images(
    train_dir: str,
    size: Optional[int] = 256,
    limit: Optional[int] = None,
    recursive: bool = True,
) -> torch.Tensor:
    """Load the training (cluster-centroid) images from a directory as ``[-1, 1]`` tensors."""
    if not train_dir or not os.path.isdir(train_dir):
        raise FileNotFoundError(f"reference/training image directory not found: {train_dir!r}")
    from ..data.datasets import list_images

    paths = list_images(train_dir, recursive=recursive)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no images found under {train_dir!r}")
    imgs = [load_image(p, size=size, resize=True) for p in paths]
    return torch.stack(imgs, dim=0)


def _to_tensor(
    data: Union[torch.Tensor, Sequence[str], Sequence[torch.Tensor]],
    size: Optional[int] = 256,
) -> torch.Tensor:
    """Coerce a tensor / list of paths / list of tensors into an ``(N, C, H, W)`` tensor in ``[-1, 1]``."""
    if isinstance(data, torch.Tensor):
        x = data.detach().float().cpu()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], str):
        return load_reference_images_paths(list(data), size=size)
    if isinstance(data, (list, tuple)) and len(data) > 0 and torch.is_tensor(data[0]):
        return torch.stack([d.detach().float().cpu() for d in data], dim=0)
    raise TypeError(f"unsupported data type for Intra-LPIPS: {type(data)!r}")


def load_reference_images_paths(paths: Sequence[str], size: Optional[int] = 256) -> torch.Tensor:
    imgs = [load_image(p, size=size, resize=True) for p in paths]
    if not imgs:
        raise ValueError("empty image list")
    return torch.stack(imgs, dim=0)
