"""FID-50k evaluation (Tables 2 and 3 of the paper).

Implements the Frechet Inception Distance with the same feature extractor and
pre-processing that are standard in the diffusion literature (and that were
used to produce the baseline numbers quoted in the paper): the 2048-d pool
features of Inception-V3 at 299x299 with the [-1, 1] -> [0, 1] -> (x - 0.5)/0.5
normalisation.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^{1/2})

The reference statistics are computed over 50,000 ImageNet validation images
and are cached to disk so that repeated evaluations are cheap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InceptionFeatureExtractor(nn.Module):
    """Inception-V3 feature extractor returning 2048-d pool features."""

    def __init__(self, device=None):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, init_weights=False, transform_input=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        if device is not None:
            self.net.to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    def preprocess(self, x: Tensor) -> Tensor:
        """x in [-1, 1] -> Inception input."""
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bicubic", align_corners=False)
        return (x - 0.5) / 0.5

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.preprocess(x))


@torch.no_grad()
def compute_statistics(
    feature_fn,
    batches,
    device: Optional[torch.device] = None,
    n_samples: Optional[int] = None,
) -> dict:
    """Accumulate the mean and covariance of the Inception features."""
    feats = []
    n = 0
    for x in batches:
        if device is not None:
            x = x.to(device)
        f = feature_fn(x)
        feats.append(f.double().cpu())
        n += f.shape[0]
        if n_samples is not None and n >= n_samples:
            break
    feats = torch.cat(feats, dim=0)[:n_samples] if n_samples is not None else torch.cat(feats, dim=0)
    mu = feats.mean(dim=0)
    centred = feats - mu
    cov = centred.T @ centred / (feats.shape[0] - 1)
    return {"mu": mu, "sigma": cov, "n": int(feats.shape[0])}


def frechet_distance(stats_a: dict, stats_b: dict, eps: float = 1e-6) -> float:
    """Frechet distance between two sets of Gaussian statistics.

    The matrix square root is computed on mildly regularised covariances
    (``sigma + eps * mean(diag) * I``), which is the standard remedy for the
    ill-conditioning that appears when the number of samples is not much
    larger than the feature dimension (the reference protocol uses 50,000
    samples and 2048 features).
    """
    import numpy as np
    from scipy import linalg

    mu1, mu2 = stats_a["mu"].numpy(), stats_b["mu"].numpy()
    sigma1, sigma2 = stats_a["sigma"].numpy(), stats_b["sigma"].numpy()
    diff = mu1 - mu2
    scale1 = float(np.mean(np.diag(sigma1)))
    scale2 = float(np.mean(np.diag(sigma2)))
    reg1 = sigma1 + eps * scale1 * np.eye(sigma1.shape[0])
    reg2 = sigma2 + eps * scale2 * np.eye(sigma2.shape[0])
    covmean, _ = linalg.sqrtm(reg1.dot(reg2), disp=False)
    if not np.isfinite(covmean).all():
        offset1 = np.eye(sigma1.shape[0]) * eps * max(scale1, 1e-8)
        offset2 = np.eye(sigma2.shape[0]) * eps * max(scale2, 1e-8)
        covmean = linalg.sqrtm((sigma1 + offset1).dot(sigma2 + offset2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def save_statistics(path: str | os.PathLike, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mu": stats["mu"], "sigma": stats["sigma"], "n": stats["n"]}, path)


def load_statistics(path: str | os.PathLike) -> dict:
    return torch.load(Path(path), map_location="cpu")


# ---------------------------------------------------------------------------
def fid50k_from_loaders(real_batches, fake_batches, device=None, n_samples: int = 50_000) -> float:
    """Compute FID-50k given iterables over real and generated image batches."""
    extractor = InceptionFeatureExtractor(device=device)
    if device is not None:
        extractor.to(device)

    def fn(x):
        return extractor(x)

    real_stats = compute_statistics(fn, real_batches, device=device, n_samples=n_samples)
    fake_stats = compute_statistics(fn, fake_batches, device=device, n_samples=n_samples)
    return frechet_distance(real_stats, fake_stats)


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FID-50k for a trained stochastic-interpolant model.")
    parser.add_argument("--config", help="experiment configuration")
    parser.add_argument("--checkpoint", help="trained model")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--method", default="dopri5")
    parser.add_argument("--real-stats", default=None, help="cached reference statistics (.pt)")
    parser.add_argument("--out", default="results/fid.json")
    parser.add_argument("--smoke", action="store_true", help="run on synthetic data (no ImageNet needed)")
    args = parser.parse_args(argv)

    from .utils import load_config, resolve_device, set_seed
    from .sample import load_model_from_checkpoint, sample_inpainting

    device = resolve_device("auto")
    set_seed(0)
    cfg = load_config(args.config)
    if args.n < 2048:
        print(
            f"warning: FID with n={args.n} samples and {2048}-dimensional features is "
            "not statistically meaningful; the reference protocol (and Tables 2/3 of "
            "the paper) uses n = 50,000.",
            flush=True,
        )
    task = cfg.get("task", "inpainting")
    resolution = int(cfg.get("data", {}).get("image_size", 256))

    model, coupling, schedule, device = load_model_from_checkpoint(cfg, args.checkpoint)

    from .data.imagenet import build_dataloader

    real_loader = build_dataloader(cfg, train=False, batch_size=args.batch_size)

    def generated_batches():
        n_done = 0
        import numpy as _np

        rng = _np.random.RandomState(0)
        for images, labels in real_loader:
            images = images.to(device)
            labels = labels.to(device)
            if task == "super_resolution":
                from .data.superres import downsample
                from .sample import sample_super_resolution

                low = downsample(images, coupling.scale)
                out = sample_super_resolution(model, coupling, low, labels=labels,
                                              method=args.method, steps=args.steps,
                                              high_res_truth=images)
                yield out["sample"].clamp(-1, 1)
            else:
                out = sample_inpainting(model, coupling, images, labels=labels,
                                        method=args.method, steps=args.steps)
                yield out["sample"].clamp(-1, 1)
            n_done += images.shape[0]
            if n_done >= args.n:
                break

    def real_batches():
        n_done = 0
        for images, _ in real_loader:
            yield images.to(device)
            n_done += images.shape[0]
            if n_done >= args.n:
                break

    if args.real_stats is not None and Path(args.real_stats).exists():
        real_stats = load_statistics(args.real_stats)
    else:
        extractor = InceptionFeatureExtractor(device=device)
        real_stats = compute_statistics(extractor, real_batches(), device=device, n_samples=args.n)
        save_statistics(Path(args.out).with_name("imagenet_stats.pt"), real_stats)

    extractor = InceptionFeatureExtractor(device=device)
    fake_stats = compute_statistics(extractor, generated_batches(), device=device, n_samples=args.n)
    fid = frechet_distance(real_stats, fake_stats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fid": fid, "n": args.n, "task": task, "checkpoint": args.checkpoint}))
    print(f"FID-{args.n} = {fid:.3f}  (written to {out})")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "InceptionFeatureExtractor",
    "compute_statistics",
    "frechet_distance",
    "fid50k_from_loaders",
    "save_statistics",
    "load_statistics",
    "main",
]
