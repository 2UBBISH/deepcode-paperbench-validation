"""Visualisation utilities: masks, reprogrammed images and t-SNE features.

These correspond to Figure 5 ("Visualization of SMM, shared patterns and output
reprogrammed images") and Figure 6 ("Feature Space Visualization Results") of
the paper.  The addendum marks both figures as *not required* for the
reproduction, but the utilities are kept small and standalone so that the
analysis can be re-run; the t-SNE helper follows the addendum detail of using
5,000 randomly selected training samples per dataset.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

__all__ = [
    "denormalize",
    "histogram_equalize",
    "save_reprogrammed_grid",
    "save_mask_figure",
    "tsne_feature_space",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalisation for display (values in [0, 1])."""
    return (x.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def histogram_equalize(x: torch.Tensor) -> torch.Tensor:
    """Per-image histogram equalisation (the paper's Figure 5 uses equalisation)."""
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        flat = x[i].flatten()
        values, inverse = flat.sort()
        ranks = torch.empty_like(inverse)
        ranks[inverse] = torch.arange(flat.numel(), dtype=flat.dtype)
        out[i] = (ranks / max(flat.numel() - 1, 1)).reshape(x[i].shape)
    return out


@torch.no_grad()
def save_reprogrammed_grid(
    model,
    loader,
    path: str,
    num_samples: int = 8,
    equalize: bool = True,
    device: Optional[torch.device] = None,
) -> str:
    """Original vs. reprogrammed images plus the learned masks (Figure 5/13-23)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = device or next(model.parameters()).device
    model.eval()
    x, y = next(iter(loader))
    x = x[:num_samples].to(device)
    mask = model.input_transform.make_mask(model.input_transform.resized_input(x))
    out = model.input_transform(x)

    originals = denormalize(x)
    reprogrammed = denormalize(out)
    if equalize:
        originals = histogram_equalize(originals)
        reprogrammed = histogram_equalize(reprogrammed)

    n = min(num_samples, x.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    if n == 1:  # pragma: no cover - cosmetic
        axes = axes.reshape(3, 1)
    for i in range(n):
        axes[0, i].imshow(originals[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"orig y={int(y[i])}", fontsize=8)
        axes[1, i].imshow(reprogrammed[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title("reprogrammed", fontsize=8)
        axes[2, i].imshow(mask[i].mean(0).cpu().numpy(), cmap="viridis")
        axes[2, i].set_title("mask", fontsize=8)
        for ax in axes[:, i]:
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_mask_figure(model, path: str, sample: Optional[torch.Tensor] = None) -> str:
    """Plot the shared pattern ``delta`` and the channel-wise mask statistics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    if model.delta is None:
        raise ValueError("this model has no shared pattern (delta)")
    delta = model.delta.detach().cpu()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for c in range(3):
        axes[c].imshow(delta[0, c].numpy(), cmap="coolwarm")
        axes[c].set_title(f"delta channel {c}")
        axes[c].axis("off")
    if sample is not None:
        with torch.no_grad():
            mask = model.input_transform.make_mask(model.input_transform.resized_input(sample))
        axes[3].imshow(mask[0].mean(0).cpu().numpy(), cmap="viridis")
        axes[3].set_title("sample mask")
    axes[3].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@torch.no_grad()
def tsne_feature_space(
    model,
    loader,
    path: str,
    num_samples: int = 5000,
    device: Optional[torch.device] = None,
    perplexity: float = 30.0,
    seed: int = 0,
) -> str:
    """t-SNE of the output-layer features before the label mapping (Figure 6).

    The addendum specifies 5,000 randomly selected training samples per dataset.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    device = device or next(model.parameters()).device
    model.eval()
    feats, labels = [], []
    collected = 0
    for x, y in loader:
        x = x.to(device)
        feats.append(model.bundle.features(model.input_transform(x)).cpu())
        labels.append(y)
        collected += x.shape[0]
        if collected >= num_samples:
            break
    features = torch.cat(feats)[:num_samples]
    labels = torch.cat(labels)[:num_samples]

    embedding = TSNE(
        n_components=2, perplexity=min(perplexity, max(features.shape[0] / 4, 5)),
        random_state=seed, init="pca",
    ).fit_transform(features.numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels.numpy(), cmap="tab20", s=4)
    fig.colorbar(scatter, ax=ax, label="target class")
    ax.set_title("output-layer features (t-SNE)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
