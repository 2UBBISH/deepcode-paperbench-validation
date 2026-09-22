"""PCA state-diversity metric of Sec. 6.4 / Figure 7.

"We compute the reconstruction error of a batch of states using k most
significant components of PCA and plot this error as a function of k.  In
general, a set that has variation along fewer dimensions of space can be
compressed with fewer principal vectors and will have lower reconstruction
error.  This metric therefore measures the extent to which the policy explores
different dimensions of state space.  We find that the rate of decrease in
reconstruction error with an increase in components is the slowest for our
method."
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Sequence

import numpy as np


def pca_reconstruction_errors(
    states: np.ndarray,
    components: Sequence[int],
    normalise: bool = True,
    max_samples: int = 200_000,
    seed: int = 0,
) -> Dict[int, float]:
    """Mean squared reconstruction error of ``states`` for each ``k``.

    The PCA basis is fitted once with ``max(components)`` principal directions
    using an SVD; keeping the first ``k`` directions of that basis gives exactly
    the ``k``-component PCA reconstruction.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.shape[0] > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(states.shape[0], size=max_samples, replace=False)
        states = states[idx]
    mean = states.mean(axis=0, keepdims=True)
    centered = states - mean
    if normalise:
        std = states.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        centered = centered / std
    max_k = int(max(components))
    # economy SVD of the centred data
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    max_k = min(max_k, vt.shape[0])
    errors: Dict[int, float] = {}
    for k in components:
        k = min(int(k), max_k)
        basis = vt[:k]
        proj = centered @ basis.T
        recon = proj @ basis
        errors[int(k)] = float(np.mean((centered - recon) ** 2))
    return errors


def analyse_pca_diversity(
    datasets: Dict[str, np.ndarray],
    components: Iterable[int] = tuple(range(1, 33)),
    normalise: bool = True,
    max_samples: int = 200_000,
) -> Dict[str, Dict[int, float]]:
    """Run :func:`pca_reconstruction_errors` for every named dataset."""
    return {
        name: pca_reconstruction_errors(
            states, list(components), normalise=normalise, max_samples=max_samples
        )
        for name, states in datasets.items()
    }


def plot_pca_diversity(
    results: Dict[str, Dict[int, float]],
    out_path: str,
    title: str = "State diversity (PCA reconstruction error)",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, errors in results.items():
        ks = sorted(errors)
        ax.plot(ks, [errors[k] for k in ks], marker="o", markersize=3, label=name)
    ax.set_xlabel("number of PCA components k")
    ax.set_ylabel("reconstruction error (MSE)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
