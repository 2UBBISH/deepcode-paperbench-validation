"""PCA-based diversity metric (SAPG paper, Sec. 6.4, Figure 7).

The metric measures how *diverse* the behaviours of a set of policies are by
fitting a low-dimensional linear model (PCA) to the joint state-action data
collected from all policies and measuring the reconstruction error as a
function of the number of principal components ``k``.

Intuition
---------
If all policies behave identically, their data lies on a low-dimensional
manifold and a small number of principal components reconstructs it almost
perfectly (reconstruction error decays quickly with ``k``).  If the policies
behave differently, the data spans a higher-dimensional manifold and the
reconstruction error decays more slowly.  SAPG is expected to show the
*slowest* decay (i.e. the most diverse policies), while PPO (a single policy)
shows the fastest decay.

The implementation follows the paper's description:

* Collect a fixed number of transitions (default 400k) from each policy.
* Standardise the features.
* Fit PCA with ``k`` components for ``k`` in ``1..k_max``.
* Report the mean squared reconstruction error (normalised by the data
  variance) for each ``k``.

The module is deliberately dependency-light: it only requires ``numpy`` and
``scikit-learn`` (for the PCA solver).  A pure-numpy fallback using the SVD is
provided so the metric can be computed even when scikit-learn is unavailable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PCADiversityMetric",
    "compute_pca_reconstruction_error",
    "collect_policy_data",
    "pca_diversity_curve",
    "plot_pca_diversity",
]


# ---------------------------------------------------------------------------
# Core PCA reconstruction error
# ---------------------------------------------------------------------------
def _fit_pca_svd(x_centered: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA via SVD and return (components, mean).

    Parameters
    ----------
    x_centered:
        Zero-mean data of shape ``[n, d]``.
    k:
        Number of principal components.

    Returns
    -------
    components:
        ``[k, d]`` orthonormal principal directions (rows).
    """
    # Economy SVD: x = U S V^T ; principal directions are rows of V^T.
    # Use full_matrices=False to keep memory bounded for large n.
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    k = int(min(k, vt.shape[0]))
    return vt[:k], k


def compute_pca_reconstruction_error(
    data: np.ndarray,
    k: int,
    mean: Optional[np.ndarray] = None,
    components: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> float:
    """Mean squared reconstruction error of ``data`` using ``k`` PCs.

    Parameters
    ----------
    data:
        Array of shape ``[n, d]`` (transitions x features).
    k:
        Number of principal components used for reconstruction.
    mean:
        Optional pre-computed feature mean (shape ``[d]``).  If ``None`` it is
        computed from ``data``.
    components:
        Optional pre-computed principal directions of shape ``[k, d]``.  If
        ``None`` they are computed from ``data`` via SVD.
    normalize:
        If ``True`` (default) the error is divided by the total variance of
        ``data`` so that curves are comparable across tasks / feature scales.

    Returns
    -------
    float
        The (optionally normalised) mean squared reconstruction error.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"data must be 2-D [n, d], got shape {x.shape}")
    n, d = x.shape
    if n == 0:
        return 0.0

    if mean is None:
        mean = x.mean(axis=0)
    x_centered = x - mean

    if components is None:
        components, k = _fit_pca_svd(x_centered, k)
    else:
        components = np.asarray(components, dtype=np.float64)
        k = int(min(k, components.shape[0]))
        components = components[:k]

    # Project onto the k principal directions and reconstruct.
    proj = x_centered @ components.T          # [n, k]
    recon = proj @ components                 # [n, d]
    residual = x_centered - recon
    mse = float(np.mean(np.sum(residual ** 2, axis=1)))

    if normalize:
        total_var = float(np.mean(np.sum(x_centered ** 2, axis=1)))
        if total_var > 0:
            mse = mse / total_var
    return mse


# ---------------------------------------------------------------------------
# High-level metric object
# ---------------------------------------------------------------------------
class PCADiversityMetric:
    """Compute PCA reconstruction-error curves for one or more policies.

    Parameters
    ----------
    k_values:
        Sequence of component counts to evaluate.  Defaults to
        ``[1, 2, 4, 8, 16, 32, 64, 128]`` (log-spaced, matching Fig. 7).
    max_samples:
        Maximum number of transitions used to fit / evaluate PCA.  The paper
        uses 400k transitions; we subsample if more are provided.
    normalize:
        Whether to normalise the reconstruction error by total variance.
    seed:
        RNG seed for subsampling.
    """

    def __init__(
        self,
        k_values: Optional[Sequence[int]] = None,
        max_samples: int = 400_000,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        if k_values is None:
            k_values = [1, 2, 4, 8, 16, 32, 64, 128]
        self.k_values: List[int] = [int(k) for k in k_values]
        self.max_samples = int(max_samples)
        self.normalize = bool(normalize)
        self.seed = int(seed)

    # -- data preparation ---------------------------------------------------
    def _prepare(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"data must be 2-D [n, d], got shape {x.shape}")
        if x.shape[0] > self.max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(x.shape[0], size=self.max_samples, replace=False)
            x = x[idx]
        return x

    # -- public API ---------------------------------------------------------
    def curve(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute the reconstruction-error curve for a single dataset.

        Returns a dict with keys ``k`` (list of ints), ``error`` (list of
        floats) and ``n_samples`` / ``n_features``.
        """
        x = self._prepare(data)
        mean = x.mean(axis=0)
        x_centered = x - mean
        # Fit the largest PCA once and slice components for each k.
        max_k = min(max(self.k_values), x_centered.shape[0], x_centered.shape[1])
        components, _ = _fit_pca_svd(x_centered, max_k)

        errors: List[float] = []
        for k in self.k_values:
            err = compute_pca_reconstruction_error(
                x,
                k=k,
                mean=mean,
                components=components,
                normalize=self.normalize,
            )
            errors.append(err)

        return {
            "k": list(self.k_values),
            "error": errors,
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
        }

    def curve_multi(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Compute curves for several named datasets (e.g. methods)."""
        return {name: self.curve(data) for name, data in datasets.items()}

    def pooled_curve(self, datasets: Iterable[np.ndarray]) -> Dict[str, Any]:
        """Compute a single curve over the union of several datasets.

        This is the metric used in the paper: data from *all* policies of a
        method is pooled, then PCA is fit to the pooled data.  A method whose
        policies are diverse produces a slower-decaying curve.
        """
        arrays = [np.asarray(d, dtype=np.float64) for d in datasets]
        arrays = [a for a in arrays if a.size > 0]
        if not arrays:
            raise ValueError("pooled_curve requires at least one non-empty dataset")
        pooled = np.concatenate(arrays, axis=0)
        return self.curve(pooled)


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------
def collect_policy_data(
    policy,
    env,
    num_transitions: int = 400_000,
    horizon: int = 16,
    device: str = "cpu",
    include_actions: bool = True,
    include_next_obs: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Roll out ``policy`` in ``env`` and collect a feature matrix.

    The feature vector for each transition is ``[obs, action]`` (optionally
    ``[obs, action, next_obs]``).  The function is simulator-agnostic and
    works with any env exposing ``reset()`` / ``step(actions)`` and any policy
    exposing ``act(obs)`` (returning either actions or ``(actions, ...)``).

    Parameters
    ----------
    policy:
        Callable policy object.  Must expose ``act(obs)``; may return a tuple
        whose first element is the action tensor/array.
    env:
        Vectorised environment with ``reset()`` and ``step(actions)``.
    num_transitions:
        Target number of transitions to collect.
    horizon:
        Number of steps per rollout chunk before re-collecting.
    device:
        Torch device string used when converting observations.
    include_actions:
        Whether to append actions to the feature vector.
    include_next_obs:
        Whether to append next observations to the feature vector.
    seed:
        Optional seed for the environment reset.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``[num_transitions, d]``.
    """
    import torch  # local import to keep module importable without torch

    obs = env.reset()
    if seed is not None and hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass

    features: List[np.ndarray] = []
    collected = 0
    while collected < num_transitions:
        steps = min(horizon, max(1, (num_transitions - collected) // max(1, env.num_envs)))
        for _ in range(steps):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
            with torch.no_grad():
                out = policy.act(obs_t)
            actions = out[0] if isinstance(out, (tuple, list)) else out
            actions_np = actions.detach().cpu().numpy()

            next_obs, _, dones, _ = env.step(actions_np)

            obs_np = np.asarray(obs, dtype=np.float64).reshape(env.num_envs, -1)
            act_np = np.asarray(actions_np, dtype=np.float64).reshape(env.num_envs, -1)
            if include_actions and include_next_obs:
                nxt_np = np.asarray(next_obs, dtype=np.float64).reshape(env.num_envs, -1)
                feat = np.concatenate([obs_np, act_np, nxt_np], axis=1)
            elif include_actions:
                feat = np.concatenate([obs_np, act_np], axis=1)
            elif include_next_obs:
                nxt_np = np.asarray(next_obs, dtype=np.float64).reshape(env.num_envs, -1)
                feat = np.concatenate([obs_np, nxt_np], axis=1)
            else:
                feat = obs_np
            features.append(feat)
            collected += feat.shape[0]

            obs = next_obs
            if np.any(dones):
                obs = env.reset()

    data = np.concatenate(features, axis=0)
    if data.shape[0] > num_transitions:
        data = data[:num_transitions]
    return data


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def pca_diversity_curve(
    datasets: Dict[str, np.ndarray],
    k_values: Optional[Sequence[int]] = None,
    max_samples: int = 400_000,
    normalize: bool = True,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Compute PCA reconstruction-error curves for several named datasets.

    Parameters
    ----------
    datasets:
        Mapping ``method_name -> feature matrix [n, d]``.  Each matrix should
        pool the transitions of all policies belonging to that method.
    k_values, max_samples, normalize, seed:
        Forwarded to :class:`PCADiversityMetric`.

    Returns
    -------
    dict
        ``method_name -> {"k": [...], "error": [...], ...}``.
    """
    metric = PCADiversityMetric(
        k_values=k_values,
        max_samples=max_samples,
        normalize=normalize,
        seed=seed,
    )
    return metric.curve_multi(datasets)


def plot_pca_diversity(
    curves: Dict[str, Dict[str, Any]],
    filename: Optional[str] = None,
    title: str = "PCA reconstruction error vs #components",
    xlabel: str = "Number of principal components (k)",
    ylabel: str = "Normalised reconstruction error",
    log_x: bool = True,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> Optional[str]:
    """Plot PCA diversity curves (reproduces Figure 7 of the paper).

    Parameters
    ----------
    curves:
        Output of :func:`pca_diversity_curve` (or :meth:`PCADiversityMetric.curve_multi`).
    filename:
        If provided, the figure is saved to this path.
    log_x:
        Whether to use a log-scaled x-axis (the paper uses log-spaced ``k``).

    Returns
    -------
    str or None
        The filename if saved, else ``None``.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    fig, ax = plt.subplots(figsize=figsize)
    for name, curve in curves.items():
        ax.plot(curve["k"], curve["error"], marker="o", label=name)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if filename:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
def _main() -> None:  # pragma: no cover - manual smoke test
    rng = np.random.default_rng(0)
    # Two synthetic "methods": one low-dimensional, one high-dimensional.
    low = rng.normal(size=(5000, 8)) @ rng.normal(size=(8, 32))
    high = rng.normal(size=(5000, 32))
    curves = pca_diversity_curve({"low-dim": low, "high-dim": high})
    for name, c in curves.items():
        print(name, "k=", c["k"])
        print(name, "err=", [round(e, 4) for e in c["error"]])


if __name__ == "__main__":  # pragma: no cover
    _main()
