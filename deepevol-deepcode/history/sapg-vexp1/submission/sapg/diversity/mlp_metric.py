"""MLP-based diversity metric (SAPG paper Sec. 6.4, Figure 8).

The paper measures policy behavioural diversity by fitting a 2-layer MLP to
reconstruct pooled state-action data collected from a *single* policy and
reporting the reconstruction error.  A policy whose behaviour is more diverse
(i.e. covers a wider region of state-action space) is harder to reconstruct
with a small network, so it yields a *higher* reconstruction error.

Concretely (Sec. 6.4 / Fig. 8):

    * Collect ~400k transitions from the policy under evaluation.
    * Fit a 2-layer MLP (ReLU activations, Adam with PyTorch defaults, L2 loss)
      to reconstruct the input features (obs, action) from themselves.
    * Sweep the hidden-layer width and report the mean squared reconstruction
      error on held-out data.

SAPG is expected to show consistently *higher* reconstruction error than PPO
across hidden sizes, indicating greater behavioural diversity.

This module is intentionally simulator-agnostic: it accepts any policy object
exposing ``act(obs)`` (or ``forward``) and any vectorized env exposing
``reset()`` / ``step(actions)``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MLPDiversityMetric",
    "compute_mlp_reconstruction_error",
    "mlp_diversity_curve",
    "plot_mlp_diversity",
    "collect_policy_data",
]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_policy_data(
    policy: Any,
    env: Any,
    num_transitions: int = 400_000,
    horizon: int = 16,
    device: str = "cpu",
    include_actions: bool = True,
    include_next_obs: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Roll out ``policy`` in ``env`` and return a ``[num_transitions, d]`` matrix.

    The feature vector for each transition is ``[obs, action]`` by default
    (matching the paper's pooled state-action data).  ``include_next_obs``
    appends the next observation as well.

    The function is simulator-agnostic: ``env`` only needs ``reset()`` and
    ``step(actions)`` and ``policy`` only needs ``act(obs)`` (a tuple return of
    ``(action, ...)`` is handled).
    """
    import torch  # local import: keep module importable without torch

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    obs = env.reset()
    obs = _to_numpy(obs)

    features: List[np.ndarray] = []
    collected = 0
    steps = 0
    max_steps = max(1, int(np.ceil(num_transitions / max(1, obs.shape[0]))) + 1)

    while collected < num_transitions and steps < max_steps:
        steps += 1
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            out = policy.act(obs_t)
            if isinstance(out, tuple):
                action = out[0]
            else:
                action = out
            action_np = _to_numpy(action)

        next_obs, _rew, _done, _info = env.step(action_np)
        next_obs = _to_numpy(next_obs)

        if include_actions:
            feat = np.concatenate([obs, action_np], axis=-1)
        else:
            feat = obs
        if include_next_obs:
            feat = np.concatenate([feat, next_obs], axis=-1)

        features.append(feat.astype(np.float32))
        collected += feat.shape[0]
        obs = next_obs

    if not features:
        raise RuntimeError("collect_policy_data: no transitions were collected")

    data = np.concatenate(features, axis=0)
    if data.shape[0] > num_transitions:
        data = data[:num_transitions]
    return data


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors / lists / arrays to a CPU float32 numpy array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:  # pragma: no cover - torch always available in practice
        pass
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# MLP autoencoder / reconstruction
# ---------------------------------------------------------------------------
def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, device: str):
    """Build a 2-layer MLP with ReLU activations (paper Sec. 6.4)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ).to(device)


def compute_mlp_reconstruction_error(
    data: np.ndarray,
    hidden_dim: int = 256,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = "cpu",
    val_fraction: float = 0.1,
    seed: int = 0,
    max_samples: Optional[int] = 400_000,
    verbose: bool = False,
) -> float:
    """Fit a 2-layer MLP to reconstruct ``data`` and return the validation MSE.

    Uses ReLU activations, Adam with PyTorch defaults and an L2 (MSE) loss,
    matching the paper's description of the MLP diversity metric.
    """
    import torch
    import torch.nn as nn

    if data.ndim != 2:
        raise ValueError(f"expected 2-D data, got shape {data.shape}")

    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=np.float32)
    if max_samples is not None and x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]

    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train = torch.as_tensor(x[train_idx], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(x[val_idx], dtype=torch.float32, device=device)

    dim = x.shape[1]
    model = _build_mlp(dim, hidden_dim, dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = x_train.shape[0]
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            batch = x_train[idx]
            pred = model(batch)
            loss = loss_fn(pred, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if verbose:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), x_val).item()
            print(f"[mlp_metric] epoch {epoch + 1}/{epochs} val_mse={val_loss:.6f}")

    model.eval()
    with torch.no_grad():
        val_mse = loss_fn(model(x_val), x_val).item()
    return float(val_mse)


# ---------------------------------------------------------------------------
# Metric container
# ---------------------------------------------------------------------------
class MLPDiversityMetric:
    """Computes MLP reconstruction-error curves for one or more policies.

    Parameters
    ----------
    hidden_sizes:
        Hidden-layer widths to sweep (Figure 8 x-axis).
    epochs, batch_size, lr:
        Optimisation hyperparameters for the reconstruction MLP.
    max_samples:
        Cap on the number of transitions used (paper uses 400k).
    device:
        Torch device string.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (16, 32, 64, 128, 256, 512),
        epochs: int = 20,
        batch_size: int = 4096,
        lr: float = 1e-3,
        max_samples: Optional[int] = 400_000,
        device: str = "cpu",
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.hidden_sizes = list(hidden_sizes)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_samples = max_samples
        self.device = device
        self.seed = seed
        self.verbose = verbose

    # -- single dataset -----------------------------------------------------
    def curve(self, data: np.ndarray) -> Dict[int, float]:
        """Return ``{hidden_size: reconstruction_error}`` for one dataset."""
        out: Dict[int, float] = {}
        for h in self.hidden_sizes:
            out[int(h)] = compute_mlp_reconstruction_error(
                data,
                hidden_dim=int(h),
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
                seed=self.seed,
                max_samples=self.max_samples,
                verbose=self.verbose,
            )
        return out

    # -- multiple datasets --------------------------------------------------
    def curve_multi(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Dict[int, float]]:
        """Return ``{name: {hidden_size: error}}`` for several datasets."""
        return {name: self.curve(data) for name, data in datasets.items()}

    # -- convenience: collect + curve --------------------------------------
    def curve_from_policies(
        self,
        policies: Dict[str, Any],
        env: Any,
        num_transitions: int = 400_000,
        horizon: int = 16,
        include_actions: bool = True,
    ) -> Dict[str, Dict[int, float]]:
        """Collect data from each policy and compute its reconstruction curve."""
        datasets: Dict[str, np.ndarray] = {}
        for name, policy in policies.items():
            datasets[name] = collect_policy_data(
                policy,
                env,
                num_transitions=num_transitions,
                horizon=horizon,
                device=self.device,
                include_actions=include_actions,
                seed=self.seed,
            )
        return self.curve_multi(datasets)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def mlp_diversity_curve(
    datasets: Dict[str, np.ndarray],
    hidden_sizes: Optional[Sequence[int]] = None,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    max_samples: Optional[int] = 400_000,
    device: str = "cpu",
    seed: int = 0,
) -> Dict[str, Dict[int, float]]:
    """Compute MLP reconstruction-error curves for named datasets."""
    metric = MLPDiversityMetric(
        hidden_sizes=hidden_sizes or (16, 32, 64, 128, 256, 512),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_samples=max_samples,
        device=device,
        seed=seed,
    )
    return metric.curve_multi(datasets)


def plot_mlp_diversity(
    curves: Dict[str, Dict[int, float]],
    filename: Optional[str] = None,
    title: str = "MLP reconstruction error vs hidden size",
    xlabel: str = "hidden size",
    ylabel: str = "reconstruction error",
    log_x: bool = True,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> Optional[str]:
    """Plot Figure 8-style curves; saves to ``filename`` if provided."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    fig, ax = plt.subplots(figsize=figsize)
    for name, curve in curves.items():
        xs = sorted(curve.keys())
        ys = [curve[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=name)
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
    plt.close(fig)
    return None
