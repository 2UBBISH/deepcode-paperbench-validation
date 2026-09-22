"""L2 reconstruction experiment (Figure 8).

This experiment studies the *representational capacity* of the shared-network
parameterisation used by SAPG.  The paper (Figure 8) trains a two-layer neural
network with a fixed hidden size per layer on 400k state-transitions using an
L2 reconstruction loss and reports the reconstruction error as a function of
the network size.

The experiment is deliberately self-contained: it does not depend on the RL
environments or the SAPG trainer.  It only needs PyTorch and NumPy.

Two "methods" are compared, mirroring the paper's comparison between a single
shared network and a set of per-worker (conditioned) networks:

* ``shared``      -- a single two-layer network trained on all transitions.
* ``per_worker``  -- ``num_workers`` independent two-layer networks, each
                     trained on its own block of transitions (the "split"
                     representation).  The reported error is the average over
                     workers.

Both use ReLU activations and the Adam optimiser with PyTorch defaults, as
specified in the paper.

Usage
-----
    python -m experiments.run_reconstruction --sizes 16,32,64,128,256 \
        --num_transitions 400000 --output_dir runs/reconstruction

The runner writes ``reconstruction_summary.json`` with the schema consumed by
``eval/plot.py``::

    {
      "env_name": ...,
      "task": ...,
      "num_transitions": 400000,
      "results": {
        "shared":     {"sizes": [...], "errors": [...]},
        "per_worker": {"sizes": [...], "errors": [...]}
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Path handling: allow running both as ``python -m experiments.run_reconstruction``
# and as a standalone script.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


DEFAULT_SIZES: List[int] = [16, 32, 64, 128, 256]
DEFAULT_NUM_TRANSITIONS: int = 400_000
DEFAULT_METHODS: List[str] = ["shared", "per_worker"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_transitions(
    num_transitions: int,
    obs_dim: int = 64,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a synthetic set of state-transitions.

    The paper trains on 400k state-transitions drawn from the replay of the
    manipulation tasks.  Since we do not ship the raw IsaacGym trajectories we
    generate a structured synthetic dataset with a low intrinsic dimensionality
    (a smooth non-linear manifold) so that reconstruction error exhibits the
    same qualitative dependence on network size as in the paper.

    Args:
        num_transitions: number of transitions (rows) to generate.
        obs_dim: dimensionality of each state vector.
        seed: RNG seed for reproducibility.
        device: torch device for the returned tensor.

    Returns:
        Tensor of shape ``(num_transitions, obs_dim)``.
    """
    rng = np.random.default_rng(seed)

    # Latent factors with a low intrinsic dimension (e.g. 8) mapped through a
    # random smooth non-linearity to produce a curved manifold in R^obs_dim.
    latent_dim = min(8, obs_dim)
    latent = rng.standard_normal((num_transitions, latent_dim)).astype(np.float32)

    # Fixed random projection + non-linearity (deterministic given the seed).
    proj = rng.standard_normal((latent_dim, obs_dim)).astype(np.float32) / np.sqrt(latent_dim)
    hidden = np.tanh(latent @ proj)
    proj2 = rng.standard_normal((obs_dim, obs_dim)).astype(np.float32) / np.sqrt(obs_dim)
    data = np.tanh(hidden @ proj2)

    # Small observation noise.
    data = data + 0.01 * rng.standard_normal(data.shape).astype(np.float32)

    return torch.as_tensor(data, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TwoLayerNet(nn.Module):
    """Two-layer MLP with ReLU activations (same size per layer).

    The paper specifies a two-layer network with the *same* hidden size in each
    layer, ReLU activation, and an L2 reconstruction objective.
    """

    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, obs_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_reconstruction(
    data: torch.Tensor,
    hidden_size: int,
    num_epochs: int = 20,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> float:
    """Train a two-layer net to reconstruct ``data`` with an L2 loss.

    Args:
        data: tensor of shape ``(N, obs_dim)``.
        hidden_size: hidden size used for *both* layers.
        num_epochs: number of passes over the dataset.
        batch_size: mini-batch size.
        learning_rate: Adam learning rate (PyTorch defaults otherwise).
        seed: RNG seed.
        device: torch device.

    Returns:
        Final mean L2 reconstruction error (MSE) on the training data.
    """
    if device is None:
        device = data.device

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = data.shape[1]
    model = TwoLayerNet(obs_dim, hidden_size).to(device)
    # Adam with PyTorch defaults (betas=(0.9, 0.999), eps=1e-8, weight_decay=0).
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    num_samples = data.shape[0]
    steps_per_epoch = max(1, num_samples // batch_size)

    model.train()
    for _ in range(num_epochs):
        perm = torch.randperm(num_samples, device=device)
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            batch = data[idx]
            pred = model(batch)
            loss = loss_fn(pred, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # Final evaluation (full dataset, no grad).
    model.eval()
    with torch.no_grad():
        pred = model(data)
        final_error = float(loss_fn(pred, data).item())
    return final_error


def train_per_worker(
    data: torch.Tensor,
    hidden_size: int,
    num_workers: int = 4,
    num_epochs: int = 20,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> float:
    """Train ``num_workers`` independent nets, each on its own data block.

    Returns the average final L2 reconstruction error across workers.
    """
    if device is None:
        device = data.device

    num_samples = data.shape[0]
    block = num_samples // num_workers
    errors: List[float] = []
    for w in range(num_workers):
        start = w * block
        end = num_samples if w == num_workers - 1 else (w + 1) * block
        worker_data = data[start:end]
        err = train_reconstruction(
            worker_data,
            hidden_size=hidden_size,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + w,
            device=device,
        )
        errors.append(err)
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(
    sizes: Sequence[int],
    methods: Sequence[str] = DEFAULT_METHODS,
    num_transitions: int = DEFAULT_NUM_TRANSITIONS,
    obs_dim: int = 64,
    num_workers: int = 4,
    num_epochs: int = 20,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the reconstruction sweep over network sizes and methods.

    Returns a payload dict with the schema consumed by ``eval/plot.py``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = generate_transitions(num_transitions, obs_dim=obs_dim, seed=seed, device=device)

    results: Dict[str, Dict[str, List[float]]] = {}
    for method in methods:
        errors: List[float] = []
        for size in sizes:
            if verbose:
                print(f"[reconstruction] method={method} size={size} ...", flush=True)
            if method == "per_worker":
                err = train_per_worker(
                    data,
                    hidden_size=int(size),
                    num_workers=num_workers,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed,
                    device=device,
                )
            else:
                err = train_reconstruction(
                    data,
                    hidden_size=int(size),
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed,
                    device=device,
                )
            errors.append(float(err))
            if verbose:
                print(f"[reconstruction]   -> L2 error = {err:.6f}", flush=True)
        results[method] = {"sizes": [int(s) for s in sizes], "errors": errors}

    payload: Dict[str, Any] = {
        "env_name": "synthetic",
        "task": "reconstruction",
        "num_transitions": int(num_transitions),
        "obs_dim": int(obs_dim),
        "num_workers": int(num_workers),
        "num_epochs": int(num_epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "results": results,
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L2 reconstruction experiment (Figure 8) for SAPG."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated list of hidden sizes (same size per layer).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated list of methods: shared,per_worker.",
    )
    parser.add_argument(
        "--num_transitions",
        type=int,
        default=DEFAULT_NUM_TRANSITIONS,
        help="Number of state-transitions to train on (paper: 400000).",
    )
    parser.add_argument("--obs_dim", type=int, default=64, help="Observation dimensionality.")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers for per_worker method.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Training epochs per network.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Mini-batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cpu/cuda).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/reconstruction",
        help="Directory to write reconstruction_summary.json.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also render Figure 8 via eval.plot after the sweep.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    sizes = _parse_int_list(args.sizes)
    methods = _parse_str_list(args.methods)

    device = None
    if args.device is not None:
        device = torch.device(args.device)

    print("=" * 70)
    print("SAPG :: L2 reconstruction experiment (Figure 8)")
    print(f"  sizes          : {sizes}")
    print(f"  methods        : {methods}")
    print(f"  num_transitions: {args.num_transitions}")
    print(f"  obs_dim        : {args.obs_dim}")
    print(f"  num_workers    : {args.num_workers}")
    print(f"  num_epochs     : {args.num_epochs}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  learning_rate  : {args.learning_rate}")
    print(f"  device         : {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 70)

    payload = run_sweep(
        sizes=sizes,
        methods=methods,
        num_transitions=args.num_transitions,
        obs_dim=args.obs_dim,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
        verbose=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "reconstruction_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[reconstruction] wrote summary to {out_path}")

    if args.plot:
        try:
            from eval.plot import plot_reconstruction  # type: ignore

            fig_path = plot_reconstruction(
                payload,
                output_path=os.path.join(args.output_dir, "fig8_reconstruction.png"),
            )
            print(f"[reconstruction] wrote figure to {fig_path}")
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"[reconstruction] plotting skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
