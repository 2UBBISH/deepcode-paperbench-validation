"""Reconstruction experiment (Figure 8).

Trains a 2-layer MLP with ReLU activations to reconstruct state-transitions
using an L2 reconstruction loss.  The experiment sweeps over network size
(the x-axis of Figure 8) and reports reconstruction quality (L2 loss) as a
function of network size.

Paper specification (from the reproduction plan):
    - 2-layer MLP (same size, size on x-axis)
    - ReLU activation
    - Adam optimizer (PyTorch defaults)
    - 400k state-transitions
    - L2 reconstruction loss
    - Compare reconstruction quality vs network size

The script is self-contained and framework-agnostic: it generates synthetic
state-transition data (or loads it from disk if provided) and trains a small
MLP to reconstruct the target transitions.  This mirrors the paper's
motivation that a modest network can reconstruct the transition data used by
SAPG, motivating the shared-network design.

Usage
-----
    python -m experiments.run_reconstruction --sizes 16 32 64 128 256 512
    python -m experiments.run_reconstruction --data path/to/transitions.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup so the script can be run directly from the project root or from
# inside the experiments/ directory.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_NUM_TRANSITIONS = 400_000
DEFAULT_STATE_DIM = 64
DEFAULT_HIDDEN_DIM = 256
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_SEED = 0
DEFAULT_SIZES: Tuple[int, ...] = (16, 32, 64, 128, 256, 512)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ReconstructionMLP(nn.Module):
    """Two-layer MLP with ReLU activations for L2 reconstruction.

    Architecture::

        input -> Linear(state_dim, hidden) -> ReLU
              -> Linear(hidden, hidden)    -> ReLU
              -> Linear(hidden, state_dim)

    The paper specifies a 2-layer MLP (two hidden layers) with ReLU
    activations, trained with Adam (PyTorch defaults) and an L2
    reconstruction loss.
    """

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def generate_transition_data(
    num_transitions: int = DEFAULT_NUM_TRANSITIONS,
    state_dim: int = DEFAULT_STATE_DIM,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic state-transition data for reconstruction.

    The data is drawn from a smooth, low-dimensional manifold embedded in
    ``state_dim`` dimensions (a random linear projection of a low-rank
    Gaussian), which mimics the structure of real RL transition data and
    makes the reconstruction task non-trivial but learnable.

    Returns
    -------
    (inputs, targets) : Tuple[torch.Tensor, torch.Tensor]
        Both of shape ``(num_transitions, state_dim)``.
    """
    rng = np.random.default_rng(seed)
    latent_dim = max(2, state_dim // 8)
    latent = rng.standard_normal((num_transitions, latent_dim)).astype(np.float32)
    projection = rng.standard_normal((latent_dim, state_dim)).astype(np.float32)
    projection /= np.sqrt(latent_dim)
    data = latent @ projection
    # Add a small amount of observation noise.
    data += 0.01 * rng.standard_normal(data.shape).astype(np.float32)

    # Targets are a deterministic (nonlinear) function of the inputs so that
    # the MLP must learn a genuine mapping rather than the identity.
    targets = np.tanh(data) + 0.1 * data

    inputs_t = torch.from_numpy(data).to(device)
    targets_t = torch.from_numpy(targets).to(device)
    return inputs_t, targets_t


def load_transition_data(
    path: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load transition data from an ``.npz`` file.

    The file must contain arrays ``inputs`` and ``targets`` (or ``obs`` and
    ``next_obs``), each of shape ``(N, state_dim)``.
    """
    data = np.load(path)
    keys = set(data.files)
    if "inputs" in keys and "targets" in keys:
        inputs = data["inputs"]
        targets = data["targets"]
    elif "obs" in keys and "next_obs" in keys:
        inputs = data["obs"]
        targets = data["next_obs"]
    else:
        raise ValueError(
            f"Unrecognized data file {path!r}; expected keys 'inputs'/'targets' "
            f"or 'obs'/'next_obs', got {sorted(keys)}"
        )
    inputs = np.asarray(inputs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_reconstruction(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    hidden_dim: int,
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a 2-layer MLP to reconstruct ``targets`` from ``inputs``.

    Uses Adam with PyTorch defaults and an L2 (mean-squared-error)
    reconstruction loss, matching the paper's Figure 8 setup.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``hidden_dim``, ``final_loss``, ``loss_history``,
        ``num_params``, ``train_time``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = int(inputs.shape[1])
    model = ReconstructionMLP(state_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # PyTorch defaults
    criterion = nn.MSELoss()  # L2 reconstruction loss

    num_samples = int(inputs.shape[0])
    num_batches = max(1, num_samples // batch_size)

    loss_history: List[float] = []
    start = time.time()
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            batch_in = inputs[idx]
            batch_tgt = targets[idx]

            pred = model(batch_in)
            loss = criterion(pred, batch_tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().item())
        epoch_loss /= num_batches
        loss_history.append(epoch_loss)
        if verbose:
            print(f"    [hidden={hidden_dim}] epoch {epoch + 1}/{epochs} "
                  f"loss={epoch_loss:.6f}")

    train_time = time.time() - start

    # Final evaluation on the full dataset (deterministic).
    model.eval()
    with torch.no_grad():
        pred = model(inputs)
        final_loss = float(criterion(pred, targets).item())

    num_params = sum(p.numel() for p in model.parameters())

    return {
        "hidden_dim": int(hidden_dim),
        "final_loss": final_loss,
        "loss_history": loss_history,
        "num_params": int(num_params),
        "train_time": float(train_time),
    }


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run_reconstruction_experiment(
    sizes: Sequence[int] = DEFAULT_SIZES,
    *,
    num_transitions: int = DEFAULT_NUM_TRANSITIONS,
    state_dim: int = DEFAULT_STATE_DIM,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    data_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full reconstruction sweep over network sizes.

    Returns
    -------
    Dict[str, Any]
        ``{"results": [...], "sizes": [...], "losses": [...],
           "num_params": [...], "config": {...}}``
    """
    if data_path is not None and os.path.exists(data_path):
        if verbose:
            print(f"Loading transition data from {data_path}")
        inputs, targets = load_transition_data(data_path, device=device)
    else:
        if verbose:
            print(f"Generating {num_transitions} synthetic state-transitions "
                  f"(state_dim={state_dim})")
        inputs, targets = generate_transition_data(
            num_transitions=num_transitions,
            state_dim=state_dim,
            seed=seed,
            device=device,
        )

    results: List[Dict[str, Any]] = []
    for size in sizes:
        if verbose:
            print(f"Training reconstruction MLP with hidden_dim={size}")
        res = train_reconstruction(
            inputs,
            targets,
            hidden_dim=int(size),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        results.append(res)
        if verbose:
            print(f"  -> final L2 loss = {res['final_loss']:.6f} "
                  f"({res['num_params']} params)")

    return {
        "results": results,
        "sizes": [int(s) for s in sizes],
        "losses": [r["final_loss"] for r in results],
        "num_params": [r["num_params"] for r in results],
        "config": {
            "num_transitions": int(inputs.shape[0]),
            "state_dim": int(inputs.shape[1]),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "seed": int(seed),
            "device": str(device),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAPG reconstruction experiment (Figure 8): 2-layer MLP "
                    "L2 reconstruction vs network size."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="Hidden dimensions to sweep over.")
    parser.add_argument("--num-transitions", type=int, default=DEFAULT_NUM_TRANSITIONS,
                        help="Number of state-transitions to use (default: 400000).")
    parser.add_argument("--state-dim", type=int, default=DEFAULT_STATE_DIM,
                        help="Dimensionality of the state-transition vectors.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of training epochs per network size.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size for Adam training.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="Learning rate (Adam, PyTorch defaults otherwise).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda).")
    parser.add_argument("--data", type=str, default=None,
                        help="Optional path to an .npz file with transition data.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to write JSON results.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("SAPG Reconstruction Experiment (Figure 8)")
        print("=" * 70)

    out = run_reconstruction_experiment(
        sizes=args.sizes,
        num_transitions=args.num_transitions,
        state_dim=args.state_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        data_path=args.data,
        verbose=verbose,
    )

    if verbose:
        print("-" * 70)
        print(f"{'hidden_dim':>12} | {'num_params':>12} | {'final L2 loss':>14}")
        print("-" * 70)
        for size, params, loss in zip(out["sizes"], out["num_params"], out["losses"]):
            print(f"{size:>12} | {params:>12} | {loss:>14.6f}")
        print("-" * 70)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        if verbose:
            print(f"Wrote results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
