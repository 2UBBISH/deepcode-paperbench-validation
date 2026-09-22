"""MLP state-diversity metric of Sec. 6.4 / Figure 8.

"We train feedforward networks with small hidden layers on the task of input
reconstruction on batches of environment states visited by our algorithm and
PPO during training.  The idea behind this is that if a batch of states has a
more diverse data distribution then it should be harder to reconstruct the
distribution using small hidden layers. [...] we find that training error is
consistently higher for our method compared to PPO across different hidden
layer sizes."

Addendum (``paper/addendum.md``): "the neural network was a two layer of the
same size (the size is shown in the x-axis of the plot).  The activation
function used was ReLU, trained with Adam optimizer using default
hyperparameters from pytorch.  Each method was trained on 400k state-transitions
on an L2 reconstruction loss."
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class ReconstructionMLP(nn.Module):
    """Two hidden layers of the *same* size with ReLU activations."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_reconstruction_mlp(
    states: np.ndarray,
    hidden_size: int,
    num_transitions: int = 400_000,
    batch_size: int = 4096,
    input_dim: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
    lr: float = 1e-3,
) -> float:
    """Train on ``num_transitions`` states and return the training error (MSE).

    Adam is used with PyTorch's default hyperparameters (``lr=1e-3``,
    ``betas=(0.9, 0.999)``) as specified in the addendum; the loss is an L2
    reconstruction loss.
    """
    torch.manual_seed(seed)
    states = np.asarray(states, dtype=np.float32)
    if input_dim is None:
        input_dim = states.shape[1]
    states_t = torch.as_tensor(states, device=device)
    n = states_t.shape[0]
    if n == 0:
        raise ValueError("No states provided")

    model = ReconstructionMLP(input_dim, int(hidden_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters())  # PyTorch defaults
    generator = torch.Generator().manual_seed(seed)

    steps = max(1, int(num_transitions // max(1, batch_size)))
    last_loss = float("nan")
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=generator).to(device)
        batch = states_t[idx]
        recon = model(batch)
        loss = ((recon - batch) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def analyse_mlp_diversity(
    datasets: Dict[str, np.ndarray],
    hidden_sizes: Iterable[int] = (2, 4, 8, 16, 32, 64),
    num_transitions: int = 400_000,
    batch_size: int = 4096,
    seeds: Sequence[int] = (0,),
) -> Dict[str, Dict[int, float]]:
    """Training error as a function of the hidden layer size, per dataset."""
    results: Dict[str, Dict[int, float]] = {}
    for name, states in datasets.items():
        errors: Dict[int, float] = {}
        for hidden in hidden_sizes:
            seed_errors = [
                train_reconstruction_mlp(
                    states,
                    hidden_size=int(hidden),
                    num_transitions=num_transitions,
                    batch_size=batch_size,
                    seed=int(seed),
                )
                for seed in seeds
            ]
            errors[int(hidden)] = float(np.mean(seed_errors))
        results[name] = errors
    return results


def plot_mlp_diversity(
    results: Dict[str, Dict[int, float]],
    out_path: str,
    title: str = "State diversity (MLP reconstruction error)",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, errors in results.items():
        sizes = sorted(errors)
        ax.plot(sizes, [errors[s] for s in sizes], marker="o", markersize=3, label=name)
    ax.set_xlabel("hidden layer size")
    ax.set_ylabel("training reconstruction error (MSE)")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
