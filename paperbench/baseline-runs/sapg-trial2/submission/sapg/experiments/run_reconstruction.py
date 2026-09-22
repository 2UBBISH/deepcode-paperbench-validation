"""Figure 8 reproduction: representation / reconstruction study.

The SAPG paper (Figure 8) studies how well a *shared* network can represent
multiple distinct policies as a function of network capacity.  The experiment
is deliberately simple and self-contained:

  * Generate a dataset of 400k state-transitions (observations) drawn from a
    mixture of ``num_workers`` distinct "task" distributions.  Each task has a
    different, randomly chosen linear map that produces the observation, so a
    network must allocate capacity to represent all of them.
  * Train a two-layer ReLU network (``in -> hidden -> hidden -> out``) with the
    Adam optimizer (PyTorch defaults) to reconstruct the observations with an
    L2 reconstruction loss.
  * Sweep the network size (hidden width) on the x-axis and plot the final
    reconstruction error on the y-axis.

The expected qualitative result is that reconstruction error decreases as the
network grows, and that a *conditioned* network (given a per-worker embedding
``phi_j``, as in SAPG's shared ``B_theta``) achieves lower error at the same
size than an unconditioned network -- motivating SAPG's shared conditioned
architecture.

Usage
-----
    python experiments/run_reconstruction.py --sizes 8 16 32 64 128 256 \
        --num-workers 8 --steps 400000 --output results/reconstruction.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Allow running both as a script and as a module.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sapg.utils import get_device, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_dataset(
    num_samples: int,
    num_workers: int,
    obs_dim: int,
    latent_dim: int,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a mixture of ``num_workers`` linear observation distributions.

    Each worker ``j`` owns a random linear map ``W_j`` (``latent_dim -> obs_dim``)
    and a random bias ``b_j``.  A latent code ``z ~ N(0, I)`` is mapped through
    the worker's map to produce an observation.  The worker id is returned as
    the conditioning signal.

    Returns
    -------
    obs : (num_samples, obs_dim)
    worker_ids : (num_samples,) int64
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Per-worker linear maps.
    W = torch.randn(num_workers, latent_dim, obs_dim, generator=g) / (latent_dim ** 0.5)
    b = torch.randn(num_workers, obs_dim, generator=g) * 0.1

    worker_ids = torch.randint(0, num_workers, (num_samples,), generator=g)
    z = torch.randn(num_samples, latent_dim, generator=g)

    W_sel = W[worker_ids]  # (N, latent_dim, obs_dim)
    b_sel = b[worker_ids]  # (N, obs_dim)
    obs = torch.bmm(z.unsqueeze(1), W_sel).squeeze(1) + b_sel

    # Add a little observation noise.
    obs = obs + 0.01 * torch.randn(obs.shape, generator=g)

    return obs.to(device), worker_ids.to(device)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ReconstructionNet(nn.Module):
    """Two-layer ReLU network with optional per-worker conditioning.

    Architecture (paper Figure 8): ``in -> hidden -> hidden -> out`` with ReLU
    activations.  When ``conditioned=True`` a learned per-worker embedding
    ``phi_j`` (dim ``phi_dim``) is concatenated to the input, mirroring SAPG's
    shared conditioned network ``B_theta(obs, phi_j)``.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        num_workers: int = 1,
        conditioned: bool = False,
        phi_dim: int = 8,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.conditioned = conditioned
        self.num_workers = num_workers

        in_dim = obs_dim
        if conditioned:
            self.phi = nn.Embedding(num_workers, phi_dim)
            nn.init.normal_(self.phi.weight, std=0.01)
            in_dim = obs_dim + phi_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, obs_dim),
        )

    def forward(self, obs: torch.Tensor, worker_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.conditioned:
            if worker_ids is None:
                worker_ids = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
            phi = self.phi(worker_ids)
            x = torch.cat([obs, phi], dim=-1)
        else:
            x = obs
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_reconstruction(
    obs: torch.Tensor,
    worker_ids: torch.Tensor,
    hidden_size: int,
    num_workers: int,
    conditioned: bool,
    steps: int = 400_000,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    log_every: int = 0,
) -> Dict[str, Any]:
    """Train a single reconstruction network and return final metrics."""
    torch.manual_seed(seed)
    model = ReconstructionNet(
        obs_dim=obs.shape[-1],
        hidden_size=hidden_size,
        num_workers=num_workers,
        conditioned=conditioned,
    ).to(device)

    # Adam with PyTorch defaults (paper: "Adam optimizer (PyTorch defaults)").
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_samples = obs.shape[0]
    steps_per_epoch = max(1, num_samples // batch_size)
    total_epochs = max(1, steps // steps_per_epoch)

    losses: List[float] = []
    step = 0
    t0 = time.time()
    for epoch in range(total_epochs):
        perm = torch.randperm(num_samples, device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            if idx.numel() == 0:
                continue
            batch_obs = obs[idx]
            batch_wid = worker_ids[idx]

            pred = model(batch_obs, batch_wid)
            loss = torch.mean((pred - batch_obs) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            step += 1
            if log_every and step % log_every == 0:
                print(f"    step {step:>7d}  loss {losses[-1]:.6f}")

    # Final evaluation over the full dataset (in chunks to bound memory).
    model.eval()
    with torch.no_grad():
        total_sq = 0.0
        total_n = 0
        for i in range(0, num_samples, 16384):
            b_obs = obs[i : i + 16384]
            b_wid = worker_ids[i : i + 16384]
            pred = model(b_obs, b_wid)
            total_sq += float(torch.sum((pred - b_obs) ** 2).cpu())
            total_n += b_obs.numel()
        final_mse = total_sq / max(1, total_n)

    return {
        "hidden_size": hidden_size,
        "conditioned": conditioned,
        "final_mse": final_mse,
        "final_loss": losses[-1] if losses else float("nan"),
        "num_steps": step,
        "wall_time": time.time() - t0,
        "loss_curve": losses[:: max(1, len(losses) // 200)] if losses else [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 8: reconstruction vs network size")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512],
                   help="Hidden widths to sweep (x-axis of Figure 8).")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Number of distinct tasks/workers in the mixture.")
    p.add_argument("--obs-dim", type=int, default=32, help="Observation dimension.")
    p.add_argument("--latent-dim", type=int, default=8, help="Latent dimension of the data.")
    p.add_argument("--num-samples", type=int, default=400_000,
                   help="Number of state-transitions in the dataset (paper: 400k).")
    p.add_argument("--steps", type=int, default=400_000,
                   help="Number of gradient steps (paper: 400k transitions).")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--conditioned", action="store_true", default=True,
                   help="Use a conditioned (per-worker phi_j) network.")
    p.add_argument("--unconditioned", action="store_true", default=True,
                   help="Also run the unconditioned baseline for comparison.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default="results/reconstruction.json")
    p.add_argument("--log-every", type=int, default=0)
    return p.parse_args(argv)


def run_reconstruction(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device) if args.device else get_device()

    results: Dict[str, Any] = {
        "config": {
            "sizes": list(args.sizes),
            "num_workers": args.num_workers,
            "obs_dim": args.obs_dim,
            "latent_dim": args.latent_dim,
            "num_samples": args.num_samples,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": list(args.seeds),
        },
        "conditioned": {},
        "unconditioned": {},
    }

    variants: List[Tuple[str, bool]] = []
    if args.conditioned:
        variants.append(("conditioned", True))
    if args.unconditioned:
        variants.append(("unconditioned", False))

    for variant_name, conditioned in variants:
        print(f"\n=== Variant: {variant_name} ===")
        for size in args.sizes:
            per_seed: List[Dict[str, Any]] = []
            for seed in args.seeds:
                set_seed(seed)
                obs, worker_ids = generate_dataset(
                    num_samples=args.num_samples,
                    num_workers=args.num_workers,
                    obs_dim=args.obs_dim,
                    latent_dim=args.latent_dim,
                    seed=seed,
                    device=device,
                )
                print(f"  size={size:>4d} seed={seed} ...", end="", flush=True)
                metrics = train_reconstruction(
                    obs=obs,
                    worker_ids=worker_ids,
                    hidden_size=size,
                    num_workers=args.num_workers,
                    conditioned=conditioned,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=seed,
                    device=device,
                    log_every=args.log_every,
                )
                print(f" mse={metrics['final_mse']:.6f} ({metrics['wall_time']:.1f}s)")
                per_seed.append(metrics)

            mean_mse = sum(m["final_mse"] for m in per_seed) / len(per_seed)
            results[variant_name][str(size)] = {
                "mean_mse": mean_mse,
                "per_seed": per_seed,
            }

    return results


def summarize(results: Dict[str, Any]) -> None:
    sizes = results["config"]["sizes"]
    print("\n" + "=" * 60)
    print("Figure 8: reconstruction error vs network size")
    print("=" * 60)
    header = f"{'size':>6} | {'conditioned':>14} | {'unconditioned':>14}"
    print(header)
    print("-" * len(header))
    for size in sizes:
        cond = results["conditioned"].get(str(size), {}).get("mean_mse", float("nan"))
        uncond = results["unconditioned"].get(str(size), {}).get("mean_mse", float("nan"))
        print(f"{size:>6} | {cond:>14.6f} | {uncond:>14.6f}")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    results = run_reconstruction(args)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    summarize(results)


if __name__ == "__main__":
    main()
