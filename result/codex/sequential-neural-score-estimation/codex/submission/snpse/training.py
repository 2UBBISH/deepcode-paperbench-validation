"""Denoising score matching objective and training loop (Eqs. 7, 11, 15).

The conditional denoising posterior score matching objective (Eq. 7) is

    J(psi) = 1/2 int_0^T lambda_t E_{p_{t|0}(theta_t|theta_0) p(x|theta_0) p(theta_0)}
             [ || s_psi(theta_t, x, t) - grad_{theta_t} log p_{t|0}(theta_t|theta_0) ||^2 ] dt

Since ``p_{t|0}`` is Gaussian with a closed-form mean/std (``snpse.sde``), the
target ``grad log p_{t|0}`` is available in closed form and the objective is a
simple Monte Carlo average.

Weighting function.  The paper leaves ``lambda_t`` unspecified beyond requiring
it to be positive.  We use the standard choice ``lambda_t = sigma_t^2`` (the
conditional variance of ``theta_t`` given ``theta_0``), which makes the
objective equivalent to the usual noise-prediction loss of score-based
generative models and keeps the loss scale comparable across noise levels.  The
choice is exposed via ``loss_weight`` so that the alternative ``lambda_t = 1``
(used, e.g., for the "likelihood weighting" ablation) can be run.
"""

from __future__ import annotations

import copy
import time
from typing import Optional, Tuple

import torch

from .sde import SDE


def forward_diffuse(z0: torch.Tensor, t: torch.Tensor, sde: SDE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``z_t ~ p_{t|0}(. | z_0)``; returns (z_t, mean, std)."""
    mean, std = sde.transition(z0, t)
    eps = torch.randn_like(z0)
    zt = mean + std * eps
    return zt, mean, std


def dsm_loss(
    score_network: torch.nn.Module,
    sde: SDE,
    z0: torch.Tensor,
    x: torch.Tensor,
    loss_weight: str = "sigma^2",
    sample_weights: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Monte Carlo estimate of the denoising score matching objective.

    Args:
        z0: (B, d) standardized parameter samples from the (proposal) prior.
        x: (B, p) corresponding simulated observations.
        loss_weight: ``"sigma^2"`` (default) or ``"none"``.
        sample_weights: (B,) optional importance weights, used by SNPSE-B
            (Eq. 15), which multiplies the per-sample loss by
            ``p(theta_0) / ptilde^r(theta_0)``.
    """
    batch = z0.shape[0]
    t = torch.rand(batch, device=z0.device, dtype=z0.dtype, generator=generator)
    zt, mean, std = forward_diffuse(z0, t, sde)
    target = -(zt - mean) / (std ** 2)
    pred = score_network(zt, x, t)
    per_sample = ((pred - target) ** 2).sum(-1)

    if loss_weight == "sigma^2":
        per_sample = per_sample * (std ** 2).reshape(batch)
    elif loss_weight == "none":
        pass
    else:
        raise ValueError(f"unknown loss_weight {loss_weight}")

    if sample_weights is not None:
        w = sample_weights.reshape(-1).to(per_sample.dtype)
        return (w * per_sample).sum() / w.sum().clamp_min(1e-12)
    return per_sample.mean()


@torch.no_grad()
def _validation_loss(
    score_network: torch.nn.Module,
    sde: SDE,
    z_val: torch.Tensor,
    x_val: torch.Tensor,
    loss_weight: str,
    w_val: Optional[torch.Tensor],
    seed: int = 0,
) -> float:
    gen = torch.Generator(device=z_val.device).manual_seed(seed)
    # a single fixed Monte Carlo draw makes the validation curve comparable
    batch = z_val.shape[0]
    t = torch.rand(batch, device=z_val.device, dtype=z_val.dtype, generator=gen)
    zt, mean, std = forward_diffuse(z_val, t, sde)
    target = -(zt - mean) / (std ** 2)
    pred = score_network(zt, x_val, t)
    per_sample = ((pred - target) ** 2).sum(-1)
    if loss_weight == "sigma^2":
        per_sample = per_sample * (std ** 2).reshape(batch)
    if w_val is not None:
        w = w_val.reshape(-1).to(per_sample.dtype)
        loss = (w * per_sample).sum() / w.sum().clamp_min(1e-12)
    else:
        loss = per_sample.mean()
    return float(loss.item())


def train_score_network(
    score_network: torch.nn.Module,
    sde: SDE,
    z: torch.Tensor,
    x: torch.Tensor,
    lr: float = 1e-4,
    batch_size: int = 50,
    max_iters: int = 3000,
    val_fraction: float = 0.15,
    patience: int = 1000,
    loss_weight: str = "sigma^2",
    sample_weights: Optional[torch.Tensor] = None,
    seed: int = 0,
    verbose: bool = False,
    time_budget_seconds: Optional[float] = None,
) -> Tuple[torch.nn.Module, dict]:
    """Train the score network with Adam and early stopping.

    Follows Appendix E.3.2: Adam with learning rate ``1e-4``, a held-out
    validation set of 15% of the data, early stopping after ``patience``
    non-improving steps (1000 by default) and a maximum of ``max_iters`` (3000
    by default) training iterations.  The network with the lowest validation
    loss is returned.
    """
    torch.manual_seed(seed)
    n = z.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(round(val_fraction * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    z_tr, x_tr = z[train_idx], x[train_idx]
    z_val, x_val = z[val_idx], x[val_idx]
    w_tr = sample_weights[train_idx] if sample_weights is not None else None
    w_val = sample_weights[val_idx] if sample_weights is not None else None

    optimizer = torch.optim.Adam(score_network.parameters(), lr=lr)
    gen = torch.Generator(device=z.device).manual_seed(seed + 1)

    best_state = copy.deepcopy(score_network.state_dict())
    best_val = float("inf")
    best_iter = 0
    history = []
    started = time.time()

    effective_batch = min(batch_size, z_tr.shape[0])
    for it in range(1, max_iters + 1):
        idx = torch.randint(0, z_tr.shape[0], (effective_batch,), generator=torch.Generator().manual_seed(seed + it))
        z_b, x_b = z_tr[idx], x_tr[idx]
        w_b = w_tr[idx] if w_tr is not None else None
        score_network.train()
        loss = dsm_loss(score_network, sde, z_b, x_b, loss_weight, w_b, generator=gen)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        val = _validation_loss(score_network, sde, z_val, x_val, loss_weight, w_val, seed=seed + 12345)
        if val < best_val - 1e-12:
            best_val = val
            best_iter = it
            best_state = copy.deepcopy(score_network.state_dict())
        history.append((it, float(loss.item()), val))
        if verbose and (it == 1 or it % 250 == 0):
            print(f"  iter {it:5d}  train {float(loss.item()):.5f}  val {val:.5f}")

        if time_budget_seconds is not None and (time.time() - started) > time_budget_seconds:
            break
        if it - best_iter >= patience:
            break

    score_network.load_state_dict(best_state)
    info = {
        "best_val_loss": best_val,
        "best_iter": best_iter,
        "num_iters": len(history),
        "num_train": int(z_tr.shape[0]),
        "num_val": int(z_val.shape[0]),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "history": history,
    }
    return score_network, info
