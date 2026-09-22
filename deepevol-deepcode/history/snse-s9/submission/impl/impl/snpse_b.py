"""SNPSE-B importance-weighted objective variant.

SNPSE-B follows the same sequential data-generation procedure as TSNPSE but
trains each score network with the denoising posterior score matching objective
weighted by the prior-to-proposal-prior density ratio. The proposal prior at
round r is the uniform mixture of all previously constructed truncated priors.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from impl.npse import (
    _StandardizedScoreModel,
    _dataset_statistics,
    _get_param,
    _sample_from_prior,
    _standardize_dataset,
)
from impl.score_network import ScoreNetworkMLP
from impl.tsnpse import Tsnpse


def _as_batch(theta: Tensor) -> Tensor:
    """Return theta as a batch tensor with shape (N, D)."""
    theta = torch.as_tensor(theta, dtype=torch.float32)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)
    return theta


def _sampler_inside(sampler, theta: Tensor) -> Tensor:
    """Return a boolean mask indicating membership in the sampler's cheap
    empirical hypercube pre-filter.
    """
    theta = _as_batch(theta)
    fn = getattr(sampler, "_inside_hypercube", None)
    if fn is not None:
        inside = fn(theta)
        if isinstance(inside, torch.Tensor):
            if inside.dtype != torch.bool:
                inside = inside > 0.5
            return inside

    posterior_samples = getattr(sampler, "_posterior_samples", None)
    if posterior_samples is not None and len(posterior_samples) > 0:
        posterior_samples = torch.as_tensor(
            posterior_samples, dtype=theta.dtype, device=theta.device
        )
        if posterior_samples.ndim == 1:
            posterior_samples = posterior_samples.unsqueeze(0)
        mins = posterior_samples.min(dim=0).values
        maxs = posterior_samples.max(dim=0).values
        return ((theta >= mins) & (theta <= maxs)).all(dim=1)

    return torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)


def _sampler_normalization(
    sampler, prior, device: str, normalization_samples: int
) -> float:
    """Approximate the normalising constant of a truncated prior by the
    prior probability of its cheap hypercube pre-filter.
    """
    cached = getattr(sampler, "_snpse_b_normalization", None)
    if cached is not None and float(cached) > 0.0:
        return float(cached)

    try:
        ref = _sample_from_prior(prior, max(16, normalization_samples), device)
    except Exception:
        return 1.0
    ref = _as_batch(ref)
    inside = _sampler_inside(sampler, ref).float()
    z = inside.mean().clamp_min(1e-3)
    try:
        sampler._snpse_b_normalization = float(z.item())
    except Exception:
        pass
    return float(z.item())


def _proposal_log_prob(
    theta: Tensor,
    prior,
    samplers,
    prior_log_prob: Tensor,
    normalization_samples: int,
) -> Tensor:
    """Log density of a uniform mixture of truncated prior samplers."""
    if not samplers:
        return prior_log_prob

    theta = _as_batch(theta)
    device = str(theta.device)
    components = []
    for sampler in samplers:
        inside = _sampler_inside(sampler, theta).float()
        z = _sampler_normalization(
            sampler, prior, device, normalization_samples
        )
        components.append(
            prior_log_prob + torch.log(inside.clamp_min(1e-12)) - math.log(z)
        )

    stacked = torch.stack(components, dim=0)
    return torch.logsumexp(stacked, dim=0) - math.log(len(samplers))


def _importance_weights(
    prior,
    theta_tensors,
    proposal_samplers,
    normalization_samples: int = 64,
) -> Tensor:
    """Compute prior-to-generating-proposal density ratios for all samples."""
    device = "cpu"
    all_weights = []

    for round_index, theta_round in enumerate(theta_tensors):
        theta = _as_batch(theta_round).to(device)
        try:
            prior_lp = prior.log_prob(theta)
        except Exception:
            prior_lp = torch.zeros(theta.shape[0], device=device)
        if prior_lp.ndim > 1:
            prior_lp = prior_lp.reshape(-1)

        proposal_lp = _proposal_log_prob(
            theta,
            prior,
            proposal_samplers[:round_index],
            prior_lp,
            normalization_samples,
        )
        log_weight = prior_lp - proposal_lp
        log_weight = torch.nan_to_num(
            log_weight, nan=-50.0, posinf=50.0, neginf=-50.0
        )
        log_weight = torch.clamp(log_weight, min=-30.0, max=30.0)
        weights = torch.exp(log_weight)
        weights = torch.where(
            torch.isfinite(weights), weights, torch.zeros_like(weights)
        )
        if weights.sum().item() <= 0.0:
            weights = torch.ones_like(weights)
        all_weights.append(weights)

    if not all_weights:
        return torch.ones(1, device=device)

    weights = torch.cat(all_weights, dim=0)
    if weights.sum().item() <= 0.0 or not torch.isfinite(weights).all():
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        if weights.sum().item() <= 0.0:
            weights = torch.ones_like(weights)
    return weights / weights.mean().clamp_min(1e-6)


def _weighted_score_matching_loss(
    score: Tensor,
    theta_t: Tensor,
    theta_0: Tensor,
    t: Tensor,
    weights: Tensor,
    sde,
) -> Tensor:
    target = sde.score_target(theta_t, theta_0, t)
    diff = score - target
    per_sample = 0.5 * (diff * diff).sum(dim=1)
    return (per_sample * weights).mean()


def _evaluate_weighted_loss(loader, model, sde, device):
    if loader is None:
        return float("inf")
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for theta_0, x, weights in loader:
            theta_0 = theta_0.to(device)
            x = x.to(device)
            weights = weights.to(device)
            n = theta_0.size(0)
            t = torch.rand(n, 1, device=device) * (sde.t_max - sde.t_min) + sde.t_min
            theta_t = sde.sample_transition(theta_0, t)
            score = model(theta_t, x, t)
            loss = _weighted_score_matching_loss(
                score, theta_t, theta_0, t, weights, sde
            )
            total += float(loss.item()) * n
            count += n
    model.train()
    return float("inf") if count == 0 else total / count


def _train_weighted_model(
    model,
    theta_z: Tensor,
    x_z: Tensor,
    weights: Tensor,
    sde,
    device: str,
    params,
):
    n_total = len(weights)
    batch_size = max(1, int(_get_param(params, "batch_size", 50)))
    validation_fraction = min(
        max(float(_get_param(params, "validation_fraction", 0.15)), 0.0), 0.5
    )
    patience = int(_get_param(params, "early_stopping_patience", 1000))
    max_iters = int(_get_param(params, "training_steps", 3000))
    learning_rate = float(_get_param(params, "learning_rate", 0.0001))
    seed = int(_get_param(params, "seed", 0))

    dataset = TensorDataset(theta_z, x_z, weights)
    if n_total >= 2:
        valid_size = max(1, int(n_total * validation_fraction))
        train_size = n_total - valid_size
        if train_size < 1:
            valid_size = 0
            train_size = n_total
        generator = torch.Generator().manual_seed(seed)
        train_ds, valid_ds = random_split(
            dataset, [train_size, valid_size], generator=generator
        ) if valid_size > 0 else (dataset, None)
    else:
        train_ds = dataset
        valid_ds = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = (
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        if valid_ds is not None
        else None
    )
    eval_loader = valid_loader if valid_loader is not None else train_loader

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    steps_no_improve = 0

    model.to(device)
    for _ in range(max_iters):
        model.train()
        for batch in train_loader:
            theta_0, x, batch_weights = batch
            theta_0 = theta_0.to(device)
            x = x.to(device)
            batch_weights = batch_weights.to(device)
            n = theta_0.size(0)
            t = torch.rand(n, 1, device=device) * (sde.t_max - sde.t_min) + sde.t_min
            theta_t = sde.sample_transition(theta_0, t)
            score = model(theta_t, x, t)
            loss = _weighted_score_matching_loss(
                score, theta_t, theta_0, t, batch_weights, sde
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_loss = _evaluate_weighted_loss(eval_loader, model, sde, device)
        if valid_loss < best_loss - 1e-12:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            steps_no_improve = 0
        else:
            steps_no_improve += 1
        if steps_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


class SNPSEB(Tsnpse):
    """Sequential NPSE variant with an importance-weighted score-matching
    objective.
    """

    sde_kind = None

    def _train_on_data(self, theta: Tensor, x: Tensor, params):
        theta = torch.as_tensor(theta, dtype=torch.float32)
        x = torch.as_tensor(x, dtype=torch.float32)
        self._ensure_dims(theta, x)
        sde = self._ensure_sde(theta)

        theta_mean, theta_std, x_mean, x_std = _dataset_statistics(theta, x)
        theta_z, x_z = _standardize_dataset(
            theta, x, theta_mean, theta_std, x_mean, x_std
        )

        normalization_samples = max(
            16, min(128, int(_get_param(params, "hpr_samples", 256)))
        )
        weights = _importance_weights(
            self.prior,
            self.theta_tensors,
            self.proposal_samplers,
            normalization_samples=normalization_samples,
        )
        weights = weights.to(theta_z.device)

        raw_model = ScoreNetworkMLP(
            theta_dim=self.theta_dim, x_dim=self.x_dim
        ).to(self.device)
        raw_model = _train_weighted_model(
            raw_model,
            theta_z,
            x_z,
            weights,
            sde,
            self.device,
            params,
        )

        self.model = _StandardizedScoreModel(
            raw_model, theta_mean, theta_std, x_mean, x_std
        ).to(self.device)
        self.sde = sde
        return self.model
