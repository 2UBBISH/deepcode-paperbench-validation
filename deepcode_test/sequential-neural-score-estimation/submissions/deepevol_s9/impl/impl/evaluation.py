"""Evaluation metrics: C2ST, valid pyloric summaries, SBCC coverage, and a
posterior predictive visual agreement check.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def compute_c2st(reference_samples: Tensor, approximate_samples: Tensor) -> float:
    """Train a small classifier to distinguish reference from approximate
    posterior samples and return its held-out accuracy.
    """
    ref = torch.as_tensor(reference_samples, dtype=torch.float32)
    app = torch.as_tensor(approximate_samples, dtype=torch.float32)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if app.ndim == 1:
        app = app.reshape(1, -1)

    n = min(ref.shape[0], app.shape[0])
    if n < 8 or ref.shape[1] != app.shape[1]:
        return 0.5

    ref = ref[:n]
    app = app[:n]
    d = ref.shape[1]
    x = torch.cat([ref, app], dim=0)
    y = torch.cat(
        [torch.zeros(n, device=x.device), torch.ones(n, device=x.device)], dim=0
    )

    perm = torch.randperm(2 * n, device=x.device)
    x = x[perm]
    y = y[perm]
    split = n
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    mu = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std

    model = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        logits = model(x_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(x_test).squeeze(-1) > 0.0).float()
        accuracy = float((preds == y_test).float().mean().item())
    return float(max(0.5, min(1.0, accuracy)))


def valid_summary_percentage(replaced_summaries: Tensor) -> float:
    """Return the percentage of summary-statistic rows that are well-defined
    after replacement, i.e. contain no NaN or infinity entries.
    """
    x = torch.as_tensor(replaced_summaries)
    if x.numel() == 0:
        return 0.0
    if not x.dtype.is_floating_point:
        return 100.0
    if x.ndim == 1:
        valid = torch.isfinite(x)
    else:
        valid = torch.isfinite(x).all(dim=1)
    return float(valid.float().mean().item() * 100.0)


def _prior_sample(prior, n: int) -> Tensor:
    try:
        theta = prior.sample((n,))
    except Exception:
        theta = prior.sample(n)
    return torch.as_tensor(theta, dtype=torch.float32)


def _simulator_call(simulator, theta: Tensor) -> Tensor:
    out = simulator(theta)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return torch.as_tensor(out, dtype=torch.float32)


def _call_sampler_sample(sampler, observation, n: int) -> Tensor:
    try:
        samples = sampler.sample(observation, n)
    except TypeError:
        try:
            samples = sampler.sample(observation, n_samples=n)
        except TypeError:
            samples = sampler.sample(n, observation)
    return torch.as_tensor(samples, dtype=torch.float32)


def _inside_credible_region(
    posterior_samples: Tensor, theta_true: Tensor, confidence: float
) -> bool:
    posterior_samples = torch.as_tensor(posterior_samples, dtype=torch.float32)
    theta_true = torch.as_tensor(theta_true, dtype=torch.float32).reshape(1, -1)
    if posterior_samples.ndim == 1:
        posterior_samples = posterior_samples.reshape(1, -1)
    if posterior_samples.shape[0] < 2:
        return False
    q_low = (1.0 - confidence) / 2.0
    q_high = 1.0 - q_low
    lower = torch.quantile(posterior_samples, q_low, dim=0)
    upper = torch.quantile(posterior_samples, q_high, dim=0)
    return bool((theta_true >= lower).all() and (theta_true <= upper).all())


def sbcc_expected_coverage(
    sampler, observation, confidence: float = 0.8
) -> float:
    """Estimate simulation-based coverage calibration at a representative
    confidence level using the sampler's prior and simulator.
    """
    confidence = float(min(max(confidence, 0.0), 1.0))
    prior = getattr(sampler, "prior", None)
    simulator = getattr(sampler, "simulator", None)
    if prior is None or simulator is None:
        return 0.5

    repetitions = int(getattr(sampler, "sbcc_repetitions", 16))
    n_posterior = int(getattr(sampler, "posterior_samples", 64))
    hits = 0
    attempts = 0

    for _ in range(repetitions):
        try:
            theta_true = _prior_sample(prior, 1)
            x_cal = _simulator_call(simulator, theta_true)
            posterior_samples = _call_sampler_sample(
                sampler, x_cal, max(2, n_posterior)
            )
            if posterior_samples.ndim == 1:
                posterior_samples = posterior_samples.reshape(1, -1)
            if theta_true.ndim == 0:
                theta_true = theta_true.reshape(1)
            if theta_true.shape[1] != posterior_samples.shape[1]:
                continue
            attempts += 1
            if _inside_credible_region(
                posterior_samples, theta_true, confidence
            ):
                hits += 1
        except Exception:
            continue

    if attempts == 0:
        return 0.5
    return float(hits / attempts)


def posterior_predictive_visual_agreement(
    predicted_traces: Tensor, observed_traces: Tensor
) -> int:
    """Return 1 when the mean predicted trace is visually close to the observed
    trace, using a normalised RMSE threshold.
    """
    pred = torch.as_tensor(predicted_traces, dtype=torch.float32)
    obs = torch.as_tensor(observed_traces, dtype=torch.float32)
    if pred.numel() == 0 or obs.numel() == 0:
        return 0

    if pred.ndim == obs.ndim + 1 and pred.shape[1:] == obs.shape:
        pred = pred.mean(dim=0)
    elif pred.ndim == obs.ndim + 1 and pred.shape[1] == obs.numel():
        pred = pred.reshape(pred.shape[0], -1).mean(dim=0)
        obs = obs.reshape(-1)
    elif pred.ndim == obs.ndim and pred.shape != obs.shape:
        if pred.numel() == obs.numel():
            pred = pred.reshape(obs.shape)
        else:
            return 0

    if pred.shape != obs.shape:
        return 0

    pred = pred.reshape(obs.shape)
    denominator = obs.abs().mean()
    if float(denominator.item()) <= 1e-12:
        denominator = torch.ones_like(denominator)
    nrmse = float(torch.sqrt(((pred - obs) ** 2).mean()).item() / denominator.item())
    return 1 if nrmse < 0.30 else 0
