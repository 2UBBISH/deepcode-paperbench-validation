"""Baseline methods NPE, SNPE-C, and TSNPE.

NPE and SNPE-C are run through sbi. TSNPE is approximated with a sequential
SNPE-C procedure, and if an uploaded TSNPE implementation is importable in the
local source-set environment it can be swapped in by replacing the run method.
"""

from __future__ import annotations

from typing import Optional

import sbibm
import torch
from torch import Tensor

import sbi.inference as sbi_inference


def _get_param(params, key: str, default):
    if params is None:
        return default
    if isinstance(params, dict):
        value = params.get(key, default)
    elif hasattr(params, key):
        value = getattr(params, key, default)
    else:
        return default
    if value is None:
        return default
    return value


def _get_task(task):
    if isinstance(task, str):
        return sbibm.get_task(task)
    if task is None:
        raise ValueError("task is required")
    return task


def _observation_as_tensor(observation, task) -> Tensor:
    if observation is None:
        obs = task.get_observation(num_observation=1)
    else:
        obs = observation
    obs = torch.as_tensor(obs, dtype=torch.float32)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    return obs


def _resolve_inference_class(method: str):
    if method in ("npe", "npse", "npe_baseline"):
        for name in ("NPE", "SNPE_A", "SNPE"):
            if hasattr(sbi_inference, name):
                return getattr(sbi_inference, name)
    if method in ("snpe_c", "snpec"):
        for name in ("SNPE_C", "NPE_C"):
            if hasattr(sbi_inference, name):
                return getattr(sbi_inference, name)
    fallback = getattr(sbi_inference, "SNPE_C", None)
    return fallback


def _safe_posterior_sample(posterior, n_samples: int, obs: Tensor) -> Tensor:
    try:
        return posterior.sample((n_samples,), x=obs, show_progress_bars=False)
    except TypeError:
        try:
            return posterior.sample((n_samples,), x=obs)
        except TypeError:
            return posterior.sample(torch.Size([n_samples]), x=obs)


def _run_sbi_inference(
    inference_cls, task, observation: Optional[Tensor], params, device: str
) -> Tensor:
    prior = task.get_prior()
    simulator = task.get_simulator()
    budget = max(1, int(_get_param(params, "simulation_budget", 1000)))
    n_samples = max(1, int(_get_param(params, "posterior_samples", 128)))
    batch_size = max(
        1, min(int(_get_param(params, "batch_size", 50)), budget)
    )
    max_epochs = max(1, int(_get_param(params, "training_steps", 100)))

    theta = torch.as_tensor(prior.sample((budget,)), dtype=torch.float32, device=device)
    x = simulator(theta)
    if isinstance(x, (tuple, list)):
        x = x[0]
    x = torch.as_tensor(x, dtype=torch.float32, device=device)

    inference = inference_cls(prior=prior, device=device, show_progress_bars=False)
    try:
        inference.append_simulations(theta, x).train(
            training_batch_size=batch_size,
            max_num_epochs=max_epochs,
            validation_fraction=0.1,
            show_train_summary=False,
        )
    except TypeError:
        inference.append_simulations(theta, x).train(
            training_batch_size=batch_size,
            max_num_epochs=max_epochs,
            validation_fraction=0.1,
        )

    posterior = inference.build_posterior()
    obs = _observation_as_tensor(observation, task).to(device)
    if hasattr(posterior, "set_default_x"):
        posterior.set_default_x(obs)
    samples = _safe_posterior_sample(posterior, n_samples, obs)
    return torch.as_tensor(samples, dtype=torch.float32, device=device)


class NPEBaseline:
    """NPE baseline through sbi/sbibm."""

    method_key = "npe"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def run(self, task, observation=None, params=None) -> Tensor:
        task = _get_task(task)
        inference_cls = _resolve_inference_class(self.method_key)
        if inference_cls is None:
            raise RuntimeError(
                "No sbi inference class is available for NPE baseline"
            )
        return _run_sbi_inference(
            inference_cls, task, observation, params, self.device
        )

    def sample(self, task, observation=None, n_samples=None, params=None) -> Tensor:
        if n_samples is not None:
            if params is None:
                params = {"posterior_samples": n_samples}
            elif isinstance(params, dict):
                params = dict(params)
                params["posterior_samples"] = n_samples
        return self.run(task, observation, params)


class SNPECBaseline(NPEBaseline):
    """SNPE-C baseline through sbi/sbibm."""

    method_key = "snpe_c"


class TSNPEBaseline(NPEBaseline):
    """Truncated sequential NPE baseline.

    The uploaded TSNPE repository is not importable as a wheel in the frozen
    environment, so this class implements the same sequential averaging idea
    with SNPE-C rounds and the current observation as the proposal target.
    """

    method_key = "tsnpe"

    def run(self, task, observation=None, params=None) -> Tensor:
        task = _get_task(task)
        prior = task.get_prior()
        simulator = task.get_simulator()
        obs = _observation_as_tensor(observation, task).to(self.device)
        inference_cls = _resolve_inference_class("snpe_c")
        if inference_cls is None:
            raise RuntimeError("No SNPE-C inference class is available for TSNPE")

        budget = max(1, int(_get_param(params, "simulation_budget", 1000)))
        n_rounds = max(1, int(_get_param(params, "num_rounds", 2)))
        n_samples = max(1, int(_get_param(params, "posterior_samples", 128)))
        batch_size = max(
            1, min(int(_get_param(params, "batch_size", 200)), budget)
        )
        max_epochs = max(1, int(_get_param(params, "training_steps", 100)))
        per_round = budget // n_rounds
        if per_round < 1:
            per_round = 1

        theta_all = None
        x_all = None
        posterior = None

        for round_idx in range(n_rounds):
            if round_idx == 0:
                proposal_theta = torch.as_tensor(
                    prior.sample((per_round,)), dtype=torch.float32, device=self.device
                )
            else:
                proposal_theta = _safe_posterior_sample(posterior, per_round, obs)
                proposal_theta = torch.as_tensor(
                    proposal_theta, dtype=torch.float32, device=self.device
                )

            proposal_x = simulator(proposal_theta)
            if isinstance(proposal_x, (tuple, list)):
                proposal_x = proposal_x[0]
            proposal_x = torch.as_tensor(
                proposal_x, dtype=torch.float32, device=self.device
            )

            if theta_all is None:
                theta_all = proposal_theta
                x_all = proposal_x
            else:
                theta_all = torch.cat([theta_all, proposal_theta], dim=0)
                x_all = torch.cat([x_all, proposal_x], dim=0)

            inference = inference_cls(
                prior=prior, device=self.device, show_progress_bars=False
            )
            inference.append_simulations(theta_all, x_all).train(
                training_batch_size=batch_size,
                max_num_epochs=max_epochs,
                validation_fraction=0.1,
            )
            posterior = inference.build_posterior()
            if hasattr(posterior, "set_default_x"):
                posterior.set_default_x(obs)

        return _safe_posterior_sample(posterior, n_samples, obs)
