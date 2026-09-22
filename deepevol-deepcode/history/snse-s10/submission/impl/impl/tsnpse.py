"""Truncated Sequential Neural Posterior Score Estimation methods.

TSNPSE runs R rounds. In each round it draws parameters from the current
truncated proposal (or the original prior in the first round), simulates data,
appends the new pairs to the accumulated dataset, retrains the score network
from scratch, and uses the trained posterior approximation to define the next
truncated proposal.
"""

from __future__ import annotations

import torch
from torch import Tensor

from impl.config import HPR_EPSILON, SDEConfig, TrialConfig
from impl.losses import denoising_posterior_score_matching_loss
from impl.npse import (
    _StandardizedScoreModel,
    _as_pair,
    _build_sde_for_kind,
    _dataset_statistics,
    _get_param,
    _sample_from_prior,
    _simulate,
    _standardize_dataset,
)
from impl.sampler import ProbabilityFlowODESampler
from impl.score_network import ScoreNetworkMLP
from impl.training import train_score_network
from impl.tsnpse_proposal import TruncatedProposalSampler


class Tsnpse:
    """Base TSNPSE implementation with configurable VE/VP dynamics."""

    sde_kind: str | None = None

    def __init__(
        self,
        prior,
        simulator,
        theta_dim: int | None = None,
        x_dim: int | None = None,
        setting: str = "",
        device: str = "cpu",
        sde_config: SDEConfig | None = None,
        task=None,
    ):
        if task is not None:
            prior = task.get_prior()
            simulator = task.get_simulator()
            theta_dim = getattr(task, "dim_parameters", theta_dim)
            x_dim = getattr(task, "dim_data", x_dim)
            setting = setting or getattr(task, "name", "")
        if prior is None or simulator is None:
            raise ValueError("prior and simulator are required")

        self.prior = prior
        self.simulator = simulator
        self.theta_dim = None if theta_dim is None else int(theta_dim)
        self.x_dim = None if x_dim is None else int(x_dim)
        self.setting = str(setting or "")
        self.device = str(device)
        self.sde_config = sde_config

        self.theta_tensors: list[Tensor] = []
        self.x_tensors: list[Tensor] = []
        self.proposal_samplers: list[TruncatedProposalSampler] = []
        self.model: _StandardizedScoreModel | None = None
        self.sde = None
        self.observation: Tensor | None = None

    @classmethod
    def from_task(cls, task, setting: str = "", device: str = "cpu", params=None):
        del params
        return cls(
            task.get_prior(),
            task.get_simulator(),
            theta_dim=getattr(task, "dim_parameters", None),
            x_dim=getattr(task, "dim_data", None),
            setting=setting or getattr(task, "name", ""),
            device=device,
            task=task,
        )

    def _ensure_dims(self, theta: Tensor, x: Tensor) -> None:
        if self.theta_dim is None:
            self.theta_dim = theta.shape[1] if theta.ndim > 1 else theta.numel()
        if self.x_dim is None:
            self.x_dim = x.shape[1] if x.ndim > 1 else x.numel()

    def _round_budget(self, r: int, n_rounds: int, params) -> int:
        if self.setting == "pyloric_network":
            if r == 0:
                return max(1, int(_get_param(params, "pyloric_initial_simulations", 30000)))
            return max(1, int(_get_param(params, "pyloric_added_per_round", 20000)))

        total = max(1, int(_get_param(params, "simulation_budget", 1000)))
        base = total // max(1, n_rounds)
        remainder = total % max(1, n_rounds)
        return base + (1 if r < remainder else 0)

    def _ensure_sde(self, theta: Tensor):
        if self.sde is None:
            self.sde = _build_sde_for_kind(
                self.sde_kind, self.setting, theta, self.sde_config
            )
        return self.sde

    def _train_on_data(self, theta: Tensor, x: Tensor, params):
        sde = self._ensure_sde(theta)
        theta_mean, theta_std, x_mean, x_std = _dataset_statistics(theta, x)
        theta_z, x_z = _standardize_dataset(
            theta, x, theta_mean, theta_std, x_mean, x_std
        )

        raw_model = ScoreNetworkMLP(
            theta_dim=self.theta_dim, x_dim=self.x_dim
        ).to(self.device)
        train_score_network(
            raw_model,
            (theta_z, x_z),
            params,
            sde=sde,
            loss_fn=denoising_posterior_score_matching_loss,
        )
        raw_model.eval()

        self.model = _StandardizedScoreModel(
            raw_model, theta_mean, theta_std, x_mean, x_std
        ).to(self.device)
        self.sde = sde
        return self.model

    def _sample_proposal(self, n: int, params=None) -> Tensor:
        n = max(1, int(n))
        if not self.proposal_samplers:
            return _sample_from_prior(self.prior, n, self.device)

        if len(self.proposal_samplers) == 1:
            return self.proposal_samplers[0].sample(n)

        indices = torch.randint(0, len(self.proposal_samplers), (n,))
        counts = torch.bincount(
            indices, minlength=len(self.proposal_samplers)
        ).tolist()
        parts = []
        for sampler, count in zip(self.proposal_samplers, counts):
            if count > 0:
                parts.append(sampler.sample(count))
        if not parts:
            return _sample_from_prior(self.prior, n, self.device)
        return torch.cat(parts, dim=0)[:n]

    def _sample_posterior(self, n: int) -> Tensor:
        if self.model is None or self.sde is None:
            raise RuntimeError("run_rounds must train a model before sampling")
        if self.observation is None:
            raise RuntimeError("observation is required")
        sampler = ProbabilityFlowODESampler(
            self.model, self.sde, self.observation
        )
        return sampler.sample(max(1, int(n)))

    def run_rounds(self, observation, params=None) -> Tensor:
        """Run the full sequential procedure and return posterior samples."""
        self.observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        n_rounds = max(1, int(_get_param(params, "num_rounds", 10)))

        self.theta_tensors = []
        self.x_tensors = []
        self.proposal_samplers = []
        self.model = None

        for r in range(n_rounds):
            budget = self._round_budget(r, n_rounds, params)
            theta_r = self._sample_proposal(budget, params)
            x_r = _simulate(self.simulator, theta_r)
            self._ensure_dims(theta_r, x_r)
            self.theta_tensors.append(theta_r.detach().cpu())
            self.x_tensors.append(x_r.detach().cpu())

            theta_all = torch.cat(self.theta_tensors, dim=0)
            x_all = torch.cat(self.x_tensors, dim=0)
            self._train_on_data(theta_all, x_all, params)

            if r < n_rounds - 1:
                hpr_samples = max(
                    1, int(_get_param(params, "hpr_samples", 256))
                )
                epsilon = float(_get_param(params, "hpr_epsilon", HPR_EPSILON))
                next_sampler = TruncatedProposalSampler(
                    self.prior,
                    self.model,
                    self.sde,
                    self.observation,
                    n_hpr_samples=hpr_samples,
                    epsilon=epsilon,
                )
                self.proposal_samplers.append(next_sampler)

        n_posterior = max(1, int(_get_param(params, "posterior_samples", 128)))
        self.posterior_samples = n_posterior
        return self._sample_posterior(n_posterior)

    def run(self, observation, params=None) -> Tensor:
        """Alias for :meth:`run_rounds`."""
        return self.run_rounds(observation, params)

    def sample(self, observation, n_samples: int | None = None, params=None) -> Tensor:
        """Draw samples after training, or from an already-trained model if run
        has previously completed."""
        if self.model is None or self.sde is None:
            return self.run_rounds(observation, params)
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        if n_samples is None:
            n_samples = int(_get_param(params, "posterior_samples", 128))
        sampler = ProbabilityFlowODESampler(self.model, self.sde, obs)
        return sampler.sample(int(n_samples))


class TsnpseVE(Tsnpse):
    """TSNPSE with the variance-exploding forward SDE."""

    sde_kind = "ve"


class TsnpseVP(Tsnpse):
    """TSNPSE with the variance-preserving forward SDE."""

    sde_kind = "vp"
