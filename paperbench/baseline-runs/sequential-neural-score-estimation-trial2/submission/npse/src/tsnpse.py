"""Truncated Sequential Neural Posterior Score Estimation (TSNPSE).

This module implements the sequential TSNPSE training procedure from the
paper:

* "Neural Posterior Score Estimation for Simulation-Based Inference"

TSNPSE repeatedly narrows the proposal prior by constructing a running
truncated-mixture proposal

    p_tilde^r(theta) \propto c^r(theta) p(theta),
    c^r(theta) = (1 / r) \sum_{s=0}^{r-1} 1{theta in Theta^s},

where ``Theta^0`` is the support of the original prior and, for ``s >= 1``,
``Theta^s`` is the highest-posterior-region (HPR) of the approximate posterior
learned in the previous round.  Each round simulates from the running proposal,
accumulates all previously simulated parameter/observation pairs, and retrains
a score network from scratch using denoising posterior score matching.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .hpr import (
    HPR_EPSILON_DEFAULT,
    HPR_N_SAMPLES_DEFAULT,
    HPRTruncation,
    PriorSupportRegion,
    TruncatedProposalSampler,
)
from .losses import npse_dsm_loss, sample_times
from .networks import ScoreNetwork, compute_standardization_stats
from .npse import build_sde, default_npse_config
from .sampling import sample_probability_flow
from .sde import SDE

__all__ = [
    "TSNPSETrainer",
    "default_tsnpse_config",
    "train_tsnpse",
]


def default_tsnpse_config(
    total_budget: int,
    rounds: int = 10,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the default configuration for sequential TSNPSE training.

    Parameters
    ----------
    total_budget:
        Total simulation budget across all rounds.
    rounds:
        Number of sequential rounds.
    overrides:
        Optional dictionary of configuration overrides.
    """
    base = default_npse_config(total_budget)

    config: Dict[str, Any] = {
        **base,
        "rounds": int(rounds),
        "per_round_budget": max(1, int(total_budget) // int(rounds)),
        # HPR truncation settings
        "epsilon": HPR_EPSILON_DEFAULT,
        "n_density_samples": HPR_N_SAMPLES_DEFAULT,
        "hutchinson_samples": 1,
        # Proposal rejection-sampling settings
        "proposal_batch_size": 512,
        "proposal_max_attempts_factor": 10000,
        # Probability-flow ODE settings used for density evaluation/sampling
        "rtol": 1e-5,
        "atol": 1e-5,
        "method": "rk45",
        # Whether to estimate an HPR region after the final round (unused but
        # kept for completeness/debugging).
        "estimate_final_region": False,
    }
    if overrides is not None:
        config.update(overrides)
    return config


class TSNPSETrainer:
    """Truncated sequential NPSE trainer.

    The trainer is conditioned on a single observed dataset ``x_obs`` and
    performs ``rounds`` sequential simulation/training rounds.  The proposal
    prior is the running truncated mixture described in the class docstring.
    """

    def __init__(
        self,
        benchmark: Any,
        sde: SDE,
        x_obs: Any,
        total_budget: int,
        rounds: int = 10,
        score_net: Optional[ScoreNetwork] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.benchmark = benchmark
        self.sde = sde
        self.total_budget = int(total_budget)
        self.rounds = int(rounds)

        self.device = getattr(benchmark, "device", torch.device("cpu"))
        self.dtype = getattr(benchmark, "dtype", torch.float32)

        self.config = default_tsnpse_config(
            total_budget=self.total_budget,
            rounds=self.rounds,
            overrides=config,
        )
        self.rounds = int(self.config.get("rounds", self.rounds))
        self.per_round_budget = int(self.config.get("per_round_budget", max(1, self.total_budget // self.rounds)))
        self.remainder_budget = self.total_budget - self.rounds * self.per_round_budget

        if seed is not None:
            self._set_seed(seed)

        # Test observation (shape (x_dim,)).
        self.x_obs = self._to_tensor(x_obs)
        if self.x_obs.dim() == 0:
            self.x_obs = self.x_obs.reshape(1)
        self.x_obs = self.x_obs.reshape(-1)

        # Score network: retrained from scratch each round on all accumulated
        # data. The passed-in network is only used to initialise dimensions.
        if score_net is not None:
            self._score_net_factory = lambda: copy.deepcopy(score_net)
        self.score_net: Optional[ScoreNetwork] = None

        # Running list of regions: Theta^0 (prior support), Theta^1, ...
        # At round r the proposal uses ``self.regions[:r]``.
        self.regions: List[Any] = [PriorSupportRegion()]

        # Accumulated parameter/observation pairs.
        self.dataset_theta: List[torch.Tensor] = []
        self.dataset_x: List[torch.Tensor] = []

        self.history: List[Dict[str, Any]] = []
        self.current_round: int = -1

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _score_fn(self, theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.score_net is None:
            raise RuntimeError("Score network has not been trained yet.")
        return self.score_net(theta, x, t)

    def _fit_standardization(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
    ):
        """Fit per-dimension standardization stats on perturbed parameters."""
        n_total = int(theta.shape[0])
        n = min(n_total, 2000)
        idx = torch.randperm(n_total, device=theta.device)[:n]
        t = torch.rand(n, device=theta.device, dtype=theta.dtype) * self.sde.T
        theta_t = self.sde.transition_sample(theta[idx], t)

        theta_mean, theta_std = compute_standardization_stats(theta_t)
        x_mean, x_std = compute_standardization_stats(x[idx])
        return theta_mean, theta_std, x_mean, x_std

    def _new_score_network(self) -> ScoreNetwork:
        if self.score_net is not None and hasattr(self, "_score_net_factory"):
            # Retraining "from scratch" is the default, but allow the caller to
            # provide a factory-like network by reinitialising its weights.
            net = ScoreNetwork(
                theta_dim=self.benchmark.theta_dim,
                x_dim=self.benchmark.x_dim,
                hidden_dim=int(self.config.get("hidden_dim", 256)),
                time_embedding_dim=int(self.config.get("time_embedding_dim", 64)),
            ).to(device=self.device, dtype=self.dtype)
        else:
            net = ScoreNetwork(
                theta_dim=self.benchmark.theta_dim,
                x_dim=self.benchmark.x_dim,
                hidden_dim=int(self.config.get("hidden_dim", 256)),
                time_embedding_dim=int(self.config.get("time_embedding_dim", 64)),
            ).to(device=self.device, dtype=self.dtype)
        return net

    # ------------------------------------------------------------------ #
    # Proposal sampling
    # ------------------------------------------------------------------ #
    def _sample_proposal(self, round_idx: int, n: int) -> torch.Tensor:
        """Sample ``n`` parameters from the proposal prior of ``round_idx``."""
        regions_for_round = self.regions[:round_idx]

        if round_idx == 0 or len(regions_for_round) == 0:
            theta = self.benchmark.prior_sample(n)
        else:
            sampler = TruncatedProposalSampler()
            for region in regions_for_round:
                sampler.add_round(region)
            theta = sampler.sample(
                n,
                self.benchmark.prior_sample,
                batch_size=int(self.config.get("proposal_batch_size", 512)),
                max_attempts_factor=int(self.config.get("proposal_max_attempts_factor", 10000)),
            )

        theta = self._to_tensor(theta)
        if theta.dim() == 1:
            theta = theta.reshape(1, -1)
        return theta

    def _simulate(self, round_idx: int, n: int) -> None:
        """Simulate parameters from the proposal and observations from the simulator."""
        theta = self._sample_proposal(round_idx, n)

        # Simulate in chunks to avoid excessive memory usage.
        x_parts: List[torch.Tensor] = []
        chunk_size = int(self.config.get("simulation_chunk_size", 4096))
        for start in range(0, theta.shape[0], chunk_size):
            theta_chunk = theta[start : start + chunk_size]
            x_chunk = self.benchmark.simulator(theta_chunk)
            x_chunk = self._to_tensor(x_chunk)
            if x_chunk.dim() == 1:
                x_chunk = x_chunk.reshape(1, -1)
            x_parts.append(x_chunk)
        x = torch.cat(x_parts, dim=0)

        self.dataset_theta.append(theta)
        self.dataset_x.append(x)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _evaluate_loss(
        self,
        net: ScoreNetwork,
        theta: torch.Tensor,
        x: torch.Tensor,
        batch_size: int,
    ) -> float:
        net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, int(theta.shape[0]), batch_size):
                theta_b = theta[start : start + batch_size]
                x_b = x[start : start + batch_size]
                t = sample_times(int(theta_b.shape[0]), self.sde.T, self.device, self.dtype)
                loss = npse_dsm_loss(
                    self.sde,
                    net,
                    theta_b,
                    x_b,
                    t,
                    weighting=self.config.get("weighting", "g2"),
                )
                total += float(loss.item()) * int(theta_b.shape[0])
                count += int(theta_b.shape[0])
        net.train()
        return total / max(1, count)

    def _train_round(self, round_idx: int) -> Dict[str, Any]:
        """Retrain the score network from scratch on all accumulated data."""
        theta_all = torch.cat(self.dataset_theta, dim=0)
        x_all = torch.cat(self.dataset_x, dim=0)

        net = self._new_score_network()

        theta_mean, theta_std, x_mean, x_std = self._fit_standardization(theta_all, x_all)
        net.set_standardization(theta_mean, theta_std, x_mean, x_std)
        self.score_net = net

        n_total = int(theta_all.shape[0])
        val_frac = float(self.config.get("val_frac", 0.15))
        n_val = max(1, int(n_total * val_frac))
        perm = torch.randperm(n_total, device=theta_all.device)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        lr = float(self.config.get("lr", 1e-4))
        batch_size = int(self.config.get("batch_size", 50))
        max_steps = int(self.config.get("max_steps", 3000))
        patience = int(self.config.get("patience", 1000))
        eval_interval = int(self.config.get("eval_interval", 25))

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        best_val = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        steps_since_improvement = 0
        step = 0

        while step < max_steps:
            if train_idx.numel() == 0:
                break

            # Sample a minibatch of parameter/observation pairs.
            batch_indices = train_idx[
                torch.randint(0, train_idx.numel(), (batch_size,), device=theta_all.device)
            ]
            theta_b = theta_all[batch_indices]
            x_b = x_all[batch_indices]

            t = sample_times(int(theta_b.shape[0]), self.sde.T, self.device, self.dtype)
            loss = npse_dsm_loss(
                self.sde,
                net,
                theta_b,
                x_b,
                t,
                weighting=self.config.get("weighting", "g2"),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(self.config.get("grad_clip", 100.0)))
            optimizer.step()

            step += 1
            steps_since_improvement += 1

            if step % eval_interval == 0:
                val_loss = self._evaluate_loss(
                    net,
                    theta_all[val_idx],
                    x_all[val_idx],
                    batch_size,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(net.state_dict())
                    steps_since_improvement = 0

                if steps_since_improvement >= patience:
                    break

        net.load_state_dict(best_state)
        self.score_net = net

        return {
            "round": round_idx,
            "final_train_loss": float(loss.item()),
            "best_val_loss": best_val,
            "steps": step,
            "n_train": int(theta_all.shape[0]),
        }

    # ------------------------------------------------------------------ #
    # HPR region estimation
    # ------------------------------------------------------------------ #
    def _estimate_region(self):
        """Estimate an HPR region for the current approximate posterior."""
        hpr = HPRTruncation(
            sde=self.sde,
            score_fn=self._score_fn,
            epsilon=float(self.config.get("epsilon", HPR_EPSILON_DEFAULT)),
            n_density_samples=int(self.config.get("n_density_samples", HPR_N_SAMPLES_DEFAULT)),
            theta_dim=self.benchmark.theta_dim,
            rtol=float(self.config.get("rtol", 1e-5)),
            atol=float(self.config.get("atol", 1e-5)),
            hutchinson_samples=int(self.config.get("hutchinson_samples", 1)),
            device=self.device,
            dtype=self.dtype,
        )
        return hpr.build_region(self.x_obs)

    # ------------------------------------------------------------------ #
    # Main loop and posterior sampling
    # ------------------------------------------------------------------ #
    def run(self) -> "TSNPSETrainer":
        """Run all sequential rounds."""
        estimate_final = bool(self.config.get("estimate_final_region", False))

        for r in range(self.rounds):
            # Distribute any remainder budget over the first few rounds.
            n = self.per_round_budget + (1 if r < self.remainder_budget else 0)
            if n <= 0:
                n = 1

            self._simulate(r, n)
            info = self._train_round(r)

            if r < self.rounds - 1 or estimate_final:
                region = self._estimate_region()
                self.regions.append(region)
                info["region_added"] = True
            else:
                info["region_added"] = False

            info["dataset_size"] = sum(int(t.shape[0]) for t in self.dataset_theta)
            info["num_regions"] = len(self.regions)
            self.history.append(info)
            self.current_round = r

        return self

    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        """Sample the final approximate posterior via probability-flow ODE."""
        if self.score_net is None:
            raise RuntimeError("No trained score network available. Call run() first.")

        return sample_probability_flow(
            self.sde,
            self._score_fn,
            n_samples,
            self.x_obs,
            theta_dim=self.benchmark.theta_dim,
            device=self.device,
            dtype=self.dtype,
            rtol=float(self.config.get("rtol", 1e-5)),
            atol=float(self.config.get("atol", 1e-5)),
            method=str(self.config.get("method", "rk45")),
        )

    def final_score_fn(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the trained posterior score function."""
        return self._score_fn


def train_tsnpse(
    benchmark: Any,
    x_obs: Any,
    total_budget: int,
    rounds: int = 10,
    sde: Optional[SDE] = None,
    sde_type: str = "ve",
    score_net: Optional[ScoreNetwork] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    verbose: bool = True,
) -> TSNPSETrainer:
    """Convenience entry point for TSNPSE training.

    Builds an SDE (if not supplied), runs the sequential training loop, and
    returns a trainer that can sample the final approximate posterior.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if sde is None:
        n_calib = 2000
        prior_samples = benchmark.prior_sample(n_calib)
        prior_samples = torch.as_tensor(
            prior_samples,
            device=getattr(benchmark, "device", torch.device("cpu")),
            dtype=getattr(benchmark, "dtype", torch.float32),
        )
        sde = build_sde(
            sde_type,
            prior_samples,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            T=1.0,
        )

    trainer = TSNPSETrainer(
        benchmark=benchmark,
        sde=sde,
        x_obs=x_obs,
        total_budget=total_budget,
        rounds=rounds,
        score_net=score_net,
        config=config,
        seed=seed,
    )

    if verbose:
        print(
            f"[TSNPSE] budget={total_budget}, rounds={trainer.rounds}, "
            f"per_round={trainer.per_round_budget}"
        )

    trainer.run()

    if verbose:
        print("[TSNPSE] finished sequential training.")

    return trainer
