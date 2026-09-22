"""Truncated Sequential Neural Posterior Score Estimation (TSNPSE).

This is Algorithm 1 of the paper (Section 3.1).  Given a budget of ``N``
simulations spread evenly over ``R`` rounds (``M = N / R`` per round):

1. In round 1 the proposals are drawn from the prior and the score network is
   trained with the standard NPSE objective (Eq. 7).
2. At the end of each round the highest-probability region of the current
   approximate posterior is used to define a truncated version ``bar p^r`` of
   the prior (Eq. 9), and the truncation boundary ``kappa`` is stored.
3. In round ``r > 1`` new parameters are drawn from ``bar p^{r-1}`` by rejection
   sampling and the score network is re-trained on the whole accumulated
   dataset, which is a Monte Carlo sample from the mixture
   ``tilde p^r = 1/r sum_{s<r} bar p^s`` used in the loss (Eq. 11).

Because the proposal is proportional to the prior on the support of the
posterior (Proposition 3.1), no correction for the proposal is needed: the
minimiser of the TSNPSE loss still equals the score of the true posterior at
``x_obs``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from .diffusion import DiffusionConfig
from .normalization import StandardizedDistribution, Standardizer
from .npse import NPSE, TrainingConfig
from .proposals import Proposal, TruncatedProposal, estimate_truncation_boundary
from .sde import compute_sigma_max


@dataclass
class TSNPSEConfig:
    num_rounds: int = 10
    num_simulations: int = 1000
    #: Optional non-uniform simulation schedule.  If set, round 1 uses
    #: ``num_simulations_initial`` samples and every later round uses
    #: ``num_simulations`` (which should then be interpreted as the
    #: per-round budget).  The pyloric experiment (Section 5.3) uses 30000
    #: initial simulations and 20000 additional simulations per round.
    num_simulations_initial: Optional[int] = None
    hpr_num_samples: int = 20000
    hpr_eps: float = 5e-4
    #: How to draw from the truncated proposal: "rejection" (Appendix E.3.3) or
    #: "sir" (the alternative mentioned at the end of Appendix E.3.3, which is
    #: preferable when the HPR is a very small fraction of the prior).
    proposal_sampling: str = "rejection"
    retrain_from_scratch: bool = True
    verbose: bool = True
    # Optional wall-clock safety valve (not used by the paper, useful for smoke
    # tests that must finish quickly on CPU).
    max_seconds: Optional[float] = None

    def simulations_for_round(self, r: int) -> int:
        if self.num_simulations_initial is None:
            M = self.num_simulations // self.num_rounds
            if M * self.num_rounds != self.num_simulations:
                raise ValueError("num_simulations must be divisible by num_rounds")
            return M
        return self.num_simulations_initial if r == 1 else self.num_simulations


class TSNPSE:
    """Sequential NPSE with truncated proposals."""

    def __init__(
        self,
        dim_parameters: int,
        dim_data: int,
        prior,
        simulator: Callable[[torch.Tensor], torch.Tensor],
        sde: str = "ve",
        sde_kwargs: Optional[dict] = None,
        sigma_min: float = 0.05,
        config: Optional[TSNPSEConfig] = None,
        diffusion_config: Optional[DiffusionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        device: str = "cpu",
        parameterisation: str = "score",
        seed: int = 0,
    ) -> None:
        self.dim_parameters = dim_parameters
        self.dim_data = dim_data
        self.prior = prior
        self.simulator = simulator
        self.sde_name = sde
        self.sde_kwargs = dict(sde_kwargs or {})
        self.sigma_min = sigma_min
        self.config = config or TSNPSEConfig()
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device
        self.parameterisation = parameterisation
        self.seed = seed

        self.theta_scaler: Optional[Standardizer] = None
        self.x_scaler: Optional[Standardizer] = None
        self.sigma_max: Optional[float] = None
        self.npse: Optional[NPSE] = None
        self.boundaries: List = []
        self.diagnostics: List[dict] = []

    # ------------------------------------------------------------------ utils
    def _simulate(self, theta: torch.Tensor) -> torch.Tensor:
        x = self.simulator(theta)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=theta.dtype)
        return x.reshape(theta.shape[0], -1).to(self.device)

    def _new_npse(self) -> NPSE:
        return NPSE(
            dim_parameters=self.dim_parameters,
            dim_data=self.dim_data,
            prior=self.prior,
            sde=self.sde_name,
            sde_kwargs=dict(self.sde_kwargs),
            sigma_min=self.sigma_min,
            diffusion_config=self.diffusion_config,
            training_config=self.training_config,
            device=self.device,
            parameterisation=self.parameterisation,
        )

    # -------------------------------------------------------------------- run
    def run(self, x_obs: torch.Tensor, verbose: Optional[bool] = None) -> NPSE:
        """Run the TSNPSE algorithm and return the final approximate posterior."""
        cfg = self.config
        verbose = cfg.verbose if verbose is None else verbose
        torch.manual_seed(self.seed)

        x_obs = x_obs.reshape(1, -1).to(self.device)
        if verbose:
            print(f"[TSNPSE] {cfg.num_rounds} rounds, budget {cfg.num_simulations}")

        theta_all: List[torch.Tensor] = []
        x_all: List[torch.Tensor] = []
        components: List[Proposal] = []
        started = time.time()

        for r in range(1, cfg.num_rounds + 1):
            round_started = time.time()
            M = cfg.simulations_for_round(r)

            # ---- (i) draw parameters from the round's proposal ---------------
            if r == 1:
                theta_r = self.prior.sample((M,)).to(self.device)
                if self.theta_scaler is None:
                    self.theta_scaler = Standardizer.fit(theta_r)
                z_r = self.theta_scaler.transform(theta_r)
            else:
                assert self.theta_scaler is not None
                z_r = components[r - 2].sample(M).to(self.device)
                theta_r = self.theta_scaler.inverse(z_r)
                # the acceptance rate of the truncated proposal is only known
                # once it has been sampled from (Appendix E.3.3)
                self.diagnostics[-1]["proposal_acceptance_rate"] = components[r - 2].boundary.acceptance_rate

            # ---- (ii) simulate ----------------------------------------------
            x_r = self._simulate(theta_r)
            if self.x_scaler is None:
                self.x_scaler = Standardizer.fit(x_r)
            if r == 1:
                # the VE SDE's sigma_max is fixed using only the first-round
                # data (see the addendum to the reproduction task)
                if self.sde_name.lower() in ("ve", "vesde") and "sigma_max" not in self.sde_kwargs:
                    self.sigma_max = compute_sigma_max(z_r, technique=1)

            theta_all.append(theta_r)
            x_all.append(x_r.detach().cpu())

            # ---- (iii) train the score network on the accumulated dataset ----
            theta_cat = torch.cat(theta_all, dim=0)
            x_cat = torch.cat(x_all, dim=0).to(self.device)
            npse = self._new_npse()
            npse.fit(
                theta_cat,
                x_cat,
                theta_scaler=self.theta_scaler,
                x_scaler=self.x_scaler,
                sigma_max=self.sigma_max,
                verbose=False,
            )
            self.npse = npse

            info = {
                "round": r,
                "num_simulations_total": int(theta_cat.shape[0]),
                "best_val_loss": npse.train_info.get("best_val_loss"),
                "best_iter": npse.train_info.get("best_iter"),
                "sigma_max": self.sigma_max,
                "seconds": time.time() - round_started,
            }

            # ---- (iv) update the truncated proposal --------------------------
            if r < cfg.num_rounds:
                boundary = estimate_truncation_boundary(
                    npse.posterior,
                    npse._standardize_x(x_obs),
                    num_samples=cfg.hpr_num_samples,
                    eps=cfg.hpr_eps,
                    round_index=r,
                    seed=self.seed + r,
                )
                self.boundaries.append(boundary)
                comp = TruncatedProposal(
                    self.prior_z,
                    boundary,
                    lambda z, post=npse.posterior: post.log_prob(npse._standardize_x(x_obs), z),
                    sampling_method=cfg.proposal_sampling,
                )
                components.append(comp)
                info["kappa"] = boundary.kappa

            self.diagnostics.append(info)
            if verbose:
                msg = (
                    f"  round {r:2d} | sims {info['num_simulations_total']:6d} "
                    f"| val {info['best_val_loss']:.4f} | {info['seconds']:.1f}s"
                )
                if "kappa" in info:
                    msg += f" | kappa {info['kappa']:.2f}"
                if "proposal_acceptance_rate" in info:
                    msg += f" | acc {info['proposal_acceptance_rate']:.4f}"
                print(msg)

            if cfg.max_seconds is not None and (time.time() - started) > cfg.max_seconds:
                if verbose:
                    print("  [TSNPSE] max_seconds reached; stopping early")
                break

        assert self.npse is not None
        return self.npse

    # ------------------------------------------------------------------ prior
    @property
    def prior_z(self) -> StandardizedDistribution:
        """The prior in standardized coordinates (needs ``theta_scaler``)."""
        assert self.theta_scaler is not None
        return StandardizedDistribution(self.prior, self.theta_scaler)
