"""Alternative sequential approaches: SNPSE-A, SNPSE-B and SNPSE-C.

These are the three additional sequential schemes introduced in Section 3.2 of
the paper and described in detail in Appendix C.  Unlike TSNPSE, all three
define the proposal directly in terms of the most recent approximation of the
posterior, ``ptilde^r(theta) = 1/r sum_{s=0}^{r-1} p_psi^s(theta | x_obs)``
(with ``p_psi^0 = p``), and therefore *do* require a correction for the
mismatch between the proposal posterior and the true posterior:

* SNPSE-A (Appendix C.2) performs a post-hoc sampling-importance-resampling
  correction with importance weights ``h_i = p(theta_i) / ptilde^r(theta_i)``.
* SNPSE-B (Appendix C.3) folds the same importance weights into the denoising
  score matching objective (Eq. 15 / Eq. 99).
* SNPSE-C (Appendix C.4) performs the correction directly in score space,
  ``stilde = s_psi + grad log ptilde_t^r(theta_t) - grad log p_t(theta_t)``.

Note on the proposal prior.  As discussed in Appendix C.2.3, the exact
proposal prior ``ptilde^r`` involves the (SIR-corrected) posterior estimates
``p_psi^s`` for ``s >= 2``, which are only known up to an intractable
normalising constant.  Following the "Approximating the Proposal Prior"
strategy of Appendix C.2.3, we approximate each mixture component
``p_psi^s(theta | x_obs)`` (``s >= 1``) by the learned proposal posterior of
round ``s``, whose density is available via the instantaneous
change-of-variables formula (Eq. 5).  For ``r = 1`` and ``r = 2`` this
approximation is exact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from .diffusion import DiffusionConfig
from .normalization import StandardizedDistribution, Standardizer
from .npse import NPSE, TrainingConfig
from .prior_score import (
    GaussianVEPriorScore,
    PerturbedPriorScore,
    UniformVEPriorScore,
    estimate_prior_score,
)
from .sde import compute_sigma_max


def _base_distribution(prior):
    """Unwrap ``torch.distributions.Independent`` (and similar) to reach the
    elementwise distribution, which is where ``low``/``high``/``mean`` live."""
    dist = prior
    for _ in range(4):
        base = getattr(dist, "base_dist", None)
        if base is None:
            break
        dist = base
    return dist


@dataclass
class VariantConfig:
    num_rounds: int = 10
    num_simulations: int = 1000
    num_candidates: int = 2  # M' = num_candidates * M for SIR in SNPSE-A
    resample_with_replacement: bool = False
    verbose: bool = True
    max_seconds: Optional[float] = None


class _SequentialBase:
    """Shared machinery for the three alternative sequential schemes."""

    def __init__(
        self,
        dim_parameters: int,
        dim_data: int,
        prior,
        simulator: Callable[[torch.Tensor], torch.Tensor],
        sde: str = "ve",
        sde_kwargs: Optional[dict] = None,
        sigma_min: float = 0.05,
        config: Optional[VariantConfig] = None,
        diffusion_config: Optional[DiffusionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.dim_parameters = dim_parameters
        self.dim_data = dim_data
        self.prior = prior
        self.simulator = simulator
        self.sde_name = sde
        self.sde_kwargs = dict(sde_kwargs or {})
        self.sigma_min = sigma_min
        self.config = config or VariantConfig()
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device
        self.seed = seed

        self.theta_scaler: Optional[Standardizer] = None
        self.x_scaler: Optional[Standardizer] = None
        self.sigma_max: Optional[float] = None
        self.sde = None
        self.npse: Optional[NPSE] = None
        self.posteriors: List[NPSE] = []
        self.theta_rounds: List[torch.Tensor] = []
        self.x_rounds: List[torch.Tensor] = []
        self.diagnostics: List[dict] = []
        self._x_obs: Optional[torch.Tensor] = None

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
        )

    def _init_round_one(self, M: int) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.prior.sample((M,)).to(self.device)
        self.theta_scaler = Standardizer.fit(theta)
        z = self.theta_scaler.transform(theta)
        x = self._simulate(theta)
        self.x_scaler = Standardizer.fit(x)
        if self.sde_name.lower() in ("ve", "vesde") and "sigma_max" not in self.sde_kwargs:
            self.sigma_max = compute_sigma_max(z, technique=1)
        return theta, x

    def _train_on_dataset(self, sample_weights: Optional[torch.Tensor] = None) -> NPSE:
        theta = torch.cat(self.theta_rounds, dim=0)
        x = torch.cat(self.x_rounds, dim=0).to(self.device)
        npse = self._new_npse()
        npse.fit(
            theta,
            x,
            theta_scaler=self.theta_scaler,
            x_scaler=self.x_scaler,
            sigma_max=self.sigma_max,
            sample_weights=sample_weights,
            verbose=False,
        )
        return npse

    def _proposal_prior_log_prob(self, r: int, z: torch.Tensor) -> torch.Tensor:
        """log ptilde^r(theta) in ``z`` coordinates (see module docstring)."""
        assert self.theta_scaler is not None and self._x_obs is not None
        logs = []
        for s in range(r):
            if s == 0:
                logs.append(self.prior_z.log_prob(z))
            else:
                logs.append(self.posteriors[s - 1].log_prob_z(self._x_obs, z))
        stacked = torch.stack(logs, dim=0)
        return torch.logsumexp(stacked, dim=0) - torch.log(torch.tensor(float(r)))

    @property
    def prior_z(self) -> StandardizedDistribution:
        assert self.theta_scaler is not None
        return StandardizedDistribution(self.prior, self.theta_scaler)

    @torch.no_grad()
    def _importance_weights(self, r: int, z: torch.Tensor) -> torch.Tensor:
        """``h_i`` proportional to ``p(theta_i) / ptilde^r(theta_i)``.

        The proposal prior is evaluated with the instantaneous
        change-of-variables formula, which is differentiated through the score
        network; the weights are therefore computed under ``torch.no_grad()``
        and detached, so that they act as constants in the loss of SNPSE-B.
        """
        log_w = self.prior_z.log_prob(z) - self._proposal_prior_log_prob(r, z)
        log_w = torch.nan_to_num(log_w.detach(), neginf=-1e30)
        return torch.exp(log_w - log_w.max())

    def _resample(self, z: torch.Tensor, w: torch.Tensor, num: int) -> torch.Tensor:
        idx = torch.multinomial(w.reshape(-1), num, replacement=self.config.resample_with_replacement)
        return z[idx]


class SNPSEA(_SequentialBase):
    """Post-hoc importance weight correction (Appendix C.2, Algorithm 3)."""

    def run(self, x_obs: torch.Tensor, verbose: Optional[bool] = None) -> NPSE:
        cfg = self.config
        verbose = cfg.verbose if verbose is None else verbose
        torch.manual_seed(self.seed)
        x_obs = x_obs.reshape(1, -1).to(self.device)
        M = cfg.num_simulations // cfg.num_rounds
        if M * cfg.num_rounds != cfg.num_simulations:
            raise ValueError("num_simulations must be divisible by num_rounds")
        started = time.time()

        for r in range(1, cfg.num_rounds + 1):
            if r == 1:
                theta_r, x_r = self._init_round_one(M)
            else:
                # (iii) sample from the approximate proposal posterior, then
                # (iv) recover posterior-estimate samples by SIR.
                prev = self.posteriors[r - 2]
                z_tilde = prev.sample_z(x_obs, cfg.num_candidates * M, seed=self.seed + r)
                self._x_obs = x_obs
                w = self._importance_weights(r - 1, z_tilde)
                z_r = self._resample(z_tilde, w, M)
                theta_r = self.theta_scaler.inverse(z_r)
                x_r = self._simulate(theta_r)

            self.theta_rounds.append(theta_r)
            self.x_rounds.append(x_r.detach().cpu())
            self._x_obs = x_obs

            npse = self._train_on_dataset()
            self.npse = npse
            self.posteriors.append(npse)
            info = {
                "round": r,
                "num_simulations_total": int(sum(t.shape[0] for t in self.theta_rounds)),
                "best_val_loss": npse.train_info.get("best_val_loss"),
                "seconds": time.time() - started,
            }
            self.diagnostics.append(info)
            if verbose:
                print(f"  [SNPSE-A] round {r:2d} | sims {info['num_simulations_total']:6d} "
                      f"| val {info['best_val_loss']:.4f}")
            if cfg.max_seconds is not None and (time.time() - started) > cfg.max_seconds:
                break
        return self.npse

    @torch.no_grad()
    def sample(self, x_obs: torch.Tensor, num_samples: int, num_candidates: Optional[int] = None) -> torch.Tensor:
        """Sample from ``p_psi^R`` via SIR (Eq. 81)."""
        r = len(self.posteriors)
        self._x_obs = x_obs.reshape(1, -1).to(self.device)
        num_candidates = num_candidates or self.config.num_candidates * num_samples
        z_tilde = self.npse.sample_z(self._x_obs, num_candidates, seed=self.seed + 1000 + r)
        w = self._importance_weights(r, z_tilde)
        z = self._resample(z_tilde, w, num_samples)
        return self.theta_scaler.inverse(z)


class SNPSEB(_SequentialBase):
    """Importance-weighted score matching objective (Appendix C.3)."""

    def run(self, x_obs: torch.Tensor, verbose: Optional[bool] = None) -> NPSE:
        cfg = self.config
        verbose = cfg.verbose if verbose is None else verbose
        torch.manual_seed(self.seed)
        x_obs = x_obs.reshape(1, -1).to(self.device)
        M = cfg.num_simulations // cfg.num_rounds
        if M * cfg.num_rounds != cfg.num_simulations:
            raise ValueError("num_simulations must be divisible by num_rounds")
        started = time.time()

        for r in range(1, cfg.num_rounds + 1):
            if r == 1:
                theta_r, x_r = self._init_round_one(M)
            else:
                prev = self.posteriors[r - 2]
                z_r = prev.sample_z(x_obs, M, seed=self.seed + r)
                theta_r = self.theta_scaler.inverse(z_r)
                x_r = self._simulate(theta_r)

            self.theta_rounds.append(theta_r)
            self.x_rounds.append(x_r.detach().cpu())
            self._x_obs = x_obs

            # importance weights p(theta) / ptilde^r(theta) for every sample
            z_all = self.theta_scaler.transform(torch.cat(self.theta_rounds, dim=0))
            w_all = self._importance_weights(r, z_all)
            w_all = w_all / w_all.mean().clamp_min(1e-12)

            npse = self._train_on_dataset(sample_weights=w_all)
            self.npse = npse
            self.posteriors.append(npse)
            info = {
                "round": r,
                "num_simulations_total": int(sum(t.shape[0] for t in self.theta_rounds)),
                "best_val_loss": npse.train_info.get("best_val_loss"),
                "ess": float((w_all.sum() ** 2 / (w_all ** 2).sum()).item()),
                "seconds": time.time() - started,
            }
            self.diagnostics.append(info)
            if verbose:
                print(f"  [SNPSE-B] round {r:2d} | sims {info['num_simulations_total']:6d} "
                      f"| val {info['best_val_loss']:.4f} | ESS {info['ess']:.1f}")
            if cfg.max_seconds is not None and (time.time() - started) > cfg.max_seconds:
                break
        return self.npse


class ScoreSpaceCorrectedNet(torch.nn.Module):
    """``stilde(theta_t, x, t) = s_psi + s_prop(theta_t, t) - grad log p_t``.

    Implements Eq. (103).  ``posterior_score`` undoes the correction to recover
    ``s_psi``, which Algorithm 5 substitutes into the probability flow ODE in
    order to sample from the posterior.
    """

    def __init__(self, base: torch.nn.Module, prop: torch.nn.Module, prior_score: PerturbedPriorScore) -> None:
        super().__init__()
        self.base = base
        self.prop = prop
        self.prior_score = prior_score

    def _proposal_prior_score(self, theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The proposal-prior score network is trained without observations, so
        it is evaluated with a dummy (zero) data input."""
        dummy = torch.zeros(theta_t.shape[0], 1, device=theta_t.device, dtype=theta_t.dtype)
        return self.prop(theta_t, dummy, t)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        if t.shape[0] == 1 and theta_t.shape[0] > 1:
            t = t.expand(theta_t.shape[0])
        return self.base(theta_t, x, t) + self._proposal_prior_score(theta_t, t) - self.prior_score(theta_t, t)

    def posterior_score(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        if t.shape[0] == 1 and theta_t.shape[0] > 1:
            t = t.expand(theta_t.shape[0])
        return (
            self.forward(theta_t, x, t)
            - self._proposal_prior_score(theta_t, t)
            + self.prior_score(theta_t, t)
        )


class SNPSEC(_SequentialBase):
    """Score-space correction (Appendix C.4, Algorithm 5).

    The paper reports that this method failed to give meaningful results
    (C2ST close to 1), which the authors attribute to the approximation error
    incurred when estimating the score of the proposal prior (Appendix C.4.3).
    The implementation below follows the "Approximating the Proposal Prior
    Score" route: in every round ``r >= 2`` a score network is trained on the
    accumulated proposal samples to approximate
    ``grad log ptilde_t^r(theta_t)`` (Eq. 123), and is combined with the
    analytically available perturbed prior score to define ``stilde``.
    """

    def __init__(self, *args, prior_score: Optional[PerturbedPriorScore] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prior_score = prior_score
        self.prop_scores: List[torch.nn.Module] = []

    def _get_prior_score(self) -> PerturbedPriorScore:
        if self.prior_score is not None:
            return self.prior_score
        dist = _base_distribution(self.prior)
        if dist is not None and self.sde_name.lower() in ("ve", "vesde") and self.sde is not None:
            cls_name = type(dist).__name__.lower()
            if "uniform" in cls_name:
                self.prior_score = UniformVEPriorScore(
                    dist.low.reshape(-1), dist.high.reshape(-1), self.sde
                )
                return self.prior_score
            if "normal" in cls_name:
                self.prior_score = GaussianVEPriorScore(
                    dist.mean.reshape(-1), dist.variance.reshape(-1), self.sde
                )
                return self.prior_score
        self.prior_score = estimate_prior_score(
            self.prior,
            self.dim_parameters,
            self.sde,
            standardizer=self.theta_scaler,
            seed=self.seed,
            device=self.device,
        )
        return self.prior_score

    def _train_proposal_prior_score(self) -> torch.nn.Module:
        """Eq. (123): denoising score matching on samples from ptilde^r."""
        from .networks import ScoreNetwork
        from .training import train_score_network

        z = self.theta_scaler.transform(torch.cat(self.theta_rounds, dim=0))
        net = ScoreNetwork(dim_parameters=self.dim_parameters, dim_data=1).to(self.device)
        dummy_x = torch.zeros(z.shape[0], 1, device=self.device)
        net, _ = train_score_network(
            net,
            self.sde,
            z,
            dummy_x,
            lr=self.training_config.lr,
            batch_size=self.training_config.batch_size,
            max_iters=self.training_config.max_iters,
            val_fraction=self.training_config.val_fraction,
            patience=self.training_config.patience,
            loss_weight=self.training_config.loss_weight,
            seed=self.training_config.seed,
        )
        return net

    def run(self, x_obs: torch.Tensor, verbose: Optional[bool] = None) -> NPSE:
        cfg = self.config
        verbose = cfg.verbose if verbose is None else verbose
        torch.manual_seed(self.seed)
        x_obs = x_obs.reshape(1, -1).to(self.device)
        M = cfg.num_simulations // cfg.num_rounds
        if M * cfg.num_rounds != cfg.num_simulations:
            raise ValueError("num_simulations must be divisible by num_rounds")
        started = time.time()

        for r in range(1, cfg.num_rounds + 1):
            if r == 1:
                theta_r, x_r = self._init_round_one(M)
            else:
                prev = self.posteriors[r - 2]
                z_r = prev.sample_z(x_obs, M, seed=self.seed + r)
                theta_r = self.theta_scaler.inverse(z_r)
                x_r = self._simulate(theta_r)

            self.theta_rounds.append(theta_r)
            self.x_rounds.append(x_r.detach().cpu())
            self._x_obs = x_obs

            if r == 1:
                npse = self._train_on_dataset()
            else:
                prop = self._train_proposal_prior_score()
                self.prop_scores.append(prop)
                prior_score = self._get_prior_score()
                from .networks import ScoreNetwork

                base = ScoreNetwork(
                    dim_parameters=self.dim_parameters, dim_data=self.dim_data
                ).to(self.device)
                net = ScoreSpaceCorrectedNet(base, prop, prior_score)
                theta = torch.cat(self.theta_rounds, dim=0)
                x = torch.cat(self.x_rounds, dim=0).to(self.device)
                npse = self._new_npse()
                npse.fit(
                    theta,
                    x,
                    theta_scaler=self.theta_scaler,
                    x_scaler=self.x_scaler,
                    sigma_max=self.sigma_max,
                    init_network=net,
                    verbose=False,
                )
                # sampling uses s_psi = stilde - s_prop + grad log p_t
                npse.sampling_network = net
                from .diffusion import DiffusionPosterior

                npse.posterior = DiffusionPosterior(
                    PosteriorScoreModule(net),
                    npse.sde,
                    self.dim_parameters,
                    npse.posterior.config,
                    self.device,
                )
                npse.net = net

            self.sde = npse.sde
            self.npse = npse
            self.posteriors.append(npse)
            info = {
                "round": r,
                "num_simulations_total": int(sum(t.shape[0] for t in self.theta_rounds)),
                "best_val_loss": npse.train_info.get("best_val_loss"),
                "seconds": time.time() - started,
            }
            self.diagnostics.append(info)
            if verbose:
                print(f"  [SNPSE-C] round {r:2d} | sims {info['num_simulations_total']:6d} "
                      f"| val {info['best_val_loss']:.4f}")
            if cfg.max_seconds is not None and (time.time() - started) > cfg.max_seconds:
                break

        self.prior_score = self._get_prior_score()
        return self.npse


class PosteriorScoreModule(torch.nn.Module):
    """Exposes the *posterior* score ``s_psi`` of a corrected network.

    ``s_psi = stilde - s_prop + grad log p_t`` (Algorithm 5, final step).
    """

    def __init__(self, corrected: ScoreSpaceCorrectedNet) -> None:
        super().__init__()
        self.corrected = corrected

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.corrected.posterior_score(theta_t, x, t)
