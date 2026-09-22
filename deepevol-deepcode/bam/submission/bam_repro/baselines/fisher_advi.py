"""Fisher-divergence ADVI baseline ("Fisher" in Figures 5.1 / E.3 / 5.2 / E.4).

The paper compares BaM against three gradient-based variational-inference
baselines that use the *same* reparameterized full-covariance Gaussian family
and the *same* ADAM optimizer, but different objectives:

* ``ADVI``   -- the (negative) ELBO                       (Algorithm 2, App. E.1)
* ``Score``  -- the score-based divergence  ``D(q ; p)`` (§2.2, §A, Prop. A.7)
* ``Fisher`` -- the Fisher divergence                     (§2.2, §A)

The Fisher divergence (Hyvarinen, 2005) is the un-weighted special case of the
weighted Fisher divergence (Barp et al., 2019),

    E_q[ || grad log(q/p) ||_M^2 ],      M in R^{D x D},

with the *identity* weight ``M = I``; the score-based divergence of eq. (2) of
the paper is recovered with ``M = Cov(q)``.  Because ``M = I`` is not invariant
under affine reparameterizations, the Fisher divergence is not affine invariant
(§2.2), unlike the score-based divergence (Theorem A.4).  For a Gaussian
variational family ``q = N(mu, Sigma)`` the objective is

    Fisher(q ; p) = E_q[ || grad log q - grad log p ||^2 ]
                  = E_q[ || -Sigma^{-1}(z - mu) - grad log p(z) ||^2 ],     (A)

i.e. exactly the score-based divergence of eq. (34) with ``Gamma_q^{-1}``
replaced by the identity matrix.  The corresponding empirical estimator is
eq. (93) of the paper's score estimator without the ``||.||_Sigma`` weighting,

    \hat{Fisher} = (1/B) sum_b || Sigma^{-1}(z_b - mu) + g_b ||^2,          (B)
                   with  g_b = grad log p(z_b),

which is what this module minimizes.  As for the score-divergence baseline the
sampled target scores are treated as constants under differentiation
(Hessian-free / black-box estimator), matching the paper's black-box setting.

The hyper-parameters follow the paper exactly:

* Gaussian synthetic targets (§5.1, E.3): learning rate selected by grid search,
  the best value was ``0.01`` for ADVI and Fisher alike.
* sinh-arcsinh synthetic targets (§5.1, E.4): grid search selected ``0.05`` for
  Fisher (``0.02`` for ADVI).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np

from ..bam.matrix_equations import ensure_spd, inverse_spd, symmetrize
from .advi import ADVI, GradientVI, GradientVIResult, LOSSES

__all__ = [
    "FisherADVI",
    "FisherADVIResult",
    "fisher_advi_fit",
    "make_fisher_advi",
    "fisher_divergence_estimate",
    "fisher_divergence_from_stats",
    "fisher_divergence_batch_form",
    "fisher_loss",
    "gaussian_fisher_divergence",
    "fisher_divergence_mc",
    "paper_fisher_learning_rate",
    "grid_search_fisher",
    "FISHER_LR_GRID",
    "PAPER_GAUSSIAN_FISHER_LR",
    "PAPER_NONGaussian_FISHER_LR",
    "DEFAULT_FISHER_LR",
]

# ---------------------------------------------------------------------------
# Learning-rate bookkeeping (paper's grid searches)
# ---------------------------------------------------------------------------

#: Learning rates considered in the paper's grid search for the gradient-based
#: baselines (ADVI / Score / Fisher).
FISHER_LR_GRID: tuple = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)

#: §5.1 / E.3 -- "For ADVI and Fisher, the selected learning rate was 0.01."
PAPER_GAUSSIAN_FISHER_LR: Dict[int, float] = {4: 0.01, 16: 0.01, 64: 0.01, 256: 0.01}

#: §5.1 / E.4 -- "The final selected learning rates were ... 0.05 for Fisher."
PAPER_NONGaussian_FISHER_LR: float = 0.05

#: Fallback when no setting is given.
DEFAULT_FISHER_LR: float = 0.01

_JITTER = 1e-12


# ---------------------------------------------------------------------------
# Divergence estimators
# ---------------------------------------------------------------------------


def fisher_divergence_estimate(
    mu,
    Sigma,
    z,
    g,
    jitter: float = _JITTER,
    xp: Any = None,
    reduction: str = "mean",
):
    """Empirical Fisher divergence, eq. (B) of the module docstring.

    ``(1/B) sum_b || Sigma^{-1}(z_b - mu) + g_b ||^2`` where ``g_b`` is the
    target score ``grad log p(z_b)`` evaluated (and held fixed) at the samples.

    Parameters
    ----------
    mu, Sigma : mean / covariance of the Gaussian variational density.
    z : ``(B, D)`` samples from ``q``.
    g : ``(B, D)`` target scores at ``z``.
    reduction : ``"mean"`` (default), ``"sum"`` or ``"none"``.
    """
    xp = xp if xp is not None else np
    mu = xp.asarray(mu)
    Sigma = xp.asarray(Sigma)
    z = xp.asarray(z)
    g = xp.asarray(g)
    dim = mu.shape[0]

    Sigma_inv = inverse_spd(ensure_spd(symmetrize(Sigma), jitter=jitter), jitter=jitter)
    # grad log q(z_b) = -Sigma^{-1}(z_b - mu);  residual = grad log q - grad log p
    resid = z - mu[None, :]
    grad_log_q = -resid @ Sigma_inv.T
    diff = grad_log_q - g
    sq = xp.sum(diff * diff, axis=1)
    if reduction == "none":
        return sq
    if reduction == "sum":
        return xp.sum(sq)
    return xp.sum(sq) / max(int(z.shape[0]), 1)


def fisher_divergence_from_stats(
    mu,
    Sigma,
    z_bar,
    g_bar,
    C,
    Gamma,
    cross=None,
    jitter: float = _JITTER,
    xp: Any = None,
):
    """Batch-statistics form of the Fisher divergence.

    Expanding eq. (B) around the batch means ``z_bar`` and ``g_bar`` gives

        (1/B) sum_b || Sigma^{-1}(z_b - mu) + g_b ||^2
          = || Sigma^{-1}(z_bar - mu) + g_bar ||^2
            + tr(Sigma^{-1} C Sigma^{-1}) + tr(Gamma) + 2 tr(Sigma^{-1} cross)

    where ``C`` and ``Gamma`` are the sample covariances of``z`` and ``g``, and
    ``cross = (1/B) sum_b (z_b - z_bar)(g_b - g_bar)^T`` (dropped when ``None``).
    Unlike the score-based divergence (eq. 98) the ``Sigma``-weighting is absent,
    reflecting ``M = I``.
    """
    xp = xp if xp is not None else np
    mu = xp.asarray(mu)
    Sigma = xp.asarray(Sigma)
    Sigma = ensure_spd(symmetrize(Sigma), jitter=jitter)
    Sigma_inv = inverse_spd(Sigma, jitter=jitter)

    z_bar = xp.asarray(z_bar)
    g_bar = xp.asarray(g_bar)
    C = xp.asarray(C)
    Gamma = xp.asarray(Gamma)

    center = Sigma_inv @ (z_bar - mu) + g_bar
    val = xp.sum(center * center)
    val = val + xp.trace(Sigma_inv @ C @ Sigma_inv)
    val = val + xp.trace(Gamma)
    if cross is not None:
        val = val + 2.0 * xp.trace(Sigma_inv @ xp.asarray(cross))
    return val


def fisher_divergence_batch_form(
    mu,
    Sigma,
    z,
    g,
    jitter: float = _JITTER,
    xp: Any = None,
):
    """Eq. (B) computed from samples (forms the batch statistics internally)."""
    xp = xp if xp is not None else np
    z = xp.asarray(z)
    g = xp.asarray(g)
    z_bar = xp.mean(z, axis=0)
    g_bar = xp.mean(g, axis=0)
    zc = z - z_bar[None, :]
    gc = g - g_bar[None, :]
    B = z.shape[0]
    C = (zc.T @ zc) / B
    Gamma = (gc.T @ gc) / B
    cross = (zc.T @ gc) / B
    return fisher_divergence_from_stats(
        mu, Sigma, z_bar, g_bar, C, Gamma, cross=cross, jitter=jitter, xp=xp
    )


#: Alias kept for symmetry with ``score_advi.score_loss``.
fisher_loss = fisher_divergence_estimate


def gaussian_fisher_divergence(
    mu,
    Sigma,
    mu_star,
    Sigma_star,
    jitter: float = _JITTER,
    xp: Any = None,
):
    """Closed-form Fisher divergence for Gaussian ``q = N(mu, Sigma)``.

    With ``z ~ q`` and target ``p = N(mu_star, Sigma_star)``,

        E_q[||grad log q - grad log p||^2]
          = tr((Sigma^{-1} - Psi^{-1}) Sigma (Sigma^{-1} - Psi^{-1}))
            + (mu - mu_star)^T Psi^{-1} Sigma Psi^{-1} (mu - mu_star)

    (``Psi = Sigma_star``).  Useful as an exact reference for the estimator
    above (Monte-Carlo error should vanish as ``B -> inf``).  Note this differs
    from the score-based divergence of Proposition A.7, which additionally
    weights by ``Cov(q) = Sigma``.
    """
    xp = xp if xp is not None else np
    Sigma = ensure_spd(symmetrize(xp.asarray(Sigma)), jitter=jitter)
    Psi = ensure_spd(symmetrize(xp.asarray(Sigma_star)), jitter=jitter)
    mu = xp.asarray(mu)
    mu_star = xp.asarray(mu_star)
    Sinv = inverse_spd(Sigma, jitter=jitter)
    Psinv = inverse_spd(Psi, jitter=jitter)
    A = Sinv - Psinv
    d = mu - mu_star
    val = xp.trace(A @ Sigma @ A)
    val = val + d @ (Psinv @ Sigma @ Psinv) @ d
    return val


def fisher_divergence_mc(
    mu,
    Sigma,
    target,
    n_samples: int = 4096,
    rng: Optional[np.random.Generator] = None,
    xp: Any = None,
):
    """Monte Carlo Fisher divergence using a ``target`` exposing ``score``."""
    xp = xp if xp is not None else np
    rng = rng if rng is not None else np.random.default_rng(0)
    mu = xp.asarray(mu)
    Sigma = ensure_spd(symmetrize(xp.asarray(Sigma)), jitter=_JITTER)
    L = np.linalg.cholesky(np.asarray(Sigma))
    eps = rng.standard_normal((int(n_samples), mu.shape[0]))
    z = np.asarray(mu)[None, :] + eps @ L.T
    score_fn = getattr(target, "score", None) or getattr(target, "grad_log_prob", None)
    g = np.asarray(score_fn(z))
    return fisher_divergence_estimate(mu, Sigma, z, g, xp=np)


# ---------------------------------------------------------------------------
# Fisher ADVI
# ---------------------------------------------------------------------------


class FisherADVI(GradientVI):
    """ADVI with the Fisher divergence in place of the (negative) ELBO.

    Everything else (full-covariance Gaussian family, reparameterized sampling,
    ADAM with ``beta1=0.9``, ``beta2=0.999``, ``eps=1e-8``) is inherited from
    :class:`~bam_repro.baselines.advi.GradientVI`, exactly as in the paper:
    "We also implemented an alternate version of ADVI using the score-based
    divergence and the Fisher divergence in place of the (negative) ELBO loss"
    (§E.1).
    """

    loss_kind = "fisher"
    name = "fisher"

    def __init__(
        self,
        mu0,
        Sigma0,
        score_fn: Optional[Callable] = None,
        target: Any = None,
        learning_rate: Union[float, Callable] = DEFAULT_FISHER_LR,
        batch_size: int = 2,
        hold_score_fixed: bool = True,
        **kwargs,
    ):
        # ``advi.GradientVI`` accepts the loss names listed in ``LOSSES``; pick
        # the canonical spelling available in this build.
        if "fisher" in LOSSES:
            loss_name = "fisher"
        elif "fisher_divergence" in LOSSES:
            loss_name = "fisher_divergence"
        else:  # pragma: no cover - defensive fallback
            loss_name = "score"
        self.hold_score_fixed = hold_score_fixed
        super().__init__(
            mu0,
            Sigma0,
            score_fn=score_fn,
            target=target,
            loss=loss_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            **kwargs,
        )

    # -- diagnostics -------------------------------------------------------
    def current_moments(self):
        """Return the current ``(mu, Sigma)`` variational parameters."""
        return np.asarray(self.mu), np.asarray(self.Sigma)

    def fisher_divergence(self, mu=None, Sigma=None, batch_size: Optional[int] = None) -> float:
        """Re-estimate the Fisher divergence at the current (or given) state."""
        mu = self.mu if mu is None else mu
        Sigma = self.Sigma if Sigma is None else Sigma
        B = int(batch_size or self.batch_size)
        z = self.sample(B)
        g = self.evaluate_scores(z)
        return float(fisher_divergence_estimate(np.asarray(mu), np.asarray(Sigma), z, g))

    # Alias for symmetry with ``ScoreADVI.score_divergence``.
    def score_divergence(self, mu=None, Sigma=None, batch_size: Optional[int] = None) -> float:
        return self.fisher_divergence(mu=mu, Sigma=Sigma, batch_size=batch_size)


#: Result container alias (the gradient-VI engines share one result type).
FisherADVIResult = GradientVIResult


def make_fisher_advi(
    mu0,
    Sigma0,
    score_fn: Optional[Callable] = None,
    target: Any = None,
    learning_rate: Union[float, Callable] = DEFAULT_FISHER_LR,
    batch_size: int = 2,
    **kwargs,
) -> GradientVI:
    """Factory returning a Fisher-divergence ADVI learner."""
    cls = FisherADVI if "fisher" in LOSSES or "fisher_divergence" in LOSSES else GradientVI
    try:
        return cls(
            mu0,
            Sigma0,
            score_fn=score_fn,
            target=target,
            learning_rate=learning_rate,
            batch_size=batch_size,
            **kwargs,
        )
    except (ValueError, KeyError, TypeError):  # pragma: no cover - defensive
        return GradientVI(
            mu0,
            Sigma0,
            score_fn=score_fn,
            target=target,
            loss="fisher_divergence",
            learning_rate=learning_rate,
            batch_size=batch_size,
            **kwargs,
        )


def fisher_advi_fit(
    mu0,
    Sigma0,
    score_fn: Optional[Callable] = None,
    target: Any = None,
    T: int = 1000,
    batch_size: int = 2,
    learning_rate: Union[float, Callable] = DEFAULT_FISHER_LR,
    track_history: bool = False,
    history_every: int = 1,
    seed: int = 0,
    **kwargs,
) -> GradientVIResult:
    """Run the Fisher-divergence ADVI baseline for ``T`` iterations.

    Mirrors ``Algorithm 2`` (Appendix E.1) with the ELBO replaced by eq. (B).
    """
    opt = make_fisher_advi(
        mu0,
        Sigma0,
        score_fn=score_fn,
        target=target,
        learning_rate=learning_rate,
        batch_size=batch_size,
        track_history=track_history,
        history_every=history_every,
        seed=seed,
        **kwargs,
    )
    return opt.run(T)


# ---------------------------------------------------------------------------
# Learning-rate selection (paper's grid search)
# ---------------------------------------------------------------------------


def paper_fisher_learning_rate(dim: Optional[int] = None, setting: str = "gaussian", default: float = DEFAULT_FISHER_LR) -> float:
    """Learning rate selected by the paper's grid search for the Fisher baseline.

    Parameters
    ----------
    dim : dimension of the target (only relevant for Gaussian targets, where the
        selected value is ``0.01`` for every ``D in {4, 16, 64, 256}``).
    setting : ``"gaussian"`` (§5.1 / E.3 -> 0.01) or ``"non_gaussian"`` /
        ``"sinh_arcsinh"`` (§5.1 / E.4 -> 0.05).
    """
    setting_norm = str(setting).lower().replace("-", "_").replace(" ", "_")
    if setting_norm in ("non_gaussian", "nongaussian", "sinh_arcsinh", "sinh_arcsinh_normal", "skew", "tail"):
        return PAPER_NONGaussian_FISHER_LR
    if dim is not None and int(dim) in PAPER_GAUSSIAN_FISHER_LR:
        return PAPER_GAUSSIAN_FISHER_LR[int(dim)]
    return default


def grid_search_fisher(
    make_fit: Callable[[float], float],
    learning_rates: Sequence[float] = FISHER_LR_GRID,
) -> Dict[str, Any]:
    """Grid-search the Fisher-ADVI learning rate.

    ``make_fit(lr)`` must run a (short) fit with that learning rate and return a
    scalar objective to be *minimized* (e.g. the smallest forward/reverse KL
    reached).   Returns a dict with the best rate, its value and the full table.
    """
    table: Dict[float, float] = {}
    best_lr, best_val = None, math.inf
    for lr in learning_rates:
        try:
            val = float(make_fit(float(lr)))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        table[float(lr)] = val
        if val < best_val:
            best_val, best_lr = val, float(lr)
    if best_lr is None:
        best_lr, best_val = float(DEFAULT_FISHER_LR), math.inf
    return {"best_learning_rate": best_lr, "best_value": best_val, "table": table}


# Re-exported for convenience (keeps baseline imports uniform).
_ADVI = ADVI
