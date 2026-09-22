"""Score-divergence ADVI baseline (labeled "Score" in the paper's figures).

Section 5.1 of the paper compares BaM against a modified ADVI method where,
"instead of the ELBO loss, we use the score-based divergence (labeled as
"Score")".  Appendix E.1 (Algorithm 2) describes ADVI, and the score version is
"an alternate version of ADVI using the score-based divergence ... in place of
the (negative) ELBO loss".

The objective minimized here is the *empirical* (batch) estimate of the
score-based divergence, which the paper introduces for the Gaussian variational
family in Section 2.2 as

.. math::
    \\mathscr{D}(q ; p) := \\int
        \\left\\| \\nabla_z \\log \\frac{q(z)}{p(z)} \\right\\|_{\\operatorname{Cov}(q)}^{2} q(z)\\, dz,

i.e. the weighted vector norm is mediated by :math:`\\operatorname{Cov}(q)`
(this is exactly what makes the divergence affine invariant -- Property 2,
Theorem A.4 -- and it recovers the weighted Fisher divergence with
:math:`M = \\operatorname{Cov}(q)`).

Appendix C.1 turns this into the batch estimator that is actually optimized.
With :math:`q = \\mathcal{N}(\\mu, \\Sigma)`, the variational score is
:math:`\\nabla \\log q(z_b) = -\\Sigma^{-1}(z_b - \\mu)`, and with
:math:`g_b = \\nabla \\log p(z_b)` the empirical divergence of eq. (93) is

.. math::
    \\widehat{\\mathscr{D}}_{q_t}(q ; p)
      = \\frac{1}{B} \\sum_{b=1}^{B}
        \\left\\| -\\Sigma^{-1}(z_b - \\mu) - g_b \\right\\|_{\\Sigma}^{2}.

Expressing it in terms of the batch statistics of eq. (95),

.. math::
    \\bar z = \\tfrac{1}{B}\\sum_b z_b, \\quad
    \\bar g = \\tfrac{1}{B}\\sum_b g_b, \\quad
    C = \\tfrac{1}{B}\\sum_b (z_b-\\bar z)(z_b-\\bar z)^{\\top}, \\quad
    \\Gamma = \\tfrac{1}{B}\\sum_b (g_b-\\bar g)(g_b-\\bar g)^{\\top},

eq. (98) gives the equivalent form

.. math::
    \\widehat{\\mathscr{D}}_{q_t}(q ; p)
      = \\operatorname{tr}(\\Gamma \\Sigma) + \\operatorname{tr}(C \\Sigma^{-1})
        + \\|\\mu - \\bar z - \\Sigma \\bar g\\|_{\\Sigma^{-1}}^{2}
        + \\text{constant},

where the trailing constant does not depend on the candidate :math:`(\\mu,
\\Sigma)`.  Both forms are implemented below and agree up to that constant.

The variational family, the ADAM optimizer and the update schedule are exactly
those of :class:`bam_repro.baselines.advi.GradientVI`; only the loss differs.
As elsewhere in this reproduction, gradients are taken through the
reparameterized samples while the *sampled* target scores are held fixed (the
Hessian-free estimator), because second-order derivatives of the target
log-density are unavailable for black-box targets.

Note on learning rates: "the score-based divergence is typically more sensitive
to the learning rate", so the paper grid-searches it.  The values it quotes for
the Gaussian targets (Section 5.1 / E.3) are recorded in
:data:`PAPER_GAUSSIAN_SCORE_LR`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from ..bam.matrix_equations import ensure_spd, inverse_spd, symmetrize
from .advi import ADVI, GradientVI, GradientVIResult

__all__ = [
    "ScoreADVI",
    "ScoreADVIResult",
    "score_divergence_estimate",
    "score_divergence_from_stats",
    "score_divergence_batch_form",
    "score_loss",
    "score_advi_fit",
    "PAPER_GAUSSIAN_SCORE_LR",
    "paper_score_learning_rate",
    "DEFAULT_SCORE_LR",
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Grid-searched learning rates quoted in the paper's protocol for the "Score"
#: baseline on the Gaussian targets of Section 5.1 / E.3, indexed by dimension.
PAPER_GAUSSIAN_SCORE_LR: Dict[int, float] = {4: 0.01, 16: 0.005, 64: 0.001, 256: 0.001}

DEFAULT_SCORE_LR: float = 0.01

_JITTER = 1e-12


# ---------------------------------------------------------------------------
# Estimators of the empirical score-based divergence (eqs. 93 and 98)
# ---------------------------------------------------------------------------
def score_divergence_estimate(mu, Sigma, z, g, jitter: float = _JITTER, xp=None):
    """Direct evaluation of eq. (93), the loss minimized by the Score baseline.

        ``(1/B) * sum_b || -Sigma^{-1}(z_b - mu) - g_b ||_Sigma^2``

    Parameters
    ----------
    mu : (D,) array
        Mean of the *candidate* Gaussian ``q`` (the object being optimized).
    Sigma : (D, D) array
        Covariance of the candidate Gaussian ``q``.
    z : (B, D) array
        Samples drawn from the current ``q_t``.
    g : (B, D) array
        Target scores ``grad log p(z_b)`` at those samples.
    jitter : float
        SPD floor used when inverting ``Sigma``.
    xp : module, optional
        Array namespace (defaults to NumPy; a JAX namespace may be supplied).

    Returns
    -------
    scalar -- batch estimate of the divergence, including the batch-dependent
    constant of eq. (98) (so it is not directly comparable to the *centered*
    form returned by :func:`score_divergence_from_stats`).
    """
    if xp is None:
        xp = np
    mu = xp.asarray(mu)
    z = xp.asarray(z)
    g = xp.asarray(g)

    Sigma_spd = ensure_spd(symmetrize(xp.asarray(Sigma)), jitter=jitter)
    Sigma_inv = inverse_spd(Sigma_spd, jitter=jitter)

    # grad log q(z_b) = -Sigma^{-1} (z_b - mu)
    diff = z - mu[None, :]
    score_q = -(diff @ Sigma_inv.T)

    # residual r_b = grad log q(z_b) - grad log p(z_b)
    r = score_q - g

    # ||r_b||_Sigma^2 = r_b^T Sigma r_b  (vectorized over the batch)
    weighted = xp.sum(r * (r @ Sigma_spd.T), axis=-1)
    return xp.mean(weighted)


def score_divergence_from_stats(
    mu, Sigma, z_bar, g_bar, C, Gamma, jitter: float = _JITTER, xp=None
):
    """Eq. (98) form of the estimator, built from the batch statistics.

        ``tr(Gamma Sigma) + tr(C Sigma^{-1})
          + ||mu - z_bar - Sigma g_bar||^2_{Sigma^{-1}}``

    The batch-only constant of eq. (98) is dropped: it has no dependence on
    ``mu`` or ``Sigma``, so it does not affect the update.
    """
    if xp is None:
        xp = np
    mu = xp.asarray(mu)
    z_bar = xp.asarray(z_bar)
    g_bar = xp.asarray(g_bar)
    C = symmetrize(xp.asarray(C))
    Gamma = symmetrize(xp.asarray(Gamma))

    Sigma_sym = symmetrize(xp.asarray(Sigma))
    Sigma_spd = ensure_spd(Sigma_sym, jitter=jitter)
    Sigma_inv = inverse_spd(Sigma_spd, jitter=jitter)

    term_gamma = xp.trace(Gamma @ Sigma_sym)
    term_C = xp.trace(C @ Sigma_inv)

    v = mu - z_bar - Sigma_sym @ g_bar
    term_mean = xp.sum(v * (v @ Sigma_inv.T))

    return term_gamma + term_C + term_mean


def score_divergence_batch_form(mu, Sigma, z, g, jitter: float = _JITTER, xp=None):
    """Eq. (98) estimator computed from samples (batch statistics formed here)."""
    if xp is None:
        xp = np
    z = xp.asarray(z)
    g = xp.asarray(g)
    B = z.shape[0]
    z_bar = xp.mean(z, axis=0)
    g_bar = xp.mean(g, axis=0)
    zc = z - z_bar[None, :]
    gc = g - g_bar[None, :]
    C = (zc.T @ zc) / B
    Gamma = (gc.T @ gc) / B
    return score_divergence_from_stats(
        mu, Sigma, z_bar, g_bar, C, Gamma, jitter=jitter, xp=xp
    )


def score_loss(mu, Sigma, z, g, jitter: float = _JITTER, xp=None):
    """Alias of :func:`score_divergence_estimate` (the Score-ADVI objective)."""
    return score_divergence_estimate(mu, Sigma, z, g, jitter=jitter, xp=xp)


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------
class ScoreADVI(GradientVI):
    """ADVI whose loss is the score-based divergence (paper label: "Score").

    Same variational family, reparameterization and ADAM optimizer as
    :class:`bam_repro.baselines.advi.ADVI` (Algorithm 2); the objective is the
    empirical score-based divergence of eqs. (93)-(98) instead of the negative
    ELBO.  This is realized by selecting the ``"score"`` loss of
    :class:`~bam_repro.baselines.advi.GradientVI`, which is exactly this
    estimator (reparameterized gradient, sampled target scores held fixed).

    Parameters
    ----------
    mu0, Sigma0 : array
        Initial variational mean (D,) and covariance (D, D).
    score_fn : callable
        ``z -> grad log p(z)`` -- the only target access this objective needs.
    batch_size : int, default 2
        Reparameterized samples per iteration (B = 2 for the Gaussian targets
        and B = 5 for the sinh-arcsinh targets in Section 5.1).
    learning_rate : float or callable, default 0.01
        ADAM step size; a callable receives the iteration index ``t``.
    hold_score_fixed : bool, default True
        Whether sampled target scores are constants under differentiation
        (the Hessian-free estimator).
    **kwargs
        Forwarded to :class:`~bam_repro.baselines.advi.GradientVI`.
    """

    loss_kind = "score"
    name = "score"

    def __init__(
        self,
        mu0,
        Sigma0,
        score_fn: Optional[Callable] = None,
        target: Optional[Any] = None,
        learning_rate: Union[float, Callable[[int], float]] = DEFAULT_SCORE_LR,
        batch_size: int = 2,
        hold_score_fixed: bool = True,
        **kwargs,
    ) -> None:
        kwargs.pop("loss", None)
        super().__init__(
            mu0=mu0,
            Sigma0=Sigma0,
            score_fn=score_fn,
            target=target,
            loss="score",
            learning_rate=learning_rate,
            batch_size=batch_size,
            **kwargs,
        )
        self.hold_score_fixed = bool(hold_score_fixed)

    # -- diagnostics -------------------------------------------------------
    def current_moments(self):
        """Return the current ``(mu, Sigma)`` as plain NumPy arrays."""
        mu = getattr(self, "mu", None)
        if mu is None:
            mu = np.asarray(self.mean)
        Sigma = getattr(self, "Sigma", None)
        if Sigma is None:
            Sigma = np.asarray(self.covariance)
        return np.asarray(mu, dtype=float), np.asarray(Sigma, dtype=float)

    def score_divergence(self, mu=None, Sigma=None, batch_size=None) -> float:
        """Estimate the score-based divergence at ``(mu, Sigma)``.

        Defaults to the optimizer's current state and batch size; the estimate
        uses a fresh batch of samples from the current variational distribution.
        """
        mu_c, Sigma_c = self.current_moments()
        if mu is not None:
            mu_c = np.asarray(mu, dtype=float)
        if Sigma is not None:
            Sigma_c = np.asarray(Sigma, dtype=float)
        B = self.batch_size if batch_size is None else int(batch_size)
        z = np.asarray(self.sample(B), dtype=float)
        g = np.asarray(self.evaluate_scores(z), dtype=float)
        return float(np.asarray(score_divergence_estimate(mu_c, Sigma_c, z, g)))

    def fisher_divergence(self, batch_size=None) -> float:
        """Fisher divergence estimate ``E_q[||grad log q - grad log p||^2]``.

        Provided for comparison with the Fisher baseline (the *unweighted*,
        non-affine-invariant special case mentioned in Section 2.2).
        """
        mu_c, _ = self.current_moments()
        Sigma = getattr(self, "Sigma", None)
        if Sigma is None:
            Sigma = np.asarray(self.covariance, dtype=float)
        B = self.batch_size if batch_size is None else int(batch_size)
        z = np.asarray(self.sample(B), dtype=float)
        g = np.asarray(self.evaluate_scores(z), dtype=float)
        Sigma_inv = inverse_spd(ensure_spd(symmetrize(np.asarray(Sigma)), jitter=_JITTER))
        r = -(z - mu_c[None, :]) @ Sigma_inv.T - g
        return float(np.mean(np.sum(r * r, axis=-1)))


ScoreADVIResult = GradientVIResult


def _make_score_advi(mu0, Sigma0, score_fn, target, batch_size, learning_rate, **kwargs):
    """Prefer ``ADVI``'s constructor and select the score loss via keyword.

    :class:`~bam_repro.baselines.advi.ADVI` is a thin subclass of
    :class:`~bam_repro.baselines.advi.GradientVI` fixing the ELBO loss, so the
    score loss is requested from ``GradientVI`` directly, with ``ADVI`` used
    whenever it turns out to accept a loss override.
    """
    base_kwargs = dict(
        mu0=mu0,
        Sigma0=Sigma0,
        score_fn=score_fn,
        target=target,
        loss="score",
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs,
    )
    try:
        return ScoreADVI(
            mu0=mu0,
            Sigma0=Sigma0,
            score_fn=score_fn,
            target=target,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
    except TypeError:  # pragma: no cover - defensive fallback
        return GradientVI(**base_kwargs)


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------
def score_advi_fit(
    mu0,
    Sigma0,
    score_fn: Optional[Callable] = None,
    target: Optional[Any] = None,
    T: int = 1000,
    batch_size: int = 2,
    learning_rate: Union[float, Callable[[int], float]] = DEFAULT_SCORE_LR,
    track_history: bool = False,
    history_every: int = 1,
    seed: int = 0,
    **kwargs,
) -> GradientVIResult:
    """Run Score-ADVI for ``T`` iterations and return the result container."""
    opt = _make_score_advi(
        mu0=mu0,
        Sigma0=Sigma0,
        score_fn=score_fn,
        target=target,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        track_history=track_history,
        history_every=history_every,
        **kwargs,
    )
    return opt.run(T)


def paper_score_learning_rate(dim: int, default: float = DEFAULT_SCORE_LR) -> float:
    """Grid-searched Score learning rate used by the paper for dimension ``dim``."""
    return float(PAPER_GAUSSIAN_SCORE_LR.get(int(dim), default))
