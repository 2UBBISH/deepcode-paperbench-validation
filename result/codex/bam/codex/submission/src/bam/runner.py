"""Shared machinery for the experiment scripts: running methods and metrics.

Everything in the paper is evaluated as a curve against the number of
*gradient evaluations* of the target (``B`` per iteration for every method, since
each iteration evaluates the target score, or the log density, at ``B`` points).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import jax
import numpy as np

from .bam import BaM, BaMConfig, lambda_constant, lambda_decay
from .baselines import GSM, GradientVI, GradientVIConfig
from .divergence import kl_gaussian, kl_gaussian_samples, score_based_divergence
from .metrics import relative_mean_error, relative_sd_error
from .targets import GaussianTarget, Target
from .utils import plot_curves


@dataclass
class MethodSpec:
    """Hyper-parameters of one method on one target."""

    name: str                       # 'BaM' | 'ADVI' | 'Score' | 'Fisher' | 'GSM'
    batch_size: int
    n_iters: int
    learning_rate: float = 0.01     # gradient-based methods
    loss: str = "elbo"              # 'elbo' | 'score' | 'fisher'
    lam: Optional[Callable[[int], float]] = None   # BaM inverse regularization
    lam_kind: str = "constant"      # bookkeeping only
    lam_value: float = 0.0

    @property
    def label(self) -> str:
        if self.name == "BaM":
            return f"BaM (B={self.batch_size})"
        return f"{self.name} (B={self.batch_size})"


def bam_spec(batch_size: int, n_iters: int, lam_value: float, decay: bool = False,
             power: float = 1.0) -> MethodSpec:
    """BaM with a constant (``lambda_t = lam_value``) or decaying inverse regularization."""
    lam = lambda_decay(lam_value, power) if decay else lambda_constant(lam_value)
    return MethodSpec(name="BaM", batch_size=batch_size, n_iters=n_iters, lam=lam,
                      lam_kind="decay" if decay else "constant", lam_value=lam_value)


def gradient_spec(name: str, batch_size: int, n_iters: int, learning_rate: float) -> MethodSpec:
    loss = {"ADVI": "elbo", "Score": "score", "Fisher": "fisher"}[name]
    return MethodSpec(name=name, batch_size=batch_size, n_iters=n_iters, learning_rate=learning_rate,
                      loss=loss)


def gsm_spec(batch_size: int, n_iters: int) -> MethodSpec:
    return MethodSpec(name="GSM", batch_size=batch_size, n_iters=n_iters)


# ----------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------


def run_method(target: Target, spec: MethodSpec, key, mu0: Optional[np.ndarray] = None,
               Sigma0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Run one method and return its iterates plus the gradient-evaluation counts."""
    if spec.name == "BaM":
        bam = BaM(target, BaMConfig(batch_size=spec.batch_size, lam=spec.lam))
        out = bam.run(key, spec.n_iters, mu0, Sigma0)
    elif spec.name in ("ADVI", "Score", "Fisher"):
        vi = GradientVI(target, GradientVIConfig(batch_size=spec.batch_size, learning_rate=spec.learning_rate,
                                                 loss=spec.loss, n_iters=spec.n_iters))
        out = vi.run(key, mu0=mu0, Sigma0=Sigma0)
    elif spec.name == "GSM":
        out = GSM(target, batch_size=spec.batch_size).run(key, spec.n_iters, mu0=mu0, Sigma0=Sigma0)
    else:
        raise ValueError(f"unknown method {spec.name!r}")
    out["spec"] = spec
    out["name"] = spec.name
    out["batch_size"] = spec.batch_size
    return out


def random_init(key, dim: int, scale: float = 0.1) -> np.ndarray:
    """``mu_0 ~ Uniform[0, 0.1]^D``, as in Appendices E.3 and E.5."""
    return scale * np.asarray(jax.random.uniform(key, (dim,)))


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------


def evaluate_metric(target: Target, out: Dict[str, np.ndarray], metric: str, key,
                    n_samples: int = 20000, stride: int = 1) -> np.ndarray:
    """Metric value at (a subset of) the iterates of a run.

    Supported metrics:

    * ``forward_kl``  -- ``KL(p ; q)`` (exact for Gaussian targets, else Monte Carlo)
    * ``reverse_kl``  -- ``KL(q ; p)`` (same)
    * ``score_div``   -- the score-based divergence ``D(q ; p)`` (Monte Carlo)
    * ``rel_mean``    -- relative posterior mean error (Section 5.2)
    * ``rel_sd``      -- relative posterior SD error (Figure E.6)
    * ``mse``         -- reconstruction MSE of the deep generative model (Section 5.3)
    """
    idxs = list(range(0, out["mu"].shape[0], stride))
    vals = np.full(len(idxs), np.nan)
    is_gauss = isinstance(target, GaussianTarget)
    for j, i in enumerate(idxs):
        mu, Sigma = out["mu"][i], out["Sigma"][i]
        if metric == "forward_kl":
            if is_gauss:
                # KL(p ; q) is the "forward" direction (Section 5.1)
                vals[j] = kl_gaussian(target.mu, target.Sigma, mu, Sigma)
            else:
                vals[j] = kl_gaussian_samples(target, mu, Sigma, n_samples, key, "forward")
        elif metric == "reverse_kl":
            if is_gauss:
                vals[j] = kl_gaussian(mu, Sigma, target.mu, target.Sigma)
            else:
                vals[j] = kl_gaussian_samples(target, mu, Sigma, n_samples, key, "reverse")
        elif metric == "score_div":
            vals[j] = score_based_divergence(target, mu, Sigma, n_samples, key)
        elif metric in ("rel_mean", "rel_sd"):
            mean, sd = target.variational_summaries(mu, Sigma)
            if metric == "rel_mean":
                vals[j] = relative_mean_error(mean, target.ref_mean, target.ref_sd)
            else:
                vals[j] = relative_sd_error(sd, target.ref_sd)
        elif metric == "mse":
            vals[j] = target.reconstruction_mse(mu)
        else:
            raise ValueError(f"unknown metric {metric!r}")
    return vals


def run_repeats(target_factory: Callable[[int], Target], spec: MethodSpec, metric: str, n_runs: int = 10,
                base_key: int = 0, n_samples: int = 20000, stride: int = 1,
                init_scale: float = 0.1) -> Dict[str, np.ndarray]:
    """Run a method on ``n_runs`` random seeds and return the metric curves.

    ``target_factory(seed)`` returns the target for a given run (so that random
    targets are re-drawn per run, as in Section 5.1).
    """
    curves = []
    grad_evals = None
    for r in range(n_runs):
        key = jax.random.PRNGKey(base_key + r)
        target = target_factory(r)
        mu0 = random_init(jax.random.PRNGKey(1000 + r), target.dim, init_scale)
        Sigma0 = np.eye(target.dim)
        out = run_method(target, spec, key, mu0, Sigma0)
        vals = evaluate_metric(target, out, metric, jax.random.PRNGKey(5000 + r), n_samples=n_samples,
                               stride=stride)
        curves.append(vals)
        grad_evals = out["grad_evals"][::stride]
    return {"grad_evals": grad_evals, "values": np.stack(curves, axis=0)}


def summarise_curves(res: Dict[str, np.ndarray], trim_after_first_finite: bool = True):
    """Mean and standard error over runs, dropping the leading ``inf``/``nan`` values."""
    x = np.asarray(res["grad_evals"], dtype=float)
    y = np.asarray(res["values"], dtype=float)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(y)
    if trim_after_first_finite and finite.any():
        last = np.max(np.where(finite.any(axis=0))[0])
        x, y = x[: last + 1], y[:, : last + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(y, axis=0)
        se = np.nanstd(y, axis=0, ddof=1) / np.sqrt(y.shape[0]) if y.shape[0] > 1 else np.zeros_like(mean)
    return {"x": x, "y": mean, "se": se, "runs": y}


__all__ = [
    "MethodSpec",
    "bam_spec",
    "gradient_spec",
    "gsm_spec",
    "run_method",
    "run_repeats",
    "evaluate_metric",
    "summarise_curves",
    "random_init",
    "plot_curves",
]
