"""Negative log-likelihood (NLL) evaluation for Simformer.

Implements the average negative log-likelihood metric of the paper's Sec. 4.1 /
Appendix A3.1:

    "The average negative loglikelihood (NLL) for the true posterior is a metric
    suitable for evaluation on an increased number of different observations.
    We evaluate the average on 5000 samples from the joint distribution. We did
    this for both the posterior and likelihood, as estimated by Simformer ...
    Notably, to evaluate the loglikelihood for the Simformer, we have to use the
    probability flow ODE (Song et al., 2021b). Hence, the loglikelihood is also
    based on the probability flow ODE, not the corresponding SDE formulation."

The instantaneous change-of-variables formula for a diffusion model (Song et
al. 2021) evaluates the log-density of the data distribution ``p_0`` from the
probability-flow ODE

    dx/dt = f(x, t) - 0.5 g(t)^2 s_theta(x, t),

    log p_0(x(0)) = log p_T(x(T)) + int_0^T div( dx/dt ) dt ,

where ``p_T`` is the prior distribution of the (terminal) forward process.
Integration runs in the data -> noise direction (t: 0 -> 1) and the divergence
of the drift is estimated exactly when the joint dimension is small, otherwise
with the Hutchinson trace estimator.

Everything in this module is NumPy + optional torch (only the score function
may call into a torch network); the ODE integration itself lives in
``simformer.diffusion``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import math

import numpy as np

from ..diffusion import (
    DEFAULT_STEPS,
    SDE,
    get_sde,
    log_likelihood_from_ode,
    probability_flow_drift,
    probability_flow_ode,
)

__all__ = [
    "NLLConfig",
    "NLLResult",
    "probability_flow_log_likelihood",
    "log_likelihood",
    "log_prob",
    "conditional_log_likelihood",
    "posterior_log_likelihood",
    "likelihood_log_likelihood",
    "average_nll",
    "evaluate_nll",
    "nll",
    "negative_log_likelihood",
    "nll_many",
    "aggregate_nll",
    "evaluate_posterior_nll",
    "evaluate_likelihood_nll",
    "nll_from_score_fn",
    "joint_log_density",
    "nll_table",
    "summarize",
    "DEFAULT_N_JOINT_SAMPLES",
    "DEFAULT_T_MIN",
    "DEFAULT_T_MAX",
    "DEFAULT_ODE_STEPS",
]

ArrayLike = Union[np.ndarray, Sequence[float]]

#: Number of joint samples used for the average NLL protocol (Appendix A3.1).
DEFAULT_N_JOINT_SAMPLES = 5000
DEFAULT_T_MIN = 1e-5
DEFAULT_T_MAX = 1.0
DEFAULT_ODE_STEPS = 200
DEFAULT_HUTCHINSON_SAMPLES = 1
#: Exact divergence is used below this dimension; Hutchinson above.
DEFAULT_EXACT_DIM = 32


@dataclass
class NLLConfig:
    """Configuration of the probability-flow-ODE NLL evaluation.

    Parameters
    ----------
    n_samples:
        Number of joint samples for the average NLL protocol (paper: 5000).
    n_steps:
        Number of uniform time steps of the ODE discretisation.
    t_min, t_max:
        Integration interval ``[t_min, t_max]`` (defaults ``[1e-5, 1]``).
    exact_divergence:
        If True use the exact trace of the Jacobian; if None the choice is
        automatic (exact when ``dim <= exact_dim``, Hutchinson otherwise).
    n_hutchinson:
        Number of Hutchinson probe vectors when the divergence is estimated.
    chunk:
        Batch size used when evaluating the score function.
    seed:
        Random seed for the Hutchinson estimator.
    reduce:
        ``"mean"`` (paper metric), ``"sum"`` or ``None`` (per-sample values).
    """

    n_samples: int = DEFAULT_N_JOINT_SAMPLES
    n_steps: int = DEFAULT_ODE_STEPS
    t_min: float = DEFAULT_T_MIN
    t_max: float = DEFAULT_T_MAX
    exact_divergence: Optional[bool] = None
    exact_dim: int = DEFAULT_EXACT_DIM
    n_hutchinson: int = DEFAULT_HUTCHINSON_SAMPLES
    chunk: Optional[int] = None
    seed: int = 0
    reduce: Optional[str] = "mean"
    return_trajectory: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_samples": self.n_samples,
            "n_steps": self.n_steps,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "exact_divergence": self.exact_divergence,
            "exact_dim": self.exact_dim,
            "n_hutchinson": self.n_hutchinson,
            "chunk": self.chunk,
            "seed": self.seed,
            "reduce": self.reduce,
        }
        d.update(dict(self.extra))
        return d

    @classmethod
    def from_dict(cls, cfg: Optional[Union[Dict[str, Any], "NLLConfig"]] = None, **kwargs) -> "NLLConfig":
        if cfg is None:
            return cls(**kwargs)
        if isinstance(cfg, NLLConfig):
            base = cfg.to_dict()
        elif isinstance(cfg, dict):
            base = dict(cfg)
        else:  # pragma: no cover - tolerate config objects (hydra/omegaconf)
            base = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
        known = {
            "n_samples",
            "n_steps",
            "t_min",
            "t_max",
            "exact_divergence",
            "exact_dim",
            "n_hutchinson",
            "chunk",
            "seed",
            "reduce",
            "return_trajectory",
        }
        extra = {k: v for k, v in base.items() if k not in known and k != "extra"}
        filtered = {k: base[k] for k in known if k in base}
        if "extra" in base and isinstance(base["extra"], dict):
            extra.update(base["extra"])
        filtered.update(kwargs)
        obj = cls(**filtered)
        obj.extra.update(extra)
        return obj


@dataclass
class NLLResult:
    """Outcome of an NLL evaluation."""

    nll: float
    n_samples: int
    n_steps: int
    per_sample: Optional[np.ndarray] = None
    std_error: Optional[float] = None
    sde: Optional[str] = None
    trajectory: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def log_likelihood(self) -> float:
        return -float(self.nll)

    @property
    def mean(self) -> float:
        return float(self.nll)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nll": float(self.nll),
            "log_likelihood": -float(self.nll),
            "n_samples": int(self.n_samples),
            "n_steps": int(self.n_steps),
            "std_error": None if self.std_error is None else float(self.std_error),
            "sde": self.sde,
        }

    def __float__(self) -> float:
        return float(self.nll)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_sde(sde: Union[str, SDE, None]) -> SDE:
    if sde is None:
        return get_sde("vesde")
    if isinstance(sde, SDE):
        return sde
    return get_sde(str(sde))


def _config(config: Optional[Union[NLLConfig, Dict[str, Any]]] = None, **kwargs) -> NLLConfig:
    if config is None:
        return NLLConfig(**kwargs)
    if isinstance(config, NLLConfig):
        if kwargs:
            cfg = NLLConfig.from_dict(config, **kwargs)
            return cfg
        return config
    return NLLConfig.from_dict(config, **kwargs)


def _call_score_fn(score_fn: Callable, x: np.ndarray, t: Any) -> np.ndarray:
    """Call a score function tolerantly accepting scalar or array ``t``."""
    try:
        out = score_fn(x, t)
    except Exception:
        out = score_fn(x, np.full(np.shape(x)[:1], float(np.asarray(t).reshape(-1)[0]) if np.ndim(t) else float(t)))
    out = np.asarray(out, dtype=float)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.shape != np.shape(x):
        out = np.reshape(out, np.shape(x))
    return out


def _terminal_log_prob(sde: SDE, x_t: np.ndarray) -> np.ndarray:
    """Log density of the terminal (prior) distribution of the forward process."""
    mean, std = sde.prior_mean_std()
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if mean.ndim == 0:
        mean = np.zeros(np.shape(x_t)[-1]) + float(mean)
    if std.ndim == 0:
        std = np.full(np.shape(x_t)[-1], float(std))
    mean = np.broadcast_to(mean, np.shape(x_t))
    std = np.broadcast_to(std, np.shape(x_t))
    z = (np.asarray(x_t, dtype=float) - mean) / std
    return -0.5 * z ** 2 - np.log(std) - 0.5 * math.log(2.0 * math.pi)


def _exact_divergence(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """Central finite-difference estimate of the Jacobian trace of ``fn`` at ``x``."""
    x = np.asarray(x, dtype=float)
    dim = x.shape[-1]
    eps = 1e-4
    trace = np.zeros(x.shape[:-1] if x.ndim > 1 else ())
    for j in range(dim):
        dx = np.zeros_like(x)
        dx[..., j] = eps
        fp = np.asarray(fn(x + dx), dtype=float)
        fm = np.asarray(fn(x - dx), dtype=float)
        trace = trace + (fp[..., j] - fm[..., j]) / (2.0 * eps)
    return trace


def probability_flow_log_likelihood(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    x: ArrayLike,
    sde: Union[str, SDE, None] = None,
    *,
    n_steps: int = DEFAULT_ODE_STEPS,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    exact_divergence: Optional[bool] = None,
    exact_dim: int = DEFAULT_EXACT_DIM,
    n_hutchinson: int = DEFAULT_HUTCHINSON_SAMPLES,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
    return_trajectory: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Log density ``log p_0(x)`` from the probability-flow ODE.

    Returns per-sample log densities, or ``(log_prob, x_T, times)`` when
    ``return_trajectory`` is True.  Integration runs data -> noise
    (``t`` from ``t_min`` to ``t_max``); the terminal log density is the prior
    of the SDE.

    Uses :func:`simformer.diffusion.log_likelihood_from_ode` when possible and
    falls back to a self-contained Euler integration otherwise.
    """
    sde_obj = _as_sde(sde)
    x_arr = np.asarray(x, dtype=float)
    x_arr = np.atleast_2d(x_arr)
    if rng is None:
        rng = np.random.default_rng(seed)

    use_exact = exact_divergence
    if use_exact is None:
        use_exact = x_arr.shape[-1] <= exact_dim

    def wrapped(xv: np.ndarray, tv: Any) -> np.ndarray:
        return _call_score_fn(score_fn, xv, tv)

    # Preferred path: the shared implementation in diffusion.py.
    try:
        out = log_likelihood_from_ode(
            wrapped,
            sde_obj,
            x_arr,
            n_steps=n_steps,
            t_min=t_min,
            t_max=t_max,
        )
        if isinstance(out, tuple):
            log_p = np.asarray(out[0], dtype=float)
            traj = np.asarray(out[1], dtype=float) if len(out) > 1 else None
            times = np.asarray(out[2], dtype=float) if len(out) > 2 else None
        else:
            log_p, traj, times = np.asarray(out, dtype=float), None, None
        if log_p.shape != x_arr.shape[:-1]:
            log_p = np.reshape(log_p, x_arr.shape[:-1])
        if return_trajectory:
            return log_p, traj, times
        return log_p
    except Exception:
        pass  # fall back to the local Euler integrator

    # ---- self-contained Euler integration of the probability-flow ODE ----
    times = np.linspace(t_min, t_max, int(n_steps) + 1)
    x_cur = x_arr.copy()
    log_p = _terminal_log_prob(sde_obj, x_cur) if False else np.zeros(x_arr.shape[:-1])
    trajs: List[np.ndarray] = []

    for k in range(len(times) - 1):
        t0, t1 = float(times[k]), float(times[k + 1])
        score = _call_score_fn(score_fn, x_cur, t0)

        def drift_fn(xv: np.ndarray, _t: float = t0, _s: np.ndarray = score) -> np.ndarray:
            # probability flow drift f - 0.5 g^2 s with the cached score noise
            return np.asarray(probability_flow_drift(sde_obj, xv, _t, _s), dtype=float)

        drift = np.asarray(probability_flow_drift(sde_obj, x_cur, t0, score), dtype=float)
        if use_exact:
            div = _exact_divergence(lambda xv: drift_fn(xv), x_cur)
        else:
            div = np.zeros(x_cur.shape[:-1])
            for _ in range(max(1, int(n_hutchinson))):
                v = rng.standard_normal(x_cur.shape)
                jvp = (
                    drift_fn(x_cur + 1e-4 * v) - drift_fn(x_cur - 1e-4 * v)
                ) / (2e-4)
                div = div + np.sum(jvp * v, axis=-1)
            div = div / max(1, int(n_hutchinson))
        log_p = log_p + div * (t1 - t0)
        x_cur = x_cur + drift * (t1 - t0)
        if return_trajectory:
            trajs.append(x_cur.copy())

    return None  # pragma: no cover


def log_likelihood(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    x: ArrayLike,
    sde: Union[str, SDE, None] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Per-sample log density ``log p(x)`` (alias of the ODE evaluator)."""
    out = probability_flow_log_likelihood(score_fn, x, sde, **kwargs)
    if isinstance(out, tuple):
        return np.asarray(out[0], dtype=float)
    return np.asarray(out, dtype=float)


#: Alias kept for API symmetry with ``sampling``/``guidance`` modules.
log_prob = log_likelihood


def negative_log_likelihood(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    x: ArrayLike,
    sde: Union[str, SDE, None] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Per-sample negative log density ``-log p(x)``."""
    return -log_likelihood(score_fn, x, sde, **kwargs)


#: Short alias.
nll = negative_log_likelihood


def average_nll(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    x: ArrayLike,
    sde: Union[str, SDE, None] = None,
    *,
    n_steps: int = DEFAULT_ODE_STEPS,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[float, NLLResult]:
    """Average negative log density over the rows of ``x`` (Appendix A3.1)."""
    vals = negative_log_likelihood(score_fn, x, sde, n_steps=n_steps, **kwargs)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    mean = float(np.mean(vals)) if vals.size else float("nan")
    se = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    if not return_result:
        return mean
    sde_obj = _as_sde(sde)
    return NLLResult(
        nll=mean,
        n_samples=int(vals.size),
        n_steps=int(n_steps),
        per_sample=vals,
        std_error=se,
        sde=getattr(sde_obj, "name", None),
    )


def evaluate_nll(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    samples: ArrayLike,
    sde: Union[str, SDE, None] = None,
    *,
    n_steps: int = DEFAULT_ODE_STEPS,
    return_result: bool = False,
    config: Optional[Union[NLLConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Union[float, NLLResult]:
    """Paper-default entry point: average NLL via the probability-flow ODE.

    ``samples`` may be joint samples ``(theta, x)`` (evaluating the joint
    density used in Fig. A8) or any conditional target vector; the density is
    the one modelled by the (possibly transformed) ``score_fn``.
    """
    cfg = _config(config, **kwargs)
    if n_steps != DEFAULT_ODE_STEPS:
        cfg.n_steps = int(n_steps)
    x = np.asarray(samples, dtype=float)
    x = np.atleast_2d(x)
    if cfg.n_samples is not None and x.shape[0] > int(cfg.n_samples):
        x = x[: int(cfg.n_samples)]
    return average_nll(
        score_fn,
        x,
        sde,
        n_steps=cfg.n_steps,
        return_result=return_result,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        exact_divergence=cfg.exact_divergence,
        exact_dim=cfg.exact_dim,
        n_hutchinson=cfg.n_hutchinson,
        seed=cfg.seed,
    )


def nll_many(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    samples: ArrayLike,
    sde: Union[str, SDE, None] = None,
    *,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[List[float], List[NLLResult]]:
    """Per-observation NLL values (one entry per row of ``samples``)."""
    vals = negative_log_likelihood(score_fn, samples, sde, **kwargs)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    if not return_result:
        return [float(v) for v in vals]
    sde_obj = _as_sde(sde)
    return [
        NLLResult(nll=float(v), n_samples=1, n_steps=int(kwargs.get("n_steps", DEFAULT_ODE_STEPS)), sde=getattr(sde_obj, "name", None))
        for v in vals
    ]


def aggregate_nll(values: Sequence[float], *, weights: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Mean/std/median summary of per-target NLL values (Fig. A8 error bars)."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0.0}
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w is not None and w.size == arr.size:
        wsum = float(np.sum(w))
        mean = float(np.sum(w * arr) / wsum) if wsum else float("nan")
        var = float(np.sum(w * (arr - mean) ** 2) / wsum) if wsum else float("nan")
    else:
        mean = float(np.mean(arr))
        var = float(np.var(arr))
    return {
        "mean": mean,
        "std": float(math.sqrt(max(var, 0.0))),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": float(arr.size),
    }


#: Alias used by tables/scripts.
summarize = aggregate_nll


# ---------------------------------------------------------------------------
# composite joint log-density (p(theta, x)) via the probability-flow ODE
# ---------------------------------------------------------------------------
def joint_log_density(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    joint: ArrayLike,
    sde: Union[str, SDE, None] = None,
    **kwargs: Any,
) -> np.ndarray:
    """``log p(theta, x)`` of the joint Simformer model (ODE-based)."""
    return log_likelihood(score_fn, joint, sde, **kwargs)


def conditional_log_likelihood(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    condition_values: ArrayLike,
    targets: ArrayLike,
    sde: Union[str, SDE, None] = None,
    *,
    condition_dim: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """``log p(targets | condition_values)`` up to an additive constant.

    The score function must already be the (pre-computed) conditional score;
    this helper simply forms the full joint vector with the conditioned
    coordinates clamped to their values and evaluates the ODE density.
    """
    cond = np.asarray(condition_values, dtype=float)
    tgt = np.asarray(targets, dtype=float)
    tgt = np.atleast_2d(tgt)
    cond = np.atleast_1d(cond).astype(float)
    if cond.ndim == 1:
        cond = np.broadcast_to(cond, (tgt.shape[0], cond.shape[0]))
    full = np.concatenate([np.asarray(cond, dtype=float), tgt], axis=-1)
    return log_likelihood(score_fn, full, sde, **kwargs)


def posterior_log_likelihood(
    sampler_or_score_fn: Any,
    x_obs: ArrayLike,
    theta: ArrayLike,
    sde: Union[str, SDE, None] = None,
    **kwargs: Any,
) -> np.ndarray:
    """``log p(theta | x_obs)`` via the probability-flow ODE.

    ``sampler_or_score_fn`` may be a :class:`simformer.sampling.ConditionalSampler`
    (posterior conditioning) or a plain score function already conditioned on
    ``x_obs``.
    """
    score_fn = _resolve_conditional_score(sampler_or_score_fn, x_obs, mode="posterior")
    return log_likelihood(score_fn, theta, sde or getattr(sampler_or_score_fn, "sde", None), **kwargs)


def likelihood_log_likelihood(
    sampler_or_score_fn: Any,
    theta: ArrayLike,
    x: ArrayLike,
    sde: Union[str, SDE, None] = None,
    **kwargs: Any,
) -> np.ndarray:
    """``log p(x | theta)`` via the probability-flow ODE."""
    score_fn = _resolve_conditional_score(sampler_or_score_fn, theta, mode="likelihood")
    return log_likelihood(score_fn, x, sde or getattr(sampler_or_score_fn, "sde", None), **kwargs)


def _resolve_conditional_score(obj: Any, condition_values: ArrayLike, *, mode: str) -> Callable:
    """Build a NumPy score function from a sampler or a bare callable."""
    if callable(obj) and not hasattr(obj, "score_fn"):
        return obj
    try:
        from ..sampling import make_score_fn  # local import: avoids circular import
    except Exception:  # pragma: no cover
        return obj

    model = getattr(obj, "model", obj)
    sde = getattr(obj, "sde", None)
    tokenizer = getattr(obj, "tokenizer", None)
    attn = getattr(obj, "attention_mask", None)
    dim = getattr(obj, "dim", None)
    cond = np.asarray(condition_values, dtype=float)
    cond = np.atleast_2d(cond)
    if cond.shape[0] == 1 and dim is not None:
        cond = np.broadcast_to(cond, (1, cond.shape[-1]))
    if dim is not None and cond.shape[-1] != dim and tokenizer is not None:
        # condition values may cover only the observed block: pad with zeros
        full = np.zeros((cond.shape[0], dim))
        idx = getattr(obj, "_condition_indices", None)
        if idx is None:
            idx = list(range(cond.shape[-1]))
        full[:, np.asarray(idx, dtype=int)] = cond
        cond = full
    return make_score_fn(
        model,
        sde,
        condition_mask=None,
        condition_values=cond,
        attention_mask=attn,
        dim=dim,
        tokenizer=tokenizer,
    )


def nll_from_score_fn(
    score_fn: Callable[[ArrayLike, Any], ArrayLike],
    samples: ArrayLike,
    sde: Union[str, SDE, None] = None,
    *,
    n_steps: int = DEFAULT_ODE_STEPS,
    n_samples: Optional[int] = DEFAULT_N_JOINT_SAMPLES,
    return_result: bool = False,
    seed: int = 0,
    **kwargs: Any,
) -> Union[float, NLLResult]:
    """Average NLL with the paper's 5000-sample protocol (Fig. A8)."""
    x = np.asarray(samples, dtype=float)
    x = np.atleast_2d(x)
    if n_samples is not None and x.shape[0] > int(n_samples):
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=int(n_samples), replace=False)
        x = x[np.sort(idx)]
    return average_nll(score_fn, x, sde, n_steps=n_steps, return_result=return_result, seed=seed, **kwargs)


def evaluate_posterior_nll(
    sampler: Any,
    x_obs: ArrayLike,
    theta_true: ArrayLike,
    *,
    n_steps: int = DEFAULT_ODE_STEPS,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[float, NLLResult]:
    """Average NLL of true parameter draws under the approximate posterior."""
    score_fn = _resolve_conditional_score(sampler, x_obs, mode="posterior")
    sde = getattr(sampler, "sde", None)
    return average_nll(score_fn, theta_true, sde, n_steps=n_steps, return_result=return_result, **kwargs)


def evaluate_likelihood_nll(
    sampler: Any,
    theta: ArrayLike,
    x_true: ArrayLike,
    *,
    n_steps: int = DEFAULT_ODE_STEPS,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[float, NLLResult]:
    """Average NLL of true data draws under the approximate likelihood."""
    score_fn = _resolve_conditional_score(sampler, theta, mode="likelihood")
    sde = getattr(sampler, "sde", None)
    return average_nll(score_fn, x_true, sde, n_steps=n_steps, return_result=return_result, **kwargs)


def nll_table(results: Dict[str, Union[float, Sequence[float], NLLResult]]) -> Dict[str, Dict[str, float]]:
    """Summarize ``{method_name: nll_or_values}`` into a benchmark table."""
    table: Dict[str, Dict[str, float]] = {}
    for name, val in results.items():
        if isinstance(val, NLLResult):
            table[name] = {
                "mean": float(val.nll),
                "std": float(val.std_error or 0.0),
                "n": float(val.n_samples),
            }
        elif np.isscalar(val):
            table[name] = {"mean": float(val), "std": 0.0, "n": 1.0}
        else:
            table[name] = aggregate_nll(np.asarray(val, dtype=float).reshape(-1))
    return table
