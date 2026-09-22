"""Section 5.1 (Figure 5.2) and Appendix E.4 (Figure E.4): non-Gaussian targets.

Targets are sinh-arcsinh normal distributions with ``D = 10`` (Jones & Pewsey,
2009), obtained by transforming a base Gaussian ``y ~ N(mu, Sigma)`` elementwise
via

    z = sinh( (1/tau) * ( sinh^{-1}(y) + s ) ),

where ``s`` controls the skew and ``tau > 0`` the tail weight (the Gaussian is
recovered for ``s = 0, tau = 1``).  Two sweeps are considered (paper Sec 5.1):

  * normal tails (``tau = 1``) with varying skew ``s = 0.2, 1.0, 1.8``;
  * no skew (``s = 0``) with varying tails ``tau = 0.1, 0.9, 1.7``.

For BaM we use the *decaying* schedule ``lambda_t = B D / (t + 1)`` (some decay
is necessary for BaM to converge on non-Gaussian targets).  ADVI and GSM use a
batch size of ``B = 5``; the gradient-based baselines use the paper's
grid-searched ADAM learning rates: 0.02 for ADVI, 0.05 for Fisher, and
``[0.01, 0.001, 0.001]`` (varying skew) / ``[0.001, 0.01, 0.01]`` (varying
tails) for Score.  All algorithms start from a random mean ``mu_0`` and
``Sigma_0 = I``; curves are averaged over 10 runs with the cost axis given by
the number of gradient evaluations (wallclock timings are out of scope).

Metrics: Monte-Carlo estimates of the forward KL ``KL(p ; q)`` and reverse KL
``KL(q ; p)``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# imports (robust to being run as a script or as part of the package)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import shim
    from ..bam.bam import BaM
    from ..bam.learning_rate import make_schedule
    from ..bam.vi_base import init_gaussian_state
    from ..baselines.advi import ADVI
    from ..baselines.fisher_advi import FisherADVI
    from ..baselines.gsm import GSM
    from ..baselines.score_advi import ScoreADVI
    from ..targets.sinh_arcsinh import (
        PAPER_DIM,
        PAPER_SKEW_VALUES,
        PAPER_TAIL_VALUES,
        SinhArcsinhTarget,
        random_sinh_arcsinh_target,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_HERE))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from bam_repro.bam.bam import BaM  # type: ignore
    from bam_repro.bam.learning_rate import make_schedule  # type: ignore
    from bam_repro.bam.vi_base import init_gaussian_state  # type: ignore
    from bam_repro.baselines.advi import ADVI  # type: ignore
    from bam_repro.baselines.fisher_advi import FisherADVI  # type: ignore
    from bam_repro.baselines.gsm import GSM  # type: ignore
    from bam_repro.baselines.score_advi import ScoreADVI  # type: ignore
    from bam_repro.targets.sinh_arcsinh import (  # type: ignore
        PAPER_DIM,
        PAPER_SKEW_VALUES,
        PAPER_TAIL_VALUES,
        SinhArcsinhTarget,
        random_sinh_arcsinh_target,
    )

try:  # metrics are optional at import time (fallbacks below)
    from ..metrics.kl_metrics import kl_divergence as _kl_divergence
except ImportError:  # pragma: no cover
    try:
        from bam_repro.metrics.kl_metrics import kl_divergence as _kl_divergence  # type: ignore
    except ImportError:  # pragma: no cover
        _kl_divergence = None


__all__ = [
    "PAPER_NON_GAUSSIAN_DIM",
    "PAPER_SKEW_SETTINGS",
    "PAPER_TAIL_SETTINGS",
    "PAPER_NON_GAUSSIAN_BATCH_SIZES",
    "PAPER_NON_GAUSSIAN_N_RUNS",
    "PAPER_NON_GAUSSIAN_METHODS",
    "PAPER_NON_GAUSSIAN_GRAD_BUDGET",
    "PAPER_NON_GAUSSIAN_SETTINGS",
    "NonGaussianExperimentResult",
    "paper_non_gaussian_settings",
    "default_learning_rate",
    "make_non_gaussian_problem",
    "build_learner",
    "evaluate_kl",
    "run_non_gaussian_replicate",
    "run_non_gaussian_experiment",
    "run_non_gaussian",
    "main",
]


# --------------------------------------------------------------------------- #
# paper settings
# --------------------------------------------------------------------------- #
PAPER_NON_GAUSSIAN_DIM: int = int(PAPER_DIM)  # 10

#: normal tails (tau = 1), varying skew s -- Table of Sec 5.1 / App E.4
PAPER_SKEW_SETTINGS: Dict[str, Dict[str, Any]] = {
    "skew0.2": {"s": 0.2, "tau": 1.0, "score_lr": 0.01},
    "skew1.0": {"s": 1.0, "tau": 1.0, "score_lr": 0.001},
    "skew1.8": {"s": 1.8, "tau": 1.0, "score_lr": 0.001},
}

#: no skew (s = 0), varying tail weight tau -- Sec 5.1 / App E.4
PAPER_TAIL_SETTINGS: Dict[str, Dict[str, Any]] = {
    "tail0.1": {"s": 0.0, "tau": 0.1, "score_lr": 0.001},
    "tail0.9": {"s": 0.0, "tau": 0.9, "score_lr": 0.01},
    "tail1.7": {"s": 0.0, "tau": 1.7, "score_lr": 0.01},
}

#: ordered list of all six target settings (paper order)
PAPER_NON_GAUSSIAN_SETTINGS: Tuple[str, ...] = tuple(PAPER_SKEW_SETTINGS) + tuple(
    PAPER_TAIL_SETTINGS
)

#: BaM uses several batch sizes; ADVI/Score/Fisher/GSM use B = 5 (App E.4)
PAPER_NON_GAUSSIAN_BASELINE_BATCH_SIZE = 5
PAPER_NON_GAUSSIAN_BATCH_SIZES: Dict[str, Tuple[int, ...]] = {
    "bam": (2, 5, 10, 20, 40),
    "baselines": (PAPER_NON_GAUSSIAN_BASELINE_BATCH_SIZE,),
}

PAPER_NON_GAUSSIAN_N_RUNS: int = 10

PAPER_NON_GAUSSIAN_METHODS: Tuple[str, ...] = ("bam", "advi", "score", "fisher", "gsm")

#: BaM schedule: decaying inverse regularization (paper Sec 5.1)
PAPER_NON_GAUSSIAN_SCHEDULE: str = "BD/(t+1)"

#: grid-searched ADAM learning rates (App E.4)
PAPER_NON_GAUSSIAN_ADVI_LR: float = 0.02
PAPER_NON_GAUSSIAN_FISHER_LR: float = 0.05

#: gradient-evaluation budgets per target setting (cost axis = grad evals)
PAPER_NON_GAUSSIAN_GRAD_BUDGET: Dict[str, int] = {
    "skew0.2": 20000,
    "skew1.0": 30000,
    "skew1.8": 40000,
    "tail0.1": 20000,
    "tail0.9": 20000,
    "tail1.7": 30000,
}

#: all algorithms are initialised with mu_0 ~ Uniform[0, mu_scale], Sigma_0 = I
PAPER_NON_GAUSSIAN_INIT_MEAN_SCALE: float = 0.1

#: number of Monte-Carlo samples for the KL estimates along each trajectory
PAPER_NON_GAUSSIAN_KL_SAMPLES: int = 4096


def _all_settings() -> List[str]:
    return list(PAPER_NON_GAUSSIAN_SETTINGS)


def setting_params(setting: str) -> Dict[str, Any]:
    """Return the (s, tau, score_lr) dictionary for a named target setting."""
    if setting in PAPER_SKEW_SETTINGS:
        return dict(PAPER_SKEW_SETTINGS[setting])
    if setting in PAPER_TAIL_SETTINGS:
        return dict(PAPER_TAIL_SETTINGS[setting])
    raise KeyError(f"unknown non-Gaussian setting {setting!r}")


def paper_non_gaussian_settings(setting: str) -> Dict[str, Any]:
    """Paper-faithful run settings for one sinh-arcsinh target setting."""
    params = setting_params(setting)
    return {
        "setting": setting,
        "dim": PAPER_NON_GAUSSIAN_DIM,
        "s": params["s"],
        "tau": params["tau"],
        "n_runs": PAPER_NON_GAUSSIAN_N_RUNS,
        "init_mean_scale": PAPER_NON_GAUSSIAN_INIT_MEAN_SCALE,
        "bam_batch_sizes": PAPER_NON_GAUSSIAN_BATCH_SIZES["bam"],
        "baseline_batch_size": PAPER_NON_GAUSSIAN_BASELINE_BATCH_SIZE,
        "schedule": PAPER_NON_GAUSSIAN_SCHEDULE,
        "grad_budget": PAPER_NON_GAUSSIAN_GRAD_BUDGET.get(setting, 30000),
        "learning_rates": {
            "advi": PAPER_NON_GAUSSIAN_ADVI_LR,
            "fisher": PAPER_NON_GAUSSIAN_FISHER_LR,
            "score": params["score_lr"],
            "gsm": None,
        },
        "kl_samples": PAPER_NON_GAUSSIAN_KL_SAMPLES,
    }


def default_learning_rate(method: str, setting: str) -> Optional[float]:
    """Grid-searched ADAM learning rate used by the paper for ``method``."""
    method = str(method).lower()
    if method in ("bam", "gsm"):
        return None  # BaM uses lambda_t; GSM has no learning rate
    if method == "advi":
        return PAPER_NON_GAUSSIAN_ADVI_LR
    if method == "fisher":
        return PAPER_NON_GAUSSIAN_FISHER_LR
    if method == "score":
        return float(setting_params(setting)["score_lr"])
    raise KeyError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# problem construction
# --------------------------------------------------------------------------- #
def make_non_gaussian_problem(
    setting: str,
    dim: int = PAPER_NON_GAUSSIAN_DIM,
    seed: int = 0,
    mu_scale: float = PAPER_NON_GAUSSIAN_INIT_MEAN_SCALE,
    target: Optional[Any] = None,
    mu0: Optional[np.ndarray] = None,
    Sigma0: Optional[np.ndarray] = None,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Build ``(target, mu_0, Sigma_0)`` for a named sinh-arcsinh setting.

    The base Gaussian of the transform is drawn once per setting (with the
    run seed) exactly like the non-Gaussian construction of Section 5.1; the
    initialisation follows the paper: random mean ``mu_0`` and ``Sigma_0 = I``.
    """
    params = setting_params(setting)
    rng = np.random.default_rng(seed)

    if target is None:
        # Seed the *target* independently of the run seed so that every method
        # and every run faces the same target distribution.
        target_rng = np.random.default_rng(12345)
        target = random_sinh_arcsinh_target(
            dim=dim, s=params["s"], tau=params["tau"], rng=target_rng
        )

    if mu0 is None:
        state = init_gaussian_state(
            dim, mean=None, cov=None, mu_scale=mu_scale, rng=rng, xp=np
        )
        mu0 = np.asarray(state.mu, dtype=np.float64)
    else:
        mu0 = np.asarray(mu0, dtype=np.float64)

    if Sigma0 is None:
        Sigma0 = np.eye(dim, dtype=np.float64)
    else:
        Sigma0 = np.asarray(Sigma0, dtype=np.float64)

    return target, mu0, Sigma0


# --------------------------------------------------------------------------- #
# learner construction
# --------------------------------------------------------------------------- #
def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs not accepted by ``cls.__init__`` (signature introspection)."""
    import inspect

    try:
        sig = inspect.signature(cls)
        accepted = set(sig.parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _resolve_schedule(schedule: Any, batch_size: int, dim: int) -> Any:
    """Resolve a schedule specification to something callable ``t -> lambda_t``."""
    if schedule is None:
        return None
    if callable(schedule):
        return schedule
    if isinstance(schedule, (int, float)):
        return float(schedule)
    name = str(schedule)
    try:
        return make_schedule(name, batch_size=batch_size, dim=dim)
    except Exception:  # pragma: no cover - manual fallback
        bd = float(batch_size) * float(dim)
        table = {
            "BD": lambda t: bd,
            "B": lambda t: float(batch_size),
            "BD/(t+1)": lambda t: bd / (t + 1.0),
            "BD/sqrt(t+1)": lambda t: bd / math.sqrt(t + 1.0),
            "B/(t+1)": lambda t: float(batch_size) / (t + 1.0),
        }
        if name not in table:
            raise
        return table[name]


def build_learner(
    method: str,
    target: Any,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    batch_size: int,
    seed: int = 0,
    learning_rate: Optional[float] = None,
    schedule: Any = PAPER_NON_GAUSSIAN_SCHEDULE,
    history_every: int = 1,
    dim: Optional[int] = None,
    **extra: Any,
) -> Any:
    """Instantiate a BaM or baseline learner for the sinh-arcsinh target."""
    method = str(method).lower()
    dim = int(dim if dim is not None else len(np.asarray(mu0)))
    common: Dict[str, Any] = {
        "seed": seed,
        "track_history": True,
        "history_every": int(history_every),
    }

    if method == "bam":
        lam = _resolve_schedule(schedule, batch_size, dim)
        kwargs = dict(
            mu0=np.asarray(mu0, dtype=np.float64),
            Sigma0=np.asarray(Sigma0, dtype=np.float64),
            score_fn=target.score,
            batch_size=int(batch_size),
            lam=lam,
            **common,
        )
        return BaM(**_filter_kwargs(BaM, kwargs))

    # The sinh-arcsinh target exposes ``log_prob``; gradient-based baselines
    # need either a target object or both score/log-prob callables.
    base_kwargs: Dict[str, Any] = dict(
        mu0=np.asarray(mu0, dtype=np.float64),
        Sigma0=np.asarray(Sigma0, dtype=np.float64),
        score_fn=target.score,
        log_prob_fn=getattr(target, "log_prob", None),
        target=target,
        batch_size=int(batch_size),
        learning_rate=(
            float(learning_rate)
            if learning_rate is not None
            else (0.02 if method in ("advi", "score", "fisher") else 0.01)
        ),
        **common,
    )

    if method == "advi":
        return ADVI(**_filter_kwargs(ADVI, base_kwargs))
    if method == "score":
        return ScoreADVI(**_filter_kwargs(ScoreADVI, base_kwargs))
    if method == "fisher":
        return FisherADVI(**_filter_kwargs(FisherADVI, base_kwargs))
    if method == "gsm":
        gsm_kwargs: Dict[str, Any] = dict(
            mu0=np.asarray(mu0, dtype=np.float64),
            Sigma0=np.asarray(Sigma0, dtype=np.float64),
            score_fn=target.score,
            target=target,
            batch_size=int(batch_size),
            **common,
        )
        return GSM(**_filter_kwargs(GSM, gsm_kwargs))
    raise KeyError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# metrics along a trajectory
# --------------------------------------------------------------------------- #
def _as_gaussian(mu: Any, Sigma: Any) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    return mu, Sigma


def evaluate_kl(
    target: Any,
    mu: Any,
    Sigma: Any,
    direction: str = "forward",
    n_samples: int = PAPER_NON_GAUSSIAN_KL_SAMPLES,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
) -> float:
    """Monte-Carlo KL between the sinh-arcsinh target and ``q = N(mu, Sigma)``.

    Uses the target's own MC helpers when available, otherwise falls back to
    ``metrics.kl_metrics.kl_divergence``.
    """
    mu, Sigma = _as_gaussian(mu, Sigma)
    if rng is None:
        rng = np.random.default_rng(seed)

    attr = "mc_forward_kl" if direction == "forward" else "mc_reverse_kl"
    fn = getattr(target, attr, None)
    if callable(fn):
        try:
            value = fn(mu, Sigma, int(n_samples))
        except TypeError:  # pragma: no cover - signature differences
            value = fn(mu, Sigma)
        return float(value)

    if _kl_divergence is not None:
        res = _kl_divergence(
            target, (mu, Sigma), direction=direction, n_samples=int(n_samples), rng=rng
        )
        return float(res)

    raise RuntimeError("no KL routine available for the sinh-arcsinh target")


# --------------------------------------------------------------------------- #
# history extraction (robust to learner attribute naming)
# --------------------------------------------------------------------------- #
def _extract_history(learner: Any) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    mus = getattr(learner, "mu_history", None)
    Sigmas = getattr(learner, "Sigma_history", None)
    if mus is None or Sigmas is None:
        hist = getattr(learner, "history", None)
        if isinstance(hist, dict):
            mus = hist.get("mu") or hist.get("mu_history")
            Sigmas = hist.get("Sigma") or hist.get("Sigma_history")
    if mus is None or Sigmas is None:  # pragma: no cover
        raise AttributeError("learner does not expose a (mu, Sigma) history")

    def _fix(h: Any, final: Any) -> List[np.ndarray]:
        h = list(h)
        if h and isinstance(h[0], dict):  # pragma: no cover
            h = [x.get("mu", x.get("Sigma")) for x in h]
        if not h:
            h = [np.asarray(final)]
        elif isinstance(h[0], (int, float)):
            h = [np.asarray(final)]
        return [np.asarray(x, dtype=np.float64) for x in h]

    return _fix(mus, getattr(learner, "mu", np.zeros(1))), _fix(
        Sigmas, getattr(learner, "Sigma", np.eye(1))
    )


def _grad_eval_axis(
    learner: Any, n_points: int, batch_size: int, history_every: int
) -> np.ndarray:
    """Gradient-evaluation abscissa for a subsampled trajectory."""
    history = getattr(learner, "grad_evals_history", None)
    if history is not None and len(history) == n_points:
        return np.asarray(history, dtype=np.float64)
    return (np.arange(n_points, dtype=np.float64) + 1.0) * float(
        batch_size * max(1, int(history_every))
    )


def _subsample(n_points: int, out_points: int) -> np.ndarray:
    if out_points is None or out_points <= 0 or n_points <= out_points:
        return np.arange(n_points)
    return np.unique(
        np.linspace(0, n_points - 1, int(out_points)).round().astype(int)
    )


# --------------------------------------------------------------------------- #
# single replicate
# --------------------------------------------------------------------------- #
def run_non_gaussian_replicate(
    setting: str,
    method: str = "bam",
    batch_size: int = 20,
    seed: int = 0,
    grad_budget: Optional[int] = None,
    n_iter: Optional[int] = None,
    learning_rate: Optional[float] = None,
    schedule: Any = PAPER_NON_GAUSSIAN_SCHEDULE,
    mu_scale: float = PAPER_NON_GAUSSIAN_INIT_MEAN_SCALE,
    dim: int = PAPER_NON_GAUSSIAN_DIM,
    target: Optional[Any] = None,
    mu0: Optional[np.ndarray] = None,
    Sigma0: Optional[np.ndarray] = None,
    history_points: int = 120,
    kl_samples: int = PAPER_NON_GAUSSIAN_KL_SAMPLES,
    return_learner: bool = False,
    **learner_kwargs: Any,
) -> Dict[str, Any]:
    """Run one replicate and return KL curves against gradient evaluations."""
    method = str(method).lower()
    batch_size = int(batch_size)
    if learning_rate is None:
        learning_rate = default_learning_rate(method, setting)

    tgt, mu_init, Sigma_init = make_non_gaussian_problem(
        setting,
        dim=dim,
        seed=seed + 1,  # init RNG (target itself is fixed across runs)
        mu_scale=mu_scale,
        target=target,
        mu0=mu0,
        Sigma0=Sigma0,
    )

    budget = int(grad_budget if grad_budget is not None else PAPER_NON_GAUSSIAN_GRAD_BUDGET.get(setting, 30000))
    if n_iter is None:
        n_iter = max(1, int(round(budget / float(batch_size))))

    # Subsample the trajectory to at most ``history_points`` snapshots.
    history_every = max(1, int(math.ceil(n_iter / float(max(1, history_points)))))

    learner = build_learner(
        method,
        tgt,
        mu_init,
        Sigma_init,
        batch_size=batch_size,
        seed=seed,
        learning_rate=learning_rate,
        schedule=schedule,
        history_every=history_every,
        dim=dim,
        **learner_kwargs,
    )

    t0 = time.time()
    if hasattr(learner, "run"):
        learner.run(int(n_iter))
    else:  # pragma: no cover
        for _ in range(int(n_iter)):
            learner.step()
    wallclock = time.time() - t0

    mus, Sigmas = _extract_history(learner)
    keep = _subsample(len(mus), int(history_points))
    mus = [mus[i] for i in keep]
    Sigmas = [Sigmas[i] for i in keep]
    grad_evals = _grad_eval_axis(learner, len(mus), batch_size, history_every)[keep]

    # Evaluate KL on the target's own RNG for reproducibility across methods.
    kl_rng = np.random.default_rng(1000 + seed)
    forward, reverse = [], []
    for mu_k, Sigma_k in zip(mus, Sigmas):
        forward.append(
            evaluate_kl(
                tgt, mu_k, Sigma_k, "forward", n_samples=kl_samples, rng=kl_rng
            )
        )
        reverse.append(
            evaluate_kl(
                tgt, mu_k, Sigma_k, "reverse", n_samples=kl_samples, rng=kl_rng
            )
        )

    final = getattr(learner, "result", None)
    final = final() if callable(final) else None
    mu_final = np.asarray(
        getattr(learner, "mu", getattr(final, "mu", mus[-1])), dtype=np.float64
    )
    Sigma_final = np.asarray(
        getattr(learner, "Sigma", getattr(final, "Sigma", Sigmas[-1])), dtype=np.float64
    )

    record: Dict[str, Any] = {
        "setting": setting,
        "dim": int(dim),
        "method": method,
        "batch_size": batch_size,
        "seed": int(seed),
        "learning_rate": learning_rate,
        "schedule": str(schedule),
        "n_iter": int(n_iter),
        "grad_evals": np.asarray(grad_evals, dtype=np.float64),
        "forward_kl": np.asarray(forward, dtype=np.float64),
        "reverse_kl": np.asarray(reverse, dtype=np.float64),
        "final_forward_kl": float(forward[-1]),
        "final_reverse_kl": float(reverse[-1]),
        "best_forward_kl": float(np.min(forward)),
        "best_reverse_kl": float(np.min(reverse)),
        "mu": mu_final,
        "Sigma": Sigma_final,
        "wallclock_seconds": float(wallclock),
        "diverged": bool(
            (not np.all(np.isfinite(forward)))
            or (not np.all(np.isfinite(reverse)))
            or float(np.max(np.abs(forward))) > 1e6
        ),
    }
    if return_learner:
        record["learner"] = learner
    return record


# --------------------------------------------------------------------------- #
# aggregate result container
# --------------------------------------------------------------------------- #
@dataclass
class NonGaussianExperimentResult:
    """Aggregated results for the sinh-arcsinh sweep (Fig. 5.2 / E.4)."""

    settings: Tuple[str, ...] = tuple(PAPER_NON_GAUSSIAN_SETTINGS)
    methods: Tuple[str, ...] = PAPER_NON_GAUSSIAN_METHODS
    n_runs: int = PAPER_NON_GAUSSIAN_N_RUNS
    config: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    curves: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    # -- aggregation ------------------------------------------------------- #
    def aggregate(self) -> "NonGaussianExperimentResult":
        """Build mean/std summary curves from the per-run records."""
        groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
        for rec in self.records:
            key = (rec["setting"], rec["method"], int(rec["batch_size"]))
            groups.setdefault(key, []).append(rec)

        summary: Dict[str, Any] = {}
        curves: Dict[str, Any] = {}
        for (setting, method, batch), recs in groups.items():
            n_pts = min(len(r["grad_evals"]) for r in recs)
            x = np.asarray(recs[0]["grad_evals"][:n_pts], dtype=np.float64)
            fwd = np.stack([np.asarray(r["forward_kl"][:n_pts]) for r in recs])
            rev = np.stack([np.asarray(r["reverse_kl"][:n_pts]) for r in recs])

            fwd_mean, fwd_std = fwd.mean(0), fwd.std(0, ddof=1 if len(recs) > 1 else 0)
            rev_mean, rev_std = rev.mean(0), rev.std(0, ddof=1 if len(recs) > 1 else 0)
            fwd_sem = fwd_std / math.sqrt(max(1, len(recs)))
            rev_sem = rev_std / math.sqrt(max(1, len(recs)))

            tag = f"{setting}|{method}|B{batch}"
            curves[tag] = {
                "setting": setting,
                "method": method,
                "batch_size": int(batch),
                "grad_evals": x,
                "forward_kl": fwd_mean,
                "forward_kl_std": fwd_std,
                "forward_kl_sem": fwd_sem,
                "reverse_kl": rev_mean,
                "reverse_kl_std": rev_std,
                "reverse_kl_sem": rev_sem,
            }
            summary[tag] = {
                "setting": setting,
                "method": method,
                "batch_size": int(batch),
                "n_runs": len(recs),
                "final_forward_kl": float(fwd_mean[-1]),
                "final_forward_kl_sem": float(fwd_sem[-1]),
                "final_reverse_kl": float(rev_mean[-1]),
                "final_reverse_kl_sem": float(rev_sem[-1]),
                "best_forward_kl": float(np.min(fwd_mean)),
                "best_reverse_kl": float(np.min(rev_mean)),
                "n_diverged": int(sum(1 for r in recs if r.get("diverged"))),
                "mean_wallclock_seconds": float(
                    np.mean([r["wallclock_seconds"] for r in recs])
                ),
                "grad_evals": float(x[-1]),
            }

        self.curves = curves
        self.summary = summary
        return self

    # -- reporting --------------------------------------------------------- #
    def table(self) -> str:
        """Pretty-printed per-(setting, method, batch) summary table."""
        if not self.summary:
            self.aggregate()
        lines = [
            f"{'setting':>10} {'method':>7} {'B':>4} {'runs':>5} "
            f"{'fwd KL':>12} {'rev KL':>12} {'div':>4}"
        ]
        lines.append("-" * len(lines[0]))
        for key in sorted(self.summary, key=_summary_sort_key):
            s = self.summary[key]
            lines.append(
                f"{s['setting']:>10} {s['method']:>7} {s['batch_size']:>4} "
                f"{s['n_runs']:>5} {s['final_forward_kl']:>12.4e} "
                f"{s['final_reverse_kl']:>12.4e} {s['n_diverged']:>4}"
            )
        return "\n".join(lines)

    def to_dict(self, include_runs: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "settings": list(self.settings),
            "methods": list(self.methods),
            "n_runs": int(self.n_runs),
            "config": self.config,
            "summary": self.summary
            if self.summary
            else (self.aggregate().summary),
        }
        if include_runs:
            runs = []
            for rec in self.records:
                run = {k: v for k, v in rec.items() if k not in ("learner",)}
                for k, v in list(run.items()):
                    if isinstance(v, np.ndarray):
                        run[k] = v.tolist()
                runs.append(run)
            payload["records"] = runs
        return payload

    def save(self, path: str) -> str:
        """Serialise results to JSON (arrays converted to lists)."""
        if not self.summary:
            self.aggregate()
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(include_runs=True), fh, indent=2, default=float)
        return path

    @classmethod
    def load(cls, path: str) -> "NonGaussianExperimentResult":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        res = cls(
            settings=tuple(payload.get("settings", PAPER_NON_GAUSSIAN_SETTINGS)),
            methods=tuple(payload.get("methods", PAPER_NON_GAUSSIAN_METHODS)),
            n_runs=int(payload.get("n_runs", PAPER_NON_GAUSSIAN_N_RUNS)),
            config=payload.get("config", {}),
            records=payload.get("records", []),
            summary=payload.get("summary", {}),
        )
        for rec in res.records:
            for k in ("grad_evals", "forward_kl", "reverse_kl", "mu", "Sigma"):
                if k in rec and isinstance(rec[k], list):
                    rec[k] = np.asarray(rec[k], dtype=np.float64)
        res.aggregate()
        return res

    # -- figures ----------------------------------------------------------- #
    def figure(
        self,
        outdir: Optional[str] = None,
        metric: str = "forward",
        group: str = "skew",
        show: bool = False,
    ) -> List[str]:
        """Figure 5.2 (forward KL) / E.4 (reverse KL): one panel per setting.

        ``group`` selects the sweep, ``"skew"`` (tau = 1) or ``"tail"`` (s = 0).
        """
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            print(f"[exp_non_gaussian] matplotlib unavailable: {exc}")
            return []

        if not self.curves:
            self.aggregate()

        if group == "skew":
            settings = list(PAPER_SKEW_SETTINGS)
        elif group == "tail":
            settings = list(PAPER_TAIL_SETTINGS)
        else:  # pragma: no cover
            settings = list(self.settings)

        attr = "forward_kl" if metric == "forward" else "reverse_kl"
        outdir = outdir or os.path.join(os.getcwd(), "results")
        os.makedirs(outdir, exist_ok=True)

        ncol = 3
        nrow = int(math.ceil(len(settings) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
        colors = {
            "bam": "tab:blue",
            "advi": "tab:orange",
            "score": "tab:green",
            "fisher": "tab:red",
            "gsm": "tab:purple",
        }

        for ax, setting in zip(axes.ravel(), settings):
            for tag, curve in sorted(self.curves.items(), key=lambda kv: _curve_sort_key(kv[1])):
                if curve["setting"] != setting:
                    continue
                method = curve["method"]
                label = method + (f" (B={curve['batch_size']})" if method == "bam" else "")
                ax.plot(
                    curve["grad_evals"],
                    curve[attr],
                    color=colors.get(method, None),
                    lw=1.6,
                    label=label,
                )
                # also draw the individual runs transparently when available
                for rec in self.records:
                    if (
                        rec["setting"] == setting
                        and rec["method"] == method
                        and int(rec["batch_size"]) == int(curve["batch_size"])
                    ):
                        ax.plot(
                            rec["grad_evals"],
                            rec[attr],
                            color=colors.get(method, None),
                            lw=0.7,
                            alpha=0.15,
                        )
            params = setting_params(setting)
            ax.set_title(f"s={params['s']:g}, tau={params['tau']:g}")
            ax.set_xlabel("gradient evaluations")
            ax.set_ylabel(f"{metric} KL")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        for ax in axes.ravel()[len(settings):]:  # pragma: no cover
            ax.axis("off")

        fig.suptitle(
            f"sinh-arcsinh targets (D={PAPER_NON_GAUSSIAN_DIM}), {metric} KL "
            f"vs gradient evaluations"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = os.path.join(outdir, f"fig_non_gaussian_{metric}_kl_{group}.png")
        fig.savefig(path, dpi=150)
        if show:  # pragma: no cover
            plt.show()
        plt.close(fig)
        return [path]


def _summary_sort_key(s: Dict[str, Any]) -> Tuple[int, int, str]:
    order = {m: i for i, m in enumerate(PAPER_NON_GAUSSIAN_METHODS)}
    return (order.get(s["method"], 99), int(s["batch_size"]), s["setting"])


def _curve_sort_key(c: Dict[str, Any]) -> Tuple[int, int]:
    order = {m: i for i, m in enumerate(PAPER_NON_GAUSSIAN_METHODS)}
    return (order.get(c["method"], 99), int(c["batch_size"]))


# --------------------------------------------------------------------------- #
# full sweep
# --------------------------------------------------------------------------- #
def run_non_gaussian_experiment(
    settings: Sequence[str] = PAPER_NON_GAUSSIAN_SETTINGS,
    n_runs: int = PAPER_NON_GAUSSIAN_N_RUNS,
    methods: Sequence[str] = PAPER_NON_GAUSSIAN_METHODS,
    bam_batch_sizes: Sequence[int] = PAPER_NON_GAUSSIAN_BATCH_SIZES["bam"],
    baseline_batch_size: int = PAPER_NON_GAUSSIAN_BASELINE_BATCH_SIZE,
    grad_budget: Optional[int] = None,
    budget_scale: float = 1.0,
    mu_scale: float = PAPER_NON_GAUSSIAN_INIT_MEAN_SCALE,
    dim: int = PAPER_NON_GAUSSIAN_DIM,
    seed: int = 0,
    history_points: int = 120,
    kl_samples: int = PAPER_NON_GAUSSIAN_KL_SAMPLES,
    schedule: Any = PAPER_NON_GAUSSIAN_SCHEDULE,
    learning_rates: Optional[Dict[str, float]] = None,
    outdir: Optional[str] = None,
    save: bool = True,
    figures: bool = True,
    verbose: bool = True,
    **learner_kwargs: Any,
) -> NonGaussianExperimentResult:
    """Run the full Section 5.1 / E.4 sinh-arcsinh sweep over the six targets."""
    settings = tuple(settings)
    methods = tuple(str(m).lower() for m in methods)
    learning_rates = dict(learning_rates or {})

    config = {
        "settings": list(settings),
        "methods": list(methods),
        "n_runs": int(n_runs),
        "dim": int(dim),
        "bam_batch_sizes": [int(b) for b in bam_batch_sizes],
        "baseline_batch_size": int(baseline_batch_size),
        "grad_budget": grad_budget,
        "budget_scale": float(budget_scale),
        "mu_scale": float(mu_scale),
        "schedule": str(schedule),
        "history_points": int(history_points),
        "kl_samples": int(kl_samples),
        "metric": "forward/reverse KL (Monte-Carlo)",
    }

    result = NonGaussianExperimentResult(
        settings=settings, methods=methods, n_runs=int(n_runs), config=config
    )

    if verbose:
        print("[exp_non_gaussian] settings:", ", ".join(settings))
        print("[exp_non_gaussian] methods :", ", ".join(methods))

    for setting in settings:
        params = setting_params(setting)
        base_budget = int(
            grad_budget
            if grad_budget is not None
            else PAPER_NON_GAUSSIAN_GRAD_BUDGET.get(setting, 30000)
        )
        budget = max(1, int(round(base_budget * float(budget_scale))))

        # One target per setting, shared by every method/run (paper protocol).
        tgt, _, _ = make_non_gaussian_problem(setting, dim=dim, seed=seed)

        for method in methods:
            if method == "bam":
                batches = [int(b) for b in bam_batch_sizes]
            else:
                batches = [int(baseline_batch_size)]

            for batch_size in batches:
                lr = learning_rates.get(method, default_learning_rate(method, setting))
                for run in range(int(n_runs)):
                    run_seed = int(seed + 1000 * run + 7 * batch_size)
                    rec = run_non_gaussian_replicate(
                        setting,
                        method=method,
                        batch_size=batch_size,
                        seed=run_seed,
                        grad_budget=budget,
                        learning_rate=lr,
                        schedule=schedule,
                        mu_scale=mu_scale,
                        dim=dim,
                        target=tgt,
                        history_points=history_points,
                        kl_samples=kl_samples,
                        **learner_kwargs,
                    )
                    result.records.append(rec)
                if verbose:
                    sub = [r for r in result.records if r["setting"] == setting
                           and r["method"] == method and int(r["batch_size"]) == batch_size]
                    print(
                        f"[exp_non_gaussian] {setting:>9} s={params['s']:g} "
                        f"tau={params['tau']:g} | {method:>6} B={batch_size:<3} "
                        f"| fwd KL={np.mean([r['final_forward_kl'] for r in sub]):.4e} "
                        f"rev KL={np.mean([r['final_reverse_kl'] for r in sub]):.4e} "
                        f"| diverged={sum(r['diverged'] for r in sub)}/{len(sub)}"
                    )

    result.aggregate()

    if save:
        outdir = outdir or os.path.join(os.getcwd(), "results")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "non_gaussian_results.json")
        result.save(path)
        if verbose:
            print(f"[exp_non_gaussian] saved {path}")

    if figures:
        outdir = outdir or os.path.join(os.getcwd(), "results")
        for metric in ("forward", "reverse"):
            for group in ("skew", "tail"):
                for p in result.figure(outdir=outdir, metric=metric, group=group):
                    if verbose:
                        print(f"[exp_non_gaussian] figure {p}")

    return result


def run_non_gaussian(quick: bool = False, **kwargs: Any) -> NonGaussianExperimentResult:
    """Convenience wrapper (``quick=True`` runs a small smoke test)."""
    if quick:
        kwargs.setdefault("n_runs", 2)
        kwargs.setdefault("budget_scale", 0.05)
        kwargs.setdefault("bam_batch_sizes", (2,))
        kwargs.setdefault("settings", ("skew0.2", "tail0.9"))
        kwargs.setdefault("history_points", 30)
        kwargs.setdefault("kl_samples", 512)
        kwargs.setdefault("figures", False)
    return run_non_gaussian_experiment(**kwargs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BaM Section 5.1 / E.4: sinh-arcsinh (non-Gaussian) targets"
    )
    parser.add_argument("--settings", nargs="*", default=list(PAPER_NON_GAUSSIAN_SETTINGS))
    parser.add_argument("--runs", type=int, default=PAPER_NON_GAUSSIAN_N_RUNS)
    parser.add_argument("--methods", nargs="*", default=list(PAPER_NON_GAUSSIAN_METHODS))
    parser.add_argument(
        "--bam-batch-sizes",
        nargs="*",
        type=int,
        default=list(PAPER_NON_GAUSSIAN_BATCH_SIZES["bam"]),
    )
    parser.add_argument(
        "--baseline-batch-size", type=int, default=PAPER_NON_GAUSSIAN_BASELINE_BATCH_SIZE
    )
    parser.add_argument("--grad-budget", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=PAPER_NON_GAUSSIAN_DIM)
    parser.add_argument("--history-points", type=int, default=120)
    parser.add_argument("--kl-samples", type=int, default=PAPER_NON_GAUSSIAN_KL_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    params = dict(
        settings=tuple(args.settings),
        n_runs=None if args.quick else args.runs,
        methods=tuple(args.methods),
        bam_batch_sizes=tuple(args.bam_batch_sizes),
        baseline_batch_size=args.baseline_batch_size,
        grad_budget=args.grad_budget,
        budget_scale=args.budget_scale,
        dim=args.dim,
        seed=args.seed,
        history_points=args.history_points,
        kl_samples=args.kl_samples,
        outdir=args.outdir,
        save=not args.no_save,
        figures=not args.no_figures,
    )
    if args.quick:
        defaults = dict(
            n_runs=2, budget_scale=0.05, bam_batch_sizes=(2,),
            settings=("skew0.2", "tail0.9"), history_points=30, kl_samples=512,
            figures=False,
        )
        defaults.update({k: v for k, v in params.items() if v is not None})
        params = defaults

    result = run_non_gaussian_experiment(**params)
    print(result.table())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
