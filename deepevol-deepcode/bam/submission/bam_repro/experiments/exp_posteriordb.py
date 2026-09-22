"""Section 5.2 / Figure 5.3 (+ Appendix E.6): PosteriorDB real-data experiments.

This module drives the three hierarchical-Bayes posterior targets of
Section 5.2 of *Batch and Match*:

    * ``ark``                  (D = 7,  Gaussian posterior)
    * ``gp-pois-regr``         (D = 13)
    * ``eight-schools-centered`` (D = 10)

For each target we compare

    * **BaM**  with the decaying learning-rate schedule ``lam_t = B D / (t + 1)``
      at batch sizes ``B in {8, 32}``, and
    * the four baselines **ADVI** (Algorithm 2, negative ELBO + Adam), **Score**
      (score-based divergence + Adam), **Fisher** (Fisher divergence + Adam) and
      **GSM** (Algorithm 3, per-sample score matching) at the same batch sizes.

Following the paper the x-axis is the number of *gradient evaluations* (wallclock
timing is explicitly out of scope), the metrics are the **relative mean error**
``||(mu - mu_hat) / sigma||_2`` and the **relative SD error**
``||(sigma - sigma_hat) / sigma||_2`` measured against HMC reference moments, and
every configuration is repeated ``n_runs = 5`` times with standard errors
reported.

The targets themselves (BridgeStan / cached HMC draws / a clearly flagged
Gaussian surrogate) live in :mod:`bam_repro.targets.posteriordb_target` and the
metrics live in :mod:`bam_repro.metrics.posteriordb_metrics`; this file is only
the experiment driver.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Relative imports with a fallback to absolute imports so the module can be run
# both as ``python -m bam_repro.experiments.exp_posteriordb`` and as a script.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import shim
    from ..bam.bam import BaM
    from ..bam.learning_rate import make_schedule
    from ..bam.vi_base import init_gaussian_state
    from ..baselines.advi import ADVI
    from ..baselines.fisher_advi import FisherADVI
    from ..baselines.gsm import GSM
    from ..baselines.score_advi import ScoreADVI
    from ..metrics.posteriordb_metrics import (
        error_curve,
        relative_errors,
        summarize_errors,
    )
    from ..targets.posteriordb_target import (
        PAPER_POSTERIORDB_BATCH_SIZES,
        PAPER_POSTERIORDB_MODELS,
        PAPER_POSTERIORDB_NAMES,
        PAPER_POSTERIORDB_N_RUNS,
        PAPER_INIT_MEAN_SCALE,
        load_posteriordb_target,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from bam_repro.bam.bam import BaM  # type: ignore
    from bam_repro.bam.learning_rate import make_schedule  # type: ignore
    from bam_repro.bam.vi_base import init_gaussian_state  # type: ignore
    from bam_repro.baselines.advi import ADVI  # type: ignore
    from bam_repro.baselines.fisher_advi import FisherADVI  # type: ignore
    from bam_repro.baselines.gsm import GSM  # type: ignore
    from bam_repro.baselines.score_advi import ScoreADVI  # type: ignore
    from bam_repro.metrics.posteriordb_metrics import (  # type: ignore
        error_curve,
        relative_errors,
        summarize_errors,
    )
    from bam_repro.targets.posteriordb_target import (  # type: ignore
        PAPER_POSTERIORDB_BATCH_SIZES,
        PAPER_POSTERIORDB_MODELS,
        PAPER_POSTERIORDB_NAMES,
        PAPER_POSTERIORDB_N_RUNS,
        PAPER_INIT_MEAN_SCALE,
        load_posteriordb_target,
    )


# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

#: Models of Section 5.2 in paper order (re-exported for convenience).
PAPER_POSTERIORDB_MODEL_LIST: Tuple[str, ...] = tuple(PAPER_POSTERIORDB_NAMES or PAPER_POSTERIORDB_NAMES)

#: Batch sizes used for *both* BaM and the baselines in Figure 5.3.
PAPER_POSTERIORDB_BATCH_SIZES = tuple(PAPER_POSTERIORDB_BATCH_SIZES or (8, 32))

#: Independent repetitions per configuration (Figure 5.3 reports standard errors).
PAPER_POSTERIORDB_N_RUNS: int = int(PAPER_POSTERIORDB_N_RUNS or 5)

#: BaM's decaying schedule: ``lam_t = B D / (t + 1)`` (Section 5.1/5.2).
PAPER_POSTERIORDB_SCHEDULE = "BD/(t+1)"

#: Grid-searched Adam learning rates for the gradient-based baselines
#: (Section 5.2 / Appendix E.6).  ADVI and Fisher use a single value across the
#: three models; Score's rate depends on the model's dimension.
PAPER_POSTERIORDB_ADVI_LR = 0.02
PAPER_POSTERIORDB_FISHER_LR = 0.05
PAPER_POSTERIORDB_SCORE_LR = {
    "ark": 0.01,                # D = 7
    "gp-pois-regr": 0.001,      # D = 13
    "eight-schools-centered": 0.001,  # D = 10
}

#: Initialisation of the variational family: mu_0 ~ Uniform[0, 0.1]^D, Sigma_0 = I.
PAPER_POSTERIORDB_INIT_MEAN_SCALE = float(PAPER_INIT_MEAN_SCALE or 0.1)

#: Gradient-evaluation budgets (the paper's cost axis).  Chosen so every method
#: reaches its plateau while remaining cheap on CPU.
PAPER_POSTERIORDB_GRAD_BUDGET = {
    "ark": 20000,
    "gp-pois-regr": 30000,
    "eight-schools-centered": 30000,
}

#: Methods compared in Figure 5.3 / E.6.
PAPER_POSTERIORDB_METHODS: Tuple[str, ...] = ("bam", "advi", "score", "fisher", "gsm")

#: Number of Monte-Carlo KL samples (only used in the optional divergence panel).
PAPER_POSTERIORDB_KL_SAMPLES = 4096

PAPER_POSTERIORDB_DEFAULTS: Dict[str, Dict[str, Any]] = {
    name: {
        "dim": int(PAPER_POSTERIORDB_MODELS[name]),
        "n_runs": PAPER_POSTERIORDB_N_RUNS,
        "batch_sizes": PAPER_POSTERIORDB_BATCH_SIZES,
        "schedule": PAPER_POSTERIORDB_SCHEDULE,
        "grad_budget": PAPER_POSTERIORDB_GRAD_BUDGET[name],
        "learning_rates": {
            "advi": PAPER_POSTERIORDB_ADVI_LR,
            "fisher": PAPER_POSTERIORDB_FISHER_LR,
            "score": PAPER_POSTERIORDB_SCORE_LR[name],
            "gsm": None,
            "bam": None,
        },
        "init_mean_scale": PAPER_POSTERIORDB_INIT_MEAN_SCALE,
    }
    for name in PAPER_POSTERIORDB_NAMES
}

#: Ordered tuple of the three paper settings.
PAPER_POSTERIORDB_SETTINGS: Tuple[str, ...] = tuple(PAPER_POSTERIORDB_NAMES)


# ---------------------------------------------------------------------------
# Small helpers (kept local to this driver)
# ---------------------------------------------------------------------------


def _filter_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments not accepted by ``cls.__init__``.

    The experiments are written to be robust to minor signature differences
    between the BaM engine and the four baseline learners.
    """
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(kwargs)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _resolve_schedule(schedule: Any, batch_size: int, dim: int) -> Any:
    """Resolve a schedule specification into a callable ``t -> lam_t``."""
    if schedule is None:
        schedule = PAPER_POSTERIORDB_SCHEDULE
    if callable(schedule) and not isinstance(schedule, str):
        return schedule
    if isinstance(schedule, (int, float)):
        return float(schedule)
    try:
        sched = make_schedule(schedule, batch_size=batch_size, dim=dim)
        if callable(sched):
            return sched
    except Exception:  # pragma: no cover - defensive
        pass
    # Manual fallback table for the paper schedules.
    key = str(schedule).replace(" ", "").lower()
    if key in ("bd", "b*d", "bd_t", "constant"):
        return lambda t: float(batch_size * dim)
    if key in ("bd/(t+1)", "bd/t", "bd_over_t"):
        return lambda t: float(batch_size * dim) / float(t + 1)
    if key in ("bd/sqrt(t+1)", "bd_over_sqrt_t"):
        return lambda t: float(batch_size * dim) / math.sqrt(float(t + 1))
    if key in ("b/(t+1)", "b_over_t"):
        return lambda t: float(batch_size) / float(t + 1)
    return lambda t: float(batch_size * dim)


def _extract_history(learner: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Best-effort extraction of the (mu, Sigma, grad_evals) traces."""
    mu_hist = getattr(learner, "mu_history", None)
    sig_hist = getattr(learner, "Sigma_history", None)
    grad_hist = getattr(learner, "grad_evals_history", None)

    hist = getattr(learner, "history", None)
    if isinstance(hist, dict):
        mu_hist = mu_hist if mu_hist is not None else hist.get("mu")
        sig_hist = sig_hist if sig_hist is not None else hist.get("Sigma")
        grad_hist = grad_hist if grad_hist is not None else hist.get("grad_evals")

    def _as_array(x: Any) -> Optional[np.ndarray]:
        if x is None:
            return None
        try:
            arr = np.asarray(x, dtype=np.float64)
        except Exception:  # pragma: no cover - defensive
            return None
        if arr.size == 0:
            return None
        return arr

    return _as_array(mu_hist), _as_array(sig_hist), _as_array(grad_hist)


def _grad_eval_axis(
    n_points: int,
    batch_size: int,
    history_every: int = 1,
    grad_history: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gradient-evaluation abscissa for a recorded trajectory."""
    if grad_history is not None and grad_history.ndim == 1 and grad_history.size == n_points:
        return grad_history.astype(np.float64)
    step = max(int(history_every), 1)
    return batch_size * step * (np.arange(n_points, dtype=np.float64) + 1.0)


def _subsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Uniformly subsample the leading axis of ``arr`` to at most ``max_points``."""
    if arr is None or arr.shape[0] <= max_points:
        return arr
    idx = np.unique(np.linspace(0, arr.shape[0] - 1, max_points).astype(int))
    return arr[idx]


def _setting_dim(setting: str) -> int:
    return int(PAPER_POSTERIORDB_MODELS[setting])


def default_learning_rate(method: str, setting: str) -> Optional[float]:
    """Paper grid-searched Adam learning rate for a baseline (``None`` for BaM/GSM)."""
    method = str(method).lower()
    if method in ("advi", "elbo"):
        return PAPER_POSTERIORDB_ADVI_LR
    if method in ("fisher", "fisher_advi", "fisher_divergence"):
        return PAPER_POSTERIORDB_FISHER_LR
    if method in ("score", "score_advi", "score_divergence"):
        return PAPER_POSTERIORDB_SCORE_LR.get(setting, 0.001)
    return None


def paper_posteriordb_settings(setting: str) -> Dict[str, Any]:
    """Paper-faithful run configuration for one posteriorDB model."""
    if setting not in PAPER_POSTERIORDB_DEFAULTS:
        raise KeyError(
            f"unknown posteriorDB setting {setting!r}; "
            f"expected one of {list(PAPER_POSTERIORDB_DEFAULTS)}"
        )
    return dict(PAPER_POSTERIORDB_DEFAULTS[setting])


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def make_posteriordb_problem(
    setting: str,
    seed: int = 0,
    mu_scale: float = PAPER_POSTERIORDB_INIT_MEAN_SCALE,
    target: Any = None,
    mu0: Optional[np.ndarray] = None,
    Sigma0: Optional[np.ndarray] = None,
    reference_samples: Any = None,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Build ``(target, mu0, Sigma0)`` for one posteriorDB model.

    The variational family is initialised as in the paper:
    ``mu_0 ~ Uniform[0, mu_scale]^D`` and ``Sigma_0 = I``.  The same target
    object is reused across methods *and* across repetitions so that all
    differences come from the optimisation, not from the problem instance.
    """
    if target is None:
        target = load_posteriordb_target(setting, reference_samples=reference_samples, seed=seed)
    dim = int(getattr(target, "dim", None) or _setting_dim(setting))

    rng = np.random.default_rng(seed)
    if mu0 is None or Sigma0 is None:
        state = init_gaussian_state(dim, mu_scale=mu_scale, rng=rng)
        mu0 = np.asarray(state.mu if hasattr(state, "mu") else state.mean, dtype=np.float64)
        Sigma0 = np.asarray(
            state.Sigma if hasattr(state, "Sigma") else state.covariance, dtype=np.float64
        )
    return target, np.asarray(mu0, dtype=np.float64), np.asarray(Sigma0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Learner construction
# ---------------------------------------------------------------------------


def build_learner(
    method: str,
    target: Any,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    batch_size: int,
    seed: int = 0,
    learning_rate: Optional[float] = None,
    schedule: Any = PAPER_POSTERIORDB_SCHEDULE,
    history_every: int = 1,
    dim: Optional[int] = None,
    **extra: Any,
) -> Any:
    """Instantiate one BaM/baseline learner with paper-faithful arguments."""
    method = str(method).lower()
    dim = int(dim if dim is not None else np.asarray(mu0).shape[0])
    score_fn = getattr(target, "score", None) or getattr(target, "grad_log_prob", None)

    if method in ("bam", "batch_and_match"):
        lam = _resolve_schedule(schedule, batch_size, dim)
        kwargs = dict(
            mu0=mu0,
            Sigma0=Sigma0,
            score_fn=score_fn,
            batch_size=batch_size,
            lam=lam,
            seed=seed,
            track_history=True,
            history_every=history_every,
        )
        kwargs.update(extra)
        return BaM(**_filter_kwargs(BaM, kwargs))

    if method in ("gsm", "gsm_redux", "score_matching"):
        kwargs = dict(
            mu0=mu0,
            Sigma0=Sigma0,
            score_fn=score_fn,
            target=target,
            batch_size=batch_size,
            seed=seed,
            track_history=True,
            history_every=history_every,
        )
        kwargs.update(extra)
        return GSM(**_filter_kwargs(GSM, kwargs))

    classes = {
        "advi": ADVI,
        "elbo": ADVI,
        "score": ScoreADVI,
        "score_advi": ScoreADVI,
        "fisher": FisherADVI,
        "fisher_advi": FisherADVI,
    }
    if method not in classes:
        raise ValueError(f"unknown method {method!r}")
    cls = classes[method]

    lr = learning_rate
    if lr is None:
        lr = default_learning_rate(method, str(getattr(target, "name", ""))) or 0.01

    kwargs = dict(
        mu0=mu0,
        Sigma0=Sigma0,
        score_fn=score_fn,
        target=target,
        batch_size=batch_size,
        learning_rate=float(lr),
        seed=seed,
        track_history=True,
        history_every=history_every,
    )
    kwargs.update(extra)
    return cls(**_filter_kwargs(cls, kwargs))


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------


def run_posteriordb_replicate(
    setting: str,
    method: str = "bam",
    batch_size: int = 8,
    seed: int = 0,
    grad_budget: Optional[int] = None,
    n_iter: Optional[int] = None,
    learning_rate: Optional[float] = None,
    schedule: Any = PAPER_POSTERIORDB_SCHEDULE,
    mu_scale: float = PAPER_POSTERIORDB_INIT_MEAN_SCALE,
    target: Any = None,
    reference_samples: Any = None,
    mu0: Optional[np.ndarray] = None,
    Sigma0: Optional[np.ndarray] = None,
    history_points: int = 120,
    history_every: Optional[int] = None,
    return_learner: bool = False,
    **learner_kwargs: Any,
) -> Dict[str, Any]:
    """Run one posteriorDB replicate and record its error trajectory.

    Returns a record with the gradient-evaluation axis plus the relative mean
    and SD error curves (``||(mu - mu_hat)/sigma||_2`` and
    ``||(sigma - sigma_hat)/sigma||_2``) evaluated against HMC moments.
    """
    if target is None:
        target = load_posteriordb_target(setting, reference_samples=reference_samples, seed=seed)
    dim = int(getattr(target, "dim", None) or _setting_dim(setting))

    if mu0 is None or Sigma0 is None:
        _, mu0, Sigma0 = make_posteriordb_problem(
            setting, seed=seed, mu_scale=mu_scale, target=target
        )

    budget = int(grad_budget or PAPER_POSTERIORDB_GRAD_BUDGET.get(setting, 20000))
    batch_size = max(int(batch_size), 1)
    n_iter = int(n_iter) if n_iter is not None else max(int(math.ceil(budget / batch_size)), 1)
    if history_every is None:
        history_every = max(int(math.ceil(n_iter / max(int(history_points), 1))), 1)

    learner = build_learner(
        method,
        target,
        mu0,
        Sigma0,
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
        try:
            result = learner.run(n_iter)
        except TypeError:  # pragma: no cover - signature difference
            result = learner.run(T=n_iter)
    else:  # pragma: no cover - every learner exposes ``run``
        result = learner
    wallclock = time.time() - t0

    if result is not None and hasattr(result, "mu") and hasattr(result, "Sigma"):
        mu_final = np.asarray(result.mu, dtype=np.float64)
        Sigma_final = np.asarray(result.Sigma, dtype=np.float64)
    elif hasattr(result, "mu") and hasattr(result, "covariance"):  # pragma: no cover
        mu_final = np.asarray(result.mu, dtype=np.float64)
        Sigma_final = np.asarray(result.covariance(), dtype=np.float64)
    else:  # pragma: no cover - fall back to the learner's live state
        mu_final = np.asarray(learner.mean, dtype=np.float64)
        cov = learner.covariance
        Sigma_final = np.asarray(cov() if callable(cov) else cov, dtype=np.float64)

    reference = getattr(target, "reference_samples", None) or reference_samples
    if reference is None:
        # No HMC draws available: fall back to the target's own moments so the
        # metrics stay computable (clearly a surrogate / diagnostic quantity).
        ref_mom = target.reference_moments()
        reference = ref_mom
    ref_arr = np.asarray(
        reference[0] if isinstance(reference, tuple) else reference, dtype=np.float64
    )

    # --- trajectory metrics -------------------------------------------------
    mu_hist, sig_hist, grad_hist = _extract_history(learner)
    if mu_hist is None:
        mu_hist = mu_final[None, :]
    if mu_hist.ndim == 1:
        mu_hist = mu_hist[None, :]
    mu_hist = _subsample(mu_hist, int(history_points))

    if sig_hist is not None and sig_hist.shape[0] >= mu_hist.shape[0]:
        sig_hist = _subsample(sig_hist, mu_hist.shape[0])
    else:
        sig_hist = None

    x_axis = _grad_eval_axis(
        mu_hist.shape[0],
        batch_size,
        history_every=history_every,
        grad_history=grad_hist,
    )

    if hasattr(target, "reference_moments"):
        ref_mean, ref_sd = target.reference_moments()
    else:  # pragma: no cover - duck-typed reference object
        ref_mean, ref_sd = np.mean(ref_arr, axis=0), np.std(ref_arr, axis=0, ddof=1)

    curves = error_curve(
        (np.asarray(ref_mean, dtype=np.float64), np.asarray(ref_sd, dtype=np.float64)),
        mu_history=mu_hist,
        Sigma_history=sig_hist,
        grad_evals=x_axis,
        model=setting,
    )

    errs = relative_errors(
        (np.asarray(ref_mean, dtype=np.float64), np.asarray(ref_sd, dtype=np.float64)),
        mu_hat=mu_final,
        cov_hat=Sigma_final,
        model=setting,
    )

    record: Dict[str, Any] = {
        "setting": setting,
        "method": str(method),
        "batch_size": batch_size,
        "seed": int(seed),
        "n_iter": int(n_iter),
        "grad_evals": x_axis.astype(np.float64),
        "relative_mean_error": np.asarray(curves["relative_mean_error"], dtype=np.float64),
        "relative_sd_error": np.asarray(curves["relative_sd_error"], dtype=np.float64),
        "final_relative_mean_error": float(errs.relative_mean_error),
        "final_relative_sd_error": float(errs.relative_sd_error),
        "mu": mu_final,
        "Sigma": Sigma_final,
        "wallclock_seconds": float(wallclock),
        "is_surrogate": bool(getattr(target, "is_surrogate", False)),
        "backend": str(getattr(target, "backend", "unknown")),
    }
    if return_learner:
        record["learner"] = learner
    return record


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------


@dataclass
class PosteriorDBExperimentResult:
    """Aggregated results of the Section 5.2 posteriorDB sweep."""

    settings: Sequence[str]
    methods: Sequence[str]
    n_runs: int
    config: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    curves: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    # -- aggregation ------------------------------------------------------
    def aggregate(self) -> "PosteriorDBExperimentResult":
        """Group runs by (setting, method, batch size) and summarize errors."""
        grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
        for rec in self.records:
            key = (rec["setting"], rec["method"], int(rec["batch_size"]))
            grouped.setdefault(key, []).append(rec)

        summary: Dict[str, Any] = {}
        curves: Dict[str, Any] = {}
        for (setting, method, bsize), recs in grouped.items():
            name = f"{setting}|{method}|B{bsize}"
            summary[name] = {
                "setting": setting,
                "method": method,
                "batch_size": bsize,
                "n_runs": len(recs),
                "relative_mean_error": summarize_errors(
                    [r["final_relative_mean_error"] for r in recs]
                ),
                "relative_sd_error": summarize_errors(
                    [r["final_relative_sd_error"] for r in recs]
                ),
            }
            # interpolate per-run curves onto a common gradient-eval grid
            grid = np.unique(
                np.concatenate([np.asarray(r["grad_evals"], dtype=np.float64) for r in recs])
            )
            rme = np.stack(
                [
                    np.interp(grid, np.asarray(r["grad_evals"], dtype=np.float64),
                              np.asarray(r["relative_mean_error"], dtype=np.float64))
                    for r in recs
                ]
            )
            rsde = np.stack(
                [
                    np.interp(grid, np.asarray(r["grad_evals"], dtype=np.float64),
                              np.asarray(r["relative_sd_error"], dtype=np.float64))
                    for r in recs
                ]
            )
            curves[name] = {
                "setting": setting,
                "method": method,
                "batch_size": bsize,
                "grad_evals": grid,
                "relative_mean_error_mean": rme.mean(axis=0),
                "relative_mean_error_stderr": _stderr(rme, axis=0),
                "relative_sd_error_mean": rsde.mean(axis=0),
                "relative_sd_error_stderr": _stderr(rsde, axis=0),
            }
        self.summary = summary
        self.curves = curves
        return self

    # -- reporting --------------------------------------------------------
    def table(self) -> str:
        if not self.summary:
            self.aggregate()
        header = (
            f"{'setting':<24}{'method':<8}{'B':>4}"
            f"{'mean err':>12}{'sd err':>12}{'runs':>6}"
        )
        lines = [header, "-" * len(header)]
        for name in sorted(self.summary):
            row = self.summary[name]
            m = row["relative_mean_error"]["mean"]
            s = row["relative_sd_error"]["mean"]
            lines.append(
                f"{row['setting']:<24}{row['method']:<8}{row['batch_size']:>4}"
                f"{m:>12.4f}{s:>12.4f}{row['n_runs']:>6}"
            )
        return "\n".join(lines)

    def to_dict(self, include_runs: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "settings": list(self.settings),
            "methods": list(self.methods),
            "n_runs": int(self.n_runs),
            "config": _jsonify(self.config),
            "summary": _jsonify(self.summary),
            "curves": _jsonify(self.curves),
        }
        if include_runs:
            out["records"] = [
                {k: v for k, v in rec.items() if k != "learner"} for rec in self.records
            ]
            out["records"] = _jsonify(out["records"])
        return out

    def save(self, path: str) -> str:
        if not self.summary:
            self.aggregate()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(include_runs=True), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "PosteriorDBExperimentResult":
        with open(path) as fh:
            data = json.load(fh)
        res = cls(
            settings=data.get("settings", []),
            methods=data.get("methods", []),
            n_runs=data.get("n_runs", 0),
            config=data.get("config", {}),
            records=data.get("records", []),
        )
        res.aggregate()
        return res

    # -- figures ----------------------------------------------------------
    def figure(
        self,
        outdir: Optional[str] = None,
        metric: str = "mean",
        show: bool = False,
        ylabel: Optional[str] = None,
        xlabel: str = "gradient evaluations",
    ) -> List[str]:
        """Plot relative error vs gradient evaluations for each model.

        ``metric`` selects ``"mean"`` (relative mean error, Figure 5.3) or
        ``"sd"`` (relative SD error, Appendix E.6).
        """
        if not self.curves:
            self.aggregate()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        key = "relative_mean_error" if metric == "mean" else "relative_sd_error"
        label = ylabel or (
            r"$\|(\mu-\hat\mu)/\sigma\|_2$"
            if metric == "mean"
            else r"$\|(\sigma-\hat\sigma)/\sigma\|_2$"
        )

        outdir = outdir or "."
        os.makedirs(outdir, exist_ok=True)
        paths: List[str] = []

        for setting in self.settings:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            entries = [v for v in self.curves.values() if v["setting"] == setting]
            entries.sort(key=lambda e: (e["method"], e["batch_size"]))
            for entry in entries:
                x = np.asarray(entry["grad_evals"], dtype=np.float64)
                y = np.asarray(entry[key + "_mean"], dtype=np.float64)
                se = np.asarray(entry[key + "_stderr"], dtype=np.float64)
                style = _style(entry["method"], entry["batch_size"])
                ax.loglog(x, y, label=style["label"], color=style["color"],
                          linestyle=style["linestyle"], linewidth=1.3)
                if se.size == y.size:
                    ax.fill_between(
                        x, np.maximum(y - se, 1e-16), y + se,
                        color=style["color"], alpha=0.15, linewidth=0,
                    )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            ax.set_title(f"{setting} (D={_setting_dim(setting)})")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            path = os.path.join(outdir, f"fig_posteriordb_{metric}_{setting}.png")
            fig.savefig(path, dpi=150)
            if not show:
                plt.close(fig)
            paths.append(path)
        return paths


def _stderr(a: np.ndarray, axis: int = 0) -> np.ndarray:
    n = a.shape[axis]
    if n <= 1:
        return np.zeros_like(a.mean(axis=axis))
    return np.std(a, axis=axis, ddof=1) / math.sqrt(n)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


_METHOD_STYLES: Dict[str, Dict[str, str]] = {
    "bam": {"color": "C3", "label": "BaM"},
    "advi": {"color": "C0", "label": "ADVI"},
    "score": {"color": "C2", "label": "Score"},
    "fisher": {"color": "C1", "label": "Fisher"},
    "gsm": {"color": "C4", "label": "GSM"},
}
_BATCH_LINESTYLES = {8: "-", 32: "--"}


def _style(method: str, batch_size: int) -> Dict[str, str]:
    base = dict(_METHOD_STYLES.get(str(method).lower(), {"color": "C5", "label": str(method)}))
    base["linestyle"] = _BATCH_LINESTYLES.get(int(batch_size), ":")
    base["label"] = f"{base['label']} (B={batch_size})"
    return base


def run_posteriordb_experiment(
    settings: Sequence[str] = PAPER_POSTERIORDB_SETTINGS,
    n_runs: int = PAPER_POSTERIORDB_N_RUNS,
    methods: Sequence[str] = PAPER_POSTERIORDB_METHODS,
    batch_sizes: Sequence[int] = PAPER_POSTERIORDB_BATCH_SIZES,
    grad_budget: Optional[int] = None,
    budget_scale: float = 1.0,
    mu_scale: float = PAPER_POSTERIORDB_INIT_MEAN_SCALE,
    reference_samples: Any = None,
    seed: int = 0,
    history_points: int = 120,
    schedule: Any = PAPER_POSTERIORDB_SCHEDULE,
    learning_rates: Optional[Dict[str, Dict[str, float]]] = None,
    outdir: Optional[str] = None,
    save: bool = True,
    figures: bool = True,
    verbose: bool = True,
    **learner_kwargs: Any,
) -> PosteriorDBExperimentResult:
    """Run the full Section 5.2 sweep over the three posteriorDB models.

    For every (model, method, batch size) combination the experiment performs
    ``n_runs`` independent repetitions; each method uses its own paper
    grid-searched Adam learning rate (BaM is hyperparameter-free with
    ``lam_t = B D/(t+1)``).
    """
    settings = list(settings)
    methods = list(methods)
    batch_sizes = [int(b) for b in batch_sizes]

    result = PosteriorDBExperimentResult(
        settings=settings,
        methods=methods,
        n_runs=int(n_runs),
        config={
            "batch_sizes": batch_sizes,
            "grad_budget": grad_budget,
            "budget_scale": float(budget_scale),
            "schedule": str(schedule),
            "mu_scale": float(mu_scale),
            "learning_rates": learning_rates or "paper",
        },
    )

    for setting in settings:
        target, mu0_base, Sigma0_base = make_posteriordb_problem(
            setting, seed=seed, mu_scale=mu_scale, reference_samples=reference_samples
        )
        conf = paper_posteriordb_settings(setting)
        budget = int(
            grad_budget if grad_budget is not None else conf["grad_budget"] * float(budget_scale)
        )
        if verbose:
            print(
                f"[posteriordb] {setting} (D={_setting_dim(setting)}) "
                f"budget={budget} grad evals"
            )

        for method in methods:
            lr = None
            if learning_rates and method in learning_rates:
                lr = learning_rates[method].get(setting, learning_rates[method])
                if isinstance(lr, dict):  # pragma: no cover - nested spec
                    lr = lr.get(setting)
            for bsize in batch_sizes:
                for run in range(int(n_runs)):
                    run_seed = int(seed + 1000 * run + 7 * bsize + 13 * _setting_dim(setting))
                    rec = run_posteriordb_replicate(
                        setting,
                        method=method,
                        batch_size=bsize,
                        seed=run_seed,
                        grad_budget=budget,
                        learning_rate=lr,
                        schedule=schedule,
                        mu_scale=mu_scale,
                        target=target,
                        mu0=mu0_base,
                        Sigma0=Sigma0_base,
                        history_points=history_points,
                        **learner_kwargs,
                    )
                    result.records.append(rec)
                    if verbose:
                        print(
                            f"  {method:<6} B={bsize:<4} run {run + 1}/{n_runs} "
                            f"rme={rec['final_relative_mean_error']:.4f} "
                            f"rsde={rec['final_relative_sd_error']:.4f}"
                        )

    result.aggregate()

    if save:
        outdir = outdir or "./results"
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "posteriordb_results.json")
        result.save(path)
        if verbose:
            print(f"[posteriordb] saved {path}")

    if figures:
        outdir = outdir or "./results"
        for metric in ("mean", "sd"):
            paths = result.figure(outdir=outdir, metric=metric)
            if verbose:
                print(f"[posteriordb] wrote {len(paths)} {metric}-error figure(s)")

    return result


def run_posteriordb(quick: bool = False, **kwargs: Any) -> PosteriorDBExperimentResult:
    """Convenience wrapper; ``quick=True`` for a cheap smoke test."""
    if quick:
        kwargs.setdefault("settings", ("ark",))
        kwargs.setdefault("n_runs", 1)
        kwargs.setdefault("batch_sizes", (8,))
        kwargs.setdefault("methods", ("bam", "advi"))
        kwargs.setdefault("budget_scale", 0.05)
        kwargs.setdefault("figures", False)
        kwargs.setdefault("save", False)
        kwargs.setdefault("verbose", True)
    return run_posteriordb_experiment(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BaM Section 5.2 posteriorDB experiments (Figure 5.3 / E.6)"
    )
    parser.add_argument("--settings", nargs="+", default=list(PAPER_POSTERIORDB_SETTINGS))
    parser.add_argument("--runs", type=int, default=PAPER_POSTERIORDB_N_RUNS)
    parser.add_argument("--methods", nargs="+", default=list(PAPER_POSTERIORDB_METHODS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(PAPER_POSTERIORDB_BATCH_SIZES))
    parser.add_argument("--grad-budget", type=int, default=None)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--history-points", type=int, default=120)
    parser.add_argument("--schedule", type=str, default=PAPER_POSTERIORDB_SCHEDULE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    run_posteriordb(
        quick=args.quick,
        settings=tuple(args.settings),
        n_runs=args.runs,
        methods=tuple(args.methods),
        batch_sizes=tuple(args.batch_sizes),
        grad_budget=args.grad_budget,
        budget_scale=args.budget_scale,
        history_points=args.history_points,
        schedule=args.schedule,
        seed=args.seed,
        outdir=args.outdir,
        save=not args.no_save,
        figures=not args.no_figures,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
