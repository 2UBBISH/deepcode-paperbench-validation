"""Gaussian-target experiment (Section 5.1, Figure 5.1 and Figure E.3).

Paper specification
-------------------
Sections 5.1 and E.3:

* Target distributions are Gaussian ``p = N(mu_star, Sigma_star)`` with
  ``Sigma_star = A A^T`` for a randomly generated ``D x D`` matrix ``A``.
* Dimensions ``D = 4, 16, 64, 256``.
* ``mu_0 ~ Uniform[0, 0.1]`` and ``Sigma_0 = I`` for every method.
* BaM uses a **constant** learning rate ``lambda_t = B D``.
* ADVI, Score, Fisher and GSM use a batch size of ``B = 2``.
  The batch size for BaM is given in the legend (``B = 20, 40``).
* Gradient-based learning rates are chosen by a grid search:
  ADVI ``0.01``, Fisher ``0.01`` and Score
  ``[0.01, 0.005, 0.001, 0.001]`` for ``D = 4, 16, 64, 256``.
* Curves are the mean over **10 runs** (individual runs plotted as
  transparent curves) of the forward KL ``KL(p ; q)`` (Figure 5.1) and the
  reverse KL ``KL(q ; p)`` (Figure E.3) against the number of **gradient
  evaluations** (the paper's cost axis; wallclock is out of scope).

This module runs the sweep, aggregates the per-run curves (mean and standard
error) and writes both a machine-readable JSON summary and the figures.
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
# imports (work both as a package and as a stand-alone script)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import style depends on how the module is executed
    from ..bam.bam import BaM
    from ..bam.learning_rate import make_schedule
    from ..baselines.advi import ADVI
    from ..baselines.fisher_advi import FisherADVI
    from ..baselines.gsm import GSM
    from ..baselines.score_advi import ScoreADVI
    from ..targets.gaussian_target import GaussianTarget, random_gaussian_target
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from bam_repro.bam.bam import BaM
    from bam_repro.bam.learning_rate import make_schedule
    from bam_repro.baselines.advi import ADVI
    from bam_repro.baselines.fisher_advi import FisherADVI
    from bam_repro.baselines.gsm import GSM
    from bam_repro.baselines.score_advi import ScoreADVI
    from bam_repro.targets.gaussian_target import GaussianTarget, random_gaussian_target


# ---------------------------------------------------------------------------
# Paper settings
# ---------------------------------------------------------------------------
PAPER_DIMS: Tuple[int, ...] = (4, 16, 64, 256)

#: Batch size used by every gradient-based baseline in Figure 5.1 / E.3.
PAPER_GAUSSIAN_BASELINE_BATCH_SIZE: int = 2

#: BaM batch sizes appearing in the Figure 5.1 legend.  Appendix E.2 mentions
#: that for the constant learning rate the curves for ``B = 20`` and ``B = 40``
#: lie on top of each other (that statement is about the D = 16 ablation).
PAPER_GAUSSIAN_BATCH_SIZES: Dict[str, Tuple[int, ...]] = {
    "bam": (20, 40),
    "baselines": (PAPER_GAUSSIAN_BASELINE_BATCH_SIZE,),
}

PAPER_GAUSSIAN_N_RUNS: int = 10

#: Grid-searched learning rates from Appendix E.3.
PAPER_GAUSSIAN_SCORE_LR: Tuple[float, ...] = (0.01, 0.005, 0.001, 0.001)
PAPER_GAUSSIAN_ADVI_LR: float = 0.01
PAPER_GAUSSIAN_FISHER_LR: float = 0.01
PAPER_GAUSSIAN_LR: Dict[str, Any] = {
    "advi": {d: PAPER_GAUSSIAN_ADVI_LR for d in PAPER_DIMS},
    "fisher": {d: PAPER_GAUSSIAN_FISHER_LR for d in PAPER_DIMS},
    "score": {d: lr for d, lr in zip(PAPER_DIMS, PAPER_GAUSSIAN_SCORE_LR)},
}

PAPER_GAUSSIAN_METHODS: Tuple[str, ...] = ("bam", "advi", "score", "fisher", "gsm")

#: Initialisation of every method (Appendix E.3).
PAPER_GAUSSIAN_INIT_MEAN_SCALE: float = 0.1

#: Number of gradient evaluations used for each dimension.  The paper does not
#: state the number of iterations; we choose a budget that lets the
#: gradient-based baselines reach their plateau while keeping the total cost
#: reasonable (the cost axis is gradient evaluations, so the budget only needs
#: to be large enough to expose the differences between methods).
PAPER_GAUSSIAN_GRAD_BUDGET: Dict[int, int] = {4: 20_000, 16: 20_000, 64: 30_000, 256: 40_000}

#: BaM schedules ablated in Figure E.2 (kept for optional reproduction).
PAPER_GAUSSIAN_BAM_SCHEDULES: Tuple[str, ...] = ("B", "BD", "B/(t+1)", "BD/(t+1)")


def paper_gaussian_settings(dim: int) -> Dict[str, Any]:
    """Return the paper's settings for a given dimension ``D``."""
    lr: Dict[str, float] = {}
    for method in ("advi", "fisher", "score"):
        table = PAPER_GAUSSIAN_LR.get(method, {})
        lr[method] = float(table.get(dim, PAPER_GAUSSIAN_ADVI_LR))
    return {
        "dim": int(dim),
        "n_runs": PAPER_GAUSSIAN_N_RUNS,
        "init_mean_scale": PAPER_GAUSSIAN_INIT_MEAN_SCALE,
        "init_cov": "I",
        "bam_batch_sizes": tuple(PAPER_GAUSSIAN_BATCH_SIZES["bam"]),
        "baseline_batch_size": PAPER_GAUSSIAN_BASELINE_BATCH_SIZE,
        "bam_schedule": "BD",  # constant lambda_t = B D
        "grad_budget": PAPER_GAUSSIAN_GRAD_BUDGET.get(dim, 20_000),
        "learning_rate": lr,
        "target_cov": "A A^T with A ~ D x D random",
        "score_learning_rate_grid": tuple(PAPER_GAUSSIAN_SCORE_LR),
    }


PAPER_GAUSSIAN_SETTINGS: Dict[int, Dict[str, Any]] = {d: paper_gaussian_settings(d) for d in PAPER_DIMS}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _filter_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments that ``fn`` does not accept."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _method_label(method: str, batch_size: int) -> str:
    if method.lower() == "bam":
        return f"BaM B={batch_size}"
    return {"advi": "ADVI", "score": "Score", "fisher": "Fisher", "gsm": "GSM"}.get(
        method.lower(), f"{method} B={batch_size}"
    )


def default_learning_rate(method: str, dim: int) -> Optional[float]:
    """Paper's grid-searched learning rate for a baseline, ``None`` for BaM."""
    method = method.lower()
    if method == "bam":
        return None
    table = PAPER_GAUSSIAN_LR.get(method, {})
    return float(table.get(dim, PAPER_GAUSSIAN_ADVI_LR))


def make_gaussian_problem(
    dim: int, seed: int = 0, mu_scale: float = PAPER_GAUSSIAN_INIT_MEAN_SCALE
) -> Tuple[GaussianTarget, np.ndarray, np.ndarray]:
    """Build ``p = N(mu_star, A A^T)`` and the paper's initialisation."""
    rng = np.random.default_rng(int(seed))
    target = random_gaussian_target(dim=int(dim), rng=rng, mean_scale=float(mu_scale))
    mu0 = rng.uniform(0.0, float(mu_scale), size=int(dim))
    Sigma0 = np.eye(int(dim), dtype=float)
    return target, mu0, Sigma0


def _forward_kl(target: Any, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """``KL(p ; q)`` for a Gaussian target (closed form)."""
    try:
        return float(target.forward_kl(mu, Sigma))
    except Exception:
        pass
    try:
        from ..metrics.kl_metrics import gaussian_forward_kl
    except ImportError:  # pragma: no cover
        from bam_repro.metrics.kl_metrics import gaussian_forward_kl
    return float(gaussian_forward_kl(target.mean, target.cov, mu, Sigma))


def _reverse_kl(target: Any, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """``KL(q ; p)`` for a Gaussian target (closed form)."""
    try:
        return float(target.reverse_kl(mu, Sigma))
    except Exception:
        pass
    try:
        from ..metrics.kl_metrics import gaussian_reverse_kl
    except ImportError:  # pragma: no cover
        from bam_repro.metrics.kl_metrics import gaussian_reverse_kl
    return float(gaussian_reverse_kl(target.mean, target.cov, mu, Sigma))


def _extract_trace(result: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mu_history, Sigma_history)`` from any experiment result object."""
    mu_hist = getattr(result, "mu_history", None)
    sig_hist = getattr(result, "Sigma_history", None)
    if mu_hist is None or sig_hist is None:
        hist = getattr(result, "history", None)
        if isinstance(hist, dict):
            mu_hist = hist.get("mu") or hist.get("mu_history")
            sig_hist = hist.get("Sigma") or hist.get("cov") or hist.get("Sigma_history")
    if mu_hist is None or sig_hist is None:
        # Fall back to the final state only.
        mu_hist = [getattr(result, "mu")]
        sig_hist = [getattr(result, "Sigma")]
    mu_arr = np.stack([np.asarray(m, dtype=float).ravel() for m in np.asarray(mu_hist, dtype=object)])
    sig_arr = np.stack([np.asarray(s, dtype=float) for s in np.asarray(sig_hist, dtype=object)])
    return mu_arr, sig_arr


def _grad_eval_axis(
    n_hist: int,
    batch_size: int,
    history_every: int,
    mu_hist: Optional[np.ndarray] = None,
    mu0: Optional[np.ndarray] = None,
    result: Any = None,
) -> np.ndarray:
    """Gradient-evaluation abscissa matching a recorded history, if possible."""
    if n_hist <= 0:
        return np.zeros(0, dtype=float)
    ge = getattr(result, "grad_evals_history", None)
    if ge is not None:
        arr = _as_float_array(ge).ravel()
        if arr.size == n_hist:
            return arr
    offset = 0
    if mu_hist is not None and mu0 is not None and n_hist:
        try:
            if not np.allclose(np.asarray(mu_hist[0]).ravel(), np.asarray(mu0).ravel(), atol=1e-10):
                offset = int(history_every)
        except Exception:  # pragma: no cover - defensive
            offset = 0
    steps = offset + np.arange(n_hist, dtype=float) * float(history_every)
    return steps * float(batch_size)


def _bam_schedule(schedule: str, batch_size: int, dim: int) -> Any:
    """Build the ``lambda_t`` schedule for BaM (constant ``BD`` by default)."""
    if schedule in (None, ""):
        schedule = "BD"
    try:
        return make_schedule(schedule, batch_size=int(batch_size), dim=int(dim))
    except Exception:  # pragma: no cover - fall back to a plain callable
        name = str(schedule).replace(" ", "").lower()
        B, D = float(batch_size), float(dim)
        if name in ("bd", "b*d", "lambda=bd"):
            return lambda t: B * D
        if name in ("b",):
            return lambda t: B
        if name in ("bd/(t+1)", "b*d/(t+1)"):
            return lambda t: B * D / (float(t) + 1.0)
        if name in ("b/(t+1)",):
            return lambda t: B / (float(t) + 1.0)
        return lambda t: B * D


def build_learner(
    method: str,
    target: Any,
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    batch_size: int,
    seed: int,
    learning_rate: Optional[float] = None,
    schedule: str = "BD",
    history_every: int = 1,
    dim: Optional[int] = None,
    **extra: Any,
) -> Any:
    """Instantiate the requested algorithm with paper-faithful settings."""
    method = str(method).lower()
    dim = int(dim if dim is not None else np.asarray(mu0).size)
    score_fn = getattr(target, "score", None)
    if score_fn is None:
        score_fn = getattr(target, "grad_log_prob", None)

    common: Dict[str, Any] = dict(
        mu0=np.asarray(mu0, dtype=float),
        Sigma0=np.asarray(Sigma0, dtype=float),
        batch_size=int(batch_size),
        seed=int(seed),
        track_history=True,
        history_every=max(1, int(history_every)),
    )
    common.update(extra)

    if method == "bam":
        kwargs = dict(
            common,
            score_fn=score_fn,
            lam=_bam_schedule(schedule, batch_size, dim) if isinstance(schedule, str) else schedule,
        )
        cls = BaM
    elif method in ("advi", "elbo"):
        kwargs = dict(
            common,
            target=target,
            score_fn=score_fn,
            loss="elbo",
            learning_rate=float(learning_rate if learning_rate is not None else PAPER_GAUSSIAN_ADVI_LR),
        )
        cls = ADVI
    elif method in ("score", "score_advi"):
        kwargs = dict(
            common,
            target=target,
            score_fn=score_fn,
            learning_rate=float(learning_rate if learning_rate is not None else 0.01),
        )
        cls = ScoreADVI
    elif method in ("fisher", "fisher_advi"):
        kwargs = dict(
            common,
            target=target,
            score_fn=score_fn,
            learning_rate=float(learning_rate if learning_rate is not None else PAPER_GAUSSIAN_FISHER_LR),
        )
        cls = FisherADVI
    elif method in ("gsm",):
        kwargs = dict(common, target=target, score_fn=score_fn)
        cls = GSM
    else:  # pragma: no cover - unknown method
        raise ValueError(f"unknown method {method!r}")

    return cls(**_filter_kwargs(cls.__init__, kwargs))


def _n_iterations(grad_budget: Optional[int], n_iter: Optional[int], batch_size: int) -> int:
    if n_iter is not None:
        return max(1, int(n_iter))
    if grad_budget is None:
        grad_budget = 20_000
    return max(1, int(round(float(grad_budget) / max(1, int(batch_size)))))


# ---------------------------------------------------------------------------
# single replicate
# ---------------------------------------------------------------------------
def run_gaussian_replicate(
    dim: int,
    method: str = "bam",
    batch_size: int = 20,
    seed: int = 0,
    grad_budget: Optional[int] = None,
    n_iter: Optional[int] = None,
    learning_rate: Optional[float] = None,
    schedule: str = "BD",
    mu_scale: float = PAPER_GAUSSIAN_INIT_MEAN_SCALE,
    target: Optional[Any] = None,
    mu0: Optional[np.ndarray] = None,
    Sigma0: Optional[np.ndarray] = None,
    history_every: Optional[int] = None,
    history_points: int = 120,
    return_learner: bool = False,
    **learner_kwargs: Any,
) -> Dict[str, Any]:
    """Run one replicate of one method on one Gaussian target.

    Returns a record dict with the gradient-evaluation abscissa and the closed
    form forward/reverse KL divergences along the whole optimisation trace.
    """
    dim = int(dim)
    method = str(method).lower()
    if target is None or mu0 is None or Sigma0 is None:
        t, m, s = make_gaussian_problem(dim, seed=int(seed), mu_scale=float(mu_scale))
        target = target if target is not None else t
        mu0 = mu0 if mu0 is not None else m
        Sigma0 = Sigma0 if Sigma0 is not None else s
    mu0 = np.asarray(mu0, dtype=float).ravel()
    Sigma0 = np.asarray(Sigma0, dtype=float)

    T = _n_iterations(grad_budget, n_iter, batch_size)
    if history_every is None:
        history_every = max(1, int(T // max(1, int(history_points))))

    if learning_rate is None:
        learning_rate = default_learning_rate(method, dim)

    t0 = time.perf_counter()
    learner = build_learner(
        method,
        target=target,
        mu0=mu0,
        Sigma0=Sigma0,
        batch_size=int(batch_size),
        seed=int(seed),
        learning_rate=learning_rate,
        schedule=schedule,
        history_every=int(history_every),
        dim=dim,
        **learner_kwargs,
    )
    result = learner.run(T)
    wallclock = time.perf_counter() - t0

    mu_hist, sig_hist = _extract_trace(result)
    n_hist = int(min(len(mu_hist), len(sig_hist)))
    mu_hist, sig_hist = mu_hist[:n_hist], sig_hist[:n_hist]

    fwd = np.empty(n_hist, dtype=float)
    rev = np.empty(n_hist, dtype=float)
    for i in range(n_hist):
        fwd[i] = _forward_kl(target, mu_hist[i], sig_hist[i])
        rev[i] = _reverse_kl(target, mu_hist[i], sig_hist[i])

    ge = _grad_eval_axis(
        n_hist,
        batch_size=int(batch_size),
        history_every=int(history_every),
        mu_hist=mu_hist,
        mu0=mu0,
        result=result,
    )

    mu_final = np.asarray(getattr(result, "mu", mu_hist[-1]), dtype=float).ravel()
    Sig_final = np.asarray(getattr(result, "Sigma", sig_hist[-1]), dtype=float)

    record: Dict[str, Any] = {
        "dim": dim,
        "method": method,
        "batch_size": int(batch_size),
        "seed": int(seed),
        "n_iter": int(T),
        "history_every": int(history_every),
        "learning_rate": None if learning_rate is None else float(learning_rate),
        "schedule": schedule if isinstance(schedule, str) else str(schedule),
        "walclock_seconds": float(wallclock),
        "grad_evals": ge,
        "forward_kl": fwd,
        "reverse_kl": rev,
        "final_forward_kl": float(fwd[-1]) if n_hist else float("nan"),
        "final_reverse_kl": float(rev[-1]) if n_hist else float("nan"),
        "best_forward_kl": float(np.nanmin(fwd)) if n_hist else float("nan"),
        "best_reverse_kl": float(np.nanmin(rev)) if n_hist else float("nan"),
        "mu": mu_final,
        "Sigma": Sig_final,
        "label": _method_label(method, batch_size),
    }
    if return_learner:
        record["learner"] = learner
        record["result"] = result
    return record


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
def _sem(stack: np.ndarray) -> np.ndarray:
    if stack.ndim != 2 or stack.shape[0] < 2:
        return np.zeros(stack.shape[-1], dtype=float)
    return np.nanstd(stack, axis=0, ddof=1) / math.sqrt(stack.shape[0])


@dataclass
class GaussianExperimentResult:
    """Aggregated results of the Gaussian-target experiment (Figure 5.1/E.3)."""

    dims: Tuple[int, ...] = PAPER_DIMS
    methods: Tuple[str, ...] = PAPER_GAUSSIAN_METHODS
    n_runs: int = PAPER_GAUSSIAN_N_RUNS
    config: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)
    curves: Dict[int, Dict[str, Dict[str, np.ndarray]]] = field(default_factory=dict)
    summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # -- aggregation -------------------------------------------------------
    def aggregate(self) -> "GaussianExperimentResult":
        groups: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = {}
        for rec in self.records:
            key = (int(rec["dim"]), str(rec["method"]), int(rec["batch_size"]))
            groups.setdefault(key, []).append(rec)

        curves: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
        summary: Dict[str, Dict[str, Any]] = {}
        for (dim, method, batch), recs in sorted(groups.items()):
            n = min(int(np.asarray(r["forward_kl"]).size) for r in recs)
            ge = _as_float_array(recs[0]["grad_evals"])[:n]
            fwd = np.stack([_as_float_array(r["forward_kl"])[:n] for r in recs])
            rev = np.stack([_as_float_array(r["reverse_kl"])[:n] for r in recs])
            label = _method_label(method, batch)
            curves.setdefault(dim, {})[label] = {
                "method": method,
                "batch_size": int(batch),
                "label": label,
                "grad_evals": ge,
                "forward_kl": np.nanmean(fwd, axis=0),
                "forward_kl_sem": _sem(fwd),
                "reverse_kl": np.nanmean(rev, axis=0),
                "reverse_kl_sem": _sem(rev),
                "forward_kl_runs": fwd,
                "reverse_kl_runs": rev,
                "n_runs": int(fwd.shape[0]),
            }
            best_fwd = np.array([r["best_forward_kl"] for r in recs], dtype=float)
            final_rev = np.array([r["final_reverse_kl"] for r in recs], dtype=float)
            summary[f"D={dim} {label}"] = {
                "dim": dim,
                "method": method,
                "batch_size": int(batch),
                "n_runs": int(fwd.shape[0]),
                "grad_evals_total": float(ge[-1]) if ge.size else 0.0,
                "best_forward_kl_mean": float(np.nanmean(best_fwd)),
                "best_forward_kl_sem": float(_sem(best_fwd.reshape(1, -1))[0]) if best_fwd.size > 1 else 0.0,
                "final_reverse_kl_mean": float(np.nanmean(final_rev)),
                "final_reverse_kl_sem": float(_sem(final_rev.reshape(1, -1))[0]) if final_rev.size > 1 else 0.0,
            }
        self.curves = curves
        self.summary = summary
        return self

    # -- reporting ---------------------------------------------------------
    def table(self) -> str:
        lines = [
            f"Gaussian targets (paper Figure 5.1 / E.3) - {self.n_runs} runs",
            "-" * 78,
            f"{'dim':>5} {'method':<14} {'B':>4} {'grad evals':>12} "
            f"{'best fwd KL':>12} {'final rev KL':>13}",
        ]
        for key in sorted(self.summary, key=lambda k: (self.summary[k]["dim"], self.summary[k]["method"])):
            row = self.summary[key]
            lines.append(
                f"{row['dim']:>5} {row['method']:<14} {row['batch_size']:>4} "
                f"{row['grad_evals_total']:>12.0f} {row['best_forward_kl_mean']:>12.4g} "
                f"{row['final_reverse_kl_mean']:>13.4g}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GaussianExperimentResult(dims={self.dims}, methods={self.methods}, "
            f"n_runs={self.n_runs}, records={len(self.records)})"
        )

    # -- serialisation -----------------------------------------------------
    def to_dict(self, include_runs: bool = True) -> Dict[str, Any]:
        def conv(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {str(k): conv(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [conv(v) for v in obj]
            return obj

        out: Dict[str, Any] = {
            "experiment": "gaussian",
            "dims": list(self.dims),
            "methods": list(self.methods),
            "n_runs": int(self.n_runs),
            "config": conv(self.config),
            "curves": conv(self.curves),
            "summary": conv(self.summary),
        }
        if include_runs:
            out["records"] = [
                {
                    k: conv(v)
                    for k, v in rec.items()
                    if k not in ("Sigma", "mu")  # keep the JSON small
                }
                for rec in self.records
            ]
        return out

    def save(self, path: str) -> str:
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "GaussianExperimentResult":
        with open(path) as fh:
            data = json.load(fh)
        res = cls(
            dims=tuple(data.get("dims", ())),
            methods=tuple(data.get("methods", ())),
            n_runs=int(data.get("n_runs", 0)),
            config=data.get("config", {}),
            records=data.get("records", []),
        )
        curves: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
        for dim, method_dict in (data.get("curves") or {}).items():
            curves[int(dim)] = {}
            for label, entry in method_dict.items():
                curves[int(dim)][label] = {
                    k: (np.asarray(v) if isinstance(v, (list, tuple)) else v) for k, v in entry.items()
                }
        res.curves = curves
        res.summary = data.get("summary", {})
        return res

    # -- plotting ----------------------------------------------------------
    def figure(
        self,
        outdir: Optional[str] = None,
        metric: str = "forward",
        show: bool = False,
        xlabel: str = "number of gradient evaluations",
    ) -> List[str]:
        """Reproduce Figure 5.1 (forward KL) or Figure E.3 (reverse KL)."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        key = "forward_kl" if str(metric).lower().startswith("f") else "reverse_kl"
        skey = key + "_sem"
        ylabel = r"$\mathrm{KL}(p\,;\,q)$" if key == "forward_kl" else r"$\mathrm{KL}(q\,;\,p)$"
        dims = sorted(self.curves)
        if not dims:
            return []
        n_cols = min(4, len(dims))
        n_rows = int(math.ceil(len(dims) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows), squeeze=False)
        for ax, dim in zip(axes.ravel(), dims):
            for label, entry in sorted(self.curves[dim].items()):
                ge = np.asarray(entry["grad_evals"], dtype=float)
                if ge.size == 0:
                    continue
                y = np.asarray(entry[key], dtype=float)
                y = np.where(y > 0, y, np.nan)
                runs = np.asarray(entry[key + "_runs"], dtype=float)
                for r in range(runs.shape[0]):
                    yy = np.where(runs[r] > 0, runs[r], np.nan)
                    ax.plot(np.maximum(ge, 1e-1), yy, alpha=0.15, linewidth=0.8, color=None)
                ax.plot(np.maximum(ge, 1e-1), y, label=label, linewidth=1.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"D = {dim}")
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(alpha=0.25, which="both")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            axes.ravel()[0].legend(fontsize=7)
        fig.tight_layout()
        paths: List[str] = []
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, f"fig_gaussian_{key}.png")
            fig.savefig(path, dpi=150)
            paths.append(path)
        if show:  # pragma: no cover
            plt.show()
        plt.close(fig)
        return paths


# ---------------------------------------------------------------------------
# experiment driver
# ---------------------------------------------------------------------------
def run_gaussian_experiment(
    dims: Sequence[int] = PAPER_DIMS,
    n_runs: int = PAPER_GAUSSIAN_N_RUNS,
    methods: Sequence[str] = PAPER_GAUSSIAN_METHODS,
    bam_batch_sizes: Sequence[int] = PAPER_GAUSSIAN_BATCH_SIZES["bam"],
    baseline_batch_size: int = PAPER_GAUSSIAN_BASELINE_BATCH_SIZE,
    grad_budget: Optional[int] = None,
    budget_scale: float = 1.0,
    mu_scale: float = PAPER_GAUSSIAN_INIT_MEAN_SCALE,
    seed: int = 0,
    history_points: int = 120,
    schedule: str = "BD",
    learning_rates: Optional[Dict[str, float]] = None,
    outdir: Optional[str] = None,
    save: bool = True,
    figures: bool = True,
    verbose: bool = True,
    **learner_kwargs: Any,
) -> GaussianExperimentResult:
    """Run the full Section 5.1 Gaussian sweep (Figure 5.1 / E.3)."""
    dims = tuple(int(d) for d in dims)
    methods = tuple(str(m).lower() for m in methods)
    learning_rates = dict(learning_rates or {})

    config = {
        "dims": list(dims),
        "n_runs": int(n_runs),
        "methods": list(methods),
        "bam_batch_sizes": [int(b) for b in bam_batch_sizes],
        "baseline_batch_size": int(baseline_batch_size),
        "budget_scale": float(budget_scale),
        "mu_scale": float(mu_scale),
        "schedule": schedule,
        "seed": int(seed),
        "learning_rates": learning_rates or "paper defaults",
        "paper": {
            "score_lr_grid": list(PAPER_GAUSSIAN_SCORE_LR),
            "advi_lr": PAPER_GAUSSIAN_ADVI_LR,
            "fisher_lr": PAPER_GAUSSIAN_FISHER_LR,
            "bam_lambda": "B D (constant)",
        },
    }

    records: List[Dict[str, Any]] = []
    for dim in dims:
        base_budget = int(grad_budget) if grad_budget is not None else PAPER_GAUSSIAN_GRAD_BUDGET.get(dim, 20_000)
        budget = max(int(round(base_budget * float(budget_scale))), 8)
        for run in range(int(n_runs)):
            run_seed = int(seed) + 1000 * int(run) + int(dim)
            target, mu0, Sigma0 = make_gaussian_problem(dim, seed=run_seed, mu_scale=float(mu_scale))
            for method in methods:
                batches = list(bam_batch_sizes) if method == "bam" else [int(baseline_batch_size)]
                for batch in batches:
                    lr = learning_rates.get(method, learning_rates.get(f"{method}_{dim}"))
                    rec = run_gaussian_replicate(
                        dim=dim,
                        method=method,
                        batch_size=int(batch),
                        seed=int(seed) + 7919 * int(run) + int(batch),
                        grad_budget=budget,
                        learning_rate=lr,
                        schedule=schedule,
                        target=target,
                        mu0=mu0,
                        Sigma0=Sigma0,
                        history_points=int(history_points),
                        **learner_kwargs,
                    )
                    records.append(rec)
        if verbose:
            print(f"[exp_gaussian] D={dim} budget={budget} grad evals, "
                  f"{n_runs} runs x {len(methods)} methods done", flush=True)

    result = GaussianExperimentResult(
        dims=dims, methods=methods, n_runs=int(n_runs), config=config, records=records
    ).aggregate()

    if verbose:
        print(result.table(), flush=True)

    paths: List[str] = []
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if save:
            paths.append(result.save(os.path.join(outdir, "exp_gaussian.json")))
        if figures:
            try:
                paths.extend(result.figure(outdir=outdir, metric="forward"))
                paths.extend(result.figure(outdir=outdir, metric="reverse"))
            except Exception as exc:  # pragma: no cover - matplotlib optional
                print(f"[exp_gaussian] figure generation skipped: {exc}", flush=True)
        if verbose and paths:
            print("[exp_gaussian] wrote:\n  " + "\n  ".join(paths), flush=True)
    return result


def run_gaussian(
    quick: bool = False,
    dims: Optional[Sequence[int]] = None,
    n_runs: Optional[int] = None,
    budget_scale: Optional[float] = None,
    bam_batch_sizes: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> GaussianExperimentResult:
    """Convenience entry point; ``quick=True`` runs a small smoke test."""
    if quick:
        dims = dims if dims is not None else (4, 16)
        n_runs = n_runs if n_runs is not None else 2
        budget_scale = budget_scale if budget_scale is not None else 0.1
        bam_batch_sizes = bam_batch_sizes if bam_batch_sizes is not None else (20,)
    if dims is not None:
        kwargs["dims"] = dims
    if n_runs is not None:
        kwargs["n_runs"] = n_runs
    if budget_scale is not None:
        kwargs["budget_scale"] = budget_scale
    if bam_batch_sizes is not None:
        kwargs["bam_batch_sizes"] = bam_batch_sizes
    return run_gaussian_experiment(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", type=int, nargs="+", default=list(PAPER_DIMS))
    parser.add_argument("--runs", type=int, default=PAPER_GAUSSIAN_N_RUNS)
    parser.add_argument("--methods", type=str, nargs="+", default=list(PAPER_GAUSSIAN_METHODS))
    parser.add_argument("--bam-batch-sizes", type=int, nargs="+", default=list(PAPER_GAUSSIAN_BATCH_SIZES["bam"]))
    parser.add_argument("--baseline-batch-size", type=int, default=PAPER_GAUSSIAN_BASELINE_BATCH_SIZE)
    parser.add_argument("--grad-budget", type=int, default=None, help="override gradient-evaluation budget")
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--history-points", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results/gaussian")
    parser.add_argument("--quick", action="store_true", help="small smoke-test configuration")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        result = run_gaussian(
            quick=True,
            methods=args.methods,
            grad_budget=args.grad_budget,
            seed=args.seed,
            outdir=args.outdir,
            figures=not args.no_figures,
            save=not args.no_save,
            baseline_batch_size=args.baseline_batch_size,
            history_points=args.history_points,
        )
    else:
        result = run_gaussian_experiment(
            dims=args.dims,
            n_runs=args.runs,
            methods=args.methods,
            bam_batch_sizes=args.bam_batch_sizes,
            baseline_batch_size=args.baseline_batch_size,
            grad_budget=args.grad_budget,
            budget_scale=args.budget_scale,
            seed=args.seed,
            history_points=args.history_points,
            outdir=args.outdir,
            figures=not args.no_figures,
            save=not args.no_save,
        )
    return 0 if result.records else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
