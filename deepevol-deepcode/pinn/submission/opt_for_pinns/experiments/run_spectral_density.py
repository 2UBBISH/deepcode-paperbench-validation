"""Spectral-density study of the PINN loss Hessian (Figures 3 and 7).

This experiment reproduces Sections 5.1-5.3 of

    "Challenges in Training PINNs: A Loss Landscape Perspective" (ICML 2024)

Concretely, for each PDE (convection / reaction / wave):

1. Train a PINN ``u(x; w)`` with **Adam+L-BFGS** (learning rate for Adam tuned by a
   grid search over ``{1e-5, ..., 1e-1}``, switch to L-BFGS after ``11k`` iterations,
   ``41000`` iterations in total), for a small grid of (width, Adam learning rate,
   seed) combinations.  The L-BFGS curvature history ``(s_k, y_k, rho_k)`` is recorded
   during training.
2. Select the "best" run per PDE with the paper's *systematic selection process*: the
   configuration with the **smallest L2RE** (the selection is over Adam lr, seed and
   width -- the paper's observed winners are width 200, Adam lr {1e-4, 1e-3, 1e-3} and
   seeds {345, 456, 567}, but they must be re-derived here, not hard-coded).
3. Estimate -- with stochastic Lanczos quadrature (native implementation or PyHessian
   if available) -- the spectral density of

     * ``H_L(w)``                                (top row of Figure 3),
     * ``H~_k^T H_L(w) H~_k``                    (L-BFGS-preconditioned, dashed lines),
     * the Hessian of each loss component (residual / initial / boundary) and its
       L-BFGS-preconditioned counterpart (bottom row of Figure 3 and Figure 7).

   Expected qualitative findings: large outlier eigenvalues of ``H_L``
   (> 1e4 convection, > 1e3 reaction, > 1e5 wave), a lot of spectral mass near 0, the
   residual component being the most ill-conditioned one, and L-BFGS preconditioning
   reducing the top eigenvalue / condition number by at least 1e3.

Usage
-----
    python experiments/run_spectral_density.py                      # full study
    python experiments/run_spectral_density.py --quick              # smoke test
    python experiments/run_spectral_density.py --pdes convection --device cuda

The runner writes ``results/spectral_density/summary.json`` plus (optionally) figures
via :mod:`src.utils.plotting`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Path bootstrap: make ``src.*`` importable no matter where the script is launched from.
# --------------------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent  # .../opt_for_pinns
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from src.pinns.loss import make_loss_fn  # noqa: E402
from src.pinns.metrics import compute_l2re  # noqa: E402
from src.pinns.model import make_pinn  # noqa: E402
from src.pinns.problems import PROBLEMS, get_problem  # noqa: E402
from src.pinns.sampling import build_sampler  # noqa: E402
from src.optimizers.combined import (  # noqa: E402
    COMBINED_TOTAL_ITERATIONS,
    DEFAULT_SWITCH_POINT,
    run_adam_lbfgs,
)
from src.optimizers.first_order import ADAM_LR_GRID  # noqa: E402

# --------------------------------------------------------------------------------------
# Optional imports (the study degrades gracefully when they are unavailable).
# --------------------------------------------------------------------------------------
try:  # spectral-density machinery (Algorithms 2/3 + SLQ)
    from src.spectral.preconditioned_mvp import (  # type: ignore
        make_spectral_operator,
        preconditioned_mvp,
    )
    from src.spectral.lbfgs_unroll import unroll_from_history  # type: ignore
    from src.spectral.spectral_density import (  # type: ignore
        SpectralDensityEstimator,
        component_operator,
        estimate_condition_number,
        pyhessian_available,
        slq_density,
    )

    SPECTRAL_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - defensive
    SPECTRAL_AVAILABLE = False
    _SPECTRAL_IMPORT_ERROR = repr(_exc)

try:
    from src.utils import plotting as _plotting  # type: ignore

    PLOTTING_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    _plotting = None
    PLOTTING_AVAILABLE = False

try:
    from src.utils.seeding import set_seed as _set_seed  # type: ignore
except Exception:  # pragma: no cover - defensive

    def _set_seed(seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(int(seed))
            try:
                import random

                random.seed(int(seed))
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# Defaults / configuration
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "problem": {
        "convection": {"beta": 40.0},
        "reaction": {"rho": 5.0},
        "wave": {"beta": 5.0, "c2": 4.0},
    },
    "network": {"depth": 3, "widths": [50, 100, 200, 400], "activation": "tanh"},
    "sampling": {
        "n_residual": 10000,
        "n_ic": 257,
        "n_bc": 101,
        "n_grid_x": 255,
        "n_grid_t": 100,
        "replace": True,
    },
    "optimizer": {
        "adam_lrs": list(ADAM_LR_GRID),
        "switch_iteration": DEFAULT_SWITCH_POINT,  # 11000 for the spectral study
        "total_iterations": COMBINED_TOTAL_ITERATIONS,  # 41000
        "lbfgs_history_size": 100,
        "lbfgs_line_search": "strong_wolfe",
        "record_limit": 64,
    },
    "spectral": {
        "n_iter": 100,  # Lanczos iterations per probe
        "n_vec": 1,  # number of Rademacher probes
        "n_grid": 200,  # density histogram resolution
        "top_k": 10,
        "backend": "native",  # "native" | "pyhessian"
        "dtype": "float64",
        "components": ["residual", "initial", "boundary"],
    },
    "selection": {"criterion": "l2re"},
    "seeds": [123, 234, 345, 456, 567],
    "pdes": ["convection", "reaction", "wave"],
    "quick": {
        "widths": [50],
        "adam_lrs": [1e-3, 1e-2],
        "seeds": [123],
        "total_iterations": 300,
        "switch_iteration": 200,
        "eval_every": 100,
        "n_iter": 30,
        "n_vec": 1,
        "n_grid": 100,
        "n_residual": 2000,
    },
}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None, quick: bool = False) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` (if available) merged with the spectral defaults."""
    cfg = dict(DEFAULT_CONFIG)
    candidate_paths: List[Path] = []
    if path:
        candidate_paths.append(Path(path))
    candidate_paths.append(_PROJECT_ROOT / "configs" / "default.yaml")
    candidate_paths.append(Path.cwd() / "configs" / "default.yaml")
    for candidate in candidate_paths:
        try:
            if candidate.is_file():
                import yaml  # local import: pyyaml is optional

                with open(candidate, "r") as handle:
                    loaded = yaml.safe_load(handle) or {}
                cfg = _deep_update(cfg, loaded)
                break
        except Exception:
            continue
    if quick:
        q = cfg.get("quick", DEFAULT_CONFIG["quick"])
        cfg = _deep_update(cfg, {k: v for k, v in q.items() if k in cfg})
        # ``quick`` keys that are lists in ``cfg`` but scalars in ``quick``.
        for scalar_key in ("total_iterations", "switch_iteration"):
            if scalar_key in q:
                cfg["optimizer"][scalar_key] = q[scalar_key]
        for scalar_key in ("n_iter", "n_vec", "n_grid"):
            if scalar_key in q:
                cfg["spectral"][scalar_key] = q[scalar_key]
        if "n_residual" in q:
            cfg["sampling"]["n_residual"] = q["n_residual"]
        if "eval_every" in q:
            cfg["eval_every"] = q["eval_every"]
    return cfg


def supported_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter ``kwargs`` down to the arguments actually accepted by ``fn``."""
    try:
        import inspect

        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of numpy/torch containers into JSON-serialisable objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    if hasattr(obj, "as_dict"):
        try:
            return to_jsonable(obj.as_dict())
        except Exception:
            return str(obj)
    return str(obj)


def _dtype_from_name(name: Any) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    text = str(name).lower()
    if "64" in text or "double" in text:
        return torch.float64
    return torch.float32


def device_from_arg(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Single Adam+L-BFGS training run
# --------------------------------------------------------------------------------------
@dataclass
class TrainingRecord:
    """Everything produced by one Adam+L-BFGS training run."""

    pde: str
    width: int
    adam_lr: float
    seed: int
    switch_iteration: int
    total_iterations: int
    final_loss: float
    best_loss: float
    l2re: float
    best_l2re: Optional[float]
    loss_history: List[float] = field(default_factory=list)
    loss_steps: List[int] = field(default_factory=list)
    l2re_history: List[float] = field(default_factory=list)
    l2re_steps: List[int] = field(default_factory=list)
    n_lbfgs_pairs: int = 0
    seconds: float = 0.0
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    lbfgs_history: Optional[Any] = None
    error: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """JSON-serialisable view of the record (without tensors/history buffers)."""
        data = {
            "pde": self.pde,
            "width": self.width,
            "adam_lr": self.adam_lr,
            "seed": self.seed,
            "switch_iteration": self.switch_iteration,
            "total_iterations": self.total_iterations,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "l2re": self.l2re,
            "best_l2re": self.best_l2re,
            "loss_history": self.loss_history,
            "loss_steps": self.loss_steps,
            "l2re_history": self.l2re_history,
            "l2re_steps": self.l2re_steps,
            "n_lbfgs_pairs": self.n_lbfgs_pairs,
            "seconds": self.seconds,
            "error": self.error,
        }
        return data


def train_adam_lbfgs(
    pde: str,
    width: int,
    adam_lr: float,
    seed: int,
    *,
    switch_iteration: int = DEFAULT_SWITCH_POINT,
    total_iterations: int = COMBINED_TOTAL_ITERATIONS,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    problem_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 250,
    history_size: int = 100,
    line_search: str = "strong_wolfe",
    record_limit: Optional[int] = 64,
    verbose: bool = False,
) -> TrainingRecord:
    """Train one PINN with Adam followed by L-BFGS and return a :class:`TrainingRecord`."""
    device = device or torch.device("cpu")
    problem = get_problem(pde, **(problem_cfg or {}))
    n_residual = int((sampling_cfg or {}).get("n_residual", 10000))

    _set_seed(seed)
    model = make_pinn(
        in_dim=int(getattr(problem, "in_dim", 2)),
        out_dim=int(getattr(problem, "out_dim", 1)),
        width=int(width),
        depth=int(depth),
        seed=seed,
        device=str(device),
    )
    model = model.to(device=device, dtype=dtype)

    sampler = build_sampler(
        problem,
        seed=seed,
        device=device,
        dtype=dtype,
        n_residual=n_residual,
        replace=bool((sampling_cfg or {}).get("replace", True)),
    )

    loss_obj, closure = make_loss_fn(model, problem, sampler=sampler, dtype=dtype)

    def eval_fn() -> float:
        return float(
            compute_l2re(
                model,
                problem,
                sampler=sampler,
                device=str(device),
                dtype=dtype,
            )
        )

    record = TrainingRecord(
        pde=pde,
        width=int(width),
        adam_lr=float(adam_lr),
        seed=int(seed),
        switch_iteration=int(switch_iteration),
        total_iterations=int(total_iterations),
        final_loss=float("nan"),
        best_loss=float("nan"),
        l2re=float("nan"),
        best_l2re=None,
    )

    t0 = time.time()
    try:
        result = run_adam_lbfgs(
            model,
            closure,
            adam_lr=float(adam_lr),
            switch_iteration=int(switch_iteration),
            total_iterations=int(total_iterations),
            eval_fn=eval_fn,
            eval_every=int(eval_every),
            record=True,
            record_limit=record_limit,
            verbose=verbose,
            lbfgs_kwargs={"history_size": int(history_size), "line_search_fn": line_search},
        )
    except Exception as exc:  # pragma: no cover - robustness for long sweeps
        record.error = repr(exc)
        record.seconds = time.time() - t0
        record.final_loss = float(loss_obj.value())
        record.best_loss = record.final_loss
        try:
            record.l2re = eval_fn()
        except Exception:
            pass
        record.state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return record

    record.seconds = time.time() - t0
    history = getattr(result, "history", None)
    if history is not None:
        to_dict = getattr(history, "to_dict", None)
        payload = to_dict() if callable(to_dict) else {}
        record.loss_steps = [int(s) for s in payload.get("steps", [])]
        record.loss_history = [float(v) for v in payload.get("losses", [])]
        record.l2re_steps = [int(s) for s in payload.get("l2re_steps", [])] or list(record.loss_steps)
        record.l2re_history = [
            float(v) for v in payload.get("l2re", []) if v is not None and math.isfinite(float(v))
        ]
        try:
            record.best_loss = float(history.best_loss)
        except Exception:
            pass
        try:
            if history.best_l2re is not None:
                record.best_l2re = float(history.best_l2re)
        except Exception:
            pass
    try:
        record.final_loss = float(result.final_loss)
    except Exception:
        record.final_loss = float(loss_obj.value())

    lbfgs_history = getattr(result, "lbfgs_history", None)
    record.lbfgs_history = lbfgs_history
    try:
        record.n_lbfgs_pairs = int(len(lbfgs_history)) if lbfgs_history is not None else 0
    except Exception:
        record.n_lbfgs_pairs = 0

    try:
        record.l2re = eval_fn()
    except Exception as exc:
        record.error = f"l2re: {exc!r}"
    record.state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return record


# --------------------------------------------------------------------------------------
# Systematic best-configuration selection (paper §4 / addendum)
# --------------------------------------------------------------------------------------
def select_best_config(
    records: Sequence[TrainingRecord],
    criterion: str = "l2re",
    *,
    allowed_keys: Sequence[str] = ("pde", "width", "adam_lr", "seed"),
) -> Dict[Any, TrainingRecord]:
    """Per-PDE argmin of ``criterion`` over (width, Adam lr, seed).

    This mirrors the paper's selection process: the configuration used for the
    spectral-density study is the one with the **smallest L2RE** among the runs of the
    Adam+L-BFGS sweep (11k switch).  Ties are broken by smaller loss, then by the
    canonical ordering of (width, lr, seed).
    """
    best: Dict[Any, TrainingRecord] = {}
    for record in records:
        key = tuple(getattr(record, k) for k in allowed_keys)[0]  # group by PDE
        value = getattr(record, criterion, None)
        if value is None or not math.isfinite(float(value)):
            continue
        value = float(value)
        current = best.get(key)
        if current is None:
            best[key] = record
            continue
        cur_value = getattr(current, criterion, float("inf"))
        cur_value = float(cur_value) if cur_value is not None else float("inf")
        if value < cur_value - 1e-15:
            best[key] = record
        elif abs(value - cur_value) <= 1e-15:
            if float(record.final_loss) < float(current.final_loss):
                best[key] = record
            elif float(record.final_loss) == float(current.final_loss):
                rank_new = (record.width, record.adam_lr, record.seed)
                rank_cur = (current.width, current.adam_lr, current.seed)
                if rank_new < rank_cur:
                    best[key] = record
    return best


# --------------------------------------------------------------------------------------
# Spectral analysis of one checkpoint
# --------------------------------------------------------------------------------------
def _build_estimator(
    model: torch.nn.Module,
    problem: Any,
    sampler: Any,
    lbfgs_history: Any,
    spectral_cfg: Dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int],
):
    """Instantiate :class:`SpectralDensityEstimator` tolerating signature drift."""
    if not SPECTRAL_AVAILABLE:
        raise RuntimeError("spectral machinery unavailable")
    kwargs = dict(
        n_iter=int(spectral_cfg.get("n_iter", 100)),
        n_vec=int(spectral_cfg.get("n_vec", 1)),
        n_grid=int(spectral_cfg.get("n_grid", 200)),
        dtype=dtype,
        device=device,
        seed=seed,
        backend=str(spectral_cfg.get("backend", "native")),
    )
    ctor = SpectralDensityEstimator
    common = dict(model=model, problem=problem, sampler=sampler, lbfgs_history=lbfgs_history)
    try:
        return ctor(**common, **supported_kwargs(ctor, kwargs))
    except TypeError:
        return ctor(model, supported_kwargs(ctor, {**common, **kwargs}))


def _density_summary(result: Any) -> Dict[str, Any]:
    """Compress a ``DensityResult`` into a JSON-friendly dictionary."""
    if result is None:
        return {}
    out: Dict[str, Any] = {}
    for key in ("condition_number", "eigenvalues_top", "grid", "density", "eigenvalues", "weights", "n_vec"):
        value = getattr(result, key, None)
        if value is None and isinstance(result, dict):
            value = result.get(key)
        if value is not None:
            out[key] = to_jsonable(value)
    for key in ("grid", "density"):
        if key in out and isinstance(out[key], list) and len(out[key]) > 400:
            # Down-sample long curve arrays for the JSON summary.
            step = max(1, len(out[key]) // 400)
            out[key] = out[key][::step]
    return out


def spectral_analysis(
    record: TrainingRecord,
    *,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    problem_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    spectral_cfg: Optional[Dict[str, Any]] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute the loss / preconditioned / per-component spectral densities."""
    spectral_cfg = spectral_cfg or {}
    device = device or torch.device("cpu")
    components = list(spectral_cfg.get("components", ["residual", "initial", "boundary"]))
    top_k = int(spectral_cfg.get("top_k", 10))

    analysis: Dict[str, Any] = {
        "pde": record.pde,
        "width": record.width,
        "adam_lr": record.adam_lr,
        "seed": record.seed,
        "n_lbfgs_pairs": record.n_lbfgs_pairs,
        "top_k": top_k,
        "spectral_available": bool(SPECTRAL_AVAILABLE),
    }
    if not SPECTRAL_AVAILABLE:
        analysis["error"] = "spectral machinery not importable"
        return analysis

    problem = get_problem(record.pde, **(problem_cfg or {}))
    n_residual = int((sampling_cfg or {}).get("n_residual", 10000))
    model = make_pinn(
        in_dim=int(getattr(problem, "in_dim", 2)),
        out_dim=int(getattr(problem, "out_dim", 1)),
        width=int(record.width),
        depth=int(depth),
        seed=int(record.seed),
        device=str(device),
    ).to(device=device, dtype=dtype)
    if record.state_dict is not None:
        state = {k: v.to(dtype=dtype, device=device) for k, v in record.state_dict.items()}
        model.load_state_dict(state)

    sampler = build_sampler(
        problem,
        seed=int(record.seed),
        device=device,
        dtype=dtype,
        n_residual=n_residual,
        replace=bool((sampling_cfg or {}).get("replace", True)),
    )
    loss_obj, _closure = make_loss_fn(model, problem, sampler=sampler, dtype=dtype)

    # Unroll the recorded L-BFGS curvature pairs (Algorithm 2).
    factors = None
    m = 0
    try:
        if record.lbfgs_history is not None and len(record.lbfgs_history) > 0:
            factors = unroll_from_history(
                record.lbfgs_history,
                m=int(spectral_cfg.get("m", min(record.n_lbfgs_pairs, 100)) or record.n_lbfgs_pairs),
                dtype=dtype,
                device=device,
            )
            m = int(getattr(factors, "m", 0))
            analysis["m"] = m
            analysis["gamma"] = float(getattr(factors, "gamma", float("nan")))
    except Exception as exc:
        analysis["unroll_error"] = repr(exc)

    estimator = None
    try:
        estimator = _build_estimator(
            model,
            problem,
            sampler,
            record.lbfgs_history,
            spectral_cfg,
            device=device,
            dtype=dtype,
            seed=int(record.seed),
        )
    except Exception as exc:
        analysis["estimator_error"] = repr(exc)

    n_params = sum(p.numel() for p in model.parameters())

    # -- (a) spectral density of H_L(w) -------------------------------------------------
    try:
        kwargs = dict(n=1, n_iter=int(spectral_cfg.get("n_iter", 100)),
                      n_vec=int(spectral_cfg.get("n_vec", 1)),
                      n_grid=int(spectral_cfg.get("n_grid", 200)),
                      top_k=top_k)
        if estimator is not None and hasattr(estimator, "loss_density"):
            result = estimator.loss_density(**supported_kwargs(estimator.loss_density, kwargs))
        else:
            from src.spectral.hvp import HessianOperator  # local import

            op = HessianOperator(lambda: loss_obj.value(), model, dtype=dtype)
            result = slq_density(
                op,
                n=1,
                n_iter=kwargs["n_iter"],
                n_vec=kwargs["n_vec"],
                n_grid=kwargs["n_grid"],
                top_k=top_k,
                dtype=dtype,
            )
        analysis["loss"] = _density_summary(result)
        if verbose:
            print(f"    H_L top eigenvalues: {analysis['loss'].get('eigenvalues_top')}")
    except Exception as exc:
        analysis["loss_error"] = repr(exc)

    # -- (b) spectral density of the L-BFGS-preconditioned Hessian ----------------------
    if factors is not None and m > 0:
        pre_operator = None
        try:
            from src.spectral.hvp import HessianOperator  # local import

            base_op = HessianOperator(lambda: loss_obj.value(), model, dtype=dtype)
            pre_operator = make_spectral_operator(factors, base_op.matvec, dtype=dtype)
        except Exception as exc:
            analysis["preconditioned_operator_error"] = repr(exc)

        if pre_operator is not None:
            try:
                kwargs = dict(n=int(getattr(pre_operator, "size", n_params + m)),
                              n_iter=int(spectral_cfg.get("n_iter", 100)),
                              n_vec=int(spectral_cfg.get("n_vec", 1)),
                              n_grid=int(spectral_cfg.get("n_grid", 200)),
                              top_k=top_k)
                result = None
                if estimator is not None and hasattr(estimator, "preconditioned_density"):
                    call = estimator.preconditioned_density
                    result = call(
                        **supported_kwargs(
                            call,
                            dict(factors=factors, history=record.lbfgs_history, m=m, **kwargs),
                        )
                    )
                if result is None:
                    result = slq_density(pre_operator, **supported_kwargs(slq_density, kwargs), dtype=dtype)
                analysis["preconditioned"] = _density_summary(result)
                if verbose:
                    print(
                        "    preconditioned top eigenvalues: "
                        f"{analysis['preconditioned'].get('eigenvalues_top')}"
                    )
            except Exception as exc:
                analysis["preconditioned_error"] = repr(exc)

            # -- (c) per-component densities (residual / initial / boundary) -----------
            for component in components:
                entry: Dict[str, Any] = {}
                try:
                    op = component_operator(model, problem, sampler=sampler, component=component, dtype=dtype)
                    kwargs = dict(
                        n=n_params,
                        n_iter=int(spectral_cfg.get("n_iter", 100)),
                        n_vec=int(spectral_cfg.get("n_vec", 1)),
                        n_grid=int(spectral_cfg.get("n_grid", 200)),
                        top_k=top_k,
                    )
                    res = slq_density(op, **supported_kwargs(slq_density, kwargs), dtype=dtype)
                    entry["loss"] = _density_summary(res)
                except Exception as exc:
                    entry["loss_error"] = repr(exc)

                try:
                    op = component_operator(model, problem, sampler=sampler, component=component, dtype=dtype)
                    pre_op = make_spectral_operator(factors, op.matvec, dtype=dtype)
                    kwargs = dict(
                        n=int(getattr(pre_op, "size", n_params + m)),
                        n_iter=int(spectral_cfg.get("n_iter", 100)),
                        n_vec=int(spectral_cfg.get("n_vec", 1)),
                        n_grid=int(spectral_cfg.get("n_grid", 200)),
                        top_k=top_k,
                    )
                    res = slq_density(pre_op, **supported_kwargs(slq_density, kwargs), dtype=dtype)
                    entry["preconditioned"] = _density_summary(res)
                except Exception as exc:
                    entry["preconditioned_error"] = repr(exc)

                analysis.setdefault("components", {})[component] = entry
                if verbose:
                    top = (entry.get("loss") or {}).get("eigenvalues_top")
                    print(f"    component {component}: top eigenvalues {top}")

    # -- (d) conditioning report ---------------------------------------------------------
    try:
        report: Dict[str, Any] = {}
        from src.spectral.hvp import HessianOperator  # local import

        base_op = HessianOperator(lambda: loss_obj.value(), model, dtype=dtype)
        kwargs = dict(n=n_params, n_iter=int(spectral_cfg.get("n_iter", 100)),
                      n_vec=int(spectral_cfg.get("n_vec", 1)), return_details=True)
        cond_loss = estimate_condition_number(base_op, **supported_kwargs(estimate_condition_number, kwargs))
        report["loss"] = to_jsonable(cond_loss)
        if factors is not None and m > 0:
            pre_op = make_spectral_operator(factors, base_op.matvec, dtype=dtype)
            kwargs = dict(n=int(getattr(pre_op, "size", n_params + m)),
                          n_iter=int(spectral_cfg.get("n_iter", 100)),
                          n_vec=int(spectral_cfg.get("n_vec", 1)), return_details=True)
            cond_pre = estimate_condition_number(pre_op, **supported_kwargs(estimate_condition_number, kwargs))
            report["preconditioned"] = to_jsonable(cond_pre)
            try:
                c_loss = float(cond_loss["condition_number"] if isinstance(cond_loss, dict) else cond_loss)
                c_pre = float(cond_pre["condition_number"] if isinstance(cond_pre, dict) else cond_pre)
                if c_pre > 0 and math.isfinite(c_loss) and math.isfinite(c_pre):
                    report["conditioning_improvement_factor"] = c_loss / c_pre
            except Exception:
                pass
        analysis["conditioning"] = report
    except Exception as exc:
        analysis["conditioning_error"] = repr(exc)

    analysis["n_parameters"] = int(n_params)
    analysis["pyhessian"] = bool(pyhessian_available()) if SPECTRAL_AVAILABLE else False
    return analysis


# --------------------------------------------------------------------------------------
# High-level driver
# --------------------------------------------------------------------------------------
def run_spectral_density(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    pdes: Optional[Sequence[str]] = None,
    widths: Optional[Sequence[int]] = None,
    adam_lrs: Optional[Sequence[float]] = None,
    seeds: Optional[Sequence[int]] = None,
    switch_iteration: Optional[int] = None,
    total_iterations: Optional[int] = None,
    device: Optional[str] = None,
    outdir: Optional[str] = None,
    verbose: bool = True,
    eval_every: Optional[int] = None,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Run the full spectral-density study (training sweep + spectral analysis)."""
    cfg = cfg or load_config()
    dev = device_from_arg(device)
    spectral_cfg = dict(cfg.get("spectral", {}))
    opt_cfg = dict(cfg.get("optimizer", {}))
    dtype = _dtype_from_name(spectral_cfg.get("dtype", "float64"))

    pdes = list(pdes if pdes is not None else cfg.get("pdes", DEFAULT_CONFIG["pdes"]))
    widths = list(widths if widths is not None else cfg.get("network", {}).get("widths", [200]))
    adam_lrs = list(adam_lrs if adam_lrs is not None else opt_cfg.get("adam_lrs", ADAM_LR_GRID))
    seeds = list(seeds if seeds is not None else cfg.get("seeds", [345, 456, 567]))
    switch = int(switch_iteration if switch_iteration is not None else opt_cfg.get("switch_iteration", 11000))
    total = int(total_iterations if total_iterations is not None else opt_cfg.get("total_iterations", 41000))
    eval_every = int(eval_every if eval_every is not None else cfg.get("eval_every", 1000))
    depth = int(cfg.get("network", {}).get("depth", 3))
    sampling_cfg = dict(cfg.get("sampling", {}))
    problem_cfgs = dict(cfg.get("problem", {}))
    outdir_path = Path(outdir) if outdir else (_PROJECT_ROOT / "results" / "spectral_density")
    outdir_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "config": {
            "pdes": pdes,
            "widths": widths,
            "adam_lrs": adam_lrs,
            "seeds": seeds,
            "switch_iteration": switch,
            "total_iterations": total,
            "dtype": str(dtype),
            "device": str(dev),
            "spectral": spectral_cfg,
        },
        "runs": [],
        "selected": {},
        "analyses": {},
        "spectral_available": bool(SPECTRAL_AVAILABLE),
        "pyhessian_available": bool(pyhessian_available()) if SPECTRAL_AVAILABLE else False,
    }

    print(
        f"[spectral] device={dev} dtype={dtype} switch={switch} total={total} "
        f"widths={widths} lrs={adam_lrs} seeds={seeds} pdes={pdes}"
    )

    records: List[TrainingRecord] = []
    for pde in pdes:
        for width in widths:
            for adam_lr in adam_lrs:
                for seed in seeds:
                    tag = f"{pde}/w{width}/lr{adam_lr:g}/s{seed}"
                    print(f"[train] {tag} ...", flush=True)
                    record = train_adam_lbfgs(
                        pde,
                        width,
                        adam_lr,
                        seed,
                        switch_iteration=switch,
                        total_iterations=total,
                        sampling_cfg=sampling_cfg,
                        problem_cfg=problem_cfgs.get(pde, {}),
                        depth=depth,
                        dtype=dtype,
                        device=dev,
                        eval_every=eval_every,
                        history_size=int(opt_cfg.get("lbfgs_history_size", 100)),
                        line_search=str(opt_cfg.get("lbfgs_line_search", "strong_wolfe")),
                        record_limit=opt_cfg.get("record_limit", 64),
                        verbose=False,
                    )
                    records.append(record)
                    summary["runs"].append(record.to_metadata())
                    try:
                        torch.save(
                            {
                                "state_dict": record.state_dict,
                                "pde": pde,
                                "width": width,
                                "adam_lr": adam_lr,
                                "seed": seed,
                                "l2re": record.l2re,
                                "final_loss": record.final_loss,
                            },
                            outdir_path / f"ckpt_{pde}_w{width}_lr{adam_lr:g}_s{seed}.pt",
                        )
                    except Exception as exc:
                        print(f"    (checkpoint save failed: {exc!r})")
                    print(
                        f"    final_loss={record.final_loss:.6e} l2re={record.l2re:.6e} "
                        f"lbgfs_pairs={record.n_lbfgs_pairs} time={record.seconds:.1f}s",
                        flush=True,
                    )

    best_per_pde = select_best_config(
        records, criterion=str(cfg.get("selection", {}).get("criterion", "l2re"))
    )
    for pde, record in best_per_pde.items():
        summary["selected"][pde] = {
            "width": record.width,
            "adam_lr": record.adam_lr,
            "seed": record.seed,
            "switch_iteration": record.switch_iteration,
            "total_iterations": record.total_iterations,
            "final_loss": record.final_loss,
            "l2re": record.l2re,
            "n_lbfgs_pairs": record.n_lbfgs_pairs,
        }
        print(
            f"[select] {pde}: width={record.width} lr={record.adam_lr:g} seed={record.seed} "
            f"loss={record.final_loss:.6e} L2RE={record.l2re:.6e}"
        )

    for pde, record in best_per_pde.items():
        print(f"[spectral] analysing {pde} ...", flush=True)
        try:
            summary["analyses"][pde] = spectral_analysis(
                record,
                sampling_cfg=sampling_cfg,
                problem_cfg=problem_cfgs.get(pde, {}),
                depth=depth,
                spectral_cfg=spectral_cfg,
                dtype=dtype,
                device=dev,
                verbose=verbose,
            )
        except Exception as exc:
            summary["analyses"][pde] = {"error": repr(exc)}
            print(f"    analysis failed: {exc!r}")

    with open(outdir_path / "summary.json", "w") as handle:
        json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)

    if make_plots and PLOTTING_AVAILABLE:
        try:
            if hasattr(_plotting, "plot_spectral_density"):
                _plotting.plot_spectral_density(
                    summary.get("analyses", {}), outdir=str(outdir_path)
                )
            if hasattr(_plotting, "plot_loss_vs_l2re") and summary.get("runs"):
                _plotting.plot_loss_vs_l2re(summary["runs"], outdir=str(outdir_path))
        except Exception as exc:
            print(f"(plotting failed: {exc!r})")

    print(f"[spectral] wrote {outdir_path / 'summary.json'}")
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spectral density of the PINN loss Hessian (Figures 3 and 7)."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--quick", action="store_true", help="Tiny smoke-test configuration.")
    parser.add_argument("--pdes", nargs="*", default=None, help="Subset of PDE names.")
    parser.add_argument("--widths", nargs="*", type=int, default=None, help="Network widths.")
    parser.add_argument("--lrs", nargs="*", type=float, default=None, help="Adam learning rates.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Random seeds.")
    parser.add_argument("--switch", type=int, default=None, help="Adam->L-BFGS switch iteration.")
    parser.add_argument("--total-iterations", type=int, default=None, help="Total iterations.")
    parser.add_argument("--eval-every", type=int, default=None, help="L2RE evaluation cadence.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, quick=args.quick)
    return run_spectral_density(
        cfg,
        pdes=args.pdes,
        widths=args.widths,
        adam_lrs=args.lrs,
        seeds=args.seeds,
        switch_iteration=args.switch,
        total_iterations=args.total_iterations,
        device=args.device,
        outdir=args.outdir,
        verbose=not args.quiet,
        eval_every=args.eval_every,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
