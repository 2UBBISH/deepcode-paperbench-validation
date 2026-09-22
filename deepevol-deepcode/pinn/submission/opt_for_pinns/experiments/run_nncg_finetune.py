"""NNCG fine-tuning experiment runner (Figures 1, 4, 5 and Tables 2, 3).

This script reproduces Section 7.3 / 7.4 of *Challenges in Training PINNs: A
Loss Landscape Perspective* (ICML 2024):

* For each PDE (convection / reaction / wave) we first obtain the **best**
  Adam+L-BFGS run.  Following the addendum, the "best" configuration is chosen
  by the *same systematic process as the paper*: the (network width, Adam
  learning rate, seed) triple attaining the **smallest L2RE** for that PDE.
  Whenever a previous optimizer-comparison study already produced checkpoints
  (``results/optimizer_comparison``), those records are re-used; otherwise the
  Adam+L-BFGS runs are (re-)executed here.

* Starting from that Adam+L-BFGS solution we continue training for an
  additional ``finetune_steps`` (=2000, per the addendum) iterations with

  - **NNCG** (Algorithm 4), tuning the damping ``mu`` over the paper grid
    ``{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}``, and
  - plain **gradient descent** as the control experiment.

* Outputs:
  - ``figure1``: loss curves (Adam, L-BFGS, Adam+L-BFGS, +NNCG) - NNCG keeps
    decreasing the loss after Adam+L-BFGS stalls (paper Figure 1).
  - ``figure4``: per-PDE loss and gradient-norm curves for NNCG vs GD
    (NNCG reduces the loss by >10x, GD makes no progress).
  - ``figure5``: pointwise absolute-error maps after Adam, after L-BFGS and
    after NNCG (saved both as ``.npz`` arrays and, when matplotlib is
    available, as heat-map figures).
  - ``table2``: loss / L2RE before fine-tuning, after NNCG and after GD.
  - ``table3``: NNCG / L-BFGS per-iteration wall-clock ratios (§7.4).

Usage
-----
    python experiments/run_nncg_finetune.py --quick
    python experiments/run_nncg_finetune.py --pdes convection wave \
        --widths 200 --lrs 1e-3 --seeds 345
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# path bootstrap: make ``import src.*`` work regardless of the CWD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1] if _HERE.parent.name == "experiments" else _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
from src.optimizers.first_order import (  # noqa: E402
    ADAM_LR_GRID,
    GradientDescent,
    run_first_order,
)
from src.optimizers.lbfgs_wrapper import run_lbfgs  # noqa: E402

# --- optional dependencies -------------------------------------------------
try:
    from src.optimizers.nncg import NNCG_MU_GRID, NNCGConfig, run_nncg  # type: ignore
    from src.optimizers.nncg import tune_mu as _tune_mu  # type: ignore

    NNCG_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    NNCG_MU_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    NNCGConfig = None  # type: ignore
    run_nncg = None  # type: ignore
    _tune_mu = None  # type: ignore
    NNCG_AVAILABLE = False

try:
    from src.utils.plotting import _plotting  # type: ignore

    PLOTTING_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    _plotting = None  # type: ignore
    PLOTTING_AVAILABLE = False

try:
    from src.utils.seeding import set_seed  # type: ignore
except Exception:  # pragma: no cover - defensive

    def set_seed(seed: Optional[int] = None, **kwargs: Any) -> Optional[int]:  # type: ignore
        import random

        if seed is None:
            return None
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        try:
            import numpy as _np

            _np.random.seed(int(seed) % (2**32 - 1))
        except Exception:
            pass
        return int(seed)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "problems": {
        "convection": {"beta": 40.0, "x_min": 0.0, "x_max": 2.0 * math.pi, "t_min": 0.0, "t_max": 1.0},
        "reaction": {"rho": 5.0, "x_min": 0.0, "x_max": 2.0 * math.pi, "t_min": 0.0, "t_max": 1.0},
        "wave": {"beta": 5.0, "c2": 4.0, "x_min": 0.0, "x_max": 1.0, "t_min": 0.0, "t_max": 1.0},
    },
    "pdes": ["convection", "reaction", "wave"],
    "network": {
        "depth": 3,
        "width": 200,
        "widths": [50, 100, 200, 400],
        "activation": "tanh",
        "in_dim": 2,
        "out_dim": 1,
    },
    "sampling": {
        "n_residual": 10000,
        "n_ic": 257,
        "n_bc": 101,
        "n_grid_x": 255,
        "n_grid_t": 100,
        "replace": True,
    },
    "optimizer": {
        "adam": {"lrs": list(ADAM_LR_GRID), "lr": 1e-3},
        "lbfgs": {
            "lr": 1.0,
            "history_size": 100,
            "line_search_fn": "strong_wolfe",
            "max_iter": 1,
            "max_eval": 25,
        },
        "combined": {
            "switch_iteration": DEFAULT_SWITCH_POINT,
            "switch_points": [1000, 11000, 31000],
            "total_iterations": COMBINED_TOTAL_ITERATIONS,
        },
        "gd": {"lr": 1e-4},
    },
    "nncg": {
        "eta": 1.0,
        "K": 2000,
        "s": 60,
        "F": 20,
        "mu": 1e-2,
        "mus": list(NNCG_MU_GRID),
        "epsilon": 1e-16,
        "M": 1000,
        "alpha": 0.1,
        "beta": 0.5,
        "max_backtracks": 100,
        "fallback_to_gradient": True,
        "tune_mu": True,
        "mu_selection": "loss",
    },
    "spectral": {"n_iter": 100, "n_vec": 1, "n_grid": 200},
    "experiment": {
        "seeds": [345, 456, 567, 678, 789],
        "n_seeds": 5,
        "switch_iteration": DEFAULT_SWITCH_POINT,
        "total_iterations": COMBINED_TOTAL_ITERATIONS,
        "finetune_steps": 2000,
        "adam_lrs": [1e-4, 1e-3, 1e-2],
        "widths": [50, 100, 200, 400],
        "eval_every": 50,
        "record_limit": 64,
        "selection": "l2re",
        "selection_keys": ["pde", "width", "adam_lr", "seed"],
        "include_baselines": True,
        "fig1_pdes": ["wave"],
        "lbfgs_timing_steps": 200,
        "max_checkpoint_files": 4000,
    },
    "runtime": {"dtype": "float64", "device": "cpu", "verbose": True},
    "paths": {
        "outdir": "results/nncg_finetune",
        "optimizer_comparison": "results/optimizer_comparison",
        "spectral_density": "results/spectral_density",
        "figures": "figures",
    },
    # Reference numbers from the paper (reporting only - never used for tuning).
    "paper_reference": {
        "table2": {
            "convection": {
                "adam_lbfgs": {"loss": 5.95e-6, "l2re": 4.19e-3},
                "nncg": {"loss": 3.63e-6, "l2re": 1.94e-3},
                "gd": {"loss": 5.95e-6, "l2re": 4.19e-3},
            },
            "reaction": {
                "adam_lbfgs": {"loss": 5.26e-6, "l2re": 1.92e-2},
                "nncg": {"loss": 2.89e-7, "l2re": 9.92e-3},
                "gd": {"loss": 5.26e-6, "l2re": 1.92e-2},
            },
            "wave": {
                "adam_lbfgs": {"loss": 1.12e-3, "l2re": 5.52e-2},
                "nncg": {"loss": 6.13e-5, "l2re": 1.27e-2},
                "gd": {"loss": 1.12e-3, "l2re": 5.52e-2},
            },
        },
        "table3": {"convection": 5.43, "reaction": 20.0, "wave": 322.0},
    },
    # ``--quick`` overrides: small budget smoke test of the full pipeline.
    "quick": {
        "pdes": ["convection"],
        "experiment": {
            "seeds": [345],
            "n_seeds": 1,
            "widths": [50],
            "adam_lrs": [1e-3],
            "switch_iteration": 1000,
            "total_iterations": 2000,
            "finetune_steps": 50,
            "eval_every": 10,
            "include_baselines": False,
            "fig1_pdes": [],
            "lbfgs_timing_steps": 20,
        },
        "nncg": {"K": 50, "s": 20, "F": 5, "M": 200, "mus": [1e-2, 1e-1]},
    },
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Optional[str] = None, quick: bool = False) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` (if present) merged over the defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    candidate = Path(path) if path else (_ROOT / "configs" / "default.yaml")
    if candidate.exists():
        try:
            import yaml  # type: ignore

            with open(candidate, "r") as handle:
                loaded = yaml.safe_load(handle) or {}
            # a legacy misspelling appears in the shipped config; normalise it
            combined = loaded.get("optimizer", {}).get("combined", {})
            if "switch_points" not in combined and "swich_points" in combined:
                combined["switch_points"] = combined.pop("swich_points")
            cfg = _deep_merge(cfg, loaded)
        except Exception as exc:  # pragma: no cover - config is optional
            print(f"[nncg] could not load {candidate}: {exc}")
    if quick:
        cfg = _deep_merge(cfg, cfg.get("quick", {}))
    return cfg


def supported_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter ``kwargs`` to those accepted by ``fn`` (guards against API drift)."""
    try:
        import inspect

        sig = inspect.signature(fn)
    except Exception:  # pragma: no cover
        return dict(kwargs)
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of numpy/torch objects into JSON-serialisable ones."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return to_jsonable(float(obj.detach().cpu().reshape(-1)[0]))
        return [to_jsonable(v) for v in obj.detach().cpu().reshape(-1).tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.ndarray):
            return [to_jsonable(v) for v in obj.reshape(-1).tolist()]
        if isinstance(obj, np.generic):
            return to_jsonable(obj.item())
    except Exception:
        pass
    for attr in ("as_dict", "to_dict"):
        if hasattr(obj, attr):
            try:
                return to_jsonable(getattr(obj, attr)())
            except Exception:
                pass
    return str(obj)


def device_from_arg(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_from_str(name: Any) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    text = str(name or "float64").lower()
    if "32" in text:
        return torch.float32
    return torch.float64


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _downsample(values: Sequence[float], limit: int = 256) -> List[float]:
    vals = [v for v in values]
    if len(vals) <= limit or limit <= 0:
        return [to_jsonable(v) for v in vals]
    stride = max(1, len(vals) // limit)
    out = vals[::stride]
    if out[-1] is not vals[-1]:
        out = list(out) + [vals[-1]]
    return [to_jsonable(v) for v in out]


# ---------------------------------------------------------------------------
# problem / model / loss setup
# ---------------------------------------------------------------------------
def build_setup(
    pde: str,
    width: int,
    seed: Optional[int] = None,
    *,
    problem_cfg: Optional[Dict[str, Any]] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    resample: bool = True,
) -> Dict[str, Any]:
    """Construct ``(problem, model, sampler, loss closure, eval_fn)`` for one run."""
    device = device or torch.device("cpu")
    problem_cfg = dict(problem_cfg or {})
    # drop keys that are not constructor kwargs of the problem classes
    allowed = {"beta", "rho", "c2", "x_min", "x_max", "t_min", "t_max", "nu", "alpha"}
    problem_cfg = {k: v for k, v in problem_cfg.items() if k in allowed}

    problem = get_problem(pde, **problem_cfg)

    if seed is not None:
        set_seed(seed)

    model = make_pinn(
        in_dim=2,
        out_dim=1,
        width=int(width),
        depth=int(depth),
        seed=seed,
    )
    model = model.to(device=device, dtype=dtype)

    sampler_kwargs: Dict[str, Any] = {}
    if sampling_cfg:
        sampler_kwargs = {
            "n_residual": sampling_cfg.get("n_residual"),
            "replace": sampling_cfg.get("replace", True),
        }
        sampler_kwargs = {k: v for k, v in sampler_kwargs.items() if v is not None}
    sampler = build_sampler(problem, seed=seed, device=device, dtype=dtype, **sampler_kwargs)

    loss_obj, closure = make_loss_fn(model, problem, sampler=sampler)

    def eval_fn(*_args: Any, **_kwargs: Any) -> float:
        return float(compute_l2re(model, problem, sampler, device=device, dtype=dtype))

    return {
        "problem": problem,
        "model": model,
        "sampler": sampler,
        "loss": loss_obj,
        "closure": closure,
        "eval_fn": eval_fn,
    }


def make_closure(model: torch.nn.Module, problem: Any, sampler: Any) -> Callable[[], torch.Tensor]:
    """Zero-argument loss closure bound to ``model`` (keeps the autograd graph)."""
    _loss_obj, closure = make_loss_fn(model, problem, sampler=sampler)
    return closure


def make_eval_fn(
    model: torch.nn.Module,
    problem: Any,
    sampler: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Callable[..., float]:
    def eval_fn(*_args: Any, **_kwargs: Any) -> float:
        return float(compute_l2re(model, problem, sampler, device=device, dtype=dtype))

    return eval_fn


def clone_model(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Deep copy a model so that fine-tuning never mutates the reference state."""
    clone = copy.deepcopy(model)
    if device is not None:
        clone = clone.to(device)
    return clone


def restore_state(model: torch.nn.Module, state: Any, device: Optional[torch.device] = None) -> bool:
    """Load a (possibly nested) state dict into ``model``; returns success flag."""
    if state is None:
        return False
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict", "weights"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except Exception:
        return False
    if device is not None:
        model.to(device)
    return True


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    """Outcome of a single training / fine-tuning stage."""

    pde: str
    stage: str
    n_steps: int = 0
    switch_iteration: int = 0
    adam_lr: float = float("nan")
    width: int = 0
    seed: int = -1
    mu: Optional[float] = None
    initial_loss: float = float("nan")
    initial_l2re: float = float("nan")
    final_loss: float = float("nan")
    best_loss: float = float("nan")
    final_l2re: float = float("nan")
    best_l2re: float = float("nan")
    final_grad_norm: float = float("nan")
    loss_history: List[float] = field(default_factory=list)
    loss_steps: List[int] = field(default_factory=list)
    l2re_history: List[float] = field(default_factory=list)
    l2re_steps: List[int] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    grad_norm_steps: List[int] = field(default_factory=list)
    seconds: float = float("nan")
    iterations: int = 0
    time_per_iteration: float = float("nan")
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # runtime-only payloads (never serialised blindly)
    model: Optional[torch.nn.Module] = field(default=None, repr=False, compare=False)
    lbfgs_history: Any = field(default=None, repr=False, compare=False)

    def to_metadata(self, history_limit: int = 256) -> Dict[str, Any]:
        payload = {
            "pde": self.pde,
            "stage": self.stage,
            "n_steps": self.n_steps,
            "switch_iteration": self.switch_iteration,
            "adam_lr": to_jsonable(self.adam_lr),
            "width": self.width,
            "seed": self.seed,
            "mu": to_jsonable(self.mu),
            "initial_loss": to_jsonable(self.initial_loss),
            "initial_l2re": to_jsonable(self.initial_l2re),
            "final_loss": to_jsonable(self.final_loss),
            "best_loss": to_jsonable(self.best_loss),
            "final_l2re": to_jsonable(self.final_l2re),
            "best_l2re": to_jsonable(self.best_l2re),
            "final_grad_norm": to_jsonable(self.final_grad_norm),
            "seconds": to_jsonable(self.seconds),
            "iterations": self.iterations,
            "time_per_iteration": to_jsonable(self.time_per_iteration),
            "loss_history": _downsample(self.loss_history, history_limit),
            "loss_steps": _downsample(self.loss_steps, history_limit),
            "l2re_history": _downsample(self.l2re_history, history_limit),
            "l2re_steps": _downsample(self.l2re_steps, history_limit),
            "grad_norm_history": _downsample(self.grad_norm_history, history_limit),
            "grad_norm_steps": _downsample(self.grad_norm_steps, history_limit),
            "extra": to_jsonable(self.extra),
            "error": self.error,
        }
        return payload


def _history_arrays(history: Any, attr: str) -> List[float]:
    """Extract a numeric list from a TrainingHistory/NNCGHistory-like object."""
    if history is None:
        return []
    values = getattr(history, attr, None)
    if values is None and isinstance(history, dict):
        values = history.get(attr)
    if values is None:
        return []
    try:
        return [float(v) for v in values]
    except TypeError:
        return []


def _history_steps(history: Any, values_len: int) -> List[int]:
    steps = getattr(history, "steps", None)
    if steps is None and isinstance(history, dict):
        steps = history.get("steps")
    try:
        steps = [int(s) for s in steps]
        if len(steps) == values_len:
            return steps
    except TypeError:
        pass
    return list(range(len_ for len_ in range(1, values_len + 1))) if values_len else []


# ---------------------------------------------------------------------------
# stage runners
# ---------------------------------------------------------------------------
def run_adam_stage(
    pde: str,
    width: int,
    adam_lr: float,
    seed: int,
    *,
    n_steps: int = DEFAULT_SWITCH_POINT,
    problem_cfg: Optional[Dict[str, Any]] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 500,
    setup: Optional[Dict[str, Any]] = None,
) -> StageResult:
    """Adam-only training (used for the "after Adam" snapshots / Figure 5)."""
    from src.optimizers.first_order import AdamOptimizer

    setup = setup or build_setup(
        pde,
        width,
        seed,
        problem_cfg=problem_cfg,
        sampling_cfg=sampling_cfg,
        depth=depth,
        dtype=dtype,
        device=device,
    )
    model, problem, sampler = setup["model"], setup["problem"], setup["sampler"]
    closure = setup["closure"]
    eval_fn = setup["eval_fn"]

    record = StageResult(
        pde=pde, stage="adam", n_steps=int(n_steps), adam_lr=float(adam_lr), width=int(width), seed=int(seed)
    )
    try:
        record.initial_loss = float(closure().detach())
    except Exception:
        record.initial_loss = float(closure())
    record.initial_l2re = _num(eval_fn())

    optimizer = AdamOptimizer(model, lr=float(adam_lr))
    t0 = time.time()
    history = run_first_order(
        model,
        closure,
        optimizer,
        int(n_steps),
        eval_fn=eval_fn,
        eval_every=int(eval_every),
    )
    record.seconds = time.time() - t0
    record.iterations = int(n_steps)
    record.time_per_iteration = record.seconds / max(1, int(n_steps))
    record.loss_history = _history_arrays(history, "losses")
    record.loss_steps = _history_steps(history, len(record.loss_history))
    record.l2re_history = _history_arrays(history, "l2re")
    record.l2re_steps = _history_steps(history, len(record.l2re_history))
    record.grad_norm_history = _history_arrays(history, "grad_norms")
    record.grad_norm_steps = _history_steps(history, len(record.grad_norm_history))
    record.final_loss = _num(record.loss_history[-1], record.initial_loss)
    final_l2re = _num(eval_fn())
    record.final_l2re = final_l2re if not math.isnan(final_l2re) else _num(record.l2re_history[-1])
    record.best_loss = _num(getattr(history, "best_loss", None), record.final_loss)
    record.best_l2re = _num(getattr(history, "best_l2re", None), record.final_l2re)
    if record.grad_norm_history:
        record.final_grad_norm = record.grad_norm_history[-1]
    record.model = model
    return record


def run_combined_stage(
    pde: str,
    width: int,
    adam_lr: float,
    seed: int,
    *,
    switch_iteration: int = DEFAULT_SWITCH_POINT,
    total_iterations: int = COMBINED_TOTAL_ITERATIONS,
    problem_cfg: Optional[Dict[str, Any]] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 500,
    record_limit: int = 64,
    lbfgs_cfg: Optional[Dict[str, Any]] = None,
    setup: Optional[Dict[str, Any]] = None,
) -> StageResult:
    """Run Adam for ``switch_iteration`` steps then L-BFGS for the remainder."""
    setup = setup or build_setup(
        pde,
        width,
        seed,
        problem_cfg=problem_cfg,
        sampling_cfg=sampling_cfg,
        depth=depth,
        dtype=dtype,
        device=device,
    )
    model, closure, eval_fn = setup["model"], setup["closure"], setup["eval_fn"]

    record = StageResult(
        pde=pde,
        stage="adam+lbfgs",
        n_steps=int(total_iterations),
        switch_iteration=int(switch_iteration),
        adam_lr=float(adam_lr),
        width=int(width),
        seed=int(seed),
    )
    try:
        record.initial_loss = float(closure().detach())
    except Exception:
        record.initial_loss = float(closure())
    record.initial_l2re = _num(eval_fn())

    lbfgs_kwargs = supported_kwargs(run_adam_lbfgs, dict(lbfgs_cfg or {}))
    t0 = time.time()
    result = run_adam_lbfgs(
        model,
        closure,
        adam_lr=float(adam_lr),
        switch_iteration=int(switch_iteration),
        total_iterations=int(total_iterations),
        eval_fn=eval_fn,
        eval_every=int(eval_every),
        record=True,
        record_limit=int(record_limit),
        **lbfgs_kwargs,
    )
    record.seconds = time.time() - t0
    record.iterations = int(total_iterations)
    record.time_per_iteration = record.seconds / max(1, int(total_iterations))

    history = getattr(result, "history", None)
    record.loss_history = _history_arrays(history, "losses")
    record.loss_steps = _history_steps(history, len(record.loss_history))
    record.l2re_history = _history_arrays(history, "l2re")
    record.l2re_steps = _history_steps(history, len(record.l2re_history))
    record.grad_norm_history = _history_arrays(history, "grad_norms")
    record.grad_norm_steps = _history_steps(history, len(record.grad_norm_history))
    record.final_loss = _num(getattr(result, "final_loss", None), record.initial_loss)
    record.best_loss = _num(getattr(result, "best_loss", None), record.final_loss)
    final_l2re = _num(eval_fn())
    record.final_l2re = final_l2re if not math.isnan(final_l2re) else _num(record.l2re_history[-1])
    record.best_l2re = _num(getattr(result, "best_l2re", None), record.final_l2re)
    if record.grad_norm_history:
        record.final_grad_norm = record.grad_norm_history[-1]
    record.lbfgs_history = getattr(result, "lbfgs_history", None)
    record.model = model
    record.extra = {"switch_step": to_jsonable(getattr(result, "switch_step", None))}
    return record


def run_lbfgs_stage(
    pde: str,
    width: int,
    seed: int,
    *,
    adam_lr: float = float("nan"),
    n_steps: int = COMBINED_TOTAL_ITERATIONS,
    problem_cfg: Optional[Dict[str, Any]] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 500,
    lbfgs_cfg: Optional[Dict[str, Any]] = None,
    setup: Optional[Dict[str, Any]] = None,
) -> StageResult:
    """L-BFGS-only baseline (used for the Figure 1 stall curve and Table 3 timing)."""
    setup = setup or build_setup(
        pde,
        width,
        seed,
        problem_cfg=problem_cfg,
        sampling_cfg=sampling_cfg,
        depth=depth,
        dtype=dtype,
        device=device,
    )
    model, closure, eval_fn = setup["model"], setup["closure"], setup["eval_fn"]

    record = StageResult(
        pde=pde,
        stage="lbfgs",
        n_steps=int(n_steps),
        adam_lr=float(adam_lr),
        width=int(width),
        seed=int(seed),
    )
    try:
        record.initial_loss = float(closure().detach())
    except Exception:
        record.initial_loss = float(closure())
    record.initial_l2re = _num(eval_fn())

    kwargs = supported_kwargs(run_lbfgs, dict(lbfgs_cfg or {}))
    kwargs.pop("history_size", None)
    t0 = time.time()
    history = run_lbfgs(
        model,
        closure,
        n_steps=int(n_steps),
        eval_fn=eval_fn,
        eval_every=int(eval_every),
        **kwargs,
    )
    record.seconds = time.time() - t0
    record.iterations = int(getattr(history, "n_steps", 0) or n_steps)
    record.time_per_iteration = record.seconds / max(1, record.iterations)

    record.loss_history = _history_arrays(history, "losses")
    record.loss_steps = _history_steps(history, len(record.loss_history))
    record.l2re_history = _history_arrays(history, "l2re")
    record.l2re_steps = _history_steps(history, len(record.l2re_history))
    record.grad_norm_history = _history_arrays(history, "grad_norms")
    record.grad_norm_steps = _history_steps(history, len(record.grad_norm_history))
    record.final_loss = _num(record.loss_history[-1] if record.loss_history else None, record.initial_loss)
    record.best_loss = _num(getattr(history, "best_loss", None), record.final_loss)
    final_l2re = _num(eval_fn())
    record.final_l2re = final_l2re if not math.isnan(final_l2re) else _num(record.l2re_history[-1])
    record.best_l2re = _num(getattr(history, "best_l2re", None), record.final_l2re)
    if record.grad_norm_history:
        record.final_grad_norm = record.grad_norm_history[-1]
    record.model = model
    return record


def run_nncg_stage(
    pde: str,
    base_model: torch.nn.Module,
    problem: Any,
    sampler: Any,
    *,
    mu: float,
    n_steps: int,
    nncg_cfg: Dict[str, Any],
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 50,
    width: int = 0,
    seed: int = -1,
    adam_lr: float = float("nan"),
    initial_loss: float = float("nan"),
    initial_l2re: float = float("nan"),
    verbose: bool = False,
) -> StageResult:
    """Fine-tune ``base_model`` with NNCG for ``n_steps`` iterations at damping ``mu``."""
    if not NNCG_AVAILABLE or run_nncg is None:  # pragma: no cover - defensive
        raise RuntimeError("NNCG implementation unavailable (src.optimizers.nncg)")

    model = clone_model(base_model, device=device)
    closure = make_closure(model, problem, sampler)
    eval_fn = make_eval_fn(model, problem, sampler, device=device, dtype=dtype)

    record = StageResult(
        pde=pde,
        stage="nncg",
        n_steps=int(n_steps),
        adam_lr=float(adam_lr),
        width=int(width),
        seed=int(seed),
        mu=float(mu),
    )
    record.initial_loss = float(initial_loss) if not math.isnan(initial_loss) else _num(
        float(closure().detach())
    )
    record.initial_l2re = float(initial_l2re) if not math.isnan(initial_l2re) else _num(eval_fn())

    kwargs = dict(nncg_cfg or {})
    kwargs.update(
        {
            "mu": float(mu),
            "n_steps": int(n_steps),
            "eval_fn": eval_fn,
            "eval_every": int(eval_every),
            "verbose": bool(verbose),
            "dtype": dtype,
            "seed": int(seed) if seed is not None and seed >= 0 else None,
        }
    )
    kwargs = supported_kwargs(run_nncg, kwargs)
    t0 = time.time()
    result = run_nncg(model, closure, **kwargs)
    record.seconds = time.time() - t0

    history = getattr(result, "history", None)
    record.loss_history = _history_arrays(history, "losses")
    record.loss_steps = _history_steps(history, len(record.loss_history))
    record.l2re_history = _history_arrays(history, "l2re")
    record.l2re_steps = _history_steps(history, len(record.l2re_history))
    record.grad_norm_history = _history_arrays(history, "grad_norms")
    record.grad_norm_steps = _history_steps(history, len(record.grad_norm_history))

    record.iterations = int(getattr(history, "n_steps", 0) or n_steps)
    record.time_per_iteration = _num(
        getattr(history, "time_per_iteration", None),
        record.seconds / max(1, record.iterations),
    )
    record.final_loss = _num(getattr(result, "final_loss", None), _num(record.loss_history[-1] if record.loss_history else None))
    record.best_loss = _num(getattr(result, "best_loss", None), record.final_loss)
    final_l2re = _num(eval_fn())
    record.final_l2re = final_l2re if not math.isnan(final_l2re) else _num(record.l2re_history[-1])
    record.best_l2re = _num(getattr(result, "best_l2re", None), record.final_l2re)
    if record.grad_norm_history:
        record.final_grad_norm = record.grad_norm_history[-1]
    record.model = model
    record.extra = {
        "mean_pcg_iterations": to_jsonable(getattr(history, "mean_pcg_iterations", None)),
        "n_refreshes": to_jsonable(getattr(history, "n_refreshes", None)),
        "total_time": to_jsonable(getattr(history, "total_time", None)),
    }
    return record


def run_gd_stage(
    pde: str,
    base_model: torch.nn.Module,
    problem: Any,
    sampler: Any,
    *,
    n_steps: int,
    lr: float,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 50,
    width: int = 0,
    seed: int = -1,
    adam_lr: float = float("nan"),
    initial_loss: float = float("nan"),
    initial_l2re: float = float("nan"),
) -> StageResult:
    """Gradient-descent control run (paper Figure 4 / Table 2)."""
    model = clone_model(base_model, device=device)
    closure = make_closure(model, problem, sampler)
    eval_fn = make_eval_fn(model, problem, sampler, device=device, dtype=dtype)

    record = StageResult(
        pde=pde,
        stage="gd",
        n_steps=int(n_steps),
        adam_lr=float(adam_lr),
        width=int(width),
        seed=int(seed),
        mu=None,
    )
    record.initial_loss = float(initial_loss) if not math.isnan(initial_loss) else _num(
        float(closure().detach())
    )
    record.initial_l2re = float(initial_l2re) if not math.isnan(initial_l2re) else _num(eval_fn())

    optimizer = GradientDescent(model, lr=float(lr))
    t0 = time.time()
    history = run_first_order(
        model,
        closure,
        optimizer,
        int(n_steps),
        eval_fn=eval_fn,
        eval_every=int(eval_every),
    )
    record.seconds = time.time() - t0
    record.iterations = int(n_steps)
    record.time_per_iteration = record.seconds / max(1, int(n_steps))

    record.loss_history = _history_arrays(history, "losses")
    record.loss_steps = _history_steps(history, len(record.loss_history))
    record.l2re_history = _history_arrays(history, "l2re")
    record.l2re_steps = _history_steps(history, len(record.l2re_history))
    record.grad_norm_history = _history_arrays(history, "grad_norms")
    record.grad_norm_steps = _history_steps(history, len(record.grad_norm_history))
    record.final_loss = _num(record.loss_history[-1] if record.loss_history else None, record.initial_loss)
    record.best_loss = _num(getattr(history, "best_loss", None), record.final_loss)
    final_l2re = _num(eval_fn())
    record.final_l2re = final_l2re if not math.isnan(final_l2re) else _num(record.l2re_history[-1])
    record.best_l2re = _num(getattr(history, "best_l2re", None), record.final_l2re)
    if record.grad_norm_history:
        record.final_grad_norm = record.grad_norm_history[-1]
    record.model = model
    record.extra = {"lr": float(lr)}
    return record


def tune_mu_stages(
    pde: str,
    base_model: torch.nn.Module,
    problem: Any,
    sampler: Any,
    *,
    mus: Sequence[float],
    n_steps: int,
    nncg_cfg: Dict[str, Any],
    select_by: str = "loss",
    **kwargs: Any,
) -> Tuple[StageResult, List[StageResult]]:
    """Grid-search the NNCG damping ``mu`` (paper grid {1e-5,...,1e-1}).

    Each trial restarts from an exact copy of the Adam+L-BFGS solution, which is
    the configuration reported in the paper (the run with the best final loss /
    L2RE is selected, ties broken by the paper's mu grid order).
    """
    results: List[StageResult] = []
    for mu in mus:
        try:
            stage = run_nncg_stage(
                pde,
                base_model,
                problem,
                sampler,
                mu=float(mu),
                n_steps=int(n_steps),
                nncg_cfg=nncg_cfg,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - defensive
            failed = StageResult(pde=pde, stage="nncg", mu=float(mu), n_steps=int(n_steps))
            failed.error = f"{type(exc).__name__}: {exc}"
            results.append(failed)
            continue
        results.append(stage)

    ok = [r for r in results if r.error is None and not math.isnan(r.final_loss)]
    if not ok:
        best = results[0] if results else StageResult(pde=pde, stage="nncg")
        return best, results

    key = "best_l2re" if select_by == "l2re" else "final_loss"
    best = min(ok, key=lambda r: (_num(getattr(r, key), float("inf")), _num(r.final_loss, float("inf"))))
    return best, results


# ---------------------------------------------------------------------------
# best Adam+L-BFGS configuration: checkpoint reuse or systematic selection
# ---------------------------------------------------------------------------
def _iter_json_records(payload: Any) -> List[Dict[str, Any]]:
    """Collect record-like dicts from a summary.json payload."""
    found: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            found.extend(_iter_json_records(item))
    elif isinstance(payload, dict):
        if {"pde", "l2re"} <= set(payload.keys()) or {"pde", "best_l2re"} <= set(payload.keys()):
            found.append(payload)
        for value in payload.values():
            if isinstance(value, (dict, list)):
                found.extend(_iter_json_records(value))
    return found


def load_comparison_records(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Read any optimizer-comparison summaries that may already exist."""
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        records.extend(_iter_json_records(payload))
    return records


def _optimizer_family(record: Dict[str, Any]) -> str:
    for key in ("family", "optimizer", "optimiser", "method", "name"):
        value = record.get(key)
        if isinstance(value, str):
            text = value.lower().replace("_", "").replace("-", "").replace("+", "")
            if "adam" in text and "lbfgs" in text:
                return "adam+lbfgs"
            if text.startswith("adam"):
                return "adam"
            if "lbfgs" in text:
                return "lbfgs"
    return ""


def _resolve_checkpoint(checkpoints_dir: Optional[Path], record: Dict[str, Any]) -> Optional[Path]:
    """Locate the checkpoint file backing a comparison record."""
    for key in ("checkpoint", "checkpoint_path", "state_dict_path", "path", "file", "ckpt"):
        value = record.get(key)
        if isinstance(value, str):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = _ROOT / candidate
            if candidate.exists():
                return candidate
    if checkpoints_dir is None or not checkpoints_dir.exists():
        return None
    files = sorted(glob.glob(str(checkpoints_dir / "*.pt")))
    if not files:
        return None
    tokens = []
    for key in ("pde", "width", "adam_lr", "lr", "seed"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            tokens.append(f"{value:g}")
        else:
            tokens.append(str(value))
    best, best_score = None, -1
    for file in files:
        name = os.path.basename(file).lower()
        score = sum(1 for token in tokens if token and token.lower() in name)
        if score > best_score:
            best, best_score = Path(file), score
    if best is not None and best_score >= 2:
        return best
    return None


def select_best_from_records(
    records: Sequence[Dict[str, Any]],
    *,
    checkpoints_dir: Optional[Path] = None,
    switch_iteration: int = DEFAULT_SWITCH_POINT,
    criterion: str = "l2re",
) -> Dict[str, Dict[str, Any]]:
    """Per-PDE argmin over (width, adam_lr, seed) - the paper's selection rule.

    Following the addendum, only Adam+L-BFGS runs that switch at 11000
    iterations are considered for the spectral / NNCG studies.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if _optimizer_family(record) != "adam+lbfgs":
            continue
        pde = record.get("pde")
        if not isinstance(pde, str):
            continue
        switch = record.get("switch_iteration")
        if switch is not None:
            try:
                if int(switch) != int(switch_iteration):
                    continue
            except (TypeError, ValueError):
                pass
        l2re = record.get("l2re", record.get("best_l2re", record.get("final_l2re")))
        loss = record.get("best_loss", record.get("final_loss"))
        l2re = _num(l2re, float("inf"))
        loss = _num(loss, float("inf"))
        key = (l2re, loss)
        entry = {
            "record": record,
            "l2re": l2re,
            "loss": loss,
            "width": record.get("width"),
            "adam_lr": record.get("adam_lr", record.get("lr")),
            "seed": record.get("seed"),
            "switch_iteration": int(switch_iteration),
            "checkpoint": _resolve_checkpoint(checkpoints_dir, record),
        }
        current = best.get(pde)
        if current is None:
            best[pde] = entry
            continue
        current_key = (
            current["l2re"] if criterion == "l2re" else current["loss"],
            current["loss"] if criterion == "l2re" else current["l2re"],
        )
        if key < current_key:
            best[pde] = entry
    return best


def sweep_best_adam_lbfgs(
    pde: str,
    *,
    widths: Sequence[int],
    adam_lrs: Sequence[float],
    seeds: Sequence[int],
    switch_iteration: int,
    total_iterations: int,
    problem_cfg: Optional[Dict[str, Any]] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 500,
    record_limit: int = 64,
    lbfgs_cfg: Optional[Dict[str, Any]] = None,
    criterion: str = "l2re",
    verbose: bool = True,
) -> Tuple[Optional[StageResult], List[StageResult]]:
    """Run the (width x Adam lr x seed) grid and keep the smallest-L2RE run."""
    trials: List[StageResult] = []
    for width in widths:
        for adam_lr in adam_lrs:
            for seed in seeds:
                try:
                    stage = run_combined_stage(
                        pde,
                        int(width),
                        float(adam_lr),
                        int(seed),
                        switch_iteration=int(switch_iteration),
                        total_iterations=int(total_iterations),
                        problem_cfg=problem_cfg,
                        sampling_cfg=sampling_cfg,
                        depth=depth,
                        dtype=dtype,
                        device=device,
                        eval_every=eval_every,
                        record_limit=record_limit,
                        lbfgs_cfg=lbfgs_cfg,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    stage = StageResult(pde=pde, stage="adam+lbfgs", width=int(width), seed=int(seed))
                    stage.adam_lr = float(adam_lr)
                    stage.error = f"{type(exc).__name__}: {exc}"
                trials.append(stage)
                if verbose:
                    print(
                        f"[nncg] {pde} w={width} lr={adam_lr:g} seed={seed} "
                        f"loss={stage.final_loss:.4e} l2re={stage.final_l2re:.4e} "
                        f"({stage.seconds:.1f}s)"
                    )
    ok = [t for t in trials if t.error is None and not math.isnan(t.final_l2re)]
    if not ok:
        ok = [t for t in trials if t.error is None]
    if not ok:
        return None, trials
    key = "final_l2re" if criterion == "l2re" else "final_loss"
    best = min(ok, key=lambda t: (_num(getattr(t, key), float("inf")), _num(t.final_loss, float("inf"))))
    return best, trials


# ---------------------------------------------------------------------------
# Figure 5: pointwise absolute errors
# ---------------------------------------------------------------------------
def absolute_error_map(
    model: torch.nn.Module,
    problem: Any,
    sampler: Any,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """|u(x,t) - u*(x,t)| on the full 255x100 interior grid (Figure 5)."""
    x_grid = getattr(sampler, "eval_grid_x", None)
    t_grid = getattr(sampler, "eval_grid_t", None)
    if x_grid is None or t_grid is None:
        x_grid, t_grid = problem.interior_grid_1d(device=device, dtype=dtype)
    x_grid = x_grid.to(dtype=dtype)
    t_grid = t_grid.to(dtype=dtype)
    xx, tt = torch.meshgrid(x_grid, t_grid, indexing="ij")  # (n_x, n_t)
    points = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1).to(device=device, dtype=dtype)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(points).detach().reshape(-1)
        exact = problem.exact(points).detach().reshape(-1)
    if was_training:
        model.train()

    err = (pred - exact).abs().reshape(x_grid.numel(), t_grid.numel())
    return {
        "error": err.cpu().numpy(),
        "x": x_grid.cpu().numpy(),
        "t": t_grid.cpu().numpy(),
        "max_error": float(err.max()),
        "l2re": float(
            torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact).clamp_min(1e-30)
        ),
    }


def plot_error_maps(maps: Dict[str, Dict[str, Any]], outfile: Path, *, title: Optional[str] = None) -> Optional[str]:
    """Heat-map figure of the absolute error at each optimizer switch point."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    stages = list(maps.keys())
    if not stages:
        return None
    fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 3.6), squeeze=False)
    for ax, stage in zip(axes[0], stages):
        payload = maps[stage]
        err = payload["error"]
        im = ax.pcolormesh(payload["x"], payload["t"], err.T, shading="auto")
        ax.set_title(stage.replace("_", " "))
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(outfile)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def build_table2(rows: Sequence[Dict[str, Any]], *, include_reference: bool = True) -> Dict[str, Any]:
    """Loss / L2RE of Adam+L-BFGS before and after NNCG / GD fine-tuning."""
    table: Dict[str, Any] = {"rows": {}, "reference": None}
    for row in rows:
        pde = row["pde"]
        table["rows"][pde] = {
            "adam_lbfgs": {
                "loss": row.get("adam_lbfgs_loss"),
                "l2re": row.get("adam_lbfgs_l2re"),
            },
            "adam_lbfgs_nncg": {
                "loss": row.get("nncg_loss"),
                "l2re": row.get("nncg_l2re"),
                "mu": row.get("nncg_mu"),
            },
            "adam_lbfgs_gd": {
                "loss": row.get("gd_loss"),
                "l2re": row.get("gd_l2re"),
            },
        }
    if include_reference:
        table["reference"] = DEFAULT_CONFIG["paper_reference"]["table2"]
    return table


def format_table2(table: Dict[str, Any]) -> str:
    lines = []
    header = f"{'PDE':<12}{'Optimizer':<22}{'Loss':>14}{'L2RE':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for pde, groups in table.get("rows", {}).items():
        for key in ("adam_lbfgs", "adam_lbfgs_nncg", "adam_lbfgs_gd"):
            entry = groups.get(key, {})
            loss = entry.get("loss")
            l2re = entry.get("l2re")
            loss_s = f"{loss:.3e}" if isinstance(loss, (int, float)) else "n/a"
            l2re_s = f"{l2re:.3e}" if isinstance(l2re, (int, float)) else "n/a"
            lines.append(f"{pde:<12}{key:<22}{loss_s:>14}{l2re_s:>14}")
        lines.append("-" * len(header))
    return "\n".join(lines)


def build_table3(rows: Sequence[Dict[str, Any]], *, include_reference: bool = True) -> Dict[str, Any]:
    """Per-iteration wall-clock ratio NNCG / L-BFGS."""
    table: Dict[str, Any] = {"rows": {}, "reference": None}
    for row in rows:
        table["rows"][row["pde"]] = {
            "nncg_time_per_iteration": row.get("nncg_time_per_iteration"),
            "lbfgs_time_per_iteration": row.get("lbfgs_time_per_iteration"),
            "ratio": row.get("time_ratio"),
            "nncg_iterations": row.get("nncg_iterations"),
            "lbfgs_iterations": row.get("lbfgs_iterations"),
        }
    if include_reference:
        table["reference"] = DEFAULT_CONFIG["paper_reference"]["table3"]
    return table


def format_table3(table: Dict[str, Any]) -> str:
    lines = [f"{'PDE':<12}{'NNCG s/it':>14}{'L-BFGS s/it':>14}{'ratio':>10}"]
    lines.append("-" * len(lines[0]))
    for pde, entry in table.get("rows", {}).items():
        nncg = entry.get("nncg_time_per_iteration")
        lbfgs = entry.get("lbfgs_time_per_iteration")
        ratio = entry.get("ratio")
        fmt = lambda v: f"{v:.4g}" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
        lines.append(f"{pde:<12}{fmt(nncg):>14}{fmt(lbfgs):>14}{fmt(ratio):>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------
def run_nncg_finetune(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    pdes: Optional[Sequence[str]] = None,
    widths: Optional[Sequence[int]] = None,
    adam_lrs: Optional[Sequence[float]] = None,
    seeds: Optional[Sequence[int]] = None,
    switch_iteration: Optional[int] = None,
    total_iterations: Optional[int] = None,
    finetune_steps: Optional[int] = None,
    device: Optional[str] = None,
    outdir: Optional[str] = None,
    verbose: bool = True,
    eval_every: Optional[int] = None,
    make_plots: bool = True,
    use_comparison_checkpoints: bool = True,
    include_baselines: Optional[bool] = None,
) -> Dict[str, Any]:
    """Fine-tune the best Adam+L-BFGS run of every PDE with NNCG and with GD."""
    cfg = copy.deepcopy(cfg or load_config())
    exp_cfg = cfg.get("experiment", {})
    nncg_cfg = cfg.get("nncg", {})
    runtime = cfg.get("runtime", {})
    paths_cfg = cfg.get("paths", {})

    pdes = list(pdes if pdes is not None else cfg.get("pdes", list(PROBLEMS.keys())))
    widths = list(widths if widths is not None else exp_cfg.get("widths", [200]))
    adam_lrs = list(adam_lrs if adam_lrs is not None else exp_cfg.get("adam_lrs", [1e-3]))
    seeds = list(seeds if seeds is not None else exp_cfg.get("seeds", [345]))
    switch = int(switch_iteration or exp_cfg.get("switch_iteration", DEFAULT_SWITCH_POINT))
    total = int(total_iterations or exp_cfg.get("total_iterations", COMBINED_TOTAL_ITERATIONS))
    finetune = int(finetune_steps or nncg_cfg.get("K", exp_cfg.get("finetune_steps", 2000)))
    eval_every = int(eval_every or exp_cfg.get("eval_every", 500))
    record_limit = int(exp_cfg.get("record_limit", 64))
    dtype = _dtype_from_str(runtime.get("dtype", "float64"))
    dev = device_from_arg(device or runtime.get("device"))
    if include_baselines is None:
        include_baselines = bool(exp_cfg.get("include_baselines", True))
    fig1_pdes = list(exp_cfg.get("fig1_pdes", []))

    outdir = Path(outdir or paths_cfg.get("outdir", "results/nncg_finetune"))
    if not outdir.is_absolute():
        outdir = _ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(paths_cfg.get("figures", "figures"))
    if not figures_dir.is_absolute():
        figures_dir = _ROOT / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "config": {
            "pdes": pdes,
            "widths": widths,
            "adam_lrs": adam_lrs,
            "seeds": seeds,
            "switch_iteration": switch,
            "total_iterations": total,
            "finetune_steps": finetune,
            "nncg_mus": list(nncg_cfg.get("mus", NNCG_MU_GRID)),
            "gd_lr": cfg.get("optimizer", {}).get("gd", {}).get("lr", 1e-4),
            "device": str(dev),
            "dtype": str(dtype),
        },
        "pdes": {},
        "table2": None,
        "table3": None,
        "figures": {},
        "histories": {"finetune": {}},
        "errors": [],
    }
    if not NNCG_AVAILABLE:
        summary["errors"].append("NNCG implementation unavailable: skipping fine-tuning stages")

    table2_rows: List[Dict[str, Any]] = []
    table3_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # locate (or produce) the best Adam+L-BFGS run of each PDE
    # ------------------------------------------------------------------
    checkpoint_dir = None
    comparison_unit = paths_cfg.get("optimizer_comparison", "results/optimizer_comparison")
    comparison_base = Path(comparison_unit)
    if not comparison_base.is_absolute():
        comparison_base = _ROOT / comparison_base
    checkpoint_dir = comparison_base / "checkpoints"

    selected: Dict[str, Dict[str, Any]] = {}
    if use_comparison_checkpoints:
        candidate_summaries = [
            comparison_base / "summary.json",
            comparison_base / "table1.json",
            comparison_base / "progress.json",
            outdir / "best_adam_lbfgs.json",
        ]
        comparison_records = load_comparison_records(candidate_summaries)
        if comparison_records:
            selected = select_best_from_records(
                comparison_records, checkpoints_dir=checkpoint_dir, switch_iteration=switch
            )
            if verbose and selected:
                print(
                    "[nncg] reusing optimizer-comparison checkpoints for "
                    + ", ".join(sorted(selected.keys()))
                )

    for pde in pdes:
        if pde not in PROBLEMS:
            summary["errors"].append(f"unknown PDE '{pde}' (available: {sorted(PROBLEMS)})")
            continue

        problem_cfg = cfg.get("problems", {}).get(pde, {})
        sampling_cfg = cfg.get("sampling", {})
        depth = int(cfg.get("network", {}).get("depth", 3))
        lbfgs_cfg = cfg.get("optimizer", {}).get("lbfgs", {})

        entry: Dict[str, Any] = {"pde": pde, "selection": {}, "stages": {}, "errors": []}
        if verbose:
            print(f"\n[nncg] === {pde} ===")

        base_stage: Optional[StageResult] = None
        base_setup: Optional[Dict[str, Any]] = None

        # --- 1) try to reuse a stored checkpoint -----------------------
        chosen = selected.get(pde)
        if chosen is not None:
            width_c = int(chosen.get("width") or (widths[0] if widths else 200))
            lr_c = float(_num(chosen.get("adam_lr"), adam_lrs[0] if adam_lrs else 1e-3))
            seed_c = int(chosen.get("seed") or (seeds[0] if seeds else 345))
            setup = build_setup(
                pde,
                width_c,
                seed_c,
                problem_cfg=problem_cfg,
                sampling_cfg=sampling_cfg,
                depth=depth,
                dtype=dtype,
                device=dev,
            )
            loaded = False
            path = chosen.get("checkpoint")
            if path is not None and Path(path).exists():
                try:
                    state = torch.load(str(path), map_location="cpu")
                    loaded = restore_state(setup["model"], state, device=dev)
                except Exception as exc:
                    entry["errors"].append(f"checkpoint load failed: {exc}")
            if loaded:
                stage = StageResult(
                    pde=pde,
                    stage="adam+lbfgs",
                    width=width_c,
                    adam_lr=lr_c,
                    seed=seed_c,
                    switch_iteration=switch,
                    n_steps=total,
                    initial_loss=float("nan"),
                    final_loss=_num(chosen.get("loss")),
                    best_loss=_num(chosen.get("loss")),
                    final_l2re=_num(chosen.get("l2re")),
                    best_l2re=_num(chosen.get("l2re")),
                )
                stage.model = setup["model"]
                base_stage = stage
                base_setup = setup
                entry["selection"] = {
                    "source": "checkpoint",
                    "checkpoint": str(path),
                    "width": width_c,
                    "adam_lr": lr_c,
                    "seed": seed_c,
                    "loss": stage.final_loss,
                    "l2re": stage.final_l2re,
                }
                if verbose:
                    print(f"[nncg] loaded {pde} checkpoint ({path.name}): l2re={stage.final_l2re:.4e}")
            else:
                entry["errors"].append("checkpoint present but could not be loaded; re-running sweep")

        # --- 2) otherwise follow the paper's selection process ---------
        if base_stage is None:
            best, trials = sweep_best_adam_lbfgs(
                pde,
                widths=widths,
                adam_lrs=adam_lrs,
                seeds=seeds,
                switch_iteration=switch,
                total_iterations=total,
                problem_cfg=problem_cfg,
                sampling_cfg=sampling_cfg,
                depth=depth,
                dtype=dtype,
                device=dev,
                eval_every=eval_every,
                record_limit=record_limit,
                lbfgs_cfg=lbfgs_cfg,
                criterion=exp_cfg.get("selection", "l2re"),
                verbose=verbose,
            )
            entry["selection_trials"] = [t.to_metadata(32) for t in trials]
            if best is None:
                entry["errors"].append("no successful Adam+L-BFGS run; skipping PDE")
                summary["pdes"][pde] = entry
                continue
            base_stage = best
            base_setup = build_setup(
                pde,
                best.width,
                best.seed,
                problem_cfg=problem_cfg,
                sampling_cfg=sampling_cfg,
                depth=depth,
                dtype=dtype,
                device=dev,
            )
            base_setup["model"].load_state_dict(best.model.state_dict())
            base_setup["closure"] = make_closure(base_setup["model"], base_setup["problem"], base_setup["sampler"])
            base_setup["eval_fn"] = make_eval_fn(
                base_setup["model"], base_setup["problem"], base_setup["sampler"], device=dev, dtype=dtype
            )
            entry["selection"] = {
                "source": "sweep",
                "width": best.width,
                "adam_lr": best.adam_lr,
                "seed": best.seed,
                "loss": best.final_loss,
                "l2re": best.final_l2re,
            }
            # persist the selected run so later stages can be replayed cheaply
            base_ckpt = ckpt_dir / f"{pde}_adam+lbfgs_w{best.width}_lr{best.adam_lr:g}_s{best.seed}.pt"
            try:
                torch.save(base_setup["model"].state_dict(), base_ckpt)
            except Exception as exc:  # pragma: no cover - disk issues
                entry["errors"].append(f"checkpoint save failed: {exc}")

        model_best = base_setup["model"]
        problem = base_setup["problem"]
        sampler = base_setup["sampler"]

        base_loss = _num(base_stage.final_loss)
        base_l2re = _num(base_stage.final_l2re)
        if math.isnan(base_loss) or math.isnan(base_l2re):
            try:
                base_loss = float(make_closure(model_best, problem, sampler)().detach())
            except Exception:
                pass
            base_l2re = _num(base_setup["eval_fn"]())
        entry["adam_lbfgs"] = {"loss": base_loss, "l2re": base_l2re, "seconds": base_stage.seconds}
        entry["stages"]["adam+lbfgs"] = base_stage.to_metadata()
        if base_stage.loss_history:
            summary["histories"].setdefault("combined", {})[pde] = {
                "steps": base_stage.loss_steps,
                "losses": _downsample(base_stage.loss_history),
                "l2re": _downsample(base_stage.l2re_history),
                "l2re_steps": _downsample(base_stage.l2re_steps),
            }

        # ------------------------------------------------------------------
        # 3) "after Adam" snapshot (Figure 5, left column)
        # ------------------------------------------------------------------
        adam_stage: Optional[StageResult] = None
        if include_baselines or pde in fig1_pdes:
            try:
                adam_stage = run_adam_stage(
                    pde,
                    base_stage.width,
                    _num(base_stage.adam_lr, 1e-3),
                    base_stage.seed,
                    n_steps=switch,
                    problem_cfg=problem_cfg,
                    sampling_cfg=sampling_cfg,
                    depth=depth,
                    dtype=dtype,
                    device=dev,
                    eval_every=eval_every,
                )
                entry["stages"]["adam"] = adam_stage.to_metadata()
                if adam_stage.loss_history:
                    summary["histories"].setdefault("adam", {})[pde] = {
                        "steps": adam_stage.loss_steps,
                        "losses": _downsample(adam_stage.loss_history),
                    }
            except Exception as exc:
                entry["errors"].append(f"adam stage failed: {exc}")

        # ------------------------------------------------------------------
        # 4) L-BFGS-only baseline (Figure 1 stall curve + Table 3 timing)
        # ------------------------------------------------------------------
        lbfgs_stage: Optional[StageResult] = None
        if include_baselines and pde in fig1_pdes:
            try:
                lbfgs_stage = run_lbfgs_stage(
                    pde,
                    base_stage.width,
                    base_stage.seed,
                    adam_lr=_num(base_stage.adam_lr, float("nan")),
                    n_steps=total,
                    problem_cfg=problem_cfg,
                    sampling_cfg=sampling_cfg,
                    depth=depth,
                    dtype=dtype,
                    device=dev,
                    eval_every=eval_every,
                    lbfgs_cfg=lbfgs_cfg,
                )
                entry["stages"]["lbfgs"] = lbfgs_stage.to_metadata()
                if lbfgs_stage.loss_history:
                    summary["histories"].setdefault("lbfgs", {})[pde] = {
                        "steps": lbfgs_stage.loss_steps,
                        "losses": _downsample(lbfgs_stage.loss_history),
                    }
            except Exception as exc:
                entry["errors"].append(f"lbfgs baseline failed: {exc}")

        lbfgs_per_iter = _num(
            lbfgs_stage.time_per_iteration if lbfgs_stage is not None else None,
            float("nan"),
        )

        # ------------------------------------------------------------------
        # 5) NNCG fine-tuning with damping grid search
        # ------------------------------------------------------------------
        nncg_best: Optional[StageResult] = None
        nncg_trials: List[StageResult] = []
        if NNCG_AVAILABLE:
            mus = list(nncg_cfg.get("mus", NNCG_MU_GRID))
            if len(mus) >= 2:
                try:
                    nncg_best, nncg_trials = tune_mu_stages(
                        pde,
                        model_best,
                        problem,
                        sampler,
                        mus=mus,
                        n_steps=finetune,
                        nncg_cfg=nncg_cfg,
                        select_by=nncg_cfg.get("mu_selection", "loss"),
                        dtype=dtype,
                        device=dev,
                        eval_every=max(1, min(eval_every, 50)),
                        width=base_stage.width,
                        seed=base_stage.seed,
                        adam_lr=_num(base_stage.adam_lr, float("nan")),
                        initial_loss=base_loss,
                        initial_l2re=base_l2re,
                        verbose=False,
                    )
                except Exception as exc:
                    entry["errors"].append(f"nncg mu search failed: {exc}")
            if nncg_best is None:
                try:
                    nncg_best = run_nncg_stage(
                        pde,
                        model_best,
                        problem,
                        sampler,
                        mu=float(nncg_cfg.get("mu", 1e-2)),
                        n_steps=finetune,
                        nncg_cfg=nncg_cfg,
                        dtype=dtype,
                        device=dev,
                        eval_every=max(1, min(eval_every, 50)),
                        width=base_stage.width,
                        seed=base_stage.seed,
                        adam_lr=_num(base_stage.adam_lr, float("nan")),
                        initial_loss=base_loss,
                        initial_l2re=base_l2re,
                    )
                    nncg_trials = [nncg_best]
                except Exception as exc:
                    entry["errors"].append(f"nncg stage failed: {exc}")

        if nncg_best is not None:
            entry["stages"]["nncg"] = nncg_best.to_metadata()
            entry["nncg"] = {
                "mu": nncg_best.mu,
                "loss": nncg_best.final_loss,
                "l2re": nncg_best.final_l2re,
                "seconds": nncg_best.seconds,
                "time_per_iteration": nncg_best.time_per_iteration,
                "iterations": nncg_best.iterations,
            }
            entry["nncg_trials"] = [
                {
                    "mu": t.mu,
                    "final_loss": t.final_loss,
                    "final_l2re": t.final_l2re,
                    "best_loss": t.best_loss,
                    "seconds": t.seconds,
                    "error": t.error,
                }
                for t in nncg_trials
            ]
            if nncg_best.loss_history:
                summary["histories"]["finetune"].setdefault(pde, {})["nncg"] = {
                    "steps": nncg_best.loss_steps,
                    "losses": _downsample(nncg_best.loss_history),
                    "grad_norms": _downsample(nncg_best.grad_norm_history),
                    "grad_steps": _downsample(nncg_best.grad_norm_steps),
                    "l2re": _downsample(nncg_best.l2re_history),
                    "l2re_steps": _downsample(nncg_best.l2re_steps),
                }
            try:
                torch.save(nncg_best.model.state_dict(), ckpt_dir / f"{pde}_nncg.pt")
            except Exception as exc:  # pragma: no cover
                entry["errors"].append(f"nncg checkpoint save failed: {exc}")

        # ------------------------------------------------------------------
        # 6) GD control run
        # ------------------------------------------------------------------
        gd_stage: Optional[StageResult] = None
        if NNCG_AVAILABLE or True:
            try:
                gd_stage = run_gd_stage(
                    pde,
                    model_best,
                    problem,
                    sampler,
                    n_steps=finetune,
                    lr=float(cfg.get("optimizer", {}).get("gd", {}).get("lr", 1e-4)),
                    dtype=dtype,
                    device=dev,
                    eval_every=max(1, min(eval_every, 50)),
                    width=base_stage.width,
                    seed=base_stage.seed,
                    adam_lr=_num(base_stage.adam_lr, float("nan")),
                    initial_loss=base_loss,
                    initial_l2re=base_l2re,
                )
                entry["stages"]["gd"] = gd_stage.to_metadata()
                if gd_stage.loss_history:
                    summary["histories"]["finetune"].setdefault(pde, {})["gd"] = {
                        "steps": gd_stage.loss_steps,
                        "losses": _downsample(gd_stage.loss_history),
                        "grad_norms": _downsample(gd_stage.grad_norm_history),
                        "grad_steps": _downsample(gd_stage.grad_norm_steps),
                        "l2re": _downsample(gd_stage.l2re_history),
                        "l2re_steps": _downsample(gd_stage.l2re_steps),
                    }
            except Exception as exc:
                entry["errors"].append(f"gd stage failed: {exc}")

        # ------------------------------------------------------------------
        # 7) per-iteration timing for Table 3 (L-BFGS vs NNCG)
        # ------------------------------------------------------------------
        if lbfgs_per_iter != 0 and not math.isnan(lbfgs_per_iter):
            pass
        else:
            timing_steps = int(exp_cfg.get("lbfgs_timing_steps", 200))
            try:
                timing_setup = build_setup(
                    pde,
                    base_stage.width,
                    base_stage.seed,
                    problem_cfg=problem_cfg,
                    sampling_cfg=sampling_cfg,
                    depth=depth,
                    dtype=dtype,
                    device=dev,
                )
                timing_setup["model"].load_state_dict(model_best.state_dict())
                timing_setup["closure"] = make_closure(
                    timing_setup["model"], timing_setup["problem"], timing_setup["sampler"]
                )
                kwargs = supported_kwargs(run_lbfgs, dict(lbfgs_cfg or {}))
                kwargs.pop("history_size", None)
                t0 = time.time()
                timing_history = run_lbfgs(
                    timing_setup["model"],
                    timing_setup["closure"],
                    n_steps=max(1, timing_steps),
                    **kwargs,
                )
                elapsed = time.time() - t0
                n_done = int(getattr(timing_history, "n_steps", 0) or timing_steps)
                lbfgs_per_iter = elapsed / max(1, n_done)
            except Exception as exc:
                entry["errors"].append(f"lbfgs timing failed: {exc}")
                lbfgs_per_iter = float("nan")

        nncg_per_iter = _num(nncg_best.time_per_iteration if nncg_best is not None else None)
        ratio = (
            nncg_per_iter / lbfgs_per_iter
            if lbfgs_per_iter and not math.isnan(lbfgs_per_iter) and not math.isnan(nncg_per_iter)
            else float("nan")
        )
        entry["timing"] = {
            "nncg_time_per_iteration": nncg_per_iter,
            "lbfgs_time_per_iteration": lbfgs_per_iter,
            "ratio": ratio,
        }

        # ------------------------------------------------------------------
        # 8) Figure 5 error maps: after Adam / after L-BFGS / after NNCG
        # ------------------------------------------------------------------
        error_maps: Dict[str, Dict[str, Any]] = {}
        sources: List[Tuple[str, Optional[torch.nn.Module]]] = [
            ("after Adam", adam_stage.model if adam_stage is not None else None),
            ("after Adam+L-BFGS", model_best),
            ("after Adam+L-BFGS+NNCG", nncg_best.model if nncg_best is not None else None),
        ]
        for stage_name, stage_model in sources:
            if stage_model is None:
                continue
            try:
                error_maps[stage_name] = absolute_error_map(
                    stage_model, problem, sampler, dtype=dtype, device=dev
                )
            except Exception as exc:  # pragma: no cover
                entry["errors"].append(f"error map ({stage_name}) failed: {exc}")
        if error_maps:
            npz_path = outdir / f"{pde}_absolute_errors.npz"
            try:
                import numpy as np  # type: ignore

                np.savez(
                    npz_path,
                    **{f"{k.replace(' ', '_')}": v["error"] for k, v in error_maps.items()},
                    x=next(iter(error_maps.values()))["x"],
                    t=next(iter(error_maps.values()))["t"],
                )
            except Exception as exc:  # pragma: no cover
                entry["errors"].append(f"error map save failed: {exc}")
            entry["absolute_errors"] = {
                name: {"max_error": payload["max_error"], "l2re": payload["l2re"]}
                for name, payload in error_maps.items()
            }
            if make_plots:
                out = plot_error_maps(
                    error_maps,
                    figures_dir / f"figure5_{pde}_absolute_errors.png",
                    title=f"{pde}: absolute error at optimizer switch points",
                )
                if out:
                    summary["figures"][f"figure5_{pde}"] = out

        # ------------------------------------------------------------------
        # 9) rows for Tables 2 and 3
        # ------------------------------------------------------------------
        table2_rows.append(
            {
                "pde": pde,
                "adam_lbfgs_loss": base_loss,
                "adam_lbfgs_l2re": base_l2re,
                "nncg_loss": _num(nncg_best.final_loss if nncg_best is not None else None),
                "nncg_l2re": _num(nncg_best.final_l2re if nncg_best is not None else None),
                "nncg_mu": nncg_best.mu if nncg_best is not None else None,
                "gd_loss": _num(gd_stage.final_loss if gd_stage is not None else None),
                "gd_l2re": _num(gd_stage.final_l2re if gd_stage is not None else None),
            }
        )
        table3_rows.append(
            {
                "pde": pde,
                "nncg_time_per_iteration": nncg_per_iter,
                "lbfgs_time_per_iteration": lbfgs_per_iter,
                "time_ratio": ratio,
                "nncg_iterations": int(nncg_best.iterations) if nncg_best is not None else 0,
                "lbfgs_iterations": int(lbfgs_stage.iterations) if lbfgs_stage is not None else 0,
            }
        )

        summary["pdes"][pde] = entry

    # ------------------------------------------------------------------
    # tables + figures
    # ------------------------------------------------------------------
    summary["table2"] = build_table2(table2_rows)
    summary["table3"] = build_table3(table3_rows)

    try:
        with open(outdir / "table2.json", "w") as handle:
            json.dump(to_jsonable(summary["table2"]), handle, indent=2)
        with open(outdir / "table3.json", "w") as handle:
            json.dump(to_jsonable(summary["table3"]), handle, indent=2)
        text2 = format_table2(summary["table2"])
        text3 = format_table3(summary["table3"])
        with open(outdir / "table2.txt", "w") as handle:
            handle.write(text2 + "\n")
        with open(outdir / "table3.txt", "w") as handle:
            handle.write(text3 + "\n")
        summary["table2_text"] = text2
        summary["table3_text"] = text3
    except Exception as exc:  # pragma: no cover
        summary["errors"].append(f"table save failed: {exc}")

    if make_plots and PLOTTING_AVAILABLE:
        summary["figures"].update(_make_figures(summary, figures_dir))

    try:
        with open(outdir / "summary.json", "w") as handle:
            json.dump(to_jsonable(summary), handle, indent=2)
    except Exception as exc:  # pragma: no cover
        print(f"[nncg] could not write summary.json: {exc}")

    if verbose:
        print("\n" + summary.get("table2_text", ""))
        print("\n" + summary.get("table3_text", ""))
        print(f"\n[nncg] results written to {outdir}")

    return summary


def _make_figures(summary: Dict[str, Any], figures_dir: Path) -> Dict[str, Optional[str]]:
    """Build Figures 1, 4 with the plotting helpers (best-effort)."""
    figures: Dict[str, Optional[str]] = {}
    try:
        plot_training_curves = getattr(_plotting, "plot_training_curves")
        plot_finetune = getattr(_plotting, "plot_finetune")
    except Exception:  # pragma: no cover
        return figures

    # Figure 1: NNCG continues to decrease the loss after Adam+L-BFGS stalls.
    for pde in summary.get("histories", {}).get("finetune", {}).keys():
        curves: Dict[str, Any] = {}
        adam_hist = summary["histories"].get("adam", {}).get(pde)
        lbfgs_hist = summary["histories"].get("lbfgs", {}).get(pde)
        combined_hist = summary["histories"].get("combined", {}).get(pde)
        nncg_hist = summary["histories"]["finetune"][pde].get("nncg")
        if adam_hist:
            curves["Adam"] = (adam_hist["steps"], adam_hist["losses"])
        if lbfgs_hist:
            curves["L-BFGS"] = (lbfgs_hist["steps"], lbfgs_hist["losses"])
        if combined_hist:
            curves["Adam+L-BFGS"] = (combined_hist["steps"], combined_hist["losses"])
        if nncg_hist:
            offset = int(summary["config"]["total_iterations"])
            curves["Adam+L-BFGS+NNCG"] = (
                [offset + int(s) for s in nncg_hist["steps"]],
                nncg_hist["losses"],
            )
        if not curves:
            continue
        try:
            out = plot_training_curves(
                curves,
                key="losses",
                xlabel="Iteration",
                ylabel="Loss",
                title=f"{pde}: NNCG continues past the Adam+L-BFGS stall",
                logy=True,
                outfile=str(figures_dir / f"figure1_{pde}_loss.png"),
            )
            figures[f"figure1_{pde}"] = out
        except Exception as exc:  # pragma: no cover
            summary.setdefault("errors", []).append(f"figure1 ({pde}) failed: {exc}")

    # Figure 4: loss and gradient-norm of NNCG vs GD after Adam+L-BFGS.
    for pde, stages in summary.get("histories", {}).get("finetune", {}).items():
        histories: Dict[str, Any] = {}
        nncg_hist = stages.get("nncg")
        gd_hist = stages.get("gd")
        if nncg_hist:
            histories["NNCG"] = nncg_hist
        if gd_hist:
            histories["GD"] = gd_hist
        if not histories:
            continue
        try:
            outs = plot_finetune(
                histories,
                outfile=str(figures_dir / f"figure4_{pde}_finetune.png"),
                title=f"{pde}: NNCG vs GD fine-tuning",
            )
            if isinstance(outs, dict):
                for key, value in outs.items():
                    figures[f"figure4_{pde}_{key}"] = value
            else:
                figures[f"figure4_{pde}"] = outs
        except Exception as exc:  # pragma: no cover
            summary.setdefault("errors", []).append(f"figure4 ({pde}) failed: {exc}")
    return figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NNCG fine-tuning experiments (Figures 1/4/5, Tables 2/3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config file")
    parser.add_argument("--quick", action="store_true", help="small smoke-test budget")
    parser.add_argument("--pdes", nargs="+", default=None, help="PDE names to run")
    parser.add_argument("--widths", nargs="+", type=int, default=None, help="network widths to select over")
    parser.add_argument("--lrs", nargs="+", type=float, default=None, help="Adam learning rates to select over")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="seeds to select over")
    parser.add_argument("--switch", type=int, default=None, help="Adam -> L-BFGS switch iteration")
    parser.add_argument("--total-iterations", type=int, default=None, help="Adam+L-BFGS iteration budget")
    parser.add_argument("--finetune-steps", type=int, default=None, help="NNCG/GD fine-tuning steps (paper: 2000)")
    parser.add_argument("--mu", nargs="+", type=float, default=None, help="NNCG damping grid")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--outdir", type=str, default=None, help="output directory")
    parser.add_argument("--eval-every", type=int, default=None, help="L2RE evaluation cadence")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="ignore existing optimizer-comparison checkpoints and re-run the selection sweep",
    )
    parser.add_argument("--no-baselines", action="store_true", help="skip the Adam / L-BFGS baselines")
    parser.add_argument("--quiet", action="store_true", help="reduce logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, quick=args.quick)
    if args.mu:
        cfg.setdefault("nncg", {})["mus"] = [float(m) for m in args.mu]
    if args.quick:
        print("[nncg] quick mode: reduced budgets (results are not paper-scale)")
    return run_nncg_finetune(
        cfg,
        pdes=args.pdes,
        widths=args.widths,
        adam_lrs=args.lrs,
        seeds=args.seeds,
        switch_iteration=args.switch,
        total_iterations=args.total_iterations,
        finetune_steps=args.finetune_steps,
        device=args.device,
        outdir=args.outdir,
        verbose=not args.quiet,
        eval_every=args.eval_every,
        make_plots=not args.no_plots,
        use_comparison_checkpoints=not args.no_checkpoints,
        include_baselines=False if args.no_baselines else None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
