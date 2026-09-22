"""Optimizer-comparison experiment runner (Fig. 2, Fig. 8, Table 1).

Reproduces Sections 2.2 / 6.1 of *Challenges in Training PINNs: A Loss Landscape
Perspective* (ICML 2024):

    * optimizers: Adam, L-BFGS, Adam+L-BFGS (switch iteration 1k / 11k / 31k);
    * 41 000 total iterations for every optimizer;
    * network widths {50, 100, 200, 400}, 3 hidden layers, tanh, Xavier init;
    * Adam learning rate tuned by grid search over {1e-5, 1e-4, 1e-3, 1e-2, 1e-1};
    * 10 000 residual / 257 IC / 101 BC points (fixed sampling protocol);
    * metrics: PINN loss L(w) (Eq. 2) and L2RE (Eq. 3) on the full evaluation set.

Aggregated outputs (min / median / max across seeds and widths) are written to
``results/optimizer_comparison/`` together with Table 1 (JSON + text) and the
paper's Fig. 2 / Fig. 8-style plots when matplotlib is available.

Every record additionally stores the L-BFGS curvature history ``(s_k, y_k, rho_k)``
which the spectral-density pipeline (``experiments/run_spectral_density.py``)
consumes, and the trained ``state_dict`` which the NNCG fine-tuning pipeline
(``experiments/run_nncg_finetune.py``) resumes from.

Usage
-----
    python experiments/run_optimizer_comparison.py --quick
    python experiments/run_optimizer_comparison.py --pdes convection --widths 200 \
        --seeds 1 2 3 --optimizers adam adam+lbfgs --switch 11000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------------------
# Import bootstrap: make `src.*` importable regardless of the current working directory.
# --------------------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent  # .../opt_for_pinns
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pinns.loss import make_loss_fn  # noqa: E402
from src.pinns.metrics import compute_l2re  # noqa: E402
from src.pinns.model import make_pinn  # noqa: E402
from src.pinns.problems import PROBLEMS, get_problem  # noqa: E402
from src.pinns.sampling import build_sampler  # noqa: E402
from src.optimizers.combined import (  # noqa: E402
    COMBINED_TOTAL_ITERATIONS,
    DEFAULT_SWITCH_POINT,
    SWITCH_POINTS,
    run_adam_lbfgs,
)
from src.optimizers.first_order import ADAM_LR_GRID, AdamOptimizer, run_first_order  # noqa: E402
from src.optimizers.lbfgs_wrapper import lbfgs_recording_state, run_lbfgs  # noqa: E402

try:  # optional: model dtype casting lives with the second-order code
    from src.spectral.hvp import cast_model_dtype, loss_and_grad
except Exception:  # pragma: no cover - fallback for partial installs
    cast_model_dtype = None  # type: ignore[assignment]

    def loss_and_grad(loss_fn, model_or_params, **kwargs):  # type: ignore[misc]
        raise RuntimeError("src.spectral.hvp is unavailable")

try:  # optional seeding helper
    from src.utils.seeding import set_seed  # type: ignore
except Exception:  # pragma: no cover
    import random

    def set_seed(seed: Optional[int] = None) -> None:
        if seed is None:
            return
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass

try:  # optional plotting helpers
    from src.utils import plotting as _plotting

    PLOTTING_AVAILABLE = bool(getattr(_plotting, "PLOTTING_AVAILABLE", True))
except Exception:  # pragma: no cover
    _plotting = None  # type: ignore[assignment]
    PLOTTING_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Default configuration
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "pdes": ["convection", "reaction", "wave"],
    "widths": [50, 100, 200, 400],
    "seeds": [1, 2, 3, 4, 5],
    "depth": 3,
    "optimizers": ["adam", "lbfgs", "adam+lbfgs"],
    "switch_points": list(SWITCH_POINTS),
    "default_switch": DEFAULT_SWITCH_POINT,
    "total_iterations": COMBINED_TOTAL_ITERATIONS,  # 41 000
    "adam_lr_grid": list(ADAM_LR_GRID),  # {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}
    # LR tuning: by default the paper's full 41k-iteration budget is used for each lr.
    "adam_lr_tuning_steps": None,  # None -> total_iterations
    "adam_lr_tuning_seed": None,  # None -> first seed in `seeds`
    "adam_lr_select_by": "loss",  # Section 6.1 tunes Adam by loss
    "adam_grad_clip": None,
    "lbfgs": {
        "lr": 1.0,
        "history_size": 100,
        "line_search_fn": "strong_wolfe",
        "max_iter": 1,
        "max_eval": 25,
    },
    "eval_every": 250,
    "record_limit": 64,
    "save_checkpoints": True,
    "history_points": 200,  # subsample length for the stored training curves
    "dtype": "float64",
    "device": "cpu",
    "output_dir": "results/optimizer_comparison",
    "sampling": {
        "n_residual": 10000,
        "n_ic": 257,
        "n_bc": 101,
        "n_grid_x": 255,
        "n_grid_t": 100,
        "replace": True,
    },
    "problems": {
        "convection": {"beta": 40.0},
        "reaction": {"rho": 5.0},
        "wave": {"beta": 5.0},
    },
    "quick": {
        "widths": [50],
        "seeds": [1],
        "optimizers": ["adam", "lbfgs", "adam+lbfgs"],
        "switch_points": [1000],
        "total_iterations": 1000,
        "eval_every": 250,
        "save_checkpoints": False,
    },
}

#: Paper (Table 1) lowest values across widths -- used only for reporting deltas.
PAPER_TABLE1_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "convection": {
        "adam": (1.40e-4, 5.96e-2),
        "lbfgs": (1.51e-5, 8.26e-3),
        "adam+lbfgs": (5.95e-6, 4.19e-3),
    },
    "reaction": {
        "adam": (4.73e-6, 2.12e-2),
        "lbfgs": (8.93e-6, 3.83e-2),
        "adam+lbfgs": (3.26e-6, 1.92e-2),
    },
    "wave": {
        "adam": (2.03e-2, 3.49e-1),
        "lbfgs": (1.84e-2, 3.35e-1),
        "adam+lbfgs": (1.12e-3, 5.52e-2),
    },
}

OPTIMIZER_LABELS = ("adam", "lbfgs", "adam+lbfgs")


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def load_config(path: Optional[str] = None, quick: bool = False) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` (if present) merged on top of :data:`DEFAULT_CONFIG`."""
    cfg: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.append(_PROJECT_ROOT / "configs" / "default.yaml")

    for cand in candidates:
        if cand and cand.is_file():
            try:
                import yaml  # type: ignore

                with open(cand, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
            except Exception as exc:  # pragma: no cover
                print(f"[warn] could not parse config {cand}: {exc}", file=sys.stderr)
                loaded = {}
            for key in ("sampling", "lbfgs", "problems"):
                if isinstance(loaded.get(key), dict):
                    cfg[key].update(loaded[key])
            for key, value in loaded.items():
                if key in ("sampling", "lbfgs", "quick", "problems"):
                    continue
                if key == "optimizer" and isinstance(value, dict):
                    for sub in ("switch_points", "total_iterations", "adam_lr_grid", "lbfgs"):
                        if sub in value and not isinstance(cfg.get(sub), dict):
                            cfg[sub] = value[sub]
                    continue
                cfg[key] = value
            break

    if quick:
        cfg.update(json.loads(json.dumps(DEFAULT_CONFIG["quick"])))
        cfg["_quick"] = True
    return cfg


def supported_kwargs(fn: Callable, kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter ``kwargs`` to the parameters actually accepted by ``fn``."""
    import inspect

    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def to_jsonable(obj: Any, max_points: Optional[int] = None) -> Any:
    """Best-effort conversion of numpy / torch / dataclass payloads to JSON."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return to_jsonable(float(obj.detach().cpu()))
        return to_jsonable(obj.detach().cpu().tolist(), max_points=max_points)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_points=max_points) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        values = list(obj)
        if max_points is not None and len(values) > max_points:
            stride = max(1, len(values) // max_points)
            values = values[::stride]
        return [to_jsonable(v, max_points=max_points) for v in values]
    if isinstance(obj, (float, int)):  # numpy scalars
        try:
            return to_jsonable(float(obj))
        except Exception:  # pragma: no cover
            return str(obj)
    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        try:
            return to_jsonable(obj.as_dict(), max_points=max_points)
        except Exception:  # pragma: no cover
            pass
    return str(obj)


def device_from_arg(device: Optional[str]) -> torch.device:
    """Resolve a device string (defaults to CPU -- the paper's protocol is device agnostic)."""
    if device is None:
        return torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable -> falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    return dev


def optimizer_family(label: str) -> str:
    """Map a run label (``adam+lbfgs@11000``) onto the Table-1 family (``adam+lbfgs``)."""
    name = str(label).split("@", 1)[0].strip().lower().replace("_", "-").replace("+", "+")
    name = name.replace("adam-lbfgs", "adam+lbfgs").replace("adam+l-bfgs", "adam+lbfgs")
    if name.startswith("adam") and "lbfgs" in name:
        return "adam+lbfgs"
    if name.startswith("adam"):
        return "adam"
    if "lbfgs" in name or "bfgs" in name:
        return "lbfgs"
    return name


def _downsample(xs: Sequence[Any], ys: Sequence[Any], n: int) -> Tuple[List[Any], List[Any]]:
    xs, ys = list(xs), list(ys)
    if n and len(xs) > n:
        stride = max(1, len(xs) // n)
        positions = list(range(0, len(xs), stride))
        if positions[-1] != len(xs) - 1:
            positions.append(len(xs) - 1)
        xs = [xs[i] for i in positions]
        ys = [ys[i] for i in positions]
    return xs, ys


def _series(history_obj: Any, key: str) -> List[Any]:
    if history_obj is None:
        return []
    values = getattr(history_obj, key, None)
    if values is None:
        return []
    try:
        return list(values)
    except TypeError:  # pragma: no cover
        return []


def make_eval_fn(
    model: torch.nn.Module,
    problem: Any,
    sampler: Any,
    *,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 50000,
) -> Callable[..., float]:
    """L2RE evaluation closure tolerant to any call signature used by the drivers."""

    def _eval(*_args: Any, **_kwargs: Any) -> float:
        try:
            value = compute_l2re(
                model,
                problem,
                sampler=sampler,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
            return float(value)
        except Exception:  # pragma: no cover - never let an eval failure kill training
            return float("nan")

    return _eval


def make_loss_and_setup(
    pde: str,
    width: int,
    seed: int,
    *,
    depth: int = 3,
    problem_cfg: Optional[Dict[str, Any]] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[str] = None,
    dtype_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Build (problem, model, sampler, loss, closure, eval_fn) for one run."""
    set_seed(seed)
    problem = get_problem(pde, **(problem_cfg or {}))
    model = make_pinn(
        in_dim=getattr(problem, "in_dim", 2),
        out_dim=getattr(problem, "out_dim", 1),
        width=width,
        depth=depth,
        seed=seed,
        device=str(device) if device is not None else "cpu",
    )
    if cast_model_dtype is not None:
        try:
            model = cast_model_dtype(model, dtype)
        except Exception:  # pragma: no cover
            model = model.to(dtype=dtype)
    else:  # pragma: no cover
        model = model.to(dtype=dtype)

    sampler_kwargs = supported_kwargs(
        build_sampler,
        {
            "seed": seed,
            "device": device,
            "dtype": dtype,
            "n_residual": None,
            "replace": True,
        },
    )
    sampler_kwargs.update(
        supported_kwargs(build_sampler, sampling_cfg or {}) if sampling_cfg else {}
    )
    sampler_kwargs["seed"] = seed
    sampler_kwargs["dtype"] = dtype
    sampler_kwargs["device"] = device
    sampler = build_sampler(problem, **sampler_kwargs)

    loss, closure = make_loss_fn(
        model,
        problem,
        sampler=sampler,
        device=device,
        dtype=dtype,
    )
    eval_fn = make_eval_fn(model, problem, sampler, device=device, dtype=dtype)
    return {
        "problem": problem,
        "model": model,
        "sampler": sampler,
        "loss": loss,
        "closure": closure,
        "eval_fn": eval_fn,
    }


# --------------------------------------------------------------------------------------
# Run record
# --------------------------------------------------------------------------------------
@dataclass
class RunRecord:
    """One trained PINN run (single pde / width / seed / optimizer configuration)."""

    pde: str
    width: int
    seed: int
    optimizer: str
    adam_lr: float
    switch_iteration: Optional[int] = None
    total_iterations: int = COMBINED_TOTAL_ITERATIONS
    final_loss: float = float("nan")
    best_loss: float = float("nan")
    l2re: float = float("nan")
    best_l2re: float = float("nan")
    grad_norm: float = float("nan")
    loss_steps: List[int] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    l2re_steps: List[int] = field(default_factory=list)
    l2re_history: List[float] = field(default_factory=list)
    grad_steps: List[int] = field(default_factory=list)
    grad_history: List[float] = field(default_factory=list)
    n_lbfgs_pairs: int = 0
    seconds: float = float("nan")
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    lbfgs_history: Any = None
    tuned: bool = False  # True when this run was produced by the Adam LR grid search
    error: Optional[str] = None

    # -- convenience -------------------------------------------------------------------
    @property
    def family(self) -> str:
        return optimizer_family(self.optimizer)

    @property
    def key(self) -> str:
        return f"{self.pde}/w{self.width}/s{self.seed}/{self.optimizer}"

    def history_dict(self) -> Dict[str, Any]:
        return {
            "steps": list(self.loss_steps),
            "losses": list(self.loss_history),
            "l2re_steps": list(self.l2re_steps),
            "l2re": list(self.l2re_history),
            "grad_steps": list(self.grad_steps),
            "grad_norms": list(self.grad_history),
        }

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "pde": self.pde,
            "width": self.width,
            "seed": self.seed,
            "optimizer": self.optimizer,
            "family": self.family,
            "adam_lr": self.adam_lr,
            "switch_iteration": self.switch_iteration,
            "total_iterations": self.total_iterations,
            "final_loss": to_jsonable(self.final_loss),
            "best_loss": to_jsonable(self.best_loss),
            "l2re": to_jsonable(self.l2re),
            "best_l2re": to_jsonable(self.best_l2re),
            "grad_norm": to_jsonable(self.grad_norm),
            "n_lbfgs_pairs": self.n_lbfgs_pairs,
            "seconds": to_jsonable(self.seconds),
            "tuned": self.tuned,
            "error": self.error,
        }

    def as_record_dict(self) -> Dict[str, Any]:
        """Flat dict consumable by ``src.utils.plotting.plot_loss_vs_l2re``."""
        return {
            "pde": self.pde,
            "width": self.width,
            "seed": self.seed,
            "optimizer": self.family,
            "label": self.optimizer,
            "loss": self.best_loss,
            "l2re": self.best_l2re,
            "final_loss": self.final_loss,
        }


def _finalize_record(
    record: RunRecord,
    *,
    model: torch.nn.Module,
    loss: Any,
    closure: Callable[[], torch.Tensor],
    error: Optional[str] = None,
) -> RunRecord:
    """Fill final loss / gradient-norm fields from the trained model."""
    record.error = error
    try:
        value = closure()
        record.final_loss = float(value.detach() if isinstance(value, torch.Tensor) else value)
    except Exception as exc:  # pragma: no cover
        if error is None:
            record.error = f"final loss failed: {exc}".strip()
    try:
        _, grad = loss_and_grad(closure, model)
        record.grad_norm = float(torch.linalg.vector_norm(grad).item())
    except Exception:  # pragma: no cover
        pass
    if not math.isfinite(record.best_loss):
        record.best_loss = record.final_loss
    if not math.isfinite(record.best_l2re) and math.isfinite(record.l2re):
        record.best_l2re = record.l2re
    if record.state_dict is None:
        try:
            record.state_dict = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        except Exception:  # pragma: no cover
            record.state_dict = None
    return record


# --------------------------------------------------------------------------------------
# Individual training runs
# --------------------------------------------------------------------------------------
def train_adam(
    pde: str,
    width: int,
    adam_lr: float,
    seed: int,
    *,
    total_iterations: int = COMBINED_TOTAL_ITERATIONS,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    problem_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 250,
    grad_clip: Optional[float] = None,
    history_points: int = 200,
    save_state: bool = True,
    tuned: bool = False,
    verbose: bool = False,
) -> RunRecord:
    """Adam baseline (Sec. 6.1): full ``total_iterations`` with learning rate ``adam_lr``."""
    parts = make_loss_and_setup(
        pde,
        width,
        seed,
        depth=depth,
        problem_cfg=problem_cfg,
        sampling_cfg=sampling_cfg,
        dtype=dtype,
        device=device,
    )
    model, closure, eval_fn = parts["model"], parts["closure"], parts["eval_fn"]

    record = RunRecord(
        pde=pde,
        width=width,
        seed=seed,
        optimizer="adam",
        adam_lr=float(adam_lr),
        switch_iteration=None,
        total_iterations=int(total_iterations),
        tuned=tuned,
    )
    t0 = time.time()
    try:
        optimizer = AdamOptimizer(model, lr=float(adam_lr), grad_clip=grad_clip)
        history = run_first_order(
            model,
            closure,
            optimizer,
            int(total_iterations),
            eval_fn=eval_fn,
            eval_every=int(eval_every),
        )
    except Exception as exc:
        record.seconds = time.time() - t0
        return _finalize_record(
            record, model=model, loss=parts["loss"], closure=closure, error=f"{type(exc).__name__}: {exc}"
        )
    record.seconds = time.time() - t0

    losses = _series(history, "losses")
    if losses:
        record.best_loss = float(min(losses))
    l2res = [v for v in _series(history, "l2re") if v is not None and math.isfinite(float(v))]
    if l2res:
        record.l2re = float(l2res[-1])
        record.best_l2re = float(min(l2res))
    record.loss_steps, record.loss_history = _downsample(
        _series(history, "steps"), losses, history_points
    )
    record.l2re_steps, record.l2re_history = _downsample(
        _series(history, "steps"), l2res, history_points
    )
    record.grad_history = [float(v) for v in losses]
    record.grad_steps = list(record.loss_steps)

    if save_state:
        state = record
    else:
        state = None
    record = _finalize_record(record, model=model, loss=parts["loss"], closure=closure)
    if not save_state:
        record.state_dict = None
    return record


def train_lbfgs(
    pde: str,
    width: int,
    seed: int,
    *,
    adam_lr: float = float("nan"),
    total_iterations: int = COMBINED_TOTAL_ITERATIONS,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    problem_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 250,
    lbfgs_cfg: Optional[Dict[str, Any]] = None,
    record_limit: int = 64,
    history_points: int = 200,
    save_state: bool = True,
    verbose: bool = False,
) -> RunRecord:
    """L-BFGS baseline: lr=1.0, memory=100, strong-Wolfe (Sec. 2.2)."""
    parts = make_loss_and_setup(
        pde,
        width,
        seed,
        depth=depth,
        problem_cfg=problem_cfg,
        sampling_cfg=sampling_cfg,
        dtype=dtype,
        device=device,
    )
    model, closure, eval_fn = parts["model"], parts["closure"], parts["eval_fn"]

    record = RunRecord(
        pde=pde,
        width=width,
        seed=seed,
        optimizer="lbfgs",
        adam_lr=None,  # type: ignore[arg-type]
        switch_iteration=None,
        total_iterations=int(total_iterations),
    )
    cfg = dict(
        lr=1.0,
        history_size=100,
        line_search_fn="strong_wolfe",
        max_iter=1,
        max_eval=25,
    )
    cfg.update(lbfgs_cfg or {})
    cfg["record_limit"] = int(record_limit)

    t0 = time.time()
    try:
        history = run_lbfgs(
            model,
            closure,
            n_steps=int(total_iterations),
            eval_fn=eval_fn,
            eval_every=int(eval_every),
            record=True,
            **cfg,
        )
    except Exception as exc:
        record.seconds = time.time() - t0
        return _finalize_record(
            record, model=model, loss=parts["loss"], closure=closure, error=f"{type(exc).__name__}: {exc}"
        )
    record.seconds = time.time() - t0

    losses = _series(history, "losses")
    if losses:
        record.best_loss = float(min(losses))
    l2res = [v for v in _series(history, "l2re") if v is not None and math.isfinite(float(v))]
    if l2res:
        record.l2re = float(l2res[-1])
        record.best_l2re = float(min(l2res))
    record.loss_steps, record.loss_history = _downsample(
        _series(history, "steps"), losses, history_points
    )
    record.l2re_steps, record.l2re_history = _downsample(
        _series(history, "steps"), l2res, history_points
    )

    rec_state = None
    try:
        rec_state = lbfgs_recording_state(model)
    except Exception:  # pragma: no cover
        rec_state = getattr(model, "_lbfgs_history", None)
    record.lbfgs_history = rec_state
    try:
        record.n_lbfgs_pairs = len(rec_state) if rec_state is not None else 0
    except Exception:  # pragma: no cover
        record.n_lbfgs_pairs = 0

    record = _finalize_record(record, model=model, loss=parts["loss"], closure=closure)
    if not save_state:
        record.state_dict = None
    return record


def train_combined(
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
    lbfgs_cfg: Optional[Dict[str, Any]] = None,
    record_limit: int = 64,
    history_points: int = 200,
    grad_clip: Optional[float] = None,
    save_state: bool = True,
    verbose: bool = False,
) -> RunRecord:
    """Adam + L-BFGS with the switch happening at ``switch_iteration`` (Sec. 6.1)."""
    parts = make_loss_and_setup(
        pde,
        width,
        seed,
        depth=depth,
        problem_cfg=problem_cfg,
        sampling_cfg=sampling_cfg,
        dtype=dtype,
        device=device,
    )
    model, closure, eval_fn = parts["model"], parts["closure"], parts["eval_fn"]

    record = RunRecord(
        pde=pde,
        width=width,
        seed=seed,
        optimizer=f"adam+lbfgs@{int(switch_iteration)}",
        adam_lr=float(adam_lr),
        switch_iteration=int(switch_iteration),
        total_iterations=int(total_iterations),
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
            record_limit=int(record_limit),
            grad_clip=grad_clip,
            lbfgs_kwargs=dict(lbfgs_cfg or {}),
        )
    except Exception as exc:
        record.seconds = time.time() - t0
        return _finalize_record(
            record, model=model, loss=parts["loss"], closure=closure, error=f"{type(exc).__name__}: {exc}"
        )
    record.seconds = time.time() - t0

    history = getattr(result, "history", result)
    losses = _series(history, "losses")
    if not losses and hasattr(result, "final_loss") and math.isfinite(float(result.final_loss or float("nan"))):
        losses = [float(result.final_loss)]
    if losses:
        record.best_loss = float(min(losses))
    l2res = [v for v in _series(history, "l2re") if v is not None and math.isfinite(float(v))]
    if l2res:
        record.l2re = float(l2res[-1])
        record.best_l2re = float(min(l2res))

    steps = _series(history, "steps")
    if not steps and losses:
        steps = list(range(len(losses)))
    record.loss_steps, record.loss_history = _downsample(steps, losses, history_points)
    record.l2re_steps, record.l2re_history = _downsample(steps, l2res, history_points)

    rec_state = getattr(result, "lbfgs_history", None)
    if rec_state is None:
        try:
            rec_state = lbfgs_recording_state(model)
        except Exception:  # pragma: no cover
            rec_state = None
    record.lbfgs_history = rec_state
    try:
        record.n_lbfgs_pairs = len(rec_state) if rec_state is not None else 0
    except Exception:  # pragma: no cover
        record.n_lbfgs_pairs = 0

    record = _finalize_record(record, model=model, loss=parts["loss"], closure=closure)
    if not save_state:
        record.state_dict = None
    return record


# --------------------------------------------------------------------------------------
# Adam learning-rate tuning (Sec. 2.2 / 6.1)
# --------------------------------------------------------------------------------------
def tune_adam_lr(
    pde: str,
    width: int,
    seed: int,
    *,
    lr_grid: Optional[Sequence[float]] = None,
    select_by: str = "loss",
    total_iterations: int = COMBINED_TOTAL_ITERATIONS,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    problem_cfg: Optional[Dict[str, Any]] = None,
    depth: int = 3,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    eval_every: int = 250,
    grad_clip: Optional[float] = None,
    history_points: int = 200,
    save_state: bool = True,
    verbose: bool = False,
) -> Tuple[float, RunRecord, List[Dict[str, Any]]]:
    """Grid search Adam's learning rate; returns ``(best_lr, best_record, all_results)``.

    The paper tunes Adam by loss (Sec. 6.1) and rebuilds an identical initialization for
    every learning rate (same seed), so the comparison is paired.
    """
    grid = [float(v) for v in (lr_grid or ADAM_LR_GRID)]
    results: List[Dict[str, Any]] = []
    best_lr, best_record, best_score = grid[0], None, float("inf")

    for lr in grid:
        record = train_adam(
            pde,
            width,
            lr,
            seed,
            total_iterations=total_iterations,
            sampling_cfg=sampling_cfg,
            problem_cfg=problem_cfg,
            depth=depth,
            dtype=dtype,
            device=device,
            eval_every=eval_every,
            grad_clip=grad_clip,
            history_points=history_points,
            save_state=save_state,
            tuned=True,
            verbose=verbose,
        )
        score = record.best_l2re if select_by == "l2re" else record.best_loss
        if score is None or not math.isfinite(float(score)):
            score = float("inf")
        results.append(
            {
                "lr": lr,
                "best_loss": record.best_loss,
                "final_loss": record.final_loss,
                "best_l2re": record.best_l2re,
                "l2re": record.l2re,
                "seconds": record.seconds,
                "error": record.error,
            }
        )
        if verbose:
            print(
                f"    [tune] {pde} w={width} lr={lr:g}: loss={record.best_loss:.4e} "
                f"l2re={record.best_l2re:.4e} ({record.seconds:.1f}s)"
            )
        if float(score) < best_score:
            best_score, best_lr, best_record = float(score), lr, record

    if best_record is None:  # pragma: no cover - grid was empty
        best_record = train_adam(
            pde, width, best_lr, seed, total_iterations=total_iterations,
            sampling_cfg=sampling_cfg, problem_cfg=problem_cfg, depth=depth, dtype=dtype,
            device=device, eval_every=eval_every, grad_clip=grad_clip, tuned=True,
        )
    return best_lr, best_record, results


# --------------------------------------------------------------------------------------
# Aggregation / Table 1
# --------------------------------------------------------------------------------------
def _stats(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"n": 0, "min": float("nan"), "median": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "n": len(clean),
        "min": float(min(clean)),
        "median": float(statistics.median(clean)),
        "max": float(max(clean)),
        "mean": float(statistics.fmean(clean)),
    }


def aggregate_records(
    records: Sequence[RunRecord],
    *,
    group_by_switch: bool = True,
) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    """Aggregate min / median / max loss and L2RE per (pde, width, optimizer).

    When ``group_by_switch`` is False the three Adam+L-BFGS switch points are pooled
    under the family key ``adam+lbfgs``.
    """
    out: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    buckets: Dict[Tuple[str, int, str], List[RunRecord]] = {}
    for rec in records:
        key = (rec.pde, int(rec.width), rec.optimizer if group_by_switch else rec.family)
        buckets.setdefault(key, []).append(rec)

    for (pde, width, label), recs in buckets.items():
        good = [r for r in recs if r.error is None]
        entry = {
            "optimizer": label,
            "family": optimizer_family(label),
            "n_runs": len(recs),
            "n_failed": len(recs) - len(good),
            "best_loss": _stats([r.best_loss for r in good]),
            "final_loss": _stats([r.final_loss for r in good]),
            "best_l2re": _stats([r.best_l2re for r in good]),
            "l2re": _stats([r.l2re for r in good]),
            "seconds": _stats([r.seconds for r in good]),
            "runs": [r.to_metadata() for r in recs],
        }
        out.setdefault(pde, {}).setdefault(width, {})[label] = entry
    return out


def best_per_optimizer(
    records: Sequence[RunRecord],
    *,
    metric: str = "best_loss",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Lowest metric over all widths/seeds, per (pde, optimizer-family).

    Reports both the metric-minimising configuration (``*_min``) and the L2RE-minimising
    configuration (``*_min_l2re``), matching how Table 1 pairs loss and L2RE.
    """
    buckets: Dict[Tuple[str, str], List[RunRecord]] = {}
    for rec in records:
        if rec.error is not None:
            continue
        buckets.setdefault((rec.pde, rec.family), []).append(rec)

    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (pde, family), recs in buckets.items():
        loss_vals = [(r, r.best_loss) for r in recs if math.isfinite(float(r.best_loss or float("nan")))]
        l2re_vals = [(r, r.best_l2re) for r in recs if math.isfinite(float(r.best_l2re or float("nan")))]
        entry: Dict[str, Any] = {"family": family, "n_runs": len(recs)}
        if loss_vals:
            rec, val = min(loss_vals, key=lambda t: t[1])
            entry.update(
                {
                    "loss_min": float(val),
                    "l2re_at_min_loss": float(rec.best_l2re),
                    "loss_min_config": {"width": rec.width, "seed": rec.seed, "optimizer": rec.optimizer},
                }
            )
        if l2re_vals:
            rec, val = min(l2re_vals, key=lambda t: t[1])
            entry.update(
                {
                    "l2re_min": float(val),
                    "loss_at_min_l2re": float(rec.best_loss),
                    "l2re_min_config": {"width": rec.width, "seed": rec.seed, "optimizer": rec.optimizer},
                }
            )
        out.setdefault(pde, {})[family] = entry
    return out


def build_table1(
    records: Sequence[RunRecord],
    *,
    include_reference: bool = True,
) -> Dict[str, Any]:
    """Table 1: lowest loss / L2RE across widths for each (PDE, optimizer)."""
    table: Dict[str, Any] = {}
    for pde, per_family in best_per_optimizer(records).items():
        rows: Dict[str, Any] = {}
        ref = PAPER_TABLE1_REFERENCE.get(pde, {}) if include_reference else {}
        for family in OPTIMIZER_LABELS:
            entry = per_family.get(family, {})
            loss = entry.get("loss_min")
            l2re = entry.get("l2re_at_min_loss")
            row = {
                "loss": loss,
                "l2re": l2re,
                "l2re_min": entry.get("l2re_min"),
                "loss_at_min_l2re": entry.get("loss_at_min_l2re"),
                "config": entry.get("loss_min_config"),
            }
            if family in ref and loss is not None:
                ref_loss, ref_l2re = ref[family]
                row["paper_loss"] = ref_loss
                row["paper_l2re"] = ref_l2re
                row["loss_ratio_to_paper"] = float(loss) / ref_loss if ref_loss else None
            rows[family] = row
        table[pde] = rows
    return table


def format_table1(table: Dict[str, Any]) -> str:
    """Render Table 1 as plain text."""
    lines: List[str] = []
    header = f"{'PDE':<12}{'Optimizer':<14}{'loss':>12}{'L2RE':>12}{'paper loss':>12}{'paper L2RE':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for pde, rows in table.items():
        for family in OPTIMIZER_LABELS:
            row = rows.get(family)
            if not row:
                continue
            loss = _fmt(row.get("loss"))
            l2re = _fmt(row.get("l2re"))
            ploss = _fmt(row.get("paper_loss"))
            pl2re = _fmt(row.get("paper_l2re"))
            lines.append(f"{pde:<12}{family:<14}{loss:>12}{l2re:>12}{ploss:>12}{pl2re:>12}")
        lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    return f"{v:.3e}"


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------
def run_optimizer_comparison(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    pdes: Optional[Sequence[str]] = None,
    widths: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    optimizers: Optional[Sequence[str]] = None,
    switch_points: Optional[Sequence[int]] = None,
    total_iterations: Optional[int] = None,
    device: Optional[str] = None,
    outdir: Optional[str] = None,
    verbose: bool = True,
    eval_every: Optional[int] = None,
    make_plots: bool = True,
    save_checkpoints: Optional[bool] = None,
    adam_lr_grid: Optional[Sequence[float]] = None,
    tune_lr: bool = True,
) -> Dict[str, Any]:
    """Sweep {PDE x width x seed x optimizer}; write Table 1 + figures."""
    cfg = dict(cfg or load_config())

    cfg_pdes = list(pdes or cfg.get("pdes") or DEFAULT_CONFIG["pdes"])
    cfg_widths = [int(w) for w in (widths or cfg.get("widths") or DEFAULT_CONFIG["widths"])]
    cfg_seeds = [int(s) for s in (seeds or cfg.get("seeds") or DEFAULT_CONFIG["seeds"])]
    cfg_opts = [str(o).lower() for o in (optimizers or cfg.get("optimizers") or DEFAULT_CONFIG["optimizers"])]
    cfg_switch = [int(s) for s in (switch_points or cfg.get("switch_points") or SWITCH_POINTS)]
    total_iters = int(total_iterations or cfg.get("total_iterations") or COMBINED_TOTAL_ITERATIONS)
    eval_every_eff = int(eval_every or cfg.get("eval_every") or 250)
    depth = int(cfg.get("depth", 3))
    dtype = getattr(torch, str(cfg.get("dtype", "float64")))
    dev = device_from_arg(device if device is not None else cfg.get("device"))
    out_root = Path(outdir or cfg.get("output_dir") or DEFAULT_CONFIG["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    lr_grid = [float(v) for v in (adam_lr_grid or cfg.get("adam_lr_grid") or ADAM_LR_GRID)]
    sampling_cfg = dict(cfg.get("sampling") or {})
    problem_cfgs = dict(cfg.get("problems") or {})
    lbfgs_cfg = dict(cfg.get("lbfgs") or {})
    record_limit = int(cfg.get("record_limit", 64))
    history_points = int(cfg.get("history_points", 200))
    if save_checkpoints is None:
        save_checkpoints = bool(cfg.get("save_checkpoints", True))
    tune_seed = cfg.get("adam_lr_tuning_seed")
    tune_seed = int(tune_seed) if tune_seed is not None else (cfg_seeds[0] if cfg_seeds else 0)
    tuning_steps = cfg.get("adam_lr_tuning_steps")
    tuning_steps = int(tuning_steps) if tuning_steps else total_iters

    for pde in cfg_pdes:
        if pde not in PROBLEMS:
            raise ValueError(f"unknown PDE '{pde}'; available: {sorted(PROBLEMS)}")

    records: List[RunRecord] = []
    tuning_log: Dict[str, List[Dict[str, Any]]] = {}
    best_lrs: Dict[str, float] = {}
    started = time.time()

    if verbose:
        print("=" * 88)
        print("Optimizer comparison (Fig. 2 / Fig. 8 / Table 1)")
        print(
            f"  PDEs={cfg_pdes}  widths={cfg_widths}  seeds={cfg_seeds}  optimizers={cfg_opts}\n"
            f"  switch points={cfg_switch}  total iterations={total_iters}  device={dev}"
        )
        print("=" * 88)

    for pde in cfg_pdes:
        problem_cfg = dict(problem_cfgs.get(pde) or {})
        for width in cfg_widths:
            key = f"{pde}/w{width}"

            # ---- Adam learning-rate tuning (paired: identical init across lrs) --------
            if tune_lr and lr_grid:
                best_lr, tuned_record, log = tune_adam_lr(
                    pde,
                    width,
                    tune_seed,
                    lr_grid=lr_grid,
                    select_by=str(cfg.get("adam_lr_select_by", "loss")),
                    total_iterations=tuning_steps,
                    sampling_cfg=sampling_cfg,
                    problem_cfg=problem_cfg,
                    depth=depth,
                    dtype=dtype,
                    device=dev,
                    eval_every=eval_every_eff,
                    grad_clip=cfg.get("adam_grad_clip"),
                    history_points=history_points,
                    save_state=save_checkpoints,
                    verbose=verbose,
                )
            else:
                best_lr = float(lr_grid[0] if lr_grid else 1e-3)
                tuned_record, log = None, []
            best_lrs[key] = float(best_lr)
            tuning_log[key] = to_jsonable(log, max_points=None) or []
            if verbose:
                print(f"  [tune] {key}: best Adam lr = {best_lr:g}")

            for seed in cfg_seeds:
                # ---- Adam ------------------------------------------------------------
                if "adam" in cfg_opts:
                    if tuned_record is not None and seed == tune_seed and tuning_steps == total_iters:
                        rec = tuned_record
                        if save_checkpoints and not rec.state_dict:
                            rec.state_dict = tuned_record.state_dict
                    else:
                        rec = train_adam(
                            pde,
                            width,
                            best_lr,
                            seed,
                            total_iterations=total_iters,
                            sampling_cfg=sampling_cfg,
                            problem_cfg=problem_cfg,
                            depth=depth,
                            dtype=dtype,
                            device=dev,
                            eval_every=eval_every_eff,
                            grad_clip=cfg.get("adam_grad_clip"),
                            history_points=history_points,
                            save_state=save_checkpoints,
                            verbose=verbose,
                        )
                    records.append(rec)
                    if verbose:
                        print(
                            f"  {pde:<11} w={width:<4} s={seed} adam        "
                            f"loss={rec.best_loss:.4e} l2re={rec.best_l2re:.4e} ({rec.seconds:.1f}s)"
                        )
                    _save_checkpoint(out_root, rec, save=save_checkpoints)

                # ---- L-BFGS ----------------------------------------------------------
                if "lbfgs" in cfg_opts:
                    rec = train_lbfgs(
                        pde,
                        width,
                        seed,
                        adam_lr=best_lr,
                        total_iterations=total_iters,
                        sampling_cfg=sampling_cfg,
                        problem_cfg=problem_cfg,
                        depth=depth,
                        dtype=dtype,
                        device=dev,
                        eval_every=eval_every_eff,
                        lbfgs_cfg=lbfgs_cfg,
                        record_limit=record_limit,
                        history_points=history_points,
                        save_state=save_checkpoints,
                        verbose=verbose,
                    )
                    records.append(rec)
                    if verbose:
                        print(
                            f"  {pde:<11} w={width:<4} s={seed} lbfgs       "
                            f"loss={rec.best_loss:.4e} l2re={rec.best_l2re:.4e} ({rec.seconds:.1f}s)"
                        )
                    _save_checkpoint(out_root, rec, save=save_checkpoints)

                # ---- Adam + L-BFGS (all switch points) -------------------------------
                if "adam+lbfgs" in cfg_opts or "adam_lbfgs" in cfg_opts:
                    for sw in cfg_switch:
                        if int(sw) >= total_iters:
                            continue
                        rec = train_combined(
                            pde,
                            width,
                            best_lr,
                            seed,
                            switch_iteration=int(sw),
                            total_iterations=total_iters,
                            sampling_cfg=sampling_cfg,
                            problem_cfg=problem_cfg,
                            depth=depth,
                            dtype=dtype,
                            device=dev,
                            eval_every=eval_every_eff,
                            lbfgs_cfg=lbfgs_cfg,
                            record_limit=record_limit,
                            history_points=history_points,
                            grad_clip=cfg.get("adam_grad_clip"),
                            save_state=save_checkpoints,
                            verbose=verbose,
                        )
                        records.append(rec)
                        if verbose:
                            print(
                                f"  {pde:<11} w={width:<4} s={seed} "
                                f"adam+lbfgs@{sw:<6} loss={rec.best_loss:.4e} "
                                f"l2re={rec.best_l2re:.4e} ({rec.seconds:.1f}s)"
                            )
                        _save_checkpoint(out_root, rec, save=save_checkpoints)

            # incremental dump so long sweeps are resumable / inspectable
            _write_progress(out_root, records, tuning_log, best_lrs, time.time() - started)

    # ---------------------------------------------------------------------------- #
    # Aggregation
    # ---------------------------------------------------------------------------- #
    by_switch = aggregate_records(records, group_by_switch=True)
    by_family = aggregate_records(records, group_by_switch=False)
    table1 = build_table1(records)
    table1_text = format_table1(table1)

    summary: Dict[str, Any] = {
        "config": to_jsonable(
            {
                "pdes": cfg_pdes,
                "widths": cfg_widths,
                "seeds": cfg_seeds,
                "optimizers": cfg_opts,
                "switch_points": cfg_switch,
                "total_iterations": total_iters,
                "eval_every": eval_every_eff,
                "adam_lr_grid": lr_grid,
                "sampling": sampling_cfg,
                "problems": problem_cfgs,
                "dtype": str(cfg.get("dtype", "float64")),
                "device": str(dev),
                "seconds": time.time() - started,
            }
        ),
        "best_adam_lr": best_lrs,
        "adam_lr_tuning": tuning_log,
        "table1": to_jsonable(table1),
        "table1_text": table1_text,
        "aggregate_by_optimizer": to_jsonable(by_switch),
        "aggregate_by_family": to_jsonable(by_family),
        "records": [r.to_metadata() for r in records],
        "histories": {r.key: r.history_dict() for r in records},
        "figure2_records": [r.as_record_dict() for r in records],
        "figure8": {
            pde: {
                str(width): {
                    family: by_family.get(pde, {}).get(width, {}).get(family, {}).get("best_l2re", {}).get("min")
                    for family in OPTIMIZER_LABELS
                    if family in by_family.get(pde, {}).get(width, {})
                }
                for width in cfg_widths
                if width in by_family.get(pde, {})
            }
            for pde in cfg_pdes
        },
    }

    (out_root / "summary.json").write_text(
        json.dumps(to_jsonable(summary, max_points=400), indent=2), encoding="utf-8"
    )
    (out_root / "table1.json").write_text(
        json.dumps(to_jsonable(table1, max_points=None), indent=2), encoding="utf-8"
    )
    (out_root / "table1.txt").write_text(table1_text, encoding="utf-8")

    if verbose:
        print("\n" + table1_text)
        print(f"[done] {len(records)} runs in {time.time() - started:.1f}s -> {out_root}")

    # ---------------------------------------------------------------------------- #
    # Figures (Fig. 2 loss-vs-L2RE scatter, Fig. 8 width sweep, convergence curves)
    # ---------------------------------------------------------------------------- #
    if make_plots and PLOTTING_AVAILABLE and _plotting is not None:
        figures: Dict[str, Any] = {}
        try:
            figures["fig2_loss_vs_l2re"] = _plotting.plot_loss_vs_l2re(
                summary["figure2_records"],
                loss_key="loss",
                l2re_key="l2re",
                group_key="optimizer",
                outfile=str(out_root / "fig2_loss_vs_l2re.png"),
                title="PINN loss vs L2RE (all PDEs / widths / seeds)",
                show=False,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Fig. 2 plot failed: {exc}", file=sys.stderr)
        try:
            figures["fig8_l2re"] = _plotting.plot_width_sweep(
                summary["figure8"],
                metric="l2re",
                outfile=str(out_root / "fig8_width_sweep_l2re.png"),
                show=False,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Fig. 8 plot failed: {exc}", file=sys.stderr)
        for pde in cfg_pdes:
            curves = {
                rec.optimizer: rec.history_dict()
                for rec in records
                if rec.pde == pde and rec.width == max(cfg_widths)
            }
            if not curves:
                continue
            try:
                figures[f"convergence_{pde}"] = _plotting.plot_training_curves(
                    curves,
                    key="losses",
                    steps_key="steps",
                    xlabel="Iteration",
                    ylabel="Loss",
                    title=f"{pde}: optimizer comparison (width {max(cfg_widths)})",
                    logy=True,
                    outfile=str(out_root / f"convergence_{pde}.png"),
                    show=False,
                )
            except Exception as exc:  # pragma: no cover
                print(f"[warn] convergence plot failed for {pde}: {exc}", file=sys.stderr)
        summary["figures"] = to_jsonable(figures)
        (out_root / "summary.json").write_text(
            json.dumps(to_jsonable(summary, max_points=400), indent=2), encoding="utf-8"
        )

    return summary


def _save_checkpoint(out_root: Path, record: RunRecord, *, save: bool) -> None:
    """Persist the trained weights + L-BFGS history for the spectral / NNCG pipelines."""
    if not save or record.state_dict is None:
        return
    ckpt_dir = out_root / "checkpoints"
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        name = f"{record.pde}_w{record.width}_s{record.seed}_{record.optimizer.replace('+', 'p')}.pt"
        payload: Dict[str, Any] = {
            "state_dict": record.state_dict,
            "metadata": record.to_metadata(),
            "adam_lr": record.adam_lr,
            "switch_iteration": record.switch_iteration,
            "optimizer": record.optimizer,
        }
        try:
            if record.lbfgs_history is not None:
                payload["lbfgs_history"] = record.lbfgs_history.as_dict()
        except Exception:
            pass
        torch.save(payload, ckpt_dir / name)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] checkpoint save failed for {record.key}: {exc}", file=sys.stderr)


def _write_progress(
    out_root: Path,
    records: Sequence[RunRecord],
    tuning_log: Dict[str, Any],
    best_lrs: Dict[str, float],
    elapsed: float,
) -> None:
    """Incremental dump of partial results (keeps long sweeps inspectable)."""
    try:
        progress = {
            "elapsed_seconds": elapsed,
            "n_records": len(records),
            "best_adam_lr": best_lrs,
            "adam_lr_tuning": to_jsonable(tuning_log, max_points=None),
            "records": [r.to_metadata() for r in records],
        }
        (out_root / "progress.json").write_text(
            json.dumps(to_jsonable(progress, max_points=None), indent=2), encoding="utf-8"
        )
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimizer comparison for PINNs (Fig. 2, Fig. 8, Table 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test sweep")
    parser.add_argument("--pdes", nargs="+", default=None, help="PDE names (convection reaction wave)")
    parser.add_argument("--widths", nargs="+", type=int, default=None, help="network widths")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="random seeds")
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        help="subset of {adam, lbfgs, adam+lbfgs}",
    )
    parser.add_argument("--switch", nargs="+", type=int, default=None, help="Adam->L-BFGS switch points")
    parser.add_argument("--total-iterations", type=int, default=None, help="total iterations per optimizer")
    parser.add_argument("--eval-every", type=int, default=None, help="L2RE evaluation cadence")
    parser.add_argument("--adam-lrs", nargs="+", type=float, default=None, help="Adam LR grid")
    parser.add_argument("--no-lr-tuning", action="store_true", help="skip the Adam LR grid search")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--outdir", type=str, default=None, help="output directory")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    parser.add_argument("--no-checkpoints", action="store_true", help="do not save model checkpoints")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, quick=args.quick)

    optimizers = args.optimizers
    if optimizers and "adam+lbfgs" in optimizers:
        optimizers = [o for o in optimizers if o != "adam_lbfgs"]

    summary = run_optimizer_comparison(
        cfg,
        pdes=args.pdes,
        widths=args.widths,
        seeds=args.seeds,
        optimizers=optimizers,
        switch_points=args.switch,
        total_iterations=args.total_iterations,
        device=args.device,
        outdir=args.outdir,
        verbose=not args.quiet,
        eval_every=args.eval_every,
        make_plots=not args.no_plots,
        save_checkpoints=False if args.no_checkpoints else None,
        adam_lr_grid=args.adam_lrs,
        tune_lr=not args.no_lr_tuning,
    )
    return summary


if __name__ == "__main__":
    main()
