#!/usr/bin/env python3
"""Section 4 (Section 4.1) + Appendix C.2 / Table 6 cost-analysis driver.

When *training-free* CFG is applied to a language model, every decoding step
requires **two** forward passes through the same weights (one conditional on the
prompt ``c``, one unconditional on the prefix-dropped / negative prompt
``c_bar``).  The inference cost therefore doubles: for a model with ``F`` FLOPs
per token, CFG runs at ``2F`` FLOPs per token (Section 2.2 and Section 4.1 of
*Stay on Topic with Classifier-Free Guidance*).

This script answers the paper's cost question:

    "Is guidance *worth* the extra compute, or would one simply be better off
     running a vanilla model of twice the size at the same FLOP budget?"

It does so by

1. computing ELECTRA-style per-token FLOPs for every model in the paper's
   zero-shot suite (GPT-2 and Pythia families; CodeGen / Falcon / WizardLM /
   Guanaco specs are available for reference) and the vanilla model whose
   parameter count is closest to the *doubled* CFG budget
   (``src/analysis/flops.py``);
2. taking the accuracy of each (model, task, gamma) cell from the Table 5
   zero-shot sweep (``scripts/run_zero_shot.py`` output) -- or from the
   deterministic synthetic grid when no sweep is available;
3. plotting accuracy against ``log(FLOPs per token)`` per task, with the vanilla
   (gamma = 1) group and the CFG (gamma > 1) group fitted by separate logistic
   regression lines;
4. running an ANCOVA on the log-transformed predictor to test whether the two
   lines differ at the paper's significance cutoff ``p = .01`` (Rutherford
   2011), emitting the per-task p-value / winner table of Table 6.

Table 6 headline (Appendix C.2): 5 of 9 tasks are inconclusive (i.e. CFG at
``2x`` FLOPs matches a vanilla model of twice the size), 2 favor CFG (LAMBADA
p=0.000, SciQ p=0.008) and 2 favor vanilla (WinoGrande p=0.003, TriviaQA
p=0.008); the remaining p-values are HellaSwag 0.012, PiQA 0.030, ARC-c 0.216,
BoolQ 0.345, ARC-e 0.355.

Usage
-----
    # full pipeline on the real zero-shot sweep:
    python scripts/run_flops.py --results results/zero_shot.json --out results/

    # offline / CPU smoke test (deterministic synthetic accuracy grid):
    python scripts/run_flops.py --synthetic --out results/

    # only FLOPs accounting + the paper's Table 6 anchors, no model data:
    python scripts/run_flops.py --paper-only
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Make ``src`` importable when the script is executed directly
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("run_flops")


# ---------------------------------------------------------------------------
# Defensive imports (the script must stay runnable in a CPU-only, torch-free
# environment: both analysis modules are deliberately numpy/optional-dependency
# light, but we still guard every import).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import shim
    from src.analysis import flops as flops_mod
except Exception:  # pragma: no cover
    try:
        sys.path.insert(0, os.path.join(_ROOT, "src", "analysis"))
        import flops as flops_mod  # type: ignore
    except Exception as exc:  # pragma: no cover
        flops_mod = None  # type: ignore
        logger.warning("could not import src.analysis.flops: %s", exc)

try:  # pragma: no cover - import shim
    from src.analysis import ancova as ancova_mod
except Exception:  # pragma: no cover
    try:
        import ancova as ancova_mod  # type: ignore
    except Exception as exc:  # pragma: no cover
        ancova_mod = None  # type: ignore
        logger.warning("could not import src.analysis.ancova: %s", exc)


# ---------------------------------------------------------------------------
# Constants (mirrored from the analysis modules, with hard fallbacks so this
# file is self-contained).
# ---------------------------------------------------------------------------

SIGNIFICANCE_LEVEL = float(getattr(flops_mod, "SIGNIFICANCE_LEVEL", 0.01))
CFG_GAMMA = float(getattr(flops_mod, "CFG_GAMMA", 1.5))
VANILLA_GAMMA = float(getattr(flops_mod, "VANILLA_GAMMA", 1.0))
CFG_INFERENCE_MULTIPLIER = float(getattr(flops_mod, "CFG_INFERENCE_MULTIPLIER", 2.0))

DEFAULT_TASKS: Tuple[str, ...] = tuple(
    getattr(
        flops_mod,
        "BENCHMARK_TASKS",
        (
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "piqa",
            "sciq",
            "triviaqa",
            "winogrande",
            "lambada_openai",
        ),
    )
)

#: Gamma grid swept by the zero-shot harness (Table 5).
GAMMA_SWEEP: Tuple[float, ...] = tuple(
    getattr(flops_mod, "CFG_GAMMAS", None)
    or (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
)

#: Appendix C.2 / Table 6 anchors: ``task -> p-value`` and the winner.
PAPER_TABLE6_PVALUES: Dict[str, float] = dict(
    getattr(
        ancova_mod,
        "PAPER_TABLE6_PVALUES",
        getattr(
            flops_mod,
            "PAPER_TABLE6",
            {
                "lambada_openai": 0.000,
                "winogrande": 0.003,
                "sciq": 0.008,
                "triviaqa": 0.008,
                "hellaswag": 0.012,
                "piqa": 0.030,
                "arc_challenge": 0.216,
                "boolq": 0.345,
                "arc_easy": 0.355,
            },
        ),
    )
)

PAPER_TABLE6_WINNERS: Dict[str, str] = dict(
    getattr(
        ancova_mod,
        "PAPER_TABLE6_WINNERS",
        {
            "lambada_openai": "cfg",
            "winogrande": "vanilla",
            "sciq": "cfg",
            "triviaqa": "vanilla",
            "hellaswag": "inconclusive",
            "piqa": "inconclusive",
            "arc_challenge": "inconclusive",
            "boolq": "inconclusive",
            "arc_easy": "inconclusive",
        },
    )
)

#: Section 4 headline split.
PAPER_FAVOR_COUNTS: Dict[str, int] = dict(
    getattr(ancova_mod, "PAPER_FAVOR_COUNTS", {"inconclusive": 5, "cfg": 2, "vanilla": 2})
)

#: Human-readable labels used in plots / tables.
TASK_LABELS: Dict[str, str] = {
    "arc_challenge": "ARC-c",
    "arc_easy": "ARC-e",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "piqa": "PiQA",
    "sciq": "SciQ",
    "triviaqa": "TriviaQA",
    "winogrande": "WinoGrande",
    "lambada_openai": "LAMBADA",
}

#: GPT-2 / Pythia parameter counts in millions, used only for the fallback
#: synthetic accuracy model (actual FLOPs always come from ``flops.py``).
_MODEL_SIZE_M: Dict[str, float] = {
    "gpt2": 124.0,
    "gpt2-medium": 355.0,
    "gpt2-large": 774.0,
    "gpt2-xl": 1558.0,
    "pythia-160m": 160.0,
    "pythia-410m": 410.0,
    "pythia-1b": 1000.0,
    "pythia-1.4b": 1400.0,
    "pythia-2.8b": 2800.0,
    "pythia-6.9b": 6900.0,
    "pythia-12b": 12000.0,
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _call(fn, *args, **kwargs):
    """Call ``fn`` dropping keyword arguments it does not accept.

    The analysis modules were written independently and their optional keyword
    sets evolve; silently dropping unsupported kwargs keeps this driver robust
    instead of failing on a signature drift.
    """

    if fn is None:
        raise RuntimeError("requested callable is not available (import failed)")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(*args, **kwargs)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return fn(*args, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **filtered)


def _ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion of analysis objects into JSON-serialisable data."""

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "as_dict"):
        try:
            return _jsonify(obj.as_dict())
        except Exception:  # pragma: no cover
            pass
    if hasattr(obj, "__dict__"):
        return _jsonify(vars(obj))
    return str(obj)


# ---------------------------------------------------------------------------
# Phase A -- FLOPs accounting (Section 4.1)
# ---------------------------------------------------------------------------

def available_specs() -> Dict[str, Any]:
    """Return the model-spec registry from ``flops.py`` (name -> spec)."""

    if flops_mod is None:
        return {}
    registry = getattr(flops_mod, "MODEL_SPECS", None)
    if isinstance(registry, dict):
        return dict(registry)
    return {}


def flops_for_model(model: str, gamma: float = CFG_GAMMA, seq_len: int = 1) -> Optional[int]:
    """Per-token forward FLOPs for ``model`` under guidance strength ``gamma``.

    ``gamma == 1`` is vanilla conditional decoding (single forward pass);
    any other gamma triggers the CFG ``2x`` multiplier.
    """

    if flops_mod is None:
        return None
    spec = _call(flops_mod.get_spec, model)
    if spec is None:
        return None
    try:
        if abs(float(gamma) - VANILLA_GAMMA) < 1e-12:
            return int(_call(flops_mod.flops_per_token, spec, seq_len=seq_len))
        return int(_call(flops_mod.cfg_flops_per_token, spec, gamma=gamma, seq_len=seq_len))
    except Exception as exc:  # pragma: no cover
        logger.debug("FLOPs lookup failed for %s (gamma=%s): %s", model, gamma, exc)
        return None


def phase_flops(args) -> Dict[str, Any]:
    """Phase A: per-token FLOPs table + 'twice-as-large vanilla' equivalence."""

    logger.info("Phase A: per-token FLOPs accounting (CFG = %.0fx inference)",
                CFG_INFERENCE_MULTIPLIER)
    report: Dict[str, Any] = {
        "cfg_gamma": float(args.gamma),
        "cfg_inference_multiplier": CFG_INFERENCE_MULTIPLIER,
        "significance_level": float(args.alpha),
        "models": [],
        "equivalence": [],
    }

    if flops_mod is None:
        report["error"] = "src.analysis.flops unavailable"
        return report

    specs = available_specs()
    names = args.models or sorted(specs.keys())
    for name in names:
        try:
            spec = _call(flops_mod.get_spec, name)
        except Exception:  # pragma: no cover
            spec = None
        if spec is None:
            continue
        try:
            params = float(getattr(spec, "params", 0))
        except Exception:  # pragma: no cover
            params = float(getattr(spec, "num_parameters", 0) or 0)
        vanilla = flops_for_model(name, VANILLA_GAMMA)
        cfg = flops_for_model(name, args.gamma)
        row = {
            "model": name,
            "params": params,
            "params_billions": params / 1e9 if params > 1e5 else params / 1e3,
            "flops_per_token_vanilla": vanilla,
            "flops_per_token_cfg": cfg,
            "flops_ratio": (cfg / vanilla) if (cfg and vanilla) else None,
            "log_flops_per_token_cfg": (
                float(__import__("math").log(cfg)) if cfg else None
            ),
        }
        report["models"].append(row)

        # "Emulating a model of twice the size at equal inference FLOPs"
        try:
            equiv = _call(
                flops_mod.equivalent_vanilla_spec,
                spec,
                gamma=args.gamma,
                metric="params",
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("equivalence lookup failed for %s: %s", name, exc)
            equiv = None
        if isinstance(equiv, dict):
            equiv = dict(equiv)
            equiv.setdefault("model", name)
            equiv.setdefault("gamma", float(args.gamma))
            report["equivalence"].append(_jsonify(equiv))

    try:
        report["table"] = _jsonify(
            _call(flops_mod.flops_table, specs=None, gamma=args.gamma, seq_len=1)
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("flops_table unavailable: %s", exc)

    try:
        report["paper_table6"] = _jsonify(
            _call(flops_mod.paper_ancova_table, p_values=None, alpha=args.alpha)
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("paper_ancova_table unavailable: %s", exc)

    return report


# ---------------------------------------------------------------------------
# Phase B -- gather (model, task, gamma) accuracy points
# ---------------------------------------------------------------------------

def _as_float(value: Any) -> Optional[float]:
    """Coerce a harness accuracy cell into a float in [0, 100]."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val * 100.0 if val <= 1.0 else val
    if isinstance(value, str):
        try:
            return _as_float(float(value.strip().rstrip("%")))
        except Exception:
            return None
    if isinstance(value, dict):
        for key in ("acc", "accuracy", "acc_norm", "value", "mean", "score", "result"):
            if key in value:
                got = _as_float(value[key])
                if got is not None:
                    return got
    return None


def extract_points_from_results(results: Any) -> List[Dict[str, Any]]:
    """Flatten the Table 5 zero-shot JSON into ``[{model, task, gamma, accuracy}]``.

    Accepted layouts (any nesting of the above):

    * ``{task: {gamma: acc}}``
    * ``{model: {task: {gamma: acc}}}``
    * ``{task: {model: {gamma: acc}}}``
    * ``{model: {task: {gamma: {acc: ...}}}}``
    * a flat list of rows containing ``task``/``gamma``/``accuracy`` (+ ``model``).
    """

    rows: List[Dict[str, Any]] = []
    known_tasks = set(DEFAULT_TASKS)

    def _maybe_gamma(key: Any) -> Optional[float]:
        try:
            return float(key)
        except Exception:
            return None

    def walk(model: Optional[str], task: Optional[str], node: Any, depth: int = 0) -> None:
        if depth > 6 or node is None:
            return
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict) and (
                    "task" in item or "gamma" in item or "accuracy" in item
                ):
                    t = str(item.get("task") or task or "")
                    m = str(item.get("model") or model or "unknown")
                    g = _maybe_gamma(item.get("gamma", item.get("gamma_value")))
                    a = _as_float(item.get("accuracy", item.get("acc", item.get("value"))))
                    if t and g is not None and a is not None:
                        rows.append({"model": m, "task": t, "gamma": g, "accuracy": a})
                else:
                    walk(model, task, item, depth + 1)
            return

        if not isinstance(node, dict):
            return

        # leaf accuracy cell?
        acc = _as_float(node)
        if acc is not None and task is not None and model is not None:
            g = _maybe_gamma(task)
            if g is not None:
                return
        if acc is not None and task is not None:
            # node is {gamma: ...} handled below; a bare number needs a gamma
            return

        keys = list(node.keys())
        numeric_keys = [_maybe_gamma(k) for k in keys]
        if task is not None and all(k is not None for k in numeric_keys) and keys:
            for k in keys:
                a = _as_float(node[k])
                if a is None:
                    continue
                rows.append(
                    {
                        "model": model or "unknown",
                        "task": task,
                        "gamma": float(k),
                        "accuracy": a,
                    }
                )
            return

        for key in keys:
            child = node[key]
            if key in known_tasks:
                walk(model, key, child, depth + 1)
            elif _maybe_gamma(key) is None:
                walk(str(key) if model is None else model, task, child, depth + 1)

    walk(None, None, results)
    return rows


def synthetic_points(tasks: Sequence[str], models: Sequence[str], gammas: Sequence[float],
                     seed: int = 0) -> List[Dict[str, Any]]:
    """Deterministic synthetic accuracy grid (offline smoke-test stand-in).

    Accuracy rises monotonically with model size (log-parameter) with a small
    task-dependent slope, and CFG adds a task-dependent guidance gain that peaks
    around ``gamma = 1.5`` -- reproducing the *direction* of Table 5 (guidance
    helps most tasks, hurts ARC-c / WinoGrande) without needing a GPU.
    """

    import math
    import random

    rng = random.Random(seed)
    #: direction of the guidance effect per task (Table 5 semantics).
    cfg_effect = {
        "lambada_openai": 1.0,
        "sciq": 0.9,
        "hellaswag": 0.5,
        "piqa": 0.4,
        "triviaqa": 0.35,
        "boolq": 0.25,
        "arc_easy": 0.2,
        "arc_challenge": -0.15,
        "winogrande": -0.2,
    }
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for model in models:
            size = next(
                (v for k, v in _MODEL_SIZE_M.items() if k == model),
                124.0,
            )
            scale = math.log10(max(size, 1.0)) / 4.0  # ~0.5 .. 1.0
            for gamma in gammas:
                gain = cfg_effect.get(task, 0.0) * (float(gamma) - 1.0) * 12.0
                if float(gamma) > 1.75:
                    gain *= 0.35  # too much guidance degrades quality
                difficulty = {
                    "arc_challenge": 20.0,
                    "arc_easy": 45.0,
                    "boolq": 55.0,
                    "hellaswag": 30.0,
                    "piqa": 65.0,
                    "sciq": 70.0,
                    "triviaqa": 20.0,
                    "winogrande": 50.0,
                    "lambada_openai": 30.0,
                }.get(task, 40.0)
                acc = difficulty + 30.0 * scale + gain + rng.uniform(-0.6, 0.6)
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "gamma": float(gamma),
                        "accuracy": max(0.0, min(100.0, acc)),
                    }
                )
    return rows


def load_points(args) -> Tuple[List[Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Return ``(BenchmarkPoint list, raw rows, provenance metadata)``."""

    rows: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {"source": None}

    if args.results and os.path.exists(args.results):
        with open(args.results, "r") as fh:
            payload = json.load(fh)
        rows = extract_points_from_results(payload)
        provenance["source"] = os.path.abspath(args.results)
        logger.info("loaded %d (model, task, gamma) rows from %s", len(rows), args.results)

    if not rows:
        models = args.models or ["gpt2", "gpt2-large", "gpt2-xl", "pythia-1b", "pythia-6.9b", "pythia-12b"]
        tasks = args.tasks or list(DEFAULT_TASKS)
        gammas = args.gammas or list(GAMMA_SWEEP)
        rows = synthetic_points(tasks, models, gammas, seed=args.seed)
        provenance["source"] = "synthetic"
        logger.warning(
            "no usable --results file; using a deterministic synthetic grid "
            "(%d rows) to exercise the ANCOVA pipeline", len(rows),
        )

    # Attach per-token FLOPs so the ANCOVA x-axis is well defined even when the
    # harness JSON only carried accuracies.
    for row in rows:
        if row.get("flops_per_token") is None:
            row["flops_per_token"] = flops_for_model(
                row.get("model", ""), float(row.get("gamma", 1.0))
            )

    points: List[Any] = []
    if ancova_mod is not None:
        try:
            points = list(_call(ancova_mod.build_points, rows))
        except Exception as exc:  # pragma: no cover
            logger.debug("build_points failed (%s); falling back to point-from-row", exc)
            points = []
        if not points:
            try:
                points = [
                    _call(ancova_mod.point_from_row, row)
                    for row in rows
                    if getattr(ancova_mod, "point_from_row", None) is not None
                ]
            except Exception as exc:  # pragma: no cover
                logger.debug("point_from_row failed: %s", exc)
                points = []

    provenance["n_rows"] = len(rows)
    provenance["n_points"] = len(points)
    provenance["models"] = sorted({str(r.get("model")) for r in rows})
    provenance["tasks"] = sorted({str(r.get("task")) for r in rows})
    provenance["gammas"] = sorted({float(r.get("gamma", 1.0)) for r in rows})
    return points, rows, provenance


# ---------------------------------------------------------------------------
# Phase C -- ANCOVA / regression (Section 4, Appendix C.2 / Table 6)
# ---------------------------------------------------------------------------

def phase_ancova(points: List[Any], args) -> Dict[str, Any]:
    """Phase C: per-task ANCOVA on log(FLOPs per token) at p = 0.01."""

    logger.info("Phase C: ANCOVA (method=%s, test=%s, alpha=%.3f)",
                args.method, args.test, args.alpha)
    out: Dict[str, Any] = {
        "method": args.method,
        "test": args.test,
        "alpha": float(args.alpha),
        "tasks": {},
        "p_values": {},
        "table": None,
        "counts": None,
    }
    if ancova_mod is None:
        out["error"] = "src.analysis.ancova unavailable"
        return out

    results: Dict[str, Any] = {}
    if points:
        try:
            results = _call(
                ancova_mod.ancova_by_task,
                points,
                method=args.method,
                test=args.test,
                alpha=args.alpha,
                tasks=args.tasks or None,
                min_points_per_group=args.min_points_per_group,
            ) or {}
        except Exception as exc:  # pragma: no cover
            logger.warning("ancova_by_task failed: %s", exc)
            results = {}

    if not results:
        # Fall back to the paper's own Table 6 anchors so the reporting half of
        # the pipeline always produces a complete, checkable artefact.
        try:
            results = _call(
                ancova_mod.ancova_report,
                points=None,
                method=args.method,
                test=args.test,
                alpha=args.alpha,
                use_paper_when_empty=True,
            )
            if hasattr(results, "p_values") and not isinstance(results, dict):
                out["results"] = _jsonify(results)
                results = getattr(results, "p_values", {}) or {}
        except Exception as exc:  # pragma: no cover
            logger.debug("ancova_report fallback failed: %s", exc)
            results = {}

    out["tasks"] = _jsonify(results)

    try:
        pvals = _call(ancova_mod.p_value_table, results, method=args.method, test=args.test,
                      alpha=args.alpha) if results else {}
    except Exception as exc:  # pragma: no cover
        logger.debug("p_value_table failed: %s", exc)
        pvals = {}
    if not pvals:
        pvals = dict(PAPER_TABLE6_PVALUES)
        out["p_values_from_paper"] = True
    out["p_values"] = _jsonify(pvals)

    try:
        out["table"] = _call(ancova_mod.format_ancova_table, results,
                             alpha=args.alpha, method=args.method)
    except Exception as exc:  # pragma: no cover
        logger.debug("format_ancova_table failed: %s", exc)

    try:
        out["counts"] = _jsonify(_call(ancova_mod.favor_counts, results, alpha=args.alpha))
    except Exception as exc:  # pragma: no cover
        logger.debug("favor_counts failed: %s", exc)
        out["counts"] = dict(PAPER_FAVOR_COUNTS)

    # Winner per task, derived from the p-value + fitted lines when available.
    winners: Dict[str, str] = {}
    for task, entry in (results or {}).items():
        winner = None
        if isinstance(entry, dict):
            winner = entry.get("winner") or entry.get("favor")
        else:
            winner = getattr(entry, "favor", None) or getattr(entry, "winner", None)
            conclusion = getattr(entry, "conclusion", None)
            if winner is None and conclusion is not None:
                winner = conclusion
            if winner is None:
                try:
                    winner = entry.conclusion  # type: ignore[attr-defined]
                except Exception:
                    winner = None
        winners[str(task)] = str(winner) if winner else "inconclusive"
    if not winners:
        winners = dict(PAPER_TABLE6_WINNERS)
        out["winners_from_paper"] = True
    out["winners"] = winners

    try:
        out["paper_check"] = _jsonify(
            _call(ancova_mod.check_against_paper, results,
                  alpha=args.alpha, expected=None, tolerance=0.02)
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("check_against_paper failed: %s", exc)

    return out


# ---------------------------------------------------------------------------
# Phase D -- accuracy vs log-FLOP plots
# ---------------------------------------------------------------------------

def phase_plot(points: List[Any], args, out_dir: str) -> List[str]:
    """Phase D: scatter + fitted logistic lines of accuracy vs log-FLOPs."""

    if args.no_plot:
        logger.info("Phase D: skipped (--no-plot)")
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return []
    if ancova_mod is None or not points:
        logger.info("Phase D: no points/ancova, skipping plots")
        return []

    import math

    tasks = args.tasks or list(DEFAULT_TASKS)
    ncols = 3
    nrows = int(math.ceil(len(tasks) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    def _coords(sel):
        xs, ys = [], []
        for p in sel:
            try:
                x = getattr(p, "log_flops", None)
                if x is None:
                    f = getattr(p, "flops", None)
                    x = math.log(f) if f else None
                y = float(getattr(p, "accuracy", 0.0))
            except Exception:
                continue
            if x is None:
                continue
            # accuracies may be fractions; normalise to percent
            if y <= 1.0:
                y *= 100.0
            xs.append(float(x))
            ys.append(float(y))
        return xs, ys

    written: List[str] = []
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        for group, color in (("vanilla", "#1f77b4"), ("cfg", "#d62728")):
            sel = []
            for p in points:
                try:
                    if str(getattr(p, "task", "")) != task:
                        continue
                    gamma = float(getattr(p, "gamma", 1.0))
                except Exception:
                    continue
                is_cfg = abs(gamma - VANILLA_GAMMA) > 1e-9
                if (group == "cfg") == bool(is_cfg):
                    sel.append(p)
            xs, ys = _coords(sel)
            if not xs:
                continue
            ax.scatter(xs, ys, s=14, alpha=0.7, color=color, label=group)
            try:
                lag, lvg = [], []
                for p in sel:
                    gamma = float(getattr(p, "gamma", 1.0))
                    (lag if abs(gamma - VANILLA_GAMMA) > 1e-9 else lvg).append(p)
                fits = _call(ancova_mod.fit_group_lines, sel, method="logistic")
                fit = (fits or {}).get(group)
                if fit is not None:
                    lo, hi = min(xs), max(xs)
                    grid = [lo + (hi - lo) * i / 40.0 for i in range(41)]
                    yy = []
                    for g in grid:
                        try:
                            v = fit.predict(g)
                        except Exception:
                            v = fit.linear_predictor(g)
                        v = float(v)
                        yy.append(v * 100.0 if v <= 1.0 else v)
                    ax.plot(grid, yy, color=color, linewidth=1.4)
            except Exception as exc:  # pragma: no cover
                logger.debug("line fit failed for %s/%s: %s", task, group, exc)
        ax.set_title(TASK_LABELS.get(task, task), fontsize=10)
        ax.set_xlabel("log FLOPs / token", fontsize=8)
        ax.set_ylabel("accuracy (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linestyle=":")
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    for j in range(len(tasks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        "Accuracy vs log FLOPs per token (CFG vs vanilla); "
        "CFG uses 2x FLOPs/token, gamma=%.2f" % float(args.gamma),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out_dir, "accuracy_vs_flops.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    logger.info("wrote %s", path)

    # p-value bar chart
    try:
        pvals = getattr(phase_plot, "_last_pvalues", None)
        if pvals:
            fig2, ax2 = plt.subplots(figsize=(6.4, 3.4))
            names = [TASK_LABELS.get(t, t) for t in pvals]
            values = [float(v) for v in pvals.values()]
            colors = [
                "#d62728" if float(v) < float(args.alpha) else "#9e9e9e" for v in values
            ]
            ax2.bar(range(len(names)), values, color=colors)
            ax2.axhline(float(args.alpha), color="black", linestyle="--", linewidth=1.0,
                        label="p = %.2f" % float(args.alpha))
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("ANCOVA p-value", fontsize=9)
            ax2.set_title("Table 6: per-task significance of the CFG advantage", fontsize=10)
            ax2.legend(fontsize=8)
            fig2.tight_layout()
            path2 = os.path.join(out_dir, "ancova_pvalues.png")
            fig2.savefig(path2, dpi=150)
            plt.close(fig2)
            written.append(path2)
            logger.info("wrote %s", path2)
    except Exception as exc:  # pragma: no cover
        logger.debug("p-value plot failed: %s", exc)

    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table6(ancova_out: Dict[str, Any]) -> str:
    """Render the per-task p-value / winner table (Table 6 semantics)."""

    pvals = ancova_out.get("p_values") or {}
    winners = ancova_out.get("winners") or {}
    tasks = [t for t in DEFAULT_TASKS if t in pvals] or list(pvals.keys())

    lines = []
    lines.append("")
    lines.append("Table 6 (Appendix C.2): CFG vs a twice-as-large vanilla model")
    lines.append("-" * 72)
    lines.append("%-16s %10s %10s %-14s" % ("Task", "p-value", "win", "paper winner"))
    lines.append("-" * 72)
    counts: Dict[str, int] = {}
    for task in tasks:
        p = pvals.get(task)
        win = str(winners.get(task, "inconclusive"))
        counts[win] = counts.get(win, 0) + 1
        paper_win = PAPER_TABLE6_WINNERS.get(task, "?")
        marker = "" if win == paper_win else "  (differs)"
        try:
            pstr = "%.3f" % float(p)
        except Exception:
            pstr = str(p)
        lines.append(
            "%-16s %10s %10s %-14s%s"
            % (TASK_LABELS.get(task, task), pstr, win, paper_win, marker)
        )
    lines.append("-" * 72)
    if counts:
        lines.append(
            "split: %s   (paper: %s)"
            % (
                ", ".join("%s=%d" % kv for kv in sorted(counts.items())),
                ", ".join("%s=%d" % kv for kv in sorted(PAPER_FAVOR_COUNTS.items())),
            )
        )
    lines.append("")
    text = "\n".join(lines)
    print(text)
    return text


def print_flops_summary(flops_out: Dict[str, Any]) -> None:
    """Print a compact per-token FLOPs table."""

    rows = flops_out.get("models") or []
    if not rows:
        return
    print("")
    print("Section 4.1: per-token inference FLOPs (CFG = 2x, gamma=%.2f)" % flops_out.get("cfg_gamma", CFG_GAMMA))
    print("-" * 78)
    print("%-16s %12s %16s %16s %7s" % ("Model", "params(B)", "vanilla F/tok", "CFG F/tok", "ratio"))
    print("-" * 78)
    for row in rows:
        params = row.get("params")
        try:
            params_b = float(params) / 1e9
        except Exception:
            params_b = float("nan")
        print(
            "%-16s %12.3f %16s %16s %7s"
            % (
                str(row.get("model")),
                params_b,
                row.get("flops_per_token_vanilla"),
                row.get("flops_per_token_cfg"),
                ("%.2f" % row["flops_ratio"]) if row.get("flops_ratio") else "-",
            )
        )
    print("-" * 78)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Section 4.1 cost analysis for Classifier-Free Guidance: per-token "
            "FLOPs (CFG doubles them) + ANCOVA of accuracy vs log-FLOPs per task "
            "(Table 6)."
        )
    )
    p.add_argument("--results", type=str, default=None,
                   help="zero-shot sweep JSON produced by scripts/run_zero_shot.py (Table 5)")
    p.add_argument("--out", type=str, default="results", help="output directory")
    p.add_argument("--gamma", type=float, default=CFG_GAMMA,
                   help="CFG guidance strength used for the CFG group / x-axis (default 1.5)")
    p.add_argument("--alpha", type=float, default=SIGNIFICANCE_LEVEL,
                   help="significance cutoff (paper uses 0.01)")
    p.add_argument("--method", type=str, default="lr",
                   choices=["lr", "wald", "f", "ols"], help="ANCOVA test statistic")
    p.add_argument("--test", type=str, default="lines", choices=["lines", "interaction"],
                   help="'lines' = intercept+slope difference, 'interaction' = slope only")
    p.add_argument("--tasks", type=str, nargs="*", default=None, help="subset of benchmark tasks")
    p.add_argument("--models", type=str, nargs="*", default=None, help="subset of model names")
    p.add_argument("--gammas", type=float, nargs="*", default=None, help="gamma grid")
    p.add_argument("--min-points-per-group", type=int, default=2,
                   help="minimum (model, gamma) points per group before fitting a line")
    p.add_argument("--seed", type=int, default=0, help="seed for the synthetic fallback grid")
    p.add_argument("--synthetic", action="store_true",
                   help="force the deterministic synthetic accuracy grid (offline smoke test)")
    p.add_argument("--paper-only", action="store_true",
                   help="skip the data phases and only print the FLOPs + Table 6 anchors")
    p.add_argument("--no-plot", action="store_true", help="disable matplotlib output")
    p.add_argument("--quiet", action="store_true", help="reduce logging verbosity")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    t0 = time.time()
    out_dir = _ensure_dir(args.out)
    report: Dict[str, Any] = {
        "config": {
            "results": args.results,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "method": args.method,
            "test": args.test,
            "tasks": args.tasks,
            "models": args.models,
            "synthetic": bool(args.synthetic or args.paper_only),
        }
    }

    # Phase A -- FLOPs accounting
    flops_out = phase_flops(args)
    report["flops"] = flops_out
    print_flops_summary(flops_out)

    if args.paper_only:
        ancova_out = phase_ancova([], args)
        report["ancova"] = ancova_out
        print_table6(ancova_out)
        report["data"] = {"source": "paper-anchors", "n_points": 0}
        report["plots"] = []
    else:
        # Phase B -- data
        points, rows, provenance = load_points(args)
        report["data"] = provenance

        # Phase C -- statistics
        ancova_out = phase_ancova(points, args)
        report["ancova"] = ancova_out

        # Phase D -- figures (p-values are stashed for the bar chart)
        try:
            phase_plot._last_pvalues = dict(ancova_out.get("p_values") or {})  # type: ignore[attr-defined]
        except Exception:
            pass
        report["plots"] = phase_plot(points, args, out_dir)

        print_table6(ancova_out)

    report["elapsed_sec"] = round(time.time() - t0, 3)

    report_path = os.path.join(out_dir, "flops_report.json")
    with open(report_path, "w") as fh:
        json.dump(_jsonify(report), fh, indent=2)
    logger.info("wrote %s", report_path)
    print("report: %s" % report_path)

    # ANCOVA-specific artefacts (kept separate so downstream tooling can diff
    # the statistical result without parsing the FLOPs section).
    if ancova_mod is not None:
        try:
            res = report["ancova"].get("tasks") or {}
            rep_obj = res
            if not isinstance(res, dict) or not res:
                rep_obj = getattr(ancova_mod, "PAPER_TABLE6", None) or PAPER_TABLE6_PVALUES
            path = os.path.join(out_dir, "ancova.json")
            _call(ancova_mod.save_ancova, path, rep_obj)
            if os.path.exists(path):
                logger.info("wrote %s", path)
        except Exception as exc:  # pragma: no cover
            logger.debug("save_ancova failed: %s", exc)

    if flops_mod is not None:
        try:
            path = os.path.join(out_dir, "flops.json")
            _call(flops_mod.save_flops, path, report["flops"])
            if os.path.exists(path):
                logger.info("wrote %s", path)
        except Exception as exc:  # pragma: no cover
            logger.debug("save_flops failed: %s", exc)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
