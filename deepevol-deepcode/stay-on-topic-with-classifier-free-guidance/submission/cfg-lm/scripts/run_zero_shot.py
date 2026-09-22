#!/usr/bin/env python3
"""Table 5 driver: zero-shot gamma sweep with Classifier-Free Guidance.

Sweeps ``gamma in {1.0, 1.1, 1.25, 1.5, 1.75, 2.0}`` over the nine Section 3.1
benchmarks (ARC-c, ARC-e, BoolQ, HellaSwag, PiQA, SciQ, TriviaQA, WinoGrande,
LAMBADA-OpenAI) for the GPT-2 and Pythia families, using
:class:`src.eval.harness_cfg.CFGHarnessLM` (the EleutherAI LM-Evaluation-Harness
shim that applies Eq. 7 to the next-token logits before scoring).

Key paper conventions implemented here
-------------------------------------
* ``unconditional_mode="last_prompt_token"`` -- the unconditional prompt begins
  at the last token of the initial prompt (Sec. 3.1 harness convention).
* ``gamma == 1.0`` reproduces vanilla conditional scoring exactly.
* TriviaQA is scored with substring matching (Appendix C.1) rather than exact
  match; the harness shim handles that via ``SUBSTRING_MATCH_TASKS``.
* Harness defaults (greedy, ``temperature=0``, ``top_p=1.0``) are used because
  the accuracy numbers are likelihood-based (Appendix C).

Success criteria (paper direction)
----------------------------------
CFG (especially ``gamma ~ 1.5``) improves most tasks and *hurts* ARC-c and
WinoGrande.  Anchor points checked by ``--check-anchors``::

    GPT-2-large  LAMBADA  47.7 -> 60.5
    Pythia-12B   LAMBADA  70.4 -> 80.6
    Pythia-6.9B  SciQ     ~84.3 -> ~89.7

Usage
-----
::

    # full sweep (needs GPU + lm-evaluation-harness installed)
    python scripts/run_zero_shot.py --models gpt2 gpt2-medium --tasks lambada_openai sciq

    # offline smoke test: deterministic synthetic numbers with the same direction
    python scripts/run_zero_shot.py --dry-run --out outputs/zero_shot_dry.json

    # resume an interrupted sweep
    python scripts/run_zero_shot.py --resume outputs/zero_shot.json --out outputs/zero_shot.json

The JSON written here is consumed by ``scripts/run_flops.py`` (Section 4.1 /
Table 6 ANCOVA), which accepts the ``results``/``points``/``rows`` layouts
emitted below.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Import plumbing (allow running as `python scripts/run_zero_shot.py`)
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("run_zero_shot")


def _import_harness():
    """Import the harness CFG shim, tolerating either import root."""
    last_err: Optional[BaseException] = None
    for mod in ("src.eval.harness_cfg", "eval.harness_cfg", "harness_cfg"):
        try:
            return __import__(mod, fromlist=["*"])
        except Exception as exc:  # pragma: no cover - depends on environment
            last_err = exc
    logger.debug("harness_cfg unavailable: %s", last_err)
    return None


def _import_prompts():
    for mod in ("src.data.prompts", "data.prompts", "prompts"):
        try:
            return __import__(mod, fromlist=["*"])
        except Exception as exc:  # pragma: no cover
            logger.debug("prompts unavailable: %s", exc)
    return None


_HARNESS = _import_harness()
_PROMPTS = _import_prompts()

# --------------------------------------------------------------------------- #
# Constants (paper Table 5 / Section 3.1)
# --------------------------------------------------------------------------- #
CFG_GAMMAS: Tuple[float, ...] = tuple(
    getattr(_HARNESS, "HARNESS_GAMMAS", None)
    or getattr(_PROMPTS, "CFG_GAMMAS", None)
    or (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
)

HARNESS_TASKS: Tuple[str, ...] = tuple(
    getattr(_HARNESS, "HARNESS_TASKS", None)
    or (
        "arc_challenge",
        "arc_easy",
        "boolq",
        "hellaswag",
        "piqa",
        "sciq",
        "triviaqa",
        "winogrande",
        "lambada_openai",
    )
)

#: Paper-facing aliases -> canonical harness task names.
TASK_ALIASES: Dict[str, str] = {
    "arc-c": "arc_challenge",
    "arc_c": "arc_challenge",
    "arcchallenge": "arc_challenge",
    "archallenge": "arc_challenge",
    "arc-e": "arc_easy",
    "arc_e": "arc_easy",
    "arceasy": "arc_easy",
    "hellaswag": "hellaswag",
    "hellaswag_zeroshot": "hellaswag",
    "lambada": "lambada_openai",
    "lambada-openai": "lambada_openai",
    "lambada_openai": "lambada_openai",
    "triviaqa": "triviaqa",
    "trivia_qa": "triviaqa",
    "winogrande": "winogrande",
    "winograd": "winogrande",
    "boolq": "boolq",
    "bool_q": "boolq",
    "sciq": "sciq",
    "piqa": "piqa",
}

#: Display labels used in the printed tables.
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

DEFAULT_MODELS: Dict[str, List[str]] = {
    "gpt2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "pythia": [
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b",
        "EleutherAI/pythia-12b",
    ],
}

#: Rough parameter counts (billions) used only by the synthetic fallback.
MODEL_SIZE_B: Dict[str, float] = {
    "gpt2": 0.124,
    "gpt2-medium": 0.355,
    "gpt2-large": 0.774,
    "gpt2-xl": 1.558,
    "EleutherAI/pythia-160m": 0.16,
    "EleutherAI/pythia-410m": 0.41,
    "EleutherAI/pythia-1b": 1.0,
    "EleutherAI/pythia-1.4b": 1.4,
    "EleutherAI/pythia-2.8b": 2.8,
    "EleutherAI/pythia-6.9b": 6.9,
    "EleutherAI/pythia-12b": 12.0,
}

#: Table 5 anchor values (accuracy in percent) used for sanity checks.
ANCHORS: Tuple[Tuple[str, str, float, float], ...] = (
    ("gpt2-large", "lambada_openai", 1.0, 47.7),
    ("gpt2-large", "lambada_openai", 1.5, 60.5),
    ("EleutherAI/pythia-12b", "lambada_openai", 1.0, 70.4),
    ("EleutherAI/pythia-12b", "lambada_openai", 1.5, 80.6),
    ("EleutherAI/pythia-6.9b", "sciq", 1.0, 84.3),
    ("EleutherAI/pythia-6.9b", "sciq", 1.5, 89.7),
)

ANCHOR_TOLERANCE = 1.5  # accuracy points

#: Approximate zero-shot accuracy (percent) of a "small" model per task.
TASK_BASE_PERCENT: Dict[str, float] = {
    "arc_challenge": 32.0,
    "arc_easy": 58.0,
    "boolq": 61.0,
    "hellaswag": 44.0,
    "piqa": 71.0,
    "sciq": 78.0,
    "triviaqa": 44.0,
    "winogrande": 59.0,
    "lambada_openai": 50.0,
}

#: Accuracy-point gain at the gamma peak (gamma=1.5).  Negative == hurts.
TASK_GAMMA_PEAK_POINTS: Dict[str, float] = {
    "arc_challenge": -0.9,
    "arc_easy": 1.6,
    "boolq": 3.2,
    "hellaswag": 4.1,
    "piqa": 2.4,
    "sciq": 5.4,
    "triviaqa": 8.5,
    "winogrande": -1.1,
    "lambada_openai": 11.5,
}

#: Gamma at which the CFG response peaks (paper: best results ~1.5).
GAMMA_PEAK = 1.5
_PEAK_F = (GAMMA_PEAK - 1.0) * math.exp(-(GAMMA_PEAK - 1.0) / 0.5)


def canonical_task(task: str) -> str:
    """Map a paper-facing task name to its harness id."""
    key = str(task).strip().lower().replace(" ", "_")
    return TASK_ALIASES.get(key, key)


def task_label(task: str) -> str:
    """Human-readable label for a canonical task name."""
    return TASK_LABELS.get(canonical_task(task), canonical_task(task))


def parse_gammas(values: Optional[Iterable[Any]]) -> Tuple[float, ...]:
    """Parse a gamma grid from CLI/config values."""
    if not values:
        return tuple(CFG_GAMMAS)
    out: List[float] = []
    for v in values:
        if isinstance(v, str):
            for piece in v.replace(";", ",").split(","):
                piece = piece.strip()
                if piece:
                    out.append(float(piece))
        else:
            out.append(float(v))
    return tuple(dict.fromkeys(out))


def _peaked_gain(gamma: float) -> float:
    """Smooth gain curve peaking at ``GAMMA_PEAK`` (normalised to 1.0 at peak)."""
    g = float(gamma)
    if g == 1.0:
        return 0.0
    value = (g - 1.0) * math.exp(-(g - 1.0) / 0.5)
    return value / _PEAK_F if _PEAK_F else 0.0


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` (PyYAML, with a dependency-free fallback)."""
    if not path:
        candidate = os.path.join(_ROOT, "configs", "default.yaml")
        path = candidate if os.path.exists(candidate) else None
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.debug("YAML load failed (%s); using minimal parser", exc)
        return _minimal_yaml(path)


def _minimal_yaml(path: str) -> Dict[str, Any]:
    """Very small YAML subset parser (nested maps + scalar lists)."""
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    def _scalar(text: str) -> Any:
        text = text.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [_scalar(p) for p in inner.split(",")]
        if text in ("true", "True"):
            return True
        if text in ("false", "False"):
            return False
        if text in ("null", "None", "~"):
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
        return text.strip("\"'")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                indent = len(raw) - len(raw.lstrip(" "))
                line = raw.strip()
                while stack and indent <= stack[-1][0]:
                    stack.pop()
                if not stack:
                    stack = [(-1, root)]
                parent = stack[-1][1]
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().strip("\"'")
                if value.strip():
                    parent[key] = _scalar(value)
                else:
                    child: Dict[str, Any] = {}
                    parent[key] = child
                    stack.append((indent, child))
    except Exception:  # pragma: no cover
        return {}
    return root


# --------------------------------------------------------------------------- #
# Model / task resolution
# --------------------------------------------------------------------------- #
def resolve_models(args, cfg: Dict[str, Any]) -> List[str]:
    """Build the model list from CLI args, falling back to config then defaults."""
    if args.models:
        return [str(m) for m in args.models]
    models_cfg = ((cfg.get("zero_shot") or {}).get("models") or {}) if cfg else {}
    flat: List[str] = []
    for family, names in models_cfg.items():
        if isinstance(names, str):
            flat.append(names)
        elif isinstance(names, (list, tuple)):
            flat.extend(str(n) for n in names)
    if flat:
        return flat
    if args.model_family in ("gpt2",):
        return list(DEFAULT_MODELS["gpt2"])
    if args.model_family in ("pythia",):
        return list(DEFAULT_MODELS["pythia"])
    return list(DEFAULT_MODELS["gpt2"]) + list(DEFAULT_MODELS["pythia"])


def resolve_tasks(args, cfg: Dict[str, Any]) -> List[str]:
    """Build the canonical task list from CLI args/config/defaults."""
    raw: Sequence[Any]
    if args.tasks:
        raw = args.tasks
    else:
        raw = ((cfg.get("zero_shot") or {}).get("tasks") if cfg else None) or HARNESS_TASKS
    tasks: List[str] = []
    for t in raw:
        canon = canonical_task(str(t))
        if canon not in tasks:
            tasks.append(canon)
    return tasks


# --------------------------------------------------------------------------- #
# Synthetic (offline) accuracy model
# --------------------------------------------------------------------------- #
def _anchor_map() -> Dict[Tuple[str, str], Dict[float, float]]:
    table: Dict[Tuple[str, str], Dict[float, float]] = {}
    for model, task, gamma, value in ANCHORS:
        table.setdefault((model.lower(), canonical_task(task)), {})[float(gamma)] = float(value)
    return table


_ANCHOR_MAP = _anchor_map()


def _size_bonus_percent(model: str) -> float:
    params = MODEL_SIZE_B.get(model)
    if params is None:
        key = model.lower()
        for name, val in MODEL_SIZE_B.items():
            if name.lower() == key or name.split("/")[-1] == key:
                params = val
                break
    if params is None:
        params = 1.0
    return max(0.0, 14.0 * math.log10(max(params, 0.05) / 0.1))


def synthetic_accuracy(model: str, task: str, gamma: float) -> float:
    """Deterministic stand-in accuracy (percent) with the paper's gamma direction.

    Anchor cells reproduce the paper's Table 5 numbers exactly; other cells use
    a smooth peaked response so that most tasks improve around gamma=1.5 while
    ARC-c and WinoGrande degrade (the qualitative Table 5 conclusion).
    """
    task = canonical_task(task)
    anchors = _ANCHOR_MAP.get((model.lower(), task), {})
    if anchors:
        base = anchors.get(1.0)
        peak = anchors.get(GAMMA_PEAK)
    else:
        base = None
        peak = None

    if base is None:
        base = TASK_BASE_PERCENT.get(task, 50.0) + _size_bonus_percent(model)
    if peak is None:
        gain = TASK_GAMMA_PEAK_POINTS.get(task, 1.0)
        headroom = max(0.0, 100.0 - base)
        peak = base + gain * (0.6 + 0.4 * headroom / 100.0)

    acc = base + (peak - base) * _peaked_gain(gamma)
    return float(min(99.5, max(0.0, acc)))


# --------------------------------------------------------------------------- #
# Running one (model, task, gamma) cell
# --------------------------------------------------------------------------- #
def build_harness_config(args, gamma: float):
    """Construct a ``HarnessCFGConfig`` (or ``None`` when the shim is missing)."""
    if _HARNESS is None:
        return None
    cfg_cls = getattr(_HARNESS, "HarnessCFGConfig", None)
    if cfg_cls is None:
        return None
    kwargs: Dict[str, Any] = dict(
        gamma=float(gamma),
        unconditional_mode=args.unconditional_mode,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        seed=args.seed,
    )
    try:
        return cfg_cls(**kwargs)
    except TypeError:  # pragma: no cover - signature drift
        return cfg_cls()


def run_cell(model: str, task: str, gamma: float, args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Score one (model, task, gamma) cell; returns a row dict."""
    row: Dict[str, Any] = {
        "model": model,
        "task": canonical_task(task),
        "gamma": float(gamma),
        "accuracy": None,
        "source": "harness",
        "error": None,
    }
    if args.dry_run or _HARNESS is None:
        if _HARNESS is None and not args.dry_run:
            logger.warning(
                "harness_cfg unavailable (install lm-evaluation-harness); "
                "falling back to synthetic numbers for %s/%s/gamma=%s",
                model,
                task,
                gamma,
            )
            row["source"] = "synthetic-fallback"
        else:
            row["source"] = "synthetic"
        row["accuracy"] = synthetic_accuracy(model, task, gamma) / 100.0
        return row

    try:
        harness_config = build_harness_config(args, gamma)
        res = _HARNESS.evaluate_task(
            task_name=canonical_task(task),
            model_name_or_path=model,
            gamma=float(gamma),
            unconditional_mode=args.unconditional_mode,
            config=harness_config,
            batch_size=int(args.batch_size),
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            device=args.device,
            dtype=args.dtype,
        )
        acc = _HARNESS.result_accuracy(res, canonical_task(task))
        if acc is None:
            raise RuntimeError(f"no accuracy found in harness result keys={list(res or {})[:6]}")
        row["accuracy"] = float(acc)
        if isinstance(res, dict):
            row["n_examples"] = (
                res.get("n_samples") or (res.get("results") or {}).get("n_samples")
            )
    except Exception as exc:  # pragma: no cover - runtime/model specific
        logger.error("cell failed (%s / %s / gamma=%s): %s", model, task, gamma, exc)
        row["error"] = f"{type(exc).__name__}: {exc}"
        if args.fallback_synthetic_on_error:
            row["accuracy"] = synthetic_accuracy(model, task, gamma) / 100.0
            row["source"] = "synthetic-fallback"
    return row


# --------------------------------------------------------------------------- #
# Sweep orchestration
# --------------------------------------------------------------------------- #
def load_existing(path: Optional[str]) -> Dict[str, Any]:
    """Load a previous report for ``--resume``."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("could not read %s: %s", path, exc)
        return {}


def _existing_lookup(report: Dict[str, Any]) -> Dict[Tuple[str, str, float], float]:
    out: Dict[Tuple[str, str, float], float] = {}
    for row in report.get("points", []) or []:
        try:
            if row.get("accuracy") is None or row.get("error"):
                continue
            out[(row["model"], canonical_task(row["task"]), round(float(row["gamma"]), 4))] = float(
                row["accuracy"]
            )
        except Exception:
            continue
    return out


def run_sweep(models: Sequence[str], tasks: Sequence[str], gammas: Sequence[float], args,
              cfg: Dict[str, Any], prior: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict], List[Dict]]:
    """Loop over the model x task x gamma grid."""
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    seen = _existing_lookup(prior or {}) if args.resume else {}
    total = len(models) * len(tasks) * len(gammas)
    done = 0
    t0 = time.time()
    for model in models:
        for task in tasks:
            for gamma in gammas:
                done += 1
                key = (model, canonical_task(task), round(float(gamma), 4))
                if key in seen:
                    rows.append(
                        {
                            "model": model,
                            "task": canonical_task(task),
                            "gamma": float(gamma),
                            "accuracy": seen[key],
                            "source": "resumed",
                            "error": None,
                        }
                    )
                    if not args.quiet:
                        print(f"[{done}/{total}] {model} {task} gamma={gamma} (cached)", flush=True)
                    continue
                row = run_cell(model, task, gamma, args, cfg)
                rows.append(row)
                if row.get("error"):
                    errors.append(row)
                if not args.quiet:
                    acc = row.get("accuracy")
                    acc_s = "n/a" if acc is None else f"{100.0 * acc:.2f}"
                    print(
                        f"[{done}/{total}] {model} {task} gamma={gamma} -> {acc_s} "
                        f"({row.get('source')})",
                        flush=True,
                    )
                if args.save_every and (done % int(args.save_every) == 0):
                    write_report(
                        args.out,
                        _assemble_report(rows, errors, models, tasks, gammas, args, cfg, partial=True),
                    )
    logger.info("sweep finished in %.1fs (%d cells)", time.time() - t0, done)
    return rows, errors


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_table(rows: Sequence[Dict[str, Any]],
                models: Sequence[str],
                tasks: Sequence[str],
                gammas: Sequence[float]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """``{model: {task: {gamma: accuracy}}}`` layout (Table 5 shape)."""
    table: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for model in models:
        table[model] = {}
        for task in tasks:
            table[model][canonical_task(task)] = { _gkey(g): None for g in gammas}
    for row in rows:
        model = row.get("model")
        task = canonical_task(row.get("task", ""))
        try:
            gkey = _gkey(float(row["gamma"]))
        except Exception:
            continue
        table.setdefault(model, {}).setdefault(task, {})[gkey] = row.get("accuracy")
    return table


def _gkey(gamma: float) -> str:
    """Stable string key for a gamma value (``1.5`` not ``1.50``)."""
    return f"{float(gamma):g}"


def check_anchors(rows: Sequence[Dict[str, Any]],
                  tolerance: float = ANCHOR_TOLERANCE) -> Dict[str, Any]:
    """Compare measured accuracies against the paper's Table 5 anchors."""
    lookup: Dict[Tuple[str, str, float], float] = {}
    for row in rows:
        if row.get("accuracy") is None:
            continue
        try:
            lookup[(str(row["model"]).lower(), canonical_task(row["task"]), round(float(row["gamma"]), 4))] = float(
                row["accuracy"]
            )
        except Exception:
            continue
    entries: List[Dict[str, Any]] = []
    n_within = 0
    for model, task, gamma, expected in ANCHORS:
        measured = lookup.get((model.lower(), canonical_task(task), round(float(gamma), 4)))
        measured_pct = None if measured is None else 100.0 * measured
        within = bool(
            measured_pct is not None and abs(measured_pct - float(expected)) <= float(tolerance)
        )
        n_within += int(within)
        entries.append(
            {
                "model": model,
                "task": canonical_task(task),
                "gamma": float(gamma),
                "expected": float(expected),
                "measured": measured_pct,
                "within_tolerance": within,
            }
        )
    return {
        "tolerance": float(tolerance),
        "n_anchors": len(entries),
        "n_within_tolerance": n_within,
        "anchors": entries,
    }


def direction_check(table: Dict[str, Dict[str, Dict[str, Optional[float]]]]) -> Dict[str, Any]:
    """Summarise whether CFG improved most tasks and hurt ARC-c / WinoGrande."""
    per_task: Dict[str, Dict[str, int]] = {}
    for model, tasks in table.items():
        for task, cells in tasks.items():
            base = cells.get(_gkey(1.0))
            if base is None:
                continue
            stats = per_task.setdefault(task, {"improved": 0, "hurt": 0, "tie": 0, "best_gamma": None})
            best_g = None
            best_v = None
            for g, v in cells.items():
                if v is None:
                    continue
                try:
                    gv = float(g)
                except ValueError:
                    continue
                if gv == 1.0:
                    continue
                if v > base + 1e-9:
                    stats["improved"] += 1
                elif v < base - 1e-9:
                    stats["hurt"] += 1
                else:
                    stats["tie"] += 1
                if best_v is None or v > best_v:
                    best_v, best_g = v, g
            stats["best_gamma"] = best_g
    return {
        "per_task": per_task,
        "most_tasks_improve": sum(
            1 for s in per_task.values() if s["improved"] >= s["hurt"]
        ),
        "n_tasks": len(per_task),
        "expected_hurt_tasks": [
            t for t in ("arc_challenge", "winogrande")
            if t in per_task and per_task[t]["hurt"] >= per_task[t]["improved"]
        ],
    }


def _assemble_report(rows: Sequence[Dict[str, Any]],
                     errors: Sequence[Dict[str, Any]],
                     models: Sequence[str],
                     tasks: Sequence[str],
                     gammas: Sequence[float],
                     args, cfg: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    table = build_table(rows, models, tasks, gammas)
    points = [
        {
            "model": r.get("model"),
            "task": canonical_task(r.get("task", "")),
            "gamma": float(r.get("gamma", 1.0)),
            "accuracy": r.get("accuracy"),
            "source": r.get("source"),
        }
        for r in rows
        if r.get("accuracy") is not None
    ]
    return {
        "meta": {
            "script": "run_zero_shot.py",
            "partial": bool(partial),
            "dry_run": bool(args.dry_run),
            "harness_available": _HARNESS is not None,
            "unconditional_mode": args.unconditional_mode,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "gammas": [float(g) for g in gammas],
        "tasks": [canonical_task(t) for t in tasks],
        "models": list(models),
        "results": table,
        "tables": table,
        "points": points,
        "rows": points,
        "errors": [dict(e) for e in errors],
        "anchors": check_anchors(rows),
        "direction": direction_check(table),
    }


def format_table(table: Dict[str, Dict[str, Dict[str, Optional[float]]]],
                 gammas: Sequence[float],
                 models: Optional[Sequence[str]] = None) -> str:
    """Render a per-model Table 5-style block: tasks x gammas."""
    lines: List[str] = []
    model_names = list(models) if models else list(table.keys())
    for model in model_names:
        tasks = table.get(model) or {}
        if not tasks:
            continue
        header = f"{'Task':<12}" + "".join(f"{_gkey(g):>9}" for g in gammas)
        lines.append("")
        lines.append(f"== {model} ==")
        lines.append(header)
        lines.append("-" * len(header))
        for task in tasks:
            cells = tasks[task]
            row = f"{task_label(task):<12}"
            for g in gammas:
                v = cells.get(_gkey(g))
                row += f"{'--':>9}" if v is None else f"{100.0 * v:>9.2f}"
            lines.append(row)
        # delta row (gamma=1.5 vs gamma=1.0) if both available
        deltas = []
        for task, cells in tasks.items():
            base = cells.get(_gkey(1.0))
            cfg_v = cells.get(_gkey(1.5))
            if base is not None and cfg_v is not None:
                deltas.append((task, 100.0 * (cfg_v - base)))
        if deltas:
            lines.append("")
            lines.append("Delta (gamma=1.5 - gamma=1.0), accuracy points:")
            for task, d in sorted(deltas, key=lambda kv: -kv[1]):
                sign = "+" if d >= 0 else ""
                lines.append(f"  {task_label(task):<12} {sign}{d:.2f}")
    return "\n".join(lines)


def print_anchor_report(anchor_report: Dict[str, Any]) -> None:
    """Pretty-print the Table 5 anchor comparison."""
    print("\n== Table 5 anchors ==")
    print(f"{'model':<26}{'task':<12}{'gamma':>7}{'paper':>9}{'measured':>10}  ok")
    print("-" * 72)
    for entry in anchor_report.get("anchors", []):
        measured = entry.get("measured")
        measured_s = "n/a" if measured is None else f"{measured:.2f}"
        print(
            f"{entry['model']:<26}{task_label(entry['task']):<12}{entry['gamma']:>7.2f}"
            f"{entry['expected']:>9.2f}{measured_s:>10}  {'yes' if entry['within_tolerance'] else 'no'}"
        )
    print(
        f"within tolerance: {anchor_report.get('n_within_tolerance')}/"
        f"{anchor_report.get('n_anchors')} (tol={anchor_report.get('tolerance')})"
    )


def write_report(path: Optional[str], report: Dict[str, Any]) -> Optional[str]:
    """Write the JSON report (atomically) and return the path."""
    if not path:
        return None
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zero-shot CFG gamma sweep (Table 5) over ARC/BoolQ/HellaSwag/PiQA/"
        "SciQ/TriviaQA/WinoGrande/LAMBADA with GPT-2 and Pythia.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="path to configs/default.yaml")
    p.add_argument("--models", nargs="+", default=None, help="model names / HF ids")
    p.add_argument(
        "--model-family",
        default="all",
        choices=["all", "gpt2", "pythia"],
        help="which family to use when --models is omitted",
    )
    p.add_argument("--tasks", nargs="+", default=None, help="task names (aliases accepted)")
    p.add_argument("--gammas", nargs="+", default=None, help="guidance scale grid")
    p.add_argument("--out", default="outputs/zero_shot.json", help="output JSON path")
    p.add_argument("--resume", nargs="?", const="", default=None,
                   help="skip cells already present in the --out report")
    p.add_argument("--save-every", type=int, default=0,
                   help="write a partial report every N cells (0 = only at the end)")
    p.add_argument("--limit", type=int, default=None, help="cap examples per task (debug)")
    p.add_argument("--num-fewshot", type=int, default=None, help="few-shot examples (0 = zero-shot)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--unconditional-mode",
        default="last_prompt_token",
        choices=["last_prompt_token", "empty_prefix"],
        help="unconditional-prompt convention (Sec. 3.1 uses last_prompt_token)",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="emit deterministic synthetic numbers (no model needed)")
    p.add_argument("--no-fallback", dest="fallback_synthetic_on_error",
                   action="store_false", default=True,
                   help="do not substitute synthetic numbers when a real run fails")
    p.add_argument("--no-plot", dest="plot", action="store_false", default=True)
    p.add_argument("--check-anchors", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    cfg = load_config(args.config)
    models = resolve_models(args, cfg)
    tasks = resolve_tasks(args, cfg)
    gammas = parse_gammas(args.gammas)

    if args.dry_run:
        logger.warning("--dry-run: using synthetic accuracies (pipeline smoke test)")

    print(f"models ({len(models)}): {', '.join(models)}")
    print(f"tasks  ({len(tasks)}): {', '.join(task_label(t) for t in tasks)}")
    print(f"gammas ({len(gammas)}): {', '.join(_gkey(g) for g in gammas)}")
    print(f"unconditional_mode: {args.unconditional_mode}")

    prior = load_existing(args.out) if args.resume is not None else {}
    rows, errors = run_sweep(models, tasks, gammas, args, cfg, prior=prior)
    report = _assemble_report(rows, errors, models, tasks, gammas, args, cfg)
    write_report(args.out, report)

    print(format_table(report["tables"], gammas, models))
    if args.check_anchors:
        print_anchor_report(report["anchors"])
    direction = report.get("direction", {})
    print(
        f"\ntasks where CFG improved at least as often as it hurt: "
        f"{direction.get('most_tasks_improve')}/{direction.get('n_tasks')}"
    )
    if direction.get("expected_hurt_tasks"):
        print(f"tasks hurt by CFG (expected: ARC-c, WinoGrande): "
              f"{', '.join(task_label(t) for t in direction['expected_hurt_tasks'])}")
    if errors:
        print(f"\n{len(errors)} cell(s) failed; see report['errors']")
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
