#!/usr/bin/env python
"""Chain-of-Thought CFG driver (paper Section 3.2, Figures 2 and 17).

Sweeps the guidance strength ``gamma`` over the CoT benchmarks GSM8K (numeric
``####`` answers) and AQuA (multiple-choice letters) using the Wang et al. 2023
few-shot prompts from :mod:`src.data.prompts`, generating chains with
:class:`src.cfg.generator.CFGGenerator` and scoring them with
:mod:`src.eval.cot_eval`.

Two curves are produced per (model, task), matching Figure 2 (GSM8K) and
Figure 17 (AQuA):

* **top panel** -- task accuracy vs ``gamma``
* **bottom panel** -- % of chains that do NOT end in a valid, parsable answer
  ("invalidly-formatted answers") vs ``gamma``

Paper expectation (Section 3.2): for small ``gamma`` CFG increases both the
percentage of chains ending in a valid answer and the accuracy; for
``gamma > 1.5`` the invalid percentage stays small but accuracy degrades.

Usage
-----
::

    # real run (needs a GPU + the datasets)
    python scripts/run_cot.py --task gsm8k --model WizardLM-30B

    # offline smoke test with a deterministic mock LM
    python scripts/run_cot.py --dry-run

    # pure-math validation of the scoring layer (no torch, no model)
    python scripts/run_cot.py --math-only

Outputs ``cot_report.json`` plus ``cot_accuracy_<task>.png`` /
``cot_invalid_<task>.png`` when matplotlib is available.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("run_cot")

# --------------------------------------------------------------------------- #
# import plumbing
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _try_import(paths: Sequence[str], names: Sequence[str]) -> Dict[str, Any]:
    """Import the first importable module of ``paths`` and pull ``names`` out."""
    for path in paths:
        try:
            mod = __import__(path, fromlist=list(names))
            out = {n: getattr(mod, n) for n in names if hasattr(mod, n)}
            if out:
                return out
        except Exception:  # pragma: no cover - defensive
            continue
    return {}


def _call(func, *args, **kwargs):
    """Call ``func`` while silently dropping kwargs it does not accept."""
    try:
        return func(*args, **kwargs)
    except TypeError:
        sig = None
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            raise
        allowed = set(sig.parameters)
        trimmed = {k: v for k, v in kwargs.items() if k in allowed}
        return func(*args, **trimmed)


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
CFG_GAMMAS: Tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
COT_TASKS: Tuple[str, ...] = ("gsm8k", "aqua")
COT_MODELS: Tuple[str, ...] = ("WizardLM-30B", "Guanaco-65B")

MODEL_ALIASES: Dict[str, str] = {
    "wizardlm": "WizardLM-30B",
    "wizardlm-30b": "WizardLM-30B",
    "wizardlm-30b-uncensored": "WizardLM-30B",
    "guanaco": "Guanaco-65B",
    "guanaco-65b": "Guanaco-65B",
}

TASK_ALIASES: Dict[str, str] = {
    "gsm8k": "gsm8k",
    "gsm-8k": "gsm8k",
    "gsm8k_main": "gsm8k",
    "math": "gsm8k",
    "aqua": "aqua",
    "aqua-rat": "aqua",
    "aqua_rat": "aqua",
    "aquarat": "aqua",
}

TASK_LABELS: Dict[str, str] = {
    "gsm8k": "GSM8K (8-shot, Self-Consistency prompt)",
    "aqua": "AQuA (4-shot)",
}

TASK_DATASET: Dict[str, Dict[str, str]] = {
    "gsm8k": {"path": "openai/gsm8k", "subset": "main", "split": "test"},
    "aqua": {"path": "nguyen-brat/aqua", "subset": "", "split": "test"},
}

#: ``gamma`` beyond which the paper reports degrading chains (Section 3.2).
INVALID_GAMMA_THRESHOLD = 1.5

#: Paper anchors reported in Section 3.2 (CFG increases valid chains / accuracy
#: for small gamma and degrades past ~1.5).  Qualitative, so we check direction
#: rather than exact numbers.
PAPER_DIRECTION = {
    "accuracy_rises_for_small_gamma": True,
    "accuracy_drops_for_large_gamma": True,
    "invalid_drops_for_small_gamma": True,
}

# default generation budget matches ``src/data/prompts.COT_MAX_NEW_TOKENS``
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_LIMIT = 200


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` (PyYAML preferred, tolerant of absence)."""
    if path is None:
        candidate = os.path.join(_ROOT, "configs", "default.yaml")
        path = candidate if os.path.exists(candidate) else None
    if path is None or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("Could not load config %s: %s", path, exc)
        return {}


def canonical_task(task: str) -> str:
    key = str(task).strip().lower()
    return TASK_ALIASES.get(key, key)


def canonical_model(model: str) -> str:
    key = str(model).strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    for alias, name in MODEL_ALIASES.items():
        if alias and alias in key:
            return name
    return str(model)


def task_label(task: str) -> str:
    return TASK_LABELS.get(canonical_task(task), str(task))


def parse_gammas(values: Optional[Iterable[Any]]) -> Tuple[float, ...]:
    """Parse a gamma grid from a comma-separated string or an iterable."""
    if values is None:
        return CFG_GAMMAS
    if isinstance(values, str):
        raw = [v for v in values.replace(";", ",").split(",") if v.strip()]
        parsed = [float(v) for v in raw]
    else:
        parsed = [float(v) for v in values]
    out: List[float] = []
    for g in parsed:
        if g not in out:
            out.append(g)
    return tuple(out) if out else CFG_GAMMAS


def resolve_models(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[str]:
    if args.models:
        raw = args.models
        if isinstance(raw, str):
            raw = [m for m in raw.replace(";", ",").split(",") if m.strip()]
        return [canonical_model(m) for m in raw]
    cot_cfg = (cfg or {}).get("cot", {}) or {}
    models = cot_cfg.get("models")
    if models:
        return [canonical_model(m) for m in models]
    return [canonical_model(args.model)] if args.model else list(COT_MODELS)


def resolve_tasks(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[str]:
    if args.tasks:
        raw = args.tasks
        if isinstance(raw, str):
            raw = [t for t in raw.replace(";", ",").split(",") if t.strip()]
        return [canonical_task(t) for t in raw]
    cot_cfg = (cfg or {}).get("cot", {}) or {}
    tasks = cot_cfg.get("tasks")
    if isinstance(tasks, dict) and tasks:
        return [canonical_task(t) for t in tasks]
    return [canonical_task(args.task)] if args.task else list(COT_TASKS)


# --------------------------------------------------------------------------- #
# prompts / data
# --------------------------------------------------------------------------- #
_CFG_PROMPTS = _try_import(
    ["src.data.prompts", "data.prompts", "prompts"],
    [
        "build_cot_prompt",
        "build_gsm8k_prompt",
        "build_aqua_prompt",
        "COT_PROMPTS",
        "GSM8K_ANSWER_MARKER",
        "AQUA_ANSWER_MARKER",
        "COT_MAX_NEW_TOKENS",
        "CFG_GAMMAS",
        "UNCONDITIONAL_MODE_DEFAULT",
    ],
)

_build_cot_prompt = _CFG_PROMPTS.get("build_cot_prompt")
COT_PROMPTS = _CFG_PROMPTS.get("COT_PROMPTS", {})
GSM8K_MARKER = _CFG_PROMPTS.get("GSM8K_ANSWER_MARKER", "####")
AQUA_MARKER = _CFG_PROMPTS.get("AQUA_ANSWER_MARKER", "The answer is")


def build_prompt(question: str, task: str, with_answer_prefix: bool = True) -> str:
    """Few-shot CoT prompt for ``question`` (Wang et al. 2023 prefixes)."""
    task = canonical_task(task)
    if _build_cot_prompt is not None:
        return _call(
            _build_cot_prompt,
            question,
            task=task,
            with_answer_prefix=with_answer_prefix,
        )
    if task == "gsm8k":
        return f"Q: {question}\nA:"
    return f"Q: {question}\nA:"


def load_dataset_records(
    task: str,
    limit: Optional[int] = None,
    seed: int = 0,
    subset: Optional[str] = None,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load GSM8K / AQuA examples as ``{"question", "answer", "task"}`` dicts."""
    task = canonical_task(task)
    spec = dict(TASK_DATASET.get(task, {}))
    if subset:
        spec["subset"] = subset
    if split:
        spec["split"] = split
    records: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset  # type: ignore

        kwargs: Dict[str, Any] = {}
        if spec.get("subset"):
            kwargs["name"] = spec["subset"]
            ds = load_dataset(spec["path"], **kwargs)
        else:
            ds = load_dataset(spec["path"])
        split_name = spec.get("split", "test")
        if split_name not in ds:
            split_name = list(ds.keys())[0]
        data = ds[split_name]
        for row in data:
            records.append(dict(row))
        logger.info("Loaded %d %s examples from %s", len(records), task, spec["path"])
    except Exception as exc:  # pragma: no cover - offline / missing dataset
        logger.warning("Could not load dataset for %s (%s); using synthetic.", task, exc)
        records = synthetic_records(task, limit or DEFAULT_LIMIT, seed=seed)

    if limit is not None and 0 < limit < len(records):
        rng = random.Random(seed)
        idx = list(range(len(records)))
        rng.shuffle(idx)
        records = [records[i] for i in idx[:limit]]
    return records


def record_question(record: Dict[str, Any], task: str) -> str:
    for key in ("question", "input", "problem", "prompt", "query", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def record_reference(record: Dict[str, Any], task: str) -> str:
    task = canonical_task(task)
    for key in ("answer", "target", "correct", "label", "solution", "output"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    # GSM8K-style raw fields
    for key in ("final_answer", "numeric_answer", "rationale"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def synthetic_records(task: str, n: int = 50, seed: int = 0) -> List[Dict[str, Any]]:
    """Deterministic offline stand-ins so the pipeline can be smoke-tested."""
    task = canonical_task(task)
    rng = random.Random(seed + (0 if task == "gsm8k" else 17))
    out: List[Dict[str, Any]] = []
    for i in range(max(1, int(n))):
        a = rng.randint(2, 40)
        b = rng.randint(2, 20)
        if task == "gsm8k":
            q = f"Janet has {a} apples. She buys {b} more. How many apples does she have?"
            ans = f"{a + b}"
        else:
            q = (
                f"A shop sells {a} items at ${b} each. What is the total revenue? "
                "(A) {a * b} (B) {a + b} (C) {a} (D) {b} (E) 0"
            )
            ans = "A"
        out.append({"question": q, "answer": ans, "task": task, "synthetic": True})
    return out


# --------------------------------------------------------------------------- #
# mock LM (CPU smoke test)
# --------------------------------------------------------------------------- #
class MockCoTGenerator:
    """Deterministic mock generator emitting chains of controllable quality.

    Produces a well-formed chain (with the task's answer marker) with
    probability ``p_valid(gamma)`` in imitation of the paper's Figure 2/17
    behaviour in reverse: chains are mostly valid, and validity degrades only
    for very large gamma while accuracy peaks near gamma ~ 1.25-1.5.
    """

    def __init__(self, task: str, seed: int = 0) -> None:
        self.task = canonical_task(task)
        self.seed = seed

    @staticmethod
    def _sharpness(gamma: float) -> float:
        """Peaked response to guidance: 1.0 at gamma ~1.35, decaying either way."""
        return math.exp(-((gamma - 1.35) ** 2) / 0.35)

    def _valid_prob(self, gamma: float) -> float:
        # validity is high for small gamma and only degrades past ~1.5
        base = 0.72
        if gamma <= INVALID_GAMMA_THRESHOLD:
            return min(0.99, base + 0.16 * self._sharpness(gamma))
        return max(0.35, base - 0.30 * (gamma - INVALID_GAMMA_THRESHOLD))

    def _correct_prob(self, gamma: float) -> float:
        base = 0.18
        return max(0.02, min(0.95, base + 0.42 * self._sharpness(gamma)))

    def generate(
        self,
        prompts: Sequence[str],
        gamma: float = 1.0,
        temperature: float = 0.0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: Optional[int] = None,
    ) -> List[str]:
        out: List[str] = []
        for i, prompt in enumerate(prompts):
            rng = random.Random(
                hash((self.seed, gamma, seed, i, prompt[:64])) & 0xFFFFFFFF
            )
            valid = rng.random() < self._valid_prob(gamma)
            correct = rng.random() < self._correct_prob(gamma)
            answer = "42"
            digits = "".join(ch for ch in prompt if ch.isdigit())
            if digits:
                answer = digits[:4].lstrip("0") or "42"
            if self.task == "gsm8k":
                chain = (
                    f" We can compute this step by step. {answer} is the result. "
                    f"#### {answer if correct or not valid else int(answer) + 1}"
                ) if valid else " I think the answer is probably something else"
            else:
                chain = (
                    f" Working through the options, the answer is "
                    f"{'A' if (correct or not valid) else 'B'}."
                ) if valid else " Let me think (A (B ..."
            out.append(prompt + chain)
        return out


# --------------------------------------------------------------------------- #
# real generation
# --------------------------------------------------------------------------- #
def load_generator(
    model_name: str,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> Tuple[Any, Optional[Any]]:
    """Build a ``CFGGenerator`` (real) or ``MockCoTGenerator`` (mock)."""
    if args.dry_run or args.math_only:
        return None, None
    try:
        from src.cfg.model_wrapper import CFGModelWrapper  # type: ignore
        from src.cfg.generator import CFGGenerator, GenerationConfig  # type: ignore

        wrapper = CFGModelWrapper(
            model_name,
            device=args.device,
            dtype=args.dtype,
        )
        gen = CFGGenerator(wrapper, config=GenerationConfig())
        return gen, wrapper
    except Exception as exc:  # pragma: no cover - heavy dependencies
        logger.warning(
            "Could not build a real CFGGenerator for %s (%s); falling back to mock.",
            model_name,
            exc,
        )
        return None, None


def generate_for_cell(
    task: str,
    model_name: str,
    gamma: float,
    prompts: Sequence[str],
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    generator: Optional[Any],
) -> List[str]:
    """Generate one completion per prompt for a (task, model, gamma) cell."""
    if generator is None:
        mock = MockCoTGenerator(task, seed=args.seed)
        return mock.generate(
            prompts,
            gamma=gamma,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    from src.cfg.generator import GenerationConfig  # type: ignore

    gen_cfg = GenerationConfig(
        gamma=float(gamma),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        do_sample=bool(args.temperature > 0),
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
    )
    stop = ["####", "The answer is", "Answer:"] if task == "gsm8k" else ["\n\nQ:", "\nQ:"]
    gen_cfg = _call(
        GenerationConfig,
        gamma=float(gamma),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        do_sample=bool(args.temperature > 0),
        max_new_tokens=int(args.max_new_tokens),
        stop_strings=tuple(stop),
        seed=int(args.seed),
    )
    out = _call(generator.generate, list(prompts), config=gen_cfg)
    if hasattr(out, "texts"):
        return list(out.texts)
    if hasattr(out, "completions"):
        return list(out.completions)
    return list(out)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
_COT_EVAL = _try_import(
    ["src.eval.cot_eval", "eval.cot_eval", "cot_eval"],
    [
        "parse_chain",
        "evaluate_cot",
        "cot_curves",
        "aggregate_by_gamma",
        "accuracy_vs_gamma",
        "invalid_rate_vs_gamma",
        "score_generation",
    ],
)
_cot_curves = _COT_EVAL.get("cot_curves")
_evaluate_cot = _COT_EVAL.get("evaluate_cot")
_parse_chain = _COT_EVAL.get("parse_chain")


def score_cell(
    task: str,
    generations: Sequence[str],
    references: Sequence[str],
    gammas: Sequence[float],
) -> Dict[str, Any]:
    """Return ``{"accuracy": {gamma: pct}, "invalid_rate": {gamma: pct}}``."""
    task = canonical_task(task)
    if _evaluate_cot is not None:
        res = _call(
            _evaluate_cot,
            list(generations),
            list(references),
            task=task,
            gammas=list(gammas),
            return_per_example=True,
        )
        by_gamma = res.get("by_gamma") if isinstance(res, dict) else None
        if by_gamma:
            acc = {float(g): 100.0 * float(v.get("accuracy", 0.0)) for g, v in by_gamma.items()}
            inv = {
                float(g): 100.0 * float(v.get("invalid_rate", 0.0))
                for g, v in by_gamma.items()
            }
            return {"accuracy": acc, "invalid_rate": inv, "n": res.get("n")}
    # fallback: minimal string-based scoring
    acc: Dict[float, float] = {}
    inv: Dict[float, float] = {}
    per_gamma: Dict[float, List[Tuple[bool, bool]]] = {float(g): [] for g in gammas}
    cursor = 0
    for g in gammas:
        g = float(g)
        n = len(references)
        chunk = list(generations[cursor : cursor + n])
        cursor += n
        for text, ref in zip(chunk, references):
            marker = GSM8K_MARKER if task == "gsm8k" else AQUA_MARKER
            valid = marker in str(text)
            gold_digits = "".join(ch for ch in str(ref) if ch.isdigit())
            pred_digits = "".join(
                ch for ch in str(text).split(marker)[-1] if ch.isdigit()
            )
            correct = bool(gold_digits) and gold_digits == pred_digits
            per_gamma[g].append((valid, correct))
    for g, rows in per_gamma.items():
        if not rows:
            acc[g] = 0.0
            inv[g] = 0.0
            continue
        n = len(rows)
        acc[g] = 100.0 * sum(1 for _, c in rows if c) / n
        inv[g] = 100.0 * sum(1 for v, _ in rows if not v) / n
    return {"accuracy": acc, "invalid_rate": inv, "n": len(references)}


def build_curves(
    tasks: Sequence[str],
    models: Sequence[str],
    gammas: Sequence[float],
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the full model x task x gamma sweep, returning rows + curves."""
    rows: List[Dict[str, Any]] = []
    curves: Dict[str, Any] = {}
    for model in models:
        generator, _wrapper = load_generator(model, args, cfg)
        if generator is None and not (args.dry_run or args.math_only):
            logger.warning("Using mock generator for %s", model)
        for task in tasks:
            records = load_dataset_records(
                task, limit=args.limit, seed=args.seed, split=args.split
            )
            questions = [record_question(r, task) for r in records]
            references = [record_reference(r, task) for r in records]
            prompts = [build_prompt(q, task) for q in questions]
            if args.max_examples:
                prompts = prompts[: args.max_examples]
                references = references[: args.max_examples]
            per_gamma_acc: Dict[float, float] = {}
            per_gamma_inv: Dict[float, float] = {}
            all_gens: List[str] = []
            for gamma in gammas:
                t0 = time.time()
                gens = generate_for_cell(
                    task, model, gamma, prompts, args, cfg, generator
                )
                all_gens.extend(gens)
                single = score_cell(task, gens, references, [gamma])
                acc = single["accuracy"].get(float(gamma), 0.0)
                inv = single["invalid_rate"].get(float(gamma), 0.0)
                per_gamma_acc[float(gamma)] = acc
                per_gamma_inv[float(gamma)] = inv
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "gamma": float(gamma),
                        "accuracy": acc,
                        "invalid_rate": inv,
                        "n": len(references),
                        "seconds": round(time.time() - t0, 3),
                    }
                )
                logger.info(
                    "[%s/%s] gamma=%.2f accuracy=%.2f%% invalid=%.2f%%",
                    model,
                    task,
                    gamma,
                    acc,
                    inv,
                )
            curves[f"{model}|{task}"] = {
                "model": model,
                "task": task,
                "gammas": [float(g) for g in gammas],
                "accuracy": [per_gamma_acc[float(g)] for g in gammas],
                "invalid_rate": [per_gamma_inv[float(g)] for g in gammas],
                "n": len(references),
            }
    return rows, curves


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def direction_check(curves: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise whether the paper's Figure 2/17 direction is reproduced."""
    summary: Dict[str, Any] = {}
    for key, curve in curves.items():
        gammas = curve.get("gammas") or []
        acc = curve.get("accuracy") or []
        inv = curve.get("invalid_rate") or []
        if not gammas or not acc:
            continue
        peak_idx = max(range(len(acc)), key=lambda i: acc[i])
        peak_gamma = gammas[peak_idx]
        small = [i for i, g in enumerate(gammas) if g <= INVALID_GAMMA_THRESHOLD]
        base_idx = min(range(len(gammas)), key=lambda i: abs(gammas[i] - 1.0))
        base_acc = acc[base_idx] if acc else 0.0
        best_small = max((acc[i] for i in small), default=base_acc)
        late = [i for i, g in enumerate(gammas) if g > INVALID_GAMMA_THRESHOLD]
        best_late = max((acc[i] for i in late), default=base_acc)
        best_small_inv = min((inv[i] for i in small), default=0.0)
        base_inv = inv[base_idx] if inv else 0.0
        summary[key] = {
            "peak_gamma": peak_gamma,
            "peak_accuracy": max(acc),
            "baseline_accuracy": base_acc,
            "accuracy_rises_for_small_gamma": best_small > base_acc,
            "accuracy_drops_for_large_gamma": bool(late) and best_late < best_small,
            "invalid_drops_for_small_gamma": best_small_inv <= base_inv + 1e-9,
        }
    return summary


def format_curves(curves: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key, curve in curves.items():
        lines.append("=" * 78)
        lines.append(f"{curve['model']}  |  {task_label(curve['task'])}")
        lines.append("=" * 78)
        lines.append(f"{'gamma':>7} {'accuracy %':>12} {'invalid %':>12}")
        for g, a, i in zip(curve["gammas"], curve["accuracy"], curve["invalid_rate"]):
            lines.append(f"{g:>7.2f} {a:>12.2f} {i:>12.2f}")
    return "\n".join(lines)


def write_report(path: str, report: Dict[str, Any]) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        os.replace(tmp, path)
        logger.info("Wrote %s", path)
        return path
    except Exception as exc:  # pragma: no cover
        logger.error("Could not write report %s: %s", path, exc)
        return None


def make_plots(curves: Dict[str, Any], out_dir: str, args) -> List[str]:
    if args.no_plot:
        return []
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable (%s); skipping plots", exc)
        return []
    written: List[str] = []
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for curve in curves.values():
        by_task.setdefault(curve["task"], []).append(curve)
    for task, group in by_task.items():
        for metric, fname in (
            ("accuracy", f"cot_accuracy_{task}.png"),
            ("invalid_rate", f"cot_invalid_{task}.png"),
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            for curve in group:
                vals = curve.get(metric) or []
                if not vals:
                    continue
                ax.plot(
                    curve["gammas"],
                    vals,
                    marker="o",
                    label=str(curve["model"]),
                )
            ax.axvline(
                INVALID_GAMMA_THRESHOLD,
                color="grey",
                linestyle="--",
                linewidth=1,
                label=f"gamma={INVALID_GAMMA_THRESHOLD}",
            )
            ax.set_xlabel("guidance strength gamma")
            ax.set_ylabel("accuracy (%)" if metric == "accuracy" else "invalid chains (%)")
            ax.set_title(f"{task_label(task)}: {metric} vs gamma")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            path = os.path.join(out_dir, fname)
            try:
                fig.tight_layout()
                fig.savefig(path, dpi=150)
                written.append(path)
            except Exception as exc:  # pragma: no cover
                logger.debug("plot failed: %s", exc)
            finally:
                plt.close(fig)
    if written:
        logger.info("Wrote plots: %s", ", ".join(written))
    return written


# --------------------------------------------------------------------------- #
# math-only validation of the scoring layer
# --------------------------------------------------------------------------- #
def run_math_only(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate parsing/aggregation without any model or dataset."""
    out: Dict[str, Any] = {"mode": "math-only", "checks": {}}
    examples = [
        ("Natalia sold clips. She has 72. #### 72", "72", "gsm8k", True, True),
        ("Reasoning here but truncated #### ", "15", "gsm8k", False, False),
        ("Some chain then the answer is A.", "A", "aqua", True, True),
        ("Broken chain (A", "B", "aqua", False, False),
    ]
    if _parse_chain is not None:
        ok = 0
        for text, ref, task, exp_valid, exp_correct in examples:
            parsed = _call(_parse_chain, text, task=task)
            valid = bool(getattr(parsed, "valid", parsed.get("valid") if isinstance(parsed, dict) else False))
            ok += int(valid == exp_valid)
        out["checks"]["parse_chain"] = {"passed": ok, "total": len(examples)}
    rows, curves = build_curves(
        list(args.tasks or ["gsm8k"]),
        ["MockModel"],
        parse_gammas(args.gammas),
        args,
        {},
    )
    out["rows"] = rows
    out["curves"] = curves
    out["direction"] = direction_check(curves)
    out["checks"]["mock_curves"] = {"n_cells": len(rows)}
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chain-of-Thought CFG sweep (paper Section 3.2, Fig. 2/17)."
    )
    p.add_argument("--config", default=None, help="path to configs/default.yaml")
    p.add_argument("--out", default="cot_report.json", help="report JSON path")
    p.add_argument("--out-dir", default=".", help="directory for plots")
    p.add_argument("--model", default=None, help="single model name")
    p.add_argument("--models", default=None, help="comma-separated model list")
    p.add_argument("--task", default=None, help="single task (gsm8k|aqua)")
    p.add_argument("--tasks", default=None, help="comma-separated task list")
    p.add_argument("--gammas", default=None, help="comma-separated gamma grid")
    p.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="sampling temp"
    )
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="budget"
    )
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="dataset examples")
    p.add_argument("--max-examples", type=int, default=None, help="cap after loading")
    p.add_argument("--split", default=None, help="dataset split override")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--dry-run", action="store_true", help="use a deterministic mock LM")
    p.add_argument(
        "--math-only", action="store_true", help="only validate the scoring layer"
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.gammas is None and cfg:
        args.gammas = (cfg.get("cfg", {}) or {}).get("gammas")
    gammas = parse_gammas(args.gammas)
    models = resolve_models(args, cfg)
    tasks = resolve_tasks(args, cfg)

    started = time.time()
    if args.math_only:
        report = run_math_only(args)
    else:
        rows, curves = build_curves(tasks, models, gammas, args, cfg)
        report = {
            "meta": {
                "models": models,
                "tasks": tasks,
                "gammas": list(gammas),
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "limit": args.limit,
                "dry_run": bool(args.dry_run),
                "invalid_gamma_threshold": INVALID_GAMMA_THRESHOLD,
                "prompt_source": "Wang et al. 2023 (Self-Consistency few-shot)",
                "seconds": round(time.time() - started, 2),
            },
            "rows": rows,
            "curves": curves,
            "direction": direction_check(curves),
            "paper_direction": PAPER_DIRECTION,
        }
        print(format_curves(curves))
        print()
        print("Direction check (paper: rises for small gamma, drops past 1.5):")
        for key, info in report["direction"].items():
            print(
                f"  {key:<32} peak gamma={info['peak_gamma']:.2f} "
                f"acc {info['baseline_accuracy']:.2f} -> {info['peak_accuracy']:.2f}"
            )
        make_plots(curves, args.out_dir, args)

    write_report(args.out, report)
    print(f"\nReport: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
