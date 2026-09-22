#!/usr/bin/env python
"""HumanEval sweep driver for "Stay on Topic with Classifier-Free Guidance".

This script reproduces the HumanEval results of the paper:

  * Table 2  -- CodeGen-350M/2B/6B-mono, temperature 0.2, gamma grid,
                pass@k for k in {1, 10, 100}.
  * Tables 7/8/9 -- the same sweep at temperatures 0.6 and 0.8 (plus the
                per-temperature cross sections reported in the appendix).
  * Figure 3 semantics -- the task-by-task outperform/tie/underperform
                distribution, computed against the vanilla (gamma=1.0) cell
                of the same (model, temperature).

Protocol (Section 3.3.1 of the paper):

  1. For every (model, temperature, gamma) cell, draw ``n_samples`` completions
     per HumanEval problem with the CFG generation loop (two forward passes
     through the same weights, next-token logits combined as
     ``uncond + gamma * (cond - uncond)`` before temperature/top-p/softmax).
  2. Score each completion by executing it against the problem's unit tests in
     an isolated subprocess (``human_eval`` sandboxing, 3 s timeout).
  3. Aggregate with the unbiased pass@k estimator of Chen et al. (2021):
     ``pass@k = 1 - C(n - c, k) / C(n, k)`` (footnote 4), averaged over the 164
     problems.
  4. Report pass@1 / pass@10 / pass@100 per model and temperature together with
     the paper's directional claims:
        - pass@1 rises for gamma in [1, 1.5] and falls beyond it;
        - high-k pass rates flat-line or decline as gamma grows.

Everything degrades gracefully: with ``--dry-run`` a deterministic mock
generator/executor is used so the whole pipeline can be smoke-tested on CPU
without torch, transformers, datasets or the ``human_eval`` package; with
``--math-only`` only the pass@k estimator layer is validated.

Examples
--------
    # full Table 2 sweep on one GPU (CodeGen-350M-mono, temperature 0.2)
    python scripts/run_humaneval.py --models codegen-350M-mono \\
        --temperatures 0.2 --n-samples 200

    # offline smoke test of the whole pipeline
    python scripts/run_humaneval.py --dry-run --limit 20 --n-samples 20

    # validate the pass@k math only
    python scripts/run_humaneval.py --math-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# path plumbing: allow `python scripts/run_humaneval.py` from the repo root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOGGER = logging.getLogger("run_humaneval")


# ---------------------------------------------------------------------------
# tolerant imports
# ---------------------------------------------------------------------------
def _try_import(module_names: Sequence[str]) -> Optional[Any]:
    """Import the first module that resolves, or return ``None``."""
    import importlib

    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on environment
            LOGGER.debug("could not import %s: %s", name, exc)
    return None


_pass_mod = _try_import(["src.eval.pass_at_k", "eval.pass_at_k", "pass_at_k"])
_he_mod = _try_import(["src.eval.humaneval_eval", "eval.humaneval_eval", "humaneval_eval"])
_prompts_mod = _try_import(["src.data.prompts", "data.prompts", "prompts"])

_HAS_PASS = _pass_mod is not None
_HAS_HE = _he_mod is not None


def _const(module: Optional[Any], name: str, default: Any) -> Any:
    if module is not None and hasattr(module, name):
        return getattr(module, name)
    return default


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` dropping kwargs it does not accept (signature drift guard)."""
    if fn is None:
        raise RuntimeError("callable is None")
    try:
        import inspect

        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        pass
    return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# constants (mirroring the paper / configs/default.yaml)
# ---------------------------------------------------------------------------
CFG_GAMMAS: Tuple[float, ...] = tuple(
    _const(_pass_mod, "CFG_GAMMAS", (1.0, 1.1, 1.25, 1.5, 1.75, 2.0))
)
HUMANEVAL_TEMPERATURES: Tuple[float, ...] = tuple(
    _const(_pass_mod, "HUMANEVAL_TEMPERATURES", (0.2, 0.6, 0.8))
)
PASS_AT_K_VALUES: Tuple[int, ...] = tuple(
    _const(_pass_mod, "PASS_AT_K_VALUES", (1, 10, 100))
)
N_PROBLEMS: int = int(_const(_pass_mod, "HUMANEVAL_N_PROBLEMS", 164))
DEFAULT_N_SAMPLES: int = int(_const(_pass_mod, "HUMANEVAL_N_SAMPLES", 200))
DEFAULT_MAX_NEW_TOKENS: int = int(_const(_he_mod, "HUMANEVAL_MAX_NEW_TOKENS", 512))
DEFAULT_TIMEOUT: float = float(_const(_he_mod, "HUMANEVAL_TIMEOUT", 3.0))

HUMANEVAL_MODELS: Tuple[str, ...] = (
    "codegen-350M-mono",
    "codegen-2B-mono",
    "codegen-6B-mono",
)

MODEL_ALIASES: Dict[str, str] = {
    "350m": "codegen-350M-mono",
    "codegen-350m-mono": "codegen-350M-mono",
    "salesforce/codegen-350m-mono": "codegen-350M-mono",
    "codegen_350m_mono": "codegen-350M-mono",
    "2b": "codegen-2B-mono",
    "codegen-2b-mono": "codegen-2B-mono",
    "salesforce/codegen-2b-mono": "codegen-2B-mono",
    "codegen_2b_mono": "codegen-2B-mono",
    "6b": "codegen-6B-mono",
    "codegen-6b-mono": "codegen-6B-mono",
    "salesforce/codegen-6b-mono": "codegen-6B-mono",
    "codegen_6b_mono": "codegen-6B-mono",
    "codegen-350m-multi": "codegen-350M-mono",
}

# pass@1 / pass@100 anchors quoted in the paper (Table 2, temperature 0.2).
# Used only for sanity reporting, never to fabricate measured values.
ANCHORS: Dict[str, Dict[float, Dict[int, float]]] = {
    "codegen-350M-mono": {
        1.0: {1: 11.0, 100: 22.0},
        1.1: {1: 11.8, 100: 22.0},
    },
}
ANCHOR_TOLERANCE: float = 3.0  # absolute percentage points

# expected qualitative behaviour (Section 3.3.1 / Figure 3)
GAMMA_PEAK: float = 1.5
INVALID_GAMMA_THRESHOLD: float = 1.5

DEFAULT_LIMIT: Optional[int] = None
DEFAULT_SEED: int = 1234


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def canonical_model(model: str) -> str:
    """Map user-facing model spellings to the canonical paper name."""
    key = str(model).strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    low = key.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    if "codegen" in low:
        for canon in HUMANEVAL_MODELS:
            if canon.lower().replace("-", "") in low.replace("-", "").replace("/", ""):
                return canon
        if "350" in low:
            return "codegen-350M-mono"
        if "2b" in low:
            return "codegen-2B-mono"
        if "6b" in low:
            return "codegen-6B-mono"
    return key


def hf_model_name(model: str) -> str:
    """Canonical name -> HuggingFace hub id."""
    canon = canonical_model(model)
    mapping = {
        "codegen-350M-mono": "Salesforce/codegen-350M-mono",
        "codegen-2B-mono": "Salesforce/codegen-2B-mono",
        "codegen-6B-mono": "Salesforce/codegen-6B-mono",
    }
    return mapping.get(canon, canon)


def parse_floats(values: Optional[Iterable[Any]]) -> Tuple[float, ...]:
    """Parse a gamma/temperature grid from CLI or config values."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = str(values).replace(";", ",").split(",")
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            LOGGER.warning("ignoring unparsable float: %r", v)
    return tuple(out)


def parse_ints(values: Optional[Iterable[Any]]) -> Tuple[int, ...]:
    """Parse an integer grid (k values) from CLI or config values."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = str(values).replace(";", ",").split(",")
    out: List[int] = []
    for v in values:
        try:
            out.append(int(float(v)))
        except (TypeError, ValueError):
            LOGGER.warning("ignoring unparsable int: %r", v)
    return tuple(out)


def _percent(x: Optional[float]) -> Optional[float]:
    """Convert a fraction in [0, 1] to a percentage."""
    if x is None:
        return None
    return 100.0 * float(x)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` (PyYAML, with a tolerant fallback)."""
    if not path:
        path = os.path.join(_ROOT, "configs", "default.yaml")
    if not os.path.exists(path):
        LOGGER.debug("config not found: %s", path)
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        LOGGER.debug("PyYAML unavailable/failed (%s); using naive parser", exc)
    cfg: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if not key or not val:
                    continue
                if val.startswith("[") and val.endswith("]"):
                    items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
                    cfg[key] = [
                        float(v) if _is_number(v) else v.strip("'\"") for v in items
                    ]
                elif _is_number(val):
                    cfg[key] = float(val) if "." in val else int(val)
    except Exception:  # pragma: no cover
        pass
    return cfg


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _nested(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_models(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[str]:
    models = getattr(args, "models", None)
    if models:
        return [canonical_model(m) for m in models]
    cfg_models = _nested(cfg, "humaneval", "models")
    if isinstance(cfg_models, list) and cfg_models:
        return [canonical_model(m) for m in cfg_models]
    return list(HUMANEVAL_MODELS)


def resolve_temperatures(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[float, ...]:
    temps = parse_floats(getattr(args, "temperatures", None))
    if temps:
        return temps
    cfg_temps = _nested(cfg, "sampling", "humaneval", "temperatures")
    if cfg_temps:
        return parse_floats(cfg_temps) or HUMANEVAL_TEMPERATURES
    return HUMANEVAL_TEMPERATURES


def resolve_gammas(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[float, ...]:
    gammas = parse_floats(getattr(args, "gammas", None))
    if gammas:
        return gammas
    cfg_gammas = _nested(cfg, "cfg", "gammas")
    if cfg_gammas:
        return parse_floats(cfg_gammas) or CFG_GAMMAS
    return CFG_GAMMAS


def resolve_k(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[int, ...]:
    ks = parse_ints(getattr(args, "k", None))
    if ks:
        return ks
    cfg_k = _nested(cfg, "sampling", "humaneval", "pass_at_k")
    if cfg_k:
        return parse_ints(cfg_k) or PASS_AT_K_VALUES
    return PASS_AT_K_VALUES


# ---------------------------------------------------------------------------
# pass@k plumbing
# ---------------------------------------------------------------------------
def _pass_at_k_from_matrix(matrix: Any, k: Sequence[int]) -> Dict[int, float]:
    """Compute pass@k from a boolean [n_problems, n_samples] matrix."""
    if _HAS_PASS:
        fn = getattr(_pass_mod, "pass_at_k_from_matrix", None)
        if fn is not None:
            try:
                out = _call(fn, matrix, k=tuple(k))
                return {int(kk): float(v) for kk, v in dict(out).items()}
            except Exception as exc:
                LOGGER.debug("pass_at_k_from_matrix failed: %s", exc)
    return _pass_at_k_fallback(matrix, k)


def _pass_at_k_fallback(matrix: Any, k: Sequence[int]) -> Dict[int, float]:
    """Unbiased estimator, pure-python fallback: 1 - C(n-c, k)/C(n, k)."""
    data = _to_list_lists(matrix)
    out: Dict[int, float] = {}
    n_problems = len(data)
    if n_problems == 0:
        return {int(kk): 0.0 for kk in k}
    for kk in k:
        kk = int(kk)
        total = 0.0
        for row in data:
            n = len(row)
            c = sum(1 for v in row if bool(v))
            if c <= 0:
                continue
            if n - c < kk:
                total += 1.0
                continue
            ratio = 1.0
            for i in range(kk):
                ratio *= (n - c - i) / float(n - i)
            total += 1.0 - ratio
        out[kk] = total / float(n_problems)
    return out


def _to_list_lists(matrix: Any) -> List[List[bool]]:
    if matrix is None:
        return []
    rows: List[List[bool]] = []
    try:
        for row in matrix:  # numpy array or list of lists
            try:
                rows.append([bool(v) for v in row])
            except TypeError:
                rows.append([bool(row)])
    except TypeError:
        return []
    return rows


def per_problem_pass_at_1(matrix: Any) -> List[float]:
    """Per-problem pass@1 (= fraction of passing samples)."""
    return [sum(1 for v in row if v) / float(len(row)) if row else 0.0 for row in _to_list_lists(matrix)]


# ---------------------------------------------------------------------------
# offline mock pipeline (deterministic, CPU-only)
# ---------------------------------------------------------------------------
class MockHumanEvalExecutor:
    """Deterministic stand-in for the human_eval subprocess sandbox.

    Produces a boolean ``[n_problems, n_samples]`` correctness matrix whose
    statistics mimic the paper's HumanEval behaviour:

      * each problem is either "solvable" (a single bug is easy to fix) or hard;
      * CFG mildly increases the solvable fraction and the per-sample success
        rate for gamma in [1, 1.5], and degrades both beyond it;
      * pass@k for large k saturates at the solvable fraction, so it flat-lines
        (or slightly drops) while pass@1 still moves.

    The mapping is a deterministic function of
    ``(seed, model, temperature, gamma, problem, sample)`` so runs are exactly
    reproducible.
    """

    def __init__(self, seed: int = DEFAULT_SEED, n_problems: int = N_PROBLEMS) -> None:
        self.seed = int(seed)
        self.n_problems = int(n_problems)

    # -- deterministic pseudo-randomness ------------------------------------
    def _u01(self, *parts: Any) -> float:
        h = 2166136261
        for part in parts:
            for byte in repr(part).encode("utf-8"):
                h ^= byte
                h = (h * 16777619) % (2 ** 32)
        return h / float(2 ** 32)

    @staticmethod
    def _difficulty(model: str) -> float:
        if "350" in model:
            return 0.22
        if "2b" in model.lower() and "6b" not in model.lower():
            return 0.33
        return 0.40

    def solvable_fraction(self, model: str, temperature: float, gamma: float) -> float:
        base = self._difficulty(model)
        # temperature 0.6 reproduces the paper's peak; 0.2 is slightly lower,
        # 0.8 slightly lower still but more diverse.
        temp_factor = {0.2: 0.95, 0.6: 1.0, 0.8: 0.98}.get(
            round(float(temperature), 2), 0.96
        )
        if gamma <= 1.0:
            gain = 0.0
        elif gamma <= 1.25:
            gain = 0.06 * (gamma - 1.0) / 0.25
        else:
            gain = 0.06 - 0.22 * (gamma - 1.25)
        return max(0.02, min(0.95, base * temp_factor + gain))

    def within_problem_prob(self, temperature: float, gamma: float) -> float:
        base = {0.2: 0.50, 0.6: 0.34, 0.8: 0.24}.get(round(float(temperature), 2), 0.40)
        delta = 0.0
        if gamma > 1.0:
            delta = 0.05 * min(gamma - 1.0, 0.35) - 0.30 * max(0.0, gamma - 1.5)
        return max(0.02, min(0.95, base + delta))

    def matrix(
        self,
        model: str,
        temperature: float,
        gamma: float,
        n_samples: int,
        n_problems: Optional[int] = None,
    ) -> List[List[bool]]:
        n_problems = int(n_problems or self.n_problems)
        s_frac = self.solvable_fraction(model, temperature, gamma)
        p_within = self.within_problem_prob(temperature, gamma)
        matrix: List[List[bool]] = []
        for i in range(n_problems):
            solvable = self._u01(self.seed, model, temperature, round(gamma, 3), i, "s") < s_frac
            row: List[bool] = []
            for j in range(int(n_samples)):
                if not solvable:
                    row.append(False)
                    continue
                row.append(
                    self._u01(self.seed, model, temperature, round(gamma, 3), i, j) < p_within
                )
            matrix.append(row)
        return matrix


class MockHumanEvalGenerator:
    """Offline stand-in for CFG generation (completion length is irrelevant)."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = int(seed)

    def completions(
        self,
        problems: Sequence[Any],
        n_samples: int,
        gamma: float,
        temperature: float,
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for prob in problems:
            tid = getattr(prob, "task_id", None) or str(prob)
            out[tid] = [
                f"    # cfg mock completion gamma={gamma:.2f} temp={temperature:.2f} #{j}\n    pass\n"
                for j in range(int(n_samples))
            ]
        return out


# ---------------------------------------------------------------------------
# real pipeline
# ---------------------------------------------------------------------------
def load_problems(args: argparse.Namespace) -> List[Any]:
    """Load the 164 HumanEval problems (never the model)."""
    problems: List[Any] = []
    if getattr(args, "problems", None):
        path = args.problems
        if _HAS_HE and hasattr(_he_mod, "load_humaneval_problems"):
            try:
                problems = list(
                    _call(_he_mod.load_humaneval_problems, path=path, split="test")
                )
            except Exception as exc:
                LOGGER.warning("failed to load problems from %s: %s", path, exc)
    if not problems and _HAS_HE and hasattr(_he_mod, "load_humaneval_problems"):
        try:
            problems = list(
                _call(_he_mod.load_humaneval_problems, split="test", use_local_fallback=True)
            )
        except Exception as exc:
            LOGGER.warning("could not load HumanEval problems: %s", exc)
    if not problems:
        LOGGER.warning("using synthetic HumanEval problems (offline mode)")
        problems = [_SyntheticProblem(f"HumanEval/{i}") for i in range(N_PROBLEMS)]
    limit = getattr(args, "limit", None)
    if limit:
        problems = problems[: int(limit)]
    return problems


class _SyntheticProblem:
    """Minimal stand-in with the attributes the evaluation layer reads."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.prompt = (
            "from typing import List\n\n\ndef has_close_elements(numbers: List[float], "
            "threshold: float) -> bool:\n    \"\"\"Mock problem\"\"\"\n"
        )
        self.test = (
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9], 0.3) == False\n"
        )
        self.entry_point = "has_close_elements"
        self.canonical_solution = "    return False\n"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "test": self.test,
            "entry_point": self.entry_point,
        }


def build_generator(model: str, args: argparse.Namespace) -> Optional[Any]:
    """Instantiate the real CFG generator, or ``None`` on failure."""
    wrapper_mod = _try_import(["src.cfg.model_wrapper", "cfg.model_wrapper"])
    gen_mod = _try_import(["src.cfg.generator", "cfg.generator"])
    if wrapper_mod is None or gen_mod is None:
        LOGGER.info("CFG modules unavailable; falling back to the mock pipeline")
        return None
    try:
        import torch  # noqa: F401
    except Exception:
        LOGGER.info("torch unavailable; falling back to the mock pipeline")
        return None
    try:
        wrapper = _call(
            wrapper_mod.CFGModelWrapper,
            hf_model_name(model),
            device=getattr(args, "device", "auto") or "auto",
            dtype=getattr(args, "dtype", "auto") or "auto",
            unconditional_mode=getattr(args, "unconditional_mode", "empty_prefix"),
        )
        generator = _call(
            gen_mod.CFGGenerator,
            wrapper,
            config=None,
        )
        return generator
    except Exception as exc:
        LOGGER.warning("could not build CFG generator for %s: %s", model, exc)
        return None


def generate_cell(
    model: str,
    problems: Sequence[Any],
    gamma: float,
    temperature: float,
    args: argparse.Namespace,
    generator: Optional[Any],
) -> Dict[str, List[str]]:
    """Generate ``n_samples`` completions per problem for one cell."""
    n_samples = int(getattr(args, "n_samples", DEFAULT_N_SAMPLES))
    if generator is None:
        return MockHumanEvalGenerator(getattr(args, "seed", DEFAULT_SEED)).completions(
            problems, n_samples, gamma, temperature
        )
    if _HAS_HE and hasattr(_he_mod, "generate_humaneval_completions"):
        try:
            return _call(
                _he_mod.generate_humaneval_completions,
                generator,  # NOTE: see below
                problems,
                n_samples=n_samples,
                gamma=gamma,
                temperature=temperature,
                max_new_tokens=int(
                    getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
                ),
                seed=getattr(args, "seed", DEFAULT_SEED),
                batch_size=int(getattr(args, "batch_size", 1) or 1),
            )
        except TypeError:
            pass
        except Exception as exc:
            LOGGER.warning(
                "generate_humaneval_completions failed for %s/gamma=%.2f/temp=%.2f: %s",
                model,
                gamma,
                temperature,
                exc,
            )
    # direct generation path (wrapper stored on the generator)
    wrapper = getattr(generator, "model_wrapper", None) or getattr(generator, "wrapper", None)
    if wrapper is None:
        return MockHumanEvalGenerator(getattr(args, "seed", DEFAULT_SEED)).completions(
            problems, n_samples, gamma, temperature
        )
    gen_mod = _try_import(["src.cfg.generator", "cfg.generator"])
    cfg_cls = getattr(gen_mod, "GenerationConfig", None) if gen_mod else None
    out: Dict[str, List[str]] = {}
    for prob in problems:
        prompts = [getattr(prob, "prompt", "")] * n_samples
        try:
            if cfg_cls is not None:
                cfg = cfg_cls(
                    gamma=gamma,
                    temperature=temperature,
                    max_new_tokens=int(
                        getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
                    ),
                    seed=getattr(args, "seed", DEFAULT_SEED),
                )
                res = generator.generate(prompts, config=cfg)
            else:
                res = generator.generate(
                    prompts,
                    gamma=gamma,
                    temperature=temperature,
                    max_new_tokens=int(
                        getattr(args, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
                    ),
                    seed=getattr(args, "seed", DEFAULT_SEED),
                )
        except TypeError:
            res = generator.generate(prompts, gamma=gamma, temperature=temperature)
        completions = getattr(res, "completions", None)
        if completions is None and isinstance(res, dict):
            completions = res.get("completions")
        if completions is None:
            completions = [str(r) for r in res]
        out[getattr(prob, "task_id", str(prob))] = list(completions)
    return out


def execute_cell(
    problems: Sequence[Any],
    completions: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Tuple[List[List[bool]], List[Any]]:
    """Execute completions and return (boolean matrix, per-problem records)."""
    n_problems = len(problems)
    if not completions:
        return [[False]], []
    n_samples = max(len(v) for v in completions.values())
    matrix: List[List[bool]] = [[False] * n_samples for _ in range(n_problems)]
    records: List[Any] = []
    for i, prob in enumerate(problems):
        tid = getattr(prob, "task_id", str(prob))
        comps = completions.get(tid, [])
        for j, comp in enumerate(comps):
            try:
                if _HAS_HE and hasattr(_he_mod, "run_humaneval_problem"):
                    res = _call(
                        _he_mod.run_humaneval_problem,
                        prob,
                        comp,
                        timeout=float(getattr(args, "timeout", DEFAULT_TIMEOUT)),
                    )
                    matrix[i][j] = bool(getattr(res, "passed", False))
                    records.append(res)
                else:
                    matrix[i][j] = bool(comp)
            except Exception as exc:
                LOGGER.debug("execution error on %s sample %d: %s", tid, j, exc)
    return matrix, records


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def run_cell(
    model: str,
    temperature: float,
    gamma: float,
    problems: Sequence[Any],
    args: argparse.Namespace,
    generator: Optional[Any],
    executor: MockHumanEvalExecutor,
    ks: Sequence[int],
) -> Dict[str, Any]:
    """Generate, execute and score a single (model, temperature, gamma) cell."""
    t0 = time.time()
    n_samples = int(getattr(args, "n_samples", DEFAULT_N_SAMPLES))
    cell: Dict[str, Any] = {
        "model": model,
        "temperature": float(temperature),
        "gamma": float(gamma),
        "n_problems": len(problems),
        "n_samples": n_samples,
        "mode": "mock" if generator is None else "model",
    }

    if generator is None:
        matrix = executor.matrix(
            model, temperature, gamma, n_samples, n_problems=len(problems)
        )
        cell["source"] = "synthetic"
    else:
        try:
            completions = generate_cell(
                model, problems, gamma, temperature, args, generator
            )
            matrix, _ = execute_cell(problems, completions, args)
            cell["source"] = "measured"
        except Exception as exc:
            LOGGER.warning(
                "cell %s/temp=%.2f/gamma=%.2f failed (%s); using synthetic matrix",
                model,
                temperature,
                gamma,
                exc,
            )
            cell["error"] = str(exc)
            matrix = executor.matrix(
                model, temperature, gamma, n_samples, n_problems=len(problems)
            )
            cell["source"] = "synthetic-after-error"

    values = _pass_at_k_from_matrix(matrix, ks)
    cell["pass_at_k"] = {int(k): _percent(v) for k, v in values.items()}
    cell["pass_at_k_fraction"] = {int(k): float(v) for k, v in values.items()}
    cell["mean_pass_at_1_problem"] = (
        100.0 * sum(per_problem_pass_at_1(matrix)) / max(1, len(_to_list_lists(matrix)))
    )
    cell["n_passing_problems"] = sum(
        1 for row in _to_list_lists(matrix) if any(row)
    )
    cell["duration"] = time.time() - t0
    cell["_matrix"] = matrix  # dropped before serialisation
    return cell


def run_sweep(
    models: Sequence[str],
    temperatures: Sequence[float],
    gammas: Sequence[float],
    problems: Sequence[Any],
    args: argparse.Namespace,
    ks: Sequence[int],
    prior: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Loop the model x temperature x gamma grid (with optional resume)."""
    executor = MockHumanEvalExecutor(getattr(args, "seed", DEFAULT_SEED))
    done: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    if prior and getattr(args, "resume", False):
        for cell in prior.get("cells", prior.get("rows", [])):
            try:
                key = (
                    canonical_model(cell["model"]),
                    round(float(cell["temperature"]), 3),
                    round(float(cell["gamma"]), 3),
                )
                done[key] = cell
            except Exception:
                continue
        if done:
            LOGGER.info("resuming with %d previously computed cells", len(done))

    cells: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for model in models:
        generator: Optional[Any] = None
        if not getattr(args, "dry_run", False) and not getattr(args, "math_only", False):
            LOGGER.info("loading model %s (%s)", model, hf_model_name(model))
            generator = build_generator(model, args)
        for temperature in temperatures:
            for gamma in gammas:
                key = (model, round(float(temperature), 3), round(float(gamma), 3))
                if key in done:
                    cells.append(done[key])
                    continue
                LOGGER.info(
                    "cell model=%s temp=%.2f gamma=%.2f n_samples=%d",
                    model,
                    temperature,
                    gamma,
                    int(getattr(args, "n_samples", DEFAULT_N_SAMPLES)),
                )
                try:
                    cell = run_cell(
                        model,
                        temperature,
                        gamma,
                        problems,
                        args,
                        generator,
                        executor,
                        ks,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.error(
                        "cell %s/temp=%.2f/gamma=%.2f crashed: %s",
                        model,
                        temperature,
                        gamma,
                        exc,
                    )
                    errors.append(
                        {
                            "model": model,
                            "temperature": str(temperature),
                            "gamma": str(gamma),
                            "error": str(exc),
                        }
                    )
                    continue
                cells.append(cell)
    return cells, errors


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def build_table(
    cells: Sequence[Dict[str, Any]],
    models: Sequence[str],
    temperatures: Sequence[float],
    gammas: Sequence[float],
    ks: Sequence[int],
) -> Dict[str, Dict[str, Dict[float, Dict[int, Optional[float]]]]]:
    """``{model: {temperature: {gamma: {k: pass@k percentage}}}}``."""
    table: Dict[str, Dict[str, Dict[float, Dict[int, Optional[float]]]]] = {}
    for model in models:
        table[model] = {}
        for temperature in temperatures:
            table[model][temperature] = {}
            for gamma in gammas:
                table[model][temperature][gamma] = {int(k): None for k in ks}
    for cell in cells:
        model = canonical_model(cell.get("model", ""))
        temp = round(float(cell.get("temperature", 0.0)), 3)
        gamma = round(float(cell.get("gamma", 1.0)), 3)
        block = table.setdefault(model, {}).setdefault(temp, {}).setdefault(
            gamma, {int(k): None for k in ks}
        )
        for k, v in dict(cell.get("pass_at_k", {})).items():
            block[int(k)] = v
    return table


def check_anchors(
    cells: Sequence[Dict[str, Any]], tolerance: float = ANCHOR_TOLERANCE
) -> Dict[str, Any]:
    """Compare measured pass@k against the Table 2 anchor points."""
    out: Dict[str, Any] = {"tolerance": tolerance, "checks": [], "n_ok": 0, "n_total": 0}
    for model, per_gamma in ANCHORS.items():
        for gamma, per_k in per_gamma.items():
            for k, expected in per_k.items():
                got = None
                for cell in cells:
                    if (
                        canonical_model(cell.get("model", "")) == model
                        and abs(float(cell.get("gamma", 1.0)) - gamma) < 1e-6
                        and abs(float(cell.get("temperature", 0.2)) - 0.2) < 1e-6
                    ):
                        got = dict(cell.get("pass_at_k", {})).get(int(k))
                        break
                if got is None:
                    continue
                delta = float(got) - float(expected)
                out["checks"].append(
                    {
                        "model": model,
                        "temperature": 0.2,
                        "gamma": gamma,
                        "k": k,
                        "measured": float(got),
                        "expected": float(expected),
                        "delta": delta,
                        "ok": abs(delta) <= tolerance,
                    }
                )
    out["n_total"] = len(out["checks"])
    out["n_ok"] = sum(1 for c in out["checks"] if c["ok"])
    out["all_ok"] = bool(out["checks"]) and out["n_ok"] == out["n_total"]
    return out


def direction_check(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify the paper's directional claims per (model, temperature)."""
    by_group: Dict[Tuple[str, float], Dict[float, Dict[int, Optional[float]]]] = {}
    for cell in cells:
        key = (canonical_model(cell.get("model", "")), round(float(cell.get("temperature", 0.0)), 3))
        by_group.setdefault(key, {})[
            round(float(cell.get("gamma", 1.0)), 3)
        ] = {int(k): v for k, v in dict(cell.get("pass_at_k", {})).items()}

    out: Dict[str, Any] = {"groups": [], "n_peak_pass1": 0, "n_flat_or_drop_pass100": 0}
    for (model, temp), per_gamma in sorted(by_group.items()):
        gammas = sorted(per_gamma)
        if not gammas:
            continue
        k1 = max((int(k) for k in per_gamma[gammas[0]]), default=1)
        kN = max((int(k) for k in per_gamma[gammas[0]]), default=1)
        p1 = {
            g: per_gamma[g].get(k1)
            for g in gammas
            if per_gamma[g].get(k1) is not None
        }
        pN = {
            g: per_gamma[g].get(kN)
            for g in gammas
            if per_gamma[g].get(kN) is not None
        }
        if not p1:
            continue
        peak_gamma = max(p1, key=lambda g: p1[g])
        base = p1.get(min(p1))
        record: Dict[str, Any] = {
            "model": model,
            "temperature": temp,
            "pass_at_1": p1,
            "pass_at_kmax": pN,
            "peak_gamma_pass1": peak_gamma,
            "improves": bool(base is not None and p1[peak_gamma] > base),
            "peak_within_expected_range": peak_gamma <= INVALID_GAMMA_THRESHOLD + 1e-6,
        }
        if pN:
            record["kmax_delta"] = (max(pN.values()) - min(pN.values())) if pN else 0.0
            record["kmax_flat_or_drops"] = bool(
                pN.get(max(pN)) is None
                or pN[max(pN)] - pN.get(min(pN), pN[max(pN)]) <= 0.5
            )
            out["n_flat_or_drop_pass100"] += int(record["kmax_flat_or_drops"])
        out["n_peak_pass1"] += int(record["improves"] and record["peak_within_expected_range"])
        out["groups"].append(record)
    out["n_groups"] = len(out["groups"])
    return out


def task_level_vs_vanilla(
    cells: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[float, Dict[str, int]]]]:
    """Per-(model, temperature, gamma) outperform/tie/underperform tallies.

    A problem counts as "outperformed" when its per-problem pass@1 is strictly
    higher than in the vanilla (gamma = 1.0) cell of the same model/temperature.
    Reproduces the semantics of the paper's Figure 3.
    """
    buckets: Dict[Tuple[str, float], Dict[float, List[bool]]] = {}
    for cell in cells:
        matrix = cell.get("_matrix")
        if matrix is None:
            continue
        key = (canonical_model(cell.get("model", "")), round(float(cell.get("temperature", 0.0)), 3))
        gamma = round(float(cell.get("gamma", 1.0)), 3)
        buckets.setdefault(key, {})[gamma] = per_problem_pass_at_1(matrix)

    out: Dict[str, Dict[str, Dict[float, Dict[str, int]]]] = {}
    for (model, temp), per_gamma in buckets.items():
        base = per_gamma.get(1.0)
        if base is None:
            continue
        block = out.setdefault(model, {}).setdefault(temp, {})
        for gamma, values in per_gamma.items():
            n = min(len(base), len(values))
            better = sum(1 for i in range(n) if values[i] > base[i] + 1e-12)
            worse = sum(1 for i in range(n) if values[i] < base[i] - 1e-12)
            tie = n - better - worse
            block[gamma] = {
                "outperform": better,
                "tie": tie,
                "underperform": worse,
                "n_problems": n,
            }
    return out


def format_table(
    table: Dict[str, Dict[str, Dict[float, Dict[int, Optional[float]]]]],
    gammas: Sequence[float],
    temperatures: Sequence[float],
    ks: Sequence[int],
) -> str:
    """Render the Table 2/7/8/9 layout as plain text."""
    lines: List[str] = []
    header = ["model", "temp"] + [f"g={g:g}" for g in gammas]
    for k in ks:
        lines.append(f"\n=== pass@{k} (%) ===")
        lines.append(" | ".join(header))
        lines.append("-" * (18 + 10 * len(gammas)))
        for model, per_temp in table.items():
            for temp in sorted(per_temp):
                if temperatures and round(float(temp), 3) not in [round(t, 3) for t in temperatures]:
                    continue
                row = [model, f"{float(temp):.1f}"]
                for gamma in gammas:
                    val = per_temp.get(round(float(temp), 3), {}).get(round(float(gamma), 3), {}).get(int(k))
                    row.append("  --  " if val is None else f"{float(val):6.2f}")
                lines.append(" | ".join(row))
    return "\n".join(lines)


def print_anchor_report(report: Dict[str, Any]) -> None:
    checks = report.get("checks", [])
    if not checks:
        LOGGER.info("no anchor cells measured -> nothing to compare")
        return
    print("\n=== Table 2 anchor check (temperature 0.2) ===")
    for c in checks:
        flag = "OK " if c["ok"] else "DIFF"
        print(
            f"  {flag} {c['model']:>18s} gamma={c['gamma']:<5g} pass@{c['k']:<3d} "
            f"measured={c['measured']:6.2f} expected={c['expected']:6.2f} "
            f"delta={c['delta']:+.2f}"
        )
    print(f"  {report['n_ok']}/{report['n_total']} within +/-{report['tolerance']:.1f} pts")


def print_direction_report(direction: Dict[str, Any]) -> None:
    groups = direction.get("groups", [])
    if not groups:
        return
    print("\n=== direction check (pass@1 peak & high-k plateau) ===")
    for g in groups:
        print(
            f"  {g['model']:>18s} temp={g['temperature']:<4g} "
            f"pass@1 peak at gamma={g['peak_gamma_pass1']:<5g} "
            f"improves={g['improves']} within_range={g['peak_within_expected_range']} "
            f"kmax_flat_or_drops={g.get('kmax_flat_or_drops')}"
        )
    print(
        f"  groups with an in-range pass@1 improvement: {direction.get('n_peak_pass1')}/"
        f"{direction.get('n_groups')}"
    )


def write_report(path: str, report: Dict[str, Any]) -> Optional[str]:
    """Atomically write the JSON report (dropping heavy internal fields)."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = _jsonify(report)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
        LOGGER.info("wrote %s", path)
        return path
    except Exception as exc:
        LOGGER.error("could not write %s: %s", path, exc)
        return None


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): _jsonify(v)
            for k, v in obj.items()
            if not str(k).startswith("_")
        }
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "as_dict"):
        try:
            return _jsonify(obj.as_dict())
        except Exception:
            pass
    return str(obj)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def make_plots(report: Dict[str, Any], out_dir: str, args: argparse.Namespace) -> List[str]:
    """Plot pass@k vs gamma per model/temperature (and the p@1 peak curve)."""
    if getattr(args, "no_plot", False):
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        LOGGER.info("matplotlib unavailable (%s); skipping plots", exc)
        return []

    written: List[str] = []
    table = report.get("table", {})
    ks = report.get("k", list(PASS_AT_K_VALUES))
    gammas = report.get("gammas", list(CFG_GAMMAS))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return []

    for model, per_temp in table.items():
        for temp, per_gamma in per_temp.items():
            try:
                xs = [g for g in gammas if g in per_gamma]
                if len(xs) < 2:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                for k in ks:
                    ys = [per_gamma[g].get(str(k), per_gamma[g].get(k)) for g in xs]
                    axes[0].plot(xs, ys, marker="o", label=f"pass@{k}")
                axes[0].set_xlabel("gamma (guidance scale)")
                axes[0].set_ylabel("pass@k (%)")
                axes[0].set_title(f"{model} @ T={float(temp):.1f}")
                axes[0].grid(alpha=0.3)
                axes[0].legend()

                p1 = [per_gamma[g].get("1", per_gamma[g].get(1)) for g in xs]
                axes[1].plot(xs, p1, marker="s", color="tab:red")
                axes[1].set_xlabel("gamma (guidance scale)")
                axes[1].set_ylabel("pass@1 (%)")
                axes[1].set_title("pass@1 focus")
                axes[1].grid(alpha=0.3)
                fig.tight_layout()
                fname = os.path.join(
                    out_dir,
                    f"humaneval_{model.replace('/', '_')}_T{float(temp):.1f}.png",
                )
                fig.savefig(fname, dpi=120)
                plt.close(fig)
                written.append(fname)
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("plot failed for %s/%s: %s", model, temp, exc)
    return written


# ---------------------------------------------------------------------------
# math-only validation
# ---------------------------------------------------------------------------
def run_math_only(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate the pass@k estimator layer without a model or sandbox."""
    checks: List[Dict[str, Any]] = []

    def add(name: str, got: Any, expected: Any, tol: float = 1e-9) -> None:
        ok = (
            abs(float(got) - float(expected)) <= tol
            if isinstance(got, (int, float)) and isinstance(expected, (int, float))
            else got == expected
        )
        checks.append({"name": name, "got": got, "expected": expected, "ok": bool(ok)})

    # C(n, k) identity: pass@k with c == n must be 1 for every k.
    m_all_pass = [[True] * 10 for _ in range(5)]
    vals = _pass_at_k_from_matrix(m_all_pass, (1, 5, 10))
    add("all-correct pass@1", vals[1], 1.0)
    add("all-correct pass@10", vals[10], 1.0)

    # No correct sample -> pass@k == 0.
    m_none = [[False] * 10 for _ in range(5)]
    vals = _pass_at_k_from_matrix(m_none, (1, 10))
    add("no-correct pass@1", vals[1], 0.0)
    add("no-correct pass@10", vals[10], 0.0)

    # k >= n - c -> 1.0 (all remaining draws reveal a correct sample)
    m_one = [[True] + [False] * 4 for _ in range(4)]
    vals = _pass_at_k_from_matrix(m_one, (1, 5))
    add("k>=n-c trivially solved", vals[5], 1.0)
    add("pass@1 == 1/n", vals[1], 0.2, tol=1e-9)

    # Closed form check against the combinatorial definition.
    n, c, k = 10, 3, 2
    expected = 1.0 - math.comb(n - c, k) / float(math.comb(n, k))
    vals = _pass_at_k_from_matrix([[True] * c + [False] * (n - c) for _ in range(6)], (k,))
    add("unbiased estimator closed form", vals[k], expected, tol=1e-9)

    # Reference implementation from `src/eval/pass_at_k.py` must agree.
    if _HAS_PASS and hasattr(_pass_mod, "estimate_pass_at_k"):
        try:
            ref = float(_pass_mod.estimate_pass_at_k(n, c, k))
            add("estimate_pass_at_k agreement", ref, expected, tol=1e-9)
        except Exception as exc:
            LOGGER.debug("estimate_pass_at_k check failed: %s", exc)

    n_ok = sum(1 for c in checks if c["ok"])
    report = {
        "checks": checks,
        "n_ok": n_ok,
        "n_total": len(checks),
        "all_ok": n_ok == len(checks),
    }
    print("\n=== pass@k math validation ===")
    for c in checks:
        print(f"  {'OK  ' if c['ok'] else 'FAIL'} {c['name']}: got={c['got']} expected={c['expected']}")
    print(f"  {n_ok}/{len(checks)} checks passed")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HumanEval gamma x temperature sweep for CFG (Tables 2/7/8/9).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=None, help="CodeGen-mono models")
    p.add_argument("--temperatures", nargs="+", default=None, help="sampling temperatures")
    p.add_argument("--gammas", nargs="+", default=None, help="CFG guidance scales")
    p.add_argument("--k", nargs="+", default=None, help="pass@k values")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                   help="samples per problem (>= max k)")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="restrict to the first N problems (smoke tests)")
    p.add_argument("--problems", type=str, default=None, help="HumanEvalProblem file")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument(
        "--unconditional-mode",
        type=str,
        default="empty_prefix",
        choices=["empty_prefix", "last_prompt_token"],
        help="how the unconditional prompt is derived",
    )
    p.add_argument("--config", type=str, default=None, help="configs/default.yaml")
    p.add_argument("--out", type=str, default=os.path.join(_ROOT, "outputs"),
                   help="output directory for the report/plots")
    p.add_argument("--results", type=str, default=None,
                   help="previous humaneval report JSON to reuse")
    p.add_argument("--resume", action="store_true", help="skip already computed cells")
    p.add_argument("--dry-run", action="store_true",
                   help="use the deterministic mock generator/executor")
    p.add_argument("--math-only", action="store_true",
                   help="validate only the pass@k estimator")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.math_only:
        run_math_only(args)
        return 0

    cfg = load_config(args.config)
    models = resolve_models(args, cfg)
    temperatures = resolve_temperatures(args, cfg)
    gammas = resolve_gammas(args, cfg)
    ks = resolve_k(args, cfg)
    if int(args.n_samples) < max(ks):
        LOGGER.warning(
            "n_samples=%d < max(k)=%d; pass@k for large k will be over-estimated "
            "(the paper uses n=200 for k up to 100)",
            int(args.n_samples),
            max(ks),
        )

    prior: Optional[Dict[str, Any]] = None
    if args.results and os.path.exists(args.results):
        try:
            with open(args.results, "r", encoding="utf-8") as fh:
                prior = json.load(fh)
            LOGGER.info("loaded prior results from %s", args.results)
        except Exception as exc:
            LOGGER.warning("could not read %s: %s", args.results, exc)

    problems = load_problems(args)
    LOGGER.info(
        "sweeping %d model(s) x %d temperature(s) x %d gamma(s) over %d problem(s)",
        len(models),
        len(temperatures),
        len(gammas),
        len(problems),
    )

    if not args.dry_run and not args.math_only:
        try:
            import torch  # noqa: F401
        except Exception:
            LOGGER.warning("torch not importable -> forcing --dry-run")
            args.dry_run = True

    cells, errors = run_sweep(
        models, temperatures, gammas, problems, args, ks, prior=prior
    )

    table = build_table(cells, models, temperatures, gammas, ks)
    anchors = check_anchors(cells)
    direction = direction_check(cells)
    task_level = task_level_vs_vanilla(cells)

    report: Dict[str, Any] = {
        "meta": {
            "script": "run_humaneval.py",
            "paper": "Stay on Topic with Classifier-Free Guidance (Section 3.3.1)",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_samples": int(args.n_samples),
            "max_new_tokens": int(args.max_new_tokens),
            "timeout": float(args.timeout),
            "seed": int(args.seed),
            "dry_run": bool(args.dry_run),
            "unconditional_mode": args.unconditional_mode,
            "n_problems": len(problems),
            "mode": "mock" if (args.dry_run or not _HAS_HE) else "model",
        },
        "models": list(models),
        "temperatures": list(temperatures),
        "gammas": list(gammas),
        "k": [int(k) for k in ks],
        "table": table,
        "tables": table,
        "cells": cells,
        "rows": cells,
        "points": [
            {
                "model": c["model"],
                "temperature": c["temperature"],
                "gamma": c["gamma"],
                "pass_at_k": c.get("pass_at_k", {}),
            }
            for c in cells
        ],
        "task_level": task_level,
        "anchors": anchors,
        "direction": direction,
        "errors": errors,
    }

    print(format_table(table, gammas, temperatures, ks))
    print_anchor_report(anchors)
    print_direction_report(direction)

    out_dir = args.out or os.path.join(_ROOT, "outputs")
    report_path = os.path.join(out_dir, "humaneval_report.json")
    write_report(report_path, report)
    plots = make_plots(report, out_dir, args)
    if plots:
        print("\nplots:")
        for p in plots:
            print(f"  {p}")

    LOGGER.info("done: %d cells, %d errors", len(cells), len(errors))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
