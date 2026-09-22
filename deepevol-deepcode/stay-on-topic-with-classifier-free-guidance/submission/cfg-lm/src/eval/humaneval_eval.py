"""HumanEval code-execution harness for the CFG-LM reproduction.

The CFG paper (Section 3.3.1) evaluates CodeGen-350M/2B/6B-mono on the HumanEval
benchmark (Chen et al. 2021):

    "The HumanEval benchmark contains 164 coding tasks in Python, with English
    prompts given by a function signature and a docstring. The model generates
    code-based continuations of the prompt, which are tested against unit tests
    to evaluate the correctness of programs."

and

    "We test different CFG strength gamma = 1.0, 1.1, 1.25, 1.5, 1.75, 2.0 and
    different temperatures, evaluating at pass@k for k = 1, 10, 100."

The pass@k definition used is that of Chen et al. (2021) (paper footnote 4):

    "k code samples are generated per problem, a problem is considered solved if
    any sample passes the unit tests, and the total fraction of problems solved
    is reported."

This module is the *execution* half of that pipeline: it turns a (problem,
completion) pair into a passed/failed verdict by running the task's unit tests
against the assembled program inside an isolated subprocess with a timeout, and
it aggregates per-problem outcomes into the unbiased pass@k estimator provided
by :mod:`src.eval.pass_at_k`.

Design notes
------------
* The official OpenAI human-eval execution protocol is reproduced (write the
  program to a temp file, execute it with a fresh interpreter, time out hard),
  but *without* requiring the ``human_eval`` package: the problem statements are
  loaded from the HuggingFace ``openai_humaneval`` dataset when available and
  from ``human_eval.data`` otherwise.
* CodeGen uses stop strings ``"\nclass", "\ndef", "\n#", "\nif", "\nprint"`` so
  that the model stops before hallucinating a new top-level definition; the
  completion is truncated at the earliest occurrence (the *earlier* entries in
  the list are cut first, matching the reference implementation which strips
  each stop token greedily).
* Every public entry point is dependency-light: only the standard library plus
  (optionally) ``numpy`` and the sibling ``pass_at_k`` module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional imports (kept soft so the module is importable in minimal setups)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised implicitly
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover
    from .pass_at_k import (  # noqa: F401
        HUMANEVAL_N_PROBLEMS,
        HUMANEVAL_N_SAMPLES,
        HUMANEVAL_TEMPERATURES,
        CFG_GAMMAS,
        compute_pass_at_k,
        count_correct,
        make_seeds,
        pass_at_k_from_matrix,
        pass_at_k_from_counts,
    )
except Exception:  # pragma: no cover - standalone execution fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pass_at_k import (  # type: ignore  # noqa: F401
        HUMANEVAL_N_PROBLEMS,
        HUMANEVAL_N_SAMPLES,
        HUMANEVAL_TEMPERATURES,
        CFG_GAMMAS,
        compute_pass_at_k,
        count_correct,
        make_seeds,
        pass_at_k_from_matrix,
        pass_at_k_from_counts,
    )

# --------------------------------------------------------------------------- #
# Constants (paper Section 3.3.1 and Appendix C.4)
# --------------------------------------------------------------------------- #
HUMANEVAL_PASS_AT_K: Tuple[int, ...] = (1, 10, 100)
"""Paper evaluates pass@1, pass@10 and pass@100 (footnote 4)."""

HUMANEVAL_N_PROBLEMS: int = 164
"""Number of coding tasks in the HumanEval benchmark (Section 3.3.1)."""

HUMANEVAL_N_SAMPLES: int = 200
"""Samples per problem: enough to estimate pass@100 reliably (n = 2k)."""

HUMANEVAL_MAX_NEW_TOKENS: int = 512
"""Per-sample generation budget (harness convention used by the paper)."""

HUMANEVAL_TIMEOUT: float = 3.0
"""Seconds allowed per program execution (OpenAI human-eval default)."""

HUMANEVAL_STOP_STRINGS: Tuple[str, ...] = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")
"""CodeGen stop strings used by the paper's code-generation evaluations."""

HUMANEVAL_TEMPERATURES: Tuple[float, ...] = (0.2, 0.6, 0.8)  # type: ignore[misc]
"""Temperatures swept in Tables 7/8/9 (Section 3.3.1, Appendix C.4)."""

HUMANEVAL_GAMMAS: Tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
"""CFG strengths swept in footnote 3."""

HUMANEVAL_ENTRY_POINT_TEMPLATE: str = "\n\n{test}\n\ncheck({entry_point})\n"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class HumanEvalProblem:
    """A single HumanEval task."""

    task_id: str
    prompt: str
    test: str
    entry_point: str
    canonical_solution: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "test": self.test,
            "entry_point": self.entry_point,
            "canonical_solution": self.canonical_solution,
        }


@dataclass
class ExecutionResult:
    """Outcome of executing one generated program."""

    task_id: str
    passed: bool
    result: str  # "passed" | "failed" | "timed out" | "error"
    completion: str = ""
    program: str = ""
    detail: str = ""
    duration: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": bool(self.passed),
            "result": self.result,
            "completion": self.completion,
            "program": self.program,
            "detail": self.detail,
            "duration": self.duration,
        }


# --------------------------------------------------------------------------- #
# Problem loading
# --------------------------------------------------------------------------- #
def _problem_from_dataset_row(row: Dict[str, Any]) -> HumanEvalProblem:
    task_id = row.get("task_id") or row.get("name") or ""
    return HumanEvalProblem(
        task_id=str(task_id),
        prompt=str(row.get("prompt", "")),
        test=str(row.get("test", "")),
        entry_point=str(row.get("entry_point", "")),
        canonical_solution=str(row.get("canonical_solution", "")),
    )


def load_humaneval_problems(
    path: Optional[str] = None,
    split: str = "test",
    use_local_fallback: bool = True,
) -> List[HumanEvalProblem]:
    """Load the 164 HumanEval problems.

    Order of preference:
      1. ``path`` (a ``.jsonl``/``.json`` file of HumanEval rows).
      2. HuggingFace ``datasets`` ``openai_humaneval``.
      3. The locally installed ``human_eval.data.read_problems`` helper.
    """

    if path is not None:
        return _read_problems_file(path)

    try:  # pragma: no cover - depends on network/cache
        from datasets import load_dataset

        ds = load_dataset("openai_humaneval", split=split)
        problems = [_problem_from_dataset_row(dict(r)) for r in ds]
        logger.info("Loaded %d HumanEval problems from openai_humaneval", len(problems))
        return problems
    except Exception as exc:  # pragma: no cover
        logger.debug("datasets load of openai_humaneval failed: %s", exc)

    if use_local_fallback:
        try:  # pragma: no cover
            from human_eval.data import read_problems

            raw = read_problems()
            problems = [_problem_from_dataset_row(dict(v)) for v in raw.values()]
            logger.info("Loaded %d HumanEval problems via human_eval.data", len(problems))
            return problems
        except Exception as exc:  # pragma: no cover
            logger.debug("human_eval.data fallback failed: %s", exc)

    logger.warning("Could not load HumanEval problems; returning an empty list.")
    return []


def _read_problems_file(path: str) -> List[HumanEvalProblem]:
    problems: List[HumanEvalProblem] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    problems.append(_problem_from_dataset_row(json.loads(line)))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            raw = list(raw.values())
        problems = [_problem_from_dataset_row(dict(r)) for r in raw]
    return problems


def problems_to_dict(
    problems: Sequence[Union[HumanEvalProblem, Dict[str, Any]]]
) -> Dict[str, HumanEvalProblem]:
    """Index problems by ``task_id``."""

    out: Dict[str, HumanEvalProblem] = {}
    for problem in problems:
        if isinstance(problem, HumanEvalProblem):
            out[problem.task_id] = problem
        else:
            parsed = _problem_from_dataset_row(dict(problem))
            out[parsed.task_id] = parsed
    return out


# --------------------------------------------------------------------------- #
# Prompt / program assembly
# --------------------------------------------------------------------------- #
def build_prompt(problem: Union[HumanEvalProblem, Dict[str, Any]]) -> str:
    """HumanEval uses the raw function-signature + docstring prompt."""

    problem = _coerce_problem(problem)
    return problem.prompt


def truncate_completion(
    completion: str, stop_strings: Optional[Sequence[str]] = HUMANEVAL_STOP_STRINGS
) -> str:
    """Cut ``completion`` at the earliest stop string (code-generation convention)."""

    if not stop_strings:
        return completion
    cut = len(completion)
    for stop in stop_strings:
        if not stop:
            continue
        idx = completion.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return completion[:cut]


def build_program(
    problem: Union[HumanEvalProblem, Dict[str, Any]],
    completion: str,
    apply_stop_strings: bool = True,
) -> str:
    """Assemble ``prompt + completion + tests + check(entry_point)``."""

    problem = _coerce_problem(problem)
    if apply_stop_strings:
        completion = truncate_completion(completion)
    tests = problem.test
    if f"check({problem.entry_point})" not in tests:
        tests = tests + HUMANEVAL_ENTRY_POINT_TEMPLATE.format(
            test="", entry_point=problem.entry_point
        )
    # Headers match the OpenAI human-eval harness so that generated solutions
    # relying on typing helpers still execute.
    return (
        "from typing import List\n"
        + problem.prompt
        + completion
        + "\n"
        + tests
        + "\n"
    )


def _coerce_problem(problem: Union[HumanEvalProblem, Dict[str, Any]]) -> HumanEvalProblem:
    if isinstance(problem, HumanEvalProblem):
        return problem
    return _problem_from_dataset_row(dict(problem))


# --------------------------------------------------------------------------- #
# Sandboxed execution
# --------------------------------------------------------------------------- #
_ERROR_PATTERNS = (
    "Traceback (most recent call last)",
    "AssertionError",
    "SyntaxError",
    "IndentationError",
    "NameError",
    "TypeError",
)


def execute_program(
    program: str,
    timeout: float = HUMANEVAL_TIMEOUT,
    python_executable: Optional[str] = None,
    env_extra: Optional[Dict[str, str]] = None,
    workdir: Optional[str] = None,
) -> Tuple[str, str, float]:
    """Run ``program`` in an isolated subprocess.

    Returns ``(result, detail, duration)`` where ``result`` is one of
    ``"passed"``, ``"failed"``, ``"timed out"`` or ``"error"``.
    """

    python_executable = python_executable or sys.executable
    tmpdir = None
    if workdir is None:
        tmpdir = tempfile.mkdtemp(prefix="humaneval_")
        workdir = tmpdir
    program_path = os.path.join(workdir, "program.py")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # avoid leaking the repo into the sandboxed run
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})

    try:
        with open(program_path, "w", encoding="utf-8") as handle:
            handle.write(program)

        start = time.time()
        try:
            proc = subprocess.run(
                [python_executable, program_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return "timed out", "", time.time() - start
        duration = time.time() - start
    finally:
        if tmpdir is not None:
            _rmtree(tmpdir)

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        return "failed", (stderr or stdout)[-4000:], duration

    # An assertion failure inside check() would have produced a nonzero exit
    # code; guard against harnesses that swallow errors.
    merged = stdout + "\n" + stderr
    for pattern in _ERROR_PATTERNS:
        if pattern in merged:
            return "failed", merged[-4000:], duration

    return "passed", "", duration


def _rmtree(path: str) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # pragma: no cover
        pass


def run_humaneval_problem(
    problem: Union[HumanEvalProblem, Dict[str, Any]],
    completion: str,
    timeout: float = HUMANEVAL_TIMEOUT,
    apply_stop_strings: bool = True,
    python_executable: Optional[str] = None,
) -> ExecutionResult:
    """Execute one generated completion against the task's unit tests."""

    problem = _coerce_problem(problem)
    program = build_program(problem, completion, apply_stop_strings=apply_stop_strings)
    result, detail, duration = execute_program(
        program, timeout=timeout, python_executable=python_executable
    )
    return ExecutionResult(
        task_id=problem.task_id,
        passed=(result == "passed"),
        result=result,
        completion=completion,
        program=program,
        detail=detail,
        duration=duration,
    )


# Alias matching the OpenAI harness naming.
check_correctness = run_humaneval_problem


def check_correctness_batch(
    problems: Sequence[Union[HumanEvalProblem, Dict[str, Any]]],
    completions: Dict[str, Union[str, Sequence[str]]],
    timeout: float = HUMANEVAL_TIMEOUT,
    apply_stop_strings: bool = True,
    max_problems: Optional[int] = None,
) -> List[ExecutionResult]:
    """Execute a (possibly multi-sample) completion dict sequentially."""

    lookup = problems_to_dict(problems)
    results: List[ExecutionResult] = []
    for i, (task_id, comps) in enumerate(completions.items()):
        if max_problems is not None and i >= max_problems:
            break
        problem = lookup.get(task_id)
        if problem is None:
            logger.debug("Skipping unknown task_id %s", task_id)
            continue
        if isinstance(comps, str):
            comps = [comps]
        for completion in comps:
            results.append(
                run_humaneval_problem(
                    problem,
                    completion,
                    timeout=timeout,
                    apply_stop_strings=apply_stop_strings,
                )
            )
    return results


# --------------------------------------------------------------------------- #
# Generation driver (uses the CFG decode loop from src/cfg)
# --------------------------------------------------------------------------- #
def generate_humaneval_completions(
    model_wrapper: Any,
    problems: Sequence[Union[HumanEvalProblem, Dict[str, Any]]],
    n_samples: int = HUMANEVAL_N_SAMPLES,
    gamma: float = 1.0,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_new_tokens: int = HUMANEVAL_MAX_NEW_TOKENS,
    seed: Optional[int] = None,
    batch_size: int = 1,
    prompt_mode: str = "completion",
) -> Dict[str, List[str]]:
    """Sample ``n_samples`` code completions per problem with CFG decoding.

    ``gamma = 1.0`` reproduces vanilla sampling; ``gamma > 1`` applies
    Classifier-Free Guidance on the raw logits (Section 2.2, Eq. 7).
    Completions are truncated at the CodeGen stop strings before being returned.
    """

    try:  # local import: keeps this module importable without torch
        from ..cfg.generator import CFGGenerator, GenerationConfig

        wrapper = model_wrapper
    except Exception:  # pragma: no cover - standalone execution
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        from cfg.generator import CFGGenerator, GenerationConfig  # type: ignore

        wrapper = model_wrapper

    config = GenerationConfig(
        gamma=float(gamma),
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        stop_strings=tuple(HUMANEVAL_STOP_STRINGS),
        stop_on_strings=True,
        seed=seed,
    )
    generator = CFGGenerator(wrapper, config=config if False else None)

    prompts = [build_prompt(p) for p in problems]
    out: Dict[str, List[str]] = {}
    for start in range(0, len(problems), max(1, batch_size)):
        chunk_problems = problems[start : start + max(1, batch_size)]
        chunk_prompts = prompts[start : start + max(1, batch_size)]
        for sample_idx in range(int(n_samples)):
            sample_config = config.replace(
                seed=None if seed is None else int(seed) + start + sample_idx
            )
            output = generator.generate(chunk_prompts, config=sample_config)
            for problem, completion in zip(chunk_problems, output.completions):
                task_id = _coerce_problem(problem).task_id
                truncated = truncate_completion(completion)
                out.setdefault(task_id, []).append(truncated)
    return out


# --------------------------------------------------------------------------- #
# Aggregation: pass@k over problems
# --------------------------------------------------------------------------- #
def results_to_counts(
    results: Sequence[Union[ExecutionResult, Dict[str, Any]]],
) -> Dict[str, Tuple[int, int]]:
    """Collapse execution results into ``{task_id: (n_samples, n_correct)}``."""

    counts: Dict[str, List[int]] = {}
    for item in results:
        if isinstance(item, ExecutionResult):
            task_id, passed = item.task_id, item.passed
        else:
            task_id = str(item.get("task_id", ""))
            passed = count_correct([item.get("passed", item.get("result", False))]) > 0
        entry = counts.setdefault(task_id, [0, 0])
        entry[0] += 1
        entry[1] += int(bool(passed))
    return {task_id: (int(n), int(c)) for task_id, (n, c) in counts.items()}


def correctness_matrix(
    completions: Dict[str, Sequence[str]],
    results: Sequence[Union[ExecutionResult, Dict[str, Any]]],
    task_ids: Optional[Sequence[str]] = None,
) -> "Any":
    """Boolean ``[n_problems, n_samples]`` correctness matrix (numpy array)."""

    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for correctness_matrix")
    if task_ids is None:
        task_ids = sorted(completions.keys())
    n_samples = max((len(completions[t]) for t in task_ids), default=0)
    matrix = np.zeros((len(task_ids), n_samples), dtype=bool)
    row_of = {task_id: i for i, task_id in enumerate(task_ids)}

    sample_idx: Dict[str, int] = {}
    for item in results:
        if isinstance(item, ExecutionResult):
            task_id, passed = item.task_id, item.passed
        else:
            task_id = str(item.get("task_id", ""))
            passed = count_correct([item.get("passed", item.get("result", False))]) > 0
        if task_id not in row_of:
            continue
        col = sample_idx.get(task_id, 0)
        if col < n_samples:
            matrix[row_of[task_id], col] = bool(passed)
        sample_idx[task_id] = col + 1
    return matrix


def evaluate_humaneval(
    problems: Sequence[Union[HumanEvalProblem, Dict[str, Any]]],
    completions: Dict[str, Union[str, Sequence[str]]],
    k: Sequence[int] = HUMANEVAL_PASS_AT_K,
    timeout: float = HUMANEVAL_TIMEOUT,
    apply_stop_strings: bool = True,
    max_problems: Optional[int] = None,
    already_executed: Optional[Sequence[Union[ExecutionResult, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Execute every completion and report pass@k for the requested k values.

    Returns a dict with:
      * ``pass_at_k``: ``{k: fraction}`` (the headline numbers of Tables 2/7/8/9);
      * ``counts``: ``{task_id: (n_samples, n_correct)}`` for scatter plots
        (Figure 3 semantics: which tasks outperform/underperform);
      * ``matrix``: boolean correctness matrix (``None`` if numpy is missing);
      * ``results``: raw :class:`ExecutionResult` objects (for JSON dumping);
      * ``n_problems``, ``n_completions``, ``n_passed_programs``.
    """

    if already_executed is not None:
        results = list(already_executed)
    else:
        results = check_correctness_batch(
            problems,
            completions,
            timeout=timeout,
            apply_stop_strings=apply_stop_strings,
            max_problems=max_problems,
        )

    counts = results_to_counts(results)
    pass_k = pass_at_k_from_counts(counts, k=tuple(k)) if counts else {int(kk): 0.0 for kk in k}

    matrix = None
    if np is not None and completions:
        try:
            matrix = correctness_matrix(completions, results)
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not build correctness matrix: %s", exc)

    return {
        "pass_at_k": pass_k,
        "counts": counts,
        "matrix": matrix,
        "results": results,
        "n_problems": len(counts),
        "n_completions": sum(n for n, _ in counts.values()),
        "n_passed_programs": sum(c for _, c in counts.values()),
    }


def evaluate_humaneval_grid(
    problems: Sequence[Union[HumanEvalProblem, Dict[str, Any]]],
    completions_by_cell: Dict[Tuple[float, float], Dict[str, Sequence[str]]],
    k: Sequence[int] = HUMANEVAL_PASS_AT_K,
    timeout: float = HUMANEVAL_TIMEOUT,
    apply_stop_strings: bool = True,
) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """Run :func:`evaluate_humaneval` for each ``(temperature, gamma)`` cell."""

    out: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for cell, completions in completions_by_cell.items():
        logger.info("HumanEval cell temperature=%s gamma=%s", cell[0], cell[1])
        out[cell] = evaluate_humaneval(
            problems,
            completions,
            k=k,
            timeout=timeout,
            apply_stop_strings=apply_stop_strings,
        )
    return out


def pass_at_k_table_from_cells(
    grid_results: Dict[Tuple[float, float], Dict[str, Any]],
    k: Sequence[int] = HUMANEVAL_PASS_AT_K,
) -> Dict[Tuple[float, float], Dict[int, float]]:
    """``{(temperature, gamma): {k: pass_at_k}}`` — the layout of Table 2/7/8/9."""

    return {
        tuple(cell): {int(kk): float(v) for kk, v in res["pass_at_k"].items()}
        for cell, res in grid_results.items()
    }


def format_table(
    table: Dict[Tuple[float, float], Dict[int, float]],
    k: Sequence[int] = HUMANEVAL_PASS_AT_K,
) -> str:
    """Render a pass@k sweep as a plain-text table (pasted into the README)."""

    header = "temperature | gamma | " + " | ".join(f"pass@{kk}" for kk in k)
    lines = [header, "-" * len(header)]
    for temperature, gamma in sorted(table.keys()):
        values = table[(temperature, gamma)]
        row = f"{temperature:>11} | {gamma:>5} | " + " | ".join(
            f"{100.0 * float(values.get(int(kk), 0.0)):>7.1f}" for kk in k
        )
        lines.append(row)
    return "\n".join(lines)


def task_level_distribution(
    counts: Dict[str, Tuple[int, int]],
    reference_counts: Dict[str, Tuple[int, int]],
    k: int = 1,
) -> Dict[str, int]:
    """Outperform / tie / underperform task tally (Figure 3 semantics).

    Compares per-task pass@k between two conditions, where each condition is a
    ``{task_id: (n_samples, n_correct)}`` mapping.
    """

    from .pass_at_k import estimate_pass_at_k

    outperform = tie = underperform = 0
    for task_id, (n, c) in counts.items():
        if task_id not in reference_counts:
            continue
        n_ref, c_ref = reference_counts[task_id]
        a = float(estimate_pass_at_k(n, c, k))
        b = float(estimate_pass_at_k(n_ref, c_ref, k))
        if a > b + 1e-12:
            outperform += 1
        elif a < b - 1e-12:
            underperform += 1
        else:
            tie += 1
    return {"outperform": outperform, "tie": tie, "underperform": underperform}


def save_results(path: str, results: Sequence[Union[ExecutionResult, Dict[str, Any]]]) -> None:
    """Dump execution results as JSONL (one program per line)."""

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in results:
            record = item.as_dict() if isinstance(item, ExecutionResult) else dict(item)
            handle.write(json.dumps(record) + "\n")


__all__ = [
    "HUMANEVAL_PASS_AT_K",
    "HUMANEVAL_N_PROBLEMS",
    "HUMANEVAL_N_SAMPLES",
    "HUMANEVAL_MAX_NEW_TOKENS",
    "HUMANEVAL_TIMEOUT",
    "HUMANEVAL_STOP_STRINGS",
    "HUMANEVAL_TEMPERATURES",
    "HUMANEVAL_GAMMAS",
    "HumanEvalProblem",
    "ExecutionResult",
    "load_humaneval_problems",
    "problems_to_dict",
    "build_prompt",
    "build_program",
    "truncate_completion",
    "execute_program",
    "run_humaneval_problem",
    "check_correctness",
    "check_correctness_batch",
    "generate_humaneval_completions",
    "results_to_counts",
    "correctness_matrix",
    "evaluate_humaneval",
    "evaluate_humaneval_grid",
    "pass_at_k_table_from_cells",
    "format_table",
    "task_level_distribution",
    "save_results",
]
