"""HumanEval evaluation with CFG-guided code generation (Section 3.3.1).

The paper evaluates CodeGen-350M/2B/6B-mono on the 164 Python problems of
HumanEval (Chen et al., 2021) at temperatures 0.2/0.6/0.8, sweeping
``gamma`` over ``{1.0, 1.1, 1.25, 1.5, 1.75, 2.0}`` and reporting the
unbiased pass@k estimator for ``k = 1, 10, 100``.

This module provides

* ``load_humaneval`` -- the benchmark (with a JSONL fallback for offline use);
* ``completion_stop_sequences`` -- the standard CodeGen stops that prevent
  the model from writing its own tests;
* ``truncate_completion`` -- post-hoc truncation of a completion that has
  already been sampled;
* ``run_humaneval_program`` -- executes ``prompt + completion + tests`` in a
  subprocess with a timeout and reports whether it passed;
* ``evaluate_samples`` -- turns ``k`` samples per problem into pass@k
  metrics, and the per-task win/tie/loss counts used by Figure 3.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .stats import estimate_pass_at_k

# Stops used by the CodeGen HumanEval configuration: they cut the model off
# before it starts writing new top-level definitions or its own tests.
STOP_SEQUENCES: Tuple[str, ...] = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    entry_point: str
    test: str
    canonical_solution: Optional[str] = None


def load_humaneval(split: str = "test", local_path: Optional[str] = None) -> List[HumanEvalProblem]:
    """Load the HumanEval problems.

    Uses ``openai/openai_humaneval`` from the HuggingFace Hub; if
    ``local_path`` is given (JSONL with the same fields) it is used instead,
    which keeps the experiment runnable offline.
    """
    records: List[dict]
    if local_path is not None and os.path.exists(local_path):
        with open(local_path) as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    else:
        from datasets import load_dataset

        dataset = load_dataset("openai/openai_humaneval", split=split)
        records = [dict(row) for row in dataset]
    return [
        HumanEvalProblem(
            task_id=row["task_id"],
            prompt=row["prompt"],
            entry_point=row["entry_point"],
            test=row["test"],
            canonical_solution=row.get("canonical_solution"),
        )
        for row in records
    ]


def truncate_completion(completion: str, stops: Sequence[str] = STOP_SEQUENCES) -> str:
    """Cut a completion at the first stop sequence."""
    cut = len(completion)
    for stop in stops:
        idx = completion.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return completion[:cut]


def build_program(problem: HumanEvalProblem, completion: str) -> str:
    """Assemble the executable program for one generated completion."""
    return (
        problem.prompt
        + truncate_completion(completion)
        + "\n\n"
        + problem.test
        + f"\n\ncheck({problem.entry_point})\n"
    )


def run_humaneval_program_inline(program: str, timeout: float = 10.0) -> Tuple[bool, str]:
    """Execute a HumanEval program in-process with a wall-clock timeout.

    This mirrors OpenAI's ``human-eval`` runner: the generated program is
    compiled and executed with ``exec`` while a ``SIGALRM`` guards against
    infinite loops.  It is roughly a thousand times faster than spawning a
    subprocess per sample (which matters: a full sweep executes several
    hundred thousand programs per model), at the cost of sharing the
    interpreter with the generated code.
    """
    import contextlib
    import io
    import signal

    def _handler(signum, frame):  # pragma: no cover - signal path
        raise TimeoutError("execution timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(timeout, 0.1))
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(program, "<candidate>", "exec"), {"__name__": "__main__"})
        return True, stdout.getvalue()
    except TimeoutError:
        return False, "timeout"
    except AssertionError:
        return False, "assertion failed"
    except SystemExit as exc:  # pragma: no cover
        return exc.code in (0, None), f"SystemExit({exc.code})"
    except BaseException as exc:  # noqa: BLE001 - the program is untrusted
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def run_humaneval_program(program: str, timeout: float = 10.0) -> Tuple[bool, str]:
    """Run a HumanEval program in a subprocess with a timeout.

    Returns ``(passed, output)``.  Execution happens in a fresh interpreter
    (rather than ``exec`` in-process) so that crashes, infinite loops and
    ``sys.exit`` calls cannot take down the evaluation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "candidate.py")
        with open(path, "w") as fh:
            fh.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, proc.stdout
    return False, (proc.stderr or proc.stdout)[-2000:]


def completion_passes(problem: HumanEvalProblem, completion: str, timeout: float = 10.0) -> bool:
    """Whether a single completion passes the problem's unit tests."""
    passed, _ = run_humaneval_program(build_program(problem, completion), timeout=timeout)
    return passed


def evaluate_samples(
    problems: Sequence[HumanEvalProblem],
    samples: Dict[str, List[str]],
    ks: Iterable[int] = (1, 10, 100),
    timeout: float = 10.0,
    execution: str = "inline",
) -> dict:
    """Compute pass@k for ``k`` samples per problem.

    Args:
        problems: the benchmark problems.
        samples: ``{task_id: [completion, ...]}``.
        ks: the ``k`` values to report.

    Returns:
        ``{"pass@k": {k: value}, "n_correct": {...}, "per_problem": {...}}``
        where ``per_problem`` records how many of the samples passed, which
        is what Figure 3 (per-task CFG win/tie/loss counts) needs.
    """
    ks = list(ks)
    if execution not in ("inline", "subprocess"):
        raise ValueError("execution must be 'inline' or 'subprocess'")
    runner = run_humaneval_program_inline if execution == "inline" else run_humaneval_program
    n_samples: List[int] = []
    n_correct: List[int] = []
    per_problem: Dict[str, int] = {}
    for problem in problems:
        completions = samples.get(problem.task_id, [])
        passed = sum(
            int(runner(build_program(problem, completion), timeout=timeout)[0])
            for completion in completions
        )
        per_problem[problem.task_id] = passed
        n_samples.append(len(completions))
        n_correct.append(passed)
    pass_k = {
        k: float(np.mean([estimate_pass_at_k(n, c, k) for n, c in zip(n_samples, n_correct)]))
        for k in ks
    }
    return {"pass@k": pass_k, "per_problem": per_problem, "n_samples": n_samples, "n_correct": n_correct}


def win_tie_loss_counts(
    baseline_pass: Dict[str, int], cfg_pass: Dict[str, int]
) -> Dict[str, int]:
    """Per-task comparison used by Figure 3 of the paper.

    A task counts as a *win* for CFG if CFG solved it in more samples than
    the baseline, a *tie* if the counts match, and a *loss* otherwise.
    """
    wins = ties = losses = 0
    for task_id, base in baseline_pass.items():
        guided = cfg_pass.get(task_id, 0)
        if guided > base:
            wins += 1
        elif guided == base:
            ties += 1
        else:
            losses += 1
    return {"cfg_wins": wins, "ties": ties, "cfg_losses": losses, "n_tasks": len(baseline_pass)}


@torch.no_grad()
def sample_solutions(
    model,
    tokenizer,
    problem: HumanEvalProblem,
    gamma: float,
    n_samples: int = 1,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    seed: Optional[int] = None,
) -> List[str]:
    """Sample ``n_samples`` completions for one problem with CFG.

    The prompt is the raw HumanEval ``prompt`` (signature + docstring), as in
    the CodeGen evaluation protocol, and completions are truncated at the
    standard stops so that the model cannot emit its own tests.
    """
    from .generation import cfg_generate

    texts = cfg_generate(
        model,
        tokenizer,
        problem.prompt,
        gamma=gamma,
        uncond_prefix_tokens=1,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        num_return_sequences=n_samples,
        seed=seed,
    )
    return [truncate_completion(text) for text in texts]
