"""A small evaluation harness that runs the paper's zero-shot benchmarks.

``evaluate_task`` evaluates one benchmark at one guidance strength; a whole
sweep is an outer loop over ``gamma`` (see ``experiments/run_zeroshot.py``),
which is how Table 5 / Figures 6-9 of the paper were produced.

The metrics follow the LM Evaluation Harness:

* multiple choice: ``acc`` (argmax of the CFG log-likelihood) and
  ``acc_norm`` (argmax of the log-likelihood normalised by the number of
  continuation tokens);
* LAMBADA: ``acc`` = fraction of items whose gold final word is the greedy
  completion of the CFG distribution;
* TriviaQA: ``substring_match`` over the answer aliases.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .generation import cfg_generate
from .scoring import CFGScorer, encode_pair
from .tasks import Task, TABLE5_TASKS, get_task, load_examples


@dataclass
class TaskResult:
    """Result of evaluating one (model, task, gamma) triple."""

    model: str
    task: str
    gamma: float
    n: int
    metrics: Dict[str, float] = field(default_factory=dict)
    n_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def substring_match(prediction: str, targets: Sequence[str]) -> bool:
    """Case-insensitive substring match against any of the answer aliases."""
    pred = _normalize_answer(prediction)
    for target in targets:
        tgt = _normalize_answer(target)
        if tgt and tgt in pred:
            return True
    return False


@torch.no_grad()
def evaluate_multiple_choice(
    model,
    tokenizer,
    task: Task,
    gamma: float,
    limit: Optional[int] = None,
    uncond_prefix_tokens: int = 1,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    return_predictions: bool = False,
):
    """Evaluate a multiple-choice / log-likelihood task under CFG."""
    docs = load_examples(task, limit=limit)
    scorer = CFGScorer(
        model,
        tokenizer,
        gamma=gamma,
        uncond_prefix_tokens=uncond_prefix_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )

    correct = 0
    correct_norm = 0
    total = 0
    n_tokens = 0
    predictions = []
    # Score several documents per forward pass: all (context, choice) pairs
    # of a batch of documents are sent to the scorer together, which is what
    # makes the full Table 5 sweep tractable.
    docs = list(docs)
    docs_per_batch = max(1, batch_size)
    for start in range(0, len(docs), docs_per_batch):
        batch = docs[start : start + docs_per_batch]
        requests: List[Tuple[str, str]] = []
        layout: List[Tuple[int, int]] = []
        for doc_index, doc in enumerate(batch):
            ctx = task.doc_to_text(doc)
            for choice_index, choice in enumerate(task.doc_to_choices(doc)):
                requests.append((ctx, choice))
                layout.append((doc_index, choice_index))
        scores = scorer.loglikelihood(requests)

        per_doc_lls: List[List[float]] = [[] for _ in batch]
        per_doc_is_greedy: List[List[bool]] = [[] for _ in batch]
        for (doc_index, _), (ll, is_greedy) in zip(layout, scores):
            per_doc_lls[doc_index].append(ll)
            per_doc_is_greedy[doc_index].append(is_greedy)

        for doc_index, doc in enumerate(batch):
            ctx = task.doc_to_text(doc)
            choices = task.doc_to_choices(doc)
            gold = task.doc_to_gold(doc)
            lls = per_doc_lls[doc_index]
            lengths = [
                max(len(encode_pair(tokenizer, ctx, choice)[1]), 1) for choice in choices
            ]
            pred = max(range(len(lls)), key=lambda i: lls[i])
            pred_norm = max(range(len(lls)), key=lambda i: lls[i] / lengths[i])
            correct += int(pred == gold)
            correct_norm += int(pred_norm == gold)
            total += 1
            n_tokens += sum(lengths)
            if return_predictions:
                predictions.append(
                    {"ctx": ctx, "choices": choices, "gold": gold, "pred": pred, "lls": lls}
                )
        if (start // docs_per_batch) % 10 == 0:
            print(f"[{task.name}] {total}/{len(docs)} docs, acc={correct / max(total, 1):.4f}")

    result = TaskResult(
        model="",
        task=task.name,
        gamma=gamma,
        n=total,
        metrics={
            "acc": correct / total if total else float("nan"),
            "acc_norm": correct_norm / total if total else float("nan"),
        },
        n_tokens=n_tokens,
    )
    return (result, predictions) if return_predictions else result


@torch.no_grad()
def evaluate_lambada(
    model,
    tokenizer,
    task: Task,
    gamma: float,
    limit: Optional[int] = None,
    uncond_prefix_tokens: int = 1,
    batch_size: int = 8,
    max_length: Optional[int] = None,
):
    """LAMBADA (OpenAI): accuracy of the greedy CFG completion of the last word."""
    docs = load_examples(task, limit=limit)
    scorer = CFGScorer(
        model,
        tokenizer,
        gamma=gamma,
        uncond_prefix_tokens=uncond_prefix_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )
    ctxs, conts = [], []
    for doc in docs:
        ctxs.append(task.doc_to_text(doc))
        conts.append(task.doc_to_targets(doc)[0])
    scores = scorer.loglikelihood(zip(ctxs, conts))
    correct = sum(int(is_greedy) for _, is_greedy in scores)
    total = len(scores)
    return TaskResult(
        model="",
        task=task.name,
        gamma=gamma,
        n=total,
        metrics={"acc": correct / total if total else float("nan")},
    )


@torch.no_grad()
def evaluate_generate_until(
    model,
    tokenizer,
    task: Task,
    gamma: float,
    limit: Optional[int] = None,
    uncond_prefix_tokens: int = 1,
    batch_size: int = 1,
    temperature: float = 0.0,
    seed: int = 1234,
    max_length: Optional[int] = None,
):
    """Generation task (TriviaQA): substring match of the generated answer."""
    docs = load_examples(task, limit=limit)
    correct = 0
    total = 0
    for i, doc in enumerate(docs):
        ctx = task.doc_to_text(doc)
        targets = task.doc_to_targets(doc)
        outputs = cfg_generate(
            model,
            tokenizer,
            ctx,
            gamma=gamma,
            uncond_prefix_tokens=uncond_prefix_tokens,
            max_new_tokens=task.max_gen_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            stop_sequences=task.stop_sequences,
            seed=seed + i,
        )
        if substring_match(outputs[0], targets):
            correct += 1
        total += 1
    return TaskResult(
        model="",
        task=task.name,
        gamma=gamma,
        n=total,
        metrics={"substring_match": correct / total if total else float("nan")},
    )


def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    gamma: float,
    limit: Optional[int] = None,
    uncond_prefix_tokens: int = 1,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    **kwargs,
) -> TaskResult:
    """Dispatch to the right evaluator for a task."""
    task = get_task(task_name) if isinstance(task_name, str) else task_name
    if task.name == "lambada_openai":
        return evaluate_lambada(
            model,
            tokenizer,
            task,
            gamma,
            limit=limit,
            uncond_prefix_tokens=uncond_prefix_tokens,
            batch_size=batch_size,
            max_length=max_length,
        )
    if task.output_type == "generate_until":
        return evaluate_generate_until(
            model, tokenizer, task, gamma, limit=limit, uncond_prefix_tokens=uncond_prefix_tokens
        )
    return evaluate_multiple_choice(
        model,
        tokenizer,
        task,
        gamma,
        limit=limit,
        uncond_prefix_tokens=uncond_prefix_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )


def evaluate_suite(
    model,
    tokenizer,
    model_name: str,
    gamma: float,
    tasks: Sequence[str] = tuple(TABLE5_TASKS),
    limit: Optional[int] = None,
    **kwargs,
) -> Dict[str, dict]:
    """Evaluate every task of the suite at one guidance strength."""
    out: Dict[str, dict] = {}
    for name in tasks:
        result = evaluate_task(model, tokenizer, name, gamma, limit=limit, **kwargs)
        result.model = model_name
        out[name] = result.to_dict()
        print(
            f"[cfglm] {model_name} | {name} | gamma={gamma} | "
            f"{ {k: round(v, 4) for k, v in result.metrics.items()} }"
        )
    return out


def save_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


def load_results(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)
