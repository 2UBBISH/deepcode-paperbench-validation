"""Zero-shot task definitions mirroring EleutherAI's LM Evaluation Harness.

Section 3.1 of the paper evaluates GPT-2, Pythia and LLaMA models on the
zero-shot benchmarks of the LM Evaluation Harness:

    ARC-challenge, ARC-easy, BoolQ, HellaSwag, PIQA, SciQ, TriviaQA,
    WinoGrande, LAMBADA (OpenAI)

Each :class:`Task` below reproduces the harness's zero-shot configuration:
its prompt template, its candidate continuations and its metric.  The
``delimiter`` between the context and a continuation follows the harness:
``" "`` for multiple-choice tasks (the harness prepends a space to every
continuation) and ``""`` when the continuation is built from the raw text
(HellaSwag, LAMBADA).

Two backends are supported:

* ``lm_eval`` -- if ``lm-evaluation-harness`` is installed, the canonical
  task objects are used directly (see ``cfglm/lm_eval_adapter.py``);
* ``native`` -- the definitions in this module, which only require
  ``datasets`` and are used by ``cfglm/harness.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class Task:
    """A zero-shot benchmark."""

    name: str
    dataset_path: str
    dataset_name: Optional[str] = None
    split: str = "test"
    output_type: str = "multiple_choice"  # or "generate_until"
    delimiter: str = " "
    description: str = ""
    # callables
    doc_to_text: Callable[[dict], str] = None
    doc_to_choices: Callable[[dict], List[str]] = None
    doc_to_gold: Callable[[dict], int] = None
    # for generation tasks
    doc_to_targets: Callable[[dict], List[str]] = None
    max_gen_tokens: int = 16
    stop_sequences: Sequence[str] = ()
    metric: str = "acc"
    metric_kwargs: Dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# individual tasks
# ----------------------------------------------------------------------
def _arc_text(doc: dict) -> str:
    return f"Question: {doc['question']}\nAnswer:"


def _arc_choices(doc: dict) -> List[str]:
    return [f" {text}" for text in doc["choices"]["text"]]


def _arc_gold(doc: dict) -> int:
    labels = doc["choices"]["label"]
    answer = doc["answerKey"]
    if answer in labels:
        return labels.index(answer)
    # ARC occasionally stores the answer key as "1".."4" while the labels are
    # "A".."D"; fall back to numeric conversion.
    return int(str(answer)) - 1


ARC_CHALLENGE = Task(
    name="arc_challenge",
    dataset_path="allenai/ai2_arc",
    dataset_name="ARC-Challenge",
    doc_to_text=_arc_text,
    doc_to_choices=_arc_choices,
    doc_to_gold=_arc_gold,
    description="AI2 Reasoning Challenge (challenge set), 0-shot.",
)

ARC_EASY = Task(
    name="arc_easy",
    dataset_path="allenai/ai2_arc",
    dataset_name="ARC-Easy",
    doc_to_text=_arc_text,
    doc_to_choices=_arc_choices,
    doc_to_gold=_arc_gold,
    description="AI2 Reasoning Challenge (easy set), 0-shot.",
)

BOOLQ = Task(
    name="boolq",
    dataset_path="google/boolq",
    split="validation",
    doc_to_text=lambda d: f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:",
    doc_to_choices=lambda d: [" yes", " no"],
    doc_to_gold=lambda d: 0 if bool(d["answer"]) else 1,
    description="BoolQ yes/no questions, 0-shot.",
)


def _hellaswag_choices(doc: dict) -> List[str]:
    return [f" {ending}" for ending in doc["endings"]]


HELLASWAG = Task(
    name="hellaswag",
    dataset_path="Rowan/hellaswag",
    split="validation",
    doc_to_text=lambda d: d["ctx"].strip(),
    doc_to_choices=_hellaswag_choices,
    doc_to_gold=lambda d: int(d["label"]),
    description="HellaSwag sentence completion, 0-shot.",
)

PIQA = Task(
    name="piqa",
    dataset_path="ybisk/piqa",
    split="validation",
    doc_to_text=lambda d: f"Question: {d['goal']}\nAnswer:",
    doc_to_choices=lambda d: [f" {d['sol1']}", f" {d['sol2']}"],
    doc_to_gold=lambda d: int(d["label"]),
    description="PIQA physical commonsense, 0-shot.",
)

SCIQ = Task(
    name="sciq",
    dataset_path="allenai/sciq",
    doc_to_text=lambda d: f"Question: {d['question']}\nAnswer:",
    doc_to_choices=lambda d: [
        f" {d['correct_answer']}",
        f" {d['distractor1']}",
        f" {d['distractor2']}",
        f" {d['distractor3']}",
    ],
    doc_to_gold=lambda d: 0,
    description="SciQ science questions, 0-shot.",
)

WINOGRANDE = Task(
    name="winogrande",
    dataset_path="allenai/winogrande",
    dataset_name="winogrande_xl",
    split="validation",
    # Partial scoring: the prefix before the blank is the context and the
    # candidate option is the continuation (the harness's WinoGrande setup).
    doc_to_text=lambda d: d["sentence"].split("_")[0],
    doc_to_choices=lambda d: [d["option1"], d["option2"]],
    doc_to_gold=lambda d: int(d["answer"]) - 1,
    delimiter="",
    description="WinoGrande coreference, 0-shot.",
)

LAMBADA = Task(
    name="lambada_openai",
    dataset_path="EleutherAI/lambada_openai",
    doc_to_text=lambda d: d["text"].rsplit(" ", 1)[0],
    doc_to_choices=None,  # single target; handled by doc_to_targets below
    doc_to_targets=lambda d: [" " + d["text"].rsplit(" ", 1)[1]],
    doc_to_gold=lambda d: 0,
    delimiter="",
    metric="acc_greedy",
    description="LAMBADA (OpenAI) last-word prediction, 0-shot.",
)

TRIVIAQA = Task(
    name="triviaqa",
    dataset_path="mandarjoshi/trivia_qa",
    dataset_name="rc.nocontext",
    split="validation",
    output_type="generate_until",
    doc_to_text=lambda d: f"Question: {d['question']}\nAnswer:",
    doc_to_targets=lambda d: list(d["answer"]["aliases"]) + [d["answer"]["value"]],
    max_gen_tokens=16,
    stop_sequences=("\n", "Question:"),
    metric="substring_match",
    description=(
        "TriviaQA, 0-shot.  Follows the LLaMA methodology but uses substring "
        "match instead of exact match (paper addendum)."
    ),
)


TASKS: Dict[str, Task] = {
    t.name: t
    for t in (ARC_CHALLENGE, ARC_EASY, BOOLQ, HELLASWAG, PIQA, SCIQ, WINOGRANDE, LAMBADA, TRIVIAQA)
}

# The order used in Table 5 / Figures 6-9 of the paper.
TABLE5_TASKS: List[str] = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "piqa",
    "sciq",
    "triviaqa",
    "winogrande",
    "lambada_openai",
]


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]


def load_examples(task: Task, limit: Optional[int] = None, split: Optional[str] = None):
    """Load the documents of a task with the HuggingFace ``datasets`` library."""
    from datasets import load_dataset

    if task.dataset_name:
        ds = load_dataset(task.dataset_path, task.dataset_name, split=split or task.split)
    else:
        ds = load_dataset(task.dataset_path, split=split or task.split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
