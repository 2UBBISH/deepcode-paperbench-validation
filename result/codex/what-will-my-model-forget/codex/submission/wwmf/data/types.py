"""The single example format used by every component of the reproduction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List

from ..utils import read_jsonl, write_jsonl


@dataclass
class Example:
    """One ``<x, y>`` pair.

    ``input``  : the prompt ``x`` that is fed to the seq2seq model.
    ``target`` : the gold answer ``y`` (exact match is computed against it).
    ``task``   : T0 task name (e.g. ``glue-mrpc``) or MMLU subject.
    ``config`` : the exact P3 template config the example came from, when known.
    ``idx``    : a stable index inside ``config`` (used to cache logits).
    """

    input: str
    target: str
    task: str = ""
    config: str = ""
    idx: int = -1
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.config or self.task}#{self.idx}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Example":
        return cls(**{k: row[k] for k in ("input", "target", "task", "config", "idx", "meta") if k in row})


Dataset = List[Example]


def save_dataset(path: str, examples: Iterable[Example]) -> int:
    return write_jsonl(path, [e.to_dict() for e in examples])


def load_dataset_file(path: str) -> Dataset:
    return [Example.from_dict(row) for row in read_jsonl(path)]


def task_counts(examples: Iterable[Example]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        counts[ex.task] = counts.get(ex.task, 0) + 1
    return counts
