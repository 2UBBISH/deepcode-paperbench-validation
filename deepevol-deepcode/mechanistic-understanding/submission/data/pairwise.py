"""Pairwise preference dataset container for the PPLM/DPO reproduction.

The paper (Section 4.2) builds the preference dataset with PPLM:

    "To generate pairwise preference data, we use sentences from Wikitext-2 as
     prompts. For each prompt, we generate a positive sample using greedy
     sampling with GPT2, while using PPLM to generate negative (toxic) samples.
     We use our toxic probe W_Toxic as our attribute classifier to guide towards
     toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations."

Each pair therefore has the structure ``(prompt, preferred=nontoxic greedy
continuation, non_preferred=PPLM toxic continuation)``.  A 90:10 split is used
for training and validation (Section 4.1 `;` the trainer validates on
``loss/valid`` with patience 10).

Because full pair generation is expensive, this module also provides resumable
sharded JSONL persistence: generated pairs are appended to numbered shards so a
run can be interrupted and restarted without losing work.
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of pairs created by the paper (Section 4.2).
N_PAIRS: int = 24576

#: Train/validation split ratio used throughout the reproduction.
VALID_RATIO: float = 0.1

#: Default artifact location for the generated pairs.
DEFAULT_PAIRS_PATH: str = os.path.join("artifacts", "data", "pairs.jsonl")

#: Default directory for resumable shards.
DEFAULT_SHARD_DIR: str = os.path.join("artifacts", "data", "pairs_shards")

#: Shard size used when writing pairs incrementally.
SHARD_SIZE: int = 512

#: Fields accepted when reading pairs from raw dictionaries / HF datasets.
PROMPT_KEYS = ("prompt", "question", "input", "text")
CHOSEN_KEYS = ("chosen", "preferred", "positive", "nontoxic", "non_toxic")
REJECTED_KEYS = ("rejected", "non_preferred", "negative", "toxic")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PairExample:
    """A single preference pair.

    Parameters
    ----------
    prompt:
        The Wikitext-2 sentence used as the generation prompt.
    preferred:
        Non-toxic continuation produced by greedy sampling with GPT2
        (the "chosen"/positive sample).
    non_preferred:
        Toxic continuation produced by PPLM guided with ``W_Toxic``
        (the "rejected"/negative sample).
    index:
        Position of the pair in the generated dataset (-1 if unknown).
    source:
        Optional provenance marker (e.g. ``"pplm"``).
    prompt_toxicity / preferred_toxicity / non_preferred_toxicity:
        Optional diagnostic scores if the generator computed them.
    meta:
        Free-form extra metadata.
    """

    prompt: str
    preferred: str
    non_preferred: str
    index: int = -1
    source: str = "pplm"
    prompt_toxicity: float = float("nan")
    preferred_toxicity: float = float("nan")
    non_preferred_toxicity: float = float("nan")
    meta: Dict = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------
    def is_valid(self) -> bool:
        """Return True when both continuations are non-empty and distinct."""
        return bool(
            self.prompt is not None
            and self.preferred
            and self.non_preferred
            and self.preferred.strip()
            and self.non_preferred.strip()
            and self.preferred != self.non_preferred
        )

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "prompt": self.prompt,
            "preferred": self.preferred,
            "non_preferred": self.non_preferred,
            "chosen": self.preferred,
            "rejected": self.non_preferred,
            "source": self.source,
            "prompt_toxicity": _nan_safe(self.prompt_toxicity),
            "preferred_toxicity": _nan_safe(self.preferred_toxicity),
            "non_preferred_toxicity": _nan_safe(self.non_preferred_toxicity),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PairExample":
        """Build a pair from a raw dictionary, tolerating alternative schemas."""
        prompt = _first_key(d, PROMPT_KEYS)
        preferred = _first_key(d, CHOSEN_KEYS)
        non_preferred = _first_key(d, REJECTED_KEYS)
        if prompt is None or preferred is None or non_preferred is None:
            raise ValueError(
                "pair dictionary must contain prompt/preferred/non_preferred "
                f"(got keys {sorted(d.keys())})"
            )
        return cls(
            prompt=str(prompt),
            preferred=str(preferred),
            non_preferred=str(non_preferred),
            index=int(d.get("index", -1)),
            source=str(d.get("source", "pplm")),
            prompt_toxicity=float(d.get("prompt_toxicity", float("nan"))),
            preferred_toxicity=float(d.get("preferred_toxicity", float("nan"))),
            non_preferred_toxicity=float(d.get("non_preferred_toxicity", float("nan"))),
            meta=dict(d.get("meta", {}) or {}),
        )

    def to_hf(self) -> Dict[str, str]:
        """Return the minimal (prompt, chosen, rejected) triple used by the trainer."""
        return {
            "prompt": self.prompt,
            "chosen": self.preferred,
            "rejected": self.non_preferred,
        }


@dataclass
class PairSplit:
    """A named subset of pairs (train or validation)."""

    pairs: List[PairExample] = field(default_factory=list)
    name: str = "train"

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[PairExample]:
        return iter(self.pairs)

    def __getitem__(self, item):
        return self.pairs[item]

    @property
    def prompts(self) -> List[str]:
        return [p.prompt for p in self.pairs]

    @property
    def chosen(self) -> List[str]:
        return [p.preferred for p in self.pairs]

    @property
    def rejected(self) -> List[str]:
        return [p.non_preferred for p in self.pairs]

    def subset(self, n: int, seed: int = 0) -> "PairSplit":
        """Deterministic sub-sample of this split."""
        if n >= len(self.pairs):
            return PairSplit(list(self.pairs), name=self.name)
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(self.pairs)), n))
        return PairSplit([self.pairs[i] for i in idx], name=self.name)

    def to_hf(self) -> List[Dict[str, str]]:
        return [p.to_hf() for p in self.pairs]

    def to_dicts(self) -> List[Dict]:
        return [p.to_dict() for p in self.pairs]


@dataclass
class PairwiseDataset:
    """Train/validation container for the preference pairs.

    Mirrors the layout of :class:`data.jigsaw.JigsawData` so downstream code can
    treat both datasets uniformly.
    """

    train: PairSplit = field(default_factory=lambda: PairSplit(name="train"))
    valid: PairSplit = field(default_factory=lambda: PairSplit(name="valid"))

    def __len__(self) -> int:
        return len(self.train) + len(self.valid)

    def as_arrays(self) -> Tuple[List[PairExample], List[PairExample]]:
        return list(self.train.pairs), list(self.valid.pairs)

    @property
    def all_pairs(self) -> List[PairExample]:
        return list(self.train.pairs) + list(self.valid.pairs)

    def stats(self) -> Dict[str, float]:
        """Basic diagnostics about the dataset."""
        return {
            "n_total": len(self),
            "n_train": len(self.train),
            "n_valid": len(self.valid),
            "valid_ratio": (len(self.valid) / len(self)) if len(self) else 0.0,
            "mean_prompt_chars": _mean_len([p.prompt for p in self.all_pairs]),
            "mean_chosen_chars": _mean_len([p.preferred for p in self.all_pairs]),
            "mean_rejected_chars": _mean_len([p.non_preferred for p in self.all_pairs]),
        }


# ---------------------------------------------------------------------------
# Construction / splitting
# ---------------------------------------------------------------------------


def stratified_split(
    pairs: Sequence[PairExample],
    valid_ratio: float = VALID_RATIO,
    seed: int = 0,
) -> PairwiseDataset:
    """Split pairs into a 90:10 train/validation split.

    The paper uses a 90:10 split for its data; we apply the same convention to
    the preference pairs.  Pairs whose preferred and non-preferred continuations
    are identical (a failed PPLM generation) are dropped, since they carry no
    preference signal.
    """
    kept = [p for p in pairs if p.is_valid()]
    if not kept:
        return PairwiseDataset(PairSplit([], "train"), PairSplit([], "valid"))

    rng = random.Random(seed)
    order = list(range(len(kept)))
    rng.shuffle(order)

    n_valid = int(round(len(kept) * valid_ratio))
    n_valid = max(0, min(n_valid, len(kept) - 1)) if len(kept) > 1 else 0

    valid_idx = set(order[:n_valid])
    train = [kept[i] for i in range(len(kept)) if i not in valid_idx]
    valid = [kept[i] for i in range(len(kept)) if i in valid_idx]

    # Re-index deterministically so the saved artifact is self-describing.
    for i, p in enumerate(train):
        p.index = i
    for i, p in enumerate(valid):
        p.index = len(train) + i

    return PairwiseDataset(PairSplit(train, "train"), PairSplit(valid, "valid"))


def build_dataset(
    pairs: Iterable[PairExample],
    valid_ratio: float = VALID_RATIO,
    seed: int = 0,
    max_pairs: Optional[int] = None,
) -> PairwiseDataset:
    """Convenience wrapper: materialise an iterable of pairs and split it."""
    materialised = list(pairs)
    if max_pairs is not None and max_pairs > 0:
        materialised = materialised[:max_pairs]
    return stratified_split(materialised, valid_ratio=valid_ratio, seed=seed)


def make_pair(
    prompt: str,
    preferred: str,
    non_preferred: str,
    index: int = -1,
    **kwargs,
) -> PairExample:
    """Small factory used by the generation script."""
    return PairExample(
        prompt=prompt,
        preferred=preferred,
        non_preferred=non_preferred,
        index=index,
        **kwargs,
    )


def deduplicate(pairs: Sequence[PairExample]) -> List[PairExample]:
    """Drop repeated (prompt, preferred, non_preferred) triples, preserving order."""
    seen = set()
    out: List[PairExample] = []
    for p in pairs:
        key = (p.prompt, p.preferred, p.non_preferred)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_pairs(pairs: Sequence[PairExample], path: str = DEFAULT_PAIRS_PATH) -> str:
    """Write pairs to a JSONL file (one pair per line)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_pairs(path: str = DEFAULT_PAIRS_PATH) -> List[PairExample]:
    """Load pairs from a JSONL file produced by :func:`save_pairs`."""
    pairs: List[PairExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pairs.append(PairExample.from_dict(json.loads(line)))
    return pairs


def load_pairwise_dataset(
    path: str = DEFAULT_PAIRS_PATH,
    valid_ratio: float = VALID_RATIO,
    seed: int = 0,
) -> PairwiseDataset:
    """Load pairs from disk and re-create the deterministic train/valid split."""
    return stratified_split(load_pairs(path), valid_ratio=valid_ratio, seed=seed)


# ---------------------------------------------------------------------------
# Resumable sharded writing (used by scripts/generate_pairs.py)
# ---------------------------------------------------------------------------


def shard_path(shard_dir: str = DEFAULT_SHARD_DIR, shard_id: int = 0) -> str:
    """Path of a numbered shard file."""
    return os.path.join(shard_dir, f"pairs-{shard_id:05d}.jsonl")


def existing_shards(shard_dir: str = DEFAULT_SHARD_DIR) -> List[str]:
    """All shard files present on disk, in numeric order."""
    paths = glob.glob(os.path.join(shard_dir, "pairs-*.jsonl"))
    return sorted(paths)


def count_existing_pairs(shard_dir: str = DEFAULT_SHARD_DIR) -> int:
    """Number of pairs already generated (used to resume pair generation)."""
    total = 0
    for path in existing_shards(shard_dir):
        total += sum(1 for line in open(path, "r", encoding="utf-8") if line.strip())
    return total


def append_pairs(
    pairs: Sequence[PairExample],
    shard_dir: str = DEFAULT_SHARD_DIR,
    shard_size: int = SHARD_SIZE,
) -> List[str]:
    """Append pairs to numbered shards, creating new shards as needed.

    Returns the list of shard files touched.  This makes generation resumable:
    restarting a run simply continues from the shard with the lowest free id.
    """
    os.makedirs(shard_dir, exist_ok=True)
    written: List[str] = []
    existing = existing_shards(shard_dir)

    # Continue filling the last (possibly partial) shard first.
    pending = list(pairs)
    if existing:
        last = existing[-1]
        n_in_last = sum(1 for line in open(last, "r", encoding="utf-8") if line.strip())
        if n_in_last < shard_size and pending:
            take = pending[: shard_size - n_in_last]
            with open(last, "a", encoding="utf-8") as fh:
                for p in take:
                    fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            pending = pending[len(take) :]
            written.append(last)
        next_id = _shard_id(existing[-1]) + 1
    else:
        next_id = 0

    while pending:
        chunk = pending[:shard_size]
        pending = pending[len(chunk) :]
        path = shard_path(shard_dir, next_id)
        with open(path, "w", encoding="utf-8") as fh:
            for p in chunk:
                fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        written.append(path)
        next_id += 1

    return written


def load_shards(shard_dir: str = DEFAULT_SHARD_DIR) -> List[PairExample]:
    """Load every shard in a directory into one pair list."""
    pairs: List[PairExample] = []
    for path in existing_shards(shard_dir):
        pairs.extend(load_pairs(path))
    return pairs


def merge_shards(
    shard_dir: str = DEFAULT_SHARD_DIR,
    out_path: str = DEFAULT_PAIRS_PATH,
) -> str:
    """Merge all shards into a single JSONL artifact."""
    return save_pairs(load_shards(shard_dir), out_path)


# ---------------------------------------------------------------------------
# HuggingFace-compatible access
# ---------------------------------------------------------------------------


def to_hf_dataset(dataset: PairwiseDataset, split: Optional[str] = None):
    """Convert pairs to a `datasets.Dataset` with prompt/chosen/rejected columns.

    ``datasets`` is imported lazily so this module stays import-safe offline.
    """
    from datasets import Dataset  # noqa: WPS433 (lazy import)

    if split == "train":
        rows = dataset.train.to_hf()
    elif split in ("valid", "validation"):
        rows = dataset.valid.to_hf()
    else:
        rows = dataset.train.to_hf() + dataset.valid.to_hf()
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_key(d: Dict, keys: Sequence[str]):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _nan_safe(x) -> Optional[float]:
    try:
        import math

        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean_len(strings: Sequence[str]) -> float:
    if not strings:
        return 0.0
    return float(sum(len(s) for s in strings) / len(strings))


def _shard_id(path: str) -> int:
    base = os.path.basename(path)
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else 0


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    demo = [
        make_pair("The cat sat on", " the mat and slept.", " the f*cking mat, damn it.", index=i)
        for i in range(20)
    ]
    ds = build_dataset(demo, valid_ratio=0.1, seed=0)
    print("stats:", ds.stats())
    tmp_dir = os.path.join("artifacts", "tmp_pairs_shards")
    append_pairs(demo, tmp_dir, shard_size=8)
    print("shards:", existing_shards(tmp_dir), "count:", count_existing_pairs(tmp_dir))
    print("round-trip:", len(load_shards(tmp_dir)))
