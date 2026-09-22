"""Jigsaw toxic comment dataset loader.

Paper (Sec. 3.1):
    "we use the Jigsaw toxic comment classification dataset (cjadams et al., 2017),
     which consists of 561,808 comments, each of which is labeled as toxic or
     non-toxic. We use a 90:10 split for training and validation."

The original paper predicts the 6 Jigsaw subtask labels independently; a comment
is counted as toxic if *any* of the six labels is 1 ("such that if any of the
six labels are 1, then the comment is toxic").  That yields 561,808 comments with
binary toxic / non-toxic labels (Section 3.1 / author clarifications).

Substitution (documented in README): the HuggingFace mirror
``thesofakillers/jigsaw-toxic-comment-classification-challenge`` is used instead
of the original Kaggle download.

This module only depends on ``datasets`` (lazily imported) and numpy/pytorch so
that import-time never touches the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

JIGSAW_HF_NAME = "thesofakillers/jigsaw-toxic-comment-classification-challenge"

# The six Jigsaw subtask labels (cjadams et al., 2017).
JIGSAW_LABELS: Tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

TEXT_COLUMN_CANDIDATES: Tuple[str, ...] = ("comment_text", "text", "comment")
LABEL_COLUMN_CANDIDATES: Tuple[str, ...] = ("label", "labels", "target")

N_TOTAL_COMMENTS = 561_808  # reported in Section 3.1


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def binarise_labels(example: Dict) -> int:
    """Return 1 if *any* Jigsaw subtask label is positive, else 0."""
    # Case 1: a single scalar label is already provided.
    for col in LABEL_COLUMN_CANDIDATES:
        if col in example:
            value = example[col]
            if isinstance(value, (list, tuple, np.ndarray)):
                return int(any(float(v) > 0.5 for v in value))
            return int(float(value) > 0.5)
    # Case 2: the six individual column labels (original Kaggle schema).
    present = [c for c in JIGSAW_LABELS if c in example]
    if present:
        return int(any(float(example[c]) > 0.5 for c in present))
    raise KeyError(
        f"Could not find any label column; example keys = {list(example.keys())}"
    )


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
@dataclass
class JigsawSplit:
    """A split of the (binarised) Jigsaw dataset."""

    texts: List[str] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def toxic_fraction(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return float(self.labels.mean())

    def subset(self, n: int, seed: int = 0) -> "JigsawSplit":
        """Deterministically take a random subset (handy for smoke tests)."""
        if n >= len(self):
            return self
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(self))[:n]
        return JigsawSplit([self.texts[i] for i in idx], self.labels[idx])


@dataclass
class JigsawData:
    """Train / validation container produced by :func:`load_jigsaw`."""

    train: JigsawSplit
    valid: JigsawSplit

    def as_arrays(self) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
        return (
            self.train.texts,
            self.train.labels,
            self.valid.texts,
            self.valid.labels,
        )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def stratified_split(
    texts: Sequence[str],
    labels: np.ndarray,
    valid_ratio: float = 0.1,
    seed: int = 0,
) -> JigsawData:
    """90:10 (default) stratified split of binarised comments."""
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    texts = list(texts)
    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) differ")

    rng = np.random.RandomState(seed)
    train_idx: List[int] = []
    valid_idx: List[int] = []
    for cls in sorted(set(labels.tolist())):
        cls_idx = np.where(labels == cls)[0]
        perm = rng.permutation(len(cls_idx))
        n_valid = int(round(valid_ratio * len(cls_idx)))
        valid_idx.extend(cls_idx[perm[:n_valid]].tolist())
        train_idx.extend(cls_idx[perm[n_valid:]].tolist())

    train_idx = sorted(train_idx)
    valid_idx = sorted(valid_idx)
    train = JigsawSplit([texts[i] for i in train_idx], labels[train_idx])
    valid = JigsawSplit([texts[i] for i in valid_idx], labels[valid_idx])
    return JigsawData(train=train, valid=valid)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_hf_splits(cache_dir: Optional[str] = None):
    from datasets import load_dataset  # lazy import

    # The mirror exposes the original Kaggle splits (train/test/...).
    try:
        ds = load_dataset(JIGSAW_HF_NAME, cache_dir=cache_dir)
    except Exception:
        ds = load_dataset(JIGSAW_HF_NAME, "default", cache_dir=cache_dir)
    return ds


def load_jigsaw_raw(
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    seed: int = 0,
) -> Tuple[List[str], np.ndarray]:
    """Load the raw mirror and return binarised (texts, labels).

    ``limit`` optionally subsamples the dataset (smoke tests / CI).
    """
    ds = _load_hf_splits(cache_dir=cache_dir)

    texts: List[str] = []
    labels: List[int] = []
    # Prefer the train split; concatenate others if the mirror is sharded.
    split_names = [s for s in ("train", "validation", "test") if s in ds]
    if not split_names:
        split_names = list(ds.keys())

    for split_name in split_names:
        split = ds[split_name]
        cols = split.column_names
        text_col = _first_existing(cols, TEXT_COLUMN_CANDIDATES)
        if text_col is None:
            raise KeyError(
                f"No text column found in split '{split_name}'; columns = {cols}"
            )
        for example in split:
            texts.append(str(example[text_col]))
            labels.append(binarise_labels(example))

    labels_arr = np.asarray(labels, dtype=np.int64)

    if limit is not None and limit < len(texts):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(texts))[:limit]
        texts = [texts[i] for i in idx]
        labels_arr = labels_arr[idx]

    return texts, labels_arr


def load_jigsaw(
    cache_dir: Optional[str] = None,
    valid_ratio: float = 0.1,
    seed: int = 0,
    limit: Optional[int] = None,
) -> JigsawData:
    """Load Jigsaw, binarise labels and return a 90:10 train/valid split."""
    texts, labels = load_jigsaw_raw(cache_dir=cache_dir, limit=limit, seed=seed)
    return stratified_split(texts, labels, valid_ratio=valid_ratio, seed=seed)


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------
def tokenize_comments(
    tokenizer,
    texts: Sequence[str],
    max_length: int = 128,
    device: str = "cpu",
):
    """Tokenise comments, padded and truncated, ready for a probe forward pass."""
    import torch  # lazy import

    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def iter_batches(
    tokenizer,
    texts: Sequence[str],
    labels: np.ndarray,
    batch_size: int = 256,
    max_length: int = 128,
    device: str = "cpu",
    shuffle: bool = True,
    seed: int = 0,
):
    """Yield ``(encodings, labels_tensor)`` batches of tokenised comments."""
    import torch  # lazy import

    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    order = np.arange(len(texts))
    if shuffle:
        order = np.random.RandomState(seed).permutation(order)

    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        batch_texts = [texts[i] for i in chunk]
        enc = tokenize_comments(
            tokenizer, batch_texts, max_length=max_length, device=device
        )
        yield enc, torch.as_tensor(labels[chunk], device=device)


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
    data = load_jigsaw(limit=64)
    print(
        f"train={len(data.train)} valid={len(data.valid)} "
        f"toxic_frac={data.train.toxic_fraction:.3f}"
    )
    batch_enc, batch_y = next(
        iter_batches(tok, data.train.texts, data.train.labels, batch_size=4)
    )
    print("batch:", {k: tuple(v.shape) for k, v in batch_enc.items()}, batch_y.shape)
