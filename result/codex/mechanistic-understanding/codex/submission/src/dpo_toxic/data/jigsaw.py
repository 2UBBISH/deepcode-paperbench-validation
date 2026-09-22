"""The Jigsaw toxic comment classification dataset (Section 3.1).

The paper trains the toxicity probe on the 561,808 comments of the Jigsaw
"Toxic Comment Classification Challenge".  We read the HuggingFace mirror
``thesofakillers/jigsaw-toxic-comment-classification-challenge`` (recommended in
the addendum) and collapse the six binary labels into a single toxic/non-toxic
label, exactly as a binary classification probe requires.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

JIGSAW_HF_ID = "thesofakillers/jigsaw-toxic-comment-classification-challenge"
JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_jigsaw(hf_id: str = JIGSAW_HF_ID, cache_dir: Optional[str] = None):
    from datasets import load_dataset

    return load_dataset(hf_id, cache_dir=cache_dir)


def _is_toxic(example: Dict) -> bool:
    for label in JIGSAW_LABELS:
        value = example.get(label, 0)
        # Some mirrors store missing labels as None (the public test split of
        # the original challenge has unlabelled rows).
        if value is not None and int(value) == 1:
            return True
    return False


def build_jigsaw_dataset(hf_id: str = JIGSAW_HF_ID,
                         cache_dir: Optional[str] = None,
                         max_train: Optional[int] = None,
                         max_val: Optional[int] = None,
                         val_fraction: float = 0.1,
                         seed: int = 0,
                         splits: Optional[Tuple[str, ...]] = None) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, Dict]:
    """Return ``(train_texts, train_labels, val_texts, val_labels, stats)``.

    The paper uses a 90:10 train/validation split.  ``max_train``/``max_val``
    cap the number of examples (useful for CPU smoke tests) and are applied
    after the split so that the small-scale runs stay faithful to the protocol.

    The HuggingFace mirror exposes the labelled challenge data as ``train``
    (159,571 comments) and ``test`` (306,328 comments, i.e. the 153,164 public
    test comments with their labels).  We concatenate the splits by default --
    the paper describes using the full 561,808-comment release, and the
    remaining discrepancy is documented in ``README.md`` (the mirror does not
    contain the 95,909 rows of the original release).
    """
    ds = load_jigsaw(hf_id=hf_id, cache_dir=cache_dir)
    if splits is None:
        splits = tuple(ds.keys())
    texts: List[str] = []
    labels: List[int] = []
    seen = set()
    for split_name in splits:
        split = ds[split_name]
        for ex in split:
            key = ex.get("id", ex["comment_text"])
            if key in seen:
                continue
            seen.add(key)
            texts.append(ex["comment_text"])
            labels.append(1 if _is_toxic(ex) else 0)
    labels_arr = np.array(labels, dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(texts))
    n_val = int(round(val_fraction * len(texts)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if max_train is not None:
        train_idx = train_idx[:max_train]
    if max_val is not None:
        val_idx = val_idx[:max_val]

    train_texts = [texts[int(i)] for i in train_idx]
    val_texts = [texts[int(i)] for i in val_idx]
    train_labels, val_labels = labels_arr[train_idx], labels_arr[val_idx]
    stats = {
        "n_total": int(len(texts)),
        "splits": list(splits),
        "n_train": int(len(train_texts)),
        "n_val": int(len(val_texts)),
        "toxic_rate_train": float(train_labels.mean()),
        "toxic_rate_val": float(val_labels.mean()),
    }
    return train_texts, train_labels, val_texts, val_labels, stats
