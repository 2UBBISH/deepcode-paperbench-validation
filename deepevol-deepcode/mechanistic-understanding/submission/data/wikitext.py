"""Wikitext-2 loader.

The paper uses Wikitext-2 (Merity et al., 2016) for two distinct purposes:

1. **PPLM prompts** -- sentences from Wikitext-2 are used as prompts for which we
   generate a *positive* (non-toxic) continuation with greedy GPT2 sampling and a
   *negative* (toxic) continuation with PPLM.  ``We use sentences from Wikitext-2
   (Merity et al., 2016) as prompts.`` (Section 4.2)
2. **Perplexity corpus** -- ``we also follow prior work (Geva et al., 2022) and measure
   perplexity on the Wikitext-2 dataset (Merity et al., 2016)`` (Section 3.3).

Additionally the F1 metric uses ``2,000 Wikipedia sentences as prompts``; the helper
:func:`wiki_sentences_for_f1` provides those (they are consumed by ``src/eval/f1.py``).

The dataset is loaded from the HuggingFace hub (``Salesforce/wikitext``,
config ``wikitext-2-raw-v1``).  Everything is lazily imported so that merely importing
this module never touches the network.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

WIKITEXT2_HF_NAME = "Salesforce/wikitext"
WIKITEXT2_CONFIG = "wikitext-2-raw-v1"

#: Number of Wikitext-2 sentences used by the F1 metric (Section 3.3).
N_F1_SENTENCES = 2_000

#: Number of preferred/non-preferred pairs created in Section 4.2.
N_PAIRS = 24_576

# Heuristic sentence splitter: split on ., !, ? followed by whitespace / end of line.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------------------
# Raw loading
# --------------------------------------------------------------------------------------


def _load_hf_splits(cache_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """Load raw Wikitext-2 splits from the HuggingFace hub.

    Returns a mapping ``{"train": [...], "validation": [...], "test": [...]}`` of the raw
    lines, so that both the sentence-level and the corpus-level consumers can be served
    from a single download.
    """
    from datasets import load_dataset

    ds = load_dataset(
        WIKITEXT2_HF_NAME,
        WIKITEXT2_CONFIG,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    splits: Dict[str, List[str]] = {}
    for name in ds.keys():
        col = "text" if "text" in ds[name].column_names else ds[name].column_names[0]
        splits[name] = [t for t in ds[name][col] if isinstance(t, str)]
    # Mirror HF naming used elsewhere in the code base.
    if "validation" in splits:
        splits["valid"] = splits["validation"]
    return splits


def load_wikitext2_raw(cache_dir: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, List[str]]:
    """Load each Wikitext-2 split as a list of raw text blocks (one per line)."""
    splits = _load_hf_splits(cache_dir=cache_dir)
    if limit is not None:
        splits = {k: v[:limit] for k, v in splits.items()}
    return splits


# --------------------------------------------------------------------------------------
# Sentence extraction (PPLM prompts / F1 prompts)
# --------------------------------------------------------------------------------------


def iter_sentences(text: str) -> Iterator[str]:
    """Yield cleaned sentences from a block of Wikitext-2 text.

    Wikitext-2 raw lines are usually already sentence-ish, but headings (``= Title =``)
    and empty lines are filtered out, and long lines are further split on sentence
    boundaries.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop section headings such as "= Plot =".
        if line.startswith("=") and line.endswith("="):
            continue
        # Skip lines that are mostly markup (tables / templates).
        if line.startswith(("{", "|", "[[Category:", "Category:")):
            continue
        for piece in _SENTENCE_SPLIT_RE.split(line):
            piece = _WHITESPACE_RE.sub(" ", piece).strip()
            if piece:
                yield piece


def split_sentences(texts: Sequence[str], min_chars: int = 40, min_words: int = 8) -> List[str]:
    """Split a sequence of raw Wikitext-2 blocks into usable prompt sentences.

    Args:
        texts: raw Wikitext-2 lines.
        min_chars: minimum number of characters for a sentence to be kept.
        min_words: minimum number of whitespace-separated words to be kept.

    Returns:
        The list of cleaned sentences, in order.
    """
    out: List[str] = []
    for text in texts:
        for sent in iter_sentences(text):
            if len(sent) < min_chars:
                continue
            if len(sent.split()) < min_words:
                continue
            out.append(sent)
    return out


def load_wikitext2_sentences(
    split: str = "train",
    cache_dir: Optional[str] = None,
    min_chars: int = 40,
    min_words: int = 8,
    limit: Optional[int] = None,
) -> List[str]:
    """Load cleaned Wikitext-2 sentences for a given split.

    ``split="train"`` (the default) yields well over the 24,576 prompts required by the
    pairwise dataset construction of Section 4.2.
    """
    splits = _load_hf_splits(cache_dir=cache_dir)
    if split not in splits:
        raise KeyError(f"Unknown Wikitext-2 split {split!r}; available: {sorted(splits)}")
    sents = split_sentences(splits[split], min_chars=min_chars, min_words=min_words)
    if limit is not None:
        sents = sents[:limit]
    return sents


def prompt_pool(
    n: Optional[int] = N_PAIRS,
    split: str = "train",
    seed: int = 0,
    cache_dir: Optional[str] = None,
    min_chars: int = 40,
    min_words: int = 8,
) -> List[str]:
    """Return ``n`` deterministic Wikitext-2 prompt sentences.

    The selection is a seeded random sample (without replacement) so that pair
    generation is reproducible across runs.  If fewer than ``n`` sentences are
    available, all of them are returned (the caller decides whether that is fatal).
    """
    import random

    sents = load_wikitext2_sentences(
        split=split, cache_dir=cache_dir, min_chars=min_chars, min_words=min_words
    )
    if n is None or n >= len(sents):
        return sents
    rng = random.Random(seed)
    return rng.sample(sents, n)


def wiki_sentences_for_f1(
    n: int = N_F1_SENTENCES,
    seed: int = 0,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """Return the ``n`` (default 2,000) Wikipedia sentences used as F1 prompts.

    Section 3.3: ``using 2,000 Wikipedia sentences as prompts, we measure the harmonic
    mean between precision and recall``.  Sentences are drawn deterministically from the
    Wikitext-2 test split (Wikipedia text), falling back to the training split when the
    test split does not contain enough material.
    """
    for split in ("test", "validation", "train"):
        try:
            sents = load_wikitext2_sentences(split=split, cache_dir=cache_dir)
        except Exception:
            continue
        if len(sents) >= n:
            import random

            return random.Random(seed).sample(sents, n)
    # Fall back to whatever we could gather from all splits.
    all_sents: List[str] = []
    for split in ("test", "validation", "train"):
        try:
            all_sents.extend(load_wikitext2_sentences(split=split, cache_dir=cache_dir))
        except Exception:
            continue
    import random

    rng = random.Random(seed)
    if n >= len(all_sents):
        return all_sents
    return rng.sample(all_sents, n)


# --------------------------------------------------------------------------------------
# Perplexity corpus
# --------------------------------------------------------------------------------------


def load_wikitext2_ppl_text(split: str = "test", cache_dir: Optional[str] = None) -> str:
    """Return the concatenated raw Wikitext-2 text used for perplexity evaluation.

    Following the common convention (and Geva et al., 2022), perplexity is measured on
    the Wikitext-2 *test* split.
    """
    splits = _load_hf_splits(cache_dir=cache_dir)
    if split not in splits:
        raise KeyError(f"Unknown Wikitext-2 split {split!r}; available: {sorted(splits)}")
    return "\n\n".join(t for t in splits[split] if t.strip())


def ppl_windows(
    token_ids: Sequence[int],
    seq_len: int = 1024,
    stride: Optional[int] = None,
) -> Iterator[List[int]]:
    """Yield (possibly strided) windows of ``seq_len`` tokens for perplexity scoring.

    The canonical sliding-window perplexity is computed by keeping the loss only on the
    newly-introduced tokens of each window; the consumer (``src/eval/perplexity.py``)
    receives the windows together with the number of ``context`` tokens to ignore.
    """
    ids = list(token_ids)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if stride is None:
        stride = seq_len
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(ids) <= seq_len:
        yield ids
        return
    for start in range(0, len(ids) - seq_len + 1, stride):
        yield ids[start : start + seq_len]
    # tail window aligned with the end of the corpus
    last_start = len(ids) - seq_len
    if (last_start % stride) != 0:
        yield ids[last_start:]


@dataclass
class PPLCorpus:
    """Tokenised Wikitext-2 corpus ready for perplexity evaluation."""

    input_ids: List[int]
    seq_len: int = 1024
    stride: Optional[int] = None
    split: str = "test"

    def windows(self) -> Iterator[List[int]]:
        return ppl_windows(self.input_ids, seq_len=self.seq_len, stride=self.stride)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.input_ids)


def load_ppl_corpus(
    tokenizer,
    split: str = "test",
    seq_len: int = 1024,
    stride: Optional[int] = None,
    cache_dir: Optional[str] = None,
    add_special_tokens: bool = False,
) -> PPLCorpus:
    """Load and tokenise the Wikitext-2 perplexity corpus.

    Args:
        tokenizer: a HuggingFace tokenizer (GPT2's tokenizer is expected to be
            un-bos'd, hence ``add_special_tokens=False`` by default).
        split: Wikitext-2 split to use (default ``"test"``).
        seq_len: context length of each perplexity window.
        stride: sliding-window stride; defaults to a non-overlapping window of ``seq_len``.
        cache_dir: optional HuggingFace cache directory.
    """
    text = load_wikitext2_ppl_text(split=split, cache_dir=cache_dir)
    enc = tokenizer(text, return_tensors=None, add_special_tokens=add_special_tokens)
    ids = enc["input_ids"]
    if isinstance(ids[0], list):  # batched tokenizers
        ids = ids[0]
    return PPLCorpus(input_ids=list(ids), seq_len=seq_len, stride=stride, split=split)


__all__ = [
    "WIKITEXT2_HF_NAME",
    "WIKITEXT2_CONFIG",
    "N_F1_SENTENCES",
    "N_PAIRS",
    "PPLCorpus",
    "load_wikitext2_raw",
    "iter_sentences",
    "split_sentences",
    "load_wikitext2_sentences",
    "prompt_pool",
    "wiki_sentences_for_f1",
    "load_wikitext2_ppl_text",
    "ppl_windows",
    "load_ppl_corpus",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    sents = load_wikitext2_sentences()
    print(f"wikitext-2 train sentences: {len(sents)}")
    print("example:", sents[0])
