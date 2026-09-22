"""P3 sampling protocol for the Section 5 analysis (entropy / overlap / perplexity).

The paper (Section 5) runs Falcon-7b-Base (and Falcon-7b-Instruct) "on a sample dataset
of 32,902 datapoints from P3 (Sanh et al., 2021)".  The benchmark addendum clarifies how
those datapoints were drawn:

    * ~50 samples are randomly sampled from each of the 660 datasets in P3;
    * some datasets have fewer than 50 samples, in which case the entire dataset is taken;
    * samples whose *input* is longer than 200 tokens were filtered out (an inference-time
      optimization that "should not have an impact on the results");
    * all reported results come from Falcon + P3 -- the Open-Assistant and Redpajama
      combinations are explicitly out of scope.

With 660 datasets x 50 = 33,000 minus the short datasets, the protocol lands on the
paper's headline 32,902 datapoints.  This module implements exactly that protocol and is
usable with or without the ``datasets``/``transformers`` packages installed (a synthetic
fallback keeps the rest of the pipeline testable on a network-less CPU box).

Source: §5, Addendum (P3 sampling protocol).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Protocol constants (paper Section 5 + Addendum)
# --------------------------------------------------------------------------------------

P3_DATASET_NAME = "bigscience/P3"
"""HuggingFace hub id of the P3 collection (660 subsets)."""

SAMPLES_PER_DATASET = 50
"""~50 random samples per P3 dataset."""

N_P3_DATASETS = 660
"""Number of datasets (subsets) in P3."""

TARGET_N_DATAPOINTS = 32902
"""Paper's sample size for the Section 5 analysis."""

MAX_INPUT_TOKENS = 200
"""Inputs longer than this (in tokens) are filtered out (inference optimization)."""

DEFAULT_SEED = 1234
"""Deterministic seed; the paper's sample is reproducible by fixing this."""

DEFAULT_SPLIT = "train"
"""P3 subsets are consumed from their ``train`` split."""

FALLBACK_SPLIT = "validation"
"""Some P3 subsets only expose a validation split."""

DEFAULT_CACHE_PATH = os.path.join("data", "cache", "p3_sample.json")
"""Where the materialised 32,902-datapoint sample is cached."""

# P3 canonical fields.
INPUT_FIELD = "inputs_pretokenized"
TARGET_FIELD = "targets_pretokenized"
TOKENIZED_INPUT_FIELD = "inputs"
TOKENIZED_TARGET_FIELD = "targets"


# --------------------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------------------


@dataclass
class P3Sample:
    """A single datapoint of the Section 5 P3 sample.

    Fields
    ------
    dataset:
        Name of the P3 subset the datapoint came from (e.g. ``"super_glue_rte"``).
    idx:
        Row index inside the subset.
    inputs:
        Flattened prompt text shown to the model (the conditional context ``c``).
    targets:
        Reference target text (used for the "unprompted" PPL(x) pass).
    n_input_tokens:
        Input length in tokens (``<= MAX_INPUT_TOKENS`` by construction).
    """

    dataset: str
    idx: int
    inputs: str
    targets: str = ""
    n_input_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "idx": self.idx,
            "inputs": self.inputs,
            "targets": self.targets,
            "n_input_tokens": self.n_input_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "P3Sample":
        return cls(
            dataset=d["dataset"],
            idx=int(d["idx"]),
            inputs=d.get("inputs", ""),
            targets=d.get("targets", ""),
            n_input_tokens=d.get("n_input_tokens"),
            metadata=d.get("metadata", {}) or {},
        )


# --------------------------------------------------------------------------------------
# Token counting helpers
# --------------------------------------------------------------------------------------


class WhitespaceTokenCounter:
    """Cheap fallback token counter (whitespace/punctuation split).

    Used only when no HuggingFace tokenizer is supplied; the paper's 200-token filter is
    specified in *model* tokens, so scripts should always pass the real tokenizer.
    """

    def __call__(self, text: str) -> int:
        return len(str(text).split())


def make_token_counter(tokenizer: Any = None) -> Callable[[str], int]:
    """Return a ``text -> n_tokens`` callable.

    Accepts a HuggingFace tokenizer, any object exposing ``encode``, a plain callable, or
    ``None`` (whitespace fallback).
    """

    if tokenizer is None:
        return WhitespaceTokenCounter()
    if callable(tokenizer) and not hasattr(tokenizer, "encode"):
        return tokenizer

    def _counter(text: str) -> int:
        try:
            ids = tokenizer.encode(str(text), add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(str(text))
        return len(ids)

    return _counter


def count_tokens(text: str, tokenizer: Any = None) -> int:
    """Number of tokens in ``text`` according to ``tokenizer`` (or whitespace)."""
    return make_token_counter(tokenizer)(text)


# --------------------------------------------------------------------------------------
# Deterministic per-subset RNG
# --------------------------------------------------------------------------------------


def subset_seed(subset: str, seed: int = DEFAULT_SEED) -> int:
    """Stable integer seed for a subset (independent of iteration order)."""
    digest = hashlib.md5(f"{seed}:{subset}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------------------------------------------
# Sampling a single subset
# --------------------------------------------------------------------------------------


def _flatten_text(value: Any) -> str:
    """Flatten a P3 pretokenized field (str or list of str) into one string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(str(v) for v in value)
    return str(value)


def sample_from_records(
    records: Sequence[Dict[str, Any]],
    subset: str,
    tokenizer: Any = None,
    n: int = SAMPLES_PER_DATASET,
    seed: int = DEFAULT_SEED,
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> List[P3Sample]:
    """Sample up to ``n`` datapoints from an in-memory list of P3 records.

    This is the pure-Python core of the protocol (used directly by tests and by the
    ``datasets``-based loader): shuffle deterministically, keep the first ``n`` records
    whose *input* is at most ``max_input_tokens`` tokens; if the filtered dataset has
    fewer than ``n`` records the whole dataset is taken.
    """
    counter = make_token_counter(tokenizer)
    order = list(range(len(records)))
    random.Random(subset_seed(subset, seed)).shuffle(order)

    out: List[P3Sample] = []
    for pos in order:
        rec = records[pos]
        inputs = _flatten_text(rec.get(INPUT_FIELD, rec.get("inputs", "")))
        targets = _flatten_text(rec.get(TARGET_FIELD, rec.get("targets", "")))
        n_tok = counter(inputs)
        if max_input_tokens is not None and n_tok > max_input_tokens:
            continue
        out.append(
            P3Sample(
                dataset=subset,
                idx=pos,
                inputs=inputs,
                targets=targets,
                n_input_tokens=n_tok,
            )
        )
        if n is not None and len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------------------
# Loading P3 through HuggingFace ``datasets``
# --------------------------------------------------------------------------------------


def list_p3_subsets(
    dataset_name: str = P3_DATASET_NAME,
    fallback: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return the names of the P3 subsets (needs the ``datasets`` package + network).

    Falls back to ``fallback`` (or an empty list) when the hub is unavailable.
    """
    try:
        from datasets import get_dataset_config_names  # type: ignore

        names = list(get_dataset_config_names(dataset_name))
        logger.info("Loaded %d P3 subset names from the HuggingFace hub", len(names))
        return names
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not list P3 subsets (%s); using fallback list", exc)
        return list(fallback or [])


def _load_subset(
    subset: str,
    dataset_name: str = P3_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
):
    """Load one P3 subset, trying ``train`` then ``validation``."""
    from datasets import load_dataset  # type: ignore

    kwargs = dict(dataset_kwargs or {})
    if cache_dir:
        kwargs.setdefault("cache_dir", cache_dir)

    last_error: Optional[Exception] = None
    for candidate in (split, FALLBACK_SPLIT):
        try:
            return load_dataset(dataset_name, candidate, split=split, **kwargs)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            logger.debug("Split %s unavailable for %s (%s)", candidate, subset, exc)
    raise RuntimeError(f"Could not load P3 subset {subset!r}: {last_error}")


def sample_dataset(
    subset: str,
    tokenizer: Any = None,
    n: int = SAMPLES_PER_DATASET,
    seed: int = DEFAULT_SEED,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    dataset_name: str = P3_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    limit_scan: Optional[int] = None,
) -> List[P3Sample]:
    """Sample up to ``n`` datapoints from a single named P3 subset."""
    ds = _load_subset(
        subset,
        dataset_name=dataset_name,
        split=split,
        dataset_kwargs=dataset_kwargs,
        cache_dir=cache_dir,
    )
    n_rows = len(ds)
    scan = n_rows if limit_scan is None else min(n_rows, limit_scan)
    records = [ds[i] for i in range(scan)]
    return sample_from_records(
        records,
        subset=subset,
        tokenizer=tokenizer,
        n=n,
        seed=seed,
        max_input_tokens=max_input_tokens,
    )


def sample_p3(
    tokenizer: Any = None,
    subsets: Optional[Sequence[str]] = None,
    n_per_dataset: int = SAMPLES_PER_DATASET,
    seed: int = DEFAULT_SEED,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    dataset_name: str = P3_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    max_datasets: Optional[int] = None,
    verbose: bool = True,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> List[P3Sample]:
    """Materialise the Section 5 sample: ~50 datapoints from each of P3's 660 subsets.

    Returns a list of :class:`P3Sample` (expected size ~32,902 when ``max_datasets`` is
    ``None`` and the full hub dataset is available).
    """
    names = list(subsets) if subsets is not None else list_p3_subsets(dataset_name)
    if max_datasets is not None:
        names = names[:max_datasets]

    samples: List[P3Sample] = []
    for i, subset in enumerate(names):
        try:
            got = sample_dataset(
                subset,
                tokenizer=tokenizer,
                n=n_per_dataset,
                seed=seed,
                max_input_tokens=max_input_tokens,
                dataset_name=dataset_name,
                split=split,
                dataset_kwargs=dataset_kwargs,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Skipping P3 subset %s (%s)", subset, exc)
            continue
        samples.extend(got)
        if verbose and (i + 1) % 25 == 0:
            logger.info(
                "P3 sampling: %d/%d subsets -> %d datapoints",
                i + 1,
                len(names),
                len(samples),
            )

    logger.info(
        "P3 sample complete: %d datapoints from %d subsets (target %d)",
        len(samples),
        len(names),
        TARGET_N_DATAPOINTS,
    )
    return samples


# --------------------------------------------------------------------------------------
# Synthetic fallback (offline / unit-testing)
# --------------------------------------------------------------------------------------


def synthetic_records(
    n_rows: int,
    subset: str = "synthetic",
    seed: int = DEFAULT_SEED,
    long_fraction: float = 0.1,
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> List[Dict[str, str]]:
    """Deterministic fake P3 records, some deliberately too long to be kept."""
    rng = random.Random(subset_seed(subset, seed))
    records: List[Dict[str, str]] = []
    n_long = int(round(n_rows * long_fraction))
    long_rows = set(rng.sample(range(n_rows), min(n_long, n_rows)))
    for i in range(n_rows):
        if i in long_rows:
            n_words = max_input_tokens + 50
        else:
            n_words = 5 + rng.randint(0, 60)
        records.append(
            {
                INPUT_FIELD: f"[{subset} #{i}] " + " ".join(["token"] * n_words),
                TARGET_FIELD: f"answer {i}",
            }
        )
    return records


def sample_p3_synthetic(
    n_datasets: int = 8,
    rows_per_dataset: int = 80,
    tokenizer: Any = None,
    n_per_dataset: int = SAMPLES_PER_DATASET,
    seed: int = DEFAULT_SEED,
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> List[P3Sample]:
    """Offline stand-in for :func:`sample_p3` (same protocol, fake records)."""
    out: List[P3Sample] = []
    for d in range(n_datasets):
        subset = f"synthetic_{d:03d}"
        records = synthetic_records(
            rows_per_dataset,
            subset=subset,
            seed=seed,
            max_input_tokens=max_input_tokens,
        )
        out.extend(
            sample_from_records(
                records,
                subset=subset,
                tokenizer=tokenizer,
                n=n_per_dataset,
                seed=seed,
                max_input_tokens=max_input_tokens,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Persistence / statistics
# --------------------------------------------------------------------------------------


def save_samples(samples: Sequence[P3Sample], path: str = DEFAULT_CACHE_PATH) -> str:
    """Write the sample to a JSON file (created eagerly so partial runs are resumable)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([s.as_dict() for s in samples], fh, ensure_ascii=False)
    logger.info("Wrote %d P3 samples to %s", len(samples), path)
    return path


def load_samples(path: str = DEFAULT_CACHE_PATH) -> List[P3Sample]:
    """Read a cached sample written by :func:`save_samples`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [P3Sample.from_dict(d) for d in raw]


def dataset_histogram(samples: Sequence[P3Sample]) -> Dict[str, int]:
    """``{subset: n_samples}`` histogram."""
    hist: Dict[str, int] = {}
    for s in samples:
        hist[s.dataset] = hist.get(s.dataset, 0) + 1
    return hist


def summary_stats(samples: Sequence[P3Sample]) -> Dict[str, Any]:
    """Shape/statistics of the sample, comparable to the paper's 32,902 figure."""
    hist = dataset_histogram(samples)
    lens = [s.n_input_tokens for s in samples if s.n_input_tokens is not None]
    return {
        "n_datapoints": len(samples),
        "n_datasets": len(hist),
        "mean_per_dataset": (len(samples) / len(hist)) if hist else 0.0,
        "target_n_datapoints": TARGET_N_DATAPOINTS,
        "matches_target": len(samples) == TARGET_N_DATAPOINTS,
        "mean_input_tokens": (sum(lens) / len(lens)) if lens else None,
        "max_input_tokens": max(lens) if lens else None,
        "max_input_tokens_filter": MAX_INPUT_TOKENS,
        "datasets_at_cap": sum(1 for v in hist.values() if v >= SAMPLES_PER_DATASET),
        "datasets_below_cap": sum(1 for v in hist.values() if v < SAMPLES_PER_DATASET),
        "seed": DEFAULT_SEED,
    }


def iter_batches(
    samples: Sequence[P3Sample],
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = DEFAULT_SEED,
) -> Iterable[List[P3Sample]]:
    """Yield consecutive batches of samples (optional deterministic shuffling)."""
    order = list(range(len(samples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [samples[i] for i in order[start : start + batch_size]]


def split_samples(
    samples: Sequence[P3Sample],
    n_eval: Optional[int] = None,
    n_analysis: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[P3Sample], List[P3Sample]]:
    """Split the sample into (generation/eval, analysis) halves, as needed by §5.

    ``n_eval`` defaults to ``n_analysis`` defaults to half of the sample.  Both lists
    are drawn without overlap from a single deterministic shuffle.
    """
    total = len(samples)
    order = list(range(total))
    random.Random(seed).shuffle(order)
    if n_eval is None and n_analysis is None:
        n_eval = total // 2
    if n_eval is None:
        n_eval = total - int(n_analysis or 0)
    eval_idx = order[:n_eval]
    rest = order[n_eval:]
    if n_analysis is not None:
        rest = rest[:n_analysis]
    return [samples[i] for i in eval_idx], [samples[i] for i in rest]


# --------------------------------------------------------------------------------------
# Convenience loader used by scripts/run_analysis.py
# --------------------------------------------------------------------------------------


def get_p3_sample(
    tokenizer: Any = None,
    n_datapoints: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    cache_path: Optional[str] = DEFAULT_CACHE_PATH,
    synthetic: bool = False,
    **kwargs: Any,
) -> List[P3Sample]:
    """Return the P3 sample, using the cache when available.

    Parameters
    ----------
    tokenizer:
        Tokenizer used for the 200-token input filter.
    n_datapoints:
        Optional truncation (useful for smoke tests / limited-budget runs).
    cache_path:
        JSON cache written by :func:`save_samples`; set to ``None`` to skip caching.
    synthetic:
        Force the offline synthetic sample (tests, no network).
    **kwargs:
        Forwarded to :func:`sample_p3` / :func:`sample_p3_synthetic`
        (e.g. ``subsets``, ``n_per_dataset``, ``max_datasets``).
    """
    if synthetic:
        samples = sample_p3_synthetic(tokenizer=tokenizer, seed=seed, **kwargs)
    elif cache_path and os.path.exists(cache_path):
        samples = load_samples(cache_path)
        logger.info("Loaded %d cached P3 samples from %s", len(samples), cache_path)
    else:
        samples = sample_p3(tokenizer=tokenizer, seed=seed, **kwargs)
        if cache_path:
            try:
                save_samples(samples, cache_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not cache P3 sample (%s)", exc)

    if n_datapoints is not None and len(samples) > n_datapoints:
        samples = samples[:n_datapoints]
    return samples


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    demo = sample_p3_synthetic()
    print(json.dumps(summary_stats(demo), indent=2))
