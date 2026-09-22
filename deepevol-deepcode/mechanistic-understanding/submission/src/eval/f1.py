"""F1 evaluation on Wikipedia sentences (Section 3.3 of the paper).

The paper follows prior work (Dinan et al., 2020; Adolphs et al., 2023) and
measures F1 with 2,000 Wikipedia sentences as prompts:

    "Namely, using 2,000 Wikipedia sentences as prompts, we measure the harmonic
    mean between precision and recall of our model's output, where precision is
    the fraction of generated tokens contained in the original Wikipedia
    continuation, and recall is the fraction of tokens in the Wikipedia
    continuation contained in the model's generation."

Concretely, for every prompt (a Wikipedia sentence) we hold out the *original
continuation* (the following Wikipedia sentence) as the reference, let the model
generate a continuation (greedy, 20 tokens -- the shared generation default used
by the toxicity / perplexity / intervention / un-alignment evaluations), and
compute

    precision = |tokens(generation) INTERSECT tokens(reference)| / |tokens(generation)|
    recall    = |tokens(generation) INTERSECT tokens(reference)| / |tokens(reference)|
    F1        = 2 * precision * recall / (precision + recall)

The paper's reproduction target for GPT2_DPO (and for the interventions) is
F1 ~ 0.195 (Tables 2 and 4), i.e. F1 should change only minimally so that
alignment/interventions do not visibly damage generation quality.

Only GPT2-medium is in reproduction scope; no Llama2 code paths are needed here.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Constants (paper / plan defaults)
# --------------------------------------------------------------------------------------

N_F1_SENTENCES = 2000
DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_SEED = 0
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_PROMPT_LENGTH = 96
DEFAULT_MIN_WORDS = 5

#: F1 reference values from the paper (Tables 2 and 4).
GPT2_F1 = 0.193
GPT2_DPO_F1 = 0.195

ARTIFACT_DIR = "artifacts/eval"
F1_RESULT_FILENAME = "f1_{model}.json"

#: Tokenisation modes for the overlap metric.
TOKENIZE_MODES = ("whitespace", "tokenizer", "set", "multiset")

_WORD_RE = re.compile(r"[a-z0-9']+")

# Matplotlib is imported lazily (through src.analysis.plots) so that this module
# can be imported on machines without a display / without the plotting stack.
try:  # pragma: no cover - import-time convenience only
    from ..model_utils import GPT2_MEDIUM, resolve_device
except Exception:  # pragma: no cover
    GPT2_MEDIUM = "openai-community/gpt2-medium"

    def resolve_device(device: Optional[str] = None):  # type: ignore
        import torch

        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


def _nan_safe(value: Any) -> Optional[float]:
    """Convert numpy/nan values into JSON-friendly floats."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


@dataclass
class F1Result:
    """Container for the token-overlap F1 evaluation of one model state."""

    model_name: str = "gpt2"
    f1: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    n_sentences: int = 0
    prompts: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    generations: List[str] = field(default_factory=list)
    per_example_f1: Optional[np.ndarray] = None
    per_example_precision: Optional[np.ndarray] = None
    per_example_recall: Optional[np.ndarray] = None
    tokenize_mode: str = "multiset"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    seed: int = DEFAULT_SEED
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- basics ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.generations)

    @property
    def mean_f1(self) -> float:
        return float(self.f1)

    @property
    def std_f1(self) -> float:
        if self.per_example_f1 is None or len(self.per_example_f1) == 0:
            return float("nan")
        return float(np.nanstd(self.per_example_f1))

    def summary(self) -> Dict[str, Any]:
        """Dict with the F1 numbers plus optional PPL / toxicity annotations."""
        out: Dict[str, Any] = {
            "model": self.model_name,
            "f1": _nan_safe(self.f1),
            "precision": _nan_safe(self.precision),
            "recall": _nan_safe(self.recall),
            "f1_std": _nan_safe(self.std_f1),
            "n_sentences": int(self.n_sentences),
            "tokenize": self.tokenize_mode,
            "max_new_tokens": int(self.max_new_tokens),
            "seed": int(self.seed),
        }
        for key in ("toxicity", "perplexity"):
            if key in self.meta:
                out[key] = _nan_safe(self.meta[key])
        return out

    # -- persistence -------------------------------------------------------------
    def to_dict(self, include_examples: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model_name": self.model_name,
            "f1": _nan_safe(self.f1),
            "precision": _nan_safe(self.precision),
            "recall": _nan_safe(self.recall),
            "f1_std": _nan_safe(self.std_f1),
            "n_sentences": int(self.n_sentences),
            "tokenize_mode": self.tokenize_mode,
            "max_new_tokens": int(self.max_new_tokens),
            "seed": int(self.seed),
            "meta": dict(self.meta),
        }
        if include_examples:
            data["prompts"] = list(self.prompts)
            data["references"] = list(self.references)
            data["generations"] = list(self.generations)
        if self.per_example_f1 is not None:
            data["per_example_f1"] = [None if v is None else float(v) for v in np.asarray(self.per_example_f1)]
            data["per_example_precision"] = [
                None if v is None else float(v) for v in np.asarray(self.per_example_precision)
            ]
            data["per_example_recall"] = [
                None if v is None else float(v) for v in np.asarray(self.per_example_recall)
            ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "F1Result":
        def _f(key: str) -> float:
            v = data.get(key)
            return float("nan") if v is None else float(v)

        def _arr(key: str) -> Optional[np.ndarray]:
            vals = data.get(key)
            if vals is None:
                return None
            return np.asarray([np.nan if v is None else float(v) for v in vals])

        return cls(
            model_name=data.get("model_name", "gpt2"),
            f1=_f("f1"),
            precision=_f("precision"),
            recall=_f("recall"),
            n_sentences=int(data.get("n_sentences", 0) or 0),
            prompts=list(data.get("prompts", []) or []),
            references=list(data.get("references", []) or []),
            generations=list(data.get("generations", []) or []),
            per_example_f1=_arr("per_example_f1"),
            per_example_precision=_arr("per_example_precision"),
            per_example_recall=_arr("per_example_recall"),
            tokenize_mode=data.get("tokenize_mode", "multiset"),
            max_new_tokens=int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS) or DEFAULT_MAX_NEW_TOKENS),
            seed=int(data.get("seed", DEFAULT_SEED) or 0),
            meta=dict(data.get("meta", {}) or {}),
        )


# --------------------------------------------------------------------------------------
# Prompt / reference construction
# --------------------------------------------------------------------------------------


def is_usable_sentence(text: str, min_words: int = DEFAULT_MIN_WORDS) -> bool:
    """Cheap heuristic filter so F1 is computed on actual prose sentences."""
    t = (text or "").strip()
    if len(t) < 20:
        return False
    if t.startswith("="):  # Wiki heading
        return False
    if t.startswith(("{", "|", "[[", "Category:")):
        return False
    return len(t.split()) >= min_words


def build_prompt_reference_pairs(
    sentences: Sequence[str],
    min_words: int = DEFAULT_MIN_WORDS,
    max_reference_chars: int = 400,
) -> List[Tuple[str, str]]:
    """Pair each sentence with the *next* sentence as its original continuation.

    The paper's reference is the "original Wikipedia continuation" of the prompt
    sentence, so consecutive sentences in the corpus form the (prompt,
    reference) pairs used for the token-overlap metric.
    """
    clean = [s.strip() for s in sentences if is_usable_sentence(s, min_words=min_words)]
    pairs: List[Tuple[str, str]] = []
    for i in range(len(clean) - 1):
        prompt = clean[i]
        reference = clean[i + 1]
        if max_reference_chars and len(reference) > max_reference_chars:
            reference = reference[:max_reference_chars].rsplit(" ", 1)[0]
        if prompt and reference:
            pairs.append((prompt, reference))
    return pairs


def load_f1_dataset(
    n: int = N_F1_SENTENCES,
    seed: int = DEFAULT_SEED,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    cache_dir: Optional[str] = None,
    min_words: int = DEFAULT_MIN_WORDS,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """Return up to ``n`` (prompt, reference) Wikipedia sentence pairs.

    Parameters
    ----------
    pairs:
        Explicit pairs (prompt, reference); when given, no data loading happens.
    n:
        Number of pairs requested (the paper uses 2,000). If the corpus holds
        fewer usable pairs, everything available is returned.
    """
    if pairs is not None:
        out = [(str(p), str(r)) for p, r in pairs]
        if n is not None and 0 < n < len(out):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(out)), n))
            out = [out[i] for i in idx]
        return out

    from ...data.wikitext import wiki_sentences_for_f1  # lazy import

    sentences = wiki_sentences_for_f1(n=n + 200, seed=seed, cache_dir=cache_dir)
    built = build_prompt_reference_pairs(sentences, min_words=min_words)
    if verbose:
        print(f"[f1] built {len(built)} (prompt, reference) pairs from Wikipedia sentences")
    if n is not None and 0 < n < len(built):
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(built)), n))
        built = [built[i] for i in idx]
    return built


# --------------------------------------------------------------------------------------
# Tokenisation and overlap F1
# --------------------------------------------------------------------------------------


def _normalise_token(tok: str) -> str:
    t = (tok or "").lower().strip()
    t = t.strip(" \t\n\r\v\f.,;:!?\"'`()[]{}<>*#-–—/$")
    return t


def tokenize_whitespace(text: str) -> List[str]:
    """Whitespace/word-boundary tokenisation (paper-level notion of token)."""
    return [m.group(0) for m in _WORD_RE.finditer((text or "").lower())]


def tokenize_with_tokenizer(tokenizer: Any, text: str) -> List[str]:
    """Token-id based tokenisation; tokens are converted to their string form."""
    if tokenizer is None:
        return tokenize_whitespace(text)
    try:
        ids = tokenizer(
            text or "",
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    except Exception:
        return tokenize_whitespace(text)
    out: List[str] = []
    for tid in ids:
        try:
            tok = tokenizer.convert_ids_to_tokens(int(tid))
        except Exception:
            tok = str(tid)
        tok = tok.replace("\u0120", " ").replace("Ġ", " ").replace("Ċ", " ")
        tok = _normalise_token(tok)
        if tok:
            out.append(tok)
    return out


def overlap_f1(
    generation: str,
    reference: str,
    tokenizer: Any = None,
    mode: str = "multiset",
    return_counts: bool = False,
):
    """Harmonic mean of precision and recall between two token sequences.

    precision = |gen INTERSECT ref| / |gen|
    recall    = |gen INTERSECT ref| / |ref|
    F1        = 2 * precision * recall / (precision + recall)

    ``mode`` selects the counting convention:
      * ``"multiset"`` (default): multiset intersection (token counts capped).
      * ``"set"``: unique-token intersection.
      * ``"whitespace"`` / ``"tokenizer"``: shorthand for multiset with the
        corresponding tokenisation.
    """
    if mode in ("whitespace", "tokenizer"):
        tokenize_mode = mode
        count_mode = "multiset"
    elif mode == "set":
        tokenize_mode = "whitespace"
        count_mode = "set"
    else:
        tokenize_mode = "whitespace"
        count_mode = "multiset"

    if tokenize_mode == "tokenizer":
        gen_tokens = tokenize_with_tokenizer(tokenizer, generation)
        ref_tokens = tokenize_with_tokenizer(tokenizer, reference)
    else:
        gen_tokens = tokenize_whitespace(generation)
        ref_tokens = tokenize_whitespace(reference)

    if count_mode == "multiset":
        gen_counts = Counter(gen_tokens)
        ref_counts = Counter(ref_tokens)
        overlap = sum(min(c, ref_counts[tok]) for tok, c in gen_counts.items())
        n_gen = sum(gen_counts.values())
        n_ref = sum(ref_counts.values())
    else:
        gen_set = set(gen_tokens)
        ref_set = set(ref_tokens)
        overlap = len(gen_set & ref_set)
        n_gen = len(gen_set)
        n_ref = len(ref_set)

    precision = overlap / n_gen if n_gen > 0 else 0.0
    recall = overlap / n_ref if n_ref > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    if return_counts:
        return f1, precision, recall
    return f1


def corpus_f1(
    generations: Sequence[str],
    references: Sequence[str],
    tokenizer: Any = None,
    mode: str = "multiset",
) -> Tuple[float, float, float]:
    """Micro-averaged F1 over a corpus of (generation, reference) pairs.

    Returns ``(f1, precision, recall)`` where the aggregated precision and recall
    are computed from the summed overlaps and lengths (micro averaging), which is
    the standard way to report a corpus-level token-overlap F1.
    """
    if len(generations) != len(references):
        raise ValueError("generations and references must have the same length")

    count_mode = "set" if mode == "set" else "multiset"
    tokenize_mode = mode if mode in ("whitespace", "tokenizer") else "whitespace"

    tot_overlap = 0
    tot_gen = 0
    tot_ref = 0
    f1s: List[float] = []
    precs: List[float] = []
    recs: List[float] = []

    for gen, ref in zip(generations, references):
        if tokenize_mode == "tokenizer":
            gen_tokens = tokenize_with_tokenizer(tokenizer, gen)
            ref_tokens = tokenize_with_tokenizer(tokenizer, ref)
        else:
            gen_tokens = tokenize_whitespace(gen)
            ref_tokens = tokenize_whitespace(ref)

        if count_mode == "multiset":
            gen_counts = Counter(gen_tokens)
            ref_counts = Counter(ref_tokens)
            overlap = sum(min(c, ref_counts[tok]) for tok, c in gen_counts.items())
            n_gen = sum(gen_counts.values())
            n_ref = sum(ref_counts.values())
        else:
            gen_set = set(gen_tokens)
            ref_set = set(ref_tokens)
            overlap = len(gen_set & ref_set)
            n_gen = len(gen_set)
            n_ref = len(ref_set)

        p = overlap / n_gen if n_gen > 0 else 0.0
        r = overlap / n_ref if n_ref > 0 else 0.0
        f = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

        f1s.append(f)
        precs.append(p)
        recs.append(r)
        tot_overlap += overlap
        tot_gen += n_gen
        tot_ref += n_ref

    precision = tot_overlap / tot_gen if tot_gen > 0 else 0.0
    recall = tot_overlap / tot_ref if tot_ref > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def per_example_f1(
    generations: Sequence[str],
    references: Sequence[str],
    tokenizer: Any = None,
    mode: str = "multiset",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-example (F1, precision, recall) arrays; used for std/plots."""
    f1s, precs, recs = [], [], []
    for gen, ref in zip(generations, references):
        f, p, r = overlap_f1(gen, ref, tokenizer=tokenizer, mode=mode, return_counts=True)
        f1s.append(f)
        precs.append(p)
        recs.append(r)
    return np.asarray(f1s), np.asarray(precs), np.asarray(recs)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def evaluate_f1(
    model: Any,
    tokenizer: Any,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    n: int = N_F1_SENTENCES,
    seed: int = DEFAULT_SEED,
    model_name: str = "gpt2",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH,
    tokenize_mode: str = "multiset",
    cache_dir: Optional[str] = None,
    generations: Optional[Sequence[str]] = None,
    verbose: bool = False,
    progress: bool = True,
) -> F1Result:
    """Token-overlap F1 on 2,000 Wikipedia sentences (Section 3.3).

    The model generates a greedy continuation of ``max_new_tokens`` tokens for
    each prompt sentence; the following Wikipedia sentence acts as the reference
    ("original Wikipedia continuation"). Precision/recall are token-overlap
    fractions and F1 is their harmonic mean.

    Parameters
    ----------
    pairs:
        Explicit (prompt, reference) pairs; otherwise loaded from Wikitext-2.
    generations:
        Pre-computed generations aligned with ``pairs``; when supplied, no
        decoding happens (used to score the exact same generations across
        metrics / interventions).
    """
    from .toxicity import generate_continuations  # local import avoids cycles

    if pairs is None:
        pairs = load_f1_dataset(n=n, seed=seed, cache_dir=cache_dir, verbose=verbose)
    pairs = [(str(p), str(r)) for p, r in pairs]
    prompts = [p for p, _ in pairs]
    references = [r for _, r in pairs]

    if generations is None:
        gens, _full = generate_continuations(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
            device=device,
            do_sample=False,
            max_prompt_length=max_prompt_length,
            verbose=verbose,
            progress=progress,
        )
    else:
        gens = ["" if g is None else str(g) for g in generations]
        if len(gens) != len(prompts):
            raise ValueError(
                f"generations ({len(gens)}) must align with prompts ({len(prompts)})"
            )

    f1, precision, recall = corpus_f1(gens, references, tokenizer=tokenizer, mode=tokenize_mode)
    ex_f1, ex_p, ex_r = per_example_f1(gens, references, tokenizer=tokenizer, mode=tokenize_mode)

    if verbose:
        print(
            f"[f1] {model_name}: F1={f1:.4f} P={precision:.4f} R={recall:.4f} "
            f"over {len(prompts)} Wikipedia sentences"
        )

    return F1Result(
        model_name=model_name,
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        n_sentences=len(prompts),
        prompts=prompts,
        references=references,
        generations=list(gens),
        per_example_f1=ex_f1,
        per_example_precision=ex_p,
        per_example_recall=ex_r,
        tokenize_mode=tokenize_mode,
        max_new_tokens=max_new_tokens,
        seed=seed,
        meta={
            "dataset": "wikitext-2 (2,000 Wikipedia sentences)",
            "device": str(resolve_device(device)) if device is not None else None,
            "mean_per_example_f1": float(np.nanmean(ex_f1)) if len(ex_f1) else None,
        },
    )


#: Convenience alias matching the paper's metric name.
f1 = evaluate_f1
wiki_f1 = evaluate_f1


def compare_f1(before: F1Result, after: F1Result) -> Dict[str, Any]:
    """Compare two F1 results (intervention / DPO deltas) as in Tables 2 and 4."""
    delta = float(after.f1 - before.f1)
    rel = delta / before.f1 if before.f1 not in (0.0,) and not math.isnan(before.f1) else float("nan")
    return {
        "before": before.model_name,
        "after": after.model_name,
        "f1_before": _nan_safe(before.f1),
        "f1_after": _nan_safe(after.f1),
        "delta": _nan_safe(delta),
        "relative_change": _nan_safe(rel),
        "precision_before": _nan_safe(before.precision),
        "precision_after": _nan_safe(after.precision),
        "recall_before": _nan_safe(before.recall),
        "recall_after": _nan_safe(after.recall),
        "n_sentences": int(before.n_sentences),
    }


# --------------------------------------------------------------------------------------
# Persistence / plotting
# --------------------------------------------------------------------------------------


def default_path(model_name: str = "gpt2", out_dir: str = ARTIFACT_DIR) -> str:
    """Default JSON artifact path for an F1 result."""
    safe = str(model_name).replace("/", "_")
    return os.path.join(out_dir, F1_RESULT_FILENAME.format(model=safe))


def save_result(path: str, result: F1Result, include_examples: bool = True) -> str:
    """Write an :class:`F1Result` to JSON, returning the path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(include_examples=include_examples), fh, indent=2)
    return path


def load_result(path: str) -> F1Result:
    """Read an :class:`F1Result` from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return F1Result.from_dict(json.load(fh))


def save_results(path: str, results: Dict[str, F1Result]) -> str:
    """Write several results (e.g. GPT2 vs GPT2_DPO) into one JSON file."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {name: res.summary() for name, res in results.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def plot_f1_comparison(
    results: Sequence[F1Result],
    labels: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
    title: str = "F1 on 2,000 Wikipedia sentences",
    ylabel: str = "F1",
    figsize: Tuple[float, float] = (6.5, 4.0),
    annotate: bool = True,
    baseline: Optional[float] = None,
):
    """Bar chart of F1 values (Table 2 / Table 4 style)."""
    from ..analysis.plots import bar_with_errors, make_figure, save_figure  # lazy

    names = list(labels) if labels is not None else [r.model_name for r in results]
    values = [float(r.f1) for r in results]
    errors = [float(r.std_f1) if r.per_example_f1 is not None else None for r in results]

    fig, ax = make_figure(figsize=figsize)
    bar_with_errors(
        ax,
        names,
        values,
        errors=errors,
        title=title,
        ylabel=ylabel,
        annotate=annotate,
    )
    if baseline is not None:
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=0.8)
    if out_path:
        return save_figure(fig, out_path)
    return None


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover - manual smoke test
    tokenizer = None
    try:
        from ..model_utils import load_model

        model, tokenizer = load_model(GPT2_MEDIUM)
    except Exception as exc:  # pragma: no cover
        print(f"[f1] could not load GPT2-medium ({exc}); running metric-only check")

    gens = ["the cat sat on the mat", "a quick brown fox jumps"]
    refs = ["the cat sat on the mat today", "the quick brown fox jumps over"]
    print("corpus F1:", corpus_f1(gens, refs, tokenizer=tokenizer))
    print("per-example F1:", overlap_f1(gens[0], refs[0], tokenizer=tokenizer, return_counts=True))

    if tokenizer is not None:
        res = evaluate_f1(
            model,
            tokenizer,
            n=8,
            verbose=True,
        )
        print(json.dumps(res.summary(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    _main()
