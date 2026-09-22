"""RealToxicityPrompts loader.

The paper (§3.3, §4.2, Figure 1) evaluates toxicity on the *challenge* subset of
RealToxicityPrompts (Gehman et al., 2020), which consists of 1,199 prompts that
elicit extremely toxic outputs from language models.  The logit-lens analysis of
Figure 1 (§4.2) additionally uses the 295 of those prompts whose greedy next
token is "sh*t".

The paper uses Perspective API to score generations; the benchmark addendum
("Useful details") instructs reproductions to use
``unitary/unbiased-toxic-roberta`` instead -- that model lives in
``src/eval/toxicity.py``, this module only supplies prompts.

All HuggingFace ``datasets`` access is deferred inside functions so that
importing this module never triggers network I/O (same convention as
``data/jigsaw.py`` and ``data/wikitext.py``).

Design notes / defaults (paper silent):
  * The HF mirror ``allenai/real-toxicity-prompts`` exposes a ``challenge``
    boolean field.  Rows with ``challenge == True`` are the 1,199 challenge
    prompts.  If that field is unavailable, we fall back to the documented RTP
    construction (prompts whose toxicity score lies in [0.5, 1.0]) and take a
    deterministic seeded sample of exactly ``N_CHALLENGE_PROMPTS`` rows.
  * Toxicity scores shipped with the dataset (Perspective-derived) are only used
    for *prompt selection / diagnostics*; the model-generated generations are
    scored with unbiased-toxic-roberta (see ``src/eval/toxicity.py``).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RTP_HF_NAME = "allenai/real-toxicity-prompts"

#: Size of the RTP "challenge" subset used for every toxicity measurement.
N_CHALLENGE_PROMPTS = 1199

#: Number of challenge prompts whose greedy next token is "sh*t" (Figure 1).
N_SHIT_PROMPTS = 295

#: Literal target token used by the paper (censored spelling in the paper text).
TARGET_TOKEN = "sh*t"

#: Token strings (GPT2 ``convert_ids_to_tokens`` style) accepted as "sh*t".
TARGET_TOKEN_VARIANTS = (
    "shit",
    "sh*t",
    "sh it",
)

#: Lower/upper bound of the RTP challenge prompt toxicity scores.
CHALLENGE_TOX_MIN = 0.5
CHALLENGE_TOX_MAX = 1.0

#: Cache file name used for the selected 295 prompts.
SHIT_PROMPTS_FILENAME = "rtp_shit_prompts.json"

PROMPT_COLUMN_CANDIDATES = ("prompt", "prompt_text", "text")
CONTINUATION_COLUMN_CANDIDATES = ("continuation", "continuation_text")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RTPPrompt:
    """A single RealToxicityPrompts prompt.

    Attributes
    ----------
    text:
        The prompt string used to condition generation.
    index:
        Position of the prompt in the original dataset (for reproducibility).
    prompt_toxicity:
        Perspective toxicity score of the prompt itself (metadata only).
    continuation:
        The human-written continuation shipped with RTP (metadata only; not
        used as a ground truth for generation).
    continuation_toxicity:
        Perspective toxicity score of the human continuation (metadata only).
    challenge:
        Whether the prompt belongs to the RTP challenge subset.
    """

    text: str
    index: int = -1
    prompt_toxicity: float = float("nan")
    continuation: str = ""
    continuation_toxicity: float = float("nan")
    challenge: bool = False
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "index": self.index,
            "prompt_toxicity": self.prompt_toxicity,
            "continuation": self.continuation,
            "continuation_toxicity": self.continuation_toxicity,
            "challenge": self.challenge,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RTPPrompt":
        return cls(
            text=d["text"],
            index=int(d.get("index", -1)),
            prompt_toxicity=float(d.get("prompt_toxicity", float("nan"))),
            continuation=d.get("continuation", ""),
            continuation_toxicity=float(d.get("continuation_toxicity", float("nan"))),
            challenge=bool(d.get("challenge", False)),
        )


# ---------------------------------------------------------------------------
# Raw loading helpers
# ---------------------------------------------------------------------------


def _nest_get(example: Dict, key: str) -> Optional[str]:
    """RTP stores prompt/continuation as nested dicts: ``{'text': ..., ...}``."""
    value = example.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("text")
    return str(value)


def _nest_toxicity(example: Dict, key: str) -> float:
    value = example.get(key)
    if value is None:
        return float("nan")
    if isinstance(value, dict):
        for field_name in ("toxicity", "severe_toxicity"):
            if field_name in value:
                try:
                    return float(value[field_name])
                except (TypeError, ValueError):
                    return float("nan")
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _row_to_prompt(example: Dict, index: int) -> Optional[RTPPrompt]:
    text = None
    for candidate in PROMPT_COLUMN_CANDIDATES:
        text = _nest_get(example, candidate)
        if text:
            break
    if not text:
        return None

    continuation = ""
    for candidate in CONTINUATION_COLUMN_CANDIDATES:
        continuation = _nest_get(example, candidate) or ""
        if continuation:
            break

    return RTPPrompt(
        text=text,
        index=index,
        prompt_toxicity=_nest_toxicity(example, "prompt"),
        continuation=continuation,
        continuation_toxicity=_nest_toxicity(example, "continuation"),
        challenge=bool(example.get("challenge", False)),
    )


def _load_hf_split(cache_dir: Optional[str] = None, split: str = "train"):
    """Lazily load the RTP mirror (import deferred to avoid network at import)."""
    from datasets import load_dataset  # local import on purpose

    return load_dataset(RTP_HF_NAME, split=split, cache_dir=cache_dir)


def load_rtp_raw(
    cache_dir: Optional[str] = None,
    split: str = "train",
    limit: Optional[int] = None,
) -> List[RTPPrompt]:
    """Load all RealToxicityPrompts rows as :class:`RTPPrompt` objects.

    ``limit`` keeps the first ``limit`` rows (smoke tests / CPU runs).
    """
    dataset = _load_hf_split(cache_dir=cache_dir, split=split)
    prompts: List[RTPPrompt] = []
    for i, example in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        prompt = _row_to_prompt(example, i)
        if prompt is not None:
            prompts.append(prompt)
    return prompts


def load_rtp_from_jsonl(path: str) -> List[RTPPrompt]:
    """Load prompts from a local JSONL file (offline / cached fallback)."""
    prompts: List[RTPPrompt] = []
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            prompt = _row_to_prompt(json.loads(line), i)
            if prompt is not None:
                prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Challenge subset
# ---------------------------------------------------------------------------


def challenge_prompts(
    cache_dir: Optional[str] = None,
    n: int = N_CHALLENGE_PROMPTS,
    seed: int = 0,
    limit: Optional[int] = None,
    jsonl_path: Optional[str] = None,
) -> List[RTPPrompt]:
    """Return the 1,199-prompt RTP "challenge" subset.

    Preference order:
      1. a local JSONL file, if ``jsonl_path`` is given;
      2. rows flagged ``challenge == True`` in the HF mirror;
      3. prompts with toxicity in [0.5, 1.0], deterministically sub-sampled to
         exactly ``n`` rows (documented RTP fallback).
    """
    if jsonl_path is not None and os.path.exists(jsonl_path):
        prompts = load_rtp_from_jsonl(jsonl_path)
    else:
        prompts = load_rtp_raw(cache_dir=cache_dir, limit=limit)

    flagged = [p for p in prompts if p.challenge]
    if len(flagged) >= n:
        return _dedupe_texts(flagged)[:n]
    if flagged:
        # Mirror flagged fewer than expected (e.g. smoke-test limit sub-slice).
        return _dedupe_texts(flagged)

    in_range = [
        p
        for p in prompts
        if not _isnan(p.prompt_toxicity)
        and CHALLENGE_TOX_MIN <= p.prompt_toxicity <= CHALLENGE_TOX_MAX
    ]
    pool = in_range if in_range else prompts
    pool = _dedupe_texts(pool)
    if len(pool) <= n:
        return pool
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pool)), n))
    return [pool[i] for i in indices]


def _isnan(value: float) -> bool:
    return value != value


def _dedupe_texts(prompts: Sequence[RTPPrompt]) -> List[RTPPrompt]:
    """Drop duplicate prompt strings while preserving order."""
    seen = set()
    out: List[RTPPrompt] = []
    for prompt in prompts:
        if prompt.text in seen:
            continue
        seen.add(prompt.text)
        out.append(prompt)
    return out


def prompt_texts(prompts: Sequence[RTPPrompt]) -> List[str]:
    """Convenience: list of prompt strings."""
    return [p.text for p in prompts]


def sample_prompts(
    prompts: Sequence[RTPPrompt], n: int, seed: int = 0
) -> List[RTPPrompt]:
    """Deterministic sub-sample (order preserved) for cheap smoke runs."""
    if n >= len(prompts):
        return list(prompts)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(prompts)), n))
    return [prompts[i] for i in indices]


# ---------------------------------------------------------------------------
# "sh*t"-eliciting prompts (Figure 1 / logit lens)
# ---------------------------------------------------------------------------


def normalise_token(token: str) -> str:
    """Normalise a GPT2 token string for target comparison.

    GPT2 uses the byte-level ``Ġ`` prefix for word-initial tokens; we strip it,
    lowercase, and drop surrounding whitespace/punctuation so that ``"Ġshit"``
    and ``"shit"`` compare equal.
    """
    token = token.replace("\u0120", " ").replace("\u010a", " ")
    token = token.strip().lower()
    return token.strip(" \t\n.!?,;:\"'*")


def is_target_token(token: str) -> bool:
    """Whether ``token`` is one of the accepted spellings of the RTP target."""
    norm = normalise_token(token)
    return any(norm == normalise_token(v) for v in TARGET_TOKEN_VARIANTS)


def _token_strings(tokenizer, token_id: int) -> List[str]:
    """Decoded/tokenised spellings for a single token id (GPT2 and BPE variants)."""
    strings: List[str] = []
    try:
        strings.append(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:  # pragma: no cover - exotic tokenizers
        pass
    try:
        strings.append(tokenizer.decode([int(token_id)]))
    except Exception:  # pragma: no cover
        pass
    return [s for s in strings if s]


def next_token_ids(model, tokenizer, texts: Sequence[str], device=None):
    """Greedy next-token id for each prompt (no generation of further tokens).

    Returns a list of ints, same length as ``texts``.
    """
    import torch  # local import on purpose

    if device is None:
        device = next(model.parameters()).device
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    # Next token after the last *unpadded* position of each sequence.
    if attention_mask is not None:
        last = attention_mask.sum(dim=1) - 1
    else:
        last = torch.full(
            (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device
        )
    logits = out.logits[torch.arange(input_ids.shape[0], device=input_ids.device), last]
    return logits.argmax(dim=-1).tolist()


def select_target_token_prompts(
    model,
    tokenizer,
    prompts: Optional[Sequence[RTPPrompt]] = None,
    n: int = N_SHIT_PROMPTS,
    cache_dir: Optional[str] = None,
    seed: int = 0,
    device=None,
    batch_size: int = 32,
    verbose: bool = False,
) -> List[RTPPrompt]:
    """Select the prompts whose greedy next token is the target ("sh*t").

    The paper (§4.2) reports 295 such prompts out of the 1,199 challenge
    prompts; we return at most the first ``n`` matches in dataset order so the
    selection is deterministic.
    """
    if prompts is None:
        prompts = challenge_prompts(cache_dir=cache_dir, seed=seed)

    selected: List[RTPPrompt] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        try:
            token_ids = next_token_ids(
                model, tokenizer, [p.text for p in batch], device=device
            )
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                print(f"[realtoxicity] batch failed: {exc}")
            continue
        for prompt, token_id in zip(batch, token_ids):
            strings = _token_strings(tokenizer, token_id)
            if any(is_target_token(s) for s in strings):
                selected.append(prompt)
        if len(selected) >= n:
            break
    if verbose:
        print(f"[realtoxicity] selected {len(selected)} target-token prompts")
    return selected[:n]


def save_prompts(prompts: Sequence[RTPPrompt], path: str) -> str:
    """Persist a prompt selection as JSON (reproduction artifact)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([p.to_dict() for p in prompts], handle, indent=2)
    return path


def load_prompts(path: str) -> List[RTPPrompt]:
    """Load a prompt selection previously written by :func:`save_prompts`."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [RTPPrompt.from_dict(d) for d in data]


def shit_prompts_path(cache_dir: Optional[str] = None) -> str:
    """Default artifact path for the 295 "sh*t"-eliciting prompts."""
    base = cache_dir or os.path.join("artifacts", "cache")
    return os.path.join(base, SHIT_PROMPTS_FILENAME)


def load_target_token_prompts(
    model,
    tokenizer,
    cache_dir: Optional[str] = None,
    n: int = N_SHIT_PROMPTS,
    device=None,
    recompute: bool = False,
    **kwargs,
) -> List[RTPPrompt]:
    """Cached accessor for the 295 "sh*t"-eliciting prompts (Figure 1)."""
    path = shit_prompts_path(cache_dir)
    if not recompute and os.path.exists(path):
        prompts = load_prompts(path)
        if prompts:
            return prompts[:n]
    prompts = select_target_token_prompts(
        model, tokenizer, prompts=None, n=n, cache_dir=cache_dir, device=device, **kwargs
    )
    if prompts:
        save_prompts(prompts, path)
    return prompts


def toxicity_stats(prompts: Sequence[RTPPrompt]) -> Dict[str, float]:
    """Diagnostics on the Perspective prompt scores shipped with the dataset."""
    scores = [p.prompt_toxicity for p in prompts if not _isnan(p.prompt_toxicity)]
    if not scores:
        return {"n": float(len(prompts)), "mean_prompt_toxicity": float("nan")}
    return {
        "n": float(len(prompts)),
        "mean_prompt_toxicity": float(sum(scores) / len(scores)),
        "min_prompt_toxicity": float(min(scores)),
        "max_prompt_toxicity": float(max(scores)),
    }


__all__ = [
    "RTP_HF_NAME",
    "N_CHALLENGE_PROMPTS",
    "N_SHIT_PROMPTS",
    "TARGET_TOKEN",
    "TARGET_TOKEN_VARIANTS",
    "RTPPrompt",
    "load_rtp_raw",
    "load_rtp_from_jsonl",
    "challenge_prompts",
    "prompt_texts",
    "sample_prompts",
    "normalise_token",
    "is_target_token",
    "next_token_ids",
    "select_target_token_prompts",
    "load_target_token_prompts",
    "save_prompts",
    "load_prompts",
    "shit_prompts_path",
    "toxicity_stats",
]


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    import argparse

    parser = argparse.ArgumentParser(description="Inspect RealToxicityPrompts loading")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--jsonl", type=str, default=None)
    args = parser.parse_args()

    prompts = challenge_prompts(limit=args.limit, jsonl_path=args.jsonl)
    print(f"loaded {len(prompts)} challenge prompts")
    print("stats:", toxicity_stats(prompts))
    if prompts:
        print("example:", prompts[0].text[:120])
