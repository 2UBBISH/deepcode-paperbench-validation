"""Perplexity evaluation on Wikitext-2 (Section 3.3).

The paper follows prior work (Geva et al., 2022) and measures perplexity on the
Wikitext-2 dataset (Merity et al., 2016) to check that the residual-stream
interventions (``x^{L-1} = x^{L-1} - alpha * W``) and DPO training do not degrade
generation quality:

* GPT2 baseline on Wikitext-2:                ~21.7  (Table 2, "NO OP")
* GPT2 with a toxic vector subtracted:        ~23.3 - 23.6 (alpha chosen so the
  perplexity matches the post-DPO model)
* GPT2_DPO (post-alignment):                  ~23.34
* GPT2 with 7 un-aligned key vectors scaled:  ~23.30 (Section 6)

Implementation notes
--------------------
* Standard HuggingFace sliding-window causal-LM perplexity: a window of
  ``seq_len`` tokens is fed at a time with ``stride`` advance; only the tokens
  that were not already scored by the previous window contribute to the
  negative log-likelihood, so every corpus token is counted exactly once.
* The same tokenization convention as the rest of the reproduction is used
  (``add_special_tokens=False`` for GPT2), shared with ``data/wikitext.py``.
* ``evaluate_perplexity`` scores whatever state ``model`` currently is in, so
  the intervention machinery can patch the model (e.g. subtract a toxic vector
  at the last layer) and simply call this function inside the patch context.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..model_utils import GPT2_MEDIUM, resolve_device

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WIKITEXT2_NAME = "wikitext-2"
DEFAULT_SPLIT = "test"
DEFAULT_SEQ_LEN = 1024
DEFAULT_STRIDE = 512
DEFAULT_BATCH_SIZE = 1

#: Reference numbers from the paper (GPT2-medium): baseline / post-DPO.
GPT2_PPL = 21.7
GPT2_DPO_PPL = 23.34

ARTIFACT_DIR = "artifacts/eval"
PERPLEXITY_RESULT_FILENAME = "perplexity_{model}.json"


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class PerplexityResult:
    """Wikitext-2 perplexity for one model state.

    Parameters
    ----------
    model_name:
        Label of the scored model (``"gpt2"``, ``"gpt2_dpo"``, ...).
    ppl:
        Token-level perplexity ``exp(mean NLL)``.
    loss:
        Mean negative log-likelihood (``log(ppl)``).
    n_tokens:
        Number of scored tokens (targets contributing to the NLL).
    n_windows:
        Number of forward passes performed.
    seq_len, stride, split:
        Window configuration used.
    """

    model_name: str = "gpt2"
    ppl: float = float("nan")
    loss: float = float("nan")
    n_tokens: int = 0
    n_windows: int = 0
    seq_len: int = DEFAULT_SEQ_LEN
    stride: int = DEFAULT_STRIDE
    split: str = DEFAULT_SPLIT
    dataset: str = WIKITEXT2_NAME
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.n_tokens)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "dataset": self.dataset,
            "split": self.split,
            "perplexity": round(float(self.ppl), 4),
            "loss": round(float(self.loss), 6),
            "n_tokens": int(self.n_tokens),
            "n_windows": int(self.n_windows),
            "seq_len": int(self.seq_len),
            "stride": int(self.stride),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ppl": float(self.ppl),
            "loss": float(self.loss),
            "n_tokens": int(self.n_tokens),
            "n_windows": int(self.n_windows),
            "seq_len": int(self.seq_len),
            "stride": int(self.stride),
            "split": self.split,
            "dataset": self.dataset,
            "meta": _jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerplexityResult":
        data = dict(data or {})
        meta = data.pop("meta", {}) or {}
        known = {
            "model_name",
            "ppl",
            "loss",
            "n_tokens",
            "n_windows",
            "seq_len",
            "stride",
            "split",
            "dataset",
        }
        kwargs = {k: v for k, v in data.items() if k in known}
        if "perplexity" in data and "ppl" not in kwargs:
            kwargs["ppl"] = data["perplexity"]
        return cls(meta=meta, **kwargs)

    # convenience aliases
    @property
    def perplexity(self) -> float:
        return float(self.ppl)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _unwrap_model(model: Any) -> Any:
    """Return the underlying HF model if ``model`` is a ``DataParallel``/wrapper."""
    return getattr(model, "module", model)


def _resolve_stride(stride: Optional[int], seq_len: int) -> int:
    if stride is None:
        return max(1, seq_len // 2)
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be a positive integer (or None)")
    return min(stride, seq_len)


@torch.inference_mode()
def _window_loss(model: Any, input_ids: torch.Tensor, n_score: int) -> Tuple[float, int]:
    """Return ``(sum NLL over the last ``n_score`` tokens, n_score)``.

    The first ``len(input_ids) - n_score`` positions are masked with ``-100`` so
    the loss only covers the tokens that were not already scored.
    """
    if n_score <= 0:
        return 0.0, 0
    input_ids = input_ids.to(next(model.parameters()).device, non_blocking=True)
    targets = input_ids.clone()
    if targets.numel() - n_score > 0:
        targets[: targets.numel() - n_score] = -100
    out = model(input_ids.unsqueeze(0), labels=targets.unsqueeze(0))
    loss = out.loss
    if loss is None:
        raise RuntimeError("model returned no loss; is it a causal LM?")
    return float(loss.detach().float().cpu()) * int(n_score), int(n_score)


# --------------------------------------------------------------------------- #
# Core perplexity
# --------------------------------------------------------------------------- #


def perplexity_from_ids(
    model: Any,
    token_ids: Sequence[int],
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = DEFAULT_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_windows: Optional[int] = None,
    model_name: str = "gpt2",
    split: str = DEFAULT_SPLIT,
    verbose: bool = False,
) -> PerplexityResult:
    """Sliding-window perplexity over a flat token-id sequence.

    Every corpus token is scored exactly once: a window covers
    ``token_ids[i:i+seq_len]`` and only the last ``end - prev_end`` positions of
    that window are scored (the prefix was already counted by the previous
    window).  The final window is aligned to the end of the corpus when the
    stride does not divide the remaining token count evenly.
    """
    model = _unwrap_model(model)
    ids = [int(t) for t in token_ids]
    n_total = len(ids)
    seq_len = int(seq_len)
    stride = _resolve_stride(stride, seq_len)

    if n_total < 2:
        return PerplexityResult(
            model_name=model_name,
            ppl=float("nan"),
            loss=float("nan"),
            n_tokens=0,
            n_windows=0,
            seq_len=seq_len,
            stride=stride,
            split=split,
            meta={"reason": "corpus shorter than 2 tokens"},
        )

    nll_sum = 0.0
    n_tokens = 0
    n_windows = 0
    prev_end = 0

    begin = 0
    while begin < n_total:
        end = min(begin + seq_len, n_total)
        if end - begin < 2:
            break
        n_score = end - prev_end
        window = torch.tensor(ids[begin:end], dtype=torch.long)
        window_nll, scored = _window_loss(model, window, n_score)
        nll_sum += window_nll
        n_tokens += scored
        n_windows += 1
        prev_end = end
        if verbose:
            print(
                f"[ppl] window {n_windows}: tokens{begin}:{end} scored={scored} "
                f"running_ppl={math.exp(nll_sum / max(n_tokens, 1)):.3f}"
            )
        if end >= n_total:
            break
        if max_windows is not None and n_windows >= int(max_windows):
            break
        begin += stride

    if n_tokens == 0:
        return PerplexityResult(
            model_name=model_name,
            ppl=float("nan"),
            loss=float("nan"),
            n_tokens=0,
            n_windows=n_windows,
            seq_len=seq_len,
            stride=stride,
            split=split,
        )

    mean_nll = nll_sum / n_tokens
    return PerplexityResult(
        model_name=model_name,
        ppl=float(math.exp(mean_nll)),
        loss=float(mean_nll),
        n_tokens=int(n_tokens),
        n_windows=int(n_windows),
        seq_len=int(seq_len),
        stride=int(stride),
        split=split,
        meta={"n_corpus_tokens": int(n_total)},
    )


def perplexity_from_text(
    model: Any,
    tokenizer: Any,
    text: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = DEFAULT_STRIDE,
    add_special_tokens: bool = False,
    model_name: str = "gpt2",
    verbose: bool = False,
) -> PerplexityResult:
    """Perplexity of a raw text string (tokenized then scored window-wise)."""
    ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    return perplexity_from_ids(
        model,
        ids,
        seq_len=seq_len,
        stride=stride,
        model_name=model_name,
        verbose=verbose,
    )


def perplexity_batched_windows(
    model: Any,
    windows: Iterable[Sequence[int]],
    batch_size: int = 8,
    model_name: str = "gpt2",
) -> PerplexityResult:
    """Non-overlapping batched perplexity over fixed-length windows.

    Convenience path used for quick checks (e.g. smoke tests) where double
    counting is avoided by construction: each window is scored in full except
    for its first token, which has no left context within the window.
    """
    model = _unwrap_model(model)
    device = next(model.parameters()).device
    nll_sum = 0.0
    n_tokens = 0
    n_windows = 0
    batch: List[torch.Tensor] = []

    def _flush(batch: List[torch.Tensor]) -> Tuple[float, int, int]:
        if not batch:
            return 0.0, 0, 0
        ids = torch.stack(batch, dim=0).to(device)
        targets = ids.clone()
        targets[:, 0] = -100
        with torch.inference_mode():
            out = model(ids, labels=targets)
        n = int((targets != -100).sum().item())
        return float(out.loss.detach().float().cpu()) * n, n, len(batch)

    for window in windows:
        batch.append(torch.tensor([int(t) for t in window], dtype=torch.long))
        if len(batch) >= int(batch_size):
            s, n, b = _flush(batch)
            nll_sum += s
            n_tokens += n
            n_windows += b
            batch = []
    s, n, b = _flush(batch)
    nll_sum += s
    n_tokens += n
    n_windows += b

    if n_tokens == 0:
        return PerplexityResult(model_name=model_name, n_windows=n_windows)
    mean_nll = nll_sum / n_tokens
    return PerplexityResult(
        model_name=model_name,
        ppl=float(math.exp(mean_nll)),
        loss=float(mean_nll),
        n_tokens=int(n_tokens),
        n_windows=int(n_windows),
    )


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    corpus: Any = None,
    split: str = DEFAULT_SPLIT,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = DEFAULT_STRIDE,
    cache_dir: Optional[str] = None,
    model_name: str = "gpt2",
    max_windows: Optional[int] = None,
    verbose: bool = False,
    device: Optional[str] = None,
) -> PerplexityResult:
    """Measure Wikitext-2 perplexity for ``model`` (Section 3.3 "Perplexity").

    Parameters
    ----------
    corpus:
        Optional pre-tokenized corpus: either a ``data.wikitext.PPLCorpus``
        (its ``input_ids`` are used) or a flat sequence of token ids.  When
        ``None`` the Wikitext-2 split is loaded and tokenized on the fly via
        :func:`data.wikitext.load_ppl_corpus`.
    split:
        Wikitext-2 split, ``"test"`` by default.
    seq_len, stride:
        Sliding-window configuration (defaults 1024 / 512).
    max_windows:
        Optional cap on the number of windows (used by ``--quick`` smoke runs).

    Notes
    -----
    Call this inside an intervention context (e.g. the residual-stream
    subtraction context manager from :mod:`src.interventions`) to obtain the
    perplexity of the *intervened* model, mirroring the paper's requirement that
    alpha is chosen so that the intervened perplexity is close to the post-DPO
    perplexity (~23.34 for GPT2-medium).
    """
    model = _unwrap_model(model)
    if device is not None:
        model.to(resolve_device(device))

    ids: Sequence[int]
    if corpus is None:
        from ...data.wikitext import load_ppl_corpus  # lazy: avoids network import

        ppl_corpus = load_ppl_corpus(
            tokenizer,
            split=split,
            seq_len=seq_len,
            stride=stride,
            cache_dir=cache_dir,
        )
        ids = list(getattr(ppl_corpus, "input_ids", []))
    elif hasattr(corpus, "input_ids"):
        ids = list(corpus.input_ids)
    else:
        ids = list(corpus)

    result = perplexity_from_ids(
        model,
        ids,
        seq_len=seq_len,
        stride=stride,
        max_windows=max_windows,
        model_name=model_name,
        split=split,
        verbose=verbose,
    )
    result.meta.setdefault("stride", result.stride)
    result.meta["device"] = str(next(model.parameters()).device)
    return result


#: Convenience alias used by evaluation scripts.
wikitext_perplexity = evaluate_perplexity


def compare_perplexity(
    before: PerplexityResult,
    after: PerplexityResult,
) -> Dict[str, Any]:
    """Relative perplexity change between two model states (Table 2/4 style)."""
    out: Dict[str, Any] = {
        "before": before.model_name,
        "after": after.model_name,
        "ppl_before": float(before.ppl),
        "ppl_after": float(after.ppl),
    }
    if before.ppl and not math.isnan(before.ppl) and before.ppl > 0:
        out["abs_delta"] = float(after.ppl - before.ppl)
        out["relative_delta"] = float((after.ppl - before.ppl) / before.ppl)
    return out


def match_alpha_for_target_ppl(
    model: Any,
    tokenizer: Any,
    make_intervention: Any,
    target_ppl: float = GPT2_DPO_PPL,
    corpus: Any = None,
    alphas: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    tolerance: float = 0.05,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = DEFAULT_STRIDE,
    model_name: str = "gpt2",
    model_factory: Any = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Choose ``alpha`` so the intervened Wikitext-2 PPL matches ``target_ppl``.

    The paper states: *"our interventions depend on how much we scale each
    vector (alpha). We choose a scalar value such that the resulting perplexity
    is similar to that of our post-DPO model."*  This helper performs that search.

    Parameters
    ----------
    model:
        Model to patch (patched in place by the intervention context manager).
    make_intervention:
        Callable ``alpha -> context manager`` applying the intervention
        (typically ``functools.partial(model_intervention, ...)`` from
        :mod:`src.interventions`).
    model_factory:
        Optional callable returning a *fresh* ``(model, tokenizer)`` triple for
        each alpha, used when the intervention mutates weights irreversibly.
        When provided, it takes precedence over ``model``.
    """
    table: List[Dict[str, float]] = []
    best: Optional[Dict[str, Any]] = None
    for alpha in alphas:
        trial_model = model
        if model_factory is not None:
            trial_model, tokenizer = model_factory()
        ctx = make_intervention(alpha)
        with ctx:
            res = evaluate_perplexity(
                trial_model,
                tokenizer,
                corpus=corpus,
                seq_len=seq_len,
                stride=stride,
                model_name=f"{model_name}_alpha{alpha}",
                verbose=False,
            )
        entry = {"alpha": float(alpha), "ppl": float(res.ppl)}
        table.append(entry)
        if verbose:
            print(f"[ppl-alpha] alpha={alpha:<8} ppl={res.ppl:.4f} target={target_ppl}")
        if math.isnan(res.ppl):
            continue
        gap = abs(res.ppl - target_ppl)
        entry["gap"] = float(gap)
        if best is None or gap < best["gap"]:
            best = {"alpha": float(alpha), "ppl": float(res.ppl), "gap": float(gap)}
        if gap <= tolerance:
            break
    return {
        "target_ppl": float(target_ppl),
        "best": best,
        "table": table,
        "tolerance": float(tolerance),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def default_path(model_name: str = "gpt2", out_dir: str = ARTIFACT_DIR) -> str:
    """Default JSON artifact path for a perplexity result."""
    safe = str(model_name).replace("/", "_")
    return os.path.join(out_dir, PERPLEXITY_RESULT_FILENAME.format(model=safe))


def save_result(path: str, result: PerplexityResult) -> str:
    """Write a :class:`PerplexityResult` to JSON (creating parent dirs)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    return path


def load_result(path: str) -> PerplexityResult:
    """Read a :class:`PerplexityResult` from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return PerplexityResult.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_perplexity_comparison(
    results: Sequence[PerplexityResult],
    labels: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
    title: str = "Wikitext-2 perplexity",
    ylabel: str = "perplexity",
    figsize: Tuple[float, float] = (6.5, 4.0),
    annotate: bool = True,
    baseline: Optional[float] = GPT2_PPL,
) -> Optional[str]:
    """Bar chart comparing perplexity across model states (Table 2/4 style)."""
    if not results:
        return None
    from ..analysis.plots import (
        bar_with_errors,
        make_figure,
        save_figure,
    )

    fig, ax = make_figure(figsize=figsize)
    names = list(labels) if labels is not None else [r.model_name for r in results]
    values = [float(r.ppl) for r in results]
    bars = bar_with_errors(ax, names, values, annotate=annotate, fmt="{:.2f}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if baseline is not None:
        ax.axhline(baseline, color="#888888", linestyle="--", linewidth=0.9)
    try:
        ax.tick_params(axis="x", rotation=15)
    except Exception:  # pragma: no cover - cosmetic only
        pass
    return save_figure(fig, out_path) if out_path else None


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #


def _main() -> int:  # pragma: no cover - manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="Wikitext-2 perplexity smoke test")
    parser.add_argument("--model", default=GPT2_MEDIUM)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=2)
    args = parser.parse_args()

    from ..model_utils import load_model

    model, tokenizer = load_model(args.model, device="cpu")
    text = (
        "The game began with a series of rapid moves . "
        "In 1971 , the club was founded by a group of enthusiasts . "
        "The city later became the capital of the region ."
    )
    res = perplexity_from_text(
        model,
        tokenizer,
        text,
        seq_len=args.seq_len,
        stride=args.stride,
        model_name=args.model,
        verbose=True,
    )
    print(res.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
