"""Section 5.3 — visualising the vocabulary reordering induced by CFG.

Paper (Section 5.3, "Visualizing Classifier-Free Guidance"):

    "Finally, we provide qualitative insights into the reordering of the
     vocabulary induced by CFG. We visualize the vocabulary at each timestep
     ranked by the difference
         log P(w_t | w_<t)  -  log P(w_T | w_hat),
     showing which tokens are encouraged or discouraged the most. In Figure 3,
     we prompt a model with c = 'The dragon flew over Paris, France',
     c_bar = empty and observe that tokens about dragons and Paris get
     upweighted while tokens about other locations ('Queensland'), dates
     ('1913'), or topics ('hostages', 'voyages') are downweighted. This
     indicates that CFG encourages and discourages tokens due to their
     relatedness to the prompt."

This module implements exactly that ranking:

    * ``paper_difference``   -> ``log P(w_t | w_<t) - log P(w_T | w_hat)``
      (the literal expression of Section 5.3, where ``w_hat`` denotes the
      continuation produced under the CFG-guided distribution);
    * ``encouragement_scores`` -> its negation, ``log P_cfg(w) - log P_ref(w)``,
      which is the quantity we sort *descending* to obtain the "most
      encouraged" column of Table 3 and *ascending* for the "most
      discouraged" column.

``w_hat`` / the guided distribution is obtained from the same weights via the
CFG logit combination of Eq. 7 (``uncond + gamma * (cond - uncond)``) applied to
the raw pre-softmax logits, with the unconditional context either the
prefix-dropped prompt (``empty_prefix``) or a negative prompt ``c_bar``
(``c_bar = empty`` for the paper's Figure 3 setting).  The reference
distribution ``P(w_t | w_<t)`` is the ordinary conditional (vanilla) next-token
distribution of the same model; ``reference="instruct"`` instead uses an
instruction-tuned model's conditional distribution, which is how the paper's
Figure 3 / Table 3 qualitatively contrasts CFG against instruction tuning.

The module is split into

    * a pure NumPy math layer (:func:`rank_vocabulary`, :func:`rank_step`,
      :func:`paper_difference`, :func:`encouragement_scores`) that is
      CPU-testable with no model, and
    * a thin, model-driven :class:`VocabReorderer` walkthrough that runs the
      dual-context forward pass from :mod:`src.cfg.model_wrapper` step by step
      and emits the Table 3 columns.

Typical usage (see ``scripts/run_analysis.py``)::

    from src.cfg import CFGModelWrapper
    from src.analysis.visualize import run_table3_walkthrough, format_table3

    wrapper = CFGModelWrapper("tiiuae/falcon-7b", unconditional_mode="empty_prefix")
    rankings = run_table3_walkthrough(wrapper, n_steps=12, gamma=1.5)
    print(format_table3(rankings))
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional dependencies (everything degrades gracefully)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only with torch installed
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False


def _fallback_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _fallback_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


def _fallback_cfg_combine(uncond, cond, gamma: float = 1.0):
    return np.asarray(uncond, dtype=np.float64) + float(gamma) * (
        np.asarray(cond, dtype=np.float64) - np.asarray(uncond, dtype=np.float64)
    )


try:  # reuse the canonical implementations from src/cfg/logits.py
    from ..cfg.logits import (  # type: ignore
        cfg_combine as _cfg_combine,
        log_softmax as _log_softmax,
        softmax as _softmax,
    )

    _HAS_LOGITS = True
except Exception:  # pragma: no cover
    try:
        from src.cfg.logits import (  # type: ignore
            cfg_combine as _cfg_combine,
            log_softmax as _log_softmax,
            softmax as _softmax,
        )

        _HAS_LOGITS = True
    except Exception:
        _HAS_LOGITS = False
        _cfg_combine = _fallback_cfg_combine  # type: ignore[assignment]
        _log_softmax = _fallback_log_softmax  # type: ignore[assignment]
        _softmax = _fallback_softmax  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: The prompt used for Figure 3 / Table 3 of the paper.
DRAGON_PROMPT = "The dragon flew over Paris, France"

#: ``c_bar = empty`` for the Figure 3 / Table 3 setting (Eq. 5 with an empty
#: negative prompt == the prompt-dropped / unconditional pass).
DRAGON_NEGATIVE_PROMPT = ""

#: Guidance strength used in the paper's qualitative Figure 3 (Section 5 uses
#: ``gamma = 1.5`` throughout the analysis).
TABLE3_GAMMA = 1.5
ANALYSIS_GAMMA = 1.5

#: Number of tokens shown per column of Table 3.
TOP_K_DISPLAY = 5

#: Paper-continuation budget for the qualitative walkthrough (the transcript in
#: Figure 3 is short: "The dragon flew over Paris, France, ...").
TABLE3_STEPS = 12

#: Default model for the Section 5 analysis (Falcon-7b-Base, per Section 5).
DEFAULT_MODEL = "tiiuae/falcon-7b"

#: Qualitative anchors reported in Section 5.3 (upweighted / downweighted).
EXPECTED_ENCOURAGED = ("dragon", "dragons", "flew", "Paris", "France")
EXPECTED_DISCOURAGED = ("Queensland", "1913", "hostages", "voyages")

#: Top-p used by the sibling Section 5 analyses; kept here so a ranking can be
#: intersected with a nucleus if desired.
TOP_P = 0.9

REFERENCE_MODES = ("cond", "vanilla", "uncond", "instruct")


# --------------------------------------------------------------------------- #
# Small array / token utilities
# --------------------------------------------------------------------------- #
def _to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor / array-like to a float64 NumPy array."""
    if _HAS_TORCH and torch is not None and isinstance(x, torch.Tensor):
        return x.detach().to(torch.float64).cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def last_position(logits: Any) -> np.ndarray:
    """Return a ``[vocab]`` array from ``[vocab]`` / ``[batch, seq, vocab]`` input."""
    arr = _to_numpy(logits)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[-1]
    if arr.ndim == 3:
        return arr[0, -1]
    raise ValueError(f"unexpected logits shape {arr.shape}")


def decode_token_ids(
    ids: Sequence[int],
    tokenizer: Any = None,
    strip: bool = True,
) -> List[str]:
    """Decode a list of token ids to display strings.

    When no tokenizer is available the ids are rendered as ``<id:N>`` so the
    table stays readable (and remains reproducible) on CPU-only machines.
    """
    out: List[str] = []
    for tid in ids:
        text: Optional[str] = None
        if tokenizer is not None:
            try:
                text = tokenizer.decode([int(tid)])
            except Exception:
                try:
                    text = tokenizer.convert_ids_to_tokens(int(tid))
                except Exception:
                    text = None
        if text is None:
            text = f"<id:{int(tid)}>"
        if strip:
            text = text.replace("\n", "\\n").strip()
        out.append(text if text else repr(text))
    return out


def rank_vocabulary(
    scores: np.ndarray,
    top_k: int = TOP_K_DISPLAY,
    descending: bool = True,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Indices of the ``top_k`` highest (or lowest) scores.

    NaN entries are pushed to the end so a masked/``-inf`` vocabulary never
    corrupts the ranking.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if mask is not None:
        m = np.asarray(mask).reshape(-1).astype(bool)
        s = np.where(m, s, -np.inf if descending else np.inf)
    order = np.argsort(-s if descending else s, kind="mergesort")
    finite = np.isfinite(s[order])
    order = order[finite] if finite.any() else order
    return order[: int(top_k)]


# --------------------------------------------------------------------------- #
# Core Section 5.3 math
# --------------------------------------------------------------------------- #
def encouragement_scores(
    logp_guided: np.ndarray,
    logp_reference: np.ndarray,
) -> np.ndarray:
    """``log P_cfg(w) - log P_ref(w)`` per vocabulary entry.

    Positive values are tokens CFG *encourages* relative to the reference
    distribution; negative values are *discouraged* tokens.
    """
    return _to_numpy(logp_guided).reshape(-1) - _to_numpy(logp_reference).reshape(-1)


def paper_difference(
    logp_reference: np.ndarray,
    logp_guided: np.ndarray,
) -> np.ndarray:
    """The literal Section 5.3 ranking expression.

    ``log P(w_t | w_<t) - log P(w_T | w_hat)`` == ``-encouragement_scores(...)``.
    The paper lists the tokens *encouraged* the most first, so the encouraging
    quantities are ``-paper_difference``; both are exposed for clarity.
    """
    return _to_numpy(logp_reference).reshape(-1) - _to_numpy(logp_guided).reshape(-1)


def guided_logprobs(
    logits_cond: Any,
    logits_uncond: Any,
    gamma: float = TABLE3_GAMMA,
    temperature: float = 1.0,
) -> np.ndarray:
    """``log softmax`` of the Eq. 7 combination of the two logit vectors.

    CFG is applied to the **raw pre-softmax logits** (matching
    :mod:`src.cfg.sampler`), then temperature, then log-softmax over the
    vocabulary.
    """
    cond = last_position(logits_cond)
    uncond = last_position(logits_uncond)
    if float(gamma) == 1.0:
        combined = cond
    elif float(gamma) == 0.0:
        combined = uncond
    else:
        combined = _to_numpy(_cfg_combine(uncond, cond, float(gamma))).reshape(-1)
    combined = combined / max(float(temperature), 1e-8)
    return _to_numpy(_log_softmax(combined)).reshape(-1)


def reference_logprobs(logits: Any, temperature: float = 1.0) -> np.ndarray:
    """``log softmax`` of the reference (e.g. vanilla conditional) logits."""
    ref = last_position(logits) / max(float(temperature), 1e-8)
    return _to_numpy(_log_softmax(ref)).reshape(-1)


def top_and_bottom_ids(
    scores: np.ndarray,
    top_k: int = TOP_K_DISPLAY,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(most encouraged ids, most discouraged ids)`` from encouragement scores."""
    hi = rank_vocabulary(scores, top_k=top_k, descending=True, mask=mask)
    lo = rank_vocabulary(-np.asarray(scores, dtype=np.float64).reshape(-1),
                         top_k=top_k, descending=True, mask=mask)
    return hi, lo


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class TokenRanking:
    """One column of Table 3 (a set of up- or down-weighted vocabulary items)."""

    ids: List[int] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    encouragement: List[float] = field(default_factory=list)
    logp_guided: List[float] = field(default_factory=list)
    logp_reference: List[float] = field(default_factory=list)
    kind: str = "encouraged"

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.ids)

    def __iter__(self):  # pragma: no cover - convenience
        return iter(self.tokens)

    @property
    def difference(self) -> List[float]:
        """Section 5.3 expression values (``-encouragement``)."""
        return [-float(v) for v in self.encouragement]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ids": [int(i) for i in self.ids],
            "tokens": list(self.tokens),
            "encouragement": [float(v) for v in self.encouragement],
            "difference": self.difference,
            "logp_guided": [float(v) for v in self.logp_guided],
            "logp_reference": [float(v) for v in self.logp_reference],
        }

    def render(self, sep: str = " | ") -> str:
        return sep.join(self.tokens)


@dataclass
class StepRanking:
    """The full vocabulary reordering at a single decoding timestep."""

    step: int = 0
    context: str = ""
    gamma: float = TABLE3_GAMMA
    reference: str = "vanilla"
    next_token: str = ""
    next_token_id: Optional[int] = None
    next_token_logp: Optional[float] = None
    encouraged: TokenRanking = field(default_factory=lambda: TokenRanking(kind="encouraged"))
    discouraged: TokenRanking = field(default_factory=lambda: TokenRanking(kind="discouraged"))
    scores: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self, include_scores: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": int(self.step),
            "context": self.context,
            "gamma": float(self.gamma),
            "reference": self.reference,
            "next_token": self.next_token,
            "next_token_id": (None if self.next_token_id is None else int(self.next_token_id)),
            "next_token_logp": (None if self.next_token_logp is None else float(self.next_token_logp)),
            "encouraged": self.encouraged.as_dict(),
            "discouraged": self.discouraged.as_dict(),
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        if include_scores and self.scores is not None:
            out["scores"] = [float(v) for v in np.asarray(self.scores).reshape(-1)]
        return out

    def format_lines(self) -> List[str]:
        return [
            f"step {self.step:>2} | ctx: {self.context!r}",
            f"        encouraged : {self.encouraged.render()}",
            f"        discouraged: {self.discouraged.render()}",
        ]


# --------------------------------------------------------------------------- #
# The one-step ranking primitive
# --------------------------------------------------------------------------- #
def rank_step(
    logits_guided: Any,
    logits_reference: Any,
    tokenizer: Any = None,
    top_k: int = TOP_K_DISPLAY,
    step: int = 0,
    context: str = "",
    gamma: float = TABLE3_GAMMA,
    reference: str = "vanilla",
    temperature: float = 1.0,
    mask: Optional[np.ndarray] = None,
) -> StepRanking:
    """Rank the whole vocabulary at one timestep (Section 5.3).

    ``logits_guided`` are the (already CFG-combined) logits; ``logits_reference``
    the unguided conditional logits.  Both may be ``[vocab]`` or ``[batch, seq,
    vocab]`` (last position is used).
    """
    lp_guided = (
        _to_numpy(logits_guided).reshape(-1)
        if logits_guided.ndim == 1
        else (last_position(logits_guided) / max(float(temperature), 1e-8))
    )
    # Always route through log-softmax so callers can pass raw *logits*.
    lp_guided = _to_numpy(_log_softmax(lp_guided)).reshape(-1)
    lp_ref = reference_logprobs(logits_reference, temperature=temperature)

    scores = encouragement_scores(lp_guided, lp_ref)
    hi, lo = top_and_bottom_ids(scores, top_k=top_k, mask=mask)

    def _make(idx: np.ndarray, kind: str) -> TokenRanking:
        ids = [int(i) for i in np.asarray(idx).reshape(-1)]
        return TokenRanking(
            ids=ids,
            tokens=decode_token_ids(ids, tokenizer),
            encouragement=[float(scores[i]) for i in ids],
            logp_guided=[float(lp_guided[i]) for i in ids],
            logp_reference=[float(lp_ref[i]) for i in ids],
            kind=kind,
        )

    argmax_id = int(np.argmax(lp_guided))
    return StepRanking(
        step=int(step),
        context=context,
        gamma=float(gamma),
        reference=reference,
        next_token=(decode_token_ids([argmax_id], tokenizer) or [""])[0],
        next_token_id=argmax_id,
        next_token_logp=float(lp_guided[argmax_id]),
        encouraged=_make(hi, "encouraged"),
        discouraged=_make(lo, "discouraged"),
        scores=scores,
    )


# --------------------------------------------------------------------------- #
# Model-driven walkthrough
# --------------------------------------------------------------------------- #
def _encode(wrapper: Any, text: str):
    """Encode ``text`` with a wrapper's tokenizer, returning ``(ids, attention)``."""
    enc = wrapper.encode(text, return_tensors="pt")
    if isinstance(enc, dict):
        ids = enc.get("input_ids")
        mask = enc.get("attention_mask")
    else:  # pragma: no cover - transformers BatchEncoding is dict-like
        ids = getattr(enc, "input_ids", enc)
        mask = getattr(enc, "attention_mask", None)
    return ids, mask


def _call_dual(
    wrapper: Any,
    input_ids: Any,
    prompt_length: int,
    negative_input_ids: Any = None,
    negative_attention_mask: Any = None,
):
    """Call ``wrapper.dual_logits`` tolerating older/newer signatures."""
    base: Dict[str, Any] = {"prompt_length": int(prompt_length)}
    if negative_input_ids is not None:
        base["negative_input_ids"] = negative_input_ids
        if negative_attention_mask is not None:
            base["negative_attention_mask"] = negative_attention_mask
    for extra in ({"only_last": True}, {}):
        try:
            return wrapper.dual_logits(input_ids, **base, **extra)  # type: ignore[call-arg]
        except TypeError:
            continue
    try:  # last resort: positional prompt_length
        return wrapper.dual_logits(input_ids, negative_input_ids, int(prompt_length))  # type: ignore[misc]
    except TypeError as exc:  # pragma: no cover
        raise TypeError(f"CFGModelWrapper.dual_logits has an unsupported signature: {exc}") from exc


def _dual_attr(dual: Any, name: str) -> Any:
    if isinstance(dual, dict):  # pragma: no cover - defensive
        return dual[name]
    return getattr(dual, name)


class VocabReorderer:
    """Step-by-step Section 5.3 vocabulary re-ranking for CFG.

    Parameters
    ----------
    model_wrapper :
        A :class:`src.cfg.model_wrapper.CFGModelWrapper` (dual-context passes).
    gamma :
        Guidance strength used to build the guided distribution (paper: 1.5).
    top_k :
        Number of tokens per encouraged/discouraged column (paper: 5).
    reference :
        Which unguided distribution to difference against:
        ``"cond"``/``"vanilla"`` (same model, ordinary conditional),
        ``"uncond"`` (prefix-dropped / negative-prompt pass) or ``"instruct"``
        (requires ``instruct_wrapper``).
    instruct_wrapper :
        Optional second wrapper (Falcon-7b-Instruct) for ``reference="instruct"``.
    """

    def __init__(
        self,
        model_wrapper: Any,
        gamma: float = TABLE3_GAMMA,
        top_k: int = TOP_K_DISPLAY,
        temperature: float = 1.0,
        unconditional_mode: str = "empty_prefix",
        instruct_wrapper: Any = None,
        reference: str = "vanilla",
        negative_prompt: Optional[str] = None,
        decode_mode: str = "cfg",
        seed: int = 0,
    ) -> None:
        self.model_wrapper = model_wrapper
        self.gamma = float(gamma)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.unconditional_mode = unconditional_mode
        self.instruct_wrapper = instruct_wrapper
        self.reference = reference if reference in REFERENCE_MODES else "vanilla"
        self.negative_prompt = negative_prompt
        self.decode_mode = decode_mode
        self.seed = int(seed)

    # -- helpers ---------------------------------------------------------- #
    @property
    def tokenizer(self) -> Any:
        return getattr(self.model_wrapper, "tokenizer", None)

    def _guided(self, dual: Any, gamma: Optional[float] = None) -> np.ndarray:
        g = self.gamma if gamma is None else float(gamma)
        if g == 1.0:
            return last_position(_dual_attr(dual, "cond"))
        if g == 0.0:
            return last_position(_dual_attr(dual, "uncond"))
        return _to_numpy(
            _cfg_combine(
                last_position(_dual_attr(dual, "uncond")),
                last_position(_dual_attr(dual, "cond")),
                g,
            )
        ).reshape(-1)

    def _reference(
        self,
        dual: Any,
        input_ids: Any,
        prompt_length: int,
        reference: Optional[str] = None,
    ) -> np.ndarray:
        ref = (reference or self.reference).lower()
        if ref in ("cond", "vanilla"):
            return last_position(_dual_attr(dual, "cond"))
        if ref == "uncond":
            return last_position(_dual_attr(dual, "uncond"))
        if ref == "instruct":
            if self.instruct_wrapper is None:
                raise ValueError("reference='instruct' requires instruct_wrapper")
            inst_dual = _call_dual(self.instruct_wrapper, input_ids, prompt_length)
            return last_position(_dual_attr(inst_dual, "cond"))
        raise ValueError(f"unknown reference mode {reference!r}; expected one of {REFERENCE_MODES}")

    def ranking_at(
        self,
        input_ids: Any,
        prompt_length: int,
        negative_input_ids: Any = None,
        negative_attention_mask: Any = None,
        step: int = 0,
        context: str = "",
        gamma: Optional[float] = None,
        reference: Optional[str] = None,
    ) -> StepRanking:
        """Rank the vocabulary for an arbitrary (already-tokenised) context."""
        dual = _call_dual(
            self.model_wrapper,
            input_ids,
            prompt_length,
            negative_input_ids,
            negative_attention_mask,
        )
        logits_g = self._guided(dual, gamma)
        logits_r = self._reference(dual, input_ids, prompt_length, reference)
        return rank_step(
            logits_g,
            logits_r,
            tokenizer=self.tokenizer,
            top_k=self.top_k,
            step=step,
            context=context,
            gamma=self.gamma if gamma is None else float(gamma),
            reference=(reference or self.reference),
            temperature=self.temperature,
        )

    # -- main walkthrough -------------------------------------------------- #
    def walkthrough(
        self,
        prompt: str = DRAGON_PROMPT,
        n_steps: int = TABLE3_STEPS,
        gamma: Optional[float] = None,
        reference: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        progress: bool = False,
    ) -> List[StepRanking]:
        """Generate a continuation, recording the Table 3 columns at every step.

        The continuation is decoded greedily from the guided distribution by
        default (``decode_mode="cfg"``); pass ``decode_mode="cond"`` to follow
        the vanilla conditional distribution while still ranking with CFG.
        """
        wrapper = self.model_wrapper
        ids, _ = _encode(wrapper, prompt)
        prompt_length = int(ids.shape[-1])

        neg_ids = neg_mask = None
        np_text = self.negative_prompt if negative_prompt is None else negative_prompt
        if np_text is not None and np_text != "":
            try:
                neg_ids, neg_mask = _encode(wrapper, np_text)
            except Exception as exc:  # pragma: no cover
                logger.warning("could not encode negative prompt %r (%s); ignoring", np_text, exc)
                neg_ids = neg_mask = None

        if eos_token_id is None:
            eos_token_id = getattr(wrapper, "eos_token_id", None)

        rankings: List[StepRanking] = []
        for step in range(int(n_steps)):
            ctx = self._decode_context(ids, prompt_length)
            record = self.ranking_at(
                ids,
                prompt_length,
                neg_ids,
                neg_mask,
                step=step,
                context=ctx,
                gamma=gamma,
                reference=reference,
            )
            rankings.append(record)

            if self.decode_mode == "cond":
                next_id = int(np.argmax(_to_numpy(last_position(_dual_attr(
                    _call_dual(wrapper, ids, prompt_length, neg_ids, neg_mask), "cond")))))
            else:
                next_id = int(record.next_token_id) if record.next_token_id is not None else 0

            if _HAS_TORCH and torch is not None and isinstance(ids, torch.Tensor):
                nxt = torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)
                ids = torch.cat([ids, nxt], dim=-1)
            else:  # pragma: no cover - numpy fallback
                ids = np.concatenate([np.asarray(ids).reshape(1, -1), np.array([[next_id]])], axis=-1)

            if progress:
                logger.info("step %d: %s", step, record.format_lines()[0])

            if eos_token_id is not None and next_id == int(eos_token_id):
                logger.debug("hit EOS at step %d", step)
                break

        return rankings

    def _decode_context(self, ids: Any, prompt_length: int) -> str:
        tokenizer = self.tokenizer
        if tokenizer is None:
            return ""
        try:
            flat = _to_numpy(ids).reshape(-1).astype(int).tolist()
        except Exception:  # pragma: no cover
            return ""
        try:
            return tokenizer.decode(flat, skip_special_tokens=True).strip()
        except Exception:  # pragma: no cover
            return ""

    def table3(
        self,
        prompt: str = DRAGON_PROMPT,
        n_steps: int = TABLE3_STEPS,
        **kwargs: Any,
    ) -> str:
        """Convenience: run the walkthrough and render the Table 3 transcript."""
        rankings = self.walkthrough(prompt=prompt, n_steps=n_steps, **kwargs)
        return format_table3(rankings, top_k=self.top_k, prompt=prompt, gamma=self.gamma)


# --------------------------------------------------------------------------- #
# Rendering / reporting
# --------------------------------------------------------------------------- #
def table3_columns(
    rankings: Sequence[StepRanking],
) -> Dict[str, List[Any]]:
    """Programmatic view of Table 3: one row per timestep."""
    return {
        "step": [r.step for r in rankings],
        "context": [r.context for r in rankings],
        "encouraged": [list(r.encouraged.tokens) for r in rankings],
        "discouraged": [list(r.discouraged.tokens) for r in rankings],
        "encouraged_ids": [list(r.encouraged.ids) for r in rankings],
        "discouraged_ids": [list(r.discouraged.ids) for r in rankings],
        "encouraged_scores": [list(r.encouraged.encouragement) for r in rankings],
        "discouraged_scores": [list(r.discouraged.encouragement) for r in rankings],
    }


def format_table3(
    rankings: Sequence[StepRanking],
    top_k: int = TOP_K_DISPLAY,
    prompt: str = DRAGON_PROMPT,
    gamma: float = TABLE3_GAMMA,
    max_steps: Optional[int] = None,
) -> str:
    """Render the Table 3 / Figure 3 transcript as plain text."""
    lines = [
        "Table 3 — vocabulary reordering induced by CFG (Section 5.3)",
        f"  prompt c = {prompt!r}   c_bar = empty   gamma = {gamma}",
        f"  ranking: log P(w_t | w_<t) - log P(w_T | w_hat)  (top {top_k} encouraged / discouraged)",
        "",
    ]
    sel = list(rankings)[: max_steps if max_steps is not None else len(rankings)]
    for r in sel:
        lines.append(f"step {r.step:>2} | {r.context!r}")
        lines.append(f"  encouraged  (+): {r.encouraged.render()}")
        lines.append(f"  discouraged (-): {r.discouraged.render()}")
    return "\n".join(lines)


def table3_dataframe(rankings: Sequence[StepRanking]):
    """Table 3 as a :class:`pandas.DataFrame` (or a list of dicts if unavailable)."""
    cols = table3_columns(rankings)
    rows = [
        {
            "step": cols["step"][i],
            "context": cols["context"][i],
            "most_encouraged": ", ".join(cols["encouraged"][i]),
            "most_discouraged": ", ".join(cols["discouraged"][i]),
        }
        for i in range(len(cols["step"]))
    ]
    if not _HAS_PANDAS:  # pragma: no cover
        return rows
    return pd.DataFrame(rows)


def count_expected_hits(
    rankings: Sequence[StepRanking],
    expected_encouraged: Sequence[str] = EXPECTED_ENCOURAGED,
    expected_discouraged: Sequence[str] = EXPECTED_DISCOURAGED,
) -> Dict[str, Any]:
    """Fraction of the paper's qualitative anchors that show up in the columns.

    Section 5.3 reports that dragon/Paris-related tokens are upweighted while
    unrelated locations/dates/topics are downweighted; this measures how well a
    run reproduces that qualitative pattern.
    """
    up = " ".join(" ".join(r.encouraged.tokens) for r in rankings).lower()
    down = " ".join(" ".join(r.discouraged.tokens) for r in rankings).lower()
    up_hits = [t for t in expected_encouraged if t.lower() in up]
    down_hits = [t for t in expected_discouraged if t.lower() in down]
    return {
        "encouraged_hits": up_hits,
        "discouraged_hits": down_hits,
        "n_encouraged_expected": len(expected_encouraged),
        "n_discouraged_expected": len(expected_discouraged),
        "encouraged_hit_rate": (len(up_hits) / len(expected_encouraged)) if expected_encouraged else 0.0,
        "discouraged_hit_rate": (len(down_hits) / len(expected_discouraged)) if expected_discouraged else 0.0,
    }


def ranking_report(
    rankings: Sequence[StepRanking],
    expected_encouraged: Sequence[str] = EXPECTED_ENCOURAGED,
    expected_discouraged: Sequence[str] = EXPECTED_DISCOURAGED,
) -> Dict[str, Any]:
    """JSON-serialisable summary of a Section 5.3 run."""
    rankings = list(rankings)
    hits = count_expected_hits(rankings, expected_encouraged, expected_discouraged)
    return {
        "n_steps": len(rankings),
        "gamma": (rankings[0].gamma if rankings else TABLE3_GAMMA),
        "reference": (rankings[0].reference if rankings else "vanilla"),
        "prompt": DRAGON_PROMPT,
        "columns": table3_columns(rankings),
        "expected": {
            "encouraged": list(expected_encouraged),
            "discouraged": list(expected_discouraged),
        },
        "hits": hits,
        "steps": [r.as_dict() for r in rankings],
    }


def check_against_paper(
    rankings: Sequence[StepRanking],
    min_encouraged_hit_rate: float = 0.25,
    min_discouraged_hit_rate: float = 0.0,
) -> Dict[str, Any]:
    """Compare a run's Table 3 columns with the paper's qualitative claim."""
    hits = count_expected_hits(rankings)
    return {
        **hits,
        "min_encouraged_hit_rate": float(min_encouraged_hit_rate),
        "min_discouraged_hit_rate": float(min_discouraged_hit_rate),
        "encouraged_ok": hits["encouraged_hit_rate"] >= float(min_encouraged_hit_rate),
        "discouraged_ok": hits["discouraged_hit_rate"] >= float(min_discouraged_hit_rate),
        "matches_paper_direction": hits["encouraged_hit_rate"] >= float(min_encouraged_hit_rate),
    }


def summarize_rankings(rankings: Sequence[StepRanking]) -> str:
    """One-line textual summary."""
    rankings = list(rankings)
    if not rankings:
        return "no steps"
    top_up = rankings[0].encouraged.render(sep=", ")
    return (
        f"{len(rankings)} steps | gamma={rankings[0].gamma} | "
        f"top encouraged at step 0: {top_up}"
    )


# --------------------------------------------------------------------------- #
# Persistence / plotting
# --------------------------------------------------------------------------- #
def save_rankings(path: str, rankings: Sequence[StepRanking]) -> str:
    """Write the step rankings (including full score vectors) to JSON."""
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    payload = {
        "prompt": DRAGON_PROMPT,
        "gamma": (list(rankings)[0].gamma if rankings else TABLE3_GAMMA),
        "steps": [r.as_dict(include_scores=True) for r in rankings],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("wrote %d step rankings to %s", len(rankings), path)
    return path


def load_rankings(path: str) -> Dict[str, Any]:
    """Load a JSON dump produced by :func:`save_rankings`."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_rankings(
    rankings: Sequence[StepRanking],
    path: Optional[str] = None,
    max_steps: int = 8,
    top_k: int = TOP_K_DISPLAY,
) -> Optional[Any]:
    """Bar plot of the encouragement scores per step (matplotlib optional)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        logger.warning("matplotlib unavailable; skipping plot")
        return None

    sel = list(rankings)[: int(max_steps)]
    if not sel:
        return None
    fig, axes = plt.subplots(len(sel), 1, figsize=(7, 1.9 * len(sel)), squeeze=False)
    for ax, rec in zip(axes[:, 0], sel):
        toks = list(reversed(rec.discouraged.tokens)) + list(rec.encouraged.tokens)[::-1]
        vals = list(reversed(rec.discouraged.encouragement)) + list(rec.encouraged.encouragement)[::-1]
        colors = ["tab:red"] * len(rec.discouraged) + ["tab:green"] * len(rec.encouraged)
        ax.barh(range(len(toks)), vals, color=colors)
        ax.set_yticks(range(len(toks)))
        ax.set_yticklabels(toks, fontsize=6)
        ax.axvline(0.0, color="k", lw=0.6)
        ax.set_title(f"step {rec.step}: {rec.context[-40:]!r}", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150)
        logger.info("wrote vocabulary-ranking plot to %s", path)
        plt.close(fig)
        return path
    return fig


# --------------------------------------------------------------------------- #
# Functional convenience API
# --------------------------------------------------------------------------- #
def run_table3_walkthrough(
    model_wrapper: Any,
    prompt: str = DRAGON_PROMPT,
    n_steps: int = TABLE3_STEPS,
    gamma: float = TABLE3_GAMMA,
    top_k: int = TOP_K_DISPLAY,
    reference: str = "vanilla",
    negative_prompt: Optional[str] = None,
    instruct_wrapper: Any = None,
    decode_mode: str = "cfg",
    **kwargs: Any,
) -> List[StepRanking]:
    """Run the Section 5.3 walkthrough and return the per-step rankings."""
    reorderer = VocabReorderer(
        model_wrapper,
        gamma=gamma,
        top_k=top_k,
        reference=reference,
        negative_prompt=negative_prompt,
        instruct_wrapper=instruct_wrapper,
        decode_mode=decode_mode,
    )
    return reorderer.walkthrough(prompt=prompt, n_steps=n_steps, **kwargs)


def rank_from_logits(
    logits_cond: Any,
    logits_uncond: Any,
    tokenizer: Any = None,
    gamma: float = TABLE3_GAMMA,
    top_k: int = TOP_K_DISPLAY,
    step: int = 0,
    context: str = "",
    reference: str = "vanilla",
    temperature: float = 1.0,
) -> StepRanking:
    """Model-free ranking entry point (useful for tests and cached logits)."""
    lp_guided = guided_logprobs(logits_cond, logits_uncond, gamma=gamma, temperature=temperature)
    lp_ref = reference_logprobs(logits_cond, temperature=temperature)
    scores = encouragement_scores(lp_guided, lp_ref)
    hi, lo = top_and_bottom_ids(scores, top_k=top_k)

    def _make(idx: np.ndarray, kind: str) -> TokenRanking:
        ids = [int(i) for i in np.asarray(idx).reshape(-1)]
        return TokenRanking(
            ids=ids,
            tokens=decode_token_ids(ids, tokenizer),
            encouragement=[float(scores[i]) for i in ids],
            logp_guided=[float(lp_guided[i]) for i in ids],
            logp_reference=[float(lp_ref[i]) for i in ids],
            kind=kind,
        )

    return StepRanking(
        step=int(step),
        context=context,
        gamma=float(gamma),
        reference=reference,
        next_token=(decode_token_ids([int(np.argmax(lp_guided))], tokenizer) or [""])[0],
        next_token_id=int(np.argmax(lp_guided)),
        next_token_logp=float(np.max(lp_guided)),
        encouraged=_make(hi, "encouraged"),
        discouraged=_make(lo, "discouraged"),
        scores=scores,
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _demo() -> None:
    """CPU-only self-test of the Section 5.3 ranking math (no model needed)."""
    rng = np.random.RandomState(0)
    vocab = 20
    logits_cond = rng.randn(vocab)
    logits_uncond = rng.randn(vocab)

    # gamma = 1 must reproduce the vanilla conditional distribution exactly.
    lp1 = guided_logprobs(logits_cond, logits_uncond, gamma=1.0)
    lp_ref = reference_logprobs(logits_cond)
    assert np.allclose(lp1, lp_ref), "gamma=1 must equal the conditional distribution"

    # gamma = 0 must reproduce the unconditional distribution.
    lp0 = guided_logprobs(logits_cond, logits_uncond, gamma=0.0)
    assert np.allclose(lp0, reference_logprobs(logits_uncond)), "gamma=0 must equal uncond"

    # Eq. 7 in log space, up to the softmax normaliser.
    lp15 = guided_logprobs(logits_cond, logits_uncond, gamma=1.5)
    raw = logits_uncond + 1.5 * (logits_cond - logits_uncond)
    assert np.allclose(lp15, raw - np.log(np.sum(np.exp(raw - raw.max()))) - raw.max(), atol=1e-8)

    # Ranking identities: lifted token shows up first, suppressed token last.
    cond = np.zeros(vocab)
    uncond = np.zeros(vocab)
    cond[7] += 6.0        # strongly conditional token -> encouraged by CFG
    uncond[3] += 4.0      # purely unconditional token -> discouraged by CFG
    rec = rank_from_logits(cond, uncond, gamma=2.0, top_k=3)
    assert rec.encouraged.ids[0] == 7, rec.encouraged.ids
    assert rec.discouraged.ids[0] == 3, rec.discouraged.ids
    assert rec.encouraged.encouragement[0] > 0 > rec.discouraged.encouragement[0]

    # Paper expression is the negation of the encouragement score.
    assert np.allclose(paper_difference(lp_ref, lp1), -encouragement_scores(lp1, lp_ref))

    cols = table3_columns([rec])
    assert len(cols["encouraged"][0]) == 3 and len(cols["discouraged"][0]) == 3

    stub = StepRanking(step=0, context="The dragon flew over Paris, France")
    stub.encouraged = TokenRanking(tokens=["dragon", "Paris"], kind="encouraged")
    stub.discouraged = TokenRanking(tokens=["Queensland", "1913"], kind="discouraged")
    report = check_against_paper([stub], min_encouraged_hit_rate=0.2, min_discouraged_hit_rate=0.2)
    assert report["encouraged_ok"] and report["discouraged_ok"], report
    assert "dragon" in format_table3([stub]).lower()
    print("[visualize] self-test OK")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _demo()
