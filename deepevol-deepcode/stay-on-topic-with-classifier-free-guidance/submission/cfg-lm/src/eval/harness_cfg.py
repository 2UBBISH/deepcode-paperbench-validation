"""EleutherAI Language-Model-Evaluation-Harness shim for Classifier-Free Guidance.

Paper: "Stay on Topic with Classifier-Free Guidance" (Sanchez et al.).

Section 3.1 evaluates zero-shot benchmarks implemented in the Language Model
Evaluation Harness (Gao et al. 2021) -- close-book QA (ARC, BoolQ, TriviaQA),
common-sense reasoning (HellaSwag, PIQA, SciQ, WinoGrande) and sentence
completion (LAMBADA-OpenAI) -- applying CFG by

    "starting the unconditional prompt at the last token of the initial prompt"
    (Section 3.1)

so that for a harness request ``(context, continuation)`` we run two forward
passes through the same LM weights:

* conditional   : ``context + continuation``           -> ``logits_cond``
* unconditional : ``context[-1:] + continuation``       -> ``logits_uncond``

and combine the per-position next-token logits in log space *before* any
softmax (Eq. 7):

    guided = logits_uncond + gamma * (logits_cond - logits_uncond)

``gamma = 1`` recovers vanilla conditional scoring, ``gamma = 0`` recovers
unconditional scoring.  Because the combined vector is not a normalized
distribution, the sequence log-likelihood is obtained by applying a log-softmax
over the combined logits (this is the log-space CFG distribution used
throughout the paper).

Appendix C.1 clarification implemented here: TriviaQA is scored with a
*substring* match instead of an exact match (see
:mod:`src.eval.triviaqa_match`), because exact matching disqualified answers
like ``"Mark Twain"`` (with quotes) or ``His name is Mark Twain``.

The module is import-safe without ``lm_eval``/``torch``/``transformers``
installed: it falls back to a plain-object base class so the helper functions
(``combine_guided_logits``, ``sequence_logprobs``, ``is_greedy_continuation``)
remain unit-testable on CPU-only machines.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #
try:  # pure-math primitives (numpy fallback inside)
    from ..cfg.logits import cfg_combine, log_softmax  # type: ignore
except Exception:  # pragma: no cover - standalone / flat import
    try:
        from src.cfg.logits import cfg_combine, log_softmax  # type: ignore
    except Exception:
        cfg_combine = None  # type: ignore
        log_softmax = None  # type: ignore

    if cfg_combine is None:  # pragma: no cover
        def _fallback_combine(  # type: ignore
            logits_uncond: np.ndarray, logits_cond: np.ndarray, gamma: float
        ) -> np.ndarray:
            """Eq. 7 in pure numpy (``uncond + gamma * (cond - uncond)``)."""
            uncond = np.asarray(logits_uncond, dtype=np.float64)
            cond = np.asarray(logits_cond, dtype=np.float64)
            return uncond + float(gamma) * (cond - uncond)

        cfg_combine = _fallback_combine  # type: ignore

try:  # torch is required for the real model path
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # lm-evaluation-harness base class (optional)
    from lm_eval.api.model import LM as _HarnessLM  # type: ignore
except Exception:  # pragma: no cover - harness not installed

    class _HarnessLM:  # type: ignore
        """Minimal stand-in for ``lm_eval.api.model.LM``."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


# --------------------------------------------------------------------------- #
# Constants (shared with configs/default.yaml and scripts/run_zero_shot.py)
# --------------------------------------------------------------------------- #
HARNESS_GAMMAS: Tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
"""Guidance strengths swept in Table 5 (Section 3.1)."""

HARNESS_TASKS: Tuple[str, ...] = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "piqa",
    "sciq",
    "triviaqa",
    "winogrande",
    "lambada_openai",
)
"""The nine zero-shot benchmarks of Table 5."""

HARNESS_DEFAULT_TEMPERATURE: float = 0.0
"""Harness defaults are greedy / argmax unless stated otherwise."""

HARNESS_DEFAULT_TOP_P: float = 1.0
HARNESS_DEFAULT_MAX_LENGTH: int = 2048

#: Tasks whose accuracy in the paper is computed with substring rather than
#: exact matching (Appendix C.1).
SUBSTRING_MATCH_TASKS: Tuple[str, ...] = ("triviaqa",)

#: ``unconditional_mode`` values understood by this shim.
#:
#: * ``"last_prompt_token"`` -- unconditional prompt begins at the last token of
#:   the initial prompt (the Section 3.1 harness convention, the default here).
#: * ``"empty_prefix"`` -- the prefix is dropped entirely.
UNCONDITIONAL_MODES: Tuple[str, ...] = ("last_prompt_token", "empty_prefix")

#: Rolling-loglikelihood (LAMBADA-OpenAI) unconditional-context conventions.
ROLLING_MODES: Tuple[str, ...] = ("shift_by_one", "empty_prefix")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class HarnessCFGConfig:
    """Configuration of the CFG scoring shim.

    Parameters
    ----------
    gamma:
        Guidance strength.  ``1.0`` == vanilla conditional scoring,
        ``0.0`` == unconditional scoring.
    unconditional_mode:
        ``"last_prompt_token"`` (Section 3.1 convention) or ``"empty_prefix"``.
        Ignored when an explicit ``negative_prompt`` is provided.
    combine_mode:
        Which positions of a multi-token continuation receive the CFG
        combination.  ``"all_tokens"`` (default, the paper's per-step rule) or
        ``"first_token"`` (only the first scored token is guided, the remaining
        tokens are scored conditionally).
    rolling_mode:
        Unconditional context used by ``loglikelihood_rolling``:
        ``"shift_by_one"`` (unconditional context is the immediately preceding
        token -- the rolling analogue of "last token of the initial prompt") or
        ``"empty_prefix"``.
    temperature, top_p:
        Decoding defaults for :meth:`CFGHarnessLM.generate_until`; the harness
        uses greedy decoding (``0.0``/``1.0``) unless set otherwise.
    max_length:
        Left-truncation budget for the context, mirroring ``lm_eval``'s
        ``max_length`` attribute.
    batch_size:
        Request batching size (accepted for harness-API compatibility).
    substring_match:
        Override for TriviaQA-style substring scoring; ``None`` auto-detects
        from the task name.
    seed:
        Optional deterministic seed for sampling-based generation.
    """

    gamma: float = 1.0
    unconditional_mode: str = "last_prompt_token"
    combine_mode: str = "all_tokens"
    rolling_mode: str = "shift_by_one"
    temperature: float = HARNESS_DEFAULT_TEMPERATURE
    top_p: float = HARNESS_DEFAULT_TOP_P
    top_k: int = 0
    max_length: int = HARNESS_DEFAULT_MAX_LENGTH
    max_gen_toks: int = 256
    batch_size: int = 1
    substring_match: Optional[bool] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.unconditional_mode not in UNCONDITIONAL_MODES:
            raise ValueError(
                f"unconditional_mode must be one of {UNCONDITIONAL_MODES}, "
                f"got {self.unconditional_mode!r}"
            )
        if self.rolling_mode not in ROLLING_MODES:
            raise ValueError(
                f"rolling_mode must be one of {ROLLING_MODES}, "
                f"got {self.rolling_mode!r}"
            )
        if self.combine_mode not in ("all_tokens", "first_token"):
            raise ValueError(
                "combine_mode must be 'all_tokens' or 'first_token', got "
                f"{self.combine_mode!r}"
            )

    def replace(self, **kwargs: Any) -> "HarnessCFGConfig":
        """Return a copy with the given fields replaced."""
        data = dict(self.__dict__)
        data.update(kwargs)
        return HarnessCFGConfig(**data)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Pure-math helpers (unit-testable without a model)
# --------------------------------------------------------------------------- #
def combine_guided_logits(
    logits_cond: np.ndarray,
    logits_uncond: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Eq. 7 combiner on numpy arrays of shape ``[..., vocab]``.

    ``guide`` is the *pre-softmax* logit level so that ``gamma = 1`` reduces
    exactly to the conditional logits (and ``gamma = 0`` to the unconditional
    ones).
    """
    if cfg_combine is None:  # pragma: no cover
        uncond = np.asarray(logits_uncond, dtype=np.float64)
        cond = np.asarray(logits_cond, dtype=np.float64)
        return uncond + float(gamma) * (cond - uncond)
    return np.asarray(
        cfg_combine(logits_uncond, logits_cond, gamma)
        if not _HAS_TORCH
        else _to_numpy(cfg_combine(_to_torch(logits_uncond), _to_torch(logits_cond), gamma))
    )


def _to_torch(array: Any) -> Any:
    if _HAS_TORCH and not torch.is_tensor(array):
        return torch.as_tensor(np.asarray(array))
    return array


def _to_numpy(value: Any) -> np.ndarray:
    if _HAS_TORCH and torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if log_softmax is not None:
        try:
            return _to_numpy(log_softmax(x)) if _HAS_TORCH else np.asarray(log_softmax(x))
        except Exception:  # pragma: no cover
            pass
    m = np.max(x, axis=-1, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def sequence_logprobs(
    logits_cond: np.ndarray,
    logits_uncond: Optional[np.ndarray],
    target_ids: Sequence[int],
    gamma: float = 1.0,
    combine_mode: str = "all_tokens",
) -> Tuple[float, np.ndarray]:
    """Sequence log-likelihood of ``target_ids`` under the CFG distribution.

    Parameters
    ----------
    logits_cond:
        ``[n_targets, vocab]`` conditional next-token logits, aligned so that
        row ``i`` predicts ``target_ids[i]``.
    logits_uncond:
        ``[n_targets, vocab]`` unconditional logits with the same alignment.
        ``None`` means vanilla conditional scoring.
    target_ids:
        The continuation token ids being scored.
    gamma:
        Guidance strength (Eq. 7).
    combine_mode:
        ``"all_tokens"`` guides every token; ``"first_token"`` guides only the
        first one and falls back to conditional scoring afterwards.

    Returns
    -------
    (total_logprob, per_token_logprob)
    """
    cond = np.asarray(logits_cond, dtype=np.float64)
    targets = np.asarray(list(target_ids), dtype=np.int64)
    n = targets.shape[0]
    if cond.ndim == 1:
        cond = cond[None, :]
    if cond.shape[0] < n:
        raise ValueError(
            f"need at least {n} conditional rows aligned to the targets, "
            f"got {cond.shape[0]}"
        )
    cond = cond[:n]

    if logits_uncond is None or float(gamma) == 1.0:
        guided = cond
    else:
        uncond = np.asarray(logits_uncond, dtype=np.float64)
        if uncond.ndim == 1:
            uncond = uncond[None, :]
        uncond = uncond[:n]
        guided = np.asarray(combine_guided_logits(cond, uncond, gamma), dtype=np.float64)
        if combine_mode == "first_token" and n > 1:
            guided = np.concatenate([guided[:1], cond[1:]], axis=0)

    logp = _log_softmax_np(guided)
    per_token = logp[np.arange(n), targets]
    return float(np.sum(per_token)), per_token


def is_greedy_continuation(
    logits_cond: np.ndarray,
    logits_uncond: Optional[np.ndarray],
    target_ids: Sequence[int],
    gamma: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> bool:
    """Harness ``is_greedy`` flag: is the continuation the argmax under CFG?"""
    cond = np.asarray(logits_cond, dtype=np.float64)
    if cond.ndim == 1:
        cond = cond[None, :]
    targets = list(target_ids)
    n = len(targets)
    cond = cond[:n]

    if logits_uncond is None or float(gamma) == 1.0:
        guided = cond
    else:
        uncond = np.asarray(logits_uncond, dtype=np.float64)
        if uncond.ndim == 1:
            uncond = uncond[None, :]
        guided = np.asarray(
            combine_guided_logits(cond, uncond[:n], gamma), dtype=np.float64
        )

    argmax = np.argmax(guided, axis=-1)
    for i, target in enumerate(targets):
        if eos_token_id is not None and target == eos_token_id and i != n - 1:
            # continuation that stops early is never "greedy"
            return False
        if int(argmax[i]) != int(target):
            return False
    return True


def tokenize_pair(
    tokenizer: Any,
    context: str,
    continuation: str,
) -> Tuple[List[int], List[int]]:
    """Tokenize a harness ``(context, continuation)`` pair the ``lm_eval`` way.

    The context is tokenized with special tokens, the continuation without, so
    that ``context_ids + continuation_ids`` is the token stream the harness
    scores (e.g. ``"Question: ...\\nAnswer:"`` + ``" yes"``).
    """
    ctx_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
    if isinstance(ctx_ids, list) and ctx_ids and isinstance(ctx_ids[0], list):
        ctx_ids = ctx_ids[0]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if isinstance(cont_ids, list) and cont_ids and isinstance(cont_ids[0], list):
        cont_ids = cont_ids[0]
    return list(ctx_ids), list(cont_ids)


def build_unconditional_inputs(
    context_ids: Sequence[int],
    continuation_ids: Sequence[int],
    mode: str = "last_prompt_token",
    bos_token_id: Optional[int] = None,
) -> Tuple[List[int], int]:
    """Build the unconditional token stream for a ``(context, continuation)``.

    Returns ``(uncond_ids, n_prefix_positions_to_skip)`` where the second value
    is how many leading positions of the unconditional stream do **not** align
    with a scored continuation token (they must be dropped before aligning).

    ``"last_prompt_token"`` (Section 3.1): the unconditional prompt begins at the
    last token of the initial prompt, i.e. the scored prefix is dropped except
    its final token.  ``"empty_prefix"``: no prompt token at all is kept.
    """
    cont = list(continuation_ids)
    ctx = list(context_ids)
    if mode == "empty_prefix":
        uncond = ([bos_token_id] if bos_token_id is not None else []) + cont
        skip = 1 if bos_token_id is not None else 0
        return uncond, skip
    if mode != "last_prompt_token":
        raise ValueError(f"unknown unconditional mode {mode!r}")
    if ctx:
        uncond = ctx[-1:] + cont
        return uncond, 0
    # empty context: fall back to an optional BOS / unconditional seed token
    uncond = ([bos_token_id] if bos_token_id is not None else []) + cont
    return uncond, 1 if bos_token_id is not None else 0


# --------------------------------------------------------------------------- #
# Forward-pass helpers used by the harness LM
# --------------------------------------------------------------------------- #
def _forward_logits(model: Any, input_ids: Any, attention_mask: Any = None) -> Any:
    """Single-no-grad forward pass returning ``[batch, seq, vocab]`` logits."""
    kwargs: Dict[str, Any] = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    out = model(input_ids=input_ids, **kwargs)
    return out.logits if hasattr(out, "logits") else out[0]


def _pad_batch(sequences: List[List[int]], pad_id: int, device: Any = None) -> Tuple[Any, Any]:
    """Right-pad a list of id sequences, returning ``(input_ids, attention_mask)``."""
    max_len = max(len(s) for s in sequences)
    ids = torch.full(
        (len(sequences), max_len), pad_id, dtype=torch.long, device=device
    )
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, : len(seq)] = 1
    return ids, mask


def request_args(request: Any) -> Tuple[Any, ...]:
    """Extract arguments from an ``lm_eval`` ``Instance`` or a plain tuple."""
    if hasattr(request, "args"):
        args = request.args
        return tuple(args) if not isinstance(args, str) else (args,)
    if isinstance(request, (tuple, list)):
        return tuple(request)
    return (request,)


# --------------------------------------------------------------------------- #
# The harness-compatible CFG language model
# --------------------------------------------------------------------------- #
class CFGHarnessLM(_HarnessLM):  # type: ignore[misc]
    """``lm_eval`` LM whose scoring logits are CFG-combined (Eq. 7).

    The class deliberately re-implements the two harness entry points
    (``loglikelihood`` and ``loglikelihood_rolling``) plus ``generate_until``
    (used by the TriviaQA task, Appendix C.1) rather than subclassing
    ``lm_eval.models.huggingface.HFLM``, because CFG requires *two* forward
    passes per scored position with the same weights.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace hub id / local path (e.g. ``"gpt2-large"``,
        ``"EleutherAI/pythia-6.9b"``).
    gamma:
        Guidance strength.
    unconditional_mode:
        ``"last_prompt_token"`` (Section 3.1) or ``"empty_prefix"``.
    model / tokenizer:
        Pre-built objects may be passed instead of ``model_name_or_path``
        (useful for tests and for reuse of an already loaded wrapper).
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        gamma: float = 1.0,
        unconditional_mode: str = "last_prompt_token",
        config: Optional[HarnessCFGConfig] = None,
        device: Any = "auto",
        dtype: Any = "auto",
        model: Any = None,
        tokenizer: Any = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        trust_remote_code: bool = False,
        negative_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg_kwargs: Dict[str, Any] = {"gamma": gamma, "unconditional_mode": unconditional_mode}
        if batch_size is not None:
            cfg_kwargs["batch_size"] = batch_size
        if max_length is not None:
            cfg_kwargs["max_length"] = max_length
        for key in (
            "combine_mode",
            "rolling_mode",
            "temperature",
            "top_p",
            "top_k",
            "max_gen_toks",
            "substring_match",
            "seed",
        ):
            if key in kwargs and kwargs[key] is not None:
                cfg_kwargs[key] = kwargs[key]
        self.config = (config or HarnessCFGConfig()).replace(**cfg_kwargs)
        self.negative_prompt = negative_prompt
        self.model_name_or_path = model_name_or_path

        self._tokenizer: Any = tokenizer
        self._model: Any = model
        self._wrapper: Any = None
        self._device = device
        self._dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.additional_kwargs = dict(kwargs)

        # harness API surface
        self.batch_size = self.config.batch_size
        self._max_length = self.config.max_length
        self._max_gen_toks = self.config.max_gen_toks
        self.cache_hook = None
        self.is_caching = False

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def create_from_arg_string(
        cls, arg_string: str = "", additional_config: Optional[Dict[str, Any]] = None
    ) -> "CFGHarnessLM":
        """``lm_eval``-style ``--model_args`` parser (``k=v,k=v``)."""
        args = dict((additional_config or {}))
        for chunk in (arg_string or "").split(","):
            if not chunk.strip():
                continue
            if "=" in chunk:
                key, value = chunk.split("=", 1)
                args[key.strip()] = value.strip()
        model_name = args.pop("pretrained", None) or args.pop("model", None)
        gamma = args.pop("gamma", 1.0)
        mode = args.pop("unconditional_mode", args.pop("cfg_mode", "last_prompt_token"))
        return cls(
            model_name_or_path=model_name,
            gamma=float(gamma),
            unconditional_mode=str(mode),
            **{k: _coerce(v) for k, v in args.items()},
        )

    def _ensure_wrapper(self) -> Any:
        """Lazily build the dual-context :class:`CFGModelWrapper`."""
        if self._wrapper is not None:
            return self._wrapper
        from ..cfg.model_wrapper import CFGModelWrapper  # local import: torch dep

        self._wrapper = CFGModelWrapper(
            model_name_or_path=self.model_name_or_path,
            device=self._device,
            dtype=self._dtype,
            unconditional_mode=self.config.unconditional_mode,
            model=self._model,
            tokenizer=self._tokenizer,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = self._wrapper.model
        self._tokenizer = self._wrapper.tokenizer
        return self._wrapper

    # ------------------------------------------------------------------ #
    # properties required by lm_eval
    # ------------------------------------------------------------------ #
    @property
    def eot_token_id(self) -> Optional[int]:
        wrapper = self._ensure_wrapper()
        return wrapper.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        wrapper = self._ensure_wrapper()
        return getattr(wrapper.tokenizer, "name_or_path", str(self.model_name_or_path))

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _score_one(
        self, context: str, continuation: str, gamma: Optional[float] = None
    ) -> Tuple[float, bool]:
        """CFG score of one ``(context, continuation)`` pair."""
        wrapper = self._ensure_wrapper()
        gamma = self.config.gamma if gamma is None else float(gamma)
        tokenizer = wrapper.tokenizer

        ctx_ids, cont_ids = tokenize_pair(tokenizer, context, continuation)
        if not cont_ids:  # nothing to score
            return 0.0, False

        max_len = self._max_length
        if len(ctx_ids) + len(cont_ids) > max_len:
            # left-truncate the context (harness convention)
            keep = max(1, max_len - len(cont_ids))
            ctx_ids = ctx_ids[-keep:]

        bos_id = getattr(wrapper, "bos_token_id", None)
        uncond_ids, skip = build_unconditional_inputs(
            ctx_ids,
            cont_ids,
            mode=self.config.unconditional_mode,
            bos_token_id=bos_id if not ctx_ids else None,
        )

        device = wrapper.device
        cond_ids_t = torch.tensor([ctx_ids + cont_ids], dtype=torch.long, device=device)
        uncond_ids_t = torch.tensor([uncond_ids], dtype=torch.long, device=device)
        cond_mask = torch.ones_like(cond_ids_t)
        uncond_mask = torch.ones_like(uncond_ids_t)

        with torch.no_grad():
            cond_logits = _forward_logits(self._model, cond_ids_t, cond_mask)[0]
            uncond_logits = _forward_logits(self._model, uncond_ids_t, uncond_mask)[0]

        n_cont = len(cont_ids)
        ctx_len = len(ctx_ids)
        # position ``ctx_len - 1 + i`` of the conditional stream predicts
        # ``cont_ids[i]``; the unconditional stream aligns at ``skip + i``.
        c_start = max(0, ctx_len - 1)
        c_slice = cond_logits[c_start : c_start + n_cont]
        u_slice = uncond_logits[skip : skip + n_cont]
        if u_slice.shape[0] < n_cont:  # safety for odd tokenizer behaviour
            u_slice = uncond_logits[:n_cont]

        cond_np = _to_numpy(c_slice).astype(np.float64)
        uncond_np = _to_numpy(u_slice).astype(np.float64)

        total, _ = sequence_logprobs(
            cond_np,
            uncond_np if gamma != 1.0 else None,
            cont_ids,
            gamma=gamma,
            combine_mode=self.config.combine_mode,
        )
        greedy = is_greedy_continuation(
            cond_np,
            uncond_np if gamma != 1.0 else None,
            cont_ids,
            gamma=gamma,
            eos_token_id=wrapper.eos_token_id,
        )
        return float(total), bool(greedy)

    def loglikelihood(
        self, requests: Iterable[Any], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """CFG-scored log-likelihoods for ``(context, continuation)`` requests."""
        results: List[Tuple[float, bool]] = []
        for request in requests:
            args = request_args(request)
            context, continuation = args[0], args[1]
            score = self._score_one(context, continuation)
            results.append(score)
            if self.cache_hook is not None:
                try:
                    self.cache_hook.add_partial("loglikelihood", (context, continuation), score)
                except Exception:  # pragma: no cover
                    pass
        return results

    def _rolling_logprob(self, text: str, gamma: Optional[float] = None) -> float:
        """CFG-scored rolling log-likelihood of ``text`` (LAMBADA-OpenAI)."""
        wrapper = self._ensure_wrapper()
        gamma = self.config.gamma if gamma is None else float(gamma)
        tokenizer = wrapper.tokenizer
        device = wrapper.device

        enc = tokenizer(text, add_special_tokens=True)
        tokens = list(enc["input_ids"])
        if isinstance(tokens) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        if not tokens:
            return 0.0

        max_len = self._max_length
        if len(tokens) > max_len:
            tokens = tokens[-max_len:]

        bos_id = getattr(wrapper, "bos_token_id", None) or tokens[0]
        total = 0.0
        for start in range(0, len(tokens), max_len):
            window = tokens[start : start + max_len]
            if not window:
                continue
            if self.config.rolling_mode == "shift_by_one":
                # unconditional context = the immediately preceding token
                uncond_window = [bos_id] + window[:-1]
                skip = 1
            else:  # empty_prefix
                uncond_window = [bos_id] + list(window)
                skip = 1
            cond_t = torch.tensor([window], dtype=torch.long, device=device)
            uncond_t = torch.tensor([uncond_window], dtype=torch.long, device=device)
            with torch.no_grad():
                cond_logits = _forward_logits(self._model, cond_t, torch.ones_like(cond_t))[0]
                uncond_logits = _forward_logits(
                    self._model, uncond_t, torch.ones_like(uncond_t)
                )[0]

            n = len(window)
            cond_np = _to_numpy(cond_logits[:n]).astype(np.float64)
            uncond_np = _to_numpy(uncond_logits[skip : skip + n]).astype(np.float64)
            if uncond_np.shape[0] < n:  # safety
                uncond_np = _to_numpy(uncond_logits[:n]).astype(np.float64)

            # position 0 of a window has no aligned unconditional context, so it
            # is scored conditionally (this only affects the first token).
            targets = list(window)
            guided_np = cond_np if float(gamma) == 1.0 else np.asarray(
                combine_guided_logits(cond_np, uncond_np, gamma), dtype=np.float64
            )
            logp = _log_softmax_np(guided_np)
            per_token = logp[np.arange(n), np.asarray(targets, dtype=np.int64)]
            if start == 0:
                per_token[0] = _log_softmax_np(cond_np[:1])[0, targets[0]]
            total += float(np.sum(per_token))
        return total

    def loglikelihood_rolling(
        self, requests: Iterable[Any], disable_tqdm: bool = False
    ) -> List[float]:
        """CFG-scored rolling log-likelihood per request (perplexity tasks)."""
        results: List[float] = []
        for request in requests:
            args = request_args(request)
            text = args[0]
            score = self._rolling_logprob(text)
            results.append(score)
            if self.cache_hook is not None:
                try:
                    self.cache_hook.add_partial("loglikelihood_rolling", (text,), score)
                except Exception:  # pragma: no cover
                    pass
        return results

    # ------------------------------------------------------------------ #
    # generation (TriviaQA, Appendix C.1)
    # ------------------------------------------------------------------ #
    def generate_until(
        self, requests: Iterable[Any], disable_tqdm: bool = False
    ) -> List[str]:
        """CFG-guided greedy generation; honours ``until`` stop sequences."""
        from ..cfg.generator import CFGGenerator, GenerationConfig

        wrapper = self._ensure_wrapper()
        generator = CFGGenerator(wrapper)
        outputs: List[str] = []
        for request in requests:
            args = request_args(request)
            context = args[0]
            gen_kwargs = dict(args[1]) if len(args) > 1 and args[1] else {}
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_new = int(gen_kwargs.get("max_gen_toks", self.config.max_gen_toks))
            do_sample = bool(gen_kwargs.get("do_sample", self.config.temperature > 0.0))
            gen_config = GenerationConfig(
                gamma=self.config.gamma,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=do_sample,
                max_new_tokens=max_new,
                stop_strings=tuple(until),
                stop_on_strings=True,
                seed=self.config.seed,
            )
            negative = self.negative_prompt
            out = generator.generate(
                context,
                config=gen_config,
                negative_prompts=[negative] if negative else None,
            )
            text = out.completions[0] if out.completions else ""
            for stop in until:
                if stop and stop in text:
                    text = text.split(stop)[0]
            outputs.append(text.strip())
        return outputs

    # ------------------------------------------------------------------ #
    # metrics (Appendix C.1 substring match)
    # ------------------------------------------------------------------ #
    def is_substring_task(self, task_name: Optional[str] = None) -> bool:
        if self.config.substring_match is not None:
            return bool(self.config.substring_match)
        if not task_name:
            return False
        return any(key in task_name for key in SUBSTRING_MATCH_TASKS)

    @staticmethod
    def match_answers(
        prediction: str,
        references: Union[str, Sequence[str]],
        substring: bool = False,
    ) -> bool:
        """Score an answer, defaulting to substring match for TriviaQA.

        ``substring=False`` performs the usual exact-match comparison after
        normalization; ``substring=True`` uses
        :func:`src.eval.triviaqa_match.substring_match` (Appendix C.1).
        """
        refs = [references] if isinstance(references, str) else list(references or [])
        if substring:
            try:
                from .triviaqa_match import substring_match  # type: ignore

                return bool(substring_match(prediction, refs))
            except Exception:  # pragma: no cover - fallback
                logger.debug("triviaqa_match unavailable; falling back to exact match")
        norm = _normalize_answer(prediction)
        return any(norm == _normalize_answer(r) for r in refs)


def _normalize_answer(text: Any) -> str:
    """Lowercase, strip quotes/whitespace/terminal punctuation."""
    s = str(text).strip().strip('"').strip("'").strip()
    while s and s[-1] in ".!?,;:":
        s = s[:-1]
    return " ".join(s.lower().split())


def _coerce(value: Any) -> Any:
    """Coerce a ``--model_args`` string value to bool/int/float when possible."""
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


# --------------------------------------------------------------------------- #
# Convenience driver used by scripts/run_zero_shot.py
# --------------------------------------------------------------------------- #
def evaluate_task(
    task_name: str,
    model_name_or_path: Optional[str] = None,
    gamma: float = 1.0,
    unconditional_mode: str = "last_prompt_token",
    config: Optional[HarnessCFGConfig] = None,
    batch_size: int = 1,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    **harness_kwargs: Any,
) -> Dict[str, Any]:
    """Run one zero-shot task through ``lm_eval`` with CFG scoring.

    Requires ``lm-evaluation-harness`` (``pip install lm-eval``); raises a
    helpful ``ImportError`` otherwise.  Returns the ``simple_evaluate`` result
    dict, augmented with the ``task``/``gamma`` bookkeeping used to assemble
    Table 5.
    """
    try:
        import lm_eval  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "lm-evaluation-harness is required for evaluate_task(); install it "
            "with `pip install lm-eval` (the paper uses the harness of Gao et "
            "al. 2021 and its sampling defaults)."
        ) from exc

    cfg = (config or HarnessCFGConfig()).replace(
        gamma=gamma, unconditional_mode=unconditional_mode, batch_size=batch_size
    )
    lm = CFGHarnessLM(
        model_name_or_path=model_name_or_path,
        config=cfg,
    )
    eval_kwargs: Dict[str, Any] = {"tasks": [task_name], "lm": lm, "batch_size": batch_size}
    if num_fewshot is not None:
        eval_kwargs["num_fewshot"] = num_fewshot
    if limit is not None:
        eval_kwargs["limit"] = limit
    eval_kwargs.update(harness_kwargs)
    results = lm_eval.simple_evaluate(**eval_kwargs)
    results.setdefault("config", {})
    results["cfg"] = {
        "task": task_name,
        "gamma": float(gamma),
        "unconditional_mode": cfg.unconditional_mode,
        "model": model_name_or_path,
    }
    return results


def result_accuracy(results: Dict[str, Any], task_name: str) -> float:
    """Pull the headline accuracy out of a ``simple_evaluate`` result dict."""
    task_results = (results.get("results") or {}).get(task_name, {})
    for key in ("acc_norm,none", "acc,none", "exact_match,none", "acc_norm", "acc"):
        if key in task_results:
            value = task_results[key]
            return float(value) if not isinstance(value, (list, tuple)) else float(
                np.mean(value)
            )
    raise KeyError(
        f"no accuracy-like metric found for task {task_name!r}; keys="
        f"{sorted(task_results.keys())}"
    )


def gamma_sweep_table(
    per_cell: Dict[Tuple[str, float], float],
    tasks: Sequence[str] = HARNESS_TASKS,
    gammas: Sequence[float] = HARNESS_GAMMAS,
) -> Dict[str, Dict[float, float]]:
    """Reshape ``{(task, gamma): accuracy}`` into a Table-5-like mapping."""
    return {
        task: {g: per_cell.get((task, g), float("nan")) for g in gammas} for task in tasks
    }


def format_table(table: Dict[str, Dict[float, float]], gammas: Sequence[float] = HARNESS_GAMMAS) -> str:
    """Plain-text rendering of the Table 5 layout."""
    header = "task".ljust(18) + "".join(f"g={g:<7}".rjust(10) for g in gammas)
    lines = [header, "-" * len(header)]
    for task, row in table.items():
        cells = []
        for g in gammas:
            value = row.get(g, float("nan"))
            cells.append("nan".rjust(10) if _isnan(value) else f"{value:>10.4f}")
        lines.append(task.ljust(18) + "".join(cells))
    return "\n".join(lines)


def _isnan(value: Any) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except Exception:
        return True


__all__ = [
    "CFGHarnessLM",
    "HarnessCFGConfig",
    "HARNESS_GAMMAS",
    "HARNESS_TASKS",
    "HARNESS_DEFAULT_TEMPERATURE",
    "HARNESS_DEFAULT_TOP_P",
    "SUBSTRING_MATCH_TASKS",
    "UNCONDITIONAL_MODES",
    "ROLLING_MODES",
    "combine_guided_logits",
    "sequence_logprobs",
    "is_greedy_continuation",
    "tokenize_pair",
    "build_unconditional_inputs",
    "evaluate_task",
    "result_accuracy",
    "gamma_sweep_table",
    "format_table",
]
