"""Dual-context HuggingFace model wrapper for Classifier-Free Guidance.

This module turns any ``AutoModelForCausalLM`` into a CFG-capable model.  At every
decoding step two forward passes are run **through the same weights**:

* the *conditional* pass on the prompt ``c`` (Eq. 7: ``log P_theta(w_i | w_<i, c)``);
* the *unconditional* pass, where the prefix ``c`` is either dropped entirely
  (``unconditional_mode="empty_prefix"``, the natural ``P_theta(w_i | w_<i)`` of
  Eq. 7) or replaced by a shorter prompt starting at the **last token of the
  initial prompt** (``unconditional_mode="last_prompt_token"``, the convention
  used for the EleutherAI LM Evaluation Harness zero-shot suite, Sec. 3.1).

A negative prompt ``c_bar`` (Eq. 5, Sec. 3.4) can be supplied instead; in that
case the *unconditional* pass is run on ``c_bar``.

The wrapper exposes:

* :meth:`CFGModelWrapper.dual_logits` - full-sequence, cache-free dual forward;
* :meth:`CFGModelWrapper.dual_logits_cached` - incremental dual forward with two
  independent kv-caches (cost doubles: ``2C``);
* :meth:`CFGModelWrapper.next_token_logits` - last-position logits for both passes,
  ready to be combined by :func:`src.cfg.logits.cfg_combine`;
* :meth:`CFGModelWrapper.token_logprobs` - log-softmax scores for both passes, used
  by the evaluation harness shim;
* a logits-hook registry so that analysis code (entropy/overlap) can consume the
  **raw pre-softmax logits** of both passes.

Everything works in the raw (pre-softmax) logit space, matching Sec. 2.2: the CFG
combination happens *before* temperature, top-p and softmax.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

try:  # transformers is optional so that the pure-math parts stay importable
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from .logits import cfg_combine, log_softmax

logger = logging.getLogger(__name__)

#: Supported unconditional-context conventions.
UNCONDITIONAL_MODES = ("empty_prefix", "last_prompt_token")


# --------------------------------------------------------------------------- #
# Small containers
# --------------------------------------------------------------------------- #
@dataclass
class DualLogits:
    """A pair of raw (pre-softmax) logit tensors from the same weights."""

    cond: torch.Tensor
    uncond: torch.Tensor

    def guided(self, gamma: float = 1.0) -> torch.Tensor:
        """Eq. 7: ``uncond + gamma * (cond - uncond)``."""
        return cfg_combine(self.uncond, self.cond, gamma)

    def logprobs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-softmax of both passes (for entropy / overlap / PPL analysis)."""
        return log_softmax(self.cond), log_softmax(self.uncond)


def resolve_dtype(dtype: Any, device: torch.device) -> torch.dtype:
    """Autodetect a sensible dtype (bf16 > fp16 > fp32) when ``dtype`` is "auto"."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or str(dtype).lower() in {"auto", "default"}:
        if device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:  # pragma: no cover - old torch
                pass
            return torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    return getattr(torch, str(dtype))


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #
class CFGModelWrapper:
    """Wraps a HuggingFace causal LM and exposes one unified dual-context API.

    Parameters
    ----------
    model_name_or_path:
        HF hub id or local path (e.g. ``"gpt2-large"``, ``"EleutherAI/pythia-1.4b"``).
    unconditional_mode:
        ``"empty_prefix"`` (drop the whole prefix; default) or
        ``"last_prompt_token"`` (unconditional prompt starts at the last token of
        the initial prompt -- the Sec. 3.1 harness convention).
    device / dtype:
        ``"auto"`` by default; fp16/bf16 autodetected on CUDA.
    prompt_length:
        Number of leading tokens of a sequence that belong to the prompt ``c``.
        It can also be provided per call to :meth:`dual_logits`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = "auto",
        dtype: Any = "auto",
        unconditional_mode: str = "empty_prefix",
        prompt_length: Optional[int] = None,
        tokenizer_name_or_path: Optional[str] = None,
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = True,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        **model_kwargs: Any,
    ) -> None:
        if unconditional_mode not in UNCONDITIONAL_MODES:
            raise ValueError(
                f"unconditional_mode must be one of {UNCONDITIONAL_MODES}, "
                f"got {unconditional_mode!r}"
            )
        self.model_name_or_path = model_name_or_path
        self.unconditional_mode = unconditional_mode

        # ---- device ------------------------------------------------------ #
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dtype = resolve_dtype(dtype, self.device)

        # ---- model / tokenizer ------------------------------------------- #
        if tokenizer is None:
            if AutoTokenizer is None:  # pragma: no cover
                raise ImportError("transformers is required to load a tokenizer")
            tok_kwargs: dict = {"trust_remote_code": trust_remote_code}
            tok_src = tokenizer_name_or_path or model_name_or_path
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_src, **tok_kwargs)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False, **tok_kwargs)
            # Decoder-only models often lack a pad token.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        self.tokenizer = tokenizer

        if model is None:
            if AutoModelForCausalLM is None:  # pragma: no cover
                raise ImportError("transformers is required to load a model")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        self.model = model.to(self.device)
        self.model.eval()

        # The wrapper is used for inference only.
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.config = getattr(self.model, "config", None)
        self.prompt_length = prompt_length

        # ---- caches (2C: one per context) -------------------------------- #
        self._past_cond: Any = None
        self._past_uncond: Any = None
        self._len_cond: int = 0
        self._len_uncond: int = 0

        self._hooks: List[Callable[[torch.Tensor, torch.Tensor], None]] = []

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def eos_token_id(self) -> Optional[int]:
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def bos_token_id(self) -> Optional[int]:
        tid = getattr(self.tokenizer, "bos_token_id", None)
        if tid is None:
            tid = getattr(self.config, "bos_token_id", None)
        return tid

    @property
    def pad_token_id(self) -> Optional[int]:
        tid = getattr(self.tokenizer, "pad_token_id", None)
        if tid is None:
            tid = getattr(self.config, "pad_token_id", None)
        if tid is None:
            tid = self.eos_token_id
        return tid

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_output_embeddings().weight.shape[0])

    # ------------------------------------------------------------------ #
    # Logits hooks (raw pre-softmax logits for both passes)
    # ------------------------------------------------------------------ #
    def add_logits_hook(self, hook: Callable[[torch.Tensor, torch.Tensor], None]) -> None:
        """Register ``hook(logits_cond, logits_uncond)`` (called on every dual pass)."""
        self._hooks.append(hook)

    def remove_logits_hook(self, hook: Callable[[torch.Tensor, torch.Tensor], None]) -> None:
        if hook in self._hooks:
            self._hooks.remove(hook)

    def _call_hooks(self, logits_cond: torch.Tensor, logits_uncond: torch.Tensor) -> None:
        for hook in self._hooks:
            hook(logits_cond, logits_uncond)

    # ------------------------------------------------------------------ #
    # Prompt bookkeeping
    # ------------------------------------------------------------------ #
    def set_prompt_length(self, prompt_length: int) -> None:
        """Declare how many leading tokens of the sequence belong to ``c``."""
        self.prompt_length = int(prompt_length)

    def _resolve_prompt_length(self, prompt_length: Optional[int], seq_len: int) -> int:
        if prompt_length is None:
            prompt_length = self.prompt_length
        if prompt_length is None:
            prompt_length = seq_len
        return max(0, min(int(prompt_length), seq_len))

    def _uncond_slice_start(self, prompt_length: int) -> int:
        """First index of the sequence that the unconditional pass keeps."""
        if self.unconditional_mode == "last_prompt_token":
            # Unconditional prompt begins at the last token of the initial prompt.
            return max(prompt_length - 1, 0)
        # empty_prefix -> the prefix is dropped entirely.
        return max(prompt_length, 0)

    def _uncond_seed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """A single seed token when the unconditional context would be empty."""
        if self.bos_token_id is not None:
            seed_id = int(self.bos_token_id)
        else:
            seed_id = int(input_ids[0, -1].item())
        return torch.full((input_ids.shape[0], 1), seed_id, dtype=input_ids.dtype, device=input_ids.device)

    def build_unconditional_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Derive the unconditional context from a conditional sequence.

        Returns ``(uncond_input_ids, uncond_attention_mask, dropped_prefix_tokens)``
        where ``dropped_prefix_tokens`` is how many leading tokens were removed (the
        unconditional cache therefore lags the conditional one by this amount).
        """
        seq_len = int(input_ids.shape[1])
        plen = self._resolve_prompt_length(prompt_length, seq_len)
        start = self._uncond_slice_start(plen)

        u_ids = input_ids[:, start:]
        if u_ids.shape[1] == 0:
            u_ids = self._uncond_seed_ids(input_ids)

        if attention_mask is not None:
            u_mask = attention_mask[:, start:]
            if u_mask.shape[1] != u_ids.shape[1]:
                u_mask = torch.ones_like(u_ids)
        else:
            u_mask = torch.ones_like(u_ids)
        return u_ids, u_mask, start

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """A single forward pass on the wrapped model; returns ``(logits, new_cache)``."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "return_dict": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if cache_position is not None:
            kwargs["cache_position"] = cache_position.to(self.device)
        if self.pad_token_id is not None:
            kwargs["pad_token_id"] = self.pad_token_id

        try:
            out = self.model(**kwargs)
        except TypeError:
            # Older / non-standard signatures: drop the optional arguments.
            kwargs.pop("cache_position", None)
            kwargs.pop("position_ids", None)
            out = self.model(**kwargs)

        logits = out.logits if hasattr(out, "logits") else out[0]
        new_cache = getattr(out, "past_key_values", None)
        return logits, new_cache

    def dual_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
        only_last: bool = False,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
    ) -> DualLogits:
        """Cache-free dual forward pass over the full sequence.

        Parameters
        ----------
        input_ids: ``[batch, seq]`` conditional sequence (prompt ``c`` + generated).
        prompt_length: number of leading prompt tokens.
        only_last: keep only the last position (next-token logits).
        negative_input_ids: optional negative prompt ``c_bar`` (Eq. 5); when given it
            replaces the prefix-dropped context as the *unconditional* anchor.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)

        logits_cond, _ = self._forward(input_ids, attention_mask, use_cache=False)

        if negative_input_ids is not None:
            neg_ids = negative_input_ids.to(self.device)
            neg_mask = (
                negative_attention_mask.to(self.device)
                if negative_attention_mask is not None
                else torch.ones_like(neg_ids)
            )
            logits_uncond, _ = self._forward(neg_ids, neg_mask, use_cache=False)
        else:
            u_ids, u_mask, _ = self.build_unconditional_inputs(input_ids, attention_mask, prompt_length)
            logits_uncond, _ = self._forward(u_ids, u_mask, use_cache=False)

        if only_last:
            logits_cond = logits_cond[:, -1, :]
            logits_uncond = logits_uncond[:, -1, :]

        self._call_hooks(logits_cond, logits_uncond)
        return DualLogits(cond=logits_cond, uncond=logits_uncond)

    # ------------------------------------------------------------------ #
    # Cached (incremental) decoding -- 2C kv-cache
    # ------------------------------------------------------------------ #
    def reset_cache(self) -> None:
        """Drop both kv-caches (call once per generation)."""
        self._past_cond = None
        self._past_uncond = None
        self._len_cond = 0
        self._len_uncond = 0

    def prime_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
    ) -> DualLogits:
        """Run the two prompt passes and store both kv-caches.

        Returns the next-token logits (``[batch, vocab]``) of both contexts.
        """
        self.reset_cache()
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(self.device)
        plen = self._resolve_prompt_length(prompt_length, int(input_ids.shape[1]))

        self._past_cond, self._len_cond, logits_cond = self._cached_prefix(
            input_ids, attention_mask, self._past_cond, 0
        )
        u_ids, u_mask, _ = self.build_unconditional_inputs(input_ids, attention_mask, plen)
        self._past_uncond, self._len_uncond, logits_uncond = self._cached_prefix(
            u_ids, u_mask, self._past_uncond, 0
        )

        lc, lu = logits_cond[:, -1, :], logits_uncond[:, -1, :]
        self._call_hooks(lc, lu)
        return DualLogits(cond=lc, uncond=lu)

    def _cached_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
        past_len: int,
    ) -> Tuple[Any, int, torch.Tensor]:
        position_ids = torch.arange(past_len, past_len + input_ids.shape[1], device=self.device).unsqueeze(0)
        cache_position = (
            torch.arange(past_len, past_len + input_ids.shape[1], device=self.device)
            if past_key_values is not None
            else None
        )
        logits, new_cache = self._forward(
            input_ids,
            attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        return new_cache, past_len + int(input_ids.shape[1]), logits

    def dual_logits_cached(
        self,
        new_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DualLogits:
        """Incremental dual pass: feed the freshly generated token(s) to both caches.

        Returns next-token logits ``[batch, vocab]`` from the conditional and the
        unconditional context.  Both contexts are advanced with the *same* new
        tokens; the unconditional cache simply starts ``offset`` tokens later,
        which is exactly the behaviour implied by Eq. 6/7 (the unconditioned
        distribution keeps the generated tokens ``w_{j<i}``).
        """
        if self._past_cond is None or self._past_uncond is None:
            raise RuntimeError("prime_cache() must be called before dual_logits_cached()")

        new_input_ids = new_input_ids.to(self.device)
        n_new = int(new_input_ids.shape[1])

        if attention_mask is not None:
            new_mask = attention_mask.to(self.device)
        else:
            new_mask = torch.ones_like(new_input_ids)

        # --- conditional ---------------------------------------------------- #
        mask_cond = torch.cat(
            [
                torch.ones((new_input_ids.shape[0], self._len_cond), dtype=new_mask.dtype, device=self.device),
                new_mask,
            ],
            dim=1,
        )
        self._past_cond, self._len_cond, logits_cond = self._cached_prefix(
            new_input_ids, mask_cond, self._past_cond, self._len_cond
        )

        # --- unconditional -------------------------------------------------- #
        mask_uncond = torch.cat(
            [
                torch.ones((new_input_ids.shape[0], self._len_uncond), dtype=new_mask.dtype, device=self.device),
                new_mask,
            ],
            dim=1,
        )
        self._past_uncond, self._len_uncond, logits_uncond = self._cached_prefix(
            new_input_ids, mask_uncond, self._past_uncond, self._len_uncond
        )

        lc, lu = logits_cond[:, -1, :], logits_uncond[:, -1, :]
        self._call_hooks(lc, lu)
        return DualLogits(cond=lc, uncond=lu)

    # ------------------------------------------------------------------ #
    # High-level helpers
    # ------------------------------------------------------------------ #
    def next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
        gamma: float = 1.0,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw CFG logits for the next token (Eq. 7, before temperature/top-p)."""
        dual = self.dual_logits(
            input_ids,
            attention_mask=attention_mask,
            prompt_length=prompt_length,
            only_last=True,
            negative_input_ids=negative_input_ids,
            negative_attention_mask=negative_attention_mask,
        )
        return dual.guided(gamma)

    def token_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-softmaxed ``[batch, seq, vocab]`` scores for both passes.

        Used by the evaluation-harness shim to score continuations under both the
        conditional and the unconditional context.
        """
        dual = self.dual_logits(
            input_ids,
            attention_mask=attention_mask,
            prompt_length=prompt_length,
            only_last=False,
            negative_input_ids=negative_input_ids,
            negative_attention_mask=negative_attention_mask,
        )
        return dual.logprobs()

    def encode(self, text: str, *, add_special_tokens: bool = True) -> torch.Tensor:
        """Tokenize ``text`` into a ``[1, seq]`` tensor on the model device."""
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"]
        return ids.to(self.device)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(model={self.model_name_or_path!r}, "
            f"mode={self.unconditional_mode!r}, dtype={self.dtype}, device={self.device})"
        )
