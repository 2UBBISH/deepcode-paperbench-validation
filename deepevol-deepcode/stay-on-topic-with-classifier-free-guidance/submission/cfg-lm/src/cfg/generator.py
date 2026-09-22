"""Autoregressive Classifier-Free Guidance generation loop.

Paper: "Stay on Topic with Classifier-Free Guidance"

At every decoding step ``i`` two forward passes are run through the *same*
language-model weights:

    logits_cond   = LM(w_{j<i}, c)        # conditional on the prompt c
    logits_uncond = LM(w_{j<i})           # prefix dropped (or a negative prompt c_bar)

and the next-token logits are combined in logit space (Eq. 7):

    log P_hat(w_i | w_{j<i}, c) = log P(w_i | w_{j<i})
                                 + gamma * ( log P(w_i | w_{j<i}, c) - log P(w_i | w_{j<i}) )

which is exactly ``uncond + gamma * (cond - uncond)`` on raw pre-softmax logits
(``src/cfg/logits.py:cfg_combine``).  Sampling (temperature -> top-p -> softmax
-> multinomial) is delegated to ``src/cfg/sampler.py``.

Standard budgets used by the paper (Section 3.2 / 3.3.1):

* HumanEval: ``max_new_tokens = 512`` (plus stopping on the end of the function),
* GSM8K / AQuA: stop when the generation contains the answer marker
  (``#### <number>`` for GSM8K, ``The answer is <X>`` for AQuA).

The loop is fully deterministic given a ``torch.Generator`` seed, which is what
makes the ``pass@k`` estimates of Tables 2/7/8/9 reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .model_wrapper import CFGModelWrapper
from .sampler import (
    ANALYSIS_GAMMA,
    CFG_GAMMAS,
    CFGSampler,
    SamplingConfig,
    get_generator,
)
from .logits import cfg_combine  # noqa: F401  (re-exported for convenience)

logger = logging.getLogger(__name__)

__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "CFGGenerator",
    "generate",
    "HUMANEVAL_MAX_NEW_TOKENS",
    "COT_MAX_NEW_TOKENS",
    "ANSWER_MARKERS",
    "truncate_at_stop_strings",
]

# --------------------------------------------------------------------------- #
# Budgets / stopping conventions (paper Sections 3.2, 3.3.1)
# --------------------------------------------------------------------------- #

#: HumanEval problems are decoded with at most this many new tokens.
HUMANEVAL_MAX_NEW_TOKENS = 512

#: Generative budget for the CoT tasks (GSM8K / AQuA).  The paper stops at the
#: answer marker; this is only an upper bound so a divergent chain cannot run
#: forever (it is exactly how the "invalid chain" fraction of Figure 2 is read).
COT_MAX_NEW_TOKENS = 512

#: Markers that terminate a chain-of-thought generation once ohe is produced.
ANSWER_MARKERS: Tuple[str, ...] = (
    "####",            # GSM8K (Self-Consistency 8-shot prompt)
    "The answer is",   # AQuA / GSM8K free-form
    "Answer:",
)


def truncate_at_stop_strings(
    text: str, stop_strings: Optional[Sequence[str]] = None
) -> Tuple[str, bool]:
    """Cut ``text`` at the earliest occurrence of any stop string.

    Returns ``(truncated_text, stopped)``.  The stop string itself is *kept* for
    GSM8K's ``####`` (it is part of the answer marker) but dropped for free-form
    markers?  -- we keep the marker uniformly, the answer parser strips it.
    """
    if not stop_strings:
        return text, False
    best: Optional[int] = None
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and (best is None or idx < best):
            best = idx
    if best is None:
        return text, False
    return text[: best + 0], False or True


@dataclass
class GenerationConfig:
    """Decoding configuration for :class:`CFGGenerator`.

    Defaults follow the EleutherAI LM Evaluation Harness (greedy, top_p=1.0) and
    are overridden by the task scripts (CoT, HumanEval) as stated in the paper.
    """

    gamma: float = 1.0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True

    max_new_tokens: int = HUMANEVAL_MAX_NEW_TOKENS
    min_new_tokens: int = 0
    stop_strings: Tuple[str, ...] = ()
    #: Decode until EOS; set to ``False`` to always fill ``max_new_tokens``
    #: (needed for HumanEval where the model typically writes "if __name__ ...").
    stop_at_eos: bool = True
    #: Stop the *whole* generation as soon as ``stop_strings`` appear.
    stop_on_strings: bool = False
    #: Prompts (e.g. CoT few-shot) are excluded from the returned completion when
    #: ``return_prompt=False``.
    seed: Optional[int] = None

    def to_sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            gamma=self.gamma,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            seed=self.seed,
        )

    def replace(self, **kwargs) -> "GenerationConfig":
        d = dict(self.__dict__)
        d.update(kwargs)
        return GenerationConfig(**d)

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass
class GenerationOutput:
    """Result of a (possibly batched) CFG generation."""

    sequences: torch.Tensor                     # [batch, prompt_len + new]
    completions: List[str]                      # decoded new tokens only
    texts: List[str]                            # decoded prompt + completion
    prompt_lengths: List[int] = field(default_factory=list)
    n_new_tokens: List[int] = field(default_factory=list)
    stopped_on: List[Optional[str]] = field(default_factory=list)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.completions)


class CFGGenerator:
    """Autoregressive decoding loop with Classifier-Free Guidance.

    One instance wraps a :class:`CFGModelWrapper` and performs the two-forward-
    pass / combine / sample loop from Equation 7.
    """

    def __init__(
        self,
        model_wrapper: CFGModelWrapper,
        config: Optional[GenerationConfig] = None,
        sampler: Optional[CFGSampler] = None,
    ) -> None:
        self.model = model_wrapper
        self.config = config or GenerationConfig()
        self.sampler = sampler or CFGSampler()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompts: Sequence[str],
        config: Optional[GenerationConfig] = None,
        prompt_length: Optional[int] = None,
        negative_prompts: Optional[Sequence[str]] = None,
        return_prompt: bool = True,
    ) -> GenerationOutput:
        """Generate completions for a batch of prompts with CFG.

        ``prompt_length`` follows the Section 3.1 convention of the zero-shot
        harness (unconditional context starts at the last prompt token).  When it
        is ``None`` the full encoded prompt length is used.  It may be a scalar
        (shared by the batch) or a per-example sequence.
        """
        cfg = config or self.config
        if isinstance(prompts, str):
            prompts = [prompts]

        enc = self.model.encode(list(prompts), return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        batch_size, plen = input_ids.shape

        neg_ids = neg_mask = None
        if negative_prompts is not None:
            nenc = self.model.encode(list(negative_prompts), return_tensors="pt")
            neg_ids = nenc["input_ids"].to(self.model.device)
            neg_mask = nenc.get("attention_mask")
            if neg_mask is not None:
                neg_mask = neg_mask.to(self.model.device)

        if prompt_length is None:
            pl_tensor = torch.full(
                (batch_size,), plen, dtype=torch.long, device=self.model.device
            )
        elif isinstance(prompt_length, int):
            pl_tensor = torch.full(
                (batch_size,), int(prompt_length), dtype=torch.long, device=self.model.device
            )
        else:
            pl_tensor = torch.as_tensor(
                list(prompt_length), dtype=torch.long, device=self.model.device
            )
            if pl_tensor.numel() == 1:
                pl_tensor = pl_tensor.expand(batch_size)

        gen = get_generator(cfg.seed, device=self.model.device)
        sampler = self.sampler.with_config(cfg.to_sampling_config())

        out_ids = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.model.device)
        stop_found: List[Optional[str]] = [None] * batch_size
        n_new = [0] * batch_size
        stop_strings = tuple(cfg.stop_strings or ())

        # Prime both kv-caches once on the prompt(s); this is the "2C" cost.
        self.model.reset_cache()
        try:
            self.model.prime_cache(
                input_ids, attention_mask=attention_mask, prompt_length=pl_tensor
            )
            use_cache = True
        except Exception as exc:  # pragma: no cover - depends on model class
            logger.debug("kv-cache priming unavailable (%s); running full forwards", exc)
            use_cache = False

        with torch.no_grad():
            for _step in range(int(cfg.max_new_tokens)):
                if use_cache and _step > 0:
                    new_tok = out_ids[:, -1:]
                    dual = self.model.dual_logits_cached(
                        new_tok,
                        attention_mask=(
                            torch.cat([attention_mask, torch.zeros_like(new_tok)], dim=1)
                            if attention_mask is not None
                            else None
                        ),
                    )
                else:
                    dual = self.model.dual_logits(
                        out_ids,
                        attention_mask=attention_mask,
                        prompt_length=pl_tensor,
                        only_last=True,
                        negative_input_ids=neg_ids,
                        negative_attention_mask=neg_mask,
                    )
                logits_cond = dual.cond
                logits_uncond = dual.uncond
                tokens, _probs = sampler.sample(logits_cond, logits_uncond, generator=gen)
                tokens = tokens.view(batch_size, 1)

                # Freeze already-finished sequences on their pad token.
                pad = self.model.pad_token_id or self.model.eos_token_id or 0
                if finished.any():
                    tokens = torch.where(
                        finished.view(batch_size, 1),
                        torch.full_like(tokens, pad),
                        tokens,
                    )
                out_ids = torch.cat([out_ids, tokens], dim=1)
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, (~finished).long().view(batch_size, 1)], dim=1
                    )
                for b in range(batch_size):
                    if not finished[b]:
                        n_new[b] += 1

                # new tokens are never "finished" by length here (handled below)
                if cfg.stop_at_eos and self.model.eos_token_id is not None:
                    finished |= tokens.view(-1) == self.model.eos_token_id

                if stop_strings:
                    for b in range(batch_size):
                        if finished[b] or stop_found[b] is not None:
                            continue
                        new_text = self.model.decode(out_ids[b, plen:])
                        hit = next((s for s in stop_strings if s in new_text), None)
                        if hit is not None:
                            stop_found[b] = hit
                            if cfg.stop_on_strings:
                                finished[b] = True

                if bool(finished.all()):
                    break

        completions: List[str] = []
        texts: List[str] = []
        for b in range(batch_size):
            gen_ids = out_ids[b, plen:]
            text = self.model.decode(gen_ids)
            eos = self.model.eos_token_id
            if eos is not None:
                # drop everything from the first EOS onwards
                row = gen_ids.tolist()
                if eos in row:
                    row = row[: row.index(eos)]
                    text = self.model.decode(row)
            if stop_strings:
                for s in stop_strings:
                    idx = text.find(s)
                    if idx != -1:
                        text = text[: idx + 0]
            completions.append(text)
            prompt_text = self.model.decode(out_ids[b, :plen])
            texts.append(prompt_text + text if return_prompt else text)

        return GenerationOutput(
            sequences=out_ids,
            completions=completions,
            texts=texts,
            prompt_lengths=[int(plen)] * batch_size,
            n_new_tokens=n_new,
            stopped_on=stop_found,
        )

    # convenience wrappers ------------------------------------------------- #

    def smart_call(self, *args, **kwargs) -> GenerationOutput:  # pragma: no cover
        return self.generate(*args, **kwargs)

    def generate_one(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        negative_prompt: Optional[str] = None,
        prompt_length: Optional[int] = None,
    ) -> str:
        neg = [negative_prompt] if negative_prompt is not None else None
        out = self.generate(
            [prompt], config=config, prompt_length=prompt_length, negative_prompts=neg
        )
        return out.completions[0]


# --------------------------------------------------------------------------- #
# Functional API
# --------------------------------------------------------------------------- #


def generate(
    model_wrapper: CFGModelWrapper,
    prompts: Sequence[str],
    gamma: float = 1.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_new_tokens: int = HUMANEVAL_MAX_NEW_TOKENS,
    stop_strings: Sequence[str] = (),
    stop_on_strings: bool = True,
    seed: Optional[int] = None,
    negative_prompts: Optional[Sequence[str]] = None,
    prompt_length: Optional[int] = None,
) -> GenerationOutput:
    """One-shot CFG generation (Eq. 7) for a batch of prompts."""
    cfg = GenerationConfig(
        gamma=gamma,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        stop_strings=tuple(stop_strings or ()),
        stop_on_strings=stop_on_strings,
        seed=seed,
    )
    return CFGGenerator(model_wrapper, cfg).generate(
        prompts,
        prompt_length=prompt_length,
        negative_prompts=negative_prompts,
    )
