"""CFG-guided generation for autoregressive language models.

Two entry points are provided:

* :class:`CFGLogitsProcessor` -- a HuggingFace ``LogitsProcessor`` that can
  be dropped into ``model.generate``.  It recomputes the unconditional (or
  negatively prompted) branch itself and mixes the two logit vectors with
  Equation 7.  This is the most portable option because the surrounding
  generation loop (sampling, stopping criteria, KV cache of the *conditional*
  branch) remains HuggingFace's.

* :func:`cfg_generate` -- a small helper that wraps ``model.generate`` with
  the processor above and returns text.

Design notes
------------
Section 3.1 of the paper: *"we implement CFG by starting the unconditional
prompt at the last token of the initial prompt"*.  We therefore build the
unconditional context from the last ``uncond_prefix_tokens`` tokens of the
prompt (default ``1``), so that the unconditional branch still receives a
minimal amount of context.  Passing ``negative_prompt`` (Equation 5) replaces
that context with the tokenised negative prompt instead.

CFG doubles the inference cost: the conditional branch is evaluated by the
standard generation loop and the unconditional branch is evaluated by this
processor on every step.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from .cfg import guidance_weight


def _as_ids(tokenizer, text_or_ids: Union[str, Sequence[int]], device) -> torch.Tensor:
    if isinstance(text_or_ids, torch.Tensor):
        return text_or_ids.to(device)
    if isinstance(text_or_ids, str):
        return tokenizer(text_or_ids, return_tensors="pt").input_ids.to(device)
    return torch.tensor([list(text_or_ids)], dtype=torch.long, device=device)


class CFGLogitsProcessor(LogitsProcessor):
    """Mix conditional logits with an unconditional (or negative) branch.

    Args:
        model: the causal LM used for generation.  It must expose
            ``__call__(input_ids=..., attention_mask=...)`` returning an
            object with a ``logits`` attribute.
        uncond_input_ids: ``[1, k]`` token ids that seed the unconditional
            branch (typically the last token of the prompt, or a negative
            prompt).
        gamma: guidance strength.
        prompt_len: number of tokens in the conditional prompt.  The
            generated tokens are taken from ``input_ids[:, prompt_len:]``.
        uncond_cache: reuse a KV cache for the unconditional branch.  When
            ``False`` (the default, and the most portable option) the whole
            unconditional sequence is re-evaluated at every step.
        clamp_gamma_min_one: the paper never uses ``gamma < 1`` for
            generation; keep the guard for clarity.
    """

    def __init__(
        self,
        model,
        uncond_input_ids: torch.Tensor,
        gamma: float,
        prompt_len: int,
        uncond_cache: bool = False,
        dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.gamma = guidance_weight(gamma)
        self.uncond_input_ids = uncond_input_ids
        self.prompt_len = int(prompt_len)
        self.uncond_cache = uncond_cache
        self.dtype = dtype
        self.verbose = verbose
        self._past = None
        self._uncond_len = 0

    # -- helpers ---------------------------------------------------------
    @torch.no_grad()
    def _uncond_logits(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Forward the unconditional branch, returning the last-step logits."""
        # ``model.generate`` may expand the batch for ``num_return_sequences``;
        # replicate the (shared) unconditional context to match.
        uncond_ids = self.uncond_input_ids
        if uncond_ids.shape[0] != generated_ids.shape[0]:
            repeat = generated_ids.shape[0] // uncond_ids.shape[0]
            uncond_ids = uncond_ids.repeat_interleave(repeat, dim=0)
        full = torch.cat([uncond_ids, generated_ids], dim=-1)
        if self.uncond_cache and self._past is not None:
            if generated_ids.shape[-1] > 0:
                step_ids = torch.cat(
                    [uncond_ids[:, :0], generated_ids[:, -1:]], dim=-1
                )
            else:
                step_ids = uncond_ids[:, -1:]
            out = self.model(input_ids=step_ids, past_key_values=self._past, use_cache=True)
            self._past = getattr(out, "past_key_values", None)
        else:
            out = self.model(input_ids=full, use_cache=self.uncond_cache)
            if self.uncond_cache:
                self._past = getattr(out, "past_key_values", None)
        logits = out.logits[:, -1, :]
        if self.dtype is not None:
            logits = logits.to(self.dtype)
        return logits

    # -- LogitsProcessor API --------------------------------------------
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_len :]
        uncond = self._uncond_logits(generated)
        # Both branches are defined over the same vocabulary.
        cond = scores
        if uncond.shape[-1] != cond.shape[-1]:  # pragma: no cover - defensive
            raise RuntimeError(
                "conditional and unconditional vocabularies differ: "
                f"{cond.shape[-1]} vs {uncond.shape[-1]}"
            )
        mixed = uncond + self.gamma * (cond.float() - uncond)
        return mixed.to(scores.dtype)


@torch.no_grad()
def cfg_generate(
    model,
    tokenizer,
    prompt: str,
    gamma: float,
    negative_prompt: Optional[str] = None,
    uncond_prefix_tokens: int = 1,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    num_return_sequences: int = 1,
    seed: Optional[int] = None,
    stop_sequences: Optional[Sequence[str]] = None,
    uncond_cache: bool = False,
    **generate_kwargs,
) -> List[str]:
    """Generate text with classifier-free guidance.

    ``gamma = 1`` reproduces vanilla sampling (the processor becomes the
    identity), which makes it easy to build matched baseline/CFG pairs.

    Returns a list of ``num_return_sequences`` completions (the prompt is
    stripped from the returned text).
    """
    device = next(model.parameters()).device
    prompt_ids = _as_ids(tokenizer, prompt, device)

    if negative_prompt is not None:
        uncond_ids = _as_ids(tokenizer, negative_prompt, device)
    else:
        k = int(uncond_prefix_tokens)
        # k = 0 means "no context at all" (a strictly unconditional branch);
        # k >= 1 keeps the last k prompt tokens, as in Section 3.1.
        uncond_ids = prompt_ids[:, -k:] if k > 0 else prompt_ids[:, :0]

    processor = CFGLogitsProcessor(
        model=model,
        uncond_input_ids=uncond_ids,
        gamma=gamma,
        prompt_len=prompt_ids.shape[-1],
        # the cached path keeps a batch of 1, so it cannot be combined with
        # num_return_sequences > 1
        uncond_cache=bool(uncond_cache) and num_return_sequences == 1,
    )

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        logits_processor=[processor],
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None and top_p < 1.0:
            gen_kwargs["top_p"] = top_p
        if top_k:
            gen_kwargs["top_k"] = top_k
    if seed is not None:
        torch.manual_seed(seed)
    gen_kwargs.update(generate_kwargs)

    output = model.generate(input_ids=prompt_ids, **gen_kwargs)
    decoded = tokenizer.batch_decode(output[:, prompt_ids.shape[-1] :], skip_special_tokens=True)
    if stop_sequences:
        trimmed = []
        for text in decoded:
            cut = len(text)
            for stop in stop_sequences:
                idx = text.find(stop)
                if idx != -1:
                    cut = min(cut, idx)
            trimmed.append(text[:cut])
        decoded = trimmed
    return decoded


@torch.no_grad()
def cfg_generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    gamma: float,
    negative_prompt: Optional[str] = None,
    uncond_prefix_tokens: int = 1,
    **kwargs,
) -> List[List[str]]:
    """Run :func:`cfg_generate` over several prompts sequentially.

    This is a thin convenience wrapper; prompts are processed one at a time
    because the unconditional branch of each generation has its own context.
    """
    return [
        cfg_generate(
            model,
            tokenizer,
            prompt,
            gamma=gamma,
            negative_prompt=negative_prompt,
            uncond_prefix_tokens=uncond_prefix_tokens,
            **kwargs,
        )
        for prompt in prompts
    ]


@torch.no_grad()
def cfg_next_token_distribution(
    model,
    tokenizer,
    prompt: str,
    gamma: float,
    negative_prompt: Optional[str] = None,
    uncond_prefix_tokens: int = 1,
) -> torch.Tensor:
    """Return ``log P_hat(w_1 | c)`` for the first generated token.

    Useful for the vocabulary-ranking visualisation of Section 5.3.
    """
    device = next(model.parameters()).device
    prompt_ids = _as_ids(tokenizer, prompt, device)
    if negative_prompt is not None:
        uncond_ids = _as_ids(tokenizer, negative_prompt, device)
    else:
        uncond_ids = prompt_ids[:, -max(1, int(uncond_prefix_tokens)) :]
    cond_logits = model(input_ids=prompt_ids).logits[:, -1, :]
    uncond_logits = model(input_ids=uncond_ids).logits[:, -1, :]
    mixed = uncond_logits + guidance_weight(gamma) * (cond_logits.float() - uncond_logits)
    return F.log_softmax(mixed.float(), dim=-1)
