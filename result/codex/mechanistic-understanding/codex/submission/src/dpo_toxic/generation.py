"""Generation helpers used by the intervention, DPO-data and F1 experiments."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .utils import batches


@torch.no_grad()
def generate_continuations(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
                           max_new_tokens: int = 20, batch_size: int = 16,
                           device: Optional[str] = None, do_sample: bool = False,
                           top_k: int = 0, temperature: float = 1.0,
                           return_prompt_lengths: bool = False,
                           context=None) -> List[str]:
    """Greedy (or top-k) generation of continuations for a list of prompts.

    ``context`` optionally supplies an object whose ``__enter__`` installs
    intervention hooks (e.g. :class:`dpo_toxic.architecture.ResidualShiftHook`).
    """
    device = device or str(next(model.parameters()).device)
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: List[str] = []
    lengths: List[int] = []
    was_training = model.training
    model.eval()
    try:
        for batch in batches(list(prompts), batch_size):
            enc = tokenizer(list(batch), return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            if do_sample:
                kwargs.update(top_k=top_k or 50, temperature=temperature)
            if context is not None:
                with context:
                    out = model.generate(**enc, **kwargs)
            else:
                out = model.generate(**enc, **kwargs)
            prompt_len = enc["input_ids"].shape[1]
            lengths.extend([prompt_len] * out.shape[0])
            for ids in out[:, prompt_len:].tolist():
                outputs.append(tokenizer.decode(ids, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
        if was_training:
            model.train()
    if return_prompt_lengths:
        return outputs, lengths
    return outputs


@torch.no_grad()
def greedy_next_tokens(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
                       batch_size: int = 16, device: Optional[str] = None) -> List[int]:
    """Argmax next token id for each prompt."""
    device = device or str(next(model.parameters()).device)
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    ids: List[int] = []
    try:
        for batch in batches(list(prompts), batch_size):
            enc = tokenizer(list(batch), return_tensors="pt", padding=True).to(device)
            logits = model(**enc).logits[:, -1]
            ids.extend(logits.argmax(dim=-1).tolist())
    finally:
        tokenizer.padding_side = old_side
    return ids


@torch.no_grad()
def topk_next_tokens(model: torch.nn.Module, tokenizer, prompt: str, k: int = 5,
                     device: Optional[str] = None, context=None) -> List[str]:
    """Top-k next tokens for a single prompt (Tables 3, 4 of the paper)."""
    device = device or str(next(model.parameters()).device)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    if context is not None:
        with context:
            logits = model(**enc).logits[0, -1]
    else:
        logits = model(**enc).logits[0, -1]
    top = torch.topk(logits, k).indices.tolist()
    return [tokenizer.decode([t]) for t in top]
