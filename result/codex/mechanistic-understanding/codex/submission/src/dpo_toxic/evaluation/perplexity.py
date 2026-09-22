"""Perplexity on Wikitext-2 (Section 3.3).

Standard sliding-window (stride = half the context) token-level perplexity, as
used by prior work for GPT2 evaluation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


@torch.no_grad()
def perplexity(model: torch.nn.Module,
               tokenizer,
               texts: Optional[Sequence[str]] = None,
               max_length: int = 1024,
               stride: Optional[int] = None,
               device: Optional[str] = None,
               dataset_split: str = "test",
               cache_dir: Optional[str] = None) -> float:
    """Token-level perplexity of ``texts`` (defaults to the Wikitext-2 test set).

    Sliding-window evaluation with ``stride = max_length / 2``: every token is
    scored exactly once, using as much left context as the model allows.  The
    loss is the model's own shifted language-modelling loss, weighted by the
    number of newly scored tokens per window.
    """
    if texts is None:
        from ..data.wikitext import load_wikitext2

        ds = load_wikitext2(cache_dir=cache_dir)[dataset_split]
        texts = [t for t in ds["text"] if t.strip()]
    stride = stride or max_length // 2
    device = device or str(next(model.parameters()).device)
    text = "\n\n".join(texts)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    seq_len = ids.size(1)
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        chunk = ids[:, begin:end]
        targets = chunk.clone()
        # only the tokens that were not scored by the previous window contribute
        targets[:, :-target_len] = -100
        outputs = model(chunk, labels=targets)
        nlls.append(outputs.loss.float() * target_len)
        prev_end = end
        if end == seq_len:
            break
    total_nll = float(torch.stack(nlls).sum())
    return float(torch.exp(torch.tensor(total_nll / seq_len)))
