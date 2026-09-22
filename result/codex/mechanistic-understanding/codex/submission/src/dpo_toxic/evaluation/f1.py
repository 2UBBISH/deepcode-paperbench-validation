"""Token-overlap F1 of generations (Section 3.3).

Following Dinan et al. (2020) and Adolphs et al. (2023): 2,000 Wikipedia
sentences are used as prompts; the model's continuation is compared with the
**original Wikipedia continuation** of the same sentence.  Precision is the
fraction of generated tokens that occur in the reference continuation, recall
the fraction of reference tokens that occur in the generation, and F1 their
harmonic mean (bag-of-tokens overlap, the standard ConvAI2-style metric).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..utils import batches


def f1_overlap(generation: str, reference: str) -> Tuple[float, float, float]:
    g, r = generation.split(), reference.split()
    if not g or not r:
        return 0.0, 0.0, 0.0
    cg, cr = Counter(g), Counter(r)
    overlap = sum((cg & cr).values())
    precision = overlap / max(len(g), 1)
    recall = overlap / max(len(r), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def build_wikipedia_eval_set(n: int = 2000, prompt_tokens: int = 10, max_continuation: int = 20,
                             tokenizer=None, split: str = "test", cache_dir: Optional[str] = None,
                             seed: int = 0) -> List[Dict[str, str]]:
    """Split Wikipedia sentences into a prompt and the true continuation."""
    from ..data.wikitext import wikitext_sentences

    if tokenizer is None:
        from ..utils import load_tokenizer

        tokenizer = load_tokenizer()
    items: List[Dict[str, str]] = []
    for text in wikitext_sentences(split=split, cache_dir=cache_dir):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < prompt_tokens + 5:
            continue
        cont_ids = ids[prompt_tokens:prompt_tokens + max_continuation]
        items.append({
            "prompt": tokenizer.decode(ids[:prompt_tokens]),
            "continuation": tokenizer.decode(cont_ids),
            "n_continuation_tokens": str(len(cont_ids)),
        })
        if len(items) >= n:
            break
    return items


@torch.no_grad()
def generation_f1(model: torch.nn.Module, tokenizer, items: Sequence[Dict[str, str]],
                  batch_size: int = 16, device: Optional[str] = None,
                  max_new_tokens: Optional[int] = None) -> Dict[str, float]:
    """Greedy generation with the same number of tokens as the reference, then F1."""
    device = device or str(next(model.parameters()).device)
    precisions, recalls, f1s = [], [], []
    for batch in batches(list(items), batch_size):
        n_new = max_new_tokens or max(int(b["n_continuation_tokens"]) for b in batch)
        enc = tokenizer([b["prompt"] for b in batch], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for ids, item in zip(gen.tolist(), batch):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            p, r, f = f1_overlap(text, item["continuation"])
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
    n = max(len(f1s), 1)
    return {"precision": sum(precisions) / n, "recall": sum(recalls) / n, "f1": sum(f1s) / n}
