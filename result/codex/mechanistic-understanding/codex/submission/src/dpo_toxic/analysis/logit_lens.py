"""Figure 1 -- the logit lens on GPT2 and GPT2_DPO.

Given the prompts of RealToxicityPrompts whose next token under GPT2 is
``"sh*t"`` (295 prompts in the paper), we apply the unembedding layer to every
intermittent layer and plot the average probability of that token at each layer
("mid" layers, i.e. after the attention block and before the MLP, are indicated
with minor ticks).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..architecture import TransformerInternals
from ..generation import greedy_next_tokens


def select_prompts_for_token(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
                             token: str = "sh*t", batch_size: int = 16,
                             device: Optional[str] = None,
                             max_prompts: Optional[int] = None) -> List[str]:
    """Prompts whose greedy next token is ``token`` (295 in the paper)."""
    target = tokenizer.encode(token, add_special_tokens=False)[0]
    keep: List[str] = []
    for batch_start in range(0, len(prompts), batch_size):
        batch = list(prompts[batch_start: batch_start + batch_size])
        ids = greedy_next_tokens(model, tokenizer, batch, batch_size=batch_size, device=device)
        for text, idx in zip(batch, ids):
            if idx == target:
                keep.append(text)
                if max_prompts and len(keep) >= max_prompts:
                    return keep
    return keep


@torch.no_grad()
def logit_lens(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
               token: str = "sh*t", batch_size: int = 8,
               device: Optional[str] = None, max_length: int = 256,
               position: str = "last") -> Dict[str, List[float]]:
    """Average probability of ``token`` at each intermittent layer.

    Returns ``{"mid": [...], "post_block": [...]}``, one entry per layer.
    ``mid`` is measured on ``x^{l-mid}`` (after attention, before MLP) and
    ``post_block`` on the residual stream after the MLP block.
    """
    from ..utils import batches

    device = device or str(next(model.parameters()).device)
    internals = TransformerInternals(model)
    n_layers = internals.n_layers
    target = tokenizer.encode(token, add_special_tokens=False)[0]
    mid_scores = [[] for _ in range(n_layers)]
    post_scores = [[] for _ in range(n_layers)]

    for batch in batches(list(prompts), batch_size):
        enc = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        mids: Dict[int, torch.Tensor] = {}
        handles = []
        for l in range(n_layers):
            def hook(module, inputs, layer=l):
                mids[layer] = inputs[0].detach()
            handles.append(internals.mlp_input_module(l).register_forward_pre_hook(hook))
        out = model(**enc, output_hidden_states=True)
        for h in handles:
            h.remove()
        idx = -1 if position == "last" else None
        for l in range(n_layers):
            x_mid = mids[l]
            logits_mid = internals.apply_unembedding(x_mid)
            probs = torch.softmax(logits_mid.float(), dim=-1)
            sel = probs[:, idx, target] if idx == -1 else probs[:, :, target].mean(dim=1)
            mid_scores[l].extend(sel.cpu().tolist())
            logits_post = internals.apply_unembedding(out.hidden_states[l + 1])
            probs_post = torch.softmax(logits_post.float(), dim=-1)
            sel_post = probs_post[:, idx, target] if idx == -1 else probs_post[:, :, target].mean(dim=1)
            post_scores[l].extend(sel_post.cpu().tolist())

    def mean(xs):
        return float(sum(xs) / max(len(xs), 1))

    return {"mid": [mean(s) for s in mid_scores],
            "post_block": [mean(s) for s in post_scores],
            "token": token, "n_prompts": len(prompts)}
