"""Section 5.1 -- how much do the parameters actually move during DPO?

The paper reports that *every* parameter of GPT2/Llama2 has a cosine similarity
greater than 0.99 with its post-DPO counterpart and an average norm difference
below 1e-5 (the GPT2 unembedding is the exception, below 1e-3).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..architecture import TransformerInternals


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1)


def compare_parameters(before: torch.nn.Module, after: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    """Per-tensor cosine similarity and mean absolute/norm difference."""
    b = dict(before.named_parameters())
    a = dict(after.named_parameters())
    out: Dict[str, Dict[str, float]] = {}
    for name, tb in b.items():
        if name not in a:
            continue
        ta = a[name]
        if ta.shape != tb.shape:
            continue
        x, y = _flatten(tb), _flatten(ta)
        cos = float(torch.dot(x, y) / (x.norm() * y.norm()).clamp(min=1e-12))
        out[name] = {
            "cosine": cos,
            "mean_abs_diff": float((x - y).abs().mean()),
            "norm_diff": float((x - y).norm()),
            "relative_norm_diff": float((x - y).norm() / x.norm().clamp(min=1e-12)),
        }
    return out


def summarize_comparison(comparison: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not comparison:
        return {}
    cosines = [v["cosine"] for v in comparison.values()]
    diffs = [v["mean_abs_diff"] for v in comparison.values()]
    return {
        "min_cosine": float(min(cosines)),
        "mean_cosine": float(sum(cosines) / len(cosines)),
        "max_mean_abs_diff": float(max(diffs)),
        "mean_abs_diff": float(sum(diffs) / len(diffs)),
        "n_tensors": len(comparison),
        "n_below_0_99": int(sum(1 for c in cosines if c < 0.99)),
    }


def compare_value_vectors(model: torch.nn.Module, other: torch.nn.Module,
                          layers: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    """``delta_MLP.v``: the shift of every value vector, per layer."""
    ia, ib = TransformerInternals(model), TransformerInternals(other)
    layers = layers if layers is not None else list(range(ia.n_layers))
    deltas, cosines = {}, {}
    for l in layers:
        va = ia.value_weight(l).detach().float().cpu()
        vb = ib.value_weight(l).detach().float().cpu()
        d = vb - va
        deltas[l] = d
        cos = (va / va.norm(dim=1, keepdim=True).clamp(min=1e-8)) @ (
            vb / vb.norm(dim=1, keepdim=True).clamp(min=1e-8)).T
        cosines[l] = torch.diagonal(cos)
    return {"deltas": deltas, "cosines": cosines}
