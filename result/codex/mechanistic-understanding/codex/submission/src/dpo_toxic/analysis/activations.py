"""Section 5.2, Figure 2 -- the drop of toxic-vector activations after DPO.

For 1,199 RealToxicityPrompts prompts we greedily generate 20 tokens with GPT2
(per the addendum) and then measure the mean activation
``m_i = sigma(x^l . k_i^l)`` of the top toxic value vectors, both before and
after DPO.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..architecture import TransformerInternals, mlp_activation
from ..generation import generate_continuations


@torch.no_grad()
def collect_mean_activations(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
                             layers: Sequence[int], batch_size: int = 8,
                             device: Optional[str] = None,
                             max_length: int = 256) -> Dict[int, torch.Tensor]:
    """Mean activation of *every* value vector for each requested layer.

    The activation is computed on the true key-matrix input (after the pre-MLP
    layernorm) and averaged over all timesteps and prompts.
    """
    from ..utils import batches

    device = device or str(next(model.parameters()).device)
    internals = TransformerInternals(model)
    sums = {l: torch.zeros(internals.d_mlp) for l in layers}
    counts = 0
    for batch in batches(list(prompts), batch_size):
        enc = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        # Hooks capture x^{l-mid}; activations are computed from the true input
        # of the key matrix (i.e. after the pre-MLP layernorm).
        collector = _MidHook(internals, list(layers))
        with collector:
            model(**enc)
        for l in layers:
            x_mid = collector.residuals[l]                     # (B, T, d)
            act = mlp_activation(internals, l, x_mid.to(device))  # (B, T, d_mlp)
            sums[l] += act.reshape(-1, act.shape[-1]).float().sum(0).cpu()
        counts += int(enc["attention_mask"].sum())
    return {l: sums[l] / max(counts, 1) for l in layers}


class _MidHook:
    """Capture ``x^{l-mid}`` (input of the pre-MLP layernorm) for given layers."""

    def __init__(self, internals: TransformerInternals, layers: List[int]):
        self.internals = internals
        self.layers = layers
        self.handles = []
        self.residuals: Dict[int, torch.Tensor] = {}

    def __enter__(self):
        for l in self.layers:
            mod = self.internals.mlp_input_module(l)

            def hook(module, inputs, layer=l):
                self.residuals[layer] = inputs[0].detach()

            self.handles.append(mod.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


def activation_drop_table(before_model: torch.nn.Module, after_model: torch.nn.Module,
                          tokenizer, prompts: Sequence[str],
                          toxic_locations: Sequence[Dict],
                          batch_size: int = 8, device: Optional[str] = None,
                          max_new_tokens: int = 20) -> Dict:
    """Figure 2: mean activations of the top toxic vectors before/after DPO.

    The 20-token generations are produced greedily by the **pre-DPO** model, as
    specified in the addendum, and then scored under both models.
    """
    generations = generate_continuations(before_model, tokenizer, prompts,
                                         max_new_tokens=max_new_tokens,
                                         batch_size=batch_size, device=device)
    sequences = [p + g for p, g in zip(prompts, generations)]
    layers = sorted({int(s["layer"]) for s in toxic_locations})
    act_before = collect_mean_activations(before_model, tokenizer, sequences, layers,
                                          batch_size=batch_size, device=device)
    act_after = collect_mean_activations(after_model, tokenizer, sequences, layers,
                                         batch_size=batch_size, device=device)
    rows = []
    for s in toxic_locations:
        l, i = int(s["layer"]), int(s["index"])
        rows.append({
            "layer": l, "index": i, "cosine_with_w_toxic": float(s.get("cosine", float("nan"))),
            "mean_activation_gpt2": float(act_before[l][i]),
            "mean_activation_dpo": float(act_after[l][i]),
            "drop": float(act_before[l][i] - act_after[l][i]),
        })
    return {"rows": rows, "n_prompts": len(prompts)}
