"""Section 5.2, Figures 3-5 -- the residual-stream offset learned by DPO.

``delta_x^{l-mid} := x^{l-mid}_{DPO} - x^{l-mid}_{GPT2}`` is the offset that
takes the residual stream out of the regions that trigger toxic value vectors.
The paper then asks where that offset comes from and finds that the *value
vectors* have shifted in the opposite direction
(``cos(delta_x^{19-mid}, delta_MLP.v_i^j) < 0`` for most ``i, j < 19``), yet
contribute *towards* ``delta_x`` because their activations are negative
(GeLU of an inactive neuron is a small negative number).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..architecture import TransformerInternals, mlp_activation


@torch.no_grad()
def collect_mid_residuals(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
                          layers: Sequence[int], batch_size: int = 8,
                          max_length: int = 256, device: Optional[str] = None
                          ) -> Dict[int, torch.Tensor]:
    """``x^{l-mid}`` for each prompt (mean-pooled over timesteps)."""
    from ..utils import batches, mean_pool

    device = device or str(next(model.parameters()).device)
    internals = TransformerInternals(model)
    collected: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    for batch in batches(list(prompts), batch_size):
        enc = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        handles = []

        def make_hook(layer):
            def hook(module, inputs):
                x = inputs[0].detach()
                collected[layer].append(
                    mean_pool(x.float(), enc["attention_mask"]).cpu())
            return hook

        for l in layers:
            handles.append(internals.mlp_input_module(l).register_forward_pre_hook(make_hook(l)))
        model(**enc)
        for h in handles:
            h.remove()
    return {l: torch.cat(v, dim=0) for l, v in collected.items()}


def mean_residual_shift(before_model: torch.nn.Module, after_model: torch.nn.Module,
                        tokenizer, prompts: Sequence[str], layer: int = 19,
                        batch_size: int = 8, device: Optional[str] = None
                        ) -> Dict[str, torch.Tensor]:
    """Mean offset ``delta_x^{l-mid}`` and the per-prompt residuals."""
    x_before = collect_mid_residuals(before_model, tokenizer, prompts, [layer],
                                     batch_size=batch_size, device=device)[layer]
    x_after = collect_mid_residuals(after_model, tokenizer, prompts, [layer],
                                    batch_size=batch_size, device=device)[layer]
    delta = x_after - x_before
    return {"x_before": x_before, "x_after": x_after, "delta": delta,
            "mean_delta": delta.mean(dim=0)}


@torch.no_grad()
def shift_vs_value_vector_shift(before_model: torch.nn.Module, after_model: torch.nn.Module,
                                delta_x: torch.Tensor, layer: int = 19) -> Dict[str, torch.Tensor]:
    """Cosine similarity of ``delta_x^{layer-mid}`` with ``delta_MLP.v_i^j`` (Figure 5)."""
    ia, ib = TransformerInternals(before_model), TransformerInternals(after_model)
    d = delta_x.float().flatten()
    d = d / d.norm().clamp(min=1e-8)
    cosines: Dict[int, torch.Tensor] = {}
    for j in range(layer):
        dv = (ib.value_weight(j).detach().float().cpu() - ia.value_weight(j).detach().float().cpu())
        norm = dv.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cosines[j] = (dv / norm) @ d
    return cosines


@torch.no_grad()
def mean_value_vector_activations(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
                                  layer: int = 19, batch_size: int = 8,
                                  device: Optional[str] = None,
                                  max_length: int = 256) -> torch.Tensor:
    """Mean activation of every value vector of one layer (orange areas, Figure 5)."""
    from ..utils import batches

    device = device or str(next(model.parameters()).device)
    internals = TransformerInternals(model)
    total, count = None, 0
    for batch in batches(list(prompts), batch_size):
        enc = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        captured = {}

        def hook(module, inputs):
            captured["x_mid"] = inputs[0].detach()

        h = internals.mlp_input_module(layer).register_forward_pre_hook(hook)
        model(**enc)
        h.remove()
        act = mlp_activation(internals, layer, captured["x_mid"].to(device))
        flat = act.reshape(-1, act.shape[-1]).float()
        total = flat.sum(0).cpu() if total is None else total + flat.sum(0).cpu()
        count += flat.shape[0]
    return total / max(count, 1)


def pca_projection(x_before: torch.Tensor, x_after: torch.Tensor,
                   mean_delta: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Figure 4: project residual streams onto (1) ``mean_delta`` and (2) the top PC.

    Dimension 1 is the (unnormalised) mean difference of the residual streams,
    dimension 2 the first principal component of the pooled residual streams of
    both models.
    """
    if mean_delta is None:
        mean_delta = (x_after - x_before).mean(dim=0)
    pooled = torch.cat([x_before, x_after], dim=0).float()
    centred = pooled - pooled.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centred, full_matrices=False)
    pc = vh[0]
    if torch.dot(pc, mean_delta.float()) < 0:  # deterministic orientation
        pc = -pc
    proj_delta_before = x_before.float() @ mean_delta.float()
    proj_delta_after = x_after.float() @ mean_delta.float()
    proj_pc_before = x_before.float() @ pc
    proj_pc_after = x_after.float() @ pc
    return {
        "pc": pc, "singular_values": s,
        "delta_axis_before": proj_delta_before, "delta_axis_after": proj_delta_after,
        "pc_axis_before": proj_pc_before, "pc_axis_after": proj_pc_after,
    }


@torch.no_grad()
def per_prompt_activations(model: torch.nn.Module, tokenizer, prompts: Sequence[str],
                           layer: int, index: int, batch_size: int = 8,
                           max_length: int = 256, device: Optional[str] = None
                           ) -> Dict[str, torch.Tensor]:
    """Per-prompt activation of one value vector (``MLP.v_index^layer``).

    Figure 4 colours each residual stream by whether it activates
    ``MLP.v_770^19``.  We return both the mean activation over the sequence and
    the maximum, so that "does this prompt activate the vector at all" can be
    read off directly.
    """
    from ..utils import batches

    device = device or str(next(model.parameters()).device)
    internals = TransformerInternals(model)
    means, maxes = [], []
    for batch in batches(list(prompts), batch_size):
        enc = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        captured = {}

        def hook(module, inputs):
            captured["x_mid"] = inputs[0].detach()

        h = internals.mlp_input_module(layer).register_forward_pre_hook(hook)
        model(**enc)
        h.remove()
        act = mlp_activation(internals, layer, captured["x_mid"].to(device),
                             indices=[index]).squeeze(-1)  # (B, T)
        mask = enc["attention_mask"].bool()
        act = act.masked_fill(~mask, float("nan"))
        means.extend(act.nanmean(dim=1).float().cpu().tolist())
        maxes.extend(act.nanmax(dim=1).float().cpu().tolist())
    return {"mean": torch.tensor(means), "max": torch.tensor(maxes),
            "activates": torch.tensor([m > 0 for m in maxes])}
