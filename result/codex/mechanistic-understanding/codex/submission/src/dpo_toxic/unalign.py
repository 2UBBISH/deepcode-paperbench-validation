"""Section 6 -- un-aligning the DPO'd models.

GPT2_DPO learns an offset that keeps the residual stream outside the toxic
regions ``gamma(MLP.k_toxic)``; scaling the *key vectors* of the most toxic
value vectors by 10x enlarges those regions and brings the toxicity back
(Table 4).

Llama2_DPO (out of scope for this reproduction because the weights are gated)
turns toxic vectors off through its gates, so setting the gate values of the
top toxic vectors to 1 -- or scaling ``W_2`` by 3x -- re-activates toxicity
(Table 5).  Both variants are implemented here so the pipeline is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .architecture import TransformerInternals


@dataclass
class UnalignConfig:
    n_vectors: int = 7           # "as few as 7 toxic key vectors" (Table 4)
    key_scale: float = 10.0      # "scale their key vectors by 10x"
    up_scale: float = 3.0        # Llama2: scale W_2 by 3x (Table 5)
    n_gate_vectors: int = 8      # Llama2: turn on 8 gate components


def top_toxic_locations(selections: Sequence[Dict], n: int) -> List[Dict]:
    return list(selections[:n])


@torch.no_grad()
def scale_toxic_key_vectors(model: torch.nn.Module, selections: Sequence[Dict],
                            cfg: Optional[UnalignConfig] = None) -> List[Dict]:
    """Multiply the selected key vectors by ``cfg.key_scale`` (Table 4, GPT2)."""
    cfg = cfg or UnalignConfig()
    internals = TransformerInternals(model)
    chosen = top_toxic_locations(selections, cfg.n_vectors)
    for s in chosen:
        internals.scale_key_vectors([int(s["layer"])], [int(s["index"])], cfg.key_scale, branch="k")
    return chosen


@torch.no_grad()
def scale_up_projection(model: torch.nn.Module, selections: Sequence[Dict],
                        cfg: Optional[UnalignConfig] = None) -> List[Dict]:
    """Llama2 variant: scale ``W_2`` (the linear branch of the GLU) by 3x (Table 5)."""
    cfg = cfg or UnalignConfig()
    internals = TransformerInternals(model)
    if internals.arch != "glu":
        raise ValueError("scale_up_projection only applies to GLU models")
    chosen = top_toxic_locations(selections, cfg.n_vectors)
    for s in chosen:
        l, i = int(s["layer"]), int(s["index"])
        w = internals.key_weight(l, "up")
        w[i] = w[i] * cfg.up_scale
        internals._write_key_weight(l, "up", w)
    return chosen


class GateOverrideHook:
    """Llama2 variant: force ``sigma(W_1 x) = 1`` for the selected neurons (Table 5)."""

    def __init__(self, model: torch.nn.Module, selections: Sequence[Dict],
                 cfg: Optional[UnalignConfig] = None):
        cfg = cfg or UnalignConfig()
        self.internals = TransformerInternals(model)
        self.chosen = top_toxic_locations(selections, cfg.n_gate_vectors)
        self._handles = []

    def __enter__(self):
        per_layer: Dict[int, List[int]] = {}
        for s in self.chosen:
            per_layer.setdefault(int(s["layer"]), []).append(int(s["index"]))
        for l, idxs in per_layer.items():
            gate = self.internals.layers[l].mlp.gate_proj

            def hook(module, inputs, output, idxs=idxs):
                output = output.clone()
                output[..., idxs] = 1.0
                return output

            self._handles.append(gate.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def run_unalign_experiment(model: torch.nn.Module, tokenizer, selections: Sequence[Dict],
                           prompts: Sequence[str], wikitext_texts: Optional[Sequence[str]] = None,
                           f1_items: Optional[Sequence[Dict]] = None,
                           toxicity_scorer=None, cfg: Optional[UnalignConfig] = None,
                           batch_size: int = 16, device: Optional[str] = None,
                           out_path: Optional[str] = None,
                           baseline: Optional[Dict[str, float]] = None) -> Dict:
    """Table 4: align -> scale key vectors -> measure toxicity / PPL / F1."""
    from .evaluation.f1 import generation_f1
    from .evaluation.perplexity import perplexity
    from .generation import generate_continuations
    from .utils import save_json

    cfg = cfg or UnalignConfig()
    result: Dict[str, Dict[str, float]] = {}
    if baseline is not None:
        result["DPO"] = dict(baseline)

    generations = generate_continuations(model, tokenizer, prompts, batch_size=batch_size, device=device)
    dpo_row = {"ppl": float(perplexity(model, tokenizer, texts=wikitext_texts, device=device))}
    if toxicity_scorer is not None:
        dpo_row["toxicity"] = float(sum(toxicity_scorer.score(generations)) / max(len(generations), 1))
    if f1_items is not None:
        dpo_row["f1"] = generation_f1(model, tokenizer, f1_items, batch_size=batch_size,
                                      device=device)["f1"]
    result.setdefault("GPT2_DPO", dpo_row)

    chosen = scale_toxic_key_vectors(model, selections, cfg)
    result["SCALE_MLP.k_toxic"] = {
        "ppl": float(perplexity(model, tokenizer, texts=wikitext_texts, device=device)),
    }
    generations = generate_continuations(model, tokenizer, prompts, batch_size=batch_size, device=device)
    if toxicity_scorer is not None:
        result["SCALE_MLP.k_toxic"]["toxicity"] = float(
            sum(toxicity_scorer.score(generations)) / max(len(generations), 1))
    if f1_items is not None:
        result["SCALE_MLP.k_toxic"]["f1"] = generation_f1(
            model, tokenizer, f1_items, batch_size=batch_size, device=device)["f1"]
    result["SCALE_MLP.k_toxic"]["scaled_vectors"] = chosen
    result["SCALE_MLP.k_toxic"]["scale"] = cfg.key_scale
    if out_path:
        save_json(result, out_path)
    return result
