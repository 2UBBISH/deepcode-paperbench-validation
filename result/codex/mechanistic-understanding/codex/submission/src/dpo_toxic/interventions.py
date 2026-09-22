"""Section 3.3 -- interventions on the residual stream with toxic vectors.

``x^{L-1} := x^{L-1} - alpha * W``

is implemented as a forward pre-hook on the last layer's pre-MLP layernorm, i.e.
the subtraction happens on the residual stream *after* attention and propagates
through the final MLP block and the unembedding, exactly as in the paper.

The scale ``alpha`` is chosen so that the perplexity of the intervened model is
comparable to the perplexity of the post-DPO model (the paper's protocol), which
we implement as a binary search over ``alpha`` on a small budget of Wikitext-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .architecture import ResidualShiftHook, TransformerInternals
from .generation import generate_continuations
from .utils import save_json


@dataclass
class InterventionSpec:
    name: str
    vector: torch.Tensor
    layer: int = -1          # -1 == last layer
    alpha: float = 1.0
    kind: str = "residual_subtract"  # "residual_subtract" | "residual_add"


def make_context(model: torch.nn.Module, spec: InterventionSpec):
    internals = TransformerInternals(model)
    layer = spec.layer if spec.layer >= 0 else internals.n_layers + spec.layer
    mode = "subtract" if spec.kind == "residual_subtract" else "add"
    return ResidualShiftHook(internals, layer, spec.vector, alpha=spec.alpha, mode=mode)


@torch.no_grad()
def calibrate_alpha(model: torch.nn.Module, tokenizer, spec: InterventionSpec,
                    target_ppl: float, texts: Optional[Sequence[str]] = None,
                    device: Optional[str] = None, tol: float = 0.15,
                    max_iter: int = 12, lo: float = 0.0, hi: float = 300.0) -> float:
    """Find ``alpha`` so the intervened perplexity matches ``target_ppl``.

    The paper scales each vector "such that the resulting perplexity is
    comparable to that of the post-DPO model".  Perplexity increases
    monotonically with ``alpha``, so a bisection is sufficient.
    """
    from .evaluation.perplexity import perplexity

    if texts is None:
        from .data.wikitext import load_wikitext2

        ds = load_wikitext2()["test"]
        texts = [t for t in ds["text"] if t.strip()]

    def ppl_at(a: float) -> float:
        spec.alpha = a
        with make_context(model, spec):
            return perplexity(model, tokenizer, texts=texts, device=device)

    best_alpha, best_ppl = 0.0, ppl_at(0.0)
    if abs(best_ppl - target_ppl) <= tol:
        spec.alpha = best_alpha
        return best_alpha
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = ppl_at(mid)
        if abs(p - target_ppl) <= tol:
            best_alpha, best_ppl = mid, p
            break
        if p < target_ppl:
            lo = mid
        else:
            hi = mid
        if abs(p - target_ppl) < abs(best_ppl - target_ppl):
            best_alpha, best_ppl = mid, p
    spec.alpha = best_alpha
    return best_alpha


def evaluate_intervention(model: torch.nn.Module, tokenizer, spec: InterventionSpec,
                          prompts: Sequence[str], toxicity_scorer=None,
                          wikitext_texts: Optional[Sequence[str]] = None,
                          f1_items: Optional[Sequence[Dict]] = None,
                          max_new_tokens: int = 20, batch_size: int = 16,
                          device: Optional[str] = None) -> Dict[str, float]:
    """Toxicity / PPL / F1 of a model under a single intervention (Table 2)."""
    from .evaluation.f1 import generation_f1
    from .evaluation.perplexity import perplexity

    def ctx_factory():
        return make_context(model, spec)

    with ctx_factory():
        generations = generate_continuations(model, tokenizer, prompts, max_new_tokens=max_new_tokens,
                                             batch_size=batch_size, device=device)
        ppl = perplexity(model, tokenizer, texts=wikitext_texts, device=device)
        f1 = None
        if f1_items is not None:
            f1 = generation_f1(model, tokenizer, f1_items, batch_size=batch_size,
                               device=device)["f1"]
    metrics: Dict[str, float] = {"ppl": float(ppl)}
    if toxicity_scorer is not None:
        metrics["toxicity"] = float(sum(toxicity_scorer.score(generations)) / max(len(generations), 1))
    if f1 is not None:
        metrics["f1"] = float(f1)
    metrics["alpha"] = float(spec.alpha)
    return metrics


def run_intervention_table(model: torch.nn.Module, tokenizer,
                           w_toxic: torch.Tensor,
                           selections: Sequence[Dict],
                           value_vectors: torch.Tensor,
                           svd_u: Optional[torch.Tensor],
                           prompts: Sequence[str],
                           target_ppl: float,
                           toxicity_scorer=None,
                           wikitext_texts: Optional[Sequence[str]] = None,
                           f1_items: Optional[Sequence[Dict]] = None,
                           out_dir: Optional[str] = "artifacts/interventions",
                           device: Optional[str] = None,
                           batch_size: int = 16) -> Dict[str, Dict]:
    """Table 2: NO-OP vs SUBTRACT {W_toxic, top MLP.v, top SVD.U}."""
    specs: List[InterventionSpec] = []
    top = selections[0]
    specs.append(InterventionSpec("SUBTRACT_W_toxic", w_toxic, layer=-1))
    specs.append(InterventionSpec(f"SUBTRACT_MLP.v_{top['index']}^{top['layer']}",
                                  value_vectors[0], layer=-1))
    if svd_u is not None:
        specs.append(InterventionSpec("SUBTRACT_SVD.U_toxic[0]", svd_u[:, 0], layer=-1))

    results: Dict[str, Dict] = {}
    from .evaluation.f1 import generation_f1
    from .evaluation.perplexity import perplexity

    if wikitext_texts is None:
        from .data.wikitext import load_wikitext2

        ds = load_wikitext2()["test"]
        wikitext_texts = [t for t in ds["text"] if t.strip()]
    generations = generate_continuations(model, tokenizer, prompts, batch_size=batch_size, device=device)
    no_op = {"ppl": float(perplexity(model, tokenizer, texts=wikitext_texts, device=device)),
             "alpha": 0.0}
    if toxicity_scorer is not None:
        no_op["toxicity"] = float(sum(toxicity_scorer.score(generations)) / max(len(generations), 1))
    if f1_items is not None:
        no_op["f1"] = generation_f1(model, tokenizer, f1_items, batch_size=batch_size,
                                    device=device)["f1"]
    results["NO_OP"] = no_op

    for spec in specs:
        calibrate_alpha(model, tokenizer, spec, target_ppl=target_ppl, texts=wikitext_texts,
                        device=device)
        results[spec.name] = evaluate_intervention(
            model, tokenizer, spec, prompts, toxicity_scorer=toxicity_scorer,
            wikitext_texts=wikitext_texts, f1_items=f1_items, batch_size=batch_size, device=device)
        if out_dir:
            save_json(results, Path(out_dir) / "intervention_table.json")
    return results
