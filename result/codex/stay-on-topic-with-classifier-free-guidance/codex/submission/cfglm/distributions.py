"""Distribution-level analyses of Section 5.

For every datapoint of the P3 sample we compare four next-token
distributions:

``prompted``   ``P(y | x)``                  vanilla conditioning
``unprompted`` ``P(x)``                      no prompt context
``cfg``        ``P_hat(y | x)``              Equation 7 with the ``gamma``
                                             used in the paper (1.5)
``instruct``    ``P_instruct(y | x)``        the instruction-tuned model

and derive the quantities the paper reports:

* Section 5.1 -- the mean entropy per token (4.7 for CFG vs 5.49 vanilla in
  the original runs) and the number of tokens inside the top-p = 90 % mass.
* Section 5.2 -- top-p overlap between CFG and instruction tuning, the
  per-dataset similarity tables, and the correlation between the
  continuation perplexities of the three models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .cfg import cfg_combine_logits, guidance_weight
from .stats import entropy_from_logits, top_p_overlap, top_p_token_set


@dataclass
class PairDistribution:
    """Per-token distributions for one (prompt, continuation) pair."""

    prompted: torch.Tensor  # [m, vocab] log-probabilities
    unprompted: torch.Tensor
    cfg: torch.Tensor
    instruct: Optional[torch.Tensor] = None

    @property
    def n_tokens(self) -> int:
        return int(self.prompted.shape[0])


@torch.no_grad()
def distribution_for_pair(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    gamma: float = 1.5,
    uncond_prefix_tokens: int = 1,
    instruct_model=None,
    device: Optional[torch.device] = None,
    max_prompt_tokens: Optional[int] = None,
) -> PairDistribution:
    """Compute the four distributions of Section 5 for a single pair."""
    device = device or next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if max_prompt_tokens is not None:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    if not prompt_ids:
        # keep the position arithmetic valid for empty prompts
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None:
            bos = getattr(tokenizer, "eos_token_id", None)
        prompt_ids = [int(bos) if bos is not None else 0]
    if not cont_ids:
        empty = torch.zeros(0, model.config.vocab_size, device=device)
        return PairDistribution(empty, empty, empty)

    uncond_ctx = prompt_ids[-max(1, uncond_prefix_tokens):] if prompt_ids else []

    def _logprobs(seq_ids, positions):
        ids = torch.tensor([seq_ids], device=device)
        logits = model(input_ids=ids).logits[0].float()
        start, stop = positions
        return F.log_softmax(logits[start:stop], dim=-1)

    def _positions(ctx):
        return (len(ctx) - 1, len(ctx) + len(cont_ids) - 1)

    prompted = _logprobs(list(prompt_ids) + cont_ids, _positions(prompt_ids))
    unprompted = _logprobs(list(uncond_ctx) + cont_ids, _positions(uncond_ctx))
    cfg_logits = cfg_combine_logits(prompted, unprompted, gamma)
    cfg = F.log_softmax(cfg_logits, dim=-1)

    instruct = None
    if instruct_model is not None:
        ids = torch.tensor([list(prompt_ids) + cont_ids], device=device)
        logits = instruct_model(input_ids=ids).logits[0].float()
        start, stop = _positions(prompt_ids)
        instruct = F.log_softmax(logits[start:stop], dim=-1)

    return PairDistribution(prompted=prompted, unprompted=unprompted, cfg=cfg, instruct=instruct)


def summarize_distributions(dist: PairDistribution, p: float = 0.9) -> Dict[str, float]:
    """Entropies, top-p sizes and overlaps for one datapoint."""
    out: Dict[str, float] = {}
    for name in ("prompted", "unprompted", "cfg"):
        lp = getattr(dist, name)
        if lp.numel() == 0:
            continue
        out[f"entropy_{name}"] = float(entropy_from_logits(lp).mean())
        out[f"topp_size_{name}"] = float(
            torch.tensor([top_p_token_set(lp_i, p).numel() for lp_i in lp], dtype=torch.float).mean()
        )
    if dist.prompted.numel():
        out["topp_overlap_cfg_prompted"] = float(
            torch.tensor(
                [top_p_overlap(a, b, p) for a, b in zip(dist.cfg, dist.prompted)], dtype=torch.float
            ).mean()
        )
    if dist.instruct is not None and dist.instruct.numel():
        out["entropy_instruct"] = float(entropy_from_logits(dist.instruct).mean())
        out["topp_overlap_cfg_instruct"] = float(
            torch.tensor(
                [top_p_overlap(a, b, p) for a, b in zip(dist.cfg, dist.instruct)], dtype=torch.float
            ).mean()
        )
        out["topp_overlap_prompted_instruct"] = float(
            torch.tensor(
                [top_p_overlap(a, b, p) for a, b in zip(dist.prompted, dist.instruct)],
                dtype=torch.float,
            ).mean()
        )
    return out


def continuation_loglikelihoods(dist: PairDistribution, cont_ids: Sequence[int]) -> Dict[str, float]:
    """Total log-likelihood of the continuation under each distribution."""
    targets = torch.tensor(list(cont_ids), dtype=torch.long, device=dist.prompted.device)
    out = {}
    for name in ("prompted", "cfg"):
        lp = getattr(dist, name)
        if lp.numel():
            out[name] = float(lp.gather(-1, targets.reshape(-1, 1)).sum())
    if dist.instruct is not None and dist.instruct.numel():
        out["instruct"] = float(dist.instruct.gather(-1, targets.reshape(-1, 1)).sum())
    return out


@torch.no_grad()
def rank_tokens_by_guidance(
    model,
    tokenizer,
    prompt: str,
    gamma: float = 1.5,
    generated: Optional[Sequence[str]] = None,
    uncond_prefix_tokens: int = 1,
    top_n: int = 5,
    device: Optional[torch.device] = None,
) -> List[dict]:
    """Reproduce the vocabulary ranking of Section 5.3 / Table 3.

    For each decoding step we rank the vocabulary by the guidance-induced
    change in log-probability,

        delta(w) = log P_hat(w | w_<t, c) - log P(w | w_<t, c)
                 = gamma * ( log P(w | w_<t, c) - log P(w | w_<t) )

    and report the ``top_n`` most up-weighted tokens and the ``top_n`` most
    down-weighted tokens, together with the token that was actually
    generated.

    Args:
        prompt: the conditioning text, e.g. ``"The dragon flew over Paris,
            France"`` as in the paper's Table 3.
        generated: the previously decoded tokens.  When ``None`` the model
            greedily decodes its own continuation (vanilla greedy), which is
            the sequence whose ranking is displayed.
    """
    device = device or next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    uncond_ctx = prompt_ids[-max(1, uncond_prefix_tokens):]

    if generated is None:
        generated_ids = []
        current = list(prompt_ids)
        for _ in range(16):
            ids = torch.tensor([current], device=device)
            logits = model(input_ids=ids).logits[0, -1].float()
            nxt = int(logits.argmax())
            generated_ids.append(nxt)
            current.append(nxt)
    else:
        generated_ids = []
        for piece in generated:
            generated_ids.extend(tokenizer(piece, add_special_tokens=False).input_ids)

    rows: List[dict] = []
    for step in range(len(generated_ids)):
        prefix_gen = generated_ids[:step]
        cond_ids = list(prompt_ids) + prefix_gen
        uncond_ids = list(uncond_ctx) + prefix_gen
        cond_logits = model(input_ids=torch.tensor([cond_ids], device=device)).logits[0, -1].float()
        uncond_logits = model(input_ids=torch.tensor([uncond_ids], device=device)).logits[0, -1].float()
        cond_lp = F.log_softmax(cond_logits, dim=-1)
        uncond_lp = F.log_softmax(uncond_logits, dim=-1)
        delta = guidance_weight(gamma) * (cond_lp - uncond_lp)
        values, indices = torch.topk(delta, top_n)
        bottom_values, bottom_indices = torch.topk(-delta, top_n)
        rows.append(
            {
                "step": step,
                "context_token": tokenizer.decode([cond_ids[-1]]) if cond_ids else "",
                "next_token": tokenizer.decode([generated_ids[step]]),
                "most_upweighted": [tokenizer.decode([i]) for i in indices.tolist()],
                "most_downweighted": [tokenizer.decode([i]) for i in bottom_indices.tolist()],
                "upweighted_delta": values.tolist(),
                "downweighted_delta": bottom_values.tolist(),
            }
        )
    return rows


def format_ranking_table(rows: Sequence[dict], top_n: int = 5) -> str:
    """Render the Section 5.3 ranking in the layout of the paper's Table 3.

    As in the paper the columns run ``bottom5 ... bottom1``: the *last*
    column holds the single most down-weighted token.
    """
    header = ["current", "top1", "top2", "top3", "top4", "top5", "...", "bottom5", "bottom4", "bottom3", "bottom2", "bottom1"]
    lines = ["\t".join(header)]
    for row in rows:
        up = row["most_upweighted"][:top_n]
        down = row["most_downweighted"][:top_n]
        cells = [row["context_token"]] + up + ["..."] + list(reversed(down))
        lines.append("\t".join(str(c).replace("\n", " ") for c in cells))
    return "\n".join(lines)
