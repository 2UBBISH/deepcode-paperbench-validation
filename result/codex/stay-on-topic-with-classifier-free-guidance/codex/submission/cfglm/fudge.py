"""FUDGE -- future-discriminator guidance (Yang & Klein, 2021).

Table 4 of the paper compares CFG against FUDGE, an external-classifier
control method.  FUDGE reweights the language model's next-token
distribution at every step by the discriminator's score of the *future*
continuation,

    p_hat(w_t | x_{<t}) ∝ p_LM(w_t | x_{<t}) *
        exp( lam * log p_phi(attribute | x_{<t}, w_t) )

and, because it needs a discriminator call per token, the paper notes it
"must be run on every time-step" and is roughly 100x slower than CFG.

Implementation note: the exact FUDGE method marginalises over future
tokens using its own rollouts; here we use the standard practical
approximation of scoring the *partial* continuation ``x_{<t} + w_t`` with
the classifier and reweighting the top-k candidates.  This preserves the
qualitative behaviour the paper relies on (a per-timestep classifier call
and steering through an external model) while keeping the code readable.  The
``lam`` coefficient plays the role of the guidance strength and is tuned the
same way as ``gamma``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def fudge_generate(
    model,
    tokenizer,
    prompt: str,
    reward: Callable[[Sequence[str]], List[float]],
    lam: float = 1.0,
    max_new_tokens: int = 32,
    top_k: int = 8,
    temperature: float = 0.7,
    do_sample: bool = True,
    seed: Optional[int] = None,
) -> str:
    """Generate text with FUDGE-style discriminator reweighting.

    Args:
        reward: a callable mapping a batch of *texts* to the classifier's
            probability of the desired attribute.
        lam: discrimination strength (FUDGE's λ).
        top_k: number of candidate tokens scored by the classifier at each
            step.
    """
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[0, -1].float()
        logprobs = F.log_softmax(logits, dim=-1)
        values, candidates = torch.topk(logprobs, top_k)
        texts = [
            prompt
            + tokenizer.decode(generated + [int(c)], skip_special_tokens=True)
            for c in candidates
        ]
        scores = torch.tensor(reward(texts), dtype=torch.float, device=device)
        scores = torch.log(scores.clamp_min(1e-6))
        reweighted = values + lam * scores
        if do_sample:
            probs = F.softmax(reweighted / max(temperature, 1e-5), dim=-1)
            choice = int(torch.multinomial(probs, 1))
        else:
            choice = int(torch.argmax(reweighted))
        token = int(candidates[choice])
        generated.append(token)
        ids = torch.cat([ids, torch.tensor([[token]], device=device)], dim=-1)
        if token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


def fudge_cost_ratio(n_steps: int, top_k: int) -> float:
    """Discriminator calls per generation, to illustrate the 100x-slower claim."""
    return float(n_steps * top_k)
