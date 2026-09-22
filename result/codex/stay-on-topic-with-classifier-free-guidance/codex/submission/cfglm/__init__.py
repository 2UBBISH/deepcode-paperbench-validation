"""Classifier-Free Guidance (CFG) for autoregressive language models.

Reference
---------
Sanchez, Spangher, Fan, Levi, Biderman.
"Stay on topic with Classifier-Free Guidance" (ICML 2024).

This package implements the core contribution of the paper: applying
classifier-free guidance at inference time to the *logits* of an
autoregressive language model.

The central equation (Equation 7 of the paper) is

    log P_hat(w_i | w_{<i}, c) = log P(w_i | w_{<i})
        + gamma * ( log P(w_i | w_<i, c) - log P(w_i | w_<i) )

which is equivalent (up to a per-step additive constant that does not
change the resulting softmax) to the logit-space mixture

    logits_cfg = (1 - gamma) * logits_uncond + gamma * logits_cond

Negative prompting (Equation 5) generalises the unconditional branch to
an arbitrary "negative" conditioning c_bar; setting c_bar = empty
recovers the standard formulation.
"""

from .cfg import (  # noqa: F401
    cfg_combine_logits,
    cfg_combine_logprobs,
    cfg_logits_from_logprobs,
    guidance_weight,
)
from .generation import (  # noqa: F401
    CFGLogitsProcessor,
    cfg_generate,
    cfg_generate_batch,
)
from .scoring import (  # noqa: F401
    CFGScorer,
    cfg_choice_scores,
    cfg_loglikelihood,
)
from .flops import (  # noqa: F401
    flops_per_token,
    transformer_flops,
)
from .stats import (  # noqa: F401
    ancova,
    entropy_from_logits,
    estimate_pass_at_k,
    spearman_correlation,
    top_p_overlap,
)

__all__ = [
    "cfg_combine_logits",
    "cfg_combine_logprobs",
    "cfg_logits_from_logprobs",
    "guidance_weight",
    "CFGLogitsProcessor",
    "cfg_generate",
    "cfg_generate_batch",
    "CFGScorer",
    "cfg_loglikelihood",
    "cfg_choice_scores",
    "flops_per_token",
    "transformer_flops",
    "ancova",
    "entropy_from_logits",
    "estimate_pass_at_k",
    "spearman_correlation",
    "top_p_overlap",
]

__version__ = "0.1.0"
