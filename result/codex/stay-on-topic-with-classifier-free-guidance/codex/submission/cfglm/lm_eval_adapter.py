"""Integration with EleutherAI's Language Model Evaluation Harness.

The paper's Table 5 comes from running the LM Evaluation Harness with a
CFG-modified model.  This module provides that model: a subclass of the
harness's ``HFLM`` that overrides the likelihood routines with the CFG
scoring of ``cfglm/scoring.py`` (Equation 7).

Usage (with ``lm-evaluation-harness`` installed)::

    python -m cfglm.lm_eval_adapter \
        --model tiiuae/falcon-7b --gamma 1.5 \
        --tasks arc_challenge,arc_easy,boolq,hellaswag,piqa,sciq,triviaqa,winogrande,lambada_openai

or programmatically::

    from cfglm.lm_eval_adapter import CFGHFLM
    lm = CFGHFLM(pretrained="gpt2-xl", gamma=1.5)
    results = lm_eval.evaluator.simple_evaluate(model=lm, tasks=["lambada_openai"])

Note that ``gamma = 1.0`` reproduces vanilla prompting exactly, so a sweep is
simply an outer loop over ``gamma``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch

try:  # the adapter is optional: the native harness does not need lm-eval
    from lm_eval.models.huggingface import HFLM

    _HFLM_AVAILABLE = True
except Exception:  # pragma: no cover
    HFLM = object  # type: ignore
    _HFLM_AVAILABLE = False

from .scoring import CFGScorer


class CFGHFLM(HFLM):  # type: ignore[misc]
    """``HFLM`` whose scores are computed with classifier-free guidance."""

    def __init__(
        self,
        gamma: float = 1.5,
        uncond_prefix_tokens: int = 1,
        negative_context: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        if not _HFLM_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "lm-evaluation-harness is not installed; use cfglm.harness instead"
            )
        self.gamma = float(gamma)
        self.uncond_prefix_tokens = int(uncond_prefix_tokens)
        self.negative_context = negative_context
        super().__init__(*args, **kwargs)
        self._scorer = None

    # ------------------------------------------------------------------
    @property
    def scorer(self) -> CFGScorer:
        if self._scorer is None:
            self._scorer = CFGScorer(
                self.model,
                self.tokenizer,
                gamma=self.gamma,
                uncond_prefix_tokens=self.uncond_prefix_tokens,
                negative_context=self.negative_context,
                batch_size=getattr(self, "batch_size", 8),
            )
        return self._scorer

    # ------------------------------------------------------------------
    # lm-eval API
    # ------------------------------------------------------------------
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
        """Compute CFG log-likelihoods for a batch of tokenised requests.

        ``requests`` is a list of ``((ctx, continuation), ctx_enc, cont_enc)``
        tuples (the exact shape varies slightly between harness versions).
        """
        from cfglm.cfg import cfg_combine_logits
        import torch.nn.functional as F

        scorer = self.scorer
        contexts: List[List[int]] = []
        continuations: List[List[int]] = []
        for request in requests:
            # support both 3-tuples and 2-tuples
            if len(request) == 3:
                _, ctx_enc, cont_enc = request
            else:  # pragma: no cover
                (_, cont_enc), ctx_enc = request
            contexts.append(list(ctx_enc))
            continuations.append(list(cont_enc))

        results: List[Tuple[float, bool]] = []
        for ctx_enc, cont_enc in zip(contexts, continuations):
            if not cont_enc:
                results.append((0.0, True))
                continue
            sequences = []
            positions = []
            uncond_ctx = scorer._uncond_context(ctx_enc)
            sequences.append(ctx_enc + cont_enc)
            positions.append((len(ctx_enc) - 1, len(ctx_enc) + len(cont_enc) - 1))
            sequences.append(uncond_ctx + cont_enc)
            positions.append((len(uncond_ctx) - 1, len(uncond_ctx) + len(cont_enc) - 1))
            lp = scorer._forward_logprobs(sequences, positions)
            targets = torch.tensor(cont_enc, dtype=torch.long, device=scorer.device)
            cond_lp = lp[0].gather(-1, targets.reshape(-1, 1)).squeeze(-1)
            uncond_lp = lp[1].gather(-1, targets.reshape(-1, 1)).squeeze(-1)
            cfg_lp = uncond_lp + scorer.gamma * (cond_lp - uncond_lp)
            cfg_logits = cfg_combine_logits(lp[0], lp[1], scorer.gamma)
            is_greedy = bool(torch.all(cfg_logits.argmax(-1) == targets))
            results.append((float(cfg_lp.sum()), is_greedy))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        """Rolling log-likelihood (used by harness perplexity tasks)."""
        scorer = self.scorer
        out = []
        for request in requests:
            # the harness passes (context, continuation) string tuples
            if isinstance(request, (tuple, list)) and len(request) == 2 and isinstance(request[0], str):
                context, continuation = request
            else:  # pragma: no cover
                context, continuation = request[0], request[1]
            ctx_ids = self.tokenizer(context, add_special_tokens=False).input_ids
            cont_ids = self.tokenizer(continuation, add_special_tokens=False).input_ids
            stats = scorer.token_stats(ctx_ids, cont_ids)
            out.append(stats.loglikelihood)
        return out


def register() -> None:
    """Register the ``cfg`` model with lm-eval (call before ``lm_eval`` runs)."""
    if not _HFLM_AVAILABLE:  # pragma: no cover
        raise ImportError("lm-evaluation-harness is not installed")
    from lm_eval.api.registry import register_model

    register_model("cfg", CFGHFLM)


def main() -> None:  # pragma: no cover - thin CLI
    import argparse

    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness with CFG")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--tasks", default="lambada_openai")
    parser.add_argument("--uncond-prefix-tokens", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import lm_eval

    register()
    lm = CFGHFLM(
        pretrained=args.model,
        gamma=args.gamma,
        uncond_prefix_tokens=args.uncond_prefix_tokens,
        batch_size=args.batch_size,
    )
    results = lm_eval.simple_evaluate(
        model=lm, tasks=args.tasks.split(","), limit=args.limit
    )
    import json

    print(json.dumps(results["results"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
