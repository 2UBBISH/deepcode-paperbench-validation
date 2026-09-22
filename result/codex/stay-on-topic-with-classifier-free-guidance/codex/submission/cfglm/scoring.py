"""Likelihood-based CFG scoring.

Most benchmarks in the paper (Section 3.1) are evaluated by *scoring*
candidate continuations rather than by sampling freely.  For a prompt
(context) ``c`` and a continuation ``w = w_1..w_m`` the harness needs

    log P(w | c) = sum_i log P(w_i | w_{<i}, c)

which under CFG becomes, token by token (Equation 7),

    log P_hat(w_i | w_{<i}, c) = log P(w_i | w_{<i})
        + gamma * ( log P(w_i | w_{<i}, c) - log P(w_i | w_{<i}) )

The unconditional branch ``P(w_i | w_{<i})`` is obtained by dropping the
prompt and keeping its final ``uncond_prefix_tokens`` tokens as a minimal
context (Section 3.1).  A negative prompt (Equation 5) replaces that
context.

This module mirrors the ``loglikelihood`` interface of EleutherAI's
Language Model Evaluation Harness so that it can be used either stand-alone
or as a drop-in replacement (see ``cfglm/lm_eval_adapter.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .cfg import cfg_combine_logits, guidance_weight

Request = Tuple[str, str]


@dataclass
class CFGTokenStats:
    """Per-token statistics for a single (context, continuation) pair."""

    cond_logprobs: torch.Tensor  # [m] log P(w_i | w_<i, c)
    uncond_logprobs: torch.Tensor  # [m] log P(w_i | w_<i)
    cfg_logprobs: torch.Tensor  # [m] log P_hat(w_i | w_<i, c)
    is_greedy: bool
    n_tokens: int

    @property
    def loglikelihood(self) -> float:
        return float(self.cfg_logprobs.sum())


def encode_pair(tokenizer, context: str, continuation: str) -> Tuple[List[int], List[int]]:
    """Tokenise a (context, continuation) pair into two id lists.

    The context and the continuation are encoded separately and then
    concatenated, which avoids BPE merges across the boundary (the LM
    harness does the same, and it matters for e.g. ``" "``-prefixed
    continuations).
    """
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    return ctx_ids, cont_ids


class CFGScorer:
    """Score continuations with classifier-free guidance.

    Args:
        model: a HuggingFace causal LM in ``eval`` mode.
        tokenizer: its tokenizer.
        gamma: guidance strength (``1.0`` = vanilla conditional scoring).
        uncond_prefix_tokens: number of prompt tokens kept as the
            unconditional context (``1`` follows Section 3.1).
        negative_context: optional negative prompt (Equation 5).  When set,
            it is used instead of the truncated prompt for the
            "unconditional" branch.
        batch_size: number of *requests* forwarded at once.
        max_length: optional truncation length for the prompt.
    """

    def __init__(
        self,
        model,
        tokenizer,
        gamma: float = 1.0,
        uncond_prefix_tokens: int = 1,
        negative_context: Optional[str] = None,
        batch_size: int = 8,
        max_length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gamma = guidance_weight(gamma)
        self.uncond_prefix_tokens = max(1, int(uncond_prefix_tokens))
        self.negative_context = negative_context
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or next(model.parameters()).device
        self._negative_ids: Optional[List[int]] = None
        if negative_context is not None:
            self._negative_ids = tokenizer(negative_context, add_special_tokens=False).input_ids

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _ensure_context(self, ctx_ids: Sequence[int]) -> List[int]:
        """Guarantee at least one context token so positions stay valid.

        Sequences with an empty context have no "previous" position to read a
        distribution from; the language model's BOS (or EOS) token is used as
        a neutral prefix, which is what the LM harness does for rolling
        log-likelihood requests.
        """
        ctx_ids = list(ctx_ids)
        if ctx_ids:
            return ctx_ids
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is None:
            bos = getattr(self.tokenizer, "eos_token_id", None)
        return [int(bos) if bos is not None else 0]

    def _uncond_context(self, ctx_ids: Sequence[int]) -> List[int]:
        if self._negative_ids is not None:
            return list(self._negative_ids)
        if not ctx_ids:
            return []
        return list(ctx_ids[-self.uncond_prefix_tokens :])

    @staticmethod
    def _left_pad(batch: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        max_len = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, ids in enumerate(batch):
            input_ids[row, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, max_len - len(ids) :] = 1
        return input_ids, attention_mask, max_len

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 0)
        return pos

    @torch.no_grad()
    def _forward_logprobs(
        self, sequences: List[List[int]], positions: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """Return ``log_softmax(logits)`` for a slice of each sequence.

        ``positions`` gives ``(start, stop)`` absolute indices for each
        sequence; the returned tensor for row ``r`` has shape
        ``[stop - start, vocab]``.
        """
        if not sequences:
            return []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        out: List[torch.Tensor] = []
        for begin in range(0, len(sequences), self.batch_size):
            chunk = sequences[begin : begin + self.batch_size]
            chunk_pos = positions[begin : begin + self.batch_size]
            input_ids, attn, max_len = self._left_pad(chunk, pad_id)
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                position_ids=self._position_ids(attn).to(self.device),
                use_cache=False,
            ).logits
            logits = logits.float()
            for row, (start, stop) in enumerate(chunk_pos):
                offset = max_len - len(chunk[row])
                out.append(F.log_softmax(logits[row, offset + start : offset + stop, :], dim=-1))
        return out

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def token_stats(self, ctx_ids: Sequence[int], cont_ids: Sequence[int]) -> CFGTokenStats:
        """Per-token conditional / unconditional / CFG log-probabilities."""
        cont = list(cont_ids)
        m = len(cont)
        if m == 0:
            empty = torch.zeros(0)
            return CFGTokenStats(empty, empty, empty, True, 0)
        ctx = self._ensure_context(ctx_ids)
        uncond_ctx = self._uncond_context(ctx)
        targets = torch.tensor(cont, dtype=torch.long, device=self.device)

        cond_seq = ctx + cont
        uncond_seq = uncond_ctx + cont
        lp = self._forward_logprobs(
            [cond_seq, uncond_seq],
            [(len(ctx) - 1, len(ctx) + m - 1), (len(uncond_ctx) - 1, len(uncond_ctx) + m - 1)],
        )
        cond_lp = lp[0].gather(-1, targets.reshape(-1, 1)).squeeze(-1)
        uncond_lp = lp[1].gather(-1, targets.reshape(-1, 1)).squeeze(-1)
        cfg_lp = uncond_lp + self.gamma * (cond_lp - uncond_lp)
        # The CFG distribution re-orders the vocabulary; rebuild it explicitly.
        cfg_logits = cfg_combine_logits(lp[0], lp[1], self.gamma)
        is_greedy = bool(torch.all(cfg_logits.argmax(-1) == targets))
        return CFGTokenStats(cond_lp, uncond_lp, cfg_lp, is_greedy, m)

    def cfg_token_logprobs(self, ctx_ids: Sequence[int], cont_ids: Sequence[int]) -> torch.Tensor:
        """Full ``[m, vocab]`` CFG log-probabilities for a continuation."""
        cont = list(cont_ids)
        m = len(cont)
        if m == 0:
            return torch.zeros(0, self.model.config.vocab_size)
        ctx = self._ensure_context(ctx_ids)
        uncond_ctx = self._uncond_context(ctx)
        lp = self._forward_logprobs(
            [ctx + cont, uncond_ctx + cont],
            [(len(ctx) - 1, len(ctx) + m - 1), (len(uncond_ctx) - 1, len(uncond_ctx) + m - 1)],
        )
        return F.log_softmax(cfg_combine_logits(lp[0], lp[1], self.gamma), dim=-1)

    def loglikelihood(self, requests: Iterable[Request]) -> List[Tuple[float, bool]]:
        """HuggingFace/LM-harness style ``loglikelihood``.

        Returns ``(total_cfg_loglikelihood, is_greedy)`` for each request,
        where ``is_greedy`` is ``True`` when every continuation token is the
        argmax of the CFG distribution.
        """
        prepared: List[Tuple[List[int], List[int]]] = []
        for context, continuation in requests:
            ctx_ids, cont_ids = encode_pair(self.tokenizer, context, continuation)
            if self.max_length is not None:
                ctx_ids = ctx_ids[-self.max_length :]
            ctx_ids = self._ensure_context(ctx_ids)
            prepared.append((ctx_ids, cont_ids))

        results: List[Tuple[float, bool]] = []
        chunk = max(1, self.batch_size)
        for start in range(0, len(prepared), chunk):
            batch = prepared[start : start + chunk]
            # Two sequences per request: conditional and unconditional.
            sequences: List[List[int]] = []
            positions: List[Tuple[int, int]] = []
            targets: List[List[int]] = []
            for ctx_ids, cont_ids in batch:
                uncond_ctx = self._uncond_context(ctx_ids)
                sequences.append(list(ctx_ids) + list(cont_ids))
                positions.append((len(ctx_ids) - 1, len(ctx_ids) + len(cont_ids) - 1))
                sequences.append(list(uncond_ctx) + list(cont_ids))
                positions.append(
                    (len(uncond_ctx) - 1, len(uncond_ctx) + len(cont_ids) - 1)
                )
                targets.append(list(cont_ids))

            logprobs = self._forward_logprobs(sequences, positions)

            for idx, (ctx_ids, cont_ids) in enumerate(batch):
                lp_cond = logprobs[2 * idx]
                lp_uncond = logprobs[2 * idx + 1]
                if not cont_ids:
                    results.append((0.0, True))
                    continue
                tgt = torch.tensor(cont_ids, dtype=torch.long, device=self.device)
                cond_lp = lp_cond.gather(-1, tgt.reshape(-1, 1)).squeeze(-1)
                uncond_lp = lp_uncond.gather(-1, tgt.reshape(-1, 1)).squeeze(-1)
                cfg_lp = uncond_lp + self.gamma * (cond_lp - uncond_lp)
                cfg_logits = cfg_combine_logits(lp_cond, lp_uncond, self.gamma)
                is_greedy = bool(torch.all(cfg_logits.argmax(-1) == tgt))
                results.append((float(cfg_lp.sum()), is_greedy))
        return results

    def loglikelihood_rolling(self, requests: Iterable[Tuple[str, ...]]) -> List[float]:
        """Rolling log-likelihood (used for the perplexity analysis of 5.2).

        The context is the prompt and the continuation is the text being
        scored, exactly as described in the paper's addendum: perplexity is
        computed on the continuation and ignores the prompt.
        """
        results: List[float] = []
        for (text,) in requests:
            ctx_ids, cont_ids = encode_pair(self.tokenizer, "", text)
            ctx_ids = self._ensure_context(ctx_ids)
            stats = self.token_stats(ctx_ids, cont_ids)
            results.append(stats.loglikelihood)
        return results

    def choice_scores(self, context: str, choices: Sequence[str]) -> List[Tuple[float, bool]]:
        """Score a list of candidate continuations for multiple-choice tasks."""
        return self.loglikelihood([(context, choice) for choice in choices])

    def predict_choice(
        self,
        context: str,
        choices: Sequence[str],
        normalize_by_length: bool = False,
        continuation_lengths: Optional[Sequence[int]] = None,
    ) -> int:
        scores = [ll for ll, _ in self.choice_scores(context, choices)]
        if normalize_by_length and continuation_lengths is not None:
            scores = [ll / max(n, 1) for ll, n in zip(scores, continuation_lengths)]
        return int(max(range(len(scores)), key=lambda i: scores[i]))


def cfg_loglikelihood(
    model,
    tokenizer,
    context: str,
    continuation: str,
    gamma: float = 1.5,
    uncond_prefix_tokens: int = 1,
    negative_context: Optional[str] = None,
) -> Tuple[float, bool]:
    """One-shot convenience wrapper around :class:`CFGScorer`."""
    scorer = CFGScorer(
        model,
        tokenizer,
        gamma=gamma,
        uncond_prefix_tokens=uncond_prefix_tokens,
        negative_context=negative_context,
    )
    return scorer.loglikelihood([(context, continuation)])[0]


def cfg_choice_scores(
    model,
    tokenizer,
    context: str,
    choices: Sequence[str],
    gamma: float = 1.5,
    uncond_prefix_tokens: int = 1,
) -> List[Tuple[float, bool]]:
    """Score several candidate continuations under CFG."""
    scorer = CFGScorer(model, tokenizer, gamma=gamma, uncond_prefix_tokens=uncond_prefix_tokens)
    return scorer.choice_scores(context, choices)
