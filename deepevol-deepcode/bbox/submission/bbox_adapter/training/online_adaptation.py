"""Online adaptation loop for BBox-Adapter (Algorithm 1).

Reproduces Section 3.4 (``Online Adaptation``) of *Lightweight Adapting for
Black-Box Large Language Models*:

```
Algorithm 1 Overview of BBOX-ADAPTER.
    Input: D = {(x_i, y_i)}_{i=1}^N : Supervised fine-tuning dataset;
           p_LLM : Unadapted black-box LLM; p_theta : Adapted LLM;
           T : Number of iterations; eta : Learning rate;
           Beam size: M; # Candidates generated per step: K.
    p_theta^(0) random initialization;
    for t = 0, ..., T-1 do
        for i = 1, ..., N do
            Sample the candidates {y_hat_{i,m}}_{m=1}^M from the
            adapted inference via Eq.(4);
            Update the positive samples y_{i+}^(t) via Eq.(5);
            Update the negative samples y_{i-}^(t) via Eq.(6);
        end for
        Compute grad_theta l(theta_t) with y_{i+}^(t) and y_{i-}^(t) via Eq.(3);
        Update the adapter via Eq.(7);
    end for
    Output: Fine-tuned theta_T after T-round iteration.
```

with Eq. (7)::

    theta_{t+1} = theta_t - eta * grad_theta l(theta_t)

and the Eq. (3) four-term objective::

    grad l(theta) = grad { -E_{y+ ~ p_data}[g(x, y+)] + alpha E[g(x, y+)^2]
                          + E_{y- ~ p_theta}[g(x, y-)] + alpha E[g(x, y-)^2] }

Paper hyperparameters (Appendix H.2): eta = 5e-6, batch size 64, 6,000 training
steps, AdamW with weight decay 0.01, maximum generation length 512, temperature
1.0.  The 6,000 steps are interpreted here as *total mini-batch gradient steps*
that may span several outer online-adaptation iterations ``T``.

The black-box LLM is only ever asked for raw text (``prompt``, ``n``,
``temperature``, ``max_len``); no log-probabilities, hidden states or gradients
are requested, and gradients flow exclusively through the small adapter
``g_theta``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper hyperparameters (Appendix H.2 / Section 4.6)
# ---------------------------------------------------------------------------

LEARNING_RATE = 5e-6             # eta
BATCH_SIZE = 64
TRAINING_STEPS = 6000
WEIGHT_DECAY = 0.01              # AdamW
MAX_LEN = 512
TEMPERATURE = 1.0                # BBox-Adapter generation temperature
DEFAULT_T = 4                    # Figure 3(b) shows a plateau from T=3-4 onwards
DEFAULT_M = 5                    # candidates sampled from p_theta_t per query
DEFAULT_K = 5                    # initial candidates prompted from p_LLM per query
DEFAULT_BEAM = 3                 # beam size (Section 4.1 Implementations)
DEFAULT_ALPHA = 1e-2             # regularizer coefficient (not specified; plan default)
ADAM_BETAS = (0.9, 0.999)
WARMUP_FRACTION = 0.1            # 10% linear warmup (not specified in the paper)
GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OnlineAdaptationConfig:
    """Hyper-parameters of Algorithm 1 (mostly Appendix H.2)."""

    # --- Algorithm 1 ---
    n_iterations: int = DEFAULT_T          # T
    n_candidates: int = DEFAULT_M          # M (candidates from p_theta_t)
    k_init: int = DEFAULT_K                # K (initial candidates from p_LLM)
    beam_size: int = DEFAULT_BEAM          # beam size k in Eq. (4)
    samples_per_beam: int = 1              # n sentence samples per beam per step
    max_steps: int = 6                     # L: max sentence-level steps
    alpha: float = DEFAULT_ALPHA           # Eq. (3) regularizer coefficient

    # --- Optimisation (H.2) ---
    lr: float = LEARNING_RATE              # eta = 5e-6
    batch_size: int = BATCH_SIZE           # 64
    max_train_steps: int = TRAINING_STEPS  # 6000
    weight_decay: float = WEIGHT_DECAY     # 0.01
    betas: Tuple[float, float] = ADAM_BETAS
    warmup_steps: int = 0                  # if 0 -> WARMUP_FRACTION * max_train_steps
    max_grad_norm: float = GRAD_CLIP
    schedule: str = "constant"

    # --- Data / model ---
    dataset: Optional[str] = None
    size: Optional[str] = None             # "0.1b" / "0.3b"
    backbone: Optional[str] = None
    sel_mode: str = "ground_truth"         # ground_truth | ai_feedback | combined
    loss_name: str = "nce"                 # nce | mlm (ablation)
    temperature: float = TEMPERATURE
    max_len: int = MAX_LEN
    max_length: int = MAX_LEN              # tokenizer max length
    outcome_supervision: bool = True
    use_beam_search: bool = True           # Eq. (4) adapted inference
    eval_every: int = 0                    # 0 = never evaluate during training
    log_every: int = 50
    save_every: int = 0
    output_dir: Optional[str] = None
    device: Optional[str] = None
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.warmup_steps is None:
            self.warmup_steps = 0
        if isinstance(self.betas, list):
            self.betas = tuple(float(b) for b in self.betas)

    @property
    def effective_warmup(self) -> int:
        if self.warmup_steps:
            return int(self.warmup_steps)
        return int(WARMUP_FRACTION * max(1, self.max_train_steps))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OnlineAdaptationConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in dict(data).items() if k in known}
        # tolerate the plan's config key names
        aliases = {
            "eta": "lr",
            "learning_rate": "lr",
            "T": "n_iterations",
            "iters": "n_iterations",
            "iterations": "n_iterations",
            "M": "n_candidates",
            "K": "k_init",
            "steps": "max_train_steps",
            "train_steps": "max_train_steps",
            "num_training_steps": "max_train_steps",
            "wd": "weight_decay",
            "positives_mode": "sel_mode",
            "positive_mode": "sel_mode",
            "loss": "loss_name",
        }
        for src, dst in aliases.items():
            if src in data and dst not in clean:
                clean[dst] = data[src]
        return cls(**clean)


# ---------------------------------------------------------------------------
# Contrastive dataset
# ---------------------------------------------------------------------------

class ContrastiveSetDataset(Dataset):
    """Groups the current positive/negative buffers into per-query contrastive sets.

    Each item is ``(question, positive_answer, [negative_answers])`` which is
    exactly the contrastive set of Eq. (1)/(2) over which the ranking NCE
    softmax is computed.
    """

    def __init__(
        self,
        questions: Sequence[str],
        positives: Sequence[str],
        negatives: Sequence[Sequence[str]],
        uids: Optional[Sequence[str]] = None,
        answer_types: Optional[Sequence[Optional[str]]] = None,
        choices: Optional[Sequence[Optional[Sequence[str]]]] = None,
    ) -> None:
        self.questions = list(questions)
        self.positives = list(positives)
        self.negatives = [list(n) for n in negatives]
        self.uids = list(uids) if uids is not None else [
            str(i) for i in range(len(self.questions))
        ]
        self.answer_types = list(answer_types) if answer_types is not None else [None] * len(
            self.questions
        )
        self.choices = list(choices) if choices is not None else [None] * len(self.questions)

        if not (len(self.questions) == len(self.positives) == len(self.negatives)):
            raise ValueError(
                "questions/positives/negatives must have equal length "
                f"({len(self.questions)}/{len(self.positives)}/{len(self.negatives)})"
            )

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "uid": self.uids[idx],
            "question": self.questions[idx],
            "positive": self.positives[idx],
            "negatives": self.negatives[idx],
            "answer_type": self.answer_types[idx],
            "choices": self.choices[idx],
        }

    @property
    def usable(self) -> int:
        """#queries with a positive and at least one negative (needed by Eq. 2)."""
        return sum(
            1
            for p, n in zip(self.positives, self.negatives)
            if p is not None and len(n) > 0
        )


def collate_contrastive(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identity collate: keeps ragged negative lists as plain Python lists."""
    return {
        "uid": [b["uid"] for b in batch],
        "question": [b["question"] for b in batch],
        "positive": [b["positive"] for b in batch],
        "negatives": [b["negatives"] for b in batch],
        "answer_type": [b["answer_type"] for b in batch],
        "choices": [b["choices"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OnlineAdapter:
    """Driver of Algorithm 1 (online adaptation of the BBox-Adapter energy ``g_theta``).

    Responsibilities
    ----------------
    1. *Initialization* (Section 3.4): prompt the black-box LLM for ``K`` responses
       per query, pick the positive with ``SEL``, keep the other ``K-1`` as the
       initial negatives.
    2. *Outer loop* ``t = 0 .. T-1``: sample ``M`` candidates from the current
       adapted inference ``p_theta_t`` (Eq. 4 via beam search), refresh positives
       (Eq. 5) and negatives (Eq. 6) using ground truth / AI feedback / combined,
       and apply outcome supervision.
    3. *Adapter update*: accumulate the ranking-NCE objective of Eq. (2)/(3) over
       mini-batches of contrastive sets and apply Eq. (7)
       ``theta_{t+1} = theta_t - eta * grad`` with AdamW (eta = 5e-6, wd = 0.01).
    """

    def __init__(
        self,
        adapter: Any,
        generator: Any = None,
        config: Optional[Any] = None,
        *,
        loss: Any = None,
        buffer: Any = None,
        beam_search: Any = None,
        selector: Any = None,
        logger_: Optional[logging.Logger] = None,
        **config_kwargs: Any,
    ) -> None:
        self.config = _coerce_config(config, config_kwargs)
        self.adapter = adapter
        self.generator = generator
        self.log = logger_ or logger

        self.device = torch.device(
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _to_device(adapter, self.device)
        if hasattr(adapter, "train"):
            adapter.train()

        self.loss_fn = loss if loss is not None else _default_loss(self.config)
        self.buffer = buffer
        self._beam = beam_search
        self._selector = selector

        params = [p for p in _parameters(adapter) if p.requires_grad]
        if not params:
            raise ValueError("adapter exposes no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(self.config.lr),
            betas=tuple(self.config.betas),
            weight_decay=float(self.config.weight_decay),
        )
        total_steps = max(1, int(self.config.max_train_steps))
        warmup = min(int(self.config.effective_warmup), total_steps)
        if self.config.schedule == "cosine":
            sched: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, total_steps - warmup)
            )
        else:
            sched = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: _lr_lambda(step, warmup)
            )
        self.scheduler = (
            torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LambdaLR(
                        self.optimizer, lr_lambda=lambda step: (step + 1) / max(1, warmup)
                    ),
                    sched,
                ],
                milestones=[warmup],
            )
            if warmup > 0
            else sched
        )

        # bookkeeping
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []
        self.energy_history: List[Dict[str, float]] = []
        self.stats: Dict[str, Any] = {
            "global_step": 0,
            "outer_iterations": 0,
            "examples_seen": 0,
            "gradient_steps": 0,
            "sel_calls": 0,
            "n_candidates_sampled": 0,
        }

    # -- construction helpers ------------------------------------------------

    def _beam_search(self) -> Any:
        if self._beam is None:
            from ..inference.beam_search import BeamSearchConfig, SentenceBeamSearch

            cfg = BeamSearchConfig(
                beam_size=int(self.config.beam_size),
                n_samples=int(self.config.samples_per_beam),
                max_steps=int(self.config.max_steps),
                temperature=float(self.config.temperature),
                max_len=int(self.config.max_len),
            )
            self._beam = SentenceBeamSearch(self.adapter, self.generator, cfg)
        return self._beam

    def _selector_fn(self) -> Callable[..., int]:
        from .buffers import make_selector

        return make_selector(self.config.sel_mode, config=None)

    # ------------------------------------------------------------------
    # Initialization (Section 3.4, "Initialization")
    # ------------------------------------------------------------------
    def initialize(
        self,
        questions: Sequence[str],
        *,
        prompts: Optional[Sequence[str]] = None,
        golds: Optional[Sequence[Any]] = None,
        answer_types: Optional[Sequence[Optional[str]]] = None,
        choices_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
        uids: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
        max_queries: Optional[int] = None,
    ) -> Any:
        """Prompt the black-box LLM for ``K`` responses/query and build t=0 buffers."""
        from .buffers import BufferConfig, QuerySamples, SampleBuffer, candidate_key, deduplicate

        if self.generator is None:
            raise ValueError("initialization requires a black-box generator")
        from ..llm.prompts import build_prompt

        k = int(k or self.config.k_init)
        selector = self._selector_fn()
        qs = list(questions)
        if max_queries is not None:
            qs = qs[: int(max_queries)]
        if prompts is None:
            prompts = [
                build_prompt(q, dataset=self.config.dataset)
                if self.config.dataset
                else q
                for q in qs
            ]

        samples = SampleBuffer()
        for i, (q, prompt) in enumerate(zip(qs, prompts)):
            gold = _at(golds, i)
            atype = _at(answer_types, i)
            choices = _at(choices_list, i)
            gen = self.generator.generate_result(
                prompt,
                n=k,
                temperature=uniform_temperature(self.config.temperature),
                max_len=int(self.config.max_len),
            )
            cands = [t for t in (gen.texts or []) if t]
            cands = deduplicate([c.strip() for c in cands if c and c.strip()])
            uid = _at(uids, i) or str(i)
            if not cands:
                samples.records[uid] = QuerySamples(
                    uid=uid, question=q, gold=gold, answer_type=atype, meta={}
                )
                continue
            idx, texts = selector(
                cands,
                gold=gold,
                answer_type=atype,
                choices=choices,
                question=q,
                return_index=True,
            ) if _wants_return_index(selector) else (selector(
                cands, gold=gold, answer_type=atype, choices=choices, question=q
            ), cands)
            idx = int(idx) if idx is not None and 0 <= int(idx) < len(texts) else 0
            rec = QuerySamples(uid=uid, question=q, gold=gold, answer_type=atype, meta={})
            rec.n_init = len(cands)
            for j, c in enumerate(texts):
                if j == idx:
                    rec.add_positive(c)
            for j, c in enumerate(texts):
                if j != idx:
                    rec.add_negative(c)
            samples.records[uid] = rec
            self.stats["sel_calls"] += 1
        self.buffer = samples
        return samples

    # ------------------------------------------------------------------
    # Step 1: sampling from the adapted inference (Eq. 4)
    # ------------------------------------------------------------------
    def sample_candidates(
        self,
        question: str,
        prompt: Optional[str] = None,
        *,
        m: Optional[int] = None,
        return_details: bool = False,
    ) -> Any:
        """``{y_hat_m}_{m=1}^M ~ p_theta_t(y | x)`` (Eq. 1), using the adapted inference."""
        m = int(m or self.config.n_candidates)
        if self.generator is None:
            raise ValueError("candidate sampling requires a black-box generator")
        bs = self._beam_search()
        from ..llm.prompts import build_prompt

        if prompt is None:
            prompt = (
                build_prompt(question, dataset=self.config.dataset)
                if self.config.dataset
                else question
            )
        if self.config.use_beam_search:
            res = bs.sample_candidates(question, prompt=prompt, n=m, return_result=return_details)
        else:
            from ..inference.beam_search import single_step_rank

            gen = self.generator.generate_result(
                prompt,
                n=m,
                temperature=float(self.config.temperature),
                max_len=int(self.config.max_len),
            )
            texts = [t for t in (gen.texts or []) if t and t.strip()]
            self.stats["n_candidates_sampled"] += len(texts)
            if return_details:
                ranked, res = single_step_rank(
                    self.adapter, texts, question, return_result=True
                )
                return [ranked], res
            ranked = single_step_rank(self.adapter, texts, question)
            return [ranked] if isinstance(ranked, str) else list(ranked)
        self.stats["n_candidates_sampled"] += 1
        return res

    # ------------------------------------------------------------------
    # Steps 2-3: buffer refresh (Eq. 5, Eq. 6) + outcome supervision
    # ------------------------------------------------------------------
    def refresh_samples(
        self,
        questions: Sequence[str],
        *,
        prompts: Optional[Sequence[str]] = None,
        golds: Optional[Sequence[Any]] = None,
        answer_types: Optional[Sequence[Optional[str]]] = None,
        choices_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
        uids: Optional[Sequence[str]] = None,
        m: Optional[int] = None,
        progress: bool = False,
    ) -> Any:
        """One pass of the Algorithm-1 inner loop over the training set."""
        from .buffers import update_query_samples
        from ..llm.prompts import build_prompt

        if self.buffer is None:
            raise ValueError("call initialize() before refresh_samples()")

        qs = list(questions)
        selector = self._selector_fn()
        for i, q in enumerate(qs):
            uid = _at(uids, i) or str(i)
            rec = self.buffer.get(uid) if hasattr(self.buffer, "get") else self.buffer.records.get(uid)
            if rec is None:
                continue
            prompt = (
                _at(prompts, i)
                or (build_prompt(q, dataset=self.config.dataset) if self.config.dataset else q)
            )
            try:
                sampled = self.sample_candidates(q, prompt=prompt, m=m)
            except Exception as exc:  # pragma: no cover - defensive
                self.log.warning("candidate sampling failed for uid=%s: %s", uid, exc)
                continue
            if not isinstance(sampled, (list, tuple)):
                sampled = [sampled]
            cands = [str(s) for s in sampled if s]
            update_query_samples(
                rec,
                cands,
                selector=selector,
                config=None,
                question=q,
                outcome_candidates=cands if self.config.outcome_supervision else None,
            )
            self.stats["sel_calls"] += 1
            if progress and (i + 1) % 50 == 0:
                self.log.info("  inner loop %d/%d queries", i + 1, len(qs))
        return self.buffer

    # ------------------------------------------------------------------
    # Step 4: adapter update (Eq. 3 -> Eq. 7)
    # ------------------------------------------------------------------
    def contrastive_loader(self, batch_size: Optional[int] = None) -> Optional[DataLoader]:
        if self.buffer is None:
            return None
        questions, positives, negatives = self.buffer.contrastive_sets(
            require_positive=True, require_negative=True
        )
        if not questions:
            return None
        uids = [getattr(r, "uid", str(i)) for i, r in enumerate(self.buffer.records.values())]
        ds = ContrastiveSetDataset(
            questions, positives, negatives, uids=uids[: len(questions)]
        )
        return DataLoader(
            ds,
            batch_size=int(batch_size or self.config.batch_size),
            shuffle=True,
            collate_fn=collate_contrastive,
            drop_last=False,
        )

    def adapter_step(self, batch: Dict[str, Any], *, update: bool = True) -> Dict[str, float]:
        """One mini-batch gradient step of Eq. (7) with the Eq. (2)/(3) objective."""
        questions = batch["question"]
        positives = batch["positive"]
        negatives = batch["negatives"]

        pos_e, neg_e, mask = self._score(questions, positives, negatives)
        if pos_e.numel() == 0 or float(mask.sum()) <= 0:
            return {"loss": 0.0, "nce": 0.0, "reg": 0.0, "pos_energy": 0.0, "neg_energy": 0.0}

        loss, stats = self._loss(pos_e, neg_e, mask)

        if update:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.max_grad_norm and self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    float(self.config.max_grad_norm),
                )
            self.optimizer.step()
            try:
                self.scheduler.step()
            except Exception:  # pragma: no cover
                pass
            self.global_step += 1
            self.stats["gradient_steps"] = self.global_step
            self.stats["global_step"] = self.global_step
            self.stats["examples_seen"] += len(questions)

        out = {
            "loss": float(loss.detach()),
            "pos_energy": float(pos_e.detach().mean()),
            "neg_energy": float(neg_e.detach()[mask].mean()) if mask.any() else 0.0,
            "lr": float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else 0.0,
            "step": self.global_step,
            "n_queries": len(questions),
            "n_negatives": int(mask.sum()),
        }
        out.update(stats)
        return out

    def _score(
        self, questions: Sequence[str], positives: Sequence[str], negatives: Sequence[Sequence[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score all (question, candidate) pairs, preserving the autograd graph."""
        from ..losses.nce import build_contrastive_tensors, score_contrastive_sets

        pos_e, neg_e, mask = score_contrastive_sets(
            self.adapter,
            questions,
            positives,
            negatives,
            batch_size=int(self.config.batch_size),
            max_length=int(self.config.max_length),
        )
        return pos_e, neg_e, mask

    def _loss(self, pos_e: torch.Tensor, neg_e: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Eq. (2) ranking NCE + Eq. (3) ``alpha * E[g^2]`` regularizer."""
        fn = self.loss_fn
        if hasattr(fn, "forward") and not isinstance(fn, torch.nn.Module):
            pass
        try:
            result = fn(
                positive_energies=pos_e,
                negative_energies=neg_e,
                mask=mask,
                return_stats=True,
            )
        except TypeError:
            result = fn(positive_energies=pos_e, negative_energies=neg_e, mask=mask)
        if isinstance(result, tuple):
            loss, stats = result
            return loss, {k: float(v) for k, v in (stats or {}).items() if _is_number(v)}
        return result, {}

    def fit_on_buffer(self, *, max_steps: Optional[int] = None, log_every: Optional[int] = None) -> Dict[str, Any]:
        """Run mini-batch gradient steps on the *current* buffers until the budget is met."""
        max_steps = int(max_steps if max_steps is not None else self.config.max_train_steps)
        log_every = int(log_every if log_every is not None else self.config.log_every)
        steps = 0
        accum = _Accumulator()
        while self.global_step < max_steps:
            loader = self.contrastive_loader()
            if loader is None or len(loader) == 0:
                break
            for batch in loader:
                if self.global_step >= max_steps:
                    break
                stats = self.adapter_step(batch, update=True)
                accum.update(stats)
                steps += 1
                if log_every and self.global_step % log_every == 0:
                    self.log.info(
                        "[step %d] loss=%.4f nce=%.4f reg=%.5f E+=%.4f E-=%.4f lr=%.2e",
                        self.global_step,
                        stats.get("loss", 0.0),
                        stats.get("nce", stats.get("loss", 0.0)),
                        stats.get("reg", 0.0),
                        stats.get("pos_energy", 0.0),
                        stats.get("neg_energy", 0.0),
                        stats.get("lr", 0.0),
                    )
                    self.energy_history.append({**stats})
        summary = accum.summary()
        summary["steps_run"] = steps
        summary["global_step"] = self.global_step
        return summary

    # ------------------------------------------------------------------
    # Top level: Algorithm 1
    # ------------------------------------------------------------------
    def run(
        self,
        questions: Sequence[str],
        *,
        prompts: Optional[Sequence[str]] = None,
        golds: Optional[Sequence[Any]] = None,
        answer_types: Optional[Sequence[Optional[str]]] = None,
        choices_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
        uids: Optional[Sequence[str]] = None,
        eval_fn: Optional[Callable[[Any, int], float]] = None,
        max_queries: Optional[int] = None,
        t: Optional[int] = None,
        m: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full Algorithm 1: ``T`` outer iterations with inner sampling/buffer refresh.

        ``eval_fn(adapter, t) -> float`` is called after every outer iteration when
        provided (used to reproduce Figure 3(b): T = 0, 1, 2, 3, 4).
        """
        T = int(t if t is not None else self.config.n_iterations)
        qs = list(questions)
        starts: List[Dict[str, Any]] = []

        if eval_fn is not None:
            score = eval_fn(self.adapter, 0)
            starts.append({"t": 0, "score": float(score), "stage": "untrained"})
            self.log.info("[t=0] untrained adapter score=%.4f", score)

        if self.buffer is None:
            self.initialize(
                qs,
                prompts=prompts,
                golds=golds,
                answer_types=answer_types,
                choices_list=choices_list,
                uids=uids,
                k=k,
                max_queries=max_queries,
            )

        for it in range(T):
            t0 = time.time()
            train_qs = qs[: int(max_queries)] if max_queries else qs
            self.refresh_samples(
                train_qs,
                prompts=prompts[: len(train_qs)] if prompts else None,
                golds=golds[: len(train_qs)] if golds else None,
                answer_types=answer_types[: len(train_qs)] if answer_types else None,
                choices_list=choices_list[: len(train_qs)] if choices_list else None,
                uids=uids[: len(train_qs)] if uids else None,
                m=m,
            )
            fit_stats = self.fit_on_buffer(
                max_steps=_per_iteration_steps(self.config, T, it)
            )
            self.stats["outer_iterations"] = it + 1
            record = {
                "t": it + 1,
                "seconds": time.time() - t0,
                "buffer_stats": self.buffer.stats() if hasattr(self.buffer, "stats") else {},
                "fit": fit_stats,
                "params": self.param_norms(),
            }
            self.history.append(record)
            self.log.info(
                "[t=%d] refreshed buffers (%s) and ran %d updates",
                it + 1,
                record["buffer_stats"],
                fit_stats.get("steps_run", 0),
            )
            if eval_fn is not None:
                score = eval_fn(self.adapter, it + 1)
                starts.append({"t": it + 1, "score": float(score)})
                self.log.info("[t=%d] score=%.4f", it + 1, score)
            if self.config.save_every and self.config.output_dir and (it + 1) % self.config.save_every == 0:
                self.save(os.path.join(self.config.output_dir, f"adapter_t{it + 1}"))

        return {
            "T": T,
            "global_step": self.global_step,
            "score_curve": starts,
            "history": self.history,
            "stats": self.stats,
            "energy_history": self.energy_history,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def param_norms(self) -> Dict[str, float]:
        total = 0.0
        loss_norm = 0.0
        head = 0.0
        for name, p in _named_parameters(self.adapter):
            n = float(p.detach().norm())
            total += n * n
            if "head" in name or "classifier" in name:
                head += n * n
            else:
                loss_norm += n * n
        return {
            "total_norm": math.sqrt(total),
            "backbone_norm": math.sqrt(loss_norm),
            "head_norm": math.sqrt(head),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if hasattr(self.adapter, "save_pretrained"):
            self.adapter.save_pretrained(path)
        if self.buffer is not None and hasattr(self.buffer, "to_dict"):
            with open(os.path.join(path, "buffer.json"), "w", encoding="utf-8") as fh:
                json.dump(self.buffer.to_dict(), fh, indent=2)
        with open(os.path.join(path, "adaptation_config.json"), "w", encoding="utf-8") as fh:
            json.dump(self.config.to_dict(), fh, indent=2, default=str)
        with open(os.path.join(path, "train_history.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {"history": self.history, "energy_history": self.energy_history, "stats": self.stats},
                fh,
                indent=2,
                default=str,
            )

    def load_buffer(self, path: str) -> Any:
        from .buffers import SampleBuffer

        with open(os.path.join(path, "buffer.json"), "r", encoding="utf-8") as fh:
            self.buffer = SampleBuffer.from_dict(json.load(fh))
        return self.buffer


# ---------------------------------------------------------------------------
# Functional entry points
# ---------------------------------------------------------------------------

def run_online_adaptation(
    adapter: Any,
    generator: Any,
    questions: Sequence[str],
    *,
    prompts: Optional[Sequence[str]] = None,
    golds: Optional[Sequence[Any]] = None,
    answer_types: Optional[Sequence[Optional[str]]] = None,
    choices_list: Optional[Sequence[Optional[Sequence[str]]]] = None,
    config: Optional[Any] = None,
    eval_fn: Optional[Callable[[Any, int], float]] = None,
    **config_kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper running Algorithm 1 end-to-end."""
    trainer = OnlineAdapter(adapter, generator, config, **config_kwargs)
    return trainer.run(
        questions,
        prompts=prompts,
        golds=golds,
        answer_types=answer_types,
        choices_list=choices_list,
        eval_fn=eval_fn,
    )


def train_adapter(
    adapter: Any,
    questions: Sequence[str],
    positives: Sequence[str],
    negatives: Sequence[Sequence[str]],
    *,
    config: Optional[Any] = None,
    steps: Optional[int] = None,
    eval_fn: Optional[Callable[[Any, int], float]] = None,
    **config_kwargs: Any,
) -> Dict[str, Any]:
    """Offline variant: train ``g_theta`` directly on given contrastive sets.

    Useful for smoke tests and for the alpha sweep, since it needs no black-box
    LLM: it simply iterates the Eq. (2)/(3) objective over fixed buffers.
    """
    from .buffers import QuerySamples, SampleBuffer

    trainer = OnlineAdapter(adapter, None, config, **config_kwargs)
    buf = SampleBuffer()
    for i, (q, p, ns) in enumerate(zip(questions, positives, negatives)):
        rec = QuerySamples(uid=str(i), question=q, gold=None, answer_type=None)
        rec.add_positive(p)
        for n in ns:
            rec.add_negative(n)
        buf.records[str(i)] = rec
    trainer.buffer = buf
    summary = trainer.fit_on_buffer(max_steps=steps or trainer.config.max_train_steps)
    return {
        "summary": summary,
        "global_step": trainer.global_step,
        "energy_history": trainer.energy_history,
        "score_curve": (
            [{"t": 0, "score": float(eval_fn(adapter, 0))}] if eval_fn is not None else []
        ),
    }


def uniform_temperature(temperature: float) -> float:
    """Clamp to the (0, 2] range accepted by the Azure/HF backends."""
    try:
        t = float(temperature)
    except (TypeError, ValueError):
        return TEMPERATURE
    return min(max(t, 1e-3), 2.0)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

class _Accumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, stats: Dict[str, float]) -> None:
        for k, v in (stats or {}).items():
            if not _is_number(v):
                continue
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0) + 1

    def summary(self) -> Dict[str, float]:
        return {k: self.sums[k] / max(1, self.counts[k]) for k in self.sums}


def _lr_lambda(step: int, warmup: int) -> float:
    if warmup and step < warmup:
        return float(step + 1) / float(warmup)
    return 1.0


def _per_iteration_steps(config: OnlineAdaptationConfig, T: int, it: int) -> int:
    """Split the 6,000-step budget across the ``T`` outer iterations (§H.2)."""
    total = int(config.max_train_steps)
    if T <= 0:
        return total
    per = total / float(T)
    # give the remainder to the last iteration so that exactly `total` steps run
    if it == T - 1:
        return max(1, total - int(per) * (T - 1))
    return max(1, int(per))


def _coerce_config(config: Any, kwargs: Dict[str, Any]) -> OnlineAdaptationConfig:
    if config is None:
        return OnlineAdaptationConfig.from_dict(kwargs or None)
    if isinstance(config, OnlineAdaptationConfig):
        return config
    if isinstance(config, dict):
        merged = dict(config)
        merged.update(kwargs or {})
        return OnlineAdaptationConfig.from_dict(merged)
    # duck-typed config object (e.g. parsed YAML namespace)
    data = {}
    for field_name in OnlineAdaptationConfig.__dataclass_fields__:  # type: ignore[attr-defined]
        if hasattr(config, field_name):
            data[field_name] = getattr(config, field_name)
    data.update(kwargs or {})
    return OnlineAdaptationConfig.from_dict(data)


def _default_loss(config: OnlineAdaptationConfig) -> Any:
    from ..losses import get_loss

    name = "mlm" if str(config.loss_name).lower().startswith("mlm") else "nce"
    if name == "nce":
        from ..losses.nce import NCELossConfig, RankingNCELoss

        return RankingNCELoss(NCELossConfig(alpha=config.alpha))
    return get_loss(name)


def _parameters(module: Any) -> List[torch.Tensor]:
    if hasattr(module, "parameters"):
        return list(module.parameters())
    if isinstance(module, dict):
        return [p for p in module.values() if isinstance(p, torch.Tensor)]
    return []


def _named_parameters(module: Any) -> Iterator[Tuple[str, torch.Tensor]]:
    if hasattr(module, "named_parameters"):
        yield from module.named_parameters()
    elif isinstance(module, torch.nn.Module):
        yield from module.named_parameters()


def _to_device(module: Any, device: torch.device) -> None:
    if hasattr(module, "to"):
        try:
            module.to(device)
        except Exception:  # pragma: no cover
            pass


def _at(seq: Optional[Sequence[Any]], i: int) -> Any:
    if seq is None:
        return None
    try:
        return seq[i]
    except (IndexError, TypeError):
        return None


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _wants_return_index(fn: Any) -> bool:
    """Detect whether a selector accepts ``return_index=True``."""
    try:
        import inspect

        sig = inspect.signature(fn)
        return "return_index" in sig.parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# Self test (offline; no network, no HuggingFace download)
# ---------------------------------------------------------------------------

def _self_test() -> Dict[str, Any]:
    """Validate Algorithm 1 mechanics with a tiny linear stand-in adapter."""
    torch.manual_seed(0)

    class TinyAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.zeros(1))

        def score_pairs(self, questions, answers, batch_size=64, max_length=None):
            vals = []
            for q, a in zip(questions, answers):
                vals.append(len(str(a)) * 0.01 + self.scale)
            return torch.stack([v.reshape(()) for v in vals])

        def energy(self, questions, answers, **kwargs):
            return self.score_pairs(questions, answers).detach()

    adapter = TinyAdapter()
    cfg = OnlineAdaptationConfig(
        max_train_steps=8,
        batch_size=2,
        log_every=0,
        warmup_steps=2,
        n_iterations=2,
    )
    trainer = OnlineAdapter(adapter, None, cfg)

    questions = ["q1", "q2", "q3", "q4"]
    positives = ["the good long answer"] * 4
    negatives = [["bad", "worse2", "bad3"]] * 4

    # --- Eq. (1)/(2) softmax over the contrastive set ---
    from ..losses.nce import compute_nce_loss

    pos_e = torch.tensor([2.0])
    neg_e = torch.tensor([[0.5, 0.1]])
    mask = torch.ones_like(neg_e, dtype=torch.bool)
    nce = compute_nce_loss(pos_e, neg_e, mask=mask)
    assert nce.item() > 0, nce
    assert abs(float(nce) - float(torch.log(torch.tensor(1.0 + math.exp(0.5 - 2.0) + math.exp(0.1 - 2.0))))) < 1e-5

    # --- gradient sign structure of Eq. (3) ---
    pos = torch.tensor([1.0, 2.0], requires_grad=True)
    negs = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    m = torch.ones_like(negs, dtype=torch.bool)
    loss = compute_nce_loss(pos, negs, mask=m)
    loss.backward()
    assert (pos.grad < 0).all(), pos.grad          # -E[g(x, y+)] term
    assert (negs.grad > 0).all(), negs.grad        # +E[g(x, y-)] term

    # --- fit_on_buffer runs the requested number of gradient steps ---
    from .buffers import QuerySamples, SampleBuffer

    buf = SampleBuffer()
    for i, q in enumerate(questions):
        rec = QuerySamples(uid=str(i), question=q, gold=None, answer_type=None)
        rec.add_positive(positives[i])
        for n in negatives[i]:
            rec.add_negative(n)
        buf.records[str(i)] = rec
    trainer.buffer = buf
    summary = trainer.fit_on_buffer(max_steps=8)
    assert trainer.global_step == 8, trainer.global_step
    assert "loss" in summary

    # --- Eq. (7): a gradient step changes theta ---
    before = adapter.scale.detach().clone()
    assert not torch.allclose(before, adapter.scale.detach())

    # --- Eq. (5)/(6) refresh with a mock black-box generator ---
    class MockGen:
        def generate_result(self, prompt, n=1, temperature=1.0, max_len=512, **kw):
            class R:
                texts = [f"{prompt} answer {j} #### {j}" for j in range(n)]
            return R()

    trainer2 = OnlineAdapter(TinyAdapter(), MockGen(), OnlineAdaptationConfig(
        n_iterations=1, max_train_steps=2, batch_size=2, log_every=0, k_init=3,
        n_candidates=3, use_beam_search=False,
    ))
    trainer2.initialize(questions, golds=["the good long answer"] * 4, k=3)
    assert trainer2.buffer.n_queries == 4
    assert all(rec.y_plus for rec in trainer2.buffer.records.values())
    assert all(rec.y_minus for rec in trainer2.buffer.records.values())
    trainer2.refresh_samples(questions, golds=["the good long answer"] * 4)
    out = trainer2.run(questions, golds=["the good long answer"] * 4, max_queries=2, t=1)
    assert out["stats"]["gradient_steps"] >= 1

    # --- per-iteration step budget respects the 6,000-step default ---
    c = OnlineAdaptationConfig(max_train_steps=6000, n_iterations=4)
    total = sum(_per_iteration_steps(c, 4, i) for i in range(4))
    assert total == 6000, total

    # --- learning rate is applied (Appendix H.2) ---
    assert OnlineAdaptationConfig().lr == 5e-6
    assert OnlineAdaptationConfig().batch_size == 64
    assert OnlineAdaptationConfig().weight_decay == 0.01
    assert OnlineAdaptationConfig().max_train_steps == 6000

    return {
        "nce_value": float(nce),
        "global_step": trainer.global_step,
        "buffer_queries": trainer2.buffer.n_queries,
        "score_curve_len": len(out["score_curve"]),
        "steps_per_iteration": [_per_iteration_steps(c, 4, i) for i in range(4)],
    }


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(_self_test(), indent=2, default=str))


__all__ = [
    "LEARNING_RATE",
    "BATCH_SIZE",
    "TRAINING_STEPS",
    "WEIGHT_DECAY",
    "TEMPERATURE",
    "DEFAULT_T",
    "DEFAULT_M",
    "DEFAULT_K",
    "DEFAULT_BEAM",
    "OnlineAdaptationConfig",
    "ContrastiveSetDataset",
    "collate_contrastive",
    "OnlineAdapter",
    "run_online_adaptation",
    "train_adapter",
    "uniform_temperature",
]
