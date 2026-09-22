"""Sentence-level beam search for BBox-Adapter adapted inference (Section 3.3).

The black-box LLM is treated purely as a *proposal generator* and the trained scalar
energy adapter ``g_theta`` is treated purely as an *evaluator*.  The adapted inference
``p_theta(y | x)`` factorizes autoregressively (Eq. (1) of Section 3.3)::

    p_theta(y | x) = p_theta(s^{1:L} | x)
                   = p_LLM(s^{1:L} | x) * exp(g_theta(s^{1:L}, x))
                   = exp(g_theta(s^{1:L}, x)) * prod_l p_LLM(s^l | x, s^{1:l-1})

Because BBox-Adapter only ever receives *text* from the black-box LLM (no hidden
states, no output log-probabilities), the proposal term ``p_LLM(s^l | x, s^{1:l-1})``
is realized as *sampling* -- the LLM is asked for ``n`` continuations per beam.  The
adapter supplies the only tractable score, ``g_theta(s^{1:l}, x)``, which is used for
top-k pruning of the beam options.

Beam search loop (Section 3.3, verbatim behaviour):

  * beam size ``k``;
  * at each step ``l`` draw ``n`` samples of ``s^l`` from
    ``p_LLM(s^l | x, s^{1:l-1})`` *for each beam*, giving ``n * k`` candidate chain
    hypotheses of ``s^{1:l}`` -- the candidate set ``C``;
  * score every candidate with ``g_theta(s^{1:l}, x)`` and keep the top-``k`` beams
    ("effectively pruning the beam options");
  * stop once a pre-defined number of ``L`` iterations is reached *or* all beams
    encounter a stop signal;
  * the adapted generation is the highest scoring option according to the adapter.

A cheaper *single-step* variant (Section 4.4, "single-step inference variant") asks the
black-box model for complete answers once and lets the adapter rank them, which requires
far fewer black-box API calls and therefore lowers the inference cost reported in Table 4.

No logprob / hidden-state / gradient access to the black-box LLM happens anywhere in this
module; :meth:`SentenceBeamSearch.run` asserts the generator interface it uses is
text-only.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # torch is needed only for adapter scoring; keep the module importable without it
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from ..data.answer_extraction import (
    ANSWER_TERMINATOR,
    contains_terminator,
    extract_final_answer,
    split_steps,
)

__all__ = [
    "STOP_SIGNAL",
    "DEFAULT_MAX_STEPS",
    "BeamSearchConfig",
    "Hypothesis",
    "BeamSearchResult",
    "SentenceBeamSearch",
    "beam_search",
    "single_step_search",
    "single_step_rank",
    "topk_indices",
    "score_candidates",
    "resolve_adapter_scores",
]

STOP_SIGNAL = ANSWER_TERMINATOR  # "####"; matches the Appendix-J prompt terminators
DEFAULT_MAX_STEPS = 6


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class BeamSearchConfig:
    """Hyper-parameters of the adapted (sentence-level beam-search) inference.

    Defaults follow Section 4.1 ("The number of beams used for training and inference is
    set as 3 by default") and Appendix H.2 (max length 512, temperature 1.0).
    """

    beam_size: int = 3                 # k
    n_samples: int = 1                 # n samples of s^l per beam per step
    max_steps: int = DEFAULT_MAX_STEPS  # L: number of sentence-level iterations
    stop_signal: str = STOP_SIGNAL
    temperature: float = 1.0
    max_len: int = 512
    normalization: str = "none"        # "none" | "length" | "sentences"
    length_penalty: float = 0.0        # optional additive penalty * #sentences
    deduplicate: bool = True
    stop_when_all_complete: bool = True
    min_step_tokens: int = 1
    step_instruction: str = (
        "Continue with the next step. Write exactly one sentence. "
        "Do not repeat the previous sentences."
    )
    continuation_sep: str = "\n"
    seed: Optional[int] = None
    include_prompt_in_energy: bool = False
    keep_candidates: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.normalization not in {"none", "length", "sentences"}:
            raise ValueError(
                f"normalization must be one of 'none','length','sentences', "
                f"got {self.normalization!r}"
            )
        if self.beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "BeamSearchConfig":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = dict(d.get("extra") or {})
        kwargs = {k: v for k, v in d.items() if k in known}
        cfg = cls(**kwargs)
        for k, v in d.items():
            if k not in known:
                extra[k] = v
        cfg.extra = extra
        return cfg


# --------------------------------------------------------------------------------------
# Data holders
# --------------------------------------------------------------------------------------
@dataclass
class Hypothesis:
    """A partial solution chain ``s^{1:l}`` together with its adapter score."""

    sentences: List[str] = field(default_factory=list)
    score: float = float("-inf")
    complete: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.strip() for s in self.sentences if s and s.strip())

    @property
    def n_sentences(self) -> int:
        return len([s for s in self.sentences if s and s.strip()])

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def append(self, sentence: str, score: Optional[float] = None) -> "Hypothesis":
        new = Hypothesis(
            sentences=list(self.sentences) + [sentence],
            score=self.score if score is None else score,
            complete=contains_terminator(sentence),
            meta=dict(self.meta),
        )
        return new

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["text"] = self.text
        return d


@dataclass
class BeamSearchResult:
    """Result of one adapted inference call."""

    question: str = ""
    best_text: str = ""
    best_score: float = float("-inf")
    best_hypothesis: Optional[Hypothesis] = None
    beams: List[Hypothesis] = field(default_factory=list)
    candidates: List[Hypothesis] = field(default_factory=list)
    n_llm_calls: int = 0
    n_candidates: int = 0
    steps_used: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    energy_history: List[Dict[str, float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "best_text": self.best_text,
            "best_score": self.best_score,
            "beams": [b.to_dict() for b in self.beams],
            "n_llm_calls": self.n_llm_calls,
            "n_candidates": self.n_candidates,
            "steps_used": self.steps_used,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "energy_history": list(self.energy_history),
            "meta": dict(self.meta),
        }


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def topk_indices(scores: Sequence[float], k: int) -> List[int]:
    """Indices of the ``k`` highest scores (stable, descending).

    Ties are broken by list order so that repeated runs are deterministic.
    """
    if k <= 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return order[: min(k, len(scores))]


def _as_list_of_texts(output: Any) -> List[str]:
    """Normalize whatever the black-box client returns into a list of strings."""
    if output is None:
        return []
    if isinstance(output, str):
        return [output]
    if isinstance(output, (list, tuple)):
        texts: List[str] = []
        for item in output:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "generation", "content", "answer", "message"):
                    if key in item and isinstance(item[key], str):
                        texts.append(item[key])
                        break
            elif hasattr(item, "text") and isinstance(getattr(item, "text"), str):
                texts.append(item.text)
        return texts
    if isinstance(output, dict):
        return _as_list_of_texts([output])
    return [str(output)]


def _first_sentence(text: str) -> str:
    """First sentence of a possibly multi-sentence continuation."""
    steps = split_steps(text)
    if steps:
        return steps[0]
    stripped = (text or "").strip()
    if not stripped:
        return ""
    for sep in ("\n", ". ", "? ", "! "):
        if sep in stripped:
            idx = stripped.find(sep)
            if sep == "\n":
                head = stripped[:idx]
            else:
                head = stripped[: idx + 1]
            head = head.strip()
            if head:
                return head
    return stripped


def _normalized_score(
    raw_score: float,
    n_sentences: int,
    n_chars: int,
    config: BeamSearchConfig,
    token_len: Optional[int] = None,
) -> float:
    """Apply the (optional) score normalization across beams.

    Section 3.3 scores a hypothesis simply with ``g_theta(s^{1:l}, x)``; the paper does
    not specify length normalization, so ``normalization="none"`` is the faithful default.
    """
    if config.normalization == "none":
        base = raw_score
    elif config.normalization == "sentences":
        base = raw_score / max(1, n_sentences)
    else:  # "length"
        denom = token_len if token_len is not None else max(1, n_chars)
        base = raw_score / max(1.0, float(denom))
    if config.length_penalty:
        base = base - config.length_penalty * n_sentences
    return float(base)


def resolve_adapter_scores(
    adapter: Any,
    question: str,
    texts: Sequence[str],
    *,
    max_length: Optional[int] = None,
) -> List[float]:
    """Score ``texts`` as candidate answers to ``question`` with the energy adapter.

    ``adapter`` may be

    * an :class:`~bbox_adapter.adapter.energy_model.EnergyModel` (uses
      ``score_pairs`` / ``energy``),
    * any object exposing ``score_pairs(pairs)``, ``energy(qs, ans)`` or
      ``score_batch(qs, ans)`` (e.g. the MLM ablation scorer), or
    * a plain callable ``f(question, texts) -> Sequence[float]`` (used by tests).

    Lower energy is better *when the callable is an energy model*; the sign convention is
    unified here so that **larger is always better** for the returned scores.
    """
    texts = list(texts)
    if not texts:
        return []

    # 1) plain callable adapter (question, texts) -> scores
    if callable(adapter) and not hasattr(adapter, "score_pairs") and not hasattr(adapter, "energy"):
        out = adapter(question, texts)
        return [float(x) for x in _as_score_sequence(out)]

    # 2) object with score_pairs / energy / score_batch
    pairs = [(question, t) for t in texts]
    out = None
    if hasattr(adapter, "score_pairs"):
        try:
            out = adapter.score_pairs(pairs, max_length=max_length)
        except TypeError:
            try:
                out = adapter.score_pairs(pairs)
            except TypeError:
                out = adapter.score_pairs([f"{q}\n{a}" for q, a in pairs])
    elif hasattr(adapter, "score_batch"):
        try:
            out = adapter.score_batch([question] * len(texts), texts, max_length=max_length)
        except TypeError:
            out = adapter.score_batch([question] * len(texts), texts)
    elif hasattr(adapter, "energy"):
        try:
            out = adapter.energy([question] * len(texts), texts, max_length=max_length)
        except TypeError:
            out = adapter.energy([question] * len(texts), texts)
    else:  # last resort: call with (question, texts)
        out = adapter(question, texts)

    scores = [float(x) for x in _as_score_sequence(out)]

    # Energy models are "lower is better"; detect the convention by named attribute and
    # convert so that the beam search always maximizes.
    if hasattr(adapter, "energy") or hasattr(adapter, "score_pairs"):
        negate = bool(getattr(adapter, "lower_energy_is_better", True))
        if negate:
            scores = [-s for s in scores]
    return scores


def _as_score_sequence(out: Any) -> List[float]:
    if torch is not None and torch is not None and isinstance(out, torch.Tensor):  # type: ignore
        out = out.detach().cpu().reshape(-1).tolist()
    if isinstance(out, (int, float)):
        return [float(out)]
    if isinstance(out, dict):
        for key in ("scores", "energy", "energies", "logprob", "logprobs"):
            if key in out:
                return _as_score_sequence(out[key])
        raise ValueError(f"cannot interpret adapter output dict keys: {list(out)}")
    result: List[float] = []
    for item in out:  # type: ignore[union-attr]
        if torch is not None and isinstance(item, torch.Tensor):  # type: ignore
            item = item.detach().cpu().reshape(-1).tolist()
            result.extend(float(x) for x in item)
        else:
            result.append(float(item))
    return result


def score_candidates(
    adapter: Any,
    question: str,
    hypotheses: Sequence[Hypothesis],
    config: Optional[BeamSearchConfig] = None,
    *,
    max_length: Optional[int] = None,
) -> List[float]:
    """Score a batch of candidate hypotheses with the adapter; returns raw scores."""
    cfg = config or BeamSearchConfig()
    texts = [_hypothesis_energy_text(h, question if cfg.include_prompt_in_energy else None) for h in hypotheses]
    raw = resolve_adapter_scores(adapter, question, texts, max_length=max_length)
    if len(raw) != len(hypotheses):
        raise RuntimeError(
            f"adapter returned {len(raw)} scores for {len(hypotheses)} candidates"
        )
    return raw


def _hypothesis_energy_text(h: Hypothesis, question: Optional[str]) -> str:
    """Text passed to ``g_theta`` for a candidate chain ``s^{1:l}``.

    The adapter was trained on ``(question, answer)`` pairs, so the question is *not*
    part of the answer text by default (``include_prompt_in_energy=False``).
    """
    return h.text


# --------------------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------------------
class SentenceBeamSearch:
    """Adapted inference: black-box LLM proposals scored/pruned by the adapter.

    Parameters
    ----------
    adapter:
        The trained energy adapter ``g_theta`` (see :mod:`bbox_adapter.adapter.energy_model`)
        or the MLM ablation scorer.  Any object exposing ``score_pairs`` / ``energy`` works,
        as does a plain callable ``f(question, texts) -> Sequence[float]``.
    generator:
        The black-box LLM.  May be a :class:`~bbox_adapter.llm.blackbox_client.BlackBoxClient`
        (exposing ``generate(prompt, n=..., temperature=..., max_len=...)``), any object with
        such a method, or a plain callable with the same signature.  The generator is only
        ever asked for **text**.
    config:
        :class:`BeamSearchConfig` (or a mapping/dict of its fields).

    Notes
    -----
    * ``n_samples`` samples are requested per beam per step, exactly as Section 3.3
      describes (``n * k`` candidates in the set ``C``).
    * No request to the black-box model contains logprob/echo parameters; the code path
      asserts this by construction (only ``prompt``, ``n``, ``temperature`` and
      ``max_len`` are ever passed).
    """

    def __init__(
        self,
        adapter: Any,
        generator: Any = None,
        config: Optional[Any] = None,
        *,
        token_counter: Any = None,
    ) -> None:
        self.adapter = adapter
        self.generator = generator
        self.config = (
            config
            if isinstance(config, BeamSearchConfig)
            else BeamSearchConfig.from_dict(config) if isinstance(config, dict) else BeamSearchConfig()
        )
        # optional cost accounting hook: object with .add(prompt, response) or a callable
        self.token_counter = token_counter

    # -- generator plumbing -----------------------------------------------------------
    def _call_generator(self, prompt: str, n: int) -> List[str]:
        """Ask the black-box model for ``n`` text continuations of ``prompt``.

        Only text is requested; never ``logprobs``/``echo``/hidden states.
        """
        if self.generator is None:
            raise ValueError("no black-box generator configured for beam search")
        gen = self.generator
        kwargs = dict(
            n=max(1, int(n)),
            temperature=self.config.temperature,
            max_len=self.config.max_len,
        )
        out = None
        if hasattr(gen, "generate"):
            try:
                out = gen.generate(prompt, **kwargs)
            except TypeError:
                try:
                    out = gen.generate(prompt, kwargs["n"])
                except TypeError:
                    out = gen.generate(prompt)
        elif hasattr(gen, "sample"):
            try:
                out = gen.sample(prompt, **kwargs)
            except TypeError:
                out = gen.sample(prompt)
        else:
            try:
                out = gen(prompt, **kwargs)
            except TypeError:
                out = gen(prompt)
        texts = _as_list_of_texts(out)
        if self.token_counter is not None:  # pragma: no cover - accounting hook
            for t in texts:
                _account(self.token_counter, prompt, t)
        return texts

    # -- prompting --------------------------------------------------------------------
    def continuation_prompt(self, prompt: str, prefix: Hypothesis) -> str:
        """Prompt asking the LLM to produce the next sentence ``s^l``."""
        if not prefix.text:
            return prompt
        return (
            f"{prompt.rstrip() + self.config.continuation_sep + prefix.text.rstrip()}"
            f"{self.config.continuation_sep}{self.config.step_instruction}"
        )

    # -- single beam-search step ------------------------------------------------------
    def step(self, prompt: str, beams: Sequence[Hypothesis]) -> Tuple[List[Hypothesis], List[Hypothesis], int]:
        """Expand every beam by ``n_samples`` sentences; return (top-k, candidates, calls)."""
        cfg = self.config
        candidates: List[Hypothesis] = []
        calls = 0
        for beam in beams:
            if beam.complete and cfg.stop_when_all_complete:
                # A beam that already emitted the stop signal is kept as-is (it is a final
                # answer); it is not extended further.
                candidates.append(beam)
                continue
            step_prompt = self.continuation_prompt(prompt, beam)
            raw_texts = self._call_generator(step_prompt, cfg.n_samples)
            calls += 1
            for text in raw_texts:
                sentence = _first_sentence(text)
                if not sentence and not contains_terminator(text):
                    continue
                candidates.append(beam.append(sentence))

        if cfg.deduplicate:
            candidates = _deduplicate(candidates)

        if not candidates:
            return list(beams), [], calls

        raw_scores = score_candidates(self.adapter, _question_of(prompt), candidates, cfg)
        scored: List[Hypothesis] = []
        for cand, raw in zip(candidates, raw_scores):
            cand = copy.copy(cand)
            cand.meta = dict(cand.meta)
            cand.meta["raw_energy_score"] = float(raw)
            cand.score = _normalized_score(
                float(raw), cand.n_sentences, cand.n_chars, cfg, token_len=cand.n_chars
            )
            scored.append(cand)
        idx = topk_indices([c.score for c in scored], cfg.beam_size)
        top = [scored[i] for i in idx]
        return top, scored, calls

    # -- full search ------------------------------------------------------------------
    def run(
        self,
        question: str,
        prompt: Optional[str] = None,
        *,
        answer_type: Optional[str] = None,
        choices: Optional[Sequence[str]] = None,
        initial_beams: Optional[Sequence[Hypothesis]] = None,
        prompt_tokens: int = 0,
    ) -> BeamSearchResult:
        """Run adapted inference on one question.

        Parameters
        ----------
        question:
            The raw question text (used for adapter scoring).
        prompt:
            The fully rendered prompt handed to the black-box LLM (Appendix J templates).
            Defaults to the question itself.
        answer_type / choices:
            Used to extract the final canonical answer for evaluation.
        """
        cfg = self.config
        if cfg.seed is not None:
            random.seed(cfg.seed)
            if torch is not None:
                torch.manual_seed(cfg.seed)

        prompt = question if prompt is None else prompt
        beams: List[Hypothesis] = list(initial_beams) if initial_beams else [Hypothesis(sentences=[])]
        result = BeamSearchResult(question=question, prompt_tokens=prompt_tokens)
        result.meta["beam_size"] = cfg.beam_size
        result.meta["n_samples"] = cfg.n_samples
        result.meta["max_steps"] = cfg.max_steps
        result.meta["config"] = cfg.to_dict()

        last_candidates: List[Hypothesis] = []
        for step_idx in range(cfg.max_steps):
            top, scored, calls = self.step(prompt, beams)
            result.n_llm_calls += calls
            result.steps_used = step_idx + 1
            result.n_candidates += len(scored)
            last_candidates = scored

            # Appendix-K style curves: energy spread of candidates / retained beams.
            result.energy_history.append(
                {
                    "step": step_idx + 1,
                    "n_candidates": float(len(scored)),
                    "mean_candidate_score": float(
                        sum(c.score for c in scored) / len(scored)
                    ) if scored else 0.0,
                    "max_candidate_score": float(max((c.score for c in scored), default=0.0)),
                    "min_candidate_score": float(min((c.score for c in scored), default=0.0)),
                    "top_beam_score": float(top[0].score) if top else 0.0,
                    "n_complete": float(sum(1 for c in scored if c.complete)),
                }
            )

            beams = top
            if not beams:
                break
            if cfg.stop_when_all_complete:
                # "Once ... all beams encounter a stop signal" -> terminate early.
                if all(b.complete for b in beams) and any(b.text for b in beams):
                    break

        # Final selection: highest adapter score among the retained beams ("The adapted
        # generation is then selected based on the highest-scoring option evaluated by
        # the adapter.").  Candidates are considered only if no beam survived.
        pool = [b for b in beams if b.text] or [b for b in last_candidates if b.text]
        if pool:
            best = max(pool, key=lambda h: h.score)
        else:
            best = Hypothesis(sentences=[], score=float("-inf"))

        result.beams = list(beams)
        result.candidates = last_candidates if cfg.keep_candidates else []
        result.best_hypothesis = best
        result.best_text = best.text
        result.best_score = float(best.score)
        if answer_type is not None:
            result.meta["answer"] = extract_final_answer(
                best.text, answer_type, choices=choices
            )
        return result

    __call__ = run

    # -- training-time helper ---------------------------------------------------------
    def sample_candidates(
        self,
        question: str,
        prompt: Optional[str] = None,
        n: int = 5,
        *,
        answer_type: Optional[str] = None,
        choices: Optional[Sequence[str]] = None,
        mode: str = "single_step",
        return_result: bool = False,
    ) -> Any:
        """Sample inference candidates from the *current adapted* model ``p_theta_t``.

        Used by the online adaptation loop (Section 3.4 / Algorithm 1) to refresh the
        negative set with "the inferences from the previous adaptation round", and by the
        AI-feedback setting to build the candidate pool shown to the gpt-4 rater.

        ``mode="single_step"`` draws ``n`` complete answers in one request (cheapest);
        ``mode="beam"`` runs the full sentence-level beam search and returns its best
        ``n`` hypotheses.
        """
        if mode == "single_step":
            return single_step_rank(
                self.adapter,
                self._call_generator(question if prompt is None else prompt, n),
                question,
                config=self.config,
                answer_type=answer_type,
                choices=choices,
                return_result=return_result,
            )
        res = self.run(question, prompt, answer_type=answer_type, choices=choices)
        pool = [b for b in res.beams if b.text]
        pool.sort(key=lambda h: h.score, reverse=True)
        texts = [b.text for b in pool[:n]]
        if return_result:
            return texts, res
        return texts


# --------------------------------------------------------------------------------------
# Module-level entry points
# --------------------------------------------------------------------------------------
def _question_of(prompt: str) -> str:
    """Best-effort recovery of the question from a rendered prompt (for adapter scoring).

    The adapter scores ``(x, y)`` pairs; when a full Appendix-J prompt is used we keep the
    whole prompt as ``x`` (it contains the question and the exemplars), which is what the
    harness does as well.
    """
    return prompt


def _deduplicate(hypotheses: Sequence[Hypothesis]) -> List[Hypothesis]:
    seen = set()
    out: List[Hypothesis] = []
    for h in hypotheses:
        key = h.text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _account(counter: Any, prompt: str, response: str) -> None:
    for name in ("add", "add_request", "record", "update", "count"):
        fn = getattr(counter, name, None)
        if callable(fn):
            try:
                fn(prompt, response)
                return
            except TypeError:
                try:
                    fn(prompt, [response])
                    return
                except TypeError:
                    continue
    if callable(counter):
        try:
            counter(prompt, response)
        except TypeError:
            counter(prompt, [response])


def beam_search(
    question: str,
    *,
    adapter: Any,
    generator: Any,
    prompt: Optional[str] = None,
    config: Optional[Any] = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    return_result: bool = False,
    **config_kwargs: Any,
) -> Any:
    """Convenience wrapper around :class:`SentenceBeamSearch` for one question."""
    cfg = config if config is not None else BeamSearchConfig(**config_kwargs)
    runner = SentenceBeamSearch(adapter, generator, cfg)
    res = runner.run(question, prompt, answer_type=answer_type, choices=choices)
    return res if return_result else res.best_text


def single_step_search(
    question: str,
    *,
    adapter: Any,
    generator: Any,
    prompt: Optional[str] = None,
    n: Optional[int] = None,
    config: Optional[Any] = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    return_result: bool = False,
    **config_kwargs: Any,
) -> Any:
    """Single-step adapted inference (Section 4.4 "single-step inference variant").

    The black-box model emits complete answers once (``n`` samples) and the adapter only
    ranks them -- no per-sentence beam expansion, hence the much lower inference cost of
    Table 4 (e.g. StrategyQA 69.87% at $2.20 per 1k questions vs $5.37 for full-step).
    """
    cfg = config if config is not None else BeamSearchConfig(**config_kwargs)
    runner = SentenceBeamSearch(adapter, generator, cfg)
    if cfg.seed is not None:
        random.seed(cfg.seed)
    prompt = question if prompt is None else prompt
    texts = runner._call_generator(prompt, n or cfg.beam_size)
    return single_step_rank(
        adapter,
        texts,
        question,
        config=cfg,
        answer_type=answer_type,
        choices=choices,
        return_result=return_result,
    )


def single_step_rank(
    adapter: Any,
    texts: Sequence[str],
    question: str,
    *,
    config: Optional[Any] = None,
    answer_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    return_result: bool = False,
) -> Any:
    """Rank already-generated complete answers with the adapter (no LLM calls).

    Returns the best text (default), or ``(best_text, BeamSearchResult)`` when
    ``return_result`` is set.  Used both by the single-step variant and by the online
    adaptation loop / AI-feedback candidate sampling.
    """
    cfg = config if isinstance(config, BeamSearchConfig) else BeamSearchConfig.from_dict(config)
    texts = [t for t in texts if t is not None]
    if not texts:
        empty = BeamSearchResult(question=question)
        return ("", empty) if return_result else ""

    if cfg.deduplicate:
        seen, uniq = set(), []
        for t in texts:
            key = t.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        texts = uniq

    hyps = [Hypothesis(sentences=split_steps(t) or [t]) for t in texts]
    if len(hyps) == 1:
        hyps = [Hypothesis(sentences=[texts[0]])]
    raw = resolve_adapter_scores(adapter, question, texts, max_length=cfg.max_len)
    scores = [
        _normalized_score(float(r), h.n_sentences, h.n_chars, cfg, token_len=h.n_chars)
        for r, h in zip(raw, hyps)
    ]
    best_idx = max(range(len(scores)), key=lambda i: (scores[i], -i))
    for h, s, r in zip(hyps, scores, raw):
        h.score = float(s)
        h.complete = contains_terminator(h.text)
        h.meta["raw_energy_score"] = float(r)

    result = BeamSearchResult(
        question=question,
        best_text=texts[best_idx],
        best_score=float(scores[best_idx]),
        best_hypothesis=hyps[best_idx],
        beams=hyps,
        candidates=hyps,
        n_llm_calls=0,
        n_candidates=len(texts),
        steps_used=1,
        meta={"regime": "single_step", "config": cfg.to_dict()},
    )
    if answer_type is not None:
        result.meta["answer"] = extract_final_answer(
            texts[best_idx], answer_type, choices=choices
        )
    return (texts[best_idx], result) if return_result else texts[best_idx]


# --------------------------------------------------------------------------------------
# Self-test (dependency-free): pruning must follow the adapter's ranking.
# --------------------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - smoke test
    cfg = BeamSearchConfig(beam_size=2, n_samples=2, max_steps=3, seed=0)

    # Adapter: prefers the candidate whose text mentions "gold".
    class FakeAdapter:
        lower_energy_is_better = True

        def score_pairs(self, pairs, max_length=None):
            return [0.0 if "gold" in a else (1.0 if "good" in a else 5.0) for _, a in pairs]

    # Generator: deterministic stream of continuations.
    class FakeGenerator:
        def __init__(self):
            self.counter = 0

        def generate(self, prompt, n=1, temperature=1.0, max_len=512):
            outs = []
            for _ in range(n):
                self.counter += 1
                outs.append(
                    "gold answer #### 1" if self.counter % 3 == 0 else f"distractor step {self.counter}"
                )
            return outs

    runner = SentenceBeamSearch(FakeAdapter(), FakeGenerator(), cfg)
    res = runner.run("q?", "Question: q?", answer_type="mcq")
    assert res.n_llm_calls > 0
    assert res.n_candidates > 0
    assert res.best_text, "beam search produced no text"
    assert any("gold" in h.text for h in res.beams), "gold beam was pruned away"

    # top-k ordering sanity
    assert topk_indices([0.1, 0.9, 0.5], 2) == [1, 2]

    # single-step ranking: best score wins
    adapter = FakeAdapter()
    best, out = single_step_rank(
        adapter,
        ["bad text", "a good step", "the gold answer"],
        "q?",
        config=cfg,
        return_result=True,
    )
    assert "gold" in best, best
    assert out.n_candidates == 3

    # normalization variants must not crash
    for norm in ("none", "sentences", "length"):
        c = BeamSearchConfig(normalization=norm)
        single_step_rank(FakeAdapter(), ["a", "b"], "q?", config=c)


if __name__ == "__main__":  # pragma: no cover
    _self_test()
    print("beam_search self-test OK")
