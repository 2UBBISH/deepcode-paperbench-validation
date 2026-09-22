"""Offline unit tests for BBox-Adapter adapted inference (Section 3.3).

Validates ``bbox_adapter.inference.beam_search`` -- the sentence-level beam search
in which the black-box LLM only *proposes* text (no logprobs / hidden states /
output probabilities are ever requested) and the trained energy adapter
``g_theta`` *evaluates* and prunes hypotheses (Eq. 4) -- plus the final
highest-scoring answer selection implemented in ``inference/selector.py``.

The tests are deliberately dependency-light: ``torch`` is used when available
(the real adapters return tensors) but everything degrades to plain Python so
the module also runs in a bare CI environment::

    python tests/test_beam_search.py
    pytest tests/test_beam_search.py -v
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # pragma: no cover - exercised indirectly
    import torch  # type: ignore
    _TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH = False

from bbox_adapter.inference import beam_search as B  # noqa: E402
from bbox_adapter.inference import selector as S  # noqa: E402
from bbox_adapter.data.answer_extraction import (  # noqa: E402
    ANSWER_TERMINATOR,
    extract_final_answer,
)

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

QUESTION = "Is it possible for a person to be allergic to sunlight?"
PROMPT = "Question: %s\nAnswer:" % QUESTION

GOOD = "good step."
BAD = "bad step."
STOP = "#### Yes."

FORBIDDEN_KWARGS = ("logprobs", "top_logprobs", "echo", "logit_bias")


class _FakeResult:
    """Stand-in for ``llm.blackbox_client.GenerationResult`` (text-only)."""

    def __init__(
        self,
        texts: Sequence[str],
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        model: str = "mock",
        temperature: float = 1.0,
        n: Optional[int] = None,
        finish_reasons: Optional[List[str]] = None,
        latency: float = 0.0,
        mock: bool = True,
        raw: Any = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.texts = list(texts)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.model = model
        self.temperature = temperature
        self.n = n if n is not None else len(self.texts)
        self.finish_reasons = finish_reasons or ["stop"] * len(self.texts)
        self.latency = latency
        self.mock = mock
        self.raw = raw
        self.meta = meta or {}

    @property
    def text(self) -> str:
        return self.texts[0] if self.texts else ""

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def __len__(self) -> int:
        return len(self.texts)

    def __iter__(self):
        return iter(self.texts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts": list(self.texts),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
        }


class _ScriptedGenerator:
    """Deterministic proposal generator that records every request."""

    def __init__(
        self,
        sentences: Sequence[str],
        *,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        repeat: bool = True,
    ) -> None:
        self.sentences = list(sentences) or [""]
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.repeat = repeat
        self.index = 0
        self.calls: List[Dict[str, Any]] = []

    # -- proposal API --------------------------------------------------------
    def _texts(self, n: int) -> List[str]:
        out: List[str] = []
        for _ in range(max(1, int(n))):
            if self.index < len(self.sentences):
                s = self.sentences[self.index]
            elif self.repeat:
                s = self.sentences[self.index % len(self.sentences)]
            else:
                s = ""
            self.index += 1
            out.append(s)
        return out

    def generate_result(self, prompt: str, n: int = 1, temperature: float = 1.0,
                        max_len: int = 512, **kwargs: Any) -> _FakeResult:
        self.calls.append(
            {"prompt": prompt, "n": n, "temperature": temperature,
             "max_len": max_len, "kwargs": dict(kwargs)}
        )
        return _FakeResult(
            self._texts(n),
            prompt_tokens=self.prompt_tokens * max(1, int(n)),
            completion_tokens=self.completion_tokens * max(1, int(n)),
            temperature=temperature,
        )

    def generate(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return self.generate_result(prompt, n=n, **kwargs).texts

    def generate_one(self, prompt: str, **kwargs: Any) -> str:
        texts = self.generate(prompt, n=1, **kwargs)
        return texts[0] if texts else ""

    def __call__(self, prompt: str, n: int = 1, **kwargs: Any) -> List[str]:
        return self.generate(prompt, n=n, **kwargs)

    # -- helpers -------------------------------------------------------------
    def forbidden_seen(self) -> List[str]:
        seen: List[str] = []
        for call in self.calls:
            for key in FORBIDDEN_KWARGS:
                if key in call["kwargs"]:
                    seen.append(key)
        return seen

    def requested_ns(self) -> List[int]:
        return [int(c["n"]) for c in self.calls]


def _good_scorer(text: str) -> float:
    """Preference score: how many 'good' sentences the text contains."""
    return float(str(text).lower().count("good"))


def _to_output(values: Sequence[float]):
    """Return a tensor when torch is available (matching the real EnergyModel)."""
    if _TORCH:
        return torch.tensor([float(v) for v in values], dtype=torch.float32)
    return [float(v) for v in values]


def _as_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if _TORCH and hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in list(value)]


class _EnergyAdapter:
    """Minimal duck-typed energy model: *lower* energy is better (Table 3 style)."""

    lower_energy_is_better = True

    def __init__(self, scorer: Callable[[str], float] = _good_scorer, offset: float = 0.0) -> None:
        self.scorer = scorer
        self.offset = offset

    @staticmethod
    def _texts(args: Sequence[Any], kwargs: Dict[str, Any]) -> List[str]:
        for candidate in reversed(list(args)):
            if isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[0], str):
                return [str(t) for t in candidate]
            if isinstance(candidate, str) and candidate:
                return [candidate]
        for key in ("answers", "answer", "texts", "hypotheses"):
            value = kwargs.get(key)
            if isinstance(value, (list, tuple)):
                return [str(t) for t in value]
            if isinstance(value, str):
                return [value]
        return []

    def _energies(self, texts: Sequence[str]) -> Sequence[float]:
        return [self.offset - self.scorer(t) for t in texts]

    def score_pairs(self, *args: Any, **kwargs: Any) -> Any:
        return _to_output(self._energies(self._texts(args, kwargs)))

    def score_batch(self, *args: Any, **kwargs: Any) -> Any:
        return _to_output(self._energies(self._texts(args, kwargs)))

    def energy(self, *args: Any, **kwargs: Any) -> Any:
        return self.score_pairs(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.score_pairs(*args, **kwargs)


class _ScoreAdapter:
    """Adapter exposing only ``score_batch`` with ambiguous sign convention."""

    def score_batch(self, *args: Any, **kwargs: Any) -> Any:
        return _to_output(_EnergyAdapter._texts(args, kwargs) and
                          [_good_scorer(t) for t in _EnergyAdapter._texts(args, kwargs)] or [])


def _resolve(adapter: Any, texts: Sequence[str]) -> List[float]:
    scores = B.resolve_adapter_scores(adapter, QUESTION, list(texts))
    out = _as_float_list(scores)
    if len(out) == 1 and len(texts) > 1:  # pragma: no cover - defensive
        return [float(out[0]) for _ in texts]
    return out


def _eq(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _val(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    value = getattr(obj, name, None)
    if value is None:
        return None
    return value(*args, **kwargs) if callable(value) else value


def _config(**kwargs: Any) -> Any:
    """Build a BeamSearchConfig tolerantly (kwargs that it rejects are dropped)."""
    try:
        return B.BeamSearchConfig(**kwargs)
    except TypeError:
        allowed = set(getattr(B.BeamSearchConfig, "__dataclass_fields__", {}) or {})
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return B.BeamSearchConfig(**filtered)


def _search(generator: Any, adapter: Any, **kwargs: Any) -> Any:
    cfg = _config(**kwargs)
    runner = B.SentenceBeamSearch(adapter, generator, cfg)
    return runner.run(QUESTION, prompt=PROMPT, answer_type="yesno")


# ---------------------------------------------------------------------------
# constants / config
# ---------------------------------------------------------------------------

def test_stop_signal_matches_answer_terminator():
    assert B.STOP_SIGNAL == ANSWER_TERMINATOR == "####"


def test_default_max_steps_constant():
    assert B.DEFAULT_MAX_STEPS >= 1


def test_beam_search_config_defaults_follow_paper():
    cfg = B.BeamSearchConfig()
    assert int(cfg.beam_size) == 3, "paper beam size is 3 (Appendix H.2)"
    assert float(cfg.temperature) == 1.0, "BBox-Adapter temperature is 1.0"
    assert int(cfg.max_len) == 512, "max generation length is 512"
    assert str(cfg.stop_signal) == "####"
    assert int(cfg.n_samples) >= 1
    assert int(cfg.max_steps) == int(B.DEFAULT_MAX_STEPS)


def test_beam_search_config_roundtrip():
    cfg = B.BeamSearchConfig(beam_size=5, n_samples=2, max_steps=4, temperature=0.7)
    payload = cfg.to_dict()
    assert payload["beam_size"] == 5 and payload["temperature"] == 0.7
    restored = B.BeamSearchConfig.from_dict(payload)
    assert restored.beam_size == 5 and restored.max_steps == 4


# ---------------------------------------------------------------------------
# top-k pruning helpers (Eq. 4)
# ---------------------------------------------------------------------------

def test_topk_indices_selects_descending_arguments():
    idx = list(B.topk_indices([0.1, 0.9, 0.5, -0.2], 2))
    assert idx == [1, 2], idx


def test_topk_indices_handles_k_larger_than_candidates():
    idx = list(B.topk_indices([0.3, 0.7], 5))
    assert sorted(idx) == [0, 1]
    assert idx[0] == 1


def test_topk_indices_is_stable_on_ties():
    idx = list(B.topk_indices([1.0, 1.0, 1.0], 2))
    assert sorted(idx) == [0, 1], idx


def test_topk_indices_accepts_tensors():
    if not _TORCH:
        return
    idx = list(B.topk_indices(torch.tensor([0.2, 2.0, 1.0]), 1))
    assert idx == [1]


# ---------------------------------------------------------------------------
# adapter scoring (evaluator role)
# ---------------------------------------------------------------------------

def test_resolve_adapter_scores_normalizes_energy_to_score():
    adapter = _EnergyAdapter()
    texts = ["bad step.", "good step.", "good good step."]
    scores = _resolve(adapter, texts)
    assert len(scores) == 3
    assert scores[2] == max(scores), scores
    assert scores[2] > scores[1] > scores[0], scores


def test_resolve_adapter_scores_score_adapter_matches_sign_convention():
    """Adapters without a declared convention must be treated consistently."""
    adapter = _ScoreAdapter()
    texts = ["bad step.", "good step.", "good good step."]
    raw = _good_scorer(texts[0]), _good_scorer(texts[1]), _good_scorer(texts[2])
    scores = _resolve(adapter, texts)
    assert len(scores) == 3
    same = all(_eq(a, b) for a, b in zip(scores, raw))
    flipped = all(_eq(a, -b) for a, b in zip(scores, raw))
    assert same or flipped, (scores, raw)


def test_score_candidates_returns_one_score_per_hypothesis():
    adapter = _EnergyAdapter()
    hyps = [B.Hypothesis(text="bad step."), B.Hypothesis(text="good step.")]
    scores = _as_float_list(B.score_candidates(adapter, QUESTION, hyps))
    assert len(scores) == 2
    assert scores[1] > scores[0]
    scores2 = _as_float_list(B.score_candidates(adapter, QUESTION, ["bad step.", "good step."]))
    assert scores2[1] > scores2[0]


# ---------------------------------------------------------------------------
# Hypothesis / result containers
# ---------------------------------------------------------------------------

def test_hypothesis_append_and_properties():
    hyp = B.Hypothesis(text="")
    hyp2 = hyp.append("First step.")
    assert hyp2 is not hyp
    assert "First step." in hyp2.text
    assert hyp2.n_sentences >= 1
    assert hyp2.n_chars == len(hyp2.text)
    assert isinstance(hyp2.to_dict(), dict)

    hyp3 = hyp2.append(STOP, score=1.5)
    assert STOP in hyp3.text
    assert _eq(hyp3.score, 1.5)
    assert hyp3.n_sentences > hyp2.n_sentences


def test_beam_search_result_to_dict_has_paper_fields():
    result = _search(_ScriptedGenerator([STOP]), _EnergyAdapter(), beam_size=2, max_steps=2)
    payload = result.to_dict()
    for key in ("question", "best_text", "best_score", "n_llm_calls", "steps_used"):
        assert key in payload, key
    assert payload["question"] == QUESTION


# ---------------------------------------------------------------------------
# sentence-level beam search behaviour
# ---------------------------------------------------------------------------

def test_beam_search_never_requests_probabilities():
    gen = _ScriptedGenerator([GOOD, BAD])
    _search(gen, _EnergyAdapter(), beam_size=2, n_samples=2, max_steps=2)
    assert gen.calls, "the generator should have been called"
    assert gen.forbidden_seen() == [], gen.forbidden_seen()
    for call in gen.calls:
        assert "logprobs" not in call["kwargs"]
        assert "echo" not in call["kwargs"]


def test_beam_search_scores_candidates_with_adapter():
    gen = _ScriptedGenerator([BAD, GOOD, BAD, GOOD], repeat=True)
    result = _search(gen, _EnergyAdapter(), beam_size=2, n_samples=2, max_steps=3)
    assert "good" in result.best_text.lower(), result.best_text
    assert result.n_candidates >= 1
    assert result.n_llm_calls >= 1


def test_beam_search_best_is_highest_scoring_beam():
    gen = _ScriptedGenerator([BAD, GOOD, BAD, GOOD], repeat=True)
    result = _search(gen, _EnergyAdapter(), beam_size=2, n_samples=2, max_steps=3)
    beams = list(getattr(result, "beams", []) or [])
    if beams:
        best = max((_val(h, "score") or 0.0) for h in beams)
        assert float(result.best_score) >= float(best) - 1e-3
        top = [h for h in beams if _eq(_val(h, "score") or 0.0, result.best_score)]
        assert result.best_text in [h.text for h in top] or any(
            h.text in result.best_text or result.best_text in h.text for h in top
        )


def test_beam_search_stops_when_all_beams_terminate():
    gen = _ScriptedGenerator([STOP])
    result = _search(gen, _EnergyAdapter(), beam_size=3, n_samples=1, max_steps=6)
    assert ANS := ANSWER_TERMINATOR in result.best_text
    assert int(result.steps_used) <= 2, result.steps_used
    assert int(result.steps_used) >= 1


def test_beam_search_respects_max_steps_without_terminator():
    gen = _ScriptedGenerator([GOOD, BAD], repeat=True)
    result = _search(gen, _EnergyAdapter(), beam_size=2, n_samples=1, max_steps=2)
    assert 1 <= int(result.steps_used) <= 2
    assert int(result.n_llm_calls) <= 3 * 2 * 2


def test_beam_search_call_budget_matches_beam_and_samples():
    gen = _ScriptedGenerator([GOOD, BAD], repeat=True)
    beam, n_samp, steps = 2, 2, 3
    result = _search(gen, _EnergyAdapter(), beam_size=beam, n_samples=n_samp, max_steps=steps)
    assert int(result.n_llm_calls) <= beam * n_samp * steps
    requested = gen.requested_ns()
    assert requested, gen.calls
    assert max(requested) <= n_samp, requested


def test_beam_search_energy_history_is_recorded():
    gen = _ScriptedGenerator([GOOD, BAD], repeat=True)
    result = _search(gen, _EnergyAdapter(), beam_size=2, n_samples=2, max_steps=2)
    history = list(getattr(result, "energy_history", []) or [])
    assert history, "Appendix-K style energy history should be recorded"
    assert len(history) == int(result.steps_used)


def test_beam_search_seed_makes_search_reproducible():
    def run_once() -> str:
        gen = _ScriptedGenerator([GOOD, BAD, GOOD, BAD], repeat=True)
        out = _search(gen, _EnergyAdapter(), beam_size=3, n_samples=2, max_steps=3, seed=0)
        return out.best_text

    assert run_once() == run_once()


def test_beam_search_functional_wrapper():
    gen = _ScriptedGenerator([GOOD, BAD, GOOD, BAD], repeat=True)
    text = B.beam_search(
        QUESTION,
        adapter=_EnergyAdapter(),
        generator=gen,
        prompt=PROMPT,
        beam_size=2,
        n_samples=2,
        max_steps=2,
        answer_type="yesno",
    )
    assert isinstance(text, str)
    assert "good" in text.lower()


def test_beam_search_starting_from_initial_beams():
    gen = _ScriptedGenerator([STOP])
    cfg = _config(beam_size=2, n_samples=1, max_steps=3)
    runner = B.SentenceBeamSearch(_EnergyAdapter(), gen, cfg)
    initial = [B.Hypothesis(text="A reasoning step.")]
    result = runner.run(QUESTION, prompt=PROMPT, answer_type="yesno", initial_beams=initial)
    assert "A reasoning step." in result.best_text or ANSWER_TERMINATOR in result.best_text


# ---------------------------------------------------------------------------
# single-step (cheaper) adapted inference
# ---------------------------------------------------------------------------

def test_single_step_rank_selects_adapter_preferred_answer():
    texts = ["#### No.", "#### Yes.", "#### Yes."]
    adapter = _EnergyAdapter()
    picked = B.single_step_rank(adapter, texts, QUESTION, answer_type="yesno")
    assert str(picked).strip().endswith("Yes."), picked
    assert extract_final_answer(str(picked), "yesno") == "Yes"


def test_single_step_rank_returns_result_when_requested():
    texts = ["#### No.", "#### Yes."]
    out = B.single_step_rank(_EnergyAdapter(), texts, QUESTION,
                             answer_type="yesno", return_result=True)
    assert isinstance(out, tuple), out
    text, result = out
    assert isinstance(result, B.BeamSearchResult)
    assert int(result.n_llm_calls) == 0, "single-step ranking performs no LLM calls"
    assert isinstance(text, str)


def test_single_step_search_proposes_once_and_ranks():
    gen = _ScriptedGenerator([BAD, GOOD, GOOD, BAD], repeat=True)
    text = B.single_step_search(
        QUESTION,
        adapter=_EnergyAdapter(),
        generator=gen,
        prompt=PROMPT,
        n=4,
        answer_type="yesno",
    )
    assert isinstance(text, str)
    assert gen.calls and max(gen.requested_ns()) <= 4
    assert "good" in text.lower()


def test_sample_candidates_single_step_mode():
    gen = _ScriptedGenerator([STOP])
    cfg = _config(beam_size=2, n_samples=1, max_steps=2)
    runner = B.SentenceBeamSearch(_EnergyAdapter(), gen, cfg)
    candidates = runner.sample_candidates(QUESTION, prompt=PROMPT, n=3, mode="single_step")
    assert isinstance(candidates, list)
    assert len(candidates) == 3
    assert gen.calls


def test_continuation_prompt_contains_prefix():
    cfg = _config()
    runner = B.SentenceBeamSearch(_EnergyAdapter(), _ScriptedGenerator([GOOD]), cfg)
    prompt = runner.continuation_prompt(PROMPT, "First step.")
    assert isinstance(prompt, str)
    assert "First step." in prompt
    assert PROMPT.splitlines()[0] in prompt


# ---------------------------------------------------------------------------
# selector (Section 3.3 final highest-scoring option)
# ---------------------------------------------------------------------------

def test_selector_picks_highest_scoring_candidate():
    candidates = [("bad answer", -2.0), ("good answer", 3.0), ("worse answer", -5.0)]
    payload = [{"text": t, "score": s} for t, s in candidates]
    out = S.select_answer(QUESTION, payload, answer_type="free")
    text = out.text if hasattr(out, "text") else str(out)
    assert "good answer" in text


def test_selector_from_beam_search_result_matches_best_beam():
    gen = _ScriptedGenerator([BAD, GOOD, BAD, GOOD], repeat=True)
    result = _search(gen, _EnergyAdapter(), beam_size=2, n_samples=2, max_steps=2)
    out = S.select_from_result(result, answer_type="yesno")
    text = out.text if hasattr(out, "text") else str(out)
    assert text.strip() != ""
    assert text in result.best_text or result.best_text in text


def test_selection_result_reports_ranked_candidates():
    payload = [{"text": "a", "score": 1.0}, {"text": "b", "score": 2.0}]
    out = S.select_answer(QUESTION, payload, answer_type="free", return_result=True)
    assert isinstance(out, S.SelectionResult)
    ranked = list(out.candidates)
    assert [c.text for c in ranked][0] == "b"
    assert ranked[0].score >= ranked[1].score


def test_rank_candidates_is_descending_and_deterministic():
    cands = [S.Candidate(text="x", score=0.5), S.Candidate(text="y", score=2.0),
             S.Candidate(text="z", score=0.5)]
    ranked = S.rank_candidates(cands)
    assert ranked[0].score == max(c.score for c in cands)
    assert ranked[0].text == "y"
    assert [c.text for c in S.rank_candidates(list(reversed(cands)))][0] == "y"


def test_rank_candidates_prefers_terminated_on_tie():
    terminated = S.Candidate(text="reasoning\n#### Yes.", score=1.0)
    plain = S.Candidate(text="reasoning only", score=1.0)
    ranked = S.rank_candidates([plain, terminated])
    assert ranked[0].text == terminated.text


def test_aggregate_scores_modes():
    assert _eq(S.aggregate_scores([1.0, 3.0], "max"), 3.0)
    assert _eq(S.aggregate_scores([1.0, 3.0], "sum"), 4.0)
    assert _eq(S.aggregate_scores([1.0, 3.0], "mean"), 2.0)
    expected = math.log(math.exp(1.0) + math.exp(3.0))
    assert abs(S.aggregate_scores([1.0, 3.0], "logsumexp") - expected) < 1e-6


def test_logsumexp_is_numerically_stable():
    big = [1000.0, 1000.0]
    assert abs(S.logsumexp(big) - (1000.0 + math.log(2.0))) < 1e-6
    assert abs(S.logsumexp([0.0]) - 0.0) < 1e-9


def test_best_by_answer_groups_identical_answers():
    cands = [
        S.Candidate(text="#### Yes.", score=1.0, answer="Yes"),
        S.Candidate(text="#### Yes.", score=2.0, answer="Yes"),
        S.Candidate(text="#### No.", score=1.5, answer="No"),
    ]
    key, score, group = S.best_by_answer(cands, aggregation="max")
    assert key == "Yes"
    assert _eq(score, 2.0)
    assert len(group) == 2


def test_normalize_candidates_accepts_mixed_inputs():
    out = S.normalize_candidates(["plain text", ("titled", 1.0), B.Hypothesis(text="hyp", score=2.0)])
    texts = [c.text for c in out]
    assert "plain text" in texts and "titled" in texts and "hyp" in texts
    assert len(out) == 3


def test_selector_config_roundtrip():
    cfg = S.SelectorConfig(aggregation="max", answer_type="yesno")
    restored = S.SelectorConfig.from_dict(cfg.to_dict())
    assert restored.aggregation == cfg.aggregation
    assert restored.answer_type == cfg.answer_type


def test_score_tie_break_is_symmetric():
    a = S.Candidate(text="a", score=1.0, index=0)
    b = S.Candidate(text="b", score=1.0, index=1)
    assert S.score_tie_break(a, b) in (-1, 0, 1)
    assert S.score_tie_break(a, b) == -S.score_tie_break(b, a)


# ---------------------------------------------------------------------------
# black-box boundary
# ---------------------------------------------------------------------------

def test_beam_search_never_touches_adapter_gradients_of_llm():
    """Only the adapter is scored; the generator receives text-only arguments."""
    gen = _ScriptedGenerator([GOOD, BAD], repeat=True)
    _search(gen, _EnergyAdapter(), beam_size=2, n_samples=1, max_steps=2)
    allowed = {"prompt", "n", "temperature", "max_len", "stop", "seed", "top_p",
               "presence_penalty", "frequency_penalty", "timeout", "extra"}
    for call in gen.calls:
        assert set(call["kwargs"]).issubset(allowed), call["kwargs"]


def test_sentence_segmentation_of_generated_text():
    hyp = B.Hypothesis(text="Step one. Step two.\n#### Yes.")
    assert hyp.n_sentences >= 2


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True) -> Dict[str, Any]:
    passed: List[str] = []
    failed: List[str] = []
    errors: List[str] = []
    tests = {name: fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)}
    for name, fn in tests.items():
        try:
            fn()
            passed.append(name)
            if verbose:
                print("PASS %s" % name)
        except AssertionError as exc:
            failed.append(name)
            if verbose:
                print("FAIL %s: %s" % (name, exc))
                traceback.print_exc()
        except Exception as exc:  # pragma: no cover
            errors.append(name)
            if verbose:
                print("ERROR %s: %r" % (name, exc))
                traceback.print_exc()
    if verbose:
        print("\n%d passed, %d failed, %d errors (torch=%s)"
              % (len(passed), len(failed), len(errors), _TORCH))
    return {"passed": passed, "failed": failed, "errors": errors}


if __name__ == "__main__":
    outcome = run_all()
    raise SystemExit(0 if not outcome["failed"] and not outcome["errors"] else 1)
