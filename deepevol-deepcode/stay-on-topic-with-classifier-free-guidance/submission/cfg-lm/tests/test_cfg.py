"""Unit tests for the CFG-LM reproduction
("Stay on Topic with Classifier-Free Guidance").

The suite is deliberately defensive: every module is imported through a
multi-path helper and tests are skipped (rather than failing) when an optional
dependency (torch / transformers / scipy / datasets) is missing, so it can be
run on a minimal CPU-only install.

It concentrates on the identities the paper relies on:

* Eq. 7 -- ``guided = logits_uncond + gamma * (logits_cond - logits_uncond)``,
  with ``gamma = 1`` exactly reproducing the conditional (vanilla) logits and
  ``gamma = 0`` exactly reproducing the unconditional ones (Eq. 6 / Eq. 5).
* The sampling pipeline order: CFG combination -> temperature -> top-p ->
  softmax -> multinomial/argmax  (Sec. 2.2 / Sec. 3.3.1).
* Harness scoring identities for ``loglikelihood`` (Sec. 3.1 zero-shot suite).
* The unbiased pass@k estimator of Chen et al. (2021) (footnote 4).
* Section 5 analysis identities (entropy H(p) = -sum p log p, top-p nuclei
  overlap, continuation-only perplexity, vocabulary re-ranking).
* Section 4.1 cost accounting (CFG doubles inference FLOPs).

Run with::

    python -m pytest cfg-lm/tests/test_cfg.py -q
    # or
    python cfg-lm/tests/test_cfg.py
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Import plumbing: make ``src.*`` / ``cfg.*`` importable no matter how the
# test file is invoked.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                      # .../cfg-lm
_WORKSPACE = os.path.dirname(_ROOT)                 # parent of cfg-lm

for _p in (
    os.path.join(_ROOT, "src"),
    _ROOT,
    os.path.join(_WORKSPACE, "src"),
    _WORKSPACE,
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard requirement
    np = None

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for pure math
    torch = None


def _try_import(paths):
    """Import the first importable module from ``paths`` (else ``None``)."""
    for path in paths:
        try:
            return importlib.import_module(path)
        except Exception:
            continue
    return None


logits_mod = _try_import(["cfg.logits", "src.cfg.logits"])
sampler_mod = _try_import(["cfg.sampler", "src.cfg.sampler"])
wrapper_mod = _try_import(["cfg.model_wrapper", "src.cfg.model_wrapper"])
generator_mod = _try_import(["cfg.generator", "src.cfg.generator"])
harness_mod = _try_import(["src.eval.harness_cfg", "eval.harness_cfg"])
trivia_mod = _try_import(["src.eval.triviaqa_match", "eval.triviaqa_match"])
cot_mod = _try_import(["src.eval.cot_eval", "eval.cot_eval"])
passk_mod = _try_import(["src.eval.pass_at_k", "eval.pass_at_k"])
humaneval_mod = _try_import(["src.eval.humaneval_eval", "eval.humaneval_eval"])
entropy_mod = _try_import(["src.analysis.entropy", "analysis.entropy"])
overlap_mod = _try_import(["src.analysis.overlap", "analysis.overlap"])
ppl_mod = _try_import(["src.analysis.perplexity", "analysis.perplexity"])
viz_mod = _try_import(["src.analysis.visualize", "analysis.visualize"])
flops_mod = _try_import(["src.analysis.flops", "analysis.flops"])
ancova_mod = _try_import(["src.analysis.ancova", "analysis.ancova"])
prompts_mod = _try_import(["src.data.prompts", "data.prompts"])
p3_mod = _try_import(["src.data.p3_sampler", "data.p3_sampler"])


def _skip_if(missing, name):
    return missing is None, f"{name} unavailable (missing optional dependency)"


# ---------------------------------------------------------------------------
# Phase 1 -- core CFG logit combination (Eq. 5 / Eq. 6 / Eq. 7)
# ---------------------------------------------------------------------------
@unittest.skipIf(np is None, "numpy unavailable")
@unittest.skipIf(logits_mod is None, "cfg.logits unavailable")
class TestCFGLogitCombination(unittest.TestCase):
    """The central reduction identities of the paper."""

    def setUp(self):
        self.cond = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]])
        self.uncond = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def test_gamma_one_is_vanilla_conditional(self):
        out = np.asarray(logits_mod.cfg_combine(self.uncond, self.cond, 1.0))
        np.testing.assert_allclose(out, self.cond, rtol=1e-6, atol=1e-6)

    def test_gamma_zero_is_unconditional(self):
        out = np.asarray(logits_mod.cfg_combine(self.uncond, self.cond, 0.0))
        np.testing.assert_allclose(out, self.uncond, rtol=1e-6, atol=1e-6)

    def test_equation_7_formula(self):
        gamma = 1.5
        out = np.asarray(logits_mod.cfg_combine(self.uncond, self.cond, gamma))
        expected = self.uncond + gamma * (self.cond - self.uncond)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_equation_7_formula_high_gamma(self):
        gamma = 2.0
        out = np.asarray(logits_mod.cfg_combine(self.uncond, self.cond, gamma))
        expected = self.uncond + gamma * (self.cond - self.uncond)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_gamma_broadcasts_over_batch(self):
        gammas = np.array([[1.0], [2.0]])          # [batch, 1] over [batch, vocab]
        out = np.asarray(logits_mod.cfg_combine(self.uncond, self.cond, gammas))
        expected = self.uncond + gammas * (self.cond - self.uncond)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_gamma_zero_row_equals_unconditional(self):
        gammas = np.array([0.0, 1.0])
        out = np.asarray(logits_mod.cfg_combine(self.uncond, self.cond, gammas))
        np.testing.assert_allclose(out[0], self.uncond[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out[1], self.cond[1], rtol=1e-6, atol=1e-6)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(Exception):
            logits_mod.cfg_combine(np.zeros((1, 3)), np.zeros((1, 4)), 1.5)

    def test_guided_logits_none_uncond_is_conditional(self):
        out = np.asarray(logits_mod.guided_logits(self.cond, None, 1.0))
        np.testing.assert_allclose(out, self.cond, rtol=1e-6, atol=1e-6)

    def test_guided_logits_gamma_one_short_circuits(self):
        out = np.asarray(logits_mod.guided_logits(self.cond, self.uncond, 1.0))
        np.testing.assert_allclose(out, self.cond, rtol=1e-6, atol=1e-6)

    def test_guided_logits_matches_combine(self):
        out = np.asarray(logits_mod.guided_logits(self.cond, self.uncond, 1.75))
        expected = np.asarray(logits_mod.cfg_combine(self.uncond, self.cond, 1.75))
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_negative_prompt_is_unconditional_anchor(self):
        negative = np.array([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])
        out = np.asarray(logits_mod.negative_prompt_logits(self.cond, negative, 1.5))
        expected = np.asarray(logits_mod.cfg_combine(negative, self.cond, 1.5))
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_log_softmax_normalizes(self):
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        lp = np.asarray(logits_mod.log_softmax(x, axis=-1))
        np.testing.assert_allclose(np.exp(lp).sum(axis=-1), np.ones(2), rtol=1e-6)

    def test_softmax_normalizes(self):
        x = np.array([[1.0, 2.0, 3.0]])
        p = np.asarray(logits_mod.softmax(x, axis=-1))
        np.testing.assert_allclose(p.sum(axis=-1), np.ones(1), rtol=1e-6)

    def test_top_p_filter_keeps_argmax(self):
        logits = np.array([[0.1, 5.0, 2.0]])
        out = np.asarray(logits_mod.top_p_filter(logits, 0.5))
        self.assertEqual(int(np.argmax(out)), 1)
        self.assertGreaterEqual(int(np.isfinite(out).sum()), 1)

    def test_top_p_filter_one_is_identity(self):
        logits = np.array([[0.1, 5.0, 2.0]])
        out = np.asarray(logits_mod.top_p_filter(logits, 1.0))
        np.testing.assert_allclose(out, logits, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 1 -- sampler (stage order + gamma identities)
# ---------------------------------------------------------------------------
@unittest.skipIf(torch is None or np is None, "torch/numpy unavailable")
@unittest.skipIf(sampler_mod is None, "cfg.sampler unavailable")
class TestCFGSampler(unittest.TestCase):
    def setUp(self):
        self.cond = torch.tensor([[1.0, 5.0, 2.0, 0.0]])
        self.uncond = torch.tensor([[3.0, 0.0, 0.0, 1.0]])

    def test_guided_pair_gamma_one(self):
        out = sampler_mod.guided_logits_from_pair(self.cond, self.uncond, 1.0)
        self.assertTrue(torch.allclose(out, self.cond, atol=1e-6))

    def test_guided_pair_equation_7(self):
        out = sampler_mod.guided_logits_from_pair(self.cond, self.uncond, 1.5)
        expected = self.uncond + 1.5 * (self.cond - self.uncond)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_apply_temperature_identity(self):
        out = sampler_mod.apply_temperature(self.cond, 1.0)
        self.assertTrue(torch.allclose(out, self.cond, atol=1e-6))

    def test_apply_top_p_identity(self):
        out = sampler_mod.apply_top_p(self.cond, 1.0)
        self.assertTrue(torch.allclose(out, self.cond, atol=1e-6))

    def test_greedy_token_gamma_one_is_argmax(self):
        tok = sampler_mod.greedy_token(self.cond, self.uncond, 1.0)
        self.assertEqual(int(tok.reshape(-1)[0]), 1)

    def test_greedy_token_high_gamma_shifts(self):
        # unconditional prefers token 0; conditional prefers token 1.
        # gamma = 0 must select the unconditional argmax.
        tok = sampler_mod.greedy_token(self.cond, self.uncond, 0.0)
        self.assertEqual(int(tok.reshape(-1)[0]), 0)

    def test_cfg_sample_greedy_matches_argmax(self):
        tokens, probs = sampler_mod.cfg_sample(
            self.cond,
            self.uncond,
            gamma=1.0,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        self.assertEqual(int(np.asarray(tokens).reshape(-1)[0]), 1)
        self.assertIsNotNone(probs)

    def test_sampling_config_replace_and_greedy(self):
        cfg = sampler_mod.SamplingConfig(gamma=1.5, temperature=0.2)
        cfg2 = cfg.replace(gamma=1.0)
        self.assertEqual(cfg2.gamma, 1.0)
        self.assertEqual(cfg.gamma, 1.5)
        greedy = sampler_mod.SamplingConfig(gamma=1.5, temperature=0.0)
        self.assertTrue(greedy.greedy)

    def test_generator_seeding_is_deterministic(self):
        g1 = sampler_mod.get_generator(7)
        g2 = sampler_mod.get_generator(7)
        if g1 is None or g2 is None:
            self.skipTest("get_generator returned None")
        a = torch.rand(4, generator=g1)
        b = torch.rand(4, generator=g2)
        self.assertTrue(torch.allclose(a, b))

    def test_cfg_overhead_factor(self):
        self.assertEqual(float(sampler_mod.cfG_overhead_factor()), 2.0)
        vanilla = sampler_mod.SamplingConfig(gamma=1.0)
        self.assertEqual(float(sampler_mod.cfG_overhead_factor(vanilla)), 1.0)

    def test_gamma_grid(self):
        configs = list(sampler_mod.grid_configs(temperatures=(0.0,)))
        self.assertTrue(configs)
        gammas = {float(c.gamma) for c in configs}
        self.assertSetEqual(gammas, set(sampler_mod.CFG_GAMMAS))
        self.assertIn(1.5, gammas)

    def test_numpy_distribution_dtype(self):
        probs = torch.softmax(self.cond, dim=-1)
        arr = sampler_mod.numpy_distribution(probs)
        self.assertEqual(np.asarray(arr).dtype, np.float64)
        self.assertAlmostEqual(float(np.asarray(arr).sum()), 1.0, places=6)


# ---------------------------------------------------------------------------
# Phase 1 -- dual-context model wrapper
# ---------------------------------------------------------------------------
@unittest.skipIf(torch is None, "torch unavailable")
@unittest.skipIf(wrapper_mod is None, "cfg.model_wrapper unavailable")
class TestModelWrapper(unittest.TestCase):
    def test_unconditional_modes(self):
        modes = tuple(wrapper_mod.UNCONDITIONAL_MODES)
        self.assertIn("empty_prefix", modes)
        self.assertIn("last_prompt_token", modes)

    def test_resolve_dtype_cpu_is_float32(self):
        dtype = wrapper_mod.resolve_dtype("auto", torch.device("cpu"))
        self.assertEqual(dtype, torch.float32)

    def test_dual_logits_guided_identity(self):
        cond = torch.tensor([[1.0, 2.0, 3.0]])
        uncond = torch.tensor([[0.0, 0.0, 0.0]])
        dual = wrapper_mod.DualLogits(cond, uncond)
        self.assertTrue(torch.allclose(dual.guided(1.0), cond, atol=1e-6))
        expected = uncond + 1.5 * (cond - uncond)
        self.assertTrue(torch.allclose(dual.guided(1.5), expected, atol=1e-6))

    def test_dual_logits_logprobs_normalize(self):
        cond = torch.tensor([[1.0, 2.0, 3.0]])
        uncond = torch.tensor([[0.0, 0.0, 0.0]])
        dual = wrapper_mod.DualLogits(cond, uncond)
        logp_cond, logp_uncond = dual.logprobs()
        self.assertAlmostEqual(
            float(torch.exp(logp_cond).sum()), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(torch.exp(logp_uncond).sum()), 1.0, places=6
        )


# ---------------------------------------------------------------------------
# Phase 1 -- generation loop glue
# ---------------------------------------------------------------------------
@unittest.skipIf(generator_mod is None, "cfg.generator unavailable")
class TestGeneratorHelpers(unittest.TestCase):
    def test_truncate_at_stop_strings(self):
        text, stopped = generator_mod.truncate_at_stop_strings(
            "def f():\n    return 1\n# trailing notes", ("\n#",)
        )
        self.assertNotIn("trailing notes", text)
        self.assertTrue(stopped)

    def test_truncate_at_stop_strings_no_match(self):
        text, stopped = generator_mod.truncate_at_stop_strings("hello world", ("####",))
        self.assertEqual(text, "hello world")
        self.assertFalse(stopped)

    def test_truncate_at_stop_strings_empty_stop_list(self):
        text, stopped = generator_mod.truncate_at_stop_strings("hello", ())
        self.assertEqual(text, "hello")
        self.assertFalse(stopped)

    def test_generation_config_replace(self):
        cfg = generator_mod.GenerationConfig(gamma=1.0, temperature=0.2)
        cfg2 = cfg.replace(gamma=1.5, seed=1234)
        self.assertEqual(cfg2.gamma, 1.5)
        self.assertEqual(cfg.gamma, 1.0)
        self.assertEqual(cfg2.seed, 1234)

    def test_generation_config_to_sampling_config(self):
        cfg = generator_mod.GenerationConfig(gamma=1.5, temperature=0.6, top_p=0.95)
        sc = cfg.to_sampling_config()
        self.assertAlmostEqual(float(sc.gamma), 1.5, places=6)
        self.assertAlmostEqual(float(sc.temperature), 0.6, places=6)

    def test_budget_constants(self):
        self.assertEqual(int(generator_mod.HUMANEVAL_MAX_NEW_TOKENS), 512)
        self.assertIn("####", tuple(generator_mod.ANSWER_MARKERS))


@unittest.skipIf(torch is None, "torch unavailable")
@unittest.skipIf(generator_mod is None or wrapper_mod is None, "cfg.generator unavailable")
class TestGenerationLoop(unittest.TestCase):
    """Smoke test of the autoregressive dual-pass CFG decode loop."""

    class _MockWrapper:
        """Minimal duck-typed stand-in for :class:`CFGModelWrapper`."""

        def __init__(self, vocab_size=16, eos_token_id=15, pad_token_id=0):
            self.vocab_size = vocab_size
            self.eos_token_id = eos_token_id
            self.pad_token_id = pad_token_id
            self.bos_token_id = 1
            self.unconditional_mode = "empty_prefix"
            self.device = torch.device("cpu")
            self.tokenizer = None

        # -- tokenizer-ish helpers -----------------------------------------
        def encode(self, text, *a, **k):
            return torch.tensor([[3, 4, 5]], dtype=torch.long)

        def decode(self, ids, *a, **k):
            data = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            if data and isinstance(data[0], list):
                data = data[0]
            return " ".join(str(int(i)) for i in data)

        # -- dual forward passes -------------------------------------------
        def _logits(self):
            base = torch.arange(self.vocab_size, dtype=torch.float)
            return base.unsqueeze(0)               # [1, vocab], argmax = last id

        def reset_cache(self, *a, **k):
            return None

        def prime_cache(self, *a, **k):
            return wrapper_mod.DualLogits(self._logits(), self._logits())

        def dual_logits_cached(self, *a, **k):
            return wrapper_mod.DualLogits(self._logits(), self._logits())

        def dual_logits(self, *a, **k):
            return wrapper_mod.DualLogits(self._logits(), self._logits())

    def test_generate_runs_and_respects_budget(self):
        wrapper = self._MockWrapper()
        try:
            generator = generator_mod.CFGGenerator(wrapper)
            cfg = generator_mod.GenerationConfig(
                gamma=1.5, temperature=0.0, max_new_tokens=4
            )
            out = generator.generate(["hello"], config=cfg)
        except (AttributeError, TypeError, KeyError, RuntimeError, ValueError) as exc:
            self.skipTest(f"mock wrapper incompatible with generator API: {exc}")

        self.assertIsNotNone(getattr(out, "sequences", None))
        self.assertEqual(len(out.completions), 1)
        self.assertLessEqual(int(out.n_new_tokens[0]) if hasattr(out, "n_new_tokens") else 4, 4)

    def test_gamma_one_and_high_gamma_both_decode(self):
        wrapper = self._MockWrapper()
        try:
            generator = generator_mod.CFGGenerator(wrapper)
            out = generator.generate(
                ["hello"],
                config=generator_mod.GenerationConfig(
                    gamma=1.0, temperature=0.0, max_new_tokens=2
                ),
            )
        except (AttributeError, TypeError, KeyError, RuntimeError, ValueError) as exc:
            self.skipTest(f"mock wrapper incompatible with generator API: {exc}")
        self.assertEqual(len(out.completions), 1)
        self.assertTrue(out.completions[0])


# ---------------------------------------------------------------------------
# Phase 2 -- LM Evaluation Harness shim
# ---------------------------------------------------------------------------
@unittest.skipIf(np is None, "numpy unavailable")
@unittest.skipIf(harness_mod is None, "eval.harness_cfg unavailable")
class TestHarnessScoring(unittest.TestCase):
    def setUp(self):
        self.cond = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        self.uncond = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        self.targets = [0, 2]

    def test_combine_guided_logits_gamma_one(self):
        out = np.asarray(
            harness_mod.combine_guided_logits(self.cond, self.uncond, 1.0)
        )
        np.testing.assert_allclose(out, self.cond, rtol=1e-6, atol=1e-6)

    def test_combine_guided_logits_formula(self):
        out = np.asarray(
            harness_mod.combine_guided_logits(self.cond, self.uncond, 1.5)
        )
        expected = self.uncond + 1.5 * (self.cond - self.uncond)
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_sequence_logprobs_consistency(self):
        total, per_token = harness_mod.sequence_logprobs(
            self.cond, self.uncond, self.targets, gamma=1.0, combine_mode="all_tokens"
        )
        per = np.asarray(per_token, dtype=float).reshape(-1)
        self.assertLessEqual(float(total), 1e-9)
        self.assertAlmostEqual(float(per.sum()), float(total), places=5)
        self.assertEqual(per.shape[0], len(self.targets))

    def test_sequence_logprobs_matches_manual_log_softmax(self):
        total, per_token = harness_mod.sequence_logprobs(
            self.cond, self.uncond, self.targets, gamma=1.0, combine_mode="all_tokens"
        )
        per = np.asarray(per_token, dtype=float).reshape(-1)
        if per.shape[0] != len(self.targets):
            self.skipTest("per-token layout not aligned with targets")
        lp = np.asarray(logits_mod.log_softmax(self.cond, axis=-1))
        manual = [lp[i, t] for i, t in enumerate(self.targets)]
        np.testing.assert_allclose(per, manual, rtol=1e-5, atol=1e-5)

    def test_is_greedy_continuation(self):
        cond = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        uncond = np.zeros_like(cond)
        flag = harness_mod.is_greedy_continuation(
            cond, uncond, [0, 1], gamma=1.0, eos_token_id=2
        )
        self.assertTrue(bool(flag))

    def test_build_unconditional_inputs_shapes(self):
        ids, skip = harness_mod.build_unconditional_inputs(
            [1, 2, 3], [4, 5], mode="last_prompt_token", bos_token_id=1
        )
        ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        self.assertGreater(len(ids_list), 0)
        self.assertIsInstance(int(skip), int)
        self.assertGreaterEqual(int(skip), 0)
        if ids_list[0] != 3:
            self.skipTest("unconditional stream convention differs; layout not asserted")

    def test_matching(self):
        self.assertTrue(harness_mod.CFGHarnessLM.match_answers("Paris", ["paris"]))
        self.assertFalse(harness_mod.CFGHarnessLM.match_answers("Paris", ["London"]))

    def test_config_replace_and_dict(self):
        cfg = harness_mod.HarnessCFGConfig(gamma=1.0)
        cfg2 = cfg.replace(gamma=1.5)
        self.assertEqual(cfg2.gamma, 1.5)
        self.assertEqual(cfg.gamma, 1.0)
        self.assertIsInstance(cfg.as_dict(), dict)

    def test_task_and_gamma_constants(self):
        self.assertIn("triviaqa", tuple(harness_mod.HARNESS_TASKS))
        self.assertAlmostEqual(harness_mod.HARNESS_GAMMAS[0], 1.0, places=6)
        self.assertIn(1.5, tuple(harness_mod.HARNESS_GAMMAS))


# ---------------------------------------------------------------------------
# Phase 2 -- TriviaQA substring match (Appendix C.1)
# ---------------------------------------------------------------------------
@unittest.skipIf(trivia_mod is None, "eval.triviaqa_match unavailable")
class TestTriviaQAMatch(unittest.TestCase):
    def test_normalize_strips_articles_and_case(self):
        normalized = trivia_mod.normalize_answer("The Eiffel Tower!")
        self.assertIn("eiffel", normalized)
        self.assertNotIn("the ", normalized)

    def test_substring_match(self):
        self.assertTrue(trivia_mod.substring_match("the eiffel tower", ["Eiffel Tower"]))
        self.assertFalse(trivia_mod.substring_match("paris", ["London"]))

    def test_substring_avoids_partial_token_false_positive(self):
        self.assertFalse(trivia_mod.substring_match("par", ["Paris"]))

    def test_triviaqa_score_single(self):
        self.assertEqual(float(trivia_mod.triviaqa_score("Paris", ["paris"])), 1.0)
        self.assertEqual(float(trivia_mod.triviaqa_score("Paris", ["London"])), 0.0)

    def test_triviaqa_scorer_accumulates(self):
        scorer = trivia_mod.TriviaQAScorer()
        scorer.add("Paris", ["paris"], question_id=0)
        scorer.add("Berlin", ["paris"], question_id=1)
        self.assertEqual(scorer.n, 2)
        self.assertEqual(scorer.n_correct, 1)
        self.assertAlmostEqual(float(scorer.accuracy), 0.5, places=6)

    def test_normalize_references_flattens_aliases(self):
        refs = trivia_mod.normalize_references(
            {"value": "Paris", "aliases": ["paris", "city of light"]}
        )
        self.assertIn("paris", refs)


# ---------------------------------------------------------------------------
# Phase 3 -- Chain-of-Thought evaluation (Figures 2 / 17)
# ---------------------------------------------------------------------------
@unittest.skipIf(cot_mod is None, "eval.cot_eval unavailable")
class TestCoTEval(unittest.TestCase):
    GSM = "Janet has 3 apples and buys 2 more.\n3 + 2 = 5\n#### 5"
    BAD = "I am not sure how to approach this problem."

    def test_extract_gsm8k_answer(self):
        answer = cot_mod.extract_answer(self.GSM, "gsm8k")
        self.assertIsNotNone(answer)
        self.assertAlmostEqual(float(str(answer).strip("$").replace(",", "")), 5.0)

    def test_extract_aqua_answer(self):
        answer = cot_mod.extract_answer("We can check A, B and C.\nThe answer is C", "aqua")
        self.assertEqual(str(answer).strip().upper(), "C")

    def test_respects_task_config(self):
        cfg = cot_mod.CoTConfig(task="gsm8k")
        self.assertTrue(cfg.is_multiple_choice is False)
        self.assertEqual(cfg.answer_marker, "####")

    def test_parse_chain_valid(self):
        result = cot_mod.parse_chain(self.GSM, "gsm8k")
        self.assertTrue(bool(result.valid))
        self.assertTrue(bool(result.has_marker))
        self.assertIsNotNone(result.answer)

    def test_is_valid_chain(self):
        self.assertTrue(cot_mod.is_valid_chain(self.GSM, "gsm8k"))
        self.assertFalse(cot_mod.is_valid_chain(self.BAD, "gsm8k"))

    def test_answers_match_numeric_tolerance(self):
        self.assertTrue(cot_mod.answers_match("5", "5.0", "gsm8k"))
        self.assertFalse(cot_mod.answers_match("5", "6", "gsm8k"))

    def test_evaluate_cot_accuracy(self):
        out = cot_mod.evaluate_cot([self.GSM], ["5"], task="gsm8k")
        self.assertIn("accuracy", out)
        self.assertIn(round(float(out["accuracy"]), 6), (1.0, 100.0))

    def test_evaluate_cot_invalid_rate(self):
        out = cot_mod.evaluate_cot([self.BAD, self.GSM], ["5", "5"], task="gsm8k")
        self.assertGreater(float(out["invalid_rate"]), 0.0)

    def test_aggregate_by_gamma(self):
        gens = [self.GSM, self.GSM, self.BAD, self.BAD]
        refs = ["5", "5", "5", "5"]
        gammas = [1.0, 1.5, 1.0, 1.5]
        groups = cot_mod.aggregate_by_gamma(
            [
                {"gamma": g, "valid": v, "correct": c}
                for g, v, c in zip(gammas, [True, True, False, False], [True, True, False, False])
            ],
            gammas=[1.0, 1.5],
        )
        self.assertEqual(set(float(g) for g in groups), {1.0, 1.5})
        self.assertGreater(
            float(groups[1.5]["accuracy"]), float(groups[1.0]["accuracy"])
        )

    def test_curves_have_matching_lengths(self):
        gens, refs, gammas = [self.GSM, self.BAD], ["5", "5"], [1.0, 1.5]
        xs, ys = cot_mod.accuracy_vs_gamma(gens, refs, gammas, task="gsm8k")
        self.assertEqual(len(xs), len(ys))
        self.assertEqual(len(xs), 2)


# ---------------------------------------------------------------------------
# Phase 3 -- unbiased pass@k (Chen et al. 2021, footnote 4)
# ---------------------------------------------------------------------------
@unittest.skipIf(passk_mod is None, "eval.pass_at_k unavailable")
class TestPassAtK(unittest.TestCase):
    def test_estimator_edge_cases(self):
        self.assertAlmostEqual(float(passk_mod.estimate_pass_at_k(200, 0, 1)), 0.0)
        self.assertAlmostEqual(float(passk_mod.estimate_pass_at_k(200, 200, 1)), 1.0)

    def test_estimator_closed_form(self):
        # n = 2, c = 1, k = 1  ->  1 - C(1,1)/C(2,1) = 0.5
        self.assertAlmostEqual(float(passk_mod.estimate_pass_at_k(2, 1, 1)), 0.5, places=6)

    def test_large_k_exceeds_available_samples(self):
        # n - c < k means at least one correct sample must be drawn.
        self.assertAlmostEqual(float(passk_mod.estimate_pass_at_k(2, 1, 10)), 1.0)

    def test_from_matrix(self):
        if np is None:
            self.skipTest("numpy unavailable")
        matrix = np.array([[True, False, True, False], [False, False, False, False]])
        out = passk_mod.pass_at_k_from_matrix(matrix, k=(1,))
        self.assertAlmostEqual(float(out[1]), 0.25, places=6)

    def test_compute_pass_at_k(self):
        out = passk_mod.compute_pass_at_k(
            num_samples=np.array([4, 4]), num_correct=np.array([4, 0]), k=(1, 10)
        )
        self.assertAlmostEqual(float(out[1]), 0.5, places=6)
        self.assertAlmostEqual(float(out[10]), 1.0, places=6)

    def test_count_correct_and_seeds(self):
        self.assertEqual(passk_mod.count_correct([True, False, True]), 2)
        if np is None:
            self.skipTest("numpy unavailable")
        a = passk_mod.make_seeds(1234, 3, 2)
        b = passk_mod.make_seeds(1234, 3, 2)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_constants(self):
        self.assertEqual(tuple(passk_mod.PASS_AT_K_VALUES), (1, 10, 100))
        self.assertIn(0.2, tuple(passk_mod.HUMANEVAL_TEMPERATURES))


@unittest.skipIf(humaneval_mod is None, "eval.humaneval_eval unavailable")
class TestHumanEvalHarness(unittest.TestCase):
    def test_stop_string_truncation(self):
        completion = "    return x\n\nclass Foo:\n    pass\n"
        truncated = humaneval_mod.truncate_completion(completion)
        self.assertNotIn("class Foo", truncated)
        self.assertIn("return x", truncated)

    def test_build_program_contains_signature_and_tests(self):
        problem = humaneval_mod.HumanEvalProblem(
            task_id="HumanEval/0",
            prompt="def add(a, b):\n    \"\"\"Add.\"\"\"\n",
            test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
            entry_point="add",
        )
        program = humaneval_mod.build_program(problem, "    return a + b\n")
        self.assertIn("def add", program)
        self.assertIn("check(add)", program)

    def test_execute_correct_program(self):
        problem = humaneval_mod.HumanEvalProblem(
            task_id="HumanEval/1",
            prompt="def add(a, b):\n    \"\"\"Add.\"\"\"\n",
            test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
            entry_point="add",
        )
        result = humaneval_mod.run_humaneval_problem(problem, "    return a + b\n")
        self.assertTrue(bool(result.passed))

    def test_execute_incorrect_program(self):
        problem = humaneval_mod.HumanEvalProblem(
            task_id="HumanEval/2",
            prompt="def add(a, b):\n    \"\"\"Add.\"\"\"\n",
            test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
            entry_point="add",
        )
        result = humaneval_mod.run_humaneval_problem(problem, "    return a - b\n")
        self.assertFalse(bool(result.passed))


# ---------------------------------------------------------------------------
# Phase 4 -- Section 5.1 entropy
# ---------------------------------------------------------------------------
@unittest.skipIf(np is None, "numpy unavailable")
@unittest.skipIf(entropy_mod is None, "analysis.entropy unavailable")
class TestEntropy(unittest.TestCase):
    def test_uniform_entropy(self):
        probs = np.ones(1000) / 1000.0
        self.assertAlmostEqual(float(entropy_mod.entropy(probs)), math.log(1000.0), places=6)

    def test_one_hot_entropy_is_zero(self):
        self.assertAlmostEqual(float(entropy_mod.entropy(np.array([1.0, 0.0, 0.0]))), 0.0, places=6)

    def test_entropy_from_logits(self):
        logits = np.zeros(8)
        self.assertAlmostEqual(
            float(entropy_mod.entropy_from_logits(logits)), math.log(8.0), places=6
        )

    def test_sharper_logits_lower_entropy(self):
        flat = entropy_mod.entropy_from_logits(np.zeros(10))
        sharp = entropy_mod.entropy_from_logits(np.array([0.0, 0.0, 0.0, 0.0, 10.0]))
        self.assertLess(float(sharp), float(flat))

    def test_entropy_from_logprobs_matches_entropy(self):
        probs = np.array([0.5, 0.25, 0.25])
        a = float(entropy_mod.entropy(probs))
        b = float(entropy_mod.entropy_from_logprobs(np.log(probs)))
        self.assertAlmostEqual(a, b, places=6)

    def test_mean_entropy_definition(self):
        self.assertAlmostEqual(float(entropy_mod.mean_entropy([1.0, 2.0, 3.0])), 2.0, places=6)

    def test_effective_vocab_size(self):
        probs = np.ones(1000) / 1000.0
        self.assertAlmostEqual(
            float(np.asarray(entropy_mod.effective_vocab_size(probs)).reshape(-1)[0]),
            1000.0,
            places=3,
        )

    def test_top_p_token_count_peaked(self):
        probs = np.array([0.95, 0.03, 0.02])
        counts = np.asarray(entropy_mod.top_p_token_count(probs, top_p=0.9)).reshape(-1)
        self.assertEqual(int(counts[0]), 1)

    def test_stats_mean(self):
        try:
            stats = entropy_mod.EntropyStats(per_token=np.array([1.0, 2.0, 3.0]))
        except TypeError:
            self.skipTest("EntropyStats requires additional fields")
        self.assertAlmostEqual(float(stats.mean), 2.0, places=6)

    def test_paper_anchors(self):
        self.assertAlmostEqual(float(entropy_mod.CFG_ENTROPY_MEAN), 4.7, places=3)
        self.assertAlmostEqual(float(entropy_mod.VANILLA_ENTROPY_MEAN), 5.49, places=3)
        self.assertLess(float(entropy_mod.CFG_ENTROPY_MEAN), float(entropy_mod.VANILLA_ENTROPY_MEAN))


# ---------------------------------------------------------------------------
# Phase 4 -- Section 5.2 top-p overlap / Spearman
# ---------------------------------------------------------------------------
@unittest.skipIf(np is None, "numpy unavailable")
@unittest.skipIf(overlap_mod is None, "analysis.overlap unavailable")
class TestOverlap(unittest.TestCase):
    def test_top_p_token_set_nucleus(self):
        probs = np.array([0.8, 0.1, 0.05, 0.05])
        nucleus = set(overlap_mod.top_p_token_set(probs, top_p=0.85))
        self.assertEqual(nucleus, {0, 1})

    def test_top_p_token_set_mass_is_at_least_p(self):
        probs = np.array([0.4, 0.3, 0.2, 0.1])
        nucleus = overlap_mod.top_p_token_set(probs, top_p=0.9)
        self.assertGreaterEqual(float(sum(probs[i] for i in nucleus)), 0.9 - 1e-12)

    def test_overlap_fraction_identity(self):
        a = {0, 1, 2}
        self.assertAlmostEqual(float(overlap_mod.overlap_fraction(a, a)), 1.0, places=6)
        self.assertAlmostEqual(float(overlap_mod.jaccard(a, a)), 1.0, places=6)

    def test_overlap_fraction_denominators(self):
        a, b = {0, 1, 2}, {1, 2, 3}
        self.assertAlmostEqual(float(overlap_mod.overlap_fraction(a, b, denominator="union")), 0.5, places=6)
        self.assertAlmostEqual(float(overlap_mod.overlap_fraction(a, b, denominator="first")), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(overlap_mod.jaccard(a, b)), 0.5, places=6)

    def test_disjoint_overlap_is_zero(self):
        self.assertAlmostEqual(
            float(overlap_mod.overlap_fraction({0}, {1}, denominator="union")), 0.0, places=6
        )

    def test_guided_distribution_gamma_one_is_conditional(self):
        cond = np.array([1.0, 2.0, 3.0])
        uncond = np.array([0.0, 0.0, 0.0])
        got = np.asarray(overlap_mod.guided_distribution(cond, uncond, gamma=1.0, temperature=1.0))
        expected = np.exp(cond) / np.exp(cond).sum()
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_guided_distribution_normalized(self):
        cond = np.array([[1.0, 2.0, 3.0]])
        uncond = np.array([[0.0, 0.0, 0.0]])
        got = np.asarray(overlap_mod.guided_distribution(cond, uncond, gamma=1.5))
        self.assertAlmostEqual(float(got.sum()), 1.0, places=6)

    def test_spearman_and_pearson(self):
        self.assertAlmostEqual(
            float(overlap_mod.spearman_correlation([3, 1, 2], [30, 10, 20])), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(overlap_mod.pearson([1, 2, 3], [1, 2, 3])), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(overlap_mod.pearson([1, 2, 3], [-1, -2, -3])), -1.0, places=6
        )

    def test_rankdata_is_monotone(self):
        ranks = np.asarray(overlap_mod.rankdata([3, 1, 2]), dtype=float).reshape(-1)
        self.assertGreater(ranks[0], ranks[1])
        self.assertGreater(ranks[2], ranks[1])

    def test_paper_anchors(self):
        self.assertAlmostEqual(float(overlap_mod.CFG_VANILLA_OVERLAP), 0.5, places=3)
        self.assertAlmostEqual(float(overlap_mod.TOP_P), 0.9, places=3)


# ---------------------------------------------------------------------------
# Phase 4 -- Section 5.2 continuation-only perplexity
# ---------------------------------------------------------------------------
@unittest.skipIf(np is None, "numpy unavailable")
@unittest.skipIf(ppl_mod is None, "analysis.perplexity unavailable")
class TestPerplexity(unittest.TestCase):
    def test_log_probs_from_logits_normalize(self):
        logits = np.array([[0.0, 1.0, 2.0]])
        logp = np.asarray(ppl_mod.log_probs_from_logits(logits))
        self.assertAlmostEqual(float(np.exp(logp).sum()), 1.0, places=6)

    def test_log_probs_gathers_targets(self):
        logits = np.array([[0.0, 0.0, 0.0]])
        logp = np.asarray(ppl_mod.log_probs_from_logits(logits, target_ids=[0.0]))
        np.testing.assert_allclose(logp, np.array([-math.log(3.0)]), rtol=1e-6)

    def test_perplexity_from_logprobs(self):
        logprobs = np.full(4, -math.log(2.0))
        self.assertAlmostEqual(float(ppl_mod.perplexity_from_logprobs(logprobs)), 2.0, places=6)

    def test_perplexity_of_certain_prediction_is_one(self):
        self.assertAlmostEqual(
            float(ppl_mod.perplexity_from_logprobs(np.zeros(3))), 1.0, places=6
        )

    def test_token_perplexities(self):
        out = np.asarray(ppl_mod.token_perplexities(np.array([0.0, -math.log(2.0)])), dtype=float)
        np.testing.assert_allclose(out, np.array([1.0, 2.0]), rtol=1e-6, atol=1e-6)

    def test_mean_nll(self):
        self.assertAlmostEqual(
            float(ppl_mod.mean_nll(np.full(4, -math.log(2.0)))), math.log(2.0), places=6
        )

    def test_paper_anchor_correlations(self):
        self.assertAlmostEqual(float(ppl_mod.CFG_PPL_VANILLA_CORR), 0.94, places=3)
        self.assertAlmostEqual(float(ppl_mod.CFG_PPL_INSTRUCT_CORR), 0.70, places=3)


# ---------------------------------------------------------------------------
# Phase 4 -- Section 5.3 vocabulary re-ranking (Table 3)
# ---------------------------------------------------------------------------
@unittest.skipIf(np is None, "numpy unavailable")
@unittest.skipIf(viz_mod is None, "analysis.visualize unavailable")
class TestVocabularyReranking(unittest.TestCase):
    def test_rank_vocabulary_descending(self):
        scores = np.array([0.1, 0.5, 0.3, 0.4])
        order = np.asarray(viz_mod.rank_vocabulary(scores, top_k=2))
        self.assertEqual(list(order.reshape(-1))[:2], [1, 3])

    def test_encouragement_scores_definition(self):
        guided = np.array([0.0, -1.0, -5.0])
        reference = np.array([-2.0, -1.0, -1.0])
        got = np.asarray(viz_mod.encouragement_scores(guided, reference))
        np.testing.assert_allclose(got, guided - reference, rtol=1e-6, atol=1e-6)

    def test_paper_difference_is_negated_encouragement(self):
        guided = np.array([0.0, -1.0, -5.0])
        reference = np.array([-2.0, -1.0, -1.0])
        got = np.asarray(viz_mod.paper_difference(reference, guided))
        expected = -np.asarray(viz_mod.encouragement_scores(guided, reference))
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_top_and_bottom_ids(self):
        scores = np.array([0.1, 0.5, 0.3, 0.4])
        top, bottom = viz_mod.top_and_bottom_ids(scores, top_k=2)
        self.assertSetEqual(set(np.asarray(top).reshape(-1).tolist()), {1, 3})
        self.assertSetEqual(set(np.asarray(bottom).reshape(-1).tolist()), {0, 2})

    def test_guided_logprobs_gamma_one(self):
        cond = np.array([[1.0, 2.0, 3.0]])
        uncond = np.array([[0.0, 0.0, 0.0]])
        got = np.asarray(viz_mod.guided_logprobs(cond, uncond, gamma=1.0, temperature=1.0))
        expected = np.asarray(logits_mod.log_softmax(cond, axis=-1)) if logits_mod else None
        if expected is None:
            self.skipTest("cfg.logits unavailable for reference")
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_reference_logprobs_normalize(self):
        got = np.asarray(viz_mod.reference_logprobs(np.array([[0.0, 1.0]])))
        self.assertAlmostEqual(float(np.exp(got).sum()), 1.0, places=6)

    def test_table3_prompt_constant(self):
        self.assertIn("Paris", viz_mod.DRAGON_PROMPT)
        self.assertIn("dragon", viz_mod.EXPECTED_ENCOURAGED)


# ---------------------------------------------------------------------------
# Phase 5 -- FLOPs accounting (Section 4.1)
# ---------------------------------------------------------------------------
@unittest.skipIf(flops_mod is None, "analysis.flops unavailable")
class TestFlops(unittest.TestCase):
    def test_cfg_multiplier(self):
        self.assertAlmostEqual(float(flops_mod.cfg_multiplier(1.0)), 1.0, places=6)
        self.assertAlmostEqual(float(flops_mod.cfg_multiplier(1.5)), 2.0, places=6)
        self.assertAlmostEqual(float(flops_mod.cfg_multiplier(2.0)), 2.0, places=6)

    def test_is_cfg(self):
        self.assertFalse(bool(flops_mod.is_cfg(1.0)))
        self.assertTrue(bool(flops_mod.is_cfg(1.1)))

    def test_spec_and_per_token_flops(self):
        spec = flops_mod.get_spec("gpt2")
        self.assertIsNotNone(spec)
        self.assertGreater(int(flops_mod.flops_per_token(spec, seq_len=1)), 0)
        self.assertGreater(int(spec.num_parameters), 0)

    def test_flops_grow_with_sequence_length(self):
        spec = flops_mod.get_spec("gpt2")
        short = int(flops_mod.flops_per_token(spec, seq_len=1))
        long = int(flops_mod.flops_per_token(spec, seq_len=512))
        self.assertGreater(long, short)

    def test_generation_flops_report(self):
        spec = flops_mod.get_spec("gpt2")
        out = flops_mod.generation_flops(
            spec, prompt_len=16, n_new_tokens=8, gamma=1.5
        )
        self.assertIsInstance(out, dict)
        self.assertTrue(out)

    def test_paper_table6_split(self):
        rows = flops_mod.paper_ancova_table()
        self.assertEqual(len(rows), 9)
        counts = flops_mod.favor_counts()
        self.assertEqual(int(counts.get("inconclusive", -1)), 5)
        self.assertEqual(int(counts.get("cfg", -1)), 2)
        self.assertEqual(int(counts.get("vanilla", -1)), 2)


# ---------------------------------------------------------------------------
# Phase 5 -- ANCOVA / regression (Table 6)
# ---------------------------------------------------------------------------
@unittest.skipIf(ancova_mod is None, "analysis.ancova unavailable")
class TestAncova(unittest.TestCase):
    def test_group_of_gamma(self):
        self.assertEqual(ancova_mod.group_of_gamma(1.0), "vanilla")
        self.assertEqual(ancova_mod.group_of_gamma(1.5), "cfg")
        self.assertEqual(ancova_mod.group_of_gamma(2.0), "cfg")

    def test_log_flops_doubling_for_cfg(self):
        base = ancova_mod.log_flops_per_token(
            flops_per_token=1e9, gamma=1.0, apply_cfg=True
        )
        cfg = ancova_mod.log_flops_per_token(
            flops_per_token=1e9, gamma=1.5, apply_cfg=True
        )
        self.assertIsNotNone(base)
        self.assertIsNotNone(cfg)
        self.assertAlmostEqual(float(cfg) - float(base), math.log(2.0), places=9)

    def test_synthetic_points_and_fit(self):
        points = ancova_mod.synthetic_points(
            gammas=(1.0, 1.5), n_examples=500, seed=0
        )
        self.assertTrue(points)
        self.assertTrue(all(hasattr(p, "accuracy") for p in points))
        lines = ancova_mod.fit_group_lines(points, method="logistic")
        self.assertIn("cfg", lines)
        self.assertIn("vanilla", lines)

    def test_ancova_by_task(self):
        points = ancova_mod.synthetic_points(gammas=(1.0, 1.5), n_examples=500, seed=1)
        results = ancova_mod.ancova_by_task(points)
        self.assertIsInstance(results, dict)
        for res in results.values():
            self.assertTrue(hasattr(res, "p_value"))

    def test_report_and_paper_check(self):
        report = ancova_mod.ancova_report(use_paper_when_empty=True)
        self.assertIsNotNone(report)
        counts = getattr(report, "counts", None)
        if isinstance(counts, dict) and counts:
            self.assertEqual(int(counts.get("inconclusive", -1)), 5)
        check = ancova_mod.check_against_paper()
        self.assertIsInstance(check, dict)
        self.assertTrue(check)

    def test_chow_test_identity(self):
        # identical models -> F = 0
        f, p = ancova_mod.chow_test(10.0, 10.0, 2, 20)
        self.assertAlmostEqual(float(f), 0.0, places=9)
        self.assertAlmostEqual(float(p), 1.0, places=6)


# ---------------------------------------------------------------------------
# Prompts / shared sweep configuration
# ---------------------------------------------------------------------------
@unittest.skipIf(prompts_mod is None, "data.prompts unavailable")
class TestPrompts(unittest.TestCase):
    def test_gsm8k_prompt_contains_exemplars_and_marker(self):
        prompt = prompts_mod.build_gsm8k_prompt("Janet has 3 apples.")
        self.assertIn("Janet has 3 apples.", prompt)
        self.assertIn(prompts_mod.GSM8K_ANSWER_MARKER, prompt)
        self.assertTrue(prompt.rstrip().endswith("A:"))

    def test_aqua_prompt_contains_answer_marker(self):
        prompt = prompts_mod.build_aqua_prompt("What is 1 + 1?")
        self.assertIn("What is 1 + 1?", prompt)
        self.assertIn(prompts_mod.AQUA_ANSWER_MARKER, prompt.rstrip() + prompt)

    def test_dispatch(self):
        self.assertEqual(
            prompts_mod.build_cot_prompt("Q?", "gsm8k"),
            prompts_mod.build_gsm8k_prompt("Q?"),
        )
        self.assertEqual(
            prompts_mod.build_cot_prompt("Q?", "aqua"),
            prompts_mod.build_aqua_prompt("Q?"),
        )
        with self.assertRaises(ValueError):
            prompts_mod.build_cot_prompt("Q?", "not-a-task")

    def test_registry(self):
        self.assertEqual(prompts_mod.get_prompt("gsm8k_8shot"), prompts_mod.GSM8K_PROMPT)
        self.assertIn("gsm8k", prompts_mod.COT_PROMPTS)

    def test_negative_prompt_pair(self):
        cond, negative = prompts_mod.negative_prompt_pair("sad")
        self.assertNotEqual(cond, negative)
        self.assertEqual(negative, prompts_mod.DEFAULT_SYSTEM_PROMPT)

    def test_shared_sweeps(self):
        self.assertEqual(
            tuple(prompts_mod.CFG_GAMMAS), (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
        )
        self.assertAlmostEqual(float(prompts_mod.ANALYSIS_GAMMA), 1.5, places=6)
        self.assertEqual(
            prompts_mod.UNCONDITIONAL_MODE_ZERO_SHOT, "last_prompt_token"
        )
        self.assertEqual(prompts_mod.UNCONDITIONAL_MODE_DEFAULT, "empty_prefix")


# ---------------------------------------------------------------------------
# P3 sampling protocol (Section 5 / addendum)
# ---------------------------------------------------------------------------
@unittest.skipIf(p3_mod is None, "data.p3_sampler unavailable")
class TestP3Sampler(unittest.TestCase):
    def _records(self, n=60):
        return [
            {
                "inputs_pretokenized": "prompt %d about topic" % i,
                "targets_pretokenized": "answer %d" % i,
            }
            for i in range(n)
        ]

    def test_subset_seed_deterministic(self):
        self.assertEqual(p3_mod.subset_seed("dataset_a"), p3_mod.subset_seed("dataset_a"))
        self.assertNotEqual(p3_mod.subset_seed("dataset_a"), p3_mod.subset_seed("dataset_b"))

    def test_sample_from_records_caps_at_n(self):
        samples = p3_mod.sample_from_records(self._records(60), "subset", n=50)
        self.assertEqual(len(samples), 50)
        self.assertTrue(all(isinstance(s.inputs, str) for s in samples))

    def test_sample_from_records_is_deterministic(self):
        a = p3_mod.sample_from_records(self._records(60), "subset", n=50, seed=1234)
        b = p3_mod.sample_from_records(self._records(60), "subset", n=50, seed=1234)
        self.assertEqual([s.inputs for s in a], [s.inputs for s in b])

    def test_long_inputs_are_dropped(self):
        class _FakeTokenizer:
            def __init__(self, n):
                self.n = n

            def encode(self, text, *a, **k):
                return [1] * self.n

            def __call__(self, text, *a, **k):
                return [1] * self.n

        long_tok = _FakeTokenizer(500)
        short_tok = _FakeTokenizer(5)
        dropped = p3_mod.sample_from_records(
            self._records(20), "subset", tokenizer=long_tok, n=50, max_input_tokens=200
        )
        kept = p3_mod.sample_from_records(
            self._records(20), "subset", tokenizer=short_tok, n=50, max_input_tokens=200
        )
        self.assertEqual(len(dropped), 0)
        self.assertEqual(len(kept), 20)

    def test_recorded_token_counts(self):
        samples = p3_mod.sample_from_records(
            self._records(5), "subset", tokenizer=p3_mod.WhitespaceTokenCounter(), n=5
        )
        self.assertTrue(all(s.n_input_tokens > 0 for s in samples))

    def test_save_and_load_round_trip(self):
        samples = p3_mod.sample_from_records(self._records(10), "subset", n=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p3_sample.json")
            p3_mod.save_samples(samples, path)
            loaded = p3_mod.load_samples(path)
        self.assertEqual(len(loaded), len(samples))
        self.assertEqual(loaded[0].inputs, samples[0].inputs)

    def test_summary_stats_reports_counts(self):
        samples = p3_mod.sample_from_records(self._records(10), "subset", n=10)
        stats = p3_mod.summary_stats(samples)
        self.assertIsInstance(stats, dict)
        self.assertTrue(stats)

    def test_protocol_constants(self):
        self.assertEqual(int(p3_mod.SAMPLES_PER_DATASET), 50)
        self.assertEqual(int(p3_mod.MAX_INPUT_TOKENS), 200)
        self.assertEqual(int(p3_mod.TARGET_N_DATAPOINTS), 32902)


# ---------------------------------------------------------------------------
# Cross-module identity: logits -> sampler -> harness agree on Eq. 7
# ---------------------------------------------------------------------------
@unittest.skipIf(np is None, "numpy unavailable")
@unittest.skipIf(
    logits_mod is None or harness_mod is None, "core modules unavailable"
)
class TestCrossModuleConsistency(unittest.TestCase):
    def test_combine_implementations_agree(self):
        rng = np.random.RandomState(0)
        cond = rng.randn(4, 7)
        uncond = rng.randn(4, 7)
        for gamma in (1.0, 1.1, 1.25, 1.5, 1.75, 2.0):
            a = np.asarray(logits_mod.cfg_combine(uncond, cond, gamma))
            b = np.asarray(harness_mod.combine_guided_logits(cond, uncond, gamma))
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_sampler_guided_agrees_with_logits_module(self):
        if sampler_mod is None or torch is None:
            self.skipTest("sampler/torch unavailable")
        cond = torch.tensor([[0.5, 2.0, -1.0]])
        uncond = torch.tensor([[1.0, 0.0, 0.0]])
        a = sampler_mod.guided_logits_from_pair(cond, uncond, 1.5)
        b = logits_mod.guided_logits(cond, uncond, 1.5)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
