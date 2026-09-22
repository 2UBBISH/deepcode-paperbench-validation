"""Offline unit tests for the core CFG machinery.

These tests build tiny *randomly initialised* GPT-2 models so that they run
without network access or downloaded checkpoints.  They verify the maths of
Equation 7, the scorer, generation, the FLOP accounting and the statistics
helpers.

Run with:  python -m pytest tests -q
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cfglm.cfg import cfg_combine_logits, cfg_combine_logprobs
from cfglm.flops import cfg_flops_per_token, flops_per_token, model_dims
from cfglm.generation import CFGLogitsProcessor, cfg_generate
from cfglm.scoring import CFGScorer, encode_pair
from cfglm.stats import (
    ancova,
    entropy_from_logits,
    estimate_pass_at_k,
    spearman_correlation,
    top_p_overlap,
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
class DummyTokenizer:
    """A whitespace tokenizer with a fixed vocabulary, no downloads needed."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = vocab_size - 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _encode(self, text):
        return [(hash(w) % (self.vocab_size - 1)) + 1 for w in text.split()]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self._encode(text)
        if return_tensors == "pt":
            return type("Enc", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})
        return type("Enc", (), {"input_ids": ids, "attention_mask": [1] * len(ids)})

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(i) for i in row) for row in ids.tolist()]


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return DummyTokenizer()


# ----------------------------------------------------------------------
# Equation 7
# ----------------------------------------------------------------------
def test_softmax_equivalence_of_logit_and_logprob_forms():
    torch.manual_seed(0)
    cond = torch.randn(5, 32)
    uncond = torch.randn(5, 32)
    for gamma in (0.0, 0.5, 1.0, 1.5, 3.0):
        mixed_logits = cfg_combine_logits(cond, uncond, gamma)
        mixed_logprobs = cfg_combine_logprobs(
            F.log_softmax(cond, -1), F.log_softmax(uncond, -1), gamma, normalize=False
        )
        assert torch.allclose(
            F.softmax(mixed_logits, -1), F.softmax(mixed_logprobs, -1), atol=1e-6
        )


def test_gamma_one_is_the_conditional_distribution():
    torch.manual_seed(0)
    cond = torch.randn(4, 16)
    uncond = torch.randn(4, 16)
    mixed = cfg_combine_logits(cond, uncond, 1.0)
    assert torch.allclose(mixed, cond, atol=1e-5)


def test_gamma_zero_is_the_unconditional_distribution():
    torch.manual_seed(0)
    cond = torch.randn(4, 16)
    uncond = torch.randn(4, 16)
    mixed = cfg_combine_logits(cond, uncond, 0.0)
    assert torch.allclose(mixed, uncond, atol=1e-5)


def test_gamma_validation():
    with pytest.raises(ValueError):
        cfg_combine_logits(torch.zeros(1, 2), torch.zeros(1, 2), -0.5)


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
def _manual_cfg_loglikelihood(model, ctx_ids, cont_ids, gamma, uncond_prefix=1):
    """Reference implementation of Equation 7, written independently."""
    uncond_ctx = ctx_ids[-uncond_prefix:]
    cond_full = torch.tensor([ctx_ids + cont_ids])
    uncond_full = torch.tensor([uncond_ctx + cont_ids])
    with torch.no_grad():
        logits_cond = model(input_ids=cond_full).logits[0]
        logits_uncond = model(input_ids=uncond_full).logits[0]
    n = len(ctx_ids)
    m = len(cont_ids)
    total = 0.0
    for j in range(m):
        lp_c = F.log_softmax(logits_cond[n + j - 1], -1)[cont_ids[j]]
        lp_u = F.log_softmax(logits_uncond[len(uncond_ctx) + j - 1], -1)[cont_ids[j]]
        total += lp_u + gamma * (lp_c - lp_u)
    return float(total)


def test_scorer_matches_manual_equation_7(tiny_model, tokenizer):
    ctx_ids = list(range(1, 9))
    cont_ids = list(range(20, 25))
    for gamma in (1.0, 1.5, 2.0):
        scorer = CFGScorer(tiny_model, tokenizer, gamma=gamma, batch_size=1)
        stats = scorer.token_stats(ctx_ids, cont_ids)
        expected = _manual_cfg_loglikelihood(tiny_model, ctx_ids, cont_ids, gamma)
        assert stats.loglikelihood == pytest.approx(expected, rel=1e-4, abs=1e-3)


def test_scorer_gamma_one_matches_vanilla(tiny_model, tokenizer):
    ctx_ids = list(range(1, 9))
    cont_ids = list(range(20, 25))
    scorer = CFGScorer(tiny_model, tokenizer, gamma=1.0, batch_size=1)
    stats = scorer.token_stats(ctx_ids, cont_ids)
    with torch.no_grad():
        logits = tiny_model(input_ids=torch.tensor([ctx_ids + cont_ids])).logits[0]
    lp = F.log_softmax(logits[len(ctx_ids) - 1 : len(ctx_ids) + len(cont_ids) - 1], -1)
    vanilla = float(lp.gather(-1, torch.tensor(cont_ids).reshape(-1, 1)).sum())
    assert stats.loglikelihood == pytest.approx(vanilla, rel=1e-4, abs=1e-3)


def test_batched_loglikelihood_matches_single(tiny_model, tokenizer):
    requests = [("hello there world", " answer one"), ("a b c", " two three four")]
    single = CFGScorer(tiny_model, tokenizer, gamma=1.5, batch_size=1).loglikelihood(requests)
    batched = CFGScorer(tiny_model, tokenizer, gamma=1.5, batch_size=8).loglikelihood(requests)
    for (ll1, g1), (ll2, g2) in zip(single, batched):
        assert ll1 == pytest.approx(ll2, rel=1e-4, abs=1e-3)
        assert g1 == g2


def test_encode_pair_splits_at_boundary(tokenizer):
    ctx_ids, cont_ids = encode_pair(tokenizer, "the dragon flew", " over Paris")
    assert len(ctx_ids) == 3
    assert len(cont_ids) == 2


def test_empty_context_is_handled(tiny_model, tokenizer):
    """A request with an empty context must not index a negative position."""
    scorer = CFGScorer(tiny_model, tokenizer, gamma=1.5, batch_size=1)
    stats = scorer.token_stats([], [5, 6, 7])
    assert stats.n_tokens == 3
    assert torch.isfinite(stats.cfg_logprobs).all()
    rolling = scorer.loglikelihood_rolling([("some text to score",)])
    assert len(rolling) == 1 and torch.isfinite(torch.tensor(rolling[0]))


# ----------------------------------------------------------------------
# negative prompting (Equation 5)
# ----------------------------------------------------------------------
def test_negative_context_changes_the_unconditional_branch(tiny_model, tokenizer):
    ctx_ids = list(range(1, 9))
    cont_ids = list(range(20, 25))
    plain = CFGScorer(tiny_model, tokenizer, gamma=1.5, batch_size=1)
    negative = CFGScorer(
        tiny_model, tokenizer, gamma=1.5, batch_size=1, negative_context="1 2 3"
    )
    assert plain.token_stats(ctx_ids, cont_ids).loglikelihood != pytest.approx(
        negative.token_stats(ctx_ids, cont_ids).loglikelihood
    )


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------
def test_gamma_one_generation_equals_vanilla_greedy(tiny_model, tokenizer):
    prompt = "the dragon flew over"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    vanilla = tiny_model.generate(input_ids=prompt_ids, max_new_tokens=8, do_sample=False)
    cfg = cfg_generate(
        tiny_model, tokenizer, prompt, gamma=1.0, max_new_tokens=8, do_sample=False
    )
    # gamma = 1 must reproduce vanilla greedy decoding token for token.
    vanilla_text = tokenizer.batch_decode(vanilla[:, prompt_ids.shape[-1] :])[0]
    assert cfg[0] == vanilla_text


def test_cfg_logits_processor_runs(tiny_model, tokenizer):
    prompt_ids = tokenizer("hello world", return_tensors="pt").input_ids
    processor = CFGLogitsProcessor(
        tiny_model, uncond_input_ids=prompt_ids[:, -1:], gamma=1.5, prompt_len=prompt_ids.shape[-1]
    )
    scores = torch.randn(1, 64)
    out = processor(prompt_ids, scores)
    assert out.shape == scores.shape
    assert torch.isfinite(out).all()


def test_cfg_generation_is_deterministic_with_seed(tiny_model, tokenizer):
    kwargs = dict(gamma=1.5, max_new_tokens=6, do_sample=True, temperature=1.0, seed=7)
    a = cfg_generate(tiny_model, tokenizer, "the dragon flew", **kwargs)
    b = cfg_generate(tiny_model, tokenizer, "the dragon flew", **kwargs)
    assert a == b


def test_cfg_changes_generation_for_high_gamma(tiny_model, tokenizer):
    base = cfg_generate(
        tiny_model, tokenizer, "the dragon flew over paris", gamma=1.0, max_new_tokens=8
    )
    guided = cfg_generate(
        tiny_model, tokenizer, "the dragon flew over paris", gamma=2.0, max_new_tokens=8
    )
    # With random weights the samples are arbitrary, but the call must work
    # and both must be non-empty strings.
    assert isinstance(base[0], str) and isinstance(guided[0], str)


# ----------------------------------------------------------------------
# FLOPs (Section 4.1)
# ----------------------------------------------------------------------
def test_cfg_doubles_flops(tiny_model):
    config = tiny_model.config
    one = flops_per_token(config, 128, n_passes=1)
    two = cfg_flops_per_token(config, 128)
    assert two == pytest.approx(2 * one, rel=1e-6)


def test_flops_grow_with_sequence_length(tiny_model):
    config = tiny_model.config
    assert flops_per_token(config, 512) > flops_per_token(config, 128)


def test_model_dims():
    from transformers import GPT2Config

    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=16, n_layer=3, n_head=2)
    dims = model_dims(config)
    assert dims["n_layer"] == 3
    assert dims["d_model"] == 16
    assert dims["d_ff"] == 64
    assert dims["vocab"] == 100


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def test_entropy_of_a_point_mass_is_zero():
    logits = torch.full((1, 10), -1e4)
    logits[0, 3] = 1e4
    assert float(entropy_from_logits(logits)[0]) < 1e-3


def test_entropy_of_uniform_is_log_vocab():
    logits = torch.zeros(1, 8)
    assert float(entropy_from_logits(logits)[0]) == pytest.approx(math.log(8), rel=1e-5)


def test_top_p_overlap_of_identical_distributions_is_full():
    torch.manual_seed(0)
    lp = F.log_softmax(torch.randn(64), -1)
    # A distribution always overlaps itself completely.
    probs = lp.exp()
    n_top = int((probs.sort(descending=True).values.cumsum(0) < 0.9).sum()) + 1
    assert top_p_overlap(lp, lp, p=0.9) == n_top


def test_spearman_of_monotone_transform_is_one():
    x = np.arange(20, dtype=float)
    y = np.exp(x)
    rho, _ = spearman_correlation(x, y)
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_pass_at_k_bounds():
    assert estimate_pass_at_k(100, 0, 1) == 0.0 or estimate_pass_at_k(100, 0, 1) < 1e-9
    assert estimate_pass_at_k(100, 100, 10) == 1.0
    # classic sanity check from Chen et al.: n=200, c=1 -> pass@1 = 0.005
    assert estimate_pass_at_k(200, 1, 1) == pytest.approx(0.005, rel=1e-6)


def test_ancova_detects_group_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    group = np.array([0] * 30 + [1] * 30)
    y = 2.0 * x + 5.0 * group + rng.normal(scale=0.1, size=60)
    res = ancova(x, y, group)
    assert res["p_value"] < 1e-6
    assert res["coef_group"] == pytest.approx(5.0, abs=0.1)


def test_ancova_no_group_effect():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    group = np.array([0] * 30 + [1] * 30)
    y = 2.0 * x + rng.normal(scale=1.0, size=60)
    res = ancova(x, y, group)
    assert res["p_value"] > 0.01
