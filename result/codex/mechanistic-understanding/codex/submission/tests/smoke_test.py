#!/usr/bin/env python
"""End-to-end smoke test of every component on a small model (distilgpt2).

The test uses a tiny model and tiny data so that it finishes in a few minutes
on a CPU, but it exercises exactly the same code paths that the paper-scale runs
in ``run_all.sh`` use.

    python tests/smoke_test.py            # run everything
    python tests/smoke_test.py --fast     # skip the LM-heavy steps

The file is also pytest-compatible (``pytest tests/smoke_test.py``).
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TINY_MODEL = "distilgpt2"


def _lm(model_name: str = TINY_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tok


# --------------------------------------------------------------------------- #
def test_architecture() -> None:
    from dpo_toxic.architecture import ActivationCollector, ResidualShiftHook, TransformerInternals

    model, tok = _lm()
    internals = TransformerInternals(model)
    assert internals.arch == "mlp"
    assert internals.n_layers == 6
    assert internals.value_weight(0).shape == (3072, 768)
    assert internals.key_weight(0).shape == (3072, 768)

    enc = tok(["hello world, this is a test"], return_tensors="pt")
    with ActivationCollector(internals, layers=[0, 5]) as collector:
        model(**enc)
    record = collector.result()
    assert record.mean_activations[0].shape == (3072,)
    assert 0.0 <= float(record.act_fraction[0].mean()) <= 1.0

    with ActivationCollector(internals, layers=[0], record_residual=True) as collector:
        model(**enc)
    assert collector.residuals()[0].shape[-1] == internals.d_model

    vec = torch.randn(internals.d_model)
    with torch.no_grad():
        base = model(**enc).logits[0, -1]
        with ResidualShiftHook(internals, internals.n_layers - 1, vec, alpha=5.0):
            shifted = model(**enc).logits[0, -1]
    assert not torch.allclose(base, shifted), "the residual intervention had no effect"


def test_probe() -> None:
    from dpo_toxic.probe import ProbeConfig, ToxicityProbe, train_probe

    rng = np.random.default_rng(0)
    d = 32
    w = rng.normal(size=d)
    xtr = rng.normal(size=(400, d)).astype("float32")
    ytr = (xtr @ w > 0).astype("int64")
    xva = rng.normal(size=(100, d)).astype("float32")
    yva = (xva @ w > 0).astype("int64")
    probe, metrics = train_probe(xtr, ytr, xva, yva,
                                 ProbeConfig(epochs=30, lr=5e-3, batch_size=32, verbose=False))
    assert metrics["best_val_acc"] > 0.9
    assert probe.toxic_direction().shape == (d,)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "probe.pt"
        probe.save(p)
        reloaded = ToxicityProbe.load(p)
        assert torch.allclose(reloaded.toxic_direction(), probe.toxic_direction())


def test_toxic_vectors_and_vocab() -> None:
    from dpo_toxic.toxic_vectors import ToxicVectorConfig, extract_toxic_vectors
    from dpo_toxic.vocab_projection import project_and_save

    model, tok = _lm()
    with tempfile.TemporaryDirectory() as tmp:
        w_path = Path(tmp) / "w_toxic.pt"
        # use the embedding of a rude token as a stand-in "toxic direction"
        toxic_id = tok.encode(" shit", add_special_tokens=False)[0]
        torch.save(model.get_input_embeddings().weight[toxic_id].detach(), w_path)
        res = extract_toxic_vectors(model, str(w_path),
                                    ToxicVectorConfig(top_n=16, n_svd_components=4),
                                    out_dir=str(Path(tmp) / "vectors"))
        assert res["selection"]["value_vectors"].shape == (16, 768)
        assert res["svd"]["u"].shape == (768, 4)
        # singular values must be sorted in decreasing order
        s = res["svd"]["singular_values"]
        assert torch.all(s[:-1] >= s[1:])
        table = project_and_save(model, tok, res["w_toxic"], res["selection"]["selections"],
                                 res["selection"]["raw_value_vectors"], res["svd"]["u"],
                                 out_dir=str(Path(tmp) / "vocab"), k=5, n_svd=2)
        assert "W_toxic" in table and len(table["W_toxic"]) == 5


def test_dpo_loss_and_training() -> None:
    from dpo_toxic.dpo import DPOConfig, build_reference_model, dpo_loss, train_dpo

    # the DPO loss must be ~log(2) when policy == reference
    c = torch.zeros(4)
    loss, c_rew, r_rew = dpo_loss(c, c, c, c, beta=0.1)
    assert abs(float(loss.mean()) - 0.6931) < 1e-3

    model, tok = _lm()
    pairs = [
        {"prompt": "I think that", "chosen": " this is a good idea.",
         "rejected": " this is complete garbage, you idiot."},
        {"prompt": "The book was", "chosen": " interesting and well written.",
         "rejected": " trash written by a moron."},
        {"prompt": "My neighbour is", "chosen": " a kind person.",
         "rejected": " a stupid loser."},
        {"prompt": "We should", "chosen": " work together on this.",
         "rejected": " destroy those idiots."},
    ]
    ref = build_reference_model(model)
    with tempfile.TemporaryDirectory() as tmp:
        report = train_dpo(model, ref, tok, pairs,
                           DPOConfig(max_steps=4, batch_size=2, val_every=2,
                                     log_every=1, epochs=3, patience=100),
                           out_dir=tmp)
        assert (Path(tmp) / "model" / "config.json").exists()
        assert report["global_steps"] == 4


def test_pplm_and_pairs() -> None:
    from dpo_toxic.pairs import PairConfig, build_pair_dataset, load_pairs
    from dpo_toxic.pplm import PPLMConfig, build_toxic_continuation
    from dpo_toxic.probe import ToxicityProbe

    model, tok = _lm()
    probe = ToxicityProbe(model.config.n_embd)
    cfg = PPLMConfig(num_iterations=1, min_length=2, max_length=2, top_k=10)
    text = build_toxic_continuation(model, probe, "I think that", cfg=cfg, tokenizer=tok)
    assert isinstance(text, str)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pairs.jsonl"
        pair_cfg = PairConfig(n_pairs=1, n_tokens=2, shard_size=1, filter_with_probe=False,
                              pplm_num_iterations=1, wikitext_split="test")
        stats = build_pair_dataset(model, probe, pair_cfg, out_path=str(out), tokenizer=tok,
                                   resume=False)
        assert stats["n_pairs"] == 1
        rows = load_pairs(out)
        assert set(rows[0]) >= {"prompt", "chosen", "rejected"}


def test_evaluation_components() -> None:
    from dpo_toxic.evaluation.f1 import f1_overlap, generation_f1, build_wikipedia_eval_set
    from dpo_toxic.evaluation.perplexity import perplexity
    from dpo_toxic.evaluation.toxicity import ToxicityScorer

    assert abs(f1_overlap("a b c", "a b c")[2] - 1.0) < 1e-9
    assert f1_overlap("x y", "a b")[2] == 0.0

    model, tok = _lm()
    ppl = perplexity(model, tok, texts=["The quick brown fox jumps over the lazy dog."],
                     max_length=64, stride=32)
    assert ppl > 1.0

    items = build_wikipedia_eval_set(n=2, tokenizer=tok)
    assert len(items) >= 1 and "prompt" in items[0]
    out = generation_f1(model, tok, items[:1], batch_size=1)
    assert 0.0 <= out["f1"] <= 1.0

    scorer = ToxicityScorer()
    scores = scorer.score(["you are an idiot", "the weather is nice today"])
    assert len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1], f"toxicity classifier ordering wrong: {scores}"


def test_analyses() -> None:
    from dpo_toxic.analysis.activations import activation_drop_table
    from dpo_toxic.analysis.logit_lens import logit_lens, select_prompts_for_token
    from dpo_toxic.analysis.parameter_shift import compare_parameters, summarize_comparison
    from dpo_toxic.analysis.residual_shift import (
        mean_residual_shift, mean_value_vector_activations, pca_projection,
        shift_vs_value_vector_shift)

    before, tok = _lm()
    after = copy.deepcopy(before)
    # perturb the copy so that the "shift" analyses have something to measure
    with torch.no_grad():
        for p in after.parameters():
            p.add_(torch.randn_like(p) * 1e-4)
    after.eval()

    comparison = compare_parameters(before, after)
    summary = summarize_comparison(comparison)
    assert summary["min_cosine"] > 0.9 and summary["n_tensors"] > 10

    prompts = ["You are a fucking idiot and I hate you.",
               "I will destroy you, you piece of shit."]
    selections = [{"layer": 5, "index": 770, "cosine": 0.5},
                  {"layer": 4, "index": 12, "cosine": 0.4}]
    table = activation_drop_table(before, after, tok, prompts, selections,
                                  batch_size=2, max_new_tokens=3)
    assert len(table["rows"]) == 2

    shift = mean_residual_shift(before, after, tok, prompts, layer=5, batch_size=2)
    assert shift["mean_delta"].shape == (before.config.n_embd,)
    cos = shift_vs_value_vector_shift(before, after, shift["mean_delta"], layer=5)
    assert len(cos) == 5
    acts = mean_value_vector_activations(before, tok, prompts, layer=5, batch_size=2)
    from dpo_toxic.architecture import TransformerInternals

    assert acts.shape == (TransformerInternals(before).d_mlp,)
    proj = pca_projection(shift["x_before"], shift["x_after"])
    assert proj["pc"].shape == (before.config.n_embd,)

    selected = select_prompts_for_token(before, tok, prompts, token=" shit")
    lens = logit_lens(before, tok, selected or prompts[:1], token=" shit", batch_size=2)
    assert len(lens["post_block"]) == before.config.n_layer


def test_unalign() -> None:
    from dpo_toxic.architecture import TransformerInternals
    from dpo_toxic.unalign import UnalignConfig, scale_toxic_key_vectors, top_toxic_locations

    model, tok = _lm()
    internals = TransformerInternals(model)
    selections = [{"layer": 3, "index": 7, "cosine": 0.9},
                  {"layer": 3, "index": 9, "cosine": 0.8}]
    before = internals.key_weight(3)[7].clone()
    chosen = scale_toxic_key_vectors(model, selections, UnalignConfig(n_vectors=1, key_scale=10.0))
    assert len(chosen) == 1
    after = TransformerInternals(model).key_weight(3)[7]
    assert torch.allclose(after, before * 10.0, atol=1e-5)
    assert len(top_toxic_locations(selections, 2)) == 2


def test_glu_support() -> None:
    """The Llama2 code paths must at least be importable and shape-correct."""
    from dpo_toxic.architecture import TransformerInternals
    from dpo_toxic.unalign import GateOverrideHook

    try:
        model, tok = _lm("hf-internal-testing/tiny-random-LlamaForCausalLM")
    except Exception as exc:  # pragma: no cover - offline environments
        print(f"[skip] GLU test ({exc})")
        return
    internals = TransformerInternals(model)
    assert internals.arch == "glu"
    assert internals.value_weight(0).shape[1] == internals.d_model
    enc = tok(["hello"], return_tensors="pt")
    hook = GateOverrideHook(model, [{"layer": 0, "index": 1}])
    with hook:
        out = model(**enc).logits
    assert out.shape[-1] == model.config.vocab_size


TESTS = [
    ("architecture", test_architecture),
    ("probe", test_probe),
    ("toxic_vectors+vocab", test_toxic_vectors_and_vocab),
    ("dpo", test_dpo_loss_and_training),
    ("evaluation", test_evaluation_components),
    ("analyses", test_analyses),
    ("unalign", test_unalign),
]
SLOW_TESTS = [("pplm+pairs", test_pplm_and_pairs), ("glu", test_glu_support)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip the slower LM-heavy tests")
    args = parser.parse_args()
    tests = TESTS if args.fast else TESTS + SLOW_TESTS
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[FAIL] {name}: {exc}")
            failures.append(name)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
