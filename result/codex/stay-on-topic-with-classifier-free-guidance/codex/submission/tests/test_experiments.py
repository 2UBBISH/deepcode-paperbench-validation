"""Offline tests for the experiment-level modules.

They exercise the prompt formatting of the zero-shot tasks, the CoT answer
parsing, the HumanEval execution/aggregation path, the Section 5 distribution
analysis and the ANCOVA pipeline, all on synthetic data so that no network
access or model download is required.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm import tasks as task_module
from cfglm.cot import answers_match, is_valid, parse_answer
from cfglm.distributions import (
    continuation_loglikelihoods,
    distribution_for_pair,
    format_ranking_table,
    rank_tokens_by_guidance,
    summarize_distributions,
)
from cfglm.humaneval import (
    HumanEvalProblem,
    build_program,
    evaluate_samples,
    truncate_completion,
    win_tie_loss_counts,
)
from cfglm.tasks import get_task


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
class FakeTokenizer:
    """Deterministic whitespace tokenizer with a broad vocabulary."""

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _encode(self, text):
        return [((abs(hash(w)) % (self.vocab_size - 2)) + 2) for w in text.split()]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self._encode(text)
        if return_tensors == "pt":
            return type("Enc", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})
        return type("Enc", (), {"input_ids": ids})

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids)

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(i) for i in row.tolist()) for row in ids]


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(1)
    config = GPT2Config(vocab_size=512, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(config).eval()


@pytest.fixture(scope="module")
def fake_tokenizer():
    return FakeTokenizer()


# ----------------------------------------------------------------------
# zero-shot task formatting
# ----------------------------------------------------------------------
def test_arc_prompt_format():
    task = get_task("arc_challenge")
    doc = {
        "question": "What is 2 + 2?",
        "choices": {"text": ["3", "4"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    assert task.doc_to_text(doc) == "Question: What is 2 + 2?\nAnswer:"
    assert task.doc_to_choices(doc) == [" 3", " 4"]
    assert task.doc_to_gold(doc) == 1


def test_boolq_prompt_format():
    task = get_task("boolq")
    doc = {"passage": "The sky is blue.", "question": "Is the sky blue", "answer": True}
    assert task.doc_to_text(doc) == "The sky is blue.\nQuestion: Is the sky blue?\nAnswer:"
    assert task.doc_to_choices(doc) == [" yes", " no"]
    assert task.doc_to_gold(doc) == 0


def test_lambada_prompt_splits_last_word():
    task = get_task("lambada_openai")
    doc = {"text": "The quick brown fox"}
    assert task.doc_to_text(doc) == "The quick brown"
    assert task.doc_to_targets(doc) == [" fox"]


def test_winogrande_uses_blank_prefix():
    task = get_task("winogrande")
    doc = {"sentence": "The trophy did not fit because _ was too big.", "option1": "the trophy",
           "option2": "the suitcase", "answer": "2"}
    assert task.doc_to_text(doc) == "The trophy did not fit because "
    assert task.doc_to_choices(doc) == ["the trophy", "the suitcase"]
    assert task.doc_to_gold(doc) == 1


def test_hellaswag_choices_are_space_prefixed():
    task = get_task("hellaswag")
    doc = {"ctx": "A man is sitting. he", "endings": ["runs", "sleeps"], "label": "1"}
    assert task.doc_to_text(doc) == "A man is sitting. he"
    assert task.doc_to_choices(doc) == [" runs", " sleeps"]
    assert task.doc_to_gold(doc) == 1


def test_table5_task_order_matches_the_paper():
    assert task_module.TABLE5_TASKS == [
        "arc_challenge",
        "arc_easy",
        "boolq",
        "hellaswag",
        "piqa",
        "sciq",
        "triviaqa",
        "winogrande",
        "lambada_openai",
    ]


# ----------------------------------------------------------------------
# harness end-to-end on a synthetic multiple-choice task
# ----------------------------------------------------------------------
def test_evaluate_multiple_choice_runs(tiny_model, fake_tokenizer, monkeypatch):
    from cfglm import harness

    docs = [
        {"question": "1 + 1 = ?", "choices": [" 2", " 3"], "gold": 0},
        {"question": "2 + 2 = ?", "choices": [" 5", " 4"], "gold": 1},
    ]
    task = task_module.Task(
        name="synthetic",
        dataset_path="synthetic",
        doc_to_text=lambda d: d["question"],
        doc_to_choices=lambda d: d["choices"],
        doc_to_gold=lambda d: d["gold"],
    )
    monkeypatch.setattr(harness, "load_examples", lambda task, limit=None, split=None: docs)
    result = harness.evaluate_multiple_choice(tiny_model, fake_tokenizer, task, gamma=1.5)
    assert result.n == 2
    assert 0.0 <= result.metrics["acc"] <= 1.0
    assert 0.0 <= result.metrics["acc_norm"] <= 1.0


# ----------------------------------------------------------------------
# Chain-of-Thought parsing
# ----------------------------------------------------------------------
def test_cot_parse_gsm8k_answer():
    assert parse_answer("So the total is 20 - 5 = 15. The answer is 15.") == "15"
    assert parse_answer("The answer is 1,234.") == "1234"


def test_cot_parse_aqua_answer():
    assert parse_answer("... The answer is (c).") == "c"


def test_cot_invalid_chain():
    assert parse_answer("I think it might be something else.") is None
    assert not is_valid("The model rambles on without concluding")


def test_cot_numeric_equivalence():
    assert answers_match("36", "36", "gsm8k")
    assert answers_match("36.0", "36", "gsm8k")
    assert not answers_match("35", "36", "gsm8k")
    assert answers_match("b", "b", "aqua")
    assert not answers_match("a", "b", "aqua")
    assert not answers_match(None, "b", "aqua")


# ----------------------------------------------------------------------
# HumanEval
# ----------------------------------------------------------------------
def test_truncate_completion_at_stops():
    text = "    return x\nclass Foo:\n    pass"
    assert truncate_completion(text) == "    return x"
    assert truncate_completion("    return x\n") == "    return x\n"


def _add_problem():
    return HumanEvalProblem(
        task_id="HumanEval/0",
        prompt="def add_one(x):\n    \"\"\"Add one.\"\"\"\n",
        entry_point="add_one",
        test="def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0\n",
    )


def test_humaneval_program_passes_and_fails():
    problem = _add_problem()
    good = "    return x + 1\n"
    bad = "    return x + 2\n"
    metrics = evaluate_samples(problem and [problem], {problem.task_id: [good, bad, good]}, ks=[1, 3])
    assert metrics["n_correct"][0] == 2
    assert metrics["per_problem"][problem.task_id] == 2
    assert 0.0 < metrics["pass@k"][1] <= 1.0


def test_humaneval_build_program_contains_check():
    program = build_program(_add_problem(), "    return x + 1\n")
    assert "def check(candidate):" in program
    assert program.strip().endswith("check(add_one)")


def test_humaneval_inline_execution_and_timeout():
    from cfglm.humaneval import run_humaneval_program_inline

    problem = _add_problem()
    good = build_program(problem, "    return x + 1\n")
    bad = build_program(problem, "    return x + 2\n")
    infinite = build_program(problem, "    while True:\n        pass\n    return x + 1\n")
    assert run_humaneval_program_inline(good, timeout=5)[0] is True
    assert run_humaneval_program_inline(bad, timeout=5)[0] is False
    passed, message = run_humaneval_program_inline(infinite, timeout=0.5)
    assert passed is False and message == "timeout"


def test_humaneval_subprocess_execution():
    from cfglm.humaneval import run_humaneval_program

    problem = _add_problem()
    assert run_humaneval_program(build_program(problem, "    return x + 1\n"), timeout=20)[0] is True


def test_win_tie_loss_counts():
    baseline = {"a": 0, "b": 1, "c": 2}
    cfg = {"a": 1, "b": 1, "c": 0}
    counts = win_tie_loss_counts(baseline, cfg)
    assert counts == {"cfg_wins": 1, "ties": 1, "cfg_losses": 1, "n_tasks": 3}


# ----------------------------------------------------------------------
# Section 5 distributions
# ----------------------------------------------------------------------
def test_distribution_summary_and_overlap(tiny_model, fake_tokenizer):
    dist = distribution_for_pair(
        tiny_model, fake_tokenizer, "the dragon flew over paris france", " and landed",
        gamma=1.5,
    )
    assert dist.n_tokens == 2
    assert dist.prompted.shape == dist.cfg.shape
    summary = summarize_distributions(dist)
    assert "entropy_prompted" in summary
    assert "entropy_cfg" in summary
    assert summary["topp_overlap_cfg_prompted"] > 0


def test_continuation_loglikelihoods_between_cond_and_cfg(tiny_model, fake_tokenizer):
    dist = distribution_for_pair(tiny_model, fake_tokenizer, "hello world", " and more", gamma=1.5)
    lls = continuation_loglikelihoods(dist, fake_tokenizer(" and more").input_ids)
    assert set(lls) == {"prompted", "cfg"}
    assert lls["cfg"] != lls["prompted"]


def test_rank_tokens_by_guidance_shape(tiny_model, fake_tokenizer):
    rows = rank_tokens_by_guidance(
        tiny_model, fake_tokenizer, "the dragon flew over paris france", gamma=1.5, top_n=3
    )
    assert len(rows) > 0
    assert len(rows[0]["most_upweighted"]) == 3
    table = format_ranking_table(rows, top_n=3)
    assert "current" in table.splitlines()[0]


# ----------------------------------------------------------------------
# ANCOVA pipeline
# ----------------------------------------------------------------------
def test_flops_ancova_pipeline(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    from run_flops_ancova import analyse_task, load_points

    records = []
    rng = np.random.default_rng(0)
    for gamma in (1.0, 1.5):
        for scale in (1e9, 2e9, 4e9, 8e9):
            acc = 0.3 + 0.05 * np.log10(scale) + (0.01 if gamma > 1 else 0.0)
            records.append(
                {
                    "task": "toy",
                    "gamma": gamma,
                    "cfg": gamma != 1.0,
                    "acc": float(np.clip(acc + rng.normal(0, 0.001), 0, 1)),
                    "flops_per_token": scale,
                }
            )
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records))
    by_task = load_points(str(path))
    res = analyse_task("toy", by_task["toy"])
    assert "p_value" in res
    assert res["winner"] in ("CFG", "Vanilla")


# ----------------------------------------------------------------------
# assistant prompts (Section 3.4 scaffolding, marked out of scope)
# ----------------------------------------------------------------------
def test_assistant_prompt_builder():
    from cfglm.assistant_prompts import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPT_SUFFIXES, build_prompt

    prompt = build_prompt(SYSTEM_PROMPT_SUFFIXES[0], "Why is the sky blue?")
    assert prompt.startswith("Instruction: " + DEFAULT_SYSTEM_PROMPT)
    assert "Prompt: Why is the sky blue?" in prompt


def test_lm_eval_adapter_imports_without_harness():
    import importlib

    module = importlib.import_module("cfglm.lm_eval_adapter")
    assert hasattr(module, "CFGHFLM")


def test_lm_eval_adapter_loglikelihood_matches_scorer(tiny_model, fake_tokenizer):
    """Exercise the harness override without lm-eval installed.

    The method is called unbound with a stand-in ``self`` that exposes the
    same attributes the real ``HFLM`` subclass would provide.
    """
    from types import SimpleNamespace

    from cfglm.lm_eval_adapter import CFGHFLM
    from cfglm.scoring import CFGScorer

    scorer = CFGScorer(tiny_model, fake_tokenizer, gamma=1.5, batch_size=2)
    fake_self = SimpleNamespace(
        scorer=scorer, tokenizer=fake_tokenizer, model=tiny_model, gamma=1.5
    )
    ctx_ids = [2, 3, 4, 5]
    cont_ids = [6, 7]
    requests = [(("ctx", "cont"), ctx_ids, cont_ids)]
    (ll, is_greedy), = CFGHFLM._loglikelihood_tokens(fake_self, requests)
    expected = scorer.token_stats(ctx_ids, cont_ids)
    assert ll == pytest.approx(expected.loglikelihood, rel=1e-4, abs=1e-3)
    assert isinstance(is_greedy, bool)


# ----------------------------------------------------------------------
# FUDGE baseline (Table 4)
# ----------------------------------------------------------------------
def test_fudge_generate_runs_with_a_dummy_reward(tiny_model, fake_tokenizer):
    from cfglm.fudge import fudge_cost_ratio, fudge_generate

    def reward(texts):
        # a deterministic "classifier": longer continuations score higher
        return [min(len(t) / 50.0, 1.0) + 1e-3 for t in texts]

    text = fudge_generate(
        tiny_model, fake_tokenizer, "hello world", reward, lam=1.0, max_new_tokens=4, top_k=4, seed=0
    )
    assert isinstance(text, str)
    assert fudge_cost_ratio(32, 8) == 256.0


def test_percent_increase():
    from cfglm.external_classifiers import percent_increase

    assert percent_increase([0.1, 0.1], [0.2, 0.2]) == pytest.approx(1.0)
    assert percent_increase([0.5], [0.5]) == pytest.approx(0.0)


def test_text_classifier_config_labels_are_importable():
    from cfglm.external_classifiers import SENTIMENT_EVAL, SENTIMENT_GUIDANCE, TOXICITY

    assert SENTIMENT_GUIDANCE.name.endswith("emotion")
    assert SENTIMENT_EVAL.name == "stevhliu/my_awesome_model"
    assert TOXICITY.name == "unitary/toxic-bert"


# ----------------------------------------------------------------------
# Self-consistency helpers (contribution 3)
# ----------------------------------------------------------------------
def test_self_consistency_majority_vote_logic():
    from collections import Counter

    from cfglm.cot import parse_answer

    chains = [
        "Reasoning... The answer is 36.",
        "Reasoning... The answer is 36.",
        "Reasoning... The answer is 35.",
        "no answer here",
    ]
    parsed = [parse_answer(c) for c in chains]
    valid = [a for a in parsed if a is not None]
    assert len(valid) == 3
    assert Counter(valid).most_common(1)[0][0] == "36"


# ----------------------------------------------------------------------
# Section 5 aggregation
# ----------------------------------------------------------------------
def _section5_rows(n=12):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "config": f"dataset_{i % 3}_prompt",
                "dataset": f"dataset {i % 3} prompt",
                "index": i,
                "n_tokens": 5,
                "entropy_prompted": 5.0 + rng.normal(0, 0.2),
                "entropy_unprompted": 5.4 + rng.normal(0, 0.2),
                "entropy_cfg": 4.7 + rng.normal(0, 0.2),
                "entropy_instruct": 4.9 + rng.normal(0, 0.2),
                "topp_size_prompted": 40 + rng.normal(0, 3),
                "topp_size_unprompted": 45 + rng.normal(0, 3),
                "topp_size_cfg": 32 + rng.normal(0, 3),
                "topp_overlap_cfg_prompted": 20 + rng.normal(0, 2),
                "topp_overlap_cfg_instruct": 10 + rng.normal(0, 2),
                "topp_overlap_prompted_instruct": 15 + rng.normal(0, 2),
                "ppl_prompted": 20 + rng.normal(0, 1),
                "ppl_cfg": 19 + rng.normal(0, 1),
                "ppl_instruct": 18 + rng.normal(0, 1),
            }
        )
    return rows


def test_section5_summary_aggregation(tmp_path):
    import importlib
    from types import SimpleNamespace

    run_section5 = importlib.import_module("run_section5")
    args = SimpleNamespace(output_dir=str(tmp_path), gamma=1.5)
    run_section5._write_analysis(_section5_rows(), SimpleNamespace(), args)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_examples"] == 12
    assert summary["gamma"] == 1.5
    # Section 5.1 reports the mean entropy of every distribution
    assert set(summary["entropy"]) == {
        "entropy_prompted",
        "entropy_unprompted",
        "entropy_cfg",
        "entropy_instruct",
    }
    # Section 5.2 reports the CFG/instruct overlap and the dataset table
    assert "topp_overlap_cfg_instruct" in summary["top_p_overlap"]
    assert summary["dataset_similarity"]
    assert "spearman_ppl_cfg_ppl_instruct" in summary["correlations"]
    assert (tmp_path / "per_example.json").exists()


# ----------------------------------------------------------------------
# Markdown reporting
# ----------------------------------------------------------------------
def test_report_tables_render():
    import importlib

    report = importlib.import_module("report_tables")
    records = [
        {"model": "gpt2", "task": "lambada_openai", "gamma": 1.0, "acc": 0.326},
        {"model": "gpt2", "task": "lambada_openai", "gamma": 1.5, "acc": 0.446},
    ]
    table = report.table5(records)
    assert "LAMBADA" in table
    assert "32.6 / 44.6" in table

    humaneval = [
        {
            "model": "codegen-350m-mono",
            "temperature": 0.2,
            "gamma": 1.0,
            "pass@k": {"1": 0.11, "10": 0.17, "100": 0.22},
        }
    ]
    rendered = report.humaneval_table(humaneval, 0.2)
    assert "11.0%" in rendered and "22.0%" in rendered
