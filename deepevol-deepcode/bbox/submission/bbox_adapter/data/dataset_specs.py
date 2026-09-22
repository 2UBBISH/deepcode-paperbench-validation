"""Dataset specifications for BBox-Adapter.

Splits are taken verbatim from the paper:

* GSM8K      : 7473 train / 1319 test              (Appendix F.1)
* StrategyQA : 2059 train / 229 test               (Appendix F.1)
* TruthfulQA : 717 train / 100 test (random sample) (Appendix F.1)
* ScienceQA  : 2000 train / 500 test (no-image)     (Appendix F.1)
* ToxiGen    : 2000 train / 500 test                (Appendix E)

The answer format of every dataset is recorded here so that the answer
extraction utilities and the prompt templates can be driven from a single
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetSpec:
    """Static description of one evaluation dataset."""

    name: str
    hf_path: str
    hf_config: Optional[str] = None
    # The split names on the HF hub (train / test)
    train_split: str = "train"
    test_split: str = "test"
    # Paper-fixed split sizes
    n_train: int = -1
    n_test: int = -1
    # Fields inside each HF record
    question_field: str = "question"
    answer_field: str = "answer"
    # How the answer must be parsed / rendered
    answer_type: str = "freeform"  # one of: yesno, numeric, mcq, truthfulqa, toxic
    # Which prompt template to use (key into prompts.PROMPT_TEMPLATES)
    prompt_key: str = ""
    # Metric used at evaluation time
    metric: str = "accuracy"
    # Extra metadata
    extra: dict = field(default_factory=dict)


STRATEGYQA = DatasetSpec(
    name="strategyqa",
    hf_path="wics/strategy-qa",
    hf_config=None,
    train_split="train",
    test_split="test",
    n_train=2059,
    n_test=229,
    question_field="question",
    answer_field="answer",
    answer_type="yesno",
    prompt_key="strategyqa",
    metric="accuracy",
    extra={"shot": 2},
)

GSM8K = DatasetSpec(
    name="gsm8k",
    hf_path="gsm8k",
    hf_config="main",
    train_split="train",
    test_split="test",
    n_train=7473,
    n_test=1319,
    question_field="question",
    answer_field="answer",
    answer_type="numeric",
    prompt_key="gsm8k",
    metric="accuracy",
    extra={"shot": 4},
)

TRUTHFULQA = DatasetSpec(
    name="truthfulqa",
    hf_path="truthful_qa",
    hf_config="generation",
    train_split="validation",  # TruthfulQA only ships a validation split
    test_split="validation",
    n_train=717,
    n_test=100,
    question_field="question",
    answer_field="best_answer",
    answer_type="truthfulqa",
    prompt_key="truthfulqa",
    metric="true_info",
    extra={"shot": 0, "random_test_seed": 0},
)

SCIENCEQA = DatasetSpec(
    name="scienceqa",
    hf_path="derek-thomas/ScienceQA",
    hf_config=None,
    train_split="train",
    test_split="test",
    n_train=2000,
    n_test=500,
    question_field="question",
    answer_field="answer",
    answer_type="mcq",
    prompt_key="scienceqa",
    metric="accuracy",
    extra={"shot": 1, "exclude_image": True, "random_subset_seed": 0},
)

TOXIGEN = DatasetSpec(
    name="toxigen",
    hf_path="toxigen/toxigen-data",
    hf_config="annotated",
    train_split="train",
    test_split="test",
    n_train=2000,
    n_test=500,
    question_field="text",
    answer_field="",
    answer_type="toxic",
    prompt_key="toxigen",
    metric="toxicity",
    extra={"random_subset_seed": 0, "temperature": 0.7},
)

ALL_SPECS: List[DatasetSpec] = [STRATEGYQA, GSM8K, TRUTHFULQA, SCIENCEQA, TOXIGEN]
QA_SPECS: List[DatasetSpec] = [STRATEGYQA, GSM8K, TRUTHFULQA, SCIENCEQA]

SPEC_BY_NAME = {s.name: s for s in ALL_SPECS}

# Deltas reported in the paper (Table 2) -- useful for regression testing the
# evaluation harness plumbing.
PAPER_TABLE2 = {
    "base": {"strategyqa": 66.59, "gsm8k": 67.51, "truthfulqa": 77.00, "scienceqa": 72.90},
    "ground_truth": {"strategyqa": 71.62, "gsm8k": 73.86, "truthfulqa": 79.70, "scienceqa": 78.53},
    "ai_feedback": {"strategyqa": 69.85, "gsm8k": 73.50, "truthfulqa": 82.10, "scienceqa": 78.30},
    "combined": {"strategyqa": 72.27, "gsm8k": 74.28, "truthfulqa": 83.60, "scienceqa": 79.40},
}

PAPER_TABLE3 = {
    "davinci-002": {"base": 33.14, "plugged": 39.99},
    "mixtral": {"base": 49.26, "plugged": 53.76},
}

PAPER_TABLE5 = {  # ablation, MLM vs NCE
    "strategyqa": {"mlm": (61.52, 60.41), "nce": (71.62, 71.18)},
    "gsm8k": {"mlm": (70.56, 70.81), "nce": (72.06, 73.86)},
}

PAPER_TABLE7 = {"toxic_pct": (41.90, 20.60), "toxicity_prob_pct": (41.02, 20.75)}


def get_spec(name: str) -> DatasetSpec:
    key = name.lower().replace("-", "").replace("_", "")
    for k, v in SPEC_BY_NAME.items():
        if k.replace("-", "").replace("_", "") == key:
            return v
    raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(SPEC_BY_NAME)}")
