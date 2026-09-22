"""Central place for every hyper-parameter reported in the paper.

Values are taken from Sec. 4.1 ("Training and Evaluation Setup") of the paper
and from ``paper/addendum.md``.  Whenever the paper is ambiguous we record the
assumption in a comment next to the value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict

# --------------------------------------------------------------------------------------
# Paper hyper-parameters (Sec. 4.1)
# --------------------------------------------------------------------------------------
#: number of parameter update steps used to fix a *single* error.
STEPS_SINGLE_ERROR = {"head": 100, "lora": 30, "full_ft": 30}

#: learning rate while fixing a single error (Sec. 4.1).
LR_SINGLE_ERROR = {
    "head": {"bart0": 1e-3, "flan_t5": 1e-4},
    "lora": {"bart0": 1e-5, "flan_t5": 1e-4},
    "full_ft": {"bart0": 1e-5, "flan_t5": 1e-4},
}

#: smaller learning rates used when *sequentially* fixing multiple errors
#: (Sec. 4.1).  The paper only spells the LoRA/full-FT values out; the head-only
#: values are inferred by applying the same 10x reduction.
LR_SEQUENTIAL = {
    "head": {"bart0": 1e-4, "flan_t5": 1e-5},
    "lora": {"bart0": 1e-6, "flan_t5": 1e-5},
    "full_ft": {"bart0": 1e-6, "flan_t5": 1e-5},
}

#: replay schedule of the model-refinement baselines (Sec. 4.2).
REPLAY_SCHEDULE = {
    "bart0_large": {"batch_size": 8, "every_n_steps": 10},
    "flan_t5_large": {"batch_size": 8, "every_n_steps": 10},
    "flan_t5_3b": {"batch_size": 4, "every_n_steps": 5},
}

#: number of examples sampled per upstream task to build ``D_PT`` (Sec. 4.1).
EXAMPLES_PER_UPSTREAM_TASK = 100

#: ``D_R`` is randomly split into 60% / 40% train/test (addendum, Sec. 4.1).
DR_TRAIN_FRACTION = 0.6

#: fraction of ``D_R`` used in one stream of the continual-refinement setup
#: (Figure 3): "1/8 of total examples in D_R in each setup".
STREAM_FRACTION = 1.0 / 8.0

#: LoRA configuration, verbatim from the addendum.
LORA_CONFIG = dict(r=16, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=["q", "v"])

#: training details of the forecasting models (Appendix B, referenced from Sec. 4/5).
FORECAST_TRAIN = dict(
    max_steps=100_000,
    batch_size=16,
    n_positive_per_batch=8,
    n_negative_per_batch=8,
    positive_loss_weight=0.1,   # alpha = 0.1 in Appendix B
    lm_learning_rate=1e-5,
    mlp_learning_rate=1e-4,
    margin=1.0,                 # preset margin of Eqn. 3
    hidden_dim=512,             # dimensionality of the 2-layer MLP / of h
)

#: "we only cache top k = 100 largest logits for each token in y_j" (Sec. 3.2).
TOPK_CACHED_LOGITS = 100


# --------------------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """A base PTLM plus the encoder used by the trainable forecasting models."""

    key: str                       # e.g. "bart0_large"
    hf_name: str                   # HF hub id of the base PTLM
    family: str                    # "bart0" | "flan_t5"
    forecast_encoder_hf_name: str  # encoder h of the forecasting models (Appendix B)
    head_only_target: str          # LM head module tuned in "head" mode
    dtype: str = "float32"
    notes: str = ""


MODEL_SPECS: Dict[str, ModelSpec] = {
    "bart0_large": ModelSpec(
        key="bart0_large",
        hf_name="inklab/bart0-large",       # BART0_Large (Lin et al., 2022a)
        family="bart0",
        forecast_encoder_hf_name="inklab/bart0-large",
        head_only_target="lm_head",
        notes="BART0 is trained on P3-Train; checkpoint distributed with ReCross.",
    ),
    "flan_t5_large": ModelSpec(
        key="flan_t5_large",
        hf_name="google/flan-t5-large",
        family="flan_t5",
        # Appendix B: FLAN-T5_small is used as the encoder for the T5 experiments.
        forecast_encoder_hf_name="google/flan-t5-small",
        head_only_target="lm_head",
    ),
    "flan_t5_3b": ModelSpec(
        key="flan_t5_3b",
        hf_name="google/flan-t5-xl",
        family="flan_t5",
        forecast_encoder_hf_name="google/flan-t5-small",
        head_only_target="lm_head",
    ),
    # Used for CPU smoke tests only.
    "flan_t5_small": ModelSpec(
        key="flan_t5_small",
        hf_name="google/flan-t5-small",
        family="flan_t5",
        forecast_encoder_hf_name="google/flan-t5-small",
        head_only_target="lm_head",
        notes="Only for scaled-down smoke tests; not a model of the paper.",
    ),
}


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one row/column of the paper tables."""

    model: str = "flan_t5_large"
    #: which dataset supplies the mispredicted examples D_R.
    refinement_data: str = "mmlu"        # "p3_test" for BART0, "mmlu" for FLAN-T5
    tuning_mode: str = "lora"            # "head" | "lora" | "full_ft"
    seed: int = 0
    #: dataset / artefact locations
    data_root: str = field(default_factory=lambda: os.environ.get("WWMF_DATA_ROOT", "data"))
    cache_root: str = field(default_factory=lambda: os.environ.get("WWMF_CACHE_ROOT", "cache"))
    output_root: str = field(default_factory=lambda: os.environ.get("WWMF_OUTPUT_ROOT", "outputs"))
    examples_per_upstream_task: int = EXAMPLES_PER_UPSTREAM_TASK
    max_input_len: int = 512
    max_target_len: int = 32
    eval_batch_size: int = 16
    #: figures/tables
    stream_fraction: float = STREAM_FRACTION
    device: str = "auto"
    #: debug / smoke-test overrides (None -> use the paper's hyper-parameters)
    steps_override: int | None = None
    lr_override: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def family(self) -> str:
        return self.model_spec().family

    def lr_single(self) -> float:
        if self.lr_override is not None:
            return self.lr_override
        return LR_SINGLE_ERROR[self.tuning_mode][self.family()]

    def lr_sequential(self) -> float:
        if self.lr_override is not None:
            return self.lr_override
        return LR_SEQUENTIAL[self.tuning_mode][self.family()]

    def steps_single(self) -> int:
        if self.steps_override is not None:
            return self.steps_override
        return STEPS_SINGLE_ERROR[self.tuning_mode]

    def model_spec(self) -> ModelSpec:
        return MODEL_SPECS[self.model]

    def replay_schedule(self) -> Dict[str, int]:
        key = self.model if self.model in REPLAY_SCHEDULE else "flan_t5_large"
        return REPLAY_SCHEDULE[key]
