"""Configuration dataclasses for BBOX-ADAPTER.

The defaults below are the values reported in the paper (Section 4.1, Appendix
H.2 and Table 8).  Where the paper does not report a value the default is
marked with ``[estimated]`` in the docstring so that the reader knows exactly
which knobs had to be chosen by us.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AdapterConfig:
    """Hyper-parameters of the energy based adapter (Section 3.2)."""

    # microsoft/deberta-v3-base (86M) and microsoft/deberta-v3-large (304M) are
    # used for StrategyQA / GSM8K / ScienceQA; bert-base-cased (110M) for
    # TruthfulQA (Appendix H.2).
    model_name: str = "microsoft/deberta-v3-base"
    adapter_size: str = "0.1B"  # "0.1B" or "0.3B"
    max_length: int = 512
    # "cls" == the [CLS]/pooler representation of the encoder is projected to a
    # scalar.  "mean" == masked mean pooling.
    pooling: str = "cls"
    dropout: float = 0.0
    # Weight decay of the AdamW optimizer (Appendix H.2).
    weight_decay: float = 0.01
    # Learning rate eta (Appendix H.2).
    learning_rate: float = 5e-6
    # Mini-batch size (Appendix H.2).
    batch_size: int = 64
    # Number of adapter updates per online iteration (Appendix H.2).
    num_train_steps: int = 6000
    # Coefficient alpha of the l2 regularisation of the energies that replaces
    # spectral normalisation (Eq. 3 + addendum).  [estimated]
    alpha: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0
    # "nce" (ours) or "mlm" (ablation of Section 4.5).
    loss_type: str = "nce"
    # Number of negative samples drawn per positive/example during training.
    negatives_per_example: int = 1
    # Use the listwise softmax form of Eq. (2) instead of the pairwise
    # sampling form of Eq. (3).  The paper's gradient (Eq. 3) is pairwise.
    listwise_softmax: bool = False
    # Candidates per example (1 positive + K-1 negatives) when Eq. (2) is used.
    num_softmax_negatives: int = 4
    grad_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    freeze_encoder_layers: int = 0


@dataclass
class InferenceConfig:
    """Adapted inference / beam search (Section 3.3)."""

    beam_size: int = 3  # k
    # Number of samples drawn per beam at every sentence step.  [estimated]
    num_samples_per_beam: int = 3  # n
    # Maximum number of sentences (L) in a generated solution.
    max_sentence_steps: int = 24
    # Maximum number of tokens produced by the black-box LLM per sentence.
    max_new_tokens: int = 64
    # Maximum length of a complete solution, in tokens (Appendix H.2).
    max_solution_tokens: int = 512
    temperature: float = 1.0  # Appendix H.2
    top_p: float = 1.0
    # "full" == sentence level beam search, "single" == sample complete answers
    # in one step and let the adapter pick the best one (Section 4.4).
    mode: str = "full"
    num_single_step_candidates: int = 5
    # Sentences collected while sampling negatives for the online loop.
    sentence_delimiter: str = "\n"
    stop_marker: str = "####"


@dataclass
class OnlineConfig:
    """Online adaptation framework (Section 3.4, Algorithm 1)."""

    num_iterations: int = 3  # T
    # Candidates sampled from the adapted inference at every iteration (M).
    num_candidates_per_question: int = 5
    # Candidates requested from the un-adapted LLM at initialisation (K).
    init_candidates_per_question: int = 5
    # "ground_truth" | "ai_feedback" | "combined"
    positive_source: str = "ground_truth"
    # Outcome supervision: candidate answers that agree with the training set
    # answer are added as extra positive samples.
    outcome_supervision: bool = True
    # Questions per online iteration (None == the whole training set).
    max_train_samples: Optional[int] = None
    # Number of questions evaluated on the dev set after every iteration
    # (0 disables the intermediate evaluation).
    dev_eval_samples: int = 0


@dataclass
class LLMConfig:
    """Black-box LLM configuration."""

    name: str = "gpt-3.5-turbo"  # gpt-3.5-turbo | davinci-002 | mixtral-8x7b | mock
    provider: str = "azure"  # azure | openai | huggingface | mock
    deployment: Optional[str] = None
    api_version: Optional[str] = None
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    seed: Optional[int] = None
    # HuggingFace specific
    dtype: str = "float16"
    device_map: str = "auto"
    # Sampling / retry behaviour
    max_retries: int = 6
    request_batch_size: int = 1


@dataclass
class RunConfig:
    """Full configuration of one experiment."""

    dataset: str = "strategyqa"
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output_dir: str = "runs/default"
    seed: int = 0

    # ------------------------------------------------------------------ utils
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)


def _build_dataclass(cls, payload: Optional[Dict[str, Any]]):
    payload = dict(payload or {})
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - fields
    if unknown:
        raise ValueError(f"Unknown fields for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def load_run_config(path: str) -> RunConfig:
    """Load a YAML/JSON run configuration (see ``configs/``)."""

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read yaml configs") from exc
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

    payload = dict(payload or {})
    return RunConfig(
        dataset=payload.get("dataset", "strategyqa"),
        adapter=_build_dataclass(AdapterConfig, payload.get("adapter")),
        inference=_build_dataclass(InferenceConfig, payload.get("inference")),
        online=_build_dataclass(OnlineConfig, payload.get("online")),
        llm=_build_dataclass(LLMConfig, payload.get("llm")),
        output_dir=payload.get("output_dir", "runs/default"),
        seed=payload.get("seed", 0),
    )


# Adapter backbones per dataset as reported in Appendix H.2.
ADAPTER_BACKBONES: Dict[str, Dict[str, str]] = {
    "strategyqa": {"0.1B": "microsoft/deberta-v3-base", "0.3B": "microsoft/deberta-v3-large"},
    "gsm8k": {"0.1B": "microsoft/deberta-v3-base", "0.3B": "microsoft/deberta-v3-large"},
    "scienceqa": {"0.1B": "microsoft/deberta-v3-base", "0.3B": "microsoft/deberta-v3-large"},
    "truthfulqa": {"0.1B": "bert-base-cased", "0.3B": "bert-base-cased"},
}


def adapter_backbone(dataset: str, size: str) -> str:
    table = ADAPTER_BACKBONES.get(dataset.lower())
    if table is None:
        return "microsoft/deberta-v3-base"
    return table.get(size, table["0.1B"])
