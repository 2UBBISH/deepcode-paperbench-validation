"""Model / tokenizer loading and the model families used in the paper.

The paper studies the GPT-2, Pythia and LLaMA families (Section 3.1), the
CodeGen-mono family (Section 3.3.1), WizardLM-30B / Guanaco-65B for
Chain-of-Thought (Section 3.2) and Falcon-7b (base + instruct) for the
analysis of Section 5.

Note (scope): LLaMA checkpoints are out of scope for the reproduction
because obtaining them requires permission; they are listed here so that the
sweep can be re-run if the user has access.  Likewise the GPT4All-J chatbot
study of Section 3.4 is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelSpec:
    """A model used in one of the paper's experiments."""

    key: str
    hf_name: str
    family: str
    params: Optional[str] = None
    n_params: Optional[float] = None  # in billions, for plotting
    in_scope: bool = True
    notes: str = ""


# ----------------------------------------------------------------------
# Section 3.1 -- zero-shot benchmarks (Table 5)
# ----------------------------------------------------------------------
GPT2_MODELS: List[ModelSpec] = [
    ModelSpec("gpt2", "gpt2", "gpt2", "s", 0.124),
    ModelSpec("gpt2-medium", "gpt2-medium", "gpt2", "m", 0.355),
    ModelSpec("gpt2-large", "gpt2-large", "gpt2", "l", 0.774),
    ModelSpec("gpt2-xl", "gpt2-xl", "gpt2", "xl", 1.558),
]

PYTHIA_MODELS: List[ModelSpec] = [
    ModelSpec("pythia-160m", "EleutherAI/pythia-160m", "pythia", "160M", 0.16),
    ModelSpec("pythia-410m", "EleutherAI/pythia-410m", "pythia", "410M", 0.41),
    ModelSpec("pythia-1b", "EleutherAI/pythia-1b", "pythia", "1B", 1.0),
    ModelSpec("pythia-1.4b", "EleutherAI/pythia-1.4b", "pythia", "1.4B", 1.4),
    ModelSpec("pythia-2.8b", "EleutherAI/pythia-2.8b", "pythia", "2.8B", 2.8),
    ModelSpec("pythia-6.9b", "EleutherAI/pythia-6.9b", "pythia", "6.9B", 6.9),
    ModelSpec("pythia-12b", "EleutherAI/pythia-12b", "pythia", "12B", 12.0),
]

LLAMA_MODELS: List[ModelSpec] = [
    ModelSpec("llama-7b", "huggyllama/llama-7b", "llama", "7B", 7.0, False,
              "LLaMA access is out of scope for the reproduction"),
    ModelSpec("llama-13b", "huggyllama/llama-13b", "llama", "13B", 13.0, False,
              "LLaMA access is out of scope for the reproduction"),
    ModelSpec("llama-30b", "huggyllama/llama-30b", "llama", "30B", 30.0, False,
              "LLaMA access is out of scope for the reproduction"),
    ModelSpec("llama-65b", "huggyllama/llama-65b", "llama", "65B", 65.0, False,
              "LLaMA access is out of scope for the reproduction"),
]

# ----------------------------------------------------------------------
# Section 3.3.1 -- HumanEval
# ----------------------------------------------------------------------
CODEGEN_MODELS: List[ModelSpec] = [
    ModelSpec("codegen-350m-mono", "Salesforce/codegen-350M-mono", "codegen", "350M", 0.35),
    ModelSpec("codegen-2b-mono", "Salesforce/codegen-2B-mono", "codegen", "2B", 2.0),
    ModelSpec("codegen-6b-mono", "Salesforce/codegen-6B-mono", "codegen", "6B", 6.0),
    ModelSpec("codegen-16b-mono", "Salesforce/codegen-16B-mono", "codegen", "16B", 16.0,
              False, "omitted in the paper due to compute constraints"),
]

# ----------------------------------------------------------------------
# Section 3.2 -- Chain-of-Thought
# ----------------------------------------------------------------------
COT_MODELS: List[ModelSpec] = [
    ModelSpec("wizardlm-30b", "WizardLM/WizardLM-30B-V1.0", "wizardlm", "30B", 30.0),
    ModelSpec("guanaco-65b", "timdettmers/guanaco-65b", "guanaco", "65B", 65.0),
]

# ----------------------------------------------------------------------
# Section 5 -- Falcon-7b
# ----------------------------------------------------------------------
FALCON_MODELS: List[ModelSpec] = [
    ModelSpec("falcon-7b", "tiiuae/falcon-7b", "falcon", "7B", 7.0),
    ModelSpec("falcon-7b-instruct", "tiiuae/falcon-7b-instruct", "falcon", "7B", 7.0),
]

FAMILIES: Dict[str, List[ModelSpec]] = {
    "gpt2": GPT2_MODELS,
    "pythia": PYTHIA_MODELS,
    "llama": LLAMA_MODELS,
    "codegen": CODEGEN_MODELS,
    "cot": COT_MODELS,
    "falcon": FALCON_MODELS,
}

ALL_MODELS: Dict[str, ModelSpec] = {
    spec.key: spec for specs in FAMILIES.values() for spec in specs
}


def resolve_model(name_or_key: str) -> ModelSpec:
    """Resolve a short key (``gpt2-medium``) or an HF name to a spec."""
    if name_or_key in ALL_MODELS:
        return ALL_MODELS[name_or_key]
    return ModelSpec(name_or_key, name_or_key, "custom")


def default_dtype(device: Optional[str] = None) -> torch.dtype:
    """``float16`` on CUDA, ``float32`` on CPU."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.float16 if device.startswith("cuda") else torch.float32


def load_model_and_tokenizer(
    name_or_key: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a causal LM and its tokenizer, in ``eval`` mode.

    Tokenizers are configured with ``padding_side="left"`` because the
    CFG scorer left-pads batches, and a pad token is set when the tokenizer
    does not define one (GPT-2, LLaMA, ...).
    """
    spec = resolve_model(name_or_key)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = default_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=revision, trust_remote_code=trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, spec
