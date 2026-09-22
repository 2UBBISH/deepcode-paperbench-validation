"""Shared helpers: seeding, device handling, json/io utilities and tokenizer loading."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Pick the best available device.

    CUDA is preferred (this is what the paper-scale reproduction runs on).  MPS
    is only used when it is requested explicitly -- empirically the MPS backend
    is much slower than the CPU backend for GPT2-medium forward passes, so it is
    not selected automatically.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def batches(items: Sequence, batch_size: int) -> Iterable:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_tokenizer(model_name: str = "gpt2-medium", padding_side: str = "right"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_lm(model_name: str = "gpt2-medium", dtype: str = "float32", device: Optional[str] = None):
    """Load a causal LM in eval mode on the requested device."""
    from transformers import AutoModelForCausalLM

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(get_device(device))
    model.eval()
    return model, tokenizer


def parse_csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over timesteps, ignoring padding. ``hidden``: (B, T, d)."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def set_determinism() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def model_dtype_str(model: torch.nn.Module) -> str:
    try:
        return str(next(model.parameters()).dtype)
    except StopIteration:  # pragma: no cover
        return "unknown"


def stack_dict(items: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of metric dicts into a single dict."""
    if not items:
        return {}
    keys = items[0].keys()
    return {k: float(np.mean([it[k] for it in items])) for k in keys}
