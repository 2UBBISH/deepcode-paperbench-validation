"""Small utilities: seeding, EMA, configuration handling and logging."""

from __future__ import annotations

import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# exponential moving average of the weights (Appendix B training recipe)
# ---------------------------------------------------------------------------
class EMA(nn.Module):
    def __init__(self, model: nn.Module, beta: float = 0.995, update_after_step: int = 100):
        super().__init__()
        self.beta = beta
        self.update_after_step = update_after_step
        self.model = deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        if self.step < self.update_after_step:
            return
        beta = min(self.beta, (1 + self.step) / (10 + self.step))
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.mul_(beta).add_(p.detach(), alpha=1 - beta)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML (or JSON) configuration file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def merge_config(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
class JsonlLogger:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def log(self, **kwargs: Any) -> None:
        record = {"wall_time": time.time() - self._t0, **kwargs}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        printable = " ".join(f"{k}={v}" for k, v in kwargs.items() if isinstance(v, (int, float, str)))
        if printable:
            print(f"[{record['wall_time']:.1f}s] {printable}", flush=True)


def _json_default(x):
    if torch.is_tensor(x):
        return x.item() if x.numel() == 1 else x.tolist()
    return str(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str | os.PathLike,
    model: nn.Module,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    config: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | os.PathLike, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location)


def to_image_range(x: Tensor, lo: float = -1.0, hi: float = 1.0) -> Tensor:
    """Clamp (and optionally rescale) a model sample to the valid image range."""
    return x.clamp(lo, hi)


__all__ = [
    "set_seed",
    "EMA",
    "load_config",
    "merge_config",
    "resolve_device",
    "JsonlLogger",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "to_image_range",
]
