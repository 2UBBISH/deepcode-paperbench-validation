"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simformer import Simformer, SimformerConfig, get_task  # noqa: E402

RESULTS = Path(os.environ.get("SIMFORMER_RESULTS", "results"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(path: Path, config: SimformerConfig, extra: Optional[dict] = None):
    payload = {**{k: v for k, v in vars(config).items()},
               **(extra or {})}
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str))


def generate_simulations(task, n: int, seed: int = 0, **kwargs):
    """Draw ``n`` joint samples ``(theta, x)`` from a task."""
    rng = np.random.default_rng(seed)
    theta, x, index, metadata = task.joint_sample(n, rng)
    return theta, x, index, metadata


def build_model(task_name: str, mask_mode: str = "dense", sde: str = "vesde",
                n_layers: int = 6, seed: int = 0,
                num_steps: int = 500, task_kwargs: Optional[dict] = None,
                **overrides) -> Tuple[object, Simformer]:
    """Instantiate the task and a Simformer with the paper configuration."""
    task = get_task(task_name, **(task_kwargs or {}))
    problem = task.problem()
    config = SimformerConfig(sde=sde, mask_mode=mask_mode, n_layers=n_layers,
                             num_steps=num_steps, seed=seed, **overrides)
    model = Simformer(problem, config)
    return task, model


def save_checkpoint(model: Simformer, path: Path, extra: Optional[dict] = None):
    ensure_dir(path.parent)
    torch.save({"state_dict": model.state_dict(),
                "config": vars(model.config),
                "problem": {
                    "name": model.problem.name,
                    "n_variables": model.problem.n_variables,
                    "n_params": model.problem.n_params,
                    "n_data": model.problem.n_data,
                },
                "extra": extra or {}}, path)


def load_checkpoint(task, path: Path, **config_overrides) -> Simformer:
    """Load a trained Simformer checkpoint.

    Checkpoints that were written by an older version of the code (e.g. without
    the ``use_fourier`` bookkeeping buffer) are loaded with ``strict=False``;
    buffers that only describe the problem layout are re-created from the task.
    """
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    config.update(config_overrides)
    config = SimformerConfig(**config)
    model = Simformer(task.problem(), config)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[load_checkpoint] re-created buffers: "
              f"missing={list(incompatible.missing_keys)} "
              f"unexpected={list(incompatible.unexpected_keys)}")
    model.eval()
    return model
