"""Shared helpers of the reduced-scale smoke runs (see run_smoke.py)."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from wwmf.config import ExperimentConfig
from wwmf.data.build import RefinementSplits
from wwmf.data.p3 import load_p3_config
from wwmf.data.types import Example
from wwmf.evaluation.metrics import exact_match

#: templates of the upstream task `glue-mrpc` (one of the 36 tasks of D_PT) that
#: FLAN-T5_small answers reasonably often -- needed for D_hat_PT to be non-empty.
SMOKE_UPSTREAM_CONFIGS: List[Tuple[str, str]] = [
    ("glue-mrpc", "glue_mrpc_paraphrase"),
    ("glue-mrpc", "glue_mrpc_same_thing"),
    ("glue-mrpc", "glue_mrpc_equivalent"),
]
#: refinement task of the smoke runs (a paraphrase-detection task the model fails on)
SMOKE_REFINEMENT_TASK = "glue-qqp"
SMOKE_REFINEMENT_CONFIG = "glue_qqp_duplicate"
SMOKE_REFINEMENT_SPLIT = "validation"


def make_smoke_config(args, model: str = "flan_t5_small") -> ExperimentConfig:
    cfg = ExperimentConfig(
        model=model,
        refinement_data="p3_test",
        tuning_mode=getattr(args, "tuning_mode", "head"),
        max_input_len=getattr(args, "max_input_len", 384),
        max_target_len=8,
        eval_batch_size=8,
        output_root=args.output,
        cache_root=os.path.join(args.output, "cache"),
        data_root=os.path.join(args.output, "data"),
    )
    cfg.steps_override = args.steps
    cfg.lr_override = args.lr
    return cfg


def build_smoke_dpt(per_task: int, seed: int = 0) -> List[Example]:
    examples: List[Example] = []
    for task, config in SMOKE_UPSTREAM_CONFIGS:
        rows = load_p3_config(config, split="train", task=task)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=min(per_task, len(rows)), replace=False)
        examples.extend(rows[int(i)] for i in sorted(idx))
    return examples


def build_smoke_dr(model, max_candidates: int, online: int, seed: int = 0):
    """Collect mispredictions of ``model`` and split them 60/40 into D_R^Train/Test."""
    rows = load_p3_config(SMOKE_REFINEMENT_CONFIG, split=SMOKE_REFINEMENT_SPLIT, task=SMOKE_REFINEMENT_TASK)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(max_candidates, len(rows)), replace=False)
    candidates = [rows[int(i)] for i in sorted(idx)]
    preds = model.predict([e.input for e in candidates], batch_size=8)
    mispredicted = [ex for ex, p in zip(candidates, preds) if exact_match([p], [ex.target]) <= 0.5]
    em = exact_match(preds, [e.target for e in candidates])
    order = np.random.default_rng(seed).permutation(len(mispredicted))
    shuffled = [mispredicted[int(i)] for i in order]
    cut = max(1, int(round(0.6 * len(shuffled))))
    splits = RefinementSplits(
        train=shuffled[:cut][:online],
        test=shuffled[cut:][:online],
        candidates=candidates,
        em=em,
    )
    return splits, len(mispredicted)
