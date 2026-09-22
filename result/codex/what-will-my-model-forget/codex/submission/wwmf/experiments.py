"""High-level drivers that reproduce each table / figure of the paper."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import FORECAST_TRAIN, ExperimentConfig
from .data.build import RefinementSplits, build_dpt, build_dr, build_ood_split, load_dr
from .data.types import Dataset, Example
from .forecasting.base import BaseForecaster, ForecastContext
from .forecasting.cache import (
    build_online_artifacts,
    build_upstream_cache,
    default_cache_path,
)
from .forecasting.logit_change import FixedLogitForecaster, TrainableLogitForecaster
from .forecasting.representation import RepresentationForecaster
from .forecasting.threshold import ThresholdForecaster
from .forecasting.types import OnlineArtifact, UpstreamCache
from .models.lm import Seq2SeqLM
from .models.tuning import prepare_model
from .refinement.replay import GTForgetReplay, RandomReplay, ReplayPool, ScoreReplay
from .refinement.stream import (
    RefinementResult,
    continual_forecasting_curves,
    evaluate_single_error_replay,
    sequential_refinement,
)
from .utils import ensure_dir, set_seed, write_json


@dataclass
class ExperimentState:
    """Everything that is shared between the methods of one (LM, dataset) setup."""

    cfg: ExperimentConfig
    base_model: Seq2SeqLM
    dpt: Dataset
    dr_train: Dataset
    dr_test: Dataset
    upstream_cache: UpstreamCache
    train_artifacts: List[OnlineArtifact] = field(default_factory=list)
    test_artifacts: List[OnlineArtifact] = field(default_factory=list)
    ctx: Optional[ForecastContext] = None

    @property
    def replay_pool(self) -> ReplayPool:
        return ReplayPool(self.upstream_cache, self.cfg.max_target_len)

    def label_lookup(self, example: Optional[Example]) -> Optional[np.ndarray]:
        if example is None:
            return None
        for art in self.train_artifacts + self.test_artifacts:
            if art.example.key == example.key:
                return art.labels
        return None

    def artifact_lookup(self, example: Optional[Example]) -> Optional[OnlineArtifact]:
        if example is None:
            return None
        for art in self.train_artifacts + self.test_artifacts:
            if art.example.key == example.key:
                return art
        return None


def prepare_state(
    cfg: ExperimentConfig,
    forecast_train_steps: Optional[int] = None,
    max_upstream: Optional[int] = None,
    max_online: Optional[int] = None,
    verbose: bool = True,
    dpt: Optional[Dataset] = None,
    dr: Optional["RefinementSplits"] = None,
) -> ExperimentState:
    """Build ``D_PT``, ``D_R``, the base model and every cached quantity.

    ``dpt``/``dr`` can be passed in to bypass the dataset download (used by the
    reduced-scale smoke runs).
    """
    set_seed(cfg.seed)
    base_model = prepare_model(cfg)
    dpt = dpt if dpt is not None else build_dpt(cfg)
    if max_upstream is not None:
        dpt = dpt[:max_upstream]
    splits = dr if dr is not None else load_dr(cfg)
    if splits is None:
        splits = build_dr(cfg, base_model)
    dr_train, dr_test = splits.train, splits.test
    if max_online is not None:
        dr_train = dr_train[:max_online]
        dr_test = dr_test[:max_online]

    upstream_cache = build_upstream_cache(
        base_model,
        dpt,
        verbose=verbose,
        cache_path=default_cache_path(cfg, "upstream", len(dpt), 0),
    )
    steps = cfg.steps_single()
    lr = cfg.lr_single()
    artifact_key = f"steps{steps}_lr{lr:g}"
    train_artifacts = build_online_artifacts(
        base_model, dr_train, upstream_cache, steps=steps, lr=lr, mode=cfg.tuning_mode,
        verbose=verbose,
        cache_path=default_cache_path(cfg, "online_train", len(dpt), len(dr_train), artifact_key),
    )
    test_artifacts = build_online_artifacts(
        base_model, dr_test, upstream_cache, steps=steps, lr=lr, mode=cfg.tuning_mode,
        verbose=verbose,
        cache_path=default_cache_path(cfg, "online_test", len(dpt), len(dr_test), artifact_key),
    )
    ctx = ForecastContext(
        cfg=cfg,
        upstream_cache=upstream_cache,
        train_artifacts=train_artifacts,
        device=base_model.device,
        seed=cfg.seed,
        verbose=verbose,
    )
    return ExperimentState(
        cfg=cfg,
        base_model=base_model,
        dpt=dpt,
        dr_train=dr_train,
        dr_test=dr_test,
        upstream_cache=upstream_cache,
        train_artifacts=train_artifacts,
        test_artifacts=test_artifacts,
        ctx=ctx,
    )


# --------------------------------------------------------------------------------------
# Table 1 / Table 2 / Figure 3 -- forecasting forgetting
# --------------------------------------------------------------------------------------
def build_forecasters(
    ctx: ForecastContext,
    train_steps: Optional[int] = None,
    include: Optional[Sequence[str]] = None,
) -> Dict[str, BaseForecaster]:
    """Instantiate and train the five methods reported in Table 1."""
    steps = train_steps if train_steps is not None else FORECAST_TRAIN["max_steps"]
    wanted = set(include) if include else {
        "threshold", "fixed_logit", "trainable_logit", "representation", "representation_wo_prior"
    }
    forecasters: Dict[str, BaseForecaster] = {}
    if "threshold" in wanted:
        forecasters["threshold"] = ThresholdForecaster().fit(ctx)
    if "fixed_logit" in wanted:
        forecasters["fixed_logit"] = FixedLogitForecaster().fit(ctx)
    if "trainable_logit" in wanted:
        forecasters["trainable_logit"] = TrainableLogitForecaster(max_steps=steps).fit(ctx)
    if "representation" in wanted:
        forecasters["representation"] = RepresentationForecaster(max_steps=steps).fit(ctx)
    if "representation_wo_prior" in wanted:
        forecasters["representation_wo_prior"] = RepresentationForecaster(
            use_prior=False, max_steps=steps
        ).fit(ctx)
    return forecasters


def evaluate_forecasters(
    forecasters: Dict[str, BaseForecaster],
    artifacts: Sequence[OnlineArtifact],
    ctx: ForecastContext,
) -> Dict[str, Dict[str, float]]:
    return {name: f.evaluate(artifacts, ctx) for name, f in forecasters.items()}


def run_forecasting_experiment(
    cfg: ExperimentConfig,
    train_steps: Optional[int] = None,
    max_upstream: Optional[int] = None,
    max_online: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Table 1 / Table 2: train on ``D_R^Train`` and evaluate on ``D_R^Test``."""
    state = prepare_state(
        cfg, train_steps=train_steps, max_upstream=max_upstream, max_online=max_online, verbose=verbose
    )
    forecasters = build_forecasters(state.ctx, train_steps=train_steps)
    metrics = evaluate_forecasters(forecasters, state.test_artifacts, state.ctx)
    result = {
        "config": cfg.to_dict(),
        "n_upstream": len(state.upstream_cache.examples),
        "n_online_train": len(state.dr_train),
        "n_online_test": len(state.dr_test),
        "base_em_dpt": state.upstream_cache.base_em,
        "results": metrics,
    }
    out_dir = ensure_dir(os.path.join(cfg.output_root, f"table1_{cfg.model}_{cfg.tuning_mode}"))
    write_json(os.path.join(out_dir, "forecasting_f1.json"), result)
    return result


def run_ood_experiment(
    cfg: ExperimentConfig,
    train_steps: Optional[int] = None,
    max_upstream: Optional[int] = None,
    max_online: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Table 2: train on P3-Test_ID, evaluate on P3-Test_ID and P3-Test_OOD."""
    base_model = prepare_model(cfg)
    dpt = build_dpt(cfg)
    if max_upstream is not None:
        dpt = dpt[:max_upstream]
    id_splits, ood_splits = build_ood_split(cfg, base_model)
    upstream_cache = build_upstream_cache(base_model, dpt, verbose=verbose)

    def _artifacts(examples):
        return build_online_artifacts(
            base_model, examples, upstream_cache, steps=cfg.steps_single(), lr=cfg.lr_single(),
            mode=cfg.tuning_mode, verbose=verbose,
        )

    id_train = _artifacts(id_splits.train[:max_online] if max_online else id_splits.train)
    id_test = _artifacts(id_splits.test[:max_online] if max_online else id_splits.test)
    ood_test = _artifacts(ood_splits.test[:max_online] if max_online else ood_splits.test)
    ctx = ForecastContext(cfg, upstream_cache, id_train, base_model.device, seed=cfg.seed, verbose=verbose)
    forecasters = build_forecasters(ctx, train_steps=train_steps, include=[
        "threshold", "trainable_logit", "representation", "representation_wo_prior"])
    metrics = {
        "id": evaluate_forecasters(forecasters, id_test, ctx),
        "ood": evaluate_forecasters(forecasters, ood_test, ctx),
    }
    out_dir = ensure_dir(os.path.join(cfg.output_root, f"table2_{cfg.model}"))
    write_json(os.path.join(out_dir, "ood_f1.json"), metrics)
    return metrics


def run_figure3(
    cfg: ExperimentConfig,
    train_steps: Optional[int] = None,
    max_upstream: Optional[int] = None,
    max_online: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Figure 3: F1 / precision / recall while continually refining the LM."""
    state = prepare_state(
        cfg, train_steps=train_steps, max_upstream=max_upstream, max_online=max_online, verbose=verbose
    )
    forecasters = build_forecasters(
        state.ctx, train_steps=train_steps,
        include=["threshold", "trainable_logit", "representation"],
    )
    curves = continual_forecasting_curves(
        cfg, state.base_model, forecasters, state.dr_test, state.upstream_cache, verbose=verbose
    )
    out_dir = ensure_dir(os.path.join(cfg.output_root, f"figure3_{cfg.model}_{cfg.tuning_mode}"))
    write_json(os.path.join(out_dir, "curves.json"), curves)
    return curves


# --------------------------------------------------------------------------------------
# Table 3 / Table 4 -- model refinement with replay
# --------------------------------------------------------------------------------------
def run_refinement_table(
    cfg: ExperimentConfig,
    single_error: bool = False,
    max_upstream: Optional[int] = None,
    max_online: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Table 3 (continual refinement) or Table 4 (single errors, separately)."""
    state = prepare_state(cfg, max_upstream=max_upstream, max_online=max_online, verbose=verbose)
    pool = state.replay_pool
    results = []

    def _run(method: str, strategy=None):
        runner = evaluate_single_error_replay if single_error else sequential_refinement
        res: RefinementResult = runner(
            cfg,
            state.base_model,
            state.dr_test,
            state.dpt,
            method=method,
            replay_strategy=strategy,
            replay_pool=pool if strategy is not None else None,
            verbose=verbose,
        )
        results.append(res.to_dict())
        return res

    _run("Vanilla FT")
    _run("Replay w/ Random", RandomReplay(seed=cfg.seed))
    forecasters = build_forecasters(
        state.ctx, train_steps=None,
        include=["threshold", "trainable_logit", "representation"],
    )
    for name, label in (
        ("threshold", "Replay w/ Threshold"),
        ("trainable_logit", "Replay w/ Trainable Logit"),
        ("representation", "Replay w/ Representation"),
    ):
        _run(label, ScoreReplay(forecasters[name], state.artifact_lookup, seed=cfg.seed))
    _run("Replay w/ GT Forget", GTForgetReplay(state.label_lookup, seed=cfg.seed))

    out_dir = ensure_dir(
        os.path.join(cfg.output_root, f"{'table4' if single_error else 'table3'}_{cfg.model}_{cfg.tuning_mode}")
    )
    write_json(os.path.join(out_dir, "refinement.json"), results)
    return results
