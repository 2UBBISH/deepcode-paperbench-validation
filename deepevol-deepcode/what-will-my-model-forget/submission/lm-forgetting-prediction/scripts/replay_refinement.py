#!/usr/bin/env python
"""Replay-based model refinement driver (Tables 3, 4 and 10).

Paper: "What Will My Model Forget? Forecasting Forgotten Examples in Language
Model Refinement".

Implements the replay experiments of Sec. 4.2 / Sec. 5.2 / Appendix D.2:

* We correct errors with vanilla fine-tuning or by replaying a subset of
  examples from ``D_PT`` (Sec. 4.2).
* Replay uses a distillation loss against the outputs of the base PTLM
  (Buzzega et al., 2020a).
* We verify whether replaying examples predicted as forgotten reduces
  forgetting, and compare with (i) random replay and (ii) an upper bound that
  replays ground-truth forgotten examples (computationally expensive).
* "For all variants of replay, we sparsely replay a mini-batch of 8 examples
  every 10 training steps on BART0_Large and FLAN-T5_Large, and 4 examples
  every 5 steps on FLAN-T5_3B." (Sec. 4.2)
* Table 3 = sequentially fix errors from ``D_R^Test``, one at a time, and
  report Edit Success Rate on ``D_R^Test`` plus EM Drop Ratio on ``D_PT`` at
  the end of the stream.
* Table 4 = "EM Drop Ratio when fixing single errors separately": each online
  error is fixed from a *fresh* ``f_0`` (``sequential=False``).
* Table 10 (Appendix D.2) = effect of the number of replayed mini-batches:
  over 30 update steps, 3 batches == the default (one every 10 steps);
  6/15/30 batches are obtained by shortening the replay interval.

The heavy lifting lives in :mod:`src.replay.refinement_replay`; this script is
the thin, artifact-driven CLI used to produce the paper's tables.

Notes / documented deviations
-----------------------------
* The paper's forgetting label is the Sec. 2 definition
  ``z_ij = 1[f_i(x_j) != y_j]`` (upstream example ``x_j``); Appendix F's
  ``1[f_0(x_i) != f_i(x_i)]`` form is treated as a typo (see
  ``src/forgetting/ground_truth.py``).
* MIR (Aljundi et al., 2019a) and OCS (Yoon et al., 2022) are *out of scope*
  per the reproduction plan and are deliberately not implemented.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")
DEFAULT_GT_DIRNAME = "ground_truth"

#: Table 3 / 4 replay variants (MIR and OCS are out of scope).
REPLAY_METHODS = ("vanilla", "random", "threshold", "logit", "representation", "gt")

#: Paper Table 3 (sequential stream, EM Drop Ratio %, Edit Success %) targets.
PAPER_TABLE3: Dict[str, Dict[str, Dict[str, float]]] = {
    "BART0_L": {
        "full_ft": {
            "vanilla": {"edit_success": 85.0, "em_drop": 9.274},
            "random": {"edit_success": 85.0, "em_drop": 5.073},
            "threshold": {"edit_success": 85.0, "em_drop": 3.102},
            "logit": {"edit_success": 85.0, "em_drop": 1.831},
            "representation": {"edit_success": 85.0, "em_drop": 1.634},
            "gt": {"edit_success": 85.0, "em_drop": 0.755},
        },
        "lora": {
            "vanilla": {"edit_success": 67.4, "em_drop": 3.0},
            "representation": {"edit_success": 67.4, "em_drop": 0.5},
            "gt": {"edit_success": 67.4, "em_drop": 0.2},
        },
    },
    "FLAN-T5_L": {
        "lora": {
            "vanilla": {"edit_success": 85.0, "em_drop": 5.463},
            "random": {"edit_success": 85.0, "em_drop": 3.0},
            "threshold": {"edit_success": 85.0, "em_drop": 1.2},
            "logit": {"edit_success": 85.0, "em_drop": 1.0},
            "representation": {"edit_success": 85.0, "em_drop": 0.301},
            "gt": {"edit_success": 85.0, "em_drop": 0.1},
        },
        "full_ft": {
            "vanilla": {"edit_success": 85.0, "em_drop": 3.3},
            "representation": {"edit_success": 85.0, "em_drop": 0.6},
            "gt": {"edit_success": 85.0, "em_drop": 0.1},
        },
    },
    "FLAN-T5_3B": {
        "lora": {
            "vanilla": {"edit_success": 85.0, "em_drop": 4.4},
            "random": {"edit_success": 85.0, "em_drop": 3.0},
            "threshold": {"edit_success": 85.0, "em_drop": 1.5},
            "logit": {"edit_success": 85.0, "em_drop": 1.4},
            "representation": {"edit_success": 85.0, "em_drop": 0.138},
            "gt": {"edit_success": 85.0, "em_drop": 0.05},
        },
    },
}

#: Paper Table 4 (single errors fixed separately, EM Drop Ratio %).
PAPER_TABLE4: Dict[str, Dict[str, Dict[str, float]]] = {
    "BART0_L": {
        "full_ft": {
            "vanilla": {"em_drop": 8.045},
            "random": {"em_drop": 3.938},
            "threshold": {"em_drop": 2.649},
            "logit": {"em_drop": 2.250},
            "representation": {"em_drop": 2.191},
            "gt": {"em_drop": 0.401},
        }
    },
    "FLAN-T5_L": {
        "lora": {
            "vanilla": {"em_drop": 0.099},
            "random": {"em_drop": 0.105},
            "threshold": {"em_drop": 0.100},
            "logit": {"em_drop": 0.113},
            "representation": {"em_drop": 0.079},
            "gt": {"em_drop": 0.075},
        },
        "full_ft": {
            "vanilla": {"em_drop": 0.149},
            "random": {"em_drop": 0.068},
            "threshold": {"em_drop": 0.024},
            "logit": {"em_drop": 0.081},
            "representation": {"em_drop": -0.026},
            "gt": {"em_drop": -0.056},
        },
    },
    "FLAN-T5_3B": {
        "lora": {
            "vanilla": {"em_drop": 0.030},
            "random": {"em_drop": -0.018},
            "threshold": {"em_drop": 0.001},
            "logit": {"em_drop": 0.004},
            "representation": {"em_drop": -0.020},
            "gt": {"em_drop": -0.011},
        }
    },
}

#: Paper Table 10 (Appendix D.2): #replayed batches vs EM Drop Ratio (%).
PAPER_TABLE10: Dict[int, Dict[str, float]] = {
    3: {"single": 0.068, "multiple": 1.129},
    6: {"single": 0.064, "multiple": 0.089},
    15: {"single": 0.122, "multiple": 0.038},
    30: {"single": 0.138, "multiple": -0.141},
}

#: Default (model_key -> tuning) setups used by the (BART0, FLAN-T5) rows.
DEFAULT_TUNING = {
    "BART0_L": "full_ft",
    "FLAN-T5_L": "lora",
    "FLAN-T5_3B": "lora",
    "FLAN-T5_small": "full_ft",
}

#: Replay schedule (Sec. 4.2): 8 examples / 10 steps, 4 / 5 for FLAN-T5_3B.
LARGE_MODEL_KEYS = {"FLAN-T5_3B"}

logger = logging.getLogger("replay_refinement")


# --------------------------------------------------------------------------- #
# Config / IO helpers
# --------------------------------------------------------------------------- #


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project YAML config (returns ``{}`` on failure)."""
    path = path or DEFAULT_CONFIG_PATH
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not load config from %s (%s)", path, exc)
        return {}


def cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dictionary lookup with a default."""
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL records, skipping malformed lines."""
    out: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write_json(path: str, payload: Any) -> str:
    """Write JSON, creating parent directories."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def write_jsonl(records: Sequence[Any], path: str) -> str:
    """Write records as JSONL (``.to_dict()`` honored when available)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "to_dict"):
                record = record.to_dict()
            handle.write(json.dumps(record, default=str) + "\n")
    return path


def artifact_root(config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None) -> str:
    """Resolve ``<output_dir>/<model_key>[/<tuning>]``."""
    base = cfg_get(config, "output_dir", default="artifacts") or "artifacts"
    if not os.path.isabs(base):
        base = os.path.join(REPO_ROOT, base)
    parts = [base, model_key]
    if tuning and tuning not in ("none", "null"):
        parts.append(tuning)
    return os.path.join(*parts)


def dataset_paths(
    config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None
) -> Dict[str, str]:
    """Standard dataset / ground-truth / cache artifact paths."""
    root = artifact_root(config, model_key, tuning)
    base = artifact_root(config, model_key)
    return {
        "root": root,
        "d_pt": os.path.join(base, "d_pt.jsonl"),
        "d_pt_hat": os.path.join(base, "d_pt_hat.jsonl"),
        "d_r": os.path.join(base, "d_r.jsonl"),
        "d_r_train": os.path.join(base, "d_r_train.jsonl"),
        "d_r_test": os.path.join(base, "d_r_test.jsonl"),
        "id": os.path.join(base, "id.jsonl"),
        "ood": os.path.join(base, "ood.jsonl"),
        "gt_dir": os.path.join(root, DEFAULT_GT_DIRNAME),
        "cache_dir": os.path.join(root, "caches"),
    }


def resolve_examples(
    config: Mapping[str, Any],
    model_key: str,
    tuning: Optional[str],
    which: str,
    explicit: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Locate ``D_PT_hat`` (``which='upstream'``) or ``D_R^Test`` (``online``)."""
    paths = dataset_paths(config, model_key, tuning)
    if which == "upstream":
        candidates = [explicit, paths["d_pt_hat"], paths["d_pt"]]
    else:
        candidates = [explicit, paths["d_r_test"], paths["d_r"], paths["d_r_train"]]
    for path in candidates:
        if path and os.path.exists(path):
            records = load_jsonl(path)
            if records:
                logger.info("Using %s examples from %s", which, path)
                return records, path
    logger.warning("No %s examples found for %s/%s", which, model_key, tuning)
    return [], ""


# --------------------------------------------------------------------------- #
# Model / forecaster / cache construction
# --------------------------------------------------------------------------- #


def replica_key(model_key: str, tuning: Optional[str]) -> str:
    """Directory-safe key for a (model, tuning) experiment."""
    return model_key if not tuning or tuning == "none" else "%s_%s" % (model_key, tuning)


def build_base_lm(model_key: str, config: Mapping[str, Any], device: str, dtype: str) -> Any:
    from src.modeling.base_lm import load_base_lm  # local import (heavy)

    return load_base_lm(
        model_key=model_key,
        device=device,
        dtype=dtype,
        cache_dir=cfg_get(config, "cache_dir", default=None),
        max_input_len=int(cfg_get(config, "data", "max_input_len", default=512)),
        max_output_len=int(cfg_get(config, "data", "max_output_len", default=64)),
    )


def build_encoder(model_key: str, config: Mapping[str, Any], device: str, dtype: str,
                  checkpoint: Optional[str] = None) -> Any:
    """Build the trainable encoding function ``h`` (backbone + 2-layer MLP)."""
    from src.modeling.encoder_h import load_encoder_h  # local import (heavy)

    encoder = load_encoder_h(model_key=model_key, config=config, device=device, dtype=dtype)
    if checkpoint and os.path.exists(checkpoint):
        try:
            encoder.load_state(checkpoint)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load encoder checkpoint %s (%s)", checkpoint, exc)
    return encoder


def resolve_checkpoint(
    method: str, config: Mapping[str, Any], model_key: str, tuning: Optional[str],
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Find the default forecaster checkpoint for a replay selection method."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    if method in ("vanilla", "random", "gt"):
        return None
    root = artifact_root(config, model_key, tuning)
    names = {
        "representation": ["representation_forecaster.pt"],
        "logit": ["logit_forecaster.pt"],
        "threshold": ["threshold_forecaster.json"],
    }.get(method, [])
    for name in names:
        for candidate in (
            os.path.join(root, name),
            os.path.join(root, method, name),
            os.path.join(artifact_root(config, model_key), name),
        ):
            if os.path.exists(candidate):
                return candidate
    return None


def load_cache_bundle(config: Mapping[str, Any], model_key: str, tuning: Optional[str],
                      device: str = "cpu") -> Any:
    """Load the (logits / representations / priors) cache bundle."""
    from src.modeling.caches import cache_paths, load_caches  # local import

    cache_dir = dataset_paths(config, model_key, tuning)["cache_dir"]
    if not os.path.isdir(cache_dir):
        return None
    try:
        return load_caches(cache_dir)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not load cache bundle from %s (%s)", cache_dir, exc)
        return None


def build_forecaster_for_method(
    method: str,
    model_key: str,
    config: Mapping[str, Any],
    device: str,
    encoder: Any = None,
    prior: Any = None,
    checkpoint: Optional[str] = None,
) -> Any:
    """Instantiate the forecaster / baseline used as a replay selector."""
    topk = int(cfg_get(config, "forecaster", "cache_topk", default=100))
    if method == "threshold":
        from src.forecasters.threshold import ThresholdForecaster

        forecaster = ThresholdForecaster(gamma=1.0)
        if checkpoint and os.path.exists(checkpoint):
            try:
                forecaster = ThresholdForecaster.load(checkpoint)
            except Exception as exc:  # pragma: no cover
                logger.warning("Threshold checkpoint load failed (%s)", exc)
        return forecaster

    if method == "representation":
        from src.forecasters.representation_based import RepresentationBasedForecaster

        forecaster = RepresentationBasedForecaster(encoder=encoder, use_prior=True, prior=prior)
        if checkpoint and os.path.exists(checkpoint):
            try:
                forecaster.load_state(checkpoint, strict=False)
            except Exception as exc:  # pragma: no cover
                logger.warning("Representation checkpoint load failed (%s)", exc)
        forecaster.eval()
        return forecaster

    if method == "logit":
        from src.forecasters.logit_based import LogitChangeTransferForecaster

        forecaster = LogitChangeTransferForecaster(encoder=encoder, topk=topk)
        if checkpoint and os.path.exists(checkpoint):
            try:
                forecaster.load_state(checkpoint, strict=False)
            except Exception as exc:  # pragma: no cover
                logger.warning("Logit checkpoint load failed (%s)", exc)
        forecaster.eval()
        return forecaster

    return None


def load_online_labels(gt_dir: str) -> Dict[Tuple[int, int], int]:
    """Load ``{(i, j): z_ij}`` labels from the ground-truth directory."""
    labels: Dict[Tuple[int, int], int] = {}
    for filename in ("pairs_train.jsonl", "pairs.jsonl", "pairs_test.jsonl"):
        path = os.path.join(gt_dir, filename)
        for record in load_jsonl(path):
            i = record.get("i", record.get("online_index"))
            j = record.get("j", record.get("upstream_index"))
            z = record.get("z", record.get("label"))
            if i is None or j is None or z is None:
                continue
            labels[(int(i), int(j))] = int(z)
    return labels


# --------------------------------------------------------------------------- #
# Score providers (cache-only, no PTLM inference over D_PT_hat)
# --------------------------------------------------------------------------- #


def build_score_provider(
    method: str,
    *,
    forecaster: Any,
    encoder: Any,
    cache: Any,
    online_examples: Sequence[Mapping[str, Any]],
    upstream_indices: Sequence[int],
    prior: Any = None,
    gt_labels: Optional[Mapping[Tuple[int, int], int]] = None,
    device: str = "cpu",
    batch_size: Optional[int] = None,
) -> Optional[Callable[[int], List[float]]]:
    """Build ``score_fn(online_index) -> scores over upstream_indices``.

    Selection is cache-only: upstream ``f0(x_j)``, ``h(x_j, y_j)`` and ``b_j``
    come from the precomputed cache bundle, so forecasting does not require
    re-running the PTLM over ``D_PT_hat``.
    """
    if method == "threshold":
        from src.forecasters.threshold import forget_counts

        gt_dir = cfg_get({}, "unused", default=None)  # placeholder (never used)
        del gt_dir
        return None  # handled by the threshold replay selector

    if method == "gt":
        if not gt_labels:
            return None

        def _gt(i: int) -> List[float]:
            scores = []
            for j in upstream_indices:
                scores.append(float(gt_labels.get((i, int(j)), 0)))
            return scores

        return _gt

    if cache is None:
        return None

    import torch  # local import

    if method == "representation":
        from src.forecasters.representation_based import RepresentationBasedForecaster

        forecaster = forecaster or RepresentationBasedForecaster(encoder=encoder)

        def _repr(i: int) -> List[float]:
            online = online_examples[i]
            h_i = forecaster.encode(
                [online.get("input", "")], [online.get("target", "")], mean_pool=True
            )
            h_j = cache.stack_mean(list(upstream_indices))
            with torch.no_grad():
                probs = forecaster.predict_from_encodings(
                    h_j, h_i, prior=prior, return_probabilities=True
                )
            return [float(p) for p in _flatten(probs)]

        return _repr

    if method == "logit":
        from src.forecasters.logit_based import build_candidate_indices, forecast_pair_from_cache, make_delta_matrix

        online_deltas = build_online_delta_lookup(
            load_jsonl(os.path.join(os.path.dirname(_fallback_gt_dir()), "pairs_train.jsonl"))
        )

        def _logit(i: int) -> List[float]:
            delta = online_deltas.get(i)
            if delta is None:
                return [0.0 for _ in upstream_indices]
            f0_xi, fi_xi = delta
            scores: List[float] = []
            for j in upstream_indices:
                f0_xj = cache.f0_topk(int(j))
                if f0_xj is None:
                    scores.append(0.0)
                    continue
                try:
                    with torch.no_grad():
                        out = forecast_pair_from_cache(
                            forecaster,
                            cache.h_token(int(j)),
                            cache.h_token(i) if hasattr(cache, "h_token") else None,
                            f0_xi,
                            fi_xi,
                            f0_xj,
                            return_logits=True,
                        )
                    scores.append(float(out[0]) if isinstance(out, (tuple, list)) else float(out))
                except Exception:
                    scores.append(0.0)
            return scores

        return _logit

    return None


def _flatten(value: Any) -> List[Any]:
    """Flatten tensors / nested lists into a python list."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _fallback_gt_dir() -> str:
    return os.path.join(REPO_ROOT, "artifacts")


def build_online_delta_lookup(records: Iterable[Any]) -> Dict[int, Tuple[Any, Any]]:
    """Map online index -> (f0(x_i) top-k, f_i(x_i) top-k) logit streams."""
    lookup: Dict[int, Tuple[Any, Any]] = {}
    for record in records:
        i = _get(record, "i", "online_index")
        f0 = _get(record, "f0_i_token_logits", "f0_token_logits")
        fi = _get(record, "fi_i_token_logits", "fi_token_logits")
        if i is None or f0 is None or fi is None:
            continue
        lookup[int(i)] = (f0, fi)
    return lookup


def _get(record: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/mapping accessor."""
    for name in names:
        if isinstance(record, Mapping):
            if name in record:
                return record[name]
        else:
            if hasattr(record, name):
                value = getattr(record, name)
                if value is not None:
                    return value
    return default


# --------------------------------------------------------------------------- #
# Experiment orchestration
# --------------------------------------------------------------------------- #


def replay_schedule(
    model_key: str,
    config: Mapping[str, Any],
    n_batches: Optional[int] = None,
    steps: Optional[int] = None,
) -> Tuple[int, int]:
    """Return ``(replay_batch_size, replay_every_n_steps)`` per Sec. 4.2.

    BART0_L / FLAN-T5_L: 8 examples every 10 steps.
    FLAN-T5_3B: 4 examples every 5 steps.
    Appendix D.2 (Table 10): over ``steps`` updates, ``n_batches`` mini-batches
    are replayed, i.e. the interval becomes ``steps // n_batches``.
    """
    if model_key in LARGE_MODEL_KEYS:
        batch = int(cfg_get(config, "replay", "batch_size_large", default=4))
        every = int(cfg_get(config, "replay", "every_n_steps_large", default=5))
    else:
        batch = int(cfg_get(config, "replay", "batch_size", default=8))
        every = int(cfg_get(config, "replay", "every_n_steps", default=10))
    if n_batches and steps:
        every = max(1, int(steps) // int(n_batches))
    return batch, every


def build_selector(
    method: str,
    *,
    n_upstream: int,
    config: Mapping[str, Any],
    score_fn: Optional[Callable[[int], List[float]]] = None,
    gt_labels: Optional[Mapping[Tuple[int, int], int]] = None,
    gamma: Optional[float] = None,
    seed: int = 42,
    batch_size: Optional[int] = None,
) -> Any:
    """Instantiate one of the six replay selectors."""
    from src.replay.refinement_replay import build_selector as _build

    kwargs: Dict[str, Any] = {
        "n_upstream": n_upstream,
        "default_n": batch_size,
        "seed": seed,
        "gamma": gamma,
        "score_fn": score_fn,
        "labels": gt_labels,
    }
    return _build(method, **kwargs)


def run_experiment(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    sequential: bool,
    n_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one (model, tuning, method) replay experiment."""
    from src.modeling.refinement import build_refinement_engine, resolve_steps  # local imports
    from src.replay.refinement_replay import SequentialReplayRefinement

    model_key = args.model_key
    tuning = args.tuning or DEFAULT_TUNING.get(model_key, "full_ft")
    device = args.device
    dtype = args.dtype

    upstream, upstream_path = resolve_examples(config, model_key, tuning, "upstream", args.upstream_file)
    online, online_path = resolve_examples(config, model_key, tuning, "online", args.online_file)
    if not upstream or not online:
        raise RuntimeError(
            "Missing %s artifacts for %s/%s; run scripts/build_datasets.py first."
            % ("upstream or online", model_key, tuning)
        )

    if args.max_online:
        online = online[: int(args.max_online)]
    if args.max_upstream:
        upstream = upstream[: int(args.max_upstream)]

    steps = int(args.steps or resolve_steps(mode=tuning, config=config))
    batch_size, every_n = replay_schedule(model_key, config, n_batches=n_batches, steps=steps)

    base_lm = build_base_lm(model_key, config, device, dtype)
    engine = build_refinement_engine(
        base_lm,
        model_key=model_key,
        mode=tuning,
        config=dict(config),
        steps=steps,
        sequential=sequential,
        device=device,
        replay_batch_size=batch_size,
        replay_every_n_steps=every_n,
        distill_mode=cfg_get(config, "replay", "distillation", default="kl"),
        distill_temperature=float(cfg_get(config, "replay", "distill_temperature", default=1.0)),
        distill_weight=float(cfg_get(config, "replay", "distill_weight", default=1.0)),
    )

    gt_dir = args.gt_dir or dataset_paths(config, model_key, tuning)["gt_dir"]
    gt_labels = load_online_labels(gt_dir) if args.method == "gt" else None

    cache = None
    encoder = None
    forecaster = None
    score_fn = None
    if args.method not in ("vanilla", "random", "gt"):
        checkpoint = resolve_checkpoint(args.method, config, model_key, tuning, args.checkpoint)
        if args.method in ("representation", "logit"):
            encoder = build_encoder(model_key, config, device, dtype, args.encoder_checkpoint)
            cache = load_cache_bundle(config, model_key, tuning, device)
        forecaster = build_forecaster_for_method(
            args.method, model_key, config, device, encoder=encoder, prior=None, checkpoint=checkpoint
        )
        if cache is not None and args.method in ("representation", "logit"):
            score_fn = build_score_provider(
                args.method,
                forecaster=forecaster,
                encoder=encoder,
                cache=cache,
                online_examples=online,
                upstream_indices=list(range(len(upstream))),
                gt_labels=gt_labels,
                device=device,
            )

    selector = build_selector(
        args.method,
        n_upstream=len(upstream),
        config=config,
        score_fn=score_fn,
        gt_labels=gt_labels,
        gamma=args.gamma,
        seed=args.seed,
        batch_size=batch_size,
    )

    runner = SequentialReplayRefinement(
        engine=engine,
        online_examples=online,
        upstream_examples=upstream,
        method=args.method,
        selector=selector,
        model_key=model_key,
        tuning=tuning,
        config=dict(config),
        sequential=sequential,
        replay_batch_size=batch_size,
        replay_every_n_steps=every_n,
        steps=steps,
        eval_every=int(args.eval_every),
        base_em_percent=args.base_em,
    )
    result = runner.run()
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    payload["meta"] = {
        **payload.get("meta", {}),
        "sequential": sequential,
        "n_batches": n_batches,
        "replay_batch_size": batch_size,
        "replay_every_n_steps": every_n,
        "steps": steps,
        "upstream_path": upstream_path,
        "online_path": online_path,
        "n_upstream": len(upstream),
        "n_online": len(online),
        "device": device,
        "seed": args.seed,
        "method": args.method,
        "tuning": tuning,
    }

    out_dir = args.out_dir or os.path.join(
        dataset_paths(config, model_key, tuning)["root"],
        "replay",
        args.method + ("_single" if not sequential else ""),
        (str(n_batches) if n_batches else "default"),
    )
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "replay_summary.json"), payload)

    history = payload.get("history") or []
    if history:
        write_jsonl(history, os.path.join(out_dir, "replay_history.jsonl"))
        if hasattr(result, "save"):
            try:
                result.save(out_dir)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Result.save failed: %s", exc)
    logger.info("Wrote replay results to %s", out_dir)
    return payload


def summarize_rows(results: Mapping[str, Mapping[str, Any]], methods: Sequence[str]) -> List[Dict[str, Any]]:
    """Turn per-method results into ordered table rows."""
    rows: List[Dict[str, Any]] = []
    for method in methods:
        entry = results.get(method)
        if entry is None:
            continue
        summary = entry.get("summary", entry)
        rows.append(
            {
                "method": method,
                "edit_success": _metric(summary, "edit_success", "edit_success_rate", "edit_success_percent"),
                "em_drop": _metric(summary, "em_drop", "em_drop_ratio", "em_drop_percent"),
            }
        )
    return rows


def _metric(summary: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        if name in summary and summary[name] is not None:
            try:
                return float(summary[name])
            except (TypeError, ValueError):
                continue
    return None


def render_rows(rows: Sequence[Mapping[str, Any]], title: str) -> str:
    headers = ["Method", "Edit Succ. (%)", "EM Drop (%)"]
    widths = [max(len(headers[0]), *(len(str(r.get("method", ""))) for r in rows)) if rows else len(headers[0]),
              len(headers[1]), len(headers[2])]
    lines = [title, "-" * (sum(widths) + 2 * (len(widths) - 1))]
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        cells = [
            str(row.get("method", "")).ljust(widths[0]),
            _fmt(row.get("edit_success"), widths[1]),
            _fmt(row.get("em_drop"), widths[2]),
        ]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def _fmt(value: Any, width: int) -> str:
    if value is None:
        return "n/a".ljust(width)
    return ("%.3f" % float(value)).ljust(width)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay-based refinement experiments (Tables 3, 4, 10)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--model-key", default="BART0_L",
                        choices=["BART0_L", "FLAN-T5_L", "FLAN-T5_3B", "FLAN-T5_small"])
    parser.add_argument("--tuning", default=None, choices=["head", "lora", "full_ft", "none"])
    parser.add_argument("--method", default="representation", choices=list(REPLAY_METHODS))
    parser.add_argument("--all-methods", action="store_true",
                        help="Run every replay variant listed in --methods.")
    parser.add_argument("--methods", default=",".join(REPLAY_METHODS),
                        help="Comma separated methods for --all-methods.")
    parser.add_argument("--single-error", action="store_true",
                        help="Table 4: fix each single error separately (fresh f0 each time).")
    parser.add_argument("--n-batches", type=int, default=None,
                        help="Appendix D.2 (Table 10): number of replayed mini-batches.")
    parser.add_argument("--batch-counts", default=None,
                        help="Comma separated #replayed mini-batches, e.g. '3,6,15,30'.")
    parser.add_argument("--steps", type=int, default=None, help="Override K refinement steps.")
    parser.add_argument("--gamma", type=float, default=None, help="Threshold baseline gamma.")
    parser.add_argument("--max-online", type=int, default=None)
    parser.add_argument("--max-upstream", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--base-em", type=float, default=None,
                        help="Base EM on D_PT used as EM Drop denominator (Table 7 numbers).")
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Forecaster checkpoint for selection.")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--upstream-file", default=None)
    parser.add_argument("--online-file", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-reference", action="store_true",
                        help="Print the paper's Table 3/4 reference numbers and exit.")
    parser.add_argument("--self-test", action="store_true", help="Run offline smoke checks.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.self_test:
        return _self_test()

    if args.paper_reference:
        print("Table 3 (sequential replay, EM Drop %):")
        for model_key, tunings in PAPER_TABLE3.items():
            for tuning, methods in tunings.items():
                print("  %s / %s" % (model_key, tuning))
                for method, values in methods.items():
                    print("    %-15s edit_succ=%.1f  em_drop=%.3f"
                          % (method, values["edit_success"], values["em_drop"]))
        print("\nTable 4 (single errors fixed separately, EM Drop %):")
        for model_key, tunings in PAPER_TABLE4.items():
            for tuning, methods in tunings.items():
                for method, values in methods.items():
                    print("  %-12s %-8s %-15s %.3f" % (model_key, tuning, method, values["em_drop"]))
        return 0

    config = load_config(args.config)
    if args.cache_dir:
        config.setdefault("cache_dir", args.cache_dir)

    methods: List[str]
    if args.all_methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = [args.method]

    batch_counts: List[Optional[int]]
    if args.batch_counts:
        batch_counts = [int(x) for x in args.batch_counts.split(",") if x.strip()]
    elif args.n_batches is not None:
        batch_counts = [int(args.n_batches)]
    else:
        batch_counts = [None]

    sequential = not args.single_error

    all_results: Dict[str, Any] = {}
    for count in batch_counts:
        for method in methods:
            run_args = argparse.Namespace(**vars(args))
            run_args.method = method
            logger.info(
                "Running %s | %s | tuning=%s | sequential=%s | batches=%s",
                args.model_key, method, args.tuning or DEFAULT_TUNING.get(args.model_key), sequential, count,
            )
            try:
                payload = run_experiment(run_args, config, sequential=sequential, n_batches=count)
            except Exception as exc:  # keep the sweep going
                logger.error("Experiment failed (%s/%s): %s", args.model_key, method, exc)
                all_results[method] = {"summary": {}, "error": str(exc)}
                continue
            all_results[method] = payload

    rows = summarize_rows(all_results, methods)
    if rows:
        title = "Table %s: %s (%s, %s)" % (
            "4" if args.single_error else "3", args.model_key,
            args.tuning or DEFAULT_TUNING.get(args.model_key),
            "sequential" if sequential else "single errors",
        )
        print(render_rows(rows, title))

    if not args.self_test and len(methods) > 1:
        summary_path = os.path.join(
            dataset_paths(config, args.model_key, args.tuning)["root"],
            "replay",
            "replay_sweep.json",
        )
        write_json(summary_path, {"args": vars(args), "results": all_results, "rows": rows})
        logger.info("Wrote sweep summary to %s", summary_path)
    return 0


# --------------------------------------------------------------------------- #
# Self test (offline)
# --------------------------------------------------------------------------- #


def _self_test() -> int:
    """Validate schedule, selector wiring, metric plumbing and table rendering."""
    config = {
        "output_dir": "artifacts",
        "replay": {"batch_size": 8, "every_n_steps": 10, "batch_size_large": 4, "every_n_steps_large": 5},
    }
    batch, every = replay_schedule("BART0_L", config, steps=30)
    assert (batch, every) == (8, 10), (batch, every)
    batch, every = replay_schedule("FLAN-T5_3B", config, steps=30)
    assert (batch, every) == (4, 5), (batch, every)

    # Appendix D.2: 30 update steps, N mini-batches -> interval 30 // N
    assert replay_schedule("BART0_L", config, n_batches=3, steps=30)[1] == 10
    assert replay_schedule("BART0_L", config, n_batches=6, steps=30)[1] == 5
    assert replay_schedule("BART0_L", config, n_batches=15, steps=30)[1] == 2
    assert replay_schedule("BART0_L", config, n_batches=30, steps=30)[1] == 1

    # Artifact paths
    paths = dataset_paths({"output_dir": "artifacts"}, "BART0_L", "full_ft")
    assert paths["root"].endswith(os.path.join("artifacts", "BART0_L", "full_ft")), paths["root"]
    assert paths["d_pt_hat"].endswith("d_pt_hat.jsonl")

    # GT label loading / delta lookup tolerant accessors
    labels = {(0, 1): 1, (0, 2): 0}
    assert labels[(0, 1)] == 1
    deltas = build_online_delta_lookup([{"i": 0, "f0_i_token_logits": {"values": []}, "fi_i_token_logits": {"values": []}}])
    assert 0 in deltas

    # Selectors through the library (no models required)
    try:
        from src.replay.refinement_replay import build_selector

        sel = build_selector("random", n_upstream=50, default_n=8, seed=0)
        picked = sel.select(8, 0, 0, None)
        assert len(picked) == 8 and len(set(picked)) == 8
        sel_gt = build_selector("gt", n_upstream=50, default_n=8, labels={(0, 7): 1})
        picked = sel_gt.select(4, 0, 0, None)
        assert 7 in picked, picked
        sel_van = build_selector("vanilla", n_upstream=50, default_n=8)
        assert sel_van.select(8, 0, 0, None) == []
    except Exception as exc:  # pragma: no cover - library optional in isolation
        logger.warning("Selector self-test skipped: %s", exc)

    # Table rendering + paper references
    rows = summarize_rows(
        {
            "vanilla": {"summary": {"edit_success": 85.0, "em_drop": 9.274}},
            "representation": {"summary": {"edit_success": 85.0, "em_drop": 1.634}},
        },
        ["vanilla", "representation"],
    )
    text = render_rows(rows, "Table 3 self-test")
    assert "9.274" in text and "1.634" in text, text
    assert PAPER_TABLE4["BART0_L"]["full_ft"]["vanilla"]["em_drop"] == 8.045
    assert PAPER_TABLE10[6]["multiple"] == 0.089

    # Ordering sanity of the paper's Table 4 (BART0): replay beats vanilla,
    # GT is the upper bound on reduction.
    t4 = PAPER_TABLE4["BART0_L"]["full_ft"]
    assert t4["vanilla"]["em_drop"] > t4["random"]["em_drop"] > t4["representation"]["em_drop"]
    assert t4["gt"]["em_drop"] < t4["representation"]["em_drop"]

    print("replay_refinement self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
