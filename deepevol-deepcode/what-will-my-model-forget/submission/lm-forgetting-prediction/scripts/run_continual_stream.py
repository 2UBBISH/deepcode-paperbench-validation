#!/usr/bin/env python
"""Continual model-refinement stream evaluation (Figure 3 of the paper).

This driver reproduces the "Generalization to Continual Model Refinement"
experiment from Sec. 5.1:

  "We create streams of errors with 1/8 of total examples in D_R in each setup.
   Figure 3 plots the curves of averaged F1, Precision, Recall of forecasting up
   to the end of the streams while we continually fix these errors. We notice
   that precision is mostly stable in the stream, while recall drops over time,
   mostly because more examples are forgotten over time."

Pipeline
--------
1. Build one stream of ``ceil(fraction * |D_R|)`` online error examples by
   shuffling ``D_R`` with a fixed seed (``--fraction`` defaults to 0.125 = 1/8).
2. Continually refine the LM over the stream (sequentially, no reset between
   examples), which is exactly the setting of ``src/replay/refinement_replay.py``
   with the ``vanilla`` replay method (optionally a replay method can be used).
   The refinement run produces, for every visited online example ``i``, the
   ground truth forgetting labels ``z_ij = 1[f_i(x_j) != y_j]`` over
   ``D_PT_hat``.
3. Forecast is computed **once at the start of the stream** and frozen for all
   subsequent steps (``stream.freeze_forecast: true``).  Forecasting is
   cache-only: upstream ``f0(x_j)`` top-k logits, ``h(x_j, y_j)`` and the
   frequency prior ``b_j`` are read from ``src/modeling/caches.py`` artifacts, so
   no PTLM forward pass is needed on ``D_PT_hat`` at forecast time.
4. For every time step ``t`` we accumulate TP/FP/FN over all pairs seen so far
   and record the running averages of F1 / Precision / Recall
   (``src/eval/metrics.average_metrics_up_to_step``); the resulting curves are
   the Figure 3 data.  Per-step (non-cumulative) values are recorded as well.

Outputs (under ``artifacts/<model_key>/<tuning>/stream/``)
---------------------------------------------------------
* ``stream_history.jsonl``   per-step records (TP/FP/FN, edit success, ...)
* ``stream_summary.json``    per-method curves + final metrics + meta
* ``figure3.json``           combined curves for all requested methods
* ``figure3.png``            optional plot (``--plot``, needs matplotlib)

The module is defensive: heavy imports (torch / transformers / project model
code) are performed lazily so that ``--self-test`` runs fully offline.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Project imports
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:  # allow "python scripts/run_continual_stream.py"
    sys.path.insert(0, REPO_ROOT)

from src.data.em_eval import is_correct  # noqa: E402  (SQuAD-2.0 style EM)
from src.eval import metrics as M  # noqa: E402  (canonical metric library)

try:  # replay driver (optional: only needed for the real refinement run)
    from src.replay.refinement_replay import (  # noqa: E402
        REPLAY_METHODS,
        SequentialReplayRefinement,
        build_selector,
        lookup_labels,
        replay_schedule_for,
    )
except Exception as _exc:  # pragma: no cover - defensive
    REPLAY_METHODS = ("vanilla", "random", "threshold", "logit", "representation", "gt")
    SequentialReplayRefinement = None  # type: ignore
    build_selector = None  # type: ignore
    replay_schedule_for = None  # type: ignore

    def rec_get_fallback(record: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(record, Mapping) and name in record:
                return record[name]
            if hasattr(record, name):
                try:
                    return getattr(record, name)
                except Exception:
                    continue
        return default

    def lookup_labels(pair_records: Iterable[Any], aggregate: str = "any") -> Dict[Tuple[int, int], int]:
        """Fallback label lookup: ``{(i, j): z_ij}`` from pair records."""
        out: Dict[Tuple[int, int], int] = {}
        for rec in pair_records or []:
            i = rec_get_fallback(rec, "i", "i_index", "online_index", "online")
            j = rec_get_fallback(rec, "j", "j_index", "upstream_index", "upstream")
            z = rec_get_fallback(rec, "z", "z_ij", "label", "forgotten")
            if i is None or j is None or z is None:
                continue
            key = (int(i), int(j))
            val = int(bool(z))
            if aggregate == "any":
                out[key] = max(out.get(key, 0), val)
            elif aggregate == "all":
                out[key] = min(out.get(key, 1), val)
            else:
                out[key] = val
        return out

    logging.getLogger(__name__).warning(
        "Could not import src.replay.refinement_replay (%s); "
        "stream driver will use precomputed pair records only.",
        _exc,
    )

logger = logging.getLogger("run_continual_stream")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
STREAM_METHODS: Tuple[str, ...] = ("representation", "threshold", "logit")
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "config.yaml")
DEFAULT_FRACTION = 0.125  # 1/8 of D_R, as described in Sec. 5.1
DEFAULT_HISTORY_FILENAME = "stream_history.jsonl"
DEFAULT_SUMMARY_FILENAME = "stream_summary.json"
DEFAULT_FIGURE3_FILENAME = "figure3.json"
DEFAULT_GT_DIRNAME = "ground_truth"

# Paper reference values (Figure 3 is qualitative; these are the Table 1 anchors)
PAPER_TABLE1 = {
    ("BART0_L", "head"): {"threshold": 62.96, "logit": 73.39, "representation": 79.32},
    ("BART0_L", "full_ft"): {"threshold": 55.75, "logit": 57.15, "representation": 67.19},
    ("FLAN-T5_L", "head"): {"threshold": 59.95, "logit": 61.09, "representation": 67.81},
    ("FLAN-T5_L", "lora"): {"threshold": 43.93, "logit": 36.54, "representation": 48.66},
    ("FLAN-T5_L", "full_ft"): {"threshold": 48.43, "logit": 40.91, "representation": 51.51},
    ("FLAN-T5_3B", "head"): {"threshold": 63.64, "logit": 55.07, "representation": 65.93},
    ("FLAN-T5_3B", "lora"): {"threshold": 41.42, "logit": 31.40, "representation": 42.99},
}

DEFAULT_TUNING = {
    "BART0_L": "full_ft",
    "FLAN-T5_L": "lora",
    "FLAN-T5_3B": "lora",
    "FLAN-T5_small": "full_ft",
}


# --------------------------------------------------------------------------- #
# Small utilities (config / IO) -- same conventions as the sibling scripts
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project YAML configuration (returns ``{}`` on failure)."""
    path = path or DEFAULT_CONFIG_PATH
    try:
        import yaml  # optional dependency
    except Exception:  # pragma: no cover
        logger.warning("pyyaml unavailable; using built-in defaults")
        return {}
    for candidate in (path, DEFAULT_CONFIG_PATH):
        if not candidate:
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            if isinstance(cfg, dict):
                return cfg
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to parse config %s: %s", candidate, exc)
    return {}


def cfg_get(cfg: Optional[Mapping[str, Any]], *keys: str, default: Any = None) -> Any:
    """Nested config lookup: ``cfg_get(cfg, 'refinement', 'steps')``."""
    cur: Any = cfg
    for key in keys:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    records: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def write_json(path: str, payload: Any) -> str:
    """Write ``payload`` as JSON, creating parent directories."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def write_jsonl(records: Sequence[Any], path: str) -> str:
    """Write records as JSONL, honouring a ``to_dict`` method when available."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            if hasattr(rec, "to_dict"):
                rec = rec.to_dict()
            fh.write(json.dumps(rec, default=str) + "\n")
    return path


def artifact_root(config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None) -> str:
    """``<output_dir>/<model_key>[/<tuning>]``."""
    out_dir = cfg_get(config, "output_dir", default="artifacts") or "artifacts"
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(REPO_ROOT, out_dir)
    parts = [out_dir, model_key]
    if tuning and tuning not in ("none", "null"):
        parts.append(tuning)
    return os.path.join(*parts)


def dataset_paths(config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None) -> Dict[str, str]:
    """Resolve the standard artifact paths for one ``(model, tuning)`` setup."""
    root = artifact_root(config, model_key, tuning)
    cache_dir = cfg_get(config, "cache_dir", default="artifacts/caches") or "artifacts/caches"
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(REPO_ROOT, cache_dir)
    return {
        "root": root,
        "d_pt": os.path.join(root, "d_pt.jsonl"),
        "d_pt_hat": os.path.join(root, "d_pt_hat.jsonl"),
        "d_r": os.path.join(root, "d_r.jsonl"),
        "d_r_train": os.path.join(root, "d_r_train.jsonl"),
        "d_r_test": os.path.join(root, "d_r_test.jsonl"),
        "gt_dir": os.path.join(root, DEFAULT_GT_DIRNAME),
        "cache_dir": os.path.join(cache_dir, model_key, tuning) if tuning else os.path.join(cache_dir, model_key),
        "stream_dir": os.path.join(root, "stream"),
    }


def rec_get(record: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute/mapping accessor."""
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            try:
                return getattr(record, name)
            except Exception:
                continue
    return default


def filter_kwargs(fn: Callable, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by ``fn`` (safe duck-typed calls)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# Data resolution
# --------------------------------------------------------------------------- #
def resolve_upstream_examples(
    config: Mapping[str, Any],
    model_key: str,
    tuning: Optional[str] = None,
    upstream_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Locate ``D_PT_hat`` (falling back to ``D_PT`` when unavailable)."""
    paths = dataset_paths(config, model_key, tuning)
    for candidate in (upstream_file, paths.get("d_pt_hat"), paths.get("d_pt")):
        if candidate and os.path.exists(candidate):
            records = load_jsonl(candidate)
            if records:
                logger.info("Upstream pool: %s (%d examples)", candidate, len(records))
                return records, candidate
    logger.warning("No upstream examples found; expected %s", paths.get("d_pt_hat"))
    return [], ""


def resolve_online_examples(
    config: Mapping[str, Any],
    model_key: str,
    tuning: Optional[str] = None,
    online_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Locate ``D_R`` (preferring the held-out ``D_R^Test`` split)."""
    paths = dataset_paths(config, model_key, tuning)
    for candidate in (online_file, paths.get("d_r_test"), paths.get("d_r")):
        if candidate and os.path.exists(candidate):
            records = load_jsonl(candidate)
            if records:
                logger.info("Online errors: %s (%d examples)", candidate, len(records))
                return records, candidate
    logger.warning("No online (D_R) examples found; expected %s", paths.get("d_r"))
    return [], ""


# --------------------------------------------------------------------------- #
# Stream construction
# --------------------------------------------------------------------------- #
def stream_size(n_online: int, fraction: float = DEFAULT_FRACTION) -> int:
    """Number of examples in one stream (Sec. 5.1 uses 1/8 of D_R)."""
    if n_online <= 0:
        return 0
    return max(1, int(math.ceil(float(fraction) * n_online)))


def make_streams(
    online_examples: Sequence[Any],
    fraction: float = DEFAULT_FRACTION,
    shuffle: bool = True,
    seed: int = 42,
    n_streams: int = 1,
) -> List[List[int]]:
    """Build ``n_streams`` disjoint streams of ``fraction * |D_R|`` examples.

    Returns a list of lists of *global* indices into ``online_examples``.  The
    paper shuffles the examples so each stream is a random (rather than
    task-contiguous) series of errors.
    """
    n = len(online_examples)
    size = stream_size(n, fraction)
    if size == 0:
        return []
    order = list(range(n))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(order)
    streams: List[List[int]] = []
    for s in range(max(1, int(n_streams))):
        chunk = order[s * size: (s + 1) * size]
        if not chunk:
            break
        streams.append(chunk)
    return streams


# --------------------------------------------------------------------------- #
# Frozen forecasts (computed once at stream start, cache-only)
# --------------------------------------------------------------------------- #
def _to_tensor(x: Any, dtype: Any = None) -> Any:
    import torch

    if isinstance(x, torch.Tensor):
        t = x
    elif x is None:
        return None
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _stack_mean_representations(cache: Any, upstream_indices: Sequence[int], device: str = "cpu") -> Any:
    """Mean-pooled upstream representations ``h(x_j, y_j)`` from the cache."""
    import torch

    if cache is None:
        raise ValueError("cache bundle is required for representation forecasting")
    if hasattr(cache, "stack_mean"):
        try:
            h = cache.stack_mean(list(upstream_indices))
        except TypeError:
            h = cache.stack_mean()
        if h is not None:
            return _to_tensor(h).float().to(device)
    reps = getattr(getattr(cache, "representations", None), "mean", None)
    if reps is None:
        raise ValueError("representation cache does not expose mean-pooled vectors")
    rows = []
    for j in upstream_indices:
        vec = reps.get(j) if hasattr(reps, "get") else reps[int(j)]
        if vec is None:
            raise ValueError("missing mean representation for upstream index %s" % (j,))
        rows.append(_to_tensor(vec).float().flatten())
    return torch.stack(rows, dim=0).to(device)


def _prior_vector(cache: Any, upstream_indices: Sequence[int], use_prior: bool, device: str = "cpu") -> Any:
    import torch

    if not use_prior or cache is None:
        return torch.zeros(len(upstream_indices), dtype=torch.float32, device=device)
    from src.forecasters.representation_based import prior_vector_from

    return _to_tensor(
        prior_vector_from(getattr(cache, "priors", None), list(upstream_indices), device=device)
    ).float().to(device)


def encode_examples(encoder: Any, examples: Sequence[Mapping[str, Any]], device: str = "cpu",
                    batch_size: int = 8, mean_pool: bool = True) -> Any:
    """Encode a list of examples with ``h`` (mean-pooled by default)."""
    import torch

    if encoder is None:
        raise ValueError("an encoder h is required to encode online examples")
    inputs = [str(rec_get(ex, "input", "prompt", default="")) for ex in examples]
    targets = [str(rec_get(ex, "target", "output", default="")) for ex in examples]
    vecs: List[Any] = []
    with torch.no_grad():
        for start in range(0, len(inputs), max(1, batch_size)):
            chunk_in = inputs[start:start + batch_size]
            chunk_tg = targets[start:start + batch_size]
            if hasattr(encoder, "encode_mean") and mean_pool:
                out = encoder.encode_mean(chunk_in, chunk_tg)
            elif hasattr(encoder, "encode"):
                try:
                    out = encoder.encode(chunk_in, chunk_tg, mean_pool=mean_pool)
                except TypeError:
                    out = encoder.encode(chunk_in, chunk_tg)
            else:
                out = encoder(chunk_in, chunk_tg)
            if isinstance(out, Mapping):
                out = out.get("h") or out.get("representations") or out.get("encoded")
            out = _to_tensor(out).float()
            if out.dim() == 3:  # [B, T, d] -> mean pool
                out = out.mean(dim=1)
            vecs.append(out.cpu())
    return torch.cat(vecs, dim=0).to(device)


def compute_frozen_forecasts(
    method: str,
    *,
    model_key: str,
    config: Mapping[str, Any],
    stream_online: Sequence[Mapping[str, Any]],
    upstream_indices: Sequence[int],
    cache: Any = None,
    encoder: Any = None,
    forecaster: Any = None,
    prior: Any = None,
    train_pairs: Optional[Sequence[Any]] = None,
    gt_pairs: Optional[Sequence[Any]] = None,
    device: str = "cpu",
    batch_size: int = 8,
    use_prior: bool = True,
    gamma: Optional[float] = None,
    topk: int = 100,
) -> Dict[str, Any]:
    """Compute the forecast for ``stream_online x upstream_indices`` once.

    Returns a dict with ``z_hat`` (list of lists, ``[n_stream][n_upstream]``),
    optional ``scores`` and metadata describing how the forecast was produced.
    The forecast is intentionally computed with the forecasting models trained
    against ``f0`` and then frozen for the whole stream, matching
    ``stream.freeze_forecast: true``.
    """
    import torch

    n_stream = len(stream_online)
    n_up = len(upstream_indices)
    info: Dict[str, Any] = {"method": method, "n_stream": n_stream, "n_upstream": n_up}
    if n_stream == 0 or n_up == 0:
        return {"z_hat": [], "scores": [], "info": info}

    if method == "representation":
        h_up = _stack_mean_representations(cache, upstream_indices, device=device)
        b = _prior_vector(cache, upstream_indices, use_prior, device=device)
        h_on = encode_examples(encoder, stream_online, device=device, batch_size=batch_size)
        if h_on.shape[-1] != h_up.shape[-1]:
            raise ValueError(
                "dimension mismatch between online h (%s) and upstream h (%s)"
                % (tuple(h_on.shape), tuple(h_up.shape))
            )
        scores = torch.sigmoid(h_on @ h_up.t() + b.unsqueeze(0))
        threshold = 0.5
        val = getattr(forecaster, "decision_threshold", None)
        if isinstance(val, (int, float)):
            threshold = float(val)
        z_hat = (scores >= threshold).int()
        info.update({"use_prior": bool(use_prior), "decision_threshold": threshold,
                     "dim": int(h_up.shape[-1]), "frozen": True})
        return {"z_hat": z_hat.cpu().tolist(), "scores": scores.cpu().tolist(), "info": info}

    if method in ("threshold", "threshold_based", "freq"):
        scores_vec: Optional[Any] = None
        gamma_val = gamma
        if forecaster is not None:
            raw = rec_get(forecaster, "gamma")
            if isinstance(raw, (int, float)):
                gamma_val = float(raw) if gamma_val is None else gamma_val
            if hasattr(forecaster, "score"):
                try:
                    counts = {int(j): float(forecaster.score(int(j))) for j in upstream_indices}
                    scores_vec = torch.tensor([counts[int(j)] for j in upstream_indices], dtype=torch.float32)
                except Exception:
                    scores_vec = None
        if scores_vec is None:
            from src.forecasters.threshold import forget_counts, tune_gamma

            pairs = list(train_pairs or gt_pairs or [])
            counts = forget_counts(pairs, n_upstream=None) if pairs else {}
            scores_vec = torch.tensor(
                [float(counts.get(int(j), counts.get(str(int(j)), 0.0))) for j in upstream_indices],
                dtype=torch.float32,
            )
            if gamma_val is None and pairs:
                tuned = tune_gamma(pairs, n_upstream=None)
                gamma_val = float(rec_get(tuned, "gamma", default=1.0) or 1.0)
        if gamma_val is None:
            gamma_val = 1.0
        z_row = (scores_vec >= float(gamma_val)).int()
        z_hat = z_row.unsqueeze(0).repeat(n_stream, 1)
        info.update({"gamma": float(gamma_val),
                     "note": "static baseline broadcast over online examples", "frozen": True})
        return {"z_hat": z_hat.cpu().tolist(), "scores": scores_vec.unsqueeze(0).cpu().tolist(), "info": info}

    if method in ("logit", "trainable_logit", "fixed_logit"):
        if forecaster is None:
            raise ValueError("logit forecasting requires a (trained) LogitChangeTransferForecaster")
        from src.forecasters.logit_based import forecast_pair_from_cache

        # f_i(x_i) - f_0(x_i) is only available from generated ground-truth pairs
        deltas: Dict[int, Tuple[Any, Any]] = {}
        for rec in list(gt_pairs or []):
            i = rec_get(rec, "i", "i_index", "online_index")
            if i is None:
                continue
            f0 = rec_get(rec, "f0_i_token_logits", "f0_token_logits")
            fi = rec_get(rec, "fi_i_token_logits", "fi_token_logits")
            if f0 is not None and fi is not None:
                deltas[int(i)] = (f0, fi)
        if not deltas:
            raise ValueError(
                "logit forecasting needs cached f0(x_i)/f_i(x_i) logit streams "
                "(run scripts/generate_ground_truth.py first)"
            )
        h_on = encode_examples(encoder, stream_online, device=device, batch_size=batch_size)
        z_rows: List[List[int]] = []
        for t in range(n_stream):
            i_global = int(rec_get(stream_online[t], "_global_index", default=t))
            key = i_global if i_global in deltas else None
            if key is None:  # tolerate local indexing of the ground-truth pairs
                for cand in (stream_online[t].get("_stream_position", t), t + 1):
                    if cand is not None and int(cand) in deltas:
                        key = int(cand)
                        break
            row = [0] * n_up
            if key is not None:
                f0_i, fi_i = deltas[key]
                h_i = h_on[t: t + 1]
                for pos, j in enumerate(upstream_indices):
                    f0_j = cache.f0_topk(int(j)) if hasattr(cache, "f0_topk") else None
                    if f0_j is None:
                        continue
                    h_j = cache.h_token(int(j)) if hasattr(cache, "h_token") else None
                    try:
                        z = forecast_pair_from_cache(
                            forecaster, h_j, h_i, f0_i, fi_i, f0_j,
                            target_ids_i=(), target_ids_j=(),
                        )
                        row[pos] = int(z) if not isinstance(z, tuple) else int(z[0])
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug("logit forecast failed for (i=%s, j=%s): %s", i_global, j, exc)
            z_rows.append(row)
        info.update({"n_online_with_deltas": len(deltas), "frozen": True})
        return {"z_hat": z_rows, "scores": z_rows, "info": info}

    raise ValueError("unknown stream method: %r (expected one of %s)" % (method, STREAM_METHODS))


# --------------------------------------------------------------------------- #
# Ground-truth labels over the stream
# --------------------------------------------------------------------------- #
def group_labels_by_online(pair_records: Sequence[Any]) -> Dict[int, Dict[int, int]]:
    """Group ground-truth pair labels as ``{online i: {upstream j: z_ij}}``."""
    grouped: Dict[int, Dict[int, int]] = {}
    for rec in pair_records or []:
        i = rec_get(rec, "i", "i_index", "online_index", "online")
        j = rec_get(rec, "j", "j_index", "upstream_index", "upstream")
        z = rec_get(rec, "z", "z_ij", "label", "forgotten")
        if i is None or j is None or z is None:
            continue
        grouped.setdefault(int(i), {})[int(j)] = int(bool(z))
    return grouped


def run_stream_refinement(
    *,
    config: Mapping[str, Any],
    model_key: str,
    tuning: Optional[str],
    online_subset: Sequence[Mapping[str, Any]],
    upstream_examples: Sequence[Mapping[str, Any]],
    replay_method: str = "vanilla",
    engine: Any = None,
    predictor: Any = None,
    eval_upstream_indices: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Continually refine the LM over the stream and collect forgetting labels.

    Returns ``(pair_records, online_records, meta)``.  The expensive sequential
    refinement is delegated to ``src.replay.refinement_replay`` (the same code
    path used for Tables 3/4) so the stream setting is identical, i.e. the model
    is *not* reset to ``f0`` between examples.
    """
    if SequentialReplayRefinement is None:
        raise RuntimeError("src.replay.refinement_replay is unavailable; cannot run the stream refinement")

    replay_batch, replay_every = (0, 1)
    if replay_schedule_for is not None:
        try:
            replay_batch, replay_every = replay_schedule_for(model_key, config, None, None)
        except TypeError:
            replay_batch, replay_every = replay_schedule_for(model_key, config)
    if replay_method in ("vanilla", "none"):
        replay_batch, replay_every = 0, 1

    selector = None
    if replay_method not in ("vanilla", "none"):
        if build_selector is None:
            raise RuntimeError("build_selector unavailable for replay-based streams")
        selector = build_selector(
            replay_method,
            n_upstream=len(upstream_examples),
            default_n=max(1, int(replay_batch)),
            seed=int(cfg_get(config, "seed", default=42) or 42),
            gamma=None,
            use_frequency=False,
            score_fn=None,
            min_score=None,
            labels=None,
            n_online=len(online_subset),
        )

    kwargs = dict(
        engine=engine,
        online_examples=list(online_subset),
        upstream_examples=list(upstream_examples),
        method="vanilla" if replay_method in ("vanilla", "none") else replay_method,
        selector=selector,
        model_key=model_key,
        tuning=tuning,
        config=dict(config),
        sequential=True,             # <-- continual refinement (no reset)
        replay_batch_size=max(1, int(replay_batch or 1)),
        replay_every_n_steps=max(1, int(replay_every or 1)),
        shuffle=False,               # the stream order is already fixed by us
        max_online=len(online_subset),
        eval_upstream_indices=list(eval_upstream_indices) if eval_upstream_indices else None,
        predict_hook=predictor,
        distill_mode=str(cfg_get(config, "replay", "distillation", default="kl") or "kl"),
        distill_temperature=float(cfg_get(config, "replay", "distill_temperature", default=1.0) or 1.0),
        distill_weight=float(cfg_get(config, "replay", "distill_weight", default=1.0) or 1.0),
    )
    driver = SequentialReplayRefinement(**filter_kwargs(SequentialReplayRefinement.__init__, kwargs))
    logger.info(
        "Running continual refinement stream: %d online examples, replay=%s (every %d steps, batch %d)",
        len(online_subset), replay_method, replay_every, replay_batch,
    )
    result = driver.run()
    pair_records = list(rec_get(result, "pair_records", default=[]) or [])
    online_records = list(rec_get(result, "online_records", default=[]) or [])
    meta = {
        "summary": rec_get(result, "summary", default={}) or {},
        "meta": rec_get(result, "meta", default={}) or {},
        "replay_method": replay_method,
        "replay_batch_size": replay_batch,
        "replay_every_n_steps": replay_every,
        "sequential": True,
    }
    return pair_records, online_records, meta


# --------------------------------------------------------------------------- #
# Stream metrics (Figure 3 curves)
# --------------------------------------------------------------------------- #
def accumulate_stream_metrics(
    stream_indices: Sequence[int],
    z_hat_rows: Sequence[Sequence[int]],
    grouped_labels: Mapping[int, Mapping[int, int]],
    upstream_indices: Sequence[int],
    online_records: Optional[Sequence[Any]] = None,
    upstream_examples: Optional[Sequence[Any]] = None,
    predictions_by_online: Optional[Mapping[int, str]] = None,
    percent: bool = True,
) -> List[Dict[str, Any]]:
    """Build the per-step history used to draw Figure 3.

    For every step ``t`` (online example ``i_t``) we compare the frozen forecast
    ``z_hat`` against the ground-truth labels of that online example and record
    the TP/FP/FN of *that step*; ``src/eval/metrics.average_metrics_up_to_step``
    then turns the history into the "averaged up to a given time step" curves
    described in Sec. 5.1.
    """
    upstream_list = [int(j) for j in upstream_indices]
    pos_map = {j: idx for idx, j in enumerate(upstream_list)}
    history: List[Dict[str, Any]] = []

    for step, i_global in enumerate(stream_indices):
        labels = grouped_labels.get(int(i_global))
        if labels is None and step in grouped_labels:  # tolerate local indexing
            labels = grouped_labels[step]
        z_hat_row = list(z_hat_rows[step]) if step < len(z_hat_rows) else [0] * len(upstream_list)

        z_true: List[int] = []
        z_pred: List[int] = []
        n_forgotten = 0
        if labels:
            for j, z in labels.items():
                idx = pos_map.get(int(j))
                if idx is None:
                    continue
                z_true.append(int(bool(z)))
                z_pred.append(int(z_hat_row[idx]))
                n_forgotten += int(bool(z))
        if z_true:
            tp, fp, fn, tn = M.confusion_counts(z_true, z_pred)
            step_metrics = M.binary_metrics(z_true, z_pred, percent=percent, with_counts=True)
        else:
            tp = fp = fn = tn = 0
            step_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        record: Dict[str, Any] = {
            "step": int(step),
            "online_index": int(i_global),
            "n_pairs": int(len(z_true)),
            "n_forgotten": int(n_forgotten),
            "forgetting_rate": float(n_forgotten / len(z_true)) if z_true else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(step_metrics.get("precision", 0.0)),
            "recall": float(step_metrics.get("recall", 0.0)),
            "f1": float(step_metrics.get("f1", 0.0)),
        }
        if online_records is not None:
            online_rec = online_records[step] if step < len(online_records) else None
            if isinstance(online_rec, Mapping) and online_rec.get("step") is not None:
                online_rec = online_rec
            es = rec_get(online_rec, "edit_success") if online_rec is not None else None
            if es is not None:
                record["edit_success"] = int(bool(es))
            for key in ("loss", "n_replay", "em_drop_percent_at_step"):
                val = rec_get(online_rec, key) if online_rec is not None else None
                if val is not None:
                    record[key] = float(val) if isinstance(val, (int, float)) else val
        history.append(record)
    return history


def running_curves(history: Sequence[Mapping[str, Any]]) -> List[Dict[str, float]]:
    """Running (cumulative) averages of F1/Precision/Recall up to each step."""
    if not history:
        return []
    try:
        curves = M.average_metrics_up_to_step(list(history))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("average_metrics_up_to_step failed (%s); using local fallback", exc)
        curves = []
        tp = fp = fn = 0
        for rec in history:
            tp += int(rec.get("tp", 0))
            fp += int(rec.get("fp", 0))
            fn += int(rec.get("fn", 0))
            prec = M.precision_from_counts(tp, fp)
            rec_ = M.recall_from_counts(tp, fn)
            curves.append({
                "step": int(rec.get("step", len(curves))),
                "precision": 100.0 * prec,
                "recall": 100.0 * rec_,
                "f1": 100.0 * M.f1_from_counts(tp, fp, fn, precision=prec, recall=rec_),
            })
    return [dict(c) for c in curves]


def summarize_stream(
    history: Sequence[Mapping[str, Any]],
    curves: Sequence[Mapping[str, Any]],
    *,
    method: str,
    model_key: str,
    tuning: Optional[str],
    fraction: float,
    stream_index: int,
    freeze_forecast: bool,
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate a stream run into a JSON-serializable summary."""
    tp = sum(int(r.get("tp", 0)) for r in history)
    fp = sum(int(r.get("fp", 0)) for r in history)
    fn = sum(int(r.get("fn", 0)) for r in history)
    tn = sum(int(r.get("tn", 0)) for r in history)
    final = dict(curves[-1]) if curves else {}
    edit_flags = [int(r.get("edit_success", 0)) for r in history if "edit_success" in r]
    summary: Dict[str, Any] = {
        "method": method,
        "model_key": model_key,
        "tuning": tuning,
        "fraction": float(fraction),
        "stream_index": int(stream_index),
        "freeze_forecast": bool(freeze_forecast),
        "n_steps": len(history),
        "n_pairs": int(tp + fp + fn + tn),
        "n_forgotten": int(tp + fn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "micro_precision": float(M.precision_from_counts(tp, fp)),
        "micro_recall": float(M.recall_from_counts(tp, fn)),
        "micro_f1": float(M.f1_from_counts(tp, fp, fn)),
        "final_running_precision": float(final.get("precision", 0.0)),
        "final_running_recall": float(final.get("recall", 0.0)),
        "final_running_f1": float(final.get("f1", 0.0)),
        "edit_success_rate": float(100.0 * sum(edit_flags) / len(edit_flags)) if edit_flags else None,
        "curves": [dict(c) for c in curves],
        "paper_reference_f1": PAPER_TABLE1.get((model_key, tuning or ""), {}).get(method),
    }
    if meta:
        summary["meta"] = dict(meta)
    return summary


def trend_statement(summary: Mapping[str, Any]) -> str:
    """Qualitative check matching the paper's Figure 3 claim (Sec. 5.1).

    The paper observes: "precision is mostly stable in the stream, while recall
    drops over time, mostly because more examples are forgotten over time."
    """
    curves = list(summary.get("curves") or [])
    if len(curves) < 4:
        return "stream too short for a trend statement"
    half = len(curves) // 2
    p_early = _mean([c.get("precision", 0.0) for c in curves[:half]])
    p_late = _mean([c.get("precision", 0.0) for c in curves[half:]])
    r_early = _mean([c.get("recall", 0.0) for c in curves[:half]])
    r_late = _mean([c.get("recall", 0.0) for c in curves[half:]])
    return (
        "precision %.2f -> %.2f (delta %+.2f); recall %.2f -> %.2f (delta %+.2f)"
        % (p_early, p_late, p_late - p_early, r_early, r_late, r_late - r_early)
    )


# --------------------------------------------------------------------------- #
# Model / artifact loading helpers (lazy, heavy)
# --------------------------------------------------------------------------- #
def build_base_lm(model_key: str, config: Mapping[str, Any], device: str, dtype: str) -> Any:
    """Load the base PTLM ``f0``."""
    from src.modeling.base_lm import load_base_lm

    return load_base_lm(
        model_key=model_key,
        device=device,
        dtype=dtype,
        cache_dir=cfg_get(config, "cache_dir"),
        max_input_len=int(cfg_get(config, "data", "max_input_len", default=512) or 512),
        max_output_len=int(cfg_get(config, "data", "max_output_len", default=64) or 64),
    )


def build_engine(base_lm: Any, model_key: str, config: Mapping[str, Any], tuning: Optional[str],
                 device: str, dtype: str) -> Any:
    """Build a *sequential* refinement engine for the stream."""
    from src.modeling.refinement import build_refinement_engine

    mode = tuning or DEFAULT_TUNING.get(model_key, "full_ft")
    kwargs = dict(
        base_lm=base_lm,
        model_key=model_key,
        mode=mode,
        config=dict(config),
        device=device,
        dtype=dtype,
        sequential=True,
        verbose=False,
    )
    try:
        return build_refinement_engine(**kwargs)
    except TypeError:
        return build_refinement_engine(**filter_kwargs(build_refinement_engine, kwargs))


def load_cache_bundle(config: Mapping[str, Any], model_key: str, tuning: Optional[str], device: str = "cpu") -> Any:
    """Load the cache bundle (logit / representation / prior caches)."""
    from src.modeling.caches import load_caches

    cache_dir = dataset_paths(config, model_key, tuning)["cache_dir"]
    if not os.path.isdir(cache_dir):
        logger.warning("cache dir %s does not exist", cache_dir)
    return load_caches(cache_dir)


def build_encoder(model_key: str, config: Mapping[str, Any], device: str, dtype: str,
                  checkpoint: Optional[str] = None) -> Any:
    """Load the encoding function ``h`` (optionally restoring trained weights)."""
    from src.modeling.encoder_h import load_encoder_h

    kwargs = dict(model_key=model_key, config=dict(config), device=device, dtype=dtype)
    try:
        encoder = load_encoder_h(**kwargs)
    except TypeError:
        encoder = load_encoder_h(**filter_kwargs(load_encoder_h, kwargs))
    if checkpoint and os.path.exists(checkpoint) and hasattr(encoder, "load_state"):
        try:
            encoder.load_state(checkpoint, strict=False)
            logger.info("Restored encoder h weights from %s", checkpoint)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not restore encoder weights from %s: %s", checkpoint, exc)
    return encoder


def build_forecaster_for_method(method: str, model_key: str, config: Mapping[str, Any],
                                device: str = "cpu", **kwargs: Any) -> Any:
    """Instantiate the forecaster used to freeze the stream forecast."""
    if method == "representation":
        from src.forecasters.representation_based import RepresentationBasedForecaster

        return RepresentationBasedForecaster(**filter_kwargs(RepresentationBasedForecaster.__init__, kwargs))
    if method in ("logit", "trainable_logit"):
        from src.forecasters.logit_based import LogitChangeTransferForecaster

        return LogitChangeTransferForecaster(**filter_kwargs(LogitChangeTransferForecaster.__init__, kwargs))
    if method in ("fixed_logit",):
        from src.forecasters.logit_based import FixedLogitForecaster

        return FixedLogitForecaster(**filter_kwargs(FixedLogitForecaster.__init__, kwargs))
    if method in ("threshold", "threshold_based", "freq"):
        from src.forecasters.threshold import ThresholdForecaster

        return ThresholdForecaster(**filter_kwargs(ThresholdForecaster.__init__, kwargs))
    raise ValueError("unknown method %r" % (method,))


def resolve_checkpoint(config: Mapping[str, Any], model_key: str, tuning: Optional[str], method: str,
                       explicit: Optional[str] = None) -> Optional[str]:
    """Best-effort lookup of a trained forecaster checkpoint."""
    if explicit:
        return explicit
    paths = dataset_paths(config, model_key, tuning)
    names = {
        "representation": ["representation_forecaster.pt", "representation_forecaster_no_prior.pt"],
        "logit": ["logit_forecaster.pt"],
        "threshold": ["threshold_forecaster.json"],
    }.get(method, [])
    candidates: List[str] = []
    for name in names:
        candidates += [
            os.path.join(paths["root"], name),
            os.path.join(paths["root"], method, name),
            os.path.join(paths["root"], "forecasters", name),
        ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Main experiment driver
# --------------------------------------------------------------------------- #
def run_method_stream(
    method: str,
    *,
    config: Mapping[str, Any],
    model_key: str,
    tuning: Optional[str],
    online_examples: Sequence[Mapping[str, Any]],
    upstream_examples: Sequence[Mapping[str, Any]],
    stream: Sequence[int],
    args: argparse.Namespace,
    pair_records: Optional[Sequence[Any]] = None,
    online_records: Optional[Sequence[Any]] = None,
    engine: Any = None,
    predictor: Any = None,
    cache: Any = None,
    encoder: Any = None,
    forecaster: Any = None,
    prior: Any = None,
) -> Dict[str, Any]:
    """Run one forecasting method over one pre-defined stream."""
    upstream_indices = list(range(len(upstream_examples)))
    if args.max_upstream:
        upstream_indices = upstream_indices[: int(args.max_upstream)]

    stream_online = []
    for pos, gidx in enumerate(stream):
        ex = dict(online_examples[gidx])
        ex["_global_index"] = int(gidx)
        ex["_stream_position"] = int(pos)
        stream_online.append(ex)

    t0 = time.time()
    forecast = compute_frozen_forecasts(
        method,
        model_key=model_key,
        config=config,
        stream_online=stream_online,
        upstream_indices=upstream_indices,
        cache=cache,
        encoder=encoder,
        forecaster=forecaster,
        prior=prior,
        train_pairs=pair_records,
        gt_pairs=pair_records,
        device=args.device,
        batch_size=args.batch_size,
        use_prior=not getattr(args, "no_prior", False),
        gamma=args.gamma,
        topk=int(cfg_get(config, "forecaster", "cache_topk", default=100) or 100),
    )
    forecast_time = time.time() - t0

    if not getattr(args, "freeze_forecast", True):
        logger.warning("--no-freeze-forecast is not supported; forecast stays frozen by design")

    grouped = group_labels_by_online(list(pair_records or []))
    history = accumulate_stream_metrics(
        stream_indices=list(stream),
        z_hat_rows=forecast["z_hat"],
        grouped_labels=grouped,
        upstream_indices=upstream_indices,
        online_records=online_records,
        upstream_examples=upstream_examples,
        percent=bool(cfg_get(config, "eval", "percent", default=True) is not False),
    )
    curves = running_curves(history)
    summary = summarize_stream(
        history,
        curves,
        method=method,
        model_key=model_key,
        tuning=tuning,
        fraction=float(args.fraction),
        stream_index=int(args.stream_index),
        freeze_forecast=bool(getattr(args, "freeze_forecast", True)),
        meta={
            "forecast_info": forecast.get("info", {}),
            "forecast_time_seconds": forecast_time,
            "n_upstream_forecast": len(upstream_indices),
            "use_prior": not getattr(args, "no_prior", False),
        },
    )
    summary["trend"] = trend_statement(summary)
    return {"history": history, "curves": curves, "summary": summary, "forecast_info": forecast.get("info", {})}


def _remap_pairs_to_global(pair_records: Sequence[Any], stream: Sequence[int]) -> List[Any]:
    """Map local stream indices in pair records back to global ``D_R`` indices."""
    stream_list = [int(i) for i in stream]
    remapped: List[Any] = []
    for rec in pair_records:
        i = rec_get(rec, "i", "i_index", "online_index")
        if i is None:
            continue
        i = int(i)
        new_i = int(stream_list[i]) if 0 <= i < len(stream_list) else i
        if isinstance(rec, Mapping):
            new_rec = dict(rec)
            new_rec["i"] = new_i
            remapped.append(new_rec)
        else:
            try:
                setattr(rec, "i", new_i)
            except Exception:
                pass
            remapped.append(rec)
    return remapped


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Full stream experiment: build the stream, refine, forecast, evaluate."""
    config = load_config(args.config)
    model_key = args.model_key
    tuning = None if args.tuning in (None, "", "none") else args.tuning
    if tuning is None:
        tuning = DEFAULT_TUNING.get(model_key, "full_ft")

    upstream_examples, upstream_path = resolve_upstream_examples(
        config, model_key, tuning, upstream_file=args.upstream_file
    )
    online_examples, online_path = resolve_online_examples(
        config, model_key, tuning, online_file=args.online_file
    )
    if not upstream_examples or not online_examples:
        logger.error("Missing data artifacts; run scripts/build_datasets.py first")
        return {"error": "missing_artifacts", "upstream_path": upstream_path, "online_path": online_path}

    if args.max_online:
        online_examples = list(online_examples)[: int(args.max_online)]

    streams = make_streams(
        online_examples,
        fraction=args.fraction,
        shuffle=not args.no_shuffle,
        seed=int(args.seed),
        n_streams=int(args.n_streams),
    )
    if not streams:
        logger.error("Could not build any stream")
        return {"error": "empty_stream"}
    stream_index = min(max(0, int(args.stream_index)), len(streams) - 1)
    stream = streams[stream_index]
    logger.info(
        "Stream %d/%d: %d online examples (fraction %.3f of %d), shuffle=%s",
        stream_index + 1, len(streams), len(stream), args.fraction, len(online_examples), not args.no_shuffle,
    )

    # ---- continual refinement over the stream (ground-truth forgetting) ---- #
    pair_records: List[Any] = []
    online_records: List[Any] = []
    refinement_meta: Dict[str, Any] = {}
    if args.pairs_file:
        pair_records = load_jsonl(args.pairs_file)
        online_records = load_jsonl(args.online_file_gt) if args.online_file_gt else []
        refinement_meta = {"source": args.pairs_file, "precomputed": True}
        logger.info("Loaded %d precomputed pair records from %s", len(pair_records), args.pairs_file)
    else:
        base_lm = build_base_lm(model_key, config, args.device, args.dtype)
        engine = build_engine(base_lm, model_key, config, tuning, args.device, args.dtype)
        predictor = engine
        try:
            pair_records, online_records, refinement_meta = run_stream_refinement(
                config=config,
                model_key=model_key,
                tuning=tuning,
                online_subset=[dict(online_examples[i], _global_index=int(i)) for i in stream],
                upstream_examples=upstream_examples,
                replay_method=args.replay_method,
                engine=engine,
                predictor=predictor,
                eval_upstream_indices=list(range(len(upstream_examples)))[: int(args.max_upstream or 0)] or None,
                batch_size=int(args.batch_size),
                device=args.device,
                verbose=bool(args.verbose),
            )
        except Exception as exc:
            logger.error("Stream refinement failed: %s", exc, exc_info=bool(args.verbose))
            return {"error": "refinement_failed", "exception": str(exc)}
        pair_records = _remap_pairs_to_global(pair_records, stream)

    grouped = group_labels_by_online(pair_records)
    n_labelled_pairs = sum(len(v) for v in grouped.values())
    n_pos = sum(sum(v.values()) for v in grouped.values())
    logger.info(
        "Ground truth: %d online examples, %d pairs, positive prevalence %.4f",
        len(grouped), n_labelled_pairs, (n_pos / n_labelled_pairs) if n_labelled_pairs else 0.0,
    )

    # ---- cache-only forecasting at stream start (frozen) ---- #
    methods: List[str] = list(STREAM_METHODS) if args.method in ("all", "figure3") else [args.method]
    cache = encoder = prior = None
    forecasters: Dict[str, Any] = {}
    if any(m in ("representation", "logit") for m in methods):
        try:
            cache = load_cache_bundle(config, model_key, tuning, device=args.device)
        except Exception as exc:
            logger.error("Could not load cache bundle: %s", exc)
            return {"error": "missing_cache", "exception": str(exc)}
        checkpoint = resolve_checkpoint(config, model_key, tuning, "representation", args.checkpoint)
        encoder = build_encoder(model_key, config, args.device, args.dtype, checkpoint=checkpoint)
        prior = getattr(cache, "priors", None)

    for method in methods:
        if method == "threshold":
            forecasters[method] = build_forecaster_for_method(
                "threshold", model_key, config, device=args.device,
                **({"gamma": args.gamma} if args.gamma else {}),
            )
            continue
        try:
            forecasters[method] = build_forecaster_for_method(
                method, model_key, config, device=args.device,
                dim=int(cfg_get(config, "encoder_h", "repr_dim", default=768) or 768),
                prior=prior,
                use_prior=not args.no_prior,
                topk=int(cfg_get(config, "forecaster", "cache_topk", default=100) or 100),
            )
        except Exception as exc:
            logger.warning("Could not build forecaster for %s: %s", method, exc)
            forecasters[method] = None

    out_dir = os.path.join(dataset_paths(config, model_key, tuning)["stream_dir"],
                           "stream%d_%s" % (stream_index, args.replay_method))
    os.makedirs(out_dir, exist_ok=True)

    results: Dict[str, Any] = {}
    combined: Dict[str, Any] = {
        "model_key": model_key, "tuning": tuning, "stream_index": stream_index,
        "fraction": args.fraction, "n_stream_examples": len(stream),
        "n_upstream": len(upstream_examples), "methods": {},
    }
    for method in methods:
        try:
            out = run_method_stream(
                method,
                config=config,
                model_key=model_key,
                tuning=tuning,
                online_examples=online_examples,
                upstream_examples=upstream_examples,
                stream=stream,
                args=args,
                pair_records=pair_records,
                online_records=online_records,
                engine=engine if not args.pairs_file else None,
                predictor=predictor if not args.pairs_file else None,
                cache=cache,
                encoder=encoder,
                forecaster=forecasters.get(method),
                prior=prior,
            )
        except Exception as exc:
            logger.error("Method %s failed on the stream: %s", method, exc, exc_info=bool(args.verbose))
            results[method] = {"error": str(exc)}
            continue
        results[method] = out
        combined["methods"][method] = out["summary"]
        write_jsonl(out["history"], os.path.join(out_dir, "%s_%s" % (method, DEFAULT_HISTORY_FILENAME)))
        write_json(out["summary"], os.path.join(out_dir, "%s_%s" % (method, DEFAULT_SUMMARY_FILENAME)))
        logger.info(
            "[%s] running stream F1=%.2f | precision=%.2f recall=%.2f | %s",
            method,
            out["summary"]["final_running_f1"],
            out["summary"]["final_running_precision"],
            out["summary"]["final_running_recall"],
            out["summary"].get("trend", ""),
        )

    combined["refinement"] = refinement_meta
    combined_path = write_json(combined, os.path.join(out_dir, DEFAULT_FIGURE3_FILENAME))

    # Optional plot via the evaluation harness (matplotlib).
    figure_path = None
    if args.plot:
        try:
            from src.eval.evaluate import write_figure3

            curves = {m: r["curves"] for m, r in results.items() if isinstance(r, dict) and r.get("curves")}
            if curves:
                figure_path = write_figure3(curves, os.path.join(out_dir, "figure3.png"), plot=True)
        except Exception as exc:
            logger.warning("Could not render Figure 3 plot: %s", exc)

    output = {
        "out_dir": out_dir,
        "figure3": combined_path,
        "figure3_png": figure_path,
        "results": {m: (r.get("summary") if isinstance(r, dict) and "summary" in r else r) for m, r in results.items()},
        "refinement": refinement_meta,
    }
    write_json(output, os.path.join(out_dir, "run_meta.json"))
    return output


# --------------------------------------------------------------------------- #
# Self-test (fully offline)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    """Offline checks: stream construction, metric accumulation, curves."""
    upstream = [{"input": "u%d" % j, "target": str(j)} for j in range(20)]
    online = [{"input": "o%d" % i, "target": str(i)} for i in range(16)]

    streams = make_streams(online, fraction=0.125, shuffle=True, seed=42)
    assert len(streams) == 1 and len(streams[0]) == 2, streams
    assert make_streams(online, fraction=0.125, shuffle=True, seed=42)[0] == streams[0]
    assert make_streams(online, fraction=0.5, shuffle=False, seed=1)[0] == list(range(8))
    assert stream_size(40, 0.125) == 5 and stream_size(0, 0.125) == 0

    stream = streams[0]
    grouped: Dict[int, Dict[int, int]] = {}
    for i in stream:
        grouped[i] = {j: int(j < (i % 7) + 1) for j in range(20)}
    z_hat = [[1 if j < 3 else 0 for j in range(20)] for _ in stream]
    history = accumulate_stream_metrics(stream, z_hat, grouped, list(range(20)))
    assert len(history) == len(stream)
    assert all("tp" in h and "f1" in h for h in history)
    curves = running_curves(history)
    assert len(curves) == len(history)
    assert all(c["f1"] >= 0.0 for c in curves)
    summary = summarize_stream(history, curves, method="representation", model_key="BART0_L",
                              tuning="head", fraction=0.125, stream_index=0, freeze_forecast=True)
    assert summary["n_steps"] == len(stream) and summary["n_pairs"] > 0
    assert 0.0 <= summary["micro_f1"] <= 1.0
    trend = trend_statement(summary)
    assert "precision" in trend and "recall" in trend

    # Remapping from local stream indices back to global D_R indices.
    recs = [{"i": 0, "j": 1, "z": 1}, {"i": 1, "j": 2, "z": 0}]
    remapped = _remap_pairs_to_global(recs, stream)
    assert [r["i"] for r in remapped] == [int(stream[0]), int(stream[1])]
    assert group_labels_by_online(remapped)[int(stream[0])][1] == 1

    # Empty / degenerate inputs must not crash.
    assert make_streams([], 0.125, True, 0) == []
    assert running_curves([]) == []
    assert accumulate_stream_metrics([], [], {}, []) == []
    assert trend_statement({"curves": []}) == "stream too short for a trend statement"

    logger.info("self-test passed (stream size=%d, pairs=%d, running F1=%.2f)",
                len(stream), summary["n_pairs"], summary["final_running_f1"])
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continual-stream forgetting forecasting (Figure 3 of the paper)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="path to config/config.yaml")
    parser.add_argument("--model-key", default="BART0_L",
                        choices=["BART0_L", "FLAN-T5_L", "FLAN-T5_3B", "FLAN-T5_small"])
    parser.add_argument("--tuning", default=None,
                        choices=[None, "head", "lora", "full_ft", "none"],
                        help="LM tuning setup used for the continual refinement")
    parser.add_argument("--method", default="all",
                        help="forecasting method: one of %s, or 'all' for Figure 3" % (", ".join(STREAM_METHODS),))
    parser.add_argument("--replay-method", default="vanilla",
                        choices=list(REPLAY_METHODS),
                        help="replay strategy used while continually refining (default: vanilla FT)")
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION,
                        help="stream size as a fraction of |D_R| (default 1/8 = 0.125)")
    parser.add_argument("--n-streams", type=int, default=1, help="number of disjoint streams to build")
    parser.add_argument("--stream-index", type=int, default=0, help="which stream to evaluate")
    parser.add_argument("--no-shuffle", action="store_true", help="keep the original D_R ordering")
    parser.add_argument("--no-prior", action="store_true", help="'w/o Prior' ablation for representation h")
    parser.add_argument("--gamma", type=float, default=None, help="override the threshold baseline gamma")
    parser.add_argument("--seed", type=int, default=None, help="random seed (defaults to config.seed)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-online", type=int, default=None, help="limit |D_R| (debug)")
    parser.add_argument("--max-upstream", type=int, default=None, help="limit |D_PT_hat| (debug)")
    parser.add_argument("--upstream-file", default=None, help="explicit D_PT_hat jsonl")
    parser.add_argument("--online-file", default=None, help="explicit D_R jsonl")
    parser.add_argument("--pairs-file", default=None,
                        help="use precomputed ground-truth pairs instead of running refinement")
    parser.add_argument("--online-file-gt", default=None, help="optional online records jsonl (with --pairs-file)")
    parser.add_argument("--checkpoint", default=None, help="forecaster checkpoint for h / forecasting")
    parser.add_argument("--freeze-forecast", dest="freeze_forecast", action="store_true", default=True)
    parser.add_argument("--no-freeze-forecast", dest="freeze_forecast", action="store_false",
                        help="kept for CLI compatibility; the forecast is always frozen (Sec. 5.1)")
    parser.add_argument("--plot", action="store_true", help="render figure3.png (needs matplotlib)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="run offline smoke tests and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        args.device = "cpu"
    if args.seed is None:
        cfg = load_config(args.config)
        args.seed = int(cfg_get(cfg, "seed", default=42) or 42)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    try:
        output = run(args)
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("interrupted")
        return 130
    except Exception as exc:
        logger.error("stream evaluation failed: %s", exc, exc_info=bool(args.verbose))
        return 1
    if isinstance(output, Mapping) and output.get("error"):
        logger.error("stream evaluation finished with error: %s", output.get("error"))
        return 2
    logger.info("Wrote stream artifacts to %s", output.get("out_dir"))
    logger.info("Figure 3 data: %s", output.get("figure3"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
