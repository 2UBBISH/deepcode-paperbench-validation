#!/usr/bin/env python
"""Run a forgetting forecaster ``g`` over ``D_PT_hat`` with cache-only inference.

This is the driver for the forecasting step of the paper ("What Will My Model
Forget? Forecasting Forgotten Examples in Language Model Refinement").

Given one online example (an error being fixed) ``<x_i, y_i>``, the forecaster
predicts ``z_hat_ij`` for *every* upstream example ``x_j`` in ``D_PT_hat``:

    z_hat_ij = 1  <=>  g predicts that ``x_j`` will be forgotten after the
                       single-example refinement ``f_0 -> f_i``.

Supported methods (all consistent with Sec. 3 of the paper):

* ``threshold``      -- frequency-threshold baseline (Eq. 1, Sec. 3.1);
* ``representation`` -- black-box ``sigmoid(<h_j, h_i> + b_j)`` (Eq. 4, Sec. 3.3);
* ``logit``          -- partially interpretable logit-change transfer
                        forecaster (Eq. 2/3, Sec. 3.2).

Efficiency (Sec. 3.2, "Efficient Inference"): upstream ``f_0(x_j)`` top-k
logits, ``h(x_j, y_j)`` representations and the frequency priors ``b_j`` are
read from the cache bundle produced by :mod:`src.modeling.caches`; the PTLM is
never re-run on ``D_PT_hat`` at forecast time.  Only the online example is
encoded (once per online example).

Outputs (written into ``--out-dir``):

* ``forecast_predictions.jsonl`` -- one line per ``(i, j)`` pair with
  ``z_hat``/``score``; or per online example with arrays when
  ``--output-format arrays`` (default) is used;
* ``forecast_summary.json``       -- prevalence, score statistics and, when
  ground-truth pairs are supplied, precision / recall / F1.

Usage::

    python scripts/forecast_forgetting.py --model-key BART0_L \
        --method representation --tuning head \
        --out-dir artifacts/BART0_L/head/forecast

The script is deliberately defensive: ground-truth comparison, thresholds and
per-method deltas are all optional so it can be run with partial artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path handling: allow ``python scripts/forecast_forgetting.py`` from anywhere.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logger = logging.getLogger("forecast_forgetting")

DEFAULT_CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")
DEFAULT_TASKS_PATH = os.path.join(ROOT, "config", "tasks.yaml")

METHODS = ("representation", "logit", "threshold")
DEFAULT_CHECKPOINT_NAMES = {
    "representation": "representation_forecaster.pt",
    "logit": "logit_forecaster.pt",
    "threshold": "threshold_forecaster.json",
}
DEFAULT_PREDICTIONS_FILENAME = "forecast_predictions.jsonl"
DEFAULT_SUMMARY_FILENAME = "forecast_summary.json"


# ---------------------------------------------------------------------------
# Config / json helpers
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project YAML config; returns ``{}`` when unavailable."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        logger.warning("config not found at %s; using defaults", path)
        return {}
    try:
        import yaml  # local import: optional dependency

        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if not isinstance(cfg, dict):
            logger.warning("config %s is not a mapping; ignoring", path)
            return {}
        return cfg
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to parse config %s (%s); using defaults", path, exc)
        return {}


def cfg_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested lookup helper: ``cfg_get(cfg, "forecaster", "max_steps")``."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSON line in %s", path)
    return records


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------
def artifact_root(config: Dict[str, Any], model_key: str, tuning: Optional[str] = None) -> str:
    out_dir = cfg_get(config, "output_dir", default="artifacts") or "artifacts"
    root = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    root = os.path.join(root, model_key)
    if tuning:
        root = os.path.join(root, tuning)
    return root


def dataset_paths(config: Dict[str, Any], model_key: str, tuning: Optional[str] = None) -> Dict[str, str]:
    root = artifact_root(config, model_key, tuning)
    return {
        "data_dir": root,
        "d_pt": os.path.join(root, "d_pt.jsonl"),
        "d_pt_hat": os.path.join(root, "d_pt_hat.jsonl"),
        "d_r": os.path.join(root, "d_r.jsonl"),
        "d_r_test": os.path.join(root, "d_r_test.jsonl"),
        "gt_dir": os.path.join(root, "ground_truth"),
        "cache_dir": os.path.join(root, "caches"),
    }


def resolve_upstream_examples(
    config: Dict[str, Any],
    model_key: str,
    tuning: Optional[str],
    upstream_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Locate ``D_PT_hat`` (falling back to ``D_PT`` with a warning)."""
    paths = dataset_paths(config, model_key, tuning)
    candidates = [upstream_file] if upstream_file else []
    candidates += [paths["d_pt_hat"], paths["d_pt"]]
    for path in candidates:
        if path and os.path.exists(path):
            examples = load_jsonl(path)
            if examples:
                if not path.endswith("d_pt_hat.jsonl"):
                    logger.warning(
                        "D_PT_hat not found; using unfiltered pool %s (%d examples)",
                        path,
                        len(examples),
                    )
                return examples, path
    raise FileNotFoundError(
        "no upstream pool found; run scripts/build_datasets.py first (looked in %s)"
        % ", ".join(str(c) for c in candidates)
    )


def resolve_online_examples(
    config: Dict[str, Any],
    model_key: str,
    tuning: Optional[str],
    online_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Locate the online error pool (``D_R^Test`` preferred, then ``D_R``)."""
    paths = dataset_paths(config, model_key, tuning)
    candidates = [online_file] if online_file else []
    candidates += [paths["d_r_test"], paths["d_r"]]
    for path in candidates:
        if path and os.path.exists(path):
            examples = load_jsonl(path)
            if examples:
                return examples, path
    raise FileNotFoundError(
        "no online (D_R) examples found; run scripts/build_datasets.py first (looked in %s)"
        % ", ".join(str(c) for c in candidates)
    )


def resolve_checkpoint(method: str, checkpoint: Optional[str], config: Dict[str, Any],
                       model_key: str, tuning: Optional[str]) -> Optional[str]:
    if checkpoint:
        return checkpoint if os.path.exists(checkpoint) else None
    root = artifact_root(config, model_key, tuning)
    for name in (DEFAULT_CHECKPOINT_NAMES[method], os.path.join("forecaster", DEFAULT_CHECKPOINT_NAMES[method])):
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Model / cache loading
# ---------------------------------------------------------------------------
def load_cache_bundle(config: Dict[str, Any], cache_dir: str, device: str) -> Any:
    """Load the LM-free cache bundle (logits / representations / priors)."""
    from src.modeling.caches import cache_paths, load_caches

    paths = cache_paths(cache_dir)
    logger.info("loading caches from %s", cache_dir)
    cache = load_caches(cache_dir)
    missing = []
    if getattr(cache, "logits", None) is None:
        missing.append("logit")
    if getattr(cache, "representations", None) is None:
        missing.append("representation")
    if getattr(cache, "priors", None) is None:
        missing.append("prior")
    if missing:
        logger.warning("cache bundle is missing: %s (%s)", ", ".join(missing), paths)
    return cache


def to_device(module: Any, device: str) -> Any:
    if module is None:
        return module
    try:
        return module.to(device)
    except Exception:  # pragma: no cover - defensive
        return module


def load_encoder_for_online(
    model_key: str,
    config: Dict[str, Any],
    device: str,
    dtype: str,
    checkpoint: Optional[str] = None,
) -> Any:
    """Build ``h`` (LM backbone + 2-layer MLP) to encode online examples.

    The upstream side is already cached, so only the online example needs an
    encoder forward pass.  When a checkpoint saved by ``train_forecaster.py``
    is available the MLP weights are restored from it, otherwise the freshly
    initialized (untrained) ``h`` is used -- enough to smoke-test the pipeline.
    """
    from src.modeling.encoder_h import load_encoder_h

    encoder = load_encoder_h(model_key=model_key, config=config, device=device, dtype=dtype)
    if checkpoint and os.path.exists(checkpoint):
        try:
            payload = encoder.load_state(checkpoint)
            logger.info("loaded h weights from %s", checkpoint)
            return encoder
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not restore h from %s (%s); using fresh h", checkpoint, exc)
    return encoder


# ---------------------------------------------------------------------------
# Forecasting back ends
# ---------------------------------------------------------------------------
def _upstream_indices(cache: Any, upstream_examples: Sequence[Dict[str, Any]]) -> List[int]:
    if cache is not None:
        try:
            idx = cache.upstream_indices()
            if idx:
                return [int(i) for i in idx]
        except Exception:  # pragma: no cover - defensive
            pass
    return list(range(len(upstream_examples)))


def forecast_with_representation(
    forecaster: Any,
    cache: Any,
    encoder: Any,
    online_examples: Sequence[Dict[str, Any]],
    upstream_indices: Sequence[int],
    use_prior: bool = True,
    device: str = "cpu",
    encode_batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Eq. 4: ``z_hat_ij = 1[sigmoid(<h_j, h_i> + b_j) >= 0.5]``."""
    import torch  # local import: heavy

    h_up = None
    if cache is not None and getattr(cache, "representations", None) is not None:
        try:
            h_up = cache.stack_mean(upstream_indices)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not stack cached h_mean (%s)", exc)
    if h_up is None:
        raise RuntimeError(
            "representation forecaster requires the cached h(x_j, y_j) mean vectors; "
            "run scripts/build_datasets.py / src.modeling.caches first"
        )

    prior_vec = None
    if use_prior and cache is not None and getattr(cache, "priors", None) is not None:
        try:
            from src.forecasters.representation_based import prior_vector_from

            prior_vec = prior_vector_from(cache.priors, upstream_indices, device=device)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not build prior vector (%s)", exc)

    results: List[Dict[str, Any]] = []
    inputs = [str(ex.get("input", "")) for ex in online_examples]
    targets = [str(ex.get("target", "")) for ex in online_examples]
    with torch.no_grad():
        h_online = forecaster.encode(
            inputs, targets, mean_pool=True, batch_size=encode_batch_size
        )
        for row, h_i in enumerate(h_online):
            probs = forecaster.predict_from_encodings(
                h_up,
                h_i.unsqueeze(0) if hasattr(h_i, "unsqueeze") else h_i,
                prior=prior_vec,
                return_probabilities=True,
            )
            if hasattr(probs, "detach"):
                probs = probs.detach().cpu().reshape(-1).tolist()
            elif isinstance(probs, dict):
                probs = list(probs.get("probabilities", probs.get("probs", [])))
                probs = [float(p) for p in probs]
            results.append(
                {
                    "online_index": int(row),
                    "online_id": online_examples[row].get("id"),
                    "scores": [float(p) for p in probs],
                }
            )
    return results


def forecast_with_logit(
    forecaster: Any,
    cache: Any,
    encoder: Any,
    online_examples: Sequence[Dict[str, Any]],
    upstream_indices: Sequence[int],
    gt_dir: Optional[str] = None,
    device: str = "cpu",
    encode_batch_size: int = 8,
    topk: int = 100,
) -> List[Dict[str, Any]]:
    """Eq. 2: ``f_hat_i(x_j) = Theta_tilde [f_i(x_i) - f_0(x_i)] + f_0(x_j)``.

    ``Theta_tilde(x_j, x_i) = h(x_j, y_j) h(x_i, y_i)^T`` uses the *cached*
    token-level ``h(x_j, y_j)`` for the upstream side and a single encoder pass
    for the online example.  The observed online logit change
    ``f_i(x_i) - f_0(x_i)`` is read from the ground-truth pair records when
    available (``--gt-dir``); otherwise the script reports a clear error
    instead of silently fabricating a delta.
    """
    import torch  # local import: heavy

    from src.forecasters.logit_based import (
        build_candidate_indices,
        forecast_pair_from_cache,
        make_delta_matrix,
    )

    delta_lookup: Dict[int, Tuple[Any, Any, Any, Any]] = {}
    if gt_dir and os.path.isdir(gt_dir):
        pairs_file = os.path.join(gt_dir, "pairs.jsonl")
        for rec in load_jsonl(pairs_file):
            i = rec.get("i")
            if i is None or i in delta_lookup:
                continue
            delta_lookup[int(i)] = (
                rec.get("f0_i_token_logits"),
                rec.get("fi_i_token_logits"),
                rec.get("target_i_ids"),
                rec.get("i_id"),
            )
    if not delta_lookup:
        logger.warning(
            "no online logit-change records found (--gt-dir); the logit forecaster "
            "needs f_i(x_i) - f_0(x_i) and cannot run without them"
        )

    results: List[Dict[str, Any]] = []
    inputs = [str(ex.get("input", "")) for ex in online_examples]
    targets = [str(ex.get("target", "")) for ex in online_examples]
    h_token_cache: Dict[int, Any] = {}
    with torch.no_grad():
        h_online_all = encoder.encode(
            inputs, targets, mean_pool=False, batch_size=encode_batch_size
        )
        for row in range(len(online_examples)):
            h_i = h_online_all[row]
            record = delta_lookup.get(row)
            if record is None:
                results.append(
                    {
                        "online_index": int(row),
                        "online_id": online_examples[row].get("id"),
                        "scores": None,
                        "skipped": "missing online logit-change record",
                    }
                )
                continue
            f0_info, fi_info, target_i_ids, _ = record
            scores: List[float] = []
            for j in upstream_indices:
                if j not in h_token_cache:
                    try:
                        h_token_cache[j] = cache.h_token(j)
                    except Exception:
                        h_token_cache[j] = None
                h_j = h_token_cache[j]
                if h_j is None:
                    scores.append(0.0)
                    continue
                try:
                    z_hat, _, _ = forecast_pair_from_cache(
                        forecaster,
                        h_j,
                        h_i,
                        f0_info,
                        fi_info,
                        cache.f0_topk(j),
                        target_ids_i=target_i_ids or (),
                        target_ids_j=(),
                        return_logits=False,
                    )
                    scores.append(float(z_hat))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("pair (%d, %d) failed: %s", row, j, exc)
                    scores.append(0.0)
            results.append(
                {
                    "online_index": int(row),
                    "online_id": online_examples[row].get("id"),
                    "scores": scores,
                }
            )
    return results


def _threshold_scores(
    forecaster: Any,
    upstream_indices: Sequence[int],
    gt_dir: Optional[str],
    n_upstream: int,
) -> Dict[int, float]:
    """Obtain ``count_j`` scores for the threshold baseline.

    Priority: counts persisted on the trained forecaster object, then counts
    recomputed from the ground-truth *training* pairs (``D_R^Train``).
    """
    for attr in ("scores", "counts", "forget_counts", "frequencies"):
        values = getattr(forecaster, attr, None)
        if isinstance(values, dict) and values:
            out = {}
            for key, val in values.items():
                try:
                    out[int(key)] = float(val)
                except (TypeError, ValueError):
                    continue
            if out:
                return out
    if gt_dir and os.path.isdir(gt_dir):
        pairs = load_jsonl(os.path.join(gt_dir, "pairs.jsonl"))
        if pairs:
            from src.forecasters.threshold import forget_counts

            try:
                counts = forget_counts(pairs, n_upstream=n_upstream)
                return {int(k): float(v) for k, v in counts.items()}
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not compute forget counts (%s)", exc)
    logger.warning(
        "threshold forecaster has no persisted counts and no ground-truth pairs; "
        "predicting 'not forgotten' for all upstream examples"
    )
    return {int(j): 0.0 for j in upstream_indices}


def forecast_with_threshold(
    forecaster: Any,
    upstream_indices: Sequence[int],
    gt_dir: Optional[str],
    n_upstream: int,
    gamma: Optional[float] = None,
    use_frequency: bool = False,
) -> List[Dict[str, Any]]:
    """Eq. 1: ``z_hat = 1[#past forgettings of x_j >= gamma]``."""
    scores = _threshold_scores(forecaster, upstream_indices, gt_dir, n_upstream)
    if gamma is None:
        gamma = float(getattr(forecaster, "gamma", 1.0) or 1.0)
        if use_frequency or getattr(forecaster, "use_frequency", False):
            total = max(int(getattr(forecaster, "n_online", 0) or 0), 1)
            scores = {k: v / total for k, v in scores.items()}
            gamma = float(getattr(forecaster, "max_gamma", gamma)) if gamma > 1 else gamma
    z = [1 if scores.get(int(j), 0.0) >= gamma else 0 for j in upstream_indices]
    return [
        {
            "online_index": None,
            "online_id": None,
            "gamma": float(gamma),
            "scores": [scores.get(int(j), 0.0) for j in upstream_indices],
            "z_hat": z,
            "static": True,
        }
    ]


# ---------------------------------------------------------------------------
# Ground-truth comparison and summaries
# ---------------------------------------------------------------------------
def load_gt_labels(gt_dir: Optional[str]) -> Dict[Tuple[int, int], int]:
    """Load ``{(i, j) -> z_ij}`` from the ground-truth pair records."""
    labels: Dict[Tuple[int, int], int] = {}
    if not gt_dir:
        return labels
    for rel in ("pairs_test.jsonl", "pairs.jsonl"):
        path = os.path.join(gt_dir, rel)
        for rec in load_jsonl(path):
            try:
                labels[(int(rec["i"]), int(rec["j"]))] = int(rec["z"])
            except (KeyError, TypeError, ValueError):
                continue
    return labels


def evaluate_predictions(
    results: Sequence[Dict[str, Any]],
    upstream_indices: Sequence[int],
    labels: Dict[Tuple[int, int], int],
) -> Dict[str, Any]:
    """Precision / recall / F1 of the forecasts against ``z_ij`` labels."""
    tp = fp = fn = tn = 0
    matched = 0
    for res in results:
        scores = res.get("scores")
        if scores is None:
            continue
        z_hat = res.get("z_hat")
        if z_hat is None:
            threshold = res.get("decision_threshold", 0.5)
            z_hat = [1 if s >= threshold else 0 for s in scores]
        i = res.get("online_index")
        for pos, j in enumerate(upstream_indices):
            if i is None:
                continue
            key = (int(i), int(j))
            if key not in labels:
                continue
            matched += 1
            truth = labels[key]
            pred = int(z_hat[pos])
            if truth == 1 and pred == 1:
                tp += 1
            elif truth == 0 and pred == 1:
                fp += 1
            elif truth == 1 and pred == 0:
                fn += 1
            else:
                tn += 1

    def _div(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall)
    return {
        "matched_pairs": matched,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_percent": 100.0 * f1,
    }


def summarize_results(
    results: Sequence[Dict[str, Any]],
    upstream_indices: Sequence[int],
    method: str,
    labels: Optional[Dict[Tuple[int, int], int]] = None,
    n_upstream: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate forecasts into a JSON-serializable summary."""
    n_pos = 0
    n_considered = 0
    score_sum = 0.0
    score_n = 0
    per_online: List[Dict[str, Any]] = []
    for res in results:
        scores = res.get("scores")
        if scores is None:
            per_online.append({"online_index": res.get("online_index"), "skipped": True})
            continue
        z_hat = res.get("z_hat")
        if z_hat is None:
            z_hat = [1 if s >= res.get("decision_threshold", 0.5) else 0 for s in scores]
        pos = int(sum(int(v) for v in z_hat))
        n_pos += pos
        n_considered += len(z_hat)
        score_sum += float(sum(float(s) for s in scores))
        score_n += len(scores)
        per_online.append(
            {
                "online_index": res.get("online_index"),
                "online_id": res.get("online_id"),
                "n_forecast_forgotten": pos,
                "n_upstream": len(z_hat),
            }
        )
    summary: Dict[str, Any] = {
        "method": method,
        "n_online": len(results),
        "n_upstream": int(n_upstream if n_upstream is not None else len(upstream_indices)),
        "n_pairs": n_considered,
        "n_forecast_forgotten": n_pos,
        "forecast_prevalence": (float(n_pos) / float(n_considered)) if n_considered else 0.0,
        "mean_score": (score_sum / score_n) if score_n else 0.0,
        "per_online": per_online,
    }
    if labels:
        summary["metrics"] = evaluate_predictions(results, upstream_indices, labels)
        gt_pos = sum(1 for v in labels.values() if int(v) == 1)
        summary["gt_prevalence"] = (float(gt_pos) / float(len(labels))) if labels else 0.0
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast forgotten upstream examples (cache-only inference)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument(
        "--model-key",
        default="BART0_L",
        choices=["BART0_L", "FLAN-T5_L", "FLAN-T5_3B", "FLAN-T5_small"],
        help="base PTLM key",
    )
    parser.add_argument(
        "--tuning",
        default="head",
        choices=["head", "lora", "full_ft", "none"],
        help="tuning setup whose artifacts/checkpoints to use",
    )
    parser.add_argument("--method", default="representation", choices=list(METHODS))
    parser.add_argument("--checkpoint", default=None, help="explicit forecaster checkpoint")
    parser.add_argument("--cache-dir", default=None, help="cache bundle directory")
    parser.add_argument("--gt-dir", default=None, help="ground-truth directory (labels + deltas)")
    parser.add_argument("--upstream-file", default=None, help="override D_PT_hat JSONL path")
    parser.add_argument("--online-file", default=None, help="override D_R^Test JSONL path")
    parser.add_argument("--out-dir", default=None, help="output directory")
    parser.add_argument("--output-format", default="arrays", choices=["arrays", "pairs"])
    parser.add_argument("--max-online", type=int, default=50, help="0 = all online examples")
    parser.add_argument("--max-upstream", type=int, default=0, help="0 = all upstream examples")
    parser.add_argument("--batch-size", type=int, default=8, help="encoding batch size")
    parser.add_argument("--gamma", type=float, default=None, help="override threshold gamma")
    parser.add_argument("--use-frequency", action="store_true", help="threshold uses normalized frequency")
    parser.add_argument(
        "--no-prior", action="store_true", help="disable the frequency prior b_j (Table 1 ablation)"
    )
    parser.add_argument("--device", default=None, help="torch device (default from config)")
    parser.add_argument("--dtype", default=None, help="torch dtype (default from config)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default from config)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_config(args.config)
    device = args.device or cfg_get(config, "device", default="cuda")
    dtype = args.dtype or cfg_get(config, "dtype", default="float32")
    seed = args.seed if args.seed is not None else int(cfg_get(config, "seed", default=42) or 42)

    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU")
            device = "cpu"
        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch is a hard requirement in practice
        logger.error("PyTorch is required to run forecasting")
        return 2

    tuning = None if args.tuning == "none" else args.tuning
    paths = dataset_paths(config, args.model_key, tuning)
    out_dir = args.out_dir or os.path.join(artifact_root(config, args.model_key, tuning), "forecast", args.method)
    if args.no_prior and args.method == "representation":
        out_dir = out_dir + "_no_prior"
    os.makedirs(out_dir, exist_ok=True)

    upstream_examples, upstream_path = resolve_upstream_examples(
        config, args.model_key, tuning, args.upstream_file
    )
    online_examples, online_path = resolve_online_examples(
        config, args.model_key, tuning, args.online_file
    )
    logger.info("upstream pool: %s (%d examples)", upstream_path, len(upstream_examples))
    logger.info("online pool:   %s (%d examples)", online_path, len(online_examples))

    if args.max_online and args.max_online > 0:
        online_examples = online_examples[: args.max_online]
        logger.info("using the first %d online examples", len(online_examples))

    cache_dir = args.cache_dir or paths["cache_dir"]
    gt_dir = args.gt_dir or paths["gt_dir"]
    upstream_indices = list(range(len(upstream_examples)))
    cache = None
    if os.path.isdir(cache_dir):
        try:
            cache = load_cache_bundle(config, cache_dir, device)
            upstream_indices = _upstream_indices(cache, upstream_examples)
            if len(upstream_indices) != len(upstream_examples):
                logger.warning(
                    "cache covers %d of %d upstream examples; forecasting over the cached subset",
                    len(upstream_indices),
                    len(upstream_examples),
                )
        except Exception as exc:
            logger.warning("could not load cache bundle from %s (%s)", cache_dir, exc)
            if args.method in ("representation", "logit"):
                logger.error("method '%s' requires the cache bundle; aborting", args.method)
                return 2
    elif args.method in ("representation", "logit"):
        logger.error("cache directory %s not found; method '%s' requires caches", cache_dir, args.method)
        return 2

    if args.max_upstream and args.max_upstream > 0:
        upstream_indices = upstream_indices[: args.max_upstream]
        logger.info("restricting forecast to the first %d upstream examples", len(upstream_indices))

    checkpoint = resolve_checkpoint(args.method, args.checkpoint, config, args.model_key, tuning)
    if checkpoint is None:
        logger.warning("no %s checkpoint found; using an untrained forecaster", args.method)

    started = time.time()
    # ------------------------------------------------------------------
    # Dispatch to the requested forecasting back end
    # ------------------------------------------------------------------
    if args.method == "threshold":
        from src.forecasters.threshold import ThresholdForecaster

        if checkpoint:
            try:
                forecaster = ThresholdForecaster.load(checkpoint)
            except Exception as exc:
                logger.warning("could not load threshold checkpoint (%s)", exc)
                forecaster = ThresholdForecaster(gamma=args.gamma or 1.0)
        else:
            forecaster = ThresholdForecaster(gamma=args.gamma or 1.0)
        results = forecast_with_threshold(
            forecaster,
            upstream_indices,
            gt_dir,
            n_upstream=len(upstream_examples),
            gamma=args.gamma,
            use_frequency=args.use_frequency,
        )
    elif args.method == "representation":
        from src.forecasters.representation_based import RepresentationBasedForecaster

        dim = int(
            cfg_get(config, "encoder_h", "repr_dim", default=cfg_get(config, "encoder_h", "mlp_hidden", default=768))
            or 768
        )
        forecaster = RepresentationBasedForecaster(dim=dim, use_prior=not args.no_prior)
        if checkpoint:
            try:
                forecaster.load_state(checkpoint)
                logger.info("loaded representation forecaster from %s", checkpoint)
            except Exception as exc:
                logger.warning("could not load checkpoint %s (%s); using untrained weights", checkpoint, exc)
        if cache is not None and not args.no_prior and getattr(cache, "priors", None) is not None:
            try:
                forecaster.set_prior(cache.priors)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not attach prior cache (%s)", exc)
        encoder = load_encoder_for_online(args.model_key, config, device, dtype, checkpoint=None)
        forecaster.set_encoder(encoder)
        forecaster = to_device(forecaster, device)
        forecaster.eval() if hasattr(forecaster, "eval") else None
        results = forecast_with_representation(
            forecaster,
            cache,
            encoder,
            online_examples,
            upstream_indices,
            use_prior=not args.no_prior,
            device=device,
            encode_batch_size=args.batch_size,
        )
    else:  # logit
        from src.forecasters.logit_based import LogitChangeTransferForecaster

        dim = int(cfg_get(config, "encoder_h", "token_dim", default=768) or 768)
        forecaster = LogitChangeTransferForecaster(dim=dim)
        if checkpoint:
            try:
                forecaster.load_state(checkpoint)
                logger.info("loaded logit forecaster from %s", checkpoint)
            except Exception as exc:
                logger.warning("could not load checkpoint %s (%s); using untrained weights", checkpoint, exc)
        encoder = load_encoder_for_online(args.model_key, config, device, dtype)
        forecaster.set_encoder(encoder)
        forecaster = to_device(forecaster, device)
        results = forecast_with_logit(
            forecaster,
            cache,
            encoder,
            online_examples,
            upstream_indices,
            gt_dir=gt_dir,
            device=device,
            encode_batch_size=args.batch_size,
            topk=int(cfg_get(config, "forecaster", "cache_topk", default=100) or 100),
        )

    elapsed = time.time() - started

    # ------------------------------------------------------------------
    # Persist predictions + summary
    # ------------------------------------------------------------------
    predictions_path = os.path.join(out_dir, DEFAULT_PREDICTIONS_FILENAME)
    with open(predictions_path, "w", encoding="utf-8") as handle:
        if args.output_format == "pairs":
            for res in results:
                scores = res.get("scores")
                if scores is None:
                    continue
                for pos, j in enumerate(upstream_indices):
                    handle.write(
                        json.dumps(
                            {
                                "i": res.get("online_index"),
                                "j": int(j),
                                "j_id": upstream_examples[j].get("id") if j < len(upstream_examples) else None,
                                "z_hat": int(
                                    res.get("z_hat", [0] * len(scores))[pos]
                                    if res.get("z_hat") is not None
                                    else (1 if float(scores[pos]) >= res.get("decision_threshold", 0.5) else 0)
                                ),
                                "score": float(scores[pos]),
                                "online_id": res.get("online_id"),
                                "method": args.method,
                                "no_prior": bool(args.no_prior),
                            },
                            default=_json_default,
                        )
                        + "\n"
                    )
        else:
            for res in results:
                handle.write(json.dumps({**res, "method": args.method}, default=_json_default) + "\n")

    labels = load_gt_labels(gt_dir)
    if labels:
        logger.info("loaded %d ground-truth labels for comparison", len(labels))
    summary = summarize_results(
        results,
        upstream_indices,
        args.method,
        labels=labels or None,
        n_upstream=len(upstream_examples),
    )
    summary.update(
        {
            "model_key": args.model_key,
            "tuning": tuning,
            "no_prior": bool(args.no_prior),
            "gamma": args.gamma,
            "checkpoint": checkpoint,
            "cache_dir": cache_dir if os.path.isdir(cache_dir) else None,
            "upstream_path": upstream_path,
            "online_path": online_path,
            "n_online_evaluated": len(online_examples),
            "n_upstream_evaluated": len(upstream_indices),
            "elapsed_seconds": elapsed,
            "predictions_file": predictions_path,
        }
    )
    summary_path = write_json(os.path.join(out_dir, DEFAULT_SUMMARY_FILENAME), summary)

    logger.info(
        "method=%s | online=%d | upstream=%d | forecast prevalence=%.4f | %.1fs",
        args.method,
        len(online_examples),
        len(upstream_indices),
        summary["forecast_prevalence"],
        elapsed,
    )
    if "metrics" in summary:
        m = summary["metrics"]
        logger.info(
            "vs ground truth (matched=%d): precision=%.2f recall=%.2f F1=%.2f",
            m["matched_pairs"],
            100 * m["precision"],
            100 * m["recall"],
            100 * m["f1"],
        )
    logger.info("wrote %s", predictions_path)
    logger.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
