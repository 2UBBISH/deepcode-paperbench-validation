#!/usr/bin/env python
"""Train the toxicity probe W_Toxic (paper Sec. 3.1).

Pipeline
--------
1. Load GPT2 (frozen) and the Jigsaw toxic-comment dataset (90:10 split).
2. Collect the residual stream of the last layer, averaged across all
   timesteps  x_bar^(L-1)  for every comment.
3. Fit the linear softmax probe  P(Toxic | x_bar) = softmax(W_Toxic x_bar)
   with ``W_Toxic`` of shape ``[d_model, 2]`` (column 1 = toxic direction).
4. Report validation accuracy (paper target: 94%) and save the probe to
   ``artifacts/probe/w_toxic.pt`` (+ ``.json`` sidecar).

Usage
-----
    python scripts/train_probe.py                       # full run
    python scripts/train_probe.py --limit 20000 --quick # smoke test
    python scripts/train_probe.py --eval-only           # load + evaluate

Notes
-----
The paper does not specify optimizer details for the probe; we use AdamW with
lr 1e-3, batch 256, up to 20 epochs and early stopping (documented default).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

# Make the project root importable when executed as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.jigsaw import JIGSAW_HF_NAME, N_TOTAL_COMMENTS, load_jigsaw  # noqa: E402
from src.probe import (  # noqa: E402
    PROBE_JSON_PATH,
    PROBE_PATH,
    TARGET_VALID_ACCURACY,
    ToxicityProbe,
    collect_probe_features,
    evaluate_probe,
    load_probe,
    probe_exists,
    save_probe,
    train_probe,
)

DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _dig(cfg: Dict[str, Any], *keys, default=None):
    """Nested lookup tolerant of missing keys."""
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_yaml(args.config)
    probe_cfg = _dig(cfg, "probe", default={}) or {}
    model_cfg = _dig(cfg, "model", default={}) or {}
    data_cfg = _dig(cfg, "data", default={}) or {}
    out_cfg = _dig(cfg, "artifacts", default={}) or {}

    settings = {
        "model_name": args.model
        or probe_cfg.get("model_name")
        or model_cfg.get("name")
        or DEFAULT_MODEL,
        "output": args.output
        or probe_cfg.get("output")
        or out_cfg.get("probe_path")
        or PROBE_PATH,
        "json_output": args.json_output
        or probe_cfg.get("json_output")
        or PROBE_JSON_PATH,
        "limit": args.limit if args.limit is not None else probe_cfg.get("limit"),
        "max_length": args.max_length
        if args.max_length is not None
        else probe_cfg.get("max_length", 128),
        "lr": args.lr if args.lr is not None else probe_cfg.get("lr", 1e-3),
        "batch_size": args.batch_size
        if args.batch_size is not None
        else probe_cfg.get("batch_size", 256),
        "feature_batch_size": args.feature_batch_size
        if args.feature_batch_size is not None
        else probe_cfg.get("feature_batch_size", 8),
        "epochs": args.epochs if args.epochs is not None else probe_cfg.get("epochs", 20),
        "patience": args.patience
        if args.patience is not None
        else probe_cfg.get("patience", 3),
        "weight_decay": args.weight_decay
        if args.weight_decay is not None
        else probe_cfg.get("weight_decay", 0.01),
        "class_weight": args.class_weight
        if args.class_weight is not None
        else probe_cfg.get("class_weight"),
        "valid_ratio": args.valid_ratio
        if args.valid_ratio is not None
        else probe_cfg.get("valid_ratio", data_cfg.get("valid_ratio", 0.1)),
        "cache_dir": args.cache_dir,
        "layer": probe_cfg.get("layer", -1),
        "position": probe_cfg.get("position", "block_out"),
        "pooling": probe_cfg.get("pooling", "mean"),
        "seed": args.seed if args.seed is not None else cfg.get("seed", 0),
        "device": args.device or cfg.get("device"),
    }
    if args.quick:
        settings["limit"] = settings["limit"] or 2000
        settings["epochs"] = min(int(settings["epochs"]), 2)
        settings["max_valid"] = 500
    return settings


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def load_gpt2(model_name: str, device: Optional[str] = None):
    from src.model_utils import load_model

    return load_model(model_name, device=device)


def collect_jigsaw_features(model, tokenizer, settings: Dict[str, Any], verbose: bool = True):
    """Load Jigsaw, split 90:10 and extract mean-pooled last-layer features."""
    data = load_jigsaw(
        cache_dir=settings.get("cache_dir"),
        valid_ratio=float(settings.get("valid_ratio", 0.1)),
        seed=int(settings.get("seed", 0)),
        limit=settings.get("limit"),
    )
    if verbose:
        print(
            f"[data] Jigsaw: {len(data.train)} train / {len(data.valid)} valid "
            f"(train toxic fraction {data.train.toxic_fraction:.3f})"
        )

    max_valid = settings.get("max_valid")
    valid_texts = data.valid.texts
    valid_labels = data.valid.labels
    if max_valid and len(valid_texts) > max_valid:
        sub = data.valid.subset(int(max_valid), seed=int(settings.get("seed", 0)))
        valid_texts, valid_labels = sub.texts, sub.labels
        if verbose:
            print(f"[data] validation downsampled to {len(valid_texts)} comments")

    t0 = time.time()
    train_feats = collect_probe_features(
        model,
        tokenizer,
        data.train.texts,
        layer=int(settings.get("layer", -1)),
        position=settings.get("position", "block_out"),
        pooling=settings.get("pooling", "mean"),
        batch_size=int(settings.get("feature_batch_size", 8)),
        max_length=int(settings.get("max_length", 128)),
        device=settings.get("device"),
        verbose=verbose,
    )
    valid_feats = collect_probe_features(
        model,
        tokenizer,
        valid_texts,
        layer=int(settings.get("layer", -1)),
        position=settings.get("position", "block_out"),
        pooling=settings.get("pooling", "mean"),
        batch_size=int(settings.get("feature_batch_size", 8)),
        max_length=int(settings.get("max_length", 128)),
        device=settings.get("device"),
        verbose=verbose,
    )
    if verbose:
        print(
            f"[features] train {train_feats.shape} valid {valid_feats.shape} "
            f"in {time.time() - t0:.1f}s"
        )
    return train_feats, data.train.labels, valid_feats, valid_labels, data


def run_train(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    print("[probe] settings:")
    for key in sorted(settings):
        print(f"    {key} = {settings[key]}")

    if probe_exists(settings["output"]) and not args.force:
        print(f"[probe] existing probe found at {settings['output']} (use --force to retrain)")

    model, tokenizer = load_gpt2(settings["model_name"], device=settings["device"])

    if args.eval_only:
        probe = load_probe(settings["output"])
        data = load_jigsaw(
            cache_dir=settings.get("cache_dir"),
            valid_ratio=float(settings.get("valid_ratio", 0.1)),
            seed=int(settings.get("seed", 0)),
            limit=settings.get("limit"),
        )
        feats = collect_probe_features(
            model,
            tokenizer,
            data.valid.texts,
            batch_size=int(settings.get("feature_batch_size", 8)),
            max_length=int(settings.get("max_length", 128)),
            device=settings.get("device"),
        )
        metrics = evaluate_probe(probe, feats, data.valid.labels)
        print("[probe] evaluation:", json.dumps(metrics, indent=2))
        return 0

    feats, labels, valid_feats, valid_labels, _ = collect_jigsaw_features(
        model, tokenizer, settings
    )

    result = train_probe(
        feats,
        labels,
        valid_features=valid_feats,
        valid_labels=valid_labels,
        model_name=settings["model_name"],
        layer=int(settings.get("layer", -1)),
        position=settings.get("position", "block_out"),
        pooling=settings.get("pooling", "mean"),
        lr=float(settings.get("lr", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
        batch_size=int(settings.get("batch_size", 256)),
        epochs=int(settings.get("epochs", 20)),
        patience=int(settings.get("patience", 3)),
        class_weight=settings.get("class_weight"),
        seed=int(settings.get("seed", 0)),
        device=settings.get("device"),
        verbose=True,
        meta={"source": "jigsaw", "hf_name": JIGSAW_HF_NAME, "n_total": N_TOTAL_COMMENTS},
    )

    print("[probe] summary:", json.dumps(result.summary(), indent=2, default=str))
    acc = float(result.valid_accuracy)
    if acc >= TARGET_VALID_ACCURACY:
        print(f"[probe] validation accuracy {acc:.4f} >= target {TARGET_VALID_ACCURACY:.2f} OK")
    else:
        print(
            f"[probe] WARNING validation accuracy {acc:.4f} below target "
            f"{TARGET_VALID_ACCURACY:.2f} (paper reports 94%)"
        )

    path = save_probe(settings["output"], result, verbose=True)
    print(f"[probe] saved probe to {path}")
    if settings.get("json_output"):
        try:
            payload = result.to_dict()
            payload["model_name"] = settings["model_name"]
            os.makedirs(os.path.dirname(os.path.abspath(settings["json_output"])), exist_ok=True)
            with open(settings["json_output"], "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            print(f"[probe] saved JSON summary to {settings['json_output']}")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[probe] could not write JSON summary: {exc}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the W_Toxic toxicity probe on Jigsaw (paper Sec. 3.1)."
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    p.add_argument("--model", default=None, help="model name or path (default GPT2-medium)")
    p.add_argument("--output", default=None, help="probe artifact path (.pt)")
    p.add_argument("--json-output", dest="json_output", default=None, help="JSON summary path")
    p.add_argument("--limit", type=int, default=None, help="cap number of Jigsaw comments")
    p.add_argument("--max-length", dest="max_length", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--feature-batch-size", dest="feature_batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    p.add_argument(
        "--class-weight",
        dest="class_weight",
        default=None,
        help="'balanced', a float (toxic class weight) or omitted",
    )
    p.add_argument("--valid-ratio", dest="valid_ratio", type=float, default=None)
    p.add_argument("--cache-dir", dest="cache_dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="retrain even if a probe exists")
    p.add_argument("--eval-only", dest="eval_only", action="store_true")
    p.add_argument("--quick", action="store_true", help="small-data smoke test")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.class_weight is not None:
        cw = str(args.class_weight)
        if cw.lower() == "balanced":
            args.class_weight = "balanced"
        else:
            try:
                args.class_weight = float(cw)
            except ValueError:
                args.class_weight = cw
    try:
        return run_train(args)
    except Exception:  # pragma: no cover - CLI diagnostics
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
