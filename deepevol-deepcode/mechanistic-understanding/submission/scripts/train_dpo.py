#!/usr/bin/env python
"""Train GPT2_DPO on PPLM-generated toxic/non-toxic preference pairs.

Reproduces the DPO phase of the paper:

    * Section 4.1 / Eq. 1 -- the DPO objective
        L_DPO = -E[ log sigma( beta * log P - beta * log N ) ]
        P = pi_theta(y+ | w) / pi_ref(y+ | w)
        N = pi_theta(y- | w) / pi_ref(y- | w)
      where ``y+`` is the preferred (non-toxic) and ``y-`` the non-preferred
      (toxic) continuation of the Wikitext-2 prompt ``w``.

    * Section 4.2 -- "We create 24,576 pairs of toxic and nontoxic
      continuations. We train until validation loss converges with a patience
      value of 10, which occurs after approximately 6,700 sample pairs."

    * Appendix E, Table 8 -- DPO hyperparameters
        learning rate            1e-6
        batch size               4
        optimizer                RMSProp
        gradient accumulation    1
        max gradient norm        10
        validation metric        loss/valid
        validation patience      10
        DPO beta                 0.1

The heavy lifting lives in :mod:`src.dpo_trainer`; this module is the CLI
orchestration layer: it resolves settings from ``configs/dpo.yaml`` + flags,
loads the preference pairs produced by ``scripts/generate_pairs.py``, trains
(optionally evaluating the resulting GPT2_DPO on toxicity / perplexity / F1),
and persists the checkpoint plus a JSON summary.

Usage
-----
    python scripts/train_dpo.py --config configs/dpo.yaml
    python scripts/train_dpo.py --pairs artifacts/data/pairs.jsonl --epochs 1 --quick
    python scripts/train_dpo.py --eval-only --evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Path handling: allow `python scripts/train_dpo.py` from the repo root.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_CONFIG = os.path.join("configs", "dpo.yaml")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort YAML config loader (returns ``{}`` when unavailable)."""
    if path is None:
        path = DEFAULT_CONFIG
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _dig(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup tolerant of missing levels."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _first(cfg: Dict[str, Any], *paths, default: Any = None) -> Any:
    """First non-``None`` value among dotted-ish lookup paths."""
    for path in paths:
        keys = path if isinstance(path, (tuple, list)) else (path,)
        value = _dig(cfg, *keys, default=None)
        if value is not None:
            return value
    return default


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args, the DPO YAML config, and paper defaults (Table 8)."""
    cfg = load_config(getattr(args, "config", None))

    # --- model / paths ---------------------------------------------------
    model_name = (
        getattr(args, "model", None)
        or _first(cfg, "model_name", ("model", "name"), ("model",), default=DEFAULT_MODEL)
    )
    output_dir = (
        getattr(args, "output_dir", None)
        or _first(cfg, "output_dir", ("training", "output_dir"), default=None)
    )
    pairs_path = (
        getattr(args, "pairs", None)
        or _first(cfg, "pairs_path", ("data", "pairs_path"), default=None)
    )
    shard_dir = getattr(args, "shard_dir", None) or _first(
        cfg, "shard_dir", ("data", "shard_dir"), default=None
    )

    # --- DPO hyperparameters (Appendix E, Table 8) -----------------------
    dpo_cfg = {}
    for section in ("dpo", "train", "training", "training_args"):
        node = cfg.get(section) if isinstance(cfg, dict) else None
        if isinstance(node, dict):
            dpo_cfg = dict(node)
            break
    # top-level hyperparameter keys are also accepted
    for key in (
        "beta",
        "learning_rate",
        "lr",
        "batch_size",
        "grad_accum",
        "max_grad_norm",
        "optimizer",
        "validation_metric",
        "patience",
        "num_epochs",
        "epochs",
        "max_length",
        "eval_steps",
        "logging_steps",
        "seed",
        "loss_type",
        "label_smoothing",
        "cache_reference_logps",
        "early_stopping",
    ):
        if isinstance(cfg, dict) and key in cfg and key not in dpo_cfg:
            dpo_cfg[key] = cfg[key]

    beta = getattr(args, "beta", None)
    lr = getattr(args, "lr", None)
    batch_size = getattr(args, "batch_size", None)
    grad_accum = getattr(args, "grad_accum", None)
    max_grad_norm = getattr(args, "max_grad_norm", None)
    optimizer = getattr(args, "optimizer", None)
    patience = getattr(args, "patience", None)
    epochs = getattr(args, "epochs", None)
    max_length = getattr(args, "max_length", None)
    seed = getattr(args, "seed", None)
    eval_steps = getattr(args, "eval_steps", None)
    logging_steps = getattr(args, "logging_steps", None)

    settings: Dict[str, Any] = {
        "model_name": model_name,
        "output_dir": output_dir,
        "pairs_path": pairs_path,
        "shard_dir": shard_dir,
        "beta": beta if beta is not None else dpo_cfg.get("beta", 0.1),
        "learning_rate": (
            lr if lr is not None else dpo_cfg.get("learning_rate", dpo_cfg.get("lr", 1e-6))
        ),
        "batch_size": (
            batch_size if batch_size is not None else dpo_cfg.get("batch_size", 4)
        ),
        "grad_accum": (
            grad_accum if grad_accum is not None else dpo_cfg.get("grad_accum", 1)
        ),
        "max_grad_norm": (
            max_grad_norm if max_grad_norm is not None else dpo_cfg.get("max_grad_norm", 10.0)
        ),
        "optimizer": (
            optimizer if optimizer is not None else dpo_cfg.get("optimizer", "rmsprop")
        ),
        "validation_metric": dpo_cfg.get("validation_metric", "loss/valid"),
        "patience": patience if patience is not None else dpo_cfg.get("patience", 10),
        "num_epochs": (
            epochs
            if epochs is not None
            else dpo_cfg.get("num_epochs", dpo_cfg.get("epochs", 1))
        ),
        "max_length": (
            max_length if max_length is not None else dpo_cfg.get("max_length", 128)
        ),
        "seed": seed if seed is not None else dpo_cfg.get("seed", 0),
        "eval_steps": (
            eval_steps if eval_steps is not None else dpo_cfg.get("eval_steps", 100)
        ),
        "logging_steps": (
            logging_steps
            if logging_steps is not None
            else dpo_cfg.get("logging_steps", 25)
        ),
        "loss_type": dpo_cfg.get("loss_type", "sigmoid"),
        "label_smoothing": dpo_cfg.get("label_smoothing", 0.0),
        "cache_reference_logps": dpo_cfg.get("cache_reference_logps", True),
        "early_stopping": dpo_cfg.get("early_stopping", True),
        "valid_ratio": _first(cfg, "valid_ratio", ("data", "valid_ratio"), default=0.1),
        "device": getattr(args, "device", None) or _first(cfg, "device", default=None),
        "quick": bool(getattr(args, "quick", False)),
        "max_pairs": getattr(args, "max_pairs", None)
        or _first(cfg, "max_pairs", ("data", "max_pairs"), default=None),
        "evaluate": bool(getattr(args, "evaluate", False))
        or bool(_first(cfg, "evaluate_after_train", default=False)),
        "n_prompts": getattr(args, "n_prompts", None)
        or _first(cfg, "n_prompts", ("eval", "n_prompts"), default=None),
    }

    # derive defaults that depend on the resolved model name
    if not settings["output_dir"]:
        settings["output_dir"] = os.path.join(
            "artifacts", "models", "gpt2_dpo" if "gpt2" in str(model_name).lower() else "dpo"
        )
    if not settings["pairs_path"]:
        try:
            from data.pairwise import DEFAULT_PAIRS_PATH  # type: ignore

            settings["pairs_path"] = DEFAULT_PAIRS_PATH
        except Exception:
            settings["pairs_path"] = os.path.join("artifacts", "data", "pairs.jsonl")

    # --- quick mode: shrink everything for smoke tests --------------------
    if settings["quick"]:
        settings["num_epochs"] = min(int(settings["num_epochs"]), 1)
        settings["max_length"] = min(int(settings["max_length"]), 64)
        settings["eval_steps"] = min(int(settings["eval_steps"]), 20)
        settings["logging_steps"] = min(int(settings["logging_steps"]), 5)
        if not settings["max_pairs"]:
            settings["max_pairs"] = 64

    return settings


# ---------------------------------------------------------------------------
# Pair loading
# ---------------------------------------------------------------------------
def resolve_pairs(
    settings: Dict[str, Any],
    verbose: bool = True,
):
    """Load the preference pairs written by ``scripts/generate_pairs.py``.

    Returns ``(train_pairs, valid_pairs)`` where each element is a
    ``data.pairwise.PairSplit``-like container (or ``None`` when nothing was
    found).  The 90:10 split is applied here when only a flat pair list exists.
    """
    from data import pairwise as pw  # local import: keeps --help fast

    pairs_path = settings.get("pairs_path")
    shard_dir = settings.get("shard_dir")
    valid_ratio = float(settings.get("valid_ratio", 0.1) or 0.1)
    seed = int(settings.get("seed", 0) or 0)

    raw: List[Any] = []
    source = None

    if pairs_path and os.path.exists(pairs_path):
        raw = list(pw.load_pairs(pairs_path))
        source = pairs_path
    elif shard_dir and os.path.isdir(shard_dir) and pw.existing_shards(shard_dir):
        raw = list(pw.load_shards(shard_dir))
        source = shard_dir
        if verbose:
            print(f"[pairs] loaded {len(raw)} pairs from shards in {shard_dir}")

    if not raw:
        if verbose:
            print(
                "[pairs] no preference pairs found at "
                f"{pairs_path!r} / {shard_dir!r}.\n"
                "        Run `python scripts/generate_pairs.py` first "
                "(PPLM toxic + greedy non-toxic continuations)."
            )
        return None, None, None

    dataset = pw.build_dataset(
        raw,
        valid_ratio=valid_ratio,
        seed=seed,
        max_pairs=settings.get("max_pairs"),
    )
    if verbose:
        stats = dataset.stats() if hasattr(dataset, "stats") else {}
        print(
            f"[pairs] {len(dataset.train)} train / {len(dataset.valid)} valid "
            f"(source: {source})"
        )
        if isinstance(stats, dict) and stats:
            shown = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:6])
            print(f"[pairs] stats: {shown}")
    return dataset.train, dataset.valid, dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def load_model_safe(name_or_path: str, device: Optional[str] = None):
    from src.model_utils import load_model  # type: ignore

    return load_model(name_or_path, device=device)


def run_train(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    verbose = not bool(getattr(args, "quiet", False))

    print("=" * 78)
    print("DPO training (paper Section 4.1 / 4.2, Appendix E Table 8)")
    print("=" * 78)
    print(f"  base model        : {settings['model_name']}")
    print(f"  output dir        : {settings['output_dir']}")
    print(f"  pairs             : {settings['pairs_path']}")
    print(
        "  hyperparameters   : beta={beta} lr={learning_rate} batch={batch_size} "
        "grad_accum={grad_accum} max_grad_norm={max_grad_norm}".format(**settings)
    )
    print(
        "                      optimizer={optimizer} metric={validation_metric} "
        "patience={patience} epochs={num_epochs}".format(**settings)
    )
    print("=" * 78)

    from src import dpo_trainer as dt  # type: ignore

    train_pairs, valid_pairs, dataset = resolve_pairs(settings, verbose=verbose)
    if train_pairs is None:
        print("[error] no training pairs available -- aborting.")
        return 2

    # --- build the DPOConfig (settings already match Table 8) ------------
    config = dt.DPOConfig(
        beta=float(settings["beta"]),
        learning_rate=float(settings["learning_rate"]),
        batch_size=int(settings["batch_size"]),
        grad_accum=int(settings["grad_accum"]),
        max_grad_norm=float(settings["max_grad_norm"]),
        optimizer=str(settings["optimizer"]),
        validation_metric=str(settings["validation_metric"]),
        patience=int(settings["patience"]),
        num_epochs=int(settings["num_epochs"]),
        max_length=int(settings["max_length"]),
        eval_steps=int(settings["eval_steps"]),
        logging_steps=int(settings["logging_steps"]),
        seed=int(settings["seed"]),
        device=settings.get("device"),
        early_stopping=bool(settings["early_stopping"]),
        cache_reference_logps=bool(settings["cache_reference_logps"]),
        loss_type=str(settings["loss_type"]),
        label_smoothing=float(settings["label_smoothing"] or 0.0),
        output_dir=settings["output_dir"],
        model_name=settings["model_name"],
    )

    # --- data plumbing ---------------------------------------------------
    if dataset is not None:
        # `train_dpo` accepts containers exposing `.to_hf()` / train / valid.
        pairs_container = dataset
    else:
        pairs_container = list(train_pairs) + list(valid_pairs)

    t0 = time.time()
    try:
        trainer, result = dt.train_dpo(
            pairs=pairs_container,
            model=None,
            tokenizer=None,
            ref_model=None,
            config=config,
            model_name=settings["model_name"],
            output_dir=settings["output_dir"],
            valid_ratio=float(settings.get("valid_ratio", 0.1) or 0.1),
            seed=int(settings["seed"]),
            device=settings.get("device"),
            verbose=verbose,
        )
    except KeyboardInterrupt:
        print("\n[interrupted] DPO training stopped by user.")
        return 130
    except Exception:  # pragma: no cover - diagnostics for long runs
        traceback.print_exc()
        print("[error] DPO training failed.")
        return 1
    elapsed = time.time() - t0

    summary = _report_result(result, elapsed=elapsed, settings=settings)

    # --- persistence ------------------------------------------------------
    out_dir = settings["output_dir"]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    summary_path = os.path.join(out_dir, "train_dpo_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)
        print(f"[save] summary -> {summary_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not write summary: {exc}")

    # also persist the standard trainer-state style result artefact
    try:
        dt.save_result(os.path.join(out_dir, "dpo_train_result.json"), result)
        print(f"[save] trainer state -> {os.path.join(out_dir, 'dpo_train_result.json')}")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not save DPOTrainResult: {exc}")

    if settings["evaluate"]:
        _evaluate_checkpoint(settings, verbose=verbose)

    return 0


def _report_result(result: Any, elapsed: float, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Print a Table-8-style training summary and return a JSON dict."""
    print("-" * 78)
    print("DPO training summary")
    print("-" * 78)

    def _get(name, default=None):
        return getattr(result, name, default)

    n_train = _get("n_train_pairs", 0) or 0
    n_valid = _get("n_valid_pairs", 0) or 0
    steps = _get("steps", 0)
    epochs = _get("epochs_run", settings.get("num_epochs", 1))
    seen = _get("train_examples_seen", 0) or 0
    best_loss = _get("best_valid_loss", None)
    best_step = _get("best_step", None)
    final_train = _get("final_train_loss", None)
    final_valid = _get("final_valid_loss", None)
    final_acc = _get("final_valid_accuracy", None)

    print(f"  train pairs            : {n_train}")
    print(f"  valid pairs            : {n_valid}")
    print(f"  optimizer steps        : {steps}")
    print(f"  epochs run             : {epochs}")
    print(f"  example pairs seen     : {seen}")
    if best_loss is not None:
        print(f"  best loss/valid        : {float(best_loss):.6f} (step {best_step})")
    if final_train is not None:
        print(f"  final train loss       : {float(final_train):.6f}")
    if final_valid is not None:
        print(f"  final valid loss       : {float(final_valid):.6f}")
    if final_acc is not None:
        print(f"  final valid accuracy   : {float(final_acc):.4f}")
    print(f"  wall clock             : {elapsed:.1f}s")
    print(f"  checkpoint             : {_get('output_dir', settings['output_dir'])}")

    # Paper reference: convergence "occurs after approximately 6,700 sample
    # pairs" (Section 4.2); flag how close we are.
    convergence_pairs = None
    history = _get("history", None) or []
    try:
        loss_keys = ("loss/valid", "valid_loss", "eval_loss")
        best = None
        for row in history:
            if not isinstance(row, dict):
                continue
            value = None
            for key in loss_keys:
                if row.get(key) is not None:
                    value = float(row[key])
                    break
            if value is None:
                continue
            if best is None or value < best[0]:
                best = (value, row)
        if best is not None:
            row = best[1]
            for key in ("examples_seen", "seen", "n_examples", "step"):
                if row.get(key) is not None:
                    value = row[key]
                    convergence_pairs = int(value) if key != "step" else int(value)
                    break
    except Exception:
        convergence_pairs = None

    print("-" * 78)
    if settings.get("evaluate"):
        print(
            "  expected GPT2_DPO metrics (paper): toxicity 0.208 | "
            "perplexity 23.34 | F1 0.195"
        )
        print("-" * 78)

    summary = {
        "model_name": _get("model_name", settings["model_name"]),
        "output_dir": _get("output_dir", settings["output_dir"]),
        "n_train_pairs": n_train,
        "n_valid_pairs": n_valid,
        "steps": steps,
        "epochs_run": epochs,
        "train_examples_seen": seen,
        "best_valid_loss": _safe_float(best_loss),
        "best_step": best_step,
        "final_train_loss": _safe_float(final_train),
        "final_valid_loss": _safe_float(final_valid),
        "final_valid_accuracy": _safe_float(final_acc),
        "convergence_examples": convergence_pairs,
        "elapsed_seconds": round(elapsed, 2),
        "config": _safe_config(settings),
        "paper_reference": {
            "n_pairs": 24576,
            "convergence_examples": 6700,
            "beta": 0.1,
            "learning_rate": 1e-6,
            "batch_size": 4,
            "optimizer": "rmsprop",
            "grad_accum": 1,
            "max_grad_norm": 10,
            "validation_metric": "loss/valid",
            "patience": 10,
            "gpt2_dpo_toxicity": 0.208,
            "gpt2_dpo_perplexity": 23.34,
            "gpt2_dpo_f1": 0.195,
        },
    }

    # Merge any extra bookkeeping the trainer recorded.
    meta = _get("meta", None)
    if isinstance(meta, dict):
        summary["meta"] = {k: _json_default(v) for k, v in meta.items()}

    return summary


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_default(v) for k, v in settings.items()}


# ---------------------------------------------------------------------------
# Optional post-training evaluation
# ---------------------------------------------------------------------------
def _evaluate_checkpoint(settings: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Score GPT2_DPO on toxicity / perplexity / F1 (Section 3.3 metrics)."""
    from src import dpo_trainer as dt  # type: ignore

    results: Dict[str, Any] = {}
    try:
        model, tokenizer = dt.load_dpo_model(settings["output_dir"], device=settings.get("device"))
    except Exception as exc:
        print(f"[warn] could not load GPT2_DPO for evaluation: {exc}")
        return results

    n_prompts = settings.get("n_prompts")
    if not n_prompts:
        try:
            from data.realtoxicity import N_CHALLENGE_PROMPTS  # type: ignore

            n_prompts = N_CHALLENGE_PROMPTS
        except Exception:
            n_prompts = 1199
    if settings.get("quick"):
        n_prompts = min(int(n_prompts), 32)

    out_dir = os.path.join("artifacts", "eval")
    os.makedirs(out_dir, exist_ok=True)

    # toxicity -----------------------------------------------------------
    try:
        from src.eval.toxicity import evaluate_toxicity, save_result  # type: ignore

        tox = evaluate_toxicity(
            model,
            tokenizer,
            model_name="gpt2_dpo",
            n_prompts=int(n_prompts),
            device=settings.get("device"),
            seed=int(settings.get("seed", 0) or 0),
            verbose=verbose,
        )
        results["toxicity"] = tox.mean_toxicity
        print(f"[eval] GPT2_DPO toxicity    : {tox.mean_toxicity:.3f} (paper 0.208)")
        save_result(os.path.join(out_dir, "toxicity_gpt2_dpo.json"), tox)
        generations = list(getattr(tox, "generations", []) or [])
        prompts = list(getattr(tox, "prompts", []) or [])
    except Exception as exc:
        print(f"[warn] toxicity evaluation skipped: {exc}")
        generations, prompts = [], []

    # perplexity ---------------------------------------------------------
    try:
        from src.eval.perplexity import evaluate_perplexity, save_result as save_ppl  # type: ignore

        ppl = evaluate_perplexity(
            model,
            tokenizer,
            model_name="gpt2_dpo",
            max_windows=32 if settings.get("quick") else None,
            verbose=verbose,
        )
        results["perplexity"] = ppl.perplexity
        print(f"[eval] GPT2_DPO perplexity  : {ppl.perplexity:.2f} (paper 23.34)")
        save_ppl(os.path.join(out_dir, "perplexity_gpt2_dpo.json"), ppl)
    except Exception as exc:
        print(f"[warn] perplexity evaluation skipped: {exc}")

    # F1 -----------------------------------------------------------------
    try:
        from src.eval.f1 import evaluate_f1, save_result as save_f1  # type: ignore

        f1_res = evaluate_f1(
            model,
            tokenizer,
            n=200 if settings.get("quick") else 2000,
            model_name="gpt2_dpo",
            device=settings.get("device"),
            seed=int(settings.get("seed", 0) or 0),
            verbose=verbose,
        )
        results["f1"] = f1_res.mean_f1
        print(f"[eval] GPT2_DPO F1          : {f1_res.mean_f1:.3f} (paper 0.195)")
        save_f1(os.path.join(out_dir, "f1_gpt2_dpo.json"), f1_res)
    except Exception as exc:
        print(f"[warn] F1 evaluation skipped: {exc}")

    results["n_prompts"] = int(n_prompts)
    results["examples"] = {
        "prompts": prompts[:3],
        "generations": generations[:3],
    }
    path = os.path.join(out_dir, "gpt2_dpo_summary.json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=_json_default)
        print(f"[save] evaluation summary -> {path}")
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _json_default(obj):
    """JSON fallback for numpy / torch / path-like scalars."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _print_plan(settings: Dict[str, Any]) -> None:
    """Show what a run would do without loading any model (``--dry-run``)."""
    print("[dry-run] resolved DPO settings:")
    for key in (
        "model_name",
        "output_dir",
        "pairs_path",
        "shard_dir",
        "beta",
        "learning_rate",
        "batch_size",
        "grad_accum",
        "max_grad_norm",
        "optimizer",
        "validation_metric",
        "patience",
        "num_epochs",
        "max_length",
        "seed",
        "valid_ratio",
        "max_pairs",
        "evaluate",
    ):
        print(f"  {key:<20}: {settings.get(key)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train GPT2_DPO with the DPO objective (Eq. 1) on PPLM-generated "
            "toxic/non-toxic preference pairs (Section 4.1/4.2, Table 8)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--model", type=str, default=None, help="Base model name or path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--pairs", type=str, default=None, help="Preference pairs JSONL path.")
    parser.add_argument("--shard-dir", type=str, default=None, help="Sharded pairs directory.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap training pairs (smoke).")
    parser.add_argument("--valid-ratio", type=float, default=None, help="Validation fraction.")

    # Table 8 hyperparameters
    parser.add_argument("--beta", type=float, default=None, help="DPO beta (Table 8: 0.1).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (Table 8: 1e-6).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (Table 8: 4).")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation (1).")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max grad norm (10).")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer (rmsprop).")
    parser.add_argument("--patience", type=int, default=None, help="Validation patience (10).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Validation frequency.")
    parser.add_argument("--logging-steps", type=int, default=None, help="Logging frequency.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. cuda).")

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After training, score GPT2_DPO on toxicity/perplexity/F1.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate.")
    parser.add_argument("--n-prompts", type=int, default=None, help="RTP prompts for evaluation.")
    parser.add_argument("--quick", action="store_true", help="Smoke-test settings.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and exit.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = resolve_settings(args)

    if args.dry_run:
        _print_plan(settings)
        return 0

    if args.eval_only:
        if not os.path.isdir(settings["output_dir"]):
            print(f"[error] no checkpoint at {settings['output_dir']}")
            return 2
        print(f"[eval-only] scoring {settings['output_dir']}")
        _evaluate_checkpoint(settings, verbose=not args.quiet)
        return 0

    try:
        return run_train(args)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
