#!/usr/bin/env python
"""Evaluate a (possibly DPO-aligned) language model on the paper's three metrics.

Reproduces the evaluation protocol of Section 3.3 of
"A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity":

* **Toxicity** -- prompt the model with the ``challenge`` subset of
  RealToxicityPrompts (1,199 prompts that elicit extremely toxic outputs) and
  score each generation. The paper uses Perspective API; this reproduction
  substitutes the openly available ``unitary/unbiased-toxic-roberta`` classifier
  (documented substitution) and reports the mean toxicity over the 1,199 prompts.
* **Perplexity** -- Wikitext-2 (Merity et al., 2016) perplexity, used to verify
  that interventions/alignment do not degrade generation quality.
* **F1** -- using 2,000 Wikipedia sentences as prompts, the harmonic mean of
  precision (fraction of generated tokens contained in the original Wikipedia
  continuation) and recall (fraction of continuation tokens contained in the
  generation).

All three metrics are computed on the *same* greedy 20-token generations so the
Table 2 / Table 4, R, and D rows are directly comparable.

Usage
-----
    python scripts/eval_model.py --model openai-community/gpt2-medium --label gpt2
    python scripts/eval_model.py --model artifacts/models/gpt2_dpo --label gpt2_dpo
    python scripts/eval_model.py --all --quick          # smoke test, few prompts

Artifacts are written to ``artifacts/eval`` (overridable with ``--out-dir``):
``toxicity_<label>.json``, ``perplexity_<label>.json``, ``f1_<label>.json`` and a
combined ``eval_<label>_summary.json`` plus comparison plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Project root on sys.path so `python scripts/eval_model.py` works directly.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Paper / reproduction constants (documented defaults are in configs/default.yaml)
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")
DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_DPO_DIR = os.path.join("artifacts", "models", "gpt2_dpo")
DEFAULT_OUT_DIR = os.path.join("artifacts", "eval")
DEFAULT_PROBE_PATH = os.path.join("artifacts", "probe", "w_toxic.pt")
DEFAULT_VECTORS_PATH = os.path.join("artifacts", "vectors", "toxic_vectors.pt")

N_CHALLENGE_PROMPTS = 1199           # Section 3.3 "challenge" subset size
N_F1_SENTENCES = 2000                # Section 3.3 F1 prompt count
DEFAULT_MAX_NEW_TOKENS = 20          # greedy continuation length
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 0

# Reference values from the paper (Tables 2 and 4).
REFERENCE: Dict[str, Dict[str, float]] = {
    "gpt2": {"toxicity": 0.453, "perplexity": 21.70, "f1": 0.193},
    "gpt2_dpo": {"toxicity": 0.208, "perplexity": 23.34, "f1": 0.195},
    "subtract_w_toxic": {"toxicity": 0.245, "perplexity": 23.56, "f1": 0.193},
    "subtract_mlp_v_770_19": {"toxicity": 0.305, "perplexity": 23.30, "f1": 0.192},
    "subtract_svd_u_toxic_0": {"toxicity": 0.268, "perplexity": 23.48, "f1": 0.193},
    "unalign_scale_k_toxic": {"toxicity": 0.458, "perplexity": 23.30, "f1": 0.195},
}

METRICS = ("toxicity", "perplexity", "f1")


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort YAML config loader (returns ``{}`` when unavailable)."""
    path = path or DEFAULT_CONFIG
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:  # pragma: no cover - config is optional
        return {}


def _dig(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup tolerant of missing intermediate keys."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args > YAML config > paper defaults into a flat settings dict."""
    cfg = load_config(getattr(args, "config", None))

    eval_cfg = _dig(cfg, "evaluation", default={}) or {}
    in_scope = _dig(cfg, "in_scope", default={}) or {}

    seed = getattr(args, "seed", None)
    if seed is None:
        seed = int(_dig(cfg, "seed", default=DEFAULT_SEED))

    max_new_tokens = getattr(args, "max_new_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = int(eval_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))

    batch_size = getattr(args, "batch_size", None)
    if batch_size is None:
        batch_size = int(eval_cfg.get("batch_size", DEFAULT_BATCH_SIZE))

    n_prompts = getattr(args, "n_prompts", None)
    if n_prompts is None:
        n_prompts = int(eval_cfg.get("n_challenge_prompts", N_CHALLENGE_PROMPTS))

    n_f1 = getattr(args, "n_f1", None)
    if n_f1 is None:
        n_f1 = int(eval_cfg.get("n_f1_sentences", N_F1_SENTENCES))

    out_dir = getattr(args, "out_dir", None) or _dig(
        cfg, "paths", "eval_dir", default=None
    ) or _dig(cfg, "eval_dir", default=DEFAULT_OUT_DIR)

    settings: Dict[str, Any] = {
        "model": getattr(args, "model", None) or DEFAULT_MODEL,
        "dpo_model": getattr(args, "dpo_model", None) or DEFAULT_DPO_DIR,
        "label": getattr(args, "label", None) or "",
        "out_dir": out_dir,
        "figure_dir": _dig(cfg, "paths", "figure_dir", default=out_dir),
        "cache_dir": getattr(args, "cache_dir", None)
        or _dig(cfg, "paths", "cache_dir", default=None),
        "seed": int(seed),
        "device": getattr(args, "device", None) or _dig(cfg, "device", default=None),
        "n_prompts": int(n_prompts),
        "n_f1": int(n_f1),
        "max_new_tokens": int(max_new_tokens),
        "batch_size": int(batch_size),
        "max_prompt_length": int(eval_cfg.get("max_prompt_length", 96)),
        "seq_len": int(eval_cfg.get("perplexity_seq_len", 1024)),
        "stride": int(eval_cfg.get("perplexity_stride", 512)),
        "ppl_split": eval_cfg.get("perplexity_split", "test"),
        "toxicity_model": eval_cfg.get(
            "toxicity_model", "unitary/unbiased-toxic-roberta"
        ),
        "do_sample": bool(eval_cfg.get("do_sample", False)),
        "temperature": float(eval_cfg.get("temperature", 1.0)),
        "top_k": int(eval_cfg.get("top_k", 0)),
        "score_full_text": bool(getattr(args, "score_full_text", False)),
        "f1_tokenize_mode": getattr(args, "f1_tokenize_mode", None) or "multiset",
        "quick": bool(getattr(args, "quick", False)),
        "save_generations": bool(getattr(args, "save_generations", False)),
        "json": not bool(getattr(args, "no_json", False)),
        "verbose": not bool(getattr(args, "quiet", False)),
        "in_scope": in_scope,
    }

    # The paper's scope is GPT2-medium only; Llama2/GLU paths are stubbed.
    if in_scope and not in_scope.get("llama2", True):
        settings["llama2_in_scope"] = False

    if settings["quick"]:
        settings["n_prompts"] = min(settings["n_prompts"], 32)
        settings["n_f1"] = min(settings["n_f1"], 64)
        settings["max_new_tokens"] = min(settings["max_new_tokens"], 10)

    return settings


# --------------------------------------------------------------------------- #
# Model / metric plumbing
# --------------------------------------------------------------------------- #
def load_model_safe(name_or_path: str, device: Optional[str] = None, verbose: bool = True):
    """Load a causal LM + tokenizer via :mod:`src.model_utils`."""
    from src.model_utils import load_model

    if verbose:
        print(f"[eval] loading model: {name_or_path}")
    return load_model(name_or_path, device=device)


def build_scorer(settings: Dict[str, Any], prompts: Optional[List[str]] = None):
    """Instantiate the toxicity scorer (unbiased-toxic-roberta substitute)."""
    from src.eval.toxicity import ToxicityScorer

    return ToxicityScorer(
        model_name=settings["toxicity_model"],
        device=settings["device"],
        batch_size=max(settings["batch_size"], 16),
    )


def run_toxicity(
    model,
    tokenizer,
    settings: Dict[str, Any],
    scorer=None,
    generations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Section 3.3 toxicity: mean score over the 1,199 RealToxicityPrompts."""
    from src.eval.toxicity import evaluate_toxicity

    verbose = settings["verbose"]
    if verbose:
        print(
            f"[eval] toxicity over {settings['n_prompts']} RealToxicityPrompts "
            f"challenge prompts ({settings['max_new_tokens']} greedy tokens)"
        )

    result = evaluate_toxicity(
        model=model,
        tokenizer=tokenizer,
        prompts=None,  # challenge subset loaded internally
        model_name=settings["label"] or settings["model"],
        scorer=scorer,
        max_new_tokens=settings["max_new_tokens"],
        batch_size=settings["batch_size"],
        seed=settings["seed"],
        device=settings["device"],
        n_prompts=settings["n_prompts"],
        cache_dir=settings["cache_dir"],
        score_full_text=settings["score_full_text"],
        verbose=verbose,
        generations=generations,
    )
    return {"result": result, "summary": result.summary()}


def run_perplexity(model, tokenizer, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Section 3.3 perplexity on Wikitext-2."""
    from src.eval.perplexity import evaluate_perplexity

    verbose = settings["verbose"]
    if verbose:
        print(
            f"[eval] Wikitext-2 perplexity (seq_len={settings['seq_len']}, "
            f"stride={settings['stride']})"
        )

    result = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        corpus=None,
        split=settings["ppl_split"],
        seq_len=settings["seq_len"],
        stride=settings["stride"],
        cache_dir=settings["cache_dir"],
        model_name=settings["label"] or settings["model"],
        verbose=verbose,
        device=settings["device"],
    )
    return {"result": result, "summary": result.summary()}


def run_f1(
    model,
    tokenizer,
    settings: Dict[str, Any],
    generations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Section 3.3 F1 against 2,000 Wikipedia continuations."""
    from src.eval.f1 import evaluate_f1

    verbose = settings["verbose"]
    if verbose:
        print(
            f"[eval] F1 on {settings['n_f1']} Wikipedia sentences "
            f"(tokenize_mode={settings['f1_tokenize_mode']})"
        )

    result = evaluate_f1(
        model=model,
        tokenizer=tokenizer,
        pairs=None,
        n=settings["n_f1"],
        seed=settings["seed"],
        model_name=settings["label"] or settings["model"],
        max_new_tokens=settings["max_new_tokens"],
        batch_size=settings["batch_size"],
        device=settings["device"],
        max_prompt_length=settings["max_prompt_length"],
        tokenize_mode=settings["f1_tokenize_mode"],
        cache_dir=settings["cache_dir"],
        generations=generations,
        verbose=verbose,
    )
    return {"result": result, "summary": result.summary()}


# --------------------------------------------------------------------------- #
# Persistence / reporting
# --------------------------------------------------------------------------- #
def _json_default(obj: Any):
    """JSON fallback for numpy / torch / path scalars."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(obj, float):
        return obj
    if hasattr(obj, "__fspath__"):
        return os.fspath(obj)
    return str(obj)


def save_json(obj: Any, path: str, verbose: bool = False) -> str:
    """Write ``obj`` as JSON, creating parent directories as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    if verbose:
        print(f"[eval] wrote {path}")
    return path


def build_summary(
    settings: Dict[str, Any],
    sections: Dict[str, Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    """Assemble the unified evaluation summary triple (toxicity/PPL/F1)."""
    summary: Dict[str, Any] = {
        "label": settings["label"] or settings["model"],
        "model": settings["model"],
        "n_challenge_prompts": settings["n_prompts"],
        "n_f1_sentences": settings["n_f1"],
        "max_new_tokens": settings["max_new_tokens"],
        "seed": settings["seed"],
        "toxicity_scorer": settings["toxicity_model"],
        "toxicity_scorer_note": (
            "substituted for Perspective API (paper Section 3.3 footnote 3)"
        ),
        "generation": "greedy" if not settings["do_sample"] else "sampling",
        "elapsed_seconds": round(float(elapsed), 2),
    }
    for metric in METRICS:
        section = sections.get(metric)
        summary[metric] = (section or {}).get("summary")
    return summary


def reference_row(label: str) -> Optional[Dict[str, float]]:
    """Return the paper's reference numbers for a label, if known."""
    key = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in REFERENCE:
        return REFERENCE[key]
    for name, values in REFERENCE.items():
        if name in key or key in name:
            return values
    return None


def _fmt(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)


def print_summary_table(summary: Dict[str, Any], reference: Optional[Dict[str, float]] = None) -> None:
    """Pretty-print the metric triple next to the paper's reference values."""
    tox = (summary.get("toxicity") or {}).get("mean_toxicity")
    ppl = (summary.get("perplexity") or {}).get("ppl")
    f1 = (summary.get("f1") or {}).get("f1")

    print("\n" + "=" * 68)
    print(f"Evaluation summary -- {summary.get('label')}")
    print("=" * 68)
    header = f"{'metric':<14}{'measured':>12}"
    if reference:
        header += f"{'paper':>12}{'delta':>12}"
    print(header)
    print("-" * 68)
    for name, value, ref_key in (
        ("toxicity", tox, "toxicity"),
        ("perplexity", ppl, "perplexity"),
        ("F1", f1, "f1"),
    ):
        line = f"{name:<14}{_fmt(value):>12}"
        if reference:
            ref = reference.get(ref_key)
            delta = None
            if value is not None and ref is not None:
                delta = float(value) - float(ref)
            line += f"{_fmt(ref):>12}{_fmt(delta):>12}"
        print(line)
    print("=" * 68)
    print(
        f"prompts={summary.get('n_challenge_prompts')} "
        f"f1_sentences={summary.get('n_f1_sentences')} "
        f"max_new_tokens={summary.get('max_new_tokens')} "
        f"seed={summary.get('seed')}"
    )
    print(f"toxicity scorer: {summary.get('toxicity_scorer')} (Perspective API substitute)")
    print()


def save_metric_artifacts(
    settings: Dict[str, Any],
    sections: Dict[str, Dict[str, Any]],
    include_generations: bool = False,
) -> Dict[str, str]:
    """Persist per-metric JSON artifacts using each module's own saver."""
    out_dir = settings["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    label = settings["label"] or "model"
    paths: Dict[str, str] = {}

    for metric in METRICS:
        section = sections.get(metric)
        if not section or section.get("result") is None:
            continue
        try:
            if metric == "toxicity":
                from src.eval.toxicity import save_result as _save

                path = os.path.join(out_dir, f"toxicity_{label}.json")
                result = section["result"]
                if include_generations:
                    _save(path, result)
                else:
                    save_json(result.to_dict(), path, verbose=settings["verbose"])
                paths["toxicity"] = path
            elif metric == "perplexity":
                from src.eval.perplexity import save_result as _save

                path = os.path.join(out_dir, f"perplexity_{label}.json")
                _save(path, section["result"])
                paths["perplexity"] = path
            else:
                from src.eval.f1 import save_result as _save

                path = os.path.join(out_dir, f"f1_{label}.json")
                _save(path, section["result"], include_examples=include_generations)
                paths["f1"] = path
        except Exception as exc:  # pragma: no cover - artifacts are best-effort
            print(f"[eval] warning: could not save {metric} artifact ({exc})")
    return paths


def maybe_plot(
    settings: Dict[str, Any],
    sections: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Bar-plot the measured metrics (Table 2 / Table 4 style)."""
    try:
        from src.eval.f1 import plot_f1_comparison
        from src.eval.perplexity import plot_perplexity_comparison
        from src.eval.toxicity import plot_toxicity_comparison
    except Exception:  # pragma: no cover
        return None

    label = settings["label"] or "model"
    figure_dir = settings["figure_dir"]
    os.makedirs(figure_dir, exist_ok=True)


    written: List[str] = []
    tox = sections.get("toxicity", {}).get("result")
    if tox is not None:
        try:
            out = os.path.join(figure_dir, f"toxicity_{label}.png")
            path = plot_toxicity_comparison(
                {label: tox}, out_path=out, title=f"Toxicity -- {label}"
            )
            if path:
                written.append(path)
        except Exception:
            pass

    ppl = sections.get("perplexity", {}).get("result")
    if ppl is not None:
        try:
            out = os.path.join(figure_dir, f"perplexity_{label}.png")
            path = plot_perplexity_comparison(
                {label: ppl}, out_path=out, title=f"Wikitext-2 perplexity -- {label}"
            )
            if path:
                written.append(path)
        except Exception:
            pass

    f1 = sections.get("f1", {}).get("result")
    if f1 is not None:
        try:
            out = os.path.join(figure_dir, f"f1_{label}.png")
            path = plot_f1_comparison(
                {label: f1}, out_path=out, title=f"F1 -- {label}"
            )
            if path:
                written.append(path)
        except Exception:
            pass

    return written[0] if written else None


# --------------------------------------------------------------------------- #
# Single-model evaluation + comparison
# --------------------------------------------------------------------------- #
def evaluate_model(
    args: argparse.Namespace,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one model on the requested metrics and persist its artifacts."""
    settings = settings or resolve_settings(args)
    metrics = list(getattr(args, "metrics", None) or METRICS)
    start = time.time()

    model, tokenizer = load_model_safe(
        settings["model"], settings["device"], verbose=settings["verbose"]
    )

    scorer = None
    if "toxicity" in metrics:
        try:
            scorer = build_scorer(settings)
        except Exception as exc:
            print(f"[eval] warning: toxicity scorer unavailable ({exc})")

    sections: Dict[str, Dict[str, Any]] = {}

    shared_generations: Optional[List[str]] = None

    if "toxicity" in metrics:
        try:
            section = run_toxicity(model, tokenizer, settings, scorer=scorer)
            sections["toxicity"] = section
            if settings.get("save_generations"):
                shared_generations = list(section["result"].generations)
        except Exception as exc:
            print(f"[eval] ERROR: toxicity evaluation failed: {exc}")
            traceback.print_exc()

    if "perplexity" in metrics:
        try:
            sections["perplexity"] = run_perplexity(model, tokenizer, settings)
        except Exception as exc:
            print(f"[eval] ERROR: perplexity evaluation failed: {exc}")
            traceback.print_exc()

    if "f1" in metrics:
        try:
            # F1 prompts differ from RTP prompts, so generations are not shared;
            # the same greedy settings are used to keep the protocol identical.
            sections["f1"] = run_f1(model, tokenizer, settings)
        except Exception as exc:
            print(f"[eval] ERROR: F1 evaluation failed: {exc}")
            traceback.print_exc()

    elapsed = time.time() - start
    summary = build_summary(settings, sections, elapsed)
    summary["metrics"] = list(sections.keys())

    artifacts = save_metric_artifacts(
        settings, sections, include_generations=settings.get("save_generations", False)
    )
    figure = maybe_plot(settings, sections)

    label = settings["label"] or "model"
    summary["artifacts"] = artifacts
    summary["figure"] = figure

    if settings["json"]:
        path = os.path.join(settings["out_dir"], f"eval_{label}_summary.json")
        save_json(summary, path, verbose=settings["verbose"])
        summary["summary_path"] = path

    print_summary_table(summary, reference_row(label))
    return summary


def compare_models(
    summaries: Dict[str, Dict[str, Any]],
    out_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Summarise the GPT2 vs GPT2_DPO (or intervention) comparison."""
    rows: Dict[str, Dict[str, Any]] = {}
    for label, summary in summaries.items():
        rows[label] = {
            "toxicity": (summary.get("toxicity") or {}).get("mean_toxicity"),
            "perplexity": (summary.get("perplexity") or {}).get("ppl"),
            "f1": (summary.get("f1") or {}).get("f1"),
            "reference": reference_row(label),
        }

    comparison: Dict[str, Any] = {"rows": rows}

    if len(summaries) >= 2:
        labels = list(summaries.keys())
        before, after = summaries[labels[0]], summaries[labels[1]]
        deltas: Dict[str, Any] = {}
        for metric, getter in (
            ("toxicity", lambda s: (s.get("toxicity") or {}).get("mean_toxicity")),
            ("perplexity", lambda s: (s.get("perplexity") or {}).get("ppl")),
            ("f1", lambda s: (s.get("f1") or {}).get("f1")),
        ):
            a, b = getter(before), getter(after)
            if a is None or b is None:
                deltas[metric] = None
                continue
            deltas[metric] = {
                "before": float(a),
                "after": float(b),
                "absolute_delta": float(b) - float(a),
                "relative_delta": (float(b) - float(a)) / float(a) if a else None,
            }
        comparison["labels"] = labels
        comparison["deltas"] = deltas

    if out_path and verbose:
        print(f"[eval] comparison written to {out_path}")
    if out_path:
        save_json(comparison, out_path, verbose=verbose)
    return comparison


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GPT2/GPT2_DPO on the paper's Section 3.3 metrics "
            "(toxicity on 1,199 RealToxicityPrompts, Wikitext-2 perplexity, "
            "and token-overlap F1 on 2,000 Wikipedia sentences)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path.")
    parser.add_argument(
        "--label",
        default=None,
        help="Short label used in artifact names (e.g. gpt2, gpt2_dpo).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate both GPT2 and GPT2_DPO and compare them.",
    )
    parser.add_argument(
        "--dpo-model",
        dest="dpo_model",
        default=DEFAULT_DPO_DIR,
        help="GPT2_DPO checkpoint directory (used with --all).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRICS),
        default=list(METRICS),
        help="Subset of metrics to evaluate.",
    )
    parser.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", dest="cache_dir", default=None)
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 (auto).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--n-prompts",
        dest="n_prompts",
        type=int,
        default=None,
        help="Number of RealToxicityPrompts challenge prompts (paper: 1199).",
    )
    parser.add_argument(
        "--n-f1",
        dest="n_f1",
        type=int,
        default=None,
        help="Number of Wikipedia sentences for F1 (paper: 2000).",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Greedy continuation length (paper: 20).",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument(
        "--f1-tokenize-mode",
        dest="f1_tokenize_mode",
        default="multiset",
        choices=["whitespace", "tokenizer", "set", "multiset"],
    )
    parser.add_argument(
        "--score-full-text",
        dest="score_full_text",
        action="store_true",
        help="Score prompt+continuation instead of the continuation only.",
    )
    parser.add_argument(
        "--save-generations",
        dest="save_generations",
        action="store_true",
        help="Include generations in saved artifacts.",
    )
    parser.add_argument("--quick", action="store_true", help="Smoke-test settings.")
    parser.add_argument("--dry-run", action="store_true", help="Print settings only.")
    parser.add_argument("--no-json", dest="no_json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if not args.quiet:
            import torch  # noqa: F401  (device info only)

            from src.model_utils import resolve_device

            device = resolve_device(args.device)
            print(f"[eval] device: {device}")

        if args.all:
            settings_before = resolve_settings(args)
            settings_before["model"] = args.model
            settings_before["label"] = args.label or "gpt2"

            args_after = argparse.Namespace(**vars(args))
            args_after.model = args.dpo_model
            args_after.label = "gpt2_dpo"
            settings_after = resolve_settings(args_after)

            if args.dry_run:
                print(json.dumps(
                    {"before": settings_before, "after": settings_after},
                    indent=2, default=_json_default,
                ))
                return 0

            summaries: Dict[str, Dict[str, Any]] = {}
            summaries[settings_before["label"]] = evaluate_model(args, settings_before)
            summaries[settings_after["label"]] = evaluate_model(args_after, settings_after)
            compare_models(
                summaries,
                out_path=os.path.join(settings_before["out_dir"], "eval_comparison.json"),
                verbose=not args.quiet,
            )
            return 0

        settings = resolve_settings(args)
        settings["model"] = args.model
        settings["label"] = args.label or os.path.basename(str(args.model).rstrip("/"))

        if args.dry_run:
            print(json.dumps(settings, indent=2, default=_json_default))
            return 0

        evaluate_model(args, settings)
        return 0

    except KeyboardInterrupt:  # pragma: no cover
        print("\n[eval] interrupted by user")
        return 130
    except Exception as exc:  # pragma: no cover
        print(f"[eval] fatal error: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
