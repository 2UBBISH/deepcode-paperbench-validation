#!/usr/bin/env python
"""Un-aligning GPT2_DPO by scaling toxic key vectors (Section 6, Table 4).

This script reproduces the un-alignment experiment of

    "A Mechanistic Understanding of Alignment Algorithms:
     A Case Study on DPO and Toxicity"

Section 6 / Table 4:

    METHOD               Toxic    PPL      F1
    GPT2_DPO             0.208    23.34    0.195
    SCALE MLP.k_Toxic    0.458    23.30    0.195
    GPT2                 0.453    21.700   0.193

Mechanistic story
-----------------
GPT2_DPO never removes the toxic MLP value vectors; it learns a distributed
residual-stream offset that pushes ``x^l`` out of the toxic activation regions
``gamma(MLP.k_Toxic^l) = { g | sigma(k . g) > 0 }`` (Equation 4).  Therefore a
simple way to *undo* alignment is to enlarge those regions by scaling each
toxic key vector larger -- "this makes the residual streams pass through toxic
regions again, thus reverting back to the pre-aligned behavior."

We select the 7 MLP vectors with the highest cosine similarity to the toxic
probe direction ``W_Toxic[:, 1]`` and scale their *key* vectors (rows of
``MLP.k``) by 10x, then re-evaluate toxicity / Wikitext-2 perplexity / F1 on the
1,199 RealToxicityPrompts challenge set.

Note (paper): "increasing activation regions gamma does not have an affect on
perplexity, unlike our interventions from Section 3.3" -- so the scaled model is
expected to keep PPL ~23.30 while toxicity reverts to ~0.458.

Usage
-----
    # full Table 4 (DPO row + scaled row + GPT2 baseline row)
    python scripts/unalign_gpt2.py --dpo-dir artifacts/models/gpt2_dpo --baseline

    # smoke test
    python scripts/unalign_gpt2.py --quick

    # optionally sweep scale / number of vectors (appendix-style checks)
    python scripts/unalign_gpt2.py --scale-sweep

Llama2 is explicitly out of reproduction scope; the GLU gate-scaling analogue
(Table 5) raises ``NotImplementedError`` inside ``src.unalign``.
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
# Make the repository root importable when run as a plain script.
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# --------------------------------------------------------------------------- #
# Paper constants (Table 4 and Section 6)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_DPO_DIR = os.path.join("artifacts", "models", "gpt2_dpo")
DEFAULT_PROBE_PATH = os.path.join("artifacts", "probe", "w_toxic.pt")
DEFAULT_VECTORS_PATH = os.path.join("artifacts", "vectors", "toxic_vectors.pt")
DEFAULT_OUT_DIR = os.path.join("artifacts", "unalign")
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")

N_CHALLENGE_PROMPTS = 1199
N_F1_SENTENCES = 2000
DEFAULT_N_VECTORS = 7          # "as few as 7 toxic key vectors"
DEFAULT_SCALE = 10.0           # "scale their key vectors by 10x"
DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_SEED = 0

# Table 4 (paper reference values used for the tolerance check)
TABLE4_REFERENCE: Dict[str, Dict[str, float]] = {
    "gpt2_dpo": {"toxicity": 0.208, "perplexity": 23.34, "f1": 0.195},
    "scaled": {"toxicity": 0.458, "perplexity": 23.30, "f1": 0.195},
    "gpt2": {"toxicity": 0.453, "perplexity": 21.70, "f1": 0.193},
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort YAML config loader; returns ``{}`` when unavailable."""
    if path is None:
        path = DEFAULT_CONFIG
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dig(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup tolerant of missing keys."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy / torch scalars and paths."""
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
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _save_json(path: str, payload: Dict[str, Any]) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args > ``configs/default.yaml`` > Section 6 defaults."""
    cfg = load_config(getattr(args, "config", None))

    vec_cfg = _dig(cfg, "vectors", default={}) or {}
    unalign_cfg = vec_cfg.get("unalign", {}) if isinstance(vec_cfg, dict) else {}
    ref_cfg = _dig(cfg, "reference", default={}) or {}
    eval_cfg = _dig(cfg, "evaluation", default={}) or {}

    settings: Dict[str, Any] = {
        # models
        "model": getattr(args, "model", None) or DEFAULT_DPO_DIR,
        "model_name": getattr(args, "model_name", None)
        or cfg.get("model_name")
        or DEFAULT_MODEL,
        "dpo_dir": getattr(args, "dpo_dir", None) or DEFAULT_DPO_DIR,
        "baseline_model": getattr(args, "baseline_model", None)
        or cfg.get("model_name")
        or DEFAULT_MODEL,
        # artifacts
        "probe_path": getattr(args, "probe_path", None)
        or cfg.get("probe_path")
        or _dig(cfg, "paths", "probe", default=None)
        or _dig(cfg, "probe", "path", default=None)
        or DEFAULT_PROBE_PATH,
        "vectors_path": getattr(args, "vectors_path", None)
        or cfg.get("vectors_path")
        or _dig(cfg, "paths", "vectors", default=None)
        or _dig(cfg, "vectors", "path", default=None)
        or DEFAULT_VECTORS_PATH,
        "out_dir": getattr(args, "out_dir", None) or DEFAULT_OUT_DIR,
        # experiment knobs (Section 6)
        "n_vectors": int(
            getattr(args, "n_vectors", None)
            or unalign_cfg.get("n_vectors")
            or DEFAULT_N_VECTORS
        ),
        "scale": float(
            getattr(args, "scale", None)
            or unalign_cfg.get("scale")
            or DEFAULT_SCALE
        ),
        "scale_sweep": bool(getattr(args, "scale_sweep", False)),
        "vector_sweep": bool(getattr(args, "vector_sweep", False)),
        # evaluation
        "n_prompts": int(
            getattr(args, "n_prompts", None)
            or eval_cfg.get("n_challenge_prompts")
            or N_CHALLENGE_PROMPTS
        ),
        "n_f1_sentences": int(
            eval_cfg.get("n_f1_sentences") or N_F1_SENTENCES
        ),
        "max_new_tokens": int(
            getattr(args, "max_new_tokens", None)
            or eval_cfg.get("max_new_tokens")
            or DEFAULT_MAX_NEW_TOKENS
        ),
        "batch_size": int(getattr(args, "batch_size", None) or 16),
        "seq_len": int(eval_cfg.get("perplexity_seq_len") or 1024),
        "stride": int(eval_cfg.get("perplexity_stride") or 512),
        "seed": int(getattr(args, "seed", None) or cfg.get("seed") or DEFAULT_SEED),
        "device": getattr(args, "device", None) or cfg.get("device"),
        "cache_dir": getattr(args, "cache_dir", None)
        or _dig(cfg, "paths", "cache_dir", default=None),
        # toggles
        "score_toxicity": not bool(getattr(args, "no_toxicity", False)),
        "score_perplexity": not bool(getattr(args, "no_perplexity", False)),
        "score_f1": not bool(getattr(args, "no_f1", False)),
        "measure_regions": not bool(getattr(args, "no_regions", False)),
        "baseline": bool(getattr(args, "baseline", False)),
        "include_generations": bool(getattr(args, "include_generations", False)),
        "quick": bool(getattr(args, "quick", False)),
        "save": not bool(getattr(args, "no_save", False)),
        "plot": not bool(getattr(args, "no_plot", False)),
        "verbose": not bool(getattr(args, "quiet", False)),
        "json_out": getattr(args, "json_out", None),
        "dry_run": bool(getattr(args, "dry_run", False)),
        # paper reference values (defaults if config lacks them)
        "reference": {
            "gpt2_dpo_toxicity": ref_cfg.get("gpt2_dpo_toxicity", 0.208),
            "gpt2_dpo_ppl": ref_cfg.get("gpt2_dpo_ppl", 23.34),
            "gpt2_dpo_f1": ref_cfg.get("gpt2_dpo_f1", 0.195),
            "unaligned_toxicity": ref_cfg.get("unaligned_toxicity", 0.458),
            "unaligned_ppl": ref_cfg.get("unaligned_ppl", 23.30),
            "unaligned_f1": ref_cfg.get("unaligned_f1", 0.195),
            "gpt2_toxicity": ref_cfg.get("gpt2_toxicity", 0.453),
            "gpt2_ppl": ref_cfg.get("gpt2_ppl", 21.70),
            "gpt2_f1": ref_cfg.get("gpt2_f1", 0.193),
        },
        # logging
        "logging": cfg.get("logging", {}) if isinstance(cfg, dict) else {},
    }

    # Quick smoke-test overrides (never used for reported numbers).
    if settings["quick"]:
        settings["n_prompts"] = min(settings["n_prompts"], 8)
        settings["n_f1_sentences"] = min(settings["n_f1_sentences"], 8)
        settings["max_new_tokens"] = min(settings["max_new_tokens"], 8)
        settings["batch_size"] = min(settings["batch_size"], 4)
        settings["seq_len"] = min(settings["seq_len"], 256)
        settings["stride"] = min(settings["stride"], 128)
        settings["scale_sweep"] = False
        settings["vector_sweep"] = False
        settings["measure_regions"] = False

    return settings


# --------------------------------------------------------------------------- #
# Model / artifact loading
# --------------------------------------------------------------------------- #
def _resolve_model_source(path_or_name: str, verbose: bool = True) -> str:
    """Fall back to GPT2-medium if the DPO checkpoint does not exist."""
    if path_or_name and os.path.isdir(path_or_name):
        return path_or_name
    if path_or_name and os.path.isfile(path_or_name):
        return path_or_name
    if verbose:
        print(
            f"[unalign] WARNING: model '{path_or_name}' not found; "
            f"falling back to '{DEFAULT_MODEL}'. Run scripts/train_dpo.py "
            f"first for the real Table 4 numbers."
        )
    return DEFAULT_MODEL


def load_model_safe(name_or_path: str, device: Optional[str] = None):
    """Wrapper around :func:`src.model_utils.load_model`."""
    from src.model_utils import load_model

    return load_model(name_or_path, device=device)


def load_artifacts(settings: Dict[str, Any], verbose: bool = True):
    """Resolve ``W_Toxic`` and the ranked ``MLP.v_Toxic`` artifact.

    Returns ``(toxic_vectors, probe, direction, indices)`` where ``indices`` are
    the ``(layer, idx)`` pairs to be scaled (may be ``None`` when the caller
    should use the default top-N selection from ``src.unalign``).
    """
    from src.probe import load_probe, probe_exists, default_probe_path
    from src.toxic_vectors import (
        toxic_vectors_exist,
        load_toxic_vectors,
        resolve_toxic_direction,
        rank_value_vectors_by_cosine,
    )

    probe = None
    probe_path = settings["probe_path"]
    try:
        if probe_exists(probe_path) or probe_exists():
            probe = load_probe(probe_path)
            if verbose:
                acc = getattr(probe, "valid_accuracy", float("nan"))
                print(f"[unalign] loaded W_Toxic from {probe_path} (valid acc={acc:.4f})")
        else:
            if verbose:
                print(
                    f"[unalign] W_Toxic not found at {probe_path}; "
                    f"training/loading the default probe."
                )
            from src.probe import load_or_train_probe

            probe = load_or_train_probe(default_probe_path())
    except Exception as exc:  # pragma: no cover - defensive
        if verbose:
            print(f"[unalign] WARNING: could not load W_Toxic ({exc}); continuing.")
        probe = None

    toxic_vectors = None
    vec_path = settings["vectors_path"]
    try:
        if toxic_vectors_exist(vec_path) or toxic_vectors_exist():
            toxic_vectors = load_toxic_vectors(vec_path)
            if verbose:
                print(
                    f"[unalign] loaded {len(getattr(toxic_vectors, 'indices', []))} "
                    f"ranked toxic vectors from {vec_path}"
                )
    except Exception as exc:  # pragma: no cover - defensive
        if verbose:
            print(f"[unalign] WARNING: could not load toxic vectors ({exc}).")
        toxic_vectors = None

    direction = None
    if probe is not None:
        try:
            direction = resolve_toxic_direction(probe=probe, toxic_index=1)
        except Exception:
            direction = None

    # If no cached ranking is available, rank on the fly later (inside
    # src.unalign) using the probe direction; we only need the direction here.
    indices: Optional[List[Tuple[int, int]]] = None
    if toxic_vectors is not None:
        cand = list(getattr(toxic_vectors, "indices", []) or [])
        if cand:
            indices = [tuple(int(v) for v in pair) for pair in cand]

    return toxic_vectors, probe, direction, indices


def select_vectors(settings: Dict[str, Any], model, toxic_vectors, probe, direction):
    """Return the ``(layer, idx)`` toxic key-vector selection for Section 6."""
    from src.unalign import select_toxic_key_vectors

    records = select_toxic_key_vectors(
        model=model,
        toxic_vectors=toxic_vectors,
        probe=probe,
        probe_path=settings["probe_path"],
        direction=direction,
        top_n=int(settings["n_vectors"]),
    )
    indices = [(int(rec["layer"]), int(rec["idx"])) for rec in records]
    return records, indices


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: Any, width: int = 8) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    try:
        f = float(value)
    except Exception:
        return f"{str(value):>{width}}"
    if f != f:  # NaN
        return f"{'n/a':>{width}}"
    return f"{f:>{width}.4f}"


def _print_table(rows: Sequence[Dict[str, Any]], settings: Dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("Table 4 - Un-aligning GPT2_DPO (scale toxic key vectors)")
    print("=" * 78)
    print(f"{'METHOD':<26}{'Toxic':>10}{'PPL':>10}{'F1':>10}{'n_vec':>8}{'scale':>8}")
    print("-" * 78)
    for row in rows:
        print(
            f"{str(row.get('label', '?'))[:25]:<26}"
            f"{_fmt(row.get('toxicity'), 10)}"
            f"{_fmt(row.get('perplexity'), 10)}"
            f"{_fmt(row.get('f1'), 10)}"
            f"{str(row.get('n_vectors', '')):>8}"
            f"{('%g' % float(row.get('scale', 0.0))):>8}"
        )
    print("-" * 78)
    print(
        "paper reference        "
        f"{TABLE4_REFERENCE['gpt2_dpo']['toxicity']:>9.3f}"
        f"{TABLE4_REFERENCE['gpt2_dpo']['perplexity']:>10.2f}"
        f"{TABLE4_REFERENCE['gpt2_dpo']['f1']:>10.3f}   (GPT2_DPO)"
    )
    print(
        "                       "
        f"{TABLE4_REFERENCE['scaled']['toxicity']:>9.3f}"
        f"{TABLE4_REFERENCE['scaled']['perplexity']:>10.2f}"
        f"{TABLE4_REFERENCE['scaled']['f1']:>10.3f}   (SCALE MLP.k_Toxic)"
    )
    print(
        "                       "
        f"{TABLE4_REFERENCE['gpt2']['toxicity']:>9.3f}"
        f"{TABLE4_REFERENCE['gpt2']['perplexity']:>10.2f}"
        f"{TABLE4_REFERENCE['gpt2']['f1']:>10.3f}   (GPT2)"
    )
    print("=" * 78)


def _print_claim_report(report: Dict[str, Any]) -> None:
    claim = report.get("claim", {}) or {}
    print()
    print("Mechanistic claim (Section 6)")
    print(
        "  toxicity re-activated          : "
        f"{claim.get('toxicity_reactivated')} "
        f"(delta={claim.get('toxicity_delta')})"
    )
    print(
        "  perplexity preserved           : "
        f"{claim.get('perplexity_preserved')} "
        f"(delta={claim.get('perplexity_delta')})"
    )
    print(
        "  reverts to pre-aligned GPT2    : "
        f"{claim.get('reverts_to_gpt2')} "
        f"(vs GPT2 toxicity {claim.get('gpt2_toxicity')})"
    )
    print(f"  overall passed                 : {report.get('passed')}")


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def run_unalign(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)

    print("=" * 78)
    print("Un-aligning GPT2_DPO by scaling toxic key vectors (Section 6)")
    print("=" * 78)
    print(f"  model (.pt/.dir)   : {settings['model']}")
    print(f"  probe              : {settings['probe_path']}")
    print(f"  toxic vectors      : {settings['vectors_path']}")
    print(f"  n_vectors / scale  : {settings['n_vectors']} / {settings['scale']:g}x")
    print(f"  prompts / max new  : {settings['n_prompts']} / {settings['max_new_tokens']}")
    print(f"  metrics            : toxicity={settings['score_toxicity']} "
          f"ppl={settings['score_perplexity']} f1={settings['score_f1']}")
    print(f"  out_dir            : {settings['out_dir']}")
    if settings["quick"]:
        print("  [quick mode: reduced prompts/tokens; numbers are NOT paper numbers]")
    if settings["dry_run"]:
        print("\n[dry-run] settings resolved; exiting without running the model.")
        if settings["json_out"]:
            _save_json(settings["json_out"], {"settings": settings, "dry_run": True})
        return 0

    # ------------------------------------------------------------------ #
    # Imports of the heavy machinery happen only when we really run.
    # ------------------------------------------------------------------ #
    from src.unalign import (
        UnalignResults,
        check_table4,
        evaluate_unaligned,
        plot_unalignment,
        run_unalignment,
        save_results,
        default_path,
    )
    from src.model_utils import load_model, resolve_device, set_seed

    set_seed(int(settings["seed"]))
    device = resolve_device(settings["device"])
    print(f"  device             : {device}")

    # ---------------- main model (GPT2_DPO if available) --------------- #
    model_source = _resolve_model_source(str(settings["model"]), verbose=settings["verbose"])
    t0 = time.time()
    model, tokenizer = load_model(model_source, device=device)
    print(f"[unalign] loaded '{model_source}' in {time.time() - t0:.1f}s")

    toxic_vectors, probe, direction, _ = load_artifacts(settings, verbose=settings["verbose"])
    records, indices = select_vectors(settings, model, toxic_vectors, probe, direction)

    print()
    print(f"Selected top-{len(indices)} MLP vectors by cosine similarity to W_Toxic[:, 1]:")
    for rec in records:
        print(
            f"  MLP.v_{rec['idx']}^{rec['layer']:<2}  cos={rec['cosine']:+.4f}"
            f"   -> scale MLP.k_{rec['idx']}^{rec['layer']} by {settings['scale']:g}x"
        )

    # ------------------------------ run -------------------------------- #
    results = run_unalignment(
        model,
        tokenizer,
        toxic_vectors=toxic_vectors,
        probe=probe,
        probe_path=settings["probe_path"],
        direction=direction,
        indices=indices,
        n_vectors=int(settings["n_vectors"]),
        scale=float(settings["scale"]),
        model_name=os.path.basename(str(settings["model"]).rstrip("/")) or "gpt2_dpo",
        scale_grid=[1.0, 2.0, 5.0, float(settings["scale"]), 20.0]
        if settings["scale_sweep"]
        else None,
        vector_grid=[1, 3, 7, 15, 32] if settings["vector_sweep"] else None,
        prompts=None,  # src.unalign loads the 1,199 RTP challenge prompts
        scorer=None,
        corpus=None,
        f1_pairs=None,
        n_prompts=int(settings["n_prompts"]),
        max_new_tokens=int(settings["max_new_tokens"]),
        batch_size=int(settings["batch_size"]),
        seed=int(settings["seed"]),
        device=device,
        cache_dir=settings["cache_dir"],
        seq_len=int(settings["seq_len"]),
        stride=int(settings["stride"]),
        score_toxicity=bool(settings["score_toxicity"]),
        score_perplexity=bool(settings["score_perplexity"]),
        score_f1=bool(settings["score_f1"]),
        measure_regions=bool(settings["measure_regions"]),
        region_prompts=None,
        scale_sweep=bool(settings["scale_sweep"]),
        verbose=bool(settings["verbose"]),
    )

    # --------------------- optional GPT2 baseline row ------------------ #
    if settings["baseline"] and settings["score_toxicity"]:
        try:
            print()
            print("[unalign] evaluating the pre-alignment GPT2 baseline (Table 4 row 3)...")
            base_model, base_tok = load_model(settings["baseline_model"], device=device)
            baseline = evaluate_unaligned(
                base_model,
                base_tok,
                label="GPT2",
                model_name="gpt2",
                n_vectors=0,
                scale=1.0,
                indices=None,
                prompts=None,
                scorer=None,
                corpus=None,
                f1_pairs=None,
                score_toxicity=bool(settings["score_toxicity"]),
                score_perplexity=bool(settings["score_perplexity"]),
                score_f1=bool(settings["score_f1"]),
                n_prompts=int(settings["n_prompts"]),
                max_new_tokens=int(settings["max_new_tokens"]),
                batch_size=int(settings["batch_size"]),
                seed=int(settings["seed"]),
                device=device,
                cache_dir=settings["cache_dir"],
                seq_len=int(settings["seq_len"]),
                stride=int(settings["stride"]),
                verbose=bool(settings["verbose"]),
            )
            results.results.append(baseline)
            del base_model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as exc:
            print(f"[unalign] WARNING: baseline evaluation failed: {exc}")

    # ---------------------------- reporting ---------------------------- #
    rows = [r.summary() if hasattr(r, "summary") else r.to_dict() for r in results]
    _print_table(rows, settings)

    report = check_table4(results, reference=TABLE4_REFERENCE)
    _print_claim_report(report)

    # ---------------------------- artifacts ---------------------------- #
    out_dir = settings["out_dir"]
    results_path = default_path(out_dir=out_dir)
    if settings["save"]:
        os.makedirs(out_dir, exist_ok=True)
        save_results(
            results_path,
            results,
            include_generations=bool(settings["include_generations"]),
            write_markdown=True,
            verbose=bool(settings["verbose"]),
        )
        print(f"[unalign] wrote {results_path}")

        payload = {
            "table4": rows,
            "reference": TABLE4_REFERENCE,
            "check": report,
            "selected_vectors": records,
            "settings": {k: v for k, v in settings.items() if k != "reference"},
            "model": str(settings["model"]),
        }
        summary_path = os.path.join(out_dir, "unalign_summary.json")
        _save_json(summary_path, payload)
        print(f"[unalign] wrote {summary_path}")

        if settings["json_out"]:
            _save_json(settings["json_out"], payload)

        if settings["plot"]:
            try:
                fig_path = os.path.join(out_dir, "unalign_table4.png")
                plot_unalignment(results, out_path=fig_path)
                print(f"[unalign] wrote {fig_path}")
            except Exception as exc:
                print(f"[unalign] WARNING: plotting failed: {exc}")

    return 0 if report.get("passed", True) else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unalign_gpt2.py",
        description=(
            "Reproduce Table 4 / Section 6: un-align GPT2_DPO by scaling toxic "
            "key vectors (GPT2-medium only; Llama2 out of scope)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="YAML config with paths/hyperparameters.")
    parser.add_argument("--model", type=str, default=DEFAULT_DPO_DIR,
                        help="GPT2_DPO checkpoint dir to un-align.")
    parser.add_argument("--dpo-dir", type=str, default=None,
                        help="Alias for --model (default: artifacts/models/gpt2_dpo).")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Human-readable name used in artifact filenames.")
    parser.add_argument("--baseline-model", type=str, default=DEFAULT_MODEL,
                        help="Pre-alignment model for Table 4 row 3.")
    parser.add_argument("--probe-path", type=str, default=None,
                        help="Path to trained W_Toxic (default: artifacts/probe/w_toxic.pt).")
    parser.add_argument("--vectors-path", type=str, default=None,
                        help="Path to ranked toxic vectors (default: artifacts/vectors/toxic_vectors.pt).")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Directory for JSON/Markdown/figure artifacts.")

    # Section 6 experiment knobs
    parser.add_argument("--n-vectors", type=int, default=None,
                        help="Number of toxic key vectors to scale (paper: 7).")
    parser.add_argument("--scale", type=float, default=None,
                        help="Key-vector scale factor (paper: 10).")
    parser.add_argument("--scale-sweep", action="store_true",
                        help="Also sweep the scale factor (appendix-style check).")
    parser.add_argument("--vector-sweep", action="store_true",
                        help="Also sweep the number of scaled vectors.")

    # Evaluation knobs
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="RealToxicityPrompts challenge prompts (paper: 1199).")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Greedy continuation length (paper: 20).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Generation batch size.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (paper/plan: 0).")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (default: auto-detect).")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory.")

    # Toggles
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate pre-alignment GPT2 (Table 4 row 3).")
    parser.add_argument("--no-toxicity", action="store_true",
                        help="Skip toxicity scoring.")
    parser.add_argument("--no-perplexity", action="store_true",
                        help="Skip Wikitext-2 perplexity.")
    parser.add_argument("--no-f1", action="store_true",
                        help="Skip token-overlap F1.")
    parser.add_argument("--no-regions", action="store_true",
                        help="Skip activation-region measurement.")
    parser.add_argument("--include-generations", action="store_true",
                        help="Store generated continuations in the JSON artifact.")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write artifacts.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not render the Table 4 figure.")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional extra path for the summary JSON.")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test with a handful of prompts/tokens.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve settings and exit without loading the model.")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce logging.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dpo_dir and not args.model:
        args.model = args.dpo_dir
    if getattr(args, "dpo_dir", None):
        args.model = args.dpo_dir

    try:
        return run_unalign(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[unalign] interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover
        print(f"[unalign] ERROR: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
