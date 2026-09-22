#!/usr/bin/env python
"""Table 2 / Table 3 residual-stream subtraction interventions (Section 3.3).

During the forward pass at the last layer we replace

    x^{L-1} = x^{L-1} - alpha * W

where ``W`` is ``W_Toxic``, an ``MLP.v_Toxic`` value vector, or an
``SVD.U_Toxic`` singular vector, and ``alpha`` is chosen such that the
resulting Wikitext-2 perplexity is similar to that of the post-DPO model
(see the paragraph after Table 3 in Section 3.3).

The three metrics reported by the paper are toxicity (RealToxicityPrompts
"challenge" subset of 1,199 prompts), perplexity (Wikitext-2) and F1
(2,000 Wikipedia sentences).  In this reproduction the Perspective API is
substituted by ``unitary/unbiased-toxic-roberta``.

Usage
-----
    python scripts/run_interventions.py                    # full Table 2
    python scripts/run_interventions.py --quick             # smoke test
    python scripts/run_interventions.py --specs w_toxic svd_u_0
    python scripts/run_interventions.py --table3            # Table 3 examples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")
DEFAULT_PROBE_PATH = os.path.join("artifacts", "probe", "w_toxic.pt")
DEFAULT_VECTORS_PATH = os.path.join("artifacts", "vectors", "toxic_vectors.pt")
DEFAULT_OUT_DIR = os.path.join("artifacts", "interventions")

#: Table 2 reference (toxicity / perplexity / F1) for GPT2-medium.
TABLE2_REFERENCE = {
    "no_op": (0.453, 21.70, 0.193),
    "w_toxic": (0.245, 23.56, 0.193),
    "mlp_v_770_19": (0.305, 23.30, 0.192),
    "svd_u_toxic_0": (0.268, 23.48, 0.193),
}

#: Supported ``--specs`` names mapped onto (kind, index) intervention specs.
SPEC_ALIASES = {
    "no_op": ("none", None),
    "none": ("none", None),
    "w_toxic": ("w_toxic", None),
    "probe": ("w_toxic", None),
    "mlp_v_770_19": ("mlp_value", (19, 770)),
    "v_770_19": ("mlp_value", (19, 770)),
    "mlp.v_770^19": ("mlp_value", (19, 770)),
    "svd_u_toxic_0": ("svd_u", 0),
    "svd_u_0": ("svd_u", 0),
    "svd.u_toxic[0]": ("svd_u", 0),
    "svd_u_toxic_1": ("svd_u", 1),
    "svd_u_toxic_2": ("svd_u", 2),
}


# --------------------------------------------------------------------------- #
# configuration helpers
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config file, returning ``{}`` when unavailable."""
    candidates = [path] if path else [
        DEFAULT_CONFIG,
        os.path.join(_ROOT, DEFAULT_CONFIG),
    ]
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    return yaml.safe_load(handle) or {}
            except Exception:
                return {}
    return {}


def _dig(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup tolerant of missing keys."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args, YAML config and defaults into a settings dict."""
    cfg = load_config(getattr(args, "config", None))
    inter_cfg = _dig(cfg, "interventions", default={}) or {}

    def pick(name: str, cfg_key: str, default: Any) -> Any:
        value = getattr(args, name, None)
        if value is None:
            value = inter_cfg.get(cfg_key)
        if value is None:
            value = _dig(cfg, "interventions", cfg_key)
        return default if value is None else value

    model_name = pick("model", "model", None) or _dig(cfg, "models", "base") \
        or _dig(cfg, "model", "name") or DEFAULT_MODEL
    probe_path = getattr(args, "probe", None) or _dig(cfg, "paths", "probe") \
        or DEFAULT_PROBE_PATH
    vectors_path = getattr(args, "vectors", None) or _dig(cfg, "paths", "vectors") \
        or DEFAULT_VECTORS_PATH
    out_dir = getattr(args, "out_dir", None) or _dig(cfg, "paths", "interventions") \
        or DEFAULT_OUT_DIR

    quick = bool(getattr(args, "quick", False))
    settings: Dict[str, Any] = {
        "model_name": model_name,
        "probe_path": probe_path,
        "vectors_path": vectors_path,
        "out_dir": out_dir,
        "layer": getattr(args, "layer", None),
        "target_ppl": getattr(args, "target_ppl", None) or 23.34,
        "alphas": _parse_alphas(getattr(args, "alphas", None)),
        "select_alpha": not bool(getattr(args, "no_alpha_search", False)),
        "position": getattr(args, "position", None) or "mid",
        "n_prompts": 32 if quick else (getattr(args, "n_prompts", None) or 1199),
        "max_new_tokens": 12 if quick else (getattr(args, "max_new_tokens", None) or 20),
        "batch_size": getattr(args, "batch_size", None) or (8 if quick else 16),
        "seed": getattr(args, "seed", None) or 0,
        "device": getattr(args, "device", None),
        "seq_len": getattr(args, "seq_len", None) or 1024,
        "stride": getattr(args, "stride", None) or 512,
        "score_toxicity": not bool(getattr(args, "no_toxicity", False)),
        "score_perplexity": not bool(getattr(args, "no_perplexity", False)),
        "score_f1": not bool(getattr(args, "no_f1", False)),
        "save": not bool(getattr(args, "no_save", False)),
        "json_out": getattr(args, "json_out", None),
        "specs": getattr(args, "specs", None),
        "quick": quick,
        "table3": bool(getattr(args, "table3", False)),
        "verbose": not bool(getattr(args, "quiet", False)),
    }
    if quick:
        settings["alphas"] = [1.0, 2.0, 4.0]
        settings["score_perplexity"] = bool(getattr(args, "perplexity", False))
    return settings


def _parse_alphas(raw: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated alpha grid such as ``"1,2,4,8"``."""
    if not raw:
        return None
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    try:
        return [float(part) for part in str(raw).split(",") if str(part).strip()]
    except ValueError:
        return None


def build_specs(names: Optional[Sequence[str]], settings: Dict[str, Any]) -> List[Any]:
    """Translate ``--specs`` names into ``InterventionSpec`` objects."""
    from src.interventions import InterventionSpec, default_specs

    default_layer = settings.get("layer")
    if not names:
        return default_specs(layer=default_layer) if default_layer is not None \
            else default_specs()

    specs = []
    for name in names:
        key = str(name).strip().lower()
        if key not in SPEC_ALIASES:
            raise SystemExit(
                f"unknown intervention spec '{name}'; choose from "
                f"{sorted(set(SPEC_ALIASES))}"
            )
        kind, index = SPEC_ALIASES[key]
        specs.append(
            InterventionSpec(
                kind=kind,
                index=index,
                label=key,
                layer=settings.get("layer"),
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# model / artifact loading
# --------------------------------------------------------------------------- #
def load_model_safe(name_or_path: str, device: Optional[str] = None):
    """Load GPT2 through :mod:`src.model_utils`."""
    from src.model_utils import load_model

    return load_model(name_or_path, device=device)


def load_artifacts(settings: Dict[str, Any], model=None, verbose: bool = True):
    """Resolve the probe direction and the extracted toxic vectors."""
    from src.interventions import resolve_probe_direction
    from src.toxic_vectors import load_toxic_vectors, toxic_vectors_exist

    direction = None
    try:
        direction = resolve_probe_direction(probe_path=settings["probe_path"])
    except Exception as exc:  # pragma: no cover - depends on artifacts
        if verbose:
            print(f"[warn] could not load W_Toxic ({exc}); using W_Toxic spec only")

    toxic_vectors = None
    path = settings["vectors_path"]
    if toxic_vectors_exist(path) or os.path.isfile(path):
        try:
            toxic_vectors = load_toxic_vectors(path)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[warn] could not load toxic vectors: {exc}")
    elif model is not None:
        try:
            from src.toxic_vectors import extract_toxic_vectors

            if verbose:
                print("[info] toxic vectors missing -> extracting from the probe")
            toxic_vectors = extract_toxic_vectors(
                model=model, probe_path=settings["probe_path"], top_n=128
            )
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[warn] toxic-vector extraction failed: {exc}")

    return direction, toxic_vectors


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def run_interventions(args: argparse.Namespace) -> int:
    """Run the Table 2 intervention suite and persist the results."""
    from src.interventions import (
        check_table2,
        default_path,
        evaluate_intervention,
        plot_intervention_results,
        run_interventions as run_suite,
        save_results,
        select_alpha_for_ppl,
    )

    settings = resolve_settings(args)
    verbose = settings["verbose"]
    t0 = time.time()

    if verbose:
        print("=" * 78)
        print("Residual-stream subtraction interventions (Section 3.3, Table 2)")
        print("=" * 78)
        print(f"  model          : {settings['model_name']}")
        print(f"  probe          : {settings['probe_path']}")
        print(f"  toxic vectors  : {settings['vectors_path']}")
        print(f"  position       : {settings['position']}")
        print(f"  output dir     : {settings['out_dir']}")
        print(f"  quick          : {settings['quick']}")
        print("=" * 78)

    model, tokenizer = load_model_safe(settings["model_name"], settings["device"])
    direction, toxic_vectors = load_artifacts(settings, model=model, verbose=verbose)

    specs = build_specs(settings.get("specs"), settings)
    if settings.get("layer") is not None:
        for spec in specs:
            if getattr(spec, "layer", None) is None:
                spec.layer = settings["layer"]

    if verbose:
        print("\n[phase] intervention suite")
        for spec in specs:
            print(f"    - {spec.resolved_label() if hasattr(spec, 'resolved_label') else spec}")

    results = run_suite(
        model,
        tokenizer,
        toxic_vectors=toxic_vectors,
        probe_path=settings["probe_path"],
        specs=specs,
        model_name=settings["model_name"],
        select_alpha=settings["select_alpha"],
        target_ppl=settings["target_ppl"],
        alpha_grid=settings["alphas"],
        corpus=None,
        n_prompts=settings["n_prompts"],
        max_new_tokens=settings["max_new_tokens"],
        batch_size=settings["batch_size"],
        seed=settings["seed"],
        device=settings["device"],
        seq_len=settings["seq_len"],
        stride=settings["stride"],
        score_toxicity=settings["score_toxicity"],
        score_perplexity=settings["score_perplexity"],
        score_f1=settings["score_f1"],
        verbose=verbose,
    )

    report: Dict[str, Any] = {
        "results": results.to_dict() if hasattr(results, "to_dict") else {},
        "table2_check": {},
        "artifacts": {},
        "elapsed_sec": time.time() - t0,
        "settings": {k: v for k, v in settings.items() if k != "alphas"},
        "model": settings["model_name"],
        "substitutions": {
            "toxicity_scorer": "unitary/unbiased-toxic-roberta (Perspective API substitute)"
        },
    }

    try:
        reference = {
            "no_op": TABLE2_REFERENCE["no_op"],
            "w_toxic": TABLE2_REFERENCE["w_toxic"],
            "mlp_v_770_19": TABLE2_REFERENCE["mlp_v_770_19"],
            "svd_u_toxic_0": TABLE2_REFERENCE["svd_u_toxic_0"],
        }
        report["table2_check"] = check_table2(results, reference=reference)
    except Exception as exc:  # pragma: no cover
        report["table2_check"] = {"error": str(exc)}

    if settings["save"]:
        out_path = default_path(settings["out_dir"])
        try:
            saved = save_results(out_path, results, write_markdown=True)
            report["artifacts"]["results_json"] = saved
            if verbose:
                print(f"[save] {saved}")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] could not save results: {exc}")

        try:
            fig_path = os.path.join(settings["out_dir"], "interventions_table2.png")
            plotted = plot_intervention_results(results, out_path=fig_path)
            if plotted:
                report["artifacts"]["figure"] = plotted
                if verbose:
                    print(f"[save] {plotted}")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] could not plot results: {exc}")

        try:
            summary_path = os.path.join(settings["out_dir"], "interventions_summary.json")
            os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, default=_json_default)
            report["artifacts"]["summary_json"] = summary_path
            if verbose:
                print(f"[save] {summary_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] could not save summary: {exc}")

    if settings.get("json_out"):
        with open(settings["json_out"], "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=_json_default)

    _print_table(results, report)
    if verbose:
        print(f"\n[done] {time.time() - t0:.1f}s")
    return 0


def run_table3(args: argparse.Namespace) -> int:
    """Reproduce the Table 3 top-k / continuation examples."""
    from src.interventions import default_specs, table3_examples

    settings = resolve_settings(args)
    model, tokenizer = load_model_safe(settings["model_name"], settings["device"])
    _, toxic_vectors = load_artifacts(settings, model=model, verbose=settings["verbose"])
    specs = build_specs(settings.get("specs"), settings) or default_specs()

    examples = table3_examples(
        model,
        tokenizer,
        toxic_vectors=toxic_vectors,
        probe_path=settings["probe_path"],
        specs=specs,
        k=5,
        max_new_tokens=12 if settings["quick"] else 20,
        alpha=1.0,
        device=settings["device"],
    )

    print("\nTable 3 examples (top-k and continuations)")
    print("=" * 78)
    for row in examples:
        print(f"\nprompt: {row.get('prompt')!r}")
        for entry in row.get("models", []):
            topk = ", ".join(entry.get("top_k", []) or [])
            print(f"  {entry.get('label', entry.get('model')):<18} top-k: {topk}")
            print(f"  {'':<18} cont. : {entry.get('continuation', '')!r}")

    if settings["save"]:
        path = os.path.join(settings["out_dir"], "table3_examples.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(examples, handle, indent=2, default=_json_default)
        print(f"\n[save] {path}")
    return 0


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy / torch scalars."""
    try:
        import numpy as np

        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
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
    return str(obj)


def _print_table(results: Any, report: Dict[str, Any]) -> None:
    """Print the Table-2 style summary and the claim check."""
    print("\nTable 2 — interventions on GPT2-medium")
    print("-" * 78)
    print(f"{'intervention':<26}{'alpha':>8}{'toxicity':>12}{'ppl':>10}{'F1':>10}")
    print("-" * 78)

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    rows = getattr(results, "results", results)
    try:
        iterator = list(rows)
    except TypeError:
        iterator = []
    for row in iterator:
        label = getattr(row, "label", "") or ""
        alpha = fmt(getattr(row, "alpha", None))
        ppl = getattr(row, "perplexity", None)
        ppl_str = "-" if ppl is None else f"{float(ppl):.2f}"
        print(
            f"{label:<26}{alpha:>8}{fmt(getattr(row, 'toxicity', None)):>12}"
            f"{ppl_str:>10}{fmt(getattr(row, 'f1', None)):>10}"
        )
    print("-" * 78)
    print("reference (paper):  no-op 0.453/21.70/0.193 | W_Toxic 0.245/23.56/0.193 | "
          "MLP.v_770^19 0.305/23.30/0.192 | SVD.U_Toxic[0] 0.268/23.48/0.193")

    check = report.get("table2_check") or {}
    if check:
        passed = check.get("passed")
        if passed is not None:
            print(f"\n[check] within tolerance for all rows: {passed}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Residual-stream subtraction interventions (Section 3.3, Table 2/3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="optional YAML config path")
    parser.add_argument("--model", default=None, help="model name or local path")
    parser.add_argument("--probe", default=None, help="path to the trained W_Toxic probe")
    parser.add_argument("--vectors", default=None, help="path to toxic_vectors.pt")
    parser.add_argument("--out-dir", default=None, help="artifact directory")
    parser.add_argument("--layer", type=int, default=None,
                        help="layer at which the subtraction is applied (default: last)")
    parser.add_argument("--position", default="mid", choices=["mid", "out"],
                        help="'mid' = after attention before MLP, 'out' = block output")
    parser.add_argument("--specs", nargs="*", default=None,
                        help=f"subset of interventions, e.g. {' '.join(sorted(set(SPEC_ALIASES))[:4])}")
    parser.add_argument("--target-ppl", type=float, default=None,
                        help="post-DPO perplexity used to select alpha (default 23.34)")
    parser.add_argument("--alphas", default=None,
                        help="comma-separated alpha grid, e.g. '1,2,4,8,12'")
    parser.add_argument("--no-alpha-search", action="store_true",
                        help="use alpha=1 instead of matching post-DPO perplexity")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="number of RealToxicityPrompts challenge prompts")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="greedy continuation length")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None, help="Wikitext-2 window length")
    parser.add_argument("--stride", type=int, default=None, help="Wikitext-2 window stride")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--no-toxicity", action="store_true", help="skip toxicity scoring")
    parser.add_argument("--no-perplexity", action="store_true", help="skip perplexity")
    parser.add_argument("--no-f1", action="store_true", help="skip F1")
    parser.add_argument("--perplexity", action="store_true",
                        help="also score perplexity in --quick mode")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    parser.add_argument("--json-out", default=None, help="optional summary JSON path")
    parser.add_argument("--table3", action="store_true",
                        help="also reproduce the Table 3 qualitative examples")
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument("--quiet", action="store_true", help="reduce logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    try:
        code = run_interventions(args)
        if getattr(args, "table3", False):
            code = max(code, run_table3(args))
        return code
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[aborted]")
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
