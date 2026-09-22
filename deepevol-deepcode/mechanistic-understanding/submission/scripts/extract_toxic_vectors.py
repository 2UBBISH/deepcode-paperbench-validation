#!/usr/bin/env python
"""Extract toxic vectors from GPT2-medium (Section 3.1 / 3.2).

This script is the CLI entry point for the second phase of the reproduction:

1.  Load the frozen GPT2-medium language model and its tokenizer.
2.  Load (or train, if missing) the linear toxicity probe ``W_Toxic`` whose
    toxic direction is column 1, ``W_Toxic[:, 1]`` (see ``scripts/train_probe.py``).
3.  Rank **every** MLP value vector of the model by cosine similarity with
    ``W_Toxic[:, 1]`` and keep the top ``N = 128`` as ``MLP.v_Toxic`` together
    with their matching key vectors ``MLP.k_Toxic`` (paper Section 3.1).
4.  Stack the selected value vectors into an ``N x d`` matrix and apply SVD to
    its **transpose** (``d x N``) to obtain the decomposed toxic vectors
    ``SVD.U_Toxic`` (paper Section 3.1: "we stack them into a N x d matrix.
    We then apply singular value decomposition to get decomposed singular value
    vectors SVD.U_Toxic").
5.  Project ``W_Toxic``, the selected ``MLP.v_Toxic`` vectors and the first
    ``SVD.U_Toxic`` vectors onto the vocabulary space to reproduce the toxic
    token groups of Table 1 (paper Section 3.2), and report a validation digest
    of which expected tokens were recovered.
6.  Persist all artifacts (``.pt`` + ``.json``) for the downstream DPO,
    intervention and mechanistic-analysis phases.

Example
-------
    python scripts/extract_toxic_vectors.py --top-n 128
    python scripts/extract_toxic_vectors.py --quick --validate

Llama2 / GLU architectures are out of scope for this reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make the repository root importable when the script is executed directly.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")
DEFAULT_OUT_DIR = os.path.join("artifacts", "vectors")
DEFAULT_PROBE_PATH = os.path.join("artifacts", "probe", "w_toxic.pt")

# Reference vector indices from the paper (Table 1) used for sanity reporting.
PAPER_REFERENCE_VECTORS: Tuple[Tuple[int, int], ...] = (
    (19, 770),
    (12, 771),
    (18, 2669),
    (13, 668),
    (16, 255),
    (12, 882),
    (19, 1438),
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load an optional YAML configuration file (returns ``{}`` on failure)."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml  # local import: config support is optional
    except Exception:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}
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


def _first_not_none(*values: Any) -> Any:
    """Return the first non-``None`` value (used for CLI > config > default)."""
    for value in values:
        if value is not None:
            return value
    return None


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI arguments, YAML config and hard-coded defaults."""
    cfg = _load_yaml(getattr(args, "config", None) or DEFAULT_CONFIG)

    model_name = _first_not_none(
        getattr(args, "model", None),
        _dig(cfg, "model", "name"),
        _dig(cfg, "models", "base"),
        _dig(cfg, "model_name"),
        DEFAULT_MODEL,
    )
    out_dir = _first_not_none(getattr(args, "out_dir", None), DEFAULT_OUT_DIR)
    probe_path = _first_not_none(
        getattr(args, "probe", None),
        _dig(cfg, "paths", "probe"),
        DEFAULT_PROBE_PATH,
    )
    top_n = int(_first_not_none(getattr(args, "top_n", None), _dig(cfg, "toxic_vectors", "top_n"), 128))
    top_k_tokens = int(
        _first_not_none(getattr(args, "top_k_tokens", None), _dig(cfg, "toxic_vectors", "top_k_tokens"), 10)
    )
    quantile = _first_not_none(getattr(args, "quantile", None), _dig(cfg, "toxic_vectors", "quantile"))

    settings: Dict[str, Any] = {
        "model_name": model_name,
        "out_dir": out_dir,
        "probe_path": probe_path,
        "top_n": top_n,
        "top_k_tokens": top_k_tokens,
        "quantile": quantile,
        "device": getattr(args, "device", None),
        "seed": int(_first_not_none(getattr(args, "seed", None), _dig(cfg, "seed"), 0)),
        "layers": None,
        "compute_svd": not bool(getattr(args, "no_svd", False)),
        "center": bool(getattr(args, "center", False)),
        "quick": bool(getattr(args, "quick", False)),
        "validate": not bool(getattr(args, "no_validate", False)),
        "force": bool(getattr(args, "force", False)),
        "save": not bool(getattr(args, "no_save", False)),
        "json_out": getattr(args, "json_out", None),
        "n_components": getattr(args, "n_components", None),
    }

    # Layer restriction, e.g. ``--layers 6,12,19``.
    layers_arg = getattr(args, "layers", None)
    if layers_arg:
        settings["layers"] = [int(x) for x in str(layers_arg).replace(" ", "").split(",") if x]

    if settings["quick"]:
        # Small smoke-test configuration: fewer vectors / fewer tokens, but the
        # full extraction path is exercised.
        settings["top_n"] = min(settings["top_n"], 16)
        settings["top_k_tokens"] = min(settings["top_k_tokens"], 5)

    return settings


# ---------------------------------------------------------------------------
# Model / probe loading
# ---------------------------------------------------------------------------
def load_gpt2(model_name: str, device: Optional[str] = None):
    """Load GPT2 (any size) and its tokenizer via :mod:`src.model_utils`."""
    from src.model_utils import load_model

    return load_model(model_name, device=device, eval_mode=True)


def announce_architecture(model, model_name: str) -> Dict[str, Any]:
    """Print and return architecture constants, warning on non-GPT2 models."""
    from src.model_utils import model_info

    info = model_info(model, name=model_name)
    print(f"[model] {info}")
    if info.n_layers != 24 or info.d_model != 1024 or info.d_mlp != 4096:
        print(
            "[warn] This reproduction targets GPT2-medium "
            "(L=24, d_model=1024, d_mlp=4096)."
        )
    if getattr(info, "is_glu", False):
        print("[warn] GLU architecture detected (Llama2-style): marked OUT OF SCOPE " "by the reproduction plan.")
    return {
        "name": info.name,
        "n_layers": int(info.n_layers),
        "d_model": int(info.d_model),
        "d_mlp": int(info.d_mlp),
        "n_heads": int(info.n_heads),
        "vocab_size": int(info.vocab_size),
        "is_glu": bool(info.is_glu),
    }


def load_or_build_probe(settings: Dict[str, Any], model, tokenizer, verbose: bool = True):
    """Load the saved ``W_Toxic`` probe, training it if it is absent."""
    from src.probe import load_or_train_probe

    probe_path = settings["probe_path"]
    exists = os.path.exists(probe_path)
    if not exists and not verbose:
        return None
    if verbose:
        print(f"[probe] {'loading' if exists else 'training'} probe at {probe_path}")
    probe = load_or_train_probe(
        path=probe_path,
        model=model,
        tokenizer=tokenizer,
        model_name=settings["model_name"],
        force=False,
        verbose=verbose,
        seed=settings["seed"],
    )
    return probe


def probe_summary(probe) -> Dict[str, Any]:
    """Extract a small JSON-serialisable digest of the probe."""
    summary: Dict[str, Any] = {}
    if probe is None:
        return summary
    try:
        summary["d_model"] = int(probe.d_model)
        summary["layer"] = int(getattr(probe, "layer", -1))
        summary["position"] = str(getattr(probe, "position", "block_out"))
        summary["pooling"] = str(getattr(probe, "pooling", "mean"))
        acc = float(getattr(probe, "valid_accuracy", float("nan")))
        if acc == acc:  # not NaN
            summary["valid_accuracy"] = acc
        direction = probe.toxic_direction
        summary["toxic_direction_norm"] = float(direction.norm().item())
    except Exception:  # pragma: no cover - defensive
        pass
    return summary


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def run_extraction(
    args: argparse.Namespace,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the full toxic-vector extraction pipeline."""
    t_start = time.time()

    from src.model_utils import set_seed

    set_seed(settings["seed"])

    model, tokenizer = load_gpt2(settings["model_name"], device=settings["device"])
    architecture = announce_architecture(model, settings["model_name"])

    # --- probe ------------------------------------------------------------
    probe = load_or_build_probe(settings, model, tokenizer, verbose=True)
    probe_info = probe_summary(probe)

    from src.toxic_vectors import (
        DEFAULT_TOP_N,
        TABLE1_EXPECTED,
        TABLE1_VECTORS,
        check_table1_tokens,
        cosine_similarity_to_direction,
        default_paths,
        extract_toxic_vectors,
        promotion_sign,
        resolve_toxic_direction,
        save_toxic_vectors,
        table1_projections,
        toxic_vectors_exist,
    )

    direction = resolve_toxic_direction(
        probe=probe,
        probe_path=settings["probe_path"],
        toxic_index=1,
    )
    direction_norm = None
    try:
        import numpy as _np

        direction_norm = float(_np.linalg.norm(direction))
    except Exception:  # pragma: no cover
        pass
    print(f"[direction] W_Toxic[:, 1] resolved (norm={direction_norm})")

    paths = default_paths(out_dir=settings["out_dir"])
    if not settings["force"] and toxic_vectors_exist(paths.get("pt")):
        from src.toxic_vectors import load_toxic_vectors

        print(f"[skip] existing artifact found at {paths.get('pt')} (use --force to overwrite)")
        toxic_vectors = load_toxic_vectors(path=paths.get("pt"))
    else:
        print(
            f"[extract] ranking MLP value vectors by cosine similarity with "
            f"W_Toxic[:, 1] (top_n={settings['top_n']}, layers={settings['layers']})"
        )
        toxic_vectors = extract_toxic_vectors(
            model=model,
            probe=probe,
            direction=direction,
            tokenizer=tokenizer,
            top_n=settings["top_n"],
            layers=settings["layers"],
            compute_svd=settings["compute_svd"],
            n_components=settings["n_components"],
            center=settings["center"],
            probe_path=settings["probe_path"],
            model_name=settings["model_name"],
            top_k_tokens=settings["top_k_tokens"],
            verbose=True,
        )

    # --- reporting --------------------------------------------------------
    indices: List[Tuple[int, int]] = list(toxic_vectors.indices)
    cosines = [float(c) for c in getattr(toxic_vectors, "cosines", []) or []]
    print(f"[extract] selected {len(indices)} value vectors (N={toxic_vectors.top_n})")
    for rank, (layer, idx) in enumerate(indices[:10]):
        cos = cosines[rank] if rank < len(cosines) else float("nan")
        print(f"    #{rank:02d} MLP.v_{idx}^{layer}  cos={cos:.4f}")

    svd_info: Dict[str, Any] = {}
    svd_vectors = getattr(toxic_vectors, "svd_u", None)
    if svd_vectors is not None:
        try:
            import numpy as _np

            svd_vectors = _np.asarray(svd_vectors)
            svd_info = {
                "n_components": int(svd_vectors.shape[0]),
                "dim": int(svd_vectors.shape[1]),
                "singular_values": [
                    float(v) for v in (getattr(toxic_vectors, "singular_values", None) or [])
                ][:10],
            }
            print(
                f"[svd] SVD.U_Toxic computed on the transposed d x N matrix "
                f"-> {svd_vectors.shape[0]} vectors of dim {svd_vectors.shape[1]}"
            )
        except Exception:  # pragma: no cover - defensive
            pass

    # --- vocabulary-space validation (Table 1) ---------------------------
    projections: Dict[str, Any] = {}
    validation: Dict[str, Any] = {}
    if settings["validate"]:
        try:
            projections = table1_projections(
                model=model,
                toxic_vectors=toxic_vectors,
                tokenizer=tokenizer,
                probe=probe,
                direction=direction,
                top_k=settings["top_k_tokens"],
                max_mlp_vectors=len(PAPER_REFERENCE_VECTORS),
            )
            for label, projection in projections.items():
                tokens = projection.tokens(0) if hasattr(projection, "tokens") else []
                print(f"[vocab] {label:<22} -> {', '.join(str(t) for t in tokens)}")
            validation = check_table1_tokens(projections, expected=TABLE1_EXPECTED, prefix_len=4)
            print(
                f"[validate] Table 1 token groups recovered: "
                f"{validation.get('n_groups_recovered', '?')}/"
                f"{validation.get('n_groups', '?')}"
            )
            for key, value in validation.items():
                if key not in {"groups", "overall"}:
                    continue
                if isinstance(value, dict):
                    print(f"[validate] {key}: {json.dumps(value, default=str)}")
        except Exception as exc:  # pragma: no cover - reporting must not fail the run
            print(f"[warn] vocabulary validation failed: {exc}")
            validation = {"error": repr(exc)}

    # --- reference-vector sanity check -----------------------------------
    reference_report: List[Dict[str, Any]] = []
    try:
        index_lookup = {tuple(map(int, pair)): i for i, pair in enumerate(indices)}
        for layer, idx in PAPER_REFERENCE_VECTORS:
            rank = index_lookup.get((layer, idx))
            entry: Dict[str, Any] = {"layer": layer, "idx": idx, "in_top_n": rank is not None}
            if rank is not None:
                entry["rank"] = int(rank)
                entry["cosine"] = cosines[rank] if rank < len(cosines) else None
                try:
                    vector = toxic_vectors.value(rank)
                    entry["vocab_sign"] = promotion_sign(1.0, vector)
                except Exception:
                    pass
            reference_report.append(entry)
        n_found = sum(1 for e in reference_report if e["in_top_n"])
        print(f"[reference] {n_found}/{len(PAPER_REFERENCE_VECTORS)} paper reference vectors found in top-N")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] reference check failed: {exc}")

    # --- persistence ------------------------------------------------------
    saved_paths: Dict[str, str] = {}
    if settings["save"]:
        try:
            pt_path = save_toxic_vectors(
                toxic_vectors,
                path=paths.get("pt"),
                out_dir=settings["out_dir"],
                save_json=True,
                verbose=True,
            )
            saved_paths["pt"] = str(pt_path)
            candidates = default_paths(out_dir=settings["out_dir"])
            for key, value in candidates.items():
                if os.path.exists(value):
                    saved_paths[key] = str(value)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] failed to save toxic vectors: {exc}")
            traceback.print_exc()

        if projections:
            try:
                proj_path = os.path.join(
                    settings["out_dir"],
                    "table1_projections.json",
                )
                os.makedirs(os.path.dirname(proj_path), exist_ok=True)
                with open(proj_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {label: (p.to_dict() if hasattr(p, "to_dict") else p) for label, p in projections.items()},
                        handle,
                        indent=2,
                        default=str,
                    )
                saved_paths["table1_projections"] = proj_path
                print(f"[save] Table 1 projections -> {proj_path}")
            except Exception as exc:  # pragma: no cover
                print(f"[warn] failed to save Table 1 projections: {exc}")

    elapsed = time.time() - t_start
    summary: Dict[str, Any] = {
        "model": settings["model_name"],
        "architecture": architecture,
        "probe": probe_info,
        "toxic_direction_norm": direction_norm,
        "top_n": int(toxic_vectors.top_n),
        "indices": [[int(l), int(i)] for l, i in indices],
        "cosines": cosines,
        "layer_counts": (
            toxic_vectors.layer_counts() if hasattr(toxic_vectors, "layer_counts") else {}
        ),
        "svd": svd_info,
        "reference_vectors": reference_report,
        "table1_validation": validation,
        "paper_table1_vectors": [[int(l), int(i)] for l, i in TABLE1_VECTORS],
        "artifacts": saved_paths,
        "elapsed_seconds": elapsed,
        "default_top_n": DEFAULT_TOP_N,
    }
    if probe is not None:
        try:
            summary["probe_direction_cosine_with_top"] = None
            summary["probe_direction_cosine_to_vector0"] = cosine_similarity_to_direction(
                toxic_vectors.value(0), direction
            )
        except Exception:
            pass

    if settings["json_out"]:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(settings["json_out"])), exist_ok=True)
            with open(settings["json_out"], "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, default=str)
            print(f"[save] summary -> {settings['json_out']}")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] failed to write summary json: {exc}")

    print(f"[done] toxic vector extraction finished in {elapsed:.1f}s")
    summary["_toxic_vectors"] = toxic_vectors
    summary["_model"] = model
    summary["_tokenizer"] = tokenizer
    summary["_probe"] = probe
    summary["_projections"] = projections
    summary["_direction"] = direction
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract toxic MLP value/key vectors and SVD.U_Toxic from GPT2-medium "
            "(paper Sections 3.1-3.2, Table 1)."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--model", default=None, help=f"Model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--probe", default=None, help=f"Path to W_Toxic probe (default: {DEFAULT_PROBE_PATH}).")
    parser.add_argument("--out-dir", default=None, help=f"Artifact directory (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--top-n", type=int, default=None, help="Number of toxic value vectors N (default: 128).")
    parser.add_argument(
        "--top-k-tokens", type=int, default=None, help="Top vocabulary tokens per vector (default: 10)."
    )
    parser.add_argument("--layers", default=None, help="Optional comma-separated layer restriction, e.g. 6,12,19.")
    parser.add_argument("--n-components", type=int, default=None, help="Number of SVD components (default: all).")
    parser.add_argument("--quantile", type=float, default=None, help="Optional cosine quantile for selection.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda / cpu.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--center", action="store_true", help="Mean-center the value-vector matrix before SVD.")
    parser.add_argument("--no-svd", action="store_true", help="Skip the SVD decomposition step.")
    parser.add_argument("--no-validate", action="store_true", help="Skip the Table 1 vocabulary validation.")
    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts to disk.")
    parser.add_argument("--force", action="store_true", help="Recompute even if artifacts already exist.")
    parser.add_argument("--quick", action="store_true", help="Smoke-test mode (fewer vectors/tokens).")
    parser.add_argument("--json-out", default=None, help="Optional path for a JSON summary of the run.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = resolve_settings(args)
    except Exception as exc:  # pragma: no cover - bad config
        print(f"[error] failed to resolve settings: {exc}")
        traceback.print_exc()
        return 2

    print("=" * 78)
    print("Toxic vector extraction (GPT2-medium) - Sections 3.1 / 3.2")
    print("=" * 78)
    for key, value in settings.items():
        print(f"  {key}: {value}")
    print("-" * 78)

    try:
        run_extraction(args, settings)
    except KeyboardInterrupt:  # pragma: no cover
        print("[abort] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[error] extraction failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
