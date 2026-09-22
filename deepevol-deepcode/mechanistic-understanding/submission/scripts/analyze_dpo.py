#!/usr/bin/env python
"""Mechanistic analysis entry point for the DPO/toxicity reproduction (GPT2-medium).

This script orchestrates the analyses that reproduce the paper's Figures 1-5 and the
underlying quantities of Tables 2-4:

* ``logit_lens``      -> Figure 1 (Section 4.2): average ``P("sh*t")`` read out of every
                         intermittent layer (``l-mid`` = after attention, before MLP) for
                         295 RealToxicityPrompts prompts, GPT2 vs GPT2_DPO.
* ``activations``     -> Figure 2 (Section 5.2, Eq. 1): mean activations
                         ``m_i = sigma(x^l . MLP.k_i^l)`` of the top MLP.v_Toxic vectors
                         over 1,199 RealToxicityPrompts (20 greedy tokens), before/after DPO,
                         plus activation-region membership ``gamma(MLP.k_i^l)``.
* ``parameter_diff``  -> Section 5.1 / Appendix C-D: every parameter in GPT2_DPO has
                         cosine similarity > 0.99 and mean norm difference < 1e-5 versus GPT2
                         (unembedding exception < 1e-3); toxic MLP vectors survive DPO.
* ``residual_shift``  -> Figures 3-5 (Section 5.1, Eq. 2): ``delta_x = x_DPO - x_GPT2`` at
                         ``l-mid``, PCA projection of the residual streams (Fig. 4),
                         ``cos(delta_x^19-mid, delta_MLP.v_i^j)`` for all ``j < l`` with the
                         mean-activation overlay (Fig. 5, Appendix D layer grid).
* ``summary``         -> writes a machine readable digest of all mechanistic checks.

All heavy imports are performed lazily inside the phase functions so that
``python scripts/analyze_dpo.py --help`` never needs a GPU or the network.

Usage
-----
    python scripts/analyze_dpo.py --phase all
    python scripts/analyze_dpo.py --phase logit_lens --quick
    python scripts/analyze_dpo.py --config configs/default.yaml --phase residual_shift
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Paths / defaults
# --------------------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ARTIFACT_DIR = "artifacts"
ANALYSIS_DIR = os.path.join(ARTIFACT_DIR, "analysis")
FIGURE_DIR = os.path.join(ARTIFACT_DIR, "figures")

# Artifacts produced by the earlier phases of the pipeline.
DEFAULT_CONFIG = os.path.join("configs", "default.yaml")
DEFAULT_DPO_DIR = os.path.join(ARTIFACT_DIR, "gpt2_dpo")
DEFAULT_PROBE_PATH = os.path.join(ARTIFACT_DIR, "probe", "W_toxic.pt")
DEFAULT_VECTORS_PATH = os.path.join(ARTIFACT_DIR, "toxic_vectors", "toxic_vectors.pt")

# Paper constants.
TARGET_VECTOR: Tuple[int, int] = (19, 770)  # MLP.v_770^19, one of the most toxic vectors
DEFAULT_LAYER = 19
N_CHALLENGE_PROMPTS = 1199
N_SHIT_PROMPTS = 295
N_TOP_VECTORS_FIG2 = 5  # Figure 2 shows 5 examples of the top MLP.v_Toxic vectors
DEFAULT_N_TOKENS = 20  # 20 greedily generated tokens (Section 5.2)

PHASES = (
    "logit_lens",
    "activations",
    "parameter_diff",
    "residual_shift",
    "summary",
    "all",
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _ensure_dir(path: str) -> str:
    """Create ``path`` (a directory) if needed and return it."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _save_json(payload: Any, path: str) -> str:
    """Persist a JSON-serialisable payload, creating parent directories."""
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy/torch scalars, arrays and tuples."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except Exception:  # pragma: no cover - numpy always present in practice
        pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:  # pragma: no cover
        pass
    return str(obj)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config, returning ``{}`` when unavailable (analysis is config-optional)."""
    cfg_path = path or DEFAULT_CONFIG
    if cfg_path and os.path.exists(cfg_path):
        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
        except Exception as exc:  # pragma: no cover - config is optional
            print(f"[analyze_dpo] could not read config {cfg_path}: {exc}")
    return {}


def _cfg_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Look up the first present key (supports dotted paths) in the config dict."""
    for key in keys:
        node: Any = cfg
        ok = True
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and node is not None:
            return node
    return default


def resolve_paths(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, str]:
    """Resolve model/artifact paths from CLI args, falling back to config and defaults."""
    paths = {
        "model": args.model
        or _cfg_get(cfg, "models.base", "model.name", "model", default="openai-community/gpt2-medium"),
        "dpo_model": args.dpo_model
        or _cfg_get(cfg, "models.dpo", "dpo.output_dir", "paths.dpo_model", default=DEFAULT_DPO_DIR),
        "probe": args.probe
        or _cfg_get(cfg, "paths.probe", "probe.output_path", default=DEFAULT_PROBE_PATH),
        "vectors": args.vectors
        or _cfg_get(cfg, "paths.vectors", "toxic_vectors.output_path", default=DEFAULT_VECTORS_PATH),
        "analysis_dir": args.out_dir
        or _cfg_get(cfg, "paths.analysis_dir", default=ANALYSIS_DIR),
        "figure_dir": args.figure_dir
        or _cfg_get(cfg, "paths.figure_dir", default=FIGURE_DIR),
    }
    return paths


def load_model_safe(name_or_path: str, device: Optional[str] = None):
    """Load a causal LM + tokenizer via :mod:`src.model_utils` (CPU-friendly float32)."""
    from src.model_utils import load_model

    return load_model(name_or_path, device=device)


def load_toxic_artifact(path: str) -> Dict[str, Any]:
    """Tolerantly load the toxic-vector artifact written by ``scripts/extract_toxic_vectors.py``.

    Accepts ``.pt``/``.pth``/``.bin`` (torch) or ``.json`` payloads and normalises the
    various plausible key spellings to ``{"indices", "key_vectors", "value_vectors"}``.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"toxic vector artifact not found: {path}")

    if path.endswith((".pt", ".pth", ".bin")):
        import torch

        payload = torch.load(path, map_location="cpu")
    else:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

    if not isinstance(payload, dict):
        # Bare tensor/first element fallback.
        payload = {"value_vectors": payload}

    def pick(*keys: str) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    indices = pick(
        "indices",
        "value_indices",
        "MLP.v_Toxic_indices",
        "mlp_v_toxic_indices",
        "top_indices",
        "value_vector_indices",
    )
    key_indices = pick("key_indices", "MLP.k_Toxic_indices", "mlp_k_toxic_indices") or indices
    value_vectors = pick(
        "value_vectors", "MLP.v_Toxic", "v_toxic", "values", "mlp_v_toxic"
    )
    key_vectors = pick("key_vectors", "MLP.k_Toxic", "k_toxic", "keys", "mlp_k_toxic")
    svd_u = pick("svd_u", "SVD.U_Toxic", "u_toxic", "U")

    # Indices may be stored as tensors / lists of [layer, idx] pairs.
    if indices is not None and not isinstance(indices, list):
        try:
            indices = [tuple(int(v) for v in pair) for pair in indices]
        except Exception:
            indices = list(indices)
    if indices is not None:
        indices = [tuple(int(v) for v in pair) for pair in indices]
    if key_indices is not None:
        key_indices = [tuple(int(v) for v in pair) for pair in key_indices]

    return {
        "indices": indices,
        "key_indices": key_indices,
        "value_vectors": value_vectors,
        "key_vectors": key_vectors,
        "svd_u": svd_u,
        "raw": payload,
    }


def top_indices(indices: Optional[Sequence[Tuple[int, int]]], k: int) -> List[Tuple[int, int]]:
    """Return the first ``k`` ``(layer, idx)`` pairs (the artifact is stored by ranking)."""
    if not indices:
        return []
    out: List[Tuple[int, int]] = []
    for pair in indices[:k]:
        out.append((int(pair[0]), int(pair[1])))
    return out


def get_key_vectors(model, indices: Sequence[Tuple[int, int]]):
    """Fetch the MLP key vectors for ``(layer, idx)`` pairs as a stacked tensor ``[n, d]``."""
    import torch

    from src.model_utils import get_key_vector

    if not indices:
        return torch.zeros(0), []
    vecs = [get_key_vector(model, int(layer), int(idx)).detach().float().cpu() for layer, idx in indices]
    return torch.stack(vecs, dim=0), [(int(l), int(i)) for l, i in indices]


# --------------------------------------------------------------------------------------
# Phase 1: logit lens (Figure 1)
# --------------------------------------------------------------------------------------


def run_logit_lens(args: argparse.Namespace, paths: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Reproduce Figure 1: average ``P("sh*t")`` per intermittent layer, GPT2 vs GPT2_DPO."""
    from data.realtoxicity import load_target_token_prompts, prompt_texts
    from src.analysis import logit_lens as ll

    print("[analyze_dpo] phase: logit_lens (Figure 1)")
    model, tokenizer = load_model_safe(paths["model"], device=args.device)

    n_prompts = N_SHIT_PROMPTS if not args.quick else min(args.max_prompts, N_SHIT_PROMPTS)
    prompts = load_target_token_prompts(
        model, tokenizer, cache_dir=args.cache_dir, n=n_prompts, device=args.device
    )
    texts = prompt_texts(prompts)
    if not texts:
        raise RuntimeError("no 'sh*t'-eliciting prompts available")
    print(f"[analyze_dpo]   {len(texts)} 'sh*t'-eliciting prompts")

    _ensure_dir(paths["figure_dir"])
    analysis = ll.analyze_logit_lens(
        model,
        paths["dpo_model"] if os.path.exists(paths["dpo_model"]) else None,
        tokenizer,
        prompts=texts,
        before_name="gpt2",
        after_name="gpt2_dpo",
        out_dir=paths["analysis_dir"],
        figure_name="figure1_logit_lens.png",
        top_k=args.top_k_shade,
        save=True,
        plot=True,
        device=args.device,
        verbose=args.verbose,
    )

    # Move the figure into the figures directory as well for convenience.
    src_fig = os.path.join(paths["analysis_dir"], "figure1_logit_lens.png")
    dst_fig = os.path.join(paths["figure_dir"], "figure1_logit_lens.png")
    if os.path.exists(src_fig) and os.path.abspath(src_fig) != os.path.abspath(dst_fig):
        try:
            import shutil

            shutil.copyfile(src_fig, dst_fig)
        except Exception as exc:  # pragma: no cover
            print(f"[analyze_dpo]   could not copy figure: {exc}")

    return {
        "phase": "logit_lens",
        "n_prompts": len(texts),
        "comparison": analysis.get("comparison") if isinstance(analysis, dict) else None,
        "figure": dst_fig if os.path.exists(dst_fig) else src_fig,
    }


def _logit_lens_after_model(tokenizer, dpo_path: Optional[str], device: Optional[str]):
    """Load GPT2_DPO if available, else return ``None`` (analysis degrades gracefully)."""
    if not dpo_path or not os.path.exists(dpo_path):
        return None
    try:
        model, _ = load_model_safe(dpo_path, device=device)
        return model
    except Exception as exc:
        print(f"[analyze_dpo]   GPT2_DPO unavailable ({exc}); base-model-only analysis")
        return None


# --------------------------------------------------------------------------------------
# Phase 2: mean activations + activation regions (Figure 2)
# --------------------------------------------------------------------------------------


def run_activations(args: argparse.Namespace, paths: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Figure 2 / Eq. 1: mean activations ``m_i`` of top MLP.v_Toxic vectors before/after DPO."""
    from data.realtoxicity import challenge_prompts, prompt_texts
    from src.analysis import activations as act

    print("[analyze_dpo] phase: activations (Figure 2)")
    model, tokenizer = load_model_safe(paths["model"], device=args.device)

    # Determine the top toxic vectors (artifact if present, else the paper's defaults).
    indices: List[Tuple[int, int]] = [TARGET_VECTOR]
    try:
        art = load_toxic_artifact(paths["vectors"])
        if art["indices"]:
            indices = top_indices(art["indices"], args.top_k_vectors)
    except FileNotFoundError:
        print(f"[analyze_dpo]   {paths['vectors']} missing; using default target vector {TARGET_VECTOR}")
    if args.quick:
        indices = indices[: min(len(indices), 2)]

    n_prompts = N_CHALLENGE_PROMPTS if not args.quick else min(args.max_prompts, N_CHALLENGE_PROMPTS)
    prompts = challenge_prompts(cache_dir=args.cache_dir)
    texts = prompt_texts(prompts)[:n_prompts]
    print(f"[analyze_dpo]   {len(texts)} RealToxicityPrompts; vectors={indices}")

    key_vectors, used_indices = get_key_vectors(model, indices)
    if len(used_indices) == 0:
        return {"phase": "activations", "skipped": "no toxic vectors"}

    result_before = act.collect_mean_activations(
        model,
        tokenizer,
        texts,
        key_vectors,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
        device=args.device,
        positions=args.positions,
        model_name="gpt2",
        verbose=args.verbose,
    )
    # Record the (layer, idx) labels that match the stacked key vectors.
    result_before.indices = used_indices

    result_after = None
    dpo_model = _logit_lens_after_model(tokenizer, paths["dpo_model"], args.device)
    if dpo_model is not None:
        key_vectors_dpo, _ = get_key_vectors(dpo_model, used_indices)
        result_after = act.collect_mean_activations(
            dpo_model,
            tokenizer,
            texts,
            key_vectors_dpo,
            n_tokens=args.n_tokens,
            batch_size=args.batch_size,
            device=args.device,
            positions=args.positions,
            model_name="gpt2_dpo",
            verbose=args.verbose,
        )
        result_after.indices = used_indices

    comparison = (
        act.compare_mean_activations(result_before, result_after)
        if result_after is not None
        else None
    )

    # Activation-region (Eq. 1) strength for the analyzed toxic vector.
    region_stats = _activation_region_stats(model, tokenizer, texts, used_indices, args)

    _ensure_dir(paths["analysis_dir"])
    _save_json(
        {
            "indices": used_indices,
            "mean_before": result_before.mean.tolist(),
            "comparison": comparison,
            "activation_region": region_stats,
        },
        os.path.join(paths["analysis_dir"], "mean_activations.json"),
    )

    figure_path = None
    if result_after is not None:
        figure_path = os.path.join(paths["figure_dir"], "figure2_mean_activations.png")
        try:
            act.plot_mean_activations(
                [result_before, result_after],
                labels=["GPT2", "GPT2_DPO"],
                indices=used_indices,
                out_path=figure_path,
            )
        except Exception as exc:  # pragma: no cover - plotting optional
            print(f"[analyze_dpo]   figure 2 failed: {exc}")
            figure_path = None

    return {
        "phase": "activations",
        "n_prompts": len(texts),
        "indices": used_indices,
        "comparison": comparison,
        "activation_region": region_stats,
        "figure": figure_path,
    }


def _activation_region_stats(model, tokenizer, texts, indices, args) -> Dict[str, Any]:
    """Fraction of residual states inside ``gamma(MLP.k_i^l)`` for each analyzed vector.

    Uses the ``l-mid`` residual states (after attention, before MLP) as the vectors ``g``
    tested against the key vectors, following Eq. 1: ``sigma(k_i^l . g) > 0``.
    """
    try:
        import numpy as np
        import torch

        from src.analysis.activations import activation_region_fraction
        from src.model_utils import capture_residual_streams, get_key_vector

        # Collect a modest number of ``l-mid`` states for the layer of each vector.
        layers = sorted({int(l) for l, _ in indices})
        per_layer: Dict[int, List[float]] = {l: [] for l in layers}
        n_texts = min(len(texts), args.region_prompts)
        max_length = 64

        for layer in layers:
            states: List[torch.Tensor] = []
            for text in texts[:n_texts]:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                with torch.no_grad(), capture_residual_streams(model) as cap:
                    model(**enc)
                    mid = cap.get_mid(layer)
                if mid is None:
                    continue
                states.append(mid.detach().reshape(-1, mid.shape[-1]))
            if not states:
                continue
            g = torch.cat(states, dim=0).float()
            for l, idx in indices:
                if int(l) != int(layer):
                    continue
                k = get_key_vector(model, int(l), int(idx)).detach().float().cpu()
                frac = float(activation_region_fraction(k, g))
                per_layer[layer].append(frac)

        return {
            f"layer_{l}": {"fractions": v, "mean": float(np.mean(v)) if v else None}
            for l, v in per_layer.items()
        }
    except Exception as exc:  # pragma: no cover - analysis is best-effort
        return {"error": str(exc)}


# --------------------------------------------------------------------------------------
# Phase 3: parameter deltas (Section 5.1 / Appendix C-D)
# --------------------------------------------------------------------------------------


def run_parameter_diff(args: argparse.Namespace, paths: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Section 5.1: ``delta_theta`` cosine similarities and norm differences."""
    from src.analysis import parameter_diff as pdiff

    print("[analyze_dpo] phase: parameter_diff (Section 5.1, Appendix C/D)")
    if not os.path.exists(paths["dpo_model"]):
        raise FileNotFoundError(
            f"GPT2_DPO not found at {paths['dpo_model']}; run scripts/train_dpo.py first"
        )

    model_before, _ = load_model_safe(paths["model"], device=args.device)
    model_after, _ = load_model_safe(paths["dpo_model"], device=args.device)

    value_indices: List[Tuple[int, int]] = [TARGET_VECTOR]
    try:
        art = load_toxic_artifact(paths["vectors"])
        if art["indices"]:
            value_indices = top_indices(art["indices"], args.top_k_vectors)
    except FileNotFoundError:
        pass

    out = pdiff.analyze_parameter_diff(
        model_before,
        model_after,
        before_name="gpt2",
        after_name="gpt2_dpo",
        out_dir=paths["analysis_dir"],
        value_indices=value_indices,
        save=True,
        verbose=args.verbose,
    )

    # Shape the claims into a compact JSON summary (avoids re-serialising the full report).
    claims = out.get("claims", {}) if isinstance(out, dict) else {}
    summary = {
        "phase": "parameter_diff",
        "claims_passed": claims.get("passed"),
        "cosine": claims.get("cosine"),
        "norm_diff": claims.get("norm_diff"),
        "unembedding": claims.get("unembedding"),
        "toxic_vectors": out.get("toxic_vectors") if isinstance(out, dict) else None,
        "report_path": os.path.join(paths["analysis_dir"], "parameter_diff.json"),
    }

    fig_path = os.path.join(paths["figure_dir"], "figure_parameter_diff.png")
    try:
        report = out.get("report") if isinstance(out, dict) else None
        if report is not None:
            _ensure_dir(paths["figure_dir"])
            pdiff.plot_parameter_cosines(report, out_path=fig_path)
    except Exception as exc:  # pragma: no cover
        print(f"[analyze_dpo]   parameter-diff figure failed: {exc}")
        fig_path = None
    summary["figure"] = fig_path
    return summary


# --------------------------------------------------------------------------------------
# Phase 4: residual-stream shift (Figures 3-5, Eq. 2)
# --------------------------------------------------------------------------------------


def run_residual_shift(args: argparse.Namespace, paths: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Figures 3-5: ``delta_x``, PCA projection, ``cos(delta_x, delta_MLP.v)`` histograms."""
    from data.realtoxicity import challenge_prompts, prompt_texts
    from src.analysis import residual_shift as rs

    print("[analyze_dpo] phase: residual_shift (Figures 3-5)")
    if not os.path.exists(paths["dpo_model"]):
        raise FileNotFoundError(
            f"GPT2_DPO not found at {paths['dpo_model']}; run scripts/train_dpo.py first"
        )

    model_before, tokenizer = load_model_safe(paths["model"], device=args.device)
    model_after, _ = load_model_safe(paths["dpo_model"], device=args.device)

    n_prompts = N_CHALLENGE_PROMPTS if not args.quick else min(args.max_prompts, N_CHALLENGE_PROMPTS)
    texts = prompt_texts(challenge_prompts(cache_dir=args.cache_dir))[:n_prompts]
    print(f"[analyze_dpo]   {len(texts)} prompts, layer {args.layer}, {args.n_tokens} tokens/prompt")

    # Collect residual streams for both models at ``layer`` (default 19).
    before = rs.collect_residual_streams(
        model_before,
        tokenizer,
        texts,
        layer=args.layer,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
        positions=args.positions,
        device=args.device,
        model_name="gpt2",
        verbose=args.verbose,
    )
    after = rs.collect_residual_streams(
        model_after,
        tokenizer,
        texts,
        layer=args.layer,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
        positions=args.positions,
        device=args.device,
        model_name="gpt2_dpo",
        verbose=args.verbose,
    )

    shift = rs.compute_residual_shift(before, after, layer=args.layer, target_index=TARGET_VECTOR)

    # Eq. 2 cosine similarities vs. the shifts of the MLP value vectors in earlier layers.
    value_deltas = rs.parameter_value_deltas(model_before, model_after)
    cosine_result = rs.parameter_delta_cosine_against_shift(
        shift.delta if shift.delta is not None else shift.mean, value_deltas, layer=args.layer
    )

    # Mean activations of each value vector during the forward pass (Figure 5 orange).
    activations = {}
    try:
        activations = rs.mean_activations_by_layer(
            model_before,
            tokenizer,
            texts,
            layers=list(range(max(0, args.layer - 4), args.layer)),
            n_tokens=args.n_tokens,
            batch_size=args.batch_size,
            device=args.device,
            positions=args.positions,
            verbose=args.verbose,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[analyze_dpo]   activation overlay failed: {exc}")

    _ensure_dir(paths["analysis_dir"])
    out_paths = rs.default_paths(layer=args.layer, prefix="gpt2")
    try:
        paths_saved = out_paths if isinstance(out_paths, dict) else {}
        shift_path = paths_saved.get("shift") or os.path.join(paths["analysis_dir"], f"shift_{args.layer}.json")
        rs.save_shift_result(shift_path, shift, save_arrays=True)
        param_path = paths_saved.get("parameter_shift") or os.path.join(
            paths["analysis_dir"], f"parameter_shift_{args.layer}.json"
        )
        rs.save_parameter_shift(param_path, cosine_result)
    except Exception as exc:  # pragma: no cover
        print(f"[analyze_dpo]   persisting shift results failed: {exc}")

    # Figures 3, 4, 5.
    _ensure_dir(paths["figure_dir"])
    figures: Dict[str, Optional[str]] = {}
    try:
        figures["figure3_residual_shift"] = rs.plot_residual_shift(
            shift, out_path=os.path.join(paths["figure_dir"], "figure3_residual_shift.png"), layer=args.layer
        )
    except Exception as exc:  # pragma: no cover
        print(f"[analyze_dpo]   figure 3 failed: {exc}")
    try:
        projection = rs.project_residual_streams(shift)
        figures["figure4_projection"] = rs.plot_projection(
            projection, out_path=os.path.join(paths["figure_dir"], "figure4_projection.png")
        )
    except Exception as exc:  # pragma: no cover
        print(f"[analyze_dpo]   figure 4 failed: {exc}")
    try:
        figures["figure5_cosine_activation"] = rs.plot_delta_cosine_histograms(
            cosine_result,
            activations or None,
            out_path=os.path.join(paths["figure_dir"], "figure5_delta_cosine.png"),
        )
    except Exception as exc:  # pragma: no cover
        print(f"[analyze_dpo]   figure 5 failed: {exc}")

    summary = {
        "phase": "residual_shift",
        "layer": args.layer,
        "n_prompts": len(texts),
        "consistency": rs.shift_consistency(shift),
        "mean_shift_norm": shift.mean_norm,
        "fraction_negative_cosine": cosine_result.fraction_negative(),
        "mean_cosine_per_layer": cosine_result.mean_per_layer(),
        "figures": figures,
    }
    _save_json(summary, os.path.join(paths["analysis_dir"], "residual_shift_summary.json"))
    return summary


# --------------------------------------------------------------------------------------
# Phase 5: summary of mechanistic checks
# --------------------------------------------------------------------------------------


def run_summary(
    args: argparse.Namespace,
    paths: Dict[str, str],
    cfg: Dict[str, Any],
    results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate the per-phase results (and any pre-existing JSON artifacts) into a digest."""
    print("[analyze_dpo] phase: summary")
    digest: Dict[str, Any] = {"results": results or {}, "artifacts": {}}
    for name in (
        "mean_activations.json",
        "parameter_diff.json",
        "parameter_diff_claims.json",
        "residual_shift_summary.json",
        "logit_lens_comparison.json",
    ):
        path = os.path.join(paths["analysis_dir"], name)
        digest["artifacts"][name] = os.path.exists(path)

    shift_summary = None
    path = os.path.join(paths["analysis_dir"], "residual_shift_summary.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                shift_summary = json.load(fh)
        except Exception:
            shift_summary = None
    if shift_summary:
        digest["fraction_negative_cosine"] = shift_summary.get("fraction_negative_cosine")
        digest["mean_shift_norm"] = shift_summary.get("mean_shift_norm")

    _save_json(digest, os.path.join(paths["analysis_dir"], "analysis_summary.json"))
    return digest


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mechanistic analysis of DPO vs GPT2 (Figures 1-5, Appendix C/D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path (optional)")
    parser.add_argument("--phase", default="all", choices=PHASES, help="which analysis phase to run")
    parser.add_argument("--model", default=None, help="base model name or local path (GPT2-medium)")
    parser.add_argument("--dpo-model", dest="dpo_model", default=None, help="GPT2_DPO directory")
    parser.add_argument("--probe", default=None, help="trained W_Toxic artifact path")
    parser.add_argument("--vectors", default=None, help="toxic-vector artifact path")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="analysis artifact directory")
    parser.add_argument("--figure-dir", dest="figure_dir", default=None, help="figure output directory")
    parser.add_argument("--cache-dir", dest="cache_dir", default=None, help="dataset cache directory")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda / cpu")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="analysis layer (paper: 19)")
    parser.add_argument("--n-tokens", dest="n_tokens", type=int, default=DEFAULT_N_TOKENS,
                        help="continuation tokens per prompt (paper: 20)")
    parser.add_argument("--positions", default="first", choices=("first", "last", "all"),
                        help="which token positions to pool")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--top-k-vectors", dest="top_k_vectors", type=int, default=N_TOP_VECTORS_FIG2,
                        help="number of top toxic vectors to analyse")
    parser.add_argument("--top-k-shade", dest="top_k_shade", type=int, default=6,
                        help="number of promoted layers to shade in Figure 1")
    parser.add_argument("--region-prompts", dest="region_prompts", type=int, default=64,
                        help="prompts used for the activation-region check")
    parser.add_argument("--max-prompts", dest="max_prompts", type=int, default=64,
                        help="prompt cap used when --quick is set")
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args)
    _ensure_dir(paths["analysis_dir"])
    _ensure_dir(paths["figure_dir"])

    if args.quick:
        args.n_tokens = min(args.n_tokens, 4)

    print(f"[analyze_dpo] paths: {json.dumps(paths, indent=2, default=_json_default)}")

    planned = (
        ["logit_lens", "activations", "parameter_diff", "residual_shift", "summary"]
        if args.phase == "all"
        else [args.phase]
    )

    results: Dict[str, Any] = {}
    for phase in planned:
        try:
            if phase == "logit_lens":
                results[phase] = run_logit_lens(args, paths, cfg)
            elif phase == "activations":
                results[phase] = run_activations(args, paths, cfg)
            elif phase == "parameter_diff":
                results[phase] = run_parameter_diff(args, paths, cfg)
            elif phase == "residual_shift":
                results[phase] = run_residual_shift(args, paths, cfg)
            elif phase == "summary":
                results[phase] = run_summary(args, paths, cfg, results)
        except FileNotFoundError as exc:
            print(f"[analyze_dpo] phase '{phase}' skipped: {exc}")
            results[phase] = {"skipped": str(exc)}
        except Exception as exc:  # pragma: no cover - keep other phases running
            print(f"[analyze_dpo] phase '{phase}' failed: {exc}")
            if args.verbose:
                traceback.print_exc()
            results[phase] = {"error": str(exc)}

    if args.phase == "all" and "summary" not in results:
        results["summary"] = run_summary(args, paths, cfg, results)

    _save_json(results, os.path.join(paths["analysis_dir"], "analyze_dpo_results.json"))
    print(f"[analyze_dpo] done; results -> {os.path.join(paths['analysis_dir'], 'analyze_dpo_results.json')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
