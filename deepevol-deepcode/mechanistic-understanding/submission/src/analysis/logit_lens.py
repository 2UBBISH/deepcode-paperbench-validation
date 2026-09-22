"""Logit-lens analysis on GPT2 / GPT2_DPO (Section 4.2, Figure 1).

Paper (Section 4.2, Figure 1):

    "Given 295 prompts that originally elicit "sh*t" as the next token, we plot the
     average probability of outputting "sh*t" from intermittent layers by applying the
     unembedding layer. Minor ticks indicate l_mid layers (after attention heads,
     before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all
     correspond to MLP layers."

So for every transformer layer ``l`` we unembed two hidden states:

* ``x^{l-mid}``  -- residual stream after attention heads, before the MLP
* ``x^{l-out}``  -- residual stream after the MLP (block output)

and record the average probability the next token is "sh*t".  The curve is stored
interleaved (``mid_0, out_0, mid_1, out_1, ...``) so it can be plotted exactly like
Figure 1, with minor ticks on the ``l-mid`` slots and grey shading over the MLP
(block-output) layers that promote the toxic token the most in the *base* model.

The module is dependency-light: ``torch``/``numpy`` are required, ``matplotlib`` is
imported lazily through :mod:`src.analysis.plots`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..model_utils import (
    capture_residual_streams,
    model_info,
    resolve_device,
    unembed_hidden_state,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

TARGET_TOKEN = "sh*t"
# Spellings that GPT2's byte-level BPE may use for the same surface form.
TARGET_TOKEN_VARIANTS: Tuple[str, ...] = (
    "sh*t",
    " shit",
    "shit",
    " Shit",
    "Shit",
    "SHIT",
    " sh*t",
)
NORMALISED_TARGETS = ("shit", "sh*t", "sh t")

N_SHIT_PROMPTS = 295
DEFAULT_N_LAYERS = 24
DEFAULT_N_TOKENS = 1
DEFAULT_TOP_K_SHADE = 6
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 128

ARTIFACT_DIR = "artifacts/analysis"
LOGIT_LENS_FILENAME = "logit_lens_{model}.json"

STATE_KINDS = ("mid", "out")


# --------------------------------------------------------------------------------------
# Token id helpers
# --------------------------------------------------------------------------------------


def _normalise_token(token: str) -> str:
    """Normalise a GPT2 vocabulary entry for comparison with "sh*t"/"shit"."""
    text = token.replace("\u0120", " ")  # byte-level space marker -> space
    text = text.strip().lower()
    text = text.strip("'\"`.,!?;:()[]{} ")
    return text


def target_token_ids(
    tokenizer,
    token: str = TARGET_TOKEN,
    variants: Sequence[str] = TARGET_TOKEN_VARIANTS,
    scan_vocab: bool = True,
) -> List[int]:
    """Return every vocabulary id that renders as ``token`` ("sh*t"/"shit").

    Both the direct encodings (``token``, ``" " + token``, capitalisations) and, if
    available, a byte-level vocabulary scan are used so the same id is found
    regardless of the tokenizer wrapper.
    """
    candidates = [token, " " + token, token.strip(), token.capitalize(),
                  " " + token.capitalize()]
    candidates.extend(variants)
    ids: List[int] = []
    for cand in dict.fromkeys(candidates):
        if not cand:
            continue
        try:
            enc = tokenizer.encode(cand, add_special_tokens=False)
        except Exception:  # pragma: no cover - tokenizer specific
            enc = []
        if len(enc) == 1:
            ids.append(int(enc[0]))

    if scan_vocab:
        vocab = getattr(tokenizer, "get_vocab", None)
        if callable(vocab):
            try:
                for tok, idx in vocab().items():
                    if _normalise_token(tok) in NORMALISED_TARGETS:
                        ids.append(int(idx))
            except Exception:  # pragma: no cover - defensive
                pass

    if not ids:
        # Fall back to the last sub-token of the surface form.
        try:
            enc = tokenizer.encode(" " + token, add_special_tokens=False)
            if enc:
                ids.append(int(enc[-1]))
        except Exception:  # pragma: no cover - defensive
            raise ValueError(f"could not resolve token id for {token!r}")

    return sorted(set(ids))


# --------------------------------------------------------------------------------------
# Hidden-state capture helpers
# --------------------------------------------------------------------------------------


def _from_container(container, layer: int):
    if container is None:
        return None
    if isinstance(container, dict):
        if layer in container:
            return container[layer]
        return container.get(str(layer))
    if isinstance(container, (list, tuple)):
        return container[layer] if 0 <= layer < len(container) else None
    if torch.is_tensor(container):
        if container.dim() >= 1 and layer < container.shape[0]:
            return container[layer]
        return None
    return None


def _state_at(capture, kind: str, layer: int, n_layers: int):
    """Fetch the ``l-mid`` or block-output state of ``layer`` from a capture object."""
    getter_name = "get_mid" if kind == "mid" else "get_block_out"
    getter = getattr(capture, getter_name, None)
    if callable(getter):
        try:
            value = getter(layer)
        except Exception:  # pragma: no cover - capture specific
            value = None
        if value is not None:
            return value
    attr_name = "mid" if kind == "mid" else "block_out"
    return _from_container(getattr(capture, attr_name, None), layer)


def collect_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    n_layers: Optional[int] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Run a forward pass and return ``({layer: x^{l-mid}}, {layer: x^{l-out}})``."""
    if n_layers is None:
        n_layers = model_info(model).n_layers
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        with capture_residual_streams(model, capture_mlp_act=False) as capture:
            model(**kwargs)
            mids = {l: _state_at(capture, "mid", l, n_layers) for l in range(n_layers)}
            outs = {l: _state_at(capture, "out", l, n_layers) for l in range(n_layers)}
    missing = [l for l in range(n_layers) if mids.get(l) is None or outs.get(l) is None]
    if missing:
        raise RuntimeError(
            "residual-stream capture did not record layers "
            f"{missing}; check src.model_utils.capture_residual_streams"
        )
    return mids, outs


def unembed_probabilities(
    model,
    hidden: torch.Tensor,
    token_ids: Sequence[int],
    positions: str = "last",
    attention_mask: Optional[torch.Tensor] = None,
    n_tokens: int = 1,
) -> torch.Tensor:
    """Probability mass on ``token_ids`` after applying the unembedding layer.

    ``hidden`` has shape ``[batch, seq, d_model]`` and the returned tensor has shape
    ``[batch]`` (probability that the next token is one of ``token_ids``).
    """
    logits = unembed_hidden_state(model, hidden)
    probs = torch.softmax(logits.float(), dim=-1)
    target = probs[..., list(token_ids)].sum(dim=-1)  # [batch, seq]

    if attention_mask is not None:
        lengths = attention_mask.to(target.device).sum(dim=-1).long().clamp(min=1)
    else:
        lengths = torch.full(
            (target.shape[0],), target.shape[1], dtype=torch.long, device=target.device
        )

    if positions in ("last", "last_1"):
        idx = (lengths - 1).clamp(min=0, max=target.shape[1] - 1)
        return target.gather(1, idx[:, None]).squeeze(1)

    if positions in ("last_n", "first", "first_n", "mean", "all"):
        mask = torch.arange(target.shape[1], device=target.device)[None, :] < lengths[:, None]
        if positions in ("last_n", "first"):
            n = max(int(n_tokens), 1)
            pos = torch.arange(target.shape[1], device=target.device)[None, :]
            if positions == "last_n":
                keep = pos >= (lengths[:, None] - n)
            else:
                keep = pos < n
            mask = mask & keep
        denom = mask.sum(dim=1).clamp(min=1).float()
        return (target * mask).sum(dim=1) / denom

    raise ValueError(f"unknown positions mode: {positions!r}")


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class LogitLensResult:
    """Average next-token probability of the target token at every intermittent layer."""

    model_name: str
    mid: np.ndarray                      # [n_layers]
    out: np.ndarray                      # [n_layers]
    token: str = TARGET_TOKEN
    token_ids: List[int] = field(default_factory=list)
    n_prompts: int = 0
    n_layers: int = DEFAULT_N_LAYERS
    positions: str = "last"
    per_prompt: Optional[np.ndarray] = None  # [n_prompts, 2 * n_layers]
    meta: Dict[str, object] = field(default_factory=dict)

    # ---- derived views -------------------------------------------------------------
    @property
    def curve(self) -> np.ndarray:
        """Interleaved ``[mid_0, out_0, mid_1, out_1, ...]`` curve as in Figure 1."""
        n = min(len(self.mid), len(self.out))
        curve = np.empty(2 * n, dtype=np.float64)
        curve[0::2] = self.mid[:n]
        curve[1::2] = self.out[:n]
        return curve

    @property
    def curve_labels(self) -> List[str]:
        labels: List[str] = []
        for layer in range(self.n_layers):
            labels.extend([f"{layer}-mid", f"{layer}-out"])
        return labels

    def layers(self, kind: str = "out") -> np.ndarray:
        return np.asarray(self.out if kind == "out" else self.mid, dtype=np.float64)

    def probability_at(self, layer: int, kind: str = "out") -> float:
        values = self.out if kind == "out" else self.mid
        return float(values[layer])

    @property
    def final_probability(self) -> float:
        """Probability at the last block output (i.e. the model's own prediction)."""
        return float(self.out[-1]) if len(self.out) else float("nan")

    def top_layers(self, top_k: int = DEFAULT_TOP_K_SHADE, kind: str = "out") -> List[int]:
        values = self.layers(kind)
        order = np.argsort(-values)
        return [int(i) for i in order[: max(int(top_k), 1)]]

    def promotion_mask(self, top_k: int = DEFAULT_TOP_K_SHADE, kind: str = "out") -> np.ndarray:
        """Boolean mask over layers marking the largest promoters of the token."""
        values = self.layers(kind)
        mask = np.zeros_like(values, dtype=bool)
        for layer in self.top_layers(top_k=top_k, kind=kind):
            mask[layer] = True
        return mask

    def summary(self, top_k: int = DEFAULT_TOP_K_SHADE) -> Dict[str, object]:
        top = self.top_layers(top_k=top_k, kind="out")
        return {
            "model": self.model_name,
            "token": self.token,
            "token_ids": list(self.token_ids),
            "n_prompts": int(self.n_prompts),
            "n_layers": int(self.n_layers),
            "positions": self.positions,
            "final_probability": self.final_probability,
            "max_out_probability": float(np.max(self.out)) if len(self.out) else None,
            "max_out_layer": int(np.argmax(self.out)) if len(self.out) else None,
            "top_mlp_layers": top,
            "mean_out_probability": float(np.mean(self.out)) if len(self.out) else None,
            "mean_mid_probability": float(np.mean(self.mid)) if len(self.mid) else None,
        }

    # ---- persistence ---------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model_name": self.model_name,
            "token": self.token,
            "token_ids": [int(i) for i in self.token_ids],
            "n_prompts": int(self.n_prompts),
            "n_layers": int(self.n_layers),
            "positions": self.positions,
            "mid": [float(v) for v in np.asarray(self.mid).ravel()],
            "out": [float(v) for v in np.asarray(self.out).ravel()],
            "meta": dict(self.meta),
        }
        if self.per_prompt is not None:
            payload["per_prompt"] = np.asarray(self.per_prompt, dtype=float).tolist()
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LogitLensResult":
        per_prompt = data.get("per_prompt")
        return cls(
            model_name=str(data.get("model_name", "model")),
            mid=np.asarray(data.get("mid", []), dtype=np.float64),
            out=np.asarray(data.get("out", []), dtype=np.float64),
            token=str(data.get("token", TARGET_TOKEN)),
            token_ids=[int(i) for i in data.get("token_ids", [])],
            n_prompts=int(data.get("n_prompts", 0) or 0),
            n_layers=int(data.get("n_layers", len(data.get("out", []) or [])) or 0),
            positions=str(data.get("positions", "last")),
            per_prompt=None if per_prompt is None else np.asarray(per_prompt, dtype=np.float64),
            meta=dict(data.get("meta", {}) or {}),
        )


# --------------------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------------------


def logit_lens(
    model,
    tokenizer,
    prompts: Sequence[str],
    token_ids: Optional[Sequence[int]] = None,
    token: str = TARGET_TOKEN,
    positions: str = "last",
    n_tokens: int = DEFAULT_N_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: Optional[str] = None,
    model_name: str = "gpt2",
    return_per_prompt: bool = True,
    verbose: bool = False,
) -> LogitLensResult:
    """Apply the unembedding layer to every intermittent layer and average P(token).

    Parameters mirror the paper's description: ``prompts`` are the 295
    RealToxicityPrompts prompts whose greedy next token is "sh*t"; ``positions='last'``
    scores the next-token distribution at the final prompt position.
    """
    if token_ids is None:
        token_ids = target_token_ids(tokenizer, token=token)
    token_ids = [int(i) for i in token_ids]
    if not len(prompts):
        raise ValueError("logit_lens requires at least one prompt")

    device = resolve_device(device)
    info = model_info(model)
    n_layers = info.n_layers

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)

    per_prompt: List[np.ndarray] = []
    model.eval()

    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        elif pad_id is not None:
            attention_mask = (input_ids != pad_id).long()

        mids, outs = collect_hidden_states(
            model, input_ids, attention_mask=attention_mask, n_layers=n_layers
        )

        mid_probs = [
            unembed_probabilities(
                model, mids[l], token_ids, positions=positions,
                attention_mask=attention_mask, n_tokens=n_tokens,
            )
            for l in range(n_layers)
        ]
        out_probs = [
            unembed_probabilities(
                model, outs[l], token_ids, positions=positions,
                attention_mask=attention_mask, n_tokens=n_tokens,
            )
            for l in range(n_layers)
        ]
        stacked = torch.stack(mid_probs + out_probs, dim=1)  # [b, 2L] mid-blocked then out-blocked
        # Reorder to interleaved [mid_0, out_0, mid_1, out_1, ...]
        interleaved = torch.stack(
            [t for pair in zip(mid_probs, out_probs) for t in pair], dim=1
        )
        per_prompt.append(interleaved.detach().cpu().numpy().astype(np.float64))

        if verbose:
            done = min(start + batch_size, len(prompts))
            print(f"[logit-lens:{model_name}] {done}/{len(prompts)} prompts")

    per_prompt_arr = np.concatenate(per_prompt, axis=0) if per_prompt else np.zeros((0, 0))
    n = n_layers
    mid = per_prompt_arr[:, 0::2].mean(axis=0) if per_prompt_arr.size else np.zeros(n)
    out = per_prompt_arr[:, 1::2].mean(axis=0) if per_prompt_arr.size else np.zeros(n)

    result = LogitLensResult(
        model_name=model_name,
        mid=mid,
        out=out,
        token=token,
        token_ids=token_ids,
        n_prompts=len(prompts),
        n_layers=n,
        positions=positions,
        per_prompt=per_prompt_arr if return_per_prompt else None,
        meta={
            "d_model": info.d_model,
            "d_mlp": info.d_mlp,
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "n_tokens": int(n_tokens),
            "device": str(device),
        },
    )
    return result


def collect_logit_lens(*args, **kwargs) -> LogitLensResult:
    """Alias of :func:`logit_lens` kept for script-level naming symmetry."""
    return logit_lens(*args, **kwargs)


def collect_logit_lens_pair(
    model_before,
    model_after,
    tokenizer,
    prompts: Sequence[str],
    before_name: str = "gpt2",
    after_name: str = "gpt2_dpo",
    **kwargs,
) -> Tuple[LogitLensResult, LogitLensResult]:
    """Run the logit lens for both the base model and the DPO model."""
    before = logit_lens(
        model_before, tokenizer, prompts, model_name=before_name, **kwargs
    )
    after = logit_lens(
        model_after, tokenizer, prompts, model_name=after_name, **kwargs
    )
    return before, after


# --------------------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------------------


def compare_logit_lens(
    before: LogitLensResult,
    after: LogitLensResult,
    top_k: int = DEFAULT_TOP_K_SHADE,
) -> Dict[str, object]:
    """Quantify the post-DPO drop in the target-token probability (Figure 1 claim)."""
    curve_before, curve_after = before.curve, after.curve
    n = min(len(curve_before), len(curve_after))
    curve_before, curve_after = curve_before[:n], curve_after[:n]
    delta = curve_after - curve_before

    top_layers = before.top_layers(top_k=top_k, kind="out")
    top_curve_idx = [2 * l + 1 for l in top_layers if 2 * l + 1 < n]

    return {
        "before_model": before.model_name,
        "after_model": after.model_name,
        "token": before.token,
        "n_prompts": int(before.n_prompts),
        "top_mlp_layers": top_layers,
        "curve_before": curve_before.tolist(),
        "curve_after": curve_after.tolist(),
        "delta": delta.tolist(),
        "mean_out_before": float(np.mean(before.out[:n // 2])),
        "mean_out_after": float(np.mean(after.out[:n // 2])),
        "mean_mid_before": float(np.mean(before.mid[:n // 2])),
        "mean_mid_after": float(np.mean(after.mid[:n // 2])),
        "top_layer_mean_before": (
            float(np.mean([curve_before[i] for i in top_curve_idx])) if top_curve_idx else None
        ),
        "top_layer_mean_after": (
            float(np.mean([curve_after[i] for i in top_curve_idx])) if top_curve_idx else None
        ),
        "final_before": before.final_probability,
        "final_after": after.final_probability,
        "max_drop": float(np.min(delta)),
        "max_drop_position": int(np.argmin(delta)),
    }


def promotion_spans(
    result: LogitLensResult,
    top_k: int = DEFAULT_TOP_K_SHADE,
    kind: str = "out",
    pad: float = 0.5,
) -> List[Tuple[float, float]]:
    """X-axis spans (Figure 1 grey shading) for the layers promoting the token most."""
    mask = result.promotion_mask(top_k=top_k, kind=kind)
    spans: List[Tuple[float, float]] = []
    offset = 0 if kind == "mid" else 1
    for layer, active in enumerate(mask):
        if not active:
            continue
        x = 2 * layer + offset
        spans.append((x - pad, x + pad))
    return spans


# --------------------------------------------------------------------------------------
# Plotting (Figure 1)
# --------------------------------------------------------------------------------------


def plot_logit_lens(
    results: Sequence[LogitLensResult],
    labels: Optional[Sequence[str]] = None,
    out_path: Optional[str] = None,
    shade_result: Optional[LogitLensResult] = None,
    top_k: int = DEFAULT_TOP_K_SHADE,
    title: str = 'Logit lens: probability of "sh*t"',
    xlabel: str = "layer",
    ylabel: str = 'P("sh*t")',
    figsize: Tuple[float, float] = (8.0, 4.5),
    log_scale: bool = True,
) -> Optional[str]:
    """Plot the intermittent-layer curves for one or more models (Figure 1 style)."""
    from .plots import (  # lazy import: keeps matplotlib optional at module import
        GPT2_DPO_NAME,
        GPT2_NAME,
        add_legend,
        color_for_model,
        line_series,
        make_figure,
        save_figure,
        shade_region,
    )

    if not results:
        raise ValueError("plot_logit_lens needs at least one LogitLensResult")
    if labels is None:
        labels = [r.model_name for r in results]

    n_layers = max(r.n_layers for r in results)
    xs = list(range(2 * n_layers))

    series: Dict[str, np.ndarray] = {}
    for label, result in zip(labels, results):
        curve = result.curve
        if len(curve) < 2 * n_layers:
            curve = np.concatenate([curve, np.full(2 * n_layers - len(curve), np.nan)])
        series[label] = curve

    colors = [
        color_for_model(label, default=(GPT2_NAME if i == 0 else GPT2_DPO_NAME))
        for i, label in enumerate(labels)
    ]
    markers = ["o" if i == 0 else "s" for i in range(len(labels))]

    fig, ax = make_figure(figsize=figsize)

    shade_source = shade_result if shade_result is not None else results[0]
    for (x0, x1) in promotion_spans(shade_source, top_k=top_k, kind="out"):
        shade_region(ax, x0, x1, color="#bdbdbd", alpha=0.35, zorder=0)

    line_series(
        ax, xs, series, colors=colors, markers=markers,
        xlabel=xlabel, ylabel=ylabel, title=title, legend=True, linewidth=1.7,
    )

    # Minor ticks mark the l-mid layers (after attention heads, before the MLP).
    ax.set_xticks(list(range(0, 2 * n_layers, 2)), minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_xlim(-0.5, 2 * n_layers - 0.5)
    if log_scale:
        try:
            ax.set_yscale("log")
        except Exception:  # pragma: no cover - defensive
            pass
    add_legend(ax)

    return save_figure(fig, out_path) if out_path else None


def plot_figure1(
    before: LogitLensResult,
    after: LogitLensResult,
    out_path: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K_SHADE,
    **kwargs,
) -> Optional[str]:
    """Reproduce Figure 1: GPT2 vs GPT2_DPO, shaded MLP promotion regions."""
    return plot_logit_lens(
        [before, after],
        labels=[before.model_name, after.model_name],
        shade_result=before,
        top_k=top_k,
        out_path=out_path,
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def default_path(model_name: str = "gpt2", out_dir: str = ARTIFACT_DIR) -> str:
    safe = str(model_name).replace("/", "_")
    return os.path.join(out_dir, LOGIT_LENS_FILENAME.format(model=safe))


def save_logit_lens(path: str, result: LogitLensResult) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=1)
    return path


def load_logit_lens(path: str) -> LogitLensResult:
    with open(path, "r", encoding="utf-8") as handle:
        return LogitLensResult.from_dict(json.load(handle))


# --------------------------------------------------------------------------------------
# High-level pipeline entry point (used by scripts/analyze_dpo.py)
# --------------------------------------------------------------------------------------


def analyze_logit_lens(
    model_before,
    model_after,
    tokenizer,
    prompts: Optional[Sequence[str]] = None,
    before_name: str = "gpt2",
    after_name: str = "gpt2_dpo",
    out_dir: str = ARTIFACT_DIR,
    figure_name: str = "figure1_logit_lens.png",
    top_k: int = DEFAULT_TOP_K_SHADE,
    save: bool = True,
    plot: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, object]:
    """Run the full Figure 1 analysis: 295 "sh*t" prompts -> curves -> plot + JSON."""
    if prompts is None:
        from ...data.realtoxicity import load_target_token_prompts  # lazy

        prompts_obj = load_target_token_prompts(
            model_before, tokenizer, n=N_SHIT_PROMPTS, verbose=verbose
        )
        prompts = [p.text if hasattr(p, "text") else str(p) for p in prompts_obj]

    before, after = collect_logit_lens_pair(
        model_before,
        model_after,
        tokenizer,
        prompts,
        before_name=before_name,
        after_name=after_name,
        verbose=verbose,
        **kwargs,
    )
    comparison = compare_logit_lens(before, after, top_k=top_k)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        save_logit_lens(default_path(before_name, out_dir), before)
        save_logit_lens(default_path(after_name, out_dir), after)
        with open(os.path.join(out_dir, "logit_lens_comparison.json"), "w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=1)

    if plot:
        os.makedirs(out_dir, exist_ok=True)
        plot_figure1(before, after, out_path=os.path.join(out_dir, figure_name), top_k=top_k)

    return {
        "before": before,
        "after": after,
        "comparison": comparison,
        "prompts": list(prompts),
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    from ..model_utils import GPT2_MEDIUM, load_model

    model, tokenizer = load_model(GPT2_MEDIUM)
    ids = target_token_ids(tokenizer)
    print("target token ids:", ids, [tokenizer.decode([i]) for i in ids])
    demo_prompts = [
        "I hate this stupid",
        "You are such a",
        "This is the worst garbage I ever saw, you",
    ]
    res = logit_lens(model, tokenizer, demo_prompts, verbose=True)
    print(json.dumps(res.summary(), indent=1))
