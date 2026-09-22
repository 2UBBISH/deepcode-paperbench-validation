#!/usr/bin/env python
"""Section 5 analysis driver for "Stay on Topic with Classifier-Free Guidance".

Runs the four Section-5 analyses of the paper on the 32,902-datapoint P3 sample
(``src/data/p3_sampler.py``) using Falcon-7b-Base and (optionally)
Falcon-7b-Instruct:

* §5.1  sampling entropy           -- ``src/analysis/entropy.py``
        H(p) = -sum_k p_k log p_k per completion token, averaged over tokens;
        paper anchors: CFG (gamma=1.5) ~4.7 vs vanilla ~5.49.
* §5.2  top-p=0.9 overlap + PPL    -- ``src/analysis/overlap.py``,
                                       ``src/analysis/perplexity.py``
        minimum top-p nuclei of CFG vs vanilla overlap ~50%; continuation-only
        perplexity correlates r~0.94 (CFG vs vanilla) and r~0.70 (instruct vs CFG).
* §5.3  vocabulary re-ranking      -- ``src/analysis/visualize.py``
        Table 3 for prompt "The dragon flew over Paris, France" (c_bar = empty).

Examples
--------
Full Falcon-7b-Base / Falcon-7b-Instruct run on the real P3 sample::

    python scripts/run_analysis.py --model tiiuae/falcon-7b \\
        --instruct-model tiiuae/falcon-7b-instruct --limit 5000

CPU smoke test with a deterministic mock LM (no GPU / no model download)::

    python scripts/run_analysis.py --dry-run --limit 32 --gamma-sweep

Pure-math sanity check of the metric implementations (no torch at all)::

    python scripts/run_analysis.py --math-only

Outputs ``<out-dir>/analysis_report.json`` plus, when matplotlib is available,
``entropy_vs_gamma.png`` / ``overlap_vs_gamma.png``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: make ``src`` importable no matter where the script is launched.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.p3_sampler import (  # noqa: E402
    DEFAULT_SEED,
    MAX_INPUT_TOKENS,
    TARGET_N_DATAPOINTS,
    get_p3_sample,
    iter_batches,
    sample_p3_synthetic,
    summary_stats,
)
from src.analysis import entropy as entropy_mod  # noqa: E402
from src.analysis import overlap as overlap_mod  # noqa: E402
from src.analysis import perplexity as ppl_mod  # noqa: E402
from src.analysis import visualize as viz_mod  # noqa: E402

logger = logging.getLogger("run_analysis")

ANALYSIS_GAMMA = getattr(entropy_mod, "ANALYSIS_GAMMA", 1.5)
GAMMA_SWEEP = (1.0, 1.25, 1.5, 1.75, 2.0)

#: prompts are truncated to this many characters before analysis (tokenizer-free cap)
PROMPT_CHAR_LIMIT = 1200


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments it actually accepts.

    Small signature drifts between the analysis modules must not break the
    driver, so unsupported kwargs are silently dropped.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(**kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        logger.debug("dropping unsupported kwargs %s for %s", dropped, fn)
    return fn(**accepted)


def _safe(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Run one analysis phase, never letting it abort the whole report."""
    t0 = time.time()
    try:
        value = fn(*args, **kwargs)
        return {"ok": True, "value": value, "seconds": round(time.time() - t0, 2)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("phase %s failed", name)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 2)}


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


# ---------------------------------------------------------------------------
# mock LM (CPU smoke test, no download / no GPU)
# ---------------------------------------------------------------------------
class _MockDual:
    """Minimal stand-in for ``cfg.model_wrapper.DualLogits``."""

    def __init__(self, cond, uncond):  # noqa: D401
        self.cond = cond
        self.uncond = uncond

    def guided(self, gamma: float = 1.0):  # pragma: no cover - convenience
        return self.uncond + float(gamma) * (self.cond - self.uncond)


class MockCFGWrapper:
    """Deterministic, dependency-light fake of :class:`CFGModelWrapper`.

    Logits are a hash-based function of the context's last token id, so the
    full Section-5 pipeline (dual forward pass -> Eq. 7 -> softmax -> metrics)
    can be exercised end to end on CPU.  The conditional logits are a sharpened
    version of the unconditional ones, therefore increasing ``gamma`` sharpens
    the guided distribution and lowers entropy -- the same qualitative
    behaviour the paper reports.
    """

    def __init__(self, vocab_size: int = 1024, seed: int = 0, device: str = "cpu", temperature_sharpening: float = 1.4):
        import torch  # local import: MockCFGWrapper is optional

        self.torch = torch
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.eos_token_id = 0
        self.bos_token_id = 0
        self.pad_token_id = 0
        self.unconditional_mode = "empty_prefix"
        self.sharpening = float(temperature_sharpening)
        self.tokenizer = None

    # -- tokenizer surface -------------------------------------------------
    def encode(self, text: Any, *args: Any, **kwargs: Any):
        torch = self.torch
        if isinstance(text, str):
            base = [1 + (ord(c) % (self.vocab_size - 1)) for c in text]
            base = base[-64:] or [1]
        else:
            base = [int(t) for t in text]
        return torch.tensor([base], dtype=torch.long, device=self.device)

    def decode(self, ids: Any, *args: Any, **kwargs: Any) -> str:
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(f"<t{i % self.vocab_size}>" for i in ids)

    # -- deterministic pseudo-logits ---------------------------------------
    def _row(self, token_id: int, seq: int) -> Any:
        torch = self.torch
        g = torch.Generator().manual_seed((int(token_id) * 2_654_435_761 + self.seed * 97 + seq) % (2**31 - 1))
        uncond = torch.randn(self.vocab_size, generator=g) * 0.5
        # conditional pass = base + a small bump on a prompt-dependent token
        target = int(token_id) % self.vocab_size
        sharp = 2.0 * self.sharpening
        delta = torch.zeros(self.vocab_size)
        delta[target] = sharp
        return uncond, uncond + delta

    def dual_logits(self, input_ids: Any, attention_mask: Any = None, prompt_length: Optional[int] = None,
                    only_last: bool = False, negative_input_ids: Any = None,
                    negative_attention_mask: Any = None) -> _MockDual:
        """Mirror the real wrapper's contract closely enough for the analyzers."""
        torch = self.torch
        ids = input_ids
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        out_cond, out_uncond = [], []
        for b in range(ids.shape[0]):
            seq = ids[b].tolist()
            n = len(seq)
            if prompt_length is None:
                uncond_seq = [self.bos_token_id]
            elif prompt_length >= 1:
                uncond_seq = seq[max(0, prompt_length - 1):]
            else:
                uncond_seq = []
            c_last = seq[-1] if n else self.bos_token_id
            u_last = uncond_seq[-1] if uncond_seq else self.bos_token_id
            steps = 1 if only_last else max(1, n - (prompt_length or 0)) + 1
            cond_rows, uncond_rows = [], []
            for s in range(steps):
                c, u = self._row(c_last, s)
                cond_rows.append(c)
                uncond_rows.append(u)
            out_cond.append(torch.stack(cond_rows))
            out_uncond.append(torch.stack(uncond_rows))
        return _MockDual(torch.stack(out_cond), torch.stack(out_uncond))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return ``(records, sample_stats)`` for the Section-5 analysis.

    ``records`` are dicts with ``inputs``/``targets`` keys, which both the P3
    sampler and :mod:`src.analysis.perplexity` understand.
    """
    if args.dataset == "synthetic":
        samples = sample_p3_synthetic(
            n_datasets=args.synthetic_datasets,
            rows_per_dataset=args.synthetic_rows,
            n_per_dataset=max(1, args.limit // max(1, args.synthetic_datasets)),
            seed=args.seed,
        )
    else:
        samples = get_p3_sample(
            subsets=None,
            n_per_dataset=50,
            seed=args.seed,
            cache_path=args.cache,
            synthetic=False,
        )
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    records = [
        {"inputs": s.inputs, "targets": s.targets, "dataset": getattr(s, "dataset", None)}
        for s in samples
    ]
    stats = summary_stats(samples) if samples else {}
    return records, stats


def record_prompts(records: Sequence[Dict[str, Any]], char_limit: int = PROMPT_CHAR_LIMIT) -> List[str]:
    prompts = []
    for r in records:
        text = r.get("inputs") or r.get("prompt") or ""
        prompts.append(str(text)[:char_limit])
    return prompts


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------
def phase_entropy(wrapper, prompts, gamma: float, max_new_tokens: int, seed: int, instruct=None) -> Any:
    analyzer = _call(
        entropy_mod.EntropyAnalyzer,
        model_wrapper=wrapper,
        gamma=gamma,
        max_new_tokens=max_new_tokens,
        unconditional_mode="empty_prefix",
        instruct_wrapper=instruct,
        seed=seed,
        device=None,
    )
    return _call(
        analyzer.compare,
        prompts=prompts,
        modes=("cfg", "vanilla", "unprompted"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        seed=seed,
        progress=True,
    )


def phase_overlap(wrapper, prompts, gamma: float, max_new_tokens: int, seed: int, instruct=None) -> Any:
    analyzer = _call(
        overlap_mod.OverlapAnalyzer,
        model_wrapper=wrapper,
        gamma=gamma,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        unconditional_mode="empty_prefix",
        instruct_wrapper=instruct,
        seed=seed,
        device=None,
    )
    return _call(
        analyzer.compare,
        prompts=prompts,
        modes=("cfg", "vanilla", "unprompted"),
        reference="vanilla",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        seed=seed,
        include_instruct=instruct is not None,
        progress=True,
    )


def phase_perplexity(wrapper, records, gamma: float, seed: int, instruct=None) -> Any:
    analyzer = _call(
        ppl_mod.PerplexityAnalyzer,
        model_wrapper=wrapper,
        gamma=gamma,
        unconditional_mode="empty_prefix",
        instruct_wrapper=instruct,
        seed=seed,
        device=None,
    )
    modes = ("cfg", "vanilla", "instruct") if instruct is not None else ("cfg", "vanilla")
    return _call(
        analyzer.compare,
        records=records,
        modes=modes,
        reference="vanilla",
        max_datapoints=None,
        include_instruct=instruct is not None,
        progress=True,
    )


def phase_table3(wrapper, gamma: float, n_steps: int, instruct=None) -> List[Any]:
    return _call(
        viz_mod.run_table3_walkthrough,
        model_wrapper=wrapper,
        prompt=viz_mod.DRAGON_PROMPT,
        n_steps=n_steps,
        gamma=gamma,
        top_k=getattr(viz_mod, "TOP_K_DISPLAY", 5),
        reference="vanilla",
        negative_prompt=getattr(viz_mod, "DRAGON_NEGATIVE_PROMPT", ""),
        instruct_wrapper=instruct,
    )


# ---------------------------------------------------------------------------
# mock-only dry run (works with no model download; needs torch)
# ---------------------------------------------------------------------------
def run_dry(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        logger.warning("torch unavailable (%s): falling back to math-only run", exc)
        return run_math_only(args, records)

    logger.info("dry-run: using MockCFGWrapper (deterministic CPU logits)")
    wrapper = MockCFGWrapper(vocab_size=args.mock_vocab, seed=args.seed)
    instruct = MockCFGWrapper(vocab_size=args.mock_vocab, seed=args.seed + 7) if args.instruct_model == "mock" else None
    prompts = record_prompts(records)[: args.limit or 8]
    return _run_pipeline(args, records, prompts, wrapper, instruct)


# ---------------------------------------------------------------------------
# math-only run: validates the metric implementations with zero model cost
# ---------------------------------------------------------------------------
def run_math_only(args: argparse.Namespace, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Exercise the pure-NumPy metric layer on synthetic distributions."""
    logger.info("math-only run: synthetic logits through the metric layer")
    rng = np.random.default_rng(args.seed)
    n_prompts = max(1, args.limit or 8)
    steps = max(1, args.max_new_tokens)
    vocab = args.mock_vocab

    ent_by_mode: Dict[str, List[float]] = {m: [] for m in ("cfg", "vanilla", "unprompted")}
    ov_cfg: List[float] = []
    ppl_by_mode: Dict[str, List[float]] = {m: [] for m in ("cfg", "vanilla")}

    for i in range(n_prompts):
        probs = {}
        for mode, gamma in (("unprompted", 0.0), ("vanilla", 1.0), ("cfg", args.gamma)):
            logits_uncond = rng.normal(0, 0.6, size=(steps, vocab))
            logits_cond = logits_uncond + rng.normal(0, 0.9, size=(steps, vocab))
            guided = entropy_mod._guided_distribution(logits_cond, logits_uncond, gamma)  # noqa: SLF001
            probs[mode] = np.asarray(guided, dtype=np.float64)
            ent_by_mode[mode].append(float(entropy_mod.mean_entropy(entropy_mod.entropy(probs[mode], axis=-1))))
        sets_cfg = overlap_mod.top_p_token_sets(probs["cfg"], top_p=0.9)
        sets_van = overlap_mod.top_p_token_sets(probs["vanilla"], top_p=0.9)
        ov_cfg.append(float(np.mean([overlap_mod.overlap_fraction(a, b) for a, b in zip(sets_cfg, sets_van)])))
        for mode in ("cfg", "vanilla"):
            lp = overlap_mod.log_softmax(np.log(np.clip(probs[mode], 1e-12, None)), axis=-1)
            targets = np.arange(steps) % vocab
            ppl_by_mode[mode].append(
                float(ppl_mod.perplexity_from_logprobs(lp[np.arange(steps), targets]))
            )

    entropy_report = {
        "modes": {m: {"mean": float(np.mean(v)), "n": len(v)} for m, v in ent_by_mode.items()},
        "cfg_lower_than_vanilla": float(np.mean(ent_by_mode["cfg"])) < float(np.mean(ent_by_mode["vanilla"])),
        "note": "synthetic distributions (math-only mode)",
    }
    overlap_report = {
        "mean": float(np.mean(ov_cfg)),
        "n": len(ov_cfg),
        "note": "synthetic distributions (math-only mode)",
    }
    ppl_report = {
        "modes": {m: {"mean_ppl": float(np.mean(v))} for m, v in ppl_by_mode.items()},
        "note": "synthetic distributions (math-only mode)",
    }
    return {
        "meta": _meta(args) | {"mode": "math-only"},
        "p3_sample": {},
        "entropy": entropy_report,
        "overlap": overlap_report,
        "perplexity": ppl_report,
        "table3": {},
        "paper_checks": _paper_checks(entropy_report, overlap_report, ppl_report),
        "gamma_sweep": [],
        "phases": {},
    }


# ---------------------------------------------------------------------------
# shared pipeline
# ---------------------------------------------------------------------------
def _run_pipeline(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    prompts: List[str],
    wrapper: Any,
    instruct: Any,
) -> Dict[str, Any]:
    do = set(args.phases)
    report: Dict[str, Any] = {"meta": _meta(args), "p3_sample": {}, "phases": {}}
    report["p3_sample"] = _safe("summary_stats", summary_stats, [])["value"] if False else {}

    # -- §5.1 entropy ------------------------------------------------------
    entropy_report: Dict[str, Any] = {}
    if "entropy" in do:
        res = _safe("entropy", phase_entropy, wrapper, prompts, args.gamma, args.max_new_tokens, args.seed, instruct)
        report["phases"]["entropy"] = {"ok": res["ok"], "seconds": res["seconds"], "error": res.get("error")}
        if res["ok"]:
            entropy_report = res["value"].as_dict() if hasattr(res["value"], "as_dict") else dict(res["value"])
            print(entropy_mod.format_entropy_table(res["value"]))
    report["entropy"] = entropy_report

    # -- §5.2 overlap ------------------------------------------------------
    overlap_report: Dict[str, Any] = {}
    if "overlap" in do:
        res = _safe("overlap", phase_overlap, wrapper, prompts, args.gamma, args.max_new_tokens, args.seed, instruct)
        report["phases"]["overlap"] = {"ok": res["ok"], "seconds": res["seconds"], "error": res.get("error")}
        if res["ok"]:
            overlap_report = res["value"].as_dict() if hasattr(res["value"], "as_dict") else dict(res["value"])
            print(overlap_mod.format_overlap_table(res["value"]))
    report["overlap"] = overlap_report

    # -- §5.2 perplexity ---------------------------------------------------
    ppl_report: Dict[str, Any] = {}
    if "perplexity" in do:
        res = _safe("perplexity", phase_perplexity, wrapper, records, args.gamma, args.seed, instruct)
        report["phases"]["perplexity"] = {"ok": res["ok"], "seconds": res["seconds"], "error": res.get("error")}
        if res["ok"]:
            ppl_report = res["value"].as_dict() if hasattr(res["value"], "as_dict") else dict(res["value"])
            print(ppl_mod.format_ppl_table(res["value"]))
    report["perplexity"] = ppl_report

    # -- §5.3 vocabulary re-ranking (Table 3) ------------------------------
    table3: Dict[str, Any] = {}
    if "table3" in do:
        res = _safe("table3", phase_table3, wrapper, args.gamma, args.table3_steps, instruct)
        report["phases"]["table3"] = {"ok": res["ok"], "seconds": res["seconds"], "error": res.get("error")}
        if res["ok"]:
            rankings = res["value"]
            table3 = {
                "prompt": getattr(viz_mod, "DRAGON_PROMPT", ""),
                "gamma": args.gamma,
                "n_steps": len(rankings),
                "columns": viz_mod.table3_columns(rankings),
                "text": viz_mod.format_table3(rankings, top_k=getattr(viz_mod, "TOP_K_DISPLAY", 5)),
                "report": _call(viz_mod.ranking_report, rankings=rankings),
            }
            print(table3["text"])
            if args.save_rankings:
                viz_mod.save_rankings(os.path.join(args.out_dir, "table3_rankings.json"), rankings)
                if args.plot:
                    viz_mod.plot_rankings(rankings, path=os.path.join(args.out_dir, "table3_ranking.png"))
    report["table3"] = table3

    report["paper_checks"] = _paper_checks(entropy_report, overlap_report, ppl_report)

    # -- optional gamma sweep (entropy/overlap vs gamma) -------------------
    report["gamma_sweep"] = []
    if args.gamma_sweep:
        for gamma in GAMMA_SWEEP:
            row: Dict[str, Any] = {"gamma": float(gamma)}
            if "entropy" in do:
                r = _safe(f"entropy g={gamma}", phase_entropy, wrapper, prompts,
                          gamma, args.max_new_tokens, args.seed, instruct)
                if r["ok"]:
                    row["entropy_mean"] = _as_float(getattr(r["value"], "mean", None))
            if "overlap" in do:
                r = _safe(f"overlap g={gamma}", phase_overlap, wrapper, prompts,
                          gamma, args.max_new_tokens, args.seed, instruct)
                if r["ok"]:
                    row["overlap_mean"] = _as_float(getattr(r["value"], "mean", None))
            report["gamma_sweep"].append(row)
            logger.info("gamma=%.2f -> %s", gamma, {k: v for k, v in row.items() if k != "gamma"})
        if args.plot:
            _plot_sweep(report["gamma_sweep"], args.out_dir)

    return report


def _meta(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "instruct_model": args.instruct_model,
        "gamma": args.gamma,
        "seed": args.seed,
        "dataset": args.dataset,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "dry_run": bool(args.dry_run),
        "paper_reference": {
            "entropy": {"cfg": getattr(entropy_mod, "CFG_ENTROPY_MEAN", 4.7),
                        "vanilla": getattr(entropy_mod, "VANILLA_ENTROPY_MEAN", 5.49)},
            "overlap": getattr(overlap_mod, "CFG_VANILLA_OVERLAP", 0.5),
            "ppl_correlation": {
                "cfg_vs_vanilla": getattr(ppl_mod, "CFG_PPL_VANILLA_CORR", 0.94),
                "instruct_vs_cfg": getattr(ppl_mod, "CFG_PPL_INSTRUCT_CORR", 0.70),
            },
            "p3_target_datapoints": TARGET_N_DATAPOINTS,
        },
    }


def _paper_checks(entropy_report: Dict[str, Any], overlap_report: Dict[str, Any],
                  ppl_report: Dict[str, Any]) -> Dict[str, Any]:
    """Compare measured Section-5 numbers against the paper's anchors."""
    checks: Dict[str, Any] = {}
    if entropy_report:
        try:
            modes = entropy_report.get("modes", entropy_report)
            cfg = _as_float(modes.get("cfg", {}).get("mean"))
            van = _as_float(modes.get("vanilla", {}).get("mean"))
            checks["entropy"] = {
                "cfg_mean": cfg,
                "vanilla_mean": van,
                "expected": [getattr(entropy_mod, "CFG_ENTROPY_MEAN", 4.7),
                             getattr(entropy_mod, "VANILLA_ENTROPY_MEAN", 5.49)],
                "direction_ok": (cfg is not None and van is not None and cfg < van),
            }
        except Exception:  # pragma: no cover
            pass
    if overlap_report:
        mean = _as_float(overlap_report.get("mean"))
        checks["overlap"] = {
            "mean": mean,
            "expected": getattr(overlap_mod, "CFG_VANILLA_OVERLAP", 0.5),
            "direction_ok": None if mean is None else abs(mean - getattr(overlap_mod, "CFG_VANILLA_OVERLAP", 0.5)) < 0.25,
        }
    if ppl_report:
        corr = ppl_report.get("correlations", {})
        checks["perplexity"] = {
            "cfg_vs_vanilla": _as_float(_pair_corr(corr, "cfg", "vanilla")),
            "instruct_vs_cfg": _as_float(_pair_corr(corr, "instruct", "cfg")),
            "expected": {
                "cfg_vs_vanilla": getattr(ppl_mod, "CFG_PPL_VANILLA_CORR", 0.94),
                "instruct_vs_cfg": getattr(ppl_mod, "CFG_PPL_INSTRUCT_CORR", 0.70),
            },
        }
    return checks


def _pair_corr(corr: Any, a: str, b: str) -> Optional[float]:
    if not isinstance(corr, dict):
        return None
    for key in (f"{a}_vs_{b}", f"{b}_vs_{a}", f"{a}-{b}", f"{b}-{a}", (a, b), (b, a)):
        if key in corr:
            entry = corr[key]
            if isinstance(entry, dict):
                for field in ("spearman", "pearson", "r", "corr"):
                    if field in entry:
                        return entry[field]
            return entry
    return None


def _plot_sweep(rows: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        logger.info("matplotlib unavailable; skipping sweep plot")
        return
    gammas = [r["gamma"] for r in rows]
    for key, fname, label in (("entropy_mean", "entropy_vs_gamma.png", "mean token entropy H(p)"),
                              ("overlap_mean", "overlap_vs_gamma.png", "top-p=0.9 overlap w/ vanilla")):
        ys = [r.get(key) for r in rows]
        if all(y is None for y in ys):
            continue
        xs = [g for g, y in zip(gammas, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(xs, yy, marker="o")
        ax.set_xlabel("gamma")
        ax.set_ylabel(label)
        ax.set_title("Section 5 analysis (CFG)")
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("wrote %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Section 5 analysis (entropy / overlap / PPL / Table 3) for CFG on P3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="tiiuae/falcon-7b", help="base LM (Falcon-7b-Base in the paper)")
    p.add_argument("--instruct-model", default="tiiuae/falcon-7b-instruct",
                   help="instruction-tuned LM; pass '' to skip, 'mock' in --dry-run")
    p.add_argument("--gamma", type=float, default=ANALYSIS_GAMMA, help="CFG guidance strength (paper: 1.5)")
    p.add_argument("--dataset", default="p3", choices=["p3", "synthetic"],
                   help="'synthetic' avoids any network access")
    p.add_argument("--limit", type=int, default=200, help="number of P3 datapoints (paper: 32902)")
    p.add_argument("--max-new-tokens", type=int, default=128, help="completion length per datapoint")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--cache", default="data/cache/p3_sample.json")
    p.add_argument("--out-dir", default="results/analysis")
    p.add_argument("--phases", default="entropy,overlap,perplexity,table3",
                   help="comma-separated subset of entropy,overlap,perplexity,table3")
    p.add_argument("--table3-steps", type=int, default=12)
    p.add_argument("--gamma-sweep", action="store_true",
                   help="also sweep gamma in {1.0,1.25,1.5,1.75,2.0} for entropy/overlap")
    p.add_argument("--plot", action="store_true", help="write PNG plots when matplotlib is available")
    p.add_argument("--save-rankings", action="store_true", help="dump Table-3 rankings to JSON")
    p.add_argument("--dry-run", action="store_true",
                   help="use a deterministic mock LM (CPU smoke test, no downloads)")
    p.add_argument("--math-only", action="store_true",
                   help="validate the metric layer on synthetic distributions only")
    p.add_argument("--mock-vocab", type=int, default=1024, help="vocab size of the mock LM")
    p.add_argument("--synthetic-datasets", type=int, default=4)
    p.add_argument("--synthetic-rows", type=int, default=32)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args.phases = [s.strip() for s in str(args.phases).split(",") if s.strip()]
    if not args.dry_run and args.dataset == "synthetic":
        args.dry_run = False
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        records, sample_stats = load_records(args)
    except Exception as exc:
        logger.warning("could not load P3 (%s); falling back to synthetic records", exc)
        args.dataset = "synthetic"
        records, sample_stats = load_records(args)
    logger.info("loaded %d datapoints (target %d; max input tokens %d)", len(records), TARGET_N_DATAPOINTS, MAX_INPUT_TOKENS)

    if args.math_only:
        report = run_math_only(args, records)
    elif args.dry_run:
        report = run_dry(args, records)
    else:
        from src.cfg.model_wrapper import CFGModelWrapper

        logger.info("loading %s (%s/%s)", args.model, args.device, args.dtype)
        wrapper = CFGModelWrapper(args.model, device=args.device, dtype=args.dtype)
        instruct = None
        if args.instruct_model:
            logger.info("loading instruct model %s", args.instruct_model)
            instruct = CFGModelWrapper(args.instruct_model, device=args.device, dtype=args.dtype)
        prompts = record_prompts(records)
        report = _run_pipeline(args, records, prompts, wrapper, instruct)

    report["p3_sample"] = sample_stats
    path = os.path.join(args.out_dir, "analysis_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("wrote %s", path)

    checks = report.get("paper_checks", {})
    if checks:
        print("\n=== Section 5 paper checks ===")
        print(json.dumps(checks, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
