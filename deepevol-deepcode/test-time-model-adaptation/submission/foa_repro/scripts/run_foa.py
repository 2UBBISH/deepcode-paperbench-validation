#!/usr/bin/env python
"""Main entry point for FOA (Forward-Optimization Adaptation).

Implements Algorithm 1 of the paper "Test-Time Model Adaptation with Only
Forward Passes" (FOA):

    Input: test batches {X_t}_{t=1..T}, model f_Theta = Head(L_i(.)),
           ID statistics {mu_i^S, sigma_i^S}_{i=0..N}, population size K.
    Initialize m^(0) = 0, Sigma^(0) = I, tau^(0) = 1 in Eqn. (6).
    for t = 1..T:
        sample K prompt solutions {p_k^t} by Eqn. (6)            (CMA `ask`)
        for k = 1..K:
            all-layer CLS features {e_n^0}_{n=1..N} of [p_k^t ; X_t]   (Eqn. 1-2)
            adjust e_N^0 to the source domain                      (Eqn. 7)
            predict y_hat_t^k = Head(e_N^0)
            fitness v_k                                            (Eqn. 5)
        update m^(t), Sigma^(t), tau^(t) from {v_k} (CMA `tell`)
        emit y_hat_t = prediction of the candidate with the best v_k

Everything is forward-only: model weights are frozen, no backward pass exists
anywhere in this file, and the only learnable object is the input prompt.

Usage
-----
    python scripts/run_foa.py --config configs/foa_imagenetc.yaml
    python scripts/run_foa.py --config configs/foa_imagenetc.yaml \
        --corruptions gaussian_noise --limit-batches 5
    python scripts/run_foa.py --config configs/foa_imagenetr.yaml --all-corruptions
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Make `import src...` work when this script is executed directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.prompt_injection import build_prompt  # noqa: E402
from src.models.vit_loader import FOA_CHECKPOINT_URL, build_vit  # noqa: E402
from src.method.activation_shifting import build_activation_shifter  # noqa: E402
from src.method.cma_wrapper import build_cma_optimizer  # noqa: E402
from src.method.fitness import build_fitness  # noqa: E402
from src.method.source_stats import (  # noqa: E402
    DEFAULT_NUM_SOURCE_SAMPLES,
    load_source_stats,
)
from src.utils.config import Config, load_config, save_config, config_to_dict  # noqa: E402


# ----------------------------------------------------------------------------
# Default corruption list of ImageNet-C (15 types, 4 categories)
# ----------------------------------------------------------------------------
IMAGENET_C_CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


# ----------------------------------------------------------------------------
# Small self-contained metrics fallback (src/eval/metrics.py is preferred and is
# used whenever importable; the fallback keeps this script runnable stand-alone).
# ----------------------------------------------------------------------------
def _fallback_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    return float((preds == targets).float().mean().item())


def _fallback_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width ECE with ``n_bins`` bins (mirrors src/eval/metrics.py)."""
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidences.size
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(correct[mask].mean())
        conf = float(confidences[mask].mean())
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def _load_metric_functions():
    """Return (accuracy_fn, ece_fn) preferring src/eval/metrics.py."""
    try:  # pragma: no cover - depends on whether metrics.py exists yet
        from src.eval import metrics as M  # type: ignore

        acc_fn = None
        for name in ("compute_accuracy", "accuracy_top1", "top1_accuracy", "accuracy"):
            if hasattr(M, name):
                acc_fn = getattr(M, name)
                break
        ece_fn = None
        for name in ("compute_ece", "expected_calibration_error", "ece"):
            if hasattr(M, name):
                ece_fn = getattr(M, name)
                break
        return acc_fn, ece_fn
    except Exception:
        return None, None


# ----------------------------------------------------------------------------
# Result bookkeeping
# ----------------------------------------------------------------------------
class ResultAccumulator:
    """Streaming accuracy / ECE accumulator (ECE needs only 2 floats/sample)."""

    def __init__(self, ece_bins: int = 15, accuracy_fn=None, ece_fn=None):
        self.ece_bins = int(ece_bins)
        self.accuracy_fn = accuracy_fn
        self.ece_fn = ece_fn
        self.num_correct = 0
        self.num_total = 0
        self._conf: List[np.ndarray] = []
        self._corr: List[np.ndarray] = []
        self.per_batch: List[Dict[str, float]] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """Accumulate one batch; returns (batch accuracy, batch ECE)."""
        with torch.no_grad():
            logits = logits.detach()
            probs = torch.softmax(logits.float(), dim=-1)
            conf, preds = probs.max(dim=-1)
            correct = (preds == targets.to(preds.device)).to(torch.float32)
        n = int(targets.numel())
        if n == 0:
            return 0.0, 0.0
        self.num_total += n
        self.num_correct += int(correct.sum().item())
        c_np = conf.cpu().numpy().astype(np.float64)
        k_np = correct.cpu().numpy().astype(np.float64)
        self._conf.append(c_np)
        self._corr.append(k_np)
        if self.accuracy_fn is not None:
            b_acc = float(self.accuracy_fn(logits, targets))
        else:
            b_acc = _fallback_accuracy(preds, targets.to(preds.device))
        if self.ece_fn is not None:
            try:
                b_ece = float(self.ece_fn(logits, targets, n_bins=self.ece_bins))
            except TypeError:
                b_ece = float(self.ece_fn(logits, targets))
        else:
            b_ece = _fallback_ece(c_np, k_np, self.ece_bins)
        self.per_batch.append({"accuracy": b_acc, "ece": b_ece, "num_samples": n})
        return b_acc, b_ece

    def compute(self) -> Dict[str, float]:
        acc = 100.0 * self.num_correct / self.num_total if self.num_total else 0.0
        if self._conf:
            conf = np.concatenate(self._conf)
            corr = np.concatenate(self._corr)
            ece = _fallback_ece(conf, corr, self.ece_bins)
        else:
            ece = 0.0
        return {
            "accuracy": float(acc),
            "ece": 100.0 * float(ece),
            "num_samples": int(self.num_total),
        }


# ----------------------------------------------------------------------------
# Data loading (lazy import; src/data/datasets.py owns the actual loaders)
# ----------------------------------------------------------------------------
def build_test_loader(cfg: Config, dataset: Optional[str] = None,
                      corruption: Optional[str] = None,
                      severity: Optional[int] = None,
                      batch_size: Optional[int] = None,
                      limit_batches: Optional[int] = None,
                      **overrides):
    """Build the ordered, single-pass online TTA loader for a benchmark."""
    data_cfg = dict(cfg.get("data", {}))
    data_cfg.update(overrides)
    if dataset is not None:
        data_cfg["dataset"] = dataset
    if corruption is not None:
        data_cfg["corruption"] = corruption
    if severity is not None:
        data_cfg["severity"] = severity
    if batch_size is not None:
        data_cfg["batch_size"] = batch_size
    # Algorithm 1 consumes a single ordered pass: never shuffle the stream.
    data_cfg["shuffle"] = False

    try:
        from src.data import datasets as D  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "src/data/datasets.py is required to run FOA; import failed with: "
            f"{exc}"
        ) from exc

    builder = None
    for name in ("build_tta_loader", "build_test_loader", "build_dataset_loader",
                 "build_loader"):
        if hasattr(D, name):
            builder = getattr(D, name)
            break
    if builder is None:
        raise RuntimeError("src/data/datasets.py exposes no loader builder.")

    try:
        loader = builder(cfg=cfg, **data_cfg)
    except TypeError:
        loader = builder(**data_cfg)
    return _MaybeLimited(loader, limit_batches)


class _MaybeLimited:
    """Wrap an iterable and optionally stop after ``limit`` batches."""

    def __init__(self, loader: Iterable, limit: Optional[int] = None):
        self.loader = loader
        self.limit = None if limit in (None, 0, -1) else int(limit)
        self.dataset = getattr(loader, "dataset", None)
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.limit is not None and i >= self.limit:
                break
            yield batch

    def __len__(self):
        try:
            n = len(self.loader)  # type: ignore[arg-type]
        except TypeError:
            return 0
        return min(n, self.limit) if self.limit is not None else n


# ----------------------------------------------------------------------------
# Core: one FOA pass over one test stream
# ----------------------------------------------------------------------------
def run_foa(cfg: Config,
            loader: Optional[Iterable] = None,
            model=None,
            prompt=None,
            optimizer=None,
            fitness=None,
            shifter=None,
            source_stats=None,
            device: Optional[str] = None,
            verbose: bool = True,
            log_every: int = 20,
            class_subset: Optional[Sequence[int]] = None,
            corruption: Optional[str] = None) -> Dict[str, Any]:
    """Run Algorithm 1 over one online test stream and return the results."""
    device = device or str(cfg.get("model", {}).get("device", "cuda") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    seed = int(cfg.get("seed", 0) or 0)

    # ---------------- model (frozen, forward only) ----------------
    if model is None:
        model_cfg = cfg.get("model", {}) or {}
        model = build_vit(
            model_name=str(model_cfg.get("name", "vit_base_patch16_224")),
            checkpoint=model_cfg.get("checkpoint"),
            pretrained=bool(model_cfg.get("pretrained", True)),
            num_classes=int(model_cfg.get("num_classes", 1000)),
            device=device,
        )
        if model_cfg.get("checkpoint") is None and verbose:
            print(f"[FOA] using augreg checkpoint: {FOA_CHECKPOINT_URL}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---------------- source ID statistics ----------------
    if source_stats is None:
        ss_cfg = cfg.get("source_stats", {}) or {}
        ss_path = ss_cfg.get("path")
        if not ss_path or not os.path.exists(str(ss_path)):
            raise FileNotFoundError(
                f"Source statistics checkpoint {ss_path!r} not found. Run "
                "`python scripts/compute_source_stats.py --config <cfg>` first."
            )
        source_stats = load_source_stats(str(ss_path), device=device)
    else:
        source_stats = source_stats.to(torch_device)

    # ---------------- learnable prompt (only mutable object) ----------------
    if prompt is None:
        p_cfg = cfg.get("prompt", {}) or {}
        prompt = build_prompt(
            embed_dim=int(getattr(model, "embed_dim", 768)),
            num_prompts=int(p_cfg.get("num_prompts", 3)),
            init=str(p_cfg.get("init", "uniform")),
            init_range=float(p_cfg.get("init_range", 0.01)),
            seed=seed,
            device=torch_device,
        )

    # ---------------- CMA-ES (Eqn. 6) ----------------
    if optimizer is None:
        optimizer = build_cma_optimizer(cfg=cfg, dim=int(prompt.prompt_dim))
    K = int(getattr(optimizer, "population_size", 0) or
            (cfg.get("cma", {}) or {}).get("population_size", 28))

    # ---------------- unsupervised fitness (Eqn. 5) ----------------
    if fitness is None:
        f_cfg = cfg.get("fitness", {}) or {}
        fitness = build_fitness(
            source_stats,
            cfg=cfg,
            lambda_base=float(f_cfg.get("lambda_base", 0.4)),
            use_entropy=bool(f_cfg.get("use_entropy", True)),
            use_discrepancy=bool(f_cfg.get("use_discrepancy", True)),
            beta=float(f_cfg.get("beta", 1.0)),
            layer_start=int(f_cfg.get("layer_start", 1)),
        )

    # ---------------- back-to-source activation shifting (Eqn. 7-9) ----------------
    if shifter is None:
        shifter = build_activation_shifter(source_stats, cfg=cfg, device=torch_device)

    # ---------------- loader ----------------
    if loader is None:
        loader = build_test_loader(cfg)

    eval_cfg = cfg.get("eval", {}) or {}
    ece_bins = int(eval_cfg.get("ece_bins", 15))
    acc_fn, ece_fn = _load_metric_functions()
    results = ResultAccumulator(ece_bins=ece_bins, accuracy_fn=acc_fn, ece_fn=ece_fn)

    subset_t = None
    if class_subset is not None:
        subset_t = torch.as_tensor(list(class_subset), dtype=torch.long,
                                   device=torch_device)

    best_history: List[float] = []
    lambda_used = None
    t0 = time.time()
    num_batches = 0

    # =========================== main loop (t = 1..T) ===========================
    for t, batch in enumerate(loader, start=1):
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("images", batch.get("x")))
            targets = batch.get("label", batch.get("labels", batch.get("y")))
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        images = images.to(torch_device, non_blocking=True)
        targets = targets.to(torch_device, non_blocking=True).long()

        # --- step 1: sample K prompt solutions p_k^(t) by Eqn. (6) ---
        candidates = np.asarray(optimizer.ask(K), dtype=np.float64)

        # Keep the shifting state identical for every candidate inside the batch:
        # d_t is formed once from mu_N(t-1) (Eqn. 8) and the EMA is updated once
        # per batch, after the best candidate has been selected (Eqn. 9).
        pre_state = shifter.state_dict() if shifter is not None else None

        values: List[float] = []
        cand_logits: List[torch.Tensor] = []
        cand_final: List[torch.Tensor] = []
        cand_terms: List[Dict[str, Any]] = []

        for k in range(candidates.shape[0]):
            prompt.set_prompt(torch.as_tensor(candidates[k], dtype=torch.float32))
            with torch.no_grad():
                out = model.forward_with_features(
                    images, prompt=prompt.as_tensor()
                )
                cls_features = out["cls_features"]
                final_cls = out["final_cls"]

                if shifter is not None:
                    # (first batch only) initialize mu_N(0) from this batch
                    if shifter.current_mean() is None or pre_state is None:
                        shifter.update(final_cls.detach())
                        pre_state = shifter.state_dict()
                    shifter.load_state_dict(pre_state)
                    # Eqn. (7): e_N^0 <- e_N^0 + gamma * (mu_N^S - mu_N(t))
                    final_shifted = shifter.shift(final_cls.detach())
                else:
                    final_shifted = final_cls.detach()
                # Head(e_N^0)
                logits = model.head(final_shifted)

            terms = fitness.evaluate(logits, cls_features)
            v = float(terms.total.item() if torch.is_tensor(terms.total) else terms.total)
            values.append(v)
            cand_logits.append(logits.detach())
            cand_final.append(final_cls.detach())
            cand_terms.append(terms.as_dict())

        # --- step 2: update m^(t), Sigma^(t), tau^(t) from {v_k} (CMA tell) ---
        optimizer.tell(values, candidates)

        # --- step 3: emit the prediction of the best-fitness candidate ---
        best = optimizer.best_index(values)
        logits_best = cand_logits[best]
        if subset_t is not None:
            logits_best = logits_best.index_select(-1, subset_t)
        b_acc, b_ece = results.update(logits_best, targets)
        best_history.append(float(values[best]))
        if lambda_used is None:
            lambda_used = float(getattr(terms, "lambda_value", 0.0) or
                                (cand_terms[0].get("lambda_value") if cand_terms else 0.0))

        # --- Eqn. (9): mu_N(t) = alpha*mu_N(X_t) + (1-alpha)*mu_N(t-1) ---
        if shifter is not None:
            if pre_state is not None:
                shifter.load_state_dict(pre_state)
            shifter.update(cand_final[best])

        num_batches += 1
        if verbose and (t % max(1, log_every) == 0):
            agg = results.compute()
            print(
                f"[FOA] batch {t:4d} | acc {agg['accuracy']:.2f} | ece {agg['ece']:.2f} "
                f"| best fitness {float(values[best]):.4f} | "
                f"mu_N drift {float(shifter.current_mean().abs().mean()):.4f}"
                if shifter is not None else
                f"[FOA] batch {t:4d} | acc {agg['accuracy']:.2f} | "
                f"best fitness {float(values[best]):.4f}",
                flush=True,
            )

    summary = results.compute()
    summary.update({
        "num_batches": num_batches,
        "population_size": K,
        "num_prompts": int(getattr(prompt, "num_prompts", 0)),
        "prompt_dim": int(getattr(prompt, "prompt_dim", 0)),
        "lambda": lambda_used,
        "shifting_enabled": shifter is not None,
        "gamma": float(getattr(shifter, "gamma", 0.0)) if shifter is not None else None,
        "alpha": float(getattr(shifter, "alpha", 0.0)) if shifter is not None else None,
        "corruption": corruption,
        "wall_clock_s": time.time() - t0,
        "per_batch": results.per_batch,
    })
    return summary


# ----------------------------------------------------------------------------
# Multi-corruption driver (Table 2 / Table 16 style reporting)
# ----------------------------------------------------------------------------
def run_over_corruptions(cfg: Config,
                         corruptions: Sequence[str],
                         device: Optional[str] = None,
                         verbose: bool = True,
                         limit_batches: Optional[int] = None) -> Dict[str, Any]:
    """Run FOA independently on each corruption and average (Table 2 format).

    A fresh prompt / CMA distribution / shifting state is used for every
    corruption, matching the per-corruption evaluation protocol.
    """
    data_cfg = cfg.get("data", {}) or {}
    severity = int(data_cfg.get("severity", 5))
    per_corruption: Dict[str, Dict[str, Any]] = {}
    for corruption in corruptions:
        if verbose:
            print(f"\n[FOA] === ImageNet-C | {corruption} | severity {severity} ===",
                  flush=True)
        _seed_everything(int(cfg.get("seed", 0) or 0))
        loader = build_test_loader(cfg, corruption=corruption, severity=severity)
        res = run_foa(cfg, loader=loader, device=device, verbose=verbose,
                      corruption=corruption)
        per_corruption[corruption] = {k: v for k, v in res.items() if k != "per_batch"}
        if verbose:
            print(f"[FOA] {corruption}: acc {res['accuracy']:.2f} "
                  f"| ece {res['ece']:.2f}", flush=True)

    accs = [v["accuracy"] for v in per_corruption.values()]
    eces = [v["ece"] for v in per_corruption.values()]
    return {
        "dataset": data_cfg.get("dataset", "imagenet-c"),
        "severity": severity,
        "num_corruptions": len(per_corruption),
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "ece": float(np.mean(eces)) if eces else 0.0,
        "per_corruption": per_corruption,
    }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FOA (Algorithm 1) on OOD data.")
    p.add_argument("--config", type=str, required=True,
                   help="Path to the YAML config (e.g. configs/foa_imagenetc.yaml).")
    p.add_argument("--extra-config", type=str, default=None,
                   help="Optional second YAML merged on top of --config.")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override data.dataset (imagenet-c/r/v2/sketch).")
    p.add_argument("--corruptions", type=str, nargs="*", default=None,
                   help="Subset of ImageNet-C corruptions to run.")
    p.add_argument("--severity", type=int, default=None, help="ImageNet-C severity.")
    p.add_argument("--all-corruptions", action="store_true",
                   help="Run all 15 ImageNet-C corruptions and average.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Where to dump the JSON results.")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="Only consume the first N batches (smoke tests).")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override model.checkpoint (npz / state_dict).")
    p.add_argument("--source-stats", type=str, default=None,
                   help="Override source_stats.path.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = [args.config] + ([args.extra_config] if args.extra_config else [])
    cfg = load_config(*paths)

    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
    if args.severity is not None:
        cfg["data"]["severity"] = args.severity
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["model"]["device"] = args.device
    if args.checkpoint is not None:
        cfg["model"]["checkpoint"] = args.checkpoint
    if args.source_stats is not None:
        cfg["source_stats"]["path"] = args.source_stats
    if args.seed is not None:
        cfg["seed"] = args.seed

    _seed_everything(int(cfg.get("seed", 0) or 0))
    verbose = not args.quiet

    data_cfg = cfg.get("data", {}) or {}
    dataset = str(data_cfg.get("dataset", "imagenet-c"))

    if dataset == "imagenet-c" and (args.all_corruptions or args.corruptions is None):
        corruptions = (list(data_cfg.get("corruptions") or IMAGENET_C_CORRUPTIONS)
                       if not args.all_corruptions
                       else list(data_cfg.get("corruptions") or IMAGENET_C_CORRUPTIONS))
        if args.corruptions:
            corruptions = list(args.corruptions)
        results = run_over_corruptions(cfg, corruptions, device=args.device,
                                       verbose=verbose,
                                       limit_batches=args.limit_batches)
    else:
        loader = build_test_loader(cfg, dataset=dataset,
                                   corruption=data_cfg.get("corruption"),
                                   limit_batches=args.limit_batches)
        results = run_foa(cfg, loader=loader, device=args.device, verbose=verbose,
                          log_every=args.log_every,
                          corruption=data_cfg.get("corruption"))

    if verbose:
        print("\n===== FOA summary =====")
        for k in ("accuracy", "ece", "num_samples", "num_batches", "population_size",
                  "num_prompts", "lambda", "wall_clock_s"):
            if k in results:
                print(f"  {k}: {results[k]}")
        if "per_corruption" in results:
            print("  per-corruption accuracy:")
            for c, v in results["per_corruption"].items():
                print(f"    {c:20s} acc {v['accuracy']:.2f} | ece {v['ece']:.2f}")

    out_path = args.output
    if out_path is None:
        out_dir = str(cfg.get("output_dir", "./outputs"))
        os.makedirs(out_dir, exist_ok=True)
        name = str((cfg.get("experiment", {}) or {}).get("name", "foa"))
        out_path = os.path.join(out_dir, f"{name}_results.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"config": config_to_dict(cfg), "results": results}, fh, indent=2)
    if verbose:
        print(f"[FOA] results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
