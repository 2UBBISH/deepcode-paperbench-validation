"""Non-i.i.d. evaluation driver for FOA (Section 4.4, Table 11).

Reproduces the robustness study of the FOA paper under two non-i.i.d. test
streams, following NOTE (Gong et al., 2022) and SAR (Niu et al., 2023):

* ``online label shifts``: the test data arrive in a *class order* (online
  imbalanced label distribution shift).  The reported number is the average
  accuracy / ECE over the 15 ImageNet-C corruptions (level 5).
* ``mixed shifts``: a *single* data stream composed of the 15 ImageNet-C
  corruptions in a random (seeded) order, i.e. multiple mixed domains with
  different distribution shifts inside one stream.
* ``mild`` (i.i.d.): the ordinary, single-corruption-pass ImageNet-C stream
  averaged over the 15 corruptions -- included here as the reference row of
  Table 11.

Reference (ViT-Base, ImageNet-C level 5, Table 11):

    ==================  ===========  ===========
    scenario            Acc. (%)     ECE (%)
    ==================  ===========  ===========
    FOA   mild          66.3         3.2
    FOA   label shift   62.1         6.6
    FOA   mixed shift   62.0         4.9
    TENT  mild          59.6         18.5
    TENT  label shift   60.2         17.7
    TENT  mixed shift   56.9         29.2
    SAR   mild          62.7         7.0
    SAR   label shift   60.8         7.5
    SAR   mixed shift   61.4         14.8
    ==================  ===========  ===========

The script re-uses :func:`run_foa.run_foa` (Algorithm 1) unchanged -- the only
difference is the stream: FOA is completely agnostic to the batch distribution
because it never updates model weights and its CMA state / feature statistics
are carried across batches in exactly the same way.

Baselines (TENT / SAR / ...) are optional: if the thin wrappers in
``src.baselines`` are importable they are run on the identical streams;
otherwise they are skipped with a warning so that the FOA numbers can still be
produced.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# make ``import src.*`` and sibling-script imports work when run directly
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_PROJECT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from src.utils.config import Config, load_config, config_to_dict, save_config  # noqa: E402

# ---------------------------------------------------------------------------
# reuse the Algorithm-1 implementation from run_foa.py
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import mechanics
    from run_foa import (  # type: ignore
        IMAGENET_C_CORRUPTIONS,
        ResultAccumulator,
        build_test_loader,
        run_foa,
        run_over_corruptions,
    )
except Exception:  # pragma: no cover - executed as a library
    from scripts.run_foa import (  # type: ignore
        IMAGENET_C_CORRUPTIONS,
        ResultAccumulator,
        build_test_loader,
        run_foa,
        run_over_corruptions,
    )


# ---------------------------------------------------------------------------
# reference numbers (Table 11) -- used for the printed comparison only
# ---------------------------------------------------------------------------
TABLE11_REFERENCE: Dict[str, Dict[str, Dict[str, float]]] = {
    "foa": {
        "mild_iid": {"accuracy": 66.3, "ece": 3.2},
        "online_label_shift": {"accuracy": 62.1, "ece": 6.6},
        "mixed_shifts": {"accuracy": 62.0, "ece": 4.9},
    },
    "tent": {
        "mild_iid": {"accuracy": 59.6, "ece": 18.5},
        "online_label_shift": {"accuracy": 60.2, "ece": 17.7},
        "mixed_shifts": {"accuracy": 56.9, "ece": 29.2},
    },
    "sar": {
        "mild_iid": {"accuracy": 62.7, "ece": 7.0},
        "online_label_shift": {"accuracy": 60.8, "ece": 7.5},
        "mixed_shifts": {"accuracy": 61.4, "ece": 14.8},
    },
}

SCENARIOS = ("mild", "label_shift", "mixed_shift", "all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _MaybeLimited:
    """Wrap a stream so that only ``limit`` batches are consumed.

    Mirrors the helper of the same name in ``run_foa.py`` (kept local so this
    script stays importable even if that private class moves).
    """

    def __init__(self, stream: Any, limit: Optional[int] = None):
        self.stream = stream
        self.limit = limit
        self.dataset = getattr(stream, "dataset", None)
        self.batch_size = getattr(stream, "batch_size", None)

    def __iter__(self):
        for i, batch in enumerate(self.stream):
            if self.limit is not None and i >= self.limit:
                break
            yield batch

    def __len__(self) -> int:
        try:
            n = len(self.stream)
        except Exception:
            return 0
        if self.limit is None:
            return n
        return min(n, self.limit)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Dotted-path read that works for both dict- and attribute-style configs."""
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _resolve_device(cfg: Any, device: Optional[str] = None) -> torch.device:
    if device is not None:
        dev = device
    else:
        dev = _cfg_get(cfg, "model", "device", default="cuda")
    dev = str(dev)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[run_non_iid] CUDA unavailable -- falling back to CPU")
        dev = "cpu"
    return torch.device(dev)


def _resolve_corruptions(cfg: Any, corruptions: Optional[Sequence[str]] = None) -> List[str]:
    if corruptions:
        return list(corruptions)
    cfg_corr = _cfg_get(cfg, "data", "corruptions", default=None)
    if cfg_corr:
        return list(cfg_corr)
    return list(IMAGENET_C_CORRUPTIONS)


# ---------------------------------------------------------------------------
# stream construction
# ---------------------------------------------------------------------------
def build_non_iid_loader(
    cfg: Any,
    scenario: str,
    corruption: Optional[str] = None,
    limit_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Any:
    """Build a single-pass non-i.i.d. stream for ``scenario``.

    ``scenario`` is one of ``"mild"`` (i.i.d., plain corruption stream),
    ``"label_shift"`` (class-ordered online imbalanced stream) or
    ``"mixed_shift"`` (all 15 corruptions randomly mixed into one stream).
    """
    from src.data.corruption_stream import build_non_iid_stream

    scenario = {
        "iid": "mild",
        "online_label_shift": "label_shift",
        "mixed_shifts": "mixed_shift",
    }.get(scenario, scenario)

    seed = int(_cfg_get(cfg, "seed", default=0))
    stream = build_non_iid_stream(
        cfg=cfg,
        scenario=scenario,
        corruption=corruption,
        device=device,
    )
    if seed is not None:
        pass  # seeding is handled inside the stream builder via cfg.seed
    return _MaybeLimited(stream, limit=limit_batches)


# ---------------------------------------------------------------------------
# FOA on one scenario
# ---------------------------------------------------------------------------
def run_foa_scenario(
    cfg: Any,
    scenario: str,
    corruptions: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    limit_batches: Optional[int] = None,
    class_subset: Optional[int] = None,
) -> Dict[str, Any]:
    """Run FOA on one non-i.i.d. scenario.

    * ``mild`` / ``label_shift``: average over the 15 corruptions (fresh
      prompt / CMA / shifting state per corruption, as in Table 2/11).
    * ``mixed_shift``: one single stream containing the 15 corruptions in a
      random order; a single FOA run.
    """
    dev = _resolve_device(cfg, device)
    scenario = {
        "iid": "mild",
        "online_label_shift": "label_shift",
        "mixed_shifts": "mixed_shift",
    }.get(scenario, scenario)

    t0 = time.time()

    if scenario == "mixed_shift":
        loader = build_non_iid_loader(
            cfg, "mixed_shift", corruption=None, limit_batches=limit_batches, device=dev
        )
        result = run_foa(
            cfg,
            loader=loader,
            device=dev,
            verbose=verbose,
            class_subset=class_subset,
            corruption="mixed",
        )
        result["scenario"] = "mixed_shift"
        result["wall_clock_s"] = result.get("wall_clock_s", time.time() - t0)
        return result

    # mild / label_shift -> per-corruption average
    corr_list = _resolve_corruptions(cfg, corruptions)
    per_corruption: Dict[str, Dict[str, Any]] = {}
    accs: List[float] = []
    eces: List[float] = []

    for corruption in corr_list:
        _seed_everything(int(_cfg_get(cfg, "seed", default=0)))
        loader = build_non_iid_loader(
            cfg, scenario, corruption=corruption, limit_batches=limit_batches, device=dev
        )
        res = run_foa(
            cfg,
            loader=loader,
            device=dev,
            verbose=verbose,
            class_subset=class_subset,
            corruption=corruption,
        )
        per_corruption[corruption] = {
            "accuracy": res["accuracy"],
            "ece": res["ece"],
            "num_samples": res["num_samples"],
        }
        accs.append(res["accuracy"])
        eces.append(res["ece"])
        if verbose:
            print(
                f"[run_non_iid][{scenario}][{corruption}] "
                f"acc={res['accuracy']:.2f} ece={res['ece']:.2f}"
            )

    mean_acc = float(sum(accs) / len(accs)) if accs else float("nan")
    mean_ece = float(sum(eces) / len(eces)) if eces else float("nan")
    return {
        "scenario": scenario,
        "accuracy": mean_acc,
        "ece": mean_ece,
        "num_corruptions": len(corr_list),
        "corruptions": list(corr_list),
        "per_corruption": per_corruption,
        "wall_clock_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# optional baselines
# ---------------------------------------------------------------------------
def _load_baseline(name: str):
    """Lazily import a baseline module from ``src.baselines``."""
    try:
        import importlib

        return importlib.import_module(f"src.baselines.{name}")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[run_non_iid] baseline '{name}' unavailable ({exc}) -- skipping")
        return None


def _baseline_step_fn(module: Any, name: str, cfg: Any, model: Any, device: Any):
    """Find a usable step/adapt callable inside a baseline wrapper module.

    The thin wrappers in ``src.baselines`` expose one of
    ``build_<name>`` / ``build`` factories returning an object with
    ``step(images)`` (TENT/SAR/CoTTA style) or ``adapt(logits, features)``
    (T3A/LAME style).  We probe for the common spellings and return ``None``
    when nothing matches, so the driver degrades gracefully.
    """
    factory = None
    for attr in (f"build_{name}", "build", "build_baseline", name):
        factory = getattr(module, attr, None)
        if callable(factory):
            break
    if factory is None:
        return None
    try:
        adapter = factory(model=model, cfg=cfg, device=device)
    except TypeError:
        try:
            adapter = factory(cfg=cfg, device=device)
        except TypeError:
            try:
                adapter = factory(model, cfg)
            except TypeError:
                return None
    for attr in ("step", "adapt", "forward_step", "predict"):
        fn = getattr(adapter, attr, None)
        if callable(fn):
            return fn if attr == "step" else fn
    return None


def run_baseline_scenario(
    name: str,
    cfg: Any,
    scenario: str,
    corruptions: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    limit_batches: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate a gradient-based baseline (TENT/SAR) on a non-i.i.d. stream.

    Returns ``None`` (and prints a warning) when the corresponding wrapper is
    not importable -- the paper reports these numbers from the official repos.
    """
    module = _load_baseline(name)
    if module is None:
        return None
    try:
        from src.models.vit_loader import build_vit
    except Exception:  # pragma: no cover
        return None

    dev = _resolve_device(cfg, device)
    scenario = {
        "iid": "mild",
        "online_label_shift": "label_shift",
        "mixed_shifts": "mixed_shift",
    }.get(scenario, scenario)

    def _run_one(loader) -> Dict[str, Any]:
        model = build_vit(
            model_name=_cfg_get(cfg, "model", "name", default="vit_base_patch16_224"),
            checkpoint=_cfg_get(cfg, "model", "checkpoint", default=None),
            pretrained=bool(_cfg_get(cfg, "model", "pretrained", default=True)),
            num_classes=int(_cfg_get(cfg, "model", "num_classes", default=1000)),
            device=str(dev),
        )
        step_fn = _baseline_step_fn(module, name, cfg, model, dev)
        acc = ResultAccumulator(ece_bins=int(_cfg_get(cfg, "eval", "ece_bins", default=15)))
        if step_fn is None:
            return {"accuracy": float("nan"), "ece": float("nan"), "num_samples": 0}
        n = 0
        for batch in loader:
            images, labels = _batch_images_labels(batch, dev)
            with torch.no_grad():
                out = step_fn(images)
            logits = out["logits"] if isinstance(out, dict) else out
            if isinstance(logits, tuple):
                logits = logits[0]
            acc.update(logits, labels)
            n += int(labels.shape[0])
        res = acc.compute()
        res["num_samples"] = n
        return res

    t0 = time.time()
    if scenario == "mixed_shift":
        loader = build_non_iid_loader(cfg, scenario, limit_batches=limit_batches, device=dev)
        out = _run_one(loader)
        out.update({"scenario": scenario, "wall_clock_s": time.time() - t0})
        return out

    corr_list = _resolve_corruptions(cfg, corruptions)
    accs, eces = [], []
    per_corruption: Dict[str, Any] = {}
    for corruption in corr_list:
        _seed_everything(int(_cfg_get(cfg, "seed", default=0)))
        loader = build_non_iid_loader(
            cfg, scenario, corruption=corruption, limit_batches=limit_batches, device=dev
        )
        res = _run_one(loader)
        per_corruption[corruption] = res
        accs.append(res["accuracy"])
        eces.append(res["ece"])
    return {
        "scenario": scenario,
        "accuracy": float(sum(accs) / len(accs)) if accs else float("nan"),
        "ece": float(sum(eces) / len(eces)) if eces else float("nan"),
        "num_corruptions": len(corr_list),
        "per_corruption": per_corruption,
        "wall_clock_s": time.time() - t0,
    }


def _batch_images_labels(batch: Any, device: Any):
    """Normalize a batch dict/sequence into (images, labels) tensors."""
    if isinstance(batch, dict):
        images = batch.get("image", batch.get("images", batch.get("x")))
        labels = batch.get("label", batch.get("labels", batch.get("y")))
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)!r}")
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return images, labels


# ---------------------------------------------------------------------------
# top level driver
# ---------------------------------------------------------------------------
def run_all_scenarios(
    cfg: Any,
    scenarios: Sequence[str] = ("mild", "label_shift", "mixed_shift"),
    methods: Sequence[str] = ("foa",),
    corruptions: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Run every requested (method, scenario) combination."""
    results: Dict[str, Dict[str, Any]] = {m: {} for m in methods}
    for method in methods:
        for scenario in scenarios:
            if verbose:
                print(f"\n=== {method.upper()} | scenario={scenario} ===")
            if method == "foa":
                res = run_foa_scenario(
                    cfg,
                    scenario,
                    corruptions=corruptions,
                    device=device,
                    verbose=verbose,
                    limit_batches=limit_batches,
                )
            else:
                res = run_baseline_scenario(
                    method,
                    cfg,
                    scenario,
                    corruptions=corruptions,
                    device=device,
                    verbose=verbose,
                    limit_batches=limit_batches,
                )
            if res is None:
                continue
            key = _scenario_key(scenario)
            res["key"] = key
            ref = TABLE11_REFERENCE.get(method, {}).get(key)
            if ref:
                res["paper_reference"] = ref
            results[method][key] = res
    return results


def _scenario_key(scenario: str) -> str:
    from src.data.corruption_stream import non_iid_summary_key

    return non_iid_summary_key(scenario)


def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a Table-11-style summary."""
    keys = ["mild_iid", "online_label_shift", "mixed_shifts"]
    header = f"{'method':<10}" + "".join(f"{k:>22}" for k in keys)
    print("\n" + "=" * len(header))
    print("Table 11 reproduction -- non-i.i.d. robustness (ImageNet-C level 5, ViT-Base)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for method, by_key in results.items():
        row = f"{method:<10}"
        for k in keys:
            r = by_key.get(k)
            if r is None:
                row += f"{'--':>22}"
            else:
                row += f"{r['accuracy']:>10.2f} / {r['ece']:>9.2f}"
        print(row)
    print("-" * len(header))
    for method in ("foa", "tent", "sar"):
        ref = TABLE11_REFERENCE.get(method)
        if not ref:
            continue
        row = f"{method + ' (paper)':<10}"
        for k in keys:
            r = ref.get(k)
            row += f"{'--':>22}" if r is None else f"{r['accuracy']:>10.2f} / {r['ece']:>9.2f}"
        print(row)
    print("=" * len(header))
    print("cells are 'Acc. (%) / ECE (%)'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FOA non-i.i.d. robustness evaluation (Table 11)"
    )
    p.add_argument("--config", type=str, default="configs/non_iid.yaml",
                   help="base YAML config (default: configs/non_iid.yaml)")
    p.add_argument("--extra-config", type=str, action="append", default=None,
                   help="additional YAML config(s) merged on top of --config")
    p.add_argument("--scenarios", type=str, default="all",
                   help="comma separated subset of {mild,label_shift,mixed_shift} or 'all'")
    p.add_argument("--methods", type=str, default="foa",
                   help="comma separated methods, e.g. 'foa' or 'foa,tent,sar'")
    p.add_argument("--corruptions", type=str, default=None,
                   help="comma separated corruption names (default: all 15)")
    p.add_argument("--severity", type=int, default=None, help="ImageNet-C severity (default 5)")
    p.add_argument("--device", type=str, default=None, help="torch device, e.g. cuda / cpu")
    p.add_argument("--seed", type=int, default=None, help="override global seed")
    p.add_argument("--output", type=str, default=None, help="JSON output path")
    p.add_argument("--limit-batches", type=int, default=None,
                   help="debug: cap the number of batches per stream")
    p.add_argument("--source-stats", type=str, default=None, help="override source stats path")
    p.add_argument("--quiet", action="store_true", help="reduce per-corruption logging")
    return p.parse_args(argv)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.severity is not None:
        cfg.setdefault("data", Config())
        cfg["data"]["severity"] = int(args.severity)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
        cfg.setdefault("cma", Config())
        cfg["cma"].setdefault("seed", int(args.seed))
    if args.source_stats is not None:
        cfg.setdefault("source_stats", Config())
        cfg["source_stats"]["path"] = args.source_stats
    if args.device is not None:
        cfg.setdefault("model", Config())
        cfg["model"]["device"] = args.device
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    paths = [args.config] if args.config else []
    if args.extra_config:
        paths.extend(args.extra_config)
    if paths:
        cfg = load_config(*paths)
    else:  # pragma: no cover - config defaults
        from src.utils.config import FOA_DEFAULTS

        cfg = Config(FOA_DEFAULTS)
    cfg = _apply_overrides(cfg, args)

    seed = int(_cfg_get(cfg, "seed", default=0))
    _seed_everything(seed)

    if args.scenarios.strip().lower() in ("all", ""):
        scenarios = ["mild", "label_shift", "mixed_shift"]
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    corruptions = (
        [c.strip() for c in args.corruptions.split(",")] if args.corruptions else None
    )

    verbose = not args.quiet
    results = run_all_scenarios(
        cfg,
        scenarios=scenarios,
        methods=methods,
        corruptions=corruptions,
        device=args.device,
        verbose=verbose,
        limit_batches=args.limit_batches,
    )

    print_summary(results)

    out_dir = _cfg_get(cfg, "output_dir", default="./outputs/non_iid")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(
        out_dir, f"{_cfg_get(cfg, 'experiment', 'name', default='foa_non_iid')}_results.json"
    )
    payload = {
        "config": config_to_dict(cfg),
        "results": results,
        "paper_reference": TABLE11_REFERENCE,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[run_non_iid] results written to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
