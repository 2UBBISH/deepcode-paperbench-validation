"""Command line interface.

Examples
--------
::

    # 1) source in-distribution statistics (32 unlabelled ImageNet val images)
    python -m foa.cli source-stats --out stats/vit_b16_in1k.pt

    # 2) one (method, corruption) evaluation
    python -m foa.cli evaluate --method FOA --dataset imagenet_c \
        --corruption gaussian_noise --severity 5 --data-root data/imagenet-c

    # 3) a full table of the paper
    python -m foa.cli table --table 2 --data-root data/imagenet-c --stats stats/vit_b16_in1k.pt

Every command writes a JSON file under ``results/`` so that the numbers can be compared
with the tables of the paper (see ``scripts/`` for ready-made wrappers).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import torch

from . import experiments as E
from .config import BaselineConfig, FOAConfig, foa_config_for_dataset
from .core.statistics import FeatureStatistics, compute_source_statistics
from .data import datasets as D
from .data import download as DL
from .evaluation.metrics import summarize
from .evaluation.runner import run_stream
from .models.prompt_vit import PromptViT


# --------------------------------------------------------------------------------------
def _device(args) -> torch.device:
    if getattr(args, "cpu", False):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stream_cfg(args, batch_size: Optional[int] = None, order: str = "random"):
    return D.StreamConfig(
        batch_size=batch_size or getattr(args, "batch_size", 64),
        shuffle=getattr(args, "shuffle", True),
        seed=getattr(args, "seed", 0),
        num_workers=getattr(args, "num_workers", 4),
        max_samples=getattr(args, "max_samples", None),
        order=order,
    )


def _save(results: dict, out: Optional[str]) -> None:
    text = json.dumps(results, indent=2)
    print(text)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as fh:
            fh.write(text)


def _load_stats(path: str, device) -> FeatureStatistics:
    stats = FeatureStatistics.load(path, map_location="cpu")
    return stats.to(device)


def _foa_cfg(args, dataset: str, **overrides) -> FOAConfig:
    cfg = foa_config_for_dataset(dataset)
    cfg.batch_size = getattr(args, "batch_size", 64)
    cfg.num_prompts = getattr(args, "num_prompts", 3)
    cfg.popsize = getattr(args, "popsize", 28)
    cfg.cma_seed = getattr(args, "seed", 0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------------------
def cmd_source_stats(args) -> None:
    device = _device(args)
    model = E.ModelSpec(
        num_prompts=0,
        checkpoint=args.checkpoint,
        prompt_pos=args.prompt_pos,
    ).build(device)
    transform = D.build_eval_transform(model, img_size=args.img_size)
    batches = E.source_statistics_batches(
        args.data_root,
        transform,
        num_samples=args.num_samples,
        seed=args.seed,
        hf_cache=args.hf_cache,
        use_hf=args.hf,
        synthetic=args.synthetic,
        img_size=args.img_size,
    )
    batches = [b.to(device) for b in batches]
    stats = compute_source_statistics(model, batches, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    stats.save(args.out)
    print(f"[source-stats] saved {stats.num_layers} layers to {args.out}")


def _evaluate_single(args, dataset: str, method: str, model, stats, corruption=None, extra_cfg=None):
    device = _device(args)
    transform = D.build_eval_transform(model, img_size=args.img_size)
    use_synthetic = getattr(args, "synthetic", False)
    loader = E.build_test_loader(
        "synthetic" if use_synthetic else dataset,
        args.data_root,
        transform,
        _stream_cfg(args, order=getattr(args, "order", "random")),
        corruption=corruption,
        severity=getattr(args, "severity", 5),
        hf_cache=args.hf_cache,
        use_hf=getattr(args, "hf", False) and dataset != "imagenet_c",
    )
    if method.lower() == "foa" and extra_cfg is not None and "interval" in extra_cfg:
        cfg = _foa_cfg(args, dataset, **extra_cfg)
    else:
        cfg = _foa_cfg(args, dataset, **(extra_cfg or {}))
    built = E.build_method(method, model, device, source_stats=stats, foa_cfg=cfg)
    result = run_stream(built, loader, max_samples=args.max_samples)
    out = result.to_dict()
    out["method"] = method
    out["dataset"] = dataset
    if corruption:
        out["corruption"] = corruption
        out["severity"] = args.severity
    return out


def cmd_evaluate(args) -> None:
    device = _device(args)
    model = E.ModelSpec(
        num_prompts=args.num_prompts,
        checkpoint=args.checkpoint,
        prompt_pos=args.prompt_pos,
    ).build(device)
    stats = _load_stats(args.stats, device) if args.stats else None
    res = _evaluate_single(args, args.dataset, args.method, model, stats, corruption=args.corruption)
    _save(res, args.out)


def cmd_table(args) -> None:
    device = _device(args)
    table = str(args.table)
    results: Dict[str, dict] = {}
    if table == "2":
        results = table2(args, device)
    elif table == "3":
        results = table3(args, device)
    elif table == "4":
        results = table4(args, device)
    elif table == "5":
        results = table5(args, device)
    elif table == "6":
        results = table6(args, device)
    elif table == "9":
        results = table9(args, device)
    elif table == "11":
        results = table11(args, device)
    elif table == "10":
        results = table10(args, device)
    else:
        raise SystemExit(f"table {table} is not implemented")
    _save(results, args.out)


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def _summaries(results: Dict[str, dict]) -> Dict[str, float]:
    accs = [v["acc"] for v in results.values() if v.get("acc") is not None]
    eces = [v["ece"] for v in results.values() if v.get("ece") is not None]
    return {
        "acc_mean": sum(accs) / len(accs) if accs else float("nan"),
        "ece_mean": sum(eces) / len(eces) if eces else float("nan"),
        "num": len(accs),
    }


def _run_method_over_corruptions(args, device, method, model, stats, update_model=None):
    out = {}
    for corruption in D.IMAGENET_C_CORRUPTIONS:
        if update_model is not None:
            model = update_model()
        res = _evaluate_single(args, "imagenet_c", method, model, stats, corruption=corruption)
        out[corruption] = res
        print(
            f"[table2] {method:8s} {corruption:18s} acc={res['acc']:.2f} ece={res['ece']:.2f}",
            flush=True,
        )
    out["average"] = _summaries({k: v for k, v in out.items() if k != "average"})
    return out


def table2(args, device) -> Dict[str, dict]:
    """Table 2: ImageNet-C level 5, full-precision ViT-Base."""
    methods = args.methods.split(",") if args.methods else ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA"]
    stats = _load_stats(args.stats, device) if args.stats else None
    results = {}
    for method in methods:
        model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
        results[method] = _run_method_over_corruptions(args, device, method, model, stats)
    return results


def table3(args, device) -> Dict[str, dict]:
    """Table 3: ImageNet-R / V2 / Sketch, full-precision ViT-Base."""
    methods = args.methods.split(",") if args.methods else ["NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "FOA"]
    stats = _load_stats(args.stats, device) if args.stats else None
    results = {}
    for dataset in ["imagenet_r", "imagenet_v2", "imagenet_sketch"]:
        for method in methods:
            model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
            res = _evaluate_single(args, dataset, method, model, stats)
            results.setdefault(method, {})[dataset] = res
            print(f"[table3] {dataset:16s} {method:8s} acc={res['acc']:.2f} ece={res['ece']:.2f}", flush=True)
        accs = [results[m][dataset]["acc"] for m in methods]
        eces = [results[m][dataset]["ece"] for m in methods]
        results.setdefault("average", {})[dataset] = {
            "acc": sum(accs) / len(accs),
            "ece": sum(eces) / len(eces),
        }
    return results


def table4(args, device) -> Dict[str, dict]:
    """Table 4: quantised (8-bit / 6-bit) ViT-Base on ImageNet-C level 5."""
    methods = args.methods.split(",") if args.methods else ["NoAdapt", "T3A", "FOA"]
    bits_list = [int(b) for b in args.bits.split(",")] if args.bits else [8, 6]
    stats_cache = {}
    results = {}
    for bits in bits_list:
        for method in methods:
            model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
            from .quantization import calibrate, quantize_vit

            # PTQ4ViT calibration with 32 training samples
            cal_batches = E.source_statistics_batches(
                args.data_root, D.build_eval_transform(model, args.img_size),
                num_samples=32, seed=args.seed, hf_cache=args.hf_cache, use_hf=args.hf,
            )
            quantize_vit(model.vit, bits=bits)
            calibrate(model.vit, [b.to(device) for b in cal_batches], search=True)
            if args.stats:
                key = f"{bits}"
                if key not in stats_cache:
                    stats_cache[key] = _load_stats(args.stats, device)
                stats = stats_cache[key]
            else:
                stats = None
            results.setdefault(f"{bits}-bit", {})[method] = _run_method_over_corruptions(
                args, device, method, model, stats
            )
    return results


def table5(args, device) -> Dict[str, dict]:
    """Table 5: ablations of FOA's components on ImageNet-C level 5."""
    settings = {
        "NoAdapt": dict(method="NoAdapt", cfg={}),
        "Entropy only": dict(
            method="FOA", cfg=dict(use_entropy=True, use_activation_discrepancy=False, use_activation_shifting=False)
        ),
        "Act. Discrepancy only": dict(
            method="FOA", cfg=dict(use_entropy=False, use_activation_discrepancy=True, use_activation_shifting=False)
        ),
        "Act. Shifting only": dict(
            method="NoAdapt", cfg={}, shifting=True
        ),
        "Act. Disc. + Shifting": dict(
            method="FOA", cfg=dict(use_entropy=False, use_activation_discrepancy=True, use_activation_shifting=True)
        ),
        "Entropy + Act. Disc.": dict(
            method="FOA", cfg=dict(use_entropy=True, use_activation_discrepancy=True, use_activation_shifting=False)
        ),
        "FOA (full)": dict(
            method="FOA", cfg=dict(use_entropy=True, use_activation_discrepancy=True, use_activation_shifting=True)
        ),
    }
    stats = _load_stats(args.stats, device) if args.stats else None
    results = {}
    for name, spec in settings.items():
        model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
        if spec.get("shifting"):
            res = _no_adapt_with_shifting(args, device, model, stats)
            results[name] = res
            continue
        per_corruption = {}
        for corruption in D.IMAGENET_C_CORRUPTIONS:
            per_corruption[corruption] = _evaluate_single(
                args, "imagenet_c", spec["method"], model, stats, corruption=corruption, extra_cfg=spec["cfg"]
            )
        per_corruption["average"] = _summaries(per_corruption)
        results[name] = per_corruption
        print(f"[table5] {name:24s} acc={per_corruption['average']['acc_mean']:.2f}", flush=True)
    return results


def _no_adapt_with_shifting(args, device, model, stats):
    """The "Act. Shifting only" row of Table 5 - the 5th line of Table 5."""
    from .core.activation_shift import BackToSourceShifting

    transform = D.build_eval_transform(model, args.img_size)
    out = {}
    for corruption in D.IMAGENET_C_CORRUPTIONS:
        loader = E.build_test_loader(
            "synthetic" if getattr(args, "synthetic", False) else "imagenet_c",
            args.data_root,
            transform,
            _stream_cfg(args),
            corruption=corruption,
            severity=args.severity,
        )
        shifting = BackToSourceShifting(stats.means[-1].clone(), alpha=0.1, gamma=1.0)

        class _ShiftOnly:
            def __init__(self):
                self.last_extra = {}

            def reset(self):
                shifting.reset()

            @torch.no_grad()
            def step(self, images):
                images = images.to(device)
                means, _ = model.cls_statistics(images, prompt=None)
                shift = shifting.update(means[-1])
                logits, _, _ = model.forward_with_prompt(images, shift=shift)
                return logits

        res = run_stream(_ShiftOnly(), loader, max_samples=args.max_samples)
        out[corruption] = res.to_dict()
    out["average"] = _summaries({k: v for k, v in out.items() if k != "average"})
    return out


def table6(args, device) -> Dict[str, dict]:
    """Table 6: FOA-I (interval update, batch size 1) on ImageNet-C gaussian noise level 5."""
    stats = _load_stats(args.stats, device) if args.stats else None
    intervals = [int(i) for i in args.intervals.split(",")] if args.intervals else [4, 8, 16, 32, 64]
    results = {}
    # NoAdapt
    model = E.ModelSpec(
        num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
    ).build(device)
    results["NoAdapt"] = _evaluate_single(
        args, "imagenet_c", "NoAdapt", model, stats, corruption="gaussian_noise"
    )
    # TENT with batch size 64
    model = E.ModelSpec(
        num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
    ).build(device)
    args.batch_size = 64
    results["TENT (BS=64)"] = _evaluate_single(
        args, "imagenet_c", "TENT", model, stats, corruption="gaussian_noise"
    )
    args.batch_size = 1
    for interval in intervals:
        for store in (["feature"] if args.interval_store == "auto" else [args.interval_store]):
            model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
            name = f"FOA-I (I={interval}, {store})"
            results[name] = _evaluate_single(
                args,
                "imagenet_c",
                "FOA",
                model,
                stats,
                corruption="gaussian_noise",
                extra_cfg={"interval": interval, "interval_store": store},
            )
            print(f"[table6] {name:26s} acc={results[name]['acc']:.2f} ece={results[name]['ece']:.2f}", flush=True)
    return results


def table9(args, device) -> Dict[str, dict]:
    """Table 9: design choices (learnable parameters x optimiser x loss)."""
    from .methods.variants import CMANormAdapt, SGDAdapt

    stats = _load_stats(args.stats, device) if args.stats else None
    variants = [
        ("TENT (norm, SGD, entropy)", lambda m: None),  # handled by build_method
        ("exp1 (prompts, SGD, entropy)", lambda m, s: SGDAdapt(m, s, "prompts", "entropy", lr=0.01)),
        ("exp2 (norm, SGD, Eqn.5)", lambda m, s: SGDAdapt(m, s, "norm", "eqn5")),
        ("exp3 (prompts, SGD, Eqn.5)", lambda m, s: SGDAdapt(m, s, "prompts", "eqn5")),
        ("exp4 (norm, CMA, Eqn.5)", lambda m, s: CMANormAdapt(m, s, "eqn5")),
        ("exp5 (norm, CMA, entropy)", lambda m, s: CMANormAdapt(m, s, "entropy")),
        ("exp6 (prompts, CMA, entropy)", lambda m, s: _foa_config_variant(args, s, "entropy")),
        ("Ours (prompts, CMA, Eqn.5)", lambda m, s: _foa_config_variant(args, s, "eqn5")),
    ]
    results = {}
    for name, factory in variants:
        model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
        per_corruption = {}
        for corruption in D.IMAGENET_C_CORRUPTIONS:
            transform = D.build_eval_transform(model, args.img_size)
            loader = E.build_test_loader(
                "imagenet_c", args.data_root, transform, _stream_cfg(args), corruption=corruption, severity=args.severity
            )
            if name.startswith("TENT"):
                method = E.build_method("TENT", model, device)
            else:
                method = factory(model, stats)
            res = run_stream(method, loader, max_samples=args.max_samples)
            per_corruption[corruption] = res.to_dict()
        per_corruption["average"] = _summaries(per_corruption)
        results[name] = per_corruption
        print(f"[table9] {name:30s} acc={per_corruption['average']['acc_mean']:.2f} ece={per_corruption['average']['ece_mean']:.2f}", flush=True)
    return results


def _foa_config_variant(args, stats, loss: str):
    from .methods import FOAMethod

    cfg = _foa_cfg(
        args,
        "imagenet_c",
        use_entropy=True,
        use_activation_discrepancy=(loss == "eqn5"),
        use_activation_shifting=False,
    )
    device = _device(args)
    model = E.ModelSpec(
        num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
    ).build(device)
    return FOAMethod(model, stats, cfg=cfg, device=device)


def table11(args, device) -> Dict[str, dict]:
    """Table 11: non-i.i.d. scenarios (online label shift / mixed domains)."""
    stats = _load_stats(args.stats, device) if args.stats else None
    results = {}
    for method in args.methods.split(",") if args.methods else ["TENT", "SAR", "FOA"]:
        per_scenario = {}
        for scenario, order in (("mild", "random"), ("online label shift", "dirichlet")):
            accs, eces = [], []
            for corruption in D.IMAGENET_C_CORRUPTIONS:
                model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
                args.order = order
                res = _evaluate_single(args, "imagenet_c", method, model, stats, corruption=corruption)
                accs.append(res["acc"])
                eces.append(res["ece"])
            per_scenario[scenario] = {
                "acc": sum(accs) / len(accs),
                "ece": sum(eces) / len(eces),
            }
        # mixed domain shift: one stream made of all 15 corruptions
        args.order = "random"
        model = E.ModelSpec(
            num_prompts=args.num_prompts, checkpoint=args.checkpoint, prompt_pos=args.prompt_pos
        ).build(device)
        per_scenario["mixed shifts"] = _evaluate_mixed(args, device, method, model, stats)
        results[method] = per_scenario
    return results


def table10(args, device) -> Dict[str, dict]:
    """Table 10: FOA on ResNet-50 and VisionMamba (ImageNet-C gaussian noise, level 5).

    ``FOA-dagger`` replaces the CMA optimiser of FOA with SGD on the affine parameters of
    the normalisation layers while keeping the fitness function of Eqn. (5).
    """
    from .methods.variants import SGDAdapt

    stats = _load_stats(args.stats, device) if args.stats else None
    backbones = args.backbones.split(",") if args.backbones else ["resnet50", "visionmamba"]
    methods = args.methods.split(",") if args.methods else [
        "NoAdapt", "BNAdapt", "TENT", "SAR", "FOA", "FOA-dagger",
    ]
    results = {}
    for backbone in backbones:
        for method in methods:
            model = E.ModelSpec(
                num_prompts=args.num_prompts, backbone=backbone
            ).build(device)
            if method.lower() in ("foa-dagger", "foadagger", "foa_dagger"):
                built = SGDAdapt(model, stats, params="norm", loss="eqn5", lr=1e-3)
            else:
                built = E.build_method(
                    method, model, device, source_stats=stats, foa_cfg=_foa_cfg(args, "imagenet_c")
                )
            transform = D.build_eval_transform(model, args.img_size)
            loader = E.build_test_loader(
                "imagenet_c",
                args.data_root,
                transform,
                _stream_cfg(args),
                corruption=args.corruption,
                severity=args.severity,
                hf_cache=args.hf_cache,
                use_hf=args.hf,
            )
            res = run_stream(built, loader, max_samples=args.max_samples)
            results.setdefault(backbone, {})[method] = res.to_dict()
            print(
                f"[table10] {backbone:12s} {method:12s} acc={res.acc:.2f} ece={res.ece:.2f}",
                flush=True,
            )
    return results


def _evaluate_mixed(args, device, method, model, stats):
    transform = D.build_eval_transform(model, args.img_size)
    loaders = [
        E.build_test_loader(
            "imagenet_c", args.data_root, transform, _stream_cfg(args), corruption=c, severity=args.severity
        )
        for c in D.IMAGENET_C_CORRUPTIONS
    ]
    mixed = D.MixedDomainStream(loaders, seed=args.seed)
    cfg = _foa_cfg(args, "imagenet_c")
    built = E.build_method(method, model, device, source_stats=stats, foa_cfg=cfg)
    res = run_stream(built, mixed, max_samples=args.max_samples)
    return res.to_dict()


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--checkpoint", default="vit_base_patch16_224.augreg_in21k_ft_in1k")
        p.add_argument("--num-prompts", type=int, default=3)
        p.add_argument("--prompt-pos", default="zero")
        p.add_argument("--img-size", type=int, default=224)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-samples", type=int, default=None)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--out", default=None, help="where to write the JSON results")

    p = sub.add_parser("source-stats", help="estimate the source in-distribution statistics")
    common(p)
    p.add_argument("--data-root", default="data/imagenet")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--hf", action="store_true", default=True)
    p.add_argument("--no-hf", dest="hf", action="store_false")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--synthetic", action="store_true", help="use random images (smoke tests)")
    p.set_defaults(func=cmd_source_stats)

    p = sub.add_parser("evaluate", help="run one method on one domain")
    common(p)
    p.add_argument("--method", default="FOA")
    p.add_argument("--dataset", default="imagenet_c")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--data-root", default="data")
    p.add_argument("--stats", default="stats/vit_b16_in1k.pt")
    p.add_argument("--popsize", type=int, default=28)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--hf", action="store_true")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--synthetic", action="store_true", help="use random images (smoke tests)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("table", help="reproduce a table of the paper")
    common(p)
    p.add_argument("--table", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--stats", default="stats/vit_b16_in1k.pt")
    p.add_argument("--methods", default=None)
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--popsize", type=int, default=28)
    p.add_argument("--bits", default=None)
    p.add_argument("--intervals", default=None)
    p.add_argument("--interval-store", default="auto")
    p.add_argument("--backbones", default=None)
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--hf", action="store_true")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--synthetic", action="store_true", help="use random images (smoke tests)")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("download", help="download the benchmarks")
    p.add_argument("--data-root", default="data")
    p.add_argument("--only", default="all")
    p.set_defaults(func=cmd_download)
    return parser


def cmd_download(args) -> None:
    root = args.data_root
    if args.only in ("all", "imagenet_c"):
        DL.download_imagenet_c(os.path.join(root, "imagenet-c"))
    if args.only in ("all", "imagenet_r"):
        DL.download_imagenet_r(os.path.join(root, "imagenet-r"))
    if args.only in ("all", "imagenet_v2"):
        DL.download_imagenet_v2(os.path.join(root, "imagenet-v2"))
    if args.only in ("all", "imagenet_sketch"):
        DL.download_imagenet_sketch(os.path.join(root, "imagenet-sketch"))
    if args.only in ("all", "imagenet"):
        DL.download_imagenet1k_val()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
