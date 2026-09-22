"""Command line entry point: run one cell of the paper's tables.

Examples
--------

.. code-block:: bash

    # Table 1: SMM with ResNet-18 on CIFAR-10 (Ilm output mapping)
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm

    # Table 1 baselines (shared masks)
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method full
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method narrow
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method medium
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method pad

    # Table 2: ViT-B/32
    python -m smm.main --config configs/vitb32.yaml --dataset cifar10 --method smm

    # Table 3 ablations
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method only_delta
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method only_mask
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method single_channel

    # Figure 4 (patch size 2**l)
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm --num-pool-layers 2

    # Appendix D.1 (other output mappings)
    python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm --label-mapping flm
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from typing import Any, Dict, Optional

from .datasets import DATASET_SPECS
from .train import TrainConfig, save_result, train


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("PyYAML is required to read YAML configs") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SMM / visual reprogramming training")
    parser.add_argument("--config", default=None, help="YAML or JSON config file")
    parser.add_argument("--dataset", default=None, choices=sorted(DATASET_SPECS))
    parser.add_argument("--backbone", default=None,
                        choices=["resnet18", "resnet50", "vit_b32"])
    parser.add_argument("--method", default=None,
                        choices=["smm", "full", "narrow", "medium", "pad",
                                 "only_delta", "only_mask", "single_channel"])
    parser.add_argument("--label-mapping", default=None, choices=["ilm", "flm", "rlm"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-pool-layers", type=int, default=None,
                        help="l: patch size 2**l (Figure 4 sweep uses l in {0,1,2,3,4})")
    parser.add_argument("--arch", default=None, choices=["torchvision", "timm_384"],
                        help="ViT-B/32 checkpoint: torchvision (resized pos-embed) or timm 384")
    parser.add_argument("--pad-width", type=int, default=None,
                        help="border width of the Pad baseline (default 28)")
    parser.add_argument("--border-width", type=int, default=None,
                        help="width of the Narrow mask (default 28)")
    parser.add_argument("--medium-width", type=int, default=None,
                        help="width of the Medium mask (default 56)")
    parser.add_argument("--hidden-channels", default=None,
                        help="comma separated mask generator widths, e.g. 16,32,32,32")
    parser.add_argument("--train-subset", type=int, default=None,
                        help="train on only N samples (fast sanity runs)")
    parser.add_argument("--test-subset", type=int, default=None,
                        help="evaluate on only N samples (fast sanity runs)")
    parser.add_argument("--save-masks-every", type=int, default=None,
                        help="store mask statistics every N epochs")
    parser.add_argument("--lr-delta", type=float, default=None)
    parser.add_argument("--lr-mask", type=float, default=None)
    parser.add_argument("--gamma-delta", type=float, default=None)
    parser.add_argument("--gamma-mask", type=float, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ilm-every", type=int, default=None,
                        help="refresh the Ilm mapping every N epochs (default 1)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="use randomly initialised backbones (smoke tests only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the model and one batch, then exit")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    values: Dict[str, Any] = load_config(args.config)
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    overrides = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "method": args.method,
        "label_mapping": args.label_mapping,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_layers": args.num_layers,
        "num_pool_layers": args.num_pool_layers,
        "arch": args.arch,
        "pad_width": args.pad_width,
        "border_width": args.border_width,
        "medium_width": args.medium_width,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "save_masks_every": args.save_masks_every,
        "lr_delta": args.lr_delta,
        "lr_mask": args.lr_mask,
        "gamma_delta": args.gamma_delta,
        "gamma_mask": args.gamma_mask,
        "data_root": args.data_root,
        "split_dir": args.split_dir,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "device": args.device,
        "num_workers": args.num_workers,
        "eval_every": args.eval_every,
        "ilm_every_n_epochs": args.ilm_every,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    if args.no_pretrained:
        values["pretrained"] = False
    if args.hidden_channels:
        if args.hidden_channels.strip().lower() in {"none", "default", ""}:
            values["hidden_channels"] = None
        else:
            values["hidden_channels"] = [
                int(c) for c in args.hidden_channels.replace(" ", "").split(",") if c
            ]
    if "milestones" in values and isinstance(values["milestones"], (list, tuple)):
        values["milestones"] = tuple(int(m) for m in values["milestones"])
    return TrainConfig(**values)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.dry_run:
        from .train import ReprogramModel

        size = cfg.resolved_image_size()
        model = ReprogramModel(
            cfg.backbone, cfg.dataset, method=cfg.method, image_size=size,
            num_layers=cfg.num_layers, num_pool_layers=cfg.num_pool_layers,
            pretrained=cfg.pretrained,
        )
        print(json.dumps({
            "dataset": cfg.dataset,
            "backbone": cfg.backbone,
            "method": cfg.method,
            "image_size": model.image_size,
            "mask_parameters": sum(p.numel() for p in model.mask_parameters()),
            "delta_parameters": int(model.delta.numel()) if model.delta is not None else 0,
            "patch_size": model.input_transform.patch_size,
        }, indent=2))
        return 0

    result = train(cfg)
    path = save_result(result, cfg)
    print(
        f"[{cfg.backbone}/{cfg.dataset}/{cfg.method}/{cfg.label_mapping}/seed{cfg.seed}] "
        f"test accuracy = {result['test_accuracy']:.2f}%  ({path})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
