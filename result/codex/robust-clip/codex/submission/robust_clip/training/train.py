"""CLI: adversarially fine-tune a CLIP vision encoder with FARE or TeCoA.

Example (reproduces the FARE^4 ViT-L/14 encoder of Table 1 for LLaVA)::

    python -m robust_clip.training.train \
        --method fare --arch ViT-L-14 --pretrained openai \
        --eps 4/255 --epochs 2 --batch-size 128 --lr 1e-5 --wd 1e-4

For the OpenFlamingo vision encoder use ``--pretrained laion2b_s32b_b82k``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from ..utils.misc import get_logger
from .adv_train import AdversarialFineTuner, FineTuneConfig, evaluate_imagenet
from .imagenet import build_imagenet_loaders


def parse_args(argv=None) -> FineTuneConfig:
    parser = argparse.ArgumentParser(description="FARE / TeCoA adversarial fine-tuning of CLIP")
    parser.add_argument("--method", choices=["fare", "tecoa"], default="fare")
    parser.add_argument("--arch", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai", help="'openai' (LLaVA) or 'laion2b_s32b_b82k' (OpenFlamingo)")
    parser.add_argument("--eps", default="2/255")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha", default="1/255")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128, help="effective batch size")
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-frac", type=float, default=0.07)
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--clean-weight", type=float, default=1.0, help="lambda of the FARE loss, Eq. (3)")
    parser.add_argument("--norm", default="l2_squared", choices=["l2_squared", "l1"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--attack-dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-dir", default=None, help="ImageFolder style directory (optional)")
    parser.add_argument("--imagenet-name", default="imagenet-1k")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None, help="debug / smoke tests")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-imagenet", action="store_true", help="report clean/robust ImageNet accuracy each epoch")
    parser.add_argument("--eval-radii", nargs="*", default=["2/255", "4/255"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    return FineTuneConfig(**vars(args))


def main(argv=None) -> None:
    config = parse_args(argv)
    logger = get_logger("robust_clip.train", os.path.join(config.output_dir, "train.log"))
    logger.info("configuration: %s", json.dumps(config.__dict__, indent=2))

    train_loader, val_loader = build_imagenet_loaders(
        batch_size=config.micro_batch_size,
        resolution=config.resolution,
        num_workers=config.num_workers,
        data_dir=config.data_dir,
        dataset_name=config.imagenet_name,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        hf_token=config.hf_token,
    )
    logger.info("train batches: %d | val batches: %d", len(train_loader), len(val_loader))

    trainer = AdversarialFineTuner(config, logger=logger)
    eval_fn = None
    if config.eval_imagenet:
        eval_fn = lambda model, loader, device: evaluate_imagenet(model, loader, device, radii=config.eval_radii)
    history = trainer.fit(
        train_loader,
        val_loader=val_loader if config.eval_imagenet else None,
        eval_fn=eval_fn,
    )
    path = trainer.save_checkpoint(epoch=config.epochs)
    logger.info("finished, checkpoint at %s, history: %s", path, history)


if __name__ == "__main__":  # pragma: no cover
    main()
