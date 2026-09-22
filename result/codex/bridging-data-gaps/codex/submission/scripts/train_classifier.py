#!/usr/bin/env python3
"""Train the source/target domain classifier of Section 5.2 / 5.5.

The supplementary material specifies: modify the last layer of the pre-trained
256x256 ImageNet classifier to output two classes, then fine-tune with Adam,
learning rate 1e-4, batch size 64, for 300 iterations, on noised images.

python scripts/train_classifier.py \
    --source-root data/ffhq256 --target-root data/few_shot/sunglasses \
    --output outputs/sunglasses/classifier.pt [--freeze-backbone]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpms_ant.backbones import load_domain_classifier
from dpms_ant.classifier import ClassifierTrainConfig, save_classifier, train_domain_classifier
from dpms_ant.data import LabelledImageDataset, ImageTensorDataset, infinite_loader, list_images
from dpms_ant.schedules import DiffusionSchedule
from dpms_ant.utils import Logger, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--output", required=True, help="path of the .pt file to write")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-shots", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = str(get_device(args.device))
    set_seed(args.seed)
    logger = Logger(args.log)

    source = ImageTensorDataset(list_images(args.source_root), image_size=args.image_size)
    target = ImageTensorDataset(
        list_images(args.target_root)[: args.num_shots], image_size=args.image_size
    )
    logger.log(f"[classifier] {len(source)} source / {len(target)} target images")

    classifier = load_domain_classifier(num_classes=2, device=device)
    labelled = LabelledImageDataset(source, target)
    batches = infinite_loader(labelled, batch_size=args.batch_size, shuffle=True)
    schedule = DiffusionSchedule().to(device)

    config = ClassifierTrainConfig(
        iterations=args.iterations,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        freeze_backbone=args.freeze_backbone,
    )
    history = train_domain_classifier(classifier, schedule, batches, config, device, logger)
    save_classifier(classifier, args.output, config)
    logger.log(
        f"[classifier] saved to {args.output} (final accuracy {history['final_accuracy']:.3f})"
    )


if __name__ == "__main__":
    main()
