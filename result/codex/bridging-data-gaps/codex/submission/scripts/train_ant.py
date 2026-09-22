#!/usr/bin/env python3
"""Run DPMs-ANT (Algorithm 1) on one of the paper's few-shot tasks.

Examples
--------
# DDPM, FFHQ -> 10-shot sunglasses (Table 1 / Table 2)
python scripts/train_ant.py --task ddpm_ffhq_sunglasses \
    --source-root data/ffhq256 --target-root data/few_shot/sunglasses \
    --output-dir outputs

# LDM backbone
python scripts/train_ant.py --task ldm_ffhq_sunglasses \
    --source-root data/ffhq256 --target-root data/few_shot/sunglasses \
    --checkpoint path/to/ldm-diffusers-dir --output-dir outputs

# the ablation rows of Figure 4
python scripts/train_ant.py --task ddpm_ffhq_sunglasses --baseline vanilla ...
python scripts/train_ant.py --task ddpm_ffhq_sunglasses --baseline full    ...
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpms_ant.configs import TASK_CONFIGS
from dpms_ant.image_experiments import ExperimentArgs, run_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ddpm_ffhq_sunglasses", choices=sorted(TASK_CONFIGS))
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--checkpoint", default=None, help="pre-trained denoiser checkpoint")
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--fid-real-root", default=None, help="larger real set for FID")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generated", type=int, default=1000, help="Intra-LPIPS set")
    parser.add_argument("--num-fid-images", type=int, default=2500)
    parser.add_argument(
        "--baseline", default="ant", choices=["ant", "vanilla", "full"],
        help="ant = our method; vanilla = traditional DDPM loss; full = fine-tune everything",
    )
    parser.add_argument("--freeze-classifier-backbone", action="store_true")
    parser.add_argument("--adaptor-granularity", default="layer", choices=["layer", "block"])
    parser.add_argument("--ddim-steps", type=int, default=0, help="0 = ancestral DDPM sampling")
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = ExperimentArgs(
        task=args.task,
        target_root=args.target_root,
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        fid_real_root=args.fid_real_root,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        num_generated=args.num_generated,
        num_fid_images=args.num_fid_images,
        baseline=args.baseline,
        classifier_freeze_backbone=args.freeze_classifier_backbone,
        adaptor_granularity=args.adaptor_granularity,
        ddim_steps=args.ddim_steps,
        save_every=args.save_every,
    )
    results = run_task(experiment)
    print(results)


if __name__ == "__main__":
    main()

