"""Zero-shot classification: clean and adversarial accuracy (Sec. 4.3).

Attack setup of the paper: "we employ the first two attacks of AutoAttack,
namely APGD with cross-entropy loss and APGD with **targeted** DLR loss (100
iterations each)".  The targeted DLR attack uses the runner-up class as the
target, as in AutoAttack.  Note that the targeted variant is *stronger* than the
untargeted one used by Mao et al. (2023).

Metrics (Table 4): clean accuracy on all samples of a dataset and robust
accuracy on 1000 samples at :math:`\\ell_\\infty` radii ``2/255`` and ``4/255``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..attacks.apgd import APGDAttack, make_loss, runner_up_classes
from ..models import load_clip_encoder
from ..utils.common import LOGGER, add_common_args, get_device, parse_epsilon, set_seed
from .zs_datasets import (
    ZERO_SHOT_DATASETS,
    ZeroShotSpec,
    build_text_embeddings,
    build_zero_shot_transform,
    class_names_of,
    load_zero_shot_dataset,
    transform_resolution,
)


def cosine_logits_fn(encoder, text_embeddings: torch.Tensor, logit_scale: Optional[float] = None):
    """``f_k(phi, x) = cos(phi(x), psi(t_k))`` (Eq. (1) of the paper)."""

    def logits_fn(images: torch.Tensor) -> torch.Tensor:
        z = encoder.encode_image(images)
        z = F.normalize(z.float(), dim=-1)
        text = text_embeddings.to(z.device, z.dtype)
        logits = z @ text.t()
        if logit_scale is not None:
            logits = logits * logit_scale
        return logits

    return logits_fn


@torch.no_grad()
def clean_accuracy(logits_fn, loader, device) -> float:
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = logits_fn(images).argmax(dim=1)
        correct += int((preds == labels).sum())
        total += labels.numel()
    return 100.0 * correct / max(1, total)


def robust_accuracy(
    logits_fn,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_iter: int = 100,
    binary: bool = False,
    momentum: float = 0.75,
    alpha: Optional[float] = None,
    grad_normalization: str = "elementwise_sign",
    n_restarts: int = 1,
) -> Dict[str, int]:
    """APGD-CE followed by APGD-DLR (targeted), chained as in AutoAttack.

    Returns the number of correctly classified samples after each stage
    (``n_total`` is the batch size).  Samples that are already misclassified on
    the clean input count as incorrect, as in every adversarial robustness
    evaluation.
    """
    n_total = int(labels.numel())
    clean_correct = (logits_fn(images).argmax(dim=1) == labels)
    idx = torch.nonzero(clean_correct, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return {"ce": 0, "dlr": 0, "n_total": n_total}

    sub_images = images[idx]
    sub_labels = labels[idx]
    attack_kwargs = dict(
        eps=eps,
        n_iter=n_iter,
        alpha=alpha,
        momentum=momentum,
        grad_normalization=grad_normalization,
        n_restarts=n_restarts,
    )

    # ---- APGD with the cross-entropy loss --------------------------------
    loss_fn = make_loss("ce", targets=sub_labels, logits_fn=logits_fn)
    attack = APGDAttack(**attack_kwargs)
    x_ce, _ = attack.perturb(loss_fn, sub_images, maximize=True)
    still_correct = logits_fn(x_ce).argmax(dim=1) == sub_labels
    n_ce = int(still_correct.sum())

    if binary:
        # The DLR loss is only defined for multi-class problems (PCAM).
        return {"ce": n_ce, "dlr": n_ce, "n_total": n_total}

    # ---- APGD with the targeted DLR loss ---------------------------------
    idx2 = torch.nonzero(still_correct, as_tuple=False).squeeze(1)
    if idx2.numel() == 0:
        return {"ce": 0, "dlr": 0, "n_total": n_total}
    sub2_images = sub_images[idx2]
    sub2_labels = sub_labels[idx2]
    targets = runner_up_classes(logits_fn(sub2_images), sub2_labels)
    loss_fn = make_loss(
        "targeted_dlr", targets=sub2_labels, target_classes=targets, logits_fn=logits_fn
    )
    attack = APGDAttack(**attack_kwargs)
    x_dlr, _ = attack.perturb(loss_fn, sub2_images, maximize=True)
    n_dlr = int((logits_fn(x_dlr).argmax(dim=1) == sub2_labels).sum())
    return {"ce": n_ce, "dlr": n_dlr, "n_total": n_total}


def evaluate_dataset(
    encoder,
    tokenizer,
    spec: ZeroShotSpec,
    args,
    device: torch.device,
) -> Dict[str, object]:
    transform = build_zero_shot_transform(
        transform_resolution(spec, native_resolution=getattr(args, "native_resolution", False))
    )
    dataset = load_zero_shot_dataset(
        spec, transform, root=args.data_root, local_files_only=args.local_files_only
    )
    class_names = class_names_of(dataset)
    if getattr(args, "max_clean_samples", None) and len(dataset) > args.max_clean_samples:
        generator = torch.Generator().manual_seed(args.seed)
        subset = torch.randperm(len(dataset), generator=generator)[:args.max_clean_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, subset)
    if class_names is None:
        raise RuntimeError(
            f"could not determine the class names of '{spec.name}'; "
            "provide a CLIP_benchmark style data root with classes.txt"
        )
    text_embeddings = build_text_embeddings(
        encoder, tokenizer, class_names, spec.templates, device=device
    )
    logits_fn = cosine_logits_fn(encoder, text_embeddings)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    clean = clean_accuracy(logits_fn, loader, device)
    LOGGER.info("[%s] clean accuracy: %.1f", spec.name, clean)

    # ---- adversarial evaluation on a random subset ------------------------
    n = min(args.n_samples, len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    sub_loader = torch.utils.data.DataLoader(
        subset, batch_size=args.attack_batch_size, shuffle=False, num_workers=args.num_workers
    )

    results: Dict[str, Dict[str, float]] = {}
    for eps in args.eps_list:
        eps = parse_epsilon(eps)
        correct_ce = total = 0
        correct_dlr = 0
        for images, labels in sub_loader:
            images = images.to(device)
            labels = labels.to(device)
            numbers = robust_accuracy(
                logits_fn,
                images,
                labels,
                eps,
                n_iter=args.n_iter,
                binary=(len(class_names) == 2),
                momentum=args.momentum,
                grad_normalization=args.grad_normalization,
                n_restarts=args.n_restarts,
            )
            correct_ce += numbers["ce"]
            correct_dlr += numbers["dlr"]
            total += numbers["n_total"]
        acc_ce = 100.0 * correct_ce / max(1, total)
        acc_dlr = 100.0 * correct_dlr / max(1, total)
        # The evaluation uses the stronger of the two attacks, i.e. a sample is
        # counted as correctly classified only if neither attack succeeded.
        results[f"{eps:.7f}"] = {"robust": acc_dlr, "apgd_ce": acc_ce}
        LOGGER.info(
            "[%s] robust accuracy @ %.6f: %.1f (APGD-CE %.1f)", spec.name, eps, acc_dlr, acc_ce
        )

    return {"clean": clean, "robust": results, "num_classes": len(class_names)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--checkpoint", default=None, help="FARE / TeCoA checkpoint")
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(ZERO_SHOT_DATASETS.keys()),
        help="which zero-shot datasets to evaluate",
    )
    parser.add_argument("--data-root", default=None, help="CLIP_benchmark style data root")
    parser.add_argument("--eps-list", nargs="+", default=["2/255", "4/255"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument(
        "--max-clean-samples", type=int, default=None,
        help="evaluate the clean accuracy on a subset only (debugging)",
    )
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument(
        "--n-restarts", type=int, default=0,
        help="the paper runs 100 iterations per attack (no restart)",
    )
    parser.add_argument("--momentum", type=float, default=0.75)
    parser.add_argument("--grad-normalization", default="elementwise_sign")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--attack-batch-size", type=int, default=64)
    parser.add_argument(
        "--native-resolution", action="store_true",
        help="evaluate CIFAR10/CIFAR100/STL10 at their stored resolution "
             "(default: resize everything to 224x224)",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default=None)
    add_common_args(parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    encoder = load_clip_encoder(
        arch=args.arch,
        pretrained=args.pretrained,
        image_size=args.image_size,
        checkpoint=args.checkpoint,
        checkpoint_key=args.checkpoint_key,
        device=device,
    )
    import open_clip

    tokenizer = open_clip.get_tokenizer(args.arch)
    encoder.eval()

    all_results = {}
    for name in args.datasets:
        spec = ZERO_SHOT_DATASETS[name]
        all_results[name] = evaluate_dataset(encoder, tokenizer, spec, args, device)

    averages = {}
    zero_shot = [v for k, v in all_results.items() if k != "imagenet"]
    if zero_shot:
        averages["avg_zero_shot_clean"] = sum(v["clean"] for v in zero_shot) / len(zero_shot)
    # the robust average over the zero-shot datasets (last column of Table 4)
    eps_keys = None
    for value in zero_shot:
        if value.get("robust"):
            eps_keys = sorted(value["robust"].keys())
            break
    for eps_key in eps_keys or []:
        values = [
            value["robust"][eps_key]["robust"]
            for value in zero_shot
            if eps_key in value.get("robust", {})
        ]
        if values:
            averages[f"avg_zero_shot_robust_{eps_key}"] = sum(values) / len(values)
    all_results["average"] = averages
    print(json.dumps(all_results, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
