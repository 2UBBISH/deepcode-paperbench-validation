"""Taxonomy soft labels and linear probing (Section 4.3.2 / Appendix E.2).

The linear probe is trained with the loss from Algorithm 1::

    L = lambda * L(CE) + L(soft_lca)

where the soft targets are the rows of the *reversed* (``1 - M_LCA``) LCA
distance matrix, i.e. the ground-truth class receives the target 1 and
semantically close classes receive large targets as well.  Following Appendix
E.2 the matrix is pre-processed with ``M_LCA = MinMax(M ** T)`` (``T = 25``).

Finally the paper interpolates in weight space between the plain cross-entropy
probe and the soft-label probe::

    W_interp = alpha * W_ce + (1 - alpha) * W_ce+soft

which is what produces the "no ID accuracy drop" rows of Tables 5/6/9.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent import process_lca_matrix

DEFAULT_LAMBDA = 0.03
DEFAULT_TEMPERATURE = 25.0


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #
def lca_alignment_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alignment_mode: str,
    lca_matrix: torch.Tensor,
    lambda_weight: float = DEFAULT_LAMBDA,
) -> torch.Tensor:
    """The paper's ``LCA_ALIGNMENT_LOSS`` (Algorithm 1, Appendix E.2).

    ``lca_matrix`` must already be the processed, min-max scaled ``(K, K)``
    matrix whose diagonal is 0.
    """
    reverse_lca_matrix = 1.0 - lca_matrix
    probs = F.softmax(logits, dim=1)
    one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).to(probs.dtype)

    standard_loss = -(one_hot_targets * torch.log(probs + 1e-12)).sum(dim=1)
    soft_targets = reverse_lca_matrix[targets]

    mode = alignment_mode.upper()
    if mode == "BCE":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        soft_loss = criterion(logits, soft_targets).mean(dim=1)
    elif mode == "CE":
        soft_loss = -(soft_targets * torch.log(probs + 1e-12)).mean(dim=1)
    else:
        raise ValueError("alignment_mode must be 'CE' or 'BCE', got %r" % alignment_mode)

    total_loss = lambda_weight * standard_loss + soft_loss
    return total_loss.mean()


# --------------------------------------------------------------------------- #
# soft label construction
# --------------------------------------------------------------------------- #
def build_soft_labels(
    lca_matrix: np.ndarray,
    temperature: float = DEFAULT_TEMPERATURE,
    tree_prefix: str = "WordNet",
) -> torch.Tensor:
    """``MinMax(M ** T)`` (plus the latent-hierarchy inversion)."""
    return process_lca_matrix(lca_matrix, tree_prefix, temperature=temperature)


def soft_label_targets(
    lca_matrix_processed: torch.Tensor, targets: Sequence[int]
) -> torch.Tensor:
    """``reverse_LCA_matrix[targets]``: the per-sample soft target vectors."""
    return 1.0 - lca_matrix_processed[torch.as_tensor(targets, dtype=torch.long)]


# --------------------------------------------------------------------------- #
# linear probe
# --------------------------------------------------------------------------- #
@dataclass
class ProbeConfig:
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_lr: float = 1e-5
    warmup_epochs: int = 1
    lambda_weight: float = DEFAULT_LAMBDA
    temperature: float = DEFAULT_TEMPERATURE
    alignment_mode: str = "CE"
    seed: int = 0


class LinearProbe(nn.Module):
    def __init__(self, in_features: int, n_classes: int = 1000):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    lca_matrix: Optional[torch.Tensor] = None,
    config: Optional[ProbeConfig] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> LinearProbe:
    """Train a linear classifier, optionally with the LCA soft loss."""
    config = config or ProbeConfig()
    torch.manual_seed(config.seed)
    n_classes = int(targets.max().item()) + 1
    model = LinearProbe(features.shape[1], n_classes).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(1, -(-len(features) // config.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=config.lr,
        total_steps=config.epochs * steps_per_epoch,
        pct_start=max(config.warmup_epochs / max(config.epochs, 1), 1e-3),
        div_factor=max(config.lr / max(config.warmup_lr, 1e-9), 1.0),
        final_div_factor=1e3,
    )

    features = features.to(device).float()
    targets = targets.to(device).long()
    if lca_matrix is not None:
        lca_matrix = lca_matrix.to(device).float()

    n = len(features)
    for epoch in range(config.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, config.batch_size):
            idx = perm[start:start + config.batch_size]
            batch_x, batch_y = features[idx], targets[idx]
            logits = model(batch_x)
            if lca_matrix is None:
                loss = F.cross_entropy(logits, batch_y)
            else:
                loss = lca_alignment_loss(
                    logits, batch_y, config.alignment_mode, lca_matrix,
                    config.lambda_weight,
                )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            epoch_loss += float(loss.item()) * len(idx)
        if verbose:
            print("[probe] epoch %d loss=%.4f" % (epoch, epoch_loss / n))
    return model


def interpolate_weights(
    probe_ce: LinearProbe, probe_soft: LinearProbe, alpha: float = 0.5
) -> LinearProbe:
    """``W_interp = alpha * W_ce + (1 - alpha) * W_ce+soft``."""
    interp = LinearProbe(
        probe_ce.fc.in_features, probe_ce.fc.out_features
    ).to(probe_ce.fc.weight.device)
    with torch.no_grad():
        interp.fc.weight.copy_(
            alpha * probe_ce.fc.weight + (1 - alpha) * probe_soft.fc.weight
        )
        interp.fc.bias.copy_(
            alpha * probe_ce.fc.bias + (1 - alpha) * probe_soft.fc.bias
        )
    return interp


@torch.no_grad()
def evaluate_probe(
    probe: LinearProbe,
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 4096,
    device: str = "cpu",
) -> Dict[str, float]:
    probe.eval().to(device)
    features = features.to(device).float()
    targets = targets.to(device).long()
    correct1 = correct5 = 0
    n = len(features)
    for start in range(0, n, batch_size):
        logits = probe(features[start:start + batch_size])
        y = targets[start:start + batch_size]
        correct1 += int((logits.argmax(dim=1) == y).sum().item())
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct5 += int((top5 == y[:, None]).any(dim=1).sum().item())
    return {"top1": correct1 / n, "top5": correct5 / n}


# --------------------------------------------------------------------------- #
# end-to-end helper
# --------------------------------------------------------------------------- #
def run_soft_label_experiment(
    backbone_features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    lca_matrix: np.ndarray,
    tree_prefix: str = "WordNet",
    config: Optional[ProbeConfig] = None,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    device: str = "cpu",
) -> Dict[str, float]:
    """Train CE and CE+soft probes and evaluate the interpolations.

    ``backbone_features`` maps a split name (``train``/``imagenet``/...) to a
    ``(features, targets)`` pair.  Returns the accuracy of the baseline, the
    soft-label probe and the best interpolation.
    """
    config = config or ProbeConfig()
    train_x, train_y = backbone_features["train"]
    soft_matrix = build_soft_labels(
        lca_matrix, temperature=config.temperature, tree_prefix=tree_prefix
    )

    probe_ce = train_linear_probe(
        train_x, train_y, None, config, device=device
    )
    probe_soft = train_linear_probe(
        train_x, train_y, soft_matrix, config, device=device
    )

    results: Dict[str, float] = {}
    for split, (feat, targ) in backbone_features.items():
        if split == "train":
            continue
        base = evaluate_probe(probe_ce, feat, targ, device=device)
        soft = evaluate_probe(probe_soft, feat, targ, device=device)
        results["%s/baseline" % split] = base["top1"]
        results["%s/soft" % split] = soft["top1"]
        for alpha in alphas:
            interp = interpolate_weights(probe_ce, probe_soft, alpha)
            acc = evaluate_probe(interp, feat, targ, device=device)
            results["%s/interp@%.2f" % (split, alpha)] = acc["top1"]
    return results


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LCA soft-label linear probing")
    parser.add_argument("--features", required=True,
                        help=".npz with train_x/train_y and test features")
    parser.add_argument("--lca-matrix", required=True, help=".npy LCA matrix")
    parser.add_argument("--tree-prefix", default="WordNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-weight", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    data = np.load(args.features)
    splits = {}
    for key in data.files:
        if key.endswith("_x"):
            splits[key[:-2]] = (
                torch.from_numpy(data[key]),
                torch.from_numpy(data[key[:-2] + "_y"]),
            )
    lca_matrix = np.load(args.lca_matrix)
    results = run_soft_label_experiment(
        splits,
        lca_matrix,
        tree_prefix=args.tree_prefix,
        config=ProbeConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_weight=args.lambda_weight,
            temperature=args.temperature,
        ),
        device=args.device,
    )
    for key, value in sorted(results.items()):
        print("%-32s %.4f" % (key, value))


if __name__ == "__main__":  # pragma: no cover
    main()
