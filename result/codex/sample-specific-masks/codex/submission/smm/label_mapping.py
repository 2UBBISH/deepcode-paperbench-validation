"""Output mapping ``f_out``: Random (Rlm), Frequent (Flm) and Iterative (Ilm).

Reproduced from Section 2.3 ("Output Mapping of Reprogramming") and
Appendix A.4 (Algorithms 2, 3 and 4) of the paper.

``f_out`` is a parameter-free, injective mapping from a subset
``Y_sub^P`` of the pre-trained label space ``Y^P`` to the target label space
``Y^T`` (``|Y_sub^P| = k^T``).

* Rlm assigns ``Y_sub^P`` at random before training and keeps it fixed.
* Flm picks, for every target class ``y^T``, the pre-trained class that is
  predicted most often on the *un-reprogrammed* (identity ``f_in``) data
  (Algorithm 2 + Algorithm 3) and keeps the mapping fixed.
* Ilm recomputes the frequency distribution with the *current* ``f_in`` before
  each training iteration and rebuilds the mapping (Algorithm 4).

All three variants greedily consume ``(y^P, y^T)`` frequency counts while
zeroing the used row and column, which guarantees injectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

__all__ = [
    "LabelMapping",
    "random_label_mapping",
    "frequency_distribution",
    "greedy_injective_mapping",
    "frequent_label_mapping",
    "iterative_label_mapping",
]


@dataclass
class LabelMapping:
    """Injective mapping ``f_out: Y_sub^P -> Y^T``.

    ``source`` lists the selected pre-trained class indices (the rows of the
    frequency matrix that were consumed) and ``target[i]`` is the target class
    mapped to ``source[i]``.
    """

    source: torch.Tensor  # (k^T,) pre-trained class indices
    target: torch.Tensor  # (k^T,) target class indices
    num_pretrained_classes: int
    num_target_classes: int
    name: str = "ilm"

    def __post_init__(self) -> None:
        self.source = self.source.to(torch.long).flatten()
        self.target = self.target.to(torch.long).flatten()
        if self.source.numel() != self.target.numel():
            raise ValueError("source and target must have the same size")
        if self.source.unique().numel() != self.source.numel():
            raise ValueError("f_out must be injective on Y_sub^P")

    def __len__(self) -> int:
        return int(self.source.numel())

    def to(self, device) -> "LabelMapping":
        return LabelMapping(
            self.source.to(device),
            self.target.to(device),
            self.num_pretrained_classes,
            self.num_target_classes,
            self.name,
        )

    def target_index_of(self, labels: torch.Tensor) -> torch.Tensor:
        """Position of the target labels inside ``self.target`` (for the loss)."""
        labels = labels.to(torch.long).flatten()
        pos = torch.full_like(labels, -1)
        for i, t in enumerate(self.target.tolist()):
            pos[labels == t] = i
        if (pos < 0).any():
            missing = labels[pos < 0].unique().tolist()
            raise ValueError(f"target classes {missing} are not covered by the mapping")
        return pos

    def as_dict(self) -> Dict[str, list]:
        return {
            "source": self.source.tolist(),
            "target": self.target.tolist(),
            "num_pretrained_classes": self.num_pretrained_classes,
            "num_target_classes": self.num_target_classes,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LabelMapping":
        return cls(
            torch.tensor(d["source"]),
            torch.tensor(d["target"]),
            int(d["num_pretrained_classes"]),
            int(d["num_target_classes"]),
            str(d.get("name", "ilm")),
        )


def random_label_mapping(
    num_pretrained_classes: int,
    num_target_classes: int,
    seed: int = 0,
    device=None,
) -> LabelMapping:
    """``f_out^Rlm``: a random injective mapping fixed before training."""
    if num_target_classes > num_pretrained_classes:
        raise ValueError("the pre-trained label space must be at least as large as the target one")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_pretrained_classes, generator=g)
    source = perm[:num_target_classes].sort().values
    target = torch.randperm(num_target_classes, generator=g)
    return LabelMapping(source.to(device), target.to(device),
                        num_pretrained_classes, num_target_classes, "rlm")


@torch.no_grad()
def frequency_distribution(
    source_predictions: Iterable[torch.Tensor],
    target_labels: Iterable[torch.Tensor],
    num_pretrained_classes: int,
    num_target_classes: int,
    device=None,
) -> torch.Tensor:
    """Algorithm 2: ``d[y^P, y^T]`` counts of pre-trained predictions per class.

    ``source_predictions`` / ``target_labels`` are iterables of (already
    batched) tensors coming from a data loader; no gradient is needed.
    """
    d = torch.zeros(num_pretrained_classes, num_target_classes, dtype=torch.long, device=device)
    for preds, labels in zip(source_predictions, target_labels):
        preds = preds.to(torch.long).flatten().to(d.device)
        labels = labels.to(torch.long).flatten().to(d.device)
        d.index_put_((preds, labels), torch.ones_like(preds), accumulate=True)
    return d


def greedy_injective_mapping(
    d: torch.Tensor,
    num_target_classes: int,
    name: str = "ilm",
    device=None,
) -> LabelMapping:
    """Algorithm 3/4 core loop: greedily match the largest counts injectively."""
    counts = d.clone().to(torch.float64)
    num_pretrained = counts.shape[0]
    source, target = [], []
    for _ in range(num_target_classes):
        idx = int(torch.argmax(counts))
        r, c = divmod(idx, counts.shape[1])
        if counts[r, c] <= 0:
            # No remaining evidence: fall back to the first unused pre-trained
            # class so that the mapping is still injective.
            used_sources = set(source)
            free = [i for i in range(num_pretrained) if i not in used_sources]
            free_targets = [t for t in range(num_target_classes) if t not in target]
            if not free or not free_targets:  # pragma: no cover - defensive
                raise RuntimeError("cannot complete an injective label mapping")
            r, c = free[0], free_targets[0]
        source.append(r)
        target.append(c)
        counts[r, :] = 0  # avoid illegal assignment to the injective function
        counts[:, c] = 0
    src = torch.tensor(source, dtype=torch.long, device=device)
    tgt = torch.tensor(target, dtype=torch.long, device=device)
    if src.unique().numel() != src.numel():  # pragma: no cover - defensive
        raise RuntimeError("greedy mapping produced a non-injective assignment")
    return LabelMapping(src, tgt, counts.shape[0], num_target_classes, name)


@torch.no_grad()
def frequent_label_mapping(
    pretrained_predictions: Iterable[torch.Tensor],
    target_labels: Iterable[torch.Tensor],
    num_pretrained_classes: int,
    num_target_classes: int,
    device=None,
) -> LabelMapping:
    """``f_out^Flm`` computed once with the identity ``f_in`` (Algorithm 3)."""
    d = frequency_distribution(
        pretrained_predictions, target_labels,
        num_pretrained_classes, num_target_classes, device,
    )
    return greedy_injective_mapping(d, num_target_classes, "flm", device)


def iterative_label_mapping(
    pretrained_predictions: Iterable[torch.Tensor],
    target_labels: Iterable[torch.Tensor],
    num_pretrained_classes: int,
    num_target_classes: int,
    device=None,
) -> LabelMapping:
    """``f_out^Ilm,(j)`` for iteration ``j`` (Algorithm 4).

    Called once per epoch with the predictions of the *current* ``f_in``.
    """
    return frequent_label_mapping(
        pretrained_predictions, target_labels,
        num_pretrained_classes, num_target_classes, device,
    )
