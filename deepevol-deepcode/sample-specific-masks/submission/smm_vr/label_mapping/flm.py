"""Frequent Label Mapping (Flm) --- Algorithm 3 of the SMM paper.

Paper reference (Appendix A.4, Algorithm 3)::

    Algorithm 3  Frequent Label Mapping (f_out^Flm)
      Input: label space of the pre-trained task Y^P, label space of the target
             task Y^T, target training set {(x_i^T, y_i^T)}_{i=1}^n,
             given pre-trained model f_P(.)
      Output: Flm f_out^Flm : Y_sub^P -> Y^T
      Initialize f_out^Flm(.) <- 0, subset Y_sub^P <- empty to store matched
                 labels, initialize f_in(.|theta) to be an identity function
                 (theta <- 0)
      # Compute frequency distribution d
      Use Algorithm 2 to obtain d
      # Compute output mapping f_out^Flm
      while size of Y_sub^P is not |Y^T| do
          Find the maximum d_{y^P, y^T} in d
          Y_sub^P <- Y_sub^P union {y^P}
          f_out^Flm(y^P) <- y^T                    # update the label mapping
          d_{y^P, t} <- 0 for t = 1, 2, ..., |Y^T| # avoiding illegal assignment
          d_{s, y^T} <- 0 for s = 1, 2, ..., |Y^P| # to the injective function
      end while

and the inverse form of the mapping used here (Appendix A.4, Eq. (10))::

    y_Flm^P = argmax_{y in Y^P} Pr{ y = f_P(f_in(x_i|theta)) | y_i = y^T }

So ``y_Flm^P`` is the ImageNet label that the (identity) reprogrammed target
model is *most frequently* predicted as, for all target samples of the class
``y^T``.  This is exactly the largest entry ``d_{y^P, y^T}`` of the frequency
distribution matrix produced by Algorithm 2
(:mod:`smm_vr.label_mapping.frequency`).

Flm is computed **once** before training and then stays fixed throughout the
optimization of ``f_in`` (this is the key difference from Ilm, which recomputes
the mapping before every epoch).

Representation
--------------
Because ``f_out`` is a one-to-one (injective) mapping from a subset
``Y_sub^P`` of the ImageNet label space onto the target label space, it can be
represented by a single integer index tensor.  We keep it in the direction that
is convenient for training, i.e.

    target_to_pretrained[t] = y^P     for every target class t in [0, |Y^T|)

and provide the inverse view

    pretrained_to_target[p] = y^T  (or IGNORE_INDEX if p not in Y_sub^P).

Training then simply selects the mapped logits::

    mapped_logits = logits[:, target_to_pretrained]

which yields a ``|Y^T|``-dimensional score vector that is used with the usual
cross-entropy loss over the target labels.  No learnable parameters are added.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .frequency import (
    compute_frequency_matrix,
    frequency_matrix_from_predictions,
)

__all__ = [
    "IGNORE_INDEX",
    "FlmMapping",
    "FlmLabelMapping",
    "build_flm_mapping",
    "compute_flm_mapping",
    "flm_mapping_from_frequency_matrix",
    "flm_mapping_from_predictions",
    "apply_label_mapping",
]

#: Sentinel stored in ``pretrained_to_target`` for ImageNet labels that are not
#: part of ``Y_sub^P`` (i.e. that are not used by the mapping).
IGNORE_INDEX: int = -1


# ---------------------------------------------------------------------------
# Core greedy assignment (Algorithm 3)
# ---------------------------------------------------------------------------
def flm_mapping_from_frequency_matrix(
    frequency_matrix: torch.Tensor,
    num_target_classes: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Run the greedy loop of Algorithm 3 on a frequency matrix ``d``.

    Parameters
    ----------
    frequency_matrix:
        ``d`` of shape ``(|Y^P|, |Y^T|)`` produced by Algorithm 2.  It is not
        modified in place; a working copy is used.
    num_target_classes:
        ``|Y^T|``.  Inferred from ``frequency_matrix.shape[1]`` when omitted.
    ignore_index:
        Value used for unmatched entries of ``pretrained_to_target``.

    Returns
    -------
    dict with keys

    ``target_to_pretrained`` : ``LongTensor`` of shape ``(|Y^T|,)``
        ``target_to_pretrained[t] = y^P`` (the ImageNet label matched to target
        class ``t``).
    ``pretrained_to_target`` : ``LongTensor`` of shape ``(|Y^P|,)``
        ``pretrained_to_target[p] = t`` for ``p in Y_sub^P`` and
        ``ignore_index`` elsewhere.  This is ``f_out^Flm`` itself.
    ``frequency_matrix`` : ``LongTensor``
        The working copy of ``d`` after all rows/columns were zeroed.
    """
    if frequency_matrix.dim() != 2:
        raise ValueError(
            "frequency_matrix must be 2-D (|Y^P|, |Y^T|), got shape "
            f"{tuple(frequency_matrix.shape)}"
        )

    num_pretrained_classes, num_target = frequency_matrix.shape
    if num_target_classes is None:
        num_target_classes = int(num_target)
    if int(num_target_classes) != int(num_target):
        raise ValueError(
            f"num_target_classes={num_target_classes} does not match the "
            f"second dimension of frequency_matrix ({num_target})"
        )
    if num_pretrained_classes < num_target_classes:
        raise ValueError(
            "the ImageNet label space must be at least as large as the target "
            f"label space to allow an injective mapping (got {num_pretrained_classes} "
            f"< {num_target_classes})"
        )

    # Working copy: "Initialize d <- {0}^{|Y^P| x |Y^T|}" then Algorithm 2 fills it.
    d = frequency_matrix.detach().to(torch.long).clone()

    target_to_pretrained = torch.full(
        (num_target_classes,), ignore_index, dtype=torch.long, device=d.device
    )
    used_pretrained = torch.zeros(
        num_pretrained_classes, dtype=torch.bool, device=d.device
    )
    used_target = torch.zeros(num_target_classes, dtype=torch.bool, device=d.device)

    flat = d.reshape(-1)

    # "while size of Y_sub^P is not |Y^T| do"
    for _ in range(num_target_classes):
        # "Find the maximum d_{y^P, y^T} in d" (argmax breaks ties by lowest
        # flattened index, i.e. deterministic and reproducible).
        pos = int(torch.argmax(flat).item())
        y_p = pos // num_target_classes
        y_t = pos % num_target_classes

        # "Y_sub^P <- Y_sub^P union {y^P}"
        # "f_out^Flm(y^P) <- y^T"
        target_to_pretrained[y_t] = y_p
        used_pretrained[y_p] = True
        used_target[y_t] = True

        # "d_{y^P, t} <- 0 for t = 1, ..., |Y^T|"  (zero the row)
        d[y_p, :] = 0
        # "d_{s, y^T} <- 0 for s = 1, ..., |Y^P|"  (zero the column)
        d[:, y_t] = 0
        flat = d.reshape(-1)

    # Any target class that could not be matched (degenerate case where a class
    # is never predicted by f_P) is filled with the remaining unused ImageNet
    # labels so that the mapping stays total and injective.
    if not bool(used_target.all()):
        free = torch.nonzero(~used_pretrained, as_tuple=False).reshape(-1)
        missing = torch.nonzero(~used_target, as_tuple=False).reshape(-1)
        k = min(free.numel(), missing.numel())
        if k > 0:
            target_to_pretrained[missing[:k]] = free[:k]
            used_pretrained[free[:k]] = True
            used_target[missing[:k]] = True

    pretrained_to_target = torch.full(
        (num_pretrained_classes,), ignore_index, dtype=torch.long, device=d.device
    )
    valid = target_to_pretrained >= 0
    if bool(valid.any()):
        pretrained_to_target[target_to_pretrained[valid]] = torch.nonzero(
            valid, as_tuple=False
        ).reshape(-1)

    return {
        "target_to_pretrained": target_to_pretrained,
        "pretrained_to_target": pretrained_to_target,
        "frequency_matrix": d,
    }


def compute_flm_mapping(
    model: nn.Module,
    data_loader,
    num_target_classes: int,
    num_pretrained_classes: Optional[int] = None,
    f_in: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    *,
    predicted: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX,
    return_frequency_matrix: bool = False,
) -> Dict[str, torch.Tensor]:
    """"Use Algorithm 2 to obtain d" then run Algorithm 3.

    ``f_in`` defaults to ``None``, i.e. the identity function ``theta <- 0`` as
    required by the initialization line of Algorithm 3.
    """
    d = compute_frequency_matrix(
        model=model,
        data_loader=data_loader,
        num_target_classes=num_target_classes,
        num_pretrained_classes=num_pretrained_classes,
        f_in=f_in,
        device=device,
        max_batches=max_batches,
        predicted=predicted,
        targets=targets,
    )
    out = flm_mapping_from_frequency_matrix(
        d, num_target_classes=num_target_classes, ignore_index=ignore_index
    )
    if not return_frequency_matrix:
        out.pop("frequency_matrix", None)
    else:
        out["frequency_matrix"] = d
    return out


def flm_mapping_from_predictions(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    num_pretrained_classes: int,
    num_target_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper: build ``d`` from cached predictions, then Algorithm 3."""
    d = frequency_matrix_from_predictions(
        predicted=predicted,
        targets=targets,
        num_pretrained_classes=num_pretrained_classes,
        num_target_classes=num_target_classes,
    )
    return flm_mapping_from_frequency_matrix(
        d, num_target_classes=num_target_classes, ignore_index=ignore_index
    )


# ---------------------------------------------------------------------------
# Module wrapper (non-parametric f_out^Flm)
# ---------------------------------------------------------------------------
def apply_label_mapping(
    logits: torch.Tensor, target_to_pretrained: torch.Tensor
) -> torch.Tensor:
    """Select the logits of the matched ImageNet classes.

    ``logits`` has shape ``(..., |Y^P|)``; the returned tensor has shape
    ``(..., |Y^T|)`` with ``out[..., t] = logits[..., target_to_pretrained[t]]``.
    """
    index = target_to_pretrained.to(device=logits.device, dtype=torch.long)
    if logits.dim() == 1:
        return logits.index_select(0, index)
    return logits.index_select(logits.dim() - 1, index)


class FlmMapping(nn.Module):
    """Non-parametric frequent label mapping ``f_out^Flm`` (Algorithm 3).

    Buffers
    -------
    ``target_to_pretrained`` : ``(num_target_classes,)`` long
    ``pretrained_to_target`` : ``(num_pretrained_classes,)`` long

    The mapping is stored as buffers (not parameters) because Flm introduces no
    trainable weights; ``update()`` is a no-op so that Flm shares the same
    interface as Ilm inside the training loop.
    """

    recomputes_each_epoch: bool = False

    def __init__(
        self,
        target_to_pretrained: torch.Tensor,
        pretrained_to_target: Optional[torch.Tensor] = None,
        num_pretrained_classes: Optional[int] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        target_to_pretrained = torch.as_tensor(
            target_to_pretrained, dtype=torch.long
        ).clone()
        if pretrained_to_target is None:
            if num_pretrained_classes is None:
                num_pretrained_classes = int(target_to_pretrained.max().item()) + 1
            pretrained_to_target = torch.full(
                (int(num_pretrained_classes),), ignore_index, dtype=torch.long
            )
            valid = target_to_pretrained >= 0
            if bool(valid.any()):
                pretrained_to_target[target_to_pretrained[valid]] = torch.nonzero(
                    valid, as_tuple=False
                ).reshape(-1)
        self.register_buffer("target_to_pretrained", target_to_pretrained)
        self.register_buffer(
            "pretrained_to_target",
            torch.as_tensor(pretrained_to_target, dtype=torch.long).clone(),
        )
        self.ignore_index = int(ignore_index)

    # -- shapes -----------------------------------------------------------
    @property
    def num_target_classes(self) -> int:
        return int(self.target_to_pretrained.numel())

    @property
    def num_pretrained_classes(self) -> int:
        return int(self.pretrained_to_target.numel())

    # -- API --------------------------------------------------------------
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Map ImageNet logits to target-class logits (Eq. (10) evaluation)."""
        return apply_label_mapping(logits, self.target_to_pretrained)

    def update(self, *args, **kwargs) -> bool:
        """Flm "remains unchanged throughout iterations" -> always a no-op."""
        return False

    def map_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Return the predicted target labels for ImageNet logits."""
        return self(logits).argmax(dim=-1)

    def is_injective(self) -> bool:
        """``f_out(y1) != f_out(y2)`` for distinct ``y1 != y2`` (see Sec. 2.3)."""
        idx = self.target_to_pretrained
        if bool((idx < 0).any()):
            return False
        return bool(torch.unique(idx).numel() == idx.numel())

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"num_target_classes={self.num_target_classes}, "
            f"num_pretrained_classes={self.num_pretrained_classes}, "
            f"injective={self.is_injective() if self.num_target_classes else 'n/a'}"
        )


#: Loud alias matching the paper's naming (``f_out^Flm``).
FlmLabelMapping = FlmMapping


def build_flm_mapping(
    model: nn.Module,
    data_loader,
    num_target_classes: int,
    num_pretrained_classes: Optional[int] = None,
    f_in: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    *,
    predicted: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX,
) -> FlmMapping:
    """End-to-end helper: Algorithm 2 + Algorithm 3 in a single ``nn.Module``."""
    out = compute_flm_mapping(
        model=model,
        data_loader=data_loader,
        num_target_classes=num_target_classes,
        num_pretrained_classes=num_pretrained_classes,
        f_in=f_in,
        device=device,
        predicted=predicted,
        targets=targets,
        max_batches=max_batches,
        ignore_index=ignore_index,
    )
    return FlmMapping(
        target_to_pretrained=out["target_to_pretrained"],
        pretrained_to_target=out["pretrained_to_target"],
        num_pretrained_classes=(
            num_pretrained_classes
            if num_pretrained_classes is not None
            else int(out["pretrained_to_target"].numel())
        ),
        ignore_index=ignore_index,
    )
