"""Random Label Mapping (Rlm) for SMM visual reprogramming.

Paper reference
---------------
Sec. 2.3 "Output Mapping of Reprogramming":

    f_out^Rlm(y | Y_sub^P) = rand({0, 1, ..., k^T})

where ``rand({...})`` means randomly selecting one element from the set and
``Y_sub^P`` is of the same size as ``Y^T`` (i.e. ``k^T``), **randomly chosen from
``Y^P`` prior to the minimization of Eq. (1)**.  Because ``f_out^Rlm`` is
injective it holds that ``f_out^Rlm(y_1) != f_out^Rlm(y_2)`` for ``y_1 != y_2``
(Elsayed et al., 2018; Chen et al., 2023).

Practical realisation implemented here (exactly equivalent to the formula):
    1. sample a random subset ``Y_sub^P`` of the ImageNet label space ``Y^P``
       with exactly ``|Y^T|`` elements (no replacement -> injective on the
       subset),
    2. sample a random bijection from ``Y_sub^P`` onto the target label space
       ``Y^T``,
    3. keep both directions as plain integer index tensors and freeze them
       *before* training (``recomputes_each_epoch = False``), in contrast with
       Ilm which refreshes the mapping before every epoch.

No parameters are introduced: the mapping is stored as integer ``buffers``
exactly like :class:`smm_vr.label_mapping.flm.FlmMapping`, so the logit
selection step (``apply_label_mapping``) can be shared by all three mappings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .flm import IGNORE_INDEX, apply_label_mapping

__all__ = [
    "RlmMapping",
    "RlmLabelMapping",
    "build_rlm_mapping",
    "random_subset",
    "random_injective_mapping",
    "rlm_mapping_from_seed",
    "IGNORE_INDEX",
    "apply_label_mapping",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_num_target_classes(
    num_target_classes: Optional[int] = None, num_classes: Optional[int] = None
) -> int:
    """Accept either spelling for the size of the target label space."""
    if num_target_classes is None:
        num_target_classes = num_classes
    if num_target_classes is None:
        raise ValueError(
            "build_rlm_mapping requires the number of target classes "
            "(num_target_classes or num_classes)."
        )
    num_target_classes = int(num_target_classes)
    if num_target_classes <= 0:
        raise ValueError(f"num_target_classes must be positive, got {num_target_classes}.")
    return num_target_classes


def random_subset(
    num_pretrained_classes: int,
    num_target_classes: int,
    *,
    seed: Optional[int] = 0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample ``Y_sub^P``: a random subset of ``Y^P`` of size ``|Y^T|``.

    The subset is drawn *without replacement*, which is what makes
    ``f_out^Rlm`` injective (Sec. 2.3).
    """
    num_pretrained_classes = int(num_pretrained_classes)
    num_target_classes = int(num_target_classes)
    if num_target_classes > num_pretrained_classes:
        raise ValueError(
            "Cannot draw an injective random label mapping: the target task has "
            f"{num_target_classes} classes but the pre-trained label space only "
            f"has {num_pretrained_classes}."
        )
    if generator is None:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
    permutation = torch.randperm(num_pretrained_classes, generator=generator)
    subset = permutation[:num_target_classes].to(torch.long)
    if device is not None:
        subset = subset.to(device)
    return subset


def random_injective_mapping(
    num_pretrained_classes: int = 1000,
    num_target_classes: Optional[int] = None,
    *,
    num_classes: Optional[int] = None,
    seed: Optional[int] = 0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Draw the random injective mapping ``f_out^Rlm``.

    Returns a dict with

    * ``subset``               -- ``(k^T,)`` ImageNet indices chosen for the task,
    * ``target_to_pretrained`` -- ``(k^T,)`` with ``target_to_pretrained[t] = y^P``,
    * ``pretrained_to_target`` -- ``(|Y^P|,)`` with ``IGNORE_INDEX`` outside the subset.
    """
    num_target_classes = _resolve_num_target_classes(num_target_classes, num_classes)
    subset = random_subset(
        num_pretrained_classes,
        num_target_classes,
        seed=seed,
        generator=generator,
        device=device,
    )

    # Random bijection from the sampled ImageNet labels onto the target labels.
    target_to_pretrained = subset[torch.randperm(
        num_target_classes, generator=generator
    )].to(torch.long)

    pretrained_to_target = torch.full(
        (int(num_pretrained_classes),), int(ignore_index), dtype=torch.long
    )
    target_indices = torch.arange(num_target_classes, dtype=torch.long)
    pretrained_to_target[target_to_pretrained] = target_indices

    # ``f_out^Rlm`` in the literal paper direction: Y_sub^P -> Y^T.
    pretrained_to_target = pretrained_to_target
    return {
        "subset": subset.to(torch.long),
        "target_to_pretrained": target_to_pretrained,
        "pretrained_to_target": pretrained_to_target,
    }


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------
class RlmMapping(nn.Module):
    """Non-parametric random output mapping ``f_out^Rlm`` (Sec. 2.3).

    The mapping is a plain *index* mapping: no learnable parameter is added and
    the mapping is *not* updated during training (``recomputes_each_epoch`` is
    ``False``), mirroring the paper's use of it as a fixed random injective
    function selected prior to the minimisation of Eq. (1).
    """

    #: Rlm is drawn once and kept fixed (contrast with Ilm's ``True``).
    recomputes_each_epoch: bool = False
    #: human readable name used by experiment runners / reporting
    name: str = "Rlm"
    #: logit handling convention consumed by the training loop
    mapping_style: str = "select"

    def __init__(
        self,
        target_to_pretrained: torch.Tensor,
        pretrained_to_target: Optional[torch.Tensor] = None,
        num_pretrained_classes: Optional[int] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        target_to_pretrained = torch.as_tensor(target_to_pretrained, dtype=torch.long).reshape(-1)

        if pretrained_to_target is None:
            if num_pretrained_classes is None:
                num_pretrained_classes = int(target_to_pretrained.max().item()) + 1
            pretrained_to_target = torch.full(
                (int(num_pretrained_classes),), int(ignore_index), dtype=torch.long
            )
            pretrained_to_target[target_to_pretrained] = torch.arange(
                target_to_pretrained.numel(), dtype=torch.long
            )
        pretrained_to_target = torch.as_tensor(pretrained_to_target, dtype=torch.long).reshape(-1)

        self.ignore_index = int(ignore_index)
        self.register_buffer("target_to_pretrained", target_to_pretrained, persistent=True)
        self.register_buffer("pretrained_to_target", pretrained_to_target, persistent=True)
        # Convenience view: the sampled ImageNet label space Y_sub^P.
        self.register_buffer("subset", target_to_pretrained.clone(), persistent=False)

    # -- properties ---------------------------------------------------------
    @property
    def num_target_classes(self) -> int:
        return int(self.target_to_pretrained.numel())

    @property
    def num_pretrained_classes(self) -> int:
        return int(self.pretrained_to_target.numel())

    # -- forward / mapping --------------------------------------------------
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Select the logits of ``Y_sub^P`` -> per-target-class scores."""
        return apply_label_mapping(logits, self.target_to_pretrained)

    def map_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode ImageNet logits into predicted *target* labels."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        selected = self.forward(logits)
        return torch.argmax(selected, dim=-1)

    def update(self, *args: Any, **kwargs: Any) -> bool:
        """No-op: ``f_out^Rlm`` is fixed before training (Sec. 2.3)."""
        return False

    # -- introspection ------------------------------------------------------
    def is_injective(self) -> bool:
        """``f_out^Rlm`` must be injective (``y1 != y2 -> f(y1) != f(y2)``)."""
        if self.target_to_pretrained.numel() == 0:
            return True
        unique = torch.unique(self.target_to_pretrained)
        return bool(unique.numel() == self.target_to_pretrained.numel())

    def extra_repr(self) -> str:
        return (
            f"num_target_classes={self.num_target_classes}, "
            f"num_pretrained_classes={self.num_pretrained_classes}, "
            f"injective={self.is_injective()}, "
            f"recomputes_each_epoch={self.recomputes_each_epoch}"
        )


#: Paper-style alias (``f_out^Rlm``).
RlmLabelMapping = RlmMapping


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_rlm_mapping(
    num_target_classes: Optional[int] = None,
    num_pretrained_classes: int = 1000,
    seed: int = 0,
    device: Optional[torch.device] = None,
    *,
    num_classes: Optional[int] = None,
    subset: Optional[torch.Tensor] = None,
    target_to_pretrained: Optional[torch.Tensor] = None,
    pretrained_to_target: Optional[torch.Tensor] = None,
    ignore_index: int = IGNORE_INDEX,
    return_dict: bool = False,
    # tolerated/ignored arguments so callers can pass a uniform signature
    classifier: Optional[nn.Module] = None,
    model: Optional[nn.Module] = None,
    data_loader: Any = None,
    f_in: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Any:
    """Build the random injective output mapping ``f_out^Rlm``.

    Parameters
    ----------
    num_target_classes / num_classes:
        size ``k^T`` of the target label space.
    num_pretrained_classes:
        size ``|Y^P|`` of the ImageNet label space (1000 by default).
    seed:
        RNG seed for the random subset + random bijection.  The paper draws the
        mapping *before* training and keeps it fixed, so the same seed yields
        the same mapping across the whole run.
    return_dict:
        when ``True`` return the raw index tensors instead of the module.

    Extra ``classifier``/``model``/``data_loader``/``f_in`` keywords are accepted
    (and ignored) so that experiment code can call every mapping builder with a
    uniform signature; Rlm is purely random and does not inspect the data.
    """
    num_target_classes = _resolve_num_target_classes(num_target_classes, num_classes)

    if target_to_pretrained is None:
        if subset is None:
            mapping = random_injective_mapping(
                num_pretrained_classes=num_pretrained_classes,
                num_target_classes=num_target_classes,
                seed=seed,
                device=device,
                ignore_index=ignore_index,
            )
        else:
            mapping = _mapping_from_subset(
                subset=subset,
                num_target_classes=num_target_classes,
                num_pretrained_classes=num_pretrained_classes,
                seed=seed,
                device=device,
                ignore_index=ignore_index,
            )
        subset = mapping["subset"]
        target_to_pretrained = mapping["target_to_pretrained"]
        pretrained_to_target = mapping["pretrained_to_target"]

    if return_dict:
        return {
            "subset": subset,
            "target_to_pretrained": target_to_pretrained,
            "pretrained_to_target": pretrained_to_target,
        }

    module = RlmMapping(
        target_to_pretrained=target_to_pretrained,
        pretrained_to_target=pretrained_to_target,
        num_pretrained_classes=num_pretrained_classes,
        ignore_index=ignore_index,
    )
    if device is not None:
        module = module.to(device)
    return module


def _mapping_from_subset(
    subset: torch.Tensor,
    num_target_classes: int,
    num_pretrained_classes: int,
    *,
    seed: int = 0,
    device: Optional[torch.device] = None,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Turn a user supplied ``Y_sub^P`` into the full random bijection."""
    subset = torch.as_tensor(subset, dtype=torch.long).reshape(-1)
    if subset.numel() != num_target_classes:
        raise ValueError(
            f"subset must contain exactly {num_target_classes} ImageNet labels, "
            f"got {subset.numel()}."
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    target_to_pretrained = subset[torch.randperm(num_target_classes, generator=generator)]
    pretrained_to_target = torch.full(
        (int(num_pretrained_classes),), int(ignore_index), dtype=torch.long
    )
    pretrained_to_target[target_to_pretrained] = torch.arange(num_target_classes, dtype=torch.long)
    if device is not None:
        subset = subset.to(device)
        target_to_pretrained = target_to_pretrained.to(device)
        pretrained_to_target = pretrained_to_target.to(device)
    return {
        "subset": subset,
        "target_to_pretrained": target_to_pretrained,
        "pretrained_to_target": pretrained_to_target,
    }


def rlm_mapping_from_seed(
    num_target_classes: int,
    num_pretrained_classes: int = 1000,
    seed: int = 0,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """Convenience wrapper returning the raw index tensors (dict form)."""
    return build_rlm_mapping(
        num_target_classes=num_target_classes,
        num_pretrained_classes=num_pretrained_classes,
        seed=seed,
        return_dict=True,
        **kwargs,
    )
