"""Training objectives of the paper (Sec. 3 and App. C.4).

FARE (Eq. (3) of the paper, "Fine-tuning with Adversarial Residual Embeddings")
-------------------------------------------------------------------------------
Given the *frozen* original CLIP image encoder :math:`\\phi_{Org}` and the
fine-tuned encoder :math:`\\phi_{FT}`, the FARE loss of an image :math:`x` is

.. math::

    L_{FARE}(x) = \\max_{\\|\\delta\\|_\\infty \\le \\epsilon}
        \\big\\|\\phi_{FT}(x + \\delta) - \\phi_{Org}(x)\\big\\|_2^2
        + \\lambda \\big\\|\\phi_{FT}(x) - \\phi_{Org}(x)\\big\\|_2^2 ,

i.e. it *regresses the fine-tuned embedding onto the embedding of the original
CLIP model*, both at the adversarial point (inner maximization, solved with
10 PGD steps during training) and at the clean point (the regularization term
with weight :math:`\\lambda`, set to 1 in all experiments of the paper).
Because the target is the embedding of the *original* model rather than a class
label, the scheme is **unsupervised**: no labels, and in particular no text
pair, are required.

Squared :math:`\\ell_2` is used because it is a monotone function of the cosine
similarity of the embeddings (App. B.4, footnote 1), which is what zero-shot
classification and LVLM connectors are built on, and because it keeps the
*non-normalized* embeddings close, see App. C.4 / Theorem A.1.

TeCoA (Mao et al., 2023), the supervised baseline of the paper
--------------------------------------------------------------

TeCoA fine-tunes CLIP with the usual image-text contrastive loss, but on
adversarially perturbed images:

.. math::

    L_{TeCoA}(x) = \\max_{\\|\\delta\\|_\\infty \\le \\epsilon}
        \\mathrm{CE}\\big(\\phi_{FT}(x+\\delta) \\cdot \\psi(t), y\\big) .

It is called *supervised* in the paper because the text tower provides the class
information ("supervised adversarial fine-tuning", Fig. 1, Table 1).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "fare_loss",
    "tecoa_loss",
    "clean_embedding_loss",
    "adversarial_embedding_loss",
    "embedding_distance",
]


def embedding_distance(a: torch.Tensor, b: torch.Tensor, norm: str = "l2_squared") -> torch.Tensor:
    """Per-sample distance between two sets of embeddings.

    ``norm`` is one of

    * ``'l2_squared'`` -- :math:`\\|a - b\\|_2^2` (Eq. (3) of the paper),
    * ``'l1'`` -- :math:`\\|a - b\\|_1` (the ablation of App. B.4),
    * ``'l2'`` -- :math:`\\|a - b\\|_2` (used for reporting).
    """
    flat_a = a.flatten(1)
    flat_b = b.flatten(1)
    if norm == "l2_squared":
        return (flat_a - flat_b).pow(2).sum(dim=1)
    if norm == "l1":
        return (flat_a - flat_b).abs().sum(dim=1)
    if norm == "l2":
        return (flat_a - flat_b).pow(2).sum(dim=1).sqrt()
    raise ValueError(f"unknown norm {norm!r}")


def fare_loss(
    phi_ft_adv: torch.Tensor,
    phi_ft_clean: torch.Tensor,
    phi_org: torch.Tensor,
    clean_weight: float = 1.0,
    norm: str = "l2_squared",
    reduction: str = "mean",
) -> torch.Tensor:
    """FARE loss, Eq. (3) of the paper.

    Parameters
    ----------
    phi_ft_adv:
        ``phi_FT(x + delta)`` -- embedding of the fine-tuned encoder at the
        *adversarial* point (the one that maximizes the distance, produced by the
        inner PGD maximization).
    phi_ft_clean:
        ``phi_FT(x)`` -- embedding of the fine-tuned encoder at the clean point.
    phi_org:
        ``phi_Org(x)`` -- embedding of the frozen original CLIP encoder.  Its
        gradient is not required; ``.detach()`` is applied internally.
    clean_weight:
        the weight :math:`\\lambda` of the clean term (1.0 in the paper).
    norm:
        ``'l2_squared'`` (paper) or ``'l1'`` (App. B.4 ablation).
    """
    phi_org = phi_org.detach()
    adv_term = embedding_distance(phi_ft_adv, phi_org, norm=norm)
    clean_term = embedding_distance(phi_ft_clean, phi_org, norm=norm)
    loss = adv_term + clean_weight * clean_term
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unknown reduction {reduction!r}")


def tecoa_loss(
    image_features_adv: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """TeCoA loss (Mao et al., 2023) -- supervised contrastive loss on adversarial images.

    ``text_features`` are the (normalized) text embeddings of the classes present
    in the batch, ``labels`` indexes into them (defaults to the identity, i.e. one
    text per sample).  ``logit_scale`` is the temperature parameter of CLIP.
    """
    logits_per_image = logit_scale * image_features_adv @ text_features.t()
    if labels is None:
        labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    return F.cross_entropy(logits_per_image, labels, reduction=reduction)


def clean_embedding_loss(phi_ft_clean: torch.Tensor, phi_org: torch.Tensor, reduction: str = "none"):
    """``L_clean(x) = ||phi_FT(x) - phi_Org(x)||_2^2`` -- Eq. (4), App. C.4 (Table 14)."""
    loss = embedding_distance(phi_ft_clean, phi_org.detach(), norm="l2_squared")
    if reduction == "mean":
        return loss.mean()
    return loss


def adversarial_embedding_loss(phi_ft_adv: torch.Tensor, phi_org: torch.Tensor, reduction: str = "none"):
    """``L_adv(x) = max_{||z-x||_inf <= eps} ||phi_FT(z) - phi_Org(x)||_2^2`` -- Eq. (5)."""
    loss = embedding_distance(phi_ft_adv, phi_org.detach(), norm="l2_squared")
    if reduction == "mean":
        return loss.mean()
    return loss
