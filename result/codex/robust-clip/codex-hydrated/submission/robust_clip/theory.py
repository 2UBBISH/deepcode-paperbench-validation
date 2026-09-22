"""Theorem 3.1 of the paper (proved in App. A).

Preserving the embedding in ℓ₂ also preserves the cosine similarities that the
zero-shot classifier uses::

    |cos(φ_FT(x), ψ(t)) − cos(φ_org(x), ψ(t))|
        ≤ min( 2/||φ_org(x)||₂ , 2/||φ_FT(x)||₂ ) · ||φ_FT(x) − φ_org(x)||₂

This is the reason why FARE can be plugged into LLaVA / OpenFlamingo without
retraining: the zero-shot logits of Eq. (1) can only move as much as the right
hand side allows, and :func:`robust_clip.training.losses.FARELoss` explicitly
minimises that term (both on clean and on adversarial inputs).

:func:`theorem_3_1_bound` and :func:`cosine_difference` implement the two sides of
the inequality; ``tests/test_smoke.py::test_theorem_3_1_holds`` checks it.
"""
from __future__ import annotations

import torch


def normalize(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_difference(phi_org: torch.Tensor, phi_ft: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Left hand side: the change of the zero-shot cosine similarity.

    ``phi_org``/``phi_ft`` are ``B x D`` image embeddings, ``psi`` is ``K x D``
    (the text embeddings); the returned tensor is ``B x K``.
    """
    return (
        normalize(phi_ft) @ normalize(psi).t() - normalize(phi_org) @ normalize(psi).t()
    ).abs()


def theorem_3_1_bound(phi_org: torch.Tensor, phi_ft: torch.Tensor) -> torch.Tensor:
    """Right hand side of Theorem 3.1, per sample (``B``)."""
    distance = (phi_ft - phi_org).norm(dim=-1)
    scale = torch.minimum(
        2.0 / phi_org.norm(dim=-1).clamp_min(1e-12),
        2.0 / phi_ft.norm(dim=-1).clamp_min(1e-12),
    )
    return scale * distance
