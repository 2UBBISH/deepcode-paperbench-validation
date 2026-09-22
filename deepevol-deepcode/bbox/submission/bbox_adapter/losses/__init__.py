"""Loss functions for BBox-Adapter.

This package contains the training objectives used to fit the lightweight
energy adapter ``g_theta(x, y)``:

* :mod:`bbox_adapter.losses.nce` -- the paper's ranking-based Noise Contrastive
  Estimation (NCE) objective derived from the energy-based-model view of the
  black-box LLM (Eq. (1) posterior, Eq. (2) loss, Eq. (3) gradient), together
  with the ``alpha * E[g_theta^2]`` energy regularizer combination.
* :mod:`bbox_adapter.losses.mlm` -- the masked-language-model ablation used in
  Section 4.5 / Table 5 (random-word masking and masked-word probability
  scoring) that is compared against the ranking NCE loss.

Nothing in this package ever differentiates through the black-box LLM: only the
scalar energies produced by the adapter carry gradient.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ranking-based NCE (main objective, Sections 3.2 / 4.5)
# ---------------------------------------------------------------------------
from .nce import (
    NCELossConfig,
    RankingNCELoss,
    binary_nce_loss,
    build_contrastive_tensors,
    compute_nce_loss,
    is_list_negatives,
    log_softmax_posterior,
    nce_gradient_terms,
    nce_loss_from_model,
    nce_objective,
    pairwise_ranking_loss,
    ranking_accuracy,
    ranking_nce_loss,
    score_contrastive_sets,
    softmax_posterior,
)

__all__ = [
    # --- configs / modules (Eq. 2 & Eq. 3) ---
    "NCELossConfig",
    "RankingNCELoss",
    # --- core losses ---
    "compute_nce_loss",
    "ranking_nce_loss",
    "nce_objective",
    "nce_gradient_terms",
    "nce_loss_from_model",
    # --- posteriors / helpers ---
    "softmax_posterior",
    "log_softmax_posterior",
    "ranking_accuracy",
    "build_contrastive_tensors",
    "score_contrastive_sets",
    "is_list_negatives",
    # --- diagnostic / ablation surrogates ---
    "binary_nce_loss",
    "pairwise_ranking_loss",
]

# ---------------------------------------------------------------------------
# Masked-LM ablation (Section 4.5, Table 5).  Imported defensively so that the
# NCE objective remains usable even if the ablation module is absent.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    from .mlm import (
        MLMLossConfig,
        MaskedLMLoss,
        apply_random_mask,
        compute_mlm_loss,
        masked_word_probability,
        mlm_loss_from_model,
        mlm_objective,
    )

    _MLM_EXPORTS = [
        "MLMLossConfig",
        "MaskedLMLoss",
        "compute_mlm_loss",
        "mlm_objective",
        "mlm_loss_from_model",
        "masked_word_probability",
        "apply_random_mask",
    ]
    __all__ += _MLM_EXPORTS
except Exception:  # noqa: BLE001 - module optional at import time
    pass


def get_loss(name: str, **kwargs):
    """Factory returning the loss module selected by ``name``.

    Parameters
    ----------
    name:
        ``"nce"`` / ``"ranking_nce"`` (default paper objective) or
        ``"mlm"`` / ``"masked"`` for the Section 4.5 ablation.
    **kwargs:
        Forwarded to the corresponding config/module constructor.
    """
    key = str(name).strip().lower().replace("-", "_")
    if key in {"nce", "ranking_nce", "ranking", "nce_loss", "default"}:
        return RankingNCELoss(**kwargs)
    if key in {"mlm", "masked", "masked_lm", "mlm_loss"}:
        globals_map = globals()
        if "MaskedLMLoss" not in globals_map:  # pragma: no cover
            raise ImportError(
                "The MLM ablation loss is unavailable (bbox_adapter.losses.mlm "
                "could not be imported)."
            )
        return MaskedLMLoss(**kwargs)  # type: ignore[name-defined]
    raise ValueError(
        f"Unknown loss '{name}'. Expected one of: 'nce', 'ranking_nce', 'mlm'."
    )
