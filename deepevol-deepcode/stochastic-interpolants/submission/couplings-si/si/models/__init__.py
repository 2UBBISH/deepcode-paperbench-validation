"""Model package for coupled stochastic interpolants.

Re-exports the velocity network :math:`\\hat b_t(x, \\xi)` (paper Appendix B) and the
optional score network :math:`\\hat g_t(x, \\xi)` (paper Section 3.1, Eq. 4/7), together
with the conditioning embeddings used by both.

The velocity net is the only network required by the reported deterministic-ODE
experiments in Section 4.  The score net (``score_net.py``) is optional and only needed
for SDE sampling (:math:`\\gamma_t \\ne 0` paths).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .embeddings import (
    ClassLabelEmbedding,
    ImageConditioningEmbedding,
    RandomOrLearnedSinusoidalPosEmb,
    SinusoidalPositionEmbeddings,
    TimeEmbedding,
    combine_embeddings,
    get_time_embedding,
    sum_embeddings,
)
from .unet import (
    APPENDIX_B_CONFIG,
    Attention,
    Block,
    Downsample,
    LinearAttention,
    PreNorm,
    Residual,
    ResnetBlock,
    UNet,
    Unet,
    Upsample,
    VelocityUNet,
    appendix_b_config,
    unet_from_config,
)

__all__ = [
    # velocity network (Appendix B)
    "VelocityUNet",
    "Unet",
    "UNet",
    "unet_from_config",
    "appendix_b_config",
    "APPENDIX_B_CONFIG",
    # U-Net building blocks
    "Attention",
    "LinearAttention",
    "ResnetBlock",
    "Block",
    "PreNorm",
    "Residual",
    "Downsample",
    "Upsample",
    # embeddings / conditioning
    "TimeEmbedding",
    "ClassLabelEmbedding",
    "ImageConditioningEmbedding",
    "SinusoidalPositionEmbeddings",
    "RandomOrLearnedSinusoidalPosEmb",
    "combine_embeddings",
    "sum_embeddings",
    "get_time_embedding",
    # optional score network
    "ScoreUNet",
    "ScoreNet",
    "score_net_from_config",
    "build_model",
    "MODELS",
]

# ---------------------------------------------------------------------------
# Optional score network (Section 3.1, Eq. 4 second line; Eq. 7 second line)
# ---------------------------------------------------------------------------
# Kept in a try/except so velocity-only training keeps working if the optional
# module is unavailable.
try:  # pragma: no cover - trivial import guard
    from .score_net import (  # type: ignore
        MODEL_CONFIG as _SCORE_CONFIG,
    )
    from .score_net import (  # type: ignore
        ScoreNet,
        ScoreUNet,
        score_net_from_config,
    )
except Exception:  # pragma: no cover
    ScoreUNet = None  # type: ignore
    ScoreNet = None  # type: ignore
    score_net_from_config = None  # type: ignore


# ---------------------------------------------------------------------------
# Registry + factory (config-friendly string dispatch)
# ---------------------------------------------------------------------------
MODELS: Dict[str, Optional[type]] = {
    "velocity": VelocityUNet,
    "velocity_unet": VelocityUNet,
    "unet": VelocityUNet,
    "b": VelocityUNet,
    "hat_b": VelocityUNet,
    "score": ScoreUNet,
    "score_unet": ScoreUNet,
    "score_net": ScoreUNet,
    "g": ScoreUNet,
    "hat_g": ScoreUNet,
}


def build_model(name: str = "velocity", **kwargs: Any):
    """Build a network by name.

    Parameters
    ----------
    name:
        One of ``"velocity"``/``"unet"``/``"b"`` (the velocity net
        :math:`\\hat b_t`) or ``"score"``/``"g"`` (the optional score net
        :math:`\\hat g_t`).
    **kwargs:
        Forwarded to the network builder.  Keys listed in ``MODEL_CONFIG``
        (e.g. ``dim_mults``, ``channels``, ``resnet_block_groups``) are picked
        out; the rest are passed through.

    Raises
    ------
    ValueError
        If ``name`` is unknown, or if the score network was requested but its
        module is unavailable.
    """
    key = str(name).lower()
    if key not in MODELS:
        raise ValueError(
            f"unknown model {name!r}; valid names: {sorted(MODELS)}"
        )
    cls = MODELS[key]
    if cls is None:
        raise ImportError(
            f"model {name!r} requested but 'si.models.score_net' is unavailable"
        )
    if cls is VelocityUNet:
        return unet_from_config(**kwargs)
    return score_net_from_config(**kwargs)  # type: ignore[misc]


def describe() -> Dict[str, str]:  # pragma: no cover - introspection helper
    """Short human-readable description of the package contents."""
    return {
        "velocity": "U-Net velocity net b_hat_t(x, xi) (Appendix B)",
        "score": "U-Net score net g_hat_t(x, xi) (Section 3.1, optional)",
        "embeddings": "time / class-label / image-shaped conditioning",
    }
