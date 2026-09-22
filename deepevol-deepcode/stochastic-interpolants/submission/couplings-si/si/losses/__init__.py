"""Losses for stochastic interpolants with data-dependent couplings.

Two simulation-free objectives are exposed:

* :func:`velocity_loss` -- the empirical velocity-regression objective
  :math:`\\hat L_b` of Eq. (22) in Section 3.4 of the paper, used for all
  reported (deterministic ODE) experiments.
* :func:`score_loss` -- the optional :math:`\\hat L_g` objective of Eq. (7)
  (Section 3.1), only needed when :math:`\\gamma_t \\neq 0` (SDE sampling paths).

Both are also available as :class:`torch.nn.Module` wrappers
(:class:`VelocityLoss`, :class:`ScoreLoss`) so that ``train.py`` can build them
straight from a config.

The heavy lifting lives in the sibling modules; this package initializer only
re-exports the public API.
"""

from __future__ import annotations

from typing import List

from .velocity_loss import (
    VelocityLoss,
    call_model,
    compute_velocity_loss,
    estimate_transport_cost,
    sample_interpolant_noise,
    sample_uniform_t,
    transport_cost_estimate,
    velocity_loss,
)

try:  # pragma: no cover - score loss is optional and may be absent
    from .score_loss import (
        ScoreLoss,
        compute_score_loss,
        score_loss,
    )
except Exception:  # pragma: no cover
    ScoreLoss = None  # type: ignore[assignment]
    compute_score_loss = None  # type: ignore[assignment]
    score_loss = None  # type: ignore[assignment]


__all__: List[str] = [
    # velocity loss (Eq. 22)
    "velocity_loss",
    "compute_velocity_loss",
    "VelocityLoss",
    # score loss (Eq. 7)
    "score_loss",
    "compute_score_loss",
    "ScoreLoss",
    # shared helpers
    "sample_uniform_t",
    "sample_interpolant_noise",
    "call_model",
    "transport_cost_estimate",
    "estimate_transport_cost",
]


def get_loss(name: str, **kwargs):
    """Build a loss module by name.

    Parameters
    ----------
    name:
        ``"velocity"``/``"velocity_loss"``/``"b"`` or
        ``"score"``/``"score_loss"``/``"g"``.
    **kwargs:
        Forwarded to the corresponding loss constructor.
    """
    key = str(name).lower()
    if key in ("velocity", "velocity_loss", "b", "l_b", "hat_b"):
        return VelocityLoss(**kwargs)
    if key in ("score", "score_loss", "g", "l_g", "hat_g"):
        if ScoreLoss is None:  # pragma: no cover
            raise ImportError(
                "si.losses.score_loss is unavailable; cannot build a score loss."
            )
        return ScoreLoss(**kwargs)
    raise ValueError(
        f"Unknown loss {name!r}. Valid names: "
        "'velocity', 'velocity_loss', 'b', 'score', 'score_loss', 'g'."
    )


LOSSES = {
    "velocity": VelocityLoss,
    "velocity_loss": VelocityLoss,
    "score": ScoreLoss,
    "score_loss": ScoreLoss,
}
