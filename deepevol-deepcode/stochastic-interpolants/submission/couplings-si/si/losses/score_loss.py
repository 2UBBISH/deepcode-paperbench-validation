"""Score-regression loss for stochastic interpolants with data-dependent couplings.

This module implements the *optional* score objective ``L_g`` of the paper, i.e. the
second line of Eq. (7) (and its conditioned generalization, Eq. (29) in Appendix A):

.. math::

    L_g(\\hat g) = \\int_0^1 \\mathbb{E}\\big[|\\hat g_t(I_t, \\xi)|^2
                   - 2\\, z \\cdot \\hat g_t(I_t, \\xi)\\big] \\, dt

whose unique minimizer is

.. math::

    g_t(x, \\xi) = \\mathbb{E}(z \\mid I_t = x, \\xi)

with :math:`I_t = \\alpha_t x_0 + \\beta_t x_1 + \\gamma_t z` and :math:`z \\sim N(0, Id)`
independent of :math:`(x_0, x_1, \\xi)`.

The function ``g`` is what makes the score available: Theorem 3.1 (resp. Theorem A.1 in
the conditioned case) shows that for every :math:`t` with :math:`\\gamma_t \\neq 0`

.. math::

    \\nabla \\log \\rho_t(x \\mid \\xi) = -\\gamma_t^{-1} g_t(x, \\xi),

which is used in the forward/backward SDEs, Eqs. (11)/(13) (resp. Eqs. (43)/(45)):

.. math::

    dX_t^F = b_t(X_t^F, \\xi)\\,dt - \\epsilon_t \\gamma_t^{-1} g_t(X_t^F, \\xi)\\,dt
             + \\sqrt{2\\epsilon_t}\\,dW_t,\\\\
    dX_t^R = b_t(X_t^R, \\xi)\\,dt + \\epsilon_t \\gamma_t^{-1} g_t(X_t^R, \\xi)\\,dt
             + \\sqrt{2\\epsilon_t}\\,dW_t .

This objective is *not* used for the deterministic probability-flow ODE experiments
reported in the paper (Table 2/3, Figs. 3-6 all use :math:`\\hat b_t` only, with
:math:`\\gamma_t = 0` for the coupled in-painting/SR couplings); it is provided for
completeness and for SDE-based sampling.

Note: the identity above requires :math:`\\gamma_t \\neq 0`. If the selected interpolant
preset has a structurally zero :math:`\\gamma_t` (e.g. ``"gamma0"``, ``"inpainting"``,
``"superres"``), then :math:`z \\perp I_t` and the minimizer of the objective is the
identically-zero function: the regression is still well posed but the resulting network
carries no score information. Pass ``strict=True`` to turn that situation into an error.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from ..interpolants.coefficients import Coefficients, get_coefficients
from ..interpolants.interpolant import Interpolant
from .velocity_loss import call_model, sample_interpolant_noise, sample_uniform_t

__all__ = [
    "score_loss",
    "compute_score_loss",
    "ScoreLoss",
    "score_from_g",
    "resolve_interpolant",
    "sample_uniform_t",
    "sample_interpolant_noise",
    "call_model",
]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def resolve_interpolant(
    interpolant: Optional[Union[Interpolant, Coefficients, str]] = None,
    coefficients: Optional[Union[Coefficients, Interpolant, str]] = None,
) -> Interpolant:
    """Resolve the many accepted spellings of "which interpolant" to an ``Interpolant``.

    Accepts an :class:`~si.interpolants.interpolant.Interpolant`, a
    :class:`~si.interpolants.coefficients.Coefficients`, or a preset name
    (``"linear"``, ``"gamma0"``, ``"inpainting"``, ``"superres"``). Defaults to the
    ``"linear"`` preset, which is the only one that carries a non-trivial score
    (:math:`\\gamma_t = \\sqrt{2t(1-t)} > 0` on ``(0, 1)``).
    """
    for candidate in (interpolant, coefficients):
        if candidate is None:
            continue
        if isinstance(candidate, Interpolant):
            return candidate
        if isinstance(candidate, Coefficients):
            return Interpolant(candidate)
        if isinstance(candidate, str):
            return Interpolant(get_coefficients(candidate))
        raise TypeError(
            "expected an Interpolant, Coefficients or preset name, got "
            f"{type(candidate).__name__}"
        )
    return Interpolant(get_coefficients("linear"))


def _reduce(per_item: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return per_item
    if reduction == "mean":
        return per_item.mean()
    if reduction == "sum":
        return per_item.sum()
    raise ValueError(f"unknown reduction {reduction!r}; expected 'mean', 'sum' or 'none'")


def _gamma_of(interp: Interpolant, t: torch.Tensor) -> torch.Tensor:
    """Return ``gamma_t`` as a tensor broadcastable against a ``(B, C, ...)`` batch."""
    _, _, gamma = interp.alpha_beta_gamma(t)
    if not torch.is_tensor(gamma):
        gamma = torch.as_tensor(gamma, dtype=t.dtype, device=t.device)
    return gamma


# --------------------------------------------------------------------------------------
# core loss
# --------------------------------------------------------------------------------------
def score_loss(
    model: Any,
    x0: Optional[torch.Tensor] = None,
    x1: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    xi: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    interpolant: Optional[Union[Interpolant, Coefficients, str]] = None,
    coefficients: Optional[Union[Coefficients, Interpolant, str]] = None,
    mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    reduction: str = "mean",
    return_dict: bool = False,
    model_fn: Optional[Callable[..., torch.Tensor]] = None,
    t_eps: float = 0.0,
    I_t: Optional[torch.Tensor] = None,
    strict: bool = False,
    warn_zero_gamma: bool = True,
    gamma_eps: float = 1e-12,
) -> Union[torch.Tensor, Dict[str, Any]]:
    r"""Empirical estimate of the score objective ``L_g`` (Eq. 7 second line / Eq. 29).

    The estimator is simulation-free; with a batch of size ``B`` it draws per-item
    :math:`t_i \sim U(0,1)`, :math:`z_i \sim N(0, I)`, forms
    :math:`I_{t_i} = \alpha_{t_i} x_0 + \beta_{t_i} x_1 + \gamma_{t_i} z_i` and returns

    .. math::

        \hat L_g = \frac{1}{B}\sum_{i=1}^B
        \big(|\hat g_{t_i}(I_{t_i}, \xi_i)|^2 - 2 z_i \cdot \hat g_{t_i}(I_{t_i}, \xi_i)\big).

    Parameters
    ----------
    model:
        The score network :math:`\hat g_t(x, \xi)` (or a callable). Ignored if ``model_fn``
        is supplied.
    x0, x1:
        Coupled base/target samples ``(x0, x1) ~ rho(x0, x1 | xi)``. Required unless
        ``I_t`` is passed directly (in which case ``x1`` is only needed to infer ``z``'s
        shape when ``z`` is not given).
    t:
        Optional per-item times in ``[0, 1]`` of shape ``(B,)``. Drawn uniformly when
        omitted.
    xi:
        Optional conditioning (mask for in-painting, upsampled low-res image for SR,
        class labels, ...), forwarded to the network.
    z:
        Optional noise sample used both in ``I_t`` and as the regression target. Drawn
        from ``N(0, Id)`` when omitted.
    mask_fn:
        Optional callable applied to the network output (e.g.
        ``InpaintingCoupling.mask_velocity``). Kept for interface symmetry with the
        velocity loss; for the coupled in-painting/SR tasks the score regression is
        degenerate (``gamma_t = 0``), see the module docstring.
    strict:
        Raise if the selected interpolant has a structurally zero ``gamma_t``.
    warn_zero_gamma:
        Emit a one-line warning (instead of raising) in the same situation.
    gamma_eps:
        Small constant protecting the ``-g / gamma`` score conversion from division by
        zero (used for the optional ``score_hat`` diagnostics entry).

    Returns
    -------
    ``torch.Tensor`` scalar (or per-item tensor when ``reduction="none"``), or a dict
    with keys ``loss``, ``per_item``, ``g_hat``, ``I_t``, ``z``, ``t``, ``gamma_t``,
    ``uses_noise``, ``batch_size`` when ``return_dict=True``.
    """
    interp = resolve_interpolant(interpolant, coefficients)

    if z is None:
        if x1 is None:
            raise ValueError(
                "score_loss needs either `z` (the regression target) or `x1` "
                "(to infer its shape/device)."
            )
        z = sample_interpolant_noise(x1, generator=generator)

    if t is None:
        t = sample_uniform_t(
            z.shape[0], device=z.device, dtype=z.dtype, generator=generator, eps=t_eps
        )

    if I_t is None:
        if x0 is None or x1 is None:
            raise ValueError(
                "score_loss needs `x0` and `x1` to build I_t, or a precomputed `I_t`."
            )
        I_t = interp.I_t(x0, x1, t, z=z)

    uses_noise = bool(interp.uses_noise())
    if not uses_noise:
        msg = (
            f"interpolant preset {interp.name!r} has gamma_t = 0: the score identity "
            "grad log rho_t = -gamma_t^{-1} g_t does not hold and the minimizer of L_g "
            "is the zero function. Score is only used for SDE paths with gamma_t != 0."
        )
        if strict:
            raise ValueError(msg)
        if warn_zero_gamma:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # network evaluation ---------------------------------------------------------------
    if model_fn is not None:
        g_hat = model_fn(I_t, t, xi, y)
    else:
        g_hat = call_model(model, I_t, t, xi=xi, y=y)
    if isinstance(g_hat, (tuple, list)):
        g_hat = g_hat[0]
    if mask_fn is not None:
        g_hat = mask_fn(g_hat)

    # L_g estimator: mean(|g_hat|^2 - 2 z . g_hat) -------------------------------------
    flat_g = g_hat.reshape(g_hat.shape[0], -1)
    flat_z = z.reshape(z.shape[0], -1)
    per_item = (flat_g * flat_g).sum(dim=1) - 2.0 * (flat_z * flat_g).sum(dim=1)
    loss = _reduce(per_item, reduction)

    if not return_dict:
        return loss

    gamma_t = _gamma_of(interp, t)
    out: Dict[str, Any] = {
        "loss": loss,
        "per_item": per_item,
        "g_hat": g_hat,
        "I_t": I_t,
        "z": z,
        "t": t,
        "gamma_t": gamma_t,
        "uses_noise": uses_noise,
        "batch_size": int(g_hat.shape[0]) if g_hat.dim() > 0 else 1,
    }
    if uses_noise:
        out["score_hat"] = score_from_g(g_hat, t, interpolant=interp, eps=gamma_eps)
    return out


def compute_score_loss(*args: Any, **kwargs: Any) -> Union[torch.Tensor, Dict[str, Any]]:
    """Alias of :func:`score_loss` (functional core of the score objective)."""
    return score_loss(*args, **kwargs)


# --------------------------------------------------------------------------------------
# score <-> g conversion (Theorem 3.1 / A.1, Eq. 3 / (28))
# --------------------------------------------------------------------------------------
def score_from_g(
    g: torch.Tensor,
    t: torch.Tensor,
    interpolant: Optional[Union[Interpolant, Coefficients, str]] = None,
    coefficients: Optional[Union[Coefficients, Interpolant, str]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Convert the learned ``g_t`` into the score via :math:`\nabla\log\rho_t = -\gamma_t^{-1} g_t`.

    Entries where :math:`|\gamma_t| < \epsilon` are set to zero, since the identity only
    holds for :math:`\gamma_t \neq 0`.
    """
    interp = resolve_interpolant(interpolant, coefficients)
    gamma = _gamma_of(interp, t)
    while gamma.dim() < g.dim():
        gamma = gamma.unsqueeze(-1)
    safe = gamma.abs() >= eps
    inv = torch.where(safe, 1.0 / torch.where(safe, gamma, torch.ones_like(gamma)), torch.zeros_like(gamma))
    return torch.where(safe, -inv * g, torch.zeros_like(g))


# --------------------------------------------------------------------------------------
# nn.Module wrapper
# --------------------------------------------------------------------------------------
class ScoreLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`score_loss`.

    Mirrors :class:`si.losses.velocity_loss.VelocityLoss`; ``forward`` takes the score
    network as its first argument so the module itself stays parameter-free (all model
    parameters are owned by the caller / by the training script).

    Parameters
    ----------
    coefficients, interpolant:
        Interpolant selection (preset name, ``Coefficients`` or ``Interpolant``).
    mask_fn:
        Optional output hook (e.g. masking observed pixels for in-painting).
    reduction:
        ``"mean"`` (default), ``"sum"`` or ``"none"``.
    return_dict:
        Default value for the ``return_dict`` flag of the underlying function.
    t_eps:
        Optional interior margin for the uniform sampling of ``t`` (0 = endpoints allowed).
    strict, warn_zero_gamma, gamma_eps:
        Forwarded to :func:`score_loss`.
    """

    def __init__(
        self,
        coefficients: Union[str, Coefficients, Interpolant, None] = "linear",
        interpolant: Optional[Union[Interpolant, Coefficients, str]] = None,
        mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reduction: str = "mean",
        return_dict: bool = False,
        t_eps: float = 0.0,
        strict: bool = False,
        warn_zero_gamma: bool = True,
        gamma_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.interpolant = resolve_interpolant(interpolant, coefficients)
        self.mask_fn = mask_fn
        self.reduction = reduction
        self.return_dict = return_dict
        self.t_eps = t_eps
        self.strict = strict
        self.warn_zero_gamma = warn_zero_gamma
        self.gamma_eps = gamma_eps

    # ------------------------------------------------------------------ properties
    @property
    def coefficients_name(self) -> str:
        return self.interpolant.name

    @property
    def uses_noise(self) -> bool:
        return bool(self.interpolant.uses_noise())

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        model: Any,
        x0: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
        xi: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: Optional[bool] = None,
        I_t: Optional[torch.Tensor] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        return score_loss(
            model,
            x0=x0,
            x1=x1,
            t=t,
            xi=xi,
            z=z,
            y=y,
            interpolant=self.interpolant,
            mask_fn=mask_fn if mask_fn is not None else self.mask_fn,
            generator=generator,
            reduction=self.reduction,
            return_dict=self.return_dict if return_dict is None else return_dict,
            t_eps=self.t_eps,
            I_t=I_t,
            strict=self.strict if strict is None else strict,
            warn_zero_gamma=self.warn_zero_gamma,
            gamma_eps=self.gamma_eps,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return (
            f"coefficients={self.coefficients_name!r}, uses_noise={self.uses_noise}, "
            f"reduction={self.reduction!r}"
        )


# --------------------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual smoke test
    """Toy checks of the L_g estimator on an exactly-solvable Gaussian problem.

    With ``x0, x1 ~ N(0, I)``, ``d`` dimensions and the ``linear`` preset, one has
    ``I_t ~ N(0, S I)`` with ``S = alpha_t^2 + beta_t^2 + gamma_t^2 = 1`` and
    ``E[z | I_t = x] = (gamma_t / S) x = gamma_t x``. Hence:

    * the zero network gives ``L_g = E|z|^2 = d``;
    * the oracle ``g(x, t) = gamma_t x`` gives ``L_g = -E|gamma_t I_t|^2 = -gamma_t^2 d``.
    """
    torch.manual_seed(0)
    d = 6
    batch = 4096

    interp = resolve_interpolant(coefficients="linear")

    class Oracle(nn.Module):
        def forward(self, x, t, xi=None, y=None):
            _, _, gamma = interp.alpha_beta_gamma(t)
            return gamma.reshape(-1, *([1] * (x.dim() - 1))) * x

    class Zero(nn.Module):
        def forward(self, x, t, xi=None, y=None):
            return torch.zeros_like(x)

    x0 = torch.randn(batch, d)
    x1 = torch.randn(batch, d)
    t = torch.full((batch,), 0.3)

    loss_zero = score_loss(Zero(), x0, x1, t=t)
    loss_oracle = score_loss(Oracle(), x0, x1, t=t)
    gamma = float(interp.alpha_beta_gamma(t)[2][0])
    expected_oracle = -(gamma ** 2) * d

    assert abs(loss_zero.item() - d) < 0.25, (loss_zero.item(), d)
    assert abs(loss_oracle.item() - expected_oracle) < 0.35, (
        loss_oracle.item(),
        expected_oracle,
    )
    assert loss_oracle.item() < loss_zero.item()

    # reduction="none" gives per-item values, and their mean reproduces the scalar loss.
    per_item = score_loss(Zero(), x0, x1, t=t, reduction="none")
    assert per_item.shape == (batch,)
    assert abs(per_item.mean().item() - loss_zero.item()) < 1e-5

    # dict output exposes the pieces required by the SDE samplers.
    info = score_loss(Oracle(), x0, x1, t=t, return_dict=True)
    assert set(["loss", "g_hat", "I_t", "z", "t", "gamma_t", "score_hat"]) <= set(info)
    # score = -gamma^{-1} g -> consistency of the conversion helper
    s1 = score_from_g(info["g_hat"], t, interpolant=interp)
    s2 = info["score_hat"]
    assert torch.allclose(s1, s2, atol=1e-6)

    # conditioned call: xi is forwarded to the network
    seen = {}

    class Ctx(nn.Module):
        def forward(self, x, t, xi=None, y=None):
            seen["xi"] = xi
            seen["y"] = y
            return torch.zeros_like(x)

    ctx = torch.randn(batch, 4)
    labels = torch.zeros(batch, dtype=torch.long)
    score_loss(Ctx(), x0, x1, t=t, xi=ctx, y=labels)
    assert seen["xi"] is ctx and seen["y"] is labels

    # gamma_t = 0 presets are flagged (and raise in strict mode).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score_loss(Zero(), x0, x1, t=t, coefficients="inpainting")
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    try:
        score_loss(Zero(), x0, x1, t=t, coefficients="inpainting", strict=True)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("strict=True should have raised for gamma_t = 0")

    # module wrapper path
    module = ScoreLoss(coefficients="linear")
    assert isinstance(module(Zero(), x0, x1, t=t), torch.Tensor)
    assert module.coefficients_name == "linear" and module.uses_noise

    print("score_loss self-test: OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
