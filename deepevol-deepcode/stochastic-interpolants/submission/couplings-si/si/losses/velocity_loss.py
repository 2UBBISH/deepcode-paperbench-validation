"""Empirical velocity-regression loss for stochastic interpolants with couplings.

This module implements the empirical approximation of the velocity objective
``L_b`` from Section 3.4 of *Stochastic Interpolants with Data-Dependent
Couplings*:

.. math::

    \\hat L_b(\\hat b) = \\frac{1}{n_b}\\sum_{i=1}^{n_b}
        \\left[\\left|\\hat b_{t_i}(I_{t_i}, \\xi)\\right|^2
              - 2 \\dot I_{t_i}\\cdot \\hat b_{t_i}(I_{t_i}, \\xi)\\right]

with per-item times :math:`t_i \\sim U([0,1])` and, when the interpolant
preset uses a structurally non-zero :math:`\\gamma_t`, per-item
:math:`z_i \\sim N(0, \\mathrm{Id})`.  ``Algorithm 1`` of the paper is exactly
the combination of the coupling (which produces ``x_0 = m(x_1) + sigma zeta``)
with this loss and a gradient step.

The loss is *simulation free*: no ODE/SDE integration is needed to evaluate the
objective, and its unique minimizer is the conditional expectation
:math:`b_t(x,\\xi) = E[\\dot I_t \\mid I_t = x, \\xi]` (Theorem 3.1 / A.1).

Notes
-----
* Tensors follow the repository-wide convention of images in ``[-1, 1]``.
* ``t`` is sampled per item in ``[0, 1]`` and is passed to the model as a 1-D
  tensor of shape ``(B,)``; interpolant evaluation broadcasts it internally.
* For the in-painting task the velocity is structurally zero on the *observed*
  pixels (``xi * I_t = xi * x_1`` for all ``t``), so the network output is
  masked there via ``mask_fn`` (e.g. ``InpaintingCoupling.mask_velocity``).
  This is a no-op for the loss value (the target derivative vanishes on those
  pixels as well) but removes wasted capacity and gradient noise.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..interpolants.interpolant import Interpolant, broadcast_t
from ..interpolants.coefficients import Coefficients, get_coefficients

__all__ = [
    "sample_uniform_t",
    "sample_interpolant_noise",
    "call_model",
    "velocity_loss",
    "compute_velocity_loss",
    "VelocityLoss",
    "transport_cost_estimate",
    "estimate_transport_cost",
]


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------
def sample_uniform_t(
    batch_size: int,
    device: Union[str, torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    eps: float = 0.0,
) -> torch.Tensor:
    """Draw ``t ~ U([0, 1])`` per item.

    Parameters
    ----------
    batch_size : int
        Number of items (``n_b`` in the paper).
    device, dtype : torch device / dtype
        Placement of the returned tensor (dtype is always floating point).
    generator : torch.Generator, optional
        Explicit RNG for reproducibility.
    eps : float
        Optional interior margin so that ``t`` stays in ``(eps, 1 - eps)``;
        useful for coefficient presets whose derivatives are singular at the
        endpoints (e.g. ``gamma_t = sqrt(2 t (1 - t))``).  The paper samples
        from ``U([0, 1])``, so the default is ``eps = 0``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch_size,)``.
    """
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if not torch.is_floating_point(torch.empty(0, dtype=dtype)):
        dtype = torch.float32
    shape = (int(batch_size),)
    if eps <= 0.0:
        t = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    else:
        lo = float(eps)
        hi = 1.0 - float(eps)
        if hi <= lo:
            raise ValueError("eps must lie in [0, 0.5)")
        t = lo + (hi - lo) * torch.rand(
            shape, device=device, dtype=dtype, generator=generator
        )
    return t


def sample_interpolant_noise(
    x1: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw ``z ~ N(0, Id)`` with the same shape/device/dtype as ``x1``."""
    return torch.randn(
        x1.shape, device=x1.device, dtype=x1.dtype, generator=generator
    )


# ---------------------------------------------------------------------------
# model invocation (tolerant of optional conditioning arguments)
# ---------------------------------------------------------------------------
def _signature_params(fn: Callable[..., Any]) -> Optional[set]:
    """Return the set of accepted parameter names of ``fn`` (or ``None``)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return None
    params = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return None  # accepts everything
        params.add(name)
    return params


def call_model(
    model: Callable[..., torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    xi: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    **extra: Any,
) -> torch.Tensor:
    """Call a velocity (or score) network with optional conditioning.

    The network is expected to accept ``forward(x, t, xi=None, y=None)`` (the
    signature used by :mod:`si.models.unet`).  To stay robust to alternative
    spellings (``cond``, ``conditioning``, ``label``, ``labels``), the accepted
    parameter names are inspected and only the relevant keyword arguments are
    forwarded.  Unknown callables that expose ``**kwargs`` receive both ``xi``
    and ``y``.
    """
    fn = getattr(model, "forward", model)
    params = _signature_params(fn)

    kwargs: Dict[str, Any] = {}
    if params is None:  # **kwargs style: pass everything
        if xi is not None:
            kwargs["xi"] = xi
        if y is not None:
            kwargs["y"] = y
        kwargs.update(extra)
    else:
        if xi is not None:
            for name in ("xi", "cond", "conditioning", "context", "mask", "c"):
                if name in params:
                    kwargs[name] = xi
                    break
        if y is not None:
            for name in ("y", "label", "labels", "class_labels", "cls"):
                if name in params:
                    kwargs[name] = y
                    break
        for key, value in extra.items():
            if key in params:
                kwargs[key] = value
    return model(x, t, **kwargs)


# ---------------------------------------------------------------------------
# core loss
# ---------------------------------------------------------------------------
def _resolve_coefficients(
    coefficients: Union[str, Coefficients, Interpolant, None],
    interpolant: Optional[Interpolant] = None,
) -> Interpolant:
    """Return an :class:`Interpolant` from a loose specification."""
    if isinstance(coefficients, Interpolant):
        return coefficients
    if interpolant is not None:
        return interpolant
    if coefficients is None:
        return Interpolant(get_coefficients("linear"))
    return Interpolant(get_coefficients(coefficients))


def _apply_mask(
    b_hat: torch.Tensor,
    mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    mask_arg: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the optional structural velocity mask."""
    if mask_fn is None:
        return b_hat
    try:
        out = mask_fn(b_hat)
    except TypeError:  # mask_fn expects (velocity, xi)
        if mask_arg is None:
            raise
        out = mask_fn(b_hat, mask_arg)
    if isinstance(out, tuple):  # tolerate (velocity, something) returns
        out = out[0]
    return out


def velocity_loss(
    model: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    xi: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    interpolant: Optional[Interpolant] = None,
    coefficients: Union[str, Coefficients, None] = None,
    mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    reduction: str = "mean",
    return_dict: bool = False,
    model_fn: Optional[Callable[..., torch.Tensor]] = None,
    t_eps: float = 0.0,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Empirical velocity loss ``\\hat L_b`` (paper Eq. 22 / Algorithm 1).

    Parameters
    ----------
    model : callable
        Velocity network ``\\hat b_t(x, xi)``; called as ``model(I_t, t, xi=xi, y=y)``.
    x0, x1 : torch.Tensor
        Coupled base/target batch.  For the data-dependent coupling,
        ``x0 = m(x1) + sigma * zeta`` should already have been produced by a
        :class:`~si.couplings.base.Coupling`.
    t : torch.Tensor, optional
        Times of shape ``(B,)`` (or broadcastable).  Drawn from ``U([0,1])``
        per item when not supplied.
    xi : torch.Tensor, optional
        Conditioning variable (missingness mask for in-painting, upsampled
        low-resolution image for super-resolution, class labels, ...).
    z : torch.Tensor, optional
        Interpolant noise ``z ~ N(0, Id)``; only used when ``gamma_t != 0``.
        Drawn on the fly when required and not supplied.
    y : torch.Tensor, optional
        Class labels forwarded to the network.
    interpolant : Interpolant, optional
        Interpolant object providing ``I_t`` / ``I_dot``.
    coefficients : str or Coefficients, optional
        Coefficient preset name (e.g. ``"linear"``, ``"gamma0"``,
        ``"inpainting"``, ``"superres"``) used when ``interpolant`` is missing.
    mask_fn : callable, optional
        Structural mask applied to the network output, e.g.
        ``coupling.mask_velocity`` for in-painting.
    generator : torch.Generator, optional
        RNG used for the internal ``t`` and ``z`` draws.
    reduction : {"mean", "sum", "none"}
        Reduction over the batch; the paper uses the mean over ``n_b``.
    return_dict : bool
        If ``True`` also return diagnostics (``loss``, ``b_hat``, ``I_t``,
        ``I_dot``, ``t``, per-item losses).
    model_fn : callable, optional
        Override for how the network is invoked (signature
        ``model_fn(model, x, t, xi, y)``); mainly useful for tests.

    Returns
    -------
    torch.Tensor or dict
        The (reduced) empirical velocity loss, optionally with diagnostics.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"x0 and x1 must have the same shape, got {tuple(x0.shape)} and "
            f"{tuple(x1.shape)}"
        )
    if x0.dim() < 1:
        raise ValueError("x0/x1 must have at least one dimension (the batch)")

    batch = x0.shape[0]
    interp = _resolve_coefficients(coefficients, interpolant)

    if t is None:
        t = sample_uniform_t(
            batch, device=x0.device, dtype=x0.dtype, generator=generator, eps=t_eps
        )
    else:
        t = t.to(device=x0.device, dtype=x0.dtype)
        if t.dim() == 0:
            t = t.expand(batch)

    needs_z = interp.uses_noise()
    if needs_z and z is None:
        z = sample_interpolant_noise(x0, generator=generator)
    if z is not None:
        z = z.to(device=x0.device, dtype=x0.dtype)

    # I_t = alpha_t x0 + beta_t x1 + gamma_t z ; I_dot = alpha_dot x0 + beta_dot x1 + gamma_dot z
    I_t, I_dot = interp(x0, x1, t, z=z, return_derivative=True)

    if model_fn is not None:
        b_hat = model_fn(model, I_t, t, xi, y)
    else:
        b_hat = call_model(model, I_t, t, xi=xi, y=y)

    b_hat = _apply_mask(b_hat, mask_fn, xi)

    per_item = (b_hat.pow(2).flatten(1).sum(dim=1)
                - 2.0 * (I_dot * b_hat).flatten(1).sum(dim=1))

    if reduction == "mean":
        loss = per_item.mean()
    elif reduction == "sum":
        loss = per_item.sum()
    elif reduction == "none":
        loss = per_item
    else:  # pragma: no cover - guard clause
        raise ValueError(
            f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}"
        )

    if not return_dict:
        return loss

    return {
        "loss": loss,
        "per_item": per_item,
        "b_hat": b_hat,
        "I_t": I_t,
        "I_dot": I_dot,
        "t": t,
        "z": z,
        "batch_size": torch.as_tensor(batch),
    }


#: Alias kept for readability in call sites / scripts.
compute_velocity_loss = velocity_loss


class VelocityLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`velocity_loss`.

    The module only stores configuration: ``x0``/``x1``/``xi`` are produced by
    the coupling at every training step (Algorithm 1).

    Parameters
    ----------
    coefficients : str or Coefficients
        Coefficient preset (default ``"linear"``).
    interpolant : Interpolant, optional
        Pre-built interpolant; overrides ``coefficients`` when given.
    mask_fn : callable, optional
        Structural output mask (``coupling.mask_velocity`` for in-painting).
    reduction : {"mean", "sum", "none"}
        Loss reduction, ``"mean"`` matching Eq. 22.
    return_dict : bool
        Forward diagnostics together with the loss.
    t_eps : float
        Interior margin for the ``U([0,1])`` time draw (default 0.0).
    """

    def __init__(
        self,
        coefficients: Union[str, Coefficients] = "linear",
        interpolant: Optional[Interpolant] = None,
        mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reduction: str = "mean",
        return_dict: bool = False,
        t_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.interpolant = interpolant or Interpolant(get_coefficients(coefficients))
        self.mask_fn = mask_fn
        self.reduction = reduction
        self.return_dict = return_dict
        self.t_eps = float(t_eps)

    # ------------------------------------------------------------------
    @property
    def coefficients_name(self) -> str:
        """Name of the coefficient preset in use."""
        return self.interpolant.name

    @property
    def uses_noise(self) -> bool:
        """Whether the interpolant has a structurally non-zero ``gamma_t``."""
        return self.interpolant.uses_noise()

    # ------------------------------------------------------------------
    def forward(
        self,
        model: Callable[..., torch.Tensor],
        x0: torch.Tensor,
        x1: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate the empirical velocity loss for one minibatch."""
        return velocity_loss(
            model,
            x0,
            x1,
            t=t,
            xi=xi,
            z=z,
            y=y,
            interpolant=self.interpolant,
            mask_fn=mask_fn or self.mask_fn,
            generator=generator,
            reduction=self.reduction,
            return_dict=self.return_dict if return_dict is None else return_dict,
            t_eps=self.t_eps,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return (
            f"coefficients={self.interpolant.name}, reduction={self.reduction}, "
            f"uses_noise={self.uses_noise}, mask_fn="
            f"{getattr(self.mask_fn, '__qualname__', self.mask_fn)}"
        )


# ---------------------------------------------------------------------------
# transport-cost diagnostics (Proposition 3.1 / Eq. 21)
# ---------------------------------------------------------------------------
@torch.no_grad()
def transport_cost_estimate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    interpolant: Optional[Interpolant] = None,
    coefficients: Union[str, Coefficients, None] = None,
    t: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    samples: int = 1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Estimate ``E[|I_dot_t|^2]``, the integrand of the transport-cost bound.

    Used to validate Proposition 3.1 / Eq. (21) empirically: with a
    data-dependent coupling ``m(x1) = x1``, ``C = sigma^2 Id`` and
    ``alpha_t = 1 - t``, ``beta_t = t``, ``gamma_t = 0`` the coupled value is
    ``d sigma^2`` whereas the independent base gives
    ``2 E[|x_1|^2] + d sigma^2``.

    Parameters
    ----------
    x0, x1 : torch.Tensor
        Base/target batch (``x0`` already coupled when assessing the coupled
        construction; independently drawn when assessing the baseline).
    samples : int
        Number of Monte-Carlo time draws to average over ``t ~ U([0,1])``
        (the bound integrates over ``t``; the paper compares the integrands at
        fixed ``t`` and also reports the empirical integral).
    """
    interp = _resolve_coefficients(coefficients, interpolant)
    batch = x1.shape[0]
    total = None
    for _ in range(max(int(samples), 1)):
        tt = t
        if tt is None:
            tt = sample_uniform_t(batch, device=x1.device, dtype=x1.dtype)
        zz = z
        if interp.uses_noise() and zz is None:
            zz = sample_interpolant_noise(x1)
        _, I_dot = interp(x0, x1, tt, z=zz, return_derivative=True)
        value = I_dot.pow(2).flatten(1).sum(dim=1)
        if reduction == "mean":
            value = value.mean()
        elif reduction == "sum":
            value = value.sum()
        total = value if total is None else total + value
    return total / max(int(samples), 1)


#: Alias matching the plan's terminology.
estimate_transport_cost = transport_cost_estimate


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    d = 4
    batch = 512

    # --- toy coupled problem: x0 = x1 + sigma zeta (m(x1) = x1, C = sigma^2 I)
    sigma = 0.5
    x1 = torch.randn(batch, d)
    zeta = torch.randn(batch, d)
    x0 = x1 + sigma * zeta

    interp = Interpolant(get_coefficients("gamma0"))
    zero_model = lambda x, t, xi=None, y=None: torch.zeros_like(x)  # noqa: E731

    loss_zero = velocity_loss(zero_model, x0, x1, interpolant=interp)
    expected = (sigma ** 2) * d * torch.ones(())  # E|I_dot|^2 = E|x1 - x0|^2
    assert torch.allclose(loss_zero, expected, atol=0.1), (loss_zero, expected)

    # --- in-painting style: target derivative is exactly recoverable
    coarse = get_coefficients("inpainting")
    interp_ip = Interpolant(coarse)
    xi_mask = (torch.rand(batch, 1, 2) > 0.3).float()
    x1_img = torch.randn(batch, 3, 2)
    zeta_img = torch.randn(batch, 3, 2)
    x0_img = xi_mask * x1_img + (1.0 - xi_mask) * zeta_img
    t = torch.rand(batch)
    I_t, I_dot = interp_ip(x0_img, x1_img, t, return_derivative=True)
    # observed pixels: I_t == x1 exactly, I_dot == 0 exactly
    observed = xi_mask.expand_as(I_t).bool()
    assert torch.allclose(I_t[observed], x1_img.expand_as(I_t)[observed], atol=1e-5)
    assert I_dot[observed].abs().max().item() < 1e-6

    # perfect model => loss = -E|I_dot|^2 <= 0
    perfect = lambda x, tt, xi=None, y=None: I_dot  # noqa: E731
    loss_perfect = velocity_loss(perfect, x0_img, x1_img, t=t, interpolant=interp_ip)
    assert torch.allclose(loss_perfect, -I_dot.pow(2).flatten(1).sum(1).mean())
    assert loss_perfect.item() <= 0.0

    # --- masked output keeps the in-painting loss unchanged on observed pixels
    def mask_velocity(v, xi):
        return v * (1.0 - xi)

    loss_masked = velocity_loss(
        perfect, x0_img, x1_img, t=t, xi=xi_mask, interpolant=interp_ip,
        mask_fn=lambda v: mask_velocity(v, xi_mask),
    )
    assert torch.allclose(loss_masked, loss_perfect, atol=1e-5)

    # --- module wrapper equivalence
    module = VelocityLoss(coefficients="gamma0")
    assert module.coefficients_name == "gamma0"
    assert not module.uses_noise
    out = module(zero_model, x0, x1)
    assert torch.allclose(out, loss_zero, atol=1e-5)

    # --- transport cost: coupled vs independent base
    coupled = transport_cost_estimate(x0, x1, interpolant=interp)
    indep_x0 = torch.randn(batch, d)
    indep = transport_cost_estimate(indep_x0, x1, interpolant=interp)
    assert coupled < indep, (coupled.item(), indep.item())

    print("velocity_loss self-test passed",
          {"coupled": round(coupled.item(), 3), "independent": round(indep.item(), 3)})


if __name__ == "__main__":  # pragma: no cover
    _self_test()
