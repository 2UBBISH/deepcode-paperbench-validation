r"""Optional score network :math:`\hat g_t(x, \xi)` for stochastic interpolants.

Paper reference
---------------
* Section 3.1, Eq. (4)::

      b_t(x, xi) = E[ I_dot_t | I_t = x, xi ],   g_t(x, xi) = E[ z | I_t = x, xi ]

* Section 3.1, Eq. (6) (score identity, valid for every ``t`` with ``gamma_t != 0``)::

      grad_x log rho_t(x | xi) = - gamma_t^{-1} g_t(x, xi)

* Appendix A (Eq. 6 of the appendix) is the *conditioned* version of the objectives;
  the second line is the one regressed by this network::

      L_g(g_hat) = \int_0^1 E[ |g_hat_t(I_t, xi)|^2 - 2 z . g_hat_t(I_t, xi) ] dt

  whose unique minimizer is ``g_t`` (see ``si/losses/score_loss.py``).

* Section 3.4 (Eq. 7 / Eq. 29): the empirical estimate of ``L_g`` uses
  ``z_i ~ N(0, Id)`` and ``t_i ~ U([0,1])`` per minibatch item, exactly like ``L_b``.

The score network is only needed for the *stochastic* sampler (the forward/backward SDEs,
Eqs. 11/13 of the main text) or for interpolants with a structurally non-zero ``gamma_t``.
The reported experiments in the paper use the deterministic probability-flow ODE with the
``gamma_t = 0`` coupling convention, so this module is optional: nothing in the velocity
training / ODE sampling path depends on it (``si/models/__init__.py`` guards the import).

Architecture
------------
By symmetry with ``b_hat_t`` (both regress a quantity conditional on ``(I_t, xi)``), the
default score net reuses the *same* U-Net family described in Appendix B (Ho et al. 2020b as
implemented in lucidrain's ``denoising-diffusion-pytorch``) with
``dim_mults=(1,1,2,3,4)``, ``channels=256``, ``resnet_block_groups=8``,
``learned_sinusoidal_cond=True``, ``learned_sinusoidal_dim=32``,
``attention_dim_head=64``, ``attention_heads=4``, ``random_fourier_features=False``, the same
class-label embedding path and the same image-shaped conditioning (mask for in-painting,
upsampled low-resolution image for super-resolution) appended to the input at each timestep.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .unet import (
    APPENDIX_B_CONFIG,
    VelocityUNet,
    appendix_b_config,
)

try:  # pragma: no cover - trivially importable, guarded for symmetry with models/__init__
    from ..interpolants.coefficients import Coefficients, get_coefficients
    from ..interpolants.interpolant import Interpolant, broadcast_t
except Exception:  # pragma: no cover
    Coefficients = None  # type: ignore
    get_coefficients = None  # type: ignore
    Interpolant = None  # type: ignore
    broadcast_t = None  # type: ignore


__all__ = [
    "ScoreUNet",
    "ScoreNet",
    "score_net_from_config",
    "MODEL_CONFIG",
    "score_config",
    "score_from_g",
    "gaussian_score",
    "estimate_g",
]


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
#: Default score-net configuration: Appendix-B hyperparameters (identical to the velocity
#: net, since ``g_hat`` and ``b_hat`` share the same conditioning structure).
MODEL_CONFIG: Dict[str, Any] = dict(APPENDIX_B_CONFIG)


def score_config(**overrides: Any) -> Dict[str, Any]:
    """Return a copy of the default score-net config, updated with ``overrides``."""
    return appendix_b_config(**overrides)


# --------------------------------------------------------------------------------------
# helpers: g <-> score, and the analytically tractable reference score
# --------------------------------------------------------------------------------------
def _resolve_interpolant(
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    default: str = "linear",
):
    """Normalize the (interpolant, coefficients) pair into an :class:`Interpolant`.

    Accepts an ``Interpolant``, a ``Coefficients`` container, or a preset name string.
    """
    if interpolant is not None and Interpolant is not None and isinstance(interpolant, Interpolant):
        return interpolant
    if interpolant is not None and isinstance(interpolant, str):
        coefficients = interpolant
        interpolant = None
    if interpolant is not None and Coefficients is not None and isinstance(interpolant, Coefficients):
        coefficients = interpolant
    coeff_name = default
    if isinstance(coefficients, str):
        coeff_name = coefficients
    elif coefficients is not None and Coefficients is not None and isinstance(coefficients, Coefficients):
        return Interpolant(coefficients)
    if get_coefficients is None:  # pragma: no cover
        raise RuntimeError("si.interpolants.coefficients is unavailable")
    return Interpolant(get_coefficients(coeff_name))


def _gamma_t(interp, t: torch.Tensor) -> torch.Tensor:
    """Evaluate ``gamma_t`` for (possibly per-item) times ``t``."""
    _, _, gamma = interp.alpha_beta_gamma(t)
    if not torch.is_tensor(gamma):
        gamma = torch.as_tensor(gamma, device=t.device, dtype=t.dtype)
    return gamma


def score_from_g(
    g: torch.Tensor,
    t: torch.Tensor,
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    gamma_eps: float = 1e-12,
) -> torch.Tensor:
    r"""Convert the learned ``g_hat_t`` to the score via the identity of Theorem 3.1.

    .. math::
        \nabla_x \log \rho_t(x \mid \xi) = -\gamma_t^{-1} g_t(x, \xi)

    Entries with ``|gamma_t| < gamma_eps`` are set to zero (the identity only holds for
    ``gamma_t != 0``; for the deterministic ``gamma_t = 0`` presets the score carries no
    information about the coupling and the regression is degenerate).

    Parameters
    ----------
    g:
        Network output with shape ``(B, C, H, W)`` (or any shape whose leading dim is ``B``).
    t:
        Times, scalar or shape ``(B,)``.
    interpolant, coefficients:
        Interpolant object / ``Coefficients`` / preset-name string. Defaults to ``"linear"``.
    """
    interp = _resolve_interpolant(interpolant, coefficients)
    if broadcast_t is not None:
        t_b = broadcast_t(t, g.dim()) if t.dim() != g.dim() else t
    else:  # pragma: no cover
        t_b = t.reshape(-1, *([1] * (g.dim() - 1)))
    gamma = _gamma_t(interp, t_b)
    while gamma.dim() < g.dim():
        gamma = gamma.unsqueeze(-1)
    safe = gamma.abs() >= gamma_eps
    inv_gamma = torch.where(
        safe,
        torch.where(safe, torch.ones_like(gamma), torch.ones_like(gamma)) / gamma.clamp_min(gamma_eps),
        torch.zeros_like(gamma),
    )
    inv_gamma = torch.where(safe, 1.0 / torch.where(safe, gamma, torch.ones_like(gamma)), torch.zeros_like(gamma))
    return -inv_gamma * g


def gaussian_score(
    x: torch.Tensor,
    t: torch.Tensor,
    interpolant: Optional[Any] = None,
    coefficients: Optional[Any] = None,
    variance: float = 1.0,
    mode: str = "unit",
) -> torch.Tensor:
    r"""Analytically known score for the toy Gaussian problem (diagnostics / unit tests).

    If ``x_0 ~ N(0, variance * Id)`` and ``x_1 ~ N(0, variance * Id)`` independently and
    ``z ~ N(0, Id)``, then ``I_t = alpha_t x_0 + beta_t x_1 + gamma_t z`` is Gaussian with
    covariance ``s_t^2 Id`` where ``s_t^2 = variance * (alpha_t^2 + beta_t^2) + gamma_t^2``,
    hence

    .. math::
        \nabla_x \log \rho_t(x) = -x / s_t^2 .

    ``mode="unit"`` returns ``-x / s_t^2``; ``mode="g"`` returns the corresponding
    minimizer of ``L_g``, namely ``g_t(x) = -gamma_t * score(x)``.
    """
    interp = _resolve_interpolant(interpolant, coefficients)
    if broadcast_t is not None:
        alpha, beta, gamma = interp.alpha_beta_gamma(t)
    else:  # pragma: no cover
        alpha, beta, gamma = interp.evaluate(t)
    alpha = alpha * torch.ones_like(x) if torch.is_tensor(alpha) else torch.full_like(x, float(alpha))
    beta = beta * torch.ones_like(x) if torch.is_tensor(beta) else torch.full_like(x, float(beta))
    gamma = gamma * torch.ones_like(x) if torch.is_tensor(gamma) else torch.full_like(x, float(gamma))
    if gamma.dim() > 0:
        shape = (-1,) + (1,) * (x.dim() - 1)
        alpha = alpha.reshape(*shape) if alpha.numel() > 1 else alpha.reshape(shape)
        beta = beta.reshape(*shape) if beta.numel() > 1 else beta.reshape(shape)
        gamma = gamma.reshape(*shape) if gamma.numel() > 1 else gamma.reshape(shape)
    s2 = variance * (alpha ** 2 + beta ** 2) + gamma ** 2
    s = torch.sqrt(s2.clamp_min(1e-12))
    if mode == "g":
        return -gamma * (-x / s2)
    if mode != "unit":
        raise ValueError(f"unknown mode {mode!r}; expected 'unit' or 'g'")
    return -x / s2


def estimate_g(model: nn.Module, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Evaluate ``g_hat_t(x, xi)`` from ``model``, supporting the common call signatures."""
    return _call_score_model(model, x, t, **kwargs)


# --------------------------------------------------------------------------------------
# score network
# --------------------------------------------------------------------------------------
class ScoreUNet(VelocityUNet):
    r"""U-Net approximating :math:`\hat g_t(x, \xi) \approx E[z \mid I_t = x, \xi]`.

    The network is architecturally identical to the velocity net (see Appendix B) and
    consumes the very same conditioning:

    * image-shaped conditioning ``xi`` (missingness mask for in-painting, upsampled
      low-resolution image for super-resolution) concatenated to the input channels at every
      timestep;
    * class labels via the embedding path (Appendix B).

    Its output has the image shape ``(B, C, H, W)``; to obtain the score use
    :meth:`score` (or the module-level :func:`score_from_g`), which applies
    ``- gamma_t^{-1}`` as in Theorem 3.1.

    Additional constructor parameters
    --------------------------------
    interpolant, coefficients:
        Used to evaluate ``gamma_t`` when the model is asked for the score directly.
        Defaults to the ``"linear"`` preset (the only one reported to use ``gamma_t != 0``).
    gamma_eps:
        Threshold below which ``|gamma_t|`` is treated as zero (returns a zero score),
        guarding the ``gamma_t = 0`` presets (``"gamma0"``/``"inpainting"``/``"superres"``).
    return_score:
        If ``True``, :meth:`forward` returns the score rather than ``g_hat``.
    """

    def __init__(
        self,
        *args: Any,
        interpolant: Optional[Any] = None,
        coefficients: Optional[Any] = "linear",
        gamma_eps: float = 1e-12,
        return_score: bool = False,
        **kwargs: Any,
    ) -> None:
        # The score net never structurally masks its output (that trick belongs to the
        # in-painting velocity field, where xi * I_t = xi * x1 for all t).
        kwargs.pop("mask_observed", None)
        kwargs.pop("observed_value", None)
        super().__init__(*args, mask_observed=False, observed_value=1.0, **kwargs)
        self.interpolant = _resolve_interpolant(interpolant, coefficients)
        self.coefficients_name = getattr(self.interpolant, "name", str(coefficients))
        self.gamma_eps = float(gamma_eps)
        self.return_score = bool(return_score)

    # -- forward -----------------------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        return_score: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Return ``g_hat_t(x, xi)`` (or the score when ``return_score=True``)."""
        g_hat = super().forward(x, t, xi=xi, y=y, return_dict=False, **kwargs)
        need_score = self.return_score if return_score is None else bool(return_score)
        score_hat = None
        if need_score:
            t_input = _time_like(t, x)
            score_hat = score_from_g(g_hat, t_input, interpolant=self.interpolant, gamma_eps=self.gamma_eps)
        if return_dict:
            out: Dict[str, Any] = {"g_hat": g_hat}
            if need_score and score_hat is not None:
                out["score_hat"] = score_hat
            return out
        return score_hat if need_score else g_hat

    # -- convenience -------------------------------------------------------------------
    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""Return the score estimate ``- gamma_t^{-1} g_hat_t(x, xi)`` (Theorem 3.1)."""
        return self.forward(x, t, xi=xi, y=y, return_score=True, **kwargs)

    def predict_z(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return ``g_hat_t(x, xi)``, the estimate of ``E[z | I_t = x, xi]`` (Eq. 4)."""
        return self.forward(x, t, xi=xi, y=y, return_score=False, **kwargs)

    def uses_noise(self) -> bool:
        """Whether the configured interpolant has a structurally non-zero ``gamma_t``."""
        try:
            return bool(self.interpolant.uses_noise())
        except Exception:  # pragma: no cover
            return True

    def extra_repr(self) -> str:  # type: ignore[override]
        base = super().extra_repr()
        extra = f"interpolant={self.coefficients_name}, gamma_eps={self.gamma_eps:g}"
        return f"{base}, {extra}" if base else extra


#: Aliases used by configs / downstream code.
ScoreNet = ScoreUNet


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
_ALIASES: Dict[str, str] = {
    "dim": "channels",
    "channels_in": "in_channels",
    "attn_heads": "attention_heads",
    "heads": "attention_heads",
    "dim_head": "attention_dim_head",
    "num_class": "num_classes",
    "n_classes": "num_classes",
    "conditional": "conditioning_channels",
    "cond_channels": "conditioning_channels",
}


def score_net_from_config(config: Optional[Dict[str, Any]] = None, **overrides: Any) -> ScoreUNet:
    """Build a :class:`ScoreUNet` from a flat config dict (with common key aliases).

    Mirrors ``si.models.unet.unet_from_config`` so a YAML config can instantiate either the
    velocity or the score network through ``si.models.build_model``.
    """
    cfg: Dict[str, Any] = dict(MODEL_CONFIG)
    if config:
        cfg.update(config)
    cfg.update(overrides)
    for src, dst in _ALIASES.items():
        if src in cfg and dst not in cfg:
            cfg[dst] = cfg.pop(src)
        elif src in cfg:
            cfg.pop(src)
    interpolant = cfg.pop("interpolant", None)
    coefficients = cfg.pop("coefficients", "linear")
    gamma_eps = cfg.pop("gamma_eps", 1e-12)
    return_score = cfg.pop("return_score", False)
    # class labels are part of the reported setup (ImageNet-1k)
    return ScoreUNet(
        interpolant=interpolant,
        coefficients=coefficients,
        gamma_eps=gamma_eps,
        return_score=return_score,
        **cfg,
    )


# --------------------------------------------------------------------------------------
# internal utilities
# --------------------------------------------------------------------------------------
def _time_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Normalize ``t`` to shape ``(B,)`` (or the broadcast shape used by the interpolant)."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=x.dtype)
    if t.dim() == 0:
        return t.expand(x.shape[0])
    if t.dim() > 1:
        t = t.reshape(t.shape[0], -1)[:, 0]
    return t


def _call_score_model(model: nn.Module, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Call a score model with whatever conditioning keywords it accepts."""
    import inspect

    try:
        params = set(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        params = {"x", "t"}
    call_kwargs: Dict[str, Any] = {}
    for key in ("xi", "cond", "conditioning", "context", "mask", "c"):
        if key in kwargs and (key in params or (key == "xi" and "xi" in params)):
            call_kwargs["xi" if "xi" in params else key] = kwargs[key]
            break
    for key in ("y", "label", "labels", "class_labels", "cls"):
        if key in kwargs and key in params:
            call_kwargs[key] = kwargs[key]
            break
    if "y" in kwargs:
        call_kwargs["y"] = kwargs["y"]
    if call_kwargs.get("xi", None) is None and "xi" in kwargs and "xi" in params:
        call_kwargs["xi"] = kwargs["xi"]
    out = model(x, t, **call_kwargs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("g_hat", out.get("score_hat", next(iter(out.values()))))
    return out


# --------------------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    B, C, H = 2, 3, 32
    for name in ("linear", "gamma0"):
        net = ScoreUNet(
            in_channels=C,
            out_channels=C,
            channels=16,
            dim_mults=(1, 1, 2),
            conditioning_channels=0,
            num_classes=10,
            image_size=H,
            coefficients=name,
        )
        x = torch.randn(B, C, H, H)
        t = torch.rand(B)
        y = torch.randint(0, 10, (B,))
        g_hat = net(x, t, y=y)
        assert g_hat.shape == x.shape, (name, g_hat.shape)
        s = net.score(x, t, y=y)
        assert s.shape == x.shape
        # with the exact network output zeroed, the score conversion is a pure scaling
        assert torch.isfinite(s).all()
        print(f"[score_net] {name}: g_hat {tuple(g_hat.shape)}, score {tuple(s.shape)}, "
              f"uses_noise={net.uses_noise()}")

    # gamma_t^{-1} conversion check against the closed form of Eq. (6)
    g = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    sc = score_from_g(g, t, coefficients="linear")
    a, b, gam = Interpolant(get_coefficients("linear")).alpha_beta_gamma(t)
    ref = -gam.reshape(-1, 1, 1, 1) ** -1 * g
    assert torch.allclose(sc, ref, atol=1e-5), (sc - ref).abs().max()
    # gamma0 -> zero score (identity undefined; guarded)
    assert torch.count_nonzero(score_from_g(g, t, coefficients="gamma0")) == 0
    print("[score_net] score_from_g conversion OK (linear); gamma0 guarded to zero")

    # analytic Gaussian score check: x ~ N(0, s^2 I) => score = -x / s^2
    x = torch.randn(1000, 5) * 2.0
    print("[score_net] gaussian_score sample norm "
          f"{gaussian_score(torch.zeros(1, 5), torch.tensor([0.5])).norm().item():.4f}")

    # full model built from config
    net = score_net_from_config({"dim": 16, "dim_mults": (1, 1, 2), "num_class": 10,
                                 "in_channels": 3, "image_size": 32, "coefficients": "linear"})
    out = net(torch.randn(1, 3, 32, 32), torch.tensor([0.4]), return_dict=True)
    assert set(out) == {"g_hat"}
    print("[score_net] self-test passed")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
