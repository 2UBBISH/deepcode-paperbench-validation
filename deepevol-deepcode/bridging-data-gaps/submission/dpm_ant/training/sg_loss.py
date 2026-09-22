"""Similarity-guided DPMs training loss (DPMs-ANT).

Implements Equation (5) of "Adapting Pretrained Diffusion Models for Few-Shot
Image Generation":

.. math::

    \\min_\\theta \\mathbb{E}_{t, x_0, \\epsilon}\\Big[\\big\\| \\epsilon_t
        - \\epsilon_\\theta(x_t, t)
        - \\hat{\\sigma}_t^2 \\gamma \\nabla_{x_t} \\log p_\\phi(y=\\mathcal{T}\\mid x_t)
      \\big\\|^2\\Big],

with (Appendix A.2)

.. math::

    \\hat{\\sigma}_t = (1 - \\bar\\alpha_{t-1}) \\sqrt{\\frac{\\alpha_t}{1-\\bar\\alpha_t}},
    \\qquad
    C_2 = \\frac{\\beta_t^2}{2 \\sigma_t^2 \\alpha_t (1-\\bar\\alpha_t)} .

The classifier term :math:`\\nabla_{x_t} \\log p_\\phi(y=\\mathcal{T}\\mid x_t)` is
an indirect (generalizing) target indicator; it is computed by the *frozen*
binary classifier ``p_phi`` and detached from the graph, so the gradient flows
only through :math:`\\epsilon_\\theta(x_t, t)` (i.e. into the adaptors ``psi``).

The module is backbone agnostic: a DDPM ``FrozenDDPMUNet``, an LDM
``FrozenLDMUNet`` or any callable ``model(x_t, t) -> eps`` works.

Notes
-----
* ``use_adv_noise``/adversarial-noise selection (Eq. 7) lives in
  :mod:`dpm_ant.training.adv_noise`; this module only implements Eq. (5).
* Setting ``gamma = 0`` (or passing ``classifier=None``) reduces the objective
  to the vanilla DDPM loss of Ho et al. (2020), which is exactly the
  "DPMs-ANT w/o similarity guidance" ablation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion.schedule import NoiseSchedule, build_schedule

try:  # pragma: no cover - optional import, keeps the module importable alone
    from ..diffusion.gaussian_diffusion import extract as _gd_extract
except Exception:  # noqa: BLE001
    _gd_extract = None


LOGGER = logging.getLogger(__name__)

__all__ = [
    "SimilarityGuidedLoss",
    "similarity_guided_loss",
    "build_sg_loss",
    "classifier_guidance_grad",
    "corrected_noise_target",
    "sigma_hat_from_schedule",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _extract(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Index ``arr`` at per-sample timesteps ``t`` and expand to ``ndim`` dims."""
    if _gd_extract is not None:
        try:
            return _gd_extract(arr, t, ndim)
        except Exception:  # noqa: BLE001 - fall back to the local version
            pass
    out = arr.to(device=t.device)[t.long()]
    while out.dim() < ndim:
        out = out.unsqueeze(-1)
    return out


def _resolve_callable(module: Any, names: Tuple[str, ...]) -> Optional[Callable]:
    """Return the first attribute of ``module`` found in ``names``."""
    if module is None:
        return None
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _unwrap_prediction(out: Any) -> torch.Tensor:
    """Handle models returning tuples/dicts (e.g. ``learn_sigma=True``)."""
    if isinstance(out, dict):
        for key in ("eps", "epsilon", "noise", "pred", "prediction", "output"):
            if key in out:
                return out[key]
        # first tensor value
        for value in out.values():
            if torch.is_tensor(value):
                return value
        raise TypeError("model output dict contains no tensor")
    if isinstance(out, (tuple, list)):
        for value in out:
            if torch.is_tensor(value):
                return value
        raise TypeError("model output sequence contains no tensor")
    if not torch.is_tensor(out):
        raise TypeError(f"unsupported model output type: {type(out)}")
    return out


def sigma_hat_from_schedule(schedule: NoiseSchedule, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
    r""":math:`\hat\sigma_t = (1-\bar\alpha_{t-1})\sqrt{\alpha_t/(1-\bar\alpha_t)}`."""
    ab = schedule.ab(t, ndim)
    ab_prev = schedule.ab_prev(t, ndim)
    alpha = schedule.alpha(t, ndim)
    denom = (1.0 - ab).clamp(min=1e-20)
    return (1.0 - ab_prev) * torch.sqrt(alpha.clamp(min=1e-20) / denom)


def classifier_guidance_grad(
    classifier: Any,
    x_t: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    target_index: Optional[int] = None,
    grad_scale: float = 1.0,
    detach: bool = True,
    create_graph: bool = False,
) -> torch.Tensor:
    r"""Return :math:`\nabla_{x_t} \log p_\phi(y=\mathcal{T}\mid x_t)`.

    Works with :class:`dpm_ant.models.classifier.PretrainedClassifier`
    (``grad_log_target``), with any module exposing ``log_prob_target`` or with
    a plain callable ``logits(x_t, t)``.  The gradient is detached by default.
    """
    if classifier is None:
        raise ValueError("classifier is None: no similarity guidance available")

    fn = _resolve_callable(classifier, ("grad_log_target",))
    if fn is not None:
        kwargs: Dict[str, Any] = {"detach": detach, "create_graph": create_graph}
        try:
            grad = fn(x_t, t, target_index=target_index, **kwargs)
        except TypeError:
            # some implementations do not expose target_index / create_graph
            try:
                grad = fn(x_t, t, **kwargs)
            except TypeError:
                grad = fn(x_t, t)
        return grad * grad_scale

    # fall back: differentiate log p_phi(y=T | x_t) w.r.t. x_t ourselves
    needs_input_grad = not x_t.requires_grad
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        log_prob_fn = _resolve_callable(classifier, ("log_prob_target",))
        if log_prob_fn is not None:
            try:
                log_prob = log_prob_fn(x_in, t, target_index) if target_index is not None else log_prob_fn(x_in, t)
            except TypeError:
                log_prob = log_prob_fn(x_in, t)
        else:
            logits_fn = _resolve_callable(classifier, ("forward", "__call__"))
            logits = _unwrap_prediction(logits_fn(x_in, t))
            logits = logits.float()
            idx = 1 if target_index is None else int(target_index)
            log_prob = F.log_softmax(logits, dim=-1)[..., idx]
        grad = torch.autograd.grad(
            outputs=log_prob.sum(),
            inputs=x_in,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
    if detach:
        grad = grad.detach()
    if needs_input_grad:
        pass
    return grad * grad_scale


def corrected_noise_target(
    eps: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    classifier: Any,
    gamma: float = 5.0,
    target_index: Optional[int] = None,
    detach_classifier_grad: bool = True,
    grad_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute ``eps_target = eps - sigma_hat_t^2 * gamma * grad log p_phi``.

    Returns ``(eps_target, sigma_hat_t)`` where both keep the batch shape of
    ``eps`` (broadcastable against ``x_t``/``eps_theta``).
    """
    ndim = x_t.dim()
    sigma_hat = sigma_hat_from_schedule(schedule, t, ndim=ndim)

    if classifier is None or float(gamma) == 0.0:
        return eps.detach() if not eps.requires_grad else eps, sigma_hat

    grad = classifier_guidance_grad(
        classifier,
        x_t,
        t,
        target_index=target_index,
        grad_scale=grad_scale,
        detach=detach_classifier_grad,
    )
    grad = grad.to(device=eps.device, dtype=eps.dtype)
    eps_target = eps - (sigma_hat ** 2) * float(gamma) * grad
    return eps_target, sigma_hat


# --------------------------------------------------------------------------- #
# main module
# --------------------------------------------------------------------------- #
class SimilarityGuidedLoss(nn.Module):
    """Equation (5): similarity-guided DPMs training loss.

    Parameters
    ----------
    classifier : nn.Module, optional
        Frozen binary classifier ``p_phi`` exposing ``grad_log_target``.
    gamma : float
        Similarity-guidance strength (paper default ``5``).
    schedule / diffusion : optional
        A :class:`NoiseSchedule` or :class:`GaussianDiffusion`.  If given, its
        schedule is used; otherwise one is built from ``num_timesteps`` /
        ``schedule_name``.
    detach_classifier_grad : bool
        Detach :math:`\nabla_{x_t}\log p_\phi` (the paper's setting; the
        classifier is frozen and only ``p_phi`` is used as an indicator).
    include_c2 : bool
        Multiply by the constant :math:`C_2` of Appendix A.2.  Off by default:
        Ho et al. (2020) drop it in the simplified objective.
    use_c2_constant : bool
        Alias of ``include_c2`` kept for config compatibility.
    """

    def __init__(
        self,
        classifier: Optional[nn.Module] = None,
        gamma: float = 5.0,
        schedule: Optional[NoiseSchedule] = None,
        diffusion: Optional[Any] = None,
        num_timesteps: int = 1000,
        schedule_name: str = "linear",
        eta: float = 0.0,
        detach_classifier_grad: bool = True,
        target_index: Optional[int] = None,
        classifier_grad_scale: float = 1.0,
        reduction: str = "mean",
        include_c2: bool = False,
        use_c2_constant: Optional[bool] = None,
        min_t: int = 1,
        device: Optional[Any] = None,
        dtype: torch.dtype = torch.float32,
        **schedule_kwargs: Any,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.gamma = float(gamma)
        self.detach_classifier_grad = bool(detach_classifier_grad)
        self.target_index = target_index
        self.classifier_grad_scale = float(classifier_grad_scale)
        self.reduction = reduction
        self.min_t = int(min_t)

        if use_c2_constant is not None:
            include_c2 = bool(use_c2_constant)
        self.include_c2 = bool(include_c2)

        if schedule is None and diffusion is not None:
            schedule = getattr(diffusion, "schedule", None)
        if schedule is None:
            schedule = build_schedule(
                num_timesteps=num_timesteps,
                schedule=schedule_name,
                eta=eta,
                device=device,
                dtype=dtype,
                **schedule_kwargs,
            )
        self.schedule = schedule
        self.num_timesteps = int(getattr(schedule, "num_timesteps", num_timesteps))

    # ------------------------------------------------------------------ #
    def set_classifier(self, classifier: Optional[nn.Module]) -> "SimilarityGuidedLoss":
        """Attach/replace the frozen classifier ``p_phi``."""
        self.classifier = classifier
        return self

    # ------------------------------------------------------------------ #
    def sigma_hat(self, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        return sigma_hat_from_schedule(self.schedule, t, ndim=ndim)

    def c2_coefficient(self, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        r"""Appendix A.2 constant :math:`C_2 = \beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))`."""
        beta = _extract(self.schedule.betas, t, ndim)
        alpha = _extract(self.schedule.alphas, t, ndim)
        ab = _extract(self.schedule.alphas_cumprod, t, ndim)
        sigma = _extract(getattr(self.schedule, "sigma", torch.zeros_like(self.schedule.betas)), t, ndim)
        denom = 2.0 * (sigma ** 2) * alpha * (1.0 - ab)
        return (beta ** 2) / denom.clamp(min=1e-20)

    # ------------------------------------------------------------------ #
    def guidance_grad(
        self,
        x_t: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        classifier: Optional[nn.Module] = None,
        gamma: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        r""":math:`\gamma \nabla_{x_t}\log p_\phi(y=\mathcal{T}\mid x_t)` (detached)."""
        clf = classifier if classifier is not None else self.classifier
        g = self.gamma if gamma is None else float(gamma)
        if clf is None or g == 0.0:
            return None
        return classifier_guidance_grad(
            clf,
            x_t,
            t,
            target_index=self.target_index,
            grad_scale=g * self.classifier_grad_scale,
            detach=self.detach_classifier_grad,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        eps_pred: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        classifier: Optional[nn.Module] = None,
        gamma: Optional[float] = None,
        eps_target: Optional[torch.Tensor] = None,
        return_info: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Evaluate Eq. (5).

        Two interchangeable calling conventions are supported:

        1. ``loss(x0=..., t=..., noise=..., model=...)``: samples ``x_t`` with
           the forward process and predicts ``eps_theta`` internally.
        2. ``loss(x_t=..., t=..., eps_target=..., eps_pred=...)``: the caller
           already built ``x_t`` (e.g. with adversarial noise) and provides the
           (possibly noise-corrected) target and the model prediction.
        """
        if x_t is None:
            if x0 is None:
                raise ValueError("forward requires either x0 or x_t")
            if t is None:
                t = self.sample_timesteps(x0.shape[0], device=x0.device)
            t = t.to(device=x0.device)
            noise = torch.randn_like(x0) if noise is None else noise
            x_t = self.schedule.q_sample(x0, t, noise)
        else:
            if t is None:
                raise ValueError("forward requires t when x_t is given")
            t = t.to(device=x_t.device)
            if noise is None and eps_target is None:
                raise ValueError("forward requires noise (or eps_target) when x_t is given")

        ndim = x_t.dim()

        # ---- target noise (with similarity guidance) -------------------- #
        if eps_target is None:
            base = noise
            eps_target, sigma_hat = corrected_noise_target(
                base,
                x_t,
                t,
                self.schedule,
                classifier if classifier is not None else self.classifier,
                gamma=self.gamma if gamma is None else float(gamma),
                target_index=self.target_index,
                detach_classifier_grad=self.detach_classifier_grad,
                grad_scale=self.classifier_grad_scale,
            )
        else:
            sigma_hat = self.sigma_hat(t, ndim)

        # ---- model prediction ------------------------------------------- #
        if eps_pred is None:
            if model is None:
                raise ValueError("forward requires model (or eps_pred)")
            eps_pred = self.predict_noise(model, x_t, t)

        eps_target = eps_target.to(device=eps_pred.device, dtype=eps_pred.dtype)
        diff = eps_target - eps_pred

        if self.include_c2:
            c2 = self.c2_coefficient(t, ndim).to(device=diff.device, dtype=diff.dtype)
            per_sample = (diff ** 2) * c2
        else:
            per_sample = diff ** 2

        if self.reduction == "none":
            loss = per_sample
        elif self.reduction == "sum":
            loss = per_sample.sum()
        else:
            loss = per_sample.mean()

        if not return_info:
            return loss

        with torch.no_grad():
            info = {
                "loss": float(loss.detach().mean()),
                "sigma_hat_mean": float(sigma_hat.detach().mean()),
                "t_mean": float(t.float().mean()),
                "gamma": self.gamma if gamma is None else float(gamma),
                "target_norm": float(eps_target.detach().float().norm().mean()),
                "pred_norm": float(eps_pred.detach().float().norm().mean()),
            }
        return loss, info

    # ------------------------------------------------------------------ #
    @staticmethod
    def predict_noise(model: Any, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Call the adapted backbone uniformly: ``eps_theta(x_t, t)``."""
        fn = _resolve_callable(model, ("epsilon_theta", "predict_noise", "predict_eps"))
        if fn is not None:
            return _unwrap_prediction(fn(x_t, t))
        if callable(model):
            try:
                return _unwrap_prediction(model(x_t, t))
            except TypeError:
                return _unwrap_prediction(model(x_t, t, None))
        raise TypeError("model is not callable and exposes no epsilon_theta")

    def sample_timesteps(
        self,
        batch_size: int,
        device: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """``t ~ Uniform({1, ..., T})`` as in Ho et al. (2020)."""
        return torch.randint(
            self.min_t,
            self.num_timesteps + 1,
            (batch_size,),
            device=device,
            generator=generator,
        ).long()

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, T={self.num_timesteps}, "
            f"detach_classifier_grad={self.detach_classifier_grad}, "
            f"include_c2={self.include_c2}, reduction={self.reduction}"
        )


# --------------------------------------------------------------------------- #
# functional API + factory
# --------------------------------------------------------------------------- #
def similarity_guided_loss(
    model: Any,
    x0: torch.Tensor,
    classifier: Optional[nn.Module] = None,
    t: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    gamma: float = 5.0,
    schedule: Optional[NoiseSchedule] = None,
    diffusion: Optional[Any] = None,
    eps_target: Optional[torch.Tensor] = None,
    x_t: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    return_info: bool = False,
    detach_classifier_grad: bool = True,
    target_index: Optional[int] = None,
    num_timesteps: int = 1000,
    schedule_name: str = "linear",
    **kwargs: Any,
) -> Any:
    """One-shot functional wrapper around :class:`SimilarityGuidedLoss`."""
    criterion = SimilarityGuidedLoss(
        classifier=classifier,
        gamma=gamma,
        schedule=schedule,
        diffusion=diffusion,
        num_timesteps=num_timesteps,
        schedule_name=schedule_name,
        reduction=reduction,
        detach_classifier_grad=detach_classifier_grad,
        target_index=target_index,
    ).to(device=x0.device if x0 is not None else (x_t.device if x_t is not None else None))
    return criterion(
        x0=x0,
        t=t,
        noise=noise,
        model=model,
        x_t=x_t,
        eps_target=eps_target,
        return_info=return_info,
        **kwargs,
    )


def build_sg_loss(
    cfg: Optional[Dict[str, Any]] = None,
    classifier: Optional[nn.Module] = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    device: Optional[Any] = None,
    **overrides: Any,
) -> SimilarityGuidedLoss:
    """Build a :class:`SimilarityGuidedLoss` from the YAML config blocks.

    Reads ``ant.gamma``, ``ant.detach_classifier_grad``,
    ``ant.classifier_target_index``, ``diffusion.*``, ``classifier.grad_scale``
    and per-task overrides from ``tasks.<task>`` / ``defaults``.
    """
    cfg = cfg or {}
    flat: Dict[str, Any] = {}

    for block in ("defaults", "ant", "diffusion"):
        sub = cfg.get(block)
        if isinstance(sub, dict):
            flat.update({k: v for k, v in sub.items() if not isinstance(v, dict)})

    classifier_cfg = cfg.get("classifier")
    if isinstance(classifier_cfg, dict):
        flat.setdefault("target_index", classifier_cfg.get("target_index"))
        flat.setdefault("classifier_grad_scale", classifier_cfg.get("grad_scale", 1.0))

    task_cfg = cfg.get("tasks", {})
    if task is not None and isinstance(task_cfg, dict) and isinstance(task_cfg.get(task), dict):
        flat.update({k: v for k, v in task_cfg[task].items() if not isinstance(v, dict)})

    flat.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
    flat.update(overrides)

    # normalise keys coming from the config files
    aliases = {
        "T": "num_timesteps",
        "beta_schedule": "schedule_name",
        "schedule": "schedule_name",
        "use_c2_constant": "include_c2",
    }
    for src, dst in aliases.items():
        if src in flat and dst not in flat:
            flat[dst] = flat.pop(src)
        elif src in flat:
            flat.pop(src)

    known = {
        "gamma",
        "num_timesteps",
        "schedule_name",
        "eta",
        "detach_classifier_grad",
        "target_index",
        "classifier_grad_scale",
        "reduction",
        "include_c2",
        "min_t",
        "beta_start",
        "beta_end",
        "cosine_s",
    }
    kwargs = {k: v for k, v in flat.items() if k in known and v is not None}

    schedule_kwargs = {
        k: kwargs.pop(k) for k in ("beta_start", "beta_end", "cosine_s") if k in kwargs
    }
    if backbone == "ldm":
        # schedule construction is backbone independent, keep the argument for
        # signature symmetry with the other build_* factories.
        pass

    return SimilarityGuidedLoss(
        classifier=classifier,
        device=device,
        **schedule_kwargs,
        **kwargs,
    )
